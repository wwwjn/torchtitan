# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration tests for the full-scale Kimi K3 Dist-MoE experiments."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from torchtitan.components.optim.optimizer import DistMuon
from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.experiments.dist_moe_kimi_k3 import config_registry
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime


@pytest.mark.parametrize(
    ("factory", "expected_context_length"),
    [
        (
            config_registry.kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512,
            2048,
        ),
        (
            config_registry.kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512_v1_5,
            4096,
        ),
    ],
)
def test_full_recipes_preserve_workload_and_use_upstream_runtime(
    factory,
    expected_context_length,
) -> None:
    """V1 variants differ only in row shape and use full recomputation."""
    config = factory()

    assert config.model.max_context_length == expected_context_length
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 4096
    assert config.training.num_tokens_per_train_step == 4096 * 64 * 64
    assert config.training.steps == 12
    assert config.training.dtype == "float32"
    assert config.training.mixed_precision_param == "bfloat16"
    assert config.training.mixed_precision_reduce == "float32"
    assert "attention_residual" in config.model.local_compile_regions
    assert config.parallelism.pipeline_parallel_degree == 8
    assert config.parallelism.data_parallel_shard_degree == 64
    assert config.parallelism.expert_parallel_degree == 64
    assert config.parallelism.num_pp_microbatches == 64
    assert config.parallelism.pipeline_parallel_schedule == "1F1B"
    assert len(config.parallelism.pipeline_parallel_module_fqns_per_model_part) == 8

    runtime = config.dist_moe
    assert isinstance(runtime, DistMoeRuntime.Config)
    assert runtime.activation_slot_capacity_factor is None
    assert runtime.scratch_capacity_factor == 1.0
    assert runtime.vmm_capacity_factor is None
    assert isinstance(config.activation_checkpoint, RegionAC.Config)
    assert config.activation_checkpoint.save_regions == []
    dist_muon_configs = [
        optimizer
        for optimizer in config.optim.optimizer.optimizers
        if isinstance(optimizer, DistMuon.Config)
    ]
    assert dist_muon_configs
    assert all(
        not optimizer.persistent_redistribution_buffers
        for optimizer in dist_muon_configs
    )

    experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert len(experts) == 92
    assert all(expert.activation == "situglu" for _, expert, _, _ in experts)


def test_phased_round_robin_balances_partial_cycle_across_ep_ranks() -> None:
    """Every EP destination receives the same row count at scratch factor one."""
    router = cast(
        config_registry._PhasedRoundRobinTokenChoiceTopKRouter,
        SimpleNamespace(num_experts=28, top_k=16),
    )
    scores_TE = torch.empty(256, router.num_experts)
    received_rows = torch.zeros(2, dtype=torch.int64)

    for ep_rank in range(2):
        ep_mesh = SimpleNamespace(
            size=lambda: 2,
            get_local_rank=lambda ep_rank=ep_rank: ep_rank,
        )
        sparse_mesh = {"ep": ep_mesh}
        with patch.object(
            config_registry,
            "spmd_sparse_mesh",
            return_value=sparse_mesh,
        ):
            expert_ids_TK = (
                config_registry._PhasedRoundRobinTokenChoiceTopKRouter._select_experts(
                    router,
                    scores_TE,
                )
            )
        counts_E = torch.bincount(
            expert_ids_TK.flatten(),
            minlength=router.num_experts,
        )
        received_rows += counts_E.reshape(2, 14).sum(dim=1)

    torch.testing.assert_close(received_rows, torch.tensor([4096, 4096]))
