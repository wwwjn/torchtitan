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

from torchtitan.components.data import HuggingFaceStreamingSource, SingleDatasetConfig
from torchtitan.components.optim.optimizer import AdamW, DistMuon
from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.experiments.dist_moe_kimi_k3 import config_registry
from torchtitan.models.common.attention.cp_attention import (
    UlyssesCPVarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.dist_moe import (
    DistMoeRoutedExperts,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan.models.common.moe import MoE
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.models.kimi_k3.model import KimiMLAAttention


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
    assert config.training.empty_device_cache_after_step
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


@pytest.mark.parametrize(
    ("ep_size", "num_experts", "num_tokens"),
    [
        (2, 28, 256),
        (64, 896, 2048),
    ],
)
def test_phased_round_robin_balances_partial_cycle_across_ep_ranks(
    ep_size: int,
    num_experts: int,
    num_tokens: int,
) -> None:
    """Every EP destination receives the same row count at scratch factor one."""
    router = cast(
        config_registry._PhasedRoundRobinTokenChoiceTopKRouter,
        SimpleNamespace(num_experts=num_experts, top_k=16),
    )
    scores_TE = torch.empty(num_tokens, router.num_experts)
    received_rows = torch.zeros(ep_size, dtype=torch.int64)

    for ep_rank in range(ep_size):
        ep_mesh = SimpleNamespace(
            size=lambda: ep_size,
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
        received_rows += counts_E.reshape(ep_size, num_experts // ep_size).sum(dim=1)

    torch.testing.assert_close(
        received_rows,
        torch.full_like(received_rows, num_tokens * router.top_k),
    )


def test_v4_2_mxfp8_recipe_matches_256_gpu_long_context_workload() -> None:
    """The Dist-MoE variant preserves V4.2's data and parallel topology."""
    config = config_registry.kimi_k3_full_long_context_v4_2_dist_moe_mxfp8_gb200_256()
    model = cast(KimiK3Model.Config, config.model)

    assert config.parallelism.pipeline_parallel_degree == 4
    assert config.parallelism.pipeline_parallel_schedule == "Interleaved1F1B"
    assert config.parallelism.pipeline_parallel_module_fqns_per_model_part is None
    assert config.parallelism.pipeline_parallel_layers_per_stage == 12
    assert config.parallelism.num_pp_microbatches == 64
    assert config.parallelism.pp_max_unsharded_active_stages == 1
    assert config.parallelism.context_parallel_degree == 16
    assert config.parallelism.context_parallel_load_balancer is None
    assert config.parallelism.data_parallel_shard_degree == 4
    assert config.parallelism.expert_parallel_degree == 64
    assert config.parallelism.fsdp_reshard_after_forward == "always"

    assert model.max_context_length == 32_768
    assert config.training.max_context_length == 32_768
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 32_768
    assert config.training.num_tokens_per_train_step == 8_388_608
    assert config.training.steps == 12
    assert config.training.disable_cuda_graphs
    assert config.training.dtype == "float32"
    assert config.training.mixed_precision_param == "bfloat16"
    assert config.training.mixed_precision_reduce == "bfloat16"

    dataset = config.dataloader.dataset
    assert isinstance(dataset, SingleDatasetConfig)
    assert isinstance(dataset.source, HuggingFaceStreamingSource.Config)
    assert dataset.source.load_dataset_kwargs["data_files"].endswith(
        "/seq32768/part-*.json.gz"
    )
    assert isinstance(
        dataset.processor,
        config_registry._PretokenizedLongContextProcessor.Config,
    )
    assert config.dataloader.max_num_documents == 1
    assert not config.profiler.enable_profiling

    runtime = config.dist_moe
    assert isinstance(runtime, DistMoeRuntime.Config)
    assert runtime.activation_slot_capacity_factor is None
    assert runtime.scratch_capacity_factor == 1.0
    assert runtime.vmm_capacity_factor is None
    assert runtime.pp_activation_slot_policy == "stage_microbatch"
    assert isinstance(config.activation_checkpoint, RegionAC.Config)
    assert config.activation_checkpoint.save_regions == []

    for _, attention, _, _ in model.traverse(KimiMLAAttention.Config):
        assert isinstance(
            attention.inner_attention,
            UlyssesCPVarlenInnerAttention.Config,
        )
    for layer in model.layers:
        if layer.delta_attention is not None:
            assert isinstance(
                layer.delta_attention.inner_kda,
                ContextParallelInnerKDA.Config,
            )
        if layer.moe is not None:
            assert isinstance(
                layer.moe.router,
                config_registry._PhasedRoundRobinTokenChoiceTopKRouter.Config,
            )

    experts = list(model.traverse(MXFP8DistMoeRoutedExperts.Config))
    assert len(experts) == 92
    assert all(
        type(expert) is MXFP8DistMoeRoutedExperts.Config for _, expert, _, _ in experts
    )
    assert all(expert.activation == "situglu" for _, expert, _, _ in experts)
    assert all(expert.situglu_beta == 4.0 for _, expert, _, _ in experts)
    assert all(expert.situglu_linear_beta == 25.0 for _, expert, _, _ in experts)
    assert all(expert.block_scaled_config.fast_math for _, expert, _, _ in experts)

    routers = list(model.traverse(MoE.Config))
    assert len(routers) == 92
    dist_muon = next(
        optimizer
        for optimizer in config.optim.optimizer.optimizers
        if isinstance(optimizer, DistMuon.Config)
    )
    assert not dist_muon.persistent_redistribution_buffers
    dense_layout = dist_muon.compute_sharding_by_fqn[
        "layers.0.delta_attention.q_proj.weight"
    ]
    assert set(dense_layout.shardings_by_mesh_axis) == {"dp_shard_cp"}
    expert_layout = dist_muon.compute_sharding_by_fqn[
        "layers.1.moe.routed_experts.w13.weight"
    ]
    assert set(expert_layout.shardings_by_mesh_axis) == {"edp_shard", "ep"}
    adamw = next(
        optimizer
        for optimizer in config.optim.optimizer.optimizers
        if isinstance(optimizer, AdamW.Config)
    )
    assert adamw.moment_dtype == "parameter"
