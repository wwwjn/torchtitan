# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
from torch.distributed.tensor import Shard

from torchtitan.components.optim.optimizer import DistMuon
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan_recipes.tests.models.kimi_k2_7 import (
    _align_dist_muon_dense_fsdp_compute_layouts,
    _dist_muon_optimizer as k2_dist_muon_optimizer,
    kimi_k2_5_debugmodel,
)
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer as k3_dist_muon_optimizer,
    kimi_k3_debugmodel,
)


@pytest.mark.parametrize(
    ("factory", "optimizer_factory"),
    [
        (kimi_k2_5_debugmodel, k2_dist_muon_optimizer),
        (kimi_k3_debugmodel, k3_dist_muon_optimizer),
    ],
)
@pytest.mark.parametrize("tp_degree", [2, 4])
@pytest.mark.parametrize("cp_degree", [1, 2])
def test_kimi_dist_muon_tp_preserves_projection_layouts(
    factory, optimizer_factory, tp_degree, cp_degree
):
    """Keep each logical projection's Muon layout on both DP and TP axes."""
    original = factory(seq_len=256)
    parallelism = replace(
        original.parallelism,
        data_parallel_shard_degree=2,
        context_parallel_degree=cp_degree,
        tensor_parallel_degree=tp_degree,
        expert_parallel_degree=4,
    )
    optimizer_config = optimizer_factory(
        original.model,
        muon_lr=8e-4,
        adamw_lr=8e-4,
        parallelism=parallelism,
    )
    optimizer_config = _align_dist_muon_dense_fsdp_compute_layouts(
        optimizer_config, parallelism=parallelism
    )
    optimizer = next(
        item
        for item in optimizer_config.optimizers
        if isinstance(item, DistMuon.Config)
    )
    layouts = optimizer.compute_sharding_by_fqn
    storage_axis = "dp_shard_cp" if cp_degree > 1 else "dp_shard"
    for fqn, layout in layouts.items():
        if ".routed_experts." in fqn:
            assert layout.shard_order_by_tensor_dim[0] == ("ep", "edp_shard")
            continue
        tp_sharding = layout.shardings_by_mesh_axis["tp"]
        assert tp_sharding == layout.shardings_by_mesh_axis[storage_axis]
        assert type(tp_sharding) in (BlockShard, Owned, Shard)

    assert (
        type(layouts["layers.0.feed_forward.w13.weight"].shardings_by_mesh_axis["tp"])
        is Shard
    )
    assert (
        type(layouts["layers.0.feed_forward.w2.weight"].shardings_by_mesh_axis["tp"])
        is Owned
    )

    for layer_id, layer in enumerate(original.model.layers):
        if hasattr(layer.attention, "q_lora_rank") and layer.attention.q_lora_rank:
            query_layout = layouts[f"layers.{layer_id}.attention.wq_b.weight"]
            assert query_layout.shardings_by_mesh_axis["tp"].block_sizes == (
                layer.attention.qk_nope_head_dim,
                layer.attention.qk_rope_head_dim,
            )


@pytest.mark.parametrize("factory", [kimi_k2_5_debugmodel, kimi_k3_debugmodel])
@pytest.mark.parametrize("tp_degree", [2, 4])
def test_kimi_dist_muon_tp_config_is_supported(factory, tp_degree):
    """Accept TP in the actual trainer configuration without changing Muon."""
    original = factory(seq_len=256)
    config = replace(
        original,
        parallelism=replace(
            original.parallelism,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=tp_degree,
            expert_parallel_degree=4,
        ),
    )
    assert config.parallelism.tensor_parallel_degree == tp_degree
    assert any(
        isinstance(item, DistMuon.Config) for item in config.optim.optimizer.optimizers
    )
