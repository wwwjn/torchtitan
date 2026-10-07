# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi DistMuon layout normalization tests."""

from torch.distributed.tensor import Shard

from torchtitan.components.optim import DistMuon, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout
from torchtitan_recipes.tests.models.kimi_k2_7 import (
    _align_dist_muon_dense_fsdp_compute_layouts,
)


def _optimizer_config() -> OptimizersContainer.Config:
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=r".*",
                compute_sharding_by_fqn={
                    "dense.weight": ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": Shard(0),
                            "tp": Shard(0),
                        },
                        shard_order_by_tensor_dim={0: ("dp_shard", "tp")},
                    ),
                    "expert.weight": ComputeLayout(
                        shardings_by_mesh_axis={"ep": Shard(0)},
                    ),
                },
                bucket_configs=(BucketConfig(patterns=("*",)),),
            )
        ]
    )


def test_dense_layouts_follow_flattened_context_parallel_fsdp_axis() -> None:
    """CP recipes name the physical FSDP parameter-storage mesh axis."""
    original = _optimizer_config()
    aligned = _align_dist_muon_dense_fsdp_compute_layouts(
        original,
        parallelism=ParallelismConfig(context_parallel_degree=16),
    )
    dist_muon = aligned.optimizers[0]
    assert isinstance(dist_muon, DistMuon.Config)

    dense = dist_muon.compute_sharding_by_fqn["dense.weight"]
    assert dict(dense.shardings_by_mesh_axis) == {
        "dp_shard_cp": Shard(0),
        "tp": Shard(0),
    }
    assert dict(dense.shard_order_by_tensor_dim) == {0: ("dp_shard_cp", "tp")}
    assert (
        dist_muon.compute_sharding_by_fqn["expert.weight"]
        is original.optimizers[0].compute_sharding_by_fqn["expert.weight"]
    )


def test_dense_layout_alignment_is_noop_without_context_parallelism() -> None:
    """The existing CP1 optimizer configuration remains unchanged."""
    optimizer = _optimizer_config()
    assert (
        _align_dist_muon_dense_fsdp_compute_layouts(
            optimizer,
            parallelism=ParallelismConfig(context_parallel_degree=1),
        )
        is optimizer
    )
