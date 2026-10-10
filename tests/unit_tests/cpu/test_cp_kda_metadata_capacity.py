# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check rank-local KDA routing capacity across packed document layouts."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from attn_gym.linear.context_parallel import ContextParallelRouting

from torchtitan.models.common.attention import (
    LinearAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("permuted", [False, True])
def test_routing_preserves_global_capacity_and_active_entries(
    cp_rank,
    permuted,
    monkeypatch,
):
    """Keep tensor shapes fixed without changing active routing or document cuts."""
    from torchtitan.distributed.context_parallel import get_token_fragments

    module = importlib.import_module(ContextParallelInnerKDA.__module__)
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(module, "spmd_mesh_group", lambda axis: group)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: cp_rank)
    permutation = torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5]]) if permuted else None
    fragments = get_token_fragments(8, cp_size=2, permutation=permutation)
    max_fragments = max(len(owned) for owned in fragments)
    layouts = ([0, 4, 8, 8, 8], [0, 2, 6, 8, 8])
    shapes = []
    for layout in layouts:
        offsets = torch.tensor(layout, dtype=torch.int32)
        metadata = LinearAttentionMetadata(
            varlen=VarlenAttentionMetadata(offsets, offsets, 8, 8),
            num_conv_history_tokens=3,
        )
        result = ContextParallelInnerKDA.prepare_cp_metadata(
            metadata,
            permutation=permutation,
        )
        routing = result.cp_routing
        compact = ContextParallelRouting.from_fragments(
            cu_seqlens_global=layout,
            fragments=fragments,
            cp_rank=cp_rank,
            device="cpu",
            conv_history=3,
        )
        capacity = len(layout) - 1 + max_fragments - 1
        assert routing.cu_seqlens.shape == (capacity + 1,)
        assert routing.entry_sources.shape == (capacity,)
        assert routing.conv_sources.shape == (capacity, 3)
        assert result.varlen is metadata.varlen
        for name, value in vars(compact).items():
            actual = getattr(routing, name)
            if isinstance(value, torch.Tensor):
                active = tuple(slice(0, length) for length in value.shape)
                torch.testing.assert_close(actual[active], value, rtol=0, atol=0)
            else:
                assert actual == value
        count = compact.cu_seqlens.numel() - 1
        assert torch.all(routing.cu_seqlens[count + 1 :] == routing.tokens)
        assert torch.all(routing.entry_sources[count:] == routing.slots)
        assert torch.all(routing.exit_sources[count:] == routing.slots)
        assert not torch.any(routing.terminal[count:])
        assert torch.all(
            routing.conv_sources[count:] == routing.world * routing.slots * 3
        )
        shapes.append(
            {
                name: value.shape
                for name, value in vars(routing).items()
                if isinstance(value, torch.Tensor)
            }
        )
    assert shapes[0] == shapes[1]
