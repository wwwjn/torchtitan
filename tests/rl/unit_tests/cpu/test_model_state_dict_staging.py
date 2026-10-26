# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.rl.generator.model_state_dict_staging import (
    ChunkedStateDictStaging,
    group_fqns_by_module,
)


def test_group_fqns_by_module_splits_each_layer():
    fqns = [
        "scale",
        "tok_embeddings.weight",
        "layers.0.attention.wq.weight",
        "layers.0.moe.experts.w1",
        "layers.1.attention.wq.weight",
        "norm.weight",
        "output.weight",
    ]
    assert group_fqns_by_module(fqns) == [
        ["scale"],
        ["tok_embeddings.weight"],
        ["layers.0.attention.wq.weight", "layers.0.moe.experts.w1"],
        ["layers.1.attention.wq.weight"],
        ["norm.weight"],
        ["output.weight"],
    ]


def test_chunks_pack_whole_modules_into_two_buffers_within_the_budget():
    # Sizes in bytes, each rounded up to a multiple of 64.
    state_dict = {
        "tok_embeddings.weight": torch.zeros(32),  # 128
        "layers.0.weight": torch.zeros(8),  # 32 -> 64
        "layers.0.bias": torch.zeros(8, dtype=torch.bfloat16),  # 16 -> 64
        "layers.1.weight": torch.zeros(16),  # 64
        "layers.2.weight": torch.zeros(16),  # 64
        "norm.weight": torch.zeros(4),  # 16 -> 64
    }
    # Half the budget is 300 bytes; a buffer holds its largest power of two, 256.
    staging = ChunkedStateDictStaging(
        state_dict, max_pinned_bytes=600, pin_memory=False
    )

    assert staging.chunks == [
        ["tok_embeddings.weight", "layers.0.weight", "layers.0.bias"],
        ["layers.1.weight", "layers.2.weight", "norm.weight"],
    ]
    for fqn, value in state_dict.items():
        staged = staging.state_dict[fqn]
        assert (staged.shape, staged.dtype) == (value.shape, value.dtype)
    buffer_nbytes = {
        staged.untyped_storage().data_ptr(): staged.untyped_storage().nbytes()
        for staged in staging.state_dict.values()
    }
    assert len(buffer_nbytes) == 2
    assert all(nbytes <= 256 for nbytes in buffer_nbytes.values())


def test_a_module_larger_than_a_buffer_is_rejected():
    state_dict = {"layers.0.weight": torch.zeros(48), "layers.1.weight": torch.zeros(4)}
    # 192 bytes do not fit in a 128-byte buffer (the largest power of two in 200).
    with pytest.raises(ValueError, match="layers.0 needs .* at least"):
        ChunkedStateDictStaging(state_dict, max_pinned_bytes=400, pin_memory=False)
