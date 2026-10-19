# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-invariant packed GDN convolution on a paged cache."""

import pytest
import torch

from torchtitan.distributed.batch_invariant import set_batch_invariance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# One channel of a Qwen3.5-0.8B decode token whose output rounds differently when
# its taps are summed in the paged kernel's history order instead of the trainer's.
_INPUTS_W = (-1.046875, 0.404296875, 1.1171875, 1.671875)
_WEIGHTS_W = (
    -0.004486083984375,
    0.0020904541015625,
    -0.0004520416259765625,
    -0.361328125,
)


def test_continuation_matches_one_pass():
    """A token whose taps reach into the cached history matches a one-pass prefill."""
    from torchtitan.models.common.attn_gym import causal_conv1d
    from torchtitan.rl.generator.paged_gdn import _conv_with_inline_history

    set_batch_invariance(True)
    channels = 64
    sequence_WC = (
        torch.tensor(_INPUTS_W, dtype=torch.bfloat16, device="cuda")
        .view(-1, 1)
        .repeat(1, channels)
    )
    weight_CW = (
        torch.tensor(_WEIGHTS_W, dtype=torch.bfloat16, device="cuda")
        .view(1, -1)
        .repeat(channels, 1)
    )
    one_pass = causal_conv1d(
        sequence_WC.unsqueeze(0),
        weight_CW,
        activation="silu",
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
    )[0, -1]

    # The last token continues slot 1, whose history holds the first three.
    state_SWC = torch.zeros(2, 3, channels, dtype=torch.bfloat16, device="cuda")
    state_SWC[1] = sequence_WC[:3]
    continued = _conv_with_inline_history(
        sequence_WC[3:],
        weight_CW,
        state_SWC,
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.tensor([True], device="cuda"),
    )[0]
    assert torch.equal(continued, one_pass)
    assert torch.equal(state_SWC[1], sequence_WC[1:])


def test_matches_paged_kernel_state_semantics():
    """Same outputs and history updates as the paged kernel on a mixed packed step."""
    from torchtitan.models.common.attn_gym import paged_causal_conv1d
    from torchtitan.rl.generator.paged_gdn import _conv_with_inline_history

    set_batch_invariance(True)
    torch.manual_seed(0)
    channels = 256
    weight_CW = (
        torch.randn(channels, 4, dtype=torch.bfloat16, device="cuda") * 0.25
    ).round(decimals=2)
    state_SWC = torch.randn(8, 3, channels, dtype=torch.bfloat16, device="cuda")
    state_SWC[0] = 0
    # A fresh slot may hold anything, NaNs included.
    state_SWC[5] = float("nan")
    # Sequences: decode row, fresh prefill (slot 5), continued chunk, a null
    # zero-length interval, and a fresh prefill shorter than the history.
    lengths = [1, 7, 5, 0, 2]
    slots_N = torch.tensor([1, 5, 3, 0, 6], dtype=torch.int32, device="cuda")
    resumes_N = torch.tensor([True, False, True, False, False], device="cuda")
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    # Small integers keep every sum exact, so both summation orders agree.
    x_TC = torch.randint(
        -4, 5, (sum(lengths), channels), device="cuda", dtype=torch.bfloat16
    )

    expected_state = state_SWC.clone()
    expected = paged_causal_conv1d(
        x_TC.unsqueeze(0),
        weight_CW,
        expected_state,
        slots_N,
        activation="silu",
        cu_seqlens=cu_seqlens,
        has_initial_state=resumes_N,
    )[0]
    actual_state = state_SWC.clone()
    actual = _conv_with_inline_history(
        x_TC, weight_CW, actual_state, slots_N, cu_seqlens, resumes_N
    )
    assert torch.equal(actual, expected)
    written = [1, 3, 5, 6]
    assert torch.equal(actual_state[written], expected_state[written])
    untouched = [0, 2, 4, 7]
    assert torch.equal(actual_state[untouched], state_SWC[untouched])
