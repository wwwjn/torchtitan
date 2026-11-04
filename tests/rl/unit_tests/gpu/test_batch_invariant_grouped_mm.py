# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-invariant ``_grouped_mm`` (MoE expert GEMM) below SM90."""

import pytest
import torch

from torchtitan.distributed.batch_invariant import set_batch_invariance

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0),
    reason="the override applies below SM90",
)


def _offsets(counts: list[int]) -> torch.Tensor:
    return torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)


def test_grouped_mm_rows_do_not_depend_on_group_sizes():
    set_batch_invariance(True)
    torch.manual_seed(0)
    num_groups, out_features, in_features = 64, 1536, 256
    weight_EOI = torch.randn(
        num_groups, out_features, in_features, device="cuda", dtype=torch.bfloat16
    )
    input_RI = torch.randn(1000, in_features, device="cuda", dtype=torch.bfloat16)
    # Group 0 holds rows 0..4 in both calls; the other groups' sizes differ.
    small = torch._grouped_mm(
        input_RI[:5],
        weight_EOI.transpose(-2, -1),
        offs=_offsets([5] + [0] * (num_groups - 1)),
    )
    large = torch._grouped_mm(
        input_RI[:750],
        weight_EOI.transpose(-2, -1),
        offs=_offsets([120] + [10] * (num_groups - 1)),
    )
    assert torch.equal(small[:5], large[:5])


def test_grouped_mm_weight_gradient_layout():
    set_batch_invariance(True)
    torch.manual_seed(0)
    counts = [3, 0, 7, 2]
    input_IR = torch.randn(16, sum(counts), device="cuda", dtype=torch.bfloat16)
    grad_RO = torch.randn(sum(counts), 24, device="cuda", dtype=torch.bfloat16)
    grad_weight_EIO = torch._grouped_mm(input_IR, grad_RO, offs=_offsets(counts))
    start = 0
    for group, count in enumerate(counts):
        expected = (
            input_IR[:, start : start + count].float()
            @ grad_RO[start : start + count].float()
        )
        torch.testing.assert_close(
            grad_weight_EIO[group].float(), expected, atol=5e-2, rtol=1e-2
        )
        start += count
