# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for allocation-failure detection in the experimental launcher."""

import pytest
import torch

from mast_dm.oom_diagnostics import is_cuda_out_of_memory


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (torch.OutOfMemoryError("allocation failed"), True),
        (torch.AcceleratorError("CUDA error: out of memory"), True),
        (torch.AcceleratorError("cudaErrorMemoryAllocation"), True),
        (torch.AcceleratorError("CUDA_ERROR_OUT_OF_MEMORY"), True),
        (torch.AcceleratorError("CUDA error: an illegal memory access"), False),
        (torch.AcceleratorError("out of memory"), False),
        (RuntimeError("CUDA error: out of memory"), False),
        (MemoryError("out of memory"), False),
    ],
)
def test_recognizes_only_typed_cuda_allocation_failures(error, expected) -> None:
    """Do not mistake an unrelated driver or CPU error for a CUDA OOM."""
    assert is_cuda_out_of_memory(error) is expected


@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_follows_wrapped_driver_allocation_errors(link: str) -> None:
    """Inspect nested errors without modifying the original exception chain."""
    inner = torch.AcceleratorError("CUDA error: out of memory")
    outer = RuntimeError("training failed")
    setattr(outer, link, inner)

    assert is_cuda_out_of_memory(outer)
    assert getattr(outer, link) is inner


def test_exception_context_cycle_terminates() -> None:
    """A cyclic exception chain must not hang the failure reporter."""
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__context__ = second
    second.__context__ = first

    assert not is_cuda_out_of_memory(first)
