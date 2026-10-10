# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for allocation-failure detection in the experimental launcher."""

import json
from unittest.mock import Mock

import pytest
import torch

from mast_dm.oom_diagnostics import (
    is_cuda_out_of_memory,
    register_graph_memory_diagnostics,
)


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


def test_graph_memory_hooks_query_only_when_invoked(monkeypatch, capsys) -> None:
    """Registration adds no CUDA work; only finished-boundary hooks query memory."""
    callbacks = []
    handles = (Mock(), Mock())

    def register(callback):
        """Retain the callback as the real global registries would."""
        callbacks.append(callback)
        return handles[len(callbacks) - 1]

    monkeypatch.setattr(torch.cuda.graphs, "register_graph_capture_end_hook", register)
    monkeypatch.setattr(torch.cuda.graphs, "register_graph_instantiate_hook", register)
    memory_info = Mock(return_value=(40, 100))
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_info)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 30)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 25)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 35)
    assert register_graph_memory_diagnostics() == handles
    memory_info.assert_not_called()

    for callback in callbacks:
        callback(object())

    records = [
        json.loads(line.removeprefix("CUDA graph memory totals: "))
        for line in capsys.readouterr().out.splitlines()
    ]
    assert [record.pop("phase") for record in records] == [
        "capture_end_pre_instantiate",
        "post_instantiate",
    ]
    assert (
        records
        == [
            {
                "device": 1,
                "allocated_bytes": 20,
                "reserved_bytes": 30,
                "peak_allocated_bytes": 25,
                "peak_reserved_bytes": 35,
                "free_bytes": 40,
                "total_bytes": 100,
            }
        ]
        * 2
    )


def test_graph_memory_query_failure_is_reported_without_raising(
    monkeypatch, capsys
) -> None:
    """A failed diagnostic must not replace an instantiation error."""
    callbacks = []

    def register(callback):
        """Capture the callbacks without initializing a device."""
        callbacks.append(callback)
        return Mock()

    monkeypatch.setattr(torch.cuda.graphs, "register_graph_capture_end_hook", register)
    monkeypatch.setattr(torch.cuda.graphs, "register_graph_instantiate_hook", register)
    query = Mock(side_effect=RuntimeError("query failed"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    register_graph_memory_diagnostics()
    callbacks[0](object())
    assert "capture_end_pre_instantiate: query failed" in capsys.readouterr().err
