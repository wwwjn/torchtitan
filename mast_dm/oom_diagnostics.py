# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recognize CUDA OOMs and report opt-in graph-boundary memory diagnostics."""

import json
import sys

import torch
from torch.utils.hooks import RemovableHandle


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize allocator and explicitly identified CUDA-driver OOMs.

    CUDA graph instantiation raises ``AcceleratorError`` rather than
    ``OutOfMemoryError``. Require a CUDA allocation-error marker for that
    broader exception type so other driver faults are not mislabeled.
    Wrapped causes and contexts are inspected without mutating exceptions
    or querying the device.

    Args:
        error: The exception raised by a CUDA training operation.

    Returns:
        Whether the exception chain contains a recognized allocation failure.
    """
    accelerator_error_type = getattr(torch, "AcceleratorError", None)
    cuda_oom_markers = (
        "CUDA error: out of memory",
        "cudaErrorMemoryAllocation",
        "CUDA_ERROR_OUT_OF_MEMORY",
    )
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, torch.OutOfMemoryError):
            return True
        if (
            accelerator_error_type is not None
            and isinstance(current, accelerator_error_type)
            and any(marker in str(current) for marker in cuda_oom_markers)
        ):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def register_graph_memory_diagnostics() -> tuple[RemovableHandle, RemovableHandle]:
    """Log memory after capture and instantiation, never inside live capture.

    The capture-end hook runs after CUDA ends capture but before graph
    instantiation. It distinguishes existing reservation from allocations
    made by the driver while instantiating the executable graph. No tensors,
    synchronization, allocator reset, or replay hooks are introduced.

    Returns:
        Capture-end and post-instantiation handles. The caller must remove
        both handles after the diagnostic run.
    """

    def report_memory(phase: str) -> None:
        """Query a finished capture without masking graph lifecycle failures."""
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            print(
                "CUDA graph memory totals: "
                + json.dumps(
                    {
                        "phase": phase,
                        "device": torch.cuda.current_device(),
                        "allocated_bytes": torch.cuda.memory_allocated(),
                        "reserved_bytes": torch.cuda.memory_reserved(),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "free_bytes": free_bytes,
                        "total_bytes": total_bytes,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            print(
                f"Failed to read CUDA graph memory at {phase}: {error}",
                file=sys.stderr,
                flush=True,
            )

    def capture_ended(graph: torch.cuda.CUDAGraph) -> None:
        """Record reservation before the executable graph is instantiated."""
        report_memory("capture_end_pre_instantiate")

    def instantiated(graph: torch.cuda.CUDAGraph) -> None:
        """Record driver headroom after successful instantiation."""
        report_memory("post_instantiate")

    return (
        torch.cuda.graphs.register_graph_capture_end_hook(capture_ended),
        torch.cuda.graphs.register_graph_instantiate_hook(instantiated),
    )
