# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Identify CUDA allocation failures without invoking device operations."""

import torch


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
