# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)


_batch_invariant_enabled: bool = False
_batch_invariant_extra_lib: torch.library.Library | None = None


def is_in_batch_invariant_mode() -> bool:
    """Return whether batch-invariant mode is active."""
    return _batch_invariant_enabled


def _sum_dim_batch_invariant(
    input: torch.Tensor,
    dim: list[int] | None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Implement ``sum.dim_IntList`` using fixed-schedule reductions."""
    from batch_invariant_ops import mean_dim

    dims = tuple(range(input.ndim)) if not dim else tuple(dim)
    dims = tuple(axis if axis >= 0 else axis + input.ndim for axis in dims)
    if len(set(dims)) != len(dims) or any(
        axis < 0 or axis >= input.ndim for axis in dims
    ):
        raise IndexError(f"invalid reduction axes {dim} for a {input.ndim}D tensor")

    output_dtype = dtype or (
        input.dtype if input.is_floating_point() or input.is_complex() else torch.int64
    )
    num_reduced_elements = math.prod(input.shape[axis] for axis in dims)
    if num_reduced_elements == 0:
        output_shape = list(input.shape)
        for axis in sorted(dims, reverse=True):
            if keepdim:
                output_shape[axis] = 1
            else:
                output_shape.pop(axis)
        return torch.zeros(output_shape, dtype=output_dtype, device=input.device)

    accumulation_dtype = torch.int64 if output_dtype == torch.bool else output_dtype
    result = (
        input if input.dtype == accumulation_dtype else input.to(accumulation_dtype)
    )
    if result.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        for axis in sorted(dims, reverse=not keepdim):
            result = mean_dim(result, axis, keepdim=keepdim)
        return result * num_reduced_elements

    # Integer, boolean, complex, and float64 reductions do not need the
    # low-precision Triton kernel. A prefix scan fixes their reduction order.
    for axis in sorted(dims, reverse=not keepdim):
        result = torch.cumsum(result, dim=axis, dtype=output_dtype).select(axis, -1)
        if keepdim:
            result = result.unsqueeze(axis)
    return result.to(output_dtype)


def _grouped_mm_batch_invariant(
    input: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Implement ``_grouped_mm`` as one batch-invariant ``mm`` per group.

    Shape legend: R = rows, E = groups, I = input features, O = output features.

    Supports the two layouts MoE uses, with ``offs`` the exclusive cumulative row
    end of each group:
    - ``[R, I] x [E, I, O] -> [R, O]``: each group's rows times its weight (forward
      and input gradient);
    - ``[I, R] x [R, O] -> [E, I, O]``: each group's columns times its rows
      (weight gradient).

    The group bounds are read on the host, so a call synchronizes with the device
    and cannot be captured in a CUDA graph.
    """
    if offs is None or bias is not None:
        raise NotImplementedError(
            "batch-invariant _grouped_mm supports offs without bias only"
        )
    output_dtype = out_dtype or input.dtype
    group_ends = offs.tolist()
    group_starts = [0] + group_ends[:-1]
    if input.ndim == 2 and mat2.ndim == 3:
        input_RI, weight_EIO = input, mat2
        output_RO = torch.zeros(
            input_RI.shape[0],
            weight_EIO.shape[-1],
            dtype=output_dtype,
            device=input.device,
        )
        for group, (start, end) in enumerate(zip(group_starts, group_ends)):
            if end > start:
                output_RO[start:end] = torch.mm(input_RI[start:end], weight_EIO[group])
        return output_RO
    if input.ndim == 2 and mat2.ndim == 2:
        input_IR, input_RO = input, mat2
        output_EIO = torch.zeros(
            len(group_ends),
            input_IR.shape[0],
            input_RO.shape[-1],
            dtype=output_dtype,
            device=input.device,
        )
        for group, (start, end) in enumerate(zip(group_starts, group_ends)):
            if end > start:
                output_EIO[group] = torch.mm(
                    input_IR[:, start:end], input_RO[start:end]
                )
        return output_EIO
    raise NotImplementedError(
        f"batch-invariant _grouped_mm does not support {input.ndim}D x {mat2.ndim}D"
    )


def set_batch_invariance(enable: bool) -> None:
    """Enable batch-invariant mode for reproducible RL training.

    Delegates ATen operator overrides to the ``batch_invariant_ops`` package
    and extends them to ``sum.dim_IntList`` and ``bmm``, producing
    bit-identical results for the same input regardless of batch composition.

    On top of that, this function applies torchtitan-specific settings:
    - NCCL env vars for deterministic inter-GPU collectives
    - Disables reduced-precision reductions and TF32

    Note: callers must set ``debug.deterministic=True`` separately.
    """
    global _batch_invariant_enabled, _batch_invariant_extra_lib
    if not enable or _batch_invariant_enabled:
        return

    # Register batch-invariant ATen overrides via upstream package
    # https://github.com/thinking-machines-lab/batch_invariant_ops
    from batch_invariant_ops import enable_batch_invariant_mode as _upstream_enable
    from vllm.model_executor.determinism.batch_invariant import (  # pyrefly: ignore[missing-import]
        bmm_batch_invariant,
    )

    _upstream_enable()
    # batch_invariant_ops does not yet override sum or bmm. Compose sum from
    # its fixed-schedule mean and reuse vLLM's fixed-schedule bmm kernel.
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("batch-invariant mode requires an accelerator")
    dispatch_key = accelerator.type.upper()
    _batch_invariant_extra_lib = torch.library.Library("aten", "IMPL")
    _batch_invariant_extra_lib.impl(
        "aten::sum.dim_IntList",
        _sum_dim_batch_invariant,
        dispatch_key,
        allow_override=True,
    )
    _batch_invariant_extra_lib.impl(
        "aten::bmm",
        bmm_batch_invariant,
        dispatch_key,
        allow_override=True,
    )
    # Below SM90, a row's _grouped_mm (MoE expert GEMM) result depends on the
    # sizes of the groups in the call, i.e. on how many tokens the batch routes
    # to each expert. Compose it from the fixed-schedule mm instead.
    if accelerator.type == "cuda" and torch.cuda.get_device_capability() < (9, 0):
        _batch_invariant_extra_lib.impl(
            "aten::_grouped_mm",
            _grouped_mm_batch_invariant,
            dispatch_key,
            allow_override=True,
        )

    # Make the TP all-reduce, the only reducing collective in a batch-invariant
    # forward, sum each element in the same cross-rank order for any message
    # size. Must be set BEFORE dist.init_process_group.
    # Tree sums every element along one fixed tree; ring's order depends on how
    # the message is chunked. Pinning the all-reduce to tree also excludes
    # NVLS and CollNet.
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    # NCCL alternates channels between two inter-node trees, so with several
    # channels the tree an element takes depends on the message size. One
    # channel keeps a single tree.
    os.environ["NCCL_MAX_NCHANNELS"] = "1"

    # Disable reduced-precision reductions: these allow cuBLAS to use
    # lower-precision accumulation that can round differently depending
    # on batch size / tile decomposition.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Disable TF32 for exact fp32 accumulation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _batch_invariant_enabled = True

    logger.info(
        "Batch-invariant mode enabled: mm, addmm, bmm, sum.dim_IntList, "
        "_log_softmax, mean.dim overridden with fixed-schedule kernels; "
        "reduced-precision reductions and TF32 disabled"
    )
