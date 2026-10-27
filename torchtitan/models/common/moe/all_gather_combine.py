# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused score-and-sum for the CUDA-graph all-gather MoE combine.

The eager combine gathers all ``G * K`` rows, scores them in place, masks other ranks'
rows and sums over K: four kernels whose traffic grows with every gathered pick, half of
them garbage rows at EP2 (35B-A3B, 8192-token prefill on A100: ~2 ms per layer). One
Triton program per gathered token reads only its local picks and writes the fp32 sum,
with the same rounding: each scored row rounds to bf16, the sum stays fp32.

Opt-in with ``TORCHTITAN_FUSED_AG_COMBINE=1``.
"""

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("TORCHTITAN_FUSED_AG_COMBINE") == "1"
# Combine with all-reduce + slice instead of reduce-scatter up to ALL_REDUCE_MAX_BYTES.
# Only where the EP group is the generator's TP group (DP = 1): the generator routes
# spmd P -> R reductions to vLLM's TP all-reduce without looking at the group.
ALL_REDUCE = os.environ.get("TORCHTITAN_EP_COMBINE_ALLREDUCE") == "1"
ALL_REDUCE_MAX_BYTES = 4 << 20


@triton.jit
def _score_sum_kernel(
    rows,
    sorted_positions,
    scores,
    is_local,
    out,
    D,
    stride_row,
    stride_score,
    stride_out,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    g = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(K):
        if tl.load(is_local + g * K + k):
            position = tl.load(sorted_positions + g * K + k).to(tl.int64)
            score = tl.load(scores + g * stride_score + k).to(tl.float32)
            row = tl.load(rows + position * stride_row + d, mask=d_mask, other=0.0)
            acc += (row.to(tl.float32) * score).to(rows.dtype.element_ty).to(tl.float32)
    tl.store(
        out + g.to(tl.int64) * stride_out + d, acc.to(out.dtype.element_ty), mask=d_mask
    )


def score_sum(
    routed_output_ND: torch.Tensor,
    sorted_positions_GK: torch.Tensor,
    topk_scores_GK: torch.Tensor,
    is_local_GK: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(G, D)``: sum over k of bf16(row[sorted_positions[g, k]] * score[g, k]) for local picks,
    accumulated in fp32 and rounded once to ``out_dtype``."""
    gathered, top_k = is_local_GK.shape
    dim = routed_output_ND.shape[1]
    assert routed_output_ND.stride(1) == 1 and topk_scores_GK.stride(1) == 1
    out_GD = torch.empty(gathered, dim, device=routed_output_ND.device, dtype=out_dtype)
    block_d = min(1024, triton.next_power_of_2(dim))
    _score_sum_kernel[(gathered, triton.cdiv(dim, block_d))](
        routed_output_ND,
        sorted_positions_GK.contiguous(),
        topk_scores_GK,
        is_local_GK.contiguous(),
        out_GD,
        dim,
        routed_output_ND.stride(0),
        topk_scores_GK.stride(0),
        out_GD.stride(0),
        K=top_k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out_GD
