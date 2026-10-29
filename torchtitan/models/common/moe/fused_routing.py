# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused inference routing for the MoE generator: fewer, larger kernels per layer.

At decode sizes a MoE layer's routing is ~35 kernels of 1.5-3 us (softmax, top-k,
gather, normalize, scale, routing map, counts; then the all-gather dispatch's id
remapping, masks, int16 sort, inverse permutation, searchsorted and diff), ~120 us per
layer, ~4.8 ms per Qwen3.5-35B-A3B EP2 decode step on A100. Two Triton kernels replace
them:

- ``softmax_topk``: one program per token computes the softmax over experts, picks the
  top k (ties go to the lower expert id), normalizes and scales the picked scores, and
  counts tokens per expert with atomics.
- ``dispatch_prep``: stable-sorts the gathered picks by local expert (each program
  ranks 64 picks against all of them) and writes the source token of each sorted row, each
  pick's sorted position, the per-expert counts and the local mask. Up to
  ``MAX_DISPATCH_ROWS`` picks; larger (prefill) batches keep the torch path, where these
  kernels are noise.

Inference only, not bitwise with the torch path: the softmax and the top-k sum round
differently, and the picks come out in descending score order (``torch.topk`` with
``sorted=False`` leaves the order unspecified), which changes the combine's summation
order. Opt-in with ``TORCHTITAN_FUSED_ROUTING=1``.
"""

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("TORCHTITAN_FUSED_ROUTING") == "1"
MAX_DISPATCH_ROWS = 4096
# gate_softmax_topk up to this many tokens; larger batches run cuBLAS's GEMM.
MAX_GATE_TOKENS = 512
_GATE_BLOCK_E = 32
_GATE_BLOCK_K = 64


@triton.jit
def _gate_partials_kernel(
    x,
    weight,
    partials,
    counts,
    T,
    E,
    stride_x,
    stride_w,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_COUNTS: tl.constexpr,
):
    # partials[split, t, e] = sum over this split's K_PER_SPLIT columns of x[t] * weight[e].
    pid_e, split, pid_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if pid_e == 0 and split == 0 and pid_t == 0:
        # Zero the top-k kernel's per-expert counts here rather than in a memset launch.
        c = tl.arange(0, BLOCK_COUNTS)
        tl.store(counts + c, tl.zeros([BLOCK_COUNTS], dtype=tl.int64), mask=c < E)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    t_mask = t < T
    e_mask = e < E
    acc = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        k = split * K_PER_SPLIT + k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x + t[:, None] * stride_x + k[None, :], mask=t_mask[:, None], other=0.0
        )
        b = tl.load(
            weight + e[None, :] * stride_w + k[:, None], mask=e_mask[None, :], other=0.0
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        partials + (split * T + t[:, None]) * E + e[None, :],
        acc,
        mask=t_mask[:, None] & e_mask[None, :],
    )


@triton.jit
def _softmax_topk_kernel(
    logits,
    scores_out,
    ids_out,
    counts,
    padding,
    E,
    stride_logits,
    stride_split,
    eps,
    scale,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    ROUTE_NORM: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0)
    e = tl.arange(0, BLOCK_E)
    e_mask = e < E
    x = tl.load(logits + t * stride_logits + e, mask=e_mask, other=0.0).to(tl.float32)
    # The gate GEMM's split-K partial sums, added in a fixed order.
    for s in tl.static_range(1, NUM_SPLITS):
        x += tl.load(
            logits + s * stride_split + t * stride_logits + e, mask=e_mask, other=0.0
        )
    x = tl.where(e_mask, x, -float("inf"))
    p = tl.exp(x - tl.max(x, 0))
    prob = p / tl.sum(p, 0)
    # One max per pick over (probability bits, reversed expert id): probabilities are >= 0,
    # so their bits order like their values, and ties go to the lower expert id. Picked and
    # padded experts become -1.
    keys = (prob.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (BLOCK_E - 1 - e).to(
        tl.int64
    )
    keys = tl.where(e_mask, keys, -1)
    o_k = tl.arange(0, K)
    picked = tl.zeros([K], dtype=tl.int64)
    for k in tl.static_range(K):
        best = tl.max(keys, 0)
        picked = tl.where(o_k == k, best, picked)
        keys = tl.where(keys == best, -1, keys)
    picked_ids = BLOCK_E - 1 - (picked & 0xFFFFFFFF)
    picked_scores = (picked >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    if ROUTE_NORM:
        picked_scores = picked_scores / (tl.sum(picked_scores, 0) + eps)
    picked_scores = picked_scores * scale
    if HAS_PADDING:
        # A padding row routes nowhere: expert id -1, score 0, not counted.
        is_padding_K = (o_k >= 0) & (tl.load(padding + t) != 0)
        tl.atomic_add(
            counts + picked_ids, tl.full([K], 1, dtype=tl.int64), mask=~is_padding_K
        )
        picked_ids = tl.where(is_padding_K, -1, picked_ids)
        picked_scores = tl.where(is_padding_K, 0.0, picked_scores)
    else:
        tl.atomic_add(counts + picked_ids, tl.full([K], 1, dtype=tl.int64))
    tl.store(scores_out + t * K + o_k, picked_scores)
    tl.store(ids_out + t * K + o_k, picked_ids)


def _topk(
    logits: torch.Tensor,
    tokens: int,
    experts: int,
    num_splits: int,
    stride_logits: int,
    stride_split: int,
    top_k: int,
    route_norm: bool,
    eps: float,
    scale: float,
    counts: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``counts``: zeroed ``(E,)`` int64 per-expert counts, or None to allocate them.

    ``padding_mask``: optional bool ``(T,)``, true for padding rows, which get expert id -1
    and score 0 and are not counted.
    """
    assert top_k == triton.next_power_of_2(top_k)
    scores = torch.empty(tokens, top_k, device=logits.device, dtype=torch.float32)
    ids = torch.empty(tokens, top_k, device=logits.device, dtype=torch.int64)
    if counts is None:
        counts = torch.zeros(experts, device=logits.device, dtype=torch.int64)
    if tokens:
        _softmax_topk_kernel[(tokens,)](
            logits,
            scores,
            ids,
            counts,
            ids if padding_mask is None else padding_mask.view(torch.uint8),
            experts,
            stride_logits,
            stride_split,
            eps,
            scale,
            K=top_k,
            NUM_SPLITS=num_splits,
            ROUTE_NORM=route_norm,
            HAS_PADDING=padding_mask is not None,
            BLOCK_E=triton.next_power_of_2(experts),
            # One warp per token keeps the K max reductions in shuffles (A100, E = 256).
            num_warps=1 if experts <= 512 else 4,
        )
    return scores, ids, counts


def softmax_topk(
    gate_TE: torch.Tensor,
    top_k: int,
    route_norm: bool,
    eps: float,
    scale: float,
    padding_mask_T: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(scores ``(T, K)`` fp32, expert ids ``(T, K)`` int64, tokens per expert ``(E,)`` int64)."""
    tokens, experts = gate_TE.shape
    assert gate_TE.stride(1) == 1
    return _topk(
        gate_TE,
        tokens,
        experts,
        1,
        gate_TE.stride(0),
        0,
        top_k,
        route_norm,
        eps,
        scale,
        padding_mask=padding_mask_T,
    )


def gate_split_k(tokens: int, experts: int, dim: int) -> int | None:
    """Split-K degree for ``gate_softmax_topk``, or None where cuBLAS's GEMM should run."""
    if tokens > MAX_GATE_TOKENS or dim % _GATE_BLOCK_K:
        return None
    programs = triton.cdiv(experts, _GATE_BLOCK_E) * triton.cdiv(
        tokens, _gate_block_t(tokens)
    )
    split = 1
    # About one program per SM (A100: 108).
    while (
        split < 16
        and programs * split * 2 <= 128
        and dim % (split * 2 * _GATE_BLOCK_K) == 0
    ):
        split *= 2
    return split


def _gate_block_t(tokens: int) -> int:
    return 16 if tokens <= 16 else 32 if tokens <= 32 else 64


def gate_softmax_topk(
    x_TD: torch.Tensor,
    weight_ED: torch.Tensor,
    top_k: int,
    route_norm: bool,
    eps: float,
    scale: float,
    padding_mask_T: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``softmax_topk`` of the bf16 router gate GEMM ``x_TD @ weight_ED.T`` (fp32 accumulate),
    without the fp32 logits: a split-K GEMM writes partial sums that the top-k kernel adds.

    At decode sizes cuBLAS runs this GEMM as a GEMM plus a split-K reduce (16 us for
    32 x 2048 x 256 on A100) and the logits round trip to memory; here it is two kernels.
    """
    tokens, dim = x_TD.shape
    experts = weight_ED.shape[0]
    split = gate_split_k(tokens, experts, dim)
    assert split is not None and x_TD.stride(1) == 1 and weight_ED.stride(1) == 1
    partials = torch.empty(
        split, tokens, experts, device=x_TD.device, dtype=torch.float32
    )
    counts = torch.empty(experts, device=x_TD.device, dtype=torch.int64)
    block_t = _gate_block_t(tokens)
    if not tokens:
        counts.zero_()
    else:
        _gate_partials_kernel[
            (triton.cdiv(experts, _GATE_BLOCK_E), split, triton.cdiv(tokens, block_t))
        ](
            x_TD,
            weight_ED,
            partials,
            counts,
            tokens,
            experts,
            x_TD.stride(0),
            weight_ED.stride(0),
            K_PER_SPLIT=dim // split,
            BLOCK_T=block_t,
            BLOCK_E=_GATE_BLOCK_E,
            BLOCK_K=_GATE_BLOCK_K,
            BLOCK_COUNTS=triton.next_power_of_2(experts),
            num_warps=4,
        )
    return _topk(
        partials,
        tokens,
        experts,
        split,
        experts,
        tokens * experts,
        top_k,
        route_norm,
        eps,
        scale,
        counts=counts,
        padding_mask=padding_mask_T,
    )


@triton.jit
def _local_key(
    expert_ids,
    i,
    N,
    first_local_expert,
    stride_ids_row,
    K: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
):
    """Local expert of pick i; other ranks' picks get NUM_LOCAL, rows past N get NUM_LOCAL + 1."""
    m = i < N
    ids = tl.load(expert_ids + (i // K) * stride_ids_row + i % K, mask=m, other=0)
    local = ids - first_local_expert
    local_ok = (local >= 0) & (local < NUM_LOCAL)
    key = tl.where(local_ok, local, NUM_LOCAL)
    return tl.where(m, key, NUM_LOCAL + 1).to(tl.int32), local_ok


@triton.jit
def _dispatch_prep_kernel(
    expert_ids,
    src_row,
    sorted_positions,
    counts_out,
    offsets_out,
    is_local_out,
    N,
    first_local_expert,
    stride_ids_row,
    K: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Stable counting sort: pick i lands after every pick with a smaller key and every
    # earlier pick with the same key. Each program ranks BLOCK picks against all N.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m_i = i < N
    key_i, local_i = _local_key(
        expert_ids, i, N, first_local_expert, stride_ids_row, K, NUM_LOCAL
    )
    position = tl.zeros([BLOCK], dtype=tl.int32)
    for j0 in range(0, N, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        key_j, _ = _local_key(
            expert_ids, j, N, first_local_expert, stride_ids_row, K, NUM_LOCAL
        )
        before = (key_j[None, :] < key_i[:, None]) | (
            (key_j[None, :] == key_i[:, None]) & (j[None, :] < i[:, None])
        )
        position += tl.sum(before.to(tl.int32), 1)
    tl.store(is_local_out + i, local_i, mask=m_i)
    tl.store(sorted_positions + i, position.to(tl.int64), mask=m_i)
    tl.store(src_row + position, (i // K).to(tl.int64), mask=m_i)
    if tl.program_id(0) == 0:
        hist = tl.zeros([BLOCK_BINS], dtype=tl.int32)
        for j0 in range(0, N, BLOCK):
            key_j, _ = _local_key(
                expert_ids,
                j0 + tl.arange(0, BLOCK),
                N,
                first_local_expert,
                stride_ids_row,
                K,
                NUM_LOCAL,
            )
            hist += tl.histogram(key_j, BLOCK_BINS)
        bins = tl.arange(0, BLOCK_BINS)
        tl.store(counts_out + bins, hist.to(tl.int64), mask=bins < NUM_LOCAL)
        tl.store(offsets_out + bins, tl.cumsum(hist, 0), mask=bins < NUM_LOCAL)


def dispatch_prep(
    gathered_expert_ids_GK: torch.Tensor, ep_rank: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counting-sort the gathered picks by local expert (others' picks last, stable).

    Returns ``src_row_N`` (gathered token of each sorted row), ``sorted_positions_GK``,
    ``num_tokens_per_local_expert_e`` (int64), ``is_local_GK`` and ``offsets_e``, the int32
    cumulative counts the grouped GEMMs take.
    """
    gathered, top_k = gathered_expert_ids_GK.shape
    rows = gathered * top_k
    assert gathered_expert_ids_GK.stride(1) == 1
    device = gathered_expert_ids_GK.device
    src_row = torch.empty(rows, device=device, dtype=torch.int64)
    sorted_positions = torch.empty(gathered, top_k, device=device, dtype=torch.int64)
    counts = torch.empty(num_local_experts, device=device, dtype=torch.int64)
    offsets = torch.empty(num_local_experts, device=device, dtype=torch.int32)
    is_local = torch.empty(gathered, top_k, device=device, dtype=torch.bool)
    block_bins = triton.next_power_of_2(num_local_experts + 2)
    block = 64
    _dispatch_prep_kernel[(triton.cdiv(rows, block),)](
        gathered_expert_ids_GK,
        src_row,
        sorted_positions,
        counts,
        offsets,
        is_local,
        rows,
        ep_rank * num_local_experts,
        gathered_expert_ids_GK.stride(0),
        K=top_k,
        NUM_LOCAL=num_local_experts,
        BLOCK_BINS=block_bins,
        BLOCK=block,
        num_warps=4,
    )
    return src_row, sorted_positions, counts, is_local, offsets
