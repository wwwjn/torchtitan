# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped GEMM for MoE experts before Hopper, without a host sync.

``torch._grouped_mm`` has no SM80 kernel: its fallback copies the group offsets to
the host and loops over experts. That syncs the host once per MoE layer and cannot
be captured in a CUDA graph. This Triton kernel reads the offsets on the device:
a small kernel turns them into a list of (expert, first row) tiles of a static
maximum length, and tiles past the live count exit at once. A layer's w13 and w2
GEMMs share the offsets, so the second reuses the first's tile list.

Opt-in for the generator with ``TORCHTITAN_SM80_GROUPED_MM=1``.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _tile_schedule_kernel(
    offsets,
    tile_expert,
    tile_row,
    num_tiles,
    groups,
    max_tiles,
    BM: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Tile t belongs to the expert whose tile range [first, end) holds t. Each program
    rebuilds the per-expert tile counts (one load of the offsets) and fills BLOCK_T tiles."""
    e = tl.arange(0, BLOCK_E)
    e_mask = e < groups
    ends = tl.load(offsets + e, mask=e_mask, other=0)
    starts = tl.load(offsets + e - 1, mask=e_mask & (e > 0), other=0)
    tiles = tl.where(e_mask, (ends - starts + BM - 1) // BM, 0)
    tile_end = tl.cumsum(tiles, 0)
    if tl.program_id(0) == 0:
        tl.store(num_tiles, tl.sum(tiles, 0))
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    # Experts whose tiles all end at or before t; padding experts end at the total.
    done = (tl.where(e_mask, tile_end, 2147483647)[None, :] <= t[:, None]).to(tl.int32)
    expert = tl.minimum(tl.sum(done, 1), groups - 1)
    first = tl.sum(
        tl.where(e[None, :] == expert[:, None], (tile_end - tiles)[None, :], 0), 1
    )
    start = tl.sum(tl.where(e[None, :] == expert[:, None], starts[None, :], 0), 1)
    t_mask = t < max_tiles
    tl.store(tile_expert + t, expert, mask=t_mask)
    tl.store(tile_row + t, start + (t - first) * BM, mask=t_mask)


@triton.jit
def _grouped_mm_kernel(
    x,
    w,
    out,
    offsets,
    tile_expert,
    tile_row,
    num_tiles,
    N,
    K,
    stride_xm,
    stride_we,
    stride_wn,
    stride_om,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= tl.load(num_tiles):
        return
    pid_n = tl.program_id(1)
    expert = tl.load(tile_expert + pid_m).to(tl.int64)
    row0 = tl.load(tile_row + pid_m)
    row_end = tl.load(offsets + expert)
    rows = row0 + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    m_mask = rows < row_end
    n_mask = cols < N
    x_ptrs = x + rows[:, None].to(tl.int64) * stride_xm + ks[None, :]
    w_ptrs = (
        w + expert * stride_we + cols[None, :].to(tl.int64) * stride_wn + ks[:, None]
    )
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_mask = ks < K - k * BK
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BK
        w_ptrs += BK
    o_ptrs = out + rows[:, None].to(tl.int64) * stride_om + cols[None, :]
    tl.store(
        o_ptrs, acc.to(out.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _block_m(rows: int, groups: int) -> int:
    rows_per_group = rows / max(groups, 1)
    if rows_per_group <= 16:
        return 16
    if rows_per_group <= 24:
        return 32
    return 64


_last_schedule: tuple | None = None


def _tile_schedule(offsets_E: torch.Tensor, rows: int, groups: int, bm: int):
    """(tile_expert, tile_row, num_tiles, max_tiles), reused once for the same offsets."""
    global _last_schedule
    if _last_schedule is not None:
        cached_offsets, cached_key, schedule = _last_schedule
        _last_schedule = None
        if cached_offsets is offsets_E and cached_key == (rows, groups, bm):
            return schedule
    # Each expert pads at most one partial tile: a static tile budget for any routing.
    max_tiles = triton.cdiv(rows, bm) + groups
    tile_expert = torch.empty(max_tiles, device=offsets_E.device, dtype=torch.int32)
    tile_row = torch.empty_like(tile_expert)
    num_tiles = torch.empty(1, device=offsets_E.device, dtype=torch.int32)
    block_t = max(16, min(64, 4096 // triton.next_power_of_2(groups)))
    _tile_schedule_kernel[(triton.cdiv(max_tiles, block_t),)](
        offsets_E.to(torch.int32),
        tile_expert,
        tile_row,
        num_tiles,
        groups,
        max_tiles,
        BM=bm,
        BLOCK_E=triton.next_power_of_2(groups),
        BLOCK_T=block_t,
    )
    schedule = (tile_expert, tile_row, num_tiles, max_tiles)
    # Holding offsets_E keeps its storage from being reused by the next layer's offsets.
    _last_schedule = (offsets_E, (rows, groups, bm), schedule)
    return schedule


def grouped_mm(
    x_RI: torch.Tensor, weight_EOI: torch.Tensor, offsets_E: torch.Tensor
) -> torch.Tensor:
    """``x_RI @ weight_EOI[e].T`` for the rows of each expert ``e``.

    Rows past ``offsets_E[-1]`` are left unwritten, as ``torch._grouped_mm`` leaves them.
    """
    rows, in_features = x_RI.shape
    groups, out_features, _ = weight_EOI.shape
    x_RI = x_RI.contiguous()
    weight_EOI = weight_EOI.contiguous()
    out_RO = x_RI.new_empty(rows, out_features)
    if rows == 0:
        return out_RO
    bm = _block_m(rows, groups)
    # A100 sweep (35B-A3B EP2 shapes): 64 x 256 tiles beat 128 x 128 by 23% at 32K rows.
    bn = 256 if bm >= 64 else 64
    tile_expert, tile_row, num_tiles, max_tiles = _tile_schedule(
        offsets_E, rows, groups, bm
    )
    grid = (max_tiles, triton.cdiv(out_features, bn))
    _grouped_mm_kernel[grid](
        x_RI,
        weight_EOI,
        out_RO,
        offsets_E.to(torch.int32),
        tile_expert,
        tile_row,
        num_tiles,
        out_features,
        in_features,
        x_RI.stride(0),
        weight_EOI.stride(0),
        weight_EOI.stride(1),
        out_RO.stride(0),
        BM=bm,
        BN=bn,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    return out_RO


def maybe_use_sm80_grouped_mm() -> None:
    """Route ``GroupedLinear`` through :func:`grouped_mm` before Hopper (opt-in)."""
    if os.environ.get("TORCHTITAN_SM80_GROUPED_MM") != "1":
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0):
        return
    from torchtitan.models.common.linear.linear import GroupedLinear

    def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
        return grouped_mm(input_RI, weight_EOI.bfloat16(), offsets_E)

    GroupedLinear._grouped_mm = _grouped_mm
