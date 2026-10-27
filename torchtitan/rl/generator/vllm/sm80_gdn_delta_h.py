# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GDN inter-chunk state recurrence for SM80, a drop-in for Attention Gym's launch.

Attention Gym keeps the whole [128, BV] fp32 state in one accumulator and reads it
through tensor descriptors, which SM80 lowers to plain loads; even at 8 warps it takes
1.5 ms per layer for one 8192-token Qwen3.5-27B TP2 prefill on A100, vs 0.30 ms for
FLA's kernel in native vLLM. This kernel follows FLA's layout (the state split into two
[64, BV] halves, plain pointers) with Attention Gym's contract: gate-scaled keys, one
cumulative log2 decay per head and chunk, paged state slots read and written in place,
null slots zero their chunk states and new values.

Outputs and final states matched Attention Gym's kernel bitwise on A100 (1 x 8192,
8 x 1024, 64 x 64, and ragged lengths with null slots and fresh states), at 0.41 vs
1.5 ms (1 x 8192). Scalar-gate GDN with K = V = 128 only; everything else goes to
Attention Gym's launch.
Opt-in with ``TORCHTITAN_SM80_GDN_DELTA_H=1``.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _delta_h_sm80_kernel(
    k,
    w,
    u,
    gk,
    h,
    v_new,
    h0,
    ht,
    state_indices,
    has_initial_state,
    cu_seqlens,
    chunk_offsets,
    T,
    h0_s0,
    h0_s1,
    h0_s2,
    h0_s3,
    ht_s0,
    ht_s1,
    ht_s2,
    ht_s3,
    H: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
):
    K: tl.constexpr = 128
    V: tl.constexpr = 128
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n.to(tl.int64) * T
        boh = i_n.to(tl.int64) * tl.cdiv(T, BT)
    NT = tl.cdiv(T, BT)

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, 64)
    o_bt = tl.arange(0, BT)
    # h is [B * capacity, H, K, V]; tokens are [B * T, H, *].
    p_h = h + (boh * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    stride_h = H * K * V

    slot = i_n.to(tl.int64)
    active = True
    load_initial = True
    if USE_STATE_INDICES:
        route = tl.load(state_indices + i_n).to(tl.int64)
        active = route > 0
        slot = tl.where(active, route, 0)
        load_initial = active
        if USE_HAS_INITIAL_STATE:
            load_initial = active & (tl.load(has_initial_state + i_n) != 0)

    if not active:
        for i_t in range(NT):
            tl.store(p_h + i_t * stride_h, tl.zeros([64, BV], dtype=h.dtype.element_ty))
            tl.store(
                p_h + i_t * stride_h + 64 * V,
                tl.zeros([64, BV], dtype=h.dtype.element_ty),
            )
            o_t = bos + i_t * BT + o_bt
            tl.store(
                v_new + (o_t[:, None] * H + i_h) * V + o_v[None, :],
                tl.zeros([BT, BV], dtype=v_new.dtype.element_ty),
                mask=(o_t < bos + T)[:, None],
            )
        return

    # State [K, V] as two [64, BV] halves (rows 0-63 and 64-127 of K).
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # State pool layout [slot, H, V, K].
        p_h0 = (
            h0
            + slot * h0_s0
            + i_h * h0_s1
            + o_v[None, :] * h0_s2
            + o_k[:, None] * h0_s3
        )
        b_h1 += tl.load(p_h0, mask=load_initial, other=0.0).to(tl.float32)
        b_h2 += tl.load(p_h0 + 64 * h0_s3, mask=load_initial, other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = bos + i_t * BT + o_bt
        m_t = o_t < bos + T
        tl.store(p_h + i_t * stride_h, b_h1.to(h.dtype.element_ty))
        tl.store(p_h + i_t * stride_h + 64 * V, b_h2.to(h.dtype.element_ty))

        p_w = w + (o_t[:, None] * H + i_h) * K + o_k[None, :]
        b_w1 = tl.load(p_w, mask=m_t[:, None], other=0.0)
        b_w2 = tl.load(p_w + 64, mask=m_t[:, None], other=0.0)
        b_v = tl.dot(b_w1, b_h1.to(b_w1.dtype))
        b_v = tl.dot(b_w2, b_h2.to(b_w2.dtype), b_v)
        p_u = u + (o_t[:, None] * H + i_h) * V + o_v[None, :]
        b_v = tl.load(p_u, mask=m_t[:, None], other=0.0).to(tl.float32) - b_v
        b_v = b_v.to(v_new.dtype.element_ty)
        tl.store(
            v_new + (o_t[:, None] * H + i_h) * V + o_v[None, :], b_v, mask=m_t[:, None]
        )

        last = tl.minimum(bos + (i_t + 1) * BT, bos + T) - 1
        decay = tl.math.exp2(tl.load(gk + last * H + i_h).to(tl.float32))
        b_h1 *= decay
        b_h2 *= decay

        # Keys transposed: [64, BT].
        p_k = k + (o_t[None, :] * H + i_h) * K + o_k[:, None]
        b_k1 = tl.load(p_k, mask=m_t[None, :], other=0.0)
        b_k2 = tl.load(p_k + 64, mask=m_t[None, :], other=0.0)
        b_h1 = tl.dot(b_k1, b_v, b_h1)
        b_h2 = tl.dot(b_k2, b_v, b_h2)

    if STORE_FINAL_STATE:
        p_ht = (
            ht
            + slot * ht_s0
            + i_h * ht_s1
            + o_v[None, :] * ht_s2
            + o_k[:, None] * ht_s3
        )
        tl.store(p_ht, b_h1.to(ht.dtype.element_ty))
        tl.store(p_ht + 64 * ht_s3, b_h2.to(ht.dtype.element_ty))


def _strides(state: torch.Tensor | None) -> tuple[int, int, int, int]:
    return tuple(state.stride()) if state is not None else (0, 0, 0, 0)


def make_delta_h_launch(original):
    """Wrap Attention Gym's ``_delta_h_launch`` with the SM80 kernel for scalar-gate K = V = 128."""

    def _delta_h_launch(
        k,
        w,
        u,
        gk,
        initial_state,
        state_indices,
        has_initial_state,
        cu_seqlens,
        chunk_offsets,
        capacity,
        final_state,
        schedule=None,
        fastmath=True,
    ):
        batch, tokens, heads, key_dim = k.shape
        value_dim = u.shape[-1]
        if not (
            gk.ndim == 3
            and key_dim == value_dim == 128
            and fastmath
            and k.dtype in (torch.bfloat16, torch.float16)
        ):
            kwargs = {"fastmath": fastmath}
            if schedule is not None:
                kwargs["schedule"] = schedule
            return original(
                k,
                w,
                u,
                gk,
                initial_state,
                state_indices,
                has_initial_state,
                cu_seqlens,
                chunk_offsets,
                capacity,
                final_state,
                **kwargs,
            )
        k, w, u, gk = k.contiguous(), w.contiguous(), u.contiguous(), gk.contiguous()
        h = k.new_empty(batch, capacity, heads, key_dim, value_dim)
        v_new = torch.empty_like(u)
        state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
        # A100 sweep over BV 32 / 64, 2-8 warps, 1-3 stages: best across 1 x 8192, 8 x 1024, 64 x 64.
        block_v = 64
        _delta_h_sm80_kernel[(state_batch * heads, value_dim // block_v)](
            k,
            w,
            u,
            gk,
            h,
            v_new,
            initial_state,
            final_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
            chunk_offsets,
            tokens,
            *_strides(initial_state),
            *_strides(final_state),
            H=heads,
            BT=64,
            BV=block_v,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            IS_VARLEN=cu_seqlens is not None,
            USE_STATE_INDICES=state_indices is not None,
            USE_HAS_INITIAL_STATE=has_initial_state is not None,
            num_warps=4,
            num_stages=2,
        )
        return h, v_new

    _delta_h_launch.sm80 = True
    return _delta_h_launch


def maybe_use_sm80_delta_h() -> bool:
    """Route Attention Gym's GDN state recurrence through the SM80 kernel (opt-in)."""
    if os.environ.get("TORCHTITAN_SM80_GDN_DELTA_H") != "1":
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0):
        return False
    import attn_gym.linear.kda.fwd.triton.chunk_delta_h as delta_h

    if not getattr(delta_h._delta_h_launch, "sm80", False):
        delta_h._delta_h_launch = make_delta_h_launch(delta_h._delta_h_launch)
    return True
