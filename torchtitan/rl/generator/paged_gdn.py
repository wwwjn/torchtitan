# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Engine-agnostic Gated DeltaNet step on a paged recurrent-state cache.

Shape legend: T = tokens in the step, C = local convolution channels,
H = local value heads, V = value head dim, K = key head dim, W = convolution width,
N = sequences in the step, S = cache slots.

The vLLM and SGLang generator engines both run TorchTitan's GDN layers through this
function on their own state pools, so the two engines execute the same kernels in
the same order. Batch-invariant execution has two additional requirements:

* The accumulated SSM cache state uses float32. Decode otherwise rounds the state
  through bfloat16 after every token, unlike a single prefill call. The convolution
  cache stays in model dtype because it only stores trailing input columns.
* Batch-invariant recurrence uses the same Attention Gym scan as the trainer.
* Batch-invariant packed convolution reads no history from the cache. Attention
  Gym's paged convolution sums an output whose taps reach into the cached history
  in a different order than an output whose taps are all in the step, which is how
  the trainer (one pass over the sequence) and the decode kernel sum it. The two
  orders can round differently, so a chunked prefill's first ``W - 1`` tokens, or a
  decode token in a step that also prefills, would depend on the schedule.
"""

import torch

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.attention.linear_attention import chunk_kernel_options
from torchtitan.models.common.attn_gym import (
    causal_conv1d_decode,
    gate_transform,
    l2norm,
    paged_causal_conv1d,
    paged_chunk_gdn,
    recurrent_gdn,
    recurrent_gdn_decode,
)


def paged_gdn_forward(
    mixed_qkv_TC: torch.Tensor,
    a_TH: torch.Tensor,
    b_TH: torch.Tensor,
    conv_weight_CW: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_H: torch.Tensor,
    output_THV: torch.Tensor,
    *,
    conv_state_SWC: torch.Tensor,
    ssm_state_SHVK: torch.Tensor,
    state_indices_N: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state_N: torch.Tensor,
    single_token: bool,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    """Run convolution and recurrence for one step, advancing the state pools in place.

    Args:
        mixed_qkv_TC: Local fused ``[query, key, value]`` channels.
        a_TH: Decay gate input.
        b_TH: Update gate input.
        conv_weight_CW: Depthwise convolution weights (no bias).
        A_log_H: Log decay rate.
        dt_bias_H: Decay time-step bias.
        output_THV: Output rows, written in place; rows past ``num_tokens`` stay.
        conv_state_SWC: Convolution history pool ``[S, W - 1, C]``.
        ssm_state_SHVK: Float32 recurrent state pool ``[S, H, V, K]``.
        state_indices_N: Int32 slot of each sequence; non-positive slots are null
            (zero output, no state write).
        cu_seqlens: Int32 packed sequence boundaries.
        has_initial_state_N: Whether each sequence resumes its slot's state.
        single_token: Every sequence processes one token (decode-style path).
        num_tokens: Real rows of the step.
        num_k_heads: Local key heads.
        num_v_heads: Local value heads.
        head_k_dim: Key head dim.
        head_v_dim: Value head dim.
    """
    if num_tokens == 0:
        return
    if single_token:
        num_decode_rows = state_indices_N.numel()
        conv_output = causal_conv1d_decode(
            mixed_qkv_TC[:num_decode_rows],
            conv_weight_CW,
            conv_state_SWC,
            activation="silu",
            state_indices=state_indices_N,
            has_initial_state=has_initial_state_N,
        )
        if not is_in_batch_invariant_mode():
            recurrent_gdn_decode(
                conv_output,
                a_TH[:num_decode_rows].unsqueeze(0),
                b_TH[:num_decode_rows].unsqueeze(0),
                A_log_H.float(),
                dt_bias_H.float(),
                ssm_state_SHVK,
                state_indices_N,
                has_initial_state=has_initial_state_N,
                scale=head_k_dim**-0.5,
                out=output_THV[:num_decode_rows].unsqueeze(0),
            )
        else:
            _recurrence(
                conv_output,
                a_TH[:num_decode_rows],
                b_TH[:num_decode_rows],
                A_log_H,
                dt_bias_H,
                output_THV[:num_decode_rows],
                ssm_state_SHVK=ssm_state_SHVK,
                cu_seqlens=cu_seqlens,
                state_indices_N=state_indices_N,
                has_initial_state_N=has_initial_state_N,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
            )
        return

    if is_in_batch_invariant_mode():
        conv_output = _conv_with_inline_history(
            mixed_qkv_TC[:num_tokens],
            conv_weight_CW,
            conv_state_SWC,
            state_indices_N,
            cu_seqlens,
            has_initial_state_N,
        )
    else:
        conv_output = paged_causal_conv1d(
            mixed_qkv_TC[:num_tokens].unsqueeze(0),
            conv_weight_CW,
            conv_state_SWC,
            state_indices_N,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state_N,
        ).squeeze(0)
    _recurrence(
        conv_output,
        a_TH[:num_tokens],
        b_TH[:num_tokens],
        A_log_H,
        dt_bias_H,
        output_THV[:num_tokens],
        ssm_state_SHVK=ssm_state_SHVK,
        cu_seqlens=cu_seqlens,
        state_indices_N=state_indices_N,
        has_initial_state_N=has_initial_state_N,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )


def _conv_with_inline_history(
    x_TC: torch.Tensor,
    conv_weight_CW: torch.Tensor,
    conv_state_SWC: torch.Tensor,
    state_indices_N: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state_N: torch.Tensor,
) -> torch.Tensor:
    """Packed convolution whose outputs sum every tap from the step's input rows.

    Each sequence's ``W - 1`` history rows (its cached history if it resumes, else
    zeros, which add exactly nothing) are placed in front of its rows, and the
    sequences run as fresh ones. The kernel then writes each slot's new history from
    the extended rows, and the history rows' own outputs are dropped. All shapes
    are static, so this is CUDA-graph safe.
    """
    num_tokens = x_TC.shape[0]
    num_seqs = state_indices_N.numel()
    history = conv_state_SWC.shape[1]
    device = x_TC.device
    # Sequence of each token, and each token's row in the extended input.
    tokens_T = torch.arange(num_tokens, device=device, dtype=torch.int32)
    seq_T = torch.searchsorted(cu_seqlens[1 : num_seqs + 1], tokens_T, right=True)
    rows_T = tokens_T + (seq_T + 1) * history
    # History rows of sequence i occupy cu_seqlens[i] + i * history + [0, history).
    seqs_N = torch.arange(num_seqs, device=device, dtype=torch.int32)
    history_rows_NW = (cu_seqlens[:num_seqs] + seqs_N * history).unsqueeze(1) + (
        torch.arange(history, device=device, dtype=torch.int32)
    )
    # A fresh slot may hold NaNs, so select rather than multiply by zero.
    resumes_N = has_initial_state_N & (state_indices_N > 0)
    history_NWC = torch.where(
        resumes_N.view(-1, 1, 1),
        conv_state_SWC[state_indices_N.clamp(min=0).long()],
        0,
    )
    x_extended_TC = x_TC.new_empty(num_tokens + num_seqs * history, x_TC.shape[1])
    x_extended_TC[rows_T.long()] = x_TC
    x_extended_TC[history_rows_NW.reshape(-1).long()] = history_NWC.reshape(
        -1, x_TC.shape[1]
    )
    output_TC = paged_causal_conv1d(
        x_extended_TC.unsqueeze(0),
        conv_weight_CW,
        conv_state_SWC,
        state_indices_N,
        activation="silu",
        cu_seqlens=cu_seqlens
        + torch.arange(num_seqs + 1, device=device, dtype=cu_seqlens.dtype) * history,
        has_initial_state=torch.zeros_like(has_initial_state_N),
    ).squeeze(0)
    return output_TC[rows_T.long()]


def _recurrence(
    conv_output_TC: torch.Tensor,
    a_TH: torch.Tensor,
    b_TH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_H: torch.Tensor,
    output_THV: torch.Tensor,
    *,
    ssm_state_SHVK: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices_N: torch.Tensor,
    has_initial_state_N: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    """Map each ``cu_seqlens`` interval to its paged SSM slot.

    Null slots skip state writes and produce zero output for padding.
    """
    num_tokens = conv_output_TC.shape[0]
    key_dim = num_k_heads * head_k_dim
    query = (
        conv_output_TC[:, :key_dim]
        .contiguous()
        .view(1, num_tokens, num_k_heads, head_k_dim)
    )
    key = (
        conv_output_TC[:, key_dim : 2 * key_dim]
        .contiguous()
        .view(1, num_tokens, num_k_heads, head_k_dim)
    )
    value = (
        conv_output_TC[:, 2 * key_dim :]
        .contiguous()
        .view(1, num_tokens, num_v_heads, head_v_dim)
    )
    # Keep the existing FP32 arithmetic while reusing the GDN gate contract.
    decay = gate_transform(
        a_TH.unsqueeze(0),
        A_log_H.float(),
        dt_bias_H.float(),
        kind="softplus",
        impl="reference",
    )
    update_gate = torch.sigmoid(b_TH).unsqueeze(0)
    query = l2norm(query, cu_seqlens=cu_seqlens)
    key = l2norm(key, cu_seqlens=cu_seqlens)

    if is_in_batch_invariant_mode():
        recurrent_output, _ = recurrent_gdn(
            query,
            key,
            value,
            decay,
            update_gate,
            ssm_state_SHVK,
            cu_seqlens=cu_seqlens,
            scale=head_k_dim**-0.5,
            state_indices=state_indices_N,
            has_initial_state=has_initial_state_N,
            # Triton autotuning breaks batch invariance.
            autotune=False,
        )
    else:
        recurrent_output = paged_chunk_gdn(
            query,
            key,
            value,
            decay,
            update_gate,
            ssm_state_SHVK,
            state_indices_N,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state_N,
            scale=head_k_dim**-0.5,
            kernel_options=chunk_kernel_options(query),
        )
    output_THV.copy_(recurrent_output[0].to(output_THV.dtype))
