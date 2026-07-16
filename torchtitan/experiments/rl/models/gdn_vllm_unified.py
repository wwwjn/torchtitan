# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified-model GDN generation core: TorchTitan's own GatedDeltaNet running
inside vLLM under the ``torchtitan_wrapper`` path, borrowing ONLY vLLM's paged
conv+ssm state cache management.

Contrast with ``gdn_vllm_titan.py`` (the ``vllm_native`` path), which keeps
vLLM's whole native GDN model and swaps just the recurrence kernel. Here every
non-GDN layer already runs TorchTitan code (via the wrapper); this module makes
the GDN layer unified too: its parameters stay TorchTitan's (they live in
``qwen3_5.model.GatedDeltaNet``), while the stateful conv + recurrence use
vLLM's native GDN helper kernels against vLLM's paged cache so continuous-batch
generation works.

Mechanism: ``VLLMGatedDeltaNetCore`` is a parameter-less ``MambaBase`` layer.
vLLM's hybrid KV-cache discovery (``get_kv_cache_spec`` -> ``MambaSpec``) sees it
via ``static_forward_context`` and allocates the paged conv_state + ssm_state.
``GatedDeltaNet`` computes its projections and gates, then delegates the conv +
recurrence to this core (drop-in for ``_causal_conv`` + ``GatedDeltaKernel``).
The 3 depthwise convs (conv_q/k/v) fuse channel-wise into the one fused conv
vLLM's causal_conv1d kernels expect -- depthwise, so identical math.

Legend (tensor shape suffixes, this module):
  T  = num actual tokens in the flattened batch (all requests concatenated)
  Ck = key conv/proj channels = num_k_heads * head_k_dim
  Cv = value channels        = num_v_heads * head_v_dim
  Hk = num key heads, Hv = num value heads, Dk = head_k_dim, Dv = head_v_dim
"""

from __future__ import annotations

import os

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# External fla (the SAME package the trainer uses). Opt-in via
# TT_GDN_WRAPPER_EXTERNAL_FLA=1 so the wrapper's GDN prefill runs the trainer's
# exact chunk + conv kernels (the vendored vLLM fla is a different copy -> a parity
# gap). We align the GENERATOR to the TRAINER because only the trainer's fla path
# has a backward (training needs it), so the trainer's kernels are the reference.
from fla.modules.convolution import (
    causal_conv1d as _external_fla_causal_conv1d,
    causal_conv1d_update as _external_fla_causal_conv1d_update,
)
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _external_fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _external_fla_fused_recurrent_gated_delta_rule,
)

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as _vllm_chunk_gated_delta_rule,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class VLLMGatedDeltaNetCore(Module, MambaBase):
    """Paged-cache GDN core for the unified (torchtitan_wrapper) path.

    Holds NO learnable parameters -- projections, conv weights, gates, norm and
    out_proj all live in the enclosing ``GatedDeltaNet``. This layer only owns
    the vLLM cache plumbing (state discovery + paged conv/ssm state) and runs
    vLLM's native GDN helper kernels against it. Decode uses the recurrent update
    path; prefill uses the varlen chunk path, matching vLLM native's split.

    Non-speculative decoding only (asserts otherwise). Requires the generator
    cudagraph off until this custom core is captured and validated separately.

    TODO(qwen3.5-gdn-unified-prefix-cache): validate vLLM align-mode prefix
    caching before enabling prefix cache for this unified GDN path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int = 4
        activation: str = "silu"

    def __init__(self, config: Config) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_spec = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )

        # TODO(qwen3.5-gdn-unified-tp): support TP by passing local head counts
        # and using the local fused conv/projection width. Current unified GDN
        # generation is validated only for pure DP / TP=1.
        if self.tp_size != 1:
            raise ValueError(
                "VLLMGatedDeltaNetCore currently supports tensor_parallel_size=1 "
                f"only, got tensor_parallel_size={self.tp_size}."
            )

        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.activation = config.activation

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        _, ssm_dtype = self.get_state_dtype()
        if ssm_dtype != torch.float32:
            raise ValueError(
                "VLLMGatedDeltaNetCore requires mamba_ssm_cache_dtype='float32' "
                f"for the triton/FLA recurrent state, got {ssm_dtype}."
            )

        # vLLM populates this via the KV-cache allocator: (conv_state, ssm_state).
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        # Side conv-state buffer for the recurrent-everywhere path (fla's conv state
        # is [num_slots, conv_dim, W] with W = kernel size, wider than vLLM's paged
        # conv_state [.., k-1], so it cannot reuse kv_cache[0]). Lazily allocated.
        self._fla_conv_state: torch.Tensor | None = None

        self.prefix = f"model.layers.{config.layer_idx}.linear_attn"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate GDN layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    # ---- MambaBase contract ------------------------------------------------
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    # ---- recurrent-everywhere BI path (TT_GDN_RECURRENT_BI=1) --------------
    def _split_qkv(
        self, mixed_qkv_slice_TC: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = mixed_qkv_slice_TC.shape[0]
        q = (
            mixed_qkv_slice_TC[:, : self.key_dim]
            .contiguous()
            .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
        )
        k = (
            mixed_qkv_slice_TC[:, self.key_dim : 2 * self.key_dim]
            .contiguous()
            .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
        )
        v = (
            mixed_qkv_slice_TC[:, 2 * self.key_dim :]
            .contiguous()
            .view(1, num_tokens, self.num_v_heads, self.head_v_dim)
        )
        return q, k, v

    def _fla_conv_state_buffer(
        self, conv_dim: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        num_slots = self.kv_cache[0].shape[0]
        w = self.conv_kernel_size
        buf = self._fla_conv_state
        if (
            buf is None
            or buf.shape[0] != num_slots
            or buf.shape[1] != conv_dim
            or buf.device != device
        ):
            buf = torch.zeros(num_slots, conv_dim, w, dtype=dtype, device=device)
            self._fla_conv_state = buf
        return buf

    def _forward_recurrent_bi(
        self,
        m: GDNAttentionMetadata,
        n: int,
        mixed_qkv_TC: torch.Tensor,
        a_THv: torch.Tensor,
        b_THv: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Recurrent-everywhere GDN: fla conv + fla RECURRENT kernel for BOTH prefill
        and decode, so decode == prefill == trainer bitwise.

        Conv uses fla's stateful pair (causal_conv1d full with output_final_state +
        causal_conv1d_update) against a side conv-state buffer keyed by the vLLM slot
        ids. Recurrence uses fla fused_recurrent against the paged ssm_state via
        gather/scatter (upstream fla recurrent has no slot indexing). Fresh prefill
        only (initial_state=None), matching the RL rollout; prefix caching is not
        supported here.
        """
        conv_dim = mixed_qkv_TC.shape[1]
        ssm_state = self.kv_cache[1]
        nsi = m.non_spec_state_indices_tensor
        assert nsi is not None
        conv_state = self._fla_conv_state_buffer(
            conv_dim, mixed_qkv_TC.dtype, mixed_qkv_TC.device
        )
        out_1THvDv = mixed_qkv_TC.new_empty(1, n, self.num_v_heads, self.head_v_dim)

        def _recurrence(
            conv_out_TC: torch.Tensor,
            seg: slice,
            slot_idx: torch.Tensor,
            cu_seqlens: torch.Tensor,
            has_state: bool,
        ) -> torch.Tensor:
            q, k, v = self._split_qkv(conv_out_TC)
            if q.shape[2] != v.shape[2]:
                rep = v.shape[2] // q.shape[2]
                q = q.repeat_interleave(rep, dim=2)
                k = k.repeat_interleave(rep, dim=2)
            # Run the recurrence in fp32: the fla recurrent kernel is NOT bitwise
            # stepped-consistent in bf16 (full-seq prefill vs 1-token decode differ by
            # ~1e-5 from bf16 input rounding), but in fp32 it matches to ~1e-8, which
            # vanishes when the output is written back to the bf16 activation buffer.
            # This is what makes decode == prefill; the trainer upcasts identically.
            q, k, v = q.float(), k.float(), v.float()
            # fp32 eager gate, identical to the trainer (model.py:514-517).
            g = (
                -torch.exp(A_log.float())
                * F.softplus(a_THv[seg].float() + dt_bias.float())
            ).unsqueeze(0)
            beta = torch.sigmoid(b_THv[seg].float()).unsqueeze(0)
            if has_state:
                # paged ssm_state is stored [.., V, K]; fla wants [.., K, V].
                initial_state = ssm_state[slot_idx].transpose(-1, -2).contiguous()
            else:
                # Fresh prefill: pass a materialized ZERO state, NOT None. fla's
                # recurrent kernel compiles USE_INITIAL_STATE from (h0 is not None),
                # so a None prefill vs a tensor decode select two different binaries
                # with divergent fp reductions (~1e-8) -> decode != prefill. A zero
                # init makes both paths the SAME binary -> bitwise resume-exact
                # (decode == prefill).
                n_seq = int(cu_seqlens.numel()) - 1
                initial_state = q.new_zeros(
                    n_seq, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
                )
            out, final_state = _external_fla_fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[slot_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)
            return out

        num_decode_tokens = m.num_decode_tokens

        # Decode segment: 1 token per sequence; resume conv + ssm state from cache.
        if m.num_decodes > 0:
            dec_slots = nsi[: m.num_decodes]
            cache = conv_state[dec_slots]  # [num_decodes, conv_dim, W] (advanced-index copy)
            conv_out, cache = _external_fla_causal_conv1d_update(
                mixed_qkv_TC[:num_decode_tokens],
                cache,
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
            )
            conv_state[dec_slots] = cache
            out_1THvDv[:, :num_decode_tokens] = _recurrence(
                conv_out,
                slice(0, num_decode_tokens),
                dec_slots,
                m.non_spec_query_start_loc[: m.num_decodes + 1],
                has_state=True,
            )

        # Prefill segment: fresh sequences; fla full conv writes conv + ssm state.
        if m.num_prefills > 0:
            assert m.prefill_state_indices is not None
            pf_start = num_decode_tokens if m.num_decodes > 0 else 0
            if m.num_decodes == 0:
                pf_cu = m.non_spec_query_start_loc  # 0-based (verified prefill path)
            else:
                assert m.prefill_query_start_loc is not None
                pf_cu = m.prefill_query_start_loc - m.prefill_query_start_loc[0]
            conv_out, conv_final = _external_fla_causal_conv1d(
                mixed_qkv_TC[pf_start:n].unsqueeze(0),
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
                cu_seqlens=pf_cu,
                output_final_state=True,
            )
            conv_out = conv_out.squeeze(0)
            conv_state[m.prefill_state_indices] = conv_final.to(conv_state.dtype)
            out_1THvDv[:, pf_start:n] = _recurrence(
                conv_out,
                slice(pf_start, n),
                m.prefill_state_indices,
                pf_cu,
                has_state=False,
            )

        return out_1THvDv

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        mixed_qkv_BTC: torch.Tensor,
        a_BTHv: torch.Tensor,
        b_BTHv: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """conv + gated-delta recurrence against vLLM's paged state.

        Args:
            mixed_qkv_BTC: (bs, seqlen, key_dim*2 + value_dim) -- concat of the
                q/k/v projections, PRE-conv (bs==1 under vLLM: one flattened row).
            a_BTHv: (bs, seqlen, num_v_heads) alpha gate input.
            b_BTHv: (bs, seqlen, num_v_heads) beta gate input.
            A_log: (num_v_heads,) decay parameter.
            dt_bias: (num_v_heads,) dt bias parameter.
            conv_weight: fused depthwise conv weight (conv_dim, kernel_size).
            conv_bias: fused conv bias (conv_dim,) or None.

        Returns:
            (bs, seqlen, num_v_heads, head_v_dim) core output (pre gated-norm).
        """
        bs, seqlen, conv_dim = mixed_qkv_BTC.shape
        if bs != 1:
            raise ValueError(
                "VLLMGatedDeltaNetCore expects vLLM's flattened batch layout "
                f"with batch size 1, got batch size {bs}."
            )
        out_BTHvDv = mixed_qkv_BTC.new_zeros(
            bs, seqlen, self.num_v_heads, self.head_v_dim
        )

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        # Dummy run (profiling / warmup): no metadata -> nothing to compute.
        if attn_metadata_raw is None:
            return out_BTHvDv
        assert isinstance(attn_metadata_raw, dict)
        m = attn_metadata_raw[self.prefix]
        assert isinstance(m, GDNAttentionMetadata)
        assert (
            m.spec_sequence_masks is None
        ), "VLLMGatedDeltaNetCore does not support speculative decoding"

        n = m.num_actual_tokens
        if n > seqlen:
            raise ValueError(
                "VLLMGatedDeltaNetCore received more actual tokens than the "
                f"flattened input length: num_actual_tokens={n}, seqlen={seqlen}."
            )
        if n == 0:
            return out_BTHvDv

        # Flatten (bs==1) to the token layout vLLM's kernels use: (T, C).
        mixed_qkv_TC = mixed_qkv_BTC.reshape(bs * seqlen, conv_dim)[:n]
        a_THv = a_BTHv.reshape(bs * seqlen, self.num_v_heads)[:n]
        b_THv = b_BTHv.reshape(bs * seqlen, self.num_v_heads)[:n]

        # conv_state stored (dim, width-1) in DS layout; SD layout needs a
        # transpose so the conv kernels see (..., dim, width-1).
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]
        nsi = m.non_spec_state_indices_tensor  # one state slot per sequence
        assert nsi is not None
        num_seq = m.num_decodes + m.num_prefills
        assert m.non_spec_query_start_loc is not None
        assert m.non_spec_query_start_loc.numel() >= num_seq + 1
        assert nsi.numel() >= num_seq

        # Recurrent-everywhere under batch-invariant mode: fla conv + fla RECURRENT
        # kernel for prefill AND decode so decode == prefill == trainer bitwise.
        # Enabled automatically whenever the generator runs batch-invariant (set via
        # set_batch_invariance in actors/generator.py); the trainer forward is likewise
        # swapped to fla recurrent under BI mode (model.py _RecurrentFwdChunkBwd).
        if is_in_batch_invariant_mode():
            out_BTHvDv[:, :n] = self._forward_recurrent_bi(
                m,
                n,
                mixed_qkv_TC,
                a_THv,
                b_THv,
                A_log=A_log,
                dt_bias=dt_bias,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
            )
            return out_BTHvDv

        # 1) Causal conv against the paged conv_state (reuse vLLM's kernels). A
        # batch with any prefill uses the varlen fn; pure-decode uses the update.
        if (
            os.environ.get("TT_GDN_WRAPPER_EXTERNAL_FLA") == "1"
            and m.num_prefills > 0
            and m.num_decodes == 0
        ):
            # Prefill-parity (PACKED / cu_seqlens path = real training): the trainer's
            # packed GatedDeltaNet conv uses fla causal_conv1d with cu_seqlens
            # (model.py:372-397), resetting at each sample boundary. Match it: run the
            # trainer's EXTERNAL fla causal_conv1d with the per-request cu_seqlens
            # instead of vLLM's paged causal_conv1d_fn. Fused q|k|v depthwise conv ==
            # the trainer's 3 separate depthwise convs (channel-wise identical).
            # fresh-prefill only; does NOT write the paged conv_state (decode needs it).
            conv_out_TC = _external_fla_causal_conv1d(
                mixed_qkv_TC.unsqueeze(0),  # [1, n, C] channels-last (as the trainer)
                weight=conv_weight,  # [C, kw]
                bias=conv_bias,
                activation="silu",
                cu_seqlens=m.non_spec_query_start_loc,
            )
            if isinstance(conv_out_TC, tuple):
                conv_out_TC = conv_out_TC[0]
            conv_out_TC = conv_out_TC.squeeze(0)  # [n, C]
        elif m.num_prefills > 0:
            conv_out_TC = causal_conv1d_fn(
                mixed_qkv_TC.transpose(0, 1),
                conv_weight,
                conv_bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=m.has_initial_state,
                cache_indices=nsi,
                query_start_loc=m.non_spec_query_start_loc,
                metadata=m,
            ).transpose(0, 1)
        else:
            conv_out_TC = causal_conv1d_update(
                mixed_qkv_TC,
                conv_state,
                conv_weight,
                conv_bias,
                self.activation,
                conv_state_indices=nsi[:n],
                validate_data=False,
            )

        # 2) Recurrent attention against paged state. Match vLLM native's split:
        # decode tokens use a recurrent update path, while prefill tokens use the
        # chunk kernel. This avoids routing mixed-batch length-1 decodes through
        # the prefill kernel and keeps the unified path closer to native GDN.
        out_1THvDv = mixed_qkv_TC.new_empty(1, n, self.num_v_heads, self.head_v_dim)

        def _split_qkv(
            mixed_qkv_slice_TC: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = mixed_qkv_slice_TC.shape[0]
            q_1THkDk = (
                mixed_qkv_slice_TC[:, : self.key_dim]
                .contiguous()
                .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
            )
            k_1THkDk = (
                mixed_qkv_slice_TC[:, self.key_dim : 2 * self.key_dim]
                .contiguous()
                .view(1, num_tokens, self.num_k_heads, self.head_k_dim)
            )
            v_1THvDv = (
                mixed_qkv_slice_TC[:, 2 * self.key_dim :]
                .contiguous()
                .view(1, num_tokens, self.num_v_heads, self.head_v_dim)
            )
            return q_1THkDk, k_1THkDk, v_1THvDv

        def _run_decode_sigmoid_update(
            start: int,
            end: int,
            state_idx: torch.Tensor,
            cu_seqlens: torch.Tensor,
        ) -> None:
            if end <= start:
                return
            q, k, v = _split_qkv(conv_out_TC[start:end])
            out, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a_THv[start:end],
                b=b_THv[start:end],
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=state_idx,
                use_qk_l2norm_in_kernel=True,
            )
            out_1THvDv[:, start:end] = out

        def _run_packed_decode(end: int, state_idx: torch.Tensor) -> None:
            if end <= 0:
                return
            out_T1HvDv = out_1THvDv[0, :end].unsqueeze(1)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=conv_out_TC[:end].contiguous(),
                a=a_THv[:end].contiguous(),
                b=b_THv[:end].contiguous(),
                A_log=A_log,
                dt_bias=dt_bias,
                scale=self.head_k_dim**-0.5,
                initial_state=ssm_state,
                out=out_T1HvDv,
                ssm_state_indices=state_idx,
                use_qk_l2norm_in_kernel=True,
            )

        def _run_prefill_chunk(
            start: int,
            end: int,
            state_idx: torch.Tensor,
            has_initial_state: torch.Tensor | None,
            cu_seqlens: torch.Tensor,
        ) -> None:
            if end <= start:
                return
            initial_state = ssm_state[state_idx]
            initial_state = initial_state.clone()
            if has_initial_state is None:
                initial_state.zero_()
            else:
                initial_state[~has_initial_state] = 0
            if os.environ.get("TT_GDN_WRAPPER_EXTERNAL_FLA") == "1":
                # Fully-unified prefill: drop fused_post_conv_prep and reproduce the
                # trainer op-for-op -- eager fp32 gate (model.py:514-517), RAW q/k with
                # in-kernel l2norm, GVA head-expand (model.py:243-247), and the trainer's
                # EXTERNAL fla chunk. So gate + l2norm + recurrence all match the trainer;
                # only the paged conv upstream remains vLLM's. Transpose the paged [V,K]
                # state <-> fla's [K,V] (default transpose_state_layout=False, trainer-match).
                q, k, v = _split_qkv(conv_out_TC[start:end])
                if q.shape[2] != v.shape[2]:
                    rep = v.shape[2] // q.shape[2]
                    q = q.repeat_interleave(rep, dim=2)
                    k = k.repeat_interleave(rep, dim=2)
                g = (
                    -torch.exp(A_log.float())
                    * F.softplus(a_THv[start:end].float() + dt_bias.float())
                ).unsqueeze(0)
                beta = torch.sigmoid(b_THv[start:end].float()).unsqueeze(0)
                # Match the trainer's PACKED path (real training uses cu_seqlens):
                #  - cu_seqlens = the per-request query_start_loc (varlen), so the
                #    chunk resets at sample boundaries exactly like the trainer.
                #  - initial_state=None (stateless), NOT a zero tensor -- fla's
                #    USE_INITIAL_STATE constexpr picks a different reduction for
                #    None vs a real tensor, so a zero tensor is not bitwise-equal.
                # Fresh-prefill only (guarded by num_decodes==0 + assumed fresh).
                # (Recurrent-everywhere mode uses _forward_recurrent_bi instead; this
                # EXTERNAL_FLA-only branch keeps the chunk kernel for prefill-parity.)
                out, final_state = _external_fla_chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=None,
                    output_final_state=True,
                    cu_seqlens=m.non_spec_query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                )
                out_1THvDv[:, start:end] = out
                ssm_state[state_idx] = final_state.transpose(-1, -2).to(ssm_state.dtype)
            else:
                q, k, v, g, beta = fused_post_conv_prep(
                    conv_output=conv_out_TC[start:end],
                    a=a_THv[start:end],
                    b=b_THv[start:end],
                    A_log=A_log,
                    dt_bias=dt_bias,
                    num_k_heads=self.num_k_heads,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=False,
                )
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
                g = g.unsqueeze(0)
                beta = beta.unsqueeze(0)
                out, final_state = _vllm_chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=m.chunk_indices,
                    chunk_offsets=m.chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                )
                out_1THvDv[:, start:end] = out
                ssm_state[state_idx] = final_state.to(ssm_state.dtype)

        num_decode_tokens = m.num_decode_tokens
        if m.num_decodes > 0:
            if m.num_prefills == 0:
                _run_packed_decode(num_decode_tokens, nsi[:num_decode_tokens])
            else:
                _run_decode_sigmoid_update(
                    0,
                    num_decode_tokens,
                    nsi[: m.num_decodes],
                    m.non_spec_query_start_loc[: m.num_decodes + 1],
                )

        if m.num_prefills > 0:
            assert m.prefill_query_start_loc is not None
            assert m.prefill_state_indices is not None
            prefill_start = num_decode_tokens if m.num_decodes > 0 else 0
            _run_prefill_chunk(
                prefill_start,
                n,
                m.prefill_state_indices,
                m.prefill_has_initial_state,
                m.prefill_query_start_loc,
            )

        out_BTHvDv[:, :n] = out_1THvDv[:, :n]
        return out_BTHvDv


def log_unified_gdn_active() -> None:
    logger.info(
        "[gdn-unified] GatedDeltaNet running as a TorchTitan unified layer with "
        "vLLM paged cache management and native GDN helper kernels"
    )
