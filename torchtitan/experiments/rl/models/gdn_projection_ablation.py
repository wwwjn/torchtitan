# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only merged input projections for Qwen3.5."""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torch.distributed.tensor import DTensor, Replicate, Shard


_MERGED_WEIGHT_CACHE: dict[tuple[int, ...], torch.Tensor] = {}
_OFFSET_WEIGHT_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_QK_ROPE_CACHE: dict[int, torch.Tensor] = {}
_GDN_PATCHED = False
_ATTENTION_PATCHED = False
_GDN_NORM_PATCHED = False
_OFFSET_NORM_PATCHED = False
_RESIDUAL_NORM_PATCHED = False
_QK_NORM_ROPE_PATCHED = False


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _merged_weight(weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
    key = tuple(weight.data_ptr() for weight in weights)
    merged = _MERGED_WEIGHT_CACHE.get(key)
    if merged is None:
        merged = torch.cat(weights, dim=0).contiguous()
        _MERGED_WEIGHT_CACHE[key] = merged
    return merged


def _clear_merged_weight_cache(*_args) -> None:
    _MERGED_WEIGHT_CACHE.clear()


def _clear_offset_weight_cache(*_args) -> None:
    _OFFSET_WEIGHT_CACHE.clear()


def _adjusted_offset_weight(
    weight: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (weight.data_ptr(), dtype)
    adjusted_weight = _OFFSET_WEIGHT_CACHE.get(key)
    if adjusted_weight is None:
        adjusted_weight = (weight.float() + 1.0).to(dtype).contiguous()
        _OFFSET_WEIGHT_CACHE[key] = adjusted_weight
    return adjusted_weight


def _qwen35_text_rope_cache(
    cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    key = cache.data_ptr()
    packed_cache = _QK_ROPE_CACHE.get(key)
    if packed_cache is None:
        half_rotary = rotary_dim // 2
        packed_cache = torch.cat(
            (
                cache[:, :half_rotary],
                cache[:, rotary_dim : rotary_dim + half_rotary],
            ),
            dim=-1,
        ).contiguous()
        _QK_ROPE_CACHE[key] = packed_cache
    return packed_cache


@torch.library.custom_op("torchtitan::merged_gdn_qkvz", mutates_args=())
def _merged_gdn_qkvz(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wz: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, _merged_weight((wq, wk, wv, wz)))


@_merged_gdn_qkvz.register_fake
def _merged_gdn_qkvz_fake(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wz: torch.Tensor,
) -> torch.Tensor:
    output_dim = wq.shape[0] + wk.shape[0] + wv.shape[0] + wz.shape[0]
    return torch.empty(*x.shape[:-1], output_dim, dtype=x.dtype, device=x.device)


@torch.library.custom_op("torchtitan::merged_gdn_ba", mutates_args=())
def _merged_gdn_ba(
    x: torch.Tensor,
    wb: torch.Tensor,
    wa: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, _merged_weight((wb, wa)))


@_merged_gdn_ba.register_fake
def _merged_gdn_ba_fake(
    x: torch.Tensor,
    wb: torch.Tensor,
    wa: torch.Tensor,
) -> torch.Tensor:
    output_dim = wb.shape[0] + wa.shape[0]
    return torch.empty(*x.shape[:-1], output_dim, dtype=x.dtype, device=x.device)


@torch.library.custom_op("torchtitan::merged_qwen35_qkv_gate", mutates_args=())
def _merged_qwen35_qkv_gate(
    x: torch.Tensor,
    wq_gate: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, _merged_weight((wq_gate, wk, wv)))


@_merged_qwen35_qkv_gate.register_fake
def _merged_qwen35_qkv_gate_fake(
    x: torch.Tensor,
    wq_gate: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
) -> torch.Tensor:
    output_dim = wq_gate.shape[0] + wk.shape[0] + wv.shape[0]
    return torch.empty(*x.shape[:-1], output_dim, dtype=x.dtype, device=x.device)


@torch.library.custom_op("torchtitan::fused_gdn_rmsnorm_gate", mutates_args=())
def _fused_gdn_rmsnorm_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

    return rmsnorm_fn(
        x,
        weight,
        None,
        z=gate,
        eps=eps,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )


@_fused_gdn_rmsnorm_gate.register_fake
def _fused_gdn_rmsnorm_gate_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("torchtitan::fused_offset_rmsnorm", mutates_args=())
def _fused_offset_rmsnorm(
    x: torch.Tensor,
    adjusted_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    output = torch.empty_like(x)
    ops.rms_norm(output, x, adjusted_weight, eps)
    return output


@_fused_offset_rmsnorm.register_fake
def _fused_offset_rmsnorm_fake(
    x: torch.Tensor,
    adjusted_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "torchtitan::fused_add_offset_rmsnorm",
    mutates_args=("x", "residual"),
)
def _fused_add_offset_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    adjusted_weight: torch.Tensor,
    eps: float,
) -> None:
    from vllm import _custom_ops as ops

    ops.fused_add_rms_norm(x, residual, adjusted_weight, eps)


@_fused_add_offset_rmsnorm.register_fake
def _fused_add_offset_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    adjusted_weight: torch.Tensor,
    eps: float,
) -> None:
    return None


def _wrap_column_projection_output(
    output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(weight, DTensor):
        return output
    output_dim = output.ndim - 1
    placements = tuple(
        Shard(output_dim) if isinstance(placement, Shard) else placement
        for placement in weight.placements
    )
    return DTensor.from_local(
        output,
        weight.device_mesh,
        placements,
        run_check=False,
    )


def _wrap_like(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if not isinstance(source, DTensor):
        return output
    return DTensor.from_local(
        output,
        source.device_mesh,
        source.placements,
        run_check=False,
    )


def apply_merged_gdn_projections() -> None:
    """Patch Qwen3.5 GDN to merge six input projections into two GEMMs.

    The model keeps its six original parameters and checkpoint keys. Their
    already TP-local weight shards are concatenated once on first use, matching
    vLLM's local ``qkvz`` and ``ba`` projection layout. A load-state hook drops
    the derived cache after RL weight synchronization.
    """
    global _GDN_PATCHED
    if _GDN_PATCHED:
        return

    from torchtitan.distributed.utils import is_in_batch_invariant_mode
    from torchtitan.models.qwen3_5.gdn import GatedDeltaNet

    original_init = GatedDeltaNet.__init__

    def init(self, config) -> None:
        original_init(self, config)
        _clear_merged_weight_cache()
        self.register_load_state_dict_post_hook(_clear_merged_weight_cache)

    def forward(self, x_TD, attention_masks=None):
        num_tokens = x_TD.shape[0]
        cu_seqlens_host = None
        if attention_masks is not None:
            with spmd.local():
                cu_seqlens = attention_masks.cu_seq_q.clone()
            cu_seqlens_host = attention_masks.cu_seq_q_host
            if cu_seqlens_host is None:
                raise ValueError(
                    "Qwen3.5 Gated DeltaNet varlen requires CPU cu_seqlens "
                    "metadata. Build VarlenMetadata with include_host_offsets=True."
                )
        else:
            cu_seqlens = torch.arange(
                0,
                num_tokens + 1,
                num_tokens,
                dtype=torch.int32,
                device=x_TD.device,
            )
            if is_in_batch_invariant_mode():
                cu_seqlens_host = (0, num_tokens)

        x = _local_tensor(x_TD)
        projection_weights = (
            self.in_proj_q.weight,
            self.in_proj_k.weight,
            self.in_proj_v.weight,
            self.in_proj_z.weight,
        )
        local_projection_weights = tuple(map(_local_tensor, projection_weights))
        qkvz = _merged_gdn_qkvz(x, *local_projection_weights)
        projection_sizes = tuple(weight.shape[0] for weight in local_projection_weights)
        query_TC, key_TC, value_TC, gate_TC = (
            _wrap_column_projection_output(part, projection_weights[0])
            for part in qkvz.split(projection_sizes, dim=-1)
        )

        gate_weights = (self.in_proj_b.weight, self.in_proj_a.weight)
        local_gate_weights = tuple(map(_local_tensor, gate_weights))
        ba = _merged_gdn_ba(x, *local_gate_weights)
        gate_sizes = tuple(weight.shape[0] for weight in local_gate_weights)
        b_TN, a_TN = (
            _wrap_column_projection_output(part, gate_weights[0])
            for part in ba.split(gate_sizes, dim=-1)
        )

        output_TNV = self.inner_gated_delta_net(
            query_TC,
            key_TC,
            value_TC,
            a_TN,
            b_TN,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            cu_seqlens_host=cu_seqlens_host,
        )
        gate_TNV = gate_TC.view(num_tokens, -1, self.value_head_dim)
        output_TNV = self.norm(output_TNV, gate_TNV)
        return self.out_proj(output_TNV.reshape(num_tokens, -1))

    GatedDeltaNet.__init__ = init
    GatedDeltaNet.forward = forward
    _GDN_PATCHED = True


def apply_merged_qwen35_attention_projection() -> None:
    """Patch Qwen3.5 full attention to merge Q/gate, K, and V projections."""
    global _ATTENTION_PATCHED
    if _ATTENTION_PATCHED:
        return

    from torchtitan.models.qwen3_5.model import Qwen35Attention

    original_init = Qwen35Attention.__init__

    def init(self, config) -> None:
        original_init(self, config)
        _clear_merged_weight_cache()
        self.register_load_state_dict_post_hook(_clear_merged_weight_cache)

    def forward(self, x_TD, attention_masks, positions=None):
        num_tokens = x_TD.shape[0]
        x = _local_tensor(x_TD)
        projection_weights = (self.wq.weight, self.wk.weight, self.wv.weight)
        local_projection_weights = tuple(map(_local_tensor, projection_weights))
        qkv_gate = _merged_qwen35_qkv_gate(x, *local_projection_weights)
        projection_sizes = tuple(weight.shape[0] for weight in local_projection_weights)
        xq_gate_TN2H, xk_TNH, xv_TNH = (
            _wrap_column_projection_output(part, projection_weights[0])
            for part in qkv_gate.split(projection_sizes, dim=-1)
        )

        xq_gate_TN2H = xq_gate_TN2H.view(num_tokens, -1, self.head_dim * 2)
        xq_TNH, gate_TNH = xq_gate_TN2H.chunk(2, dim=-1)
        xk_TNH = xk_TNH.view(num_tokens, -1, self.head_dim)
        xv_TNH = xv_TNH.view(num_tokens, -1, self.head_dim)

        xq_TNH = self.q_norm(xq_TNH)
        xk_TNH = self.k_norm(xk_TNH)

        xq_TNR, xq_TNP = (
            xq_TNH[..., : self.rotary_dim],
            xq_TNH[..., self.rotary_dim :],
        )
        xk_TNR, xk_TNP = (
            xk_TNH[..., : self.rotary_dim],
            xk_TNH[..., self.rotary_dim :],
        )
        xq_TNR, xk_TNR = self.rope(xq_TNR, xk_TNR, positions)
        xq_TNH = torch.cat([xq_TNR, xq_TNP], dim=-1)
        xk_TNH = torch.cat([xk_TNR, xk_TNP], dim=-1)

        out_TNH = self.inner_attention(
            xq_TNH,
            xk_TNH,
            xv_TNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        out_TNH = out_TNH * torch.sigmoid(gate_TNH)
        return self.wo(out_TNH.view(num_tokens, -1))

    Qwen35Attention.__init__ = init
    Qwen35Attention.forward = forward
    _ATTENTION_PATCHED = True


def apply_fused_qwen35_qk_norm_rope() -> None:
    """Fuse Qwen3.5 text QK normalization, partial RoPE, and gate split."""
    global _QK_NORM_ROPE_PATCHED
    if _QK_NORM_ROPE_PATCHED:
        return

    apply_merged_qwen35_attention_projection()

    from torchtitan.models.qwen3_5.model import Qwen35Attention
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

    original_forward = Qwen35Attention.forward

    def forward(self, x_TD, attention_masks, positions=None):
        if positions is None or positions.ndim != 1:
            return original_forward(self, x_TD, attention_masks, positions)

        num_tokens = x_TD.shape[0]
        x = _local_tensor(x_TD)
        projection_weights = (self.wq.weight, self.wk.weight, self.wv.weight)
        local_projection_weights = tuple(map(_local_tensor, projection_weights))
        qkv_gate = _merged_qwen35_qkv_gate(x, *local_projection_weights)
        projection_sizes = tuple(weight.shape[0] for weight in local_projection_weights)
        xq_gate, xk, xv = qkv_gate.split(projection_sizes, dim=-1)

        q_weight = _adjusted_offset_weight(
            _local_tensor(self.q_norm.weight),
            torch.float32,
        )
        k_weight = _adjusted_offset_weight(
            _local_tensor(self.k_norm.weight),
            torch.float32,
        )
        rope_cache = _qwen35_text_rope_cache(
            _local_tensor(self.rope.cache),
            self.rotary_dim,
        )
        q, k, gate = fused_qk_rmsnorm_rope_gate(
            xq_gate,
            xk,
            q_weight,
            k_weight,
            rope_cache,
            _local_tensor(positions),
            self.q_norm.eps,
            xq_gate.shape[-1] // (2 * self.head_dim),
            xk.shape[-1] // self.head_dim,
            self.head_dim,
            self.rotary_dim,
        )

        xq_TNH = _wrap_column_projection_output(
            q.view(num_tokens, -1, self.head_dim),
            projection_weights[0],
        )
        xk_TNH = _wrap_column_projection_output(
            k.view(num_tokens, -1, self.head_dim),
            projection_weights[1],
        )
        xv_TNH = _wrap_column_projection_output(
            xv.view(num_tokens, -1, self.head_dim),
            projection_weights[2],
        )
        gate_TNH = _wrap_column_projection_output(
            gate.view(num_tokens, -1, self.head_dim),
            projection_weights[0],
        )

        out_TNH = self.inner_attention(
            xq_TNH,
            xk_TNH,
            xv_TNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        out_TNH = out_TNH * torch.sigmoid(gate_TNH)
        return self.wo(out_TNH.view(num_tokens, -1))

    Qwen35Attention.forward = forward
    _QK_NORM_ROPE_PATCHED = True


def apply_fused_gdn_rmsnorm_gate() -> None:
    """Patch Qwen3.5 GDN output normalization to use vLLM's fused kernel."""
    global _GDN_NORM_PATCHED
    if _GDN_NORM_PATCHED:
        return

    from torchtitan.models.qwen3_5.gdn import RMSNormGated

    def forward(self, x, gate):
        x_local = _local_tensor(x)
        gate_local = _local_tensor(gate)
        weight_local = _local_tensor(self.weight)
        output = _fused_gdn_rmsnorm_gate(
            x_local.contiguous(),
            gate_local.contiguous(),
            weight_local.contiguous(),
            self.eps,
        )
        return _wrap_like(output, x)

    RMSNormGated.forward = forward
    _GDN_NORM_PATCHED = True


def apply_fused_qwen35_offset_rmsnorm() -> None:
    """Patch Qwen3.5 offset RMSNorm to use vLLM's fused CUDA kernel."""
    global _OFFSET_NORM_PATCHED
    if _OFFSET_NORM_PATCHED:
        return

    from torchtitan.models.qwen3_5.model import OffsetRMSNorm

    original_init = OffsetRMSNorm.__init__

    def init(self, config) -> None:
        original_init(self, config)
        _clear_offset_weight_cache()
        self.register_load_state_dict_post_hook(_clear_offset_weight_cache)

    def forward(self, x):
        x_local = _local_tensor(x)
        weight_local = _local_tensor(self.weight)
        adjusted_weight = _adjusted_offset_weight(weight_local, x_local.dtype)
        output = _fused_offset_rmsnorm(
            x_local.contiguous(),
            adjusted_weight,
            self.eps,
        )
        return _wrap_like(output, x)

    OffsetRMSNorm.__init__ = init
    OffsetRMSNorm.forward = forward
    _OFFSET_NORM_PATCHED = True


def _fused_add_qwen35_offset_rmsnorm(module, x, residual):
    x_local = _local_tensor(x)
    residual_local = _local_tensor(residual)
    weight_local = _local_tensor(module.weight)
    adjusted_weight = _adjusted_offset_weight(weight_local, x_local.dtype)
    _fused_add_offset_rmsnorm(
        x_local,
        residual_local,
        adjusted_weight,
        module.eps,
    )
    return x, residual


def apply_fused_qwen35_residual_norm() -> None:
    """Thread residuals through Qwen3.5 layers for fused add plus RMSNorm."""
    global _RESIDUAL_NORM_PATCHED
    if _RESIDUAL_NORM_PATCHED:
        return

    apply_fused_qwen35_offset_rmsnorm()

    from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper
    from torchtitan.models.qwen3_5.model import Qwen35Model

    original_forward = VLLMModelWrapper.forward

    def forward(
        self,
        input_ids=None,
        positions=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if not isinstance(self.model, Qwen35Model):
            return original_forward(
                self,
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        with self.spmd_context():
            hidden_states = self.model.tok_embeddings(input_ids)
            residual = None
            for layer in self.model.layers.values():
                if residual is None:
                    residual = hidden_states
                    hidden_states = layer.attention_norm(hidden_states)
                else:
                    hidden_states, residual = _fused_add_qwen35_offset_rmsnorm(
                        layer.attention_norm,
                        hidden_states,
                        residual,
                    )

                if layer.full_attn:
                    hidden_states = layer.attn(
                        hidden_states,
                        None,
                        positions,
                    )
                else:
                    hidden_states = layer.attn(hidden_states, None)

                hidden_states, residual = _fused_add_qwen35_offset_rmsnorm(
                    layer.ffn_norm,
                    hidden_states,
                    residual,
                )
                if layer.moe_enabled:
                    hidden_states = layer.moe(hidden_states)
                else:
                    hidden_states = layer.feed_forward(hidden_states)

            hidden_states, _ = _fused_add_qwen35_offset_rmsnorm(
                self.model.norm,
                hidden_states,
                residual,
            )

        if isinstance(hidden_states, DTensor):
            assert all(
                isinstance(placement, Replicate)
                for placement in hidden_states.placements
            )
            hidden_states = hidden_states._local_tensor
        return hidden_states

    VLLMModelWrapper.forward = forward
    _RESIDUAL_NORM_PATCHED = True


__all__ = [
    "apply_merged_gdn_projections",
    "apply_merged_qwen35_attention_projection",
    "apply_fused_qwen35_qk_norm_rope",
    "apply_fused_gdn_rmsnorm_gate",
    "apply_fused_qwen35_offset_rmsnorm",
    "apply_fused_qwen35_residual_norm",
]
