# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang support for TorchTitan's Gated DeltaNet (hybrid linear-attention models).

Shape legend: T = tokens in the step, C = local convolution channels,
H = local value heads, V = value head dim, K = key head dim, W = convolution width,
B = requests in the step, S = state slots.

SGLang keeps a hybrid model's recurrent state in its mamba pool, next to the KV
pool of the full-attention layers, and wraps the full-attention backend and a
linear-attention backend in ``HybridLinearAttnBackend``. TorchTitan models plug in
through SGLang's linear-attention model registry:

- ``TorchTitanHybridConfig`` is the HF config SGLang reads: it lists the
  full-attention and GDN layers and the per-slot state shapes. The convolution
  history is ``[W - 1, C]`` per slot (channels contiguous) and the SSM state is
  float32, as Attention Gym's paged kernels require.
- ``TorchTitanGDNAttnBackend`` is the linear-attention backend: it builds each
  step's slot indices, sequence boundaries and initial-state flags.
- ``SGLangInnerGDN`` runs a GDN layer on its slots through ``paged_gdn_forward``,
  the function the vLLM engine runs, so both engines execute the same kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel
from transformers import AutoConfig, PretrainedConfig

from torchtitan.models.common.attention import InnerAttention, LinearAttentionMetadata
from torchtitan.rl.generator.paged_gdn import paged_gdn_forward

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


class TorchTitanHybridConfig(PretrainedConfig):
    """HF config of a TorchTitan model with full-attention and GDN layers."""

    model_type = "torchtitan_hybrid"

    def __init__(
        self,
        *,
        layer_types: list[str] | None = None,
        linear_num_key_heads: int = 0,
        linear_num_value_heads: int = 0,
        linear_key_head_dim: int = 0,
        linear_value_head_dim: int = 0,
        linear_conv_kernel_dim: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_types = layer_types or []
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        tp = get_parallel().attn_tp_size
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        conv_dim = 2 * key_dim + value_dim
        for name, count in (
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
        ):
            if count % tp != 0:
                raise ValueError(
                    f"{name} ({count}) must be divisible by the attention tensor "
                    f"parallel size ({tp})"
                )
        shape = Mamba2StateShape(
            # Attention Gym reads [W - 1, C] histories with contiguous channels.
            conv=[(self.linear_conv_kernel_dim - 1, conv_dim // tp)],
            temporal=(
                self.linear_num_value_heads // tp,
                self.linear_value_head_dim,
                self.linear_key_head_dim,
            ),
            disable_conv_window_dedup=True,
            intermediate_size=value_dim,
            conv_dim=conv_dim,
            ssm_state_size=self.linear_key_head_dim,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            num_k_heads_per_tp=self.linear_num_key_heads // tp,
            conv_shard_groups=[key_dim, key_dim, value_dim],
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            # The convolution history stores trailing input columns in model dtype;
            # the recurrent state accumulates in float32.
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        )


_registered = False


def register_hybrid_config() -> None:
    """Register ``TorchTitanHybridConfig`` with transformers and SGLang. Idempotent."""
    global _registered
    if _registered:
        return
    AutoConfig.register(
        TorchTitanHybridConfig.model_type, TorchTitanHybridConfig, exist_ok=True
    )
    register_linear_attn_model(
        LinearAttnModelSpec(
            config_class=TorchTitanHybridConfig,
            backend_class_name=(
                "torchtitan.rl.generator.sglang.gdn.TorchTitanGDNAttnBackend"
            ),
            # Marks the model as keeping recurrent state, so SGLang's tree cache
            # owns the mamba slots and frees them when requests finish or are
            # retracted. Prefix reuse stays off: it would need the backend to
            # checkpoint recurrent states at prefix boundaries, so the engine
            # disables the radix cache for these models.
            uses_mamba_radix_cache=True,
            support_mamba_cache=False,
        )
    )
    _registered = True


@dataclass(kw_only=True, slots=True)
class _GDNMetadata:
    """Per-step state metadata every GDN layer of one forward reuses."""

    # int32 mamba pool slot of each request. SGLang reserves slot 0 for padding,
    # which the paged kernels treat as null.
    state_indices: torch.Tensor  # [B]
    cu_seqlens: torch.Tensor  # [B + 1] int32
    has_initial_state: torch.Tensor  # [B] bool
    single_token: bool
    num_tokens: int


class TorchTitanGDNAttnBackend(MambaAttnBackendBase):
    """Linear-attention backend for TorchTitan GDN layers.

    A request resumes its slot's state when it already has computed tokens (decode,
    or a continued chunked prefill). A step uses the single-token path when every
    request processes one token, as the vLLM engine decides.
    """

    def init_forward_metadata(self, forward_batch) -> None:
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_idle():
            self.forward_metadata = None
            return
        if not (forward_mode.is_decode() or forward_mode.is_extend()):
            raise NotImplementedError(
                f"{type(self).__name__} supports extend and decode, got {forward_mode}"
            )
        batch_size = forward_batch.batch_size
        state_indices_B = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        ).to(torch.int32)
        if forward_mode.is_decode():
            cu_seqlens = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=self.device
            )
            has_initial_state_B = torch.ones(
                batch_size, dtype=torch.bool, device=self.device
            )
            single_token = True
            num_tokens = batch_size
        else:
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            has_initial_state_B = forward_batch.extend_prefix_lens > 0
            single_token = max(forward_batch.extend_seq_lens_cpu) == 1
            num_tokens = sum(forward_batch.extend_seq_lens_cpu)
        self.forward_metadata = _GDNMetadata(
            state_indices=state_indices_B,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state_B,
            single_token=single_token,
            num_tokens=num_tokens,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate the static metadata buffers decode CUDA graphs read."""
        del max_num_tokens
        self._graph_state_indices = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._graph_cu_seqlens = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._graph_has_initial_state = torch.ones(
            max_bs, dtype=torch.bool, device=self.device
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        """Fill the static decode buffers for a capture or replay of ``forward_batch``.

        Padded rows have request pool row 0, whose mamba slot is the null slot 0.
        """
        del in_capture
        if not forward_batch.forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} captures decode CUDA graphs only, got "
                f"{forward_batch.forward_mode}"
            )
        batch_size = forward_batch.batch_size
        self._graph_state_indices[:batch_size].copy_(
            self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices[:batch_size]
            )
        )
        self.forward_metadata = _GDNMetadata(
            state_indices=self._graph_state_indices[:batch_size],
            cu_seqlens=self._graph_cu_seqlens[: batch_size + 1],
            has_initial_state=self._graph_has_initial_state[:batch_size],
            single_token=True,
            num_tokens=batch_size,
        )

    def layer_state(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return GDN layer ``layer_id``'s ``[S, W - 1, C]`` and ``[S, H, V, K]`` pools."""
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        return cache.conv[0], cache.temporal


class SGLangInnerGDN(InnerAttention):
    """GDN inner computation on SGLang's mamba pool.

    The enclosing ``common.attention.gdn.GDN`` owns all parameters; this module
    owns no state. Both it and the pool are head-sharded under tensor parallelism.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        layer_idx: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int = 4

    def __init__(self, config: Config) -> None:
        super().__init__()
        tp = get_parallel().attn_tp_size
        if config.num_k_heads % tp != 0 or config.num_v_heads % tp != 0:
            raise ValueError(
                f"num_k_heads ({config.num_k_heads}) and num_v_heads "
                f"({config.num_v_heads}) must both be divisible by the attention "
                f"tensor parallel size ({tp})."
            )
        self.layer_idx = config.layer_idx
        self.local_num_k_heads = config.num_k_heads // tp
        self.local_num_v_heads = config.num_v_heads // tp
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        """Run the GDN step on rank-local tensors."""
        del attention_metadata
        assert key_head_dim == self.head_k_dim
        assert value_head_dim == self.head_v_dim
        mixed_qkv_TC = torch.cat([query_TC, key_TC, value_TC], dim=-1)
        conv_weight_CW = torch.cat(
            [conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W], dim=0
        ).squeeze(1)
        assert conv_weight_CW.shape[-1] == self.conv_kernel_size

        # Padding rows (expert parallelism, idle steps) keep zero output.
        output_THV = mixed_qkv_TC.new_zeros(
            mixed_qkv_TC.shape[0], self.local_num_v_heads, self.head_v_dim
        )
        backend = get_attn_backend().linear_attn_backend
        metadata = backend.forward_metadata
        if metadata is None:
            return output_THV
        conv_state_SWC, ssm_state_SHVK = backend.layer_state(self.layer_idx)
        paged_gdn_forward(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            conv_weight_CW,
            A_log_H,
            dt_bias_H,
            output_THV,
            conv_state_SWC=conv_state_SWC,
            ssm_state_SHVK=ssm_state_SHVK,
            state_indices_N=metadata.state_indices,
            cu_seqlens=metadata.cu_seqlens,
            has_initial_state_N=metadata.has_initial_state,
            single_token=metadata.single_token,
            num_tokens=metadata.num_tokens,
            num_k_heads=self.local_num_k_heads,
            num_v_heads=self.local_num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        return output_THV
