# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang attention backend and inner attention for TorchTitan models.

Shape legend: T = tokens in the step, H = local query heads, G = local KV heads,
K = query/key head dim, V = value head dim, B = requests in the step,
P = KV pages per request.

The backend runs PyTorch's FlashAttention varlen kernel on SGLang's paged KV pool,
the same kernel and arguments the trainer and the vLLM ``CUSTOM`` backend use, so
batch-invariant mode keeps logits bitwise identical across the three.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.arg_groups.choices import (
    add_attention_backend_choices,
    ATTENTION_BACKEND_CHOICES,
)
from sglang.srt.layers.attention.attention_registry import register_attention_backend
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.runtime_context import get_parallel
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.attention import InnerAttention
from torchtitan.observability.logging import warn_once
from torchtitan.tools.utils import get_cuda_flash_attention_impl

logger = logging.getLogger(__name__)

# Name of the attention backend in SGLang's registry (``ServerArgs.attention_backend``).
TORCHTITAN_ATTENTION_BACKEND = "torchtitan_varlen"


@dataclass(kw_only=True, slots=True)
class _VarlenMetadata:
    """Per-step metadata every layer of one forward reuses."""

    cu_seqlens_q: torch.Tensor  # [B + 1] int32
    cu_seqlens_k: torch.Tensor | None  # [B + 1] int32, FA2 only
    seqused_k: torch.Tensor  # [B] int32
    max_seqlen_q: int
    max_seqlen_k: int
    block_table: torch.Tensor  # [B, P] int32 page ids


class TorchTitanVarlenAttnBackend(AttentionBackend):
    """SGLang attention backend that calls ``torch.nn.attention.varlen.varlen_attn_out``.

    Pages: SGLang's ``req_to_token`` maps each request position to a KV slot. With
    ``page_size`` > 1 the slots of one page are contiguous, so the page id of a
    position is ``slot // page_size``; the block table keeps one entry per page.
    FA2 (SM < 9.0) needs pages of a multiple of 256 tokens, as in vLLM.
    """

    def __init__(self, model_runner):
        super().__init__()
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.max_context_len = model_runner.model_config.context_len
        self.forward_metadata: _VarlenMetadata | None = None

        flash_attention_impl = get_cuda_flash_attention_impl()
        if flash_attention_impl is not None:
            if current_flash_attention_impl() != flash_attention_impl:
                activate_flash_attention_impl(flash_attention_impl)
        else:
            warn_once(
                logger,
                "FA3/FA4 not available on this CUDA architecture, falling back to FA2.",
            )
        self._uses_fa2 = current_flash_attention_impl() in (None, "FA2")
        if self._uses_fa2 and self.page_size % 256 != 0:
            raise ValueError(
                f"FA2 paged attention needs page_size to be a multiple of 256, got "
                f"{self.page_size}"
            )

    def init_forward_metadata(self, forward_batch) -> None:
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_idle():
            self.forward_metadata = None
            return
        if not (forward_mode.is_decode() or forward_mode.is_extend()):
            raise NotImplementedError(
                f"{type(self).__name__} supports extend and decode, got {forward_mode}"
            )
        seq_lens_B = forward_batch.seq_lens.to(torch.int32)
        batch_size = seq_lens_B.shape[0]
        device = seq_lens_B.device
        if forward_mode.is_decode():
            cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            max_seqlen_q = 1
        else:
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            max_seqlen_q = max(forward_batch.extend_seq_lens_cpu)
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max())
        token_slots_BL = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]
        page_starts_BP = token_slots_BL[:, :: self.page_size]
        # FlashAttention checks stride(-1) == 1, which a strided view can miss even
        # when PyTorch calls it contiguous (e.g. one page: shape [B, 1], strides
        # (1, P)), so write the page ids into a dense table.
        block_table_BP = torch.empty(
            page_starts_BP.shape, dtype=torch.int32, device=device
        )
        torch.floor_divide(page_starts_BP, self.page_size, out=block_table_BP)
        # FA3 and FA4 infer key lengths from block_table + seqused_k for paged KV
        # (FA4 rejects block_table together with cu_seqlens_k); FA2 needs cu_seqlens_k.
        cu_seqlens_k = None
        if self._uses_fa2:
            cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seq_lens_B, dim=0, dtype=torch.int32), (1, 0)
            )
        self.forward_metadata = _VarlenMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seq_lens_B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_table=block_table_BP,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate the static metadata buffers decode CUDA graphs read.

        A graph records the buffer addresses and the ``max_seqlen_*`` Python ints of
        its capture, so replays refill the buffers in place and pass the context
        length as ``max_seqlen_k`` (the kernel bounds keys by ``seqused_k``).
        """
        del max_num_tokens
        max_num_pages = -(-self.max_context_len // self.page_size)
        self._graph_cu_seqlens_q = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._graph_cu_seqlens_k = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._graph_seqused_k = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._graph_block_table = torch.zeros(
            max_bs, max_num_pages, dtype=torch.int32, device=self.device
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        # Padded rows attend to one key.
        return 1

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        """Fill the static decode buffers for a capture or replay of ``forward_batch``."""
        del in_capture
        if not forward_batch.forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} captures decode CUDA graphs only, got "
                f"{forward_batch.forward_mode}"
            )
        batch_size = forward_batch.batch_size
        seq_lens_B = forward_batch.seq_lens[:batch_size]
        self._graph_seqused_k[:batch_size].copy_(seq_lens_B)
        if self._uses_fa2:
            torch.cumsum(
                seq_lens_B,
                dim=0,
                dtype=torch.int32,
                out=self._graph_cu_seqlens_k[1 : batch_size + 1],
            )
        token_slots_BL = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[:batch_size], : self.max_context_len
        ]
        self._graph_block_table[:batch_size].copy_(
            token_slots_BL[:, :: self.page_size] // self.page_size
        )
        self.forward_metadata = _VarlenMetadata(
            cu_seqlens_q=self._graph_cu_seqlens_q[: batch_size + 1],
            cu_seqlens_k=(
                self._graph_cu_seqlens_k[: batch_size + 1] if self._uses_fa2 else None
            ),
            seqused_k=self._graph_seqused_k[:batch_size],
            max_seqlen_q=1,
            max_seqlen_k=self.max_context_len,
            block_table=self._graph_block_table[:batch_size],
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        metadata = self.forward_metadata
        assert metadata is not None, "init_forward_metadata must run first"

        num_heads = layer.tp_q_head_num
        q_THK = q.view(-1, num_heads, layer.qk_head_dim)
        key_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
        )
        value_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        out_THV = q_THK.new_empty(q_THK.shape[0], num_heads, layer.v_head_dim)

        extra_kwargs: dict[str, Any] = {}
        # FA2's auto num_splits heuristic produces NaN intermittently with paged KV
        # (pytorch/pytorch#179760); FA3 picks num_splits from the sequence length,
        # which changes the reduction tree, so batch-invariant mode fixes it to 1.
        if self._uses_fa2 or is_in_batch_invariant_mode():
            extra_kwargs["num_splits"] = 1
        if num_heads > layer.tp_k_head_num:
            extra_kwargs["enable_gqa"] = True
        if layer.sliding_window_size > 0:
            window_size = (layer.sliding_window_size - 1, 0)
        else:
            window_size = (-1, 0)

        torch.nn.attention.varlen.varlen_attn_out(
            out_THV,
            q_THK,
            key_cache,
            value_cache,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            metadata.max_seqlen_q,
            metadata.max_seqlen_k,
            scale=layer.scaling,
            window_size=window_size,
            block_table=metadata.block_table,
            seqused_k=metadata.seqused_k,
            **extra_kwargs,
        )
        return out_THV.view(-1, num_heads * layer.v_head_dim)


@register_attention_backend(TORCHTITAN_ATTENTION_BACKEND)
def create_torchtitan_varlen_backend(runner) -> TorchTitanVarlenAttnBackend:
    return TorchTitanVarlenAttnBackend(runner)


if TORCHTITAN_ATTENTION_BACKEND not in ATTENTION_BACKEND_CHOICES:
    add_attention_backend_choices([TORCHTITAN_ATTENTION_BACKEND])


class SGLangInnerAttention(InnerAttention):
    """Adapter from TorchTitan's tensor layout to SGLang's ``RadixAttention``.

    ``RadixAttention`` writes this layer's KV to SGLang's pool and dispatches to the
    active attention backend; the forward batch comes from SGLang's forward context,
    which ``SGLangModelWrapper.forward`` publishes.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        layer_id: int
        """Index of this layer's KV in SGLang's KV pool."""
        num_heads: int
        num_kv_heads: int
        head_dim: int
        value_head_dim: int | None = None
        scale: float | None = None
        sliding_window_size: int | None = None
        """Causal sliding-window size (``None`` => full attention)."""

        @property
        def is_full_attention(self) -> bool:
            return self.sliding_window_size is None

    def __init__(self, config: Config) -> None:
        super().__init__()
        tp_degree = get_parallel().attn_tp_size
        for name, count in (
            ("num_heads", config.num_heads),
            ("num_kv_heads", config.num_kv_heads),
        ):
            if count % tp_degree != 0:
                raise ValueError(
                    f"{name} ({count}) must be divisible by the attention tensor "
                    f"parallel size ({tp_degree})"
                )
        self.value_head_dim = (
            config.value_head_dim
            if config.value_head_dim is not None
            else config.head_dim
        )
        self.scale = (
            config.scale if config.scale is not None else config.head_dim**-0.5
        )
        self.radix_attn = RadixAttention(
            num_heads=config.num_heads // tp_degree,
            head_dim=config.head_dim,
            scaling=self.scale,
            num_kv_heads=config.num_kv_heads // tp_degree,
            layer_id=config.layer_id,
            v_head_dim=self.value_head_dim,
            sliding_window_size=config.sliding_window_size or -1,
            prefix=f"model.layers.{config.layer_id}.attention.inner_attention",
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: None = None,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run SGLang paged attention on local (non-DTensor) tensors.

        Args:
            q_THK: ``(num_tokens, num_heads, query/key_head_dim)``
            k_THK: ``(num_tokens, num_kv_heads, query/key_head_dim)``
            v_THV: ``(num_tokens, num_kv_heads, value_head_dim)``
            scale: Optional per-call scale; must match the scale set at construction.

        Returns:
            ``(num_tokens, num_heads, value_head_dim)``.
        """
        if attention_metadata is not None:
            raise ValueError(
                "SGLangInnerAttention does not support attention_metadata; SGLang "
                "manages causal masking and the KV cache internally."
            )
        if scale is not None and scale != self.scale:
            raise ValueError(
                f"Model attention scale {scale} != SGLang scale {self.scale}"
            )
        num_tokens = q_THK.shape[0]
        out_TD = self.radix_attn(q_THK, k_THK, v_THV, get_forward_batch())
        return out_TD.view(num_tokens, -1, self.value_head_dim)
