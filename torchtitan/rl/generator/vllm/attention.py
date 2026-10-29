# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import spmd_types as spmd

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torch.nn.attention.varlen import AuxRequest
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    set_default_quant_scales,
)
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.observability.logging import warn_once
from torchtitan.tools.utils import get_cuda_flash_attention_impl

logger = logging.getLogger(__name__)


# Opt-in with TORCHTITAN_VLLM_FA2_SPLIT_MIXED=1 (with TORCHTITAN_VLLM_FA2_KERNEL=1).
_SPLIT_MIXED_BATCHES = os.environ.get("TORCHTITAN_VLLM_FA2_SPLIT_MIXED") == "1"
# Decode requests per single-token FA2 call: FA2 splits the KV only while
# batch * kv_heads stays below ~0.8 * 2 * SMs.
_DECODE_CHUNK = 64
# KV splits for single-token (decode) FA2 calls; 0 keeps FA2's heuristic, which sizes
# splits for two CTAs per SM (one fits at head_dim 256) and stops splitting above
# ~0.8 * 2 * SMs rows. A100, Qwen3.5-35B-A3B TP2, 5-45K keys, 16 splits: 32 decodes
# 685 -> 621 us per layer; 25 decodes among 256 padded rows 1942 -> 504 us.
# Opt-in with TORCHTITAN_VLLM_FA2_DECODE_SPLITS=16.
_DECODE_NUM_SPLITS = int(os.environ.get("TORCHTITAN_VLLM_FA2_DECODE_SPLITS", "0"))


# Opt-in with TORCHTITAN_TRTLLM_DECODE=1 (SM100 / SM103); the decodes of mixed batches too
# with TORCHTITAN_TRTLLM_SPLIT_MIXED=1.
_TRTLLM_DECODE = os.environ.get("TORCHTITAN_TRTLLM_DECODE") == "1"
_TRTLLM_SPLIT_MIXED = (
    _TRTLLM_DECODE and os.environ.get("TORCHTITAN_TRTLLM_SPLIT_MIXED") == "1"
)
# FA4-only alternative to TORCHTITAN_TRTLLM_SPLIT_MIXED: the decodes of mixed batches as a
# separate FA4 call (_forward_mixed_fa4). Opt-in with TORCHTITAN_FA4_SPLIT_MIXED=1.
_FA4_SPLIT_MIXED = (
    not _TRTLLM_SPLIT_MIXED and os.environ.get("TORCHTITAN_FA4_SPLIT_MIXED") == "1"
)
# FA4's swap-AB split-KV decode kernel for decode requests (Dao-AILab/flash-attention #2973, with
# head_dim 256 / SM103 / 128-token pages from the FA4 tree on PYTHONPATH; see _fa4_swapab_decode).
# Opt-in with TORCHTITAN_FA4_SWAPAB_DECODE=1; the decodes of mixed batches too with
# TORCHTITAN_FA4_SWAPAB_SPLIT_MIXED=1.
_FA4_SWAPAB_DECODE = (
    not _TRTLLM_DECODE and os.environ.get("TORCHTITAN_FA4_SWAPAB_DECODE") == "1"
)
_FA4_SWAPAB_SPLIT_MIXED = (
    _FA4_SWAPAB_DECODE and os.environ.get("TORCHTITAN_FA4_SWAPAB_SPLIT_MIXED") == "1"
)
# vLLM's FlashInfer workspace size; TRT-LLM-gen keeps zero-initialized split counters in it.
_TRTLLM_WORKSPACE_BYTES = 256 << 20
_trtllm_workspaces: dict[torch.device, torch.Tensor] = {}


def _trtllm_workspace(device: torch.device) -> torch.Tensor:
    workspace = _trtllm_workspaces.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _TRTLLM_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        _trtllm_workspaces[device] = workspace
    return workspace


def _trtllm_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
) -> None:
    """Causal paged decode attention (one query token per request) with FlashInfer's
    TRT-LLM-gen kernel, which native vLLM and SGLang run on Blackwell.

    On GB300 with vLLM's 128-token kernel pages, FA4 decode reads the KV at 5.1-5.8 TB/s
    and TRT-LLM-gen at 5.9-6.4 TB/s (Qwen3.5, 8-128 requests at 2-90K keys): 666 -> 561 us
    per layer for 32 requests at 35B-A3B. It reads the block table and sequence lengths on the GPU,
    so CUDA graphs replay it without host-side planning. ``max_seqlen_k`` is the model's
    max length while vLLM captures, as for its own FlashInfer backend.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    # Padded FULL-graph batches: one query row per (possibly padded) request.
    batch = seqused_k.shape[0]
    trtllm_batch_decode_with_kv_cache(
        query[:batch],
        (key_cache, value_cache),
        _trtllm_workspace(query.device),
        block_table,
        seqused_k,
        max_seqlen_k,
        bmm1_scale=softmax_scale,
        out=output[:batch],
        kv_layout="NHD",
    )


def _fa4_swapab_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Causal paged decode attention (one query token per request) on FA4's swap-AB kernel.

    FA4's standard kernel puts the (packed GQA) query rows on the MMA's 128-row dimension, which
    one decode token per request leaves mostly empty; at head_dim 256 on GB300 it reads the KV at
    4.3-5.6 TB/s. The swap-AB kernel puts 128 KV tokens on that dimension and the query heads on N,
    as TRT-LLM-gen's decode does, and splits each request's KV: 6.6-7.1 TB/s for Qwen3.5-35B-A3B
    at 4-256 requests and 1-160K keys, vs 5.4-6.4 for TRT-LLM-gen. It takes a dense
    (batch, 1, heads, head_dim) query, so the call goes to FA4's interface directly; the backend
    is forced so an ineligible call fails instead of falling back. All on-device, so FULL CUDA
    graphs replay it; the split count follows the batch and the block table's capacity.
    """
    from flash_attn.cute.interface import _flash_attn_fwd

    # Padded FULL-graph batches: one query row per (possibly padded) request.
    batch = seqused_k.shape[0]
    previous = os.environ.get("FLASH_ATTENTION_SM100_BACKEND")
    os.environ["FLASH_ATTENTION_SM100_BACKEND"] = "swapab"
    try:
        _flash_attn_fwd(
            query[:batch].unsqueeze(1),
            key_cache,
            value_cache,
            seqused_k=seqused_k,
            page_table=block_table,
            softmax_scale=softmax_scale,
            causal=True,
            out=output[:batch].unsqueeze(1),
        )
    finally:
        if previous is None:
            del os.environ["FLASH_ATTENTION_SM100_BACKEND"]
        else:
            os.environ["FLASH_ATTENTION_SM100_BACKEND"] = previous


def _fa2_single_token(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
) -> None:
    """Causal paged FA2 on one query token per request, ``_DECODE_NUM_SPLITS`` KV splits.

    vLLM's ``flash_attn_varlen_func`` rejects ``num_splits > 1`` for FA2; its op takes it
    on the single-token path, where the query heads of a KV head become the query rows.
    """
    torch.ops._vllm_fa2_C.varlen_fwd(
        query,
        key_cache,
        value_cache,
        output,
        cu_seqlens_q,
        torch.empty_like(cu_seqlens_q),  # unused with seqused_k
        seqused_k,
        None,  # leftpad_k
        block_table,
        None,  # alibi_slopes
        1,  # max_seqlen_q
        max_seqlen_k,
        0.0,  # dropout_p
        softmax_scale,
        False,  # zero_tensors
        True,  # is_causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        False,  # return_softmax
        _DECODE_NUM_SPLITS,
        None,  # generator
    )


@register_backend(AttentionBackendEnum.CUSTOM)
class TorchTitanVarlenInnerAttentionBackend(FlashAttentionBackend):
    """Custom vLLM attention backend using PyTorch's native FlashAttention kernel.

    This class is not directly referenced in user code. It is registered into
    vLLM's attention backend registry via the ``@register_backend`` decorator
    and selected at runtime when the vLLM engine is configured to use a CUSTOM
    attention backend.

    Inheriting from ``FlashAttentionBackend`` is not strictly required for all
    backends, but it is convenient here to reuse metadata construction logic.
    """

    @staticmethod
    def get_name():
        # vLLM requires any custom attention backend to return "CUSTOM" as its
        # name so the backend registry can look it up correctly.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TorchTitanVarlenInnerAttentionImpl

    @staticmethod
    def get_builder_cls():
        class TorchTitanVarlenInnerAttentionMetadataBuilder(
            FlashAttentionMetadataBuilder
        ):
            _cudagraph_support = AttentionCGSupport.ALWAYS
            # Decodes first, for the mixed-batch splits (_forward_mixed_fa2,
            # _forward_mixed_trtllm, _forward_mixed_fa4).
            reorder_batch_threshold = (
                1
                if _SPLIT_MIXED_BATCHES
                or _TRTLLM_SPLIT_MIXED
                or _FA4_SPLIT_MIXED
                or _FA4_SWAPAB_SPLIT_MIXED
                else None
            )

        return TorchTitanVarlenInnerAttentionMetadataBuilder


class TorchTitanVarlenInnerAttentionDiffKVBackend(FlashAttentionDiffKVBackend):
    """PyTorch varlen backend using vLLM's unequal K/V cache support."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TorchTitanVarlenInnerAttentionDiffKVImpl

    @staticmethod
    def get_builder_cls():
        return TorchTitanVarlenInnerAttentionBackend.get_builder_cls()


class TorchTitanVarlenInnerAttentionImpl(FlashAttentionImpl):
    """
    Custom vLLM attention backend impl using PyTorch's native FlashAttention varlen API.
    Instead of using vLLM's FlashAttention kernel, this implementation takes the kernel
    dependency from torch directly while supporting the same interface.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # optional post-attention epilogue transform (out, lse) -> out
        # set by VLLMModelWrapper in vllm_wrapper.py
        self.out_transform = None

        self.enable_gqa = self.num_heads > self.num_kv_heads

        flash_attention_impl = get_cuda_flash_attention_impl()
        if flash_attention_impl is not None:
            # activate_flash_attention_impl() will restore internal global state
            # and re-run register function, so we want to only call it once.
            if current_flash_attention_impl() != flash_attention_impl:
                try:
                    activate_flash_attention_impl(flash_attention_impl)
                except (ImportError, RuntimeError, ValueError) as error:
                    capability = torch.cuda.get_device_capability()
                    raise RuntimeError(
                        f"{flash_attention_impl} is required on detected SM "
                        f"{capability[0]}.{capability[1]}, but activation failed."
                    ) from error
        else:
            warn_once(
                logger,
                "FA3/FA4 not available on this CUDA architecture, falling back to FA2. ",
            )
        # Hill-climb prototype (opt-in): on FA2 GPUs, torch's paged FA2 reads decode KV at
        # ~330-550 GB/s (num_splits=1, 1024-token pages: 3.2 ms per Qwen3.5-27B TP2 layer at
        # 32 x 16K) while vLLM's FA2 build reads it at ~1.4 TB/s (0.73 ms). Outside
        # batch-invariant mode, run vLLM's kernel instead.
        self.use_vllm_fa2 = (
            flash_attention_impl is None
            and os.environ.get("TORCHTITAN_VLLM_FA2_KERNEL") == "1"
        )
        if _TRTLLM_DECODE:
            # Before CUDA graph capture, so the workspace is not a graph-pool allocation.
            _trtllm_workspace(torch.device("cuda", torch.cuda.current_device()))

    # Based on vLLM's FlashAttentionImpl.forward():
    # https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape is ``[num_blocks, num_kv_heads, block_size,
                2 * head_size]``, with K and V packed in the last dimension.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        # Breakable cuda_graph: under VLLM_USE_BREAKABLE_CUDAGRAPH the recorded
        # forward closure pins capture-time args (attn_metadata None, varlen metadata
        # absent). Re-read live per-layer metadata + kv_cache from the forward context
        # (vLLM refreshes them before each replay) BEFORE the None check, else replay
        # short-circuits to output.fill_(0) (zeroed attention). No-op outside capture.
        attn_metadata, _, kv_cache, _ = get_attention_context(layer.layer_name)

        if attn_metadata is None:
            # Profiling / CUDA graph dummy-capture run (no real metadata yet).
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        assert attn_type not in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ), "Encoder-only attention not supported yet."

        # vLLM #44455 packs K and V into the content dimension for full-attention
        # layers, including those in hybrid models. Restore the layout expected by
        # varlen attention before splitting them.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        assert not self.kv_cache_dtype.startswith(
            "fp8"
        ), "FP8 KV cache not supported yet."

        assert not attn_metadata.use_cascade, "Cascade not supported yet."

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        assert self.dcp_world_size == 1, "DCP not supported yet."

        if not attn_metadata.causal:
            raise RuntimeError("Non-causal attention not supported yet.")

        # vLLM assigns sliding_window_size = None to (-1, -1) w/ optional causal flag
        # but varlen only encode with sliding window. so we need to convert vllm (-1, -1) to (-1, 0)
        # for proper full causal attention instead of bidirectional
        if self.sliding_window == (-1, -1):
            sliding_window_size = (-1, 0)
        else:
            # by default vLLM sets attention type = DECODER, which will set (W-1, 0)
            sliding_window_size = self.sliding_window

        assert self.alibi_slopes is None, "Alibi slopes not supported yet."

        if (
            self.use_vllm_fa2
            and self.out_transform is None
            and not is_in_batch_invariant_mode()
        ):
            if max_seqlen_k < max_seqlen_q:
                # vLLM's CUDA graph memory profiling runs dummy batches whose keys (length
                # 1) are shorter than their queries, e.g. 8192 tokens over 48 requests at
                # max_num_seqs=48, where paged causal FA2 reads out of bounds. Real batches
                # never have that shape.
                return output[:num_actual_tokens].zero_()
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            if (
                _SPLIT_MIXED_BATCHES
                and max_seqlen_q > 1
                and self.sliding_window == (-1, -1)
                and cu_seqlens_q.shape[0] - 1 <= num_actual_tokens
            ):
                self._forward_mixed_fa2(
                    flash_attn_varlen_func,
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                )
                return output[:num_actual_tokens]
            if (
                _DECODE_NUM_SPLITS
                and max_seqlen_q == 1
                and self.sliding_window == (-1, -1)
            ):
                _fa2_single_token(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_k,
                    self.scale,
                )
                return output[:num_actual_tokens]
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=(
                    None
                    if self.sliding_window == (-1, -1)
                    else list(self.sliding_window)
                ),
                block_table=block_table,
                fa_version=2,
                num_splits=attn_metadata.max_num_splits,
            )
            return output[:num_actual_tokens]

        if (
            _TRTLLM_DECODE
            and self.out_transform is None
            and self.sliding_window == (-1, -1)
            and not is_in_batch_invariant_mode()
        ):
            if max_seqlen_q == 1:
                _trtllm_decode(
                    query,
                    key_cache,
                    value_cache,
                    output,
                    block_table,
                    seqused_k,
                    max_seqlen_k,
                    self.scale,
                )
                return output[:num_actual_tokens]
            if (
                _TRTLLM_SPLIT_MIXED
                and max_seqlen_k >= max_seqlen_q
                and cu_seqlens_q.shape[0] - 1 <= num_actual_tokens
            ):
                self._forward_mixed_trtllm(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                )
                return output[:num_actual_tokens]

        if (
            _FA4_SWAPAB_DECODE
            and self.out_transform is None
            and self.sliding_window == (-1, -1)
            and not is_in_batch_invariant_mode()
        ):
            if max_seqlen_q == 1:
                _fa4_swapab_decode(
                    query,
                    key_cache,
                    value_cache,
                    output,
                    block_table,
                    seqused_k,
                    self.scale,
                )
                return output[:num_actual_tokens]
            if (
                _FA4_SWAPAB_SPLIT_MIXED
                and max_seqlen_k >= max_seqlen_q
                and cu_seqlens_q.shape[0] - 1 <= num_actual_tokens
            ):
                self._forward_mixed_swapab(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                )
                return output[:num_actual_tokens]

        if (
            _FA4_SPLIT_MIXED
            and max_seqlen_q > 1
            and max_seqlen_k >= max_seqlen_q
            and self.out_transform is None
            and self.sliding_window == (-1, -1)
            and not is_in_batch_invariant_mode()
            and current_flash_attention_impl() == "FA4"
            and cu_seqlens_q.shape[0] - 1 <= num_actual_tokens
        ):
            self._forward_mixed_fa4(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                output[:num_actual_tokens],
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
            )
            return output[:num_actual_tokens]

        # FA3 and FA4 infer key lengths from block_table + seqused_k for paged
        # KV. FA4 explicitly rejects page_table together with cu_seqlens_k.
        # FA2 requires cu_seqlens_k to be explicitly set.
        fa_impl = current_flash_attention_impl()
        if fa_impl in ("FA3", "FA4"):
            cu_seqlens_k = None
        else:
            num_seqs = seqused_k.shape[0]
            cu_seqlens_k = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)
        extra_kwargs: dict[str, Any] = {}

        # TODO(pytorch/pytorch#179760): FA2's auto num_splits heuristic
        # produces NaN intermittently with paged KV (block_table). Force
        # num_splits=1 as a workaround until the root cause is fixed
        # upstream. current_flash_attention_impl() returns None when FA2
        # is the implicit default (SM < 9.0). For FA3, only force
        # num_splits=1 in batch-invariant mode (determinism).
        if fa_impl in (None, "FA2") or is_in_batch_invariant_mode():
            extra_kwargs["num_splits"] = 1

        if self.enable_gqa:
            extra_kwargs["enable_gqa"] = True

        if self.out_transform is not None:
            extra_kwargs["return_aux"] = AuxRequest(lse=True)

        result = torch.nn.attention.varlen.varlen_attn_out(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=sliding_window_size,
            block_table=block_table,
            seqused_k=seqused_k,
            **extra_kwargs,
        )
        if self.out_transform is None:
            return result

        out, lse = result
        out = self.out_transform(out, lse.transpose(0, 1))
        output[:num_actual_tokens].copy_(out)
        return output[:num_actual_tokens]

    def _forward_mixed_trtllm(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> None:
        """A batch with prefills: the decodes on TRT-LLM-gen, the prefills on FA4.

        One FA4 call over a mixed batch plans for the prefill's query length: it splits no
        decode request's KV, and at head_dim 256 with a query-head group that does not divide
        128 (Qwen3.5-27B: 6) it also drops PackGQA and reads each decode request's KV once per
        query head. Attention was 49 ms of an 85 ms mixed step for Qwen3.5-27B on GB300, vs
        9 ms for SGLang. As ``_forward_mixed_fa2`` does for
        FA2: the runner orders decodes first, so request ``i < num_decodes`` owns token
        ``i``; TRT-LLM-gen runs rows ``[0, num_requests)`` with the prefills' key lengths
        masked to zero, then FA4 runs the prefills with the decodes as empty queries.
        All on-device, so FULL CUDA graphs keep one graph per token count.
        """
        is_decode = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) == 1
        # Prefill requests get no keys here; their rows are rewritten below.
        _trtllm_decode(
            query,
            key_cache,
            value_cache,
            output,
            block_table,
            torch.where(is_decode, seqused_k, 0),
            max_seqlen_k,
            self.scale,
        )
        torch.nn.attention.varlen.varlen_attn_out(
            output,
            query,
            key_cache,
            value_cache,
            torch.maximum(cu_seqlens_q, is_decode.sum(dtype=cu_seqlens_q.dtype)),
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=(-1, 0),
            block_table=block_table,
            seqused_k=seqused_k,
            enable_gqa=self.enable_gqa,
        )

    def _forward_mixed_swapab(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> None:
        """A batch with prefills: the decodes on FA4's swap-AB kernel, the prefills on FA4.

        ``_forward_mixed_trtllm`` with ``_fa4_swapab_decode`` for the decodes: the runner
        orders decodes first, the decode call runs rows ``[0, num_requests)`` with the
        prefills' key lengths masked to zero, then FA4's standard kernel runs the prefills
        with the decodes as empty queries and rewrites the prefill rows.
        """
        is_decode = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) == 1
        _fa4_swapab_decode(
            query,
            key_cache,
            value_cache,
            output,
            block_table,
            torch.where(is_decode, seqused_k, 0),
            self.scale,
        )
        torch.nn.attention.varlen.varlen_attn_out(
            output,
            query,
            key_cache,
            value_cache,
            torch.maximum(cu_seqlens_q, is_decode.sum(dtype=cu_seqlens_q.dtype)),
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=(-1, 0),
            block_table=block_table,
            seqused_k=seqused_k,
            enable_gqa=self.enable_gqa,
        )

    def _forward_mixed_fa4(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> None:
        """A batch with prefills as two FA4 calls: the decodes, then the prefills.

        ``_forward_mixed_trtllm`` without TRT-LLM-gen. The decode call passes
        ``max_seqlen_q=1``, so FA4 keeps PackGQA and its split-KV heuristic for the decode
        requests; the prefills see no queries in it, as the decodes see none in the prefill
        call. The runner orders decodes first.
        """
        num_decodes = ((cu_seqlens_q[1:] - cu_seqlens_q[:-1]) == 1).sum(
            dtype=cu_seqlens_q.dtype
        )
        for cu_seqlens, max_q in (
            (torch.minimum(cu_seqlens_q, num_decodes), 1),
            (torch.maximum(cu_seqlens_q, num_decodes), max_seqlen_q),
        ):
            torch.nn.attention.varlen.varlen_attn_out(
                output,
                query,
                key_cache,
                value_cache,
                cu_seqlens,
                None,
                max_q,
                max_seqlen_k,
                scale=self.scale,
                window_size=(-1, 0),
                block_table=block_table,
                seqused_k=seqused_k,
                enable_gqa=self.enable_gqa,
            )

    def _forward_mixed_fa2(
        self,
        flash_attn_varlen_func,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> None:
        """Attention for a batch that mixes decodes and prefills, as two kinds of FA2 calls.

        FA2 splits the KV only on its single-token path and rejects ``num_splits > 1`` for
        paged varlen with longer queries. One call over a mixed batch therefore reads each
        decode request's whole KV with one CTA per head: 3.3-5.4 ms per layer for
        Qwen3.5-35B-A3B TP2 replay steps on A100 vs 0.38 ms for a pure-decode step. The
        runner orders decodes first (``reorder_batch_threshold = 1``), so request
        ``i < num_decodes`` owns token ``i``. Decodes run on the single-token path, in one
        call with ``_DECODE_NUM_SPLITS`` or else in chunks small enough for FA2's split
        heuristic (it does not split a batch of 256 padded requests). The prefills then run
        with the decodes as empty queries. A100, 31 decodes at 15-45K keys plus a 512-token
        prefill: 6.5 -> 2.3 ms per layer.
        """
        num_reqs = cu_seqlens_q.shape[0] - 1
        is_decode = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) == 1
        decode_seqused_k = torch.where(is_decode, seqused_k, 0)
        one_token_each = torch.arange(
            num_reqs + 1, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device
        )
        # Prefill requests get no keys here; their rows are rewritten below.
        if _DECODE_NUM_SPLITS:
            _fa2_single_token(
                query[:num_reqs],
                key_cache,
                value_cache,
                output[:num_reqs],
                one_token_each,
                decode_seqused_k,
                block_table,
                max_seqlen_k,
                self.scale,
            )
        else:
            for start in range(0, num_reqs, _DECODE_CHUNK):
                end = min(start + _DECODE_CHUNK, num_reqs)
                flash_attn_varlen_func(
                    q=query[start:end],
                    k=key_cache,
                    v=value_cache,
                    out=output[start:end],
                    cu_seqlens_q=one_token_each[: end - start + 1],
                    max_seqlen_q=1,
                    seqused_k=decode_seqused_k[start:end],
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=block_table[start:end],
                    fa_version=2,
                    num_splits=0,
                )
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=torch.maximum(cu_seqlens_q, is_decode.sum(dtype=torch.int32)),
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=block_table,
            fa_version=2,
            num_splits=0,
        )


class TorchTitanVarlenInnerAttentionDiffKVImpl(TorchTitanVarlenInnerAttentionImpl):
    """PyTorch varlen implementation using vLLM's unequal K/V cache writer."""

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        return FlashAttentionDiffKVImpl.do_kv_cache_update(
            self, layer, key, value, kv_cache, slot_mapping
        )


class _VLLMAttention(Attention, Module):
    """vLLM attention that participates in the TorchTitan Module protocol."""

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # Set vLLM's 4 attention scales to their default 1.0 now: the model is built empty,
        # and the weight-sync buffers copy these values before vLLM sets them.
        set_default_quant_scales(self)


class VLLMInnerAttention(InnerAttention):
    """Adapter from TorchTitan tensor layout to ``vllm.Attention``.

    vLLM's ``Attention`` layer manages KV-cache and paged attention internally,
    and expects ``(num_tokens, num_heads, head_dim)`` inputs.

    Used as ``inner_attention`` in GQAttention via Config-based construction.
    """

    # vLLM requires a unique prefix per Attention layer for
    # static_forward_context registration.
    # TODO: Pass layer_id through the build chain instead of using a
    # global counter. The counter breaks with pipeline parallelism
    # where layers are built on different ranks.
    _layer_counter: itertools.count = itertools.count()
    vllm_attn_scale_buffer_names: ClassVar[tuple[str, ...]] = (
        "_k_scale",
        "_prob_scale",
        "_q_scale",
        "_v_scale",
    )

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        hidden_size: int
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

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_degree = vllm_config.parallel_config.tensor_parallel_size

        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads

        if num_kv_heads < tp_degree:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be >= "
                f"tensor_parallel_size ({tp_degree})"
            )
        if num_kv_heads % tp_degree != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )
        if num_heads % tp_degree != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )

        num_heads = num_heads // tp_degree
        num_kv_heads = num_kv_heads // tp_degree
        head_dim = config.head_dim
        value_head_dim = (
            config.value_head_dim if config.value_head_dim is not None else head_dim
        )
        scale = config.scale if config.scale is not None else head_dim**-0.5

        self.hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        # TODO: This need to be compatible with Pipeline Parallelism
        layer_id = next(VLLMInnerAttention._layer_counter)
        diff_kv_kwargs: dict[str, Any] = {}
        if value_head_dim != head_dim:
            FlashAttentionDiffKVBackend.set_head_size_v(value_head_dim)
            diff_kv_kwargs = {
                "head_size_v": value_head_dim,
                "attn_backend": TorchTitanVarlenInnerAttentionDiffKVBackend,
            }
        self.vllm_attn = _VLLMAttention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=config.sliding_window_size,
            prefix=f"model.layers.{layer_id}.attention.inner_attention",
            **diff_kv_kwargs,
        )
        # Set sharding config for the inner vLLM attention module, which has scale
        # buffers that need to be replicated on all ranks in the dense mesh
        replicated_dense = dense_param_placement(tp=spmd.R)
        self.vllm_attn._sharding_config = ShardingConfig(
            state_shardings={
                name: replicated_dense for name in self.vllm_attn_scale_buffer_names
            },
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
        """Run vLLM paged attention on local (non-DTensor) tensors.

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
                "VLLMInnerAttention does not support attention_metadata; vLLM "
                "manages causal masking and the KV-cache internally."
            )

        if scale is not None and scale != self.scale:
            raise ValueError(
                f"Model attention scale {scale} != vLLM scale {self.scale}"
            )

        if self.value_head_dim != self.head_dim:
            # V may be a strided view (e.g. MLA splits it from a fused per-head
            # KV projection), but vLLM's cache writer assumes V heads are packed
            # with stride value_head_dim, so repack before updating the cache.
            v_THV = v_THV.contiguous()
        out_TD = self.vllm_attn(q_THK, k_THK, v_THV)

        # vLLM's flash attention backend may pad the token count (e.g.
        # round up to an even number), which introduces a new symbolic
        # shape under torch.compile.  Narrow to trim this padding.
        num_tokens = q_THK.shape[0]
        out_TD = out_TD.narrow(0, 0, num_tokens)
        return out_TD.view(num_tokens, -1, self.value_head_dim)


def get_attention_dimensions(
    attention_config, model_dim: int
) -> tuple[int, int, int, int]:
    """Return query heads, KV heads, Q/K head dim, and value head dim."""
    num_heads = attention_config.n_heads
    num_kv_heads = getattr(attention_config, "n_kv_heads", None) or num_heads
    if hasattr(attention_config, "qk_nope_head_dim"):
        head_dim = attention_config.qk_nope_head_dim + attention_config.qk_rope_head_dim
        value_head_dim = attention_config.v_head_dim
    else:
        head_dim = getattr(attention_config, "head_dim", None)
        head_dim = head_dim if head_dim is not None else model_dim // num_heads
        value_head_dim = head_dim
    return num_heads, num_kv_heads, head_dim, value_head_dim
