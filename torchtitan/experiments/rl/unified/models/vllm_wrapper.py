# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base wrapper for TorchTitan models to work with vLLM V1 engine.

This module provides TorchTitanVLLMModel: Core model class that adapts
TorchTitan models for vLLM.
"""

import os
import time
from functools import partial

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
    create_job_config_from_vllm_config,
    create_parallel_dims_from_vllm_config,
)

from torchtitan.experiments.rl.unified.models.utils import replace_with_vllm_attention
from torchtitan.models.qwen3.model.model import precompute_rope_cache
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.protocols.train_spec import ParallelizeFunction

from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)

# Enable timing instrumentation via environment variable
# Check at runtime to handle worker process spawning
def _is_timing_enabled() -> bool:
    return os.environ.get("TORCHTITAN_VLLM_TIMING", "0") == "1"


# Global timing stats accumulator for TorchTitan model
_tt_timing_stats: dict[str, list[float]] = {}


def _log_timing(name: str, elapsed_ms: float, phase: str = ""):
    """Log timing information.

    Args:
        name: The operation name
        elapsed_ms: Time in milliseconds
        phase: Optional phase identifier (prefill/decode)
    """
    if _is_timing_enabled():
        full_name = f"{phase}.{name}" if phase else name
        logger.info(f"[TORCHTITAN_TIMING] {full_name}: {elapsed_ms:.3f} ms")
        # Accumulate stats
        if full_name not in _tt_timing_stats:
            _tt_timing_stats[full_name] = []
        _tt_timing_stats[full_name].append(elapsed_ms)


def get_torchtitan_timing_stats() -> dict[str, dict[str, float]]:
    """Get aggregated timing statistics for TorchTitan model."""
    result = {}
    for name, times in _tt_timing_stats.items():
        if times:
            result[name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
    return result


def reset_torchtitan_timing_stats():
    """Reset all TorchTitan timing statistics."""
    global _tt_timing_stats
    _tt_timing_stats = {}


def print_torchtitan_timing_summary():
    """Print a summary of TorchTitan timing statistics."""
    stats = get_torchtitan_timing_stats()
    if not stats:
        logger.info("[TORCHTITAN_TIMING] No timing data collected")
        return

    logger.info("\n" + "=" * 80)
    logger.info("[TORCHTITAN_TIMING] TIMING SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"{'Operation':<50} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
    )
    logger.info("-" * 100)

    # Sort by total time descending
    for name, data in sorted(stats.items(), key=lambda x: -x[1]["total_ms"]):
        logger.info(
            f"{name:<50} {data['count']:>8} {data['total_ms']:>12.2f} "
            f"{data['avg_ms']:>10.3f} {data['min_ms']:>10.3f} {data['max_ms']:>10.3f}"
        )
    logger.info("=" * 80)


class TorchTitanVLLMModelWrapper(nn.Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models. Implemented
    required interface required by vLLM Engine.
    Doc: https://docs.vllm.ai/en/latest/contributing/model/basic/
    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Parallelism setup and DTensor conversion between torchtitan and vLLM
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_cls: type[ModelProtocol],
        model_args: BaseModelArgs,
        state_dict_adapter: type[BaseStateDictAdapter],
        parallelize_fn: ParallelizeFunction,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components
        self.model_cls = model_cls
        self.state_dict_adapter = state_dict_adapter
        self.parallelize_fn = parallelize_fn

        # Use TorchTitan model args directly (no HF config mapping)
        self.config = model_args
        logger.info(f"Creating {self.model_cls.__name__} with config: {model_args}")
        self.model = self.model_cls(model_args)

        # Setup RoPE cache extension function if provided
        self.rope_cache_extension_fn = partial(
            precompute_rope_cache,
            dim=self.config.head_dim,
            base=self.config.rope_theta,
        )

        # Create ParallelDims and JobConfig from vLLM config at runtime
        # vLLM config contains the tensor_parallel_size from command-line args
        # and this will be consistent across all worker processes
        self.parallel_dims = create_parallel_dims_from_vllm_config(vllm_config)
        self.parallel_config = create_job_config_from_vllm_config(
            vllm_config=vllm_config,
        )
        # Replace attention with vLLM paged attention
        tp_size = self.parallel_dims.tp
        if tp_size > 1:
            assert (
                model_args.n_heads % tp_size == 0
            ), "Only support when n_heads can be divided by tp_size"

        replace_with_vllm_attention(self.model, tp_degree=tp_size)

        # NOTE: We need to apply parallelize within model.__init__ because vllm
        # doesn't separate model creation and parallelism application and instead
        # requires parallelization to be done inside model constructor.
        self.model = parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            job_config=self.parallel_config,
        )

    def _extend_rope_cache_if_needed(
        self, rope_cache: torch.Tensor, max_position: int
    ) -> torch.Tensor:
        """
        Extend RoPE cache if needed during vLLM profiling stage.

        Args:
            rope_cache: Current RoPE cache tensor
            max_position: Maximum position index needed

        Returns:
            Extended RoPE cache if needed, otherwise original cache
        """
        required_len = max_position + 1

        # No extension needed
        if required_len <= rope_cache.shape[0]:
            return rope_cache

        # If no extension function provided, return original cache
        if self.rope_cache_extension_fn is None:
            logger.warning(
                f"RoPE cache extension needed (required_len={required_len}, "
                f"current_len={rope_cache.shape[0]}) but no rope_cache_extension_fn provided. "
                "Returning original cache."
            )
            return rope_cache

        # Handle DTensor case
        is_dtensor = isinstance(rope_cache, DTensor)
        if is_dtensor:
            device_mesh = rope_cache.device_mesh
            local_rope_cache = rope_cache.to_local()
            device = local_rope_cache.device
            dtype = local_rope_cache.dtype
        else:
            device = rope_cache.device
            dtype = rope_cache.dtype

        # Use provided extension function
        try:
            extended_cache = self.rope_cache_extension_fn(self.config, required_len)
            extended_cache = extended_cache.to(device=device, dtype=dtype)
        except Exception as e:
            logger.warning(
                f"Failed to extend RoPE cache using rope_cache_extension_fn: {e}. "
                "Returning original cache."
            )
            return rope_cache

        # Convert back to DTensor if needed
        if is_dtensor:
            rope_cache = DTensor.from_local(
                extended_cache,
                device_mesh=device_mesh,
                placements=[Replicate()],
            )
        else:
            rope_cache = extended_cache

        return rope_cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs [total_tokens] (1D varlen format)
            positions: Position indices [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        forward_start = time.perf_counter()

        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Determine phase: prefill if num_tokens > 1, decode otherwise
        num_tokens = input_ids.shape[0]
        phase = "prefill" if num_tokens > 1 else "decode"
        if _is_timing_enabled():
            logger.info(f"[TORCHTITAN_TIMING] Phase: {phase}, num_tokens: {num_tokens}")

        # Convert vLLM interface to TorchTitan interface
        # vLLM: [total_tokens] → TorchTitan: [batch_size, seq_len]
        tokens_2d = input_ids.unsqueeze(0)

        # Get embeddings
        embed_start = time.perf_counter()
        h = self.model.tok_embeddings(tokens_2d)
        torch.cuda.synchronize()
        _log_timing(
            "forward.tok_embeddings", (time.perf_counter() - embed_start) * 1000, phase
        )

        # Get RoPE cache (handle model-specific attribute names)
        # Use hasattr to avoid ambiguous boolean value error with tensors
        if hasattr(self.model, "rope_cache"):
            rope_attr = self.model.rope_cache
        elif hasattr(self.model, "freqs_cis"):
            rope_attr = self.model.freqs_cis
        else:
            rope_attr = None

        # Extend RoPE cache if needed (vLLM profiling may use 2x max_seq_len)
        if positions is not None:
            max_position = positions.max().item()
        else:
            max_position = 0

        rope_cache = self._extend_rope_cache_if_needed(rope_attr, max_position)
        positions = positions.unsqueeze(0)

        # Pass through transformer layers
        layers_start = time.perf_counter()
        for i, layer in enumerate(self.model.layers.values()):
            layer_start = time.perf_counter()
            h = layer(h, rope_cache, attention_masks=None, positions=positions)
            if (
                _is_timing_enabled() and i < 3
            ):  # Only log first 3 layers to avoid too much output
                torch.cuda.synchronize()
                _log_timing(
                    f"forward.layer_{i}",
                    (time.perf_counter() - layer_start) * 1000,
                    phase,
                )
        torch.cuda.synchronize()
        _log_timing(
            "forward.transformer_layers",
            (time.perf_counter() - layers_start) * 1000,
            phase,
        )

        # When parallelism is applied, get full tensor before return to vLLM Engine
        # The original placement is Shard(1) (shard on sequence dimension, as it will prepare for sequence parallel in `self.norm`).
        # vLLM's engine expects plain, non-distributed tensors to slice the last token for each request.
        if isinstance(h, DTensor):
            dtensor_start = time.perf_counter()
            h = h.full_tensor()
            torch.cuda.synchronize()
            _log_timing(
                "forward.dtensor_to_full",
                (time.perf_counter() - dtensor_start) * 1000,
                phase,
            )

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            batch_size, seq_len, hidden_size = h.shape
            h = h.view(batch_size * seq_len, hidden_size)

        torch.cuda.synchronize()
        _log_timing(
            "forward.total", (time.perf_counter() - forward_start) * 1000, phase
        )
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        compute_logits_start = time.perf_counter()

        # Determine phase based on hidden states size
        # For compute_logits, typically 1 token per request for decode, more for prefill
        num_tokens = hidden_states.shape[0]
        phase = "prefill" if num_tokens > 1 else "decode"

        # When TP is applied, we return the full tensor (plain tensor) to vLLM engine
        # at the end of TorchTitanVLLMModelWrapper.forward().
        # We need to wrap the input from vLLM engine back to DTensor with Replicate() placement.
        if self.parallel_dims.tp_enabled:
            dtensor_start = time.perf_counter()
            hidden_states = DTensor.from_local(
                hidden_states,
                device_mesh=self.parallel_dims.get_mesh("tp"),
                placements=[
                    Replicate(),
                ],
            )
            _log_timing(
                "compute_logits.dtensor_from_local",
                (time.perf_counter() - dtensor_start) * 1000,
                phase,
            )

        norm_start = time.perf_counter()
        h = self.model.norm(hidden_states)
        torch.cuda.synchronize()
        _log_timing(
            "compute_logits.norm", (time.perf_counter() - norm_start) * 1000, phase
        )

        output_start = time.perf_counter()
        logits = self.model.output(h)
        torch.cuda.synchronize()
        _log_timing(
            "compute_logits.output", (time.perf_counter() - output_start) * 1000, phase
        )

        _log_timing(
            "compute_logits.total",
            (time.perf_counter() - compute_logits_start) * 1000,
            phase,
        )
        return logits

    def load_weights(self, weights_iter):
        """
        Load weights from HF checkpoint using the provided state dict adapter.
        vLLM engine would call this function to load model weights.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """
        # Collect weights from iterator
        hf_state_dict = {}
        for name, tensor in weights_iter:
            hf_state_dict[name] = tensor

        # Use adapter to convert HF → TorchTitan format
        adapter = self.state_dict_adapter(
            model_args=self.config,
            hf_assets_path=None,
        )

        torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        model_state_dict = {k: v for k, v in self.model.state_dict().items()}

        # Convert to DTensor if target is DTensor
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )

        # Load state dict
        set_model_state_dict(
            model=self.model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=False),
        )

        loaded_params = {f"model.{name}" for name in torchtitan_state_dict.keys()}

        return loaded_params
