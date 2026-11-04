# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registers TorchTitan models with SGLang.

SGLang builds its model config from an HF ``config.json`` and resolves the model
class from the config's ``architectures`` through its ``ModelRegistry``. The
TorchTitan model config is the source of truth: ``model_config_to_sglang_hf_config``
produces the HF-shaped config SGLang reads before it builds the model (KV pool
sizing, context length, vocabulary), as vLLM's TorchTitan config parser does, and
``register_to_sglang`` registers a model class that closes over the TorchTitan
config.
"""

from __future__ import annotations

from typing import Any

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.unified_model import get_attention_dimensions

# Architecture name the TorchTitan model class is registered under in SGLang.
SGLANG_MODEL_NAME = "TorchTitanForCausalLM"


def model_config_to_sglang_hf_config(cfg: Decoder.Config) -> dict[str, Any]:
    """Return the HF config SGLang reads for ``cfg``, written to its ``config.json``.

    SGLang sizes the KV pool from the layer count, KV heads and head dims, caps the
    context at ``max_position_embeddings``, and resolves the model class from
    ``architectures``. ``model_type`` only selects the HF config class that parses
    these generic fields; the TorchTitan wrapper builds the model itself.
    """
    attention = cfg.first_full_attention
    if attention is None:
        raise ValueError(
            f"Model config {type(cfg).__qualname__} has no full attention layer"
        )
    num_heads, num_kv_heads, head_dim, value_head_dim = get_attention_dimensions(
        attention, cfg.dim
    )
    hf_config: dict[str, Any] = {
        "architectures": [SGLANG_MODEL_NAME],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "v_head_dim": value_head_dim,
        "num_hidden_layers": len(cfg.layers),
        "max_position_embeddings": cfg.max_context_length,
        "tie_word_embeddings": getattr(cfg, "enable_weight_tying", False),
        "torch_dtype": "bfloat16",
    }
    return hf_config


def register_to_sglang(
    model_config: Decoder.Config,
    *,
    parallelism: InferenceParallelismConfig,
    checkpointer_config: CheckpointManager.Config | None,
) -> None:
    """Register the TorchTitan model class with SGLang's ``ModelRegistry``.

    Must run in the process that builds SGLang's scheduler, before the build.

    Args:
        model_config: TorchTitan decoder model config.
        parallelism: Inference parallelism; the wrapper builds the TorchTitan
            parallelism context from it.
        checkpointer_config: Optional CheckpointManager configuration for initial
            weight loading. ``None`` for the RL loop, where weights arrive from
            TorchStore.
    """
    from sglang.srt.models.registry import ModelRegistry

    # Imports SGLang layers; deferred so importing this module stays cheap.
    from torchtitan.rl.generator.sglang.model_wrapper import SGLangModelWrapper

    class SGLangModelFromSpec(SGLangModelWrapper):
        def __init__(self, *, config, quant_config=None, prefix: str = "", **kwargs):
            if quant_config is not None:
                raise ValueError("Quantized TorchTitan models are not supported")
            super().__init__(
                model_config=model_config,
                parallelism=parallelism,
                checkpointer_config=checkpointer_config,
            )

    SGLangModelFromSpec.__name__ = SGLANG_MODEL_NAME
    SGLangModelFromSpec.__qualname__ = SGLANG_MODEL_NAME
    ModelRegistry.models[SGLANG_MODEL_NAME] = SGLangModelFromSpec
