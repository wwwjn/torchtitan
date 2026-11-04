# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wrapper that runs a TorchTitan model as an SGLang model.

Shape legend: T = tokens in the step, B = requests in the step, D = model dim,
V = vocabulary size.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import spmd_types as spmd
import torch
import torch.distributed as dist
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._state_dict_utils import _create_cpu_state_dict
from torch.distributed.tensor import DTensor, Replicate

# Registers the "loss" local-compile region. The generator shares the trainer's
# model config, whose local_compile_regions may name it, but never builds a loss.
from torchtitan.components import loss as _loss  # noqa: F401
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import TrainingConfig
from torchtitan.distributed import maybe_apply_numa_binding
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import Attention, LinearAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.sglang.attention import SGLangInnerAttention
from torchtitan.rl.generator.unified_model import (
    get_attention_dimensions,
    load_initial_weights,
    load_state_dict_by_fsdp_group,
    unshard_and_free_sharded_params,
)

logger = logging.getLogger(__name__)


def _replace_sglang_layer_configs(model_config: Decoder.Config) -> Decoder.Config:
    """Replace inner attention configs with SGLang paged-attention variants."""
    new_layers = []
    for layer_idx, layer_cfg in enumerate(model_config.layers):
        attention_cfg = getattr(layer_cfg, "attention", None)
        if isinstance(attention_cfg, LinearAttention.Config):
            raise NotImplementedError(
                f"Layer {layer_idx} uses {type(attention_cfg).__qualname__}; the SGLang "
                "engine does not support linear attention yet."
            )
        if isinstance(attention_cfg, Attention.Config):
            (
                num_heads,
                num_kv_heads,
                head_dim,
                value_head_dim,
            ) = get_attention_dimensions(attention_cfg, model_config.dim)
            sglang_attention_cfg = SGLangInnerAttention.Config(
                layer_id=layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                value_head_dim=value_head_dim,
                scale=attention_cfg.softmax_scale,
                sliding_window_size=getattr(attention_cfg, "sliding_window_size", None),
                sharding_config=attention_cfg.inner_attention.sharding_config,
            )
            layer_cfg = dataclasses.replace(
                layer_cfg,
                attention=dataclasses.replace(
                    attention_cfg, inner_attention=sglang_attention_cfg
                ),
            )
        new_layers.append(layer_cfg)
    return dataclasses.replace(model_config, layers=new_layers)


class SGLangModelWrapper(Module):
    """Runs a TorchTitan model behind SGLang's model interface.

    SGLang's model loader constructs this class with the HF-shaped config and calls
    ``load_weights``; ``ModelRunner`` calls ``forward(input_ids, positions,
    forward_batch)`` and samples from the returned ``LogitsProcessorOutput``. The
    wrapper builds the TorchTitan model on the meta device, parallelizes it with the
    generator's layout, and loads initial weights through TorchTitan's checkpointer,
    as the vLLM wrapper does.
    """

    def __init__(
        self,
        *,
        model_config: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        checkpointer_config: CheckpointManager.Config | None,
    ):
        super().__init__()
        training_parallelism = parallelism.to_training()
        self.config = _replace_sglang_layer_configs(copy.deepcopy(model_config))
        self.parallelism_context = ParallelismContext(
            dp_replicate=training_parallelism.data_parallel_replicate_degree,
            dp_shard=training_parallelism.data_parallel_shard_degree,
            cp=training_parallelism.context_parallel_degree,
            tp=training_parallelism.tensor_parallel_degree,
            pp=training_parallelism.pipeline_parallel_degree,
            ep=training_parallelism.expert_parallel_degree,
            world_size=dist.get_world_size(),
            enable_sequence_parallel=training_parallelism.enable_sequence_parallel,
        )

        with self.parallelism_context.activate_spmd(), torch.device("meta"):
            self.model = self.config.build()
        self.model._skip_lm_head = True
        if getattr(self.model, "vision_encoder", None) is not None:
            self.model.vision_encoder = None

        self.model = self.model.parallelize(
            parallelism_context=self.parallelism_context,
            training=TrainingConfig(),
            parallelism=training_parallelism,
            local_compile_regions=self.config.local_compile_regions,
            activation_remat=None,
            dump_folder="",
        )
        # Preserve compute storage addresses across model state dict loads for CUDA graphs.
        assert isinstance(self.model, FSDPModule)
        self.model.set_keep_unsharded_storage(True)

        device = torch.device("cuda", torch.cuda.current_device())
        self.model.to_empty(device=device)
        with torch.no_grad(), self.parallelism_context.activate_spmd():
            self.model.init_weights(buffer_device=None)
        load_initial_weights(self.model, self.config, checkpointer_config)

        # Build pinned CPU receive buffers for model state dict loads while the model
        # is still sharded. Bind first so first-touch places them on the NUMA node
        # local to this rank's GPU.
        maybe_apply_numa_binding(torch.cuda.current_device(), "cuda")
        with torch.device("cpu"):
            self._prefetched_model_state_dict = _create_cpu_state_dict(
                self.model.state_dict(), pin_memory=True
            )

        # Unshard now so SGLang profiles GPU memory with the representation the
        # forward uses before it sizes the KV pool.
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                unshard_and_free_sharded_params(module)

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self._prefetched_model_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        """Copy new weights into ``self.model`` (see ``load_state_dict_by_fsdp_group``).

        Args:
            state_dict: Keyed like ``self.model.state_dict()``, e.g. the generator's
                pinned CPU copy.
        """
        return load_state_dict_by_fsdp_group(self.model, state_dict)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """SGLang model API: a no-op, since ``__init__`` already loaded the weights.

        The iterator is not consumed, so SGLang reads no checkpoint shards.
        """
        del weights

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        **kwargs,
    ) -> LogitsProcessorOutput:
        """SGLang model API: return next-token logits for each request in the step.

        Args:
            input_ids: ``[T]`` token ids of the step.
            positions: ``[T]`` position of each token in its sequence.
            forward_batch: SGLang's ``ForwardBatch`` for the step.

        Returns:
            ``LogitsProcessorOutput`` with ``next_token_logits`` of shape ``[B, V]``.
        """
        # An idle step still runs the model: with data parallelism, expert
        # parallelism's all-to-alls span every DP rank, so a rank without requests
        # runs its (possibly empty) padding rows through the MoE layers too.
        num_tokens = input_ids.shape[0]
        num_padding_tokens = self._num_padding_tokens(num_tokens)
        if num_padding_tokens:
            input_ids = torch.nn.functional.pad(input_ids, (0, num_padding_tokens))
            positions = torch.nn.functional.pad(positions, (0, num_padding_tokens))
        with set_forward_batch(forward_batch), self.parallelism_context.activate_spmd():
            h_TD = self.model(input_ids, attention_metadata=None, positions=positions)
        # Inference disables sequence parallelism, so final hidden states are
        # replicated before logits.
        if isinstance(h_TD, DTensor):
            assert all(isinstance(p, Replicate) for p in h_TD.placements)
            h_TD = h_TD._local_tensor
        h_TD = h_TD[:num_tokens]

        if forward_batch.forward_mode.is_extend():
            # One row per request: its last token predicts the next one.
            last_index_B = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int64)
                - 1
            )
            h_BD = h_TD[last_index_B]
        else:
            h_BD = h_TD
        if h_BD.shape[0] == 0:
            # An idle step samples nothing. Every rank of a TP group shares its DP
            # rank's idle step, so skipping the logits all-gather is collective.
            return LogitsProcessorOutput(
                next_token_logits=h_BD.new_empty(
                    0, self.config.vocab_size, dtype=torch.float32
                )
            )
        return LogitsProcessorOutput(next_token_logits=self.compute_logits(h_BD))

    def _num_padding_tokens(self, num_tokens: int) -> int:
        """Rows to append before the TorchTitan forward.

        With EP and TP, the MoE block splits each step's tokens evenly across the
        TP ranks of the expert group (as vLLM's TorchTitan runner pads for), and an
        idle step still needs rows to run. The padding rows run through the model,
        attention gives them zeros and writes no KV, and the forward drops them
        before the logits.
        """
        tp = self.parallelism_context.tp
        expert_sequence_parallel = self.parallelism_context.ep_enabled and tp > 1
        if num_tokens == 0:
            # An idle step (data parallelism) runs padding rows only: the model's
            # reshapes need at least one row, and expert SP one per TP rank.
            return tp if expert_sequence_parallel else 1
        return -num_tokens % tp if expert_sequence_parallel else 0

    def compute_logits(self, h_BD: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits of ``h_BD``, gathered over tensor parallelism."""
        with self.parallelism_context.activate_spmd():
            logits_BV = self.model.lm_head(h_BD)
            # lm_head returns vocab-sharded logits under TP.
            if self.parallelism_context.tp_enabled:
                mesh = current_spmd_mesh()
                assert mesh is not None
                logits_BV = spmd.redistribute(
                    logits_BV,
                    mesh.get_group("tp"),
                    src=spmd.S(-1),
                    dst=spmd.R,
                    backward_options={"op_dtype": logits_BV.dtype},
                )
        return logits_BV
