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
from types import SimpleNamespace
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
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import Attention, LinearAttention
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
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
        if isinstance(attention_cfg, GDN.Config):
            from torchtitan.rl.generator.sglang.gdn import SGLangInnerGDN

            layer_cfg = dataclasses.replace(
                layer_cfg,
                attention=dataclasses.replace(
                    attention_cfg,
                    inner_attention=SGLangInnerGDN.Config(
                        layer_idx=layer_idx,
                        num_k_heads=(
                            attention_cfg.in_proj_q.out_features
                            // attention_cfg.key_head_dim
                        ),
                        num_v_heads=(
                            attention_cfg.in_proj_v.out_features
                            // attention_cfg.value_head_dim
                        ),
                        head_k_dim=attention_cfg.key_head_dim,
                        head_v_dim=attention_cfg.value_head_dim,
                        conv_kernel_size=attention_cfg.conv_kernel_size,
                        sharding_config=attention_cfg.inner_attention.sharding_config,
                    ),
                ),
            )
        elif isinstance(attention_cfg, LinearAttention.Config):
            raise NotImplementedError(
                f"Layer {layer_idx} uses {type(attention_cfg).__qualname__}; the SGLang "
                "engine supports GDN linear attention only."
            )
        elif isinstance(attention_cfg, Attention.Config):
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


# vLLM's FlashInfer all-reduce size caps in MiB, by SM version and TP size
# (vllm.compilation.passes.fusion.allreduce_rms_fusion.FI_ALLREDUCE_FUSION_MAX_SIZE_MB).
_FLASHINFER_ALL_REDUCE_MAX_MB = {
    100: {2: 64, 4: 32, 8: 1},
    103: {2: 64, 4: 64, 8: 4},
}


def _build_flashinfer_all_reduce(cpu_group: dist.ProcessGroup, hidden_dim: int):
    """FlashInfer's MNNVL all-reduce workspace for ``[T, hidden_dim]`` bf16 inputs.

    The all-reduce vLLM runs for tensor parallelism on SM100+ (vLLM's
    ``FlashInferAllReduce``): one-shot and two-shot kernels on an NVLink
    multicast workspace, CUDA graph capturable. Returns ``(workspace, max_tokens)``,
    or None where it is unavailable (no FlashInfer, other GPUs, no multicast).
    """
    import os
    import random

    try:
        import flashinfer.comm as flashinfer_comm
        from flashinfer.comm.mnnvl import TorchDistBackend
    except ImportError:
        return None
    major, minor = torch.cuda.get_device_capability()
    world_size = dist.get_world_size(cpu_group)
    max_mb = _FLASHINFER_ALL_REDUCE_MAX_MB.get(major * 10 + minor, {}).get(world_size)
    if not max_mb:
        return None
    max_tokens = int(max_mb * 2**20) // (hidden_dim * torch.bfloat16.itemsize)
    # The workspace names its shared buffers from Python's random module, which
    # every rank seeds alike.
    random_state = random.getstate()
    random.seed(int.from_bytes(os.urandom(16), byteorder="big"))
    try:
        workspace = flashinfer_comm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=dist.get_rank(cpu_group),
            max_token_num=max_tokens,
            hidden_dim=hidden_dim,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(group=cpu_group),
            group=cpu_group,
        )
    except Exception as error:
        logger.warning("FlashInfer all-reduce is unavailable: %s", error)
        return None
    finally:
        random.setstate(random_state)
    if not getattr(workspace, "mc_ptr", 0):
        workspace.destroy()
        return None
    return workspace, max_tokens


def _route_tp_all_reduce_through_custom_ar(
    tp_group: dist.ProcessGroup, hidden_dim: int
):
    """Run the model's tensor-parallel sum all-reduces on the all-reduce vLLM uses.

    vLLM's wrapper routes them to vLLM's TP all-reduce in perf mode
    (``_patch_vllm_all_reduce``); NCCL's ring all-reduce is latency-bound at decode
    sizes. Like vLLM, ``[T, hidden_dim]`` inputs take FlashInfer's MNNVL all-reduce
    where available (``_build_flashinfer_all_reduce``), others SGLang's custom
    all-reduce. Both are built on SGLang's attention-TP group, which has the TP
    ranks of this DP rank. The custom all-reduce always takes its unregistered path: it copies
    the input into a buffer exchanged once over IPC, so CUDA graphs capture it
    without registering graph memory over IPC, which expandable segments (VMM)
    memory does not support. A captured call still reads its peers' buffer
    addresses from a table that ``register_graph_buffers`` fills after capture
    (``SGLangModelWrapper.register_cuda_graph_buffers``). Inputs it does not take
    (over its size limit) and other groups keep the original collective. Returns
    the custom all-reduce, or None when unavailable. Two entry points:

    1. ``torch.ops._c10d_functional.all_reduce``, which DTensor's Partial ->
       Replicate redistribution calls.
    2. ``spmd.redistribute`` from Partial to Replicate or Invariant without grad,
       which issues an in-place ``dist.all_reduce`` the first cannot see.
    """
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    attn_tp_group = get_attn_tp_group()
    if attn_tp_group.ranks != dist.get_process_group_ranks(tp_group):
        raise RuntimeError(
            f"SGLang's attention-TP ranks {attn_tp_group.ranks} differ from the "
            f"model's TP ranks {dist.get_process_group_ranks(tp_group)}"
        )
    custom_ar = CustomAllreduce(
        group=attn_tp_group.cpu_group, device=torch.cuda.current_device()
    )
    flashinfer_ar = _build_flashinfer_all_reduce(attn_tp_group.cpu_group, hidden_dim)
    if custom_ar.disabled and flashinfer_ar is None:
        logger.warning("No custom TP all-reduce is available; TP uses NCCL")
        return None
    if flashinfer_ar is not None:
        import flashinfer.comm as flashinfer_comm

    def reduce(input: torch.Tensor, group_name) -> torch.Tensor | None:
        """The sum over TP of ``input``, or None to leave it to the original op."""
        if group_name != tp_group.group_name:
            return None
        if (
            flashinfer_ar is not None
            and input.dtype == torch.bfloat16
            and input.is_contiguous()
            and input.shape[-1] == hidden_dim
        ):
            workspace, max_tokens = flashinfer_ar
            num_tokens = input.numel() // hidden_dim
            if num_tokens <= max_tokens and workspace.is_buffer_size_sufficient(
                tp_size=tp_group.size(),
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                dtype=input.dtype,
            ):
                out = flashinfer_comm.allreduce_fusion(
                    input=input.view(num_tokens, hidden_dim),
                    workspace=workspace,
                    pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
                    launch_with_pdl=True,
                    # vLLM's PDL_ADVANCE_LAUNCH_TOKENS.
                    trigger_completion_at_end=num_tokens > 16,
                )
                return out.view(input.shape)
        if not custom_ar.disabled and custom_ar.should_custom_ar(input):
            return custom_ar._all_reduce_impl(input, registered=False)
        return None

    c10d = torch.ops._c10d_functional
    original_all_reduce = c10d.all_reduce

    def all_reduce(input, reduce_op, group_name, config=None):
        if not config and reduce_op == "sum":
            out = reduce(input, group_name)
            if out is not None:
                return out
        return original_all_reduce(input, reduce_op, group_name, config=config)

    c10d.all_reduce = all_reduce

    original_redistribute = spmd.redistribute

    def redistribute(x, group, *, src, dst, **kwargs):
        if src == spmd.P and dst in (spmd.R, spmd.I) and not torch.is_grad_enabled():
            out = reduce(x, getattr(group, "group_name", None))
            if out is not None:
                return out
        return original_redistribute(x, group, src=src, dst=dst, **kwargs)

    spmd.redistribute = redistribute
    return None if custom_ar.disabled else custom_ar


def _max_prefill_graph_tokens() -> int | None:
    """The largest token bucket SGLang captures prefill CUDA graphs for, if any."""
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    if server_args.disable_cuda_graph or server_args.cuda_graph_config is None:
        return None
    return max(server_args.cuda_graph_config.prefill.bs or [], default=None)


def _max_num_tokens_per_step(tp: int) -> int:
    """The most tokens one step runs on a DP rank, rounded up to a multiple of TP.

    A prefill step runs at most ``chunked_prefill_size`` tokens (a prefill CUDA
    graph at most its largest bucket), and a decode step one token per running
    request.
    """
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    if (
        server_args.chunked_prefill_size is None
        or server_args.chunked_prefill_size <= 0
    ):
        raise ValueError(
            "Dist-MoE needs chunked prefill, which bounds the tokens of a prefill step"
        )
    num_tokens = max(
        server_args.chunked_prefill_size,
        _max_prefill_graph_tokens() or 0,
        server_args.max_running_requests,
    )
    return -(-num_tokens // tp) * tp


class SGLangTransformerBody(torch.nn.Module):
    """The TorchTitan model's transformer body behind SGLang's layer-model interface.

    SGLang's prefill CUDA graphs capture the body of ``model.model``:
    ``forward(input_ids, positions, forward_batch)`` returns the step's hidden
    states, and the wrapper's forward computes the logits eagerly on top. SGLang
    finds the layers' attention through ``layers``.
    """

    def __init__(
        self,
        decoder: Decoder,
        parallelism_context: ParallelismContext,
        dist_moe_runtime: DistMoeRuntime | None = None,
        max_prefill_graph_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.parallelism_context = parallelism_context
        self._dist_moe_runtime = dist_moe_runtime
        self._max_prefill_graph_tokens = max_prefill_graph_tokens
        # SGLang's CUDA graph setup finds a full-attention layer's RadixAttention at
        # layer.self_attn.attn and a linear-attention layer at layer.linear_attn; it
        # captures prefill graphs only when it finds one for every layer.
        from torchtitan.rl.generator.sglang.gdn import SGLangInnerGDN

        self.layers = []
        for layer in decoder.layers.values():
            inner_attention = layer.attention.inner_attention
            if isinstance(inner_attention, SGLangInnerAttention):
                view = SimpleNamespace(
                    self_attn=SimpleNamespace(attn=inner_attention.radix_attn)
                )
            elif isinstance(inner_attention, SGLangInnerGDN):
                view = SimpleNamespace(linear_attn=inner_attention)
            else:
                view = SimpleNamespace()
            self.layers.append(view)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch
    ) -> torch.Tensor:
        """Return the hidden states ``[T, D]`` of the step's tokens.

        Args:
            input_ids: ``[T]`` token ids of the step.
            positions: ``[T]`` position of each token in its sequence.
            forward_batch: SGLang's ``ForwardBatch`` for the step.
        """
        # An idle step still runs the model: with data parallelism, expert
        # parallelism's all-to-alls span every DP rank, so a rank without requests
        # runs its (possibly empty) padding rows through the MoE layers too.
        num_tokens = input_ids.shape[0]
        num_padding_tokens = self._num_padding_tokens(num_tokens)
        if num_padding_tokens:
            input_ids = torch.nn.functional.pad(input_ids, (0, num_padding_tokens))
            positions = torch.nn.functional.pad(positions, (0, num_padding_tokens))
        model_kwargs = {}
        if self._dist_moe_runtime is not None:
            model_kwargs["padding_mask"] = self._padding_mask(
                forward_batch, input_ids.shape[0]
            )
            self._dist_moe_runtime.set_num_local_input_tokens_per_call(
                self._dist_moe_num_local_input_tokens(forward_batch, input_ids.shape[0])
            )
        with set_forward_batch(forward_batch), self.parallelism_context.activate_spmd():
            h_TD = self.decoder(
                input_ids, attention_metadata=None, positions=positions, **model_kwargs
            )
        # Inference disables sequence parallelism, so final hidden states are
        # replicated before logits.
        if isinstance(h_TD, DTensor):
            assert all(isinstance(p, Replicate) for p in h_TD.placements)
            h_TD = h_TD._local_tensor
        return h_TD[:num_tokens]

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

    @staticmethod
    def _padding_mask(forward_batch, num_rows: int) -> torch.Tensor:
        """``[num_rows]`` bool, true for the rows that are padding, without a host sync.

        Only the MoE reads this mask: Dist-MoE routes those rows to expert -1, so
        they cost no dispatch or expert work. SGLang never allocates KV slot 0 and
        points every padding row's ``out_cache_loc`` at it: the rows a CUDA graph
        replay pads its batch or token bucket with, and an idle DP rank's
        fabricated rows. Eager steps give KV slots to the real tokens only, so the
        rows past ``out_cache_loc`` (``_num_padding_tokens``) are padding too.
        """
        out_cache_loc = forward_batch.out_cache_loc
        return torch.nn.functional.pad(
            out_cache_loc == 0, (0, num_rows - out_cache_loc.shape[0]), value=True
        )

    def _dist_moe_num_local_input_tokens(self, forward_batch, num_rows: int) -> int:
        """Return the Dist-MoE row count every EP rank passes in this step.

        Every EP rank must pass the same count, and the routed tokens are split
        evenly over TP, so this is the step's row count over TP. A CUDA graph
        records the count of its capture.
        """
        tp = self.parallelism_context.tp
        if self.parallelism_context.dp_shard > 1:
            # The EP group spans the DP ranks, so every DP rank uses the largest
            # count (SGLang's data-parallel attention keeps each rank's own).
            if torch.cuda.is_current_stream_capturing():
                if forward_batch.forward_mode.is_extend():
                    # DP ranks replay prefill graphs of their own token buckets
                    # in the same step, so every prefill graph calls Dist-MoE at
                    # the largest bucket's size.
                    assert self._max_prefill_graph_tokens is not None
                    num_tokens = self._max_prefill_graph_tokens
                else:
                    # Every DP rank replays the decode graph of the largest DP
                    # rank's batch size.
                    num_tokens = num_rows
            else:
                num_tokens = max(forward_batch.global_num_tokens_cpu)
            num_rows = num_tokens + self._num_padding_tokens(num_tokens)
        # Otherwise DP = 1: the EP group is the TP group, which shares this
        # step's batch.
        assert num_rows % tp == 0, f"TP={tp} does not divide {num_rows} rows"
        return num_rows // tp


class SGLangModelWrapper(Module):
    """Runs a TorchTitan model behind SGLang's model interface.

    SGLang's model loader constructs this class with the HF-shaped config and calls
    ``load_weights``; ``ModelRunner`` calls ``forward(input_ids, positions,
    forward_batch)`` and samples from the returned ``LogitsProcessorOutput``. The
    wrapper builds the TorchTitan model on the meta device, parallelizes it with the
    generator's layout, and loads initial weights through TorchTitan's checkpointer,
    as the vLLM wrapper does. ``model`` is the transformer body SGLang's prefill CUDA
    graphs capture (``SGLangTransformerBody``); ``decoder`` is the TorchTitan model.
    """

    def __init__(
        self,
        *,
        model_config: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        checkpointer_config: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None = None,
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
            model = self.config.build()
        model._skip_lm_head = True
        if getattr(model, "vision_encoder", None) is not None:
            model.vision_encoder = None

        model = model.parallelize(
            parallelism_context=self.parallelism_context,
            training=TrainingConfig(),
            parallelism=training_parallelism,
            local_compile_regions=self.config.local_compile_regions,
            activation_remat=None,
            dump_folder="",
        )
        # Preserve compute storage addresses across model state dict loads for CUDA graphs.
        assert isinstance(model, FSDPModule)
        model.set_keep_unsharded_storage(True)

        device = torch.device("cuda", torch.cuda.current_device())
        model.to_empty(device=device)
        with torch.no_grad(), self.parallelism_context.activate_spmd():
            model.init_weights(buffer_device=None)
        runtime = self._build_dist_moe_runtime(dist_moe_runtime, model, device)
        load_initial_weights(model, self.config, checkpointer_config)
        self._tp_custom_ar = None
        if self.parallelism_context.tp_enabled and not is_in_batch_invariant_mode():
            # Off in batch-invariant mode: the custom all-reduce picks its algorithm
            # by size, which changes a token's reduction order with the batch.
            with self.parallelism_context.activate_spmd():
                tp_group = current_spmd_mesh().get_group("tp")
            self._tp_custom_ar = _route_tp_all_reduce_through_custom_ar(
                tp_group, self.config.dim
            )

        # Build pinned CPU receive buffers for model state dict loads while the model
        # is still sharded. Bind first so first-touch places them on the NUMA node
        # local to this rank's GPU.
        maybe_apply_numa_binding(torch.cuda.current_device(), "cuda")
        with torch.device("cpu"):
            self._prefetched_model_state_dict = _create_cpu_state_dict(
                model.state_dict(), pin_memory=True
            )

        # Unshard now so SGLang profiles GPU memory with the representation the
        # forward uses before it sizes the KV pool.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                unshard_and_free_sharded_params(module)
        self.model = SGLangTransformerBody(
            model,
            self.parallelism_context,
            dist_moe_runtime=runtime,
            max_prefill_graph_tokens=(
                None if runtime is None else _max_prefill_graph_tokens()
            ),
        )

    def _build_dist_moe_runtime(
        self,
        runtime_config: DistMoeRuntime.Config | None,
        model: Decoder,
        device: torch.device,
    ) -> DistMoeRuntime | None:
        """Build the runtime the model's Dist-MoE experts run under, if they exist.

        As in the vLLM wrapper, the context is built once, before any CUDA graph
        capture, and sized for the most tokens a step on this rank can run.
        """
        has_dist_moe = any(
            isinstance(module, DistMoeRoutedExperts) for module in model.modules()
        )
        if not has_dist_moe:
            if runtime_config is not None:
                logger.warning(
                    "Ignoring dist_moe_runtime: the generator model has no "
                    "Dist-MoE routed experts."
                )
            return None
        if runtime_config is None:
            raise ValueError(
                "The generator model uses Dist-MoE routed experts but no "
                "dist_moe_runtime was given. Pass a "
                "DistMoeRuntime.Config(inference=True) to the generator."
            )
        if not runtime_config.inference:
            raise ValueError(
                f"dist_moe_runtime must set inference=True; got {runtime_config!r}"
            )
        # An inference context rejects any operand carrying requires_grad.
        for module in model.modules():
            if isinstance(module, DistMoeRoutedExperts):
                module.requires_grad_(False)
        return runtime_config.build(
            model_parts=[model],
            parallelism_context=self.parallelism_context,
            device=device,
            num_tokens_per_microbatch_per_dp_rank=_max_num_tokens_per_step(
                self.parallelism_context.tp
            ),
            pp_schedule=None,
        )

    def register_cuda_graph_buffers(self) -> None:
        """Give the TP custom all-reduce its peers' buffer addresses for the CUDA
        graphs captured so far; collective over the TP ranks. The engine calls it
        once SGLang has captured its graphs, before the first replay."""
        if self._tp_custom_ar is not None:
            self._tp_custom_ar.register_graph_buffers()

    @property
    def decoder(self) -> Decoder:
        """The TorchTitan model."""
        return self.model.decoder

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self._prefetched_model_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        """Copy new weights into ``self.decoder`` (see ``load_state_dict_by_fsdp_group``).

        Args:
            state_dict: Keyed like ``self.decoder.state_dict()``, e.g. the generator's
                pinned CPU copy.
        """
        return load_state_dict_by_fsdp_group(self.decoder, state_dict)

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
        # A prefill CUDA graph replay stands in for the body and returns the hidden
        # states of the step's tokens.
        h_TD = self.model(input_ids, positions, forward_batch)

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

    def compute_logits(self, h_BD: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits of ``h_BD``, gathered over tensor parallelism."""
        with self.parallelism_context.activate_spmd():
            logits_BV = self.decoder.lm_head(h_BD)
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
