# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang inference engine for the RL generator, running TorchTitan models.

SGLang's ``Engine`` spawns a scheduler process per GPU and talks to it over ZMQ. The
RL generator instead owns one process per GPU (launched by Monarch or torchrun) and
drives the engine with SPMD calls from its engine loop, so this engine builds
SGLang's ``Scheduler`` in-process on each rank and runs one scheduler iteration per
``step()``, the way the vLLM engine uses vLLM's ``external_launcher``:

    vLLM  (external_launcher)       SGLang (this engine)
    LLMEngine.add_request      ->   Scheduler.process_input_requests([TokenizedGenerateReqInput])
    LLMEngine.step             ->   get_next_batch_to_run + run_batch + process_batch_result
    RequestOutput              ->   the scheduler's BatchTokenIDOutput payloads, collected
                                    in-process instead of sent to the detokenizer

Every TP rank runs its own scheduler on the same requests in the same order, as
SGLang's own TP schedulers do after rank 0 broadcasts the inputs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from array import array
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import GenerationEngine
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)
from torchtitan.tools.utils import has_cuda_capability

logger = logging.getLogger(__name__)

# ServerArgs fields this engine derives from the generator config.
_OWNED_SERVER_ARGS = frozenset(
    {
        "model_path",
        "tokenizer_path",
        "skip_tokenizer_init",
        "dtype",
        "tp_size",
        "dp_size",
        "attention_backend",
        "page_size",
        "mem_fraction_static",
        "max_running_requests",
        "chunked_prefill_size",
        "context_length",
        "json_model_override_args",
        "disable_cuda_graph",
        "disable_overlap_schedule",
        "disable_custom_all_reduce",
        "random_seed",
        "enable_priority_scheduling",
        "schedule_low_priority_values_first",
        "load_format",
    }
)


@dataclass(kw_only=True, slots=True)
class _OutputAccumulator:
    """One request's output, accumulated across the scheduler's output payloads."""

    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    num_cached_tokens: int | None = None


class _OutputCollector:
    """Stands in for the scheduler's detokenizer socket and keeps the payloads.

    The scheduler sends a ``BatchTokenIDOutput`` whenever requests make progress it
    reports; a non-streaming request reports when it finishes (and every
    ``SGLANG_FORCE_STREAM_INTERVAL`` tokens). Each payload carries the tokens and
    logprobs since the previous one.
    """

    def __init__(self) -> None:
        self._outputs: dict[str, _OutputAccumulator] = {}
        self._finished: list[FinishedRequest] = []

    def send_output(self, payload, *args, **kwargs) -> None:
        rids = getattr(payload, "rids", None)
        if rids is None:
            return
        for i, rid in enumerate(rids):
            output = self._outputs.setdefault(rid, _OutputAccumulator())
            output.token_ids.extend(payload.output_ids[i])
            if payload.output_token_logprobs_val is not None:
                output.token_logprobs.extend(payload.output_token_logprobs_val[i])
            output.num_cached_tokens = payload.cached_tokens[i]
            finished_reason = payload.finished_reasons[i]
            if finished_reason is None:
                continue
            del self._outputs[rid]
            self._finished.append(
                FinishedRequest(
                    request_id=rid,
                    token_ids=output.token_ids,
                    token_logprobs=output.token_logprobs,
                    finish_reason=_finish_reason(finished_reason),
                    metrics_inputs=RequestMetricsInputs(
                        num_cached_tokens=output.num_cached_tokens, has_stats=False
                    ),
                )
            )

    def pop_finished(self) -> list[FinishedRequest]:
        finished, self._finished = self._finished, []
        return finished


def _finish_reason(finished_reason: dict[str, Any]) -> str:
    """Map SGLang's finish reason to vLLM's names, which the generator reports."""
    reason_type = finished_reason["type"]
    if reason_type == "length":
        return "length"
    if reason_type == "stop":
        return "stop"
    return reason_type


class SGLangEngine(GenerationEngine):
    """An SGLang scheduler running a TorchTitan model, behind the calls the engine loop makes.

    See ``GenerationEngine`` for the build arguments and the threading contract.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GenerationEngine.Config):
        enable_cuda_graph: bool = False
        """Capture SGLang's decode CUDA graphs. Off: every step runs eager."""

        page_size: int | None = None
        """KV page size in tokens. None: 256 with FA2 (SM < 9.0), which requires a
        multiple of 256, else 16 (vLLM's default block size)."""

        extra_server_args: dict[str, Any] = field(default_factory=dict)
        """Additional keyword arguments for SGLang's ``ServerArgs``, for options
        without a dedicated field. Fields this engine derives from the generator
        config raise a ``ValueError``."""

    def __init__(
        self,
        config: Config,
        *,
        model: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        model_path: str,
        max_num_seqs: int,
        model_dtype: str,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        debug: DebugConfig,
        reset_kv_cache_on_model_state_dict_load: bool,
        rank: int,
        generator_name: str,
        output_dir: str,
    ):
        del generator_name, output_dir
        if parallelism.data_parallel_degree != 1:
            raise NotImplementedError(
                "SGLangEngine does not support data_parallel_degree > 1 yet"
            )
        if parallelism.expert_parallel_degree != 1:
            raise NotImplementedError(
                "SGLangEngine does not support expert parallelism yet"
            )
        if parallelism.enable_sequence_parallel:
            raise NotImplementedError(
                "SGLangEngine does not support sequence parallelism"
            )
        if dist_moe_runtime is not None:
            raise NotImplementedError("SGLangEngine does not support Dist-MoE yet")
        attention = model.first_full_attention
        if attention is None or not isinstance(
            attention.inner_attention, VarlenInnerAttention.Config
        ):
            raise ValueError("SGLangEngine requires varlen attention in the model")
        if (
            config.enable_cuda_graph
            and model.first_moe is not None
            and not has_cuda_capability(9, 0)
        ):
            raise ValueError(
                "enable_cuda_graph requires SM90+ for MoE models: below SM90, the "
                "expert GEMM (torch._grouped_mm) reads group sizes on the host, "
                "which a CUDA graph cannot capture"
            )
        overlapping = sorted(config.extra_server_args.keys() & _OWNED_SERVER_ARGS)
        if overlapping:
            raise ValueError(
                f"extra_server_args cannot set ServerArgs that SGLangEngine controls: "
                f"{overlapping}"
            )

        self.config = config
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )
        self._max_context_length = model.max_context_length
        self._vocab_size = model.vocab_size
        self._tp_degree = parallelism.tensor_parallel_degree
        self._rank = rank

        from torchtitan.rl.generator.sglang.registry import (
            model_config_to_sglang_overrides,
            register_to_sglang,
        )

        register_to_sglang(
            model, parallelism=parallelism, checkpointer_config=checkpointer
        )
        page_size = config.page_size
        if page_size is None:
            page_size = 16 if has_cuda_capability(9, 0) else 256
        server_args_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tokenizer_path=model_path,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            dtype=model_dtype,
            tp_size=parallelism.tensor_parallel_degree,
            dp_size=1,
            attention_backend="torchtitan_varlen",
            page_size=page_size,
            mem_fraction_static=gpu_memory_limit,
            max_running_requests=max_num_seqs,
            context_length=model.max_context_length,
            json_model_override_args=json.dumps(
                model_config_to_sglang_overrides(model)
            ),
            disable_cuda_graph=not config.enable_cuda_graph,
            # The generator steps the scheduler synchronously.
            disable_overlap_schedule=True,
            # The TorchTitan model's collectives run on its own process groups; SGLang's
            # custom all-reduce would only cost its startup (a kernel JIT build).
            disable_custom_all_reduce=True,
            # Smaller priority runs first, as in the vLLM engine; EngineRequest.group_id
            # explains the order.
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
            log_level="warning",
        )
        if max_num_batched_tokens is not None:
            server_args_kwargs["chunked_prefill_size"] = max_num_batched_tokens
        if debug.seed is not None:
            server_args_kwargs["random_seed"] = debug.seed
        server_args_kwargs.update(config.extra_server_args)
        self._scheduler = self._build_scheduler(server_args_kwargs, rank=rank)
        self._install_hooks()

    def _build_scheduler(self, server_args_kwargs: dict[str, Any], *, rank: int):
        """Build SGLang's scheduler in this process: what ``run_scheduler_process``
        does in SGLang's scheduler child, minus the process setup the launcher owns."""
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.plugins import load_plugins
        from sglang.srt.runtime_context import publish, spawn_world_rank, SpawnRanks
        from sglang.srt.server_args import PortArgs, ServerArgs

        # Registers the attention backend with SGLang.
        import torchtitan.rl.generator.sglang.attention  # noqa: F401

        # As vLLM's external_launcher does: this process is one rank of the launcher's
        # world (torchrun or Monarch env vars); SGLang reuses an initialized default
        # process group instead of creating its own.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", device_id=torch.device("cuda", local_rank)
            )

        load_plugins()
        server_args = ServerArgs(**server_args_kwargs)
        server_args.resolve_once()
        server_args.check_server_args()
        tp_rank = rank % self._tp_degree
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
                dp_rank=None,
                gpu_id=local_rank,
            ),
        )
        port_args = PortArgs.init_new(server_args)
        logger.info("Initializing SGLang scheduler in-process (rank %d)", rank)
        return Scheduler(server_args, port_args)

    def _install_hooks(self) -> None:
        """Swap in the parity sampler and collect outputs in-process."""
        from torchtitan.rl.generator.sglang.sampler import TorchTitanSampler

        scheduler = self._scheduler
        # The renderer's stop_token_ids (which include EOS) must be the only stops, as
        # in the vLLM engine. SGLang's ignore_eos would also disable stop_token_ids, so
        # drop the model's EOS ids, which the scheduler gives every request it creates.
        scheduler.model_config.hf_eos_token_id = set()
        model_runner = scheduler.tp_worker.model_runner
        self._sampler = TorchTitanSampler()
        model_runner.sampler = self._sampler

        # The sampler needs each row's request (for its seeded generator); a
        # ForwardBatch keeps no reference to the ScheduleBatch it came from.
        tp_worker = scheduler.tp_worker
        forward_batch_generation = tp_worker.forward_batch_generation
        sampler = self._sampler

        def forward_batch_generation_with_reqs(batch, *args, **kwargs):
            if batch is not None:
                sampler.set_batch_reqs(batch.reqs)
            return forward_batch_generation(batch, *args, **kwargs)

        tp_worker.forward_batch_generation = forward_batch_generation_with_reqs

        self._output_collector = _OutputCollector()
        scheduler.output_streamer.send_to_detokenizer = self._output_collector

    def _get_model(self):
        return self._scheduler.tp_worker.model_runner.model

    @property
    def data_parallel_size(self) -> int:
        return 1

    @property
    def data_parallel_rank(self) -> int:
        return 0

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self._get_model().prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return those rejected before admission."""
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        rejected_requests: list[tuple[str, Exception]] = []
        tokenized_requests = []
        for request in requests:
            error = self._validate(request)
            if error is not None:
                rejected_requests.append((request.request_id, error))
                continue
            sampling_params = _build_sampling_params(request.sampling)
            sampling_params.normalize(None)
            sampling_params.verify(self._vocab_size)
            tokenized_requests.append(
                TokenizedGenerateReqInput(
                    rid=request.request_id,
                    input_text=None,
                    input_ids=array("q", request.prompt_token_ids),
                    input_embeds=None,
                    mm_inputs=None,
                    token_type_ids=None,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=-1,
                    top_logprobs_num=0,
                    token_ids_logprob=None,
                    stream=False,
                    priority=max(request.group_id, -1),
                    cache_salt=(
                        None
                        if self._reset_kv_cache_on_model_state_dict_load
                        else str(request.min_policy_version)
                    ),
                )
            )
        if tokenized_requests:
            self._scheduler.process_input_requests(tokenized_requests)
            # The generator returns output-token logprobs only. A request that returns
            # logprobs starts SGLang's input logprobs at its prompt length, so a
            # retracted request re-prefilling its generated tokens would need input
            # logprobs for them, which the wrapper does not compute; start them past
            # any fill. This also lifts the cap SGLang puts on prefix-cache matches
            # at the logprob start.
            admitted = {request.rid for request in tokenized_requests}
            for req in self._scheduler.waiting_queue:
                if req.rid in admitted:
                    req.logprob_start_len = sys.maxsize
        return rejected_requests

    def _validate(self, request: EngineRequest) -> Exception | None:
        """Reject what SGLang would abort after admission, so it never enters the engine."""
        num_prompt_tokens = len(request.prompt_token_ids)
        if num_prompt_tokens == 0:
            return ValueError("prompt is empty")
        if num_prompt_tokens >= self._max_context_length:
            return ValueError(
                f"prompt has {num_prompt_tokens} tokens; the context length is "
                f"{self._max_context_length}"
            )
        if any(t < 0 or t >= self._vocab_size for t in request.prompt_token_ids):
            return ValueError("prompt token id out of vocabulary range")
        return None

    def has_unfinished_requests(self) -> bool:
        scheduler = self._scheduler
        # A prefill batch's requests join running_batch only when the next step
        # schedules, so a just-prefilled request is still in last_batch.
        last_batch = scheduler.last_batch
        return (
            len(scheduler.waiting_queue) > 0
            or scheduler.chunked_req is not None
            or any(not req.finished() for req in scheduler.running_batch.reqs)
            or (
                last_batch is not None
                and any(not req.finished() for req in last_batch.reqs)
            )
        )

    def step(self) -> list[FinishedRequest]:
        """Run one scheduler iteration, a TP collective, and return the requests it finished."""
        scheduler = self._scheduler
        with torch.no_grad():
            plan = scheduler.get_next_batch_to_run(
                running_batch=scheduler.running_batch,
                last_batch=scheduler.last_batch,
            )
            scheduler.running_batch = plan.running_batch
            batch = plan.batch_to_run
            scheduler.cur_batch_for_debug = batch
            if batch:
                result = scheduler.run_batch(batch)
                scheduler.process_batch_result(batch, result)
            scheduler.last_batch = batch
        finished = self._output_collector.pop_finished()
        self._sampler.release(request.request_id for request in finished)
        return finished

    def load_prefetched_model_state_dict(self) -> None:
        """Copy the prefetched model state dict into the model, and reset the KV cache when
        configured."""
        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            PauseGenerationReqInput,
        )

        scheduler = self._scheduler
        if self._reset_kv_cache_on_model_state_dict_load:
            # Retract running requests to the waiting queue (their KV is freed and
            # recomputed under the new weights when they resume), then drop the cache.
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
        model = self._get_model()
        model.load_state_dict(model.prefetched_model_state_dict)
        if self._reset_kv_cache_on_model_state_dict_load:
            if not scheduler.flush_cache():
                raise RuntimeError("SGLang did not flush its KV cache after retracting")
            scheduler.continue_generation(
                ContinueGenerationReqInput(torch_empty_cache=False)
            )

    def shutdown(self) -> None:
        """Process teardown is the launcher's; release nothing beyond Python references."""
        self._scheduler = None


def _build_sampling_params(sampling: SamplingConfig):
    """Translate a ``SamplingConfig`` into SGLang ``SamplingParams``.

    The engine loads no tokenizer and drops the model's EOS ids (see
    ``SGLangEngine._install_hooks``), so the renderer's ``stop_token_ids`` (which
    include EOS) are the only stops, as in the vLLM engine. ``ignore_eos`` stays off:
    in SGLang it also disables ``stop_token_ids``.
    """
    from sglang.srt.sampling.sampling_params import SamplingParams

    return SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_new_tokens=sampling.max_tokens,
        stop_token_ids=sampling.stop_token_ids,
        sampling_seed=sampling.seed,
        skip_special_tokens=False,
    )
