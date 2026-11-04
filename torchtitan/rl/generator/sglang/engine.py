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

import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from array import array
from dataclasses import dataclass, field
from types import SimpleNamespace
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
        "enable_dp_attention",
        "attention_backend",
        "page_size",
        "mem_fraction_static",
        "max_running_requests",
        "chunked_prefill_size",
        "context_length",
        "disable_cuda_graph",
        "disable_overlap_schedule",
        "disable_custom_all_reduce",
        "random_seed",
        "enable_priority_scheduling",
        "schedule_low_priority_values_first",
        "load_format",
        "enable_session_radix_cache",
        "cuda_graph_config",
        "disable_chunked_prefix_cache",
        "bf16_gemm_backend",
        "enable_mixed_chunk",
    }
)

# Most tokens a step runs when the generator sets no max_num_batched_tokens: vLLM's
# default, which also bounds its FULL CUDA graphs.
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048


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
        self._arrival_ts: dict[str, float] = {}

    def set_arrival_time(self, request_id: str, arrival_time: float) -> None:
        """Record a request's ``EngineRequest.arrival_time`` (wall clock) on the
        scheduler's ``time.perf_counter`` clock, so its TTFT counts the time it waited
        in the generator before admission, as vLLM's does."""
        self._arrival_ts[request_id] = time.perf_counter() - (
            time.time() - arrival_time
        )

    def send_output(self, payload, *args, **kwargs) -> None:
        rids = getattr(payload, "rids", None)
        if rids is None:
            return
        # In-process, SGLang's pickle IPC (its default) hands over the requests' live
        # SchedulerReqTimeStats; over msgpack IPC they arrive empty.
        time_stats = getattr(payload, "time_stats", None)
        if not isinstance(time_stats, list):
            time_stats = None
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
            arrival_ts = self._arrival_ts.pop(rid, None)
            self._finished.append(
                FinishedRequest(
                    request_id=rid,
                    token_ids=output.token_ids,
                    token_logprobs=output.token_logprobs,
                    finish_reason=_finish_reason(finished_reason),
                    metrics_inputs=_request_metrics_inputs(
                        output,
                        time_stats[i] if time_stats is not None else None,
                        arrival_ts,
                        (
                            getattr(payload, "retraction_counts", None)
                            or [0] * len(rids)
                        )[i],
                    ),
                )
            )

    def pop_finished(self) -> list[FinishedRequest]:
        finished, self._finished = self._finished, []
        return finished


def _request_metrics_inputs(
    output: _OutputAccumulator,
    stats: Any,
    arrival_ts: float | None,
    num_retractions: int,
) -> RequestMetricsInputs:
    """The vLLM ``RequestStateStats`` fields the generator's metrics read, from SGLang's
    scheduler timestamps: waiting-queue entry, first forward, end of prefill (the
    first token) and completion."""
    if stats is None or arrival_ts is None or not stats.prefill_finished_time:
        return RequestMetricsInputs(
            num_cached_tokens=output.num_cached_tokens, has_stats=False
        )
    return RequestMetricsInputs(
        num_cached_tokens=output.num_cached_tokens,
        has_stats=True,
        queued_ts=stats.wait_queue_entry_time,
        scheduled_ts=stats.forward_entry_time,
        first_token_ts=stats.prefill_finished_time,
        last_token_ts=stats.completion_time or stats.prefill_finished_time,
        first_token_latency=stats.prefill_finished_time - arrival_ts,
        num_generation_tokens=len(output.token_ids),
        num_preemptions=num_retractions,
    )


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
        """Capture full CUDA graphs of decode and prefill steps, as vLLM's FULL CUDA
        graph mode does. Decode graphs cover up to ``max_num_seqs`` requests. Prefill
        graphs cover token buckets (powers of two) up to ``max_num_batched_tokens``
        with ``max_num_seqs`` request slots. Off: every step runs eager."""

        overlap_schedule: bool = True
        """Overlap the scheduler's CPU work with the GPU, as SGLang's server does by
        default: a step launches its batch's forward, then processes the previous
        batch's results while the forward runs, so ``step()`` returns the requests
        that finished one step earlier. Off: a step runs its batch and processes its
        results before it returns, leaving the GPU idle during that CPU work."""

        share_group_prefill: bool = True
        """Prefill a group's shared prompt once. Requests of one group with the same
        prompt (GRPO samples) wait until the first one's prefill is done, then read
        the prompt's KV from the radix cache and prefill only their last token. A
        prefill batch otherwise computes the prompt once per sample: SGLang adds a
        prompt to its radix cache after the batch that computes it. Needs the radix
        cache (not GDN models)."""

        page_size: int | None = None
        """KV page size in tokens. None: 256 with FA2 (SM < 9.0), which requires a
        multiple of 256, else 16 (vLLM's default block size)."""

        hold_session_kv: bool = False
        """Keep each multi-turn session's history cached between its turns.

        Requests carry their ``routing_session_id`` as SGLang's radix session id
        (``enable_session_radix_cache``): when a turn finishes, the prompt and output
        KV it caches is tagged with its session, and eviction takes untagged KV first.
        ``release_sessions`` drops the tags when a rollout or its group ends.

        Unlike ``VLLMSessionKVHolder``, which pins blocks, the tag only orders eviction,
        so tagged KV is still evicted once no untagged KV is left. Works with either
        ``reset_kv_cache_on_model_state_dict_load``: a reset drops the tags with the
        cache. Not for GDN models, which run without a radix cache."""

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
        if parallelism.expert_parallel_degree not in (
            1,
            parallelism.data_parallel_degree * parallelism.tensor_parallel_degree,
        ):
            raise ValueError(
                "expert_parallel_degree must be 1 or data_parallel_degree * "
                "tensor_parallel_degree"
            )
        if parallelism.enable_sequence_parallel:
            raise NotImplementedError(
                "SGLangEngine does not support sequence parallelism"
            )
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
        self._dp_degree = parallelism.data_parallel_degree
        self._rank = rank

        from torchtitan.rl.generator.sglang.registry import (
            is_hybrid_linear_attention,
            model_config_to_sglang_hf_config,
            register_to_sglang,
        )

        register_to_sglang(
            model,
            parallelism=parallelism,
            checkpointer_config=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
        )
        page_size = config.page_size
        if page_size is None:
            page_size = 16 if has_cuda_capability(9, 0) else 256
        # SGLang reads only the HF config from its model path: the wrapper loads
        # weights from the checkpointer, and the engine loads no tokenizer. Give it
        # a config derived from the TorchTitan model instead of the checkpoint's.
        del model_path
        self._sglang_model_dir = tempfile.mkdtemp(prefix="torchtitan_sglang_")
        with open(os.path.join(self._sglang_model_dir, "config.json"), "w") as f:
            json.dump(model_config_to_sglang_hf_config(model), f)
        server_args_kwargs: dict[str, Any] = dict(
            model_path=self._sglang_model_dir,
            tokenizer_path=self._sglang_model_dir,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            dtype=model_dtype,
            # Generator DP supplies ranks for expert parallelism, whose all-to-alls
            # span every DP rank, so the DP ranks must step together. That is
            # SGLang's data-parallel attention: one world of DP * TP ranks in which
            # attention runs data-parallel over DP and tensor-parallel over TP, and a
            # rank without requests runs an idle batch in a step any rank runs.
            tp_size=parallelism.data_parallel_degree
            * parallelism.tensor_parallel_degree,
            dp_size=parallelism.data_parallel_degree,
            enable_dp_attention=parallelism.data_parallel_degree > 1,
            attention_backend="torchtitan_varlen",
            page_size=page_size,
            mem_fraction_static=gpu_memory_limit,
            max_running_requests=max_num_seqs,
            context_length=model.max_context_length,
            disable_overlap_schedule=not config.overlap_schedule,
            # The TorchTitan model's collectives run on its own process groups; SGLang's
            # custom all-reduce would only cost its startup (a kernel JIT build).
            disable_custom_all_reduce=True,
            # The TorchTitan model runs its own linears. On SM100 SGLang's default
            # ("cutedsl") imports FlashInfer GEMM kernels newer than the FlashInfer
            # the vLLM nightly pins.
            bf16_gemm_backend="torch",
            # Run the running requests' decode tokens in the same step as a prefill
            # chunk, as vLLM's scheduler does, instead of pausing them for a
            # prefill-only step. With steady arrivals (multi-turn rollouts),
            # prefill-only steps would stall every running request.
            enable_mixed_chunk=True,
            # Smaller priority runs first, as in the vLLM engine; EngineRequest.group_id
            # explains the order.
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
            log_level="warning",
        )
        max_num_batched_tokens = (
            max_num_batched_tokens or _DEFAULT_MAX_NUM_BATCHED_TOKENS
        )
        server_args_kwargs["chunked_prefill_size"] = max_num_batched_tokens
        if config.enable_cuda_graph:
            server_args_kwargs.update(
                _full_cuda_graph_server_args(
                    max_num_seqs=max_num_seqs,
                    max_prefill_tokens=max_num_batched_tokens,
                )
            )
        else:
            server_args_kwargs["disable_cuda_graph"] = True
        if debug.seed is not None:
            server_args_kwargs["random_seed"] = debug.seed
        server_args_kwargs.update(config.extra_server_args)
        if is_hybrid_linear_attention(model):
            # Reusing a cached prefix of a GDN model would need its recurrent state
            # at the prefix boundary, which the GDN backend does not checkpoint.
            if not server_args_kwargs.get("disable_radix_cache", True):
                raise ValueError("GDN models require disable_radix_cache=True")
            server_args_kwargs["disable_radix_cache"] = True
        if config.hold_session_kv:
            if server_args_kwargs.get("disable_radix_cache", False):
                raise ValueError(
                    "hold_session_kv needs SGLang's radix cache, which this model "
                    "runs without"
                )
            server_args_kwargs["enable_session_radix_cache"] = True
            # SGLang logs every session release at INFO.
            logging.getLogger(
                "sglang.srt.mem_cache.unified_cache.session_ref_tracker"
            ).setLevel(logging.WARNING)
        # Sessions admitted on this rank, by group, so a group's release closes its
        # sessions (hold_session_kv).
        self._session_groups: dict[str, int] = {}
        # share_group_prefill: the first request of each (group, prompt, cache salt)
        # whose prefill is not done yet, and the requests waiting behind it.
        self._share_group_prefill = config.share_group_prefill and not (
            server_args_kwargs.get("disable_radix_cache", False)
        )
        self._prefill_heads: dict[tuple, str] = {}
        self._waiting_siblings: dict[str, list] = {}
        self._head_reqs: dict[str, Any] = {}
        self._group_sessions: dict[int, set[str]] = {}
        self._scheduler = self._build_scheduler(server_args_kwargs, rank=rank)
        self._install_hooks()
        # Keep SGLang's ~1.2M startup objects out of later collections, as vLLM's
        # engine does after its startup (freeze_gc_heap): the engine loop runs a full
        # gc.collect() after every weight load, which took ~0.9 s per load otherwise.
        gc.collect()
        gc.freeze()

    def _build_scheduler(self, server_args_kwargs: dict[str, Any], *, rank: int):
        """Build SGLang's scheduler in this process: what ``run_scheduler_process``
        does in SGLang's scheduler child, minus the process setup the launcher owns."""
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.plugins import load_plugins

        _restore_triton_next_power_of_2()
        from sglang.srt.runtime_context import publish, spawn_world_rank, SpawnRanks
        from sglang.srt.server_args import ServerArgs

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
        dp_rank = rank // self._tp_degree if self._dp_degree > 1 else None
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0),
                dp_rank=dp_rank,
                gpu_id=local_rank,
            ),
        )
        port_args = _in_process_port_args()
        # Every rank's step runs collectives with every other DP rank, so the DP
        # ranks agree on whether any of them has work (see has_unfinished_requests).
        self._dp_cpu_group = (
            dist.new_group(backend="gloo") if self._dp_degree > 1 else None
        )
        logger.info("Initializing SGLang scheduler in-process (rank %d)", rank)
        scheduler = Scheduler(server_args, port_args)
        # The scheduler has captured its CUDA graphs.
        scheduler.tp_worker.model_runner.model.register_cuda_graph_buffers()
        if scheduler.enable_overlap:
            # What Scheduler.run_event_loop and event_loop_overlap set up before
            # their first iteration; step() runs one iteration at a time instead.
            from sglang.srt.managers.utils import allocate_distinct_stream

            if (
                scheduler.schedule_stream.cuda_stream
                == scheduler.forward_stream.cuda_stream
            ):
                scheduler.schedule_stream = allocate_distinct_stream(
                    torch.cuda, (scheduler.forward_stream,)
                )
            scheduler._war_barrier_enabled = True
            scheduler.enable_skip_finishing_decode = True
        return scheduler

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
        # ForwardBatch keeps no reference to the ScheduleBatch it came from. A batch
        # that does not finish a request's prefill launches while that request is the
        # scheduler's chunked_req. Its inflight_middle_chunks count would lag: with
        # overlap, the previous chunk's result is processed after the next launches.
        tp_worker = scheduler.tp_worker
        forward_batch_generation = tp_worker.forward_batch_generation
        sampler = self._sampler

        def forward_batch_generation_with_reqs(batch, *args, **kwargs):
            if batch is not None:
                sampler.set_batch_reqs(batch.reqs, partial_req=scheduler.chunked_req)
            return forward_batch_generation(batch, *args, **kwargs)

        tp_worker.forward_batch_generation = forward_batch_generation_with_reqs

        self._output_collector = _OutputCollector()
        scheduler.output_streamer.send_to_detokenizer = self._output_collector

    def _get_model(self):
        return self._scheduler.tp_worker.model_runner.model

    @property
    def data_parallel_size(self) -> int:
        return self._dp_degree

    @property
    def data_parallel_rank(self) -> int:
        return self._rank // self._tp_degree

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self._get_model().prefetched_model_state_dict

    @property
    def holds_session_kv(self) -> bool:
        return self.config.hold_session_kv

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return those rejected before admission."""
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        rejected_requests: list[tuple[str, Exception]] = []
        tokenized_requests = []
        new_heads = []
        for request in requests:
            error = self._validate(request)
            if error is not None:
                rejected_requests.append((request.request_id, error))
                continue
            self._output_collector.set_arrival_time(
                request.request_id, request.arrival_time
            )
            sampling_params = _build_sampling_params(request.sampling)
            sampling_params.normalize(None)
            sampling_params.verify(self._vocab_size)
            tokenized_request = TokenizedGenerateReqInput(
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
                session_id=(
                    request.routing_session_id if self.config.hold_session_kv else None
                ),
            )
            if self._share_group_prefill:
                key = (
                    request.group_id,
                    tuple(request.prompt_token_ids),
                    tokenized_request.cache_salt,
                )
                head = self._prefill_heads.get(key)
                if head is not None:
                    self._waiting_siblings[head].append(tokenized_request)
                else:
                    self._prefill_heads[key] = request.request_id
                    self._waiting_siblings[request.request_id] = []
                    new_heads.append(request.request_id)
                    tokenized_requests.append(tokenized_request)
            else:
                tokenized_requests.append(tokenized_request)
            if self.config.hold_session_kv:
                self._session_groups[request.routing_session_id] = request.group_id
                self._group_sessions.setdefault(request.group_id, set()).add(
                    request.routing_session_id
                )
        admitted = self._admit(tokenized_requests)
        for rid in new_heads:
            # A request the scheduler did not queue (it aborted it) caches nothing.
            self._head_reqs[rid] = admitted.get(rid) or SimpleNamespace(
                finished=lambda: True
            )
        if new_heads:
            self._admit_waiting_siblings()
        return rejected_requests

    def _admit(self, tokenized_requests: list) -> dict[str, Any]:
        """Hand ``tokenized_requests`` to the scheduler; return their ``Req``s by id."""
        if not tokenized_requests:
            return {}
        with self._schedule_stream():
            self._scheduler.process_input_requests(tokenized_requests)
        # The generator returns output-token logprobs only. A request that returns
        # logprobs starts SGLang's input logprobs at its prompt length, so a
        # retracted request re-prefilling its generated tokens would need input
        # logprobs for them, which the wrapper does not compute; start them past
        # any fill. This also lifts the cap SGLang puts on prefix-cache matches
        # at the logprob start.
        rids = {request.rid for request in tokenized_requests}
        admitted = {}
        for req in self._scheduler.waiting_queue:
            if req.rid in rids:
                req.logprob_start_len = sys.maxsize
                admitted[req.rid] = req
        return admitted

    def _admit_waiting_siblings(self) -> None:
        """Admit the siblings of every group request whose prefill is done: the
        scheduler has cached its prompt once it has an output token."""
        done = [
            rid
            for rid, req in self._head_reqs.items()
            if req.finished() or len(req.output_ids) > 0
        ]
        if not done:
            return
        siblings = []
        for rid in done:
            del self._head_reqs[rid]
            siblings += self._waiting_siblings.pop(rid)
        done_set = set(done)
        self._prefill_heads = {
            key: rid for key, rid in self._prefill_heads.items() if rid not in done_set
        }
        self._admit(siblings)

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
        """Whether any admitted request is still running, on any DP rank.

        With DP > 1, a rank whose own requests are done still steps while another DP
        rank has work (SGLang gives it an idle batch), so this is a collective.
        """
        unfinished = self._has_local_unfinished_requests()
        if self._dp_cpu_group is None:
            return unfinished
        flag = torch.tensor([int(unfinished)], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self._dp_cpu_group)
        return bool(flag.item())

    def _has_local_unfinished_requests(self) -> bool:
        scheduler = self._scheduler
        # A prefill batch's requests join running_batch only when the next step
        # schedules, so a just-prefilled request is still in last_batch.
        last_batch = scheduler.last_batch
        return (
            len(scheduler.waiting_queue) > 0
            or any(self._waiting_siblings.values())
            # With overlap, a launched batch's results wait for the next step.
            or len(scheduler.result_queue) > 0
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
        with torch.no_grad(), self._schedule_stream():
            plan = scheduler.get_next_batch_to_run(
                running_batch=scheduler.running_batch,
                last_batch=scheduler.last_batch,
            )
            scheduler.running_batch = plan.running_batch
            batch = plan.batch_to_run
            scheduler.cur_batch_for_debug = batch
            if scheduler.enable_overlap:
                self._run_overlapped(batch)
            elif batch:
                result = scheduler.run_batch(batch)
                scheduler.process_batch_result(batch, result)
            scheduler.last_batch = batch
            # SGLang's watchdog kills the process when a batch is current but makes
            # no progress for watchdog_timeout. Between steps the generator may
            # leave the engine idle for long (trainer steps, weight syncs).
            scheduler.cur_batch_for_debug = None
        if self._head_reqs:
            self._admit_waiting_siblings()
        finished = self._output_collector.pop_finished()
        self._sampler.release(request.request_id for request in finished)
        return finished

    def _run_overlapped(self, batch) -> None:
        """Launch ``batch`` and process the previous batch's results meanwhile: one
        iteration of SGLang's ``Scheduler.event_loop_overlap``."""
        scheduler = self._scheduler
        # Consecutive prefill batches run without overlap (SGLang's TTFT policy).
        disable_overlap = scheduler.is_disable_overlap_for_batch(
            batch, last_batch=scheduler.last_batch
        )
        if disable_overlap and scheduler.result_queue:
            scheduler.process_batch_result(*scheduler.result_queue.popleft())
        batch_result = None
        if batch:
            batch_result = scheduler.run_batch(batch)
            scheduler._apply_war_barrier()
            scheduler.result_queue.append((batch.copy(), batch_result))
        if scheduler.last_batch and not disable_overlap:
            scheduler.process_batch_result(*scheduler.result_queue.popleft())
        if scheduler.is_generation:
            scheduler.launch_batch_sample_if_needed(batch_result, batch)

    def _schedule_stream(self):
        """The stream SGLang's event loop runs the scheduler's work on."""
        return torch.cuda.stream(self._scheduler.schedule_stream)

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        """Drop the session tags of ``session_ids`` and of every session of
        ``group_ids``; their KV stays cached, evicted like any untagged KV."""
        if not self.config.hold_session_kv:
            return
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        # Every rank gets every release; close only sessions this rank admitted.
        released = {s for s in session_ids if s in self._session_groups}
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank
        # must change its radix cache the same way.
        for session_id in sorted(released):
            group_sessions = self._group_sessions.get(
                self._session_groups.pop(session_id)
            )
            if group_sessions is not None:
                group_sessions.discard(session_id)
            # A request of the session still running tags nothing when it finishes.
            self._scheduler.close_session(CloseSessionReqInput(session_id=session_id))

    def log_stats(self, step: int) -> None:
        if self.config.hold_session_kv:
            logger.info(
                "Session KV at step %d: %d sessions open",
                step,
                len(self._session_groups),
            )

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
            with self._schedule_stream():
                scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
        model = self._get_model()
        # With overlap, the last step's forward may still be running, and the next
        # step's must read the new weights.
        torch.cuda.synchronize()
        model.load_state_dict(model.prefetched_model_state_dict)
        torch.cuda.synchronize()
        if self._reset_kv_cache_on_model_state_dict_load:
            if not scheduler.flush_cache():
                raise RuntimeError("SGLang did not flush its KV cache after retracting")
            if self.config.hold_session_kv:
                # The flush also reset the session tags and their generations; give
                # the waiting requests (retracted or not yet started) current ones,
                # or their turns would tag nothing when they finish.
                for req in scheduler.waiting_queue:
                    if req.session_id is not None:
                        req.session_generation = (
                            scheduler.tree_cache.ensure_session_generation(
                                req.session_id
                            )
                        )
            scheduler.continue_generation(
                ContinueGenerationReqInput(torch_empty_cache=False)
            )

    def shutdown(self) -> None:
        """Process teardown is the launcher's; release Python references and the config."""
        self._scheduler = None
        shutil.rmtree(self._sglang_model_dir, ignore_errors=True)


def _restore_triton_next_power_of_2() -> None:
    """Let Triton kernels call ``triton.next_power_of_2`` again after SGLang's import.

    SGLang replaces it with a plain Python function (``sglang.srt.utils.common``),
    which Triton refuses inside a ``@triton.jit`` function (Triton's own is a
    ``constexpr_function``): Dist-MoE's routing kernel then fails to compile. Wrap
    SGLang's version, which host callers keep using, as a ``constexpr_function``.
    """
    import triton
    from triton.runtime.jit import ConstexprFunction

    if not isinstance(triton.next_power_of_2, ConstexprFunction):
        triton.next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)


def _in_process_port_args():
    """IPC endpoints for a scheduler that receives and returns no messages.

    The in-process scheduler takes requests from ``add_requests`` and returns
    outputs through ``_OutputCollector``, so its sockets carry nothing; private
    ``ipc://`` names keep ranks from contending for the TCP ports SGLang derives
    for data-parallel attention.
    """
    import uuid

    from sglang.srt.server_args import PortArgs

    def ipc_name() -> str:
        return f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

    return PortArgs(
        tokenizer_ipc_name=ipc_name(),
        scheduler_input_ipc_name=ipc_name(),
        detokenizer_ipc_name=ipc_name(),
        # Unused: SGLang reuses the launcher's initialized process group.
        nccl_port=0,
        rpc_ipc_name=ipc_name(),
        metrics_ipc_name=ipc_name(),
        tokenizer_worker_ipc_name=None,
        decoupled_spec_ipc_config=None,
        instance_id=uuid.uuid4().hex[:12],
    )


def _full_cuda_graph_server_args(
    *, max_num_seqs: int, max_prefill_tokens: int
) -> dict[str, Any]:
    """ServerArgs for full CUDA graphs of decode and prefill steps."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )

    prefill_buckets = [1 << i for i in range(max_prefill_tokens.bit_length())]
    if max_prefill_tokens not in prefill_buckets:
        prefill_buckets.append(max_prefill_tokens)
    return dict(
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(
                backend=Backend.FULL,
                bs=prefill_buckets,
                full_prefill_max_req=max_num_seqs,
            ),
        ),
        # Full prefill graphs otherwise capture a separate cached-prefix topology,
        # which only MLA backends implement; this backend reads cached prefixes
        # through its block table like any other KV.
        disable_chunked_prefix_cache=True,
    )


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
