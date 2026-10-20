# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM inference engine for the RL generator: every vLLM call the generator makes."""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging,
    KVConnectorStats,
)
from vllm.exceptions import VLLMClientError
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.utils.network_utils import get_ip, get_open_port, make_zmq_path
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig
from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.generator.engine import GenerationEngine
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)
from torchtitan.rl.generator.vllm.batch_invariance import (
    force_logprobs_fn_for_batch_invariance,
)
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
)
from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
from torchtitan.rl.generator.vllm.stat_logger import (
    StatLoggerContext,
    VllmOtelStatLogger,
)
from torchtitan.tools.utils import has_cuda_capability

logger = logging.getLogger(__name__)


def add_extra_vllm_engine_args(
    engine_kwargs: dict[str, Any], extra_engine_args: dict[str, Any]
) -> None:
    """Add ``VLLMEngine.Config.extra_engine_args`` to ``engine_kwargs`` in place.

    Raises ``ValueError`` for keys TorchTitan already set in ``engine_kwargs`` or
    derives from a config field, so they cannot silently diverge from it.
    """
    # EngineArgs that TorchTitan sets only when the mapped config field is set (or
    # derives from it), so they may be absent from engine_kwargs.
    config_field_by_engine_arg = {
        "max_num_batched_tokens": "Generator.Config.max_num_batched_tokens",
        "compilation_config": "VLLMEngine.Config.cuda_graph",
        "seed": "Generator.Config.debug.seed",
    }
    owned_keys = sorted(
        extra_engine_args.keys()
        & (engine_kwargs.keys() | config_field_by_engine_arg.keys())
    )
    if owned_keys:
        details = ", ".join(
            (
                f"{key} (use {config_field_by_engine_arg[key]})"
                if key in config_field_by_engine_arg
                else key
            )
            for key in owned_keys
        )
        raise ValueError(
            f"extra_engine_args cannot set EngineArgs that TorchTitan controls: {details}"
        )
    engine_kwargs.update(extra_engine_args)


KVConnectorName = Literal["NixlConnector", "MooncakeConnector"]


def add_kv_transfer_config(
    engine_kwargs: dict[str, Any],
    prefill_decode_role: Literal["prefill", "decode"],
    *,
    connector: KVConnectorName = "NixlConnector",
    extra_config: dict[str, Any] | None = None,
    engine_id: str | None = None,
) -> dict[str, Any] | None:
    """Add vLLM's KV connector to ``engine_kwargs`` in place: a prefill generator produces KV,
    a decode generator consumes it.

    NIXL: peers reach this engine's KV listener at this host and port; the prefill request's
    output carries the handle a decode generator needs.

    Mooncake: a prefill engine serves a bootstrap address where decode workers find its
    workers, and its outputs carry no handle. Returns the handle fields for a prefill engine
    (``remote_engine_id``, ``remote_bootstrap_addr``); each request adds its ``transfer_id``.
    Returns None otherwise.
    """
    kv_role = "kv_producer" if prefill_decode_role == "prefill" else "kv_consumer"
    extra = dict(extra_config or {})
    if connector == "NixlConnector":
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = get_ip()
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(get_open_port())
        engine_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role=kv_role,
            kv_connector_extra_config=extra,
        )
        return None
    if connector != "MooncakeConnector":
        raise ValueError(f"unsupported KV connector {connector!r}")
    if engine_id is None:
        raise ValueError("MooncakeConnector needs an engine_id")
    handle = None
    if prefill_decode_role == "prefill":
        # Several prefill generators can share a host, so each binds its own port.
        # TODO: with TP > 1 every rank of the engine must agree on this port and engine_id.
        bootstrap_port = get_open_port()
        os.environ["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(bootstrap_port)
        handle = {
            "do_remote_prefill": True,
            "remote_engine_id": engine_id,
            # vLLM queries ``remote_bootstrap_addr + "/query"``, so this is a full URL.
            "remote_bootstrap_addr": make_zmq_path("http", get_ip(), bootstrap_port),
        }
    engine_kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
        engine_id=engine_id,
        kv_connector_extra_config=extra,
    )
    return handle


class _KVTransferStatLogger(StatLoggerBase):
    """Collects the KV connector's transfer stats from each engine step until ``drain``.

    The connector records them on the engine that moves the KV: the prefill engine for
    Mooncake (it writes into the decode engine's blocks), the decode engine for NIXL (it reads).
    """

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0) -> None:
        self._kv_connector_logging = KVConnectorLogging(vllm_config.kv_transfer_config)
        # `record` runs on the engine loop thread, `drain` on the generator's.
        self._lock = threading.Lock()

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is not None and scheduler_stats.kv_connector_stats:
            with self._lock:
                self._kv_connector_logging.observe(scheduler_stats.kv_connector_stats)

    def log_engine_initialized(self) -> None:
        pass

    def drain(self) -> KVConnectorStats | None:
        """The stats recorded since the last call, or None if there were none."""
        with self._lock:
            stats = self._kv_connector_logging.transfer_stats_accumulator
            self._kv_connector_logging.reset()
        return stats


def _kv_transfer_metrics(stats: KVConnectorStats) -> dict[str, float]:
    """The connector's reduced stats as ``kv_transfer/*`` scalars, e.g. vLLM's
    "Avg xfer time (ms)" -> ``kv_transfer/avg_xfer_time_ms``."""
    metrics = {
        "kv_transfer/" + re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_"): value
        for name, value in stats.reduce().items()
    }
    # Both connectors keep each transfer's duration in seconds under this key; the reduced
    # stats stop at P90, and the slowest hand-off sets the decode generator's wait.
    durations = stats.data.get("transfer_duration")
    if durations:
        metrics["kv_transfer/max_xfer_time_ms"] = max(durations) * 1000
    return metrics


class VLLMEngine(GenerationEngine):
    """The generator's vLLM ``LLMEngine``, behind the calls the engine loop makes.

    See ``GenerationEngine`` for the build arguments and the threading contract;
    ``LLMEngine`` is not thread-safe.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GenerationEngine.Config):
        cuda_graph: VLLMCudaGraphConfig = field(default_factory=VLLMCudaGraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        extra_engine_args: dict[str, Any] = field(default_factory=dict)
        """Additional keyword arguments passed to vLLM's ``EngineArgs``, for engine
        options without a dedicated field (e.g. ``{"watermark": 0.03}``). vLLM
        validates them. Keys that TorchTitan sets itself, or derives from a config
        field (e.g. ``max_num_batched_tokens``), raise a ``ValueError``; see
        ``add_extra_vllm_engine_args``."""

        stat_logger: VllmOtelStatLogger.Config | None = None
        """Optional logger instantiated on TP rank 0 to export vLLM metrics."""

        session_kv_holder: VLLMSessionKVHolder.Config | None = None
        """Keep each multi-turn session's cached prefix on GPU between its turns (see
        `VLLMSessionKVHolder`). None: off."""

        kv_connector: KVConnectorName = "NixlConnector"
        """vLLM KV connector that moves a prompt's KV from a prefill generator to a decode
        generator. Used only with prefill/decode disaggregation (``num_prefill_generators``).
        ``"MooncakeConnector"`` needs Mooncake's transfer engine
        (``mooncake-transfer-engine``)."""

        kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
        """Passed to vLLM as ``kv_connector_extra_config``, e.g. for Mooncake
        ``{"mooncake_protocol": "rdma", "device_name": "mlx5_0,mlx5_1"}``."""

        @property
        def kv_transfer_listener_thread_name(self) -> str | None:
            # vLLM's NIXL connector binds its listener in this thread, and engine init waits for
            # it with no timeout. Mooncake's engine init does not wait on a listener.
            if self.kv_connector == "NixlConnector":
                return "nixl_handshake_listener"
            return None

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
        prefill_decode_role: Literal["prefill", "decode"] | None = None,
    ):
        if (
            config.session_kv_holder is not None
            and reset_kv_cache_on_model_state_dict_load
        ):
            raise ValueError(
                "session_kv_holder needs reset_kv_cache_on_model_state_dict_load=False: vLLM "
                "cannot reset the prefix cache while blocks are held"
            )
        # TODO: on a prefill generator the holder would take a second reference on blocks
        # vLLM delay-frees for the transfer; untested.
        if config.session_kv_holder is not None and prefill_decode_role is not None:
            raise ValueError(
                "prefill and decode generators need session_kv_holder=None"
            )
        self.config = config
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )
        self._prefill_decode_role = prefill_decode_role
        # Mooncake only: this prefill engine's handle fields, and each in-flight prefill
        # request's transfer id (set at submit, attached to the output at finish).
        self._mooncake_handle: dict[str, Any] | None = None
        self._transfer_ids: dict[str, str] = {}
        self._kv_engine_id = f"{generator_name}-{uuid.uuid4().hex[:8]}"
        self._kv_transfer_stat_logger: _KVTransferStatLogger | None = None
        self._setup(
            model,
            parallelism=parallelism,
            checkpointer=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
            batch_invariant=debug.batch_invariant,
            # Transferring a hybrid model's KV needs the GDN conv state stored dim-first.
            conv_state_dim_first=prefill_decode_role is not None,
        )
        engine_args = self._build_engine_args(
            model,
            parallelism=parallelism,
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            model_dtype=model_dtype,
            gpu_memory_limit=gpu_memory_limit,
            max_num_batched_tokens=max_num_batched_tokens,
            seed=debug.seed,
        )
        tp_degree = parallelism.tensor_parallel_degree
        stat_loggers = None
        if rank % tp_degree == 0:
            stat_loggers = self._build_stat_loggers(
                StatLoggerContext(
                    rank=rank,
                    tp_rank=0,
                    dp_rank=rank // tp_degree,
                    generator_name=generator_name,
                    output_dir=output_dir,
                )
            )
        logger.info("Initializing LLMEngine from EngineArgs...")
        self._engine = LLMEngine.from_engine_args(
            engine_args, stat_loggers=stat_loggers
        )
        # Every rank holds the same blocks in its own scheduler; only the engine loop thread
        # touches them.
        self._session_kv_holder: VLLMSessionKVHolder | None = None
        if config.session_kv_holder is not None:
            self._session_kv_holder = config.session_kv_holder.build(
                scheduler=self._engine.engine_core.engine_core.scheduler
            )

    @staticmethod
    def _setup(
        model: Decoder.Config,
        *,
        parallelism: InferenceParallelismConfig,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        batch_invariant: bool,
        conv_state_dim_first: bool = False,
    ) -> None:
        """Process-wide vLLM setup before the engine build: register the TorchTitan model
        and config parser with vLLM, and select vLLM's model runner."""
        register_to_vllm(
            model,
            parallelism=parallelism,
            checkpointer_config=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
            conv_state_dim_first=conv_state_dim_first,
        )
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if batch_invariant:
            # The vLLM v2 logprob Triton kernel bypasses the batch-invariant aten
            # overrides; route it through trainer's function to match the trainer exactly.
            force_logprobs_fn_for_batch_invariance()

    def _build_engine_args(
        self,
        model: Decoder.Config,
        *,
        parallelism: InferenceParallelismConfig,
        model_path: str,
        max_num_seqs: int,
        model_dtype: str,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        seed: int | None,
    ) -> EngineArgs:
        """Build the vLLM ``EngineArgs`` (see ``GenerationEngine`` for the arguments)."""
        attention = model.first_full_attention
        assert attention is not None, "No full attention layer found in the model spec."
        inner_attention_config = attention.inner_attention
        assert isinstance(
            inner_attention_config,
            (VarlenInnerAttention.Config, FlexInnerAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

        enable_ep = parallelism.expert_parallel_degree > 1
        engine_kwargs = dict(
            # ``model`` is the path to the HF checkpoint directory. The
            # config is sourced from TorchTitan's model config via
            # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
            # read), but vLLM still uses this path to locate the
            # tokenizer assets and the safetensors weight shards.
            model=model_path,
            trust_remote_code=True,
            # Use the torchtitan custom config parser (registered by
            # register_to_vllm in _setup). It builds PretrainedConfig from
            # model config instead of reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=model_dtype,
            tensor_parallel_size=parallelism.tensor_parallel_degree,
            data_parallel_size=parallelism.data_parallel_degree,
            # NOTE: Monarch launches the generator workers and sets the torch
            # elastic distributed env; with external_launcher, vLLM uses that
            # world to build its process groups. vLLM does not take an
            # explicit EP degree: when this boolean is set, it converts all
            # DP * TP ranks into the expert-parallel group for MoE layers.
            enable_expert_parallel=enable_ep,
            worker_cls=TORCHTITAN_WORKER_CLS,
            # Monarch already spawned TP workers via proc mesh. "external_launcher"
            # tells vLLM to run one worker per process (no subprocess spawning)
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_limit,
            enforce_eager=self.config.cuda_graph.mode == "NONE",
            attention_config=AttentionConfig(
                backend=(
                    AttentionBackendEnum.FLEX_ATTENTION
                    if isinstance(inner_attention_config, FlexInnerAttention.Config)
                    else AttentionBackendEnum.CUSTOM
                ),
            ),
            # Enables RequestOutput.metrics, so generator metrics can be returned
            disable_log_stats=False,
            enable_cumem_allocator=False,
            # Token-in-token-out: prompts and outputs are token ids, so vLLM
            # needs no tokenizer. This also drops the tokenizer's eos_token_id
            # as a stop; the generation config's eos ids are dropped by
            # ignore_eos in _build_sampling_params.
            skip_tokenizer_init=True,
        )
        engine_kwargs["max_model_len"] = model.max_context_length
        # Return logprobs of the distribution vLLM samples from (after temperature). vLLM's default
        # returns the raw model's logprobs, before temperature.
        engine_kwargs["logprobs_mode"] = "processed_logprobs"
        engine_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        # Smaller `priority` runs first and is preempted last; we pass `EngineRequest.group_id`
        # (-1 for all validation), so older groups have higher priority (see its docstring).
        # TODO: validation (priority -1) runs before training. Revisit once validation overlaps training.
        engine_kwargs["scheduling_policy"] = "priority"
        # Which sliding-window / Mamba (GDN) state blocks are hashed into the prefix
        # cache; full-attention groups hash every block regardless.
        # - 0 (vLLM default since v0.29): only the replay boundary (the last block
        #   boundary before the prompt's final token) and shared-prefix junctions.
        #   Other prompt blocks and every block crossed during decode are freed
        #   unhashed, so a multi-turn rollout's next turn, whose prompt contains the
        #   previous completion, cannot reuse anything past the previous prompt.
        # - None: every full block boundary, as before v0.29. Blocks are still freed
        #   when the request moves past them; hashed ones stay reusable until evicted.
        engine_kwargs["prefix_cache_retention_interval"] = None
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = self.config.cuda_graph.get_vllm_compilation_config(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            expert_sequence_parallel_size=parallelism.expert_sequence_parallel_size,
            enable_sequence_parallel=parallelism.enable_sequence_parallel,
        )
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if seed is not None:
            engine_kwargs["seed"] = seed
        if self._prefill_decode_role is not None:
            self._mooncake_handle = add_kv_transfer_config(
                engine_kwargs,
                self._prefill_decode_role,
                connector=self.config.kv_connector,
                extra_config=self.config.kv_connector_extra_config,
                engine_id=self._kv_engine_id,
            )
        add_extra_vllm_engine_args(engine_kwargs, self.config.extra_engine_args)
        return EngineArgs(**engine_kwargs)

    def _build_stat_loggers(self, context: StatLoggerContext) -> list | None:
        """Build the vLLM stat logger factories: ``config.stat_logger``, and the KV transfer
        stats of a prefill or decode engine. None if there are none."""
        factories = []
        stat_logger_config = self.config.stat_logger
        if stat_logger_config is None:
            logger.info(
                "VllmOtelStatLogger inactive because stat_logger=None. To record vLLM "
                "metrics, set VLLMEngine.Config.stat_logger to VllmOtelStatLogger.Config() "
                "and set OTEL_METRICS_EXPORTER=jsonl or otlp"
            )
        else:

            def build_stat_logger(vllm_config, engine_index):
                return stat_logger_config.build(
                    vllm_config=vllm_config,
                    engine_index=engine_index,
                    context=context,
                )

            factories.append(build_stat_logger)

        if self._prefill_decode_role is not None:

            def build_kv_transfer_stat_logger(vllm_config, engine_index):
                self._kv_transfer_stat_logger = _KVTransferStatLogger(
                    vllm_config, engine_index
                )
                return self._kv_transfer_stat_logger

            factories.append(build_kv_transfer_stat_logger)
        return factories or None

    @property
    def data_parallel_size(self) -> int:
        """The data-parallel size vLLM derived for this engine."""
        return self._engine.vllm_config.parallel_config.data_parallel_size

    @property
    def data_parallel_rank(self) -> int:
        """The data-parallel rank vLLM derived for this engine."""
        return self._engine.vllm_config.parallel_config.data_parallel_rank

    @property
    def holds_session_kv(self) -> bool:
        return self._session_kv_holder is not None

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        ``load_prefetched_model_state_dict``. Safe to fill from any thread while no load runs."""
        return self._get_model()._prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        vLLM rejected."""
        rejected_requests: list[tuple[str, Exception]] = []
        for request in requests:
            prompt = {"prompt_token_ids": request.prompt_token_ids}
            if not self._reset_kv_cache_on_model_state_dict_load:
                # Salt by the pinned version so a request only reuses KV
                # computed under that version.
                prompt["cache_salt"] = str(request.min_policy_version)
            # vLLM validates a request (prompt length, sampling params, token ids) as it
            # admits it: render_cmpl and InputProcessor.process_inputs raise
            # VLLMClientError (e.g. VLLMValidationError, not a ValueError subclass) or
            # ValueError before LLMEngine.add_request registers the request with the
            # output processor and engine core, so a rejected request leaves no engine
            # state (recheck on a vLLM upgrade). The checks are deterministic, so every TP
            # rank of this replica rejects the same requests.
            try:
                # render_cmpl is vLLM's input pipeline (tokenize is a no-op for tokenized
                # prompts); the high-level entry stays resilient to vLLM internals vs
                # vllm.inputs.tokens_input.
                [engine_input] = self._engine.renderer.render_cmpl([prompt])
                kv_transfer_params = request.kv_transfer_params
                transfer_id = None
                if self._prefill_decode_role == "prefill":
                    kv_transfer_params = {"do_remote_decode": True}
                    if self._mooncake_handle is not None:
                        # Mooncake pairs the two hops by this id.
                        transfer_id = uuid.uuid4().hex
                        kv_transfer_params["transfer_id"] = transfer_id
                # vLLM appends a random suffix to the id; the scheduler knows only that one.
                internal_request_id = self._engine.add_request(
                    request_id=request.request_id,
                    prompt=engine_input,
                    params=_build_sampling_params(
                        request.sampling,
                        prefill_decode_role=self._prefill_decode_role,
                        kv_transfer_params=kv_transfer_params,
                    ),
                    arrival_time=request.arrival_time,
                    priority=max(request.group_id, -1),
                )
                if transfer_id is not None:
                    self._transfer_ids[request.request_id] = transfer_id
                if self._session_kv_holder is not None:
                    self._session_kv_holder.track(
                        internal_request_id,
                        request.routing_session_id,
                        request.group_id,
                    )
            except (ValueError, VLLMClientError) as exc:
                rejected_requests.append((request.request_id, exc))
        return rejected_requests

    def has_unfinished_requests(self) -> bool:
        """Whether any admitted request is still running."""
        return self._engine.has_unfinished_requests()

    def step(self) -> list[FinishedRequest]:
        """Run one engine step, a TP collective, and return the requests it finished."""
        with torch.no_grad():
            request_outputs = self._engine.step()
        if self._session_kv_holder is not None:
            self._session_kv_holder.after_step()
        finished = [
            _finished_request(request_output) for request_output in request_outputs
        ]
        if self._mooncake_handle is not None:
            # Mooncake's prefill outputs carry no handle: build it from this engine's address
            # and the request's transfer id.
            finished = [
                dataclasses.replace(
                    request,
                    kv_transfer_params={
                        **self._mooncake_handle,
                        "transfer_id": self._transfer_ids.pop(request.request_id),
                    },
                )
                if request.request_id in self._transfer_ids
                else request
                for request in finished
            ]
        return finished

    def holds_kv_for_transfer(self) -> bool:
        """Prefill engines only: whether finished requests still hold KV for a decode engine.
        vLLM frees it in an engine step once the pull is done, so the loop keeps stepping."""
        return (
            self._prefill_decode_role == "prefill"
            and self._engine.engine_core.engine_core.scheduler.has_finished_requests()
        )

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        if self._session_kv_holder is not None:
            self._session_kv_holder.release(session_ids, group_ids)

    def log_stats(self, step: int) -> None:
        if self._session_kv_holder is not None:
            logger.info(
                "Session KV at step %d: %d sessions held, %d evicted so far",
                step,
                self._session_kv_holder.num_sessions,
                self._session_kv_holder.num_evicted,
            )
        # TODO: with generator DP > 1 this reports only DP rank 0's transfers.
        if self._kv_transfer_stat_logger is not None:
            stats = self._kv_transfer_stat_logger.drain()
            if stats is not None and not stats.is_empty():
                metrics = _kv_transfer_metrics(stats)
                logger.info(
                    "KV transfer at step %d: %s",
                    step,
                    ", ".join(f"{name}={value:g}" for name, value in metrics.items()),
                )
                sl.log_trace_scalar(metrics)

    def load_prefetched_model_state_dict(self) -> None:
        """Copy the prefetched model state dict from pinned CPU memory into the model, and
        reset the KV cache when configured."""
        model = self._get_model()
        model.load_state_dict(model._prefetched_model_state_dict)
        if self._reset_kv_cache_on_model_state_dict_load:
            # Always reset running requests too: the only reason to reset is a strict
            # recompute under the new weights. Keeping running requests' KV while hiding
            # old KV from new requests is already what the default (no reset) does via the salt.
            self._engine.reset_prefix_cache(
                reset_running_requests=True,
            )

    def shutdown(self) -> None:
        """Close renderer-local resources.

        With `external_launcher`, vLLM reuses the process group and actor lifetime that
        Monarch owns. Calling vLLM's internal `engine_core.shutdown()` can block while
        Monarch is also trying to stop the same actor mesh, so process teardown is left to
        `ProcMesh.stop()`.
        """
        renderer = getattr(self._engine, "renderer", None)
        if renderer is not None:
            logger.info("Shutting down vLLM renderer")
            renderer.shutdown()
        self._session_kv_holder = None

    def _get_model(self):
        """The ``VLLMModelWrapper`` the engine runs."""
        return self._engine.model_executor.driver_worker.get_model()


def _build_sampling_params(
    sampling: SamplingConfig,
    *,
    prefill_decode_role: Literal["prefill", "decode"] | None = None,
    kv_transfer_params: dict[str, Any] | None = None,
) -> SamplingParams:
    """Translate a `SamplingConfig` into vLLM `SamplingParams` (n=1).

    ``seed`` and ``stop_token_ids`` are carried on the ``SamplingConfig``
    (the controller fills ``stop_token_ids`` and the rollouter offsets
    ``seed`` per sample), so each sample in a group is a distinct ``n=1``
    request that stays diverse and bitwise-reproducible.

    The engine loads no tokenizer, so its ``eos_token_id`` is not a stop,
    but vLLM still adds the generation config's ``eos_token_id`` (checkpoint
    ``generation_config.json`` or the ``vllm_registry`` HF config).
    ``ignore_eos`` turns that off, so the renderer's ``stop_token_ids``
    (which include EOS) are the only stops.
    """
    max_tokens, extra_args = sampling.max_tokens, None
    stop_token_ids = sampling.stop_token_ids
    if prefill_decode_role == "prefill":
        # Only the prompt's KV is kept; the decode generator samples every output token.
        max_tokens = 1
        # The connectors keep a prefill request's KV for the decode generator only when the
        # request stops at its length cap, so a sampled stop token must not end it first.
        stop_token_ids = None
        extra_args = {
            "kv_transfer_params": kv_transfer_params or {"do_remote_decode": True}
        }
    elif kv_transfer_params is not None:
        extra_args = {"kv_transfer_params": kv_transfer_params}
    return SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=max_tokens,
        n=1,  # always expects a single sample per request. Caller can call N times.
        stop_token_ids=stop_token_ids,
        # Drops the generation config's eos ids, which vLLM merges into
        # stop_token_ids even with skip_tokenizer_init.
        ignore_eos=True,
        seed=sampling.seed,
        logprobs=0,  # return only the sampled token's logprob (for the GRPO ratio)
        # Token ids in, token ids and logprob floats out: stops are token ids and nothing reads
        # text, so skip vLLM's per-token detokenization and per-token logprob dicts.
        detokenize=False,
        flat_logprobs=True,
        # Return each request's result once, when it is fully done, instead of streaming partial
        # outputs as tokens arrive.
        # TODO(async-rl): use RequestOutputKind.CUMULATIVE for exact per-token
        #   (start_token, version) boundaries; today we keep only the per-turn min/max.
        output_kind=RequestOutputKind.FINAL_ONLY,
        extra_args=extra_args,
    )


def _finished_request(request_output: RequestOutput) -> FinishedRequest:
    """Convert a finished ``RequestOutput`` into a ``FinishedRequest``."""
    # We enforce n=1 in sampling params -> exactly one CompletionOutput per finished request
    # Here we just sanity check it (a single engine.step may still finish several requests).
    if len(request_output.outputs) != 1:
        raise ValueError(
            f"expected n=1 (one sample per request), got "
            f"{len(request_output.outputs)} for {request_output.request_id}"
        )

    # flat_logprobs=True: vLLM returns logprobs as plain lists instead of one dict per token.
    # logprobs=0 keeps only the sampled token, so `.logprobs` has exactly one float per generated token.
    completion_output = request_output.outputs[0]
    return FinishedRequest(
        request_id=request_output.request_id,
        token_ids=list(completion_output.token_ids),
        token_logprobs=list(completion_output.logprobs.logprobs),
        finish_reason=completion_output.finish_reason,
        metrics_inputs=_extract_request_metrics_inputs(request_output),
        kv_transfer_params=request_output.kv_transfer_params,
    )


def _extract_request_metrics_inputs(
    request_output: RequestOutput,
) -> RequestMetricsInputs:
    """Pull the raw metric inputs off a finished ``RequestOutput``.

    For an `add_request` call, vLLM returns a RequestOutput carrying a single
    `RequestStateStats`.
    """
    stats = request_output.metrics
    if stats is None:
        return RequestMetricsInputs(
            num_cached_tokens=request_output.num_cached_tokens, has_stats=False
        )
    return RequestMetricsInputs(
        num_cached_tokens=request_output.num_cached_tokens,
        has_stats=True,
        queued_ts=stats.queued_ts,
        scheduled_ts=stats.scheduled_ts,
        first_token_ts=stats.first_token_ts,
        last_token_ts=stats.last_token_ts,
        first_token_latency=stats.first_token_latency,
        num_generation_tokens=stats.num_generation_tokens,
        num_preemptions=stats.num_preemptions,
    )
