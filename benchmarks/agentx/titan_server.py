"""TitanRL's generator, standalone, behind the Verifiers generation endpoint.

The RL loop puts the same pieces together: a ``Generator`` (engine loop, dispatcher,
TorchTitan model in vLLM or SGLang) whose ``generate`` backs ``GenerationServer``'s
``/inference/v1/generate``. Here the generator loads its initial HF weights itself
instead of receiving them from a trainer, and the server adds benchmark routes:

    X-TorchTitan-Metrics      response header: the request's generator metrics
                              (time_to_first_token_ms, inter_token_latency_ms,
                              queue_time_ms, num_cached_tokens, ...)
    POST /torchtitan/release  {"session_ids": [...], "group_ids": [...]}: a rollout
                              or group ended (drops pinned cache salts / held KV)
    POST /torchtitan/profile  {"seconds": 5, "trace": path}: CUDA kernel table (rank 0)
    POST /torchtitan/weight_sync
                              load the prefetched weights as policy version v+1, as
                              a trainer push does (with --weight-sync-buffers they
                              hold the current weights, so outputs do not change;
                              --reset-kv-on-sync also drops the KV)
    POST /v1/chat/completions OpenAI chat, for the official AgentX harness (AIPerf):
                              vLLM's chat front end and native vLLM's chat sampling,
                              in vLLM's render server (--chat-render-url) or in-process

    torchrun --nproc_per_node=1 titan_server.py --engine vllm \\
        --model-path <hf model dir> --port 8300
"""

from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()

import argparse
import asyncio
import contextvars
import datetime
import importlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed.distributed_c10d as c10d
from aiohttp import web

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig, ParallelismConfig
from torchtitan.rl.examples.verifiers.generation_server import GenerationServer
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

logger = logging.getLogger("titan_server")

# The current request's metrics, filled by the wrapped GenerateFn and read by the
# handler that awaits it (same task, so the context carries over).
_METRICS: contextvars.ContextVar[dict] = contextvars.ContextVar("metrics")

# Chat requests waiting for their first token: request id -> callback taking the output token ids.
# The chat handler (rank 0) registers it; the patched engine step (engine loop thread) fires it.
_FIRST_TOKEN_WAITERS: dict[str, Callable[[list[int]], None]] = {}

# The vLLM SamplingParams fields that decide what a chat request generates. The handler copies them
# from the params native vLLM's chat endpoint would build; every TP rank rebuilds the params.
_CHAT_SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
    "frequency_penalty", "seed", "max_tokens", "min_tokens", "ignore_eos", "stop_token_ids",
    "logprobs",
)


@dataclass(kw_only=True, slots=True)
class _ChatSampling(SamplingConfig):
    """A chat request's sampling: native vLLM's SamplingParams fields for it."""

    vllm_fields: dict | None = None


def _install_chat_engine_hooks() -> None:
    """Let the vLLM engine run chat requests as native vLLM does (on every rank).

    TitanRL's engine builds RL sampling (EOS off, the renderer's stop ids, sampled-token logprobs)
    and returns each request once, finished. A chat request runs native vLLM's sampling fields
    with no logprobs instead, and each step reports newly started requests to the handler.
    """
    from torchtitan.rl.generator.vllm import engine as vllm_engine
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    build_sampling_params = vllm_engine._build_sampling_params
    finished_request = vllm_engine._finished_request
    step = vllm_engine.VLLMEngine.step

    def build_chat_sampling_params(sampling):
        fields = getattr(sampling, "vllm_fields", None)
        if fields is None:
            return build_sampling_params(sampling)
        # Token ids out: the handler decodes once at the end (native vLLM detokenizes in its API
        # server process, off the engine core).
        return SamplingParams(
            **fields, n=1, detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY
        )

    def finished_chat_request(request_output):
        output = request_output.outputs[0]
        if output.logprobs is not None:
            return finished_request(request_output)
        return vllm_engine.FinishedRequest(
            request_id=request_output.request_id,
            token_ids=list(output.token_ids),
            token_logprobs=[0.0] * len(output.token_ids),
            finish_reason=output.finish_reason,
            metrics_inputs=vllm_engine._extract_request_metrics_inputs(request_output),
        )

    def step_and_report_first_tokens(self):
        finished = step(self)
        if _FIRST_TOKEN_WAITERS:
            processor = self._engine.output_processor
            for request_id in list(_FIRST_TOKEN_WAITERS):
                internal_ids = processor.external_req_ids.get(request_id)
                state = processor.request_states.get(internal_ids[0]) if internal_ids else None
                if state is not None and not state.is_prefilling:
                    callback = _FIRST_TOKEN_WAITERS.pop(request_id, None)
                    if callback is not None:
                        callback(list(state.detokenizer.output_token_ids))
        return finished

    vllm_engine._build_sampling_params = build_chat_sampling_params
    vllm_engine._finished_request = finished_chat_request
    vllm_engine.VLLMEngine.step = step_and_report_first_tokens


@dataclass(kw_only=True, slots=True)
class _RenderedChat:
    """A chat request after vLLM's chat front end: what the engine runs and the response needs."""

    prompt_ids: list[int]
    sampling_fields: dict
    model: str | None
    stream: bool
    include_usage: bool
    skip_special_tokens: bool


def _chat_error(message: str, status: int) -> web.Response:
    kind = "BadRequestError" if status == 400 else "InternalServerError"
    return web.json_response(
        {"error": {"message": message, "type": kind, "code": status}}, status=status
    )


class BenchGenerationServer(GenerationServer):
    def __init__(self, config: GenerationServer.Config, generator: Generator) -> None:
        super().__init__(config)
        self.generator = generator
        self._sync_lock = asyncio.Lock()
        # The native servers' --served-model-name.
        self.model_id = "model"
        self._chat = None
        # vLLM's GPU-less render server (`vllm launch render`), a separate process like native
        # vLLM's API server: rendering ~80K-token prompts there keeps it off the GIL that the
        # engine loop thread needs. None renders in this process.
        self.chat_render_url: str | None = None
        # --release-on-final-turn: release a session's held KV after the request the harness marks
        # as its conversation's last turn (X-Session-Final-Turn: 1), as the RL rollouter releases
        # a rollout's sessions when it ends.
        self.release_on_final_turn = False
        self.final_turn_releases = 0
        self._http = None
        self._tokenizer = None

    def set_generate_fn(self, generate_fn) -> None:
        async def generate(prompt_token_ids, **kwargs):
            completion = await generate_fn(prompt_token_ids, **kwargs)
            slot = _METRICS.get(None)
            if slot is not None and completion is not None:
                for metric in completion.metrics:
                    if isinstance(metric.value, m.Max):
                        slot[metric.key.split("/", 1)[-1]] = metric.value.value
            return completion

        super().set_generate_fn(generate)

    async def start(self) -> None:
        # GenerationServer.start with the benchmark routes added.
        app = web.Application(client_max_size=1 << 30)
        app.router.add_get("/healthz", self._handle_health_request)
        app.router.add_get("/v1/models", self._handle_models_request)
        app.router.add_post("/inference/v1/generate", self._handle_generate_request)
        app.router.add_post("/torchtitan/release", self._handle_release)
        app.router.add_post("/torchtitan/weight_sync", self._handle_weight_sync)
        app.router.add_post("/torchtitan/profile", self._handle_profile)
        app.router.add_get("/server_info", self._handle_server_info)
        app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        if self.chat_render_url:
            await self._connect_render_server()
        else:
            self._chat = self._build_chat_front_end()
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, self.host, self.requested_port).start()
        self.bound_port = self.requested_port

    async def _handle_generate_request(self, request: web.Request) -> web.Response:
        slot: dict = {}
        token = _METRICS.set(slot)
        try:
            response = await super()._handle_generate_request(request)
        finally:
            _METRICS.reset(token)
        if slot:
            response.headers["X-TorchTitan-Metrics"] = json.dumps(slot)
        return response

    async def _handle_release(self, request: web.Request) -> web.Response:
        body = await request.json()
        session_ids = list(body.get("session_ids") or [])
        group_ids = list(body.get("group_ids") or [])
        for session_id in session_ids:
            self.pop_generation_metadata(session_id)
        if session_ids:
            await self.generator.release_sessions(session_ids)
        if group_ids:
            await self.generator.release_groups(group_ids)
        return web.json_response({"ok": True})

    async def _release_session(self, session_id: str) -> None:
        self.pop_generation_metadata(session_id)
        await self.generator.release_sessions([session_id])
        self.final_turn_releases += 1
        if self.final_turn_releases in (1, 10) or self.final_turn_releases % 500 == 0:
            logger.info("final-turn releases: %d", self.final_turn_releases)

    async def _connect_render_server(self) -> None:
        """Wait for the render server, and load the tokenizer that decodes outputs here."""
        import aiohttp
        from transformers import AutoTokenizer

        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
        for _ in range(600):
            try:
                async with self._http.get(f"{self.chat_render_url}/health") as response:
                    if response.status == 200:
                        break
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError(f"render server {self.chat_render_url} not ready after 600 s")
        engine = self.generator._engine_loop._engine._engine
        self._tokenizer = AutoTokenizer.from_pretrained(engine.vllm_config.model_config.tokenizer)
        logger.info("chat front end: vLLM render server at %s", self.chat_render_url)

    async def _render_chat(self, request: web.Request) -> "_RenderedChat | web.Response":
        """The chat request through vLLM's chat front end: the render server's
        ``/v1/chat/completions/render`` (prompt ids and the SamplingParams native vLLM's chat
        endpoint builds), or the same steps in this process."""
        if self.chat_render_url:
            import msgspec
            from vllm import SamplingParams

            body = await request.read()
            async with self._http.post(
                f"{self.chat_render_url}/v1/chat/completions/render",
                data=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                payload = await response.read()
                if response.status != 200:
                    return web.Response(
                        body=payload, status=response.status, content_type="application/json"
                    )
            rendered = msgspec.json.decode(payload)
            # The render server omits SamplingParams fields at their defaults; converting back to
            # SamplingParams restores them, as vLLM's own transport does.
            params = msgspec.convert(rendered["sampling_params"], SamplingParams)
            stream_options = rendered.get("stream_options") or {}
            return _RenderedChat(
                prompt_ids=rendered["token_ids"],
                sampling_fields={name: getattr(params, name) for name in _CHAT_SAMPLING_FIELDS},
                model=rendered.get("model"),
                stream=bool(rendered.get("stream")),
                include_usage=bool(stream_options.get("include_usage")),
                skip_special_tokens=params.skip_special_tokens,
            )

        import pydantic
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
        from vllm.renderers.inputs.preprocess import extract_prompt_components

        try:
            chat = ChatCompletionRequest.model_validate(await request.json())
        except (ValueError, pydantic.ValidationError) as error:
            return _chat_error(str(error), 400)
        rendered = await self._chat.render_chat(chat)
        if isinstance(rendered, ErrorResponse):
            return web.json_response(rendered.model_dump(), status=rendered.error.code)
        model_config = self._chat.model_config
        prompt_ids = list(extract_prompt_components(model_config, rendered[1][0]).token_ids)
        try:
            max_tokens = get_max_tokens(
                model_config.max_model_len,
                chat.max_completion_tokens
                if chat.max_completion_tokens is not None
                else chat.max_tokens,
                len(prompt_ids),
                self._chat_defaults,
                model_config.override_generation_config.get("max_new_tokens"),
                truncate_prompt_tokens=chat.truncate_prompt_tokens,
            )
            params = chat.to_sampling_params(max_tokens, self._chat_defaults)
        except ValueError as error:
            return _chat_error(str(error), 400)
        return _RenderedChat(
            prompt_ids=prompt_ids,
            sampling_fields={name: getattr(params, name) for name in _CHAT_SAMPLING_FIELDS},
            model=chat.model,
            stream=bool(chat.stream),
            include_usage=bool(chat.stream_options and chat.stream_options.include_usage),
            skip_special_tokens=chat.skip_special_tokens,
        )

    def _build_chat_front_end(self):
        """vLLM's OpenAI chat front end over this engine's model config, as ``vllm serve`` builds
        it with default flags: the checkpoint's chat template, no tool or reasoning parser."""
        import copy

        from vllm.renderers.online_renderer import OnlineRenderer
        from vllm.renderers.registry import renderer_from_config

        engine = getattr(self.generator._engine_loop._engine, "_engine", None)
        if engine is None or not hasattr(engine, "vllm_config"):
            return None
        # The engine skips its tokenizer (token ids in and out); the front end needs it.
        model_config = copy.copy(engine.vllm_config.model_config)
        object.__setattr__(model_config, "skip_tokenizer_init", False)
        vllm_config = copy.copy(engine.vllm_config)
        object.__setattr__(vllm_config, "model_config", model_config)
        front_end = OnlineRenderer(
            model_config,
            renderer_from_config(vllm_config),
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        front_end.warmup()
        # Native vLLM's chat defaults: the checkpoint's generation config.
        self._chat_defaults = model_config.get_diff_sampling_param()
        logger.info("chat front end ready, sampling defaults %s", self._chat_defaults)
        return front_end

    async def _handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """OpenAI ``/v1/chat/completions``, for the official AgentX harness (AIPerf).

        The prompt goes through vLLM's chat front end and the sampling through native vLLM's chat
        defaults, so both systems get the same prompt tokens and sampling. The engine returns a
        request once, finished: a stream sends the first token when it exists and the rest at
        the end, so the harness's TTFT and mean ITL hold, its per-chunk latencies do not.
        """
        if (self._chat is None and not self.chat_render_url) or self.generate_fn is None:
            return _chat_error("no chat front end on this engine", 404)
        chat = await self._render_chat(request)
        if isinstance(chat, web.Response):
            return chat
        prompt_ids = chat.prompt_ids
        sampling = _ChatSampling(
            max_tokens=chat.sampling_fields["max_tokens"],
            stop_token_ids=list(chat.sampling_fields["stop_token_ids"] or []),
            vllm_fields=chat.sampling_fields,
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        # AIPerf sends one correlation id per conversation (X-Correlation-ID, or the name given
        # with --session-header).
        session_id = (
            request.headers.get("X-Session-ID")
            or request.headers.get("X-Correlation-ID")
            or request_id
        )
        loop = asyncio.get_running_loop()
        first_token: asyncio.Future = loop.create_future()

        def on_first_token(token_ids: list[int]) -> None:  # engine loop thread
            loop.call_soon_threadsafe(
                lambda: first_token.done() or first_token.set_result(token_ids)
            )

        slot: dict = {}
        context = _METRICS.set(slot)
        try:
            if chat.stream:
                _FIRST_TOKEN_WAITERS[request_id] = on_first_token
            generation = asyncio.ensure_future(
                self.generate_fn(
                    prompt_ids,
                    request_id=request_id,
                    group_id=0,
                    routing_session_id=session_id,
                    sampling_config=sampling,
                )
            )
        finally:
            _METRICS.reset(context)
        if self.release_on_final_turn and request.headers.get("X-Session-Final-Turn") == "1":
            generation.add_done_callback(
                lambda _: asyncio.ensure_future(self._release_session(session_id))
            )

        tokenizer = self._tokenizer if self.chat_render_url else self._chat.renderer.tokenizer

        def decode(token_ids: list[int]) -> str:
            return tokenizer.decode(token_ids, skip_special_tokens=chat.skip_special_tokens)

        def usage(completion) -> dict:
            return {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion.token_ids),
                "total_tokens": len(prompt_ids) + len(completion.token_ids),
                "prompt_tokens_details": {
                    "cached_tokens": int(slot.get("num_cached_tokens", 0))
                },
            }

        created = int(time.time())
        if not chat.stream:
            try:
                completion = await generation
            except Exception as error:
                logger.exception("chat generation failed for %s", request_id)
                return _chat_error(str(error), 500)
            message = {"role": "assistant", "content": decode(completion.token_ids)}
            return web.json_response(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": chat.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "logprobs": None,
                            "finish_reason": completion.finish_reason,
                        }
                    ],
                    "usage": usage(completion),
                }
            )

        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)

        async def send(event: dict | str) -> None:
            data = event if isinstance(event, str) else json.dumps(event)
            await response.write(f"data: {data}\n\n".encode())

        def chunk(delta: dict, finish_reason: str | None = None) -> dict:
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": chat.model,
                "choices": [choice],
            }

        sent = None
        try:
            await asyncio.wait({generation, first_token}, return_when=asyncio.FIRST_COMPLETED)
            if first_token.done() and not generation.done():
                sent = decode(first_token.result())
                # A partial UTF-8 character decodes to U+FFFD: leave it for the final chunk.
                if sent.endswith("\ufffd"):
                    sent = ""
                await send(chunk({"role": "assistant", "content": ""}))
                if sent:
                    await send(chunk({"content": sent}))
            completion = await generation
        except Exception as error:
            logger.exception("chat generation failed for %s", request_id)
            await send({"error": {"message": str(error), "type": "InternalServerError", "code": 500}})
            await send("[DONE]")
            return response
        finally:
            _FIRST_TOKEN_WAITERS.pop(request_id, None)
        text = decode(completion.token_ids)
        if sent is None:
            await send(chunk({"role": "assistant", "content": ""}))
            sent = ""
        if not text.startswith(sent):
            logger.warning("%s: first-token text %r is not a prefix of the output", request_id, sent)
        await send(chunk({"content": text[len(sent):]}, completion.finish_reason))
        if chat.include_usage:
            await send(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": chat.model,
                    "choices": [],
                    "usage": usage(completion),
                }
            )
        await send("[DONE]")
        await response.write_eof()
        return response

    async def _handle_server_info(self, request: web.Request) -> web.Response:
        """The engine's effective VllmConfig in the format of native vLLM's dev-mode
        ``/server_info?config_format=json``, so run_node.sh can diff the two systems' settings."""
        import pydantic
        import vllm
        from vllm.config import VllmConfig

        engine = getattr(self.generator._engine_loop._engine, "_engine", None)
        if engine is None or not hasattr(engine, "vllm_config"):
            return web.json_response({"error": "not a vLLM engine"}, status=404)
        config = pydantic.TypeAdapter(VllmConfig).dump_python(engine.vllm_config, mode="json", fallback=str)
        return web.json_response(
            {"vllm_config": config, "vllm_version": vllm.__version__},
            dumps=lambda o: json.dumps(o, default=str),
        )

    async def _handle_profile(self, request: web.Request) -> web.Response:
        """CUDA kernels of every thread on this rank for ``seconds`` while traffic runs
        (the engine loop thread included): a kernel table, and a Chrome trace at ``trace``."""
        from torch.profiler import profile, ProfilerActivity

        body = await request.json() if request.can_read_body else {}
        prof = profile(activities=[ProfilerActivity.CUDA])
        prof.start()
        await asyncio.sleep(float(body.get("seconds", 5.0)))
        prof.stop()
        if body.get("trace"):
            prof.export_chrome_trace(body["trace"])
        return web.Response(
            text=prof.key_averages().table(sort_by="cuda_time_total", row_limit=int(body.get("rows", 40)))
        )

    async def _handle_weight_sync(self, request: web.Request) -> web.Response:
        # A trainer push: the generator applies its prefetched buffers as the next
        # policy version. In-flight requests keep running (the endpoint never drains).
        reason = getattr(self.generator, "_weight_sync_disabled", None)
        if reason:
            return web.json_response({"error": reason}, status=409)
        async with self._sync_lock:
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await self.generator.load_model_state_dict(self.generator.policy_version + 1)
            return web.json_response(
                {
                    "policy_version": self.generator.policy_version,
                    "load_s": loop.time() - t0,
                }
            )



# An idle benchmark server outlives gloo's 30 min default: ranks > 0 block on the engine
# loop's decision broadcast and time out. The RL loop never idles that long.
c10d.default_pg_timeout = datetime.timedelta(days=7)

def _torchtitan_expert_layout(weight: torch.Tensor, gated: bool) -> torch.Tensor:
    """TorchTitan's layout of an expert weight that ``trtllm_moe.relayout`` rewrote in place:
    the inverse of vLLM's ``convert_moe_weights_to_flashinfer_trtllm_block_layout`` (rows
    permuted for the fused SwiGLU epilogue, then 128-byte K blocks moved outermost).
    w13 is ``[E, 2, F, D]`` (gate, up), w2 ``[E, D, F]``.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    weight = weight.to_local() if hasattr(weight, "to_local") else weight
    num_experts = weight.shape[0]
    rows = weight[0].numel() // weight.shape[-1]
    row_bytes = weight.shape[-1] * weight.element_size()
    template = torch.empty(rows, row_bytes, dtype=torch.uint8, device=weight.device)
    if gated:
        permute = _maybe_get_cached_w3_w1_permute_indices({}, template, 128, is_gated_act_gemm=True)
        permute = (permute + rows // 2) % rows
    else:
        permute = get_w2_permute_indices_with_cache({}, template, 128)
    # Kernel layout: [E, K blocks, rows, 128 B] with row m holding original row permute[m].
    blocks = weight.reshape(-1).view(torch.uint8).view(num_experts, row_bytes // 128, rows, 128)
    original = torch.empty(num_experts, rows, row_bytes // 128, 128, dtype=torch.uint8, device=weight.device)
    original[:, permute.to(weight.device)] = blocks.permute(0, 2, 1, 3)
    return original.view(-1).view(weight.dtype).view(weight.shape)


def _fill_prefetch_buffers(generator: Generator) -> None:
    """Copy the loaded weights into the generator's pinned weight-sync buffers.

    In the RL loop a trainer push fills them; here a sync must leave the outputs
    unchanged. The generator's FSDP mesh has one rank, so each parameter's
    unsharded tensor (what the forward reads) holds exactly its sharded copy.
    """
    engine_loop = generator._engine_loop

    def fill():
        engine = engine_loop._engine
        wrapper = engine._get_model()
        # The TorchTitan decoder: VLLMModelWrapper.model, SGLang's wrapper .decoder.
        model = getattr(wrapper, "decoder", None) or wrapper.model
        buffers = generator._prefetched_model_state_dict
        # Climb steps that rewrite expert weights in place after each load leave the live
        # tensors in their kernel layout. A trainer pushes TorchTitan's layout, so the
        # TRT-LLM-gen experts are turned back into it; the TP experts (A100) have no inverse.
        if any(hasattr(m, "_tp_weight") for m in model.modules()):
            generator._weight_sync_disabled = "experts are in a kernel layout; no buffers to sync from"
            return
        # Dist-MoE block-major experts (TORCHTITAN_DIST_MOE_BLOCK_MAJOR=1) have no inverse here.
        if any(getattr(m, "block_major_weights", False) for m in model.modules()):
            generator._weight_sync_disabled = "Dist-MoE experts are block-major; no buffers to sync from"
            return
        live = dict(model.named_parameters())
        live.update(dict(model.named_buffers()))
        kernel_layout = {
            f"{prefix}.{name}.weight": name == "w13"
            for prefix, module in model.named_modules()
            if hasattr(module, "_trtllm_w13")
            for name in ("w13", "w2")
        }
        missing = [k for k in buffers if k not in live] + [k for k in kernel_layout if k not in buffers]
        if missing:
            raise RuntimeError(f"no live tensor or sync buffer for {len(missing)} keys, e.g. {missing[:3]}")
        with torch.no_grad():
            for key, buffer in buffers.items():
                src = live[key]
                src = src.to_local() if hasattr(src, "to_local") else src
                if key in kernel_layout:
                    # One layer's temporary at a time.
                    src = _torchtitan_expert_layout(src, gated=kernel_layout[key])
                dst = buffer.to_local() if hasattr(buffer, "to_local") else buffer
                if dst.shape != src.shape:
                    raise RuntimeError(f"{key}: buffer {tuple(dst.shape)} != live {tuple(src.shape)}")
                dst.copy_(src)
        torch.cuda.synchronize()

    generator._engine_loop_thread.call(fill)


def _build_generator(args) -> Generator:
    if args.engine == "vllm":
        _install_chat_engine_hooks()
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=args.tp, expert_parallel_degree=args.ep
    )
    _, model_config = build_rl_model_configs(
        importlib.import_module(args.model_module).build_model_config,
        args.flavor,
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=parallelism,
        fp32_lm_head=not args.bf16_lm_head,
        seq_len=args.max_model_len,
        attn_backend="varlen",
    )
    if args.override:
        # As the RL controller applies config.override to the generator model (e.g. Felipe's
        # torchtitan_recipes.overrides.triton_routed_experts.triton_routed_experts).
        from torchtitan.config import apply_overrides, OverrideConfig

        apply_overrides(OverrideConfig(imports=list(args.override)), model_config)
    if args.batch_invariant or args.no_compile_regions:
        # Batch-invariant mode needs it, as the batch-invariant RL recipes set it.
        model_config.local_compile_regions = []
    if args.batch_invariant and (args.session_kv_holder or args.kv_offload_gib):
        raise ValueError("--batch-invariant resets the KV cache on weight sync: no KV holder or CPU KV tier")
    if args.engine == "vllm":
        from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
        from torchtitan.rl.generator.vllm.engine import VLLMEngine

        holder = None
        if args.session_kv_holder or args.kv_offload_gib:
            from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder

            holder_kwargs = {}
            if args.kv_offload_gib:
                # Felipe's up#5299: idle session KV goes to host RAM instead of being recomputed.
                from torchtitan.rl.generator.vllm.cpu_kv_offload import CPUKVOffload

                holder_kwargs["cpu_kv_offload"] = CPUKVOffload.Config(size_gib=args.kv_offload_gib)
                holder_kwargs["hold_gpu_blocks"] = args.session_kv_holder
            holder = VLLMSessionKVHolder.Config(**holder_kwargs)
        # tools/gb300/config_parity.py reads this line: the holder is a server flag, not in VllmConfig.
        logger.info(
            "session KV holder: %s, CPU KV offload: %s GiB",
            "on" if args.session_kv_holder else "off",
            args.kv_offload_gib,
        )
        # --capture-sizes p0: the P0 RL recipe's explicit sizes (vLLM's decode spacing plus a
        # prefill ladder to 16,384), so the server pads steps as the RL generators do. Default:
        # the tree's auto sizes (powers of 2 on the P0 tree, the dense ladder on the climb tree).
        capture_sizes = None
        if args.capture_sizes == "p0":
            capture_sizes = (
                [1, 2, 4]
                + list(range(8, 257, 8))
                + list(range(272, 1025, 16))
                + list(range(1536, 4097, 512))
                + list(range(6144, 16385, 2048))
            )
        engine = VLLMEngine.Config(
            cuda_graph=VLLMCudaGraphConfig(
                mode=args.vllm_cuda_graph_mode, capture_sizes=capture_sizes
            ),
            extra_engine_args=json.loads(args.extra_engine_args),
            session_kv_holder=holder,
        )
    else:
        from torchtitan.rl.generator.sglang.engine import SGLangEngine

        # CUDA graphs are off by default in SGLangEngine.Config; on, as the vLLM
        # engine's graph modes are.
        engine = SGLangEngine.Config(
            enable_cuda_graph=True, hold_session_kv=args.hold_session_kv
        )
    config = Generator.Config(
        model=model_config,
        parallelism=parallelism,
        engine=engine,
        model_dtype="bfloat16",
        gpu_memory_limit=args.gpu_memory_limit,
        max_num_batched_tokens=args.max_num_batched_tokens,
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True, initial_load_path=args.model_path
        ),
        sampling=SamplingConfig(stop_token_ids=[]),
        # --batch-invariant: TitanRL's bitwise mode (the trainer side: tools/gb300/x2_score.py --mode bi).
        debug=DebugConfig(seed=0, batch_invariant=args.batch_invariant, deterministic=args.batch_invariant),
        reset_kv_cache_on_model_state_dict_load=args.reset_kv_on_sync or args.batch_invariant,
        max_engine_steps_between_decisions=args.max_engine_steps,
    )
    if args.dist_moe:
        # Dist-MoE routed experts, as the trainer runs them, built the way the RL recipe builds
        # an EP > 1 generator: scratch for EP x the balanced rows (vLLM's profiling batch can
        # route every row to one EP rank). Needs the stock RoutedExperts config (no --override).
        from torchtitan.config.transform import apply_transforms
        from torchtitan.config.transform.dist_moe import DistMoeTransform
        from torchtitan.models.common.dist_moe import DistMoeRuntime

        config.dist_moe_runtime = DistMoeRuntime.Config(
            inference=True, scratch_capacity_factor=float(args.ep)
        )
        config = apply_transforms(config, [DistMoeTransform()])
        import dist_moe

        # Which dist_moe copy runs (DIST_MOE_DIR picks it in gb300/run_node.sh).
        logger.info("dist_moe from %s", os.path.dirname(dist_moe.__file__))
    return config.build(
        model_path=args.model_path,
        max_num_seqs=args.max_num_seqs,
        output_dir=args.output_dir,
        rank=int(os.environ.get("RANK", "0")),
        generator_name="titan_server",
    )


async def _serve(args) -> None:
    generator = _build_generator(args)
    if args.weight_sync_buffers:
        _fill_prefetch_buffers(generator)
    await generator.start_engine_loop()
    if int(os.environ.get("RANK", "0")) != 0:
        await generator.wait_engine_loop()
        return
    server = BenchGenerationServer(
        GenerationServer.Config(
            host=args.host, port=args.port, max_rollout_tokens=args.max_model_len
        ),
        generator,
    )
    server.set_generate_fn(generator.generate)
    server.chat_render_url = args.chat_render_url
    server.release_on_final_turn = args.release_on_final_turn
    await server.start()
    logger.info("titan_server ready on http://%s:%d", args.host, server.port)
    # Which FA4 the attention path loaded (upstream overlay or the env's wheel, gb300/titan_env.sh).
    fa4 = sys.modules.get("flash_attn.cute.interface")
    logger.info("FA4 module: %s", getattr(fa4, "__file__", "not loaded"))
    print(f"TITAN_SERVER_READY http://{args.host}:{server.port}", flush=True)
    await generator.wait_engine_loop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-module", default="torchtitan.models.qwen3")
    p.add_argument("--flavor", default="8B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--tp", type=int, default=1)
    # Same limits as the native servers (sut/vllm_native.sh, sut/sglang_native.sh).
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-limit", type=float, default=0.85)
    # RL defaults; the flags ablate them.
    p.add_argument("--capture-sizes", choices=["auto", "p0"], default="auto",
                   help="CUDA graph capture sizes: the tree's default, or the P0 RL recipe's")
    p.add_argument("--vllm-cuda-graph-mode", default="FULL_DECODE_ONLY",
                   help="the 9B Terminal-Bench recipe's pin; main's default is FULL")
    p.add_argument("--max-engine-steps", type=int, default=16,
                   help="Generator.Config.max_engine_steps_between_decisions")
    p.add_argument("--bf16-lm-head", action="store_true", help="drop the RL fp32 lm_head")
    p.add_argument("--ep", type=int, default=1, help="1, or tp (full EP over the TP ranks)")
    p.add_argument("--dist-moe", action="store_true",
                   help="Dist-MoE routed experts (DistMoeTransform + DistMoeRuntime(inference=True)), "
                        "as the trainer runs them; for EP > 1 (EP=1 keeps the local dispatch)")
    p.add_argument("--hold-session-kv", action="store_true", help="SGLang engine")
    p.add_argument("--session-kv-holder", action="store_true",
                   help="vLLM engine: hold each session's prefix on GPU between turns (up#5220)")
    p.add_argument("--kv-offload-gib", type=float, default=0.0,
                   help="vLLM engine: CPU KV offload per generator (up#5299); launch with PYTHONHASHSEED=0 "
                        "and PYTORCH_ALLOC_CONF=expandable_segments:False")
    p.add_argument("--extra-engine-args", default="{}", help='JSON, e.g. {"watermark": 0.03}')
    p.add_argument("--reset-kv-on-sync", action="store_true")
    p.add_argument("--batch-invariant", action="store_true",
                   help="batch-invariant mode (bitwise trainer parity); no compile regions, no KV holder")
    p.add_argument("--no-compile-regions", action="store_true", help="drop the model's local compile regions")
    p.add_argument("--weight-sync-buffers", action="store_true",
                   help="fill the weight-sync buffers so /torchtitan/weight_sync keeps outputs")
    p.add_argument("--release-on-final-turn", action="store_true",
                   help="release a session after the request with X-Session-Final-Turn: 1 "
                        "(AIPerf with sut/aiperf_final_turn_shim)")
    p.add_argument("--override", action="append", default=[],
                   help="config override target (module.function) for the generator model; repeat")
    p.add_argument("--output-dir", default="/tmp/titan_server")
    p.add_argument("--chat-render-url", default=None,
                   help="vLLM's GPU-less render server (`vllm launch render`) for /v1/chat/completions; "
                        "unset renders in this process")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
