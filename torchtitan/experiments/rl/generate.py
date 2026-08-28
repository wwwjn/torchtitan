#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example inference script using TorchTitan models with vLLM LLMEngine.

This script uses the RL unified config_registry to configure both
the vLLM engine and sampling parameters.

Run: torchrun --nproc_per_node=4 \
      torchtitan/experiments/rl/generate.py --config rl_grpo_qwen3_30b_a3b_varlen
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import CompileConfig
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.examples.alphabet_sort import config_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
)
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.tools.utils import has_cuda_capability


logger = init_logger(__name__)


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TorchTitan RL/vLLM generator config standalone."
    )
    parser.add_argument(
        "--config",
        default="rl_grpo_qwen3_0_6b_varlen",
        help="RL config_registry function to instantiate.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Sort these names alphabetically and put the final answer inside "
            "<alphabetical_sorted>...</alphabetical_sorted>: Charlie, Alice, Bob."
        ),
        help="User prompt to generate from.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send --prompt directly to vLLM instead of rendering a chat prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure generation throughput with synthetic token ID prompts.",
    )
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--input-len", type=int, default=None)
    benchmark.add_argument("--warmup-runs", type=int, default=2)
    benchmark.add_argument("--num-runs", type=int, default=5)
    benchmark.add_argument("--ignore-eos", action="store_true")
    benchmark.add_argument("--profile", action="store_true")
    benchmark.add_argument(
        "--profile-dir", default="/tmp/qwen3_5_inference_hillclimb/traces"
    )
    benchmark.add_argument("--profile-tag", default=None)
    benchmark.add_argument(
        "--output-json",
        default=None,
        help="Write the final measured pass's prompt and generated token IDs.",
    )

    overrides = parser.add_argument_group("overrides")
    overrides.add_argument("--model-path", default=None)
    overrides.add_argument("--tp", type=int, default=None)
    overrides.add_argument("--max-seq-len", type=int, default=None)
    overrides.add_argument("--compile", choices=["off", "aot_eager"], default=None)
    overrides.add_argument("--cudagraph", choices=["on", "off"], default=None)
    overrides.add_argument(
        "--cudagraph-mode",
        choices=["FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL"],
        default=None,
    )
    overrides.add_argument("--max-num-batched-tokens", type=int, default=None)
    overrides.add_argument(
        "--safetensors-load-strategy",
        choices=["lazy", "prefetch", "eager"],
        default=None,
        help="Checkpoint loading strategy; prefetch is useful for Manifold mounts.",
    )
    overrides.add_argument(
        "--native",
        action="store_true",
        help="Benchmark vLLM's native Hugging Face model implementation.",
    )
    overrides.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Use the same NCCL all-reduce fallback for both benchmark paths.",
    )
    overrides.add_argument(
        "--fused-gdn-projections",
        action="store_true",
        help="Merge Qwen3.5 GDN qkvz and ba input projections.",
    )
    overrides.add_argument(
        "--fused-mlp",
        action="store_true",
        help="Use TorchTitan's fused SwiGLU gate+up projection override.",
    )
    return parser.parse_args()


def _apply_overrides(config, args: argparse.Namespace) -> None:
    if args.model_path is not None:
        config.hf_assets_path = args.model_path

    if args.tp is not None:
        config.generator.parallelism = dataclasses.replace(
            config.generator.parallelism,
            tensor_parallel_degree=args.tp,
        )

    if args.compile is not None:
        config.compile = (
            CompileConfig(enable=False)
            if args.compile == "off"
            else CompileConfig(enable=True, backend="aot_eager")
        )

    if args.cudagraph is not None or args.cudagraph_mode is not None:
        config.generator.cudagraph = dataclasses.replace(
            config.generator.cudagraph,
            enable=(
                config.generator.cudagraph.enable
                if args.cudagraph is None
                else args.cudagraph == "on"
            ),
            mode=(
                config.generator.cudagraph.mode
                if args.cudagraph_mode is None
                else args.cudagraph_mode
            ),
        )

    if args.max_num_batched_tokens is not None:
        config.generator.max_num_batched_tokens = args.max_num_batched_tokens

    if args.fused_mlp:
        fused_swiglu = "torchtitan.overrides.fused_swiglu.fused_swiglu"
        override_imports = list(config.generator.override.imports)
        if fused_swiglu not in override_imports:
            override_imports.append(fused_swiglu)
            config.generator.override = dataclasses.replace(
                config.generator.override,
                imports=override_imports,
            )


def _build_compilation_config(config, *, max_num_seqs: int, native: bool):
    generator_config = config.generator
    compilation_config = generator_config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=generator_config.max_num_batched_tokens,
        expert_sequence_parallel_size=(
            generator_config.parallelism.expert_sequence_parallel_size
        ),
        enable_sequence_parallel=(
            generator_config.parallelism.enable_sequence_parallel
        ),
    )
    if native and config.compile.enable:
        compilation_config = dataclasses.replace(
            compilation_config,
            mode=CompilationMode.VLLM_COMPILE,
            backend="eager",
        )
    return compilation_config


def _build_engine(config, args: argparse.Namespace, *, max_num_seqs: int):
    generator_config = config.generator
    model_spec = config.model_spec
    model_path = config.hf_assets_path

    if (
        generator_config.cudagraph.enable
        and generator_config.cudagraph.mode == "FULL_AND_PIECEWISE"
    ):
        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    enable_ep = generator_config.parallelism.expert_parallel_degree > 1
    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype=generator_config.model_dtype,
        tensor_parallel_size=generator_config.parallelism.tensor_parallel_degree,
        data_parallel_size=generator_config.parallelism.data_parallel_degree,
        enable_expert_parallel=enable_ep,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=generator_config.gpu_memory_limit,
        enforce_eager=not generator_config.cudagraph.enable,
        disable_log_stats=False,
        max_model_len=(
            model_spec.model.max_context_length
            if args.max_seq_len is None
            else args.max_seq_len
        ),
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=not args.benchmark,
        compilation_config=_build_compilation_config(
            config, max_num_seqs=max_num_seqs, native=args.native
        ),
    )
    if generator_config.max_num_batched_tokens is not None:
        engine_kwargs[
            "max_num_batched_tokens"
        ] = generator_config.max_num_batched_tokens
    if args.safetensors_load_strategy is not None:
        engine_kwargs["safetensors_load_strategy"] = args.safetensors_load_strategy
    if generator_config.debug.seed is not None:
        engine_kwargs["seed"] = generator_config.debug.seed
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256

    if args.native:
        return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))

    if args.fused_gdn_projections:
        from torchtitan.experiments.rl.models.gdn_projection_ablation import (
            apply_merged_gdn_projections,
        )

        apply_merged_gdn_projections()

    register_to_vllm(
        model_spec,
        parallelism=generator_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        ),
        override=generator_config.override,
    )
    set_batch_invariance(generator_config.debug.batch_invariant)
    if generator_config.debug.batch_invariant:
        from torchtitan.experiments.rl.batch_invariance import (
            patch_bmm_for_batch_invariance,
        )

        patch_bmm_for_batch_invariance()

    attention_backend = model_spec.model.first_full_attention_backend
    if not isinstance(
        attention_backend, (VarlenAttention.Config, FlexAttention.Config)
    ):
        raise ValueError("Only varlen and flex attention backends are supported.")
    engine_kwargs.update(
        config_format=TORCHTITAN_CONFIG_FORMAT,
        worker_cls=TORCHTITAN_WORKER_CLS,
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLEX_ATTENTION
                if isinstance(attention_backend, FlexAttention.Config)
                else AttentionBackendEnum.CUSTOM
            ),
        ),
    )
    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


def _resolve_sampling(config, args: argparse.Namespace):
    sampling = config.generator.sampling
    return (
        sampling.temperature if args.temperature is None else args.temperature,
        sampling.top_p if args.top_p is None else args.top_p,
        sampling.max_tokens if args.max_tokens is None else args.max_tokens,
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _make_token_id_lists(
    *, vocab_size: int, input_len: int, batch_size: int
) -> list[list[int]]:
    token_id_lists = []
    token_span = vocab_size - 1024
    for batch_idx in range(batch_size):
        base = (batch_idx * 1009) % token_span + 256
        token_id_lists.append([base + (idx % 257) for idx in range(input_len)])
    return token_id_lists


def _run_pass(engine, prompts, sampling_params) -> tuple[int, list[list[int]]]:
    for request_idx, prompt in enumerate(prompts):
        engine.add_request(str(request_idx), prompt, sampling_params)
    generated_token_ids = {}
    while engine.has_unfinished_requests():
        for request_output in engine.step():
            if request_output.finished:
                generated_token_ids[int(request_output.request_id)] = list(
                    request_output.outputs[0].token_ids
                )
    ordered_token_ids = [
        generated_token_ids[request_idx] for request_idx in range(len(prompts))
    ]
    return sum(map(len, ordered_token_ids)), ordered_token_ids


def _profile_one_pass(engine, prompts, sampling_params, args) -> None:
    from torch.profiler import profile, ProfilerActivity

    rank = int(os.environ.get("RANK", "0"))
    model_tag = "native" if args.native else "torchtitan"
    trace_name = args.profile_tag or (
        f"{model_tag}_bs{args.batch_size}_in{args.input_len}_out{args.max_tokens}"
    )
    os.makedirs(args.profile_dir, exist_ok=True)
    trace_path = os.path.join(args.profile_dir, f"{trace_name}_rank{rank}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        _run_pass(engine, prompts, sampling_params)
        _sync()
    profiler.export_chrome_trace(trace_path)
    if _is_rank0():
        print(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=30,
            ),
            flush=True,
        )
        print(f"Profiler trace written: {trace_path}", flush=True)


def benchmark(config, args: argparse.Namespace) -> None:
    if args.input_len is None:
        raise ValueError("--input-len is required with --benchmark")

    temperature, top_p, max_tokens = _resolve_sampling(config, args)
    max_num_seqs = max(args.max_num_seqs, args.batch_size)
    build_start = time.perf_counter()
    engine = _build_engine(config, args, max_num_seqs=max_num_seqs)
    _sync()
    build_time = time.perf_counter() - build_start

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        ignore_eos=args.ignore_eos,
        seed=config.generator.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    token_id_lists = _make_token_id_lists(
        vocab_size=config.model_spec.model.vocab_size,
        input_len=args.input_len,
        batch_size=args.batch_size,
    )
    prompts = engine.renderer.render_cmpl(
        [{"prompt_token_ids": token_ids} for token_ids in token_id_lists]
    )

    for _ in range(args.warmup_runs):
        _run_pass(engine, prompts, sampling_params)
    _sync()

    if args.profile:
        _profile_one_pass(engine, prompts, sampling_params, args)
        _sync()

    durations = []
    generated_counts = []
    final_generated_token_ids = []
    for _ in range(args.num_runs):
        _sync()
        start = time.perf_counter()
        num_generated_tokens, final_generated_token_ids = _run_pass(
            engine, prompts, sampling_params
        )
        generated_counts.append(num_generated_tokens)
        _sync()
        durations.append(time.perf_counter() - start)

    if not _is_rank0():
        return

    throughputs = [
        num_tokens / duration
        for num_tokens, duration in zip(generated_counts, durations)
    ]
    mean_throughput = statistics.mean(throughputs)
    throughput_std = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    model_name = "vllm-native" if args.native else "torchtitan"
    compile_name = config.compile.backend if config.compile.enable else "off"
    print("\n" + "=" * 72, flush=True)
    print(f"BENCHMARK model={model_name} config={args.config}", flush=True)
    print(
        f"  TP={config.generator.parallelism.tensor_parallel_degree} "
        f"compile={compile_name} "
        f"cudagraph={config.generator.cudagraph.mode if config.generator.cudagraph.enable else 'off'}",
        flush=True,
    )
    print(
        f"  batch={args.batch_size} input={args.input_len} output={max_tokens} "
        f"warmup={args.warmup_runs} runs={args.num_runs} "
        f"build={build_time:.1f}s (excluded)",
        flush=True,
    )
    for run_idx, (duration, num_tokens, throughput) in enumerate(
        zip(durations, generated_counts, throughputs)
    ):
        print(
            f"  run {run_idx}: {duration * 1000:8.1f} ms "
            f"{num_tokens:6d} tok {throughput:8.1f} tok/s",
            flush=True,
        )
    print(
        f"  THROUGHPUT: {mean_throughput:.1f} +/- {throughput_std:.1f} tok/s",
        flush=True,
    )
    print("=" * 72 + "\n", flush=True)

    if args.output_json is not None:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w") as output_file:
            json.dump(
                {
                    "prompt_token_ids": token_id_lists,
                    "generated_token_ids": final_generated_token_ids,
                },
                output_file,
            )
        print(f"Output token IDs written: {args.output_json}", flush=True)


def generate() -> None:
    args = _parse_args()

    config_factory = getattr(config_registry, args.config, None)
    if not callable(config_factory):
        raise ValueError(f"Unknown RL config {args.config!r}")
    config = config_factory()
    _apply_overrides(config, args)
    if args.benchmark:
        benchmark(config, args)
        return
    if args.native:
        raise ValueError("--native is only supported with --benchmark")

    if args.fused_gdn_projections:
        from torchtitan.experiments.rl.models.gdn_projection_ablation import (
            apply_merged_gdn_projections,
        )

        apply_merged_gdn_projections()

    gen_config = config.generator
    model_spec = config.model_spec
    model_path = config.hf_assets_path
    max_num_seqs = args.max_num_seqs
    is_rank0 = _is_rank0()

    # FULL_AND_PIECEWISE reads VLLM_USE_BREAKABLE_CUDAGRAPH at import time (the
    # @eager_break_during_capture decorator in rl/models/attention.py).
    if (
        gen_config.cudagraph.enable
        and gen_config.cudagraph.mode == "FULL_AND_PIECEWISE"
    ):
        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"

    # Register TorchTitan model with vLLM before engine creation
    register_to_vllm(
        model_spec,
        parallelism=gen_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        ),
        override=config.generator.override,
    )
    logger.info("Registered TorchTitan model with vLLM")

    attention_backend = model_spec.model.first_full_attention_backend
    if attention_backend is None:
        raise ValueError("No full-attention layer found in the model spec.")
    if not isinstance(
        attention_backend, (VarlenAttention.Config, FlexAttention.Config)
    ):
        raise ValueError("Only varlen and flex attention backends are supported.")

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    set_batch_invariance(gen_config.debug.batch_invariant)
    if gen_config.debug.batch_invariant:
        # batch_invariant_ops doesn't cover bmm; the MoE router gate is a bmm in
        # the vLLM inference graph, so override it generator-side (not in core).
        from torchtitan.experiments.rl.batch_invariance import (
            patch_bmm_for_batch_invariance,
        )

        patch_bmm_for_batch_invariance()
    enable_ep = gen_config.parallelism.expert_parallel_degree > 1

    logger.debug("Initializing vLLM LLMEngine with TorchTitan model")
    logger.debug(f"Model: {model_path}")
    logger.debug(
        f"Tensor Parallel Size: {gen_config.parallelism.tensor_parallel_degree}"
    )
    logger.debug(f"Expert Parallel Enabled: {enable_ep}")

    # Create EngineArgs from config
    engine_kwargs = dict(
        # Model configuration
        model=model_path,
        trust_remote_code=True,
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        # Parallelism configuration
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        data_parallel_size=gen_config.parallelism.data_parallel_degree,
        enable_expert_parallel=enable_ep,
        worker_cls=TORCHTITAN_WORKER_CLS,
        # Use external_launcher only when launched via torchrun (multi-GPU);
        # for single-GPU, let vLLM pick the default executor.
        distributed_executor_backend=("external_launcher"),
        # Memory and performance
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=not gen_config.cudagraph.enable,
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLEX_ATTENTION
                if isinstance(attention_backend, FlexAttention.Config)
                else AttentionBackendEnum.CUSTOM
            ),
        ),
        disable_log_stats=False,
    )
    engine_kwargs["max_model_len"] = model_spec.model.max_context_length
    engine_kwargs["max_num_seqs"] = max_num_seqs
    if gen_config.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = gen_config.max_num_batched_tokens
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    expert_sequence_parallel_size = gen_config.parallelism.expert_sequence_parallel_size
    vllm_compilation_config = gen_config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=gen_config.max_num_batched_tokens,
        expert_sequence_parallel_size=expert_sequence_parallel_size,
        enable_sequence_parallel=gen_config.parallelism.enable_sequence_parallel,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed
    engine_args = EngineArgs(**engine_kwargs)

    logger.debug("Initializing LLMEngine from EngineArgs...")
    engine = LLMEngine.from_engine_args(engine_args)

    logger.debug("vLLM LLMEngine initialized successfully")

    renderer = config.renderer.build(tokenizer_path=model_path)
    stop_token_ids = list(renderer.get_stop_token_ids())

    # Create sampling parameters from config
    sampling = gen_config.sampling
    temperature = sampling.temperature if args.temperature is None else args.temperature
    top_p = sampling.top_p if args.top_p is None else args.top_p
    max_tokens = sampling.max_tokens if args.max_tokens is None else args.max_tokens
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stop_token_ids=stop_token_ids or None,
        seed=gen_config.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    logger.debug(
        f"Sampling params: temperature={temperature}, "
        f"top_p={top_p}, max_tokens={max_tokens}"
    )

    prompt = args.prompt
    logger.debug(f"Prompt: {prompt}")

    # Add request to engine
    logger.debug("Adding request to engine...")
    request_id = "0"
    if args.raw_prompt:
        engine_input = prompt
    else:
        prompt_token_ids = renderer.render_ids(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            add_generation_prompt=True,
        )
        engine_input = engine.renderer.render_cmpl(
            [{"prompt_token_ids": prompt_token_ids}]
        )[0]
        if is_rank0:
            print(f"Prompt token count: {len(prompt_token_ids)}", flush=True)
            print(f"Stop token ids: {stop_token_ids}", flush=True)
    engine.add_request(request_id, engine_input, sampling_params)

    # Generate text by stepping through engine
    logger.debug("Generating text...")
    while engine.has_unfinished_requests():
        request_outputs = engine.step()

        # Process finished requests
        for request_output in request_outputs:
            if request_output.finished:
                generated_text = request_output.outputs[0].text
                output_token_ids = request_output.outputs[0].token_ids

                # Print results
                logger.debug("Generation complete")
                if is_rank0:
                    print(f"\nConfig: {args.config}", flush=True)
                    print(f"Prompt: {prompt}", flush=True)
                    print(f"Generated token count: {len(output_token_ids)}", flush=True)
                    print(f"Generated text: {generated_text!r}\n", flush=True)


if __name__ == "__main__":
    generate()
