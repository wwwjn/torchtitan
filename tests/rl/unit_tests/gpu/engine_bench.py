# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time generation through a ``GenerationEngine``, the same way for every engine.

Both engines run the same TorchTitan model config (local compile regions included),
in perf mode (not batch-invariant) by default, with the same batch limits and
requests; only ``--engine`` differs between two launches:

    torchrun --nproc_per_node=1 -m tests.rl.unit_tests.gpu.engine_bench \\
        --engine vllm --model qwen3_0_6b --workloads decode:64x128x512 --out /tmp/v.json

A workload ``kind:NxPxO`` admits N requests of P random prompt tokens and O output
tokens (no stop tokens, so every request generates exactly O). ``grpo:GxSxPxO``
admits G prompts with S samples each, sharing the prompt (prefix-cache hits).
Each workload runs ``--repeats`` times with fresh prompts; every repeat reports its
wall time, its step times and the median step time of its decode phase, which is
the inter-token latency at N running requests.
"""

from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()

import argparse
import importlib
import json
import os
import random
import statistics
import time

import torch
import torch.distributed as dist

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig, ParallelismConfig
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

from tests.rl.unit_tests.gpu.engine_parity import MODELS

# name -> (model builder module, flavor, default HF checkpoint); pass --model-path
# for the ones without a default.
MODELS = {
    **MODELS,
    "qwen3_4b": ("torchtitan.models.qwen3", "4B", "./assets/hf/Qwen3-4B"),
    "qwen3_14b": ("torchtitan.models.qwen3", "14B", "./assets/hf/Qwen3-14B"),
}


def _build_engine(args):
    module_name, flavor, default_model_path = MODELS[args.model]
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=args.tp,
        data_parallel_degree=args.dp,
        expert_parallel_degree=args.ep,
    )
    _, model_config = build_rl_model_configs(
        importlib.import_module(module_name).build_model_config,
        flavor,
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=parallelism,
        seq_len=args.seq_len,
        attn_backend="varlen",
    )
    if args.batch_invariant or args.no_compile:
        model_config.local_compile_regions = []
    dist_moe_runtime = None
    if args.dist_moe:
        from torchtitan.config.transform import transform_model_config_
        from torchtitan.config.transform.dist_moe import DistMoeTransform
        from torchtitan.models.common.dist_moe import DistMoeRuntime

        model_config = transform_model_config_(model_config, [DistMoeTransform()])
        dist_moe_runtime = DistMoeRuntime.Config(
            inference=True,
            scratch_capacity_factor=args.dist_moe_scratch_capacity_factor,
        )
    debug = DebugConfig(batch_invariant=args.batch_invariant, seed=0)
    set_batch_invariance(debug.batch_invariant)
    Generator._set_determinism(debug)

    if args.engine == "vllm":
        from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
        from torchtitan.rl.generator.vllm.engine import VLLMEngine

        engine_config = VLLMEngine.Config(
            cuda_graph=VLLMCudaGraphConfig(mode=args.vllm_cuda_graph_mode),
            extra_engine_args=json.loads(args.vllm_extra_engine_args),
        )
    else:
        from torchtitan.rl.generator.sglang.engine import SGLangEngine

        engine_config = SGLangEngine.Config(
            enable_cuda_graph=args.vllm_cuda_graph_mode != "NONE",
            overlap_schedule=not args.sglang_no_overlap,
            extra_server_args=json.loads(args.sglang_extra_server_args),
        )
    model_path = args.model_path or os.path.abspath(default_model_path)
    engine = engine_config.build(
        model=model_config,
        parallelism=parallelism,
        model_path=model_path,
        max_num_seqs=args.max_num_seqs,
        model_dtype="bfloat16",
        gpu_memory_limit=args.gpu_memory_limit,
        max_num_batched_tokens=args.max_num_batched_tokens,
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True, initial_load_path=model_path
        ),
        dist_moe_runtime=dist_moe_runtime,
        debug=debug,
        reset_kv_cache_on_model_state_dict_load=False,
        rank=int(os.environ.get("RANK", "0")),
        generator_name="engine_bench",
        output_dir="/tmp/engine_bench",
    )
    return engine, model_config.vocab_size


def _parse_workload(spec: str) -> tuple[str, int, int, int, int]:
    """``kind:NxPxO`` or ``grpo:GxSxPxO`` -> (kind, prompts, samples, P, O)."""
    kind, _, shape = spec.partition(":")
    dims = [int(d) for d in shape.split("x")]
    if kind == "grpo":
        num_prompts, num_samples, prompt_len, output_len = dims
    elif kind == "stream":
        # stream:NxPxOxK: N requests, one admitted every K steps.
        (num_prompts, prompt_len, output_len, _), num_samples = dims, 1
    else:
        (num_prompts, prompt_len, output_len), num_samples = dims, 1
    return kind, num_prompts, num_samples, prompt_len, output_len


def _requests(
    spec: str, repeat: int, vocab_size: int, temperature: float
) -> list[EngineRequest]:
    kind, num_prompts, num_samples, prompt_len, output_len = _parse_workload(spec)
    rng = random.Random(f"{spec}/{repeat}")
    requests = []
    for i in range(num_prompts):
        # Token ids from the lower vocabulary, away from special tokens.
        prompt = [
            rng.randrange(1000, min(vocab_size, 100_000)) for _ in range(prompt_len)
        ]
        for j in range(num_samples):
            request = EngineRequest(
                request_id=f"{spec}/{repeat}/{i}/{j}",
                prompt_token_ids=prompt,
                sampling=SamplingConfig(
                    temperature=temperature,
                    max_tokens=output_len,
                    stop_token_ids=[],
                ),
                group_id=i,
                routing_session_id=f"{spec}/{repeat}/{i}/{j}",
                arrival_time=float(len(requests)),
            )
            request.min_policy_version = 0
            requests.append(request)
    return requests


_STEP_TRACE: list = []
# BENCH_TRACE_GC=1: (generation, ms) of every collection, to spot GC stalls.
_GC_PAUSES: list = []


def _track_gc() -> None:
    import gc

    start = {}

    def callback(phase, info):
        if phase == "start":
            start["t"] = time.perf_counter()
        else:
            _GC_PAUSES.append(
                (info["generation"], (time.perf_counter() - start["t"]) * 1e3)
            )

    gc.callbacks.append(callback)


def _run(engine, requests: list[EngineRequest], admit_every: int = 0) -> dict:
    """Admit ``requests`` (all at once, or one every ``admit_every`` steps) and step
    until all finish; per-request latency is admission to the step that returns it."""
    dp_rank, dp_size = engine.data_parallel_rank, engine.data_parallel_size
    requests = requests[dp_rank::dp_size]
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    pending = list(requests) if admit_every else []
    admitted_at = {}
    if not admit_every:
        assert not engine.add_requests(requests)
        admitted_at = dict.fromkeys((r.request_id for r in requests), start)
    step_ms, latencies_s = [], []
    output_tokens = 0
    while pending or engine.has_unfinished_requests():
        if pending and len(step_ms) % admit_every == 0:
            request = pending.pop(0)
            admitted_at[request.request_id] = time.perf_counter()
            assert not engine.add_requests([request])
        t0 = time.perf_counter()
        finished = engine.step()
        now = time.perf_counter()
        step_ms.append((now - t0) * 1e3)
        scheduler = getattr(engine, "_scheduler", None)
        if os.environ.get("BENCH_TRACE_STEPS") and scheduler is not None:
            batch = scheduler.last_batch
            _STEP_TRACE.append(
                None
                if batch is None
                else (
                    str(batch.forward_mode).split(".")[-1],
                    batch.batch_size(),
                    len(scheduler.running_batch.reqs),
                    len(scheduler.waiting_queue),
                    round(step_ms[-1], 2),
                )
            )
        output_tokens += sum(len(f.token_ids) for f in finished)
        latencies_s += [now - admitted_at[f.request_id] for f in finished]
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    return dict(
        wall_s=wall,
        num_steps=len(step_ms),
        step_ms=step_ms,
        num_finished=len(latencies_s),
        output_tokens=output_tokens,
        latencies_s=latencies_s,
    )


def _summary(spec: str, result: dict) -> dict:
    _, num_prompts, num_samples, prompt_len, output_len = _parse_workload(spec)
    step_ms = result["step_ms"]
    # The last output_len - 1 steps of a wave whose requests all start together
    # are its decode phase; skip the first decode steps, which overlap prefill.
    decode = step_ms[-(output_len - 1) :][output_len // 8 :] if output_len > 8 else []
    latencies = sorted(result["latencies_s"])
    return dict(
        workload=spec,
        latency_s_p50=round(latencies[len(latencies) // 2], 3) if latencies else None,
        latency_s_p90=round(latencies[int(len(latencies) * 0.9)], 3)
        if latencies
        else None,
        wall_s=round(result["wall_s"], 3),
        num_steps=result["num_steps"],
        output_tokens=result["output_tokens"],
        output_tok_per_s=round(result["output_tokens"] / result["wall_s"], 1),
        decode_step_ms_p50=round(statistics.median(decode), 3) if decode else None,
        first_steps_ms=[round(x, 1) for x in step_ms[:8]],
        step_ms_max=round(max(step_ms), 1) if step_ms else None,
    )


def _time_weight_loads(engine, args, rank: int) -> None:
    """Time ``load_prefetched_model_state_dict`` (the engine pause of a weight sync)."""
    from torch.profiler import profile, ProfilerActivity

    times_ms = []
    for i in range(args.weight_loads):
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if i == args.weight_loads - 1 and args.profile_stack:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
            ) as prof:
                engine.load_prefetched_model_state_dict()
            if rank == 0:
                prof.export_chrome_trace(args.out + ".load_trace.json")
        else:
            engine.load_prefetched_model_state_dict()
        torch.cuda.synchronize()
        times_ms.append(round((time.perf_counter() - start) * 1e3, 1))
    # The engine loop runs a full collection after every weight load.
    import gc

    gc_start = time.perf_counter()
    gc.collect()
    gc_ms = round((time.perf_counter() - gc_start) * 1e3, 1)
    if rank == 0:
        print(
            "WEIGHT_LOAD",
            args.engine,
            json.dumps(
                dict(ms=times_ms, gc_collect_ms=gc_ms, gc_tracked=len(gc.get_objects()))
            ),
            flush=True,
        )


def _profile(engine, args, vocab_size: int, rank: int) -> None:
    """Profile one wave of ``--profile``: GPU kernel time per step vs wall per step.

    The profiler slows later CUDA graph replays, so it runs after the timed waves.
    """
    from torch.autograd import DeviceType
    from torch.profiler import profile, ProfilerActivity

    requests = _requests(args.profile, 1000, vocab_size, args.temperature)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=args.profile_stack,
    ) as prof:
        result = _run(engine, requests)
    if rank != 0:
        return
    prof.export_chrome_trace(args.out + ".trace.json")
    events = prof.key_averages()
    kernels = [e for e in events if e.device_type == DeviceType.CUDA]
    kernel_ms = sum(e.self_device_time_total for e in kernels) / 1e3
    num_steps = result["num_steps"]
    print(
        "PROFILE",
        args.engine,
        json.dumps(
            dict(
                workload=args.profile,
                num_steps=num_steps,
                wall_ms_per_step=round(result["wall_s"] * 1e3 / num_steps, 3),
                kernel_ms_per_step=round(kernel_ms / num_steps, 3),
                kernels_per_step=round(sum(e.count for e in kernels) / num_steps, 1),
            )
        ),
        flush=True,
    )
    top = sorted(kernels, key=lambda e: -e.self_device_time_total)[:25]
    for e in top:
        print(
            f"KERNEL {args.engine} {e.self_device_time_total / 1e3 / num_steps:8.3f} ms/step "
            f"{e.count / num_steps:6.1f}/step {e.key[:110]}"
        )
    cpu_top = sorted(events, key=lambda e: -e.self_cpu_time_total)[:25]
    for e in cpu_top:
        print(
            f"CPU {args.engine} {e.self_cpu_time_total / 1e3 / num_steps:8.3f} ms/step "
            f"{e.count / num_steps:6.1f}/step {e.key[:110]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="qwen3_0_6b", choices=sorted(MODELS))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-limit", type=float, default=0.8)
    parser.add_argument(
        "--vllm-cuda-graph-mode",
        choices=["NONE", "FULL_DECODE_ONLY", "FULL"],
        default="FULL",
        help="SGLang: anything but NONE captures FULL decode and prefill graphs",
    )
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dist-moe", action="store_true")
    parser.add_argument("--dist-moe-scratch-capacity-factor", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--workloads",
        default="decode:64x128x512,prefill:32x2048x1,grpo:16x8x1024x256",
    )
    parser.add_argument("--warmup", default="decode:16x128x32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", default=None, help="workload to profile")
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--weight-loads", type=int, default=0)
    parser.add_argument("--sglang-no-overlap", action="store_true")
    parser.add_argument("--sglang-extra-server-args", default="{}")
    parser.add_argument("--vllm-extra-engine-args", default="{}")
    args = parser.parse_args()

    init_start = time.perf_counter()
    engine, vocab_size = _build_engine(args)
    if os.environ.get("BENCH_TRACE_GC"):
        _track_gc()
    init_s = time.perf_counter() - init_start
    rank = int(os.environ.get("RANK", "0"))
    _run(engine, _requests(args.warmup, -1, vocab_size, args.temperature))
    runs = []
    for spec in args.workloads.split(","):
        for repeat in range(args.repeats):
            admit_every = (
                int(spec.rpartition("x")[2]) if spec.startswith("stream:") else 0
            )
            result = _run(
                engine,
                _requests(spec, repeat, vocab_size, args.temperature),
                admit_every,
            )
            summary = _summary(spec, result)
            runs.append(dict(summary, repeat=repeat, step_ms=result["step_ms"]))
            if rank == 0:
                print("BENCH", args.engine, json.dumps(summary), flush=True)
                if _GC_PAUSES:
                    slow = sorted(
                        (round(ms, 1), gen) for gen, ms in _GC_PAUSES if ms > 5
                    )
                    print(
                        "GC",
                        args.engine,
                        spec,
                        len(_GC_PAUSES),
                        "collections; >5 ms:",
                        slow[-10:],
                        flush=True,
                    )
                    _GC_PAUSES.clear()
    if args.weight_loads:
        _time_weight_loads(engine, args, rank)
    if args.profile:
        _profile(engine, args, vocab_size, rank)
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(
                dict(
                    engine=args.engine,
                    args=vars(args),
                    init_s=init_s,
                    runs=runs,
                    step_trace=_STEP_TRACE,
                ),
                f,
            )
    engine.shutdown()
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
