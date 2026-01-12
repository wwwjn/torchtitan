#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM Timing Analysis Script

This script compares timing for vLLM native model vs vLLM TorchTitan model.
It instruments key functions to measure time consumption for:
- Prefill phase
- Decode phase
- forward() function
- compute_logits() function
- _model_forward() function
- _prepare_inputs() function
- _preprocess() function
- _build_attention_metadata() function
- _sample() function

Usage:
    # Run with environment variable timing enabled (recommended)
    VLLM_TIMING=1 TORCHTITAN_VLLM_TIMING=1 python scripts/vllm_timing_analysis.py \
        --model-path Qwen/Qwen3-1.7B \
        --checkpoint /path/to/hf/checkpoint \
        --tp 1 \
        --num-prompts 5 \
        --max-tokens 128

    # Analyze existing log file
    python scripts/vllm_timing_analysis.py \
        --analyze-log /path/to/vllm_output.log

Environment Variables:
    VLLM_TIMING=1              Enable timing in gpu_model_runner.py
    TORCHTITAN_VLLM_TIMING=1   Enable timing in vllm_wrapper.py
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torchtitan.experiments.rl import unified  # noqa: F401


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
script_logger = logging.getLogger(__name__)


@dataclass
class TimingEntry:
    """Single timing measurement."""

    name: str
    elapsed_ms: float
    phase: str = ""  # prefill or decode


@dataclass
class TimingAggregator:
    """Aggregates timing measurements."""

    def __init__(self):
        self.entries: dict[str, list[float]] = defaultdict(list)
        self.by_phase: dict[str, dict[str, list[float]]] = {
            "prefill": defaultdict(list),
            "decode": defaultdict(list),
        }

    def add(self, name: str, elapsed_ms: float, phase: str = ""):
        self.entries[name].append(elapsed_ms)
        if phase in self.by_phase:
            self.by_phase[phase][name].append(elapsed_ms)

    def get_summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, times in self.entries.items():
            if times:
                result[name] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }
        return result

    def get_phase_summary(self, phase: str) -> dict[str, dict[str, float]]:
        result = {}
        if phase not in self.by_phase:
            return result
        for name, times in self.by_phase[phase].items():
            if times:
                result[name] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }
        return result


class LogParser:
    """Parse timing logs from vLLM and TorchTitan.

    This parser detects benchmark boundaries and associates timing logs
    with the correct benchmark run (native vs torchtitan).
    """

    # Pattern: [VLLM_TIMING] phase.operation: xxx.xxx ms
    # or: [VLLM_TIMING] Phase: prefill, num_reqs: 1, ...
    VLLM_TIMING_RE = re.compile(
        r"\[VLLM_TIMING\]\s+"
        r"(?:Phase:\s+(\w+),\s+.*?|"  # Phase info line
        r"((?:\w+\.)*\w+):\s+([\d.]+)\s+ms)"  # Timing line
    )

    TORCHTITAN_TIMING_RE = re.compile(
        r"\[TORCHTITAN_TIMING\]\s+"
        r"(?:Phase:\s+(\w+),\s+.*?|"
        r"((?:\w+\.)*\w+):\s+([\d.]+)\s+ms)"
    )

    # Benchmark boundary markers
    WARMUP_RE = re.compile(r"Warmup run\.\.\.")

    def __init__(self, debug=False, default_benchmark=None):
        # Separate aggregators for each benchmark
        self.native_aggregator = TimingAggregator()
        self.torchtitan_aggregator = TimingAggregator()

        # Current state
        # default_benchmark: if set, use this when no benchmark marker is detected
        self._current_benchmark = default_benchmark  # "native" or "torchtitan"
        self._current_phase = ""
        self._in_warmup = False
        self._debug = debug
        self._seen_benchmarks = set()  # Track which benchmarks we've seen

        # Also keep the old aggregators for backward compatibility
        self.vllm_aggregator = self.native_aggregator

        # For internal TorchTitan wrapper timing (if available)
        self.torchtitan_internal_aggregator = TimingAggregator()

    def parse_line(self, line: str):
        """Parse a single log line."""
        # Detect benchmark boundaries - check TorchTitan first since it's more specific
        if "Running vLLM TorchTitan Benchmark" in line:
            self._current_benchmark = "torchtitan"
            self._seen_benchmarks.add("torchtitan")
            self._in_warmup = False
            if self._debug:
                print("[DEBUG] Starting TorchTitan benchmark")
            return

        if "Running vLLM Native Benchmark" in line:
            self._current_benchmark = "native"
            self._seen_benchmarks.add("native")
            self._in_warmup = False
            if self._debug:
                print("[DEBUG] Starting Native benchmark")
            return

        if self.WARMUP_RE.search(line):
            self._in_warmup = True
            if self._debug:
                print(f"[DEBUG] Entering warmup, benchmark={self._current_benchmark}")
            return

        if "Timed run" in line:
            self._in_warmup = False
            if self._debug:
                print(f"[DEBUG] Exiting warmup, benchmark={self._current_benchmark}")
            return

        # Skip warmup logs
        if self._in_warmup:
            return

        # Parse VLLM_TIMING - associate with current benchmark
        if "[VLLM_TIMING]" in line:
            match = self.VLLM_TIMING_RE.search(line)
            if match:
                phase_info = match.group(1)
                name = match.group(2)
                value_str = match.group(3)

                if phase_info:
                    self._current_phase = phase_info
                elif name and value_str:
                    try:
                        elapsed_ms = float(value_str)
                        # Extract phase from name if present (e.g., "prefill._model_forward")
                        if name.startswith(("prefill.", "decode.")):
                            parts = name.split(".", 1)
                            phase = parts[0]
                            op_name = parts[1] if len(parts) > 1 else name
                        else:
                            phase = self._current_phase
                            op_name = name

                        # Add to the correct aggregator based on current benchmark
                        if self._current_benchmark == "native":
                            self.native_aggregator.add(op_name, elapsed_ms, phase)
                            if self._debug:
                                print(
                                    f"[DEBUG] Added to NATIVE: {op_name} = {elapsed_ms:.3f}ms (phase={phase})"
                                )
                        elif self._current_benchmark == "torchtitan":
                            self.torchtitan_aggregator.add(op_name, elapsed_ms, phase)
                            if self._debug:
                                print(
                                    f"[DEBUG] Added to TORCHTITAN: {op_name} = {elapsed_ms:.3f}ms (phase={phase})"
                                )
                        else:
                            # No benchmark set - skip with debug message
                            if self._debug:
                                print(f"[DEBUG] Skipping {op_name} (no benchmark set)")
                    except ValueError:
                        pass

        # Parse TORCHTITAN_TIMING - internal wrapper timing (always goes to torchtitan)
        if "[TORCHTITAN_TIMING]" in line:
            match = self.TORCHTITAN_TIMING_RE.search(line)
            if match:
                phase_info = match.group(1)
                name = match.group(2)
                value_str = match.group(3)

                if phase_info:
                    pass  # Just phase info
                elif name and value_str:
                    try:
                        elapsed_ms = float(value_str)
                        if name.startswith(("prefill.", "decode.")):
                            parts = name.split(".", 1)
                            phase = parts[0]
                            op_name = parts[1] if len(parts) > 1 else name
                        else:
                            phase = self._current_phase
                            op_name = name
                        self.torchtitan_internal_aggregator.add(
                            op_name, elapsed_ms, phase
                        )
                        if self._debug:
                            print(
                                f"[DEBUG] Added to TORCHTITAN_INTERNAL: {op_name} = {elapsed_ms:.3f}ms"
                            )
                    except ValueError:
                        pass

    def parse_file(self, filepath: str):
        """Parse an entire log file."""
        with open(filepath, "r") as f:
            for line in f:
                self.parse_line(line)

        # Print what benchmarks were detected
        if self._debug or not self._seen_benchmarks:
            print(
                f"[DEBUG] Detected benchmarks: {self._seen_benchmarks if self._seen_benchmarks else 'None'}"
            )

    def print_summary(self):
        """Print formatted summary of timing data."""

        def print_table(title: str, summary: dict):
            if not summary:
                print(f"\n{title}")
                print("  (No data)")
                return
            print(f"\n{title}")
            print("-" * 100)
            print(
                f"{'Operation':<45} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
            )
            print("-" * 100)
            for name, data in sorted(summary.items(), key=lambda x: -x[1]["total_ms"]):
                print(
                    f"{name:<45} {data['count']:>8} {data['total_ms']:>12.2f} "
                    f"{data['avg_ms']:>10.3f} {data['min_ms']:>10.3f} {data['max_ms']:>10.3f}"
                )

        print("\n" + "=" * 100)
        print("TIMING ANALYSIS SUMMARY")
        print("=" * 100)

        # Native benchmark timing
        print_table(
            "vLLM NATIVE Benchmark - All Operations:",
            self.native_aggregator.get_summary(),
        )
        print_table(
            "vLLM NATIVE - Prefill Phase:",
            self.native_aggregator.get_phase_summary("prefill"),
        )
        print_table(
            "vLLM NATIVE - Decode Phase:",
            self.native_aggregator.get_phase_summary("decode"),
        )

        # TorchTitan benchmark timing (from gpu_model_runner.py)
        print_table(
            "vLLM + TORCHTITAN Benchmark - All Operations:",
            self.torchtitan_aggregator.get_summary(),
        )
        print_table(
            "vLLM + TORCHTITAN - Prefill Phase:",
            self.torchtitan_aggregator.get_phase_summary("prefill"),
        )
        print_table(
            "vLLM + TORCHTITAN - Decode Phase:",
            self.torchtitan_aggregator.get_phase_summary("decode"),
        )

        # TorchTitan internal timing (from vllm_wrapper.py, if available)
        internal_summary = self.torchtitan_internal_aggregator.get_summary()
        if internal_summary:
            print_table(
                "TorchTitan Model Internal Timing (vllm_wrapper.py):", internal_summary
            )

        # Print comparison table
        self._print_comparison()

        print("\n" + "=" * 100)

    def _print_comparison(self):
        """Print side-by-side comparison of native vs torchtitan."""
        native_summary = self.native_aggregator.get_summary()
        tt_summary = self.torchtitan_aggregator.get_summary()

        if not native_summary or not tt_summary:
            return

        print("\n" + "=" * 100)
        print("COMPARISON: vLLM Native vs vLLM + TorchTitan")
        print("=" * 100)
        print(
            f"{'Operation':<35} {'Native Avg(ms)':>15} {'TorchTitan Avg(ms)':>18} {'Diff(ms)':>12} {'Ratio':>10}"
        )
        print("-" * 90)

        # Key operations to compare
        key_ops = [
            "execute_model.total",
            "_model_forward",
            "compute_logits",
            "_prepare_inputs",
            "_preprocess",
            "_build_attention_metadata",
            "postprocess",
            "_sample",
        ]

        for op in key_ops:
            native_data = native_summary.get(op)
            tt_data = tt_summary.get(op)

            if native_data and tt_data:
                native_avg = native_data["avg_ms"]
                tt_avg = tt_data["avg_ms"]
                diff = tt_avg - native_avg
                ratio = tt_avg / native_avg if native_avg > 0 else 0
                print(
                    f"{op:<35} {native_avg:>15.3f} {tt_avg:>18.3f} {diff:>+12.3f} {ratio:>9.2f}x"
                )
            elif native_data:
                print(
                    f"{op:<35} {native_data['avg_ms']:>15.3f} {'N/A':>18} {'N/A':>12} {'N/A':>10}"
                )
            elif tt_data:
                print(
                    f"{op:<35} {'N/A':>15} {tt_data['avg_ms']:>18.3f} {'N/A':>12} {'N/A':>10}"
                )

        # Also compare by phase
        for phase in ["prefill", "decode"]:
            native_phase = self.native_aggregator.get_phase_summary(phase)
            tt_phase = self.torchtitan_aggregator.get_phase_summary(phase)

            if native_phase.get("execute_model.total") and tt_phase.get(
                "execute_model.total"
            ):
                native_avg = native_phase["execute_model.total"]["avg_ms"]
                tt_avg = tt_phase["execute_model.total"]["avg_ms"]
                diff = tt_avg - native_avg
                ratio = tt_avg / native_avg if native_avg > 0 else 0
                print(
                    f"{phase}.execute_model.total{'':<14} {native_avg:>15.3f} {tt_avg:>18.3f} {diff:>+12.3f} {ratio:>9.2f}x"
                )


@dataclass
class TimingStats:
    """Timing statistics for a single run."""

    approach: str  # "vllm_native" or "vllm_torchtitan"

    # Overall timing
    total_time_ms: float = 0.0

    # Per-phase timing (averaged per token)
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0

    # Per-function timing (averaged per call)
    forward_time_ms: float = 0.0
    compute_logits_time_ms: float = 0.0
    model_forward_time_ms: float = 0.0
    prepare_inputs_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    build_attn_time_ms: float = 0.0
    sample_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0

    # Counts
    num_forward_calls: int = 0
    num_compute_logits_calls: int = 0
    num_tokens_generated: int = 0
    num_prefill_tokens: int = 0

    # Detailed timing lists (per call)
    forward_times: list = field(default_factory=list)
    compute_logits_times: list = field(default_factory=list)
    model_forward_times: list = field(default_factory=list)


class TimingContext:
    """Global context to collect timing data."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.compute_logits_times = []
        self.model_forward_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        self.prepare_inputs_times = []
        self.build_attn_times = []
        self.sample_times = []
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def record_forward(self, elapsed_ms: float):
        if self.enabled:
            self.forward_times.append(elapsed_ms)

    def record_compute_logits(self, elapsed_ms: float):
        if self.enabled:
            self.compute_logits_times.append(elapsed_ms)

    def record_model_forward(self, elapsed_ms: float):
        if self.enabled:
            self.model_forward_times.append(elapsed_ms)

    def record_preprocess(self, elapsed_ms: float):
        if self.enabled:
            self.preprocess_times.append(elapsed_ms)

    def record_postprocess(self, elapsed_ms: float):
        if self.enabled:
            self.postprocess_times.append(elapsed_ms)

    def record_prepare_inputs(self, elapsed_ms: float):
        if self.enabled:
            self.prepare_inputs_times.append(elapsed_ms)

    def record_build_attn(self, elapsed_ms: float):
        if self.enabled:
            self.build_attn_times.append(elapsed_ms)

    def record_sample(self, elapsed_ms: float):
        if self.enabled:
            self.sample_times.append(elapsed_ms)

    def get_stats(
        self, approach: str, total_time_ms: float, num_tokens: int, num_prefill: int
    ) -> TimingStats:
        stats = TimingStats(approach=approach)
        stats.total_time_ms = total_time_ms
        stats.num_tokens_generated = num_tokens
        stats.num_prefill_tokens = num_prefill

        def avg(times):
            return sum(times) / len(times) if times else 0.0

        stats.forward_times = self.forward_times.copy()
        stats.forward_time_ms = avg(self.forward_times)
        stats.num_forward_calls = len(self.forward_times)

        stats.compute_logits_times = self.compute_logits_times.copy()
        stats.compute_logits_time_ms = avg(self.compute_logits_times)
        stats.num_compute_logits_calls = len(self.compute_logits_times)

        stats.model_forward_times = self.model_forward_times.copy()
        stats.model_forward_time_ms = avg(self.model_forward_times)

        stats.preprocess_time_ms = avg(self.preprocess_times)
        stats.postprocess_time_ms = avg(self.postprocess_times)
        stats.prepare_inputs_time_ms = avg(self.prepare_inputs_times)
        stats.build_attn_time_ms = avg(self.build_attn_times)
        stats.sample_time_ms = avg(self.sample_times)

        # Estimate prefill vs decode
        if len(self.forward_times) > 1:
            stats.prefill_time_ms = self.forward_times[0]
            stats.decode_time_ms = avg(self.forward_times[1:])
        elif len(self.forward_times) == 1:
            stats.prefill_time_ms = self.forward_times[0]

        return stats


# Global timing context
TIMING_CTX = TimingContext()


class TimingLogHandler(logging.Handler):
    """Custom log handler to capture timing logs."""

    def __init__(self, parser: LogParser):
        super().__init__()
        self.parser = parser

    def emit(self, record):
        try:
            msg = self.format(record)
            self.parser.parse_line(msg)
        except Exception:
            pass


def setup_timing_log_capture(
    default_benchmark=None,
) -> tuple[LogParser, TimingLogHandler]:
    """Setup log capture for timing analysis.

    Args:
        default_benchmark: Default benchmark type ("native" or "torchtitan") when not detected
    """
    parser = LogParser(default_benchmark=default_benchmark)
    handler = TimingLogHandler(parser)
    handler.setLevel(logging.INFO)

    # Add to root logger to capture all timing logs
    logging.getLogger().addHandler(handler)

    # Also add to vllm logger
    try:
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.addHandler(handler)
    except Exception:
        pass

    return parser, handler


def cleanup_timing_log_capture(handler: TimingLogHandler):
    """Remove the timing log handler."""
    logging.getLogger().removeHandler(handler)
    try:
        logging.getLogger("vllm").removeHandler(handler)
    except Exception:
        pass


def patch_vllm_for_timing():
    """
    Patch vLLM classes to add timing instrumentation.
    Call this before creating any vLLM LLM instances.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        # Save original methods
        original_model_forward = GPUModelRunner._model_forward

        def timed_model_forward(self, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_model_forward(self, *args, **kwargs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            TIMING_CTX.record_model_forward(elapsed_ms)
            return result

        GPUModelRunner._model_forward = timed_model_forward
        print("Patched GPUModelRunner._model_forward for timing")

    except ImportError as e:
        print(f"Warning: Could not patch vLLM for timing: {e}")
    except Exception as e:
        print(f"Warning: Error patching vLLM: {e}")


def patch_torchtitan_wrapper_for_timing():
    """
    Patch TorchTitanVLLMModelWrapper for timing instrumentation.
    """
    try:
        from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
            TorchTitanVLLMModelWrapper,
        )

        # Save original methods
        original_forward = TorchTitanVLLMModelWrapper.forward
        original_compute_logits = TorchTitanVLLMModelWrapper.compute_logits

        def timed_forward(self, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_forward(self, *args, **kwargs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            TIMING_CTX.record_forward(elapsed_ms)
            return result

        def timed_compute_logits(self, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_compute_logits(self, *args, **kwargs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            TIMING_CTX.record_compute_logits(elapsed_ms)
            return result

        TorchTitanVLLMModelWrapper.forward = timed_forward
        TorchTitanVLLMModelWrapper.compute_logits = timed_compute_logits
        print("Patched TorchTitanVLLMModelWrapper for timing")

    except ImportError as e:
        print(f"Warning: Could not patch TorchTitan wrapper for timing: {e}")
    except Exception as e:
        print(f"Warning: Error patching TorchTitan wrapper: {e}")


def run_vllm_native_benchmark(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    tp: int = 1,
    temperature: float = 0.0,
) -> TimingStats:
    """Run benchmark with vLLM native model."""
    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}")
    print("Running vLLM Native Benchmark")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"TP: {tp}")
    print(f"Max tokens: {max_tokens}")
    print(f"Num prompts: {len(prompts)}")

    # Reset timing context
    TIMING_CTX.reset()

    # Create engine
    engine = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=tp,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Warmup
    print("Warmup run...")
    _ = engine.generate(prompts[:1], sampling_params)
    torch.cuda.synchronize()

    # Actual run with timing
    print("Timed run...")
    TIMING_CTX.reset()
    TIMING_CTX.enable()

    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - start) * 1000

    TIMING_CTX.disable()

    # Count tokens
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_prefill = sum(len(prompt.split()) for prompt in prompts)  # Approximate

    stats = TIMING_CTX.get_stats(
        "vllm_native", total_time_ms, total_tokens, num_prefill
    )

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    print(f"Total time: {total_time_ms:.2f}ms")
    print(f"Tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens / (total_time_ms / 1000):.2f} tokens/sec")

    return stats


def run_vllm_torchtitan_benchmark(
    checkpoint_path: str,
    prompts: list[str],
    max_tokens: int,
    tp: int = 1,
    temperature: float = 0.0,
) -> TimingStats:
    """Run benchmark with vLLM TorchTitan model."""
    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}")
    print("Running vLLM TorchTitan Benchmark")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"TP: {tp}")
    print(f"Max tokens: {max_tokens}")
    print(f"Num prompts: {len(prompts)}")

    # Reset timing context
    TIMING_CTX.reset()

    # Create engine with TorchTitan model
    engine = LLM(
        model=checkpoint_path,
        hf_overrides={
            "architectures": ["Qwen3TorchTitanForCausalLM"],
        },
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=tp,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Warmup
    print("Warmup run...")
    _ = engine.generate(prompts[:1], sampling_params)
    torch.cuda.synchronize()

    # Actual run with timing
    print("Timed run...")
    TIMING_CTX.reset()
    TIMING_CTX.enable()

    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - start) * 1000

    TIMING_CTX.disable()

    # Count tokens
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_prefill = sum(len(prompt.split()) for prompt in prompts)  # Approximate

    stats = TIMING_CTX.get_stats(
        "vllm_torchtitan", total_time_ms, total_tokens, num_prefill
    )

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    print(f"Total time: {total_time_ms:.2f}ms")
    print(f"Tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens / (total_time_ms / 1000):.2f} tokens/sec")

    return stats


def print_comparison(native_stats: TimingStats, torchtitan_stats: TimingStats):
    """Print comparison between native and TorchTitan timing."""

    print(f"\n{'=' * 80}")
    print("TIMING COMPARISON: vLLM Native vs vLLM TorchTitan")
    print(f"{'=' * 80}")

    print(f"\n{'Metric':<35} {'Native':>15} {'TorchTitan':>15} {'Ratio':>10}")
    print("-" * 80)

    def print_row(name, native_val, tt_val, unit="ms"):
        if native_val > 0 and tt_val > 0:
            ratio = tt_val / native_val
            print(
                f"{name:<35} {native_val:>12.2f}{unit:>3} {tt_val:>12.2f}{unit:>3} {ratio:>9.2f}x"
            )
        elif native_val > 0:
            print(f"{name:<35} {native_val:>12.2f}{unit:>3} {'N/A':>15} {'N/A':>10}")
        elif tt_val > 0:
            print(f"{name:<35} {'N/A':>15} {tt_val:>12.2f}{unit:>3} {'N/A':>10}")

    print_row("Total Time", native_stats.total_time_ms, torchtitan_stats.total_time_ms)
    print_row(
        "Prefill Time (avg)",
        native_stats.prefill_time_ms,
        torchtitan_stats.prefill_time_ms,
    )
    print_row(
        "Decode Time (avg per step)",
        native_stats.decode_time_ms,
        torchtitan_stats.decode_time_ms,
    )
    print_row(
        "Forward Time (avg)",
        native_stats.forward_time_ms,
        torchtitan_stats.forward_time_ms,
    )
    print_row(
        "Compute Logits Time (avg)",
        native_stats.compute_logits_time_ms,
        torchtitan_stats.compute_logits_time_ms,
    )
    print_row(
        "Model Forward Time (avg)",
        native_stats.model_forward_time_ms,
        torchtitan_stats.model_forward_time_ms,
    )

    print("-" * 80)

    print(f"\n{'Count':<35} {'Native':>15} {'TorchTitan':>15}")
    print("-" * 60)
    print(
        f"{'Tokens Generated':<35} {native_stats.num_tokens_generated:>15} {torchtitan_stats.num_tokens_generated:>15}"
    )
    print(
        f"{'Forward Calls':<35} {native_stats.num_forward_calls:>15} {torchtitan_stats.num_forward_calls:>15}"
    )
    print(
        f"{'Compute Logits Calls':<35} {native_stats.num_compute_logits_calls:>15} {torchtitan_stats.num_compute_logits_calls:>15}"
    )

    # Throughput
    native_tps = (
        native_stats.num_tokens_generated / (native_stats.total_time_ms / 1000)
        if native_stats.total_time_ms > 0
        else 0
    )
    tt_tps = (
        torchtitan_stats.num_tokens_generated / (torchtitan_stats.total_time_ms / 1000)
        if torchtitan_stats.total_time_ms > 0
        else 0
    )

    print(f"\n{'Throughput (tokens/sec)':<35} {native_tps:>15.2f} {tt_tps:>15.2f}")
    if native_tps > 0:
        print(
            f"{'Relative Performance':<35} {'1.00x':>15} {tt_tps / native_tps:>14.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="vLLM Timing Analysis")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model path for native vLLM",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to HuggingFace checkpoint for TorchTitan",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument(
        "--num-prompts", type=int, default=5, help="Number of prompts to run"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max tokens to generate per prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="timing_analysis.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--skip-native", action="store_true", help="Skip native vLLM benchmark"
    )
    parser.add_argument(
        "--skip-torchtitan", action="store_true", help="Skip TorchTitan benchmark"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--analyze-log",
        type=str,
        default=None,
        help="Analyze existing log file instead of running benchmarks",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output for log parsing"
    )

    args = parser.parse_args()

    # If analyzing log file, just parse and print summary
    if args.analyze_log:
        print(f"Analyzing log file: {args.analyze_log}")
        log_parser = LogParser(debug=args.debug)
        log_parser.parse_file(args.analyze_log)
        log_parser.print_summary()

        # Save parsed results
        results = {
            "native": log_parser.native_aggregator.get_summary(),
            "torchtitan": log_parser.torchtitan_aggregator.get_summary(),
            "torchtitan_internal": log_parser.torchtitan_internal_aggregator.get_summary(),
            "by_phase": {
                "prefill": {
                    "native": log_parser.native_aggregator.get_phase_summary("prefill"),
                    "torchtitan": log_parser.torchtitan_aggregator.get_phase_summary(
                        "prefill"
                    ),
                },
                "decode": {
                    "native": log_parser.native_aggregator.get_phase_summary("decode"),
                    "torchtitan": log_parser.torchtitan_aggregator.get_phase_summary(
                        "decode"
                    ),
                },
            },
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
        return

    # Enable timing environment variables if not already set
    if os.environ.get("VLLM_TIMING") != "1":
        os.environ["VLLM_TIMING"] = "1"
        print("Enabled VLLM_TIMING=1")
    if os.environ.get("TORCHTITAN_VLLM_TIMING") != "1":
        os.environ["TORCHTITAN_VLLM_TIMING"] = "1"
        print("Enabled TORCHTITAN_VLLM_TIMING=1")

    # Apply timing patches
    patch_vllm_for_timing()
    patch_torchtitan_wrapper_for_timing()

    # Determine which benchmarks will run
    will_run_native = not args.skip_native
    will_run_torchtitan = not args.skip_torchtitan and args.checkpoint

    # Set default benchmark for log capture
    # If only one benchmark is running, set it as default
    if will_run_torchtitan and not will_run_native:
        default_benchmark = "torchtitan"
    elif will_run_native and not will_run_torchtitan:
        default_benchmark = "native"
    else:
        default_benchmark = None  # Both or neither - will detect from markers

    # Setup log capture
    log_parser, log_handler = setup_timing_log_capture(
        default_benchmark=default_benchmark
    )

    # Prepare prompts
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()][: args.num_prompts]
    else:
        prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to paint.",
            "What are the benefits of using Python for data science?",
            "Describe the process of photosynthesis.",
            "How does a neural network work?",
            "What is the difference between AI and machine learning?",
            "Explain quantum computing to a beginner.",
            "What are the main programming paradigms?",
        ][: args.num_prompts]

    print(f"Using {len(prompts)} prompts")

    results = {}

    # Run native vLLM benchmark
    if not args.skip_native:
        native_stats = run_vllm_native_benchmark(
            model_path=args.model_path,
            prompts=prompts,
            max_tokens=args.max_tokens,
            tp=args.tp,
        )
        results["native"] = asdict(native_stats)
    else:
        native_stats = None

    # Run TorchTitan vLLM benchmark
    if not args.skip_torchtitan and args.checkpoint:
        torchtitan_stats = run_vllm_torchtitan_benchmark(
            checkpoint_path=args.checkpoint,
            prompts=prompts,
            max_tokens=args.max_tokens,
            tp=args.tp,
        )
        results["torchtitan"] = asdict(torchtitan_stats)
    else:
        torchtitan_stats = None

    # Print comparison
    if native_stats and torchtitan_stats:
        print_comparison(native_stats, torchtitan_stats)

    # Print detailed timing from log capture
    print("\n" + "=" * 80)
    print("DETAILED TIMING FROM LOG CAPTURE")
    print("=" * 80)
    log_parser.print_summary()

    # Add log captured timing to results
    results["log_captured"] = {
        "native": log_parser.native_aggregator.get_summary(),
        "torchtitan": log_parser.torchtitan_aggregator.get_summary(),
        "torchtitan_internal": log_parser.torchtitan_internal_aggregator.get_summary(),
        "by_phase": {
            "prefill": {
                "native": log_parser.native_aggregator.get_phase_summary("prefill"),
                "torchtitan": log_parser.torchtitan_aggregator.get_phase_summary(
                    "prefill"
                ),
            },
            "decode": {
                "native": log_parser.native_aggregator.get_phase_summary("decode"),
                "torchtitan": log_parser.torchtitan_aggregator.get_phase_summary(
                    "decode"
                ),
            },
        },
    }

    # Cleanup
    cleanup_timing_log_capture(log_handler)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
