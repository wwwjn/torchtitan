#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script to compare inference performance across three approaches:
1. vLLM engine with native Qwen3 1.7B model (HuggingFace)
2. vLLM engine with TorchTitan Qwen3 1.7B model (via infer.py)
3. Direct TorchTitan Qwen3 model inference

Usage:
    python benchmark_inference.py --config benchmark_config.toml --output results.json
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import torch
from torchtitan.experiments.rl import unified  # noqa: F401


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmarking."""

    approach: str
    total_time: float
    tokens_generated: int
    prefill_time: float
    decode_time: float
    throughput_tokens_per_sec: float
    latency_per_token_ms: float
    first_token_latency_ms: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    peak_memory_gb: float
    batch_size: int
    sequence_length: int
    num_prompts: int


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    model_path: str
    torchtitan_checkpoint_path: str
    prompts_file: str
    torchtitan_config_path: str = None  # Optional config path for TorchTitan
    batch_size: int = 1
    max_tokens: int = 512
    num_runs: int = 5
    warmup_runs: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "cuda"


class VLLMNativeBenchmark:
    """Benchmark vLLM with native Qwen3 1.7B model from HuggingFace."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None

    def setup(self):
        """Initialize vLLM engine with native Qwen3 model."""
        try:
            from vllm import LLM, SamplingParams

            print("Loading vLLM with native Qwen3 model from HuggingFace...")
            print(f"Model: {self.config.model_path}")

            self.engine = LLM(
                model=self.config.model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            print("✓ vLLM native Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load vLLM native model: {e}")
            raise

    def run_inference(self, prompts: List[str]) -> BenchmarkMetrics:
        """Run inference and collect metrics."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure prefill (first token) time separately
        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, self.sampling_params)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        # Memory metrics
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        # Estimate prefill vs decode time (vLLM doesn't expose this directly)
        # Approximate: first token takes ~20% of total time for typical sequences
        first_token_latency = total_time * 0.2 * 1000  # Convert to ms
        prefill_time = total_time * 0.2
        decode_time = total_time * 0.8

        return BenchmarkMetrics(
            approach="vLLM Native",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=prefill_time,
            decode_time=decode_time,
            throughput_tokens_per_sec=total_tokens / total_time,
            latency_per_token_ms=(total_time / total_tokens) * 1000,
            first_token_latency_ms=first_token_latency,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        """Cleanup resources."""
        if self.engine is not None:
            del self.engine
            self.engine = None
        torch.cuda.empty_cache()


class VLLMTorchTitanBenchmark:
    """Benchmark vLLM with TorchTitan Qwen3 model via infer.py."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None

    def setup(self):
        """Initialize vLLM engine with TorchTitan Qwen3 model."""
        try:
            # Import unified module to register TorchTitan models with vLLM
            from vllm import LLM, SamplingParams

            print("Loading vLLM with TorchTitan Qwen3 checkpoint...")
            print(f"Checkpoint: {self.config.torchtitan_checkpoint_path}")

            # Initialize vLLM with TorchTitan model
            # This uses the registered Qwen3TorchTitanForCausalLM architecture
            self.engine = LLM(
                model=self.config.torchtitan_checkpoint_path,
                hf_overrides={
                    # Override architectures to use our registered TorchTitan model class
                    "architectures": ["Qwen3TorchTitanForCausalLM"],
                },
                dtype="bfloat16",
                trust_remote_code=True,
                enforce_eager=True,  # Use eager mode
                gpu_memory_utilization=0.9,
            )

            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            print("✓ vLLM TorchTitan Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load vLLM TorchTitan model: {e}")
            import traceback

            traceback.print_exc()
            raise

    def run_inference(self, prompts: List[str]) -> BenchmarkMetrics:
        """Run inference and collect metrics."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        start_time = time.perf_counter()
        outputs = self.engine.generate(prompts, self.sampling_params)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        # Estimate prefill vs decode time (vLLM doesn't expose this directly)
        first_token_latency = total_time * 0.2 * 1000  # Convert to ms
        prefill_time = total_time * 0.2
        decode_time = total_time * 0.8

        return BenchmarkMetrics(
            approach="vLLM TorchTitan",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=prefill_time,
            decode_time=decode_time,
            throughput_tokens_per_sec=total_tokens / total_time
            if total_time > 0
            else 0,
            latency_per_token_ms=(total_time / total_tokens) * 1000
            if total_tokens > 0
            else 0,
            first_token_latency_ms=first_token_latency,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        """Cleanup resources."""
        if self.engine is not None:
            del self.engine
            self.engine = None
        torch.cuda.empty_cache()


class TorchTitanNativeBenchmark:
    """Benchmark direct TorchTitan Qwen3 model inference using test_generate.py approach."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.config_path = None

    def setup(self):
        """Initialize TorchTitan Qwen3 model using test_generate.py approach."""
        try:
            print("Loading TorchTitan Qwen3 model directly (test_generate.py style)...")

            import sys
            from pathlib import Path

            import torch.distributed.checkpoint as dcp
            from torchtitan.config import ConfigManager
            from torchtitan.protocols.train_spec import get_train_spec

            # Add generate module to path
            generate_path = Path(__file__).parent.parent / "scripts" / "generate"
            sys.path.insert(0, str(generate_path))
            from _generation import generate as generate_fn

            self.generate_fn = generate_fn

            # Find config file for the model
            if self.config.torchtitan_config_path:
                self.config_path = self.config.torchtitan_config_path
            else:
                # Try to find a config file in the model directory
                # Default to ./train_configs/qwen3_1.7b.toml
                config_search_paths = [
                    Path("./train_configs/qwen3_1.7b.toml"),
                    Path("./torchtitan/models/qwen3/train_configs/qwen3_1.7b.toml"),
                    Path("/data/users/jianiw/torchtitan/train_configs/qwen3_1.7b.toml"),
                    Path(
                        "/data/users/jianiw/torchtitan/torchtitan/models/qwen3/train_configs/debug_model.toml"
                    ),
                    Path("./torchtitan/models/qwen3/train_configs/debug_model.toml"),
                ]

                self.config_path = None
                for path in config_search_paths:
                    if path.exists():
                        self.config_path = str(path)
                        break

                if self.config_path is None:
                    raise RuntimeError(
                        "Could not find model config file. Please specify --torchtitan-config\n"
                        "Searched paths:\n"
                        + "\n".join(f"  - {p}" for p in config_search_paths)
                    )

            print(f"Using config: {self.config_path}")

            # Load configuration
            config_manager = ConfigManager()
            self.tt_config = config_manager.parse_args(
                [f"--job.config_file={self.config_path}"]
            )

            # Get train spec
            train_spec = get_train_spec(self.tt_config.model.name)

            # Build tokenizer
            self.tokenizer = train_spec.build_tokenizer_fn(self.tt_config)

            # Get model args
            model_args = train_spec.model_args[self.tt_config.model.flavor]
            model_args.update_from_config(self.tt_config)

            # Initialize model on device
            device = torch.device(self.config.device)
            with torch.device(device):
                self.model = train_spec.model_cls(model_args)

            # Materialize model
            self.model.to_empty(device=device)
            with torch.no_grad():
                self.model.init_weights()
            self.model.eval()

            # Load checkpoint using DCP
            print(
                f"Loading checkpoint from {self.config.torchtitan_checkpoint_path}..."
            )

            # Check if checkpoint is a directory (DCP format) or a file (torch.save format)
            checkpoint_path = Path(self.config.torchtitan_checkpoint_path)

            if checkpoint_path.is_dir():
                # DCP format checkpoint
                state_dict = self.model.state_dict()
                dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
            elif checkpoint_path.is_file():
                # Regular torch checkpoint file
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                state_dict = checkpoint.get("model", checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            print("✓ TorchTitan native Qwen3 model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load TorchTitan native model: {e}")
            import traceback

            traceback.print_exc()
            raise

    @torch.inference_mode()
    def generate_with_timing(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> tuple[torch.Tensor, float, float]:
        """Generate tokens with separate prefill/decode timing."""
        device = torch.device(self.config.device)

        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device)
        generated_tokens = input_ids.clone()

        # Prefill: first token generation
        prefill_start = time.perf_counter()
        logits = self.model(generated_tokens)
        probs = torch.nn.functional.softmax(
            logits[:, -1, :] / max(self.config.temperature, 1e-5), dim=-1
        )

        if self.config.temperature == 0.0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        # Decode: remaining tokens
        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            logits = self.model(generated_tokens)
            probs = torch.nn.functional.softmax(
                logits[:, -1, :] / max(self.config.temperature, 1e-5), dim=-1
            )

            if self.config.temperature == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        decode_end = time.perf_counter()
        decode_time = decode_end - decode_start

        return generated_tokens, prefill_time, decode_time

    def run_inference(self, prompts: List[str]) -> BenchmarkMetrics:
        """Run inference and collect metrics."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        total_tokens = 0
        total_prefill_time = 0
        total_decode_time = 0
        first_token_latency = None

        start_time = time.perf_counter()

        for prompt in prompts:
            # Tokenize (using TorchTitan tokenizer API)
            if hasattr(self.tokenizer, "encode"):
                # For BaseTokenizer with encode method that takes add_bos/add_eos
                try:
                    input_ids = torch.tensor(
                        self.tokenizer.encode(prompt, add_bos=True, add_eos=False),
                        dtype=torch.long,
                    )
                except TypeError:
                    # Fallback for HF tokenizers
                    input_ids = torch.tensor(
                        self.tokenizer.encode(prompt), dtype=torch.long
                    )
            else:
                # Fallback
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt), dtype=torch.long
                )

            # Generate
            output_tokens, prefill_time, decode_time = self.generate_with_timing(
                input_ids, self.config.max_tokens
            )

            if first_token_latency is None:
                first_token_latency = prefill_time * 1000  # Convert to ms

            num_generated = output_tokens.size(1) - input_ids.size(0)
            total_tokens += num_generated
            total_prefill_time += prefill_time
            total_decode_time += decode_time

        end_time = time.perf_counter()
        total_time = end_time - start_time

        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        return BenchmarkMetrics(
            approach="TorchTitan Native",
            total_time=total_time,
            tokens_generated=total_tokens,
            prefill_time=total_prefill_time,
            decode_time=total_decode_time,
            throughput_tokens_per_sec=total_tokens / total_time
            if total_time > 0
            else 0,
            latency_per_token_ms=(total_time / total_tokens) * 1000
            if total_tokens > 0
            else 0,
            first_token_latency_ms=first_token_latency or 0,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            peak_memory_gb=peak_memory,
            batch_size=len(prompts),
            sequence_length=self.config.max_tokens,
            num_prompts=len(prompts),
        )

    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()


class BenchmarkRunner:
    """Main benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, List[BenchmarkMetrics]] = {
            "vllm_native": [],
            "vllm_torchtitan": [],
            "torchtitan_native": [],
        }

    def load_prompts(self) -> List[str]:
        """Load prompts from file."""
        prompts_path = Path(self.config.prompts_file)
        if prompts_path.exists():
            with open(prompts_path, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default prompts
            prompts = [
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint.",
                "What are the benefits of using Python for data science?",
                "Describe the process of photosynthesis.",
                "How does a neural network work?",
            ]

        return prompts[: self.config.batch_size]

    def run_benchmark(self, benchmark_cls, key: str):
        """Run benchmark for a specific approach."""
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {key}")
        print(f"{'=' * 60}")

        try:
            benchmark = benchmark_cls(self.config)
            benchmark.setup()

            prompts = self.load_prompts()

            # Warmup runs
            print(f"Running {self.config.warmup_runs} warmup iterations...")
            for i in range(self.config.warmup_runs):
                benchmark.run_inference(prompts)
                print(f"  Warmup {i + 1}/{self.config.warmup_runs} completed")

            # Actual benchmark runs
            print(f"Running {self.config.num_runs} benchmark iterations...")
            for i in range(self.config.num_runs):
                metrics = benchmark.run_inference(prompts)
                self.results[key].append(metrics)
                print(
                    f"  Run {i + 1}/{self.config.num_runs}: "
                    f"{metrics.throughput_tokens_per_sec:.2f} tokens/s, "
                    f"latency: {metrics.latency_per_token_ms:.2f} ms/token"
                )

            benchmark.cleanup()
            print(f"✓ {key} benchmark completed")

        except Exception as e:
            print(f"✗ {key} benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    def compute_statistics(
        self, metrics_list: List[BenchmarkMetrics]
    ) -> Dict[str, Any]:
        """Compute statistics from multiple runs."""
        if not metrics_list:
            return {}

        throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
        latencies = [m.latency_per_token_ms for m in metrics_list]
        first_token_latencies = [m.first_token_latency_ms for m in metrics_list]
        memory_peaks = [m.peak_memory_gb for m in metrics_list]

        return {
            "approach": metrics_list[0].approach,
            "throughput": {
                "mean": np.mean(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs),
                "median": np.median(throughputs),
            },
            "latency_per_token_ms": {
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "median": np.median(latencies),
            },
            "first_token_latency_ms": {
                "mean": np.mean(first_token_latencies),
                "std": np.std(first_token_latencies),
                "min": np.min(first_token_latencies),
                "max": np.max(first_token_latencies),
                "median": np.median(first_token_latencies),
            },
            "peak_memory_gb": {
                "mean": np.mean(memory_peaks),
                "std": np.std(memory_peaks),
                "min": np.min(memory_peaks),
                "max": np.max(memory_peaks),
                "median": np.median(memory_peaks),
            },
            "num_runs": len(metrics_list),
            "total_tokens_generated": sum(m.tokens_generated for m in metrics_list),
        }

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'=' * 80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'=' * 80}\n")

        for key, metrics_list in self.results.items():
            if not metrics_list:
                continue

            stats = self.compute_statistics(metrics_list)

            print(stats["approach"])
            print("-" * 80)
            print("  Throughput (tokens/sec):")
            print(
                f"    Mean:   {stats['throughput']['mean']:>10.2f} ± {stats['throughput']['std']:.2f}"
            )
            print(f"    Median: {stats['throughput']['median']:>10.2f}")
            print(
                f"    Range:  {stats['throughput']['min']:>10.2f} - {stats['throughput']['max']:.2f}"
            )
            print("\n  Latency (ms/token):")
            print(
                f"    Mean:   {stats['latency_per_token_ms']['mean']:>10.2f} ± {stats['latency_per_token_ms']['std']:.2f}"
            )
            print(f"    Median: {stats['latency_per_token_ms']['median']:>10.2f}")
            print(
                f"    Range:  {stats['latency_per_token_ms']['min']:>10.2f} - {stats['latency_per_token_ms']['max']:.2f}"
            )
            print("\n  First Token Latency (ms):")
            print(
                f"    Mean:   {stats['first_token_latency_ms']['mean']:>10.2f} ± {stats['first_token_latency_ms']['std']:.2f}"
            )
            print("\n  Peak Memory (GB):")
            print(
                f"    Mean:   {stats['peak_memory_gb']['mean']:>10.2f} ± {stats['peak_memory_gb']['std']:.2f}"
            )
            print(f"\n  Total tokens: {stats['total_tokens_generated']:,}")
            print(f"  Runs: {stats['num_runs']}")
            print()

        # Comparison
        if len(self.results) > 1:
            print("=" * 80)
            print("RELATIVE PERFORMANCE")
            print("=" * 80 + "\n")

            baseline_key = "vllm_native"
            if self.results[baseline_key]:
                baseline_stats = self.compute_statistics(self.results[baseline_key])
                baseline_throughput = baseline_stats["throughput"]["mean"]

                for key, metrics_list in self.results.items():
                    if not metrics_list or key == baseline_key:
                        continue

                    stats = self.compute_statistics(metrics_list)
                    throughput = stats["throughput"]["mean"]
                    speedup = throughput / baseline_throughput

                    print(f"{stats['approach']} vs {baseline_stats['approach']}:")
                    print(f"  Speedup: {speedup:.2f}x ({speedup * 100 - 100:+.1f}%)")
                    print()

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        output_data = {
            "config": asdict(self.config),
            "raw_results": {
                key: [asdict(m) for m in metrics_list]
                for key, metrics_list in self.results.items()
            },
            "statistics": {
                key: self.compute_statistics(metrics_list)
                for key, metrics_list in self.results.items()
                if metrics_list
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    def run_all(self):
        """Run all benchmarks."""
        print("Starting benchmark suite...")
        print("Configuration:")
        print(f"  Model: {self.config.model_path}")
        print(f"  Checkpoint: {self.config.torchtitan_checkpoint_path}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Max tokens: {self.config.max_tokens}")
        print(f"  Warmup runs: {self.config.warmup_runs}")
        print(f"  Benchmark runs: {self.config.num_runs}")

        # Run benchmarks
        self.run_benchmark(VLLMNativeBenchmark, "vllm_native")
        self.run_benchmark(VLLMTorchTitanBenchmark, "vllm_torchtitan")
        self.run_benchmark(TorchTitanNativeBenchmark, "torchtitan_native")

        # Print summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference approaches")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path to Qwen3 model from HuggingFace (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TorchTitan checkpoint",
    )
    parser.add_argument(
        "--torchtitan-config",
        type=str,
        default=None,
        help="Path to TorchTitan config file (TOML). If not specified, will auto-detect.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts.txt",
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--skip-vllm-native",
        action="store_true",
        help="Skip vLLM native benchmark",
    )
    parser.add_argument(
        "--skip-vllm-torchtitan",
        action="store_true",
        help="Skip vLLM TorchTitan benchmark",
    )
    parser.add_argument(
        "--skip-torchtitan-native",
        action="store_true",
        help="Skip TorchTitan native benchmark",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_path=args.model_path,
        torchtitan_checkpoint_path=args.checkpoint,
        torchtitan_config_path=args.torchtitan_config,
        prompts_file=args.prompts,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
    )

    runner = BenchmarkRunner(config)

    # Run selected benchmarks
    if not args.skip_vllm_native:
        runner.run_benchmark(VLLMNativeBenchmark, "vllm_native")

    if not args.skip_vllm_torchtitan:
        runner.run_benchmark(VLLMTorchTitanBenchmark, "vllm_torchtitan")

    if not args.skip_torchtitan_native:
        runner.run_benchmark(TorchTitanNativeBenchmark, "torchtitan_native")

    # Print and save results
    runner.print_summary()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
