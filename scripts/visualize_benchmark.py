#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualize benchmark results from benchmark_inference.py

Usage:
    python visualize_benchmark.py benchmark_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_throughput(stats, output_dir):
    """Plot throughput comparison."""
    approaches = [s["approach"] for s in stats.values()]
    means = [s["throughput"]["mean"] for s in stats.values()]
    stds = [s["throughput"]["std"] for s in stats.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(approaches))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="black")

    # Color bars
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)
    ax.set_title("Inference Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 5, f"{mean:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "throughput.png", dpi=300)
    print(f"✓ Saved throughput plot to {output_dir}/throughput.png")
    plt.close()


def plot_latency(stats, output_dir):
    """Plot latency comparison."""
    approaches = [s["approach"] for s in stats.values()]
    means = [s["latency_per_token_ms"]["mean"] for s in stats.values()]
    stds = [s["latency_per_token_ms"]["std"] for s in stats.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(approaches))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="black")

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel("Latency (ms/token)", fontsize=12)
    ax.set_title("Per-Token Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i, mean + std + 0.1, f"{mean:.2f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "latency.png", dpi=300)
    print(f"✓ Saved latency plot to {output_dir}/latency.png")
    plt.close()


def plot_memory(stats, output_dir):
    """Plot memory usage comparison."""
    approaches = [s["approach"] for s in stats.values()]
    means = [s["peak_memory_gb"]["mean"] for s in stats.values()]
    stds = [s["peak_memory_gb"]["std"] for s in stats.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(approaches))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="black")

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel("Peak Memory (GB)", fontsize=12)
    ax.set_title("Peak GPU Memory Usage", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i, mean + std + 0.05, f"{mean:.2f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "memory.png", dpi=300)
    print(f"✓ Saved memory plot to {output_dir}/memory.png")
    plt.close()


def plot_first_token_latency(stats, output_dir):
    """Plot first token latency comparison."""
    approaches = [s["approach"] for s in stats.values()]
    means = [s["first_token_latency_ms"]["mean"] for s in stats.values()]
    stds = [s["first_token_latency_ms"]["std"] for s in stats.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(approaches))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="black")

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel("First Token Latency (ms)", fontsize=12)
    ax.set_title("Time to First Token (TTFT)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 1, f"{mean:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "first_token_latency.png", dpi=300)
    print(f"✓ Saved first token latency plot to {output_dir}/first_token_latency.png")
    plt.close()


def plot_combined_summary(stats, output_dir):
    """Plot a combined summary dashboard."""
    approaches = [s["approach"] for s in stats.values()]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(approaches))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    # Throughput
    throughputs = [s["throughput"]["mean"] for s in stats.values()]
    bars1 = ax1.bar(x, throughputs, alpha=0.8, edgecolor="black")
    for bar, color in zip(bars1, colors):
        bar.set_color(color)
    ax1.set_ylabel("Tokens/sec", fontsize=11)
    ax1.set_title("Throughput", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches, fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Latency
    latencies = [s["latency_per_token_ms"]["mean"] for s in stats.values()]
    bars2 = ax2.bar(x, latencies, alpha=0.8, edgecolor="black")
    for bar, color in zip(bars2, colors):
        bar.set_color(color)
    ax2.set_ylabel("ms/token", fontsize=11)
    ax2.set_title("Per-Token Latency", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(approaches, fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Memory
    memory = [s["peak_memory_gb"]["mean"] for s in stats.values()]
    bars3 = ax3.bar(x, memory, alpha=0.8, edgecolor="black")
    for bar, color in zip(bars3, colors):
        bar.set_color(color)
    ax3.set_ylabel("GB", fontsize=11)
    ax3.set_title("Peak Memory", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(approaches, fontsize=10)
    ax3.grid(axis="y", alpha=0.3)

    # First Token Latency
    ftl = [s["first_token_latency_ms"]["mean"] for s in stats.values()]
    bars4 = ax4.bar(x, ftl, alpha=0.8, edgecolor="black")
    for bar, color in zip(bars4, colors):
        bar.set_color(color)
    ax4.set_ylabel("ms", fontsize=11)
    ax4.set_title("First Token Latency", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(approaches, fontsize=10)
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Inference Benchmark Summary", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "summary.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved summary dashboard to {output_dir}/summary.png")
    plt.close()


def plot_speedup(stats, output_dir):
    """Plot relative speedup compared to baseline."""
    baseline_key = list(stats.keys())[0]
    baseline_throughput = stats[baseline_key]["throughput"]["mean"]

    approaches = []
    speedups = []

    for key, stat in stats.items():
        if key == baseline_key:
            continue
        approaches.append(stat["approach"])
        speedup = stat["throughput"]["mean"] / baseline_throughput
        speedups.append(speedup)

    if not approaches:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(approaches))
    bars = ax.bar(x, speedups, alpha=0.8, edgecolor="black")

    # Color bars based on speedup (green if >1, red if <1)
    for bar, speedup in zip(bars, speedups):
        bar.set_color("#2ecc71" if speedup >= 1.0 else "#e74c3c")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Baseline")
    ax.set_ylabel("Relative Speedup", fontsize=12)
    ax.set_title(
        f'Speedup Relative to {stats[baseline_key]["approach"]}',
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for i, speedup in enumerate(speedups):
        label = f"{speedup:.2f}x\n({(speedup - 1) * 100:+.1f}%)"
        ax.text(i, speedup + 0.02, label, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "speedup.png", dpi=300)
    print(f"✓ Saved speedup plot to {output_dir}/speedup.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results_file", type=str, help="Path to benchmark results JSON")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        data = json.load(f)

    stats = data.get("statistics", {})
    if not stats:
        print("Error: No statistics found in results file")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")

    # Generate all plots
    plot_throughput(stats, output_dir)
    plot_latency(stats, output_dir)
    plot_memory(stats, output_dir)
    plot_first_token_latency(stats, output_dir)
    plot_combined_summary(stats, output_dir)
    plot_speedup(stats, output_dir)

    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - throughput.png")
    print("  - latency.png")
    print("  - memory.png")
    print("  - first_token_latency.png")
    print("  - summary.png")
    print("  - speedup.png")


if __name__ == "__main__":
    main()
