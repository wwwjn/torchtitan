#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Benchmark script that runs all three benchmark suites with proper TP handling:
# 1. vLLM Native - runs with python (vLLM handles TP internally via tensor_parallel_size)
# 2. vLLM TorchTitan - runs with python (vLLM handles TP internally)
# 3. TorchTitan Native - runs with torchrun when TP > 1

set -e

# Default values
CHECKPOINT=""
MODEL_PATH="Qwen/Qwen3-1.7B"
TP=1
BATCH_SIZE=4
MAX_TOKENS=512
NUM_RUNS=5
WARMUP_RUNS=2
OUTPUT_DIR="benchmark_results"
TORCHTITAN_CONFIG=""
PROMPTS_FILE="scripts/prompts.txt"

# Profiling options
PROFILE=false
PROFILE_DIR="./profiler_traces"

# Benchmark selection (all enabled by default)
RUN_VLLM_NATIVE=true
RUN_VLLM_TORCHTITAN=true
RUN_TORCHTITAN_NATIVE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --torchtitan-config)
            TORCHTITAN_CONFIG="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --profile-dir)
            PROFILE_DIR="$2"
            shift 2
            ;;
        --skip-vllm-native)
            RUN_VLLM_NATIVE=false
            shift
            ;;
        --skip-vllm-torchtitan)
            RUN_VLLM_TORCHTITAN=false
            shift
            ;;
        --skip-torchtitan-native)
            RUN_TORCHTITAN_NATIVE=false
            shift
            ;;
        --only-vllm-native)
            RUN_VLLM_NATIVE=true
            RUN_VLLM_TORCHTITAN=false
            RUN_TORCHTITAN_NATIVE=false
            shift
            ;;
        --only-vllm-torchtitan)
            RUN_VLLM_NATIVE=false
            RUN_VLLM_TORCHTITAN=true
            RUN_TORCHTITAN_NATIVE=false
            shift
            ;;
        --only-torchtitan-native)
            RUN_VLLM_NATIVE=false
            RUN_VLLM_TORCHTITAN=false
            RUN_TORCHTITAN_NATIVE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --checkpoint <path> [options]"
            echo ""
            echo "Required:"
            echo "  --checkpoint <path>        Path to HuggingFace checkpoint"
            echo ""
            echo "Options:"
            echo "  --model-path <path>        HuggingFace model path (default: Qwen/Qwen3-1.7B)"
            echo "  --tp <int>                 Tensor parallelism size (default: 1)"
            echo "  --batch-size <int>         Batch size (default: 4)"
            echo "  --max-tokens <int>         Max tokens to generate (default: 512)"
            echo "  --num-runs <int>           Number of benchmark runs (default: 5)"
            echo "  --warmup-runs <int>        Number of warmup runs (default: 2)"
            echo "  --output-dir <path>        Output directory for results (default: benchmark_results)"
            echo "  --torchtitan-config <path> TorchTitan config file path"
            echo "  --prompts <path>           Prompts file path (default: scripts/prompts.txt)"
            echo ""
            echo "Profiling options:"
            echo "  --profile                  Enable PyTorch profiler for performance tracing"
            echo "  --profile-dir <path>       Directory to save profiler traces (default: ./profiler_traces)"
            echo ""
            echo "Benchmark selection:"
            echo "  --skip-vllm-native         Skip vLLM native benchmark"
            echo "  --skip-vllm-torchtitan     Skip vLLM TorchTitan benchmark"
            echo "  --skip-torchtitan-native   Skip TorchTitan native benchmark"
            echo "  --only-vllm-native         Only run vLLM native benchmark"
            echo "  --only-vllm-torchtitan     Only run vLLM TorchTitan benchmark"
            echo "  --only-torchtitan-native   Only run TorchTitan native benchmark"
            echo ""
            echo "Examples:"
            echo "  # Run all benchmarks with TP=1"
            echo "  $0 --checkpoint /path/to/checkpoint --tp 1"
            echo ""
            echo "  # Run all benchmarks with TP=2"
            echo "  $0 --checkpoint /path/to/checkpoint --tp 2"
            echo ""
            echo "  # Run only TorchTitan native with TP=4"
            echo "  $0 --checkpoint /path/to/checkpoint --tp 4 --only-torchtitan-native"
            echo ""
            echo "  # Run benchmarks with profiling enabled"
            echo "  $0 --checkpoint /path/to/checkpoint --profile --profile-dir ./traces"
            echo ""
            echo "  # Profile only vLLM native benchmark"
            echo "  $0 --checkpoint /path/to/checkpoint --only-vllm-native --profile --num-runs 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Use --help for usage information"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_inference.py"

# Common arguments
COMMON_ARGS="--checkpoint $CHECKPOINT --model-path $MODEL_PATH --tp $TP --batch-size $BATCH_SIZE --max-tokens $MAX_TOKENS --num-runs $NUM_RUNS --warmup-runs $WARMUP_RUNS --prompts $PROMPTS_FILE"

if [ -n "$TORCHTITAN_CONFIG" ]; then
    COMMON_ARGS="$COMMON_ARGS --torchtitan-config $TORCHTITAN_CONFIG"
fi

# Add profiling arguments if enabled
if [ "$PROFILE" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --profile --profile-dir $PROFILE_DIR"
    mkdir -p "$PROFILE_DIR"
fi

echo "============================================================"
echo "Benchmark Configuration"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Model Path: $MODEL_PATH"
echo "Tensor Parallelism: $TP"
echo "Batch Size: $BATCH_SIZE"
echo "Max Tokens: $MAX_TOKENS"
echo "Num Runs: $NUM_RUNS"
echo "Warmup Runs: $WARMUP_RUNS"
echo "Output Directory: $OUTPUT_DIR"
if [ "$PROFILE" = true ]; then
    echo ""
    echo "Profiling: ENABLED"
    echo "Profile Directory: $PROFILE_DIR"
else
    echo ""
    echo "Profiling: disabled"
fi
echo ""
echo "Benchmarks to run:"
echo "  vLLM Native: $RUN_VLLM_NATIVE"
echo "  vLLM TorchTitan: $RUN_VLLM_TORCHTITAN"
echo "  TorchTitan Native: $RUN_TORCHTITAN_NATIVE"
echo "============================================================"
echo ""

# Run vLLM Native benchmark (uses python, vLLM handles TP internally)
if [ "$RUN_VLLM_NATIVE" = true ]; then
    echo ""
    echo "============================================================"
    echo "Running vLLM Native Benchmark (TP=$TP)"
    echo "============================================================"
    echo ""

    python "$BENCHMARK_SCRIPT" \
        $COMMON_ARGS \
        --output "$OUTPUT_DIR/vllm_native_tp${TP}.json" \
        --skip-vllm-torchtitan \
        --skip-torchtitan-native

    echo ""
    echo "vLLM Native benchmark completed. Results: $OUTPUT_DIR/vllm_native_tp${TP}.json"
fi

# Run vLLM TorchTitan benchmark (uses python, vLLM handles TP internally)
if [ "$RUN_VLLM_TORCHTITAN" = true ]; then
    echo ""
    echo "============================================================"
    echo "Running vLLM TorchTitan Benchmark (TP=$TP)"
    echo "============================================================"
    echo ""

    python "$BENCHMARK_SCRIPT" \
        $COMMON_ARGS \
        --output "$OUTPUT_DIR/vllm_torchtitan_tp${TP}.json" \
        --skip-vllm-native \
        --skip-torchtitan-native

    echo ""
    echo "vLLM TorchTitan benchmark completed. Results: $OUTPUT_DIR/vllm_torchtitan_tp${TP}.json"
fi

# Run TorchTitan Native benchmark (uses torchrun for TP > 1)
if [ "$RUN_TORCHTITAN_NATIVE" = true ]; then
    echo ""
    echo "============================================================"
    echo "Running TorchTitan Native Benchmark (TP=$TP)"
    echo "============================================================"
    echo ""

    if [ "$TP" -gt 1 ]; then
        # Use torchrun for TP > 1
        torchrun --nproc_per_node=$TP "$BENCHMARK_SCRIPT" \
            $COMMON_ARGS \
            --output "$OUTPUT_DIR/torchtitan_native_tp${TP}.json" \
            --skip-vllm-native \
            --skip-vllm-torchtitan
    else
        # Use regular python for TP = 1
        python "$BENCHMARK_SCRIPT" \
            $COMMON_ARGS \
            --output "$OUTPUT_DIR/torchtitan_native_tp${TP}.json" \
            --skip-vllm-native \
            --skip-vllm-torchtitan
    fi

    echo ""
    echo "TorchTitan Native benchmark completed. Results: $OUTPUT_DIR/torchtitan_native_tp${TP}.json"
fi

echo ""
echo "============================================================"
echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR/"
if [ "$PROFILE" = true ]; then
    echo "Profiler traces saved to: $PROFILE_DIR/"
    echo ""
    echo "To visualize traces:"
    echo "  - Perfetto: https://ui.perfetto.dev/"
    echo "  - TensorBoard: tensorboard --logdir=$PROFILE_DIR"
fi
echo "============================================================"

# List all result files
echo ""
echo "Result files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No result files found"

# List profiler trace directories if profiling was enabled
if [ "$PROFILE" = true ]; then
    echo ""
    echo "Profiler trace directories:"
    ls -la "$PROFILE_DIR"/ 2>/dev/null || echo "No profiler traces found"
fi
