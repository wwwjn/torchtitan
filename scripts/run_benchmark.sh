#!/bin/bash
# Example script to run the inference benchmark

# Configuration
CHECKPOINT_PATH="/path/to/your/torchtitan/checkpoint.pt"
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).json"

# Run benchmark
echo "Starting inference benchmark..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

python scripts/benchmark_inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --model-path "$MODEL_PATH" \
    --prompts scripts/benchmark_prompts.txt \
    --batch-size 4 \
    --max-tokens 512 \
    --num-runs 5 \
    --warmup-runs 2 \
    --output "$OUTPUT_FILE"

# Check if benchmark succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Benchmark completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "Generating visualizations..."

    python scripts/visualize_benchmark.py "$OUTPUT_FILE" \
        --output-dir "benchmark_plots_$(date +%Y%m%d_%H%M%S)"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Visualizations generated successfully!"
    fi
else
    echo ""
    echo "✗ Benchmark failed!"
    exit 1
fi
