# Inference Benchmarking Suite

This directory contains scripts to benchmark and compare inference performance across three different approaches for running **Qwen3 1.7B** model.

## Approaches Compared

1. **vLLM Native**: vLLM engine with native Qwen3 1.7B model from HuggingFace
   - Model: `Qwen/Qwen3-1.7B` (base model)
   - Uses `Qwen3ForCausalLM` architecture

2. **vLLM TorchTitan**: vLLM engine with TorchTitan Qwen3 checkpoint (using `infer.py`)
   - Loads TorchTitan-trained Qwen3 checkpoint into vLLM
   - Uses the same architecture as native vLLM

3. **TorchTitan Native**: Direct inference using TorchTitan Qwen3 model without vLLM
   - Uses TorchTitan's Qwen3 model implementation
   - Follows the same approach as `test_generate.py`

## Installation

```bash
# Install vLLM
pip install vllm

# Install other dependencies
pip install numpy transformers torch

# Optional: Set HuggingFace token to avoid rate limits
export HF_TOKEN=your_token_here
# Or login via: huggingface-cli login
```

> **Note**: You may see a warning about unauthenticated HuggingFace requests. This is harmless but you can set `HF_TOKEN` to enable faster downloads and higher rate limits.

## Usage

### Basic Usage

```bash
# Auto-detect config (searches for ./train_configs/qwen3_1.7b.toml by default)
python scripts/benchmark_inference.py \
    --checkpoint /path/to/torchtitan/checkpoint.pt \
    --model-path Qwen/Qwen3-1.7B \
    --prompts scripts/benchmark_prompts.txt \
    --output results.json

# Or specify config explicitly
python scripts/benchmark_inference.py \
    --checkpoint /path/to/torchtitan/checkpoint.pt \
    --torchtitan-config /path/to/config.toml \
    --model-path Qwen/Qwen3-1.7B \
    --prompts scripts/benchmark_prompts.txt \
    --output results.json
```

### Advanced Usage

```bash
python scripts/benchmark_inference.py \
    --checkpoint /path/to/checkpoint.pt \
    --torchtitan-config /path/to/config.toml \
    --model-path Qwen/Qwen3-1.7B \
    --prompts scripts/benchmark_prompts.txt \
    --batch-size 8 \
    --max-tokens 1024 \
    --num-runs 10 \
    --warmup-runs 3 \
    --output detailed_results.json
```

### Skip Specific Benchmarks

```bash
# Only run vLLM benchmarks, skip native TorchTitan
python scripts/benchmark_inference.py \
    --checkpoint /path/to/checkpoint.pt \
    --skip-torchtitan-native

# Only run TorchTitan native
python scripts/benchmark_inference.py \
    --checkpoint /path/to/checkpoint.pt \
    --skip-vllm-native \
    --skip-vllm-torchtitan
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **Required** | Path to TorchTitan Qwen3 checkpoint |
| `--torchtitan-config` | str | `./train_configs/qwen3_1.7b.toml` | Path to TorchTitan config (TOML) file. Auto-detects if not specified. |
| `--model-path` | str | `Qwen/Qwen3-1.7B` | HuggingFace Qwen3 model path (base model) |
| `--prompts` | str | `prompts.txt` | File with prompts (one per line) |
| `--batch-size` | int | 4 | Number of prompts to process |
| `--max-tokens` | int | 512 | Max tokens to generate per prompt |
| `--num-runs` | int | 5 | Number of benchmark runs |
| `--warmup-runs` | int | 2 | Number of warmup runs |
| `--output` | str | `benchmark_results.json` | Output file for results |
| `--skip-vllm-native` | flag | False | Skip vLLM native benchmark |
| `--skip-vllm-torchtitan` | flag | False | Skip vLLM TorchTitan benchmark |
| `--skip-torchtitan-native` | flag | False | Skip TorchTitan native benchmark |

## Output Metrics

The benchmark collects the following metrics for each approach:

### Performance Metrics
- **Throughput**: Tokens generated per second
- **Latency per token**: Milliseconds per token (decode)
- **First token latency**: Time to generate first token (prefill)
- **Total time**: End-to-end inference time
- **Prefill time**: Time spent in prefill phase
- **Decode time**: Time spent in decode phase

### Memory Metrics
- **Memory allocated**: CUDA memory allocated
- **Memory reserved**: CUDA memory reserved
- **Peak memory**: Maximum CUDA memory used

### Statistics
For each metric, the following statistics are computed across runs:
- Mean
- Standard deviation
- Min/Max
- Median

## Example Output

```
================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================

vLLM Native
--------------------------------------------------------------------------------
  Throughput (tokens/sec):
    Mean:       245.32 ± 12.45
    Median:     243.56
    Range:      228.91 - 267.83

  Latency (ms/token):
    Mean:         4.08 ± 0.21
    Median:       4.11
    Range:        3.74 - 4.37

  First Token Latency (ms):
    Mean:        45.23 ± 2.34

  Peak Memory (GB):
    Mean:         3.42 ± 0.05

  Total tokens: 10,240
  Runs: 5

vLLM TorchTitan
--------------------------------------------------------------------------------
  Throughput (tokens/sec):
    Mean:       238.45 ± 15.67
    Median:     235.12
    Range:      215.34 - 259.87
  ...

TorchTitan Native
--------------------------------------------------------------------------------
  Throughput (tokens/sec):
    Mean:       187.92 ± 8.34
    Median:     189.45
    Range:      175.23 - 198.76
  ...

================================================================================
RELATIVE PERFORMANCE
================================================================================

vLLM TorchTitan vs vLLM Native:
  Speedup: 0.97x (-2.8%)

TorchTitan Native vs vLLM Native:
  Speedup: 0.77x (-23.4%)
```

## Output JSON Format

The results are saved in JSON format with the following structure:

```json
{
  "config": {
    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "torchtitan_checkpoint_path": "/path/to/checkpoint.pt",
    "batch_size": 4,
    "max_tokens": 512,
    "num_runs": 5,
    "warmup_runs": 2
  },
  "raw_results": {
    "vllm_native": [
      {
        "approach": "vLLM Native",
        "total_time": 8.234,
        "tokens_generated": 2048,
        "throughput_tokens_per_sec": 248.72,
        ...
      }
    ],
    ...
  },
  "statistics": {
    "vllm_native": {
      "approach": "vLLM Native",
      "throughput": {
        "mean": 245.32,
        "std": 12.45,
        "min": 228.91,
        "max": 267.83,
        "median": 243.56
      },
      ...
    },
    ...
  }
}
```

## Customizing Prompts

Create a text file with one prompt per line:

```
# prompts.txt
What is machine learning?
Explain neural networks.
How does backpropagation work?
```

Then pass it to the benchmark:

```bash
python scripts/benchmark_inference.py \
    --checkpoint checkpoint.pt \
    --prompts prompts.txt
```

## Tips for Accurate Benchmarking

1. **Warmup runs**: Use at least 2-3 warmup runs to allow CUDA kernels to compile
2. **Multiple runs**: Run at least 5 iterations to get stable statistics
3. **Consistent environment**: Close other GPU applications, use the same CUDA version
4. **Batch size**: Test with different batch sizes (1, 4, 8, 16) to see scaling
5. **Sequence length**: Test with different max_tokens (256, 512, 1024, 2048)
6. **GPU utilization**: Monitor `nvidia-smi` during benchmarks

## Troubleshooting

### HuggingFace Warnings

**Warning**: `You are sending unauthenticated requests to the HF Hub`
- This is harmless and doesn't affect functionality
- To suppress it, set your HuggingFace token:
  ```bash
  export HF_TOKEN=your_token_here
  # or
  huggingface-cli login
  ```

**Info**: `Resolved architecture: Qwen3ForCausalLM`
- This is **correct** for Qwen3 1.7B models
- vLLM will correctly detect the Qwen3 architecture

### Out of Memory (OOM)
- Reduce `--batch-size`
- Reduce `--max-tokens`
- For vLLM, the script sets `gpu_memory_utilization=0.9`, you can reduce this

### Import Errors
```bash
# Make sure vLLM is installed
pip install vllm

# Make sure transformers is installed
pip install transformers

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Checkpoint Loading Issues
- Ensure checkpoint path is correct
- Check that checkpoint contains `model` or `model_args` keys
- Verify checkpoint was saved from compatible TorchTitan version

## Advanced: Custom Benchmark Classes

You can extend the benchmark with custom approaches:

```python
from benchmark_inference import BenchmarkRunner, BenchmarkConfig, BenchmarkMetrics

class MyCustomBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def setup(self):
        # Initialize your model
        pass

    def run_inference(self, prompts: List[str]) -> BenchmarkMetrics:
        # Run inference and return metrics
        pass

    def cleanup(self):
        # Cleanup resources
        pass

# Add to runner
config = BenchmarkConfig(...)
runner = BenchmarkRunner(config)
runner.run_benchmark(MyCustomBenchmark, "my_custom")
```

## Visualizing Results

You can use the JSON output to create visualizations:

```python
import json
import matplotlib.pyplot as plt

with open('benchmark_results.json') as f:
    data = json.load(f)

stats = data['statistics']

approaches = list(stats.keys())
throughputs = [stats[k]['throughput']['mean'] for k in approaches]
errors = [stats[k]['throughput']['std'] for k in approaches]

plt.bar(approaches, throughputs, yerr=errors)
plt.ylabel('Throughput (tokens/sec)')
plt.title('Inference Performance Comparison')
plt.savefig('benchmark_chart.png')
```

## Contributing

To add new metrics or benchmark approaches, follow the existing pattern:

1. Create a new benchmark class inheriting the interface
2. Implement `setup()`, `run_inference()`, and `cleanup()` methods
3. Return a `BenchmarkMetrics` object with all fields populated
4. Add your benchmark to the runner

## License

BSD-style license (see LICENSE file in root directory)
