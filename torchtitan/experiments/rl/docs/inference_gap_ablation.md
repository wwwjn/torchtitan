# Inference performance hill-climb results

This file records inference performance experiments in chronological order. Do
not rewrite prior results when adding a new experiment.

## Qwen3.5-27B TP4, H100, 2026-08-28

### Environment

- Branch: `perf/qwen35-inference-hillclimb`
- Base commit: `a069326d9`
- Model: `Qwen/Qwen3.5-27B`
- Checkpoint: `/data/users/jianiw/model/Qwen3.5-27B-local`
- Hardware: 4x NVIDIA H100 96 GB
- Driver: 580.126.09
- PyTorch: 2.14.0.dev20260723+cu130
- vLLM: 0.1.dev1+g27ffbfde8.d20260724
- Triton: 3.8.0
- Tensor parallel degree: 4
- Compilation: disabled
- CUDA graph mode: `FULL_AND_PIECEWISE`
- Collective backend: NCCL fallback for both TorchTitan and native vLLM

### Validation

Each retained rung was checked against native vLLM with deterministic greedy
generation using batch size 2, input length 64, and output length 8. The prompt
and generated token ID JSON files matched exactly. The focused CPU suite ended
with 30 passing tests and 1 skipped test.

### W1: decode-oriented comparison

Configuration: batch size 8, input length 1024, output length 128, maximum
sequence length 2048, maximum batched tokens 2048, 2 warmups, and 3 measured
runs. Throughput is aggregate generated tokens per second.

| Rung | Commit | Throughput | Gain from prior |
| --- | --- | ---: | ---: |
| Native vLLM | - | 446.1 +/- 0.3 tok/s | - |
| TorchTitan baseline | `9669afbea` | 265.0 +/- 0.9 tok/s | - |
| Packed GDN decode | `3766d97f2` | 302.8 +/- 0.5 tok/s | +14.3% |
| Merged GDN projections | `3689e4db9` | 315.3 +/- 1.2 tok/s | +4.1% |
| Fused SwiGLU | `8bfcbd144` | 319.1 +/- 0.3 tok/s | +1.2% |
| Merged full-attention projections | `41d200fc9` | 322.5 +/- 0.4 tok/s | +1.1% |
| Fused GDN RMSNorm and gate | `cf0f1ce0c` | 337.9 +/- 0.4 tok/s | +4.8% |
| Fused offset RMSNorm | `61241e36a` | 410.6 +/- 0.7 tok/s | +21.5% |
| Fused residual add and RMSNorm | `eaa2b3543` | 414.7 +/- 0.5 tok/s | +1.0% |
| Fused QK norm and RoPE | `c6176ee24` | 433.3 +/- 0.7 tok/s | +4.5% |
| FlashInfer GDN prefill | `2723a5a0f` | 440.8 +/- 2.8 tok/s | +1.7% |

The final path is 1.66x the initial TorchTitan baseline and reaches 98.8% of
native vLLM throughput on W1.

### W2: larger-batch and long-context comparison

Configuration: batch size 32, input length 4096, output length 1024, maximum
sequence length 8192, maximum batched tokens 8192, 1 warmup, and 1 measured run.

| Implementation | Throughput |
| --- | ---: |
| Native vLLM | 1330.9 tok/s |
| Optimized TorchTitan | 1597.1 tok/s |

The optimized TorchTitan path is 20.0% faster than native vLLM on W2.

### Root causes and accepted changes

- Decode used generic state gathering, transposition, and scattering around GDN.
  The packed recurrent decode kernel now updates paged state directly.
- Qwen3.5 represented native packed projections as separate linears. The
  inference adapter now caches TP-local concatenated weights while retaining the
  original checkpoint keys and invalidating derived weights after weight sync.
- SwiGLU gate and up projections now use TorchTitan's existing fused override.
- GDN output normalization and gating now use vLLM's fused FLA kernel.
- Offset RMSNorm now uses vLLM's fused CUDA RMSNorm with cached effective
  `(1 + weight)` values.
- Residuals are threaded across decoder layers so residual addition is fused
  into the following offset RMSNorm.
- Text full-attention layers use vLLM's fused QK RMSNorm, partial RoPE, and gate
  extraction kernel.
- Pure prefill uses vLLM's paged causal-convolution and FlashInfer GDN path;
  mixed prefill/decode retains the existing FLA path.

### Reproduction

Use the following suffix for the optimized TorchTitan path:

```bash
--compile off \
--cudagraph on \
--cudagraph-mode FULL_AND_PIECEWISE \
--disable-custom-all-reduce \
--flashinfer-gdn-prefill \
--fused-gdn-projections \
--fused-mlp \
--fused-attention-projections \
--fused-qk-norm-rope \
--fused-gdn-norm \
--fused-offset-norm \
--fused-residual-norm
```

For W1, add:

```bash
--benchmark --config rl_grpo_qwen3_5_27b_varlen \
--model-path /data/users/jianiw/model/Qwen3.5-27B-local \
--tp 4 --batch-size 8 --input-len 1024 --max-tokens 128 \
--temperature 0 --ignore-eos --warmup-runs 2 --num-runs 3 \
--max-seq-len 2048 --max-num-batched-tokens 2048
```

For W2, change the workload arguments to:

```bash
--batch-size 32 --input-len 4096 --max-tokens 1024 \
--warmup-runs 1 --num-runs 1 \
--max-seq-len 8192 --max-num-batched-tokens 8192
```

Launch the module with four processes:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m torchtitan.experiments.rl.generate <arguments>
```
