# SGLang generator engine

`SGLangEngine` runs the generator's TorchTitan model inside SGLang, as an alternative
to `VLLMEngine`. Both implement `GenerationEngine`, so a recipe selects SGLang with:

```python
from torchtitan.rl.generator.sglang.engine import SGLangEngine

generator=Generator.Config(..., engine=SGLangEngine.Config(enable_cuda_graph=True))
```

or, for an existing recipe, with the `sglang_engine` config override:

```bash
python -m torchtitan.rl.train --module torchtitan_recipes.rl.alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen_batch_invariant \
    --override torchtitan_recipes.overrides.sglang_engine.sglang_engine
```

In batch-invariant mode (`DebugConfig(batch_invariant=True, deterministic=True)`), the
SGLang engine returns the same tokens and bitwise-identical logprobs as the vLLM engine
for the same requests, including seeded temperature sampling. It does so regardless of
how either engine batches, chunks, retracts or preempts the requests.

## Design

| Piece | File | What it does |
|---|---|---|
| In-process scheduler | `engine.py` | SGLang's `Engine` spawns a scheduler process per GPU behind ZMQ. The RL generator owns one process per GPU and drives the engine with SPMD calls, so `SGLangEngine` builds SGLang's `Scheduler` in-process on each rank (as `VLLMEngine` uses vLLM's `external_launcher`) and runs one scheduler iteration per `step()`. Requests enter through `Scheduler.process_input_requests`; outputs are collected from the scheduler's output payloads instead of a detokenizer socket. |
| Model | `model_wrapper.py`, `registry.py` | `SGLangModelWrapper` builds the TorchTitan model on the meta device, parallelizes it, and loads weights through TorchTitan's checkpointer, as the vLLM wrapper does. It is registered in SGLang's `ModelRegistry`, and the TorchTitan config overrides the HF config fields SGLang reads (`json_model_override_args`). |
| Attention | `attention.py` | The `torchtitan_varlen` attention backend runs `torch.nn.attention.varlen.varlen_attn_out` on SGLang's paged KV pool. That is the kernel, with the same arguments, used by the trainer and by vLLM's `CUSTOM` backend. `SGLangInnerAttention` adapts TorchTitan's attention layout to SGLang's `RadixAttention`. Supports decode CUDA graphs. |
| Sampler | `sampler.py` | Reproduces vLLM's v1 sampler for the RL generator's sampling (fp32 logits, temperature, processed logprobs, exponential-race sampling with per-request seeded generators), so both engines sample identically. |

Shared, engine-agnostic helpers live in `torchtitan/rl/generator/unified_model.py`.

## Behavior that differs from SGLang defaults

- **Stop tokens.** The renderer's `stop_token_ids` are the only stops. The engine
  drops the model's EOS ids instead of setting `ignore_eos`, because in SGLang
  `ignore_eos` also disables `stop_token_ids`.
- **Logprobs.** Only output-token logprobs are returned. Input logprobs are off, which
  also covers generated tokens re-prefilled after a retraction.
- **Weight reload.** With `reset_kv_cache_on_model_state_dict_load=True`, a reload
  retracts running requests (`pause_generation(mode="retract")`), flushes the cache,
  and resumes. The retracted requests recompute their KV under the new weights.
  Per-request generators survive retraction, so a retraction never changes the
  sampled tokens.

## Install

SGLang is optional. Install it into the RL environment (`scripts/install_rl_env.sh`)
without dependencies, so it cannot replace the nightly torch:

```bash
git clone https://github.com/sgl-project/sglang && cd sglang/python
SGLANG_BUILD_RUST_EXTS=none pip install --no-deps --no-build-isolation -e .
pip install --no-deps sglang-kernel ipython traitlets blobfile easydict gguf orjson \
    zstandard torch_memory_saver soundfile build pycryptodomex lxml setproctitle scipy
```

The `sglang-kernel` wheel uses the stable ABI and loads with the nightly torch.

## Support

| Model / layout | Status |
|---|---|
| Dense, full attention, TP=1 | Supported; bitwise parity with `VLLMEngine` |
| Dense, full attention, TP > 1 | Supported; bitwise parity with `VLLMEngine` at the same TP degree (tested at TP=2 and TP=4) |
| MoE, no expert parallelism (TP=1) | Supported; bitwise parity with `VLLMEngine`. Below SM90, decode CUDA graphs are rejected: the expert GEMM reads group sizes on the host |

Not supported yet: expert parallelism, generator data parallelism, linear attention
(GDN, KDA), Dist-MoE, sequence parallelism, prefill CUDA graphs, multi-turn session KV
holding (`VLLMSessionKVHolder`), and per-step engine metrics.

## Tests

- `tests/rl/unit_tests/cpu/test_sglang_engine.py`: config translation, and the sampler
  against vLLM's sampling ops.
- `tests/rl/unit_tests/gpu/test_sglang_engine_parity.py`: runs both engines on Qwen3-0.6B
  and compares every token and logprob bitwise, across admission orders, chunked
  prefill, CUDA graphs, KV-pool pressure (SGLang retractions) and a weight reload
  during generation.
