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
| Model | `model_wrapper.py`, `registry.py` | `SGLangModelWrapper` builds the TorchTitan model on the meta device, parallelizes it, and loads weights through TorchTitan's checkpointer, as the vLLM wrapper does. It is registered in SGLang's `ModelRegistry`. SGLang reads only an HF `config.json` from its model path, so the engine writes one derived from the TorchTitan model config into a private directory, as vLLM's TorchTitan config parser does. |
| Attention | `attention.py` | The `torchtitan_varlen` attention backend runs `torch.nn.attention.varlen.varlen_attn_out` on SGLang's paged KV pool. That is the kernel, with the same arguments, used by the trainer and by vLLM's `CUSTOM` backend. `SGLangInnerAttention` adapts TorchTitan's attention layout to SGLang's `RadixAttention`. Supports decode CUDA graphs. |
| Expert and data parallelism | `engine.py`, `model_wrapper.py` | With EP and TP, the wrapper pads each step's tokens to a TP multiple, as the vLLM runner does. Generator DP exists to supply EP ranks, whose all-to-alls span every DP rank, so DP > 1 runs SGLang's data-parallel attention: one world of DP x TP ranks, attention data-parallel over DP and tensor-parallel over TP, and a rank without requests runs an idle batch through the model whenever another rank has work. `has_unfinished_requests` is collective across ranks. |
| GDN linear attention | `gdn.py`, `../paged_gdn.py` | Hybrid models (GDN layers with full-attention layers) use SGLang's linear-attention model registry: the engine's HF config lists the layer types and the per-slot state shapes, SGLang keeps the GDN states in its mamba pool next to the KV pool, and `TorchTitanGDNAttnBackend` builds each step's slot indices and sequence boundaries. `SGLangInnerGDN` runs `paged_gdn_forward`, the function the vLLM engine's GDN layer runs. The SSM state is float32. The radix cache is disabled: reusing a prefix would need the recurrent state at the prefix boundary. Supports decode CUDA graphs. |
| Session KV | `engine.py` | With `hold_session_kv`, requests carry their `routing_session_id` as SGLang's radix session id (`enable_session_radix_cache`). When a turn finishes, the prompt and output KV it caches is tagged with its session, and eviction takes untagged KV first, so a session's history survives other requests between its turns. `release_sessions` drops the tags of ended rollouts and groups. Unlike `VLLMSessionKVHolder`, which pins blocks, the tag only orders eviction. It works with or without `reset_kv_cache_on_model_state_dict_load`. |
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
| MoE, expert parallelism with EP = DP x TP | Supported; bitwise parity with `VLLMEngine` (tested at TP=2/EP=2, DP=2/TP=1/EP=2, DP=2/TP=2/EP=4). DP > 1 runs SGLang's data-parallel attention |
| Hybrid GDN + full attention (Qwen3.5) | Supported; bitwise parity with `VLLMEngine` (tested at TP=1 and TP=2, with chunked prefill, mixed prefill/decode steps and CUDA graphs on both engines). No radix cache |
| Multi-turn session KV (`hold_session_kv`) | Dense and MoE models. Outputs stay bitwise-identical to `VLLMEngine`; only cache hits change. Not for GDN models, which have no radix cache |

Not supported yet: KDA linear attention, Dist-MoE, sequence parallelism, prefill CUDA graphs, and per-step engine
metrics.

## Tests

- `tests/rl/unit_tests/cpu/test_sglang_engine.py`: config translation, and the sampler
  against vLLM's sampling ops.
- `tests/rl/unit_tests/gpu/test_sglang_engine_parity.py`: runs both engines on Qwen3-0.6B,
  a random-weight Qwen3 MoE and Qwen3.5-0.8B (GDN), and compares every token and
  logprob bitwise, across parallel layouts, admission orders, chunked prefill, CUDA
  graphs, KV-pool pressure (SGLang retractions) and a weight reload during generation.
  Its multi-turn cases also check that session KV holding keeps each session's history
  cached while other sessions churn the KV pool between turns.
