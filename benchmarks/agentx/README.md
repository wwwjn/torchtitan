# AgentX generator benchmark: TitanRL vs native vLLM

[AgentX](https://github.com/SemiAnalysisAI/agentx-harness) replays Claude Code sessions (multi-turn,
1-256K-token contexts) against an OpenAI chat endpoint and reports output tokens/s. This directory runs
TitanRL's generator (a TorchTitan model inside vLLM's engine) and native vLLM side by side, on the same
node, with the same vLLM build and the same settings.

## Run

You need:
- a node with Blackwell GPUs (SM100 / SM103: GB300, GB200, B200), 2 GPUs per one-GPU cell, 4 for TP2;
- ~200 GiB free host RAM;
- `git`, `uv` and internet access.

```bash
git clone -b agentx-repro https://github.com/wwwjn/torchtitan.git
bash torchtitan/benchmarks/agentx/setup.sh          # into ./agentx_work; MODEL=<dir> skips the model download
torchtitan/benchmarks/agentx/run_agentx.sh out/c32_s42 32 42   # 32 sessions, workload seed 42, 3,600 s
python torchtitan/benchmarks/agentx/summary.py out/*
```

`DURATION=600` gives a 10-minute look. The other knobs are listed at the top of `run_agentx.sh`. They
are GPU KV size, CPU KV tier, GPUs per system, TP and the session KV holder, and each applies to both
systems equally.

Paper cells: one GPU with 32 and 64 sessions; TP2 with 64 and 128 sessions; seeds 42 / 43 / 44. Compare
the titan / vllm ratio from one node.

## What runs

- **TitanRL:** branch `bench-climb-triton-gdn-sab`.
  - Base: torchtitan main.
  - Generator optimizations behind env flags: fused GDN input projection, fused MoE routing,
    shared-expert stream, fused add + RMSNorm, CPU KV tier.
  - Triton routed experts.
  - GDN generator PRs #5668-#5671, #5690.
  - FA4 swap-AB split-KV decode for the head_dim-256 attention layers.
- **FA4:** flash-attention 98c1cee, plus PR #2973 (swap-AB split-KV decode, unmerged), plus a local patch
  admitting head_dim 256, SM103 and 128-token pages, with a split heuristic targeting 1,024 CTAs at
  head_dim 256. `fa4_swapab_hd256.patch` is the whole diff on 98c1cee.
- **Native vLLM:** the same vLLM nightly (`1.0.0.dev20261018+cu132`), auto backends (TRT-LLM-gen attention
  and MoE), compile on, async scheduling, the same batch / KV / CPU-tier settings and CUDA-graph sizes.

TP2 TitanRL (`TP=2`) uses the trainer's Dist-MoE expert backend. That needs branch
`bench-climb-distmoe-rep4-gdn-sab` (`TP2=1 bash setup.sh`) and the `dist_moe` package via
`DIST_MOE_DIR`. The package is Meta-internal: ask the TitanRL team for a copy.

## Files

| File | What |
|---|---|
| `setup.sh` | Venv with the pinned torch / vLLM nightlies (shared by both systems), TorchTitan tree, patched FA4, harness venv, model |
| `run_agentx.sh` | One cell: both servers, the official harness against each for the same window, result table |
| `titan_server.py` | TitanRL's generator behind an OpenAI chat endpoint |
| `summary.py` | Result table from AIPerf's `profile_export_aiperf.json` |
| `aiperf_final_turn_shim/` | Marks each conversation's last turn for TitanRL's session KV holder (`HOLDER=1`) |
| `constraints.txt` | Every package pinned to the benchmarked env (`PIN=full bash setup.sh`) |
| `fa4_swapab_hd256.patch` | FA4 swap-AB decode, PR #2973 + head_dim-256 patch, on flash-attention 98c1cee |
