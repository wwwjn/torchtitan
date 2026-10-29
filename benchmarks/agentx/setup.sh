#!/usr/bin/env bash
# One-time setup for the AgentX generator benchmark (TitanRL vs native vLLM) on a Blackwell node
# (GB300 / GB200 / B200: SM100 or SM103). Needs git, uv and internet access (GitHub, PyPI,
# download.pytorch.org, Hugging Face). Everything lands under $WORK [./agentx_work]:
#
#   env/                 Python 3.12 venv shared by TitanRL and native vLLM (one torch / vLLM build)
#   torchtitan/          TitanRL tree for one GPU (EP = 1)
#   torchtitan-tp2/      TitanRL tree for TP2 / EP2 Dist-MoE (only with TP2=1)
#   fa4/flash_attn/cute  FA4 with the swap-AB split-KV decode kernel (PR #2973 + hd256 / SM103 patch)
#   agentx-venv/         the official harness (agentx-harness 1.0.12, AIPerf)
#   Qwen3.5-35B-A3B/     model weights (skipped when MODEL points at an existing copy)
#
#   bash setup.sh                       # one-GPU rows
#   TP2=1 bash setup.sh                 # also the TP2 tree (Dist-MoE itself: see README)
#   PIN=full bash setup.sh              # pin every package to the paper env, not just torch / vLLM
#   PIN=none bash setup.sh              # latest nightlies (both systems still share one build)
# Re-running skips steps whose output exists.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(realpath -m "${WORK:-$PWD/agentx_work}")
TT_REPO=${TT_REPO:-https://github.com/wwwjn/torchtitan.git}
mkdir -p "$WORK"
cd "$WORK"

echo "== TorchTitan trees"
[[ -d torchtitan ]] || git clone --single-branch -b bench-climb-triton-gdn-sab "$TT_REPO" torchtitan
if [[ ${TP2:-0} == 1 && ! -d torchtitan-tp2 ]]; then
  git clone --single-branch -b bench-climb-distmoe-rep4-gdn-sab "$TT_REPO" torchtitan-tp2
fi

echo "== Python env (torch / vLLM nightlies of 2026-10-18, cu132)"
if [[ ! -d env ]]; then
  pins=()
  case ${PIN:-core} in
    core) printf '%s\n' torch==2.16.0.dev20261018+cu132 torchvision==0.30.0.dev20261018+cu132 \
            vllm==1.0.0.dev20261018+cu132 flashinfer-python==0.6.18.post1 >pins.txt
          pins=(--constraint "$WORK/pins.txt") ;;
    full) pins=(--constraint "$HERE/constraints.txt") ;;
    none) ;;
    *) echo "PIN must be core, full or none" >&2; exit 2 ;;
  esac
  uv venv env --python 3.12
  # TorchTitan's RL installer: requirements, torch / torchvision / vLLM nightlies, FA4 wheel, nvcc
  # for FlashInfer's JIT (--flash-attn 4: also when this runs on a node without GPUs).
  (source env/bin/activate && cd torchtitan && bash scripts/install_rl_env.sh --cuda cu132 --nvcc --flash-attn 4 "${pins[@]}")
  # The CuTe DSL stack the patched FA4 source was run with (the installer takes the newest FA4 wheel).
  [[ ${PIN:-core} == none ]] || VIRTUAL_ENV=$WORK/env uv pip install "flash-attn-4[cu13]==4.0.0b35" \
    nvidia-cutlass-dsl==4.7.1 quack-kernels==0.6.5 apache-tvm-ffi==0.1.11
fi
# vLLM turns on TRT-LLM-gen attention (native vLLM at its fastest) only when it sees FlashInfer's
# cubin package, published on FlashInfer's index (not PyPI) at the flashinfer-python version.
if ! env/bin/python -c 'import flashinfer_cubin' 2>/dev/null; then
  VIRTUAL_ENV=$WORK/env uv pip install --extra-index-url https://flashinfer.ai/whl/ --index-strategy unsafe-best-match \
    "flashinfer-cubin==$(env/bin/python -c 'from importlib.metadata import version; print(version("flashinfer-python"))')"
fi

echo "== FA4 swap-AB decode (flash-attention main 98c1cee + PR #2973 + local hd256 / SM103 / 128-token-page patch)"
if [[ ! -d fa4/flash_attn/cute ]]; then
  [[ -d fa4-src ]] || git clone https://github.com/Dao-AILab/flash-attention.git fa4-src
  git -C fa4-src checkout -q 98c1cee0ed59307833841cdbf567fe336ceec13a
  git -C fa4-src apply "$HERE/fa4_swapab_hd256.patch"
  # Only flash_attn/cute goes on TitanRL's PYTHONPATH: the full flash_attn package imports FA2's CUDA
  # extension, which is not built here.
  mkdir -p fa4/flash_attn
  cp -r fa4-src/flash_attn/cute fa4/flash_attn/
fi

echo "== Official AgentX harness"
if [[ ! -d agentx-venv ]]; then
  uv venv agentx-venv --python 3.12
  VIRTUAL_ENV=$WORK/agentx-venv uv pip install agentx-harness==1.0.12
fi

echo "== Model"
if [[ -z ${MODEL:-} && ! -f Qwen3.5-35B-A3B/config.json ]]; then
  env/bin/hf download Qwen/Qwen3.5-35B-A3B --local-dir Qwen3.5-35B-A3B
fi

echo "== Import check"
# The vLLM nightly's _C_stable_libtorch does not link cuBLAS: preload it (run_agentx.sh does too).
cu=$(env/bin/python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13/lib
env LD_PRELOAD="$cu/libcublas.so.13:$cu/libcublasLt.so.13" PYTHONPATH="$WORK/torchtitan:$WORK/fa4" env/bin/python -c '
import torch, vllm
from flash_attn.cute import swapab_dispatch
from torchtitan.models import qwen3_5
print("torch", torch.__version__, "| vllm", vllm.__version__, "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
echo "setup done: $WORK"
