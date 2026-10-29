#!/usr/bin/env bash
# AgentX generator benchmark on one node: TitanRL and native vLLM at the same time on separate GPUs,
# each driven by the official harness for the same window, then a side-by-side table. No scheduler
# needed: run it on a bare node or inside any allocation that has the GPUs (2 at TP=1, 4 at TP=2).
#
#   run_agentx.sh <out_dir> [sessions] [seed]          # defaults: 32 sessions, seed 42
#
# Paper cells: TP=1 with 32 and 64 sessions, TP=2 with 64 and 128, seeds 42 / 43 / 44, 3,600 s.
# Env (defaults follow setup.sh's layout):
#   WORK         setup.sh's directory [./agentx_work]
#   MODEL        Qwen3.5-35B-A3B HF directory [$WORK/Qwen3.5-35B-A3B]
#   TP           1 or 2 [1]; TP=2: TitanRL attention TP2 + Dist-MoE EP2 (needs DIST_MOE_DIR),
#                native vLLM TP2 + --enable-expert-parallel
#   TT_DIR       TorchTitan tree [$WORK/torchtitan; $WORK/torchtitan-tp2 at TP=2]
#   FA4_DIR      directory holding the patched flash_attn/cute [$WORK/fa4]
#   SYSTEMS      servers to run ["titan vllm"]
#   GPUS_TITAN / GPUS_VLLM   [0 / 1 at TP=1; 0,1 / 2,3 at TP=2]
#   KV_GIB       GPU KV cache per server, the same on both sides [180 on GB300; ~100 on 180 GB GPUs]
#   CPU_KV_GIB   CPU KV tier per server, both sides [100; needs that much free host RAM per server]
#   DURATION     measurement window, s [3600; 600 for a quick look]
#   HOLDER=1     TitanRL keeps a finished turn's KV until the harness marks the conversation's last
#                turn (the "+ holder" rows; native vLLM has no such policy)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "${1:?usage: run_agentx.sh <out_dir> [sessions] [seed]}")
SESSIONS=${2:-32} SEED=${3:-42}
WORK=$(realpath -m "${WORK:-$PWD/agentx_work}")
: "${MODEL:=$WORK/Qwen3.5-35B-A3B}" "${TP:=1}" "${FA4_DIR:=$WORK/fa4}" "${SYSTEMS:=titan vllm}"
: "${KV_GIB:=180}" "${CPU_KV_GIB:=100}" "${DURATION:=3600}" "${HOLDER:=0}"
if [[ $TP == 1 ]]; then
  : "${TT_DIR:=$WORK/torchtitan}" "${GPUS_TITAN:=0}" "${GPUS_VLLM:=1}"
else
  : "${TT_DIR:=$WORK/torchtitan-tp2}" "${GPUS_TITAN:=0,1}" "${GPUS_VLLM:=2,3}"
  [[ $SYSTEMS != *titan* ]] || : "${DIST_MOE_DIR:?TP=2 TitanRL needs DIST_MOE_DIR (the dist_moe package, see README)}"
fi
mkdir -p "$OUT/logs"

source "$WORK/env/bin/activate"
# No system CUDA toolkit needed: FlashInfer's JIT uses the pip nvcc. The vLLM nightly's
# _C_stable_libtorch does not link cuBLAS, so preload it.
cu=$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13
if [[ -d $cu ]]; then
  export CUDA_HOME=${CUDA_HOME:-$cu} PATH=$cu/bin:$PATH
  export LD_PRELOAD=$cu/lib/libcublas.so.13:$cu/lib/libcublasLt.so.13
fi
export PYTHONUNBUFFERED=1 VLLM_SERVER_DEV_MODE=1
kv_bytes=$((KV_GIB * 1024 * 1024 * 1024))
max_len=262144 batched=16384
pids=()
killtree() { local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do killtree "$c"; done; kill "$1" 2>/dev/null; }
trap 'for p in "${pids[@]}"; do killtree "$p"; done' EXIT
declare -A URL

start_titan() {  # TitanRL's generator (TorchTitan model in vLLM's engine) behind an OpenAI chat endpoint
  local port=8100
  # Chat prompts render in vLLM's GPU-less render server, a separate process as in native vLLM.
  CUDA_VISIBLE_DEVICES= vllm launch render "$MODEL" --host 127.0.0.1 --port $((port + 50)) \
    --served-model-name model --max-model-len $max_len --language-model-only >"$OUT/logs/titan_render.log" 2>&1 &
  pids+=($!)
  local moe=(--override torchtitan_recipes.overrides.triton_routed_experts.triton_routed_experts) moe_env=()
  [[ $TP == 2 ]] && moe=(--dist-moe --ep 2) moe_env=(TORCHTITAN_MOE_TP_SPLIT_VIEWS=1)
  local holder=() holder_env=(SESSION_KV_HOLDER=0)
  [[ $HOLDER == 1 ]] && holder=(--session-kv-holder --release-on-final-turn) holder_env=(SESSION_KV_HOLDER=1)
  env PYTHONHASHSEED=0 PYTORCH_ALLOC_CONF=expandable_segments:False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    TORCHTITAN_FUSED_GDN_PROJ=1 TORCHTITAN_FUSED_ROUTING=1 TORCHTITAN_MOE_SHARED_EXPERT_STREAM=1 \
    TORCHTITAN_FUSED_AG_COMBINE=1 TORCHTITAN_KV_OFFLOAD_PROMPT_ONLY=1 TORCHTITAN_FUSED_ADD_RMSNORM=1 \
    TORCHTITAN_SCHEDULING_POLICY=fcfs TORCHTITAN_FA4_SWAPAB_DECODE=1 TORCHTITAN_FA4_SWAPAB_SPLIT_MIXED=1 \
    FA4_DIR="$FA4_DIR" "${moe_env[@]}" "${holder_env[@]}" \
    PYTHONPATH="$TT_DIR:$FA4_DIR${DIST_MOE_DIR:+:$DIST_MOE_DIR}" CUDA_VISIBLE_DEVICES=$GPUS_TITAN \
    torchrun --nproc_per_node="$TP" --master_port 29500 "$HERE/titan_server.py" --engine vllm \
      --model-path "$MODEL" --port $port --tp "$TP" --model-module torchtitan.models.qwen3_5 --flavor 35B-A3B \
      --max-model-len $max_len --max-num-seqs 256 --max-num-batched-tokens $batched --gpu-memory-limit 0.9 \
      --weight-sync-buffers --output-dir "$OUT/titan_server" --chat-render-url http://127.0.0.1:$((port + 50)) \
      --vllm-cuda-graph-mode FULL --kv-offload-gib "$CPU_KV_GIB" --max-engine-steps 1 "${holder[@]}" "${moe[@]}" \
      --extra-engine-args "{\"kv_cache_memory_bytes\":$kv_bytes}" >"$OUT/logs/titan.log" 2>&1 &
  pids+=($!)
  URL[titan]=http://127.0.0.1:$port
}

start_vllm() {  # native vLLM at its fastest: auto backends (TRT-LLM-gen attention + MoE), compile on
  local port=8200 ladder ep=()
  # TitanRL's CUDA-graph capture sizes, up to the 16,384-token batch.
  ladder="[1,2,4$(printf ',%d' $(seq 8 8 256) $(seq 272 16 1024) $(seq 1536 512 4096) $(seq 6144 2048 16384))]"
  [[ $TP == 2 ]] && ep=(--enable-expert-parallel)
  env VLLM_HAS_FLASHINFER_CUBIN=1 CUDA_VISIBLE_DEVICES=$GPUS_VLLM \
    vllm serve "$MODEL" --served-model-name model --host 127.0.0.1 --port $port --seed 0 \
      --tensor-parallel-size "$TP" "${ep[@]}" --max-model-len $max_len --max-num-seqs 256 \
      --max-num-batched-tokens $batched --gpu-memory-utilization 0.92 --kv-cache-memory-bytes $kv_bytes \
      --kv-offloading-size "$CPU_KV_GIB" --enable-prefix-caching --prefix-cache-retention-interval None \
      --enable-prompt-tokens-details --async-scheduling --language-model-only \
      --compilation-config "{\"max_cudagraph_capture_size\":16384,\"cudagraph_capture_sizes\":$ladder}" \
      >"$OUT/logs/vllm.log" 2>&1 &
  pids+=($!)
  URL[vllm]=http://127.0.0.1:$port
}

for s in $SYSTEMS; do "start_$s"; done
deadline=$((SECONDS + 2400))  # model load + compile + CUDA-graph capture: ~10 min each
for s in $SYSTEMS; do
  until curl -sf -m 3 -o /dev/null "${URL[$s]}/v1/models"; do
    (( SECONDS < deadline )) || { echo "$s did not come up"; tail -30 "$OUT/logs/$s.log"; exit 1; }
    sleep 10
  done
  echo "$(date +%T) $s ready at ${URL[$s]}"
done

# Both systems get the same harness run at the same time (same workload sample: --random-seed).
seed_arg=(); [[ $SEED == 42 ]] || seed_arg=(--random-seed "$SEED")  # 42 is the scenario default
shim=(-u PYTHONPATH); [[ $HOLDER == 1 ]] && shim=(PYTHONPATH="$HERE/aiperf_final_turn_shim")
cpids=()
for s in $SYSTEMS; do
  env -u LD_PRELOAD "${shim[@]}" "$WORK/agentx-venv/bin/agentx" profile --url "${URL[$s]}" --model model \
    --tokenizer "$MODEL" --output-artifact-dir "$OUT/$s" \
    --public-dataset semianalysis_cc_traces_weka_062126_256k --concurrency "$SESSIONS" \
    --benchmark-duration "$DURATION" "${seed_arg[@]}" >"$OUT/$s.stdout" 2>&1 &
  cpids+=($!)
done
echo "$(date +%T) harness running: $SESSIONS sessions, seed $SEED, ${DURATION}s"
wait "${cpids[@]}"
python "$HERE/summary.py" "$OUT" | tee "$OUT/summary.md"
