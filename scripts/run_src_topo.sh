#!/usr/bin/env bash
# Copy of /raid/vllm-runtime/musetest/run_src.sh (the original stays untouched) that also puts the host folder on
# PYTHONPATH (the muse_topo vLLM plugin is found through its dist-info there) and exports env/host.env.
# Dummy-weight load test + pp/tg speed for one model, using the sm120-dev venv with the
# branch source tree /raid/vllm-runtime/sm120-dev/src on PYTHONPATH.
# Usage: run_src_topo.sh <hf_repo> <tag> [extra vllm serve args...]
# The host folder is the one this script lives in (so <mirror>/scripts/run_src_topo.sh uses <mirror>); HOST_DIR overrides.
# MUSE_TOPO_ENABLE=0 gives the NCCL-only baseline through the same launch path.
# Env: SRC (source tree, default $DEV/src), PORT (8100), MAXLEN (131072), GPU_UTIL (0.90), READY_TIMEOUT_S (5400), SKIP_CTX (0), TP (8)
#      LOAD_FORMAT (default "dummy" = random weights; "auto" loads the real checkpoint from the HF cache, offline),
#      CHAT_CHECK (0; 1 = greedy chat sanity requests via bench/chat_check.py right after READY, result in $OUT/chat.json)
# Each bench uses a distinct --seed so random prompts never share prefixes.
# Method mirrors /home/user/trivial/benchmark_results.md:
#   pure speed: concurrency 1, random 512 in / 128 out, 16 prompts, --ignore-eos
#     pp t/s = 512 / mean TTFT (s), tg t/s = 1000 / mean TPOT (ms)
#   long context: concurrency 1, output 256, 3 prompts, input = MAXLEN x {10,50,90}% (cap MAXLEN-512)
set -u
HOST=${HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=$1; TAG=$2; shift 2; EXTRA=("$@")
BASE=/raid/vllm-runtime
OUT=$BASE/musetest/results/$TAG
SEED=${SEED:-1000}
PORT=${PORT:-8100}
MAXLEN=${MAXLEN:-131072}
GPU_UTIL=${GPU_UTIL:-0.90}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-5400}
SKIP_CTX=${SKIP_CTX:-0}
mkdir -p "$OUT"
STATUS="$OUT/status.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

DEV=$BASE/sm120-dev
source "$DEV/.venv/bin/activate"
export PYTHONPATH=$HOST:${SRC:-$DEV/src}
export FLASHINFER_WORKSPACE_BASE=$DEV/fi_workspace TRITON_CACHE_DIR=$DEV/triton_cache
export CUTE_DSL_CACHE_DIR=$DEV/cute_cache VLLM_CACHE_ROOT=$DEV/vllm_cache TORCHINDUCTOR_CACHE_DIR=$DEV/inductor_cache
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=7200
# Host env: NCCL_P2P_LEVEL=SYS (bitwise identical to the default, see env/host.env), MUSE_TOPO_ENABLE=1, opt-ins.
source "$HOST/env/host.env"

# Refuse to start if any GPU already holds memory (another job owns the GPUs).
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "${used:-0}" -gt 2000 ]; then log "ABORT gpus_busy used_mib=$used"; exit 2; fi

LOAD_FORMAT=${LOAD_FORMAT:-dummy}
if [ "$LOAD_FORMAT" = auto ]; then LF=(); else LF=(--load-format "$LOAD_FORMAT"); fi
SERVE_ARGS=(serve "$REPO" "${LF[@]}" --tensor-parallel-size "${TP:-8}"
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL"
  --host 127.0.0.1 --port "$PORT" "${EXTRA[@]}")
printf '%q ' vllm "${SERVE_ARGS[@]}" > "$OUT/serve_cmd.txt"; echo >> "$OUT/serve_cmd.txt"
log "START $REPO tag=$TAG maxlen=$MAXLEN load_format=$LOAD_FORMAT extra=${EXTRA[*]:-none}"
log "host=$HOST MUSE_TOPO_ENABLE=$MUSE_TOPO_ENABLE NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL SRC=${SRC:-$DEV/src} large_bf16_exchange=${MUSE_TOPO_LARGE_BF16_EXCHANGE:-0}"
t0=$(date +%s)
# Own session so the whole server process tree can be stopped as one group.
setsid vllm "${SERVE_ARGS[@]}" > "$OUT/serve.log" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$OUT/serve.pid"

stop_server() {
  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SPID" 2>/dev/null
  # Wait until the GPUs are free again (a killed worker frees memory 1-3 min later).
  for _ in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "${u:-0}" -lt 2000 ] && break; sleep 5
  done
  log "STOPPED gpu_used_mib=${u:-?}"
}

state=LOADING
while :; do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then state=READY; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then state=LOAD_FAIL; break; fi
  if [ $(( $(date +%s) - t0 )) -gt "$READY_TIMEOUT_S" ]; then state=LOAD_TIMEOUT; break; fi
  sleep 10
done
load_s=$(( $(date +%s) - t0 ))
log "$state load_s=$load_s"
if [ "$state" != READY ]; then
  tail -60 "$OUT/serve.log" > "$OUT/serve_tail.txt"
  stop_server; exit 1
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/gpu_mem_ready.txt"

if [ "${CHAT_CHECK:-0}" = 1 ]; then
  timeout 3600 python3 "$HOST/bench/chat_check.py" --port "$PORT" --model "$REPO" --out "$OUT/chat.json" > "$OUT/chat.log" 2>&1
  log "CHAT rc=$? $(grep CHAT_SUMMARY "$OUT/chat.log")"
fi

bench() {  # name in out prompts
  local name=$1 in=$2 outl=$3 n=$4
  timeout 3600 vllm bench serve --backend openai --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/completions --model "$REPO" --dataset-name random \
    --random-input-len "$in" --random-output-len "$outl" --num-prompts "$n" \
    --max-concurrency 1 --ignore-eos --seed "$SEED" --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT" --result-filename "$name.json" \
    > "$OUT/$name.log" 2>&1
  log "BENCH $name in=$in out=$outl n=$n seed=$SEED rc=$?"
  SEED=$((SEED + 1))
}

bench c1_512_128 512 128 16
if [ "$SKIP_CTX" != 1 ]; then
  for pct in 10 50 90; do
    in=$(( MAXLEN * pct / 100 )); cap=$(( MAXLEN - 512 )); [ "$in" -gt "$cap" ] && in=$cap
    bench "ctx${pct}" "$in" 256 3
  done
fi
stop_server
log "DONE $REPO"
