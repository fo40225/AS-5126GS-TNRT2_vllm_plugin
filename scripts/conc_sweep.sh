#!/usr/bin/env bash
# Concurrency sweep with real weights, one server session per tag, random 512/128 prompts (copy of the r2 sweep script
# /raid/vllm-runtime/work/r2-measure/b/sweep_b.sh, reduced to one shape and parametrized). The caller holds the GPU lock.
# usage: [HOST_DIR=<host folder>] [SRC=<vllm tree>] [CONCS="1 4 16 64 128"] [REPEATS=2] [SEED=9000] conc_sweep.sh <tag>
#   Variant switches are passed through the environment (e.g. MUSE_TOPO_MEDIUM_BX=0). Results: musetest/results/<tag>/.
set -u
TAG=${1:?tag}
REPO=unsloth/Muse-Spark-Open-816B-A42B-NVFP4
HOST=${HOST_DIR:-/raid/vllm-runtime/host_topo}
BASE=/raid/vllm-runtime
R=$BASE/musetest/results
OUT=$R/$TAG
SEED=${SEED:-9000}
PORT=${PORT:-8100}
MAXLEN=131072
CONCS=${CONCS:-"1 4 16 64 128"}
REPEATS=${REPEATS:-2}
IN=${IN:-512}
OUTL=${OUTL:-128}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/status.txt" > /dev/null; }

DEV=$BASE/sm120-dev
source "$DEV/.venv/bin/activate"
export PYTHONPATH=$HOST:${SRC:-$DEV/src}
export FLASHINFER_WORKSPACE_BASE=$DEV/fi_workspace TRITON_CACHE_DIR=$DEV/triton_cache
export CUTE_DSL_CACHE_DIR=$DEV/cute_cache VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$DEV/vllm_cache} TORCHINDUCTOR_CACHE_DIR=$DEV/inductor_cache
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=7200
export HOST_DIR=$HOST
source "$HOST/env/host.env"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "${used:-0}" -gt 2000 ]; then log "ABORT gpus_busy used_mib=$used"; exit 2; fi

SERVE_ARGS=(serve "$REPO" --tensor-parallel-size 8
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.95
  --host 127.0.0.1 --port "$PORT"
  --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}')
printf '%q ' vllm "${SERVE_ARGS[@]}" > "$OUT/serve_cmd.txt"; echo >> "$OUT/serve_cmd.txt"
log "START $REPO tag=$TAG maxlen=$MAXLEN load_format=auto seed_base=$SEED concs=($CONCS) repeats=$REPEATS in=$IN out=$OUTL"
log "host=$HOST SRC=${SRC:-$DEV/src} flags $(env | grep -E '^(VLLM_MUSE_SPARK|MUSE_TOPO|NCCL_)' | sort | tr '\n' ' ')"
t0=$(date +%s)
setsid vllm "${SERVE_ARGS[@]}" > "$OUT/serve.log" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$OUT/serve.pid"

stopped=0
stop_server() {
  [ "$stopped" = 1 ] && return; stopped=1
  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "${u:-0}" -lt 2000 ] && break; sleep 5
  done
  log "STOPPED gpu_used_mib=${u:-?}"
}
trap 'log "SIGNAL"; stop_server; exit 130' INT TERM

state=LOADING
while :; do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then state=READY; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then state=LOAD_FAIL; break; fi
  if [ $(( $(date +%s) - t0 )) -gt 5400 ]; then state=LOAD_TIMEOUT; break; fi
  sleep 10
done
log "$state load_s=$(( $(date +%s) - t0 ))"
if [ "$state" != READY ]; then
  tail -60 "$OUT/serve.log" > "$OUT/serve_tail.txt"
  stop_server; exit 1
fi
log "TOPO_PLUGIN registered_lines=$(grep -c 'muse_topo plugin registered' "$OUT/serve.log")"

bench() {  # name in out concurrency prompts seed
  local name=$1 in=$2 outl=$3 c=$4 n=$5 seed=$6
  timeout 3600 vllm bench serve --backend openai --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/completions --model "$REPO" --dataset-name random \
    --random-input-len "$in" --random-output-len "$outl" --num-prompts "$n" \
    --max-concurrency "$c" --ignore-eos --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$OUT" --result-filename "$name.json" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  local chk
  chk=$(python3 -I - "$OUT/$name.json" "$n" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ok = d["completed"] == int(sys.argv[2]) and d["failed"] == 0
    print("completed=%d failed=%d out_tps=%.1f tot_tps=%.1f ttft=%.1f tpot=%.2f %s" % (
        d["completed"], d["failed"], d["output_throughput"], d["total_token_throughput"],
        d["mean_ttft_ms"], d["mean_tpot_ms"], "OK" if ok else "BAD"))
except Exception as e:
    print("NORESULT %r" % (e,))
PY
)
  log "BENCH $name in=$in out=$outl c=$c n=$n seed=$seed rc=$rc $chk"
  case "$chk" in
    *" OK") ;;
    *) if [ "$rc" != 0 ] || ! curl -sf -o /dev/null --max-time 20 "http://127.0.0.1:$PORT/health"; then
         log "SERVER_DEAD after $name"; tail -60 "$OUT/serve.log" > "$OUT/serve_tail.txt"
         stop_server; exit 1
       fi ;;
  esac
}

bench warmup 1024 32 32 64 $((SEED - 1))
i=0
for rep in $(seq 1 "$REPEATS"); do
  for c in $CONCS; do
    n=$(( 4 * c )); [ "$n" -lt 16 ] && n=16
    bench "s${IN}_${OUTL}_c${c}_r${rep}" "$IN" "$OUTL" "$c" "$n" $(( SEED + i ))
    i=$(( i + 1 ))
  done
done
stop_server
log "DONE $REPO tag=$TAG"
