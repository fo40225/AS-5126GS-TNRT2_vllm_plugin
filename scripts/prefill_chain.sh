#!/bin/bash
# Prefill latency of a list of server configurations in ONE lock hold (the GPU lock must be held by the caller):
#   flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock bash scripts/prefill_chain.sh <configs.txt>
# Every line of the configs file is "name|ENV=value ENV=value ..."; # starts a comment. For each line a server is started with
# these variables (dummy weights unless LOAD_FORMAT=auto; host folder on PYTHONPATH, env/host.env sourced; SRC = vLLM tree),
# bench/prefill_probe.py measures single-request prefill latencies (env SIZES, REPS), the server is stopped. One line per
# configuration in $WORK/prefill_chain/<configs>/summary.txt, server logs next to it.
# Example (2026-10-24): "base|VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP=0" / "overlap|VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP=1".
HOST=${HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK=${WORK:-/raid/vllm-runtime/work/host}
CONF=$1; CHAIN=$(basename "$CONF" .txt)
OUTD=$WORK/prefill_chain/$CHAIN; mkdir -p "$OUTD"
DEV=/raid/vllm-runtime/sm120-dev
wait_free() {
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "${u:-0}" -lt 2000 ] && return 0; sleep 5
  done; return 1
}
wait_free || { echo "ABORT gpus busy" >> "$OUTD/summary.txt"; exit 2; }
while IFS='|' read -r name envs; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  wait_free || { echo "ABORT gpus busy before $name" >> "$OUTD/summary.txt"; break; }
  (
    source $DEV/.venv/bin/activate
    export PYTHONPATH=$HOST:${SRC:-$DEV/src}
    export FLASHINFER_WORKSPACE_BASE=$DEV/fi_workspace TRITON_CACHE_DIR=$DEV/triton_cache
    export CUTE_DSL_CACHE_DIR=$DEV/cute_cache VLLM_CACHE_ROOT=$DEV/vllm_cache TORCHINDUCTOR_CACHE_DIR=$DEV/inductor_cache
    export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
    export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=7200
    source $HOST/env/host.env
    for kv in $envs; do export "$kv"; done
    LF=(); [ "${LOAD_FORMAT:-dummy}" = dummy ] && LF=(--load-format dummy)
    cd "$OUTD"
    setsid vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 "${LF[@]}" --tensor-parallel-size 8 --enable-expert-parallel \
      --max-model-len 131072 --gpu-memory-utilization 0.95 --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 \
      --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --host 127.0.0.1 --port 8100 > "$OUTD/$name.serve.log" 2>&1 < /dev/null &
    SPID=$!
    t0=$(date +%s); state=LOADING
    while :; do
      curl -sf -o /dev/null http://127.0.0.1:8100/health && { state=READY; break; }
      kill -0 $SPID 2>/dev/null || { state=LOAD_FAIL; break; }
      [ $(( $(date +%s) - t0 )) -gt 1800 ] && { state=LOAD_TIMEOUT; break; }
      sleep 5
    done
    if [ $state = READY ]; then
      python3 "$HOST/bench/prefill_probe.py" 8100 "$name" ${REPS:-3} 2>&1 | tee -a "$OUTD/summary.txt"
    else
      echo "TUNE $name $state" | tee -a "$OUTD/summary.txt"
      tail -30 "$OUTD/$name.serve.log" > "$OUTD/$name.tail.txt"
    fi
    kill -TERM -- -$SPID 2>/dev/null
    for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
    kill -KILL -- -$SPID 2>/dev/null
  )
done < "$CONF"
wait_free
touch "$OUTD/DONE"
