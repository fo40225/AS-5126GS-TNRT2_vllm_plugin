#!/usr/bin/env bash
# S1-10 job, runs inside serve_sd.sh (SD_OUT, SD_HARNESS, PORT, REPO, STATUS exported; the dev venv python is first on PATH).
# Env knobs (all optional):
#   S110_JOBS="facts decode reuse chat smoke"   which phases to run, in this order
#   S110_ARM, S110_PROFILE                       labels only (written into the status lines)
#   DECODE_C="1 16 64 128", DECODE_REPS=2        real-prompt greedy decode sweep (bench_sd.sh DATASET=real), no reuse
#   REUSE_ARGS="--doc-len 32768 --ws-factor 1.5 --passes 3 --clients 4"   passed to reuse_driver.py --scenario cycle
#   CHAT_ARGS="--convs 64 --turns 5 --turn-tokens 6000"                    passed to reuse_driver.py --scenario chat
#   SMOKE_LEN=65536                              smoke: one long prompt, GPU prefix cache reset, re-send (CPU hit proof)
# Writes numa_<label>.txt snapshots, facts_*.txt, decode rows (real_*), reuse_cycle.json, reuse_chat.json, metrics_*.txt.
set -u
OUT=${SD_OUT:?}; H=${SD_HARNESS:?}; PORT=${PORT:-8100}; REPO=${REPO:?}; STATUS=${STATUS:-$OUT/status.txt}
T=${S110_TOOLS:-/raid/vllm-runtime/work/s1-10/tools}
SPID=$(cat "$OUT/serve.pid")
JOBS=${S110_JOBS:-"facts decode"}
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
want() { case " $JOBS " in *" $1 "*) return 0;; esac; return 1; }
alive() { curl -sf -o /dev/null --max-time 20 "http://127.0.0.1:$PORT/health"; }
snap() { "$T/numa_snap.sh" "$OUT" "$1" "$SPID" "${2:-first}" | tee -a "$STATUS"; }
conn_metrics() {  # label
  curl -s --max-time 30 "http://127.0.0.1:$PORT/metrics" | grep -E '^vllm:(simple_kv_offload|kv_offload|external_prefix_cache|prefix_cache|num_preemptions)' \
    | grep -v -E '_bucket|_created' > "$OUT/metrics_$1.txt"
}
# layout of the native connector's region for region_nodes.py (the owner check): bytes per block-id by profile, blocks per chunk from the launch line
export S110_BPB=$([ "${S110_PROFILE:-spec}" = spec ] && echo 20480 || echo 65536)
if grep -q "OffloadingConnector" "$OUT/serve_cmd.txt" 2>/dev/null; then
  export S110_BPC=$(grep -o 'blocks_per_chunk[^0-9]*[0-9]*' "$OUT/serve_cmd.txt" | grep -o '[0-9]*$' | head -1)
else
  unset S110_BPB
fi
pool=$(grep -a -m1 -o 'GPU KV cache size: [0-9,]* tokens' "$OUT/serve.log" | grep -o '[0-9,]*' | tr -d ',')
log "S110_JOB arm=${S110_ARM:-?} profile=${S110_PROFILE:-?} jobs=[$JOBS] pool_tokens=${pool:-unknown}"

if want facts; then
  grep -a -n -E "NUMA|numactl|Binding|affinity" "$OUT/serve.log" | cut -c1-260 | head -60 > "$OUT/facts_numa_log.txt"
  grep -a -n -i -E "offload|CPU blocks|CPU tensors|mmap|host_register|KV cache layout|Allocating|connector|hybrid|group sizes|padding|Maximum concurrency|GPU KV cache size|Model loading took|init engine" \
    "$OUT/serve.log" | cut -c1-330 | head -120 > "$OUT/facts_offload_log.txt"
  : > "$OUT/facts_worker_env.txt"
  for pid in $(ps -eo pid=,sid= | awk -v s="$SPID" '$2==s{print $1}'); do
    e=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -E '^(PYTORCH_CUDA_ALLOC_CONF|VLLM_KV_CACHE_LAYOUT|_VLLM_INTERNAL_NUMACTL_ARGS|VLLM_WORKER_MULTIPROC_METHOD)=' | sort | tr '\n' ' ')
    [ -n "$e" ] && echo "pid=$pid $e" >> "$OUT/facts_worker_env.txt"
  done
  log "FACTS env of $(wc -l < "$OUT/facts_worker_env.txt") procs: expandable_segments_set=$(grep -c 'PYTORCH_CUDA_ALLOC_CONF' "$OUT/facts_worker_env.txt") layout=$(grep -h -o 'VLLM_KV_CACHE_LAYOUT=[A-Z]*' "$OUT/facts_worker_env.txt" | sort | uniq -c | tr '\n' ' ')"
  log "FACTS layout line: $(grep -a -m1 'KV cache layout' "$OUT/serve.log" | sed 's/^.*\] //')"
  log "FACTS binding: $(grep -a -c 'Binding worker subprocess' "$OUT/serve.log") worker lines, auto-detected: $(grep -a -m1 -o 'Auto-detected NUMA nodes for GPUs: .*' "$OUT/serve.log" | tr -d '\n')"
  grep -a -m16 'Binding worker subprocess\|Binding EngineCore' "$OUT/serve.log" | sed 's/^.*\] //' | while read -r l; do log "  $l"; done
  log "FACTS connector lines: $(grep -a -c -i -E 'SimpleCPUOffload|OffloadingConnector|CPUOffloadingSpec|shared_offload' "$OUT/serve.log")"
  grep -a -i -E 'SimpleCPUOffloadScheduler: Allocating|SimpleCPUOffloadWorker \[CPU\]|Created mmap file|Unlinked mmap|SimpleCPUOffloadConnector: role=' "$OUT/serve.log" | sed 's/^.*\] //' | sort | uniq -c | sort -rn | head -12 | while read -r l; do log "  $l"; done
  snap ready all
  log "NODE_MEM_READY $(grep -E '^node[0-7] ' "$OUT/numa_ready.txt" | awk '{printf "%s:%s %s %s %s | ", $1,$3,$5,$6,$11}' | cut -c1-600)"
  conn_metrics ready
  if [ "${S110_SNAP:-0}" = 1 ]; then
    wpid=$(ps -eo pid=,sid=,args= | awk -v s="$SPID" '$2==s && /Worker_TP0_EP0/ {print $1; exit}')
    log "SNAP dump of worker pid=${wpid:-none}"
    [ -n "${wpid:-}" ] && kill -USR2 "$wpid"
    sleep 20
    log "SNAP files: $(ls -la "${S17X_DIR:-$OUT/x}"/snap_*.pickle 2>&1 | awk '{print $5, $9}' | tr '\n' ' ')"
  fi
fi

if want smoke; then
  log "SMOKE start len=${SMOKE_LEN:-65536}"
  python3 -I "$T/reuse_driver.py" --port "$PORT" --model "$REPO" --out "$OUT/reuse_smoke.json" --pool-tokens "${pool:-1600000}" --scenario smoke \
    --ab-lens "${SMOKE_LEN:-65536}" > "$OUT/reuse_smoke.log" 2>&1
  log "SMOKE rc=$?"
  grep -E '^REUSE_(SEND|WAIT|RESET)' "$OUT/reuse_smoke.log" | cut -c1-420 | while read -r l; do log "$l"; done
  snap after_smoke all
  conn_metrics after_smoke
fi

if want decode; then
  alive || { log "SERVER_DEAD before decode"; exit 1; }
  # Real-prompt greedy closed loop at fixed concurrency, n = max(16, 4c) requests, every request with its own cache_salt (cycle mode of the
  # S1-9 driver): no request of any row, rep or arm can hit the GPU prefix cache or the connector tier, so the rows measure the decode
  # path (plus the stores of the eager offload mode) and never reuse. (bench_sd.sh resets only the GPU prefix cache between rows, which
  # a CPU tier survives: its repeated prompts would hit the tier.)
  CD=${S110_CONC_DRIVER:-$T/conc_driver_s110.py}
  for rep in $(seq 1 "${DECODE_REPS:-2}"); do
    for c in ${DECODE_C:-1 16 64 128}; do
      n=$(( 4 * c )); [ "$n" -lt 16 ] && n=16
      name="real_r${rep}_c${c}"
      python3 -I "$H/spec_scrape.py" snap --port "$PORT" --out "$OUT/$name.spec_before.json" > /dev/null 2>&1
      timeout 900 python3 -I "$CD" --set gsm80,mtbench --mode greedy --concurrency "$c" --num-prompts "$n" --max-tokens 128 --effort xhigh \
        --cycle --salt-prefix "s110${S110_ARM:-x}r${rep}c${c}x" --port "$PORT" --model "$REPO" --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
      rc=$?
      spec=$(python3 -I "$H/spec_scrape.py" delta --port "$PORT" --before "$OUT/$name.spec_before.json" --out "$OUT/$name.spec.json" 2>/dev/null)
      log "DEC $name c=$c n=$n rc=$rc $(python3 -I -c "
import json
d=json.load(open('$OUT/$name.json'))
print('completed=%d failed=%d out_tps=%.1f tpot=%.2f ttft_p50=%.1f' % (d['completed'], d['failed'], d['output_throughput'], d['mean_tpot_ms'], d['median_ttft_ms']))" 2>&1 | tail -1) $spec"
      alive || { log "SERVER_DEAD after $name"; exit 1; }
    done
  done
  conn_metrics after_decode
  snap after_decode all
fi

if want reuse; then
  alive || { log "SERVER_DEAD before reuse"; exit 1; }
  log "REUSE start pool=${pool:-0} args=[${REUSE_ARGS:-}]"
  # shellcheck disable=SC2086
  timeout "${REUSE_TIMEOUT:-7200}" python3 -I "$T/reuse_driver.py" --port "$PORT" --model "$REPO" --out "$OUT/reuse_cycle.json" \
    --pool-tokens "${pool:-1600000}" --scenario cycle ${REUSE_ARGS:-} > "$OUT/reuse_cycle.log" 2>&1
  log "REUSE rc=$?"
  grep -E '^REUSE_(ROW|DONE|CYCLE|WAIT)' "$OUT/reuse_cycle.log" | cut -c1-300 | while read -r l; do log "$l"; done
  conn_metrics after_reuse
  snap after_reuse all
fi

if want chat; then
  alive || { log "SERVER_DEAD before chat"; exit 1; }
  log "CHAT start pool=${pool:-0} args=[${CHAT_ARGS:-}]"
  # shellcheck disable=SC2086
  timeout "${CHAT_TIMEOUT:-7200}" python3 -I "$T/reuse_driver.py" --port "$PORT" --model "$REPO" --out "$OUT/reuse_chat.json" \
    --pool-tokens "${pool:-1600000}" --scenario chat ${CHAT_ARGS:-} > "$OUT/reuse_chat.log" 2>&1
  log "CHAT rc=$?"
  grep -E '^REUSE_(ROW|DONE)' "$OUT/reuse_chat.log" | cut -c1-300 | while read -r l; do log "$l"; done
  conn_metrics after_chat
  snap end all
fi
log "S110_JOB_DONE"
