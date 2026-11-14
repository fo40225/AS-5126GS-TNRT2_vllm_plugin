#!/usr/bin/env bash
# S1-10 gate: determinism hold of the KV CPU offload (e2e_s110.py x 4 cold processes: ON a, OFF a, ON b, OFF b), one profile per call, inside ONE GPU-lock hold:
#   setsid nohup flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock env GATE_TOOLS=<dir> bash master_det.sh spec|ns256 > det.out 2>&1 < /dev/null &
# Result: $W/det_<profile>.txt = PASS | FAIL | ARTIFACT (an OOM, a missing pass file or a missing connector / binding evidence is an ARTIFACT, retried once; only a
# comparison mismatch is a FAIL), $W/det_out/<tag>.* dumps, logs in $W/det_logs/.
# ON = gate tree + gate host mirror, env/offload.env sourced: SimpleCPUOffloadConnector + numa_bind=True + the exact tier bytes (45 GiB per rank), the offload-on exact KV bytes;
# OFF = the same tree and host, MUSE_OFFLOAD=0: the S1-9 bytes. Profile spec: RadixArk DFlash2 K=7 as the production line (host spec_k7.env), mns 128, MNBT 8199, cap 256, max-model-len 32768;
# profile ns256: no speculation, mns 256, MNBT 8192, cap 256, host.env only.
set -u
PROFILE=${1:?usage: master_det.sh spec|ns256}
B=/raid/vllm-runtime/sm120-dev
W=/raid/vllm-runtime/work/s1-10
T=${GATE_TOOLS:-$W/gate}
TREE=${TREE_G:-$B/src_s110g}
HOST=${HOST_G:-/raid/vllm-runtime/host_topo_s110}
OUT=$W/det_out; LOGS=$W/det_logs
mkdir -p "$OUT" "$LOGS"
LOG=$W/master_det.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
CLEAN_ENV="VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS MUSE_SPEC_SERVE_ARGS MUSE_SPEC_KV_BYTES MUSE_SPEC_MNBT MUSE_SPEC_GRAPH_SP_MIN_TOKENS MUSE_SPEC_GRAPH_CAP MUSE_SPEC_KV_BYTES_OVERRIDE VLLM_MUSE_SPARK_SM120_DRAFT_FP8 VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD VLLM_KV_CACHE_LAYOUT PYTORCH_CUDA_ALLOC_CONF MUSE_TOPO_LOG_OPS MUSE_OFFLOAD MUSE_OFFLOAD_NUMA_BIND MUSE_OFFLOAD_PROFILE MUSE_OFFLOAD_PER_RANK_BYTES MUSE_OFFLOAD_BPB MUSE_OFFLOAD_KV_BYTES MUSE_OFFLOAD_KV_BYTES_OVERRIDE MUSE_OFFLOAD_SERVE_ARGS MUSE_OFFLOAD_ERROR _MUSE_OFFLOAD_SPEC_OVERRIDE_SET"
# shellcheck disable=SC2086
unset $CLEAN_ENV

gpus_free() { local i u; for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'); [ "${u:-0}" -lt 2000 ] && return 0; sleep 5; done; return 1; }

# e2e_run <tag> <on|off>
e2e_run() {
  local tag=$1 arm=$2
  gpus_free || log "WARN gpus busy before e2e $tag"
  log "E2E START $tag arm=$arm profile=$PROFILE"
  rm -f "$OUT/$tag".*.json
  (
    # shellcheck disable=SC2086
    unset $CLEAN_ENV
    if [ "$arm" = on ]; then export MUSE_OFFLOAD=1; else export MUSE_OFFLOAD=0; fi
    export MUSE_OFFLOAD_PROFILE=$PROFILE
    # shellcheck disable=SC1091
    source "$HOST/env/offload.env" || { echo "offload.env failed: ${MUSE_OFFLOAD_ERROR:-?}"; exit 3; }
    if [ "$PROFILE" = spec ]; then
      # shellcheck disable=SC1091
      source "$HOST/env/spec_k7.env"
      kv=$MUSE_SPEC_KV_BYTES; mns=128; mnbt=8199; spec=1
    else
      kv=16957243392; [ "$arm" = on ] && kv=$MUSE_OFFLOAD_KV_BYTES; mns=256; mnbt=8192; spec=0
    fi
    exp=$([ "$PROFILE" = spec ] && { [ "$arm" = on ] && echo 11525693440 || echo 11573411840; } || echo 16957243392)
    [ "$kv" = "$exp" ] || { echo "bytes $kv != expected $exp"; exit 3; }
    env SRC=$TREE GPU=0,1,2,3,4,5,6,7 HOST_DIR=$HOST VLLM_ENABLE_V1_MULTIPROCESSING=0 E2E_ARM=$arm E2E_KV_BYTES=$kv E2E_PER_RANK_BYTES=${MUSE_OFFLOAD_PER_RANK_BYTES:-0} \
      E2E_SPEC=$spec E2E_MNS=$mns E2E_MNBT=$mnbt E2E_CAP=256 E2E_GEN_TOKENS=128 \
      timeout 5400 bash "$HOST/scripts/runsrc_topo.sh" "$T/e2e_s110.py" "$OUT/$tag.json" > "$LOGS/$tag.log" 2>&1
  )
  local rc=$?
  log "E2E END $tag rc=$rc passes=$(ls "$OUT/$tag".*.json 2>/dev/null | wc -l) oom=$(grep -a -c -E 'OutOfMemoryError|CUDA out of memory' "$LOGS/$tag.log") bind_lines=$(grep -a -c 'Binding worker subprocess' "$LOGS/$tag.log") conn_workers=$(grep -a -c 'SimpleCPUOffloadWorker \[CPU\]' "$LOGS/$tag.log") mm_embeds=$(grep -a -c 'enable_mm_embeds is True' "$LOGS/$tag.log") $(grep -a -h 'E2E_S110 DONE' "$LOGS/$tag.log" | tail -1 | cut -c1-60)"
  sleep 20
}

# artifact test of one process: tag arm -> 0 = fine
proc_ok() {
  local tag=$1 arm=$2
  [ "$(grep -a -c -E 'OutOfMemoryError|CUDA out of memory' "$LOGS/$tag.log" 2>/dev/null)" = 0 ] || return 1
  [ "$(ls "$OUT/$tag".*.json 2>/dev/null | wc -l)" -ge 12 ] || return 1
  grep -a -q 'E2E_S110 DONE' "$LOGS/$tag.log" || return 1
  if [ "$arm" = on ]; then
    [ "$(grep -a -c 'Binding worker subprocess' "$LOGS/$tag.log")" -ge 8 ] || return 1
    [ "$(grep -a -c 'SimpleCPUOffloadWorker \[CPU\]' "$LOGS/$tag.log")" -ge 8 ] || return 1
    [ "$(grep -a -c 'enable_mm_embeds is True' "$LOGS/$tag.log")" = 0 ] || return 1
  else
    [ "$(grep -a -c 'SimpleCPUOffload' "$LOGS/$tag.log")" = 0 ] || return 1
    [ "$(grep -a -c 'Binding worker subprocess' "$LOGS/$tag.log")" = 0 ] || return 1
  fi
  return 0
}

N=d_$PROFILE
rm -f "$W/det_$PROFILE.txt"
for spec in "${N}_ONa on" "${N}_OFFa off" "${N}_ONb on" "${N}_OFFb off"; do
  set -- $spec
  e2e_run "$1" "$2"
  proc_ok "$1" "$2" || { log "DET ARTIFACT in $1 (OOM, missing passes or evidence): retry once"; e2e_run "$1" "$2"; }
done
bad=""
for spec in "${N}_ONa on" "${N}_OFFa off" "${N}_ONb on" "${N}_OFFb off"; do set -- $spec; proc_ok "$1" "$2" || bad="$bad $1"; done
if [ -n "$bad" ]; then echo ARTIFACT > "$W/det_$PROFILE.txt"; log "DETERMINISM $PROFILE ARTIFACT (after the retry:$bad): not a mismatch, left unresolved"; exit 0; fi
python3 -I "$T/cmp_e2e_s110.py" "$OUT" "${N}_ONa" "${N}_ONb" "${N}_OFFa" "${N}_OFFb" > "$OUT/compare_$PROFILE.txt" 2>&1
while IFS= read -r l; do log "E2E_CMP_$PROFILE $l"; done < <(grep -E 'E2E_GATE|FAIL|MISSING' "$OUT/compare_$PROFILE.txt" | head -40)
if grep -q 'E2E_GATE=PASS' "$OUT/compare_$PROFILE.txt"; then echo PASS > "$W/det_$PROFILE.txt"; else echo FAIL > "$W/det_$PROFILE.txt"; fi
log "DETERMINISM $PROFILE $(cat "$W/det_$PROFILE.txt")"
