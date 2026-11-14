#!/usr/bin/env bash
# Pinned-memory bandwidth microbench (item 2). Runs under the GPU lock (locked_sd.sh microbench.sh). GPUs hold almost no memory
# here (1 GiB each). Part 1: one GPU at a time, buffer on the local node / a same-socket remote node / a cross-socket node
# (nodes without GPUs, so nothing else loads them). Part 2: all 8 GPUs at once for three placements.
# GPU -> node: 0,1->3  2,3->0  4,5->7  6,7->4. Socket 0 = nodes 0-3, socket 1 = nodes 4-7; GPU-less nodes 1,2,5,6.
set -u
N=/raid/vllm-runtime/work/numa
OUT=${OUT:-$N/microbench}
mkdir -p "$OUT"
export SRC=/raid/vllm-runtime/sm120-dev/src
declare -A LOCAL=([0]=3 [1]=3 [2]=0 [3]=0 [4]=7 [5]=7 [6]=4 [7]=4)
# same-socket remote (GPU-less) and cross-socket (GPU-less) node per GPU
declare -A SAME=([0]=2 [1]=2 [2]=1 [3]=1 [4]=5 [5]=5 [6]=6 [7]=6)
declare -A CROSS=([0]=5 [1]=5 [2]=6 [3]=6 [4]=1 [5]=1 [6]=2 [7]=2)
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/status.txt"; }
log "MICROBENCH start"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
log "gpu_used_mib=$used"
run() {  # gpu placements [extra args]
  local g=$1 pl=$2; shift 2
  GPU=$g /raid/vllm-runtime/sm120-dev/runsrc.sh "$N/pinned_bw.py" --gpu-node "${LOCAL[$g]}" --placements "$pl" --out "$OUT/results.jsonl" "$@" 2>&1 | grep -v -E "^$|Warning|warn" | tail -n +1 >> "$OUT/run.log"
}
# Part 1: single GPU
for g in 0 2 4 6 1 5; do
  log "single gpu=$g local=${LOCAL[$g]} same=${SAME[$g]} cross=${CROSS[$g]}"
  run "$g" "local:${LOCAL[$g]},same:${SAME[$g]},cross:${CROSS[$g]}" --tag single
done
# Part 2: concurrent, all 8 GPUs, common start 40 s after launch (process start + page population take ~25 s)
conc() {  # name placement-fn
  local name=$1; shift
  local start; start=$(( $(date +%s) + 45 ))
  log "concurrent $name start_at=$start"
  local pids=()
  for g in 0 1 2 3 4 5 6 7; do
    node=$("$@" "$g")
    ( GPU=$g /raid/vllm-runtime/sm120-dev/runsrc.sh "$N/pinned_bw.py" --gpu-node "${LOCAL[$g]}" --placements "$name:$node" \
        --sizes 4096,65536,1048576 --seconds 4 --start-at "$start" --tag "conc_$name" --out "$OUT/results.jsonl" >> "$OUT/run_conc_$name.log" 2>&1 ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}
loc() { echo "${LOCAL[$1]}"; }
one() { echo 1; }
crossf() { echo "${CROSS[$1]}"; }
conc local loc
conc allnode1 one
conc cross crossf
log "MICROBENCH done"
echo DONE > "$OUT/DONE"
