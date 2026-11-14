#!/usr/bin/env bash
# S1-10 copy of the S1-9 L11 exact_boot_l11.sh with XEXTRA (more `vllm serve` arguments: --numa-bind, --kv-transfer-config ...) appended to the serve arguments.
# S1-9 L11 copy (+ CAP, HOST_DIR) of the S1-9 L5 copy (+ MNBT) of the S1-9 L3b copy of the S1-7 step 3 exact_boot.sh (stress with the extra phases H, I, L): one boot with an exact --kv-cache-memory-bytes (or the default path when KV is empty), NVML sampler from
# the process launch, stress job, analysis. Run it under the GPU lock: locked_sd.sh <this script> <tag>.
# env: KV (bytes | empty = default path), MNS (128), SPEC (preset | empty), TREE (default sm120-dev/src),
#      PROBE=1 (allocator-stats probe; needs TREE=src_s17-exact), REPS, PHASES, CONC_A, PL_PROBE, STRESS (1|0),
#      MNBT (--max-num-batched-tokens), CG256=1 (--max-cudagraph-capture-size 256) or CAP=<n> (--max-cudagraph-capture-size n), HOST_DIR (host folder mirror, default host_topo),
#      PROB=probabilistic|greedy (draft sampling, default probabilistic)
set -u
TAG=${1:?tag}
B=/raid/vllm-runtime/sm120-dev
H=/raid/vllm-runtime/work/sd/harness
E=/raid/vllm-runtime/work/s1-7/exact
G=${L3B_G:-/raid/vllm-runtime/work/s1-9/L11/g}   # S1-9 L3 copies: stress_l3.py (phases H, I, L, --eligible), job_x_l3b.sh
export L3B_G=$G
R=/raid/vllm-runtime/musetest/results
PY=$B/.venv/bin/python
OUT=$R/$TAG
mkdir -p $OUT/x
export TREE=${TREE:-$B/src}
export MNS=${MNS:-128}
HOST_DIR=${HOST_DIR:-/raid/vllm-runtime/host_topo}
EXTRA=""
[ -n "${KV:-}" ] && EXTRA="--kv-cache-memory-bytes $KV"
[ "${CG256:-0}" = 1 ] && EXTRA="$EXTRA --max-cudagraph-capture-size 256"
[ -n "${CAP:-}" ] && EXTRA="$EXTRA --max-cudagraph-capture-size $CAP"
[ -n "${MNBT:-}" ] && EXTRA="$EXTRA --max-num-batched-tokens $MNBT"
[ -n "${XEXTRA:-}" ] && EXTRA="$EXTRA $XEXTRA"
SPEC_OVER=""
if [ -n "${SPEC:-}" ]; then
  SPEC_OVER="{\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"${PROB:-probabilistic}\"}"
fi
echo "TAG=$TAG KV=${KV:-default} MNS=$MNS SPEC=${SPEC:-none} TREE=$TREE PROBE=${PROBE:-0} REPS=${REPS:-1} CG256=${CG256:-0} CAP=${CAP:-none} HOST_DIR=$HOST_DIR EXTRA='$EXTRA'" > $OUT/x/boot_args.txt
cat $OUT/x/boot_args.txt
for i in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  [ "${u:-0}" -lt 2000 ] && break; sleep 5
done
setsid $PY -I $E/smi_sampler.py $OUT/x 0.01 > $OUT/x/sampler.log 2>&1 < /dev/null &
SAMPLER=$!
date +%s.%N > $OUT/x/launch_epoch
if [ "${PROBE:-0}" = 1 ]; then export S17X=1 S17X_DIR=$OUT/x; fi
env SRC=$TREE HOST_DIR=$HOST_DIR VLLM_LOGGING_LEVEL=DEBUG JOB_TIMEOUT_S=${JOB_TIMEOUT_S:-5400} PORT=8100 \
    MAX_NUM_SEQS=$MNS SPEC_CONFIG="${SPEC:-}" SPEC_OVERRIDES="$SPEC_OVER" SERVE_EXTRA="$EXTRA" \
    MNS=$MNS REPS=${REPS:-1} PHASES=${PHASES:-A1,B,C1,C2,D,H,I,L,M,X} CONC_A=${CONC_A:-256} PL_PROBE=${PL_PROBE:-0} STRESS=${STRESS:-1} \
  $H/serve_sd.sh $TAG -- $G/job_x_l3b.sh > $OUT/serve_sd.out 2>&1
sleep 2
kill -TERM $SAMPLER 2>/dev/null
for i in 1 2 3 4 5; do kill -0 $SAMPLER 2>/dev/null || break; sleep 1; done
kill -KILL $SAMPLER 2>/dev/null
$PY -I $E/analyze.py $OUT ${MNS} > $OUT/x/summary.txt 2>&1
cat $OUT/x/summary.txt
