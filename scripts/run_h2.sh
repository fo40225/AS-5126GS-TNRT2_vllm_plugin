#!/bin/bash
# Server side: the vLLM-integration (plugin) validation sequence in one GPU-lock acquisition
# (start it with scripts/locked_job.sh). usage: [SRC=<vllm tree>] [HOST_DIR=<host folder>] run_h2.sh <tag> [steps...]
# steps: plugin    8-process vLLM TP group through the plugin: eager / graph / fallback / mixed sequences (bench/test_plugin.py)
#        e2e_topo1 e2e_topo2   offline 6-layer dummy-weight run with the topo path (VLLM_ENABLE_V1_MULTIPROCESSING=0, greedy)
#        e2e_nccl  the same run with MUSE_TOPO_ENABLE=0 (NCCL baseline of the same tree)
#        lp_topo lp_nccl   teacher-forced prompt-logprob dumps (bench/e2e_prompt_lp.py) with / without the topo path
#        lp_compare        statistics of |d logprob| nccl vs topo;  ops   per-op summary of the e2e_topo1 log
#        real_topo1 real_topo2 real_nccl real_compare   FULL 62-layer model with the real NVFP4 weights (bench/e2e_real.py):
#                  greedy continuations + teacher-forced NLL, topo twice (bitwise expected) and NCCL
#        compare   topo vs topo (bitwise expected) and topo vs NCCL (token / logprob difference) with nccl/compare.py
# Logs and result dumps go to $MUSE_TEST_SCRATCH/h2/ (created here; never into the host folder).
tag=$1; shift
H=${HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
W=${MUSE_TEST_SCRATCH:-/raid/vllm-runtime/work/plugin-tests}/h2
mkdir -p $W
export SRC=${SRC:-/raid/vllm-runtime/sm120-dev/src}
export GPU=0,1,2,3,4,5,6,7
e2e() {  # name, extra env...
  name=$1; shift
  env VLLM_ENABLE_V1_MULTIPROCESSING=0 MUSE_TOPO_LOG_OPS=2 "$@" timeout 2400 \
    $H/scripts/runsrc_topo.sh $H/nccl/e2e_bitwise.py $W/out_${tag}_$name.json --repeat 1 > $W/e2e_${tag}_$name.log 2>&1
}
e2e_lp() {  # name, extra env...
  name=$1; shift
  env VLLM_ENABLE_V1_MULTIPROCESSING=0 MUSE_TOPO_LOG_OPS=2 "$@" timeout 2400 \
    $H/scripts/runsrc_topo.sh $H/bench/e2e_prompt_lp.py $W/lp_${tag}_$name.json > $W/lp_${tag}_$name.log 2>&1
}
real() {  # name, extra env...
  name=$1; shift
  env VLLM_ENABLE_V1_MULTIPROCESSING=0 MUSE_TOPO_LOG_OPS=1 "$@" timeout 3000 \
    $H/scripts/runsrc_topo.sh $H/bench/e2e_real.py $W/real_${tag}_$name.json > $W/real_${tag}_$name.log 2>&1
}
for s in "$@"; do
  echo "=== step $s $(date +%T)"
  case $s in
    plugin) BENCH_WATCHDOG_S=1200 timeout 1500 $H/scripts/runsrc_topo.sh $H/bench/test_plugin.py ${PLUGIN_ARGS:-} ;;
    e2e_topo1) e2e topo1 MUSE_TOPO_ENABLE=1 ;;
    e2e_topo2) e2e topo2 MUSE_TOPO_ENABLE=1 ;;
    e2e_nccl) e2e nccl MUSE_TOPO_ENABLE=0 ;;
    lp_topo) e2e_lp topo MUSE_TOPO_ENABLE=1 ;;
    lp_nccl) e2e_lp nccl MUSE_TOPO_ENABLE=0 ;;
    lp_compare) python3 $H/bench/e2e_prompt_lp.py --compare $W/lp_${tag}_nccl.json $W/lp_${tag}_topo.json ;;
    ops) python3 $H/bench/summarize_ops.py $W/e2e_${tag}_${OPS_LOG:-topo1}.log ;;
    real_topo1) real topo1 MUSE_TOPO_ENABLE=1 ;;
    real_topo2) real topo2 MUSE_TOPO_ENABLE=1 ;;
    real_nccl) real nccl MUSE_TOPO_ENABLE=0 ;;
    real_compare)
      C=$H/nccl/compare.py
      echo "--- topo1 vs topo2 (separate processes)"; python3 $C $W/real_${tag}_topo1.json $W/real_${tag}_topo2.json | tail -45
      echo "--- nccl vs topo1"; python3 $C $W/real_${tag}_nccl.json $W/real_${tag}_topo1.json
      grep -h "NLL\|load_time" $W/real_${tag}_nccl.log $W/real_${tag}_topo1.log $W/real_${tag}_topo2.log
      grep -h "SAMPLE" $W/real_${tag}_topo1.log | cut -c1-200 ;;
    compare)
      C=$H/nccl/compare.py
      echo "--- in-process repeat inside topo1"; python3 $C --inproc $W/out_${tag}_topo1.json
      echo "--- topo1 vs topo2 (separate processes)"; python3 $C $W/out_${tag}_topo1.json $W/out_${tag}_topo2.json
      echo "--- nccl vs topo1"; python3 $C $W/out_${tag}_nccl.json $W/out_${tag}_topo1.json ;;
  esac
  echo "=== step $s rc=$? $(date +%T)"
done
