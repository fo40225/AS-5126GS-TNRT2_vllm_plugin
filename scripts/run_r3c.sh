#!/bin/bash
# Server side: the validation / measurement sequence of the medium all-reduce (r3c) in one GPU-lock acquisition.
# usage: [HOST_DIR=<host folder>] [SRC=<vLLM tree>] run_r3c.sh <tag> [steps...]
# steps: medium_test medium_test_quick h1a_quick h1b_quick bench_quick bench_full bench_cfg
tag=$1; shift
H=${HOST_DIR:-/raid/vllm-runtime/host_topo}
W=/raid/vllm-runtime/work/host
# exported: py_topo.sh puts it on PYTHONPATH, and test_medium_bx.py (seq group) imports the vLLM Triton kernels from it
export SRC=${SRC:-/raid/vllm-runtime/sm120-dev/src}
cd $H/bench
for s in "$@"; do
  echo "=== step $s $(date +%T)"
  case $s in
    medium_test) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh test_medium_bx.py ;;
    medium_test_quick) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh test_medium_bx.py --quick ;;
    h1a_quick) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh tests_h1a.py --quick --soak-replays 20000 ;;
    h1b_quick) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh tests_h1b.py --quick --groups ar,bits,rs,ag,det,bx,seq ;;
    bench_quick) BENCH_WATCHDOG_S=1500 bash ../scripts/py_topo.sh bench_medium.py --out $W/results_${tag}_quick.jsonl --quick ;;
    bench_full) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_medium.py --out $W/results_${tag}_full.jsonl ;;
    bench_dispatch) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh bench_medium.py --out $W/results_${tag}_dispatch.jsonl --no-sweep ;;
  esac
  echo "=== step $s rc=$? $(date +%T)"
done
