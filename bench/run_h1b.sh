#!/bin/bash
# Server side: the H1b validation sequence in one GPU-lock acquisition. usage: run_h1b.sh <tag> [steps...]
# Run it through the lock helper:  bash scripts/locked_job.sh <tag> <timeout_s> bash bench/run_h1b.sh <tag> <steps...>
# steps: tests_quick tests tests_soak timeout epoch
tag=$1; shift
H=${HOST_DIR:-/raid/vllm-runtime/host_topo}
cd $H/bench
for s in "$@"; do
  echo "=== step $s $(date +%T)"
  case $s in
    tests_quick) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh tests_h1b.py --quick --soak-replays 2000 ;;
    tests) BENCH_WATCHDOG_S=1500 bash ../scripts/py_topo.sh tests_h1b.py --soak-replays 20000 ;;
    tests_soak) BENCH_WATCHDOG_S=1500 bash ../scripts/py_topo.sh tests_h1b.py --groups soak --soak-replays 100000 ;;
    timeout) BENCH_WATCHDOG_S=120 bash ../scripts/py_topo.sh test_timeout_large.py ;;
    epoch) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh test_epoch.py ;;
  esac
  echo "=== step $s rc=$? $(date +%T)"
done
