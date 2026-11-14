#!/bin/bash
# Server side: the H1a validation sequence in one GPU-lock acquisition. usage: run_h1a.sh <tag> [steps...]
# steps: smoke tests tests_quick ctx initfail other other_full topology timeout epoch unit
tag=$1; shift
H=${HOST_DIR:-/raid/vllm-runtime/host_topo}
cd $H/bench
for s in "$@"; do
  echo "=== step $s $(date +%T)"
  case $s in
    smoke) BENCH_WATCHDOG_S=240 bash ../scripts/py_topo.sh smoke.py ;;
    tests) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh tests_h1a.py ;;
    tests_quick) BENCH_WATCHDOG_S=300 bash ../scripts/py_topo.sh tests_h1a.py --quick --soak-replays 20000 ;;
    ctx) BENCH_WATCHDOG_S=120 bash ../scripts/py_topo.sh check_ctx.py ;;
    initfail) BENCH_WATCHDOG_S=120 bash ../scripts/py_topo.sh test_init_failure.py ;;
    other) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh test_h1a.py --quick ;;
    topology) CUDA_VISIBLE_DEVICES= bash ../scripts/py_topo.sh test_topology.py ;;
    timeout) BENCH_WATCHDOG_S=120 bash ../scripts/py_topo.sh test_timeout.py ;;
    epoch) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh test_epoch.py ;;
    other_full) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh test_h1a.py ;;
    unit) CUDA_VISIBLE_DEVICES= bash ../scripts/py_topo.sh test_plugin_unit.py ;;
  esac
  echo "=== step $s rc=$? $(date +%T)"
done
