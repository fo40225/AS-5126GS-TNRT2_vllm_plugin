#!/bin/bash
# Server side: the H1a validation sequence in one GPU-lock acquisition. usage: run_h1a.sh <tag> [steps...]
# steps: smoke tests tests_quick bench_quick bench_full bench_public bench_small bench_ab other other_full topology unit initfail timeout epoch
tag=$1; shift
H=${HOST_DIR:-/raid/vllm-runtime/host_topo}
W=/raid/vllm-runtime/work/host
cd $H/bench
for s in "$@"; do
  echo "=== step $s $(date +%T)"
  case $s in
    smoke) BENCH_WATCHDOG_S=240 bash ../scripts/py_topo.sh smoke.py ;;
    tests) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh tests_h1a.py ;;
    tests_quick) BENCH_WATCHDOG_S=300 bash ../scripts/py_topo.sh tests_h1a.py --quick --soak-replays 20000 ;;
    bench_quick) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh bench_h1a.py --out $W/results_${tag}_quick.jsonl --sweep --quick --ts 1,4,16,64,512 --hids 8192 --ag-ts 1,8,64 ;;
    bench_full) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1a.py --out $W/results_${tag}_full.jsonl --sweep ;;
    trace) MUSE_TOPO_TRACE=1 BENCH_WATCHDOG_S=300 bash ../scripts/py_topo.sh trace_h1a.py ;;
    trace_pdl) MUSE_TOPO_TRACE=1 BENCH_WATCHDOG_S=300 bash ../scripts/py_topo.sh trace_h1a.py --pdl ;;
    bench_public) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1a.py --out $W/results_${tag}_public.jsonl --public --ts 1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512 --ag-ts 1,2,4,8,16,24,32,48,64,96,128,192,256 ;;
    bench_small) BENCH_WATCHDOG_S=1500 bash ../scripts/py_topo.sh bench_h1a.py --out $W/results_${tag}_small.jsonl --sweep --ts 1,2,4,8,16,32 --hids 8192,4096 --ag-ts 1,2,4,8,16,32 ;;
    bench_ab) for pdl in 0 1; do MUSE_TOPO_PDL=$pdl BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh bench_h1a.py --out $W/results_${tag}_pdl$pdl.jsonl --sweep --quick --ts 1,2,4,8,16,32 --hids 8192,4096 --ag-ts 1,2,4,8,16 ; done ;;
    fi_spacer) (cd $W/customar_copy && BENCH_WATCHDOG_S=900 AR_WATCHDOG_S=900 bash ./run_py.sh bench_spacer.py --out $W/fi_spacer_${tag}.jsonl --spacers 50 --port 29631) ;;
    ctx) BENCH_WATCHDOG_S=120 bash ../scripts/py_topo.sh check_ctx.py ;;
    relay) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh host_relay_probe.py --out $W/relay_${tag}.jsonl ;;
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
