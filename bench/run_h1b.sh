#!/bin/bash
# Server side: the H1b validation sequence in one GPU-lock acquisition. usage: run_h1b.sh <tag> [steps...]
# Run it through the lock helper:  bash scripts/locked_job.sh <tag> <timeout_s> bash bench/run_h1b.sh <tag> <steps...>
# steps: tests_quick tests tests_soak bench_ar bench_rs bench_ag sweep_ar sweep_rs sweep_ag ce_probe a2a_probe a2a_ce a2a_mix trace trace16 search_ar search_rs search_ag timeout epoch
tag=$1; shift
H=${HOST_DIR:-/raid/vllm-runtime/host_topo}
W=/raid/vllm-runtime/work/host
cd $H/bench
for s in "$@"; do
  echo "=== step $s $(date +%T)"
  case $s in
    tests_quick) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh tests_h1b.py --quick --soak-replays 2000 ;;
    tests) BENCH_WATCHDOG_S=1500 bash ../scripts/py_topo.sh tests_h1b.py --soak-replays 20000 ;;
    tests_soak) BENCH_WATCHDOG_S=1500 bash ../scripts/py_topo.sh tests_h1b.py --groups soak --soak-replays 100000 ;;
    bench_ar) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_ar.jsonl --what ar --mibs 1,1.5,2,2.5,3,4,6,8,16,32,64,128 ;;
    bench_rs) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_rs.jsonl --what rs --mibs 0.125,0.25,0.5,1,2,4,8,16,32,64,128 ;;
    bench_ag) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_ag.jsonl --what ag --mibs 0.25,0.5,1,2,4,8,16,32,64,128 ;;
    sweep_ar) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_sweep_ar.jsonl --what ar --mibs 4,16,128 --sweep ;;
    sweep_rs) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_sweep_rs.jsonl --what rs --mibs 16,128 --sweep ;;
    sweep_ag) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_sweep_ag.jsonl --what ag --mibs 16,128 --sweep ;;
    ce_probe) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh ce_cut_probe.py --out $W/h1b_${tag}_ce_probe.jsonl ;;
    a2a_probe) BENCH_WATCHDOG_S=1200 bash ../scripts/py_topo.sh a2a_probe.py --out $W/h1b_${tag}_a2a_probe.jsonl ;;
    a2a_mix) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh a2a_probe.py --out $W/h1b_${tag}_a2a_mix.jsonl --only step_mix --sizes 2097152,4194304 ;;
    a2a_ce) BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh a2a_probe.py --out $W/h1b_${tag}_a2a_ce.jsonl --only ce,ce_seq,ce_seq_cut --sizes 524288,1048576,4194304 ;;
    trace) MUSE_TOPO_TRACE=1 BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh trace_h1b.py --mib 128 ;;
    trace16) MUSE_TOPO_TRACE=1 BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh trace_h1b.py --mib 16 ;;
    search_ar) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_search_ar.jsonl --what ar --mibs 1,2,3,4,6,8,16,32,64,128 --search ;;
    search_rs) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_search_rs.jsonl --what rs --mibs 1,2,4,8,16,32,64,128 --search ;;
    search_ag) BENCH_WATCHDOG_S=3000 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_search_ag.jsonl --what ag --mibs 2,4,8,16,32,64,128 --search ;;
    small_ar) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_small_ar.jsonl --what ar --mibs 1,1.5,2,2.5,3,4 --search ;;
    small_rs) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_small_rs.jsonl --what rs --mibs 0.125,0.25,0.5,1,2 --search ;;
    small_ag) BENCH_WATCHDOG_S=1800 bash ../scripts/py_topo.sh bench_h1b.py --out $W/h1b_${tag}_small_ag.jsonl --what ag --mibs 0.25,0.5,1,2,4 --search ;;
    timeout) BENCH_WATCHDOG_S=120 bash ../scripts/py_topo.sh test_timeout_large.py ;;
    epoch) BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh test_epoch.py ;;
  esac
  echo "=== step $s rc=$? $(date +%T)"
done
