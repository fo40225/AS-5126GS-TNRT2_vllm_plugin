#!/bin/bash
# Server side, one GPU-lock acquisition: tests, medium all-reduce sweep, table generation, dispatch check, real-weight
# concurrency A/B (bf16 exchange of the medium all-reduce off / on) and the GSM8K smoke of the "on" configuration.
# usage: HOST_DIR=<host folder> r3c_chain.sh <tag> [stages]     stages (default all): test sweep reg disp e2e acc
# Results: work/r3c-medium-ar/<tag>/ (rows, tuning.json, table.md, logs) and musetest/results/<tag>_conc_bx{0,1}, <tag>_acc.
tag=$1; stages=${2:-"test sweep reg disp e2e acc"}
H=${HOST_DIR:-/raid/vllm-runtime/host_topo_r3c-medium-ar}
M=/raid/vllm-runtime/work/r3c-medium-ar
W=/raid/vllm-runtime/work/host
SRC=${SRC:-/raid/vllm-runtime/sm120-dev/src_r3c-medium-ar}
R=/raid/vllm-runtime/musetest/results
O=$M/$tag
export HOST_DIR=$H
mkdir -p "$O"
has() { case " $stages " in *" $1 "*) return 0;; esac; return 1; }
say() { echo "=== $* $(date +%T)"; }
cd "$H/bench"
ok_kernel=1

if has test; then
  say "stage test"
  # hard correctness groups gate the real-weight stages; the accuracy group (soft thresholds, informational) runs apart
  BENCH_WATCHDOG_S=1200 bash ../scripts/py_topo.sh test_medium_bx.py --groups emu,bits,det,disp,seq > "$O/test_medium.log" 2>&1
  rc=$?; say "test_medium_bx (hard groups) rc=$rc"; tail -4 "$O/test_medium.log"; grep '^FAIL' "$O/test_medium.log" | head
  [ "$rc" != 0 ] && ok_kernel=0
  BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh test_medium_bx.py --groups acc > "$O/test_medium_acc.log" 2>&1
  say "test_medium_bx (acc group, informational) rc=$?"; grep -E 'accuracy|PASS|FAIL' "$O/test_medium_acc.log" | head
fi

if has sweep; then
  say "stage sweep"
  BENCH_WATCHDOG_S=6000 bash ../scripts/py_topo.sh bench_medium.py --out "$O/sweep.jsonl" > "$O/sweep.log" 2>&1
  say "bench_medium rc=$?"; grep -c INVALID "$O/sweep.log"
  python3 -I make_table_medium.py "$O/sweep.jsonl" --json "$O/tuning.json" --md "$O/table.md" > "$O/table.txt" 2>&1
  say "make_table rc=$?"; cat "$O/table.txt"
fi

if has reg; then
  say "stage reg"
  BENCH_WATCHDOG_S=600 bash ../scripts/py_topo.sh tests_h1a.py --quick --soak-replays 20000 > "$O/h1a_quick.log" 2>&1
  say "tests_h1a rc=$?"; tail -3 "$O/h1a_quick.log"
  BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh tests_h1b.py --quick --groups ar,bits,rs,ag,det,bx,seq > "$O/h1b_quick.log" 2>&1
  say "tests_h1b rc=$?"; tail -3 "$O/h1b_quick.log"
fi

if has disp && [ -f "$O/tuning.json" ]; then
  say "stage disp"
  MUSE_TOPO_AR_TUNING_FILE=$O/tuning.json BENCH_WATCHDOG_S=900 bash ../scripts/py_topo.sh bench_medium.py \
    --out "$O/dispatch.jsonl" --no-sweep > "$O/dispatch.log" 2>&1
  say "dispatch bench rc=$?"
fi

if has e2e && [ "$ok_kernel" = 1 ]; then
  say "stage e2e"
  TUNE=""; [ -f "$O/tuning.json" ] && TUNE=$O/tuning.json
  for v in 0 1; do
    if [ "$v" = 0 ]; then extra="MUSE_TOPO_MEDIUM_BX=0"; else extra="MUSE_TOPO_MEDIUM_BX=1 MUSE_TOPO_AR_TUNING_FILE=$TUNE"; fi
    env HOST_DIR=$H SRC=$SRC CONCS="1 16 64 128" REPEATS=2 $extra bash "$H/scripts/conc_sweep.sh" ${tag}_conc_bx$v > "$O/conc_bx$v.out" 2>&1
    say "conc_sweep bx$v rc=$?"; grep BENCH "$R/${tag}_conc_bx$v/status.txt" | sed 's/^/  /'
    # did the switch reach the workers? (plugin summary line) and did anything go wrong in the serve log
    echo "  plugin lines bx$v:"; grep -h -E 'TopoComm active|all_reduce tuning from' "$R/${tag}_conc_bx$v/serve.log" | sort -u | head -4 | cut -c1-400 | sed 's/^/    /'
    echo "  serve log problems bx$v: $(grep -c -E 'spin timeout|Traceback|muse_topo.*(fallback|disabled)' "$R/${tag}_conc_bx$v/serve.log")"
  done
  grep -q 'medium bf16 exchange off' "$R/${tag}_conc_bx0/serve.log" || echo "WARNING: bx0 serve log does not show 'medium bf16 exchange off'"
  grep -q 'medium bf16 exchange from' "$R/${tag}_conc_bx1/serve.log" && flag_ok=1 || { flag_ok=0; echo "WARNING: bx1 serve log does not show 'medium bf16 exchange from'"; }
  python3 -I - "$R/${tag}_conc_bx0" "$R/${tag}_conc_bx1" > "$O/e2e_summary.txt" 2>&1 <<'PY'
import json, glob, sys, os
def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "s512_128_c*_r*.json"))):
        j = json.load(open(f)); c = int(os.path.basename(f).split("_c")[1].split("_")[0])
        out.setdefault(c, []).append((j["output_throughput"], j["mean_tpot_ms"]))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
for c in sorted(set(a) | set(b)):
    ra, rb = a.get(c, []), b.get(c, [])
    fa = lambda r: ("%.1f tok/s tpot %.2f" % (sum(x[0] for x in r) / len(r), sum(x[1] for x in r) / len(r))) if r else "-"
    gain = (sum(x[0] for x in rb) / len(rb)) / (sum(x[0] for x in ra) / len(ra)) - 1 if ra and rb else float("nan")
    print("c%-3d  bx0: %-34s bx1: %-34s out tok/s gain %+.2f%%" % (c, fa(ra), fa(rb), 100 * gain))
PY
  cat "$O/e2e_summary.txt"
fi

if has acc && [ "$ok_kernel" = 1 ]; then
  gain=$(grep '^c64 ' "$O/e2e_summary.txt" 2>/dev/null | sed 's/.*gain \([-+0-9.]*\)%.*/\1/')
  say "stage acc (c64 gain ${gain:-n/a}%)"
  # run the smoke unless the real-weight A/B is clearly negative (policy: the final configuration needs the smoke)
  if [ "${flag_ok:-1}" = 1 ] && { [ -z "$gain" ] || python3 -I -c "import sys; sys.exit(0 if float('$gain') >= -0.5 else 1)"; }; then
    TUNE=""; [ -f "$O/tuning.json" ] && TUNE=$O/tuning.json
    env HOST_DIR=$H SRC=$SRC MUSE_TOPO_MEDIUM_BX=1 MUSE_TOPO_AR_TUNING_FILE=$TUNE CONC=64 bash "$M/acc/acc_run.sh" ${tag}_acc > "$O/acc.out" 2>&1
    say "acc rc=$?"; grep -E 'SCORE|AGREE|PASS' "$R/${tag}_acc/status.txt"
  else
    say "acc skipped (c64 gain below -0.5% or the flag did not reach the workers)"
  fi
fi
say "chain done"
