#!/bin/bash
# Regression test of scripts/preflight_offload.sh on synthetic node meminfo files (S1-10 gate review). No GPU: bash bench/test_preflight_offload.sh
here=$(cd "$(dirname "$0")/.." && pwd)
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
fails=0
# mkn <node> <MemFree MiB> <file cache MiB>
mkn() { mkdir -p "$d/node$1"; printf 'Node %s MemFree: %d kB\nNode %s Active(file): %d kB\nNode %s Inactive(file): %d kB\n' "$1" $(( $2 * 1024 )) "$1" $(( $3 * 1024 / 2 )) "$1" $(( $3 * 1024 / 2 )) > "$d/node$1/meminfo"; }
check() { # desc want_rc want_msg [env...]
  local desc=$1 want=$2 msg=$3; shift 3
  out=$(env -i PATH="$PATH" MUSE_OFFLOAD_NODE_ROOT="$d" "$@" bash "$here/scripts/preflight_offload.sh" 2>&1); rc=$?
  if [ "$rc" != "$want" ] || ! echo "$out" | grep -q -- "$msg"; then echo "FAIL case '$desc': rc=$rc wanted $want / '$msg'"; echo "$out" | sed 's/^/    /'; fails=$((fails+1)); else echo "ok   $desc"; fi
}
# the idle state (no server) derived from the 0.33.0 phase 3 A_N figures (server up: node 3 119,263, node 0 118,644, node 7 105,586, node 4 119,641 MiB) plus W_N
# (two workers 8,192 MiB, node 3 + EngineCore 1,229, node 7 + API server 1,331); the shipped P* = 48.26 GiB leaves 6.9 GiB on node 7 (M = 6.6 GiB: no warning)
# (the node 7 fixture is looser than A_7 + W_7 = 105,586 + 9,523 = 115,109 MiB, which would leave 6,759 MiB; the measured phase 3 preflight left on node 7 was 7,091-7,253 MiB)
mkn 0 20000 106844; mkn 3 20000 108700; mkn 4 20000 107900; mkn 7 10000 105400
check "shipped tier fits the 0.33.0 nodes" 0 "PREFLIGHT OK"
mkn 7 10000 60000
check "node 7 short by 40 GiB fails" 1 "PREFLIGHT FAIL"
mkn 7 10000 105400
check "a smaller tier fits" 0 "PREFLIGHT OK" MUSE_OFFLOAD_PER_RANK_BYTES=26214400000
mkn 4 20000 92000
check "node 4 inside the warn band passes with a warning" 0 "WARN" 
mkn 4 20000 107900
check "tier off: nothing to check" 0 "nothing to check" MUSE_OFFLOAD=0
rm "$d/node3/meminfo"
check "unreadable meminfo" 2 "cannot read"
echo "$fails case(s) failed"; exit $(( fails > 0 ))
