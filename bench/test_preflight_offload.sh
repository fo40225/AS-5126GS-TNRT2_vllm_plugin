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
# the idle server state of the study: node 7 A_N = 98,892 + 8,192 + 1,331 MiB without a server
mkn 0 20000 97738; mkn 3 20000 98000; mkn 4 20000 98000; mkn 7 10000 98400
check "shipped tier fits the study's nodes" 0 "PREFLIGHT OK"
mkn 7 10000 60000
check "node 7 short by 40 GiB fails" 1 "PREFLIGHT FAIL"
mkn 7 10000 98400
check "a smaller tier fits" 0 "PREFLIGHT OK" MUSE_OFFLOAD_PER_RANK_BYTES=26214400000
mkn 4 20000 87000
check "node 4 inside the warn band passes with a warning" 0 "WARN" 
mkn 4 20000 98000
check "tier off: nothing to check" 0 "nothing to check" MUSE_OFFLOAD=0
rm "$d/node3/meminfo"
check "unreadable meminfo" 2 "cannot read"
echo "$fails case(s) failed"; exit $(( fails > 0 ))
