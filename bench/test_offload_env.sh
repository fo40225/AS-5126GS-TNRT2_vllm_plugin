#!/bin/bash
# Regression test of env/offload.env (S1-10 gate review): every source rebuilds the derived state, a profile typo is an error, the exact GPU bytes and the
# tier travel together. No GPU, no vLLM: usage  bash bench/test_offload_env.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
fails=0
# t <description> <expected stdout> <bash snippet run in a clean shell with the env files available as $E/host.env ...>
t() {
  local desc=$1 want=$2 code=$3 got
  got=$(env -i PATH="$PATH" E="$here/env" bash -c "$code" 2>/dev/null)
  if [ "$got" = "$want" ]; then echo "ok   $desc"; else echo "FAIL case '$desc'"; echo "    wanted: $want"; echo "    got:    $got"; fails=$((fails+1)); fi
}
# Note: `VAR=x source file` restores VAR after the source (bash), so the documented off switch is `export MUSE_OFFLOAD=0` before sourcing.
# the expected numbers below are the 0.33.0-derived values of env/offload.env (S1-10R phase 3, s1-10r/phase3_an.json and phase3_stress.json; update both together)
SPEC=11645337600; NS256=16854810624; PB=51814072320
t "spec defaults: bytes, bpb, spec override, serve args" "$SPEC 20480 $SPEC 1 1" \
  'source $E/offload.env; echo "$MUSE_OFFLOAD_KV_BYTES $MUSE_OFFLOAD_BPB $MUSE_SPEC_KV_BYTES_OVERRIDE $MUSE_OFFLOAD $(echo "$MUSE_OFFLOAD_SERVE_ARGS" | grep -c -- "--numa-bind --kv-transfer-config")"'
t "serve args carry the exact per-rank bytes and 8x as the total" "1 1" \
  'source $E/offload.env; echo "$(echo "$MUSE_OFFLOAD_SERVE_ARGS" | grep -c "cpu_bytes_to_use_per_rank\":'$PB'}") $(echo "$MUSE_OFFLOAD_SERVE_ARGS" | grep -c "cpu_bytes_to_use\":'$((PB*8))',")"'
t "spec_k7.env after offload.env uses the stepped bytes" "$SPEC" \
  'source $E/offload.env; source $E/spec_k7.env; echo $MUSE_SPEC_KV_BYTES'
t "ns256 then spec in ONE shell: spec bytes, not the ns256 bytes (review finding)" "$SPEC $SPEC 20480" \
  'MUSE_OFFLOAD_PROFILE=ns256 source $E/offload.env; MUSE_OFFLOAD_PROFILE=spec source $E/offload.env; source $E/spec_k7.env; echo "$MUSE_OFFLOAD_KV_BYTES $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD_BPB"'
t "spec then ns256 in one shell: the spec override is gone, ns256 bytes and bpb" "$NS256 65536 [] " \
  'source $E/offload.env; MUSE_OFFLOAD_PROFILE=ns256 source $E/offload.env; echo "$MUSE_OFFLOAD_KV_BYTES $MUSE_OFFLOAD_BPB [${MUSE_SPEC_KV_BYTES_OVERRIDE:-}] "'
t "unknown profile (typo): rc 1, no tier, no bytes, error set" "rc=1 off=0 args=[] err=1 kv=[] ov=[]" \
  'MUSE_OFFLOAD_PROFILE=ns265 source $E/offload.env 2>/dev/null; rc=$?; echo "rc=$rc off=$MUSE_OFFLOAD args=[$MUSE_OFFLOAD_SERVE_ARGS] err=$([ -n "$MUSE_OFFLOAD_ERROR" ] && echo 1) kv=[${MUSE_OFFLOAD_KV_BYTES:-}] ov=[${MUSE_SPEC_KV_BYTES_OVERRIDE:-}]"'
t "typo after a good source in the same shell: no stale bytes or override survive" "rc=1 off=0 args=[] kv=[] ov=[]" \
  'source $E/offload.env; MUSE_OFFLOAD_PROFILE=ns265 source $E/offload.env 2>/dev/null; rc=$?; echo "rc=$rc off=$MUSE_OFFLOAD args=[$MUSE_OFFLOAD_SERVE_ARGS] kv=[${MUSE_OFFLOAD_KV_BYTES:-}] ov=[${MUSE_SPEC_KV_BYTES_OVERRIDE:-}]"'
t "MUSE_OFFLOAD=0 after a tier-on source: no args, our override dropped" "off=0 args=[] ov=[]" \
  'source $E/offload.env; export MUSE_OFFLOAD=0; source $E/offload.env; echo "off=$MUSE_OFFLOAD args=[$MUSE_OFFLOAD_SERVE_ARGS] ov=[${MUSE_SPEC_KV_BYTES_OVERRIDE:-}]"'
t "MUSE_OFFLOAD=0 keeps an operator MUSE_SPEC_KV_BYTES_OVERRIDE (cap 384 launches)" "11346923520 11346923520" \
  'export MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520 MUSE_OFFLOAD=0; source $E/offload.env; MUSE_SPEC_GRAPH_CAP=384 source $E/spec_k7.env; echo "${MUSE_SPEC_KV_BYTES_OVERRIDE:-} $MUSE_SPEC_KV_BYTES"'
t "tier on discards an operator MUSE_SPEC_KV_BYTES_OVERRIDE (the bytes are an offload property)" "$SPEC" \
  'export MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520; source $E/offload.env; echo $MUSE_SPEC_KV_BYTES_OVERRIDE'
t "MUSE_OFFLOAD_KV_BYTES_OVERRIDE replaces the offload bytes everywhere" "11346000000 11346000000 11346000000" \
  'MUSE_OFFLOAD_KV_BYTES_OVERRIDE=11346000000 source $E/offload.env; source $E/spec_k7.env; echo "$MUSE_OFFLOAD_KV_BYTES $MUSE_SPEC_KV_BYTES_OVERRIDE $MUSE_SPEC_KV_BYTES"'
t "numa bind off: no --numa-bind, connector kept" "0 1" \
  'MUSE_OFFLOAD_NUMA_BIND=0 source $E/offload.env; echo "$(echo "$MUSE_OFFLOAD_SERVE_ARGS" | grep -c -- "--numa-bind") $(echo "$MUSE_OFFLOAD_SERVE_ARGS" | grep -c SimpleCPUOffloadConnector)"'
t "per-rank bytes override reaches the JSON (a multiple of the block lcm stays the caller's duty)" "1" \
  'MUSE_OFFLOAD_PER_RANK_BYTES=26214400000 source $E/offload.env; echo "$(echo "$MUSE_OFFLOAD_SERVE_ARGS" | grep -c "cpu_bytes_to_use_per_rank\":26214400000}")"'
t "ns128 profile: 0.33.0-derived bytes and block size" "rc=0 off=1 kv=17257267200 bpb=65536" \
  'MUSE_OFFLOAD_PROFILE=ns128 source $E/offload.env 2>/dev/null; rc=$?; echo "rc=$rc off=$MUSE_OFFLOAD kv=$MUSE_OFFLOAD_KV_BYTES bpb=$MUSE_OFFLOAD_BPB"'
t "0.33.0 tier: P* is whole blocks of L, spec and ns (2,529,984 / 790,620) and the ns256 profile bytes" "0 2529984 790620 16854810624" \
  'source $E/offload.env; echo "$((MUSE_OFFLOAD_PER_RANK_BYTES % 327680)) $((MUSE_OFFLOAD_PER_RANK_BYTES / 20480)) $((MUSE_OFFLOAD_PER_RANK_BYTES / 65536)) $(MUSE_OFFLOAD_PROFILE=ns256 source $E/offload.env; echo $MUSE_OFFLOAD_KV_BYTES)"'
t "ns128 with an explicit MUSE_OFFLOAD_KV_BYTES_OVERRIDE replaces the bytes" "rc=0 off=1 kv=17000000000 bpb=65536" \
  'MUSE_OFFLOAD_PROFILE=ns128 MUSE_OFFLOAD_KV_BYTES_OVERRIDE=17000000000 source $E/offload.env 2>/dev/null; rc=$?; echo "rc=$rc off=$MUSE_OFFLOAD kv=$MUSE_OFFLOAD_KV_BYTES bpb=$MUSE_OFFLOAD_BPB"'
t "ns128 with the tier off (the OFF arm of the stress / gate boots)" "rc=0 off=0 err=[]" \
  'export MUSE_OFFLOAD=0 MUSE_OFFLOAD_PROFILE=ns128; source $E/offload.env 2>/dev/null; rc=$?; echo "rc=$rc off=$MUSE_OFFLOAD err=[${MUSE_OFFLOAD_ERROR:-}]"'
t "host.env + spec_k7.env alone: offload not enabled" "[] 11720212480" \
  'source $E/host.env; source $E/spec_k7.env; echo "[${MUSE_OFFLOAD_SERVE_ARGS:-}] $MUSE_SPEC_KV_BYTES"'
echo "$fails case(s) failed"
exit $(( fails > 0 ))
