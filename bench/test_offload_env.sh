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
# S1-10R adoption: spec_k7.env sources offload.env itself and re-derives everything, so the operator input must be EXPORTED before spec_k7.env (a one-shot prefix on an earlier source is not carried over)
t "MUSE_OFFLOAD_KV_BYTES_OVERRIDE (exported) replaces the offload bytes everywhere" "11346000000 11346000000 11346000000" \
  'export MUSE_OFFLOAD_KV_BYTES_OVERRIDE=11346000000; source $E/offload.env; source $E/spec_k7.env; echo "$MUSE_OFFLOAD_KV_BYTES $MUSE_SPEC_KV_BYTES_OVERRIDE $MUSE_SPEC_KV_BYTES"'
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
t "MUSE_OFFLOAD=0 with host.env + spec_k7.env alone: offload not enabled, the pre-adoption bytes" "[] 11720212480 0" \
  'export MUSE_OFFLOAD=0; source $E/host.env; source $E/spec_k7.env; echo "[${MUSE_OFFLOAD_SERVE_ARGS:-}] $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD"'
# ---- S1-10R adoption (A1 / A2 / A3): env/spec_k7.env sources env/offload.env for the speculative profile
GATED='--max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 11645337600 --numa-bind --kv-transfer-config {"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":414512578560,"cpu_bytes_to_use_per_rank":51814072320}}'
LINE='echo "$MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD_SERVE_ARGS"'
t "A2: the documented production line equals the gated serve_extra.txt of s110r_fulltier_spec_a5 byte for byte" "$GATED" \
  "source \$E/host.env; source \$E/spec_k7.env; $LINE"
t "A2: the documented line after the gate's explicit offload.env source is the same string" "$GATED" \
  "source \$E/host.env; source \$E/offload.env; source \$E/spec_k7.env; $LINE"
t "A2: MUSE_OFFLOAD=0, the same line is the pre-adoption launch (no connector, no --numa-bind, 11,720,212,480 B)" "--max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 11720212480" \
  "export MUSE_OFFLOAD=0; source \$E/host.env; source \$E/spec_k7.env; echo \$($LINE)"
t "A1: spec_k7.env alone defaults to offload ON: flag, profile, bytes, override" "1 spec 11645337600 11645337600" \
  'source $E/spec_k7.env; echo "$MUSE_OFFLOAD $MUSE_OFFLOAD_PROFILE $MUSE_SPEC_KV_BYTES $MUSE_SPEC_KV_BYTES_OVERRIDE"'
t "A1: a shell that exports MUSE_OFFLOAD_PROFILE=ns256 still gets the spec profile (forced for that source)" "spec 20480 11645337600 11645337600" \
  'export MUSE_OFFLOAD_PROFILE=ns256; source $E/offload.env; source $E/spec_k7.env; echo "$MUSE_OFFLOAD_PROFILE $MUSE_OFFLOAD_BPB $MUSE_OFFLOAD_KV_BYTES $MUSE_SPEC_KV_BYTES"'
t "A1: ns256 sourced first (serve args, override), then spec_k7.env: spec bytes and args" "$GATED" \
  "MUSE_OFFLOAD_PROFILE=ns256 source \$E/offload.env; source \$E/spec_k7.env; $LINE"
t "A1: sourced twice in one shell: same result" "$GATED" \
  "source \$E/spec_k7.env; source \$E/spec_k7.env; $LINE"
t "A1: sourced from another directory with a relative path (BASH_SOURCE, no cwd dependence)" "11645337600" \
  'cd "$E" && cd .. && source env/spec_k7.env; echo $MUSE_SPEC_KV_BYTES'
t "A1: no helper variable is left behind: the shell variables after spec_k7.env equal those of the gated order" "same" \
  'a=$(source $E/offload.env; source $E/spec_k7.env; compgen -v | grep -v -x -E "a|b" | sort | md5sum); b=$(source $E/spec_k7.env; compgen -v | grep -v -x -E "a|b" | sort | md5sum); [ "$a" = "$b" ] && echo same || echo differ'
t "A1: an exported MUSE_OFFLOAD_KV_BYTES_OVERRIDE reaches the bytes through the nested source" "11346000000" \
  'export MUSE_OFFLOAD_KV_BYTES_OVERRIDE=11346000000; source $E/spec_k7.env; echo $MUSE_SPEC_KV_BYTES'
t "A1: a one-shot prefix on an EARLIER offload.env source is not carried over (inputs must be exported)" "11645337600" \
  'MUSE_OFFLOAD_KV_BYTES_OVERRIDE=11346000000 source $E/offload.env; source $E/spec_k7.env; echo $MUSE_SPEC_KV_BYTES'
t "A1: MUSE_OFFLOAD=2 is an error (rc 1), nothing is derived" "rc=1" \
  'export MUSE_OFFLOAD=2; source $E/spec_k7.env 2>/dev/null; echo "rc=$?"'
t "A1: MUSE_OFFLOAD= (empty) is the default ON" "1 11645337600" \
  'export MUSE_OFFLOAD=; source $E/spec_k7.env; echo "$MUSE_OFFLOAD $MUSE_SPEC_KV_BYTES"'
t "A1: MUSE_OFFLOAD=0 sources nothing of offload.env: no MUSE_OFFLOAD_* variable but the switch itself is exported" "MUSE_OFFLOAD" \
  'export MUSE_OFFLOAD=0; source $E/host.env; source $E/spec_k7.env; export -p | grep -o -E "MUSE_OFFLOAD[A-Z_]*" | sort -u | tr "\n" " " | sed "s/ *$//"'
t "A1: MUSE_OFFLOAD=0 after an ON source in the same shell drops the serve args and the override (stale state)" "off=0 args=[] ov=[] kv=11720212480" \
  'source $E/spec_k7.env; export MUSE_OFFLOAD=0; source $E/spec_k7.env; echo "off=$MUSE_OFFLOAD args=[$MUSE_OFFLOAD_SERVE_ARGS] ov=[${MUSE_SPEC_KV_BYTES_OVERRIDE:-}] kv=$MUSE_SPEC_KV_BYTES"'
# review r1 B1: a CHILD shell inherits the ON exports but not offload.env's shell-local marker of its own override
t "B1: child shell, parent ON exports, MUSE_OFFLOAD=0: no override, no serve args, the no-tier bytes" "off=0 args=[] ov=[] kv=11720212480" \
  'source $E/host.env; source $E/spec_k7.env; MUSE_OFFLOAD=0 bash -c "source \$E/host.env; source \$E/spec_k7.env; echo \"off=\$MUSE_OFFLOAD args=[\$MUSE_OFFLOAD_SERVE_ARGS] ov=[\${MUSE_SPEC_KV_BYTES_OVERRIDE:-}] kv=\$MUSE_SPEC_KV_BYTES\""'
t "B1: same with export MUSE_OFFLOAD=0 in the parent after the ON source" "off=0 args=[] ov=[] kv=11720212480" \
  'source $E/host.env; source $E/spec_k7.env; export MUSE_OFFLOAD=0; bash -c "source \$E/host.env; source \$E/spec_k7.env; echo \"off=\$MUSE_OFFLOAD args=[\$MUSE_OFFLOAD_SERVE_ARGS] ov=[\${MUSE_SPEC_KV_BYTES_OVERRIDE:-}] kv=\$MUSE_SPEC_KV_BYTES\""'
t "B1: child shell, parent ON exports, MUSE_OFFLOAD=0: the documented line is the pre-adoption launch" "--max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 11720212480" \
  "CH='source \$E/host.env; source \$E/spec_k7.env; echo \$($LINE)'; source \$E/host.env; source \$E/spec_k7.env; export MUSE_OFFLOAD=0; bash -c \"\$CH\""
t "B1: child shell with ON exports and MUSE_OFFLOAD unset: still ON (the gated line)" "$GATED" \
  "CH='source \$E/host.env; source \$E/spec_k7.env; $LINE'; source \$E/host.env; source \$E/spec_k7.env; bash -c \"\$CH\""
t "B1: an operator override that differs from the ON bytes survives MUSE_OFFLOAD=0 in a child shell with ON exports" "ov=11346923520 kv=11346923520" \
  'source $E/host.env; source $E/spec_k7.env; export MUSE_OFFLOAD=0 MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520; bash -c "source \$E/spec_k7.env; echo \"ov=\$MUSE_SPEC_KV_BYTES_OVERRIDE kv=\$MUSE_SPEC_KV_BYTES\""'
# review r1 M3: the cap > 256 warning (bytes not derived) is printed on the ON path too, and not for an operator override
t "M3: MUSE_SPEC_GRAPH_CAP=512 with the tier on prints the 'not derived' warning" "1" \
  'export MUSE_SPEC_GRAPH_CAP=512; source $E/spec_k7.env 2>&1 >/dev/null | grep -c "not derived"'
t "M3: MUSE_SPEC_GRAPH_CAP=512 with MUSE_OFFLOAD=0 and an operator override prints no warning" "0" \
  'export MUSE_OFFLOAD=0 MUSE_SPEC_GRAPH_CAP=512 MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520; source $E/spec_k7.env 2>&1 >/dev/null | grep -c "not derived"'
t "A1: MUSE_OFFLOAD=0 keeps an operator MUSE_SPEC_KV_BYTES_OVERRIDE without offload.env (cap 384 launches)" "11346923520 11346923520" \
  'export MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520 MUSE_OFFLOAD=0; MUSE_SPEC_GRAPH_CAP=384 source $E/spec_k7.env; echo "${MUSE_SPEC_KV_BYTES_OVERRIDE:-} $MUSE_SPEC_KV_BYTES"'
# S1-11 review minors: m1 fixed in code, m3 / m4 closed by pinning the behaviour
t "m1: a child shell that inherits the ON exports and runs ON again prints no 'discarded' NOTE (the value is the profile's own)" "0" \
  'source $E/host.env; source $E/spec_k7.env; bash -c "source \$E/host.env; source \$E/spec_k7.env" 2>&1 >/dev/null | grep -c discarded'
t "m1: an operator override that differs from the ON bytes still prints the NOTE (and is discarded)" "1 11645337600" \
  'export MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520; n=$(source $E/spec_k7.env 2>&1 >/dev/null | grep -c discarded); source $E/spec_k7.env 2>/dev/null; echo "$n $MUSE_SPEC_KV_BYTES"'
t "m1: the exports of a child ON-after-ON equal those of the parent (only the NOTE changed)" "same" \
  'source $E/host.env; source $E/spec_k7.env; a=$(export -p | grep -v -E "SHLVL|_=" | sort | md5sum); b=$(bash -c "source \$E/host.env; source \$E/spec_k7.env; export -p" 2>/dev/null | grep -v -E "SHLVL|_=" | sort | md5sum); [ "$a" = "$b" ] && echo same || echo differ'
t "m3 (closed by test): an explicit operator override equal to the ON bytes with MUSE_OFFLOAD=0 is honoured (ON bytes without the tier; launch/lib.sh and both post-boot checks refuse it)" "off=0 args=[] kv=11645337600" \
  'export MUSE_OFFLOAD=0 MUSE_SPEC_KV_BYTES_OVERRIDE=11645337600; source $E/spec_k7.env 2>/dev/null; echo "off=$MUSE_OFFLOAD args=[${MUSE_OFFLOAD_SERVE_ARGS:-}] kv=$MUSE_SPEC_KV_BYTES"'
t "m4 (closed by test): MUSE_OFFLOAD=0 with MUSE_OFFLOAD_PROFILE=ns256 exported: the launch-relevant values equal the plain OFF launch (extra MUSE_OFFLOAD_* exports are not launch-relevant)" "kv=11720212480 args=[] sargs=--max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192" \
  'export MUSE_OFFLOAD=0 MUSE_OFFLOAD_PROFILE=ns256; source $E/host.env; source $E/spec_k7.env 2>/dev/null; echo "kv=$MUSE_SPEC_KV_BYTES args=[${MUSE_OFFLOAD_SERVE_ARGS:-}] sargs=$MUSE_SPEC_SERVE_ARGS"'
# A3 literal proofs: every exported variable (export -p, sorted) of the default launch equals the gated order's; the OFF launch equals main 942acddc48's
exps() { env -i PATH="$PATH" "${@:2}" bash -c "$1" 2>/dev/null | sort; }
if [ "$(exps "source $here/env/host.env; source $here/env/spec_k7.env; export -p")" = "$(exps "source $here/env/host.env; source $here/env/offload.env; source $here/env/spec_k7.env; export -p")" ]; then
  echo "ok   A3: every exported variable of host.env + spec_k7.env (default) equals the gated host.env + offload.env + spec_k7.env"
else echo "FAIL A3: default and gated exports differ"; fails=$((fails+1)); fi
if [ "$(exps "export MUSE_OFFLOAD_PROFILE=ns256; source $here/env/host.env; source $here/env/spec_k7.env; export -p")" = "$(exps "source $here/env/host.env; source $here/env/offload.env; source $here/env/spec_k7.env; export -p")" ]; then
  echo "ok   A3: same with a shell that exports MUSE_OFFLOAD_PROFILE=ns256"
else echo "FAIL A3: ns256 shell and gated exports differ"; fails=$((fails+1)); fi
mt=$(mktemp -d)
if git -C "$here" cat-file -e 942acddc48^{commit} 2>/dev/null && git -C "$here" archive 942acddc48 env | tar -x -C "$mt"; then
  if [ "$(exps "export MUSE_OFFLOAD=0; source $here/env/host.env; source $here/env/spec_k7.env; export -p")" = "$(exps "export MUSE_OFFLOAD=0; source $mt/env/host.env; source $mt/env/spec_k7.env; export -p")" ]; then
    echo "ok   A3: MUSE_OFFLOAD=0 exports exactly what main 942acddc48 exports (host.env + spec_k7.env)"
  else echo "FAIL A3: MUSE_OFFLOAD=0 exports differ from main 942acddc48"; fails=$((fails+1)); fi
else echo "skip A3: main 942acddc48 not available in this checkout (the main history must stay reachable for this proof)"; fi
rm -rf "$mt"
echo "$fails case(s) failed"
exit $(( fails > 0 ))
