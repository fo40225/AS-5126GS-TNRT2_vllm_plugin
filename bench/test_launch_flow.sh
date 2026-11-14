#!/bin/bash
# Regression test of the launch scripts launch/serve_spec.sh, launch/serve_nonspec.sh and launch/stop_server.sh (S1-11) with a STUBBED server: every profile with both
# MUSE_OFFLOAD values, the refusals (preflight rc 1 / 2, busy GPUs, bad input), a server that dies or never becomes healthy, a signal while waiting, every post-boot
# check failing, --keep-on-fail and the pairing guard. No GPU, no vLLM: the stub (bench/fixtures/launch_stub/vllm) prints the check-relevant lines of the real boot logs
# (bench/fixtures/launch/) and answers /health and /metrics. Usage  bash bench/test_launch_flow.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
fx=$here/bench/fixtures
stub=$fx/launch_stub
d=$(mktemp -d)
fails=0
port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
cleanup() {
  local f
  for f in "$d"/*/serve.pid; do [ -r "$f" ] && bash "$here/launch/stop_server.sh" "$(dirname "$f")" > /dev/null 2>&1; done
  rm -rf "$d"
}
trap cleanup EXIT

pass() { echo "ok   $1"; }
bad() { echo "FAIL $1"; shift; for l in "$@"; do echo "    $l"; done; fails=$((fails + 1)); }
alive() { kill -0 "$(cat "$d/$1/serve.pid" 2> /dev/null)" 2> /dev/null; }

# run <name> <script> [env assignments...] -- <script args...>: runs a launch script from a clean shell with the stub hooks; sets rc and out=$d/<name>
run() {
  local name=$1 script=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -rf "${d:?}/$name"
  out=$d/$name
  env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none \
    MUSE_LAUNCH_PORT=$port MUSE_LAUNCH_POLL_S=0.2 MUSE_LAUNCH_READY_S=60 MUSE_LAUNCH_STOP_POLL_S=0.2 STUB_PORT=$port STUB_MARK="$d/$name.mark" "${envs[@]}" \
    bash "$here/launch/$script" "$@" --out "$out" > "$d/$name.stdout" 2>&1
  rc=$?
}
result() { grep -a "^RESULT" "$d/$1/status.txt" 2> /dev/null | tail -n 1; }

# expect_ok <name> <desc> <script> <profile arg or ''> <state> <tier> [env assignments...]: a full launch that must pass and leave a stoppable server
expect_ok() {
  local name=$1 desc=$2 script=$3 parg=$4 state=$5 tier=$6; shift 6
  run "$name" "$script" STUB_STATE="$state" "$@" -- $parg
  local ok=1 msgs=()
  [ $rc = 0 ] || { ok=0; msgs+=("rc=$rc wanted 0"); }
  [ "$(result "$name")" = "RESULT pass" ] || { ok=0; msgs+=("status: $(result "$name")"); }
  alive "$name" || { ok=0; msgs+=("the server is not running after a passing launch"); }
  local f
  for f in serve_cmd.txt env.txt pid_environ.txt cache_config_info.txt serve.pid status.txt; do [ -s "$out/$f" ] || { ok=0; msgs+=("missing $f"); }; done
  # the line the server was started with and the environment it inherited are the gated boot's (checkout path -> host_topo mirror, the one recorded normalization)
  sed "s#$here#/raid/vllm-runtime/host_topo#g" "$out/serve_cmd.txt" | cmp -s - "$fx/launch_ref/${state}.serve_cmd.txt" || { ok=0; msgs+=("serve_cmd.txt differs from the gated boot's"); }
  cmp -s "$out/pid_environ.txt" "$out/pid_environ_expected.txt" || { ok=0; msgs+=("the live environment of the server differs from the script's"); }
  [ "$tier" = 1 ] && { grep -q 'PREFLIGHT rc=0' "$out/status.txt" || { ok=0; msgs+=("no preflight line on a tier-ON launch"); }; }
  [ "$tier" = 0 ] && { grep -q 'PREFLIGHT skipped' "$out/status.txt" || { ok=0; msgs+=("preflight ran on a tier-OFF launch"); }; }
  local pid; pid=$(cat "$out/serve.pid" 2> /dev/null)
  bash "$here/launch/stop_server.sh" "$out" > /dev/null 2>&1
  kill -0 "$pid" 2> /dev/null && { ok=0; msgs+=("stop_server.sh left pid $pid running"); }
  if [ $ok = 1 ]; then pass "$desc"; else bad "$desc" "${msgs[@]}"; fi
}

# expect_fail <name> <desc> <rc wanted> <message regex or ''> <server running afterwards: 0|1> <script> <profile arg or ''> <env assignments...> -- <more script args>
expect_fail() {
  local name=$1 desc=$2 want=$3 msg=$4 live=$5 script=$6 parg=$7; shift 7
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  run "$name" "$script" "${envs[@]}" -- $parg "$@"
  local ok=1 msgs=()
  [ $rc = "$want" ] || { ok=0; msgs+=("rc=$rc wanted $want"); }
  if [ -n "$msg" ] && ! cat "$d/$name.stdout" "$out/status.txt" 2> /dev/null | grep -q -E -- "$msg"; then ok=0; msgs+=("no line matching '$msg'"); fi
  if [ "$live" = 1 ]; then alive "$name" || { ok=0; msgs+=("the server should still run (--keep-on-fail)"); }
  elif [ -r "$out/serve.pid" ] && alive "$name"; then ok=0; msgs+=("the server was left running after a failed launch"); fi
  [ "$live" = 0 ] && [ -s "$d/$name.mark" ] && [ "$want" = 2 ] && { ok=0; msgs+=("a refused launch started the server"); }
  if [ $ok = 1 ]; then pass "$desc"; else bad "$desc" "${msgs[@]}" "stdout: $(tail -n 3 "$d/$name.stdout" | tr '\n' '|' | cut -c1-300)"; fi
  [ "$live" = 1 ] && bash "$here/launch/stop_server.sh" "$out" > /dev/null 2>&1
}

# ---- every profile, both MUSE_OFFLOAD values (the default, the explicit value, the other value)
expect_ok spec_def "spec, MUSE_OFFLOAD unset: tier ON, the adoption line, checks pass" serve_spec.sh "" spec_1 1
expect_ok spec_on "spec, MUSE_OFFLOAD=1" serve_spec.sh "" spec_1 1 MUSE_OFFLOAD=1
expect_ok spec_off "spec, MUSE_OFFLOAD=0: no tier, no preflight, the previous launch" serve_spec.sh "" spec_0 0 MUSE_OFFLOAD=0
expect_ok ns256_def "ns256, MUSE_OFFLOAD unset: tier ON" serve_nonspec.sh ns256 ns256_1 1
expect_ok ns256_on "ns256, MUSE_OFFLOAD=1" serve_nonspec.sh ns256 ns256_1 1 MUSE_OFFLOAD=1
expect_ok ns256_off "ns256, MUSE_OFFLOAD=0: the 0.33.0 validation line" serve_nonspec.sh ns256 ns256_0 0 MUSE_OFFLOAD=0
expect_ok ns128_def "ns128, MUSE_OFFLOAD unset: tier OFF (S1-11 Amendment 2)" serve_nonspec.sh ns128 ns128_0 0
expect_ok ns128_off "ns128, MUSE_OFFLOAD=0" serve_nonspec.sh ns128 ns128_0 0 MUSE_OFFLOAD=0
expect_ok ns128_on "ns128, MUSE_OFFLOAD=1 opts in: the adoption line" serve_nonspec.sh ns128 ns128_1 1 MUSE_OFFLOAD=1
# a dirty calling shell must not change the launch (the gated boots started from a clean slate)
expect_ok dirty "spec from a dirty shell (MUSE_SPEC_MNST=0, FUSED_BOUNDARY=0, NCCL_P2P_LEVEL, SPEC_EXPECT_FB=0, an inherited ON override): the adoption line" serve_spec.sh "" spec_1 1 \
  MUSE_SPEC_MNST=0 VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=0 NCCL_P2P_LEVEL=NVL SPEC_EXPECT_FB=0 MUSE_OFFLOAD_PER_RANK_BYTES=1 MUSE_TOPO_ENABLE=0 VLLM_PLUGINS=x _MUSE_FOO=1
expect_ok dirty2 "ns256 from a dirty shell with unnamed NCCL_* / VLLM_* exports (NCCL_ALGO, NCCL_DEBUG, VLLM_ATTENTION_BACKEND, VLLM_LOGGING_LEVEL, CUDA_VISIBLE_DEVICES): the adoption line and environment" serve_nonspec.sh ns256 ns256_1 1 \
  NCCL_ALGO=Ring NCCL_DEBUG=INFO NCCL_PROTO=Simple VLLM_ATTENTION_BACKEND=X VLLM_LOGGING_LEVEL=DEBUG CUDA_VISIBLE_DEVICES=0
grep -q 'CHK env PASS' "$d/dirty2/status.txt" && pass "  ... and the environment check passed" || bad "  ... and the environment check passed" "$(grep CHK "$d/dirty2/status.txt")"

# the stub hooks are logged as HOOK lines (a stale MUSE_LAUNCH_* export is visible in the record)
grep -q 'HOOK .*MUSE_LAUNCH_PREFLIGHT=' "$d/spec_def/status.txt" && pass "MUSE_LAUNCH_* hooks are logged as a HOOK line in status.txt" || bad "no HOOK line in status.txt"

# ---- refusals: nothing is started
expect_fail pf1 "preflight rc 1 refuses the ON launch (spec), nothing started" 2 'REFUSED preflight_offload.sh rc=1' 0 serve_spec.sh "" STUB_STATE=spec_1 STUB_PREFLIGHT_RC=1 --
expect_fail pf2 "preflight rc 2 refuses the ON launch (ns256)" 2 'REFUSED preflight_offload.sh rc=2' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_1 STUB_PREFLIGHT_RC=2 --
expect_fail pf3 "preflight rc 1 refuses ns128 with MUSE_OFFLOAD=1" 2 'REFUSED preflight' 0 serve_nonspec.sh ns128 STUB_STATE=ns128_1 STUB_PREFLIGHT_RC=1 MUSE_OFFLOAD=1 --
expect_ok pf4 "a failing preflight does not matter without the tier (spec MUSE_OFFLOAD=0)" serve_spec.sh "" spec_0 0 MUSE_OFFLOAD=0 STUB_PREFLIGHT_RC=1
expect_ok pf5 "a failing preflight does not matter for ns128 at its default (no tier)" serve_nonspec.sh ns128 ns128_0 0 STUB_PREFLIGHT_RC=1
expect_fail busy "busy GPUs refuse the launch" 2 'REFUSED gpus_busy' 0 serve_spec.sh "" STUB_STATE=spec_1 STUB_GPU_MIB=50000 MUSE_LAUNCH_GPU_TRIES=2 MUSE_LAUNCH_GPU_WAIT_S=0.1 --
expect_fail badoff "MUSE_OFFLOAD=2 is refused (spec)" 2 'MUSE_OFFLOAD must be 0 or 1' 0 serve_spec.sh "" MUSE_OFFLOAD=2 --
expect_fail badoff2 "MUSE_OFFLOAD=yes is refused (ns256)" 2 'MUSE_OFFLOAD must be 0 or 1' 0 serve_nonspec.sh ns256 MUSE_OFFLOAD=yes --
expect_fail usage1 "serve_nonspec.sh without a profile is a usage error" 64 'usage' 0 serve_nonspec.sh "" --
expect_fail usage2 "serve_nonspec.sh with an unknown profile is a usage error" 64 'usage' 0 serve_nonspec.sh ns512 --
expect_fail usage3 "an unknown flag is a usage error" 64 'usage' 0 serve_spec.sh "" -- --bogus
# the pairing guard (review m3): the ON bytes never go to a launch without the tier, and the tier never goes with bytes above the table
expect_fail pair1 "m3: --keep-env with MUSE_OFFLOAD=0 and an explicit override equal to the ON bytes is refused" 2 'the offload-on bytes of spec' 0 serve_spec.sh "" \
  STUB_STATE=spec_0 MUSE_OFFLOAD=0 MUSE_SPEC_KV_BYTES_OVERRIDE=11645337600 -- --keep-env
expect_fail pair2 "--keep-env, tier ON, an offload-on bytes override above the table is refused" 2 'above the offload-on bytes' 0 serve_spec.sh "" \
  STUB_STATE=spec_1 MUSE_OFFLOAD_KV_BYTES_OVERRIDE=11700000000 -- --keep-env
expect_fail pair3 "--keep-env, ns128 without the tier but MUSE_OFFLOAD_SERVE_ARGS in the shell is refused" 2 'MUSE_OFFLOAD_SERVE_ARGS is set' 0 serve_nonspec.sh ns128 \
  STUB_STATE=ns128_0 'MUSE_OFFLOAD_SERVE_ARGS=--numa-bind' -- --keep-env

# ---- the server does not come up
expect_fail die "the server dies before READY: exit 3, nothing left" 3 'DIED' 0 serve_spec.sh "" STUB_STATE=spec_1 STUB_MODE=die --
expect_fail hang "the server never becomes healthy: timeout, exit 3, stopped" 3 'TIMEOUT' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_1 STUB_MODE=hang MUSE_LAUNCH_READY_S=2 --

# a signal (TERM, INT, HUP) while waiting for READY stops the server (own session) and exits 3 (review r2 m3: HUP like TERM / INT)
for sig in TERM INT HUP; do
  rm -rf "${d:?}/sig_$sig"
  (env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none MUSE_LAUNCH_PORT=$port \
    MUSE_LAUNCH_POLL_S=0.2 MUSE_LAUNCH_READY_S=60 MUSE_LAUNCH_STOP_POLL_S=0.2 STUB_PORT=$port STUB_STATE=spec_1 STUB_MODE=hang bash "$here/launch/serve_spec.sh" --out "$d/sig_$sig" > "$d/sig_$sig.stdout" 2>&1; echo $? > "$d/sig_$sig.rc") &
  bgpid=$!
  for _ in $(seq 1 50); do [ -s "$d/sig_$sig/serve.pid" ] && break; sleep 0.2; done
  sigpid=$(cat "$d/sig_$sig/serve.pid" 2> /dev/null)
  launcher=$(ps -eo pid,args | awk -v o="$d/sig_$sig" 'index($0, "serve_spec.sh") && index($0, o) && !index($0, "awk") {print $1; exit}')
  [ -n "$launcher" ] && kill -$sig "$launcher"
  wait $bgpid
  if [ "$(cat "$d/sig_$sig.rc" 2> /dev/null)" = 3 ] && ! kill -0 "$sigpid" 2> /dev/null && grep -q 'SIGNAL received' "$d/sig_$sig/status.txt"; then pass "SIG$sig to the launcher while waiting: server stopped, exit 3"
  else bad "SIG$sig to the launcher while waiting" "rc=$(cat "$d/sig_$sig.rc" 2> /dev/null) launcher=$launcher server alive=$(kill -0 "$sigpid" 2> /dev/null && echo yes || echo no)"; fi
done

# ---- every post-boot check fails on its own (a mutated real log): the server is stopped and the exit is 1
expect_fail ck_kv "spec: the KV check fails (tokens differ): exit 1, server stopped" 1 'CHK kv FAIL' 0 serve_spec.sh "" STUB_STATE=spec_1 'STUB_LOG_SED=s/1,639,394/1,639,395/' --
grep -q 'CHK spec PASS' "$d/ck_kv/status.txt" && grep -q 'CHK offload PASS' "$d/ck_kv/status.txt" && pass "  ... and only the KV check failed" || bad "  ... and only the KV check failed" "$(grep CHK "$d/ck_kv/status.txt")"
expect_fail ck_spec "spec: check_spec_launch fails (sequence-parallel floor)" 1 'CHK spec FAIL' 0 serve_spec.sh "" STUB_STATE=spec_1 'STUB_LOG_SED=s/batches of at least 384 tokens/batches of at least 400 tokens/' --
grep -q 'CHK offload PASS' "$d/ck_spec/status.txt" && grep -q 'CHK kv PASS' "$d/ck_spec/status.txt" && pass "  ... and only the spec check failed" || bad "  ... and only the spec check failed" "$(grep CHK "$d/ck_spec/status.txt")"
expect_fail ck_off "spec: check_offload_launch fails (no NUMA binding lines)" 1 'CHK offload FAIL' 0 serve_spec.sh "" STUB_STATE=spec_1 'STUB_LOG_SED=/Binding worker subprocess/d' --
expect_fail ck_off2 "spec: check_offload_launch fails (the log has no connector although the tier is ON)" 1 'CHK offload FAIL' 0 serve_spec.sh "" 'STUB_STATE=spec_0' --
expect_fail ck_off3 "spec OFF: check_offload_launch fails (a connector line in a tier-less boot)" 1 'CHK offload FAIL' 0 serve_spec.sh "" STUB_STATE=spec_0 MUSE_OFFLOAD=0 'STUB_LOG_SED=$a SimpleCPUOffloadScheduler: Allocating 5 offload blocks' --
expect_fail ck_ns_kv "ns256: the KV check fails (blocks differ)" 1 'CHK kv FAIL' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_1 'STUB_CCI_SED=s/num_gpu_blocks="257184"/num_gpu_blocks="1"/' --
expect_fail ck_ns_cap "ns256 OFF: the KV check fails (the parsed capture cap differs)" 1 'CHK kv FAIL' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_0 MUSE_OFFLOAD=0 "STUB_LOG_SED=s/'max_cudagraph_capture_size': 256/'max_cudagraph_capture_size': 512/" --
expect_fail ck_ns_mns "ns128: the KV check fails (parsed max_num_seqs differs)" 1 'CHK kv FAIL' 0 serve_nonspec.sh ns128 STUB_STATE=ns128_0 "STUB_LOG_SED=s/'max_num_seqs': 128/'max_num_seqs': 256/" --
expect_fail ck_ns_off "ns128 OFF: check_offload_launch fails (a connector line in a tier-less boot)" 1 'CHK offload FAIL' 0 serve_nonspec.sh ns128 STUB_STATE=ns128_0 'STUB_LOG_SED=$a SimpleCPUOffloadScheduler: Allocating 5 offload blocks' --
expect_fail ck_ns_on "ns128 ON: check_offload_launch fails (workers disagree with the scheduler)" 1 'CHK offload FAIL' 0 serve_nonspec.sh ns128 MUSE_OFFLOAD=1 STUB_STATE=ns128_1 'STUB_LOG_SED=0,/790620 CPU blocks/s//790619 CPU blocks/' --
# the state of the boot must match the launch: the tier-ON fixture under a tier-OFF launch fails (bytes, tokens, connector)
expect_fail ck_mix "ns256 launched without the tier but the boot logged the tier-ON numbers" 1 'CHK (kv|offload) FAIL' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_1 MUSE_OFFLOAD=0 --

# ---- the live environment of the server is compared with the clean-shell derivation (review B1): an extra, a changed and a missing variable each fail it alone
expect_fail env1 "spec: the server process carries an extra NCCL_ALGO: only the environment check fails (exit 1, server stopped)" 1 'CHK env FAIL' 0 serve_spec.sh "" STUB_STATE=spec_1 STUB_EXPORT=NCCL_ALGO=Ring --
grep -q 'CHK spec PASS' "$d/env1/status.txt" && grep -q 'CHK offload PASS' "$d/env1/status.txt" && grep -q 'CHK kv PASS' "$d/env1/status.txt" && grep -q '^> NCCL_ALGO=Ring' "$d/env1/check_env.txt" && pass "  ... and only the environment check failed, naming the variable" || bad "  ... and only the environment check failed" "$(grep CHK "$d/env1/status.txt")"
expect_fail env2 "ns256: a changed value (VLLM_PLUGINS) in the server process fails the environment check" 1 'CHK env FAIL' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_1 STUB_EXPORT=VLLM_PLUGINS=other --
expect_fail env3 "ns128: a variable missing in the server process (NCCL_P2P_LEVEL) fails the environment check" 1 'CHK env FAIL' 0 serve_nonspec.sh ns128 STUB_STATE=ns128_0 STUB_UNSET=NCCL_P2P_LEVEL --
# --keep-env: the reference is the launching shell's own record, so its exports are expected in the server
expect_ok keepenv "--keep-env: an inherited NCCL_ALGO is part of the launch and of the reference record (the environment check passes)" serve_spec.sh "--keep-env" spec_1 1 NCCL_ALGO=Ring
grep -q 'KEEP_ENV' "$d/keepenv/status.txt" && grep -q '^NCCL_ALGO=Ring' "$d/keepenv/pid_environ.txt" && pass "  ... and the record says so" || bad "  ... and the record says so"

# ---- --keep-on-fail
expect_fail keep "--keep-on-fail: a failed check exits 1 and the server keeps running (pid in serve.pid); stop_server.sh stops it" 1 'FAILED 1 post-boot' 1 serve_nonspec.sh ns256 \
  STUB_STATE=ns256_1 'STUB_CCI_SED=s/num_gpu_blocks="257184"/num_gpu_blocks="1"/' -- --keep-on-fail
pid=$(cat "$d/keep/serve.pid"); kill -0 "$pid" 2> /dev/null && bad "stop_server.sh stops the kept server" "pid $pid still running" || pass "stop_server.sh stopped the kept server"
expect_fail keep2 "--keep-on-fail does not keep a server that never became healthy" 3 'TIMEOUT' 0 serve_spec.sh "" STUB_STATE=spec_1 STUB_MODE=hang MUSE_LAUNCH_READY_S=2 -- --keep-on-fail

# ---- review r2 m1: the serve line the server really runs. Live /proc/<pid>/cmdline = serve_cmd.txt = the clean-shell derivation at check time, and vLLM's own parse of
# the arguments (the "non-default args" line) carries the same options. Every mutation below changes ONE of the three and must fail the line check (exit 1, server stopped).
grep -q 'CHK line PASS' "$d/spec_def/status.txt" && grep -q 'CHK line PASS' "$d/ns128_on/status.txt" && pass "the unmutated launches pass the serve-line check (CHK line PASS)" || bad "CHK line PASS missing" "$(grep CHK "$d/spec_def/status.txt")"
grep -q 'NOTE --keep-env' "$d/keepenv/check_line.txt" && pass "--keep-env: the serve-line check runs against the launching shell's own line (noted, no derivation)" || bad "--keep-env serve-line check" "$(cat "$d/keepenv/check_line.txt")"
K6='s/(num_speculative_tokens[^0-9]*)7/\16/'
line_case() { # <name> <desc> <regex> <env assignments...>   (spec launch; line_case_ns for ns256)
  local name=$1 desc=$2 re=$3; shift 3
  expect_fail "$name" "$desc" 1 'CHK line FAIL' 0 serve_spec.sh "" STUB_STATE=spec_1 "$@" --
  grep -q -E -- "$re" "$d/$name/check_line.txt" && pass "  ... naming the difference" || bad "  ... naming the difference ($re)" "$(cat "$d/$name/check_line.txt")"
}
line_case ln_k_cmd "m1: the live command line has num_speculative_tokens 6 instead of 7 (serve_cmd.txt and the log say 7)" 'live command line differs' "STUB_ARG_SED=$K6"
line_case ln_k_log "m1: the startup log's non-default args say num_speculative_tokens 6 (the command line and serve_cmd.txt say 7)" "num_speculative_tokens" "STUB_LOG_SED=s/'num_speculative_tokens': 7/'num_speculative_tokens': 6/"
line_case ln_mnbt_cmd "m1: the live command line has --max-num-batched-tokens 8192 instead of 8199" 'live command line differs' 'STUB_ARG_SED=s/^8199$/8192/'
line_case ln_mnbt_log "m1: the log's non-default args say max_num_batched_tokens 8192" "max_num_batched_tokens" "STUB_LOG_SED=s/'max_num_batched_tokens': 8199/'max_num_batched_tokens': 8192/"
line_case ln_cap_cmd "m1: the live command line has --max-cudagraph-capture-size 512 instead of 256" 'live command line differs' 'STUB_ARG_SED=s/^256$/512/'
line_case ln_cap_log "m1: the log's non-default args say max_cudagraph_capture_size 512" "max_cudagraph_capture_size" "STUB_LOG_SED=s/'max_cudagraph_capture_size': 256/'max_cudagraph_capture_size': 512/"
grep -q 'CHK spec PASS' "$d/ln_k_cmd/status.txt" && grep -q 'CHK offload PASS' "$d/ln_k_cmd/status.txt" && grep -q 'CHK kv PASS' "$d/ln_k_cmd/status.txt" && grep -q 'CHK env PASS' "$d/ln_k_cmd/status.txt" \
  && pass "  ... the other four checks pass on the K=6 command line: only the new check sees it" || bad "  ... only the line check fails (K=6 command line)" "$(grep CHK "$d/ln_k_cmd/status.txt")"
# serve_cmd.txt, the live line and the log all say K=6: only the clean-shell derivation (the env files say 7) can tell
line_case ln_derive "m1: serve_cmd.txt, the live command line and the log all say K=6, the env files derive K=7" 're-derived from the env files' \
  "STUB_SERVECMD_SED=$K6" "STUB_ARG_SED=$K6" "STUB_LOG_SED=s/'num_speculative_tokens': 7/'num_speculative_tokens': 6/"
expect_fail ln_ns "m1: ns256: the live command line has --max-num-seqs 128 instead of 256 and the capture cap 128 (the sed hits both 256 arguments)" 1 'CHK line FAIL' 0 serve_nonspec.sh ns256 STUB_STATE=ns256_1 'STUB_ARG_SED=s/^256$/128/' --
grep -q -E 'live command line differs' "$d/ln_ns/check_line.txt" && pass "  ... naming the difference" || bad "  ... naming the difference" "$(cat "$d/ln_ns/check_line.txt")"
# the serve line built in the launching shell must equal the clean-shell derivation BEFORE anything starts: a copy of the host folder whose launch.env lets an exported
# LAUNCH_PORT through (not scrubbed) is refused with exit 2 and nothing is started
cp_root=$d/hostcopy; mkdir -p "$cp_root"; cp -r "$here/launch" "$here/env" "$here/scripts" "$cp_root/"
sed -i 's/^LAUNCH_PORT=8100$/LAUNCH_PORT=${LAUNCH_PORT:-8100}/' "$cp_root/env/launch.env"
rm -rf "${d:?}/pre_refuse"
env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none MUSE_LAUNCH_PORT=$port \
  STUB_PORT=$port STUB_STATE=spec_1 STUB_MARK="$d/pre_refuse.mark" LAUNCH_PORT=9999 bash "$cp_root/launch/serve_spec.sh" --out "$d/pre_refuse" > "$d/pre_refuse.stdout" 2>&1; rc=$?
if [ $rc = 2 ] && grep -q 'REFUSED the serve line built in this shell differs from the clean-shell derivation' "$d/pre_refuse.stdout" && [ ! -s "$d/pre_refuse.mark" ]; then pass "m1: a serve line that differs from the clean-shell derivation is refused before the start (exit 2, nothing started)"
else bad "m1: pre-start serve-line refusal" "rc=$rc $(tail -n 2 "$d/pre_refuse.stdout" | tr '\n' '|')"; fi

# ---- review r2 M2: the session of the server is the one the child recorded after setsid, never one read with ps before it ran
sentinel_run() { # <name> <seconds to wait before a TERM or 0> <env assignments...>: the launcher runs in its own session next to a sentinel process of that session
  local name=$1 term_after=$2; shift 2
  rm -rf "${d:?}/$name" "${d:?}/$name.rc"; mkdir -p "$d/$name"
  (setsid bash -c 'sleep 300 & echo $! > "$1/sentinel"; shift; exec "$@"' _ "$d/$name" env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" \
    MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none MUSE_LAUNCH_PORT=$port MUSE_LAUNCH_POLL_S=0.2 MUSE_LAUNCH_READY_S=60 MUSE_LAUNCH_STOP_POLL_S=0.2 STUB_PORT=$port \
    STUB_MARK="$d/$name.mark" "$@" bash "$here/launch/serve_spec.sh" --out "$d/$name/out" > "$d/$name.stdout" 2>&1; echo $? > "$d/$name.rc") &
  local bg=$!
  if [ "$term_after" != 0 ]; then
    for _ in $(seq 1 50); do [ -s "$d/$name/out/serve.pid" ] && break; sleep 0.1; done
    sleep "$term_after"
    local l; l=$(ps -eo pid,args | awk -v o="$d/$name/out" 'index($0, "serve_spec.sh") && index($0, o) && !index($0, "awk") {print $1; exit}')
    [ -n "$l" ] && kill -TERM "$l"
  fi
  wait $bg
  sent=$(cat "$d/$name/sentinel" 2> /dev/null)
  sent_alive=no; kill -0 "$sent" 2> /dev/null && sent_alive=yes
  [ -n "$sent" ] && kill "$sent" 2> /dev/null
}
# a late setsid (1.5 s): the old capture read the LAUNCHER's session id in that window
sentinel_run race_ok 0 MUSE_LAUNCH_SETSID="$stub/slow_setsid" STUB_STATE=spec_1
o=$d/race_ok/out; read -r rp rs rg rt < "$o/server.sess"
if [ "$(cat "$d/race_ok.rc")" = 0 ] && grep -q 'RESULT pass' "$o/status.txt" && [ "$rp" = "$(cat "$o/serve.pid")" ] && [ "$rs" = "$rp" ] && [ "$rg" = "$rp" ] \
   && [ "$rs" != "$(ps -o sid= -p "$sent" 2> /dev/null | tr -d ' ')" ] && grep -q "LAUNCHED pid=$rp sid=$rp " "$o/status.txt" && [ "$sent_alive" = yes ]; then pass "M2: a late setsid: the launch passes, server.sess names the server's own session (pid = sid = pgrp), sentinel untouched"
else bad "M2: late setsid, passing launch" "rc=$(cat "$d/race_ok.rc") record='$rp $rs $rg $rt' sentinel=$sent_alive $(grep -a -E 'LAUNCHED|RESULT' "$o/status.txt" | tr '\n' '|')"; fi
bash "$here/launch/stop_server.sh" "$o" > /dev/null 2>&1
alive_pid() { kill -0 "$1" 2> /dev/null && echo yes || echo no; }
# the same late setsid on a failure path (the server dies right after READY would be due): the stop must hit the server's session only
sentinel_run race_die 0 MUSE_LAUNCH_SETSID="$stub/slow_setsid" STUB_STATE=spec_1 STUB_MODE=die
o=$d/race_die/out
if [ "$(cat "$d/race_die.rc")" = 3 ] && grep -q 'RESULT fail DIED' "$o/status.txt" && [ "$sent_alive" = yes ] && [ "$(alive_pid "$(cat "$o/serve.pid")")" = no ]; then pass "M2: a late setsid and a server that dies: exit 3, the launcher's own session (sentinel) is untouched"
else bad "M2: late setsid, failure path" "rc=$(cat "$d/race_die.rc") sentinel=$sent_alive $(grep -a -E 'LAUNCHED|STOP|RESULT' "$o/status.txt" | tr '\n' '|')"; fi
sentinel_run race_hang 0 MUSE_LAUNCH_SETSID="$stub/slow_setsid" STUB_STATE=spec_1 STUB_MODE=hang MUSE_LAUNCH_READY_S=2
o=$d/race_hang/out
if [ "$(cat "$d/race_hang.rc")" = 3 ] && grep -q 'RESULT fail TIMEOUT' "$o/status.txt" && [ "$sent_alive" = yes ] && [ "$(alive_pid "$(cat "$o/serve.pid")")" = no ]; then pass "M2: a late setsid and a server that never becomes healthy: stopped, sentinel untouched"
else bad "M2: late setsid, timeout path" "rc=$(cat "$d/race_hang.rc") sentinel=$sent_alive $(grep -a -E 'LAUNCHED|STOP|RESULT' "$o/status.txt" | tr '\n' '|')"; fi
# TERM inside the setsid delay window (no session record yet): the child pid is signalled alone, nothing else
sentinel_run race_term 0.5 MUSE_LAUNCH_SETSID="$stub/slow_setsid" STUB_SETSID_DELAY=4 STUB_STATE=spec_1 STUB_MODE=hang
o=$d/race_term/out
sleep 0.2
if [ "$(cat "$d/race_term.rc")" = 3 ] && grep -q 'SIGNAL received' "$o/status.txt" && grep -q 'signalled pid .* only' "$o/status.txt" && [ "$sent_alive" = yes ] && [ "$(alive_pid "$(cat "$o/serve.pid")")" = no ] && [ ! -s "$d/race_term.mark" ]; then pass "M2: TERM inside the late-setsid window: exit 3, the child pid alone is stopped, no group signal, sentinel untouched, no server started"
else bad "M2: TERM inside the setsid window" "rc=$(cat "$d/race_term.rc") sentinel=$sent_alive $(grep -a -E 'LAUNCHED|SIGNAL|STOP|RESULT' "$o/status.txt" | tr '\n' '|')"; fi
# the launcher refuses a record that says the server is in the launcher's own session (setsid replaced by a no-op): the pid is stopped alone, the session of the launcher is not signalled
printf '#!/bin/bash\nexec "$@"\n' > "$d/fake_setsid"; chmod +x "$d/fake_setsid"
sentinel_run race_nosetsid 0 MUSE_LAUNCH_SETSID="$d/fake_setsid" STUB_STATE=spec_1 STUB_MODE=hang
o=$d/race_nosetsid/out
if [ "$(cat "$d/race_nosetsid.rc")" = 3 ] && grep -q 'SESSION_UNKNOWN' "$o/status.txt" && grep -q 'signalled pid .* only' "$o/status.txt" && [ "$sent_alive" = yes ] && [ "$(alive_pid "$(cat "$o/serve.pid")")" = no ]; then pass "M2: a server that is not in a session of its own is refused (SESSION_UNKNOWN, exit 3), only its pid is stopped, sentinel untouched"
else bad "M2: no setsid" "rc=$(cat "$d/race_nosetsid.rc") sentinel=$sent_alive $(grep -a -E 'LAUNCHED|SESSION|STOP|RESULT' "$o/status.txt" | tr '\n' '|')"; fi

# stop_server.sh: only a recorded session whose leader matches the recorded pid and start time, never a session that contains the caller
mkdir -p "$d/ss1"
setsid -f bash -c 'exec -a vllm_foreign sleep 120' < /dev/null > /dev/null 2>&1
fpid=$(ps -eo pid,args | awk '$2 == "vllm_foreign" && $3 == "120" {print $1; exit}')
if [ -n "$fpid" ]; then
  fstart=$(sed -E 's/^.*\) //' "/proc/$fpid/stat" | awk '{print $20}'); fsid=$(ps -o sid= -p "$fpid" | tr -d ' ')
  echo "$fpid" > "$d/ss1/serve.pid"
  echo "$fpid $fsid $fsid $((fstart + 1))" > "$d/ss1/server.sess"
  bash "$here/launch/stop_server.sh" "$d/ss1" > "$d/ss1.out" 2>&1; rc=$?
  if [ $rc = 3 ] && kill -0 "$fpid" 2> /dev/null && grep -q 'not the recorded session leader' "$d/ss1.out"; then pass "stop_server.sh: a leader whose start time differs from the record (a recycled pid) is not signalled (exit 3)"
  else bad "stop_server.sh with a wrong start time" "rc=$rc $(cat "$d/ss1.out")"; fi
  echo "$fpid $fsid $fsid $fstart" > "$d/ss1/server.sess"
  bash "$here/launch/stop_server.sh" "$d/ss1" > "$d/ss1.out" 2>&1; rc=$?
  sleep 0.3
  if [ $rc = 0 ] && ! kill -0 "$fpid" 2> /dev/null; then pass "stop_server.sh: the recorded session with the matching start time is stopped (exit 0)"
  else bad "stop_server.sh with the right record" "rc=$rc $(cat "$d/ss1.out")"; kill "$fpid" 2> /dev/null; fi
else bad "stop_server.sh tests: could not start the foreign process"; fi
# a record that names the caller's own session (or its parent's) is refused whatever the process looks like
mkdir -p "$d/ss2"
(exec -a vllm_samesess sleep 120) &
spid2=$!
sleep 0.2
mysid=$(ps -o sid= -p $$ | tr -d ' '); st2=$(sed -E 's/^.*\) //' "/proc/$spid2/stat" | awk '{print $20}')
echo "$spid2" > "$d/ss2/serve.pid"; echo "$spid2 $mysid $mysid $st2" > "$d/ss2/server.sess"
bash "$here/launch/stop_server.sh" "$d/ss2" > "$d/ss2.out" 2>&1; rc=$?
if [ $rc = 3 ] && kill -0 "$spid2" 2> /dev/null && grep -q 'contains this script' "$d/ss2.out"; then pass "stop_server.sh: a session that contains the caller (or its parent) is never signalled (exit 3)"
else bad "stop_server.sh with the caller's session" "rc=$rc $(cat "$d/ss2.out")"; fi
kill "$spid2" 2> /dev/null
# a session whose leader is gone but whose workers live on (a crashed API server): the orphans of this launch (same boot, cwd = out dir, started after the leader) are cleaned one by one
mkdir -p "$d/ss3"
(cd "$d/ss3" && setsid bash "$here/launch/lc_child.sh" "$d/ss3/server.sess" bash -c '(exec -a vllm_orphan sleep 120 > /dev/null 2>&1 &) ; sleep 0.3; exit 0' < /dev/null > /dev/null 2>&1) &
for _ in $(seq 1 30); do [ -s "$d/ss3/server.sess" ] && break; sleep 0.1; done
wait $! 2> /dev/null
sleep 0.2
read -r lp _ < "$d/ss3/server.sess"; echo "$lp" > "$d/ss3/serve.pid"
opid=$(ps -eo pid,args | awk '$2 == "vllm_orphan" {print $1; exit}')
if [ -n "$opid" ] && ! kill -0 "$lp" 2> /dev/null; then
  bash "$here/launch/stop_server.sh" "$d/ss3" > "$d/ss3.out" 2>&1; rc=$?
  sleep 0.3
  if [ $rc = 0 ] && ! kill -0 "$opid" 2> /dev/null; then pass "stop_server.sh: leader gone, the orphaned member of the recorded session that belongs to this launch is stopped"
  else bad "stop_server.sh orphan cleanup" "rc=$rc $(cat "$d/ss3.out")"; kill "$opid" 2> /dev/null; fi
else bad "stop_server.sh orphan setup" "orphan=$opid leader alive=$(kill -0 "$lp" 2> /dev/null && echo yes || echo no)"; [ -n "$opid" ] && kill "$opid" 2> /dev/null; fi

# review p2 r1 B1: a leaderless session is swept only when its members provably belong to the launch. A stale or forged record whose session number is carried by
# an unrelated leaderless session (members with another cwd), a record from another boot, or a record without a boot id must not signal anything (exit 3, members alive).
bootid=$(cat /proc/sys/kernel/random/boot_id)
leaderless() { # <name> <cwd of the member> -> sets lp (the gone leader), mp (the member): a session whose leader exited and whose member keeps running
  local n=$1 cwd=$2; mkdir -p "$d/$n"
  (cd "$cwd" && setsid bash -c '(exec -a vllm_member sleep 120 > /dev/null 2>&1 &) ; sleep 0.3; echo $$ > "$1/leader"; exit 0' _ "$d/$n" < /dev/null > /dev/null 2>&1) &
  for _ in $(seq 1 30); do [ -s "$d/$n/leader" ] && break; sleep 0.1; done
  sleep 0.5
  lp=$(cat "$d/$n/leader"); mp=$(ps -eo pid,sid,args | awk -v s="$lp" '$2 == s && $3 == "vllm_member" {print $1; exit}')
  lst=$(ps -p "$lp" > /dev/null 2>&1 && sed -E 's/^.*\) //' "/proc/$lp/stat" | awk '{print $20}')
  [ -n "$mp" ] && [ -n "$lp" ] && ! kill -0 "$lp" 2> /dev/null
}
mkdir -p "$d/elsewhere"
for variant in foreign_cwd other_boot no_boot; do
  case $variant in foreign_cwd) cwd=$d/elsewhere ;; *) cwd=$d/lg_$variant ;; esac
  mkdir -p "$d/lg_$variant"
  if leaderless "lg_$variant" "$cwd"; then
    # a start time earlier than the member's (a leader that started before it)
    mst=$(sed -E 's/^.*\) //' "/proc/$mp/stat" | awk '{print $20}'); lstart=$((mst - 5))
    echo "$lp" > "$d/lg_$variant/serve.pid"
    case $variant in
      foreign_cwd) echo "$lp $lp $lp $lstart $bootid" > "$d/lg_$variant/server.sess" ;;
      other_boot) echo "$lp $lp $lp $lstart 00000000-0000-0000-0000-000000000000" > "$d/lg_$variant/server.sess" ;;
      no_boot) echo "$lp $lp $lp $lstart" > "$d/lg_$variant/server.sess" ;;
    esac
    bash "$here/launch/stop_server.sh" "$d/lg_$variant" > "$d/lg_$variant.out" 2>&1; rc=$?
    if [ $rc = 3 ] && kill -0 "$mp" 2> /dev/null && grep -q 'NOT SIGNALLED' "$d/lg_$variant.out"; then pass "B1: leaderless session, stale record ($variant): the member is not signalled (exit 3), it is reported"
    else bad "B1: leaderless session, stale record ($variant)" "rc=$rc member alive=$(kill -0 "$mp" 2> /dev/null && echo yes || echo no) $(cat "$d/lg_$variant.out")"; fi
    # and with a correct record (same boot, cwd = out dir) the same member is the launch's own and is stopped
    if [ $variant = other_boot ]; then
      echo "$lp $lp $lp $lstart $bootid" > "$d/lg_$variant/server.sess"
      bash "$here/launch/stop_server.sh" "$d/lg_$variant" > "$d/lg_$variant.out2" 2>&1; rc=$?
      sleep 0.3
      if [ $rc = 0 ] && ! kill -0 "$mp" 2> /dev/null; then pass "B1: the same session with a matching record (boot, cwd = out dir, start time) is swept (exit 0)"
      else bad "B1: matching record sweep" "rc=$rc $(cat "$d/lg_$variant.out2")"; fi
    fi
    kill "$mp" 2> /dev/null
  else bad "B1 setup ($variant)" "leader=$lp member=$mp"; [ -n "$mp" ] && kill "$mp" 2> /dev/null; fi
done

# review p2 r1 P1: a zombie (an unreaped, exited child) is neither alive nor a member to signal
bash -c '
  source "$1/launch/session.sh"
  setsid bash -c "sleep 0.1 & echo \$! > $2/zpid; exec -a vllm_zparent sleep 3" < /dev/null > /dev/null 2>&1 &
  sleep 0.8; zp=$(cat $2/zpid)
  st=$(ps -o stat= -p "$zp" | tr -d " ")
  case "$st" in Z*) ;; *) echo "setup: not a zombie ($st)"; exit 9 ;; esac
  if sx_alive "$zp"; then echo "zombie reported alive"; exit 1; fi
  sid=$(ps -o sid= -p "$zp" | tr -d " "); if sx_members "$sid" | grep -qx "$zp"; then echo "zombie listed as a member"; exit 1; fi
  exit 0' _ "$here" "$d"; zrc=$?
if [ $zrc = 0 ]; then pass "P1: a zombie child is not alive and not a session member (the TERM wait of an INT-aborted launch does not stall)"
else bad "P1: zombie handling" "rc=$zrc"; fi

# review p2 r1 P2: a launcher that runs with job control (bash -m) still records the real server (setsid must not fork away from $!)
rm -rf "${d:?}/jobctl"
env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none \
  MUSE_LAUNCH_PORT=$port MUSE_LAUNCH_POLL_S=0.2 MUSE_LAUNCH_READY_S=60 MUSE_LAUNCH_STOP_POLL_S=0.2 STUB_PORT=$port STUB_STATE=spec_1 STUB_MARK="$d/jobctl.mark" SHELLOPTS=monitor \
  bash -m "$here/launch/serve_spec.sh" --out "$d/jobctl" > "$d/jobctl.stdout" 2>&1; rc=$?
read -r jp js jg jt jb < "$d/jobctl/server.sess" 2> /dev/null
if [ $rc = 0 ] && [ "$jp" = "$(cat "$d/jobctl/serve.pid" 2> /dev/null)" ] && [ "$js" = "$jp" ] && [ "$jg" = "$jp" ] && [ "$jb" = "$bootid" ] && kill -0 "$jp" 2> /dev/null; then
  pass "P2: a launcher with job control on (bash -m): server.sess names the running server (pid = sid = pgrp = serve.pid), boot id recorded"
else bad "P2: job control" "rc=$rc record='$jp $js $jg $jt $jb' serve.pid=$(cat "$d/jobctl/serve.pid" 2> /dev/null) $(grep -a -E 'LAUNCHED|SESSION|RESULT' "$d/jobctl/status.txt" | tr '\n' '|')"; fi
bash "$here/launch/stop_server.sh" "$d/jobctl" > /dev/null 2>&1

# ---- --dry-run: the records only, no server
run dry serve_spec.sh STUB_STATE=spec_1 -- --dry-run
if [ $rc = 0 ] && [ ! -s "$d/dry.mark" ] && [ -s "$d/dry/serve_cmd.txt" ] && [ -s "$d/dry/env.txt" ] && [ ! -e "$d/dry/serve.pid" ] && [ "$(result dry)" = "RESULT dryrun" ]; then pass "--dry-run writes serve_cmd.txt and env.txt and starts nothing"
else bad "--dry-run" "rc=$rc mark=$([ -s "$d/dry.mark" ] && echo yes || echo no)"; fi
# review r2 M1: a relative --out is resolved to an absolute path before the server subshell cds into it (launch and dry run)
mkdir -p "$d/relcwd"
for mode in launch dry; do
  rm -rf "${d:?}/relcwd/relout_$mode"
  rargs=(); [ $mode = dry ] && rargs=(--dry-run)
  (cd "$d/relcwd" && env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none \
    MUSE_LAUNCH_PORT=$port MUSE_LAUNCH_POLL_S=0.2 MUSE_LAUNCH_READY_S=60 MUSE_LAUNCH_STOP_POLL_S=0.2 STUB_PORT=$port STUB_STATE=spec_1 \
    bash "$here/launch/serve_spec.sh" "${rargs[@]}" --out "relout_$mode" > "$d/relcwd/$mode.stdout" 2>&1); rc=$?
  ro=$d/relcwd/relout_$mode
  if [ $mode = launch ]; then
    if [ $rc = 0 ] && [ "$(grep -a '^RESULT' "$ro/status.txt" | tail -n 1)" = "RESULT pass" ] && [ -s "$ro/serve.log" ] && [ ! -e "$ro/relout_$mode" ] && grep -q "out=$ro " "$ro/status.txt"; then pass "relative --out: the launch passes, the files are in the absolute directory (M1)"
    else bad "relative --out (launch)" "rc=$rc $(grep -a '^RESULT' "$ro/status.txt" 2> /dev/null | tail -n 1) $(tail -n 2 "$d/relcwd/$mode.stdout" | tr '\n' '|' | cut -c1-300)"; fi
    bash "$here/launch/stop_server.sh" "$ro" > /dev/null 2>&1
  else
    if [ $rc = 0 ] && [ -s "$ro/serve_cmd.txt" ] && [ -s "$ro/env.txt" ] && grep -q "out=$ro " "$ro/status.txt"; then pass "relative --out: the dry run writes into the absolute directory (M1)"
    else bad "relative --out (dry run)" "rc=$rc"; fi
  fi
done
# stop_server.sh refuses a pid that is not a vllm process
mkdir -p "$d/notvllm"; sleep 30 & echo $! > "$d/notvllm/serve.pid"; spid=$!
bash "$here/launch/stop_server.sh" "$d/notvllm" > /dev/null 2>&1; rc=$?
if [ $rc = 2 ] && kill -0 "$spid" 2> /dev/null; then pass "stop_server.sh does not touch a pid that is not a vllm process"; else bad "stop_server.sh on a foreign pid" "rc=$rc"; fi
kill "$spid" 2> /dev/null

# ---- review p2 r2 F1: the stop path with a missing or tampered session record. The launcher keeps the tuple it accepted: when server.sess is gone it signals that session (the same
# guards apply); when the record disagrees with the leader the stop is refused and the log must not say STOPPED.
freeport() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
f1_launch() { # <name>: starts a hanging stub launch in the background; sets f1pid (the launcher, from this subshell's exec) and waits for the LAUNCHED line with the session
  local name=$1 p; p=$(freeport)
  rm -rf "${d:?}/$name" "${d:?}/$name.rc"
  (exec env -i PATH="$PATH" HOME="$HOME" MUSE_LAUNCH_VLLM="$stub/vllm" MUSE_LAUNCH_SMI="$stub/nvidia-smi" MUSE_LAUNCH_PREFLIGHT="$stub/preflight.sh" MUSE_LAUNCH_VENV=none MUSE_LAUNCH_PORT=$p \
    MUSE_LAUNCH_POLL_S=0.2 MUSE_LAUNCH_READY_S=60 MUSE_LAUNCH_STOP_POLL_S=0.2 STUB_PORT=$p STUB_STATE=spec_1 STUB_MODE=hang bash "$here/launch/serve_spec.sh" --out "$d/$name" > "$d/$name.stdout" 2>&1) &
  f1pid=$!
  for _ in $(seq 1 100); do grep -q -a 'LAUNCHED pid=.* sid=' "$d/$name/status.txt" 2> /dev/null && break; sleep 0.1; done
  grep -q -a 'LAUNCHED pid=.* sid=' "$d/$name/status.txt" 2> /dev/null
}
# 1. the record file is removed after the launcher accepted it; a TERM to the launcher must still stop the server
if f1_launch f1_gone; then
  f1srv=$(cat "$d/f1_gone/serve.pid")
  rm -f "$d/f1_gone/server.sess"
  kill -TERM "$f1pid"; wait "$f1pid"; rc=$?
  if [ $rc = 3 ] && ! kill -0 "$f1srv" 2> /dev/null && grep -q -a 'using the tuple the launcher accepted' "$d/f1_gone/status.txt" && grep -q -a 'STOPPED gpu_used_mib' "$d/f1_gone/status.txt" \
     && ! grep -q -a 'NOT_STOPPED' "$d/f1_gone/status.txt"; then pass "F1: server.sess removed after the launch was accepted: a TERM still stops the server from the accepted tuple (exit 3, STOPPED)"
  else bad "F1: record removed, TERM to the launcher" "rc=$rc server alive=$(kill -0 "$f1srv" 2> /dev/null && echo yes || echo no) $(grep -a -E 'SIGNAL|STOP|RESULT' "$d/f1_gone/status.txt" | tr '\n' '|' | cut -c1-400)"; kill "$f1srv" 2> /dev/null; fi
else bad "F1 setup (record removed)" "no LAUNCHED line"; kill "$f1pid" 2> /dev/null; fi
# 2. the record is rewritten with another start time (a stale / forged record): the stop is refused, nothing is signalled and the log must not claim STOPPED
if f1_launch f1_forged; then
  f1srv=$(cat "$d/f1_forged/serve.pid")
  cp "$d/f1_forged/server.sess" "$d/f1_forged.sess.good"
  read -r a1 a2 a3 a4 a5 < "$d/f1_forged/server.sess"
  echo "$a1 $a2 $a3 $((a4 + 1)) $a5" > "$d/f1_forged/server.sess"
  kill -TERM "$f1pid"; wait "$f1pid"; rc=$?
  if [ $rc = 3 ] && kill -0 "$f1srv" 2> /dev/null && grep -q -a 'not the recorded session leader' "$d/f1_forged/status.txt" && grep -q -a 'NOT_STOPPED' "$d/f1_forged/status.txt" \
     && ! grep -q -a 'STOPPED gpu_used_mib' "$d/f1_forged/status.txt"; then pass "F1: a record that disagrees with the leader: the stop is refused, the log says NOT_STOPPED and not STOPPED, the server was not signalled (exit 3)"
  else bad "F1: forged record, TERM to the launcher" "rc=$rc server alive=$(kill -0 "$f1srv" 2> /dev/null && echo yes || echo no) $(grep -a -E 'SIGNAL|STOP|RESULT' "$d/f1_forged/status.txt" | tr '\n' '|' | cut -c1-400)"; fi
  cp "$d/f1_forged.sess.good" "$d/f1_forged/server.sess"
  bash "$here/launch/stop_server.sh" "$d/f1_forged" > /dev/null 2>&1
  kill -0 "$f1srv" 2> /dev/null && kill "$f1srv" 2> /dev/null
else bad "F1 setup (forged record)" "no LAUNCHED line"; kill "$f1pid" 2> /dev/null; fi

echo "$fails case(s) failed"
exit $(( fails > 0 ))
