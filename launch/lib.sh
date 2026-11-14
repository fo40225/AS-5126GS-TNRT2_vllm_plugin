#!/bin/bash
# Shared flow of the production launch scripts (launch/serve_spec.sh, launch/serve_nonspec.sh). Sourced, never run.
#
# What a launch script does, in this order:
#   1. reads its operator inputs (MUSE_OFFLOAD and the MUSE_LAUNCH_* hooks below), then scrubs every inherited MUSE_* / VLLM_* / NCCL_* / SPEC_* export (and
#      CUDA_VISIBLE_DEVICES) of the calling shell, so a dirty shell cannot change the launch (the gated boots started from a clean slate the same way; --keep-env
#      opts out). The scrub is backed by a hard post-boot check (check env): the live environment of the server, filtered like the gated boots' pid_environ.txt,
#      must equal the one a clean `env -i` dry run of the same script derives (with --keep-env: the launching shell's own record);
#   2. exports the runtime block of the gated boots (venv, caches, PYTHONPATH = host folder + vLLM tree) and sources env/host.env and the profile files;
#   3. builds the serve line and writes serve_cmd.txt and env.txt exactly like the gated boots (printf %q of every argument; the same env filter);
#   4. refuses to start when the GPUs are busy or, for a tier-ON launch, scripts/preflight_offload.sh fails;
#   5. starts the server in its own session (launch/lc_child.sh records the child's own pid / session / start time in server.sess after setsid), waits for /health, saves the
#      READY facts (cache_config_info.txt, pid_environ.txt, swap record);
#   6. runs the post-boot checks (env, line = the live command line / serve_cmd.txt / a fresh clean-shell derivation / vLLM's parsed arguments, then the profile checks). A failed
#      check stops the server (its recorded session) and exits 1, unless --keep-on-fail was given.
# On success the server keeps running (detached, own session); stop it with launch/stop_server.sh <out dir>.
#
# Exit codes: 0 = READY and every check passed (server left running); 1 = a post-boot check failed (server stopped unless --keep-on-fail);
#             2 = refused to start (bad input, busy GPUs, failed preflight, env file error): nothing was started; 3 = the server never became READY
#             (died, timed out or was interrupted; stopped); 64 = usage.
# Every MUSE_LAUNCH_* hook that is set is logged as a HOOK line in status.txt (a stale export such as a preflight stub is visible in the record).
# --dry-run builds everything up to the serve line and writes serve_cmd.txt and env.txt, then exits 0 without any GPU, venv, preflight or server step.
#
# Inputs read BEFORE the scrub (shell variables of this script, never exported):
#   MUSE_OFFLOAD            0 | 1 (empty = the profile default: spec 1, ns256 1, ns128 0)
#   MUSE_LAUNCH_VLLM        the vllm executable (default vllm); the serve_cmd.txt always names `vllm`
#   MUSE_LAUNCH_PORT        moves only the health / metrics probe URL (tests); the serve line always carries LAUNCH_PORT of env/launch.env
#   MUSE_LAUNCH_POLL_S      seconds between /health polls (default 5)
#   MUSE_LAUNCH_READY_S     ready timeout in seconds (default LAUNCH_READY_TIMEOUT_S of env/launch.env)
#   MUSE_LAUNCH_PREFLIGHT   preflight script (default scripts/preflight_offload.sh)
#   MUSE_LAUNCH_SMI         nvidia-smi executable (default nvidia-smi)
#   MUSE_LAUNCH_VENV        venv activate script (default $LAUNCH_DEV/.venv/bin/activate; `none` = do not activate, tests)
#   MUSE_LAUNCH_STOP_ITERS  polls to wait for the TERM before the KILL (default 60), MUSE_LAUNCH_STOP_POLL_S seconds apart (default 2)
#   MUSE_LAUNCH_SETSID      the setsid executable (default setsid; tests delay it to widen the race window of the session capture)
#   MUSE_LAUNCH_SESS_WAIT_S seconds to wait for the server's own session record (default 30)
#   MUSE_LAUNCH_GPU_TRIES   tries (MUSE_LAUNCH_GPU_WAIT_S = 5 s apart) for the GPUs to be free (< 2000 MiB in use in total) before the launch is refused (default 90)

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/session.sh"

lc_usage() {
  echo "usage: $1 $([ "$2" = nonspec ] && echo "<ns128|ns256> ")[--out DIR] [--keep-on-fail] [--keep-env] [--dry-run]" >&2
  exit 64
}

# lc_init <spec|nonspec> <script args...>: parse the arguments, capture the operator inputs, scrub, locate the host folder
lc_init() {
  LC_KIND=$1; shift
  LC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  LC_PROFILE=spec
  if [ "$LC_KIND" = nonspec ]; then
    case "${1:-}" in ns128|ns256) LC_PROFILE=$1; shift ;; *) lc_usage serve_nonspec.sh nonspec ;; esac
  fi
  LC_OUT=; LC_KEEP_FAIL=0; LC_KEEP_ENV=0; LC_DRY=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --out) [ $# -ge 2 ] || lc_usage "$0" "$LC_KIND"; LC_OUT=$2; shift 2 ;;
      --keep-on-fail) LC_KEEP_FAIL=1; shift ;;
      --keep-env) LC_KEEP_ENV=1; shift ;;
      --dry-run) LC_DRY=1; shift ;;
      *) lc_usage "$0" "$LC_KIND" ;;
    esac
  done
  LC_OFFLOAD_IN=${MUSE_OFFLOAD-}
  LC_HOOKS=$(env | grep -a -E '^MUSE_LAUNCH_' | LC_ALL=C sort | tr '\n' ' ')
  LC_VLLM=${MUSE_LAUNCH_VLLM:-vllm}
  LC_PROBE_PORT=${MUSE_LAUNCH_PORT:-}
  LC_POLL_S=${MUSE_LAUNCH_POLL_S:-5}
  LC_READY_S_IN=${MUSE_LAUNCH_READY_S:-}
  LC_PREFLIGHT=${MUSE_LAUNCH_PREFLIGHT:-$LC_ROOT/scripts/preflight_offload.sh}
  LC_SMI=${MUSE_LAUNCH_SMI:-nvidia-smi}
  LC_VENV_IN=${MUSE_LAUNCH_VENV-}
  LC_SETSID=${MUSE_LAUNCH_SETSID:-setsid}
  LC_SESS_WAIT_S=${MUSE_LAUNCH_SESS_WAIT_S:-30}
  LC_STOP_ITERS=${MUSE_LAUNCH_STOP_ITERS:-60}
  LC_STOP_POLL_S=${MUSE_LAUNCH_STOP_POLL_S:-2}
  LC_GPU_TRIES=${MUSE_LAUNCH_GPU_TRIES:-90}
  LC_GPU_WAIT_S=${MUSE_LAUNCH_GPU_WAIT_S:-5}
  case "$LC_OFFLOAD_IN" in ""|0|1) ;; *) echo "launch: MUSE_OFFLOAD must be 0 or 1, got '$LC_OFFLOAD_IN'" >&2; exit 2 ;; esac
  if [ "$LC_KEEP_ENV" = 0 ]; then
    local v
    # every VLLM_* and NCCL_* export goes (the script sets the ones it needs itself, after this point), not only a list of named ones
    for v in $(compgen -e | grep -E '^(_MUSE_|MUSE_|VLLM_|NCCL_|SPEC_|R2E_|FP8_F|FCSHARD_)'); do unset "$v"; done
    unset PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES
  fi
  # the variables the env files read: a shell variable of ours, never an export
  unset HOST
  HOST=$LC_ROOT
}

# lc_clean_derive <dir>: a dry run of the same script in a clean shell (env -i, the operator's MUSE_OFFLOAD only) into <dir>: the serve line, env.txt and
# pid_environ_expected.txt derived from the env files instead of the launching shell. rc 1 when the derivation fails.
lc_clean_derive() {
  local script=serve_spec.sh parg=() ref=$1
  if [ "$LC_KIND" = nonspec ]; then script=serve_nonspec.sh; parg=("$LC_PROFILE"); fi
  rm -rf "$ref"
  env -i PATH="$PATH" HOME="${HOME:-/}" ${LC_OFFLOAD_IN:+MUSE_OFFLOAD=$LC_OFFLOAD_IN} bash "$LC_ROOT/launch/$script" "${parg[@]}" --dry-run --out "$ref" > "$ref.stdout" 2>&1 || return 1
  [ -s "$ref/pid_environ_expected.txt" ] && [ -s "$ref/serve_cmd.txt" ]
}

# lc_clean_environ: the environment the server must inherit, derived from the env files in a clean shell. Prints the file name; returns 1 when the derivation fails.
lc_clean_environ() {
  lc_clean_derive "$LC_OUT/clean_ref" || return 1
  echo "$LC_OUT/clean_ref/pid_environ_expected.txt"
}

# lc_check_env <reference> <live>: the live environment of the server (filtered) equals the reference exactly; prints the difference otherwise
lc_check_env() {
  local d
  d=$(diff "$1" "$2") && { echo "OK live environ = reference ($(wc -l < "$1") lines)"; return 0; }
  echo "FAIL live environment of the server differs from the clean-shell derivation (< expected, > live):"; echo "$d"
  return 1
}

# lc_log <message>: status line to stdout and to $LC_OUT/status.txt (once the output directory exists)
lc_log() {
  local m; m="$(date '+%F %T') $*"
  echo "$m"
  [ -n "${LC_STATUS:-}" ] && echo "$m" >> "$LC_STATUS"
}

lc_refuse() { lc_log "REFUSED $*"; [ -n "${LC_STATUS:-}" ] && echo "RESULT refused" >> "$LC_STATUS"; exit 2; }

# the process environment the server will inherit, like the gated boots (venv, caches, host folder + vLLM tree on PYTHONPATH)
lc_runtime_env() {
  LC_TREE=$LAUNCH_TREE
  if [ "$LC_DRY" = 0 ]; then
    local venv=${LC_VENV_IN:-$LAUNCH_DEV/.venv/bin/activate}
    if [ "$venv" != none ]; then
      [ -r "$venv" ] || lc_refuse "venv activate script $venv is not readable"
      # shellcheck disable=SC1090
      source "$venv"
    fi
  fi
  export PYTHONPATH=$HOST:$LC_TREE
  export FLASHINFER_WORKSPACE_BASE=$LAUNCH_DEV/fi_workspace TRITON_CACHE_DIR=$LAUNCH_DEV/triton_cache
  export CUTE_DSL_CACHE_DIR=$LAUNCH_DEV/cute_cache VLLM_CACHE_ROOT=$LAUNCH_DEV/vllm_cache TORCHINDUCTOR_CACHE_DIR=$LAUNCH_DEV/inductor_cache
  export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
  export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=$LAUNCH_ENGINE_READY_TIMEOUT_S
}

# lc_source_env <profile>: host.env, launch.env and the profile files; sets LC_TIER (1 | 0), LC_KVB and the profile arrays
lc_source_env() {
  LAUNCH_ENV_FILE=$LC_ROOT/env/launch.env
  [ -r "$LAUNCH_ENV_FILE" ] || lc_refuse "cannot read $LAUNCH_ENV_FILE"
  # shellcheck disable=SC1090
  source "$LAUNCH_ENV_FILE"
  lc_runtime_env
  # shellcheck disable=SC1091
  source "$LC_ROOT/env/host.env"
  if [ "$LC_KIND" = spec ]; then
    export VLLM_SERVER_DEV_MODE=$LAUNCH_SPEC_DEV_MODE
    [ -n "$LC_OFFLOAD_IN" ] && export MUSE_OFFLOAD=$LC_OFFLOAD_IN
    # shellcheck disable=SC1091
    source "$LC_ROOT/env/spec_k7.env" || lc_refuse "env/spec_k7.env failed"
    LC_TIER=$MUSE_OFFLOAD
    LC_KVB=$MUSE_SPEC_KV_BYTES
    LC_MNS=$LAUNCH_SPEC_MNS
    LC_CAP=
  else
    local p=${LC_PROFILE^^}
    eval "LC_MNS=\$LAUNCH_${p}_MNS LC_CAP=\$LAUNCH_${p}_CAP LC_KVB_OFF=\$LAUNCH_${p}_KV_BYTES LC_TIER=\$LAUNCH_${p}_OFFLOAD_DEFAULT"
    [ -n "$LC_OFFLOAD_IN" ] && LC_TIER=$LC_OFFLOAD_IN
    if [ "$LC_TIER" = 1 ]; then
      export MUSE_OFFLOAD_PROFILE=$LC_PROFILE
      MUSE_OFFLOAD=1
      # shellcheck disable=SC1091
      source "$LC_ROOT/env/offload.env" || lc_refuse "env/offload.env failed: ${MUSE_OFFLOAD_ERROR:-no error text}"
      LC_KVB=$MUSE_OFFLOAD_KV_BYTES
    else
      # no tier: env/offload.env is not sourced at all (even with MUSE_OFFLOAD=0 it exports the ON bytes), so the environment is host.env alone
      LC_KVB=$LC_KVB_OFF
    fi
  fi
}

# lc_on_bytes <profile>: the offload-on bytes of a profile, from env/offload.env in a clean shell (an exported override of this shell cannot move them)
lc_on_bytes() { env -i PATH="$PATH" bash -c "MUSE_OFFLOAD=1 MUSE_OFFLOAD_PROFILE=$1; source '$LC_ROOT/env/offload.env' && echo \$MUSE_OFFLOAD_KV_BYTES" 2> /dev/null; }

# lc_pairing: the exact bytes and the tier travel together (S1-10R); refuse a line that would break the pairing
lc_pairing() {
  local p b
  if [ "$LC_TIER" = 1 ]; then
    [ -n "${MUSE_OFFLOAD_SERVE_ARGS:-}" ] || lc_refuse "tier ON but \$MUSE_OFFLOAD_SERVE_ARGS is empty"
    b=$(lc_on_bytes "$LC_PROFILE")
    [ -n "$b" ] && [ "$LC_KVB" -gt "$b" ] && lc_refuse "tier ON with --kv-cache-memory-bytes $LC_KVB above the offload-on bytes $b of $LC_PROFILE"
  else
    [ -z "${MUSE_OFFLOAD_SERVE_ARGS:-}" ] || lc_refuse "tier OFF but \$MUSE_OFFLOAD_SERVE_ARGS is set"
    for p in spec ns128 ns256; do
      [ "$LC_KVB" = "$(lc_on_bytes $p)" ] && lc_refuse "tier OFF with --kv-cache-memory-bytes $LC_KVB, the offload-on bytes of $p (the bytes only fit with the tier)"
    done
  fi
  return 0
}

# lc_build_serve: LC_SERVE, the argument array of the gated boots (`vllm` first, so printf %q reproduces serve_cmd.txt)
lc_build_serve() {
  LC_SERVE=(vllm serve "$LAUNCH_MODEL" --tensor-parallel-size "$LAUNCH_TP" --max-model-len "$LAUNCH_MAX_MODEL_LEN"
    --gpu-memory-utilization "$LAUNCH_GPU_UTIL" --host "$LAUNCH_HOST" --port "$LAUNCH_PORT" --enable-expert-parallel
    --max-num-seqs "$LC_MNS" --kv-cache-dtype "$LAUNCH_KV_DTYPE" --quantization "$LAUNCH_QUANT"
    --limit-mm-per-prompt "$LAUNCH_LIMIT_MM" $LAUNCH_PARSER_ARGS)
  if [ "$LC_KIND" = spec ]; then
    LC_SERVE+=(--speculative-config
      "{\"draft_sample_method\":\"$LAUNCH_SPEC_DRAFT_SAMPLE\",\"method\":\"$LAUNCH_SPEC_METHOD\",\"model\":\"$LAUNCH_SPEC_DRAFTER\",\"num_speculative_tokens\":$LAUNCH_SPEC_K,\"rejection_sample_method\":\"$LAUNCH_SPEC_REJECTION\"}"
      $LAUNCH_SPEC_EXTRA_ARGS $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes "$LC_KVB" ${MUSE_OFFLOAD_SERVE_ARGS:-})
  else
    [ -n "$LC_CAP" ] && LC_SERVE+=(--max-cudagraph-capture-size "$LC_CAP")
    LC_SERVE+=(--kv-cache-memory-bytes "$LC_KVB" ${MUSE_OFFLOAD_SERVE_ARGS:-})
  fi
}

# lc_records: serve_cmd.txt and env.txt, written like the gated boots
lc_records() {
  printf '%q ' "${LC_SERVE[@]}" > "$LC_OUT/serve_cmd.txt"; echo >> "$LC_OUT/serve_cmd.txt"
  env | grep -E '^(_MUSE_|MUSE_|VLLM_MUSE|NCCL_|VLLM_PLUGINS|VLLM_SERVER|PYTHONPATH)' | LC_ALL=C sort > "$LC_OUT/env.txt"
  # what pid_environ.txt (the live environment of the server, saved after READY) must equal: the same filter on this environment
  env | grep -E '^(_MUSE_|MUSE_|VLLM_|NCCL_|PYTHONPATH|HF_HUB_OFFLINE|CUDA_HOME)' | LC_ALL=C sort > "$LC_OUT/pid_environ_expected.txt"
}

lc_swap_used() { awk '/^SwapTotal/ {t=$2} /^SwapFree/ {f=$2} END {printf "%.1f", (t - f) / 1024}' /proc/meminfo; }
lc_vm() { awk -v k="$1" '$1 == k {print $2}' /proc/vmstat; }
lc_gpus_used() { "$LC_SMI" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'; }
lc_gpus_free() {
  local i u
  for i in $(seq 1 "${1:-90}"); do
    u=$(lc_gpus_used 2> /dev/null)
    [ -n "$u" ] && [ "$u" -lt 2000 ] && return 0
    sleep "$LC_GPU_WAIT_S"
  done
  return 1
}

# lc_await_session: waits for the server's own session record (launch/lc_child.sh writes it after setsid); sets LC_SID from it. rc 0 recorded and plausible,
# 1 the child died or no record appeared in time, 2 a record that is not a session of its own (the stop then signals the pid alone, never a group).
lc_await_session() {
  local n=$(( LC_SESS_WAIT_S * 10 )) i
  for i in $(seq 1 "$n"); do
    if sx_read_record "$LC_OUT/server.sess"; then
      if [ "$SX_PID" = "$LC_SPID" ] && [ "$SX_SID" = "$SX_PID" ] && [ "$SX_PGRP" = "$SX_PID" ] && sx_guard "$SX_SID"; then LC_SID=$SX_SID; return 0; fi
      LC_ODD_PID=$SX_PID
      return 2
    fi
    kill -0 "$LC_SPID" 2> /dev/null || { sx_read_record "$LC_OUT/server.sess" && continue; return 1; }
    sleep 0.1
  done
  return 1
}

# lc_kill_pid <pid>: TERM then KILL of one process we started, no group, no sweep (used when the server's session is not known)
lc_kill_pid() {
  local i
  kill -TERM "$1" 2> /dev/null
  for i in $(seq 1 "$LC_STOP_ITERS"); do kill -0 "$1" 2> /dev/null || return 0; sleep "$LC_STOP_POLL_S"; done
  kill -KILL "$1" 2> /dev/null
}

# lc_stop: TERM the recorded server session, then KILL it and its leftovers (never a pattern kill). Only the session in $LC_OUT/server.sess is signalled, and only when
# its leader is the recorded pid with the recorded start time and the session does not contain this launcher or its parent (launch/session.sh); without a usable
# record only the pid we started is signalled.
lc_stop() {
  [ -n "${LC_SPID:-}" ] || return 0
  local msg
  if [ -n "${LC_SID:-}" ]; then
    msg=$(sx_stop "$LC_OUT/server.sess" "$LC_STOP_ITERS" "$LC_STOP_POLL_S")
    [ -n "$msg" ] && lc_log "STOP $(echo "$msg" | tr '\n' '|')"
  else
    lc_kill_pid "$LC_SPID"
    [ -n "${LC_ODD_PID:-}" ] && [ "$LC_ODD_PID" != "$LC_SPID" ] && lc_kill_pid "$LC_ODD_PID"
    lc_log "STOP no usable session record: signalled pid $LC_SPID only (no session or group)"
  fi
  LC_STOPPED=1
  lc_gpus_free 90 || lc_log "WARN GPU memory still in use after the stop"
  lc_log "STOPPED gpu_used_mib=$(lc_gpus_used 2> /dev/null)"
}

lc_on_exit() {
  if [ "${LC_ARMED:-0}" = 1 ] && [ "${LC_STOPPED:-0}" = 0 ]; then
    lc_log "EXIT while the server was not handed over: stopping it"
    lc_stop
    echo "RESULT aborted" >> "$LC_STATUS"
  fi
}

# lc_check <name> <command...>: a hard post-boot check, output in $LC_OUT/check_<name>.txt
lc_check() {
  local name=$1 rc; shift
  "$@" > "$LC_OUT/check_$name.txt" 2>&1; rc=$?
  if [ $rc = 0 ]; then lc_log "CHK $name PASS $(tail -n 1 "$LC_OUT/check_$name.txt" | cut -c1-300)"
  else lc_log "CHK $name FAIL rc=$rc $(tail -n 1 "$LC_OUT/check_$name.txt" | cut -c1-300)"; LC_NFAIL=$((LC_NFAIL + 1)); fi
}

# lc_main: the whole flow after lc_init + lc_source_env
lc_main() {
  lc_pairing
  lc_build_serve
  local name=$LC_PROFILE
  [ -n "$LC_OUT" ] || LC_OUT=$PWD/launch_${name}_tier${LC_TIER}_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$LC_OUT" || { echo "launch: cannot create $LC_OUT" >&2; exit 2; }
  # absolute before any cd: the server subshell does cd "$LC_OUT" and then opens "$LC_OUT/serve.log" (a relative --out would resolve twice)
  LC_OUT=$(cd "$LC_OUT" && pwd) || { echo "launch: cannot enter $LC_OUT" >&2; exit 2; }
  LC_STATUS=$LC_OUT/status.txt
  lc_records
  lc_log "START profile=$name tier=$LC_TIER bytes=$LC_KVB mns=$LC_MNS tree=$LC_TREE host=$HOST out=$LC_OUT dry_run=$LC_DRY"
  [ -z "$LC_HOOKS" ] || lc_log "HOOK test hooks set in the calling shell (they replace the real vllm / preflight / nvidia-smi / venv): $LC_HOOKS"
  if [ "$LC_DRY" = 1 ]; then
    lc_log "DRYRUN serve_cmd.txt and env.txt written; no preflight, no GPU, no server"
    echo "RESULT dryrun" >> "$LC_STATUS"
    return 0
  fi
  # the reference environment of the server: a clean-shell derivation (a leak of the calling shell cannot hide in it; the serve line is compared the same way); --keep-env: this shell's own record
  if [ "$LC_KEEP_ENV" = 0 ]; then
    LC_ENV_REF=$(lc_clean_environ) || lc_refuse "cannot derive the clean-shell environment (see $LC_OUT/clean_ref.stdout)"
    cmp -s "$LC_ENV_REF" "$LC_OUT/pid_environ_expected.txt" || lc_refuse "the scrubbed environment of this shell differs from the clean-shell derivation (diff $LC_ENV_REF $LC_OUT/pid_environ_expected.txt)"
    cmp -s "$LC_OUT/clean_ref/serve_cmd.txt" "$LC_OUT/serve_cmd.txt" || lc_refuse "the serve line built in this shell differs from the clean-shell derivation (diff $LC_OUT/clean_ref/serve_cmd.txt $LC_OUT/serve_cmd.txt)"
  else
    LC_ENV_REF=$LC_OUT/pid_environ_expected.txt
    lc_log "KEEP_ENV the environment check compares with the launching shell's own record, not with a clean shell"
  fi
  # the node-oom protection recommended by preflight_offload.sh: an OOM kill must hit this session first
  echo 1000 > /proc/self/oom_score_adj 2> /dev/null
  lc_gpus_free "$LC_GPU_TRIES" || lc_refuse "gpus_busy used_mib=$(lc_gpus_used 2> /dev/null)"
  if [ "$LC_TIER" = 1 ]; then
    bash "$LC_PREFLIGHT" > "$LC_OUT/preflight.txt" 2>&1; local prc=$?
    lc_log "PREFLIGHT rc=$prc $(tail -n 1 "$LC_OUT/preflight.txt" | cut -c1-200)"
    [ $prc = 0 ] || lc_refuse "preflight_offload.sh rc=$prc: the host memory cannot take the tier (nothing started; MUSE_OFFLOAD=0 for a launch without the tier)"
  else
    lc_log "PREFLIGHT skipped (no tier)"
  fi
  local port=${LC_PROBE_PORT:-$LAUNCH_PORT} url ready_s=${LC_READY_S_IN:-$LAUNCH_READY_TIMEOUT_S}
  url=http://$LAUNCH_HOST:$port
  local sw0 v0in v0out t0 state=LOADING
  sw0=$(lc_swap_used); v0in=$(lc_vm pswpin); v0out=$(lc_vm pswpout)
  t0=$(date +%s)
  LC_STOPPED=0; LC_ARMED=1; LC_NFAIL=0
  trap lc_on_exit EXIT
  trap 'lc_log "SIGNAL received"; lc_stop; echo "RESULT aborted signal" >> "$LC_STATUS"; exit 3' TERM INT HUP
  rm -f "$LC_OUT/server.sess"
  (cd "$LC_OUT" && exec "$LC_SETSID" bash "$LC_ROOT/launch/lc_child.sh" "$LC_OUT/server.sess" "$LC_VLLM" "${LC_SERVE[@]:1}" > "$LC_OUT/serve.log" 2>&1 < /dev/null) &
  LC_SPID=$!
  echo "$LC_SPID" > "$LC_OUT/serve.pid"
  # the child records its own session after setsid; reading it with ps right after the fork could return this launcher's session
  lc_await_session; local arc=$?
  if [ $arc != 0 ]; then
    lc_log "LAUNCHED pid=$LC_SPID out=$LC_OUT"
    lc_log "SESSION_UNKNOWN rc=$arc: no usable session record of the server in ${LC_SESS_WAIT_S}s (child alive: $(kill -0 "$LC_SPID" 2> /dev/null && echo yes || echo no))"
    lc_stop
    echo "RESULT fail SESSION_UNKNOWN" >> "$LC_STATUS"
    LC_ARMED=0
    exit 3
  fi
  lc_log "LAUNCHED pid=$LC_SPID sid=$LC_SID out=$LC_OUT"
  while :; do
    if curl -sf -o /dev/null "$url/health"; then state=READY; break; fi
    if ! kill -0 "$LC_SPID" 2> /dev/null; then state=DIED; break; fi
    [ $(( $(date +%s) - t0 )) -gt "$ready_s" ] && { state=TIMEOUT; break; }
    sleep "$LC_POLL_S"
  done
  local load_s=$(( $(date +%s) - t0 ))
  lc_log "$state load_s=$load_s"
  if [ $state != READY ]; then
    lc_stop
    echo "RESULT fail $state" >> "$LC_STATUS"
    LC_ARMED=0
    exit 3
  fi
  lc_ready_facts "$url" "$sw0" "$v0in" "$v0out"
  lc_checks
  if [ "$LC_NFAIL" != 0 ]; then
    if [ "$LC_KEEP_FAIL" = 1 ]; then
      lc_log "FAILED $LC_NFAIL post-boot check(s); --keep-on-fail: the server keeps running (pid $LC_SPID, stop with launch/stop_server.sh $LC_OUT)"
      echo "RESULT fail checks kept" >> "$LC_STATUS"
      LC_ARMED=0
    else
      lc_log "FAILED $LC_NFAIL post-boot check(s): stopping the server"
      lc_stop
      echo "RESULT fail checks" >> "$LC_STATUS"
      LC_ARMED=0
    fi
    exit 1
  fi
  LC_ARMED=0
  trap - EXIT TERM INT HUP
  lc_log "READY and every post-boot check passed: $name tier=$LC_TIER, server pid $LC_SPID (own session), out $LC_OUT; stop with launch/stop_server.sh $LC_OUT"
  echo "RESULT pass" >> "$LC_STATUS"
}

# lc_ready_facts <url> <swap at launch> <pswpin at launch> <pswpout at launch>
lc_ready_facts() {
  local url=$1 sw0=$2 vi=$3 vo=$4 sw1
  sw1=$(lc_swap_used)
  lc_log "SWAP launch_mib=$sw0 ready_mib=$sw1 pswpin_delta=$(( $(lc_vm pswpin) - vi )) pswpout_delta=$(( $(lc_vm pswpout) - vo ))"
  "$LC_SMI" --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits > "$LC_OUT/gpu_mem_ready.txt" 2> /dev/null
  curl -s --max-time 20 "$url/metrics" | grep -a -E '^vllm:cache_config_info' | head -2 > "$LC_OUT/cache_config_info.txt"
  tr '\0' ' ' < "/proc/$LC_SPID/cmdline" > "$LC_OUT/pid_cmdline.txt" 2> /dev/null
  tr '\0' '\n' < "/proc/$LC_SPID/environ" 2> /dev/null | grep -a -E '^(_MUSE_|MUSE_|VLLM_|NCCL_|PYTHONPATH|HF_HUB_OFFLINE|CUDA_HOME)' | LC_ALL=C sort > "$LC_OUT/pid_environ.txt"
  lc_log "READY_FACTS gpu_mem_ready.txt cache_config_info.txt pid_cmdline.txt pid_environ.txt ($(wc -l < "$LC_OUT/pid_environ.txt") environ lines)"
}

# lc_check_line: the serve line the server really runs (review r2 m1). The live /proc/<pid>/cmdline equals serve_cmd.txt, serve_cmd.txt equals the line a clean env -i dry run
# re-derives from the env files NOW (the pre-start derivation could be stale; --keep-env: not compared, the shell's own line is the reference), and vLLM's "non-default args"
# line carries exactly these options with these values (scripts/check_serve_line.py).
lc_check_line() {
  local ref=()
  if [ "$LC_KEEP_ENV" = 0 ]; then
    lc_clean_derive "$LC_OUT/clean_check" || { echo "FAIL cannot re-derive the serve line in a clean shell (see $LC_OUT/clean_check.stdout)"; return 1; }
    ref=(--derived "$LC_OUT/clean_check/serve_cmd.txt")
  else
    echo "NOTE --keep-env: serve_cmd.txt is not compared with a clean-shell derivation"
  fi
  python3 -I "$LC_ROOT/scripts/check_serve_line.py" --serve-cmd "$LC_OUT/serve_cmd.txt" --pid "$LC_SPID" --log "$LC_OUT/serve.log" "${ref[@]}"
}

# lc_checks: the post-boot checks of the profile (expectations come from this shell's env files)
lc_checks() {
  local log=$LC_OUT/serve.log
  lc_check env lc_check_env "$LC_ENV_REF" "$LC_OUT/pid_environ.txt"
  lc_check line lc_check_line
  if [ "$LC_KIND" = spec ]; then
    lc_check spec env SPEC_EXPECT_FB=1 SPEC_EXPECT_OFFLOAD="$MUSE_OFFLOAD" bash "$LC_ROOT/scripts/check_spec_launch.sh" "$log"
  fi
  if [ "$LC_TIER" = 1 ]; then
    lc_check offload bash "$LC_ROOT/scripts/check_offload_launch.sh" "$log"
  else
    lc_check offload env MUSE_OFFLOAD_EXPECT=off bash "$LC_ROOT/scripts/check_offload_launch.sh" "$log"
  fi
  lc_check kv bash "$LC_ROOT/scripts/check_kv_launch.sh" "$LC_PROFILE" "$LC_TIER" "$log"
}
