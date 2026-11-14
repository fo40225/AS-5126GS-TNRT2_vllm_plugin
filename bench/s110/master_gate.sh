#!/usr/bin/env bash
# S1-10 GATE master: runs the boots listed in a RUNS file, one after the other, inside ONE GPU-lock hold taken by the caller:
#   setsid nohup flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock env GATE_TOOLS=<dir> bash master_gate.sh <runs_file> > master.out 2>&1 < /dev/null &
# (nothing in here takes the lock again; serve_sd.sh is called directly.) Successor of master_s110.sh: the launch line is assembled from the host environment files
# themselves (env/offload.env, env/spec_k7.env), so the gate boots exactly the documented recipe.
# A RUNS line is a set of shell assignments (blank lines and #-lines ignored):
#   tag=s110g_spec_ON_1 profile=spec|ns256|ns128 arm=CTL|OFF|ON jobs="facts steady decode reuse"
#   steady_cells="1:2,16:1,64:1,128:1" decode_c="1 16 64 128" reuse_args="--passes 3" rag_args="" fulltier_args="" preempt_args="" smoke_len=65536
#   extra="<more serve args, no spaces inside JSON>" xenv="KEY=VAL ..." (no spaces in a value; @OUT@ = the result dir)  wd=<watchdog %> wdmin=<watchdog MiB>
#   xboot=1 kv=<exact --kv-cache-memory-bytes of the stress boot> reps=<2> phases=<stress phases>: an S1-7 stress boot on the probe tree (exact_boot_s110.sh)
# Arms (the same GPU bytes rule as the pre-registered ADOPTION_RULE_FINAL.txt):
#   CTL = production tree (sm120-dev/src) + production host mirror (host_topo), no offload;
#   OFF = gate tree (src_s110g = production + 9388ed1133) + gate host mirror (host_topo_s110) with MUSE_OFFLOAD=0 sourced (no bind, no connector, S1-9 bytes);
#   ON  = gate tree + gate host mirror, env/offload.env sourced before env/spec_k7.env: --numa-bind + SimpleCPUOffloadConnector, 45 GiB per rank, stepped bytes.
# Exact GPU bytes per (profile, arm) are asserted against the table below before the launch (a mismatch aborts the boot, nothing is started).
# Per boot: node-memory sampler (2 s), GPU NVML sampler, a watchdog that TERMs the server session when the unreclaimable memory of ANY node reaches WD percent or
# MemFree + file cache of any node falls below WDMIN MiB, and raises oom_score_adj of the session processes to 1000 (an OOM kill must hit this session, never another
# user's process); postcheck = launch checks (check_spec_launch.sh, check_offload_launch.sh), OOM count, READY seconds. File $W/STOP ends the hold at the next boot.
set -u
RUNS_FILE=${1:?usage: master_gate.sh <runs_file>}
B=/raid/vllm-runtime/sm120-dev
W=/raid/vllm-runtime/work/s1-10
T=${GATE_TOOLS:-$W/gate}
H=/raid/vllm-runtime/work/sd/harness
R=${RESULTS_ROOT:-/raid/vllm-runtime/musetest/results}
TREE_PROD=$B/src
TREE_G=${TREE_G:-$B/src_s110g}
TREE_GX=${TREE_GX:-$B/src_s110gx}
HOST_PROD=/raid/vllm-runtime/host_topo
HOST_G=${HOST_G:-/raid/vllm-runtime/host_topo_s110}
PORT=${PORT:-8100}
LOG=${MASTER_LOG:-$W/master_gate.log}
PB='{"rejection_sample_method":"block","draft_sample_method":"probabilistic"}'
mkdir -p "$W" "$R"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
CLEAN_ENV="VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS MUSE_SPEC_SERVE_ARGS MUSE_SPEC_KV_BYTES MUSE_SPEC_MNBT MUSE_SPEC_GRAPH_SP_MIN_TOKENS MUSE_SPEC_GRAPH_CAP MUSE_SPEC_KV_BYTES_OVERRIDE VLLM_MUSE_SPARK_SM120_DRAFT_FP8 VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD VLLM_KV_CACHE_LAYOUT PYTORCH_CUDA_ALLOC_CONF MUSE_TOPO_LOG_OPS MUSE_OFFLOAD MUSE_OFFLOAD_NUMA_BIND MUSE_OFFLOAD_PROFILE MUSE_OFFLOAD_PER_RANK_BYTES MUSE_OFFLOAD_BPB MUSE_OFFLOAD_KV_BYTES MUSE_OFFLOAD_KV_BYTES_OVERRIDE MUSE_OFFLOAD_SERVE_ARGS MUSE_OFFLOAD_ERROR _MUSE_OFFLOAD_SPEC_OVERRIDE_SET"
# shellcheck disable=SC2086
unset $CLEAN_ENV

# expected exact --kv-cache-memory-bytes per profile and arm (ADOPTION_RULE_FINAL.txt "ARMS")
expect_bytes() {  # profile arm [kv of an ns128 ON stress]
  case "$1:$2" in
    spec:CTL|spec:OFF) echo 11573411840 ;;
    spec:ON) echo 11525693440 ;;
    ns256:CTL|ns256:OFF|ns256:ON) echo 16957243392 ;;
    ns128:CTL|ns128:OFF) echo 17357799424 ;;
    ns128:ON) echo "${3:-}" ;;
    *) echo "" ;;
  esac
}

gpus_free() {
  local i u
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    [ "${u:-0}" -lt 2000 ] && return 0; sleep 5
  done
  return 1
}

# watchdog <out_dir> <limit_percent> [min_avail_mib]: runs until killed (see the header)
watchdog() {
  local out=$1 lim=$2 minav=${3:-2048} spid n tot anon shm pct pid free fa av
  while :; do
    sleep 1
    spid=$(cat "$out/serve.pid" 2>/dev/null) || continue
    for pid in $(ps -eo pid=,sid= | awk -v s="$spid" '$2==s{print $1}'); do
      [ "$(cat /proc/$pid/oom_score_adj 2>/dev/null)" = 1000 ] || echo 1000 > /proc/$pid/oom_score_adj 2>/dev/null
    done
    for n in 0 1 2 3 4 5 6 7; do
      read -r tot anon shm free fa <<< "$(awk '/MemTotal/ {t=$4} /AnonPages/ {a=$4} /Shmem:/ {s=$4} /MemFree/ {f=$4} /Active\(file\)|Inactive\(file\)/ {c+=$4} END {print t, a, s, f, c}' /sys/devices/system/node/node$n/meminfo)"
      pct=$(( (anon + shm) * 100 / tot ))
      av=$(( (free + fa) / 1024 ))
      if [ "$pct" -ge "$lim" ] || [ "$av" -lt "$minav" ]; then
        echo "$(date '+%F %T') WATCHDOG node$n anon+shmem=${pct}% (limit ${lim}%), avail_safe=${av} MiB (min ${minav}): TERM server session $spid" | tee -a "$out/status.txt" "$LOG"
        kill -TERM -- "-$spid" 2>/dev/null; sleep 30; kill -KILL -- "-$spid" 2>/dev/null
        return 0
      fi
    done
  done
}

boot() {
  local tag profile arm jobs steady_cells decode_c reuse_args rag_args fulltier_args preempt_args chat_args smoke_len extra xenv xboot kv reps phases wd wdmin
  local host tree out bytes want mns kvm pidw nodepid wdpid nvpid t0 ok=no
  tag=; profile=spec; arm=OFF; jobs="facts steady decode"; steady_cells="1:2,16:1,64:1,128:1"; decode_c="1 16 64 128"; reuse_args=""; rag_args=""; fulltier_args=""
  preempt_args=""; chat_args=""; smoke_len=65536; extra=""; xenv=""; xboot=0; kv=""; reps=2; phases=""; wd=95; wdmin=2048
  eval "$1"
  [ -e "$W/STOP" ] && { log "STOP present: skip $tag"; return 1; }
  out=$R/$tag
  host=$HOST_G; tree=$TREE_G
  [ "$arm" = CTL ] && { host=$HOST_PROD; tree=$TREE_PROD; }
  [ "$xboot" = 1 ] && tree=$TREE_GX
  want=$(expect_bytes "$profile" "$arm" "$kv")
  [ -n "$want" ] || { log "ABORT $tag: no expected bytes for profile=$profile arm=$arm (ns128 ON needs kv=)"; return 2; }
  gpus_free || { log "WARN gpus busy before $tag"; return 1; }
  if [ "$arm" = ON ] && [ "$xboot" != 1 ]; then
    # host-memory preflight of the tier (a node that cannot give 2 x the per-rank bytes ends in a constrained OOM kill or a thrashing populate)
    pf=$(MUSE_OFFLOAD=1 bash "$HOST_G/scripts/preflight_offload.sh" 2>&1); pfrc=$?
    log "PREFLIGHT $tag rc=$pfrc $(echo "$pf" | tr '\n' '|' | cut -c1-400)"
    [ "$pfrc" = 0 ] || { log "SKIP $tag: preflight failed"; return 1; }
  fi
  rm -rf "$out"; mkdir -p "$out/x"
  xenv=${xenv//@OUT@/$out}
  log "BOOT $tag arm=$arm profile=$profile tree=$(basename $tree) host=$(basename $host) jobs=[$jobs] expect_bytes=$want extra=[$extra] xenv=[$xenv] xboot=$xboot"
  t0=$(date +%s)
  "$T/sample_nodes.sh" "$out" 2 > /dev/null 2>&1 &
  nodepid=$!
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free --format=csv,noheader,nounits -l 1 > "$out/nvml.csv" 2>/dev/null &
  nvpid=$!
  watchdog "$out" "$wd" "$wdmin" &
  wdpid=$!
  (
    # shellcheck disable=SC2086
    unset $CLEAN_ENV
    CFG=(); SE=""; XE=""
    if [ "$arm" != CTL ]; then
      if [ "$arm" = ON ]; then export MUSE_OFFLOAD=1; else export MUSE_OFFLOAD=0; fi
      export MUSE_OFFLOAD_PROFILE=$profile
      [ "$profile" = ns128 ] && [ "$arm" = ON ] && export MUSE_OFFLOAD_KV_BYTES_OVERRIDE=$kv
      # shellcheck disable=SC1091
      source "$host/env/offload.env" || { echo "env/offload.env failed: ${MUSE_OFFLOAD_ERROR:-?}" >&2; exit 3; }
    fi
    if [ "$profile" = spec ]; then
      # shellcheck disable=SC1091
      source "$host/env/spec_k7.env"
      bytes=$MUSE_SPEC_KV_BYTES; mns=128
      SE="$MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $bytes"
      CFG+=("SPEC_CONFIG=dflash2_radixark" "SPEC_OVERRIDES=$PB")
      XSPEC=dflash2_radixark; XMNBT=8199; XCAP=256; XPH=${phases:-A1,B,C1,C2,D,H,I,L,M,X}
    elif [ "$profile" = ns256 ]; then
      bytes=16957243392; [ "$arm" = ON ] && bytes=$MUSE_OFFLOAD_KV_BYTES; mns=256
      SE="--max-cudagraph-capture-size 256 --kv-cache-memory-bytes $bytes"
      CFG+=("SPEC_CONFIG=")
      XSPEC=""; XMNBT=""; XCAP=256; XPH=${phases:-A1,B,C1,C2,D,H,I,X}
    else
      bytes=17357799424; [ "$arm" = ON ] && bytes=$MUSE_OFFLOAD_KV_BYTES; mns=128
      SE="--kv-cache-memory-bytes $bytes"
      CFG+=("SPEC_CONFIG=")
      XSPEC=""; XMNBT=""; XCAP=""; XPH=${phases:-A1,B,C1,C2,D,H,I,X}
    fi
    [ "$bytes" = "$want" ] || { echo "bytes $bytes != expected $want for $profile/$arm" >&2; exit 3; }
    if [ "$arm" = ON ]; then SE="$SE $MUSE_OFFLOAD_SERVE_ARGS"; XE="$MUSE_OFFLOAD_SERVE_ARGS"; fi
    [ -n "$extra" ] && { SE="$SE $extra"; XE="$XE $extra"; }
    echo "$SE" > "$out/serve_extra.txt"
    if [ "$xboot" = 1 ]; then
      # S1-7 stress boot (exact_boot_s110.sh: NVML sampler, allocator probe, stress phases, summary.json): needs the probe tree
      env TREE=$tree HOST_DIR=$host KV=${kv:-$bytes} MNS=$mns SPEC=$XSPEC CAP=$XCAP MNBT=$XMNBT PROBE=1 REPS=$reps PHASES=$XPH CONC_A=256 \
        XEXTRA="$XE" L3B_G=/raid/vllm-runtime/work/s1-9/L11/g PORT=$PORT "${CFG[@]}" $xenv bash "$T/exact_boot_s110.sh" "$tag" > "$out/xboot.out" 2>&1
      exit 0
    fi
    env SRC=$tree HOST_DIR=$host JOB_TIMEOUT_S=${JOB_TIMEOUT_S:-21600} BENCH_TIMEOUT=900 PORT=$PORT RESULTS_ROOT=$R MAX_NUM_SEQS=$mns \
      "SERVE_EXTRA=$SE" "${CFG[@]}" $xenv S110_TOOLS=$T S110_JOBS="$jobs" S110_ARM=$arm S110_PROFILE=$profile DECODE_C="$decode_c" STEADY_CELLS="$steady_cells" \
      REUSE_ARGS="$reuse_args" RAG_ARGS="$rag_args" FULLTIER_ARGS="$fulltier_args" PREEMPT_ARGS="$preempt_args" CHAT_ARGS="$chat_args" SMOKE_LEN=$smoke_len \
      "$H/serve_sd.sh" "$tag" -- bash "$T/job_gate.sh" > "$out/serve_sd.out" 2>&1
  )
  kill "$wdpid" "$nodepid" "$nvpid" 2>/dev/null; wait "$wdpid" "$nodepid" "$nvpid" 2>/dev/null
  grep -q "DONE unsloth" "$out/status.txt" 2>/dev/null && ok=yes
  if [ "$xboot" = 1 ]; then
    local fl=1256; [ "$profile" != spec ] && fl=500
    log "XREAD $(python3 -I /raid/vllm-runtime/work/s1-9/L11/g/xread.py "$out/x/summary.json" --mib-floor $fl 2>&1 | tail -1 | cut -c1-400)"
  fi
  log "BOOT_END $tag done=$ok seconds=$(( $(date +%s) - t0 ))"
  postcheck "$out" "$profile" "$arm" "$host" "$tag" "$kv"
  [ "$ok" = yes ]
}

# per-boot summary: peak unreclaimable memory per node, READY line, OOM count, GPU memory, the launch checks
postcheck() {
  local out=$1 profile=$2 arm=$3 host=$4 tag=$5 kv=$6 sp of
  awk '{n=$2; split($5,a,"="); split($6,s,"="); split($3,f,"="); split($4,fi,"=");
        u=a[2]+s[2]; if (u>mx[n]) mx[n]=u; if (!(n in mn) || f[2]<mn[n]) mn[n]=f[2]; if (fi[2]>mf[n]) mf[n]=fi[2]}
        END {for (n in mx) printf "%s peak_anon_plus_shmem_mib=%d min_free_mib=%d max_filecache_mib=%d\n", n, mx[n], mn[n], mf[n]}' "$out/node_mem_ts.txt" 2>/dev/null | sort > "$out/node_peaks.txt"
  log "NODE_PEAKS $tag $(awk '{printf "%s:%s ", $1, $2}' "$out/node_peaks.txt" | sed 's/peak_anon_plus_shmem_mib=//g')"
  { echo "== $tag"; grep -a -h -E "Model loading took|Graph capturing finished|GPU KV cache size|Application startup complete|init engine|SimpleCPUOffload|KV cache layout|Allocating" "$out/serve.log" | sed 's/^(\([A-Za-z_0-9]*\) pid=[0-9]*) /\1 /' | cut -c1-260 | head -60; } > "$out/boot_times.txt" 2>&1
  log "BOOT_TIMES $tag $(grep -a -E 'READY|LOADING|LOAD_FAIL' "$out/status.txt" | tail -1) gpu_ready_mib=$(awk -F', ' '{gsub(/ MiB/,"",$2); s+=$2; n++} END {if (n) printf "%d", s/n}' "$out/gpu_mem_ready.txt" 2>/dev/null) oom_lines=$(grep -a -c -E 'OutOfMemoryError|CUDA out of memory' "$out/serve.log" 2>/dev/null) watchdog=$(grep -c WATCHDOG "$out/status.txt" 2>/dev/null)"
  if [ "$profile" = spec ]; then
    sp=$( ( export MUSE_OFFLOAD=$([ "$arm" = ON ] && echo 1 || echo 0) MUSE_OFFLOAD_PROFILE=$profile
            # shellcheck disable=SC1091
            source "$host/env/host.env"; [ "$arm" != CTL ] && source "$host/env/offload.env"; source "$host/env/spec_k7.env"; bash "$host/scripts/check_spec_launch.sh" "$out/serve.log" ) 2>&1 | tail -3 | tr '\n' ' ' )
    echo "$sp" > "$out/spcheck.txt"
    log "SPCHECK $tag $sp"
  fi
  if [ "$arm" = ON ]; then
    of=$( ( export MUSE_OFFLOAD=1 MUSE_OFFLOAD_PROFILE=$profile
            [ "$profile" = ns128 ] && export MUSE_OFFLOAD_KV_BYTES_OVERRIDE=$kv
            # shellcheck disable=SC1091
            source "$HOST_G/env/offload.env"; bash "$HOST_G/scripts/check_offload_launch.sh" "$out/serve.log" ) 2>&1 | tail -2 | tr '\n' ' ' )
  else
    of=$( ( MUSE_OFFLOAD=0 bash "$HOST_G/scripts/check_offload_launch.sh" "$out/serve.log" ) 2>&1 | tail -2 | tr '\n' ' ' )
  fi
  echo "$of" > "$out/offcheck.txt"
  log "OFFCHECK $tag $of"
}

log "HOLD start runs=$RUNS_FILE gate_tree=$TREE_G gate_host=$HOST_G md5(host.env prod/gate)=$(md5sum $HOST_PROD/env/host.env | cut -c1-8)/$(md5sum $HOST_G/env/host.env | cut -c1-8) md5(spec_k7.env prod/gate)=$(md5sum $HOST_PROD/env/spec_k7.env | cut -c1-8)/$(md5sum $HOST_G/env/spec_k7.env | cut -c1-8) md5(offload.env)=$(md5sum $HOST_G/env/offload.env | cut -c1-12)"
while IFS= read -r line; do
  case $line in ''|'#'*) continue ;; esac
  boot "$line" || log "RUN FAILED: $line"
done < "$RUNS_FILE"
log "HOLD end"
