#!/usr/bin/env bash
# S1-10 master: runs the boots listed in a RUNS file, one after the other, inside ONE GPU-lock hold taken by the caller:
#   setsid nohup flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock bash master_s110.sh <runs_file> > master.out 2>&1 < /dev/null &
# (nothing in here takes the lock again; serve_sd.sh is called directly.)
# A RUNS line is a set of shell assignments (blank lines and #-lines ignored):
#   tag=s110_a_off_n profile=spec|ns256 arm=OFF_N|OFF_B|SIM_N|SIM_B|NAT_N|NAT_B jobs="facts decode reuse chat smoke"
#   prb=<cpu bytes per rank>   (SIM: cpu_bytes_to_use_per_rank = prb; NAT: cpu_bytes_to_use = 8 * prb, total over the ranks)
#   bpc=<native blocks_per_chunk, default 1>  lazy=0|1 (SIM lazy_offload)  extra="<more serve args, no spaces inside JSON>"
#   reuse_args="..." chat_args="..." decode_c="..." wd=<watchdog anon+shmem percent, default 95> wdmin=<watchdog min MemFree+file cache MiB per node, default 2048>   (optional overrides)
#   xboot=1 kv=<exact --kv-cache-memory-bytes> reps=<2> phases=<stress phases>: an S1-7 stress boot (exact_boot_s110.sh, probe tree needed in tree=)
#   tree=<vllm tree, default $TREE> xenv="KEY=VAL ..." (@OUT@ = the result dir; extra environment of the server, no spaces inside a value)
# Arms: *_B = vLLM --numa-bind (every worker under numactl --cpunodebind=N --membind=N, N = the GPU's node), *_N = no binding;
#   SIM = SimpleCPUOffloadConnector, NAT = native OffloadingConnector with VLLM_KV_CACHE_LAYOUT=LBNHC, OFF = no connector.
# Profiles (the S1-9 production lines): spec = host.env + env/spec_k7.env, RadixArk DFlash2 K=7, mns 128, cap 256, MNBT 8199,
#   --kv-cache-memory-bytes 11573411840; ns256 = host.env only, mns 256, cap 256, --kv-cache-memory-bytes 16957243392.
# Per boot: node-memory sampler (2 s), GPU NVML sampler, a watchdog that TERMs the server session when the unreclaimable memory
# (anon + shmem) of ANY node reaches WD percent of its size and raises oom_score_adj of the session processes to 1000 (an OOM
# kill must hit this session, never another user's process). File $W/STOP ends the hold at the next boot boundary.
set -u
RUNS_FILE=${1:?usage: master_s110.sh <runs_file>}
B=/raid/vllm-runtime/sm120-dev
W=/raid/vllm-runtime/work/s1-10
T=$W/tools
H=/raid/vllm-runtime/work/sd/harness
R=${RESULTS_ROOT:-/raid/vllm-runtime/musetest/results}
TREE=${TREE:-$B/src}
HOST=${HOST_DIR:-/raid/vllm-runtime/host_topo}
PORT=${PORT:-8100}
LOG=${MASTER_LOG:-$W/master.log}
BYTES_SPEC=11573411840
BYTES_NS256=16957243392
PB='{"rejection_sample_method":"block","draft_sample_method":"probabilistic"}'
mkdir -p "$W" "$R"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
CLEAN_ENV="VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS MUSE_SPEC_SERVE_ARGS MUSE_SPEC_KV_BYTES MUSE_SPEC_MNBT MUSE_SPEC_GRAPH_SP_MIN_TOKENS MUSE_SPEC_GRAPH_CAP MUSE_SPEC_KV_BYTES_OVERRIDE VLLM_MUSE_SPARK_SM120_DRAFT_FP8 VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD VLLM_KV_CACHE_LAYOUT PYTORCH_CUDA_ALLOC_CONF MUSE_TOPO_LOG_OPS"
# shellcheck disable=SC2086
unset $CLEAN_ENV

gpus_free() {
  local i u
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    [ "${u:-0}" -lt 2000 ] && return 0; sleep 5
  done
  return 1
}

# watchdog <out_dir> <limit_percent> [min_avail_mib]: runs until killed. Trips (TERM, then KILL of the server session) when ANY node
# has anon + shmem >= limit_percent of its size, or when MemFree + file cache (excluding shmem; the 'avail_safe' of numa_snap.sh) of
# any node falls below min_avail_mib (default 2048): node 7 carries ~13 GiB of memory of a foreign driver, so a percentage of the
# node total alone would trip too late there. It also raises oom_score_adj of every process of the session to 1000.
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
  local tag profile arm jobs prb bpc lazy extra reuse_args chat_args decode_c wd wdmin smoke_len tree xenv xboot kv reps phases
  local BIND CONN kvt out bytes mns specv pidw nodepid wdpid nvpid t0 ok=no
  tag=; profile=spec; arm=OFF_N; jobs="facts decode"; prb=0; bpc=1; lazy=0; extra=""; reuse_args=""; chat_args=""; decode_c="1 16 64 128"; wd=95; wdmin=2048; smoke_len=65536; tree=$TREE; xenv=""; xboot=0; kv=""; reps=2; phases=""
  eval "$1"
  [ -e "$W/STOP" ] && { log "STOP present: skip $tag"; return 1; }
  out=$R/$tag
  case $arm in
    OFF_N) BIND=0; CONN=none ;;   OFF_B) BIND=1; CONN=none ;;
    SIM_N) BIND=0; CONN=simple ;; SIM_B) BIND=1; CONN=simple ;;
    NAT_N) BIND=0; CONN=native ;; NAT_B) BIND=1; CONN=native ;;
    *) log "ABORT unknown arm $arm"; return 2 ;;
  esac
  kvt=""
  if [ "$CONN" = simple ]; then
    kvt='{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'$(( prb * 8 ))',"cpu_bytes_to_use_per_rank":'$prb
    [ "$lazy" = 1 ] && kvt="$kvt"',"lazy_offload":true'
    kvt="$kvt"'}}'
  elif [ "$CONN" = native ]; then
    kvt='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'$(( prb * 8 ))',"blocks_per_chunk":'$bpc'}}'
  fi
  gpus_free || { log "WARN gpus busy before $tag"; return 1; }
  rm -rf "$out"; mkdir -p "$out/x"
  xenv=${xenv//@OUT@/$out}
  log "BOOT $tag tree=$(basename $tree) xenv=[$xenv] profile=$profile arm=$arm jobs=[$jobs] prb=$prb bpc=$bpc lazy=$lazy extra=[$extra] kvt=[$kvt]"
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
    CFG=()
    SE=""
    if [ "$profile" = spec ]; then
      # shellcheck disable=SC1091
      source "$HOST/env/spec_k7.env"
      [ "$MUSE_SPEC_KV_BYTES" = "$BYTES_SPEC" ] || { echo "spec bytes $MUSE_SPEC_KV_BYTES != $BYTES_SPEC" >&2; exit 3; }
      bytes=$BYTES_SPEC; mns=128
      SE="$MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $bytes"
      CFG+=("SPEC_CONFIG=dflash2_radixark" "SPEC_OVERRIDES=$PB")
    else
      bytes=$BYTES_NS256; mns=256
      SE="--max-cudagraph-capture-size 256 --kv-cache-memory-bytes $bytes"
      CFG+=("SPEC_CONFIG=")
    fi
    [ "$BIND" = 1 ] && SE="$SE --numa-bind"
    [ -n "$kvt" ] && SE="$SE --kv-transfer-config $kvt"
    [ -n "$extra" ] && SE="$SE $extra"
    [ "$CONN" = native ] && export VLLM_KV_CACHE_LAYOUT=LBNHC
    if [ "$xboot" = 1 ]; then
      # S1-7 stress boot (exact_boot_s110.sh: NVML sampler, allocator probe, stress phases, summary.json): needs a probe tree
      XE=""; [ "$BIND" = 1 ] && XE="--numa-bind"; [ -n "$kvt" ] && XE="$XE --kv-transfer-config $kvt"; [ -n "$extra" ] && XE="$XE $extra"
      if [ "$profile" = spec ]; then XSPEC=dflash2_radixark; XMNBT=8199; XPH=${phases:-A1,B,C1,C2,D,H,I,L,M,X}
      else XSPEC=""; XMNBT=""; XPH=${phases:-A1,B,C1,C2,D,H,I,X}; fi
      env TREE=$tree HOST_DIR=$HOST KV=${kv:-$bytes} MNS=$mns SPEC=$XSPEC CAP=256 MNBT=$XMNBT PROBE=1 REPS=$reps PHASES=$XPH CONC_A=256 \
        XEXTRA="$XE" L3B_G=/raid/vllm-runtime/work/s1-9/L11/g PORT=$PORT "${CFG[@]}" $xenv bash "$T/exact_boot_s110.sh" "$tag" > "$out/xboot.out" 2>&1
      exit 0
    fi
    env SRC=$tree HOST_DIR=$HOST JOB_TIMEOUT_S=${JOB_TIMEOUT_S:-14400} BENCH_TIMEOUT=900 PORT=$PORT RESULTS_ROOT=$R MAX_NUM_SEQS=$mns \
      "SERVE_EXTRA=$SE" "${CFG[@]}" $xenv S110_TOOLS=$T S110_JOBS="$jobs" S110_ARM=$arm S110_PROFILE=$profile DECODE_C="$decode_c" \
      REUSE_ARGS="$reuse_args" CHAT_ARGS="$chat_args" SMOKE_LEN=$smoke_len \
      "$H/serve_sd.sh" "$tag" -- bash "$T/job_s110.sh" > "$out/serve_sd.out" 2>&1
  )
  kill "$wdpid" "$nodepid" "$nvpid" 2>/dev/null; wait "$wdpid" "$nodepid" "$nvpid" 2>/dev/null
  grep -q "DONE unsloth" "$out/status.txt" 2>/dev/null && ok=yes
  if [ "$xboot" = 1 ]; then
    local fl=1256; [ "$profile" != spec ] && fl=500
    log "XREAD $(python3 -I /raid/vllm-runtime/work/s1-9/L11/g/xread.py "$out/x/summary.json" --mib-floor $fl 2>&1 | tail -1 | cut -c1-400)"
  fi
  log "BOOT_END $tag done=$ok seconds=$(( $(date +%s) - t0 ))"
  postcheck "$out" "$profile" "$tag"
  [ "$ok" = yes ]
}

# per-boot summary: peak unreclaimable memory per node, key log timestamps, the launch check of the spec profile
postcheck() {
  local out=$1 profile=$2 tag=$3
  awk '{n=$2; split($5,a,"="); split($6,s,"="); split($3,f,"="); split($4,fi,"=");
        u=a[2]+s[2]; if (u>mx[n]) mx[n]=u; if (!(n in mn) || f[2]<mn[n]) mn[n]=f[2]; if (fi[2]>mf[n]) mf[n]=fi[2]}
        END {for (n in mx) printf "%s peak_anon_plus_shmem_mib=%d min_free_mib=%d max_filecache_mib=%d\n", n, mx[n], mn[n], mf[n]}' "$out/node_mem_ts.txt" 2>/dev/null | sort > "$out/node_peaks.txt"
  log "NODE_PEAKS $tag $(awk '{printf "%s:%s ", $1, $2}' "$out/node_peaks.txt" | sed 's/peak_anon_plus_shmem_mib=//g')"
  { echo "== $tag"; grep -a -h -E "Model loading took|Graph capturing finished|GPU KV cache size|Application startup complete|init engine|SimpleCPUOffload|Created mmap|Unlinked mmap|KV cache layout|Allocating" "$out/serve.log" | sed 's/^(\([A-Za-z_0-9]*\) pid=[0-9]*) /\1 /' | cut -c1-260 | head -60; } > "$out/boot_times.txt" 2>&1
  log "BOOT_TIMES $tag $(grep -a -E 'READY|LOADING|LOAD_FAIL' "$out/status.txt" | tail -1)"
  if [ "$profile" = spec ]; then
    local sp
    sp=$( ( source "$HOST/env/host.env"; source "$HOST/env/spec_k7.env"; bash "$HOST/scripts/check_spec_launch.sh" "$out/serve.log" ) 2>&1 | tail -3 | tr '\n' ' ' )
    echo "$sp" > "$out/spcheck.txt"
    log "SPCHECK $tag $sp"
  fi
}

log "HOLD start runs=$RUNS_FILE tree=$TREE host=$HOST md5(host.env)=$(md5sum "$HOST/env/host.env" | cut -c1-12) md5(spec_k7.env)=$(md5sum "$HOST/env/spec_k7.env" | cut -c1-12)"
while IFS= read -r line; do
  case $line in ''|'#'*) continue ;; esac
  boot "$line" || log "RUN FAILED: $line"
done < "$RUNS_FILE"
log "HOLD end"
