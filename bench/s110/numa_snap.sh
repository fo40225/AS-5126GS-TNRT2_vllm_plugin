#!/usr/bin/env bash
# NUMA snapshot of one running vLLM server session (S1-10).
# usage: numa_snap.sh <out_dir> <label> <session_id> [all|first]     (writes <out_dir>/numa_<label>.txt, prints a one-line summary)
# Contents: per-node memory incl. an estimate of MemAvailable (MemFree + file cache + SReclaimable), every process of the
# session with RSS / RssAnon / RssShmem / VmPin / allowed CPUs and memory nodes, the GPU index of every worker, the numa_maps
# summary per worker (all or the first) and the page nodes of the KV-offload shm mapping of the first worker (native connector).
OUT=${1:?}; LABEL=${2:?}; SID=${3:?}; WHICH=${4:-first}
N=${S110_TOOLS:-/raid/vllm-runtime/work/s1-10/tools}
F=$OUT/numa_$LABEL.txt
PY=python3
{
echo "=== numa_snap $LABEL $(date '+%F %T') session=$SID"
echo "--- node meminfo (MiB); avail_est = MemFree + Active(file) + Inactive(file) + SReclaimable (upper bound of MemAvailable); avail_safe = without the reclaimable slab; other_unacc = total - free - FilePages - Anon - Slab - KernelStack - PageTables (non-reclaimable memory nobody of ours owns)"
for n in 0 1 2 3 4 5 6 7; do
  awk -v n="$n" '{k=$3; sub(":","",k); v[k]=$4/1024} END {
    other = v["MemTotal"]-v["MemFree"]-v["FilePages"]-v["AnonPages"]-v["SReclaimable"]-v["SUnreclaim"]-v["KernelStack"]-v["PageTables"]
    printf "node%d total=%d free=%d filecache=%d anon=%d shmem=%d mapped=%d mlocked=%d unevictable=%d slab_recl=%d avail_est=%d avail_safe=%d slab_unrecl=%d other_unacc=%d\n", n, v["MemTotal"], v["MemFree"], v["FilePages"], v["AnonPages"], v["Shmem"], v["Mapped"], v["Mlocked"], v["Unevictable"], v["SReclaimable"], v["MemFree"]+v["Active(file)"]+v["Inactive(file)"]+v["SReclaimable"], v["MemFree"]+v["Active(file)"]+v["Inactive(file)"], v["SUnreclaim"], other }' /sys/devices/system/node/node$n/meminfo
done
echo "--- global: $(grep -E '^(MemAvailable|Shmem|Mlocked|Unevictable):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
echo "--- processes of the server session"
ps -eo pid,sid,comm,nlwp,rss,args --no-headers | awk -v s="$SID" '$2==s' | cut -c1-200 | while read -r pid sid comm nlwp rss rest; do
  cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/$pid/status 2>/dev/null)
  mems=$(awk '/Mems_allowed_list/ {print $2}' /proc/$pid/status 2>/dev/null)
  pin=$(awk '/VmPin/ {print int($2/1024)}' /proc/$pid/status 2>/dev/null)
  ra=$(awk '/RssAnon/ {print int($2/1024)}' /proc/$pid/status 2>/dev/null)
  rs=$(awk '/RssShmem/ {print int($2/1024)}' /proc/$pid/status 2>/dev/null)
  echo "pid=$pid comm=$comm threads=$nlwp rss_mib=$((rss/1024)) rssanon_mib=${ra:-?} rssshmem_mib=${rs:-?} vmpin_mib=${pin:-?} cpus=[$cpus] mems=[$mems] $(echo "$rest" | cut -c1-70)"
done
echo "--- gpu index of compute processes"
declare -A BUS2IDX
while IFS=', ' read -r idx bus; do BUS2IDX[$bus]=$idx; done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)
WPIDS=()
while IFS=', ' read -r pid bus; do
  if [ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$SID" ]; then
    echo "gpu=${BUS2IDX[$bus]:-?} pid=$pid cpus=[$(awk '/Cpus_allowed_list/ {print $2}' /proc/$pid/status)] mems=[$(awk '/Mems_allowed_list/ {print $2}' /proc/$pid/status)]"
    WPIDS+=("${BUS2IDX[$bus]:-9}:$pid")
  fi
done < <(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | sort -u)
IFS=$'\n' WPIDS=($(printf '%s\n' "${WPIDS[@]}" | sort -n)); unset IFS
echo "--- numa_maps summary (MiB per node)"
SEL=("${WPIDS[@]}"); [ "$WHICH" = first ] && SEL=("${WPIDS[0]:-}")
TMPD=$(mktemp -d /tmp/numa_snap.XXXXXX)
for gp in "${SEL[@]}"; do
  [ -n "$gp" ] || continue
  g=${gp%%:*}; pid=${gp##*:}
  ( $PY -I $N/numa_maps_summary.py "$pid" > "$TMPD/$g.txt" 2>&1 ) &
done
wait
for gp in "${SEL[@]}"; do
  [ -n "$gp" ] || continue
  g=${gp%%:*}; echo "gpu=$g $(cat "$TMPD/$g.txt")"
done
rm -rf "$TMPD"
echo "--- offload mapping page nodes (move_pages, first worker)"
[ -n "${WPIDS[0]:-}" ] && { $PY -I $N/region_nodes.py "${WPIDS[0]##*:}" 2>&1; }
} > "$F" 2>&1
echo "numa_snap $LABEL -> $F"
