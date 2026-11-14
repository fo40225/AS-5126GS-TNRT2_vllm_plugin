#!/usr/bin/env bash
# Background sampler: per-NUMA-node memory every INTERVAL seconds (MiB) into <out_dir>/node_mem_ts.txt (runs until killed).
# usage: sample_nodes.sh <out_dir> [interval_s]    (also <out_dir>/vmstat_ts.txt: the reclaim / major-fault counters of /proc/vmstat)
OUT=${1:?}; IV=${2:-2}
while :; do
  t=$(date +%T)
  for n in 0 1 2 3 4 5 6 7; do
    awk -v n="$n" -v t="$t" '{k=$3; sub(":","",k); v[k]=$4/1024} END {
      printf "%s node%d free=%d file=%d anon=%d shmem=%d mlocked=%d unevict=%d\n", t, n, v["MemFree"], v["FilePages"], v["AnonPages"], v["Shmem"], v["Mlocked"], v["Unevictable"] }' /sys/devices/system/node/node$n/meminfo
  done >> "$OUT/node_mem_ts.txt"
  echo "$t $(awk '/^(pgmajfault|pgscan_direct|pgsteal_direct|allocstall_normal|allocstall_movable|pgscan_kswapd|numa_pages_migrated|thp_fault_alloc) / {printf "%s=%s ", $1, $2}' /proc/vmstat)" >> "$OUT/vmstat_ts.txt"
  sleep "$IV"
done
