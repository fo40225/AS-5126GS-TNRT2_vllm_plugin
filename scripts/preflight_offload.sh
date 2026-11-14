#!/bin/bash
# Host-memory preflight of the NUMA-local KV CPU tier (S1-10 gate review). Run it BEFORE a launch with env/offload.env sourced (or
# MUSE_OFFLOAD_PER_RANK_BYTES set): the tier is pinned, non-reclaimable and bound with strict membind, so a node that cannot give it up ends in a
# constrained OOM kill (largest eligible RSS: one of our workers, or another user's big process) or in a populate that thrashes the page cache.
#
# Per GPU node N (GPU 0,1 -> 3; 2,3 -> 0; 4,5 -> 7; 6,7 -> 4):
#   A_N     = MemFree + Active(file) + Inactive(file)          of /sys/devices/system/node/nodeN/meminfo (MiB; the tier's populate evicts the page cache)
#   NEED_N  = 2 * per_rank_bytes + W_N                         W_N = what the server itself puts on the node besides the tier: two workers (3.7 GiB anon +
#                                                              0.3 GiB shmem each) and, on node 3 the EngineCore (1.2 GiB), on node 7 the API server (1.3 GiB)
#   LEFT_N  = A_N - NEED_N                                     predicted reclaimable memory left once the server is up
# FAIL (exit 1) when any LEFT_N < MUSE_OFFLOAD_MIN_LEFT_MIB (default 4,096: the floor of the original G3 guard); WARN when LEFT_N < the margin M = 6.6 GiB of the
# sizing formula (docs/s110_numa_offload.md section 4). Measured with the shipped 45 GiB tier on node 7: 10.4 GiB left at READY.
# Only a server of ours may be running: workers already up are counted in neither A_N nor W_N here (a second server would make the numbers meaningless).
# OOM protection (not done here): start the server session with `echo 1000 > /proc/self/oom_score_adj` so that an OOM kill of this launch hits the server and
# never a foreign process (the S1-10 harness did this for every process of its session).
# 0.32.0-DERIVED (S1-10R): the default per-rank bytes (45 GiB), the margin M = 6.6 GiB and the W_N process footprints (workers 3.7 GiB anon + 0.3 GiB shmem, EngineCore 1.2 GiB,
# API server 1.3 GiB) are 2026-11-07 measurements on the 0.32.0 line, before three reboots and the NIC mode change; S1-10R phase 3 re-measures A_N and the footprints on 0.33.0.
# usage: [MUSE_OFFLOAD_PER_RANK_BYTES=<bytes>] [MUSE_OFFLOAD_MIN_LEFT_MIB=4096] [MUSE_OFFLOAD_NODE_ROOT=/sys/devices/system/node] preflight_offload.sh
# Exit 0 = every GPU node can take the tier, 1 = a node is short, 2 = unreadable meminfo.
root=${MUSE_OFFLOAD_NODE_ROOT:-/sys/devices/system/node}
per_rank=${MUSE_OFFLOAD_PER_RANK_BYTES:-48318382080}
min_left=${MUSE_OFFLOAD_MIN_LEFT_MIB:-4096}
margin_mib=6758          # M = 6.6 GiB
if [ "${MUSE_OFFLOAD:-1}" = 0 ]; then echo "preflight_offload: MUSE_OFFLOAD=0, nothing to check"; exit 0; fi
tier_mib=$(( 2 * per_rank / 1048576 ))
rc=0
printf '%-6s %12s %12s %12s %s\n' node A_N_MiB NEED_MiB LEFT_MiB verdict
for n in 3 0 7 4; do
  f=$root/node$n/meminfo
  [ -r "$f" ] || { echo "preflight_offload: cannot read $f"; exit 2; }
  a=$(awk '/MemFree/ {f=$4} /Active\(file\)|Inactive\(file\)/ {c+=$4} END {printf "%d", (f + c) / 1024}' "$f")
  w=8192                                       # two workers
  [ "$n" = 3 ] && w=$(( w + 1229 ))            # EngineCore
  [ "$n" = 7 ] && w=$(( w + 1331 ))            # API server
  need=$(( tier_mib + w ))
  left=$(( a - need ))
  v=ok
  if [ "$left" -lt "$min_left" ]; then v="FAIL (left < ${min_left} MiB)"; rc=1
  elif [ "$left" -lt "$margin_mib" ]; then v="WARN (left < M = ${margin_mib} MiB)"; fi
  printf '%-6s %12d %12d %12d %s\n' "node$n" "$a" "$need" "$left" "$v"
done
[ "$rc" = 0 ] && echo "PREFLIGHT OK: per-rank tier $per_rank B (2 ranks per node = $tier_mib MiB) fits every GPU node" || echo "PREFLIGHT FAIL: a GPU node cannot take the tier (free memory taken by other users? lower MUSE_OFFLOAD_PER_RANK_BYTES, a multiple of 327,680)"
exit $rc
