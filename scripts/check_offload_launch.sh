#!/bin/bash
# Confirms in the log of a boot that the NUMA-local KV CPU offload of env/offload.env took effect (S1-10).
# Usage: [MUSE_OFFLOAD_EXPECT=simple|off] [MUSE_OFFLOAD_EXPECT_BIND=1|0] [MUSE_OFFLOAD_EXPECT_BYTES=<per rank>] [MUSE_OFFLOAD_EXPECT_BPB=<bytes per block-id>]
#        check_offload_launch.sh <serve.log>
# Defaults come from the sourcing shell (env/offload.env exports MUSE_OFFLOAD, MUSE_OFFLOAD_NUMA_BIND, MUSE_OFFLOAD_PER_RANK_BYTES, MUSE_OFFLOAD_BPB).
# What it checks (all from lines the vLLM branch already logs):
#   binding   : eight "Binding worker subprocess (local_rank=<i>, gpu_index=<i>) to NUMA node <n>" lines with n = the GPU's node (3 3 0 0 7 7 4 4 on this box)
#               and one EngineCore line; with the bind off there must be none (and then the tier is NOT node-local by construction);
#   connector : one "SimpleCPUOffloadScheduler: Allocating <N> offload blocks" and eight "SimpleCPUOffloadWorker [CPU]: <K> tensors, <N> CPU blocks" lines;
#               the scheduler's N and every worker's N must be equal (a rank with fewer blocks than the scheduler hands out would corrupt memory),
#               N must equal floor(per_rank_bytes / bytes_per_block) and K the same on every worker;
#   layout    : "Using LBNHC KV cache layout." (the production layout; the native OffloadingConnector would flip it to LBHNC);
#   mm path   : no "enable_mm_embeds is True" line (a vLLM tree without the S1-10 receives_remote_kv fix flips a kv_both connector into the multimodal embedding path:
#               a 128 MiB encoder buffer per GPU, the model on input embeddings; it also voids the exact KV bytes);
#   problems  : no "Pinned memory not available", no "cudaHostRegister failed", no "cpu_bytes_to_use_per_rank (... ) != cpu_bytes_to_use/world_size" disagreement.
# When serve_cmd.txt (written by the harness) sits next to the log it must carry --numa-bind exactly when the bind is expected, a
# --kv-transfer-config naming SimpleCPUOffloadConnector with the same cpu_bytes_to_use_per_rank, and a --kv-cache-memory-bytes that is not above
# MUSE_OFFLOAD_EXPECT_KV_MAX (default MUSE_OFFLOAD_KV_BYTES, the S1-7 bytes of the profile with the tier on).
# Exit 0 = in effect; 1 = a line with other values, no line, or a mismatch; 2 = usage or unreadable log.
if [ $# -ne 1 ]; then echo "usage: check_offload_launch.sh <serve.log>" >&2; exit 2; fi
log=$1
[ -r "$log" ] || { echo "check_offload_launch: cannot read $log" >&2; exit 2; }
want=${MUSE_OFFLOAD_EXPECT:-$([ "${MUSE_OFFLOAD:-1}" = 0 ] && echo off || echo simple)}
bind=${MUSE_OFFLOAD_EXPECT_BIND:-${MUSE_OFFLOAD_NUMA_BIND:-1}}
bytes=${MUSE_OFFLOAD_EXPECT_BYTES:-${MUSE_OFFLOAD_PER_RANK_BYTES:-}}
bpb=${MUSE_OFFLOAD_EXPECT_BPB:-${MUSE_OFFLOAD_BPB:-}}
cmd="$(dirname "$log")/serve_cmd.txt"
fail() { echo "FAIL $*"; exit 1; }

# ---- binding
declare -A node_of=([0]=3 [1]=3 [2]=0 [3]=0 [4]=7 [5]=7 [6]=4 [7]=4)
nbind=$(grep -a -c "Binding worker subprocess" "$log")
if [ "$bind" = 1 ]; then
  [ "$nbind" = 8 ] || fail "expected 8 'Binding worker subprocess' lines, found $nbind (--numa-bind missing, numactl rejected, or the spawn method is not spawn)"
  for i in 0 1 2 3 4 5 6 7; do
    grep -a -q "Binding worker subprocess (local_rank=$i, gpu_index=$i) to NUMA node ${node_of[$i]}" "$log" ||
      fail "rank $i is not bound to NUMA node ${node_of[$i]}: $(grep -a "Binding worker subprocess (local_rank=$i," "$log" | head -1 | sed 's/^.*\] //')"
  done
  grep -a -q "Binding EngineCore subprocess" "$log" || fail "no EngineCore binding line"
  if grep -a -q "falling back to" "$log" && grep -a "numactl args" "$log" | grep -a -q "rejected"; then fail "numactl args were rejected (no membind)"; fi
else
  [ "$nbind" = 0 ] || fail "bind expected off but $nbind 'Binding worker subprocess' lines are in the log"
fi

# ---- connector
if [ "$want" = off ]; then
  grep -a -q "SimpleCPUOffload" "$log" && fail "offload expected off but the log has SimpleCPUOffload lines"
  echo "OK offload=off bind=$bind"; exit 0
fi
sched=$(grep -a -o -E "SimpleCPUOffloadScheduler: Allocating [0-9]+ offload blocks" "$log" | grep -o -E "[0-9]+" | sort -u)
[ "$(echo "$sched" | grep -c .)" = 1 ] || fail "expected one SimpleCPUOffloadScheduler allocation line, found: ${sched:-none}"
workers=$(grep -a -o -E "SimpleCPUOffloadWorker \[CPU\]: [0-9]+ tensors, [0-9]+ CPU blocks" "$log" | sed -E 's/.*: ([0-9]+) tensors, ([0-9]+) CPU blocks/\1 \2/' | sort | uniq -c)
[ "$(echo "$workers" | grep -c .)" = 1 ] || fail "workers disagree on tensors / blocks (count tensors blocks): $(echo $workers)"
nw=$(echo "$workers" | awk '{print $1}'); tens=$(echo "$workers" | awk '{print $2}'); wblocks=$(echo "$workers" | awk '{print $3}')
[ "$nw" = 8 ] || fail "expected 8 worker allocation lines, found $nw"
[ "$wblocks" = "$sched" ] || fail "scheduler hands out $sched CPU blocks but the workers allocated $wblocks"
if [ -n "$bytes" ] && [ -n "$bpb" ]; then
  exp=$(( bytes / bpb ))
  [ "$exp" = "$sched" ] || fail "per-rank bytes $bytes / $bpb B per block-id = $exp blocks, the log has $sched"
fi
grep -a -q "Using LBNHC KV cache layout" "$log" || fail "no 'Using LBNHC KV cache layout' line (a connector that flips the layout?)"
grep -a -q "enable_mm_embeds is True" "$log" && fail "'enable_mm_embeds is True' in the log: the vLLM tree takes the kv_both connector for a disaggregated KV consumer (no receives_remote_kv fix): +128 MiB of GPU memory and the model runs from input embeddings, so the exact KV bytes of this profile do not hold"
grep -a -q "Pinned memory not available" "$log" && fail "pinned memory not available (the tier is pageable)"
grep -a -q "cudaHostRegister failed" "$log" && fail "cudaHostRegister failed"
grep -a -q "cpu_bytes_to_use_per_rank .* != " "$log" && fail "cpu_bytes_to_use_per_rank disagrees with cpu_bytes_to_use / world_size (set only one)"

# ---- launch line next to the log
kvmax=${MUSE_OFFLOAD_EXPECT_KV_MAX:-${MUSE_OFFLOAD_KV_BYTES:-}}
if [ -r "$cmd" ]; then
  # the exact --kv-cache-memory-bytes of an offload-on launch is stepped down for the page tables of the tier (S1-7 rule): never above MUSE_OFFLOAD_KV_BYTES
  kvb=$(grep -o -E -e "--kv-cache-memory-bytes[ =][0-9]+" "$cmd" | head -1 | grep -o -E "[0-9]+$")
  if [ -n "$kvmax" ]; then
    [ -n "$kvb" ] || fail "serve_cmd.txt has no --kv-cache-memory-bytes (the exact bytes are part of the offload profile)"
    [ "$kvb" -le "$kvmax" ] || fail "--kv-cache-memory-bytes $kvb is above the offload-on maximum $kvmax (the tier's page tables take GPU memory: S_live of the stress boot would fall below the floor)"
  fi
  if [ "$bind" = 1 ]; then grep -q -e "--numa-bind" "$cmd" || fail "serve_cmd.txt has no --numa-bind"; else grep -q -e "--numa-bind" "$cmd" && fail "serve_cmd.txt has --numa-bind but the bind is expected off"; fi
  grep -q "SimpleCPUOffloadConnector" "$cmd" || fail "serve_cmd.txt has no SimpleCPUOffloadConnector"
  if [ -n "$bytes" ]; then grep -q -E "cpu_bytes_to_use_per_rank(\\\\)?\":$bytes([^0-9]|$)" "$cmd" || fail "serve_cmd.txt does not carry cpu_bytes_to_use_per_rank=$bytes"; fi
fi
echo "OK offload=simple bind=$bind blocks=$sched tensors=$tens workers=$nw layout=LBNHC${bytes:+ per_rank=$bytes}"
exit 0
