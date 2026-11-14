#!/bin/bash
# Regression test of scripts/check_offload_launch.sh (S1-10: binding lines, scheduler / worker block agreement, layout, serve_cmd.txt, fail-closed
# expectations) on synthetic boot logs. No GPU, no vLLM: usage  bash bench/test_check_offload_launch.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
check=$here/scripts/check_offload_launch.sh
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
fails=0
node_of=(3 3 0 0 7 7 4 4)
# mk <name> <bind 1|0> <scheduler blocks> <worker blocks> [tensors] [extra log line]
mk() {
  mkdir -p "$d/$1"
  {
    if [ "$2" = 1 ]; then
      echo "(EngineCore pid=1) INFO 11-07 10:00:00 [numa_utils.py:395] Binding EngineCore subprocess (local_rank=0) to NUMA nodes 0,3,4,7"
      for i in 0 1 2 3 4 5 6 7; do echo "(EngineCore pid=1) INFO 11-07 10:00:00 [numa_utils.py:363] Binding worker subprocess (local_rank=$i, gpu_index=$i) to NUMA node ${node_of[$i]}"; done
    fi
    echo "(EngineCore pid=1) INFO 11-07 10:00:30 [simple_cpu_offload_connector.py:151] SimpleCPUOffloadConnector: role=SCHEDULER, per_rank=24.00 GB, world_size=8, mode=eager, backend=cpu, disk=none"
    echo "(EngineCore pid=1) INFO 11-07 10:00:30 [manager.py:196] SimpleCPUOffloadScheduler: Allocating $3 offload blocks (24.00 GB, mode=eager, backend=cpu)"
    for i in 0 1 2 3 4 5 6 7; do echo "(Worker_TP$i pid=2) INFO 11-07 10:00:40 [worker.py:230] SimpleCPUOffloadWorker [CPU]: ${5:-5} tensors, $4 CPU blocks (24.00 GB)"; done
    echo "INFO 11-07 10:00:40 [gpu_model_runner.py:1] Using LBNHC KV cache layout."
    [ -n "${6:-}" ] && echo "$6"
  } > "$d/$1/serve.log"
  # serve_cmd.txt as the harness writes it (printf %q of every argument)
  echo 'vllm serve m --kv-cache-memory-bytes 11500000000 --numa-bind --kv-transfer-config \{\"kv_connector\":\"SimpleCPUOffloadConnector\"\,\"kv_role\":\"kv_both\"\,\"kv_connector_extra_config\":\{\"cpu_bytes_to_use\":206158397440\,\"cpu_bytes_to_use_per_rank\":25769799680\}\}' > "$d/$1/serve_cmd.txt"
  [ "$2" = 0 ] && sed -i 's/ --numa-bind//' "$d/$1/serve_cmd.txt"
}
# run <name> <rc wanted> <description> [env assignments...] -- <args to the script>
run() {
  local name=$1 want=$2 desc=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  out=$(env -i PATH="$PATH" "${envs[@]}" bash "$check" "$@" 2>&1); rc=$?
  if [ "$rc" != "$want" ]; then echo "FAIL case '$desc': rc=$rc wanted $want"; echo "$out" | sed 's/^/    /'; fails=$((fails+1))
  elif [ -n "${MSG:-}" ] && ! echo "$out" | grep -q -- "$MSG"; then echo "FAIL case '$desc': rc ok but the message lacks '$MSG'"; echo "$out" | sed 's/^/    /'; fails=$((fails+1))
  else echo "ok   $desc"; fi
}
# fixture numbers (a 24 GiB tier, spec bytes per block-id 20,480, a KV maximum of 11.5 GB) are synthetic and deliberately not the values of env/offload.env (0.33.0: P* 51,814,072,320, spec maximum 11,645,337,600); the cases test the check logic, not the values
B=25769799680; BPB=20480; N=$(( B / BPB )); KVMAX=11500000000
# the full expectation of a launch (what env/offload.env exports): every case of the simple mode needs all three
X="MUSE_OFFLOAD_EXPECT_BYTES=$B MUSE_OFFLOAD_EXPECT_BPB=$BPB MUSE_OFFLOAD_EXPECT_KV_MAX=$KVMAX"
mk good 1 $N $N
run good 0 "bound, 8 workers agree with the scheduler, bytes, bpb and the kv maximum match" $X -- "$d/good/serve.log"
MSG='per-rank tier bytes unknown' run good 1 "no expectation at all (fail-closed)" -- "$d/good/serve.log"
MSG='per-rank tier bytes unknown' run good 1 "per-rank bytes unknown (fail-closed)" MUSE_OFFLOAD_EXPECT_BPB=$BPB MUSE_OFFLOAD_EXPECT_KV_MAX=$KVMAX -- "$d/good/serve.log"
MSG='bytes per block-id unknown' run good 1 "bytes per block-id unknown (fail-closed)" MUSE_OFFLOAD_EXPECT_BYTES=$B MUSE_OFFLOAD_EXPECT_KV_MAX=$KVMAX -- "$d/good/serve.log"
MSG='KV bytes maximum unknown' run good 1 "kv maximum unknown (fail-closed)" MUSE_OFFLOAD_EXPECT_BYTES=$B MUSE_OFFLOAD_EXPECT_BPB=$BPB -- "$d/good/serve.log"
run good 0 "expectation from the sourcing shell (env/offload.env exports)" MUSE_OFFLOAD=1 MUSE_OFFLOAD_PER_RANK_BYTES=$B MUSE_OFFLOAD_BPB=$BPB MUSE_OFFLOAD_KV_BYTES=$KVMAX -- "$d/good/serve.log"
MSG='bind expected off' run good 1 "bind expected off but the log is bound" $X MUSE_OFFLOAD_NUMA_BIND=0 -- "$d/good/serve.log"
MSG='expected off but' run good 1 "offload expected off but the log has the connector" MUSE_OFFLOAD=0 -- "$d/good/serve.log"
MSG='per-rank bytes' run good 1 "wrong per-rank bytes" MUSE_OFFLOAD_EXPECT_BYTES=$((B / 2)) MUSE_OFFLOAD_EXPECT_BPB=$BPB MUSE_OFFLOAD_EXPECT_KV_MAX=$KVMAX -- "$d/good/serve.log"
run good 0 "kv bytes at the maximum" $X -- "$d/good/serve.log"
MSG='above the offload-on maximum' run good 1 "kv bytes above the offload-on maximum" MUSE_OFFLOAD_EXPECT_BYTES=$B MUSE_OFFLOAD_EXPECT_BPB=$BPB MUSE_OFFLOAD_EXPECT_KV_MAX=11400000000 -- "$d/good/serve.log"
MSG='reported an error' run good 1 "env/offload.env reported an error (unknown profile)" $X MUSE_OFFLOAD_ERROR="unknown MUSE_OFFLOAD_PROFILE=ns265" -- "$d/good/serve.log"
MSG='neither 0 nor 1' run good 1 "MUSE_OFFLOAD is neither 0 nor 1" $X MUSE_OFFLOAD=yes -- "$d/good/serve.log"
mk nocmd 1 $N $N
rm "$d/nocmd/serve_cmd.txt"
MSG='no serve_cmd.txt' run nocmd 1 "no serve_cmd.txt next to the log (fail-closed)" $X -- "$d/nocmd/serve.log"
run nocmd 0 "no serve_cmd.txt, log-only check accepted" $X MUSE_OFFLOAD_ALLOW_LOG_ONLY=1 -- "$d/nocmd/serve.log"
mk nbnodes 1 $N $N
sed -i 's/ --numa-bind / --numa-bind-nodes 3 0 7 4 /' "$d/nbnodes/serve_cmd.txt"
MSG='no --numa-bind' run nbnodes 1 "launch line has only --numa-bind-nodes (not the bind flag)" $X -- "$d/nbnodes/serve.log"
mk nbend 1 $N $N
sed -i 's/ --numa-bind / /; s/$/ --numa-bind/' "$d/nbend/serve_cmd.txt"
run nbend 0 "--numa-bind as the last argument of the launch line" $X -- "$d/nbend/serve.log"
mk nokv 1 $N $N
sed -i 's/ --kv-cache-memory-bytes 11500000000//' "$d/nokv/serve_cmd.txt"
MSG='no --kv-cache-memory-bytes' run nokv 1 "no --kv-cache-memory-bytes in the launch line while a maximum is demanded" $X -- "$d/nokv/serve.log"
mk nobind 0 $N $N
run nobind 0 "bind off and expected off" $X MUSE_OFFLOAD_NUMA_BIND=0 -- "$d/nobind/serve.log"
MSG='Binding worker subprocess' run nobind 1 "bind expected on but the log has no binding" $X -- "$d/nobind/serve.log"
mk mism 1 $N $((N - 1))
MSG='scheduler hands out' run mism 1 "workers allocated fewer blocks than the scheduler hands out" $X -- "$d/mism/serve.log"
mk tens 1 $N $N 5
sed -i '0,/5 tensors/s//16 tensors/' "$d/tens/serve.log"
MSG='disagree on tensors' run tens 1 "one worker with another tensor count" $X -- "$d/tens/serve.log"
mk pin 1 $N $N 5 "(Worker_TP0 pid=2) WARNING 11-07 [worker.py:1] Pinned memory not available. CPU offload performance may be degraded."
MSG='pinned memory not available' run pin 1 "pinned memory not available" $X -- "$d/pin/serve.log"
mk reg 1 $N $N 5 "WARNING cudaHostRegister failed (code=2)"
MSG='cudaHostRegister failed' run reg 1 "cudaHostRegister failed" $X -- "$d/reg/serve.log"
mk dis 1 $N $N 5 "WARNING cpu_bytes_to_use_per_rank (24.00 GB) != cpu_bytes_to_use/world_size (1.00 GB). Using per-rank value."
MSG='disagrees with cpu_bytes_to_use' run dis 1 "per-rank and total bytes disagree" $X -- "$d/dis/serve.log"
mk mm 1 $N $N 5 "(Worker_TP0 pid=2) INFO 11-07 [encoder_budget.py:98] enable_mm_embeds is True; modalities handled as embedding-only: ('audio', 'video', 'image')"
MSG='enable_mm_embeds is True' run mm 1 "tree without the receives_remote_kv fix (enable_mm_embeds is True)" $X -- "$d/mm/serve.log"
mk lay 1 $N $N
sed -i 's/LBNHC/LBHNC/' "$d/lay/serve.log"
MSG='LBNHC' run lay 1 "layout flipped" $X -- "$d/lay/serve.log"
mk seven 1 $N $N
sed -i 's/(local_rank=7, gpu_index=7) to NUMA node 4/(local_rank=7, gpu_index=7) to NUMA node 7/' "$d/seven/serve.log"
MSG='rank 7 is not bound' run seven 1 "rank 7 bound to the wrong node" $X -- "$d/seven/serve.log"
mk seven2 1 $N $N
sed -i '/local_rank=3,/d' "$d/seven2/serve.log"
MSG='found 7' run seven2 1 "a missing binding line" $X -- "$d/seven2/serve.log"
mk cmdbad 1 $N $N
echo "vllm serve m --kv-cache-memory-bytes $KVMAX --numa-bind" > "$d/cmdbad/serve_cmd.txt"
MSG='SimpleCPUOffloadConnector' run cmdbad 1 "serve_cmd.txt without the connector (readable log)" $X -- "$d/cmdbad/serve.log"
run cmdbad 2 "unreadable log" $X -- "$d/nope.log"
run cmdbad 2 "usage" $X --
mkdir -p "$d/off"; { echo "(Worker_TP0 pid=2) INFO nothing about offload"; } > "$d/off/serve.log"
run off 0 "offload off, no bind, no connector lines" MUSE_OFFLOAD=0 MUSE_OFFLOAD_NUMA_BIND=0 -- "$d/off/serve.log"
run off 0 "offload off without MUSE_OFFLOAD_NUMA_BIND: the off launch expects no bind lines (an integrated profile exports it as 1)" MUSE_OFFLOAD=0 MUSE_OFFLOAD_NUMA_BIND=1 -- "$d/off/serve.log"
MSG='bind expected off' run good 1 "off launch whose log carries the binding lines" MUSE_OFFLOAD=0 -- "$d/good/serve.log"
run off 0 "offload off (explicit expectation)" MUSE_OFFLOAD_EXPECT=off MUSE_OFFLOAD_EXPECT_BIND=0 -- "$d/off/serve.log"
MSG='SimpleCPUOffloadScheduler allocation line' run off 1 "connector expected but the log has none" $X MUSE_OFFLOAD_EXPECT=simple MUSE_OFFLOAD_EXPECT_BIND=0 -- "$d/off/serve.log"
echo "$fails case(s) failed"
exit $(( fails > 0 ))
