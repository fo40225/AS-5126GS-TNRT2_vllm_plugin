#!/bin/bash
# Regression test of scripts/check_offload_launch.sh (S1-10: binding lines, scheduler / worker block agreement, layout, serve_cmd.txt) on
# synthetic boot logs. No GPU, no vLLM: usage  bash bench/test_check_offload_launch.sh   Exit 0 = all cases as expected.
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
  if [ "$rc" != "$want" ]; then echo "FAIL case '$desc': rc=$rc wanted $want"; echo "$out" | sed 's/^/    /'; fails=$((fails+1)); else echo "ok   $desc"; fi
}
B=25769799680; BPB=20480; N=$(( B / BPB ))
mk good 1 $N $N
run good 0 "bound, 8 workers agree with the scheduler, bytes and bpb match" MUSE_OFFLOAD_EXPECT_BYTES=$B MUSE_OFFLOAD_EXPECT_BPB=$BPB -- "$d/good/serve.log"
run good 0 "no byte expectation" -- "$d/good/serve.log"
run good 1 "bind expected off but the log is bound" MUSE_OFFLOAD_NUMA_BIND=0 -- "$d/good/serve.log"
run good 1 "offload expected off but the log has the connector" MUSE_OFFLOAD=0 -- "$d/good/serve.log"
run good 1 "wrong per-rank bytes" MUSE_OFFLOAD_EXPECT_BYTES=$((B / 2)) MUSE_OFFLOAD_EXPECT_BPB=$BPB -- "$d/good/serve.log"
run good 0 "kv bytes at the maximum" MUSE_OFFLOAD_EXPECT_KV_MAX=11500000000 -- "$d/good/serve.log"
run good 1 "kv bytes above the offload-on maximum" MUSE_OFFLOAD_EXPECT_KV_MAX=11400000000 -- "$d/good/serve.log"
mk nokv 1 $N $N
sed -i 's/ --kv-cache-memory-bytes 11500000000//' "$d/nokv/serve_cmd.txt"
run nokv 1 "no --kv-cache-memory-bytes in the launch line while a maximum is demanded" MUSE_OFFLOAD_EXPECT_KV_MAX=11500000000 -- "$d/nokv/serve.log"
run nokv 0 "no maximum demanded" -- "$d/nokv/serve.log"
mk nobind 0 $N $N
run nobind 0 "bind off and expected off" MUSE_OFFLOAD_NUMA_BIND=0 -- "$d/nobind/serve.log"
run nobind 1 "bind expected on but the log has no binding" -- "$d/nobind/serve.log"
mk mism 1 $N $((N - 1))
run mism 1 "workers allocated fewer blocks than the scheduler hands out" -- "$d/mism/serve.log"
mk tens 1 $N $N 5
sed -i '0,/5 tensors/s//16 tensors/' "$d/tens/serve.log"
run tens 1 "one worker with another tensor count" -- "$d/tens/serve.log"
mk pin 1 $N $N 5 "(Worker_TP0 pid=2) WARNING 11-07 [worker.py:1] Pinned memory not available. CPU offload performance may be degraded."
run pin 1 "pinned memory not available" -- "$d/pin/serve.log"
mk reg 1 $N $N 5 "WARNING cudaHostRegister failed (code=2)"
run reg 1 "cudaHostRegister failed" -- "$d/reg/serve.log"
mk dis 1 $N $N 5 "WARNING cpu_bytes_to_use_per_rank (24.00 GB) != cpu_bytes_to_use/world_size (1.00 GB). Using per-rank value."
run dis 1 "per-rank and total bytes disagree" -- "$d/dis/serve.log"
mk mm 1 $N $N 5 "(Worker_TP0 pid=2) INFO 11-07 [encoder_budget.py:98] enable_mm_embeds is True; modalities handled as embedding-only: ('audio', 'video', 'image')"
run mm 1 "tree without the receives_remote_kv fix (enable_mm_embeds is True)" -- "$d/mm/serve.log"
mk lay 1 $N $N
sed -i 's/LBNHC/LBHNC/' "$d/lay/serve.log"
run lay 1 "layout flipped" -- "$d/lay/serve.log"
mk seven 1 $N $N
sed -i 's/(local_rank=7, gpu_index=7) to NUMA node 4/(local_rank=7, gpu_index=7) to NUMA node 7/' "$d/seven/serve.log"
run seven 1 "rank 7 bound to the wrong node" -- "$d/seven/serve.log"
mk seven2 1 $N $N
sed -i '/local_rank=3,/d' "$d/seven2/serve.log"
run seven2 1 "a missing binding line" -- "$d/seven2/serve.log"
mk cmdbad 1 $N $N
echo "vllm serve m --numa-bind" > "$d/cmdbad/serve_cmd.txt"
run cmdbad 1 "serve_cmd.txt without the connector (readable log)" -- "$d/cmdbad/serve.log"
run cmdbad 2 "unreadable log" -- "$d/nope.log"
run cmdbad 2 "usage" --
mkdir -p "$d/off"; { echo "(Worker_TP0 pid=2) INFO nothing about offload"; } > "$d/off/serve.log"
run off 0 "offload off, no bind, no connector lines" MUSE_OFFLOAD=0 MUSE_OFFLOAD_NUMA_BIND=0 -- "$d/off/serve.log"
run off 0 "offload off (explicit expectation)" MUSE_OFFLOAD_EXPECT=off MUSE_OFFLOAD_EXPECT_BIND=0 -- "$d/off/serve.log"
run off 1 "connector expected but the log has none" MUSE_OFFLOAD_EXPECT=simple MUSE_OFFLOAD_EXPECT_BIND=0 -- "$d/off/serve.log"
echo "$fails case(s) failed"
exit $(( fails > 0 ))
