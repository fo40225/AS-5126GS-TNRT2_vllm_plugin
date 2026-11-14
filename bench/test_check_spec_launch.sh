#!/bin/bash
# Regression test of scripts/check_spec_launch.sh (S1-9 L11: the graph-floor line, the cap > 256 bytes rule and the default floor of a
# larger capture list) on synthetic boot logs. No GPU, no vLLM: usage  bash bench/test_check_spec_launch.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
check=$here/scripts/check_spec_launch.sh
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
fails=0
sp_line() { echo "INFO MuseSpark SM120: sequence-parallel residual stream for batches of at least $1 tokens (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=384, CUDA graphs up to $2 tokens)."; }
gsp_line() { echo "INFO MuseSpark SM120: sequence parallel inside FULL decode CUDA graphs for batches of at least $1 tokens (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=$2; eager, mixed and PIECEWISE batches keep the floor of $3 tokens)."; }
# mk <name> <eager floor> <cap> <graph floor line args or ''> [bytes]
mk() {
  mkdir -p "$d/$1"
  { sp_line "$2" "$3"; [ -n "$4" ] && gsp_line $4; echo "Chunked prefill is enabled with max_num_batched_tokens=8199"; echo "Graph capturing finished in 30 secs, took 0.95 GiB"; } > "$d/$1/serve.log"
  echo "vllm serve m --max-cudagraph-capture-size $3 --max-num-batched-tokens 8199 --kv-cache-memory-bytes ${5:-9000000000}" > "$d/$1/serve_cmd.txt"
}
# expect <name> <rc wanted> <description> [env assignments...] -- <args to the script>
run() {
  local name=$1 want=$2 desc=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  out=$(env -i PATH="$PATH" "${envs[@]}" bash "$check" "$@" 2>&1); rc=$?
  if [ "$rc" != "$want" ]; then echo "FAIL case '$desc': rc=$rc wanted $want"; echo "$out" | sed 's/^/    /'; fails=$((fails+1)); else echo "ok   $desc"; fi
}
mk prod 384 256 ""
run prod 0 "production boot, no graph floor, no expectation" -- "$d/prod/serve.log"
run prod 0 "production boot, graph floor demanded off" SPEC_EXPECT_GRAPH_SP=off -- "$d/prod/serve.log"
run prod 1 "production boot, graph floor 136 demanded" SPEC_EXPECT_GRAPH_SP=136 -- "$d/prod/serve.log"
run prod 1 "production boot, flag exported in the shell but no log line" MUSE_SPEC_GRAPH_SP_MIN_TOKENS=136 -- "$d/prod/serve.log"
run prod 1 "production boot, vLLM env set in the shell but no log line" VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=136 -- "$d/prod/serve.log"
mk a1a 384 256 "136 136 384"
run a1a 0 "A1a boot (graphs 256, graph floor 136)" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1a/serve.log"
run a1a 0 "A1a boot, expectation from the shell flag" MUSE_SPEC_GRAPH_SP_MIN_TOKENS=136 -- "$d/a1a/serve.log"
run a1a 1 "A1a boot, graph floor demanded off" SPEC_EXPECT_GRAPH_SP=off -- "$d/a1a/serve.log"
run a1a 1 "A1a boot, another graph floor demanded" SPEC_EXPECT_GRAPH_SP=264 -- "$d/a1a/serve.log"
mk a1a_big 384 256 "136 136 384" 11000000000
run a1a_big 1 "A1a boot above the cap-256 bytes" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1a_big/serve.log"
mk a1a_small 384 256 "129 1 384"
run a1a_small 0 "flag 1 raised to 129 on the line" MUSE_SPEC_GRAPH_SP_MIN_TOKENS=1 -- "$d/a1a_small/serve.log"
mk a1a_low 384 256 "120 120 384"
run a1a_low 1 "graph floor inside the muse decode engine range" -- "$d/a1a_low/serve.log"
mk a1a_eager 384 256 "136 136 400"
run a1a_eager 1 "graph line names another eager floor" -- "$d/a1a_eager/serve.log"
mk a1b 385 384 "136 136 385" 10900000000
run a1b 1 "cap 384 (shell profile) without SPEC_CAP_MAX_BYTES" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1b/serve.log"
run a1b 0 "cap 384 (shell profile) with its bytes (default floor cap + 1)" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b/serve.log"
run a1b 1 "cap 384 above its bytes" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10800000000 -- "$d/a1b/serve.log"
run a1b 1 "cap 384 boot checked as the production profile (cap 256)" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1b/serve.log"
run a1b 0 "cap 384, floor and cap given" SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b/serve.log" 385 384
run a1b 1 "cap 384 with the cap-256 floor" SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b/serve.log" 384 384
mk a1b_json 385 384 "136 136 385" 10900000000
echo 'vllm serve m --compilation-config \{\"cudagraph_capture_sizes\":\[1,2,4,8,256,264,384\],\"max_cudagraph_capture_size\":384\} --max-num-batched-tokens 8199 --kv-cache-memory-bytes 10900000000' > "$d/a1b_json/serve_cmd.txt"
run a1b_json 0 "cap 384 as an escaped compilation-config JSON (explicit capture list)" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_json/serve.log"
echo 'vllm serve m --compilation-config {"max_cudagraph_capture_size":384} --max-num-batched-tokens 8199 --kv-cache-memory-bytes 10900000000' > "$d/a1b_json/serve_cmd.txt"
run a1b_json 0 "cap 384 as a plain compilation-config JSON" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_json/serve.log"
echo 'vllm serve m --compilation-config \{\"max_cudagraph_capture_size\":256\} --max-num-batched-tokens 8199 --kv-cache-memory-bytes 10900000000' > "$d/a1b_json/serve_cmd.txt"
run a1b_json 1 "serve_cmd.txt names another escaped cap (256) than the boot (384)" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_json/serve.log"
mk a1b_nograph 385 384 "" 10900000000
run a1b_nograph 1 "cap 384 without the graph line while it is demanded" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_nograph/serve.log"
run a1b_nograph 0 "cap 384, graph floor off (capture list only)" MUSE_SPEC_GRAPH_CAP=384 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_nograph/serve.log"
[ $fails = 0 ] && echo "ALL CASES PASSED" || echo "$fails case(s) FAILED"
exit $((fails > 0))
