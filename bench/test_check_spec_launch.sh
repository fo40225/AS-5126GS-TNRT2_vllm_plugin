#!/bin/bash
# Regression test of scripts/check_spec_launch.sh (S1-9 L11: the graph-floor line, the cap > 256 bytes rule and the default floor of a
# larger capture list, S1-14 F2: the scheduled-token cap) on synthetic boot logs. No GPU, no vLLM: usage  bash bench/test_check_spec_launch.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
check=$here/scripts/check_spec_launch.sh
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
fails=0
sp_line() { echo "INFO MuseSpark SM120: sequence-parallel residual stream for batches of at least $1 tokens (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=384, CUDA graphs up to $2 tokens)."; }
# gsp_line <graph floor> <env> <eager floor> [engine limit, default 128]
gsp_line() { echo "INFO MuseSpark SM120: sequence parallel inside FULL decode CUDA graphs for batches of at least $1 tokens (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=$2; decode MoE engines serve up to ${4:-128} tokens; eager, mixed and PIECEWISE batches keep the floor of $3 tokens)."; }
# the graph manager's line of every worker (8): <flagged> <range or none> <other FULL> <PIECEWISE>
flagged_line() { for _ in 1 2 3 4 5 6 7 8; do echo "INFO CUDA graphs: $1 FULL decode graphs are flagged full_decode_graph (token counts $2), $3 other FULL graphs, $4 PIECEWISE graphs."; echo "Model loading took 70.8 GiB"; done; }
# vLLM's startup line of the API server (arg_utils / startup.py:96): the parsed non-default args; <mnst> empty = no --max-num-scheduled-tokens on the command line
nda_line() { echo "(APIServer pid=1) INFO 11-10 19:49:51 [startup.py:96] non-default args: {'model_tag': 'm', 'tensor_parallel_size': 8, 'kv_cache_memory_bytes': 11720212480, 'max_num_batched_tokens': 8199${1:+, 'max_num_scheduled_tokens': $1}, 'max_num_seqs': 128, 'max_cudagraph_capture_size': 256}"; }
# mk <name> <eager floor> <cap> <graph floor line args or ''> [bytes] [flagged line args, default: 33 FULL decode graphs 8..256 for cap 256, 48 8..384 for cap 384]
# (MNST=<n> in the environment of mk sets the scheduled-token cap of the log and the command line, default 8192; MNST= (empty) = no flag)
mk() {
  mkdir -p "$d/$1"
  local fl="${6:-}" mst="${MNST-8192}"
  if [ -z "$fl" ]; then if [ "$3" = 384 ]; then fl="48 8..384 1 56"; else fl="32 8..256 1 40"; fi; fi
  { sp_line "$2" "$3"; [ -n "$4" ] && gsp_line $4; flagged_line $fl; echo "Chunked prefill is enabled with max_num_batched_tokens=8199"; nda_line "$mst"; echo "Graph capturing finished in 30 secs, took 0.95 GiB"; } > "$d/$1/serve.log"
  echo "vllm serve m --max-cudagraph-capture-size $3 --max-num-batched-tokens 8199${mst:+ --max-num-scheduled-tokens $mst} --kv-cache-memory-bytes ${5:-9000000000}" > "$d/$1/serve_cmd.txt"
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
run a1a 0 "A1a boot, expectation from the vLLM variable of the sourced profile" VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=136 -- "$d/a1a/serve.log"
run a1a 1 "A1a boot, profile exports a different vLLM floor (264)" VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=264 -- "$d/a1a/serve.log"
run prod 1 "production boot, profile exports the graph floor but the boot has no line" VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=136 -- "$d/prod/serve.log"
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
echo 'vllm serve m --compilation-config \{\"cudagraph_capture_sizes\":\[1,2,4,8,256,264,384\],\"max_cudagraph_capture_size\":384\} --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 10900000000' > "$d/a1b_json/serve_cmd.txt"
run a1b_json 0 "cap 384 as an escaped compilation-config JSON (explicit capture list)" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_json/serve.log"
echo 'vllm serve m --compilation-config {"max_cudagraph_capture_size":384} --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 10900000000' > "$d/a1b_json/serve_cmd.txt"
run a1b_json 0 "cap 384 as a plain compilation-config JSON" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_json/serve.log"
echo 'vllm serve m --compilation-config \{\"max_cudagraph_capture_size\":256\} --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 10900000000' > "$d/a1b_json/serve_cmd.txt"
run a1b_json 1 "serve_cmd.txt names another escaped cap (256) than the boot (384)" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_json/serve.log"
mk a1b_nograph 385 384 "" 10900000000
run a1b_nograph 1 "cap 384 without the graph line while it is demanded" MUSE_SPEC_GRAPH_CAP=384 SPEC_EXPECT_GRAPH_SP=136 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_nograph/serve.log"
run a1b_nograph 0 "cap 384, graph floor off (capture list only)" MUSE_SPEC_GRAPH_CAP=384 SPEC_CAP_MAX_BYTES=10900000000 -- "$d/a1b_nograph/serve.log"
# S1-9 L11 review fixes: the graph floor is armed but no flagged FULL decode graph is at or above it
mk a1a_inert 384 256 "136 136 384" 9000000000 "0 none 40 0"
run a1a_inert 1 "armed floor, no FULL decode graph (FULL mixed / PIECEWISE only): inert" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1a_inert/serve.log"
mk a1a_lowgraphs 384 256 "136 136 384" 9000000000 "8 8..64 33 40"
run a1a_lowgraphs 1 "armed floor 136, flagged graphs only up to 64 tokens" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1a_lowgraphs/serve.log"
mkdir -p "$d/a1a_noflagline"; grep -v 'CUDA graphs:' "$d/a1a/serve.log" > "$d/a1a_noflagline/serve.log"; cp "$d/a1a/serve_cmd.txt" "$d/a1a_noflagline/"
run a1a_noflagline 1 "armed floor, boot of a tree without the flagged-graph line" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1a_noflagline/serve.log"
mkdir -p "$d/a1a_mixedlines"; { cat "$d/a1a/serve.log"; echo "INFO CUDA graphs: 30 FULL decode graphs are flagged full_decode_graph (token counts 8..240), 1 other FULL graphs, 40 PIECEWISE graphs."; } > "$d/a1a_mixedlines/serve.log"; cp "$d/a1a/serve_cmd.txt" "$d/a1a_mixedlines/"
run a1a_mixedlines 1 "workers disagree on the flagged graphs" SPEC_EXPECT_GRAPH_SP=136 -- "$d/a1a_mixedlines/serve.log"
run prod 0 "production boot prints flagged=32@8..256 without arming" -- "$d/prod/serve.log"
# the engine limit comes from the line: DECODE_MAX_TOKENS=64 gives engine limit 64, floor 65
mk eng64 384 256 "65 1 384 64"
run eng64 0 "engine limit 64: flag 1 raised to 65 on the line" MUSE_SPEC_GRAPH_SP_MIN_TOKENS=1 -- "$d/eng64/serve.log"
run eng64 0 "engine limit 64: graph floor 65 demanded" SPEC_EXPECT_GRAPH_SP=65 -- "$d/eng64/serve.log"
mk eng64_raw 384 256 "64 1 384 64"
run eng64_raw 1 "engine limit 64 but the line says floor 64 (not above the engine)" -- "$d/eng64_raw/serve.log"
mk eng8 384 256 "9 1 384 8"
run eng8 0 "engine limit 8 (b12x only): floor 9" MUSE_SPEC_GRAPH_SP_MIN_TOKENS=1 -- "$d/eng8/serve.log"
mk wrongraise 384 256 "200 136 384"
run wrongraise 1 "graph floor 200 is not max(env 136, engine 128 + 1)" -- "$d/wrongraise/serve.log"
# S1-13 review: the flag-2 pairing (r2e on + aux projection on the target's rows + FP8 draft flag 2 / cutlass) against FP8_F2_MAX_BYTES: 11,720,212,480 is
# accepted, one byte more is rejected (the a1a_big case above only fails through the r2e-off maximum, so a wrong flag-2 constant would not be noticed)
mk_f2() {
  mk "$1" 384 256 "" "$2"
  { cat "$d/$1/serve.log"; echo "INFO MuseSpark SM120: prefill batches of at least 512 tokens run as 2 microbatches (r2e)."
    echo "INFO MuseSpark SM120: the aux projection runs in the target's forward (VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD=2)."
    for _ in 1 2 3 4 5 6 7 8; do
      echo "INFO DFlash2 draft weights in FP8 (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2, kernel cutlass): 30 linear layers quantized"
      echo "INFO DFlash2 candidate head in FP8 (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2)"
    done; } > "$d/$1/serve.log.new" && mv "$d/$1/serve.log.new" "$d/$1/serve.log"
}
mk_f2 f2_at 11720212480
run f2_at 0 "flag-2 pairing (r2e, aux projection, FP8 2 cutlass) at 11,720,212,480" -- "$d/f2_at/serve.log"
mk_f2 f2_over 11720212481
run f2_over 1 "flag-2 pairing, one byte above FP8_F2_MAX_BYTES (11,720,212,481)" -- "$d/f2_over/serve.log"
# S1-14 F2: the scheduled-token cap (--max-num-scheduled-tokens 8192) in the log line and the command line
run prod 0 "production boot with the cap 8192 (log line and flag), default expectation" -- "$d/prod/serve.log"
run prod 0 "production boot, cap 8192 demanded explicitly" SPEC_EXPECT_MNST=8192 -- "$d/prod/serve.log"
run prod 1 "production boot, another cap (4096) demanded" SPEC_EXPECT_MNST=4096 -- "$d/prod/serve.log"
run prod 1 "production boot has the cap but 'no cap' is demanded" SPEC_EXPECT_MNST=off -- "$d/prod/serve.log"
run prod 1 "production boot has the cap but 0 is demanded (= off)" SPEC_EXPECT_MNST=0 -- "$d/prod/serve.log"
run prod 2 "garbage cap expectation" SPEC_EXPECT_MNST=8k -- "$d/prod/serve.log"
MNST= mk nocap 384 256 ""
run nocap 1 "boot without the flag (the plain 0.33.0 port: 8199-token chunks), default expectation" -- "$d/nocap/serve.log"
run nocap 1 "boot without the flag, cap 8192 demanded" SPEC_EXPECT_MNST=8192 -- "$d/nocap/serve.log"
run nocap 0 "boot without the flag, off demanded" SPEC_EXPECT_MNST=off -- "$d/nocap/serve.log"
run nocap 0 "boot without the flag, 0 demanded" SPEC_EXPECT_MNST=0 -- "$d/nocap/serve.log"
MNST=4096 mk othercap 384 256 ""
run othercap 1 "boot with cap 4096, default expectation 8192" -- "$d/othercap/serve.log"
run othercap 0 "boot with cap 4096, 4096 demanded" SPEC_EXPECT_MNST=4096 -- "$d/othercap/serve.log"
run othercap 1 "boot with cap 4096, off demanded" SPEC_EXPECT_MNST=off -- "$d/othercap/serve.log"
# log and command line disagree: the flag is missing from serve_cmd.txt (log line present), or only in the command line (no log entry)
mkdir -p "$d/cmdonly"; cp "$d/nocap/serve.log" "$d/cmdonly/serve.log"; sed 's/ --max-num-batched-tokens 8199/ --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192/' "$d/nocap/serve_cmd.txt" > "$d/cmdonly/serve_cmd.txt"
run cmdonly 1 "flag on the command line but not in the parsed args of the log" -- "$d/cmdonly/serve.log"
run cmdonly 1 "flag on the command line, off demanded" SPEC_EXPECT_MNST=off -- "$d/cmdonly/serve.log"
mkdir -p "$d/logonly"; cp "$d/prod/serve.log" "$d/logonly/serve.log"; sed 's/ --max-num-scheduled-tokens 8192//' "$d/prod/serve_cmd.txt" > "$d/logonly/serve_cmd.txt"
run logonly 1 "cap in the log line but not on the command line of serve_cmd.txt" -- "$d/logonly/serve.log"
mkdir -p "$d/noargs"; grep -v 'non-default args' "$d/prod/serve.log" > "$d/noargs/serve.log"; cp "$d/prod/serve_cmd.txt" "$d/noargs/"
run noargs 1 "no non-default args line in the log: the cap cannot be confirmed" -- "$d/noargs/serve.log"
run noargs 0 "no non-default args line, off demanded and not on the command line" SPEC_EXPECT_MNST=off -- "$d/nocap/serve.log"
mkdir -p "$d/nocmd"; cp "$d/prod/serve.log" "$d/nocmd/serve.log"
run nocmd 0 "log only (no serve_cmd.txt), cap 8192 in the parsed args" -- "$d/nocmd/serve.log"
mkdir -p "$d/mixedcap"; { cat "$d/prod/serve.log"; nda_line 4096; } > "$d/mixedcap/serve.log"; cp "$d/prod/serve_cmd.txt" "$d/mixedcap/"
run mixedcap 1 "two non-default args lines with different caps" -- "$d/mixedcap/serve.log"
mkdir -p "$d/sub"; { cat "$d/prod/serve.log"; echo "INFO other line mentioning 'max_num_scheduled_tokens': 4096 outside the args line"; } > "$d/sub/serve.log"; cp "$d/prod/serve_cmd.txt" "$d/sub/"
run sub 0 "a max_num_scheduled_tokens text outside the non-default args line is ignored" -- "$d/sub/serve.log"
# the OK line names the cap, and the KV bytes of the capped profile are the bytes of the flag-2 pairing (the cap moves no bytes)
out=$(env -i PATH="$PATH" bash "$check" "$d/f2_at/serve.log" 2>&1); if echo "$out" | grep -q 'mnbt=8199 mnst=8192'; then echo "ok   the OK line carries mnbt=8199 mnst=8192"; else echo "FAIL OK line: $out"; fails=$((fails+1)); fi
# env/spec_k7.env: off is a property of sourcing the profile
got=$(env -i PATH="$PATH" bash -c "MUSE_SPEC_GRAPH_SP_MIN_TOKENS=136; source $here/env/spec_k7.env; a=\${VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS:-unset}; MUSE_SPEC_GRAPH_SP_MIN_TOKENS=0; source $here/env/spec_k7.env; echo \$a \${VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS:-unset}" 2>&1)
if [ "$got" = "136 unset" ]; then echo "ok   spec_k7.env: a re-source with the flag at 0 unsets the vLLM variable"; else echo "FAIL spec_k7.env re-source: got '$got' wanted '136 unset'"; fails=$((fails+1)); fi
got=$(env -i PATH="$PATH" VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=264 MUSE_SPEC_GRAPH_SP_MIN_TOKENS=0 bash -c "source $here/env/spec_k7.env; echo \${VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS:-unset}" 2>&1)
if [ "$got" = "unset" ]; then echo "ok   spec_k7.env: a stray vLLM variable does not survive the profile at flag 0"; else echo "FAIL spec_k7.env stray variable: got '$got'"; fails=$((fails+1)); fi
# the adopted default (S1-9 L11 gate): the profile arms the graph floor 136 on the production capture list, bytes unchanged
got=$(env -i PATH="$PATH" bash -c "source $here/env/spec_k7.env; echo \${VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS:-unset} \$MUSE_SPEC_GRAPH_CAP \$MUSE_SPEC_KV_BYTES \"\$MUSE_SPEC_SERVE_ARGS\"" 2>&1)
if [ "$got" = "136 256 11720212480 --max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192" ]; then echo "ok   spec_k7.env: the default profile is A1a (graph floor 136, cap 256, mnst 8192, production bytes of the 0.33.0 line)"; else echo "FAIL spec_k7.env default profile: got '$got'"; fails=$((fails+1)); fi
got=$(env -i PATH="$PATH" MUSE_SPEC_MNST=0 bash -c "source $here/env/spec_k7.env; echo \$MUSE_SPEC_KV_BYTES \"\$MUSE_SPEC_SERVE_ARGS\"" 2>&1)
if [ "$got" = "11720212480 --max-cudagraph-capture-size 256 --max-num-batched-tokens 8199" ]; then echo "ok   spec_k7.env: MUSE_SPEC_MNST=0 drops the flag, bytes unchanged"; else echo "FAIL spec_k7.env MNST=0: got '$got'"; fails=$((fails+1)); fi
got=$(env -i PATH="$PATH" MUSE_SPEC_MNST=4096 bash -c "source $here/env/spec_k7.env; echo \$MUSE_SPEC_KV_BYTES \"\$MUSE_SPEC_SERVE_ARGS\"" 2>&1)
if [ "$got" = "11720212480 --max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 4096" ]; then echo "ok   spec_k7.env: MUSE_SPEC_MNST=4096 overrides the cap, bytes unchanged"; else echo "FAIL spec_k7.env MNST=4096: got '$got'"; fails=$((fails+1)); fi
[ $fails = 0 ] && echo "ALL CASES PASSED" || echo "$fails case(s) FAILED"
exit $((fails > 0))
