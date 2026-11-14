#!/bin/bash
# Confirms in the log of a speculative boot that the launch profile of env/spec_k7.env took effect.
# Usage: check_spec_launch.sh <serve.log> [expected_sp_floor [expected_graph_cap]]    (defaults 384 and 256)
# The vLLM branch logs once per process: "sequence-parallel residual stream for batches of at least <floor> tokens
# (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=<env>, CUDA graphs up to <cap> tokens)". The floor is
# max(env, 257, cap + 1): the env value alone does not make a launch (floor 513 with the default capture list).
# When serve_cmd.txt (written by the harness) sits next to the log, it must carry --max-cudagraph-capture-size <cap> too
# (not checked for cap 512, the default capture list, which has no flag).
# Exit 0 = profile in effect; 1 = a line with other values, no line, or the flag is missing; 2 = usage.
log=${1:?usage: check_spec_launch.sh <serve.log> [floor [cap]]}
floor=${2:-384}
cap=${3:-256}
[ -r "$log" ] || { echo "check_spec_launch: cannot read $log" >&2; exit 2; }
pattern='sequence-parallel residual stream for batches of at least [0-9]+ tokens \(VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=[0-9]+, CUDA graphs up to [0-9]+ tokens\)'
seen=$(grep -a -o -E "$pattern" "$log" | sed -E 's/.*at least ([0-9]+) tokens.*CUDA graphs up to ([0-9]+) tokens\)/\1 \2/' | sort | uniq -c)
if [ -z "$seen" ]; then
  echo "FAIL no sequence-parallel line in $log (flag off, another model, or the boot did not get that far)"; exit 1
fi
if [ "$(echo "$seen" | wc -l)" != 1 ] || ! echo "$seen" | grep -q -E "^ *[0-9]+ $floor $cap\$"; then
  echo "FAIL expected floor $floor with graphs up to $cap; found (count floor cap):"; echo "$seen"; exit 1
fi
pool=$(grep -a -o -E 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB' "$log" | sort | uniq -c | tr '\n' ';')
rc=0
cmd=$(dirname "$log")/serve_cmd.txt
if [ "$cap" != 512 ] && [ -r "$cmd" ] && ! grep -q -E -e "--max-cudagraph-capture-size $cap( |$)" "$cmd"; then
  echo "FAIL serve_cmd.txt has no --max-cudagraph-capture-size $cap"; rc=1
fi
[ $rc = 0 ] && echo "OK floor=$floor graphs<=$cap ($pool)"
exit $rc
