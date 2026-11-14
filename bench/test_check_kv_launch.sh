#!/bin/bash
# Regression test of scripts/check_kv_launch.sh (S1-11: exact bytes, KV tokens / blocks, max-num-seqs, the ns256 capture cap, the speculative state) on the check-relevant lines of the
# real boot logs of the six production states (bench/fixtures/launch/, references in bench/fixtures/launch_ref/) and on mutations of them. No GPU, no vLLM:
# usage  bash bench/test_check_kv_launch.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
check=$here/scripts/check_kv_launch.sh
fx=$here/bench/fixtures
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
fails=0
# the independent expectations (literals from the gated boots, not read from the env files): bytes, tokens, blocks
declare -A BYTES=([spec_1]=11645337600 [spec_0]=11720212480 [ns256_1]=16854810624 [ns256_0]=16957243392 [ns128_1]=17257267200 [ns128_0]=17357799424)
declare -A TOK=([spec_1]=1639394 [spec_0]=1649935 [ns256_1]=2893281 [ns256_0]=2910864 [ns128_1]=2962366 [ns128_0]=2979623)
declare -A BLK=([spec_1]=568620 [spec_0]=572276 [ns256_1]=257184 [ns256_0]=258747 [ns128_1]=263325 [ns128_0]=264859)
# mk <name> <state>: a boot directory (serve.log, cache_config_info.txt, serve_cmd.txt) of a real state
mk() {
  mkdir -p "$d/$1"
  cp "$fx/launch/$2.log" "$d/$1/serve.log"; cp "$fx/launch/$2.cache_config_info.txt" "$d/$1/cache_config_info.txt"; cp "$fx/launch_ref/$2.serve_cmd.txt" "$d/$1/serve_cmd.txt"
}
# run <name> <rc wanted> <description> <profile> <tier>
run() {
  local name=$1 want=$2 desc=$3
  out=$(env -i PATH="$PATH" bash "$check" "$4" "$5" "$d/$name/serve.log" 2>&1); rc=$?
  if [ "$rc" != "$want" ]; then echo "FAIL case '$desc': rc=$rc wanted $want"; echo "$out" | sed 's/^/    /'; fails=$((fails+1))
  elif [ -n "${MSG:-}" ] && ! echo "$out" | grep -q -- "$MSG"; then echo "FAIL case '$desc': rc ok but the message lacks '$MSG'"; echo "$out" | sed 's/^/    /'; fails=$((fails+1))
  else echo "ok   $desc"; fi
}
for st in spec_1 spec_0 ns256_1 ns256_0 ns128_1 ns128_0; do
  p=${st%_*}; t=${st#*_}
  mk $st $st
  MSG="bytes=${BYTES[$st]} tokens=${TOK[$st]} blocks=${BLK[$st]}" run $st 0 "$st: the real boot passes with the independent numbers (${BYTES[$st]} B, ${TOK[$st]} tokens, ${BLK[$st]} blocks)" $p $t
  # the other tier state of the same profile must fail
  run $st 1 "$st: checked as the other tier state fails" $p $((1 - t))
done
# profile mix-ups
run ns256_1 1 "an ns256 boot checked as ns128 fails" ns128 1
run ns128_0 1 "an ns128 boot checked as ns256 fails" ns256 0
run spec_1 1 "a speculative boot checked as ns256 fails" ns256 1
run ns256_1 1 "a non-speculative boot checked as spec fails" spec 1
# usage
out=$(env -i PATH="$PATH" bash "$check" ns512 1 "$d/ns256_1/serve.log" 2>&1); [ $? = 2 ] && echo "ok   unknown profile: usage error (2)" || { echo "FAIL unknown profile"; fails=$((fails+1)); }
out=$(env -i PATH="$PATH" bash "$check" ns256 2 "$d/ns256_1/serve.log" 2>&1); [ $? = 2 ] && echo "ok   tier 2: usage error (2)" || { echo "FAIL tier 2"; fails=$((fails+1)); }
out=$(env -i PATH="$PATH" bash "$check" ns256 1 "$d/none.log" 2>&1); [ $? = 2 ] && echo "ok   unreadable log: exit 2" || { echo "FAIL unreadable log"; fails=$((fails+1)); }
# the launch line
mk m_bytes ns256_1; sed -i 's/--kv-cache-memory-bytes 16854810624/--kv-cache-memory-bytes 16957243392/' "$d/m_bytes/serve_cmd.txt"
MSG='serve_cmd.txt --kv-cache-memory-bytes' run m_bytes 1 "ns256 ON with the no-tier bytes on the launch line" ns256 1
mk m_mns ns128_0; sed -i 's/--max-num-seqs 128/--max-num-seqs 256/' "$d/m_mns/serve_cmd.txt"
MSG='serve_cmd.txt --max-num-seqs' run m_mns 1 "ns128 with --max-num-seqs 256 on the launch line" ns128 0
mk m_nocap ns256_0; sed -i 's/ --max-cudagraph-capture-size 256//' "$d/m_nocap/serve_cmd.txt"
MSG='serve_cmd.txt --max-cudagraph-capture-size' run m_nocap 1 "ns256 without the capture cap flag" ns256 0
mk m_cap128 ns128_0; sed -i 's/ --kv-cache-memory-bytes/ --max-cudagraph-capture-size 256 --kv-cache-memory-bytes/' "$d/m_cap128/serve_cmd.txt"
MSG='serve_cmd.txt --max-cudagraph-capture-size' run m_cap128 1 "ns128 with a capture cap flag (the profile has none)" ns128 0
mk m_spec ns256_0; sed -i 's/ --kv-cache-memory-bytes/ --speculative-config x --kv-cache-memory-bytes/' "$d/m_spec/serve_cmd.txt"
MSG='--speculative-config in a non-speculative' run m_spec 1 "a speculative flag on a non-speculative launch line" ns256 0
mk m_nospec spec_0; sed -i 's/ --speculative-config [^ ]*//' "$d/m_nospec/serve_cmd.txt"
MSG='no --speculative-config' run m_nospec 1 "the speculative profile without --speculative-config" spec 0
mk m_nocmd ns256_1; rm "$d/m_nocmd/serve_cmd.txt"
MSG='no serve_cmd.txt' run m_nocmd 1 "no serve_cmd.txt next to the log (fail-closed)" ns256 1
# the startup log
mk l_nda ns256_1; sed -i "s/'max_num_seqs': 256/'max_num_seqs': 128/" "$d/l_nda/serve.log"
MSG='max_num_seqs=256' run l_nda 1 "the parsed arguments carry another max_num_seqs" ns256 1
mk l_cap ns256_1; sed -i "s/'max_cudagraph_capture_size': 256/'max_cudagraph_capture_size': 512/" "$d/l_cap/serve.log"
MSG='max_cudagraph_capture_size=256' run l_cap 1 "the parsed arguments carry another capture cap" ns256 1
mk l_capon128 ns128_0; sed -i "s/'max_num_seqs': 128/'max_num_seqs': 128, 'max_cudagraph_capture_size': 256/" "$d/l_capon128/serve.log"
MSG='has no capture flag' run l_capon128 1 "ns128: a capture cap in the parsed arguments" ns128 0
mk l_kvb ns128_1; sed -i "s/'kv_cache_memory_bytes': 17257267200/'kv_cache_memory_bytes': 17357799424/" "$d/l_kvb/serve.log"
MSG='kv_cache_memory_bytes=17257267200' run l_kvb 1 "the parsed arguments carry the no-tier bytes of an ON boot" ns128 1
mk l_nonda ns128_0; sed -i '/non-default args/d' "$d/l_nonda/serve.log"
MSG='no "non-default args" line' run l_nonda 1 "no non-default args line" ns128 0
mk l_spec ns256_0; sed -i 's/speculative_config=None/speculative_config=SpeculativeConfig(method=dflash/' "$d/l_spec/serve.log"
MSG="speculative_config=None" run l_spec 1 "a speculative engine config in a non-speculative profile" ns256 0
mk l_nospec spec_1; sed -i 's/speculative_config=SpeculativeConfig(method=.dflash./speculative_config=None/' "$d/l_nospec/serve.log"
MSG="'speculative_config=None' in the log of the speculative" run l_nospec 1 "speculative_config=None in the speculative profile" spec 1
# the KV size
mk k_tok ns256_1; sed -i 's/2,893,281/2,893,282/' "$d/k_tok/serve.log"
MSG='GPU KV cache size: expected 2893281' run k_tok 1 "one token off in the GPU KV cache size line" ns256 1
mk k_none spec_0; sed -i '/GPU KV cache size/d' "$d/k_none/serve.log"
MSG="no 'GPU KV cache size" run k_none 1 "no KV size line" spec 0
mk k_two spec_0; echo "(EngineCore pid=1) INFO x [kv_cache_utils.py:1] GPU KV cache size: 1,000 tokens, Maximum concurrency" >> "$d/k_two/serve.log"
MSG='GPU KV cache size: expected' run k_two 1 "two different KV size lines" spec 0
mk k_blk ns128_0; sed -i 's/num_gpu_blocks="264859"/num_gpu_blocks="264860"/' "$d/k_blk/cache_config_info.txt"
MSG='num_gpu_blocks' run k_blk 1 "one block off in cache_config_info" ns128 0
mk k_ctok ns128_0; sed -i 's/kv_cache_size_tokens="2979623"/kv_cache_size_tokens="2979000"/' "$d/k_ctok/cache_config_info.txt"
MSG='kv_cache_size_tokens' run k_ctok 1 "tokens in cache_config_info differ" ns128 0
mk k_cbytes ns128_0; sed -i 's/kv_cache_memory_bytes="17357799424"/kv_cache_memory_bytes="17257267200"/' "$d/k_cbytes/cache_config_info.txt"
MSG='kv_cache_memory_bytes' run k_cbytes 1 "bytes in cache_config_info differ" ns128 0
mk k_nocci spec_1; rm "$d/k_nocci/cache_config_info.txt"
MSG='no cache_config_info.txt' run k_nocci 1 "no cache_config_info.txt (fail-closed)" spec 1
# the FULL decode graphs of the capture cap
mk g_hi ns256_1; sed -i 's/token counts 1\.\.256/token counts 1..128/' "$d/g_hi/serve.log"
MSG='largest FULL decode graph is 128' run g_hi 1 "ns256 with FULL decode graphs only up to 128" ns256 1
mk g_dis ns128_1; sed -i '0,/token counts 1\.\.128/s//token counts 1..64/' "$d/g_dis/serve.log"
MSG='disagree' run g_dis 1 "one worker logged another graph range" ns128 1
mk g_none ns128_0; sed -i '/FULL decode graphs are flagged/d' "$d/g_none/serve.log"
MSG='no flagged FULL decode graph line' run g_none 1 "no flagged graph line" ns128 0
mk g_sp ns128_0; sed -i 's/CUDA graphs up to 256 tokens/CUDA graphs up to 128 tokens/' "$d/g_sp/serve.log"
run g_sp 0 "the sequence-parallel line (256 for ns128 too) is not the capture-cap evidence" ns128 0
echo "$fails case(s) failed"
exit $(( fails > 0 ))
