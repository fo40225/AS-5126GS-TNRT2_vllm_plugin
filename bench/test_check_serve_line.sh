#!/bin/bash
# Regression test of scripts/check_serve_line.py (S1-11 review r2 m1), built from the real boot logs of the adoption / validation boots (bench/fixtures/launch/*.log: the
# "non-default args" lines are the real ones) and the real serve lines (bench/fixtures/launch_ref/*.serve_cmd.txt). For every state: the unmutated inputs PASS; then each
# mutation of ONLY the parsed "non-default args" line, of ONLY the live command line, or of ONLY serve_cmd.txt (against the live line and the clean-shell derivation) must FAIL.
# Usage  bash bench/test_check_serve_line.sh   Exit 0 = all cases as expected.
here=$(cd "$(dirname "$0")/.." && pwd)
exec python3 -I - "$here" <<'PY'
import os, re, shlex, subprocess, sys, tempfile

here = sys.argv[1]
fx = os.path.join(here, "bench", "fixtures")
chk = os.path.join(here, "scripts", "check_serve_line.py")
tmp = tempfile.mkdtemp()
fails = 0
n = 0

def run(cmd_args, serve_cmd, log, derived=None):
    c = os.path.join(tmp, "cmdline"); s = os.path.join(tmp, "serve_cmd.txt"); l = os.path.join(tmp, "serve.log")
    open(c, "wb").write(b"\0".join(a.encode() for a in cmd_args) + b"\0")
    open(s, "w").write(serve_cmd); open(l, "w").write(log)
    argv = [sys.executable, "-I", chk, "--serve-cmd", s, "--cmdline-file", c, "--log", l]
    if derived is not None:
        d = os.path.join(tmp, "derived.txt"); open(d, "w").write(derived); argv += ["--derived", d]
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()

def live(serve_cmd):
    # the shape of the real command line: <venv>/python <venv>/bin/vllm serve <args>
    w = shlex.split(serve_cmd)
    return ["/raid/vllm-runtime/sm120-dev/.venv/bin/python", "/raid/vllm-runtime/sm120-dev/.venv/bin/vllm"] + w[1:]

def check(desc, want_rc, got, must=None):
    global fails, n
    n += 1
    rc, out = got
    ok = rc == want_rc and (must is None or re.search(must, out))
    print(("ok   " if ok else "FAIL ") + desc)
    if not ok:
        fails += 1
        print("    rc=%s wanted %s: %s" % (rc, want_rc, out[:400]))

# mutations: (name, text to find, replacement) on the python repr (log) / the JSON or plain argument (command line)
spec_muts = [
    ("num_speculative_tokens 7 -> 6", "'num_speculative_tokens': 7", "'num_speculative_tokens': 6", '"num_speculative_tokens":7', '"num_speculative_tokens":6'),
    ("rejection block -> strict", "'rejection_sample_method': 'block'", "'rejection_sample_method': 'strict'", '"rejection_sample_method":"block"', '"rejection_sample_method":"strict"'),
    ("draft_sample probabilistic -> greedy", "'draft_sample_method': 'probabilistic'", "'draft_sample_method': 'greedy'", '"draft_sample_method":"probabilistic"', '"draft_sample_method":"greedy"'),
    ("method dflash -> eagle", "'method': 'dflash'", "'method': 'eagle'", '"method":"dflash"', '"method":"eagle"'),
    ("drafter snapshot", "snapshots/46c90f412549c472941a68382d91a99e5e6372a0", "snapshots/46c90f412549c472941a68382d91a99e5e6372a1", "snapshots/46c90f412549c472941a68382d91a99e5e6372a0", "snapshots/46c90f412549c472941a68382d91a99e5e6372a1"),
    ("max_num_batched_tokens 8199 -> 8192", "'max_num_batched_tokens': 8199", "'max_num_batched_tokens': 8192", None, None),
    ("max_num_scheduled_tokens 8192 -> 4096", "'max_num_scheduled_tokens': 8192", "'max_num_scheduled_tokens': 4096", None, None),
    ("capture cap 256 -> 512", "'max_cudagraph_capture_size': 256", "'max_cudagraph_capture_size': 512", None, None),
    ("per_request_spec_decode_metrics detailed -> basic", "'per_request_spec_decode_metrics': 'detailed'", "'per_request_spec_decode_metrics': 'basic'", None, None),
]
common_muts = [
    ("gpu_memory_utilization 0.95 -> 0.9", "'gpu_memory_utilization': 0.95", "'gpu_memory_utilization': 0.9", "0.95", "0.9"),
    ("max_model_len 131072 -> 65536", "'max_model_len': 131072", "'max_model_len': 65536", "131072", "65536"),
    ("kv_cache_dtype fp8 -> auto", "'kv_cache_dtype': 'fp8'", "'kv_cache_dtype': 'auto'", None, None),
    ("tensor_parallel_size 8 -> 4", "'tensor_parallel_size': 8", "'tensor_parallel_size': 4", None, None),
    ("quantization modelopt_fp4 -> fp8", "'quantization': 'modelopt_fp4'", "'quantization': 'fp8'", None, None),
    ("expert parallel off", "'enable_expert_parallel': True, ", "", None, None),
    ("max_num_seqs", "'max_num_seqs': 128", "'max_num_seqs': 64", None, None),
    ("kv_cache_memory_bytes 11645337600 -> 11645337601", "'kv_cache_memory_bytes': 11645337600", "'kv_cache_memory_bytes': 11645337601", None, None),
    ("port", "'port': 8100", "'port': 8101", None, None),
    ("limit_mm_per_prompt image 0 -> 1", "'limit_mm_per_prompt': {'image': 0", "'limit_mm_per_prompt': {'image': 1", None, None),
    ("numa_bind off", "'numa_bind': True, ", "", None, None),
    ("tier connector", "kv_connector='SimpleCPUOffloadConnector'", "kv_connector='LMCacheConnectorV1'", None, None),
    ("cpu_bytes_to_use_per_rank", "'cpu_bytes_to_use_per_rank': 51814072320", "'cpu_bytes_to_use_per_rank': 51814072321", None, None),
    ("kv_role", "kv_role='kv_both'", "kv_role='kv_producer'", None, None),
    ("a parser option the command line does not have", "'tool_call_parser': 'muse'", "'tool_call_parser': 'muse', 'enable_prefix_caching': False", None, None),
    ("model", "'model': 'unsloth", "'model': 'other", None, None),
]

# the kv-transfer JSON as serve_cmd.txt holds it (shell-quoted): the two fields that a deletion must not hide from the log comparison
KVC = r'\"kv_connector\":\"SimpleCPUOffloadConnector\"\,'
KVR = r'\"kv_role\":\"kv_both\"\,'

def log_line(log):
    return [l for l in log.splitlines() if "non-default args" in l][0]

for state in ("spec_1", "spec_0", "ns256_1", "ns256_0", "ns128_1", "ns128_0"):
    log = open(os.path.join(fx, "launch", state + ".log")).read()
    sc = open(os.path.join(fx, "launch_ref", state + ".serve_cmd.txt")).read()
    check("%s: real log + real serve line + live command line + derivation: PASS" % state, 0, run(live(sc), sc, log, sc), r"^OK")
    check("%s: no derivation given (--keep-env shape): PASS" % state, 0, run(live(sc), sc, log), r"^OK")
    check("%s: the live process is `python script vllm serve`, an extra interpreter option in front does not matter" % state, 0,
          run(["/usr/bin/python3", "-I"] + live(sc)[:2] + live(sc)[2:], sc, log, sc), r"^OK")
    ll = log_line(log)
    for name, a, b, ca, cb in spec_muts + common_muts:
        if a not in ll:
            continue
        # 1. only the parsed "non-default args" line is mutated
        check("%s: log only: %s" % (state, name), 1, run(live(sc), sc, log.replace(a, b, 1), sc), r"non-default args|carries options|not in the parsed")
        # 2. only the live command line is mutated (the log and serve_cmd.txt stay); a plain argument is matched as a whole argument
        w = shlex.split(sc)
        m = re.search(r"'([a-z_]+)': ('?)([^',}]+)'?", a)
        if ca is not None:
            argv = [x.replace(ca, cb) for x in live(sc)]
        else:
            val = re.sub(r".*: '?([^',}]+)'?.*", r"\1", a); new = re.sub(r".*: '?([^',}]+)'?.*", r"\1", b)
            argv = [new if x == val else x for x in live(sc)]
        if argv != live(sc):
            check("%s: live command line only: %s" % (state, name), 1, run(argv, sc, log, sc), r"live command line differs")

    # review p2 r2 F3: kv_connector / kv_role deleted from the kv-transfer JSON in ALL copies of the line (live, serve_cmd.txt, derivation) while the log keeps them
    if "SimpleCPUOffloadConnector" in sc:
        for field, frag in (("kv_connector", KVC), ("kv_role", KVR)):
            assert frag in sc, frag
            ed = sc.replace(frag, "")
            check("%s: %s deleted from the line (all copies), log unchanged: FAIL" % (state, field), 1, run(live(ed), ed, log, ed), r"kv_transfer_config differs: %s: line .*no %s" % (field, field))
        # review docs r1 m4: kv_connector_extra_config (the tier size) deleted from the kv-transfer JSON in all copies while the log keeps it
        KVE = r'\,\"kv_connector_extra_config\":'
        k = sc.index(KVE)
        ed = sc[:k] + r'\}' + sc[sc.index(r'\}\}', k) + 4:]
        assert ed != sc and "extra_config" not in ed, ed
        check("%s: kv_connector_extra_config deleted from the line (all copies), log unchanged: FAIL" % state, 1, run(live(ed), ed, log, ed), r"kv_transfer_config differs: kv_connector_extra_config: line .*no kv_connector_extra_config")
        # the same edit on the line AND the log (a boot that really ran without them) is consistent for this leg
        ed = sc.replace(KVC, "").replace(KVR, "")
        elog = log.replace("kv_connector='SimpleCPUOffloadConnector'", "kv_connector=None").replace("kv_role='kv_both'", "kv_role=None")
        check("%s: both fields deleted from the line and None in the log: this leg passes" % state, 0, run(live(ed), ed, elog, ed), r"^OK")

    # the line the process runs vs the file: one argument dropped / added / reordered
    L = live(sc)
    check("%s: live command line with an extra argument" % state, 1, run(L + ["--enable-prefix-caching"], sc, log, sc), r"live command line differs")
    check("%s: live command line with the last argument dropped" % state, 1, run(L[:-1], sc, log, sc), r"live command line differs")
    check("%s: no `vllm serve` in the live command line" % state, 1, run(["/usr/bin/python3", "x.py", "--port", "1"], sc, log, sc), r"no `vllm serve`")
    # serve_cmd.txt differs from the clean-shell derivation (stale env files, an edited line): live line and log carry the same value, only the derivation leg can see it
    mut = sc.replace("--max-num-seqs 128", "--max-num-seqs 64").replace("--max-num-seqs 256", "--max-num-seqs 64")
    mlog = log.replace("'max_num_seqs': 128", "'max_num_seqs': 64").replace("'max_num_seqs': 256", "'max_num_seqs': 64")
    check("%s: serve_cmd.txt = live = log but not the clean-shell derivation: FAIL" % state, 1, run(live(mut), mut, mlog, sc), r"re-derived from the env files")
    check("%s: no non-default args line in the log: FAIL" % state, 1, run(L, sc, "\n".join(x for x in log.splitlines() if "non-default args" not in x), sc), r"no .non-default args. line")

print("%d case(s), %d failed" % (n, fails))
sys.exit(1 if fails else 0)
PY
