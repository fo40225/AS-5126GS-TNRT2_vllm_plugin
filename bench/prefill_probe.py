"""Prefill latency probe: single requests of several lengths (1 output token, random token ids), against a running server.

usage: prefill_probe.py <port> <name> [reps]       env SIZES=4300,8192,16384,65536 (prompt lengths), REPO (served model)
Prints one line "TUNE <name> <n>:<mean s>(min <s>, <tokens/s>) ..." (the mean of `reps` requests per size, 2 for >= 64k),
after a warm-up of every size. scripts/prefill_chain.sh runs it for a list of server configurations."""
import json, os, random, sys, time, urllib.request, urllib.error

port, name = int(sys.argv[1]), sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MODEL = os.environ.get("REPO", "unsloth/Muse-Spark-Open-816B-A42B-NVFP4")
base = "http://127.0.0.1:%d" % port


def complete(n_in, seed):
    rnd = random.Random(seed)
    prompt = [rnd.randint(1000, 100000) for _ in range(n_in)]
    data = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 1, "min_tokens": 1, "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        r.read()
    return time.time() - t0


sizes = [int(x) for x in os.environ.get("SIZES", "4300,8192,16384,65536").split(",")]
for n in sorted({min(n, 16384) for n in sizes}) + [max(sizes)]:
    complete(n, 5)  # warm-up
    complete(n, 6)
res = {}
for n in sizes:
    k = max(2, reps - 1) if n >= 65536 else reps
    ts = [complete(n, 1000 + 7 * n + i) for i in range(k)]
    res[n] = (sum(ts) / len(ts), min(ts))
print("TUNE %-28s " % name + " ".join("%d:%.3fs(min %.3f, %.0f t/s)" % (n, m, lo, n / m) for n, (m, lo) in res.items()), flush=True)
