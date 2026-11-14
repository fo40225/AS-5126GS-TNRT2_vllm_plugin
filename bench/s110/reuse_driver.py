#!/usr/bin/env python3
"""S1-10 prefix-reuse driver for the KV CPU-offload A/B (stdlib only; run with python3 -I).

Random token-id prompts (no tokenizer, distinct per document / conversation, so nothing is shared by accident) against a
RUNNING server (/v1/completions with a token-id prompt, greedy, max_tokens 16, streamed). Scenarios:

  cycle   Long documents cycle through a working set that is larger than the GPU prefix cache:
          1. two long prompts (A = 64k, B = 118k tokens) are sent cold and immediately again (GPU hit),
          2. pass 1 over D documents of --doc-len tokens (cold; D * doc-len = --ws-factor x the GPU pool, so it evicts A and B),
          3. A and B again (re-admission: CPU-tier hit with offload, recompute without),
          4. passes 2..N over the same documents in the same order with a fresh random question suffix (LRU of the GPU pool
             hits nothing; an offload tier that holds the working set hits everything).
          --clients requests are in flight (closed loop). Per request: TTFT, total time, prompt length, metric deltas per pass.
  chat    --convs conversations of --turns turns, round-robin over the conversations (the reuse distance of a history is
          the whole working set): turn k sends history(k-1) + --turn-tokens new tokens + --reply-tokens of the assistant
          reply (random ids), so every turn reuses the previous prompt as its prefix.

Hit proof: the /metrics counters of the prefix caches (vllm:prefix_cache_*, vllm:external_prefix_cache_* = connector hits) and of
both connectors (vllm:simple_kv_offload_*, vllm:kv_offload_*) are diffed per pass. The TTFT alone proves nothing.
Stores are asynchronous: wait_stores() waits until the connector's pending-store gauges are 0 and its counters stand still.

usage: reuse_driver.py --port 8100 --model M --out FILE.json --pool-tokens N [--scenario cycle|chat|smoke, comma separated]
       [--doc-len 32768] [--ws-factor 1.5] [--passes 3] [--clients 4] [--ab-lens 65536,117964]
       [--convs 64] [--turns 5] [--turn-tokens 6000] [--reply-tokens 128] [--seed-base 777000] [--max-docs N]
"""
import argparse
import hashlib
import http.client
import json
import math
import random
import re
import statistics
import sys
import threading
import time

KEEP = re.compile(r"^vllm:(prefix_cache|external_prefix_cache|simple_kv_offload|kv_offload|num_preemptions|kv_cache_usage|spec_decode_num|"
                  r"request_prefill_kv_computed|prompt_tokens_by_source|prompt_tokens_cached|prompt_tokens_total)")
PENDING = re.compile(r"(pending_store|write_usage)")


def metrics(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("GET", "/metrics")
    body = c.getresponse().read().decode()
    c.close()
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or not KEEP.match(line):
            continue
        name, _, val = line.rpartition(" ")
        base = name.split("{")[0]
        # drop the engine / model_name labels (long, constant); keep the others (source, outcome, ...)
        labels = re.findall(r'(\w+)="([^"]*)"', name[len(base):])
        keep = ",".join("%s=%s" % (k, v) for k, v in labels if k not in ("engine", "model_name"))
        key = base + ("{" + keep + "}" if keep else "")
        if base.endswith("_created") or base.endswith("_bucket"):
            continue
        if base.endswith("_info"):
            continue
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def delta(a, b):
    d = {}
    for k in set(a) | set(b):
        if "kv_cache_usage" in k or PENDING.search(k):
            continue
        x = b.get(k, 0.0) - a.get(k, 0.0)
        if abs(x) > 0:
            d[k] = round(x, 3)
    return dict(sorted(d.items()))


def gen_tokens(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def send(port, model, prompt, max_tokens=16, timeout=7200):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "logprobs": 1})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    try:
        c.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        status = r.status
        ttft = None
        text = []
        lps = []
        toks = []
        vals = []
        while True:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            try:
                ch = json.loads(payload)["choices"][0]
                text.append(ch.get("text", ""))
                lp = ch.get("logprobs") or {}
                # the logprob sha covers the token strings and the exact logprob values (repr round-trips)
                lps.append("%r|%r" % (lp.get("tokens"), lp.get("token_logprobs")))
                toks.extend(lp.get("tokens") or [])
                vals.extend(lp.get("token_logprobs") or [])
            except (ValueError, KeyError, IndexError):
                pass
        total = time.perf_counter() - t0
    except Exception as e:  # a dead server must show up as a failed request, not as a crash of the driver
        return {"status": -1, "ttft_s": None, "total_s": time.perf_counter() - t0, "text": "", "lps": "", "toks": [], "vals": [], "error": repr(e)}
    finally:
        c.close()
    return {"status": status, "ttft_s": ttft, "total_s": total, "text": "".join(text), "lps": "".join(lps), "toks": toks, "vals": vals}


def post(port, path, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path)
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    return r.status, body


def wait_stores(port, quiet_s=6.0, max_s=900.0):
    """Wait until the connector's pending-store gauges are 0 and every offload counter stands still for quiet_s."""
    t0 = time.time()
    last = None
    stable_since = time.time()
    while time.time() - t0 < max_s:
        m = metrics(port)
        pend = sum(v for k, v in m.items() if PENDING.search(k))
        sig = tuple(sorted((k, v) for k, v in m.items() if ("offload" in k or "external_prefix" in k) and not PENDING.search(k)))
        if not sig:  # no connector metrics at all: nothing to wait for
            time.sleep(1.0)
            return 0.0
        if sig != last or pend > 0:
            last = sig
            stable_since = time.time()
        elif time.time() - stable_since >= quiet_s:
            return round(time.time() - t0, 1)
        time.sleep(1.0)
    return round(time.time() - t0, 1)


def reset_gpu_cache(port, tries=30):
    """POST /reset_prefix_cache (GPU pool only; needs VLLM_SERVER_DEV_MODE=1); retried while requests are running."""
    for i in range(tries):
        status, body = post(port, "/reset_prefix_cache")
        if status == 200:
            return i + 1
        time.sleep(2.0)
    return -1


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1))
    return round(s[k], 3)


def summarize(recs):
    ok = [r for r in recs if r["status"] == 200 and r["ttft_s"] is not None]
    tt = [r["ttft_s"] for r in ok]
    return {"n": len(recs), "ok": len(ok), "ttft_p50": pct(tt, 50), "ttft_p90": pct(tt, 90), "ttft_p99": pct(tt, 99),
            "ttft_mean": round(statistics.mean(tt), 3) if tt else None, "ttft_max": round(max(tt), 3) if tt else None}


class Run:
    def __init__(self, a):
        self.a = a
        self.recs = []
        self.lock = threading.Lock()
        self.info = {"args": vars(a), "passes": {}}

    def one(self, prompt, label, pas, idx):
        r = send(self.a.port, self.a.model, prompt)
        rec = {"label": label, "pass": pas, "idx": idx, "len": len(prompt), "ttft_s": None if r["ttft_s"] is None else round(r["ttft_s"], 3),
               "total_s": round(r["total_s"], 3), "status": r["status"], "t_end": round(time.time(), 2),
               "text_sha": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
               "lp_sha": hashlib.sha1(r["lps"].encode()).hexdigest()[:12], "text_head": r["text"][:48],
               "toks": r["toks"], "vals": r["vals"]}
        if r.get("error"):
            rec["error"] = r["error"]
        with self.lock:
            self.recs.append(rec)
        return rec

    def timed_single(self, prompt, label, pas="single"):
        m0 = metrics(self.a.port)
        rec = self.one(prompt, label, pas, 0)
        time.sleep(1.0)
        rec["metric_delta"] = delta(m0, metrics(self.a.port))
        print("REUSE_SEND %-24s len=%-7d ttft=%s total=%s status=%s delta=%s" % (
            label, len(prompt), rec["ttft_s"], rec["total_s"], rec["status"], json.dumps(rec["metric_delta"], sort_keys=True)), flush=True)
        return rec

    def run_pass(self, name, prompts, labels, clients):
        """Closed loop: `clients` requests in flight; prompts is a list of token lists; returns summary."""
        m0 = metrics(self.a.port)
        t0 = time.time()
        nxt = [0]
        lock = threading.Lock()
        recs = []

        def worker():
            while True:
                with lock:
                    i = nxt[0]
                    nxt[0] += 1
                if i >= len(prompts):
                    return
                rec = self.one(prompts[i], labels[i], name, i)
                recs.append(rec)

        ths = [threading.Thread(target=worker) for _ in range(clients)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.time() - t0
        m1 = metrics(self.a.port)
        toks = sum(len(p) for p in prompts)
        s = summarize(recs)
        s.update({"wall_s": round(wall, 1), "prompt_tokens": toks, "tok_per_s": round(toks / wall, 1), "metric_delta": delta(m0, m1)})
        self.info["passes"][name] = s
        print("REUSE_PASS %s %s" % (name, json.dumps(s, sort_keys=True)), flush=True)
        return s

    # ------------------------------------------------------------------ cycle
    def cycle(self):
        a = self.a
        lens = [int(x) for x in a.ab_lens.split(",") if x]
        base = a.seed_base
        # warm-up: a short prompt so first-use costs stay out of the rows
        send(a.port, a.model, gen_tokens(base - 1, 2048), max_tokens=4)
        longs = {L: gen_tokens(base + 100 + L, L) for L in lens}
        for L in lens:
            self.timed_single(longs[L], "ab_cold_%d" % L)
            self.timed_single(longs[L] + gen_tokens(base + 900 + L, 64), "ab_gpuhit_%d" % L)
        w = wait_stores(a.port)
        print("REUSE_WAIT_STORES after_ab %.1fs" % w, flush=True)
        ndocs = max(2, int(math.ceil(a.ws_factor * a.pool_tokens / a.doc_len)))
        if a.max_docs:
            ndocs = min(ndocs, a.max_docs)
        docs = [gen_tokens(base + 5000 + i, a.doc_len) for i in range(ndocs)]
        self.info["cycle"] = {"docs": ndocs, "doc_len": a.doc_len, "working_set_tokens": ndocs * a.doc_len, "pool_tokens": a.pool_tokens}
        print("REUSE_CYCLE docs=%d doc_len=%d working_set=%d pool=%d" % (ndocs, a.doc_len, ndocs * a.doc_len, a.pool_tokens), flush=True)
        for p in range(1, a.passes + 1):
            suff = [gen_tokens(base + 70000 + i, 64) for i in range(ndocs)]  # same suffix in every pass: the sha is comparable
            prompts = [docs[i] + suff[i] for i in range(ndocs)]
            self.run_pass("cycle_p%d" % p, prompts, ["doc%d_p%d" % (i, p) for i in range(ndocs)], a.clients)
            w = wait_stores(a.port)
            print("REUSE_WAIT_STORES after_p%d %.1fs" % (p, w), flush=True)
            if p == 1:
                for L in lens:
                    self.timed_single(longs[L] + gen_tokens(base + 900 + L, 64), "ab_readmit_%d" % L)

    # ------------------------------------------------------------------ smoke
    def smoke(self):
        """Per length of --ab-lens: one prompt cold, GPU hit x2, GPU prefix cache reset (the connector tier is kept), re-send x2.
        With a connector the re-sends are connector hits; without one they are recomputes. Equal outputs of the second GPU hit and of
        the connector hits (same prompt, same hit length) are the determinism check of the hit source; every request runs alone."""
        a = self.a
        base = a.seed_base + 9_000_000
        send(a.port, a.model, gen_tokens(base - 1, 2048), max_tokens=4)
        for L in [int(x) for x in a.ab_lens.split(",") if x]:
            p = gen_tokens(base + L, L)
            sfx = gen_tokens(base + 1 + L, 64)
            tag = "smoke_%d_" % L
            self.timed_single(p, tag + "cold")
            self.timed_single(p + sfx, tag + "gpuhit1")
            self.timed_single(p + sfx, tag + "gpuhit2")
            w = wait_stores(a.port)
            n = reset_gpu_cache(a.port)
            print("REUSE_RESET L=%d wait_stores_s=%s reset_tries=%s" % (L, w, n), flush=True)
            self.timed_single(p + sfx, tag + "after_reset1")
            wait_stores(a.port)
            reset_gpu_cache(a.port)
            self.timed_single(p + sfx, tag + "after_reset2")

    # ------------------------------------------------------------------ chat
    def chat(self):
        a = self.a
        base = a.seed_base + 2_000_000
        hist = [[] for _ in range(a.convs)]
        for turn in range(1, a.turns + 1):
            prompts = []
            for c in range(a.convs):
                hist[c] = hist[c] + gen_tokens(base + c * 1000 + turn, a.turn_tokens)
                prompts.append(list(hist[c]))
            self.run_pass("chat_t%d" % turn, prompts, ["conv%d_t%d" % (c, turn) for c in range(a.convs)], a.clients)
            wait_stores(a.port)
            for c in range(a.convs):  # the assistant reply joins the history as random ids
                hist[c] = hist[c] + gen_tokens(base + 500_000 + c * 1000 + turn, a.reply_tokens)
        self.info["chat"] = {"convs": a.convs, "turns": a.turns, "turn_tokens": a.turn_tokens, "final_len": len(hist[0]),
                             "working_set_tokens": a.convs * len(hist[0]), "pool_tokens": a.pool_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-tokens", type=int, required=True, help="GPU KV pool size in tokens (serve log 'GPU KV cache size')")
    ap.add_argument("--scenario", default="cycle")
    ap.add_argument("--doc-len", type=int, default=32768)
    ap.add_argument("--ws-factor", type=float, default=1.5)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--ab-lens", default="65536,117964")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--convs", type=int, default=64)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--turn-tokens", type=int, default=6000)
    ap.add_argument("--reply-tokens", type=int, default=128)
    ap.add_argument("--seed-base", type=int, default=777000)
    a = ap.parse_args()
    run = Run(a)
    for sc in a.scenario.split(","):
        if sc == "cycle":
            run.cycle()
        elif sc == "chat":
            run.chat()
        elif sc == "smoke":
            run.smoke()
        else:
            print("unknown scenario %s" % sc, file=sys.stderr)
            return 2
    run.info["sends"] = run.recs
    json.dump(run.info, open(a.out, "w"), indent=1)
    print("REUSE_TABLE name n ok ttft_p50 ttft_p99 wall_s tok_per_s")
    for k, s in run.info["passes"].items():
        print("REUSE_ROW %s %d %d %s %s %s %s" % (k, s["n"], s["ok"], s["ttft_p50"], s["ttft_p99"], s["wall_s"], s["tok_per_s"]))
    bad = [r for r in run.recs if r["status"] != 200]
    print("REUSE_DONE requests=%d failed=%d" % (len(run.recs), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
