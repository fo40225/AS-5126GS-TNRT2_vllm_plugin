"""S1-10 GPU check tools on the 0.33.0 production trees (in-process engine, VLLM_ENABLE_V1_MULTIPROCESSING=0, TP8 + EP8).

Modes (env CHK): `spec` / `ns256` = the S1-10R Phase 2 investigations (held blocks after a reset, schedule trace, single-stream replays; kept as they ran);
`d4` = the gate check of ADOPTION_RULE_FINAL_AMENDMENT3 (D4'), one process per profile (E2E_SPEC=1 spec, 0 ns256), output CHK_OUT/d4_verdict_<profile>.json:
  (a) KV byte equality: the 85 det prompts in greedy, passes C / H / X (batched); every block restored from the tier in X is compared, on every rank, in ALL
      regions (= all KV groups, the draft's included), with its tier copy and with its pass-H GPU copy (compare_loads in chk_worker.py);
  NC  mandatory negative control in the same process: one byte of one tier block is corrupted before a restore, the compare must report exactly that block
      (and a corrupted GPU copy must be reported by the X-vs-tier leg); otherwise the verdict is ARTIFACT;
  (b) single-stream equality: all 85 det requests in greedy and in seeded t1k64 (seed 1000 + index), each ALONE: cold, H (GPU hit), GPU-only reset after the
      stores drained, X (tier hit); token ids and (where requested) logprobs equal H vs X, for spec also the accepted / drafted counts.
Verdict PASS | FAIL | ARTIFACT (any error or missing piece is ARTIFACT, never PASS). Every reset uses robust_reset (engine steps, no request) because a
request leaves a finished-request store pending in the connector and Scheduler.reset_prefix_cache checks the GPU pool before the connector reset releases it.
Driven by the gate tool e2e_s110.py (85 det prompts, dump helpers).
env: CHK=spec|ns256|d4  CHK_OUT=<dir>  E2E_KV_BYTES  E2E_PER_RANK_BYTES  E2E_SPEC  E2E_MNS  E2E_MNBT  E2E_MNST  E2E_CAP  E2E_GEN_TOKENS  E2E_MAXLEN
The pure functions (aggregate_compare, eval_nc, final_verdict) need neither torch nor vllm (bench/test_chk_d4_verdict.py).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chk_worker as W  # noqa: E402
import e2e_s110 as E  # noqa: E402  (byte-identical copy of the S1-10 gate tool)

CHK = os.environ.get("CHK", "")
OUT = os.environ.get("CHK_OUT", "")
GEN = E.GEN_TOKENS
RES = {"chk": CHK, "env": {k: v for k, v in os.environ.items() if k.startswith(("E2E_", "MUSE_", "VLLM_MUSE", "NCCL_", "CHK"))}}


def log(*a):
    print("CHK", time.strftime("%H:%M:%S"), *a, flush=True)


def save():
    p = f"{OUT}/result_{CHK}.json"
    with open(p + ".tmp", "w") as f:
        json.dump(RES, f, indent=1, default=str)
    os.replace(p + ".tmp", p)


def safe(name, fn):
    """Run one probe; a failure is recorded and does not lose the other results."""
    import traceback

    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        RES.setdefault("errors", {})[name] = traceback.format_exc()[-3000:]
        log("PROBE_ERROR", name, repr(e))
        save()
        return None


# ---------------- D4' pure functions (no torch / vllm; tested by bench/test_chk_d4_verdict.py) ----------------
N_DET = 85
B_KINDS = ("greedy", "t1k64")


def merge_rank_results(chunks):
    """chunks = [(offset, [per-rank compare_digests dict])]: one dict per rank with GLOBAL pair indices."""
    merged = {}
    for off, res in chunks:
        for r in res:
            m = merged.setdefault(r["rank"], {"rank": r["rank"], "n": 0, "n_regions": r["n_regions"], "xc_bad": [], "hx_bad": [], "n_no_h": 0,
                                              "n_missing_h": 0, "examples": []})
            if m["n_regions"] != r["n_regions"]:
                m["n_regions"] = -1
            m["n"] += r["n"]
            m["xc_bad"] += [off + k for k in r["xc_bad"]]
            m["hx_bad"] += [off + k for k in r["hx_bad"]]
            m["n_no_h"] += r["n_no_h"]
            m["n_missing_h"] += r["n_missing_h"]
            m["examples"] += [dict(e, k=off + e["k"]) for e in r["examples"]][:8]
    return [merged[k] for k in sorted(merged)]


def aggregate_compare(rank_results, pairs, expected_ranks, n_void=0, void_examples=()):
    """rank_results: per rank (merge_rank_results); pairs: [(key_hex, group, req, gpu_id, cpu_id, n_h_candidates)] of the restored blocks."""
    def ent(rank, k):
        return [rank, pairs[k][0], pairs[k][1], pairs[k][2]]

    groups_loaded, groups_with_h = {}, {}
    for key, g, req, gb, cb, nh in pairs:
        groups_loaded[g] = groups_loaded.get(g, 0) + 1
        if nh:
            groups_with_h[g] = groups_with_h.get(g, 0) + 1
    r0 = rank_results[0] if rank_results else {"n_no_h": 0}
    return {"ranks": [r["rank"] for r in rank_results], "expected_ranks": expected_ranks, "n_regions": sorted({r["n_regions"] for r in rank_results}),
            "n_loaded": len(pairs), "n_void": n_void, "void_examples": list(void_examples)[:6], "n_with_h": len(pairs) - r0["n_no_h"],
            "n_missing_h": sum(r["n_missing_h"] for r in rank_results), "n_block_rank_xc": len(pairs) * len(rank_results),
            "n_block_rank_hx": sum(r["n"] - r["n_no_h"] for r in rank_results),
            "groups_loaded": groups_loaded, "groups_with_h": groups_with_h,
            "xc_bad": [ent(r["rank"], k) for r in rank_results for k in r["xc_bad"]],
            "hx_bad": [ent(r["rank"], k) for r in rank_results for k in r["hx_bad"]],
            "examples": [e for r in rank_results for e in r["examples"]][:8]}


def eval_kv(kv, n_groups, uncovered_requests):
    """-> (artifact reasons, number of differing block-rank pairs {xc, hx}); a compare that could not cover what it claims is an artifact."""
    if not kv:
        return ["(a) no KV compare result"], None
    art = []
    if kv["ranks"] != list(range(kv["expected_ranks"])):
        art.append(f"(a) ranks reported {kv['ranks']} != 0..{kv['expected_ranks'] - 1}")
    if len(kv["n_regions"]) != 1 or kv["n_regions"][0] <= 0:
        art.append(f"(a) region counts differ between ranks: {kv['n_regions']}")
    if kv["n_loaded"] <= 0:
        art.append("(a) no block was restored from the tier in pass X")
    if kv["n_void"]:
        art.append(f"(a) {kv['n_void']} restored blocks without a matching hash on the GPU and tier side (unverifiable): {kv['void_examples']}")
    if kv["n_missing_h"]:
        art.append(f"(a) {kv['n_missing_h']} restored blocks whose pass-H digest is missing")
    if kv["n_with_h"] <= 0:
        art.append("(a) no restored block has a pass-H GPU copy")
    unc = [g for g in range(n_groups) if not kv["groups_loaded"].get(g)]
    unh = [g for g in range(n_groups) if kv["groups_loaded"].get(g) and not kv["groups_with_h"].get(g)]
    if unc:
        art.append(f"(a) KV groups with no restored block in pass X: {unc} of {n_groups}")
    if unh:
        art.append(f"(a) KV groups with no restored block that has a pass-H copy: {unh}")
    if uncovered_requests:
        art.append(f"(a) requests with cached tokens in pass X but no captured tier load: {list(uncovered_requests)[:10]}")
    return art, {"xc": len(kv["xc_bad"]), "hx": len(kv["hx_bad"])}


def _set(entries):
    return {(e[0], e[1]) for e in entries}


def eval_nc(nc):
    """The negative control must report exactly the corrupted block: -> (ok, reasons)."""
    if not nc:
        return False, ["no negative control result"]
    why = []
    base, c1, c2 = nc.get("baseline"), nc.get("after_tier_corrupt"), nc.get("after_gpu_corrupt")
    if not (base and c1 and c2):
        return False, ["negative control incomplete: " + ",".join(k for k in ("baseline", "after_tier_corrupt", "after_gpu_corrupt") if not nc.get(k))]
    t, g = nc["tier_target"], nc["gpu_target"]
    if base["n_loaded"] <= 0 or base["xc_bad"] or base["hx_bad"]:
        why.append("baseline restore of the control request is not clean (or empty)")
    if not nc.get("tier_digest_changed"):
        why.append("the tier corruption did not change the tier digest")
    if not nc.get("tier_undo_ok"):
        why.append("the tier corruption was not undone")
    if not nc.get("target_in_loads"):
        why.append("the corrupted tier block was not restored in the control pass")
    want_t = {(t["rank"], t["key"])}
    if _set(c1["hx_bad"]) != want_t:
        why.append(f"after the tier corruption the H-vs-X leg reported {sorted(_set(c1['hx_bad']))}, expected exactly {sorted(want_t)}")
    if c1["xc_bad"]:
        why.append(f"after the tier corruption the X-vs-tier leg reported {sorted(_set(c1['xc_bad']))} (the restored GPU copy equals the corrupted tier copy, expected none)")
    want_g = {(g["rank"], g["key"])}
    if _set(c2["xc_bad"]) != want_g:
        why.append(f"after the GPU corruption the X-vs-tier leg reported {sorted(_set(c2['xc_bad']))}, expected exactly {sorted(want_g)}")
    if _set(c2["hx_bad"]) != want_t | want_g:
        why.append(f"after the GPU corruption the H-vs-X leg reported {sorted(_set(c2['hx_bad']))}, expected exactly {sorted(want_t | want_g)}")
    return not why, why


def final_verdict(D):
    """D: {profile, spec, n_groups, errors, tracer_errors, a, a_uncovered_requests, nc, b}; any error or missing piece -> ARTIFACT, a real difference -> FAIL."""
    art, fail, warn = [], [], []
    for name, tb in (D.get("errors") or {}).items():
        art.append(f"probe error {name}")
    if D.get("tracer_errors"):
        art.append(f"tracer errors {D['tracer_errors'][:3]}")
    a_art, bad = eval_kv(D.get("a"), D.get("n_groups", 0), D.get("a_uncovered_requests") or [])
    nc_ok, nc_why = eval_nc(D.get("nc"))
    art += a_art
    art += ["negative control: " + w for w in nc_why]
    if bad and (bad["xc"] or bad["hx"]):
        if nc_ok:
            fail.append(f"(a) restored KV bytes differ from the tier copy ({bad['xc']} block-ranks) or the pass-H GPU copy ({bad['hx']} block-ranks): connector bug")
        else:
            art.append("(a) KV bytes differ but the negative control is not valid")
    recs = D.get("b") or []
    good = [r for r in recs if not r.get("error")]
    seen = {}
    for r in good:
        seen[(r["req"], r["kind"])] = r
    want = {(i, k) for i in range(D.get("n_det", N_DET)) for k in B_KINDS}
    if want - set(seen):
        art.append(f"(b) {len(want - set(seen))} of {len(want)} single-stream records missing or failed, e.g. {sorted(want - set(seen))[:5]}")
    if len(good) != len(seen):
        art.append("(b) duplicate single-stream records")
    for (i, k), r in sorted(seen.items()):
        if not r["tokens_equal"]:
            fail.append(f"(b) request {i} {k}: token ids differ H vs X (first diff at {r['first_diff']})")
        if r.get("lp_requested") and not r["lp_equal"]:
            fail.append(f"(b) request {i} {k}: logprobs differ H vs X (max abs {r['lp_max_abs']})")
        if D.get("spec") and not r["spec_equal"]:
            fail.append(f"(b) request {i} {k}: accepted/drafted counts differ H vs X ({r['spec']['H']} vs {r['spec']['X']})")
        if r["cached"]["H"] != r["cached"]["X"]:
            warn.append(f"(b) request {i} {k}: cached tokens H {r['cached']['H']} != X {r['cached']['X']}")
    if seen and not any(r["cached"]["X"] > 0 for r in seen.values()):
        art.append("(b) no single-stream pass X was a tier hit")
    if D.get("spec") and seen:
        for ph in ("H", "X"):
            if not sum((r["spec"][ph] or {}).get("drafted", 0) for r in seen.values()):
                art.append(f"(b) spec profile but no drafted tokens counted in pass {ph}")
    verdict = "FAIL" if fail else ("ARTIFACT" if art else "PASS")
    return {"verdict": verdict, "reasons_fail": fail, "reasons_artifact": art, "warnings": warn[:40], "n_warnings": len(warn), "nc_valid": nc_ok}



def main():
    os.makedirs(OUT, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.v1.core.kv_cache_utils import get_group_id
    from vllm.v1.request import RequestStatus

    t0 = time.time()
    kw = {}
    kw["kv_cache_memory_bytes"] = int(os.environ["E2E_KV_BYTES"])
    if os.environ.get("E2E_SPEC", "1") == "1":
        drafter = os.environ.get("E2E_DRAFTER", "/raid/.cache/huggingface/hub/models--RadixArk--Muse-Spark-Open-DFlash2/snapshots/46c90f412549c472941a68382d91a99e5e6372a0")
        kw["speculative_config"] = {"method": "dflash", "model": drafter, "num_speculative_tokens": 7, "rejection_sample_method": "block", "draft_sample_method": "probabilistic"}
    if int(os.environ.get("E2E_MNST", "0")) > 0:
        kw["max_num_scheduled_tokens"] = int(os.environ["E2E_MNST"])
    per_rank = int(os.environ["E2E_PER_RANK_BYTES"])
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
                                                kv_connector_extra_config={"cpu_bytes_to_use": per_rank * 8, "cpu_bytes_to_use_per_rank": per_rank})
    kw["numa_bind"] = True
    llm = LLM(model=E.MODEL, quantization="modelopt_fp4", kv_cache_dtype="fp8", max_model_len=int(os.environ.get("E2E_MAXLEN", "32768")), tensor_parallel_size=8,
              enable_expert_parallel=True, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
              gpu_memory_utilization=float(os.environ.get("E2E_GPU_UTIL", "0.95")), max_num_seqs=int(os.environ.get("E2E_MNS", "128")), seed=0,
              max_num_batched_tokens=int(os.environ.get("E2E_MNBT", "8199")), enable_prefix_caching=True,
              compilation_config={"max_cudagraph_capture_size": int(os.environ.get("E2E_CAP", "256"))}, **kw)
    log(f"loaded in {time.time() - t0:.1f} s profile={CHK}")
    RES["load_s"] = round(time.time() - t0, 1)

    core = llm.llm_engine.engine_core.engine_core
    sched = core.scheduler
    kvm = sched.kv_cache_manager
    pool = kvm.block_pool
    conn = sched.connector
    while conn is not None and not hasattr(conn, "scheduler_manager"):
        conn = (getattr(conn, "_connectors", None) or [None])[0]
    mgr = conn.scheduler_manager
    RES["async_scheduling"] = bool(getattr(sched, "async_scheduling", None))
    RES["mgr"] = {"num_cpu_blocks": mgr.num_cpu_blocks, "n_groups": len(kvm.kv_cache_config.kv_cache_groups), "num_gpu_blocks": pool.num_gpu_blocks,
                  "group_block_sizes": list(mgr.group_block_sizes), "lazy": mgr._lazy_mode, "expected_workers": mgr._expected_worker_count}

    # RPC smoke test right after load: a serialization failure shows here and not 40 minutes later.
    sm = safe("smoke", lambda: llm.collective_rpc(W.smoke))
    RES["smoke"] = sm
    log("smoke", json.dumps(sm)[:600])
    save()

    prompts = E.build_prompts(llm.get_tokenizer())
    plen = [len(p) for p in prompts]
    n = len(prompts)
    inputs = [{"prompt_token_ids": p} for p in prompts]
    RES["plen"] = plen

    def sp(**k):
        return SamplingParams(max_tokens=GEN, ignore_eos=True, **k)

    # ---------------- schedule tracer ----------------
    class Tracer:
        def __init__(self):
            self.on = False
            self.steps = []
            self.watch_all = False
            self.snap_last = None
            self.load_last = None
            self.collect_loads = False
            self.loads = {}
            self.errors = []
            orig = sched.schedule

            def wrapped(*a, **k):
                out = orig(*a, **k)
                try:
                    self.record(out)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(repr(e))
                return out

            sched.schedule = wrapped

        @staticmethod
        def ext(rid):
            return rid.rsplit("-", 1)[0]

        def record(self, out):
            if self.on:
                nrun, nwait = sched.get_request_counts()
                nremote = sum(1 for r in sched.requests.values() if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
                self.steps.append({"step": len(self.steps), "sched": {self.ext(r): c for r, c in out.num_scheduled_tokens.items()},
                                   "tot": out.total_num_scheduled_tokens, "run": nrun, "wait": nwait, "remote": nremote})
            if self.collect_loads:
                # D4': every load of the pass, keyed by the engine request id (the entry lives only until the load completes)
                for rid, st in mgr._reqs_to_load.items():
                    if rid not in self.loads:
                        self.loads[rid] = (list(st.transfer_meta.gpu_block_ids), list(st.transfer_meta.cpu_block_ids))
            if self.watch_all:
                # the load entry exists only between update_state_after_alloc (inside schedule) and the load completion: capture it here, not at a later step
                for rid, st in mgr._reqs_to_load.items():
                    if self.load_last is None:
                        self.load_last = (list(st.transfer_meta.gpu_block_ids), list(st.transfer_meta.cpu_block_ids))
                for rid in out.num_scheduled_tokens:
                    try:
                        self.snap_last = [list(x) for x in kvm.get_block_ids(rid)]
                    except Exception as e:  # noqa: BLE001
                        self.errors.append("snap " + repr(e))

        def begin(self):
            self.on = True
            self.steps = []

        def end(self, outs):
            self.on = False
            m = {o.request_id: i for i, o in enumerate(outs)}
            res = []
            for s in self.steps:
                res.append({"step": s["step"], "sched": {str(m.get(k, "x" + k)): c for k, c in s["sched"].items()}, "tot": s["tot"], "run": s["run"],
                            "wait": s["wait"], "remote": s["remote"]})
            return res

    tr = Tracer()

    class SpecCount:
        """Accepted / drafted tokens per generate call (alone in the engine = per request). The scheduler counts them in make_spec_decoding_stats, which
        returns early when the stat logger is off (offline LLM default), so the call is wrapped and the numbers are taken before it."""

        def __init__(self):
            self.cur = None
            self.errors = []
            self.enabled = os.environ.get("E2E_SPEC", "1") == "1" and hasattr(sched, "make_spec_decoding_stats")
            if self.enabled:
                orig = sched.make_spec_decoding_stats

                def wrapped(*a, **k):
                    try:
                        if self.cur is not None:
                            d = int(k["num_draft_tokens"]) - int((k.get("num_invalid_spec_tokens") or {}).get(k["request_id"], 0))
                            self.cur["drafted"] += d
                            self.cur["accepted"] += int(k["num_accepted_tokens"])
                            self.cur["calls"] += 1
                    except Exception as e:  # noqa: BLE001
                        self.errors.append(repr(e))
                    return orig(*a, **k)

                sched.make_spec_decoding_stats = wrapped

        def begin(self):
            self.cur = {"drafted": 0, "accepted": 0, "calls": 0} if self.enabled else None

        def take(self):
            c, self.cur = self.cur, None
            return c

    SC = SpecCount()

    def run_pass(tag, ins, sps, pl, traced=True):
        t1 = time.time()
        if traced:
            tr.begin()
        outs = llm.generate(ins, sps, use_tqdm=False)
        run_pass.last_outs = outs
        trace = tr.end(outs) if traced else None
        cached = sum(o.num_cached_tokens or 0 for o in outs)
        log(f"pass {tag} took {time.time() - t1:.1f} s cached_tokens={cached} of {sum(pl)} steps={len(trace) if trace else '-'}")
        RES.setdefault("pass_post", {})[tag] = {"used": used(), "pending_stores": mgr.has_pending_stores(), "pending_finished": len(mgr._pending_finished_stores),
                                                 "inflight_store_events": len(mgr._store_event_to_blocks), "has_requests": sched.has_requests()}
        rows = E.dump(outs, "req", pl)
        with open(f"{OUT}/{CHK}.{tag}.json", "w") as f:
            json.dump({"requests": rows}, f)
        if trace is not None:
            with open(f"{OUT}/{CHK}.{tag}.trace.json", "w") as f:
                json.dump(trace, f)
        return rows, trace

    # ---------------- state dumps ----------------
    def used():
        return pool.num_gpu_blocks - pool.get_num_free_blocks() - 1

    def tagmap():
        tags = {}

        def add(bids, t):
            for b in bids:
                tags.setdefault(b, []).append(t)

        for ev, t in mgr._store_event_to_blocks.items():
            add(t.gpu_block_ids, f"store_inflight:{ev}")
        for ev, t in mgr._abandoned_store_event_to_blocks.items():
            add(t.gpu_block_ids, f"store_abandoned:{ev}")
        for t in mgr._pending_finished_stores:
            add(t.gpu_block_ids, "pending_finished_store")
        for r, s in mgr._reqs_to_load.items():
            add(s.transfer_meta.gpu_block_ids, f"load:{r}")
        for r, s in mgr._abandoned_reqs_to_load.items():
            add(s.transfer_meta.gpu_block_ids, f"abandoned_load:{r}")
        add(mgr._in_flight_store_gpu_blocks, "in_flight_set")
        for r in list(sched.requests):
            try:
                for g in kvm.get_block_ids(r):
                    add(g, f"req:{r}")
            except Exception:  # noqa: BLE001
                pass
        return tags

    def dump_state(label, full=False):
        held = [b for b in pool.blocks if b.ref_cnt > 0 and not b.is_null]
        tags = tagmap()
        by_group = {}
        by_kind = {}
        untracked = []
        rows = []
        for b in held:
            g = get_group_id(b.block_hash) if b.block_hash is not None else -1
            by_group[g] = by_group.get(g, 0) + 1
            tg = tags.get(b.block_id, [])
            kinds = sorted({t.split(":")[0] for t in tg})
            for k in kinds:
                by_kind[k] = by_kind.get(k, 0) + 1
            if not tg:
                untracked.append(b.block_id)
            rows.append([b.block_id, b.ref_cnt, g, kinds])
        d = {
            "label": label, "n_used": used(), "n_held": len(held), "by_group": by_group, "by_kind": by_kind, "untracked_ids": untracked,
            "held": rows if (full or len(rows) <= 64) else rows[:64],
            "requests": {str(s): sum(1 for r in sched.requests.values() if r.status == s) for s in {r.status for r in sched.requests.values()}},
            "mgr": {
                "store_event_to_blocks": {k: len(v.gpu_block_ids) for k, v in mgr._store_event_to_blocks.items()},
                "abandoned_store_event_to_blocks": {k: len(v.gpu_block_ids) for k, v in mgr._abandoned_store_event_to_blocks.items()},
                "store_event_pending_counts": dict(mgr._store_event_pending_counts),
                "pending_finished_stores": [len(t.gpu_block_ids) for t in mgr._pending_finished_stores],
                "abandoned_reqs_to_load": list(mgr._abandoned_reqs_to_load),
                "reqs_to_load": list(mgr._reqs_to_load),
                "reqs_to_store": {r: {"events": sorted(s.store_events), "finished": s.finished} for r, s in mgr._reqs_to_store.items()},
                "store_event_to_reqs": {k: list(v) for k, v in mgr._store_event_to_reqs.items()},
                "in_flight_store_gpu_blocks": len(mgr._in_flight_store_gpu_blocks),
                "pending_cpu_hits": list(mgr._pending_cpu_hits),
                "load_event_to_reqs": {k: list(v) for k, v in mgr._load_event_to_reqs.items()},
                "store_event_counter": mgr._store_event_counter,
                "has_pending_stores": mgr.has_pending_stores(),
                "cpu_free_blocks": mgr.cpu_block_pool.get_num_free_blocks(), "num_cpu_blocks": mgr.num_cpu_blocks,
            },
            "has_requests": sched.has_requests(),
        }
        if full:
            d["workers"] = safe("worker_state", lambda: llm.collective_rpc(W.store_state))
        log("DUMP", label, json.dumps({k: d[k] for k in ("n_used", "n_held", "by_group", "by_kind", "untracked_ids", "has_requests")}), "mgr", json.dumps(d["mgr"])[:700])
        return d

    def drain(max_steps=400):
        k = 0
        while sched.has_requests() and k < max_steps:
            llm.llm_engine.step()
            k += 1
            time.sleep(0.02)
        return k

    def one_token_request(nt):
        llm.generate([{"prompt_token_ids": prompts[0][:nt]}], SamplingParams(max_tokens=1), use_tqdm=False)

    def e2e_reset(connector, tag, ntok=16, maxtry=40, dump_at=()):
        """The gate driver's retry loop (a 1-token request between attempts); returns (retries, trail)."""
        trail = []
        for i in range(maxtry):
            ok = llm.reset_prefix_cache(reset_connector=connector)
            trail.append({"try": i, "ok": ok, "used": used(), "pending_stores": mgr.has_pending_stores()})
            if ok:
                log(f"reset({'full' if connector else 'gpu'}) {tag} ok after {i} retries, trail used={[t['used'] for t in trail]}")
                return i, trail
            if i in dump_at:
                trail[-1]["dump"] = dump_state(f"{tag}_try{i}", full=(i == dump_at[-1]))
            one_token_request(ntok)
            time.sleep(0.5)
        log(f"reset({'full' if connector else 'gpu'}) {tag} FAILED after {maxtry}, trail used={[t['used'] for t in trail]}")
        return None, trail

    def robust_reset(connector, tag, maxtry=100):
        """Reset retry that sends NO request: it drains the engine (idle steps) between attempts."""
        for i in range(maxtry):
            if llm.reset_prefix_cache(reset_connector=connector):
                return i
            drain()
            time.sleep(0.1)
        raise RuntimeError(f"robust reset kept failing before {tag}: {dump_state(tag + '_robust_fail', full=True)}")

    def cmp_rows(a, b):
        diff = [i for i in range(len(a)) if a[i]["token_ids"] != b[i]["token_ids"]]
        return diff

    # ---------------- greedy C/H/X with trace ----------------
    greedy = [sp(temperature=0.0) for _ in range(n)]

    def chx(tag):
        r, trail_c = e2e_reset(True, f"{tag}_C")
        C, tC = run_pass(f"{tag}_C", inputs, greedy, plen)
        H, tH = run_pass(f"{tag}_H", inputs, greedy, plen)
        r2, trail_x = e2e_reset(False, f"{tag}_X")
        X, tX = run_pass(f"{tag}_X", inputs, greedy, plen)
        s = {"retries": [r, r2], "trail_c": trail_c, "trail_x": trail_x,
             "diff_XH": cmp_rows(X, H), "diff_HC": cmp_rows(H, C), "diff_XC": cmp_rows(X, C),
             "cached": {"C": [x["num_cached_tokens"] for x in C], "H": [x["num_cached_tokens"] for x in H], "X": [x["num_cached_tokens"] for x in X]}}
        s["n_diff_XH"] = len(s["diff_XH"])
        s["n_diff_HC"] = len(s["diff_HC"])
        s["n_diff_XC"] = len(s["diff_XC"])
        # schedule digest: per step membership, sets that differ between H and X
        for nm, t in (("C", tC), ("H", tH), ("X", tX)):
            s[f"steps_{nm}"] = len(t)
            s[f"steps_remote_{nm}"] = sum(1 for x in t if x["remote"] > 0)
            s[f"max_remote_{nm}"] = max([x["remote"] for x in t] or [0])
            s[f"tot_{nm}"] = [x["tot"] for x in t]
            s[f"nreq_{nm}"] = [len(x["sched"]) for x in t]
            s[f"remote_{nm}"] = [x["remote"] for x in t]
        for rq in ("15", "59", "0"):
            for nm, t in (("H", tH), ("X", tX)):
                s[f"req{rq}_{nm}_steps"] = [[x["step"], x["sched"][rq], len(x["sched"]), x["tot"]] for x in t if rq in x["sched"]]
        log(f"{tag}: n_diff XH={s['n_diff_XH']} {s['diff_XH']} HC={s['n_diff_HC']} XC={s['n_diff_XC']} steps C/H/X={s['steps_C']}/{s['steps_H']}/{s['steps_X']} "
            f"remote-steps H/X={s['steps_remote_H']}/{s['steps_remote_X']} max remote X={s['max_remote_X']}")
        return s

    if CHK != "d4":
        RES["chx_greedy"] = safe("chx", lambda: chx("greedy"))
        RES["tracer_errors"] = tr.errors[:10]
        save()

    if CHK == "spec":
        # ---------- reset reproduction, order A (the task order): GPU-only resets until they succeed, then the full reset ----------
        def reproduce(label, gpu_first):
            rec = {"label": label}
            rec["S0"] = dump_state(f"{label}_before")
            if gpu_first:
                g, gtrail = e2e_reset(False, f"{label}_gpu_only")
                rec["gpu_only_retries"] = g
                rec["gpu_only_trail"] = gtrail
                rec["S1"] = dump_state(f"{label}_after_gpu_only")
            tr.watch_all = True
            ok = llm.reset_prefix_cache(reset_connector=True)
            rec["first_full_ok"] = ok
            rec["first_full_used"] = used()
            if ok:
                tr.watch_all = False
                return rec
            rec["first_full_dump"] = dump_state(f"{label}_full_try0", full=True)
            # the harness loop: a 16-token request between attempts
            trail = [{"try": 0, "ok": False, "used": used()}]
            ok = False
            for i in range(1, 11):
                tr.snap_last = None
                one_token_request(16)
                retry_blocks = tr.snap_last
                time.sleep(0.5)
                ent = {"try": i}
                if i in (1, 2, 3):
                    # state at the moment of the call, i.e. before this attempt's reset: what pins the GPU blocks?
                    pre = dump_state(f"{label}_pre_try{i}", full=(i == 1))
                    held_ids = {r[0] for r in pre["held"]}
                    rb = {b for g in (retry_blocks or []) for b in g}
                    ent["pre_dump"] = pre
                    ent["pre_held_ids"] = sorted(held_ids)
                    ent["pre_held_in_retry_request_blocks"] = len(held_ids & rb)
                    ent["n_retry_request_blocks_nonnull"] = len(rb - {0})
                ok = llm.reset_prefix_cache(reset_connector=True)
                ent.update({"ok": ok, "used": used(), "pending_stores": mgr.has_pending_stores()})
                if not ok and i in (3, 10):
                    ent["dump"] = dump_state(f"{label}_full_try{i}", full=(i == 10))
                trail.append(ent)
                log(f"{label} harness-loop try {i} ok={ok} used={used()}")
                if ok:
                    break
            tr.watch_all = False
            rec["harness_loop"] = trail
            rec["harness_loop_ok"] = ok
            if not ok:
                # discriminator: no request, only engine steps
                rec["drain_steps"] = drain()
                rec["after_drain_dump"] = dump_state(f"{label}_after_drain", full=True)
                ok2 = llm.reset_prefix_cache(reset_connector=True)
                rec["reset_after_drain_ok"] = ok2
                rec["reset_after_drain_used"] = used()
                if not ok2:
                    rec["drain_steps2"] = drain()
                    time.sleep(1.0)
                    rec["reset_after_drain2_ok"] = llm.reset_prefix_cache(reset_connector=True)
                    rec["after_drain2_dump"] = dump_state(f"{label}_after_drain2", full=True)
                log(f"{label}: reset after drain ok={rec.get('reset_after_drain_ok')} used={used()}")
                # variant: pin again with a 16-token request, then retry with a request below one block (no store)
                if rec.get("reset_after_drain_ok") or rec.get("reset_after_drain2_ok"):
                    one_token_request(16)
                    var = []
                    ok3 = False
                    for i in range(10):
                        ok3 = llm.reset_prefix_cache(reset_connector=True)
                        var.append({"try": i, "ok": ok3, "used": used()})
                        if ok3:
                            break
                        one_token_request(8)
                        time.sleep(0.5)
                    rec["short_request_loop"] = var
                    rec["short_request_loop_ok"] = ok3
                    log(f"{label}: short (8-token) request loop ok={ok3} tries={len(var)} used={[v['used'] for v in var]}")
                    # back-to-back calls: the 16-token request first (pins its blocks), then two resets in a row with nothing in between
                    one_token_request(16)
                    time.sleep(0.5)
                    b1 = llm.reset_prefix_cache(reset_connector=True)
                    b2 = llm.reset_prefix_cache(reset_connector=True)
                    rec["back_to_back"] = {"first": b1, "second": b2}
                    log(f"{label}: back-to-back resets after a 16-token request: first={b1} second={b2}")
            return rec

        RES["repro_A"] = safe("repro_A", lambda: reproduce("A", True))
        save()
        a = RES.get("repro_A") or {}
        if a.get("first_full_ok") and a.get("S0", {}).get("n_held") is not None:
            # order A did not reproduce on the first full reset: rerun C/H/X and use the gate's original order (full reset straight after X)
            RES["chx_greedy_B"] = safe("chx_B", lambda: chx("greedyB"))
            RES["repro_B"] = safe("repro_B", lambda: reproduce("B", False))
            save()
        elif a and not a.get("first_full_ok"):
            RES["repro_B_skipped"] = "order A reproduced the failure"
        save()

        # V0: the gate driver's complete flow (4 modes x C/H/X with the gate's own retry loops); a RuntimeError of the gate driver shows up as a failed trail
        def gate_flow():
            t1k64 = [sp(temperature=1.0, top_k=64, top_p=1.0, seed=1000 + i) for i in range(n)]
            mixed = []
            for i in range(n):
                kind = i % 4
                if kind == 0:
                    mixed.append(sp(temperature=1.0, top_k=64, top_p=1.0, seed=4000 + i))
                elif kind == 1:
                    mixed.append(sp(temperature=1.0, top_k=64, top_p=0.95, seed=4000 + i))
                elif kind == 2:
                    mixed.append(sp(temperature=1.0, top_k=64, frequency_penalty=0.2, seed=4000 + i))
                else:
                    mixed.append(sp(temperature=0.0))
            lp_idx = [i for i in range(n) if 600 <= plen[i] <= 9100][:E.LP_PROMPTS]
            lp_ins = [inputs[i] for i in lp_idx]
            lp_sps = [sp(temperature=0.0, logprobs=5) for _ in lp_idx]
            lp_pl = [plen[i] for i in lp_idx]
            out = {}
            for mode, ins, sps, pl in (("greedy", inputs, greedy, plen), ("t1k64", inputs, t1k64, plen), ("mixed", inputs, mixed, plen), ("lp", lp_ins, lp_sps, lp_pl)):
                r, trail = e2e_reset(True, f"gate_{mode}_C", dump_at=(1, 3))
                out[mode] = {"full_reset_retries": r, "full_trail": trail}
                if r is None:
                    out[mode]["final_dump"] = dump_state(f"gate_{mode}_C_final", full=True)
                    break
                run_pass(f"gate_{mode}_C", ins, sps, pl, traced=False)
                run_pass(f"gate_{mode}_H", ins, sps, pl, traced=False)
                r2, trail2 = e2e_reset(False, f"gate_{mode}_X")
                out[mode]["gpu_reset_retries"] = r2
                out[mode]["gpu_trail"] = trail2
                run_pass(f"gate_{mode}_X", ins, sps, pl, traced=False)
                RES["v0_gate_flow"] = out
                save()
            return out

        RES["v0_gate_flow"] = safe("v0_gate_flow", gate_flow)
        save()

        # V1: full reset while the stores of a cold pass are still in flight (right after C), then the harness loop / drain experiment
        def v1():
            robust_reset(True, "v1_pre")
            run_pass("v1_C", inputs, greedy, plen, traced=False)
            return reproduce("V1", False)

        RES["v1_after_C"] = safe("v1_after_C", v1)
        save()

        # V2: full reset right after a cold pass of a different mode, with a GPU-only loop first (the task order) while stores may be in flight
        def v2():
            robust_reset(True, "v2_pre")
            run_pass("v2_C", inputs, greedy, plen, traced=False)
            return reproduce("V2", True)

        RES["v2_after_C_gpu_first"] = safe("v2_after_C_gpu_first", v2)
        save()
        log("DONE spec")
        return

    # ================= ns256 =================
    def lpdict(rows_lp):
        return rows_lp

    def lp_cmp(a, b):
        if a is None or b is None:
            return {"equal": a == b, "max_abs": None}
        if len(a) != len(b):
            return {"equal": False, "max_abs": None, "len": [len(a), len(b)]}
        mx = 0.0
        eq = True
        for x, y in zip(a, b):
            if x == y:
                continue
            eq = False
            if x is None or y is None:
                mx = float("inf")
                continue
            for t in set(x) & set(y):
                mx = max(mx, abs(x[t][0] - y[t][0]))
            if set(x) != set(y):
                mx = max(mx, 1e9)
        return {"equal": eq, "max_abs": mx}

    def mk_sp(i, kind):
        if kind == "greedy_lp":
            return sp(temperature=0.0, logprobs=5)
        if kind == "greedy":
            return sp(temperature=0.0)
        if kind == "t1k64":
            return sp(temperature=1.0, top_k=64, top_p=1.0, seed=1000 + i)
        if kind == "mixed":
            return sp(temperature=1.0, top_k=64, top_p=1.0, seed=4000 + i)
        raise ValueError(kind)

    def gen_one(i, kind):
        tr.snap_last = None
        tr.load_last = None
        SC.begin()
        o = llm.generate([inputs[i]], [mk_sp(i, kind)], use_tqdm=False)[0]
        c = o.outputs[0]
        return {"token_ids": list(c.token_ids), "logprobs": E.lp_dicts(c.logprobs), "cached": o.num_cached_tokens, "spec": SC.take()}

    def block_keys(snap):
        """[(key_hex, group, block_id)] of the hashed, non-null blocks of a request snapshot."""
        res = []
        for g, ids in enumerate(snap or []):
            for b in ids:
                blk = pool.blocks[b]
                if blk.is_null or blk.block_hash is None:
                    continue
                res.append((bytes(blk.block_hash).hex(), g, b))
        return res

    def hash_ids(gpu_ids, cpu_ids):
        r = llm.collective_rpc(W.hash_blocks, args=(list(gpu_ids), list(cpu_ids)))
        r = sorted(r, key=lambda x: x["rank"])
        return r

    def cpu_lookup(key_hex):
        from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

        blk = mgr.cpu_block_pool.cached_block_hash_to_block.get_one_block(BlockHashWithGroupId(bytes.fromhex(key_hex)))
        return None if blk is None else blk.block_id

    def single_stream(i, kind, hash_kv=False):
        rec = {"req": i, "plen": plen[i], "kind": kind}
        rec["reset_c"] = robust_reset(True, f"ss{i}{kind}_C")
        tr.watch_all = True
        C = gen_one(i, kind)
        H = gen_one(i, kind)
        hk = None
        if hash_kv:
            keysH = block_keys(tr.snap_last)
            gids = [b for _, _, b in keysH]
            hk = {"keysH": keysH, "dig_H": hash_ids(gids, [])}
        rec["drain_steps"] = drain()
        if hash_kv:
            cpu_ids_H = [cpu_lookup(k) for k, _, _ in hk["keysH"]]
            have = [j for j, c in enumerate(cpu_ids_H) if c is not None]
            hk["cpu_by_hash"] = {"n_keys": len(cpu_ids_H), "n_found": len(have)}
            r = hash_ids([], [cpu_ids_H[j] for j in have])
            hk["dig_cpuH"] = r
            hk["have"] = have
        rec["reset_x"] = robust_reset(False, f"ss{i}{kind}_X")
        X = gen_one(i, kind)
        tr.watch_all = False
        if hash_kv:
            pairs = tr.load_last
            hk["pairs_n"] = None if pairs is None else len(pairs[0])
            keyX = []
            if pairs is not None:
                for gb, cb in zip(*pairs):
                    blk = pool.blocks[gb]
                    keyX.append((bytes(blk.block_hash).hex() if blk.block_hash is not None else None, gb, cb))
            hk["n_pairs_nohash"] = sum(1 for k in keyX if k[0] is None)
            rX = hash_ids([g for _, g, _ in keyX], [c for _, _, c in keyX])
            hk["keyX"] = keyX
            hk["dig_X"] = rX
            rec["kv"] = kv_compare(hk)
        for nm, a, b in (("HX", H, X), ("CH", C, H), ("CX", C, X)):
            lc = lp_cmp(a["logprobs"], b["logprobs"])
            rec[nm] = {"tokens_equal": a["token_ids"] == b["token_ids"], "first_diff": next((k for k in range(min(len(a["token_ids"]), len(b["token_ids"])))
                                                                                          if a["token_ids"][k] != b["token_ids"][k]), None),
                       "lp_equal": lc["equal"], "lp_max_abs": lc["max_abs"]}
        rec["cached"] = {"C": C["cached"], "H": H["cached"], "X": X["cached"]}
        with open(f"{OUT}/ss_{i}_{kind}.json", "w") as f:
            json.dump({"C": C, "H": H, "X": X}, f)
        log(f"ss req {i} ({plen[i]} tok) {kind}: cached C/H/X={rec['cached']} HX tokens_equal={rec['HX']['tokens_equal']} first_diff={rec['HX']['first_diff']} "
            f"lp_equal={rec['HX']['lp_equal']} lp_max_abs={rec['HX']['lp_max_abs']} CH={rec['CH']['tokens_equal']} CX={rec['CX']['tokens_equal']}")
        return rec

    def kv_compare(hk):
        keysH = hk["keysH"]
        have = hk["have"]
        keyX = hk["keyX"]
        out = {"n_H_blocks": len(keysH), "n_cpu_found_by_hash": len(have), "n_restored": len(keyX), "n_restored_nohash": hk["n_pairs_nohash"]}
        hmap = {keysH[j][0]: j for j in range(len(keysH))}
        xmap = {k[0]: j for j, k in enumerate(keyX) if k[0] is not None}
        common = sorted(set(hmap) & set(xmap))
        out["n_common_H_X"] = len(common)
        out["n_restored_not_in_H"] = len(set(xmap) - set(hmap))
        hpos = {j: p for p, j in enumerate(have)}
        ranks = []
        tot = {"HX": 0, "HX_bad": 0, "XC": 0, "XC_bad": 0, "HC": 0, "HC_bad": 0}
        bad_examples = []
        for r in range(len(hk["dig_H"])):
            dH = hk["dig_H"][r]["gpu"]
            dCH = hk["dig_cpuH"][r]["cpu"]
            dX = hk["dig_X"][r]["gpu"]
            dXC = hk["dig_X"][r]["cpu"]
            nreg = hk["dig_H"][r]["n_regions"]
            rk = {"rank": r, "n_regions": nreg, "HX_bad": 0, "XC_bad": 0, "HC_bad": 0}
            for k in common:
                jh, jx = hmap[k], xmap[k]
                tot["HX"] += 1
                if dH[jh] != dX[jx]:
                    tot["HX_bad"] += 1
                    rk["HX_bad"] += 1
                    if len(bad_examples) < 6:
                        bad_examples.append({"rank": r, "key": k[:12], "group": keysH[jh][1], "regions_bad": [q for q in range(nreg) if dH[jh][16 * q:16 * q + 16] != dX[jx][16 * q:16 * q + 16]][:8]})
                tot["XC"] += 1
                if dX[jx] != dXC[jx]:
                    tot["XC_bad"] += 1
                    rk["XC_bad"] += 1
                if jh in hpos:
                    tot["HC"] += 1
                    if dH[jh] != dCH[hpos[jh]]:
                        tot["HC_bad"] += 1
                        rk["HC_bad"] += 1
            ranks.append(rk)
        out["totals_all_ranks"] = tot
        out["ranks"] = ranks
        out["bad_examples"] = bad_examples
        gcount = {}
        for k in common:
            g = keysH[hmap[k]][1]
            gcount[g] = gcount.get(g, 0) + 1
        out["common_by_group"] = gcount
        out["bytes_equal"] = bool(tot["HX"] > 0 and tot["HX_bad"] == 0 and tot["XC_bad"] == 0 and tot["HC_bad"] == 0 and tot["XC"] > 0)
        log("KV compare", json.dumps({k: out[k] for k in ("n_H_blocks", "n_cpu_found_by_hash", "n_restored", "n_common_H_X", "totals_all_ranks", "common_by_group", "bytes_equal")}))
        return out

    # ================= D4' (ADOPTION_RULE_FINAL_AMENDMENT3) =================
    def run_d4():
        from vllm.v1.core.kv_cache_utils import get_group_id

        tp = 8
        spec_profile = os.environ.get("E2E_SPEC", "1") == "1"
        profile = "spec" if spec_profile else "ns256"
        vpath = f"{OUT}/d4_verdict_{profile}.json"
        groups = kvm.kv_cache_config.kv_cache_groups

        def group_info(gr):
            ksp = gr.kv_cache_spec
            subs = getattr(ksp, "kv_cache_specs", None)
            draft = bool(getattr(ksp, "is_draft", False)) or (isinstance(subs, dict) and any(getattr(x, "is_draft", False) for x in subs.values()))
            return {"n_layers": len(gr.layer_names), "first_layer": gr.layer_names[0] if gr.layer_names else None, "spec_type": type(ksp).__name__, "is_draft": draft}

        D = {"profile": profile, "spec": spec_profile, "n_groups": len(groups), "n_det": n, "tp": tp, "groups": [group_info(g) for g in groups]}
        D["draft_groups"] = [g for g, x in enumerate(D["groups"]) if x["is_draft"]]
        D["smoke_regions"] = (sm or [{}])[0].get("region_names")
        RES["d4"] = D

        def write_verdict(final=False):
            D["errors"] = dict(RES.get("errors") or {})
            D["tracer_errors"] = tr.errors[:10] + SC.errors[:10]
            v = final_verdict(D)
            out = {"tool": "chk_driver.py d4", "rule": "ADOPTION_RULE_FINAL_AMENDMENT3 D4'", "profile": profile, "finished": final, "time": time.strftime("%F %T"),
                   "load_s": RES.get("load_s"), "env": RES["env"], **v, "detail": D}
            with open(vpath + ".tmp", "w") as f:
                json.dump(out, f, indent=1, default=str)
            os.replace(vpath + ".tmp", vpath)
            save()
            return v

        def subset(idxs):
            return [inputs[i] for i in idxs], [greedy[i] for i in idxs], [plen[i] for i in idxs]

        def all_blocks():
            hm = {}
            for b in pool.blocks:
                if b.block_hash is not None and not b.is_null:
                    hm.setdefault(bytes(b.block_hash), []).append(b.block_id)
            return hm

        def cold_hot(tag, idxs):
            """Full reset, cold pass C, GPU-hit pass H, drain; the pass-H GPU copies of every hashed block are digested on the workers. -> ({hash: [gpu ids]}, record)"""
            ins, sps_, pl = subset(idxs)
            rec = {"n_requests": len(idxs), "reset_c_retries": robust_reset(True, f"{tag}_C")}
            run_pass(f"{tag}_C", ins, sps_, pl, traced=False)
            rowsH, _ = run_pass(f"{tag}_H", ins, sps_, pl, traced=False)
            rec["drain_h_steps"] = drain()
            hm = all_blocks()
            ids = sorted({b for v in hm.values() for b in v})
            r = llm.collective_rpc(W.snap_h, args=(ids,))
            rec["snap_h"] = {"n_blocks": len(ids), "n_keys": len(hm), "n_dup_keys": sum(1 for v in hm.values() if len(v) > 1), "ranks": sorted(x["rank"] for x in r)}
            rec["cached_H"] = [x["num_cached_tokens"] for x in rowsH]
            return hm, rec

        def restore_pass(tag, idxs):
            """GPU-only reset (after the drain), tier-hit pass X with the loads of every request captured. -> (rows, loads, request-id map, record)"""
            ins, sps_, pl = subset(idxs)
            rec = {"reset_x_retries": robust_reset(False, f"{tag}_X")}
            tr.loads = {}
            tr.collect_loads = True
            try:
                rows, _ = run_pass(f"{tag}_X", ins, sps_, pl, traced=False)
            finally:
                tr.collect_loads = False
            m = {o.request_id: k for k, o in enumerate(run_pass.last_outs)}
            rec["drain_x_steps"] = drain()
            rec["cached_X"] = [x["num_cached_tokens"] for x in rows]
            rec["n_load_requests"] = len(tr.loads)
            return rows, dict(tr.loads), m, rec

        def build_pairs(loads, m, idxs, hm):
            """[(key hex, group, request, gpu id, cpu id, number of pass-H candidates)] of the restored blocks; a pair whose GPU and tier block do not carry
            the same hash (block reused or evicted meanwhile) is void."""
            pairs, void, vex, by_req = [], 0, [], {}
            for rid, (gl, cl) in loads.items():
                k = m.get(Tracer.ext(rid))
                req = idxs[k] if k is not None else -1
                for gb, cb in zip(gl, cl):
                    g = pool.blocks[gb].block_hash
                    c = mgr.cpu_block_pool.blocks[cb].block_hash
                    if g is None or c is None or bytes(g) != bytes(c):
                        void += 1
                        if len(vex) < 6:
                            vex.append([req, gb, cb, None if g is None else bytes(g).hex()[:16], None if c is None else bytes(c).hex()[:16]])
                        continue
                    pairs.append((bytes(g).hex(), get_group_id(g), req, gb, cb, len(hm.get(bytes(g), ()))))
                    by_req[req] = by_req.get(req, 0) + 1
            return pairs, void, vex, by_req

        def compare_pairs(pairs, hm, void=0, vex=()):
            step = 20000
            chunks = []
            for off in range(0, len(pairs), step):
                part = pairs[off:off + step]
                res = llm.collective_rpc(W.compare_loads, args=([p[3] for p in part], [p[4] for p in part], [hm.get(bytes.fromhex(p[0]), []) for p in part]))
                chunks.append((off, res))
            return aggregate_compare(merge_rank_results(chunks), pairs, tp, void, vex)

        def stage_a():
            idxs = list(range(n))
            hm, rec = cold_hot("d4a", idxs)
            rows, loads, m, rec2 = restore_pass("d4a", idxs)
            pairs, void, vex, by_req = build_pairs(loads, m, idxs, hm)
            kv = compare_pairs(pairs, hm, void, vex)
            D["a_uncovered_requests"] = [idxs[k] for k, row in enumerate(rows) if (row["num_cached_tokens"] or 0) > 0 and by_req.get(idxs[k], 0) == 0]
            D["a"] = kv
            D["a_rec"] = {"cold_hot": rec, "restore": rec2, "loaded_blocks_per_request": {str(k): v for k, v in sorted(by_req.items())}}
            log("D4'(a)", json.dumps({k: kv[k] for k in ("n_loaded", "n_void", "n_with_h", "n_block_rank_xc", "n_block_rank_hx", "groups_loaded", "groups_with_h")}),
                "xc_bad", len(kv["xc_bad"]), "hx_bad", len(kv["hx_bad"]), "uncovered", D["a_uncovered_requests"])

        def digests_of(res):
            return {r["rank"]: r["dg"] for r in res}

        def stage_nc():
            """Negative control: corrupt one byte of one tier block (rank 3, last region) before a restore of one request; the compare must report that block
            on the H-vs-X leg and nothing else; then one byte of a restored GPU block (rank 6) must be reported by the X-vs-tier leg."""
            nc = D["nc"] = {}
            i0 = next((i for i in range(n) if 2500 <= plen[i] <= 9100), None)
            nc["request"] = i0
            idxs = [i0]
            hm, rec = cold_hot("d4nc", idxs)
            nc["cold_hot"] = rec
            rows, loads, m, rec1 = restore_pass("d4nc1", idxs)
            pairs, void, vex, _ = build_pairs(loads, m, idxs, hm)
            nc["baseline"] = compare_pairs(pairs, hm, void, vex)
            cand = [p for p in pairs if p[5]]
            gmax = max(p[1] for p in cand)
            tg = [p for p in cand if p[1] == gmax]
            t = tg[len(tg) // 2]
            nreg = nc["baseline"]["n_regions"][0]
            rank_t, region_t = 3, nreg - 1
            before = digests_of(llm.collective_rpc(W.tier_digest, args=([t[4]],)))
            cor = [c for c in llm.collective_rpc(W.corrupt_tier, args=(rank_t, t[4], region_t)) if c]
            after = digests_of(llm.collective_rpc(W.tier_digest, args=([t[4]],)))
            nc["tier_target"] = {"rank": rank_t, "key": t[0], "group": t[1], "cpu_id": t[4], "region": region_t, "worker": cor}
            nc["tier_digest_changed"] = bool(len(cor) == 1 and after[rank_t] != before[rank_t] and all(after[r] == before[r] for r in before if r != rank_t))
            try:
                rows2, loads2, m2, rec2 = restore_pass("d4nc2", idxs)
                pairs2, void2, vex2, _ = build_pairs(loads2, m2, idxs, hm)
                nc["target_in_loads"] = any(p[0] == t[0] for p in pairs2)
                nc["after_tier_corrupt"] = compare_pairs(pairs2, hm, void2, vex2)
                cand2 = [p for p in pairs2 if p[0] != t[0] and p[5]]
                other = [p for p in cand2 if p[1] != t[1]] or cand2
                g2 = other[len(other) // 2]
                rank_g, region_g = 6, 0
                corg = [c for c in llm.collective_rpc(W.corrupt_gpu, args=(rank_g, g2[3], region_g)) if c]
                nc["gpu_target"] = {"rank": rank_g, "key": g2[0], "group": g2[1], "gpu_id": g2[3], "region": region_g, "worker": corg}
                nc["after_gpu_corrupt"] = compare_pairs(pairs2, hm, void2, vex2)
            finally:
                # undo the tier corruption (XOR twice) and prove it; the next stage starts with a full reset anyway
                llm.collective_rpc(W.corrupt_tier, args=(rank_t, t[4], region_t))
                undone = digests_of(llm.collective_rpc(W.tier_digest, args=([t[4]],)))
                nc["tier_undo_ok"] = bool(undone == before)
            ok, why = eval_nc(nc)
            log("D4' negative control", "OK" if ok else "FAILED", why)

        def single(i, kind, lp_req):
            k_sp = "greedy_lp" if lp_req else kind
            rec = {"req": i, "kind": kind, "plen": plen[i], "lp_requested": lp_req}
            robust_reset(True, f"d4b{i}{kind}_C")
            C = gen_one(i, k_sp)
            H = gen_one(i, k_sp)
            drain()
            robust_reset(False, f"d4b{i}{kind}_X")
            X = gen_one(i, k_sp)
            drain()
            lc = lp_cmp(H["logprobs"], X["logprobs"]) if lp_req else {"equal": True, "max_abs": None}
            th, tx = H["token_ids"], X["token_ids"]
            sh, sx = H["spec"], X["spec"]
            rec.update(tokens_equal=th == tx, first_diff=next((k for k in range(min(len(th), len(tx))) if th[k] != tx[k]), None if len(th) == len(tx) else min(len(th), len(tx))),
                       lp_equal=lc["equal"], lp_max_abs=lc["max_abs"], cached={"C": C["cached"] or 0, "H": H["cached"] or 0, "X": X["cached"] or 0},
                       spec={"C": C["spec"], "H": sh, "X": sx},
                       spec_equal=True if not SC.enabled else bool(sh is not None and sx is not None and (sh["drafted"], sh["accepted"]) == (sx["drafted"], sx["accepted"])))
            with open(f"{OUT}/d4_ss_{profile}_{i}_{kind}.json", "w") as f:
                json.dump({"C": C, "H": H, "X": X}, f)
            return rec

        def stage_b():
            recs = D["b"] = []
            lp_idx = set([i for i in range(n) if 600 <= plen[i] <= 9100][:E.LP_PROMPTS])
            D["b_lp_requests"] = sorted(lp_idx)
            for kind in B_KINDS:
                for i in range(n):
                    r = safe(f"d4b_{i}_{kind}", lambda: single(i, kind, kind == "greedy" and i in lp_idx))
                    recs.append(r if r is not None else {"req": i, "kind": kind, "error": True})
                    if len(recs) % 10 == 0:
                        log(f"D4'(b) {len(recs)} of {2 * n} done")
                        write_verdict()

        write_verdict()
        safe("d4_a", stage_a)
        write_verdict()
        safe("d4_nc", stage_nc)
        write_verdict()
        safe("d4_b", stage_b)
        v = write_verdict(final=True)
        log("D4' VERDICT", profile, v["verdict"], "fail:", v["reasons_fail"][:5], "artifact:", v["reasons_artifact"][:5])

    if CHK == "d4":
        run_d4()
        log("DONE d4")
        return

    ss = []
    plan = [(15, "greedy_lp"), (59, "greedy_lp"), (0, "greedy_lp"), (80, "greedy_lp"), (39, "greedy_lp"), (68, "greedy_lp"), (72, "greedy_lp"), (36, "greedy_lp"),
            (0, "greedy"), (80, "greedy"), (0, "t1k64"), (39, "t1k64"), (68, "t1k64"), (72, "t1k64"), (36, "mixed"), (15, "t1k64"), (59, "t1k64")]
    RES["single_stream"] = ss
    for i, kind in plan:
        r = safe(f"ss_{i}_{kind}", lambda: single_stream(i, kind, hash_kv=(i == 0 and kind == "greedy_lp")))
        if r is not None:
            ss.append(r)
            save()
    save()
    log("DONE ns256")


if __name__ == "__main__":
    main()
