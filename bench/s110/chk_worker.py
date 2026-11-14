"""S1-10R Phase 2 / S1-10 D4' GPU checks: worker-side probes run through LLM.collective_rpc (callables take the worker as first argument).

D4' (ADOPTION_RULE_FINAL_AMENDMENT3): snap_h digests the GPU-resident blocks of pass H, compare_loads digests the blocks restored from the tier in pass X
(GPU copy and CPU tier copy, ALL regions of the block = all KV groups incl. the draft's) and compares them with each other and with the pass-H copy on the
worker, so only mismatch lists travel back; corrupt_tier / corrupt_gpu / tier_digest are the negative control. The digest functions import torch lazily.
"""
import hashlib



def _handler():
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    conn = get_kv_transfer_group()
    seen = [conn]
    while seen:
        c = seen.pop()
        h = getattr(c, "worker_handler", None)
        if h is not None:
            return h
        seen.extend(getattr(c, "_connectors", []) or [])
    raise RuntimeError("no SimpleCPUOffloadWorker handler found")


def smoke(self):
    """Trivial probe right after load: rank, device, region count, tier blocks and bytes."""
    import torch

    h = _handler()
    gpu = h.gpu_kv_caches or {}
    cpu = h.cpu_kv_caches or {}
    return {
        "rank": int(getattr(self, "rank", -1)),
        "device": str(h.device),
        "n_regions": len(gpu),
        "num_cpu_blocks": int(h.num_cpu_blocks),
        "gpu_bytes_per_block": int(sum(t.stride(0) * t.element_size() for t in gpu.values())),
        "cpu_total_bytes": int(sum(t.numel() * t.element_size() for t in cpu.values())),
        "cpu_pinned": bool(next(iter(cpu.values())).is_pinned()) if cpu else None,
        "cuda": torch.cuda.is_available(),
        "region_names": sorted(gpu),
        "region_dtypes": sorted({str(t.dtype) for t in gpu.values()}),
    }


def store_state(self):
    h = _handler()
    return {
        "rank": int(getattr(self, "rank", -1)),
        "store_events": [i for i, _ in h._store_events],
        "load_events": [i for i, _ in h._load_events],
        "store_hwm": h._store_hwm,
        "load_hwm": h._load_hwm,
        "completed_store_events": dict(h._completed_store_events),
        "pending_store_event_indices": sorted(h._pending_store_event_indices),
        "pending_load_event_indices": sorted(h._pending_load_event_indices),
    }


def _digest_rows(tensors, names, ids, device_gpu):
    import torch

    if not ids:
        return []
    per_region = []
    for n in names:
        t = tensors[n]
        idx = torch.tensor(ids, dtype=torch.long, device=t.device)
        rows = t.index_select(0, idx).cpu()
        per_region.append(rows)
    out = []
    for k in range(len(ids)):
        parts = []
        for rows in per_region:
            parts.append(hashlib.blake2b(rows[k].numpy().tobytes(), digest_size=8).hexdigest())
        out.append("".join(parts))
    return out


def hash_blocks(self, gpu_ids, cpu_ids):
    """Per-block digest (8 B blake2b per region, concatenated) of GPU blocks and of tier blocks."""
    import torch

    h = _handler()
    torch.cuda.synchronize()
    names = sorted(h.gpu_kv_caches)
    res = {"rank": int(getattr(self, "rank", -1)), "n_regions": len(names)}
    res["gpu"] = _digest_rows(h.gpu_kv_caches, names, list(gpu_ids), True)
    res["cpu"] = _digest_rows(h.cpu_kv_caches, names, list(cpu_ids), False) if cpu_ids else []
    return res


# ---------------- D4' tools ----------------
def _u8(t):
    """Reinterpret a region tensor as bytes (no copy)."""
    import torch

    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def digests(tensors, names, ids, chunk=2048):
    """Per block id: bytes = concatenation of one 8-byte blake2b per region (in `names` order), over the raw bytes of the block in that region."""
    import torch

    out = []
    for s in range(0, len(ids), chunk):
        part = ids[s:s + chunk]
        per_region = []
        for n in names:
            t = _u8(tensors[n])
            idx = torch.tensor(part, dtype=torch.long, device=t.device)
            rows = t.index_select(0, idx).cpu().reshape(len(part), -1).numpy()
            per_region.append(rows)
        for k in range(len(part)):
            out.append(b"".join(hashlib.blake2b(rows[k], digest_size=8).digest() for rows in per_region))
    return out


def _rank(self):
    return int(getattr(self, "rank", -1))


def snap_h(self, gpu_ids):
    """Digest the GPU blocks `gpu_ids` (pass H, after the stores drained) and keep them on the worker."""
    import torch

    h = _handler()
    torch.cuda.synchronize()
    names = sorted(h.gpu_kv_caches)
    ids = [int(i) for i in gpu_ids]
    dg = digests(h.gpu_kv_caches, names, ids)
    self._chk_h = dict(zip(ids, dg))   # kept on the worker object: no dependence on how the module is imported in the worker process
    return {"rank": _rank(self), "n": len(ids), "n_regions": len(names)}


def compare_loads(self, gpu_x, cpu_ids, gpu_h):
    """Pass X: pair k = (restored GPU block gpu_x[k], tier block cpu_ids[k], candidate pass-H GPU blocks gpu_h[k] (list, may be empty)).
    Returns the indices k whose GPU copy differs from the tier copy (xc_bad) and from every pass-H candidate (hx_bad; a pair without a candidate is not
    compared with H and counted in n_no_h)."""
    import torch

    h = _handler()
    torch.cuda.synchronize()
    names = sorted(h.gpu_kv_caches)
    dx = digests(h.gpu_kv_caches, names, [int(i) for i in gpu_x])
    dc = digests(h.cpu_kv_caches, names, [int(i) for i in cpu_ids])
    return compare_digests(_rank(self), len(names), dx, dc, gpu_h, getattr(self, "_chk_h", {}))


def compare_digests(rank, n_regions, dx, dc, gpu_h, hdig):
    """Pure part of compare_loads (no torch): the mismatch bookkeeping."""
    xc_bad, hx_bad, n_no_h, n_missing_h, examples = [], [], 0, 0, []
    for k in range(len(dx)):
        if dx[k] != dc[k]:
            xc_bad.append(k)
            if len(examples) < 8:
                examples.append({"k": k, "leg": "xc", "regions": [q for q in range(n_regions) if dx[k][8 * q:8 * q + 8] != dc[k][8 * q:8 * q + 8]][:8]})
        cands = gpu_h[k]
        if not cands:
            n_no_h += 1
            continue
        refs = [hdig.get(int(c)) for c in cands]
        if any(r is None for r in refs):
            n_missing_h += 1
        refs = [r for r in refs if r is not None]
        if refs and dx[k] not in refs:
            hx_bad.append(k)
            if len(examples) < 8:
                examples.append({"k": k, "leg": "hx", "regions": [q for q in range(n_regions) if dx[k][8 * q:8 * q + 8] != refs[0][8 * q:8 * q + 8]][:8]})
    return {"rank": rank, "n": len(dx), "n_regions": n_regions, "xc_bad": xc_bad, "hx_bad": hx_bad, "n_no_h": n_no_h, "n_missing_h": n_missing_h,
            "examples": examples}


def tier_digest(self, cpu_ids):
    """Digests of tier blocks (used to prove the negative control's corruption and its undo)."""
    h = _handler()
    names = sorted(h.gpu_kv_caches)
    return {"rank": _rank(self), "dg": [d.hex() for d in digests(h.cpu_kv_caches, names, [int(i) for i in cpu_ids])]}


def _xor_byte(t, block, region_bytes_off, mask):
    flat = _u8(t)[block].reshape(-1)
    old = int(flat[region_bytes_off])
    flat[region_bytes_off] = old ^ mask
    return old, int(flat[region_bytes_off])


def corrupt_tier(self, rank, cpu_id, region, mask=0x5A):
    """XOR one byte (the middle of the block's slice in region `region`) of tier block `cpu_id`; acts only on worker `rank`. Applying it twice undoes it."""
    if _rank(self) != rank:
        return None
    h = _handler()
    name = sorted(h.gpu_kv_caches)[region]
    t = h.cpu_kv_caches[name]
    off = _u8(t)[cpu_id].numel() // 2
    old, new = _xor_byte(t, cpu_id, off, mask)
    return {"rank": rank, "cpu_id": cpu_id, "region": region, "region_name": name, "offset": off, "old": old, "new": new}


def corrupt_gpu(self, rank, gpu_id, region, mask=0x5A):
    """Same for a GPU block (negative control of the X-vs-tier leg); acts only on worker `rank`."""
    import torch

    if _rank(self) != rank:
        return None
    h = _handler()
    torch.cuda.synchronize()
    name = sorted(h.gpu_kv_caches)[region]
    t = h.gpu_kv_caches[name]
    off = _u8(t)[gpu_id].numel() // 2
    old, new = _xor_byte(t, gpu_id, off, mask)
    torch.cuda.synchronize()
    return {"rank": rank, "gpu_id": gpu_id, "region": region, "region_name": name, "offset": off, "old": old, "new": new}
