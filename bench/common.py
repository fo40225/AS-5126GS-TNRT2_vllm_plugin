"""Shared helpers of the muse_topo standalone tests and benchmarks (8 processes, one per GPU, no vLLM model)."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

WORLD = 8
MiB = 1 << 20


def init_rank(rank: int, port: int, nccl: bool = False):
    """gloo CPU group (+ optional vLLM PyNcclCommunicator on the same group, as vLLM's device communicator)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    wd = int(os.environ.get("BENCH_WATCHDOG_S", "0"))
    if wd > 0:
        import faulthandler

        faulthandler.dump_traceback_later(wd, exit=True)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    cpu_group = dist.new_group(backend="gloo")
    pynccl = None
    if nccl:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        pynccl = PyNcclCommunicator(group=cpu_group, device=dev)
    return dev, cpu_group, pynccl


def log(rank, *a):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# data generators (deterministic in (kind, shape, seed, rank))
# ---------------------------------------------------------------------------------------------------------------
def make_input(rank, shape, seed, dev, dtype=torch.bfloat16, kind="randn"):
    g = torch.Generator(device=dev)
    g.manual_seed((seed * 1315423911 + rank * 7919 + 17) % (1 << 62))
    x = torch.randn(shape, device=dev, dtype=torch.float32, generator=g)
    if kind == "mixed":
        # real-ish activations: per-column scale 2^[-6,6] (outlier channels), sparse large outliers, some tiny values
        cols = shape[-1]
        e = torch.randint(-6, 7, (cols,), device=dev, generator=g).float()
        x = x * torch.exp2(e)
        m = torch.rand(shape, device=dev, generator=g) < 1e-3
        x = torch.where(m, x * 64.0, x)
    elif kind == "zeros":
        # signed zeros and zeros mixed with values; exercises sign of zero in the fp32 chain
        s = torch.rand(shape, device=dev, generator=g)
        x = torch.where(s < 0.5, torch.zeros_like(x), torch.where(s < 0.75, -torch.zeros_like(x), x))
    return x.to(dtype)


def contract_ref(xs, members):
    """The arithmetic contract in torch (IEEE fp32 adds, one final rounding)."""
    dt = xs[0].dtype
    ps = []
    for isl in members:
        acc = xs[isl[0]].float()
        for r in isl[1:]:
            acc = acc + xs[r].float()
        ps.append(acc)
    out = (ps[0] + ps[1]).to(dt)
    return canon_words(out)


def canon_words(t):
    """Words equal to 0xFFFFFFFF become 0xFFFFFFFE (the sentinel canonicalization; only affects NaN payloads)."""
    if t.element_size() == 4:
        v = t.view(torch.int32)
        return torch.where(v == -1, torch.full_like(v, -2), v).view(t.dtype)
    v = t.view(torch.int16).reshape(-1, 2) if t.numel() % 2 == 0 else None
    if v is None:
        return t
    bad = (v[:, 0] == -1) & (v[:, 1] == -1)
    v = v.clone()
    v[bad, 0] = -2
    return v.reshape(t.shape).view(t.dtype)


def bits(t):
    return t.contiguous().view(torch.uint8)


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(bits(a), bits(b)))


def digest(t):
    return hashlib.sha256(bits(t).cpu().numpy().tobytes()).hexdigest()[:16]


def all_equal_across_ranks(t, cpu_group):
    d = [None] * WORLD
    dist.all_gather_object(d, digest(t), group=cpu_group)
    return all(x == d[0] for x in d)


def gather_all(vals, cpu_group):
    out = [None] * WORLD
    dist.all_gather_object(out, vals, group=cpu_group)
    return out


# ---------------------------------------------------------------------------------------------------------------
# accuracy vs exact fp64 sum
# ---------------------------------------------------------------------------------------------------------------
def exact_sum(xs):
    s = torch.zeros(xs[0].shape, dtype=torch.float64, device=xs[0].device)
    for x in xs:
        s += x.double()
    return s


def round_single(s64, dtype):
    """float64 -> dtype with ONE round-to-nearest-even. (torch converts float64 to bf16/fp16 through float32, i.e. it
    rounds twice, which can differ from the correctly rounded value when the float32 intermediate is a halfway case.)
    Normal range of the target dtype: integer rounding of the float64 mantissa; below it (subnormals) and for
    non-finite values the plain torch conversion is used."""
    if dtype == torch.float32:
        return s64.to(dtype)
    mant, tiny = {torch.bfloat16: (7, 2.0**-126), torch.float16: (10, 2.0**-14)}[dtype]
    drop = 52 - mant
    a = s64.abs()
    mag = a.view(torch.int64)
    lsb = (mag >> drop) & 1
    mag = (mag + ((1 << (drop - 1)) - 1) + lsb) & ~((1 << drop) - 1)  # carries into the exponent when needed
    r = torch.copysign(mag.view(torch.float64), s64)
    use = torch.isfinite(s64) & (a >= tiny)
    return torch.where(use, r, s64).to(dtype)


def err_stats(y, s64):
    """Error of the low-precision result y vs the exact fp64 sum, in ulps of the exact value (bf16 ulp = 2^-7 rel)."""
    mant = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}[y.dtype]
    yd = y.double()
    nz = s64 != 0
    ex = torch.floor(torch.log2(s64.abs().clamp_min(1e-300)))
    ulp = torch.exp2(ex - mant)
    e = ((yd - s64).abs() / ulp)[nz]
    ref = round_single(s64, y.dtype)  # single rounding of the exact sum (see round_single)
    return {
        "mean_ulp": round(e.mean().item(), 4),
        "max_ulp": round(e.max().item(), 3),
        "correctly_rounded_pct": round(100 * (y == ref).double().mean().item(), 3),
    }


# ---------------------------------------------------------------------------------------------------------------
# timing: CUDA graph of N back-to-back calls (optionally with a spacer kernel before each)
# ---------------------------------------------------------------------------------------------------------------
def spin_cycles_for_us(us):
    """torch.cuda._sleep takes SM clock cycles; torch.cuda.clock_rate() is in MHz (= cycles per microsecond)."""
    return int(us * torch.cuda.clock_rate())


def build_graph(call, n, spacer_cycles=0):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            if spacer_cycles:
                torch.cuda._sleep(spacer_cycles)
            call()
    return g


def time_graph(g, ncalls, cpu_group, target_s=0.05, trials=5, max_reps=400):
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(cpu_group)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    one = max(gather_all(s.elapsed_time(e) / 1e3, cpu_group))
    reps = int(max(1, min(max_reps, target_s / max(one, 1e-6))))
    reps = min(gather_all(reps, cpu_group))
    per = []
    for _ in range(trials):
        torch.cuda.synchronize()
        dist.barrier(cpu_group)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            g.replay()
        e.record()
        e.synchronize()
        per.append(s.elapsed_time(e) * 1e3 / (reps * ncalls))
    return per


def med_over_ranks(per, cpu_group):
    """Median over trials on each rank, then the max over ranks (the slowest rank defines the collective)."""
    return max(gather_all(statistics.median(per), cpu_group))


def append_jsonl(path, rows, rank):
    if rank != 0:
        return
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
