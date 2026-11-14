"""Shared helpers of the muse_topo standalone tests and benchmarks (8 processes, one per GPU)."""
import hashlib
import os
import statistics
import time

import torch
import torch.distributed as dist

WORLD = 8
MiB = 1 << 20


def init_rank(rank, port, nccl=False, watchdog_s=0):
    import faulthandler

    if watchdog_s:
        faulthandler.dump_traceback_later(watchdog_s, exit=True)  # dump stacks and exit if a rank hangs
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    if nccl:
        dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=WORLD, device_id=dev)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    cpu_group = dist.new_group(backend="gloo")
    return dev, cpu_group


def log(rank, *a):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_input(rank, shape, seed, dev, dtype=torch.bfloat16, kind="randn"):
    """Deterministic per-rank input. kinds: randn | realish (per-column scales, outliers) | zeros (+-0 mixes)."""
    g = torch.Generator(device=dev)
    g.manual_seed((seed * 1315423911 + rank * 7919 + 17) % (1 << 62))
    x = torch.randn(shape, device=dev, dtype=torch.float32, generator=g)
    if kind == "realish":
        cols = shape[-1]
        sc = torch.exp2(torch.randint(-6, 6, (cols,), device=dev, generator=g).float())
        x = x * sc
        mask = torch.rand(shape, device=dev, generator=g) < 0.002
        x = torch.where(mask, x * 300.0, x)
    elif kind == "zeros":
        # +0 / -0 mixes by column class: all -0, all +0, rank-dependent +-0, exactly cancelling values
        cls = (torch.arange(shape[-1], device=dev) % 4).expand(shape)
        pz = torch.zeros(shape, device=dev, dtype=torch.float32)
        nz = torch.full(shape, -0.0, device=dev, dtype=torch.float32)
        canc = torch.full(shape, 1.5 if rank % 2 == 0 else -1.5, device=dev, dtype=torch.float32)
        x = torch.where(cls == 0, nz, torch.where(cls == 1, pz, torch.where(cls == 2, nz if rank % 2 else pz, canc)))
    return x.to(dtype)


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32).reshape(-1)


def bits_equal(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def digest(t):
    return hashlib.sha1(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def gather_obj(v, cpu_group):
    out = [None] * WORLD
    dist.all_gather_object(out, v, group=cpu_group)
    return out


def contract_ref(xs, dtype):
    """The arithmetic contract in torch: P_I = ((x0+x1)+x2)+x3 in fp32, out = round(P_0 + P_1)."""
    f = [x.float() for x in xs]
    p0 = ((f[0] + f[1]) + f[2]) + f[3]
    p1 = ((f[4] + f[5]) + f[6]) + f[7]
    return (p0 + p1).to(dtype)


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
    """Error of y against the fp64 exact sum in ulps of y's dtype, plus the fraction that is correctly rounded."""
    yd = y.double()
    nz = s64 != 0
    mant = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}[y.dtype]
    ex = torch.floor(torch.log2(s64.abs().clamp_min(1e-300)))
    ulp = torch.exp2(ex - mant)
    e = ((yd - s64).abs() / ulp)[nz]
    ref = round_single(s64, y.dtype)  # single rounding of the exact sum (see round_single)
    return {
        "mean_ulp": round(e.mean().item(), 4),
        "max_ulp": round(e.max().item(), 3),
        "rn_exact_pct": round(100 * (y == ref).double().mean().item(), 3),
    }


def exact_sum(xs):
    s = torch.zeros(xs[0].shape, dtype=torch.float64, device=xs[0].device)
    for x in xs:
        s += x.double()
    return s


def time_graph(g, ncalls, cpu_group, target_s=0.05, trials=5, max_reps=400):
    """Median/min per-call time (us) of a graph holding ncalls collectives, replayed back to back."""
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(cpu_group)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    one = max(gather_obj(s.elapsed_time(e) / 1e3, cpu_group))
    reps = int(max(1, min(max_reps, target_s / max(one, 1e-6))))
    reps = min(gather_obj(reps, cpu_group))
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
    med = statistics.median(per)
    allmed = gather_obj(med, cpu_group)
    return {"us": round(max(allmed), 3), "us_rank0": round(med, 3), "us_min": round(min(per), 3)}
