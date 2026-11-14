"""In-kernel timeline of the decode-size collectives (needs MUSE_TOPO_TRACE=1 at build time: a separate JIT build).

usage: MUSE_TOPO_TRACE=1 py_topo.sh bench/trace_h1a.py [--pdl]
Block 0 / thread 0 stamps %globaltimer at the phase boundaries; stamps are per rank (clocks are not synchronized
across GPUs), so only durations are meaningful. Reported: median over 40 graph replays of the last of 20 calls.
"""
import argparse
import statistics
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

NAMES = {
    "ar2": ["start->a_sent", "a_sent->a_recv", "a_recv->p_sent", "p_sent->p_recv", "p_recv->y_sent", "y_sent->c_recv"],
    "ar1": ["start->a_sent", "a_sent->a_recv", "a_recv->p_sent", "p_sent->p_recv", "p_recv->end"],
    "ag": ["start->pushed", "pushed->partner_fwd", "partner_fwd->all_recv", "all_recv->end"],
}


def worker(rank, args):
    try:
        dev, cpu_group, _ = init_rank(rank, args.port)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD)
        comm.set_pdl(args.pdl)
        comm._c.trace_enable()
        cases = []
        for T, hid in ((1, 8192), (1, 4096), (4, 8192), (16, 8192)):
            x = make_input(rank, (T, hid), 3, dev)
            out = torch.empty_like(x)
            n = x.numel() * 2
            cases.append((f"ar2 [{T},{hid}]", "ar2", lambda x=x, out=out, n=n: comm._c.all_reduce(x, out, 2, max(1, min(128, -(-(n // 64) // 256))), 256, 1)))
            if n <= comm.ar1_max_bytes:
                cases.append((f"ar1 [{T},{hid}]", "ar1", lambda x=x, out=out, n=n: comm._c.all_reduce(x, out, 1, max(1, min(128, -(-(n // 16) // 128))), 128, 1)))
        for T, w in ((1, 512), (1, 1024), (8, 1024)):
            x = make_input(rank, (T, w), 3, dev)
            out = torch.empty((T, 8 * w), dtype=x.dtype, device=dev)
            n = x.numel() * 2
            cases.append((f"ag-ll [{T},{w}] dim-1", "ag", lambda x=x, out=out, n=n: comm._c.all_gather(x, out, 0, 1, max(1, min(128, -(-(n // 4) // 128))), 128, 1)))
        for name, kind, call in cases:
            g = build_graph(call, 20)
            rows = []
            for _ in range(40):
                g.replay()
                torch.cuda.synchronize()
                t = comm._c.trace()
                m = len(NAMES[kind]) + 1
                rows.append([(t[i + 1] - t[i]) / 1000.0 for i in range(m - 1)] + [(t[m - 1] - t[0]) / 1000.0])
            med = [statistics.median(col) for col in zip(*rows)]
            allm = gather_all(med, cpu_group)
            if rank == 0:
                cols = NAMES[kind] + ["total"]
                avg = [statistics.mean(r[i] for r in allm) for i in range(len(cols))]
                mx = [max(r[i] for r in allm) for i in range(len(cols))]
                print(f"{name:24s} " + "  ".join(f"{c}={a:.2f}({m:.2f})" for c, a, m in zip(cols, avg, mx)), flush=True)
            del g
        dist.barrier(cpu_group)
        comm.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdl", action="store_true")
    ap.add_argument("--port", type=int, default=29715)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
