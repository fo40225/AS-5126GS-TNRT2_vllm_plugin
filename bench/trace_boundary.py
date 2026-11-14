"""In-kernel timeline of the fused collective + epilogue kernels (needs MUSE_TOPO_TRACE=1 at build time: a separate JIT build).

usage: MUSE_TOPO_TRACE=1 py_topo.sh bench/trace_boundary.py [--cold] [--ts 1,4,8]
Block 0 / thread 0 stamps %globaltimer at the phase boundaries; stamps are per rank (clocks are not synchronized across GPUs), so
only durations are meaningful. Reported: median over 40 graph replays of the last of 20 calls; mean (and max) over ranks.
--cold: a 256 MiB copy runs before every call, so the operands of the epilogue come from HBM.
"""
import argparse
import statistics
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

D = 8192
AR_STAMPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
AR_NAMES = ["start->a_sent", "a_sent->a_recv", "a_recv->p_sent", "p_sent->p_recv", "p_recv->y_sent", "y_sent->c_recv", "c_recv->tree1",
            "tree1->new", "new->tree2", "tree2->end"]
AG_STAMPS = [0, 1, 2, 3, 7, 8, 9, 10]
AG_NAMES = ["start->pushed", "pushed->partner_fwd", "partner_fwd->all_recv", "all_recv->tree1", "tree1->new", "new->tree2", "tree2->end"]


def worker(rank, args):
    try:
        dev, cpu_group, _ = init_rank(rank, args.port)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD)
        comm._c.trace_enable()
        g = torch.Generator(device=dev)
        g.manual_seed(5)
        alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
        beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
        post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
        pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
        big_a = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        big_b = torch.randn(64 << 20, device=dev)
        cases = []
        for T in args.ts:
            x = make_input(rank, (T, D), 3, dev, torch.bfloat16, "mixed")
            stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
            shard = make_input(rank, (T, 1024), 4, dev, torch.bfloat16, "mixed")
            cases.append((f"ar+boundary T={T}", AR_STAMPS, AR_NAMES,
                          lambda x=x, stream=stream: comm.all_reduce_boundary(x, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)))
            if T <= 32:
                cases.append((f"ag+boundary T={T}", AG_STAMPS, AG_NAMES,
                              lambda shard=shard, stream=stream: comm.all_gather_boundary(shard, stream, alpha, beta, pre_w, post_w,
                                                                                         1e-8, 1e-6, 1.0, 1.0)))
        for name, stamps, names, call in cases:
            def step(call=call):
                if args.cold:
                    big_a.copy_(big_b)
                call()

            gr = build_graph(step, 20)
            rows = []
            for _ in range(40):
                gr.replay()
                torch.cuda.synchronize()
                t = comm._c.trace()
                ts = [t[i] for i in stamps]
                rows.append([(ts[i + 1] - ts[i]) / 1000.0 for i in range(len(ts) - 1)] + [(ts[-1] - ts[0]) / 1000.0])
            med = [statistics.median(col) for col in zip(*rows)]
            allm = gather_all(med, cpu_group)
            if rank == 0:
                cols = names + ["total"]
                avg = [statistics.mean(r[i] for r in allm) for i in range(len(cols))]
                mx = [max(r[i] for r in allm) for i in range(len(cols))]
                print(f"{name:20s} " + "  ".join(f"{c}={a:.2f}({m:.2f})" for c, a, m in zip(cols, avg, mx)), flush=True)
            del gr
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
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--ts", type=lambda s: [int(v) for v in s.split(",")], default=[1, 4, 8])
    ap.add_argument("--port", type=int, default=29771)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
