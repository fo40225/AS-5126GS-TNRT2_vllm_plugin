"""Smoke test: forced algorithms on a few shapes against the contract reference. usage: py_topo.sh bench/smoke.py"""
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403


def worker(rank, port):
    try:
        dev, cpu_group, _ = init_rank(rank, port)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD)
        log(rank, "TopoComm up", comm.info.summary, f"workspace {comm.workspace_bytes / 2**20:.1f} MiB")
        members = comm.info.members
        bad = 0
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for shape in ((1, 8192), (4, 8192), (2, 4096), (16, 8192), (64, 8192)):
                for kind in ("randn", "mixed", "zeros"):
                    xs = [make_input(r, shape, 11, dev, dtype, kind) for r in range(WORLD)]
                    ref = contract_ref(xs, members)
                    x = xs[rank]
                    nb = x.numel() * x.element_size()
                    for algo in (1, 2, 3):
                        if algo == 1 and nb > comm.ar1_max_bytes:
                            continue
                        y = comm._all_reduce(x, algo, 128 if algo != 3 else 256, 1)
                        torch.cuda.synchronize()
                        ok = bits_equal(y, ref)
                        same = all_equal_across_ranks(y, cpu_group)
                        if not (ok and same):
                            bad += 1
                            log(rank, f"FAIL AR algo{algo} {dtype} {shape} {kind} ok={ok} same={same}")
        log(rank, f"AR forced-algo checks done, failures={bad}")
        for shape, dim in (((1, 512), -1), ((1, 1024), -1), ((8, 1024), -1), ((3, 512), 0), ((8, 1024), 0)):
            xs = [make_input(r, shape, 5, dev) for r in range(WORLD)]
            ref = torch.cat(xs, dim=dim if dim != -1 else len(shape) - 1)
            for algo in (1, 2):
                if algo == 2 and (xs[rank].numel() * 2) % 16:
                    continue
                y = comm._all_gather(xs[rank], len(shape) - 1 if dim == -1 else 0, 0 if dim == -1 else 1, algo,
                                     128 if algo == 1 else 256, 1)
                torch.cuda.synchronize()
                ok = bits_equal(y, ref)
                if not ok:
                    bad += 1
                log(rank, f"AG algo{algo} {shape} dim={dim} ok={ok}")
        log(rank, "SMOKE", "PASS" if bad == 0 else f"FAIL({bad})")
        torch.cuda.synchronize()
        comm.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    mp.spawn(worker, args=(29711,), nprocs=WORLD, join=True)
