"""Regression test: when one rank skips a LARGE collective the others must abort with an error instead of hanging.

usage: py_topo.sh bench/test_timeout_large.py
Rank 3 never calls the large all-reduce / reduce-scatter / all-gather; every other rank calls them one after another
(each in a fresh process) with a 3 s spin timeout and must die (exit code 3) well before 30 s. Prints RESULT PASS/FAIL.
"""
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403


def worker(rank, port, which):
    try:
        dev, cpu_group, _ = init_rank(rank, port)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD, timeout_s=3.0)
        x = make_input(rank, (2048, 4096), 1, dev)  # 16 MiB bf16
        y = comm.all_reduce(x)  # warm-up that everybody makes (large path)
        torch.cuda.synchronize()
        dist.barrier(cpu_group)
        if rank == 3:
            time.sleep(15)
            os._exit(0)
        t0 = time.time()
        try:
            if which == "ar":
                comm.all_reduce(x)
            elif which == "rs":
                comm.reduce_scatter(x)
            else:
                comm.all_gather(make_input(rank, (2048, 512), 2, dev), -1)
            torch.cuda.synchronize()
            print(f"rank {rank}: {which} returned without the missing rank?!", flush=True)
            os._exit(0)
        except Exception as e:
            print(f"rank {rank}: {which} aborted after {time.time() - t0:.1f}s, comm.error()={comm.error():#x}: {str(e)[:80]}",
                  flush=True)
            os._exit(3)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    allok = True
    for which, port in (("ar", 29741), ("rs", 29742), ("ag", 29743)):
        ctx = mp.start_processes(worker, args=(port, which), nprocs=WORLD, join=False, start_method="spawn")
        t0 = time.time()
        while time.time() - t0 < 60 and any(p.exitcode is None for p in ctx.processes):
            time.sleep(0.5)
        codes = [p.exitcode for p in ctx.processes]
        for p in ctx.processes:
            if p.exitcode is None:
                p.kill()
        el = time.time() - t0
        ok = all(c == 3 for r, c in enumerate(codes) if r != 3) and codes[3] == 0 and el < 30
        print(f"[{which}] exit codes {codes} after {el:.1f}s -> {'PASS' if ok else 'FAIL'}", flush=True)
        allok &= ok
        time.sleep(5)
    print("RESULT", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)
