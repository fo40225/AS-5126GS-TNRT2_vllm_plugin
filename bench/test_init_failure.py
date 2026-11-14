"""A failure on one rank during TopoComm construction must make every rank raise TopoUnavailable (no hang).

usage: py_topo.sh bench/test_init_failure.py
Rank 3's JIT load raises; all ranks must raise TopoUnavailable within a few seconds and still be able to use NCCL.
"""
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403


def worker(rank, port):
    try:
        dev, cpu_group, _ = init_rank(rank, port)
        from muse_topo import TopoComm, TopoUnavailable, ops

        if rank == 3:
            def boom():
                raise RuntimeError("injected build failure")

            ops.load = boom
        t0 = time.time()
        try:
            TopoComm(cpu_group, dev, rank, WORLD)
            print(f"rank {rank}: constructed?!", flush=True)
            os._exit(2)
        except TopoUnavailable as e:
            print(f"rank {rank}: TopoUnavailable after {time.time() - t0:.1f}s: {str(e)[:100]}", flush=True)
        dist.barrier(cpu_group)  # the gloo group must still be usable
        torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    if rank == 0:
        print("RESULT PASS", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    mp.spawn(worker, args=(29718,), nprocs=WORLD, join=True)
