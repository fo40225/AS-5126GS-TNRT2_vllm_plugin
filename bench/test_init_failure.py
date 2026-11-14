"""A failure on one rank during TopoComm construction must make every rank raise TopoUnavailable (no hang).

usage: py_topo.sh bench/test_init_failure.py [--cases build,tuning]
Case "build": rank 3's JIT load raises. Case "tuning": rank 3 gets an unreadable MUSE_TOPO_AR_TUNING_FILE and rank 5 a
bad MUSE_TOPO_MEDIUM_BX_MIN_BYTES (the parse happens before the first cross-rank step and is reported through the
stage-1 agreement). In both cases all ranks must raise TopoUnavailable within a few seconds and still be able to use
NCCL.
"""
import argparse
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403


def worker(rank, port, case):
    try:
        dev, cpu_group, _ = init_rank(rank, port)
        from muse_topo import TopoComm, TopoUnavailable, ops

        if case == "build" and rank == 3:
            def boom():
                raise RuntimeError("injected build failure")

            ops.load = boom
        elif case == "tuning" and rank == 3:
            os.environ["MUSE_TOPO_AR_TUNING_FILE"] = "/nonexistent/muse_topo_tuning.json"
        elif case == "tuning" and rank == 5:
            os.environ["MUSE_TOPO_MEDIUM_BX_MIN_BYTES"] = "64K"
        t0 = time.time()
        try:
            TopoComm(cpu_group, dev, rank, WORLD)
            print(f"rank {rank}: constructed?!", flush=True)
            os._exit(2)
        except TopoUnavailable as e:
            print(f"rank {rank}: TopoUnavailable after {time.time() - t0:.1f}s: {str(e)[:100]}", flush=True)
            if case == "tuning" and "tuning" not in str(e):
                print(f"rank {rank}: reason does not name the tuning failure", flush=True)
                os._exit(3)
        dist.barrier(cpu_group)  # the gloo group must still be usable
        torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    if rank == 0:
        print(f"RESULT PASS ({case})", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="build,tuning")
    args = ap.parse_args()
    for i, case in enumerate(args.cases.split(",")):
        mp.spawn(worker, args=(29718 + i, case), nprocs=WORLD, join=True)
