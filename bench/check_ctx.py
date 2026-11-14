"""Check that TopoComm does not create CUDA contexts on the other 7 GPUs (per-process GPU memory via nvidia-smi).
usage: py_topo.sh bench/check_ctx.py"""
import subprocess
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403


def worker(rank, port):
    try:
        dev, cpu_group, _ = init_rank(rank, port)
        from muse_topo import TopoComm

        torch.zeros(1, device=dev)
        torch.cuda.synchronize()
        dist.barrier(cpu_group)
        before = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout
        dist.barrier(cpu_group)
        comm = TopoComm(cpu_group, dev, rank, WORLD)
        y = comm.all_reduce(make_input(rank, (4, 8192), 1, dev))
        torch.cuda.synchronize()
        dist.barrier(cpu_group)
        after = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout
        if rank == 0:
            mine = lambda txt: sorted(l for l in txt.strip().splitlines() if l)
            nb, na = len(mine(before)), len(mine(after))
            print("compute-app entries before TopoComm:", nb, "after:", na, flush=True)
            print("after:\n" + after, flush=True)
            print("RESULT", "PASS" if na == nb else "FAIL (extra contexts)", flush=True)
        dist.barrier(cpu_group)
        comm.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    mp.spawn(worker, args=(29716,), nprocs=WORLD, join=True)
