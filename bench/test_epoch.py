"""Regression test of the 64-bit call counter / flags of the muse_topo kernels (review finding: 32-bit signed epoch).

usage: py_topo.sh bench/test_epoch.py [--quick]      (run under the GPU lock; ~1-2 min)
Every kernel launch advances a per-GPU call counter that selects the double-buffer parity and the flag / LL8 tags.
The counter used to be 32 bit with a signed compare: at the 2^31-th launch the flag kernels waited for a flag above
INT_MAX (spin timeout, trap) or accepted stale flags (silent wrong data). The counter and the flags are 64 bit now;
this test forces the counter (test hook Comm.set_epoch, only forward, only while idle) to the values around the old
failure points and checks every result bitwise.

  A. fresh communicators (every flag column and LL8 slot is pristine, i.e. all-ones) at 0x7fffffff, 0x80000000, 0x90000000,
     0xffffffff, 0x1ffffffff, 2^62 and 0x1fffffffc / 0x1fffffffd with the LL8 all-gather first (a tag of 0xFFFFFFFF would
     equal the pristine all-ones slot); rank 0 joins 30 ms late so that every receiver polls before its senders stored: flag all-gather, LL8 all-gather, sentinel
     all-reduce, large all-reduce (flag all-gather role), forced flag all-reduce.
  B. one communicator, mixed ops with random per-rank delays, counter moved forward across 2^31, 2^32, a high 64-bit value
     and the LL8 tag wrap (2^33), 30 rounds of 6 ops each: any wrong bit or trap fails the test.
A regression traps the CUDA context (spin timeout 20 s), which this script reports as a failure of the whole run.
CSRC_DIR=<kernel tree> builds and tests another tree (negative control: it needs the Comm.set_epoch / get_epoch hooks).
"""
import argparse
import os
import random
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

FAILS = []
# A fresh LL8 slot holds the all-ones word; with tag = (e >> 1) + 1 mod 2^32 the launches 0x1fffffffc / 0x1fffffffd would
# use the tag 0xFFFFFFFF and accept the pristine slot of a late sender as data
LL8_FIRST = (0x1FFFFFFFC, 0x1FFFFFFFD)


def delay(prng):
    r = prng.random()
    if r < 0.4:
        return
    us = prng.uniform(5, 60) if r < 0.8 else prng.uniform(100, 600)
    torch.cuda._sleep(int(us * torch.cuda.clock_rate()))


def op_list(comm, rank, dev, members, seed):
    """(name, call, reference) for six ops that cover the flag, LL8 and sentinel protocols and the large engine."""
    bf = torch.bfloat16
    out = []
    xs = [make_input(r, (32, 512), seed, dev, bf) for r in range(WORLD)]
    out.append(("ag flag 32KiB dim0", lambda: comm.all_gather(xs[rank], 0), torch.cat(xs, 0)))
    xs2 = [make_input(r, (300, 512), seed + 7, dev, bf) for r in range(WORLD)]
    out.append(("ag large 300KiB dim-1", lambda: comm.all_gather(xs2[rank], 1), torch.cat(xs2, 1)))
    xs3 = [make_input(r, (128, 8192), seed + 13, dev, bf) for r in range(WORLD)]
    out.append(("ar large 2MiB", lambda: comm.all_reduce(xs3[rank]), contract_ref(xs3, members)))
    xs4 = [make_input(r, (16, 4096), seed + 17, dev, bf) for r in range(WORLD)]
    out.append(("ar algo3 flag 128KiB", lambda: comm._all_reduce(xs4[rank], 3, 256, 1), contract_ref(xs4, members)))
    xs5 = [make_input(r, (1, 512), seed + 19, dev, bf) for r in range(WORLD)]
    out.append(("ag LL8 1KiB", lambda: comm.all_gather(xs5[rank], 0), torch.cat(xs5, 0)))
    xs6 = [make_input(r, (4, 8192), seed + 23, dev, bf) for r in range(WORLD)]
    out.append(("ar sentinel 64KiB", lambda: comm.all_reduce(xs6[rank]), contract_ref(xs6, members)))
    return out


def sync_all(cpu_group):
    torch.cuda.synchronize()
    dist.barrier(cpu_group)


def phase_a(rank, dev, cpu_group, epochs):
    from muse_topo import TopoComm

    for ep in epochs:
        comm = TopoComm(cpu_group, dev, rank, WORLD, timeout_s=20.0)
        comm._c.set_epoch(ep)
        sync_all(cpu_group)
        if rank == 0:
            torch.cuda._sleep(int(30e3 * torch.cuda.clock_rate()))  # rank 0 joins 30 ms late
        nbad = 0
        ops = op_list(comm, rank, dev, comm.info.members, 11)
        if ep in LL8_FIRST:  # the LL8 all-gather must be the first launch (its tag is the one under test)
            ops.sort(key=lambda o: o[0] != "ag LL8 1KiB")
        for name, fn, ref in ops:
            y = fn()
            torch.cuda.synchronize()
            if not bits_equal(y, ref):
                nbad += 1
                print(f"FAIL[r{rank}] fresh comm at epoch {ep:#x}: {name}", flush=True)
        got = comm._c.get_epoch()
        allbad = sum(gather_all(nbad, cpu_group))
        log(rank, f"A fresh communicator, counter {ep:#x} -> {got:#x} (6 ops, 7 launches): bad results over all ranks {allbad}")
        FAILS.append(allbad)
        if not ep + 6 <= got <= ep + 16:  # one launch per op, the large ops use 1-2 launches
            print(f"FAIL[r{rank}] counter {got:#x} after the 6 ops does not follow {ep:#x} by one per launch", flush=True)
            FAILS.append(1)
        sync_all(cpu_group)
        comm.close()
        dist.barrier(cpu_group)


def phase_b(rank, dev, cpu_group, starts, rounds):
    from muse_topo import TopoComm

    comm = TopoComm(cpu_group, dev, rank, WORLD, timeout_s=20.0)
    members = comm.info.members
    prng = random.Random(1234 + rank)
    for label, start in starts:
        sync_all(cpu_group)
        comm._c.set_epoch(start)
        sync_all(cpu_group)
        nbad = 0
        for call in range(rounds):
            pend = []
            for name, fn, ref in op_list(comm, rank, dev, members, 1000 + call):
                delay(prng)
                pend.append((name, fn(), ref))
            torch.cuda.synchronize()
            for name, y, ref in pend:
                if not bits_equal(y, ref):
                    nbad += 1
                    print(f"FAIL[r{rank}] B {label} round {call} {name} (counter {comm._c.get_epoch():#x})", flush=True)
        allbad = sum(gather_all(nbad, cpu_group))
        log(rank, f"B {label}: counter {start:#x} -> {comm._c.get_epoch():#x}, bad results over all ranks {allbad}")
        FAILS.append(allbad)
    sync_all(cpu_group)
    comm.close()


def run(rank, args):
    if os.environ.get("CSRC_DIR"):  # negative control: build and test another kernel tree (e.g. the former 32-bit one)
        from pathlib import Path

        import muse_topo.ops as ops

        ops._CSRC_DIR = Path(os.environ["CSRC_DIR"])
    dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
    ea = [0x7FFFFFFF, 0x80000000, 0x90000000, 0xFFFFFFFF, 0x1FFFFFFFC, 0x1FFFFFFFD, 0x1FFFFFFFF, 1 << 62]
    sb = [("control (low counter)", 100), ("cross 2^31", 0x7FFFFFFF - 40), ("cross 2^32", 0xFFFFFFFF - 40),
          ("high 0x1_9000_0000", 0x190000000), ("cross 2^33 (LL8 tag wrap)", 0x1FFFFFFFF - 40)]
    if args.quick:
        ea, sb = [ea[0], ea[2], ea[4]], sb[:2]
    phase_a(rank, dev, cpu_group, ea)
    phase_b(rank, dev, cpu_group, sb, 10 if args.quick else 30)
    tot = sum(FAILS)
    alltot = sum(gather_all(tot, cpu_group))
    log(rank, "EPOCH TEST RESULT", "PASS" if alltot == 0 else f"FAIL total={alltot}")
    sys.stdout.flush()
    os._exit(0 if alltot == 0 else 1)


def worker(rank, args):
    try:
        run(rank, args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--port", type=int, default=29873)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
