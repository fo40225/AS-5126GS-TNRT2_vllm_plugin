#!/usr/bin/env python3
"""Node of the pages of the unlinked /dev/shm/vllm_offload_* mapping of one worker, sampled with move_pages(2).

usage: region_nodes.py <pid> [--stride N] [--head PAGES]
 * head: the first PAGES pages (default 8192 = 32 MiB) one by one, printed as run-length encoded node sequence. The mmap
   rows interleave the 8 workers' slots, so with rank-local placement the runs follow GPU order 3,3,0,0,7,7,4,4 (two slots
   per node) within every row, repeated.
 * stride: every Nth page over the whole mapping (default 64), printed as a histogram per node.
move_pages without a target node only reads the node (status); it never migrates anything. Needs ptrace access to <pid>.
"""
import ctypes
import re
import sys

PAGE = 4096


def find_mapping(pid):
    with open("/proc/%s/maps" % pid) as f:
        for line in f:
            if "vllm_offload" in line:
                a, b = line.split()[0].split("-")
                return int(a, 16), int(b, 16)
    return None


def query(libnuma, pid, addrs):
    n = len(addrs)
    pages = (ctypes.c_void_p * n)(*addrs)
    status = (ctypes.c_int * n)()
    rc = libnuma.move_pages(ctypes.c_int(pid), ctypes.c_ulong(n), pages, None, status, ctypes.c_int(0))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "move_pages failed rc=%d" % rc)
    return list(status)


def main():
    pid = int(sys.argv[1])
    stride = 64
    head = 8192
    if "--stride" in sys.argv:
        stride = int(sys.argv[sys.argv.index("--stride") + 1])
    if "--head" in sys.argv:
        head = int(sys.argv[sys.argv.index("--head") + 1])
    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    m = find_mapping(pid)
    if m is None:
        print("no vllm_offload mapping in pid %d" % pid)
        return
    start, end = m
    npages = (end - start) // PAGE
    print("pid %d mapping %x-%x %.2f GiB (%d pages)" % (pid, start, end, (end - start) / 2**30, npages))
    # head, one by one in batches
    seq = []
    for i in range(0, min(head, npages), 2048):
        seq += query(libnuma, pid, [start + (i + j) * PAGE for j in range(min(2048, head - i, npages - i))])
    runs = []
    for s in seq:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    print("head %d pages as runs node:pages (first 80 runs of %d): %s" % (
        len(seq), len(runs), " ".join("%d:%d" % (n, c) for n, c in runs[:80])))
    # owner check: with S110_BPB (bytes per block-id) and S110_BPC (blocks per chunk) the layout is known: row = 8 rank cells of bpb x bpc bytes,
    # rank r is on node NODE_OF[r]; a prime page stride avoids the aliasing of the plain strided sample below
    import os
    if os.environ.get("S110_BPB"):
        cell = int(os.environ["S110_BPB"]) * int(os.environ.get("S110_BPC", "1"))
        row = 8 * cell
        node_of = [3, 3, 0, 0, 7, 7, 4, 4]
        step = 61
        idx = list(range(0, npages, step))
        got = []
        for i in range(0, len(idx), 4096):
            got += query(libnuma, pid, [start + k * PAGE for k in idx[i:i + 4096]])
        ok = 0
        bad = {}
        for k, g in zip(idx, got):
            r = ((k * PAGE) % row) // cell
            if g == node_of[r]:
                ok += 1
            else:
                bad[(r, g)] = bad.get((r, g), 0) + 1
        print("owner check (every %d-th page, %d samples): on the owner's node %.2f %%, elsewhere %d; (rank, node) of the misplaced: %s" % (
            step, len(idx), 100.0 * ok / len(idx), len(idx) - ok, sorted(bad.items(), key=lambda x: -x[1])[:8]))
    # strided sample over everything
    hist = {}
    for i in range(0, npages, stride * 2048):
        idx = range(i, min(i + stride * 2048, npages), stride)
        for s in query(libnuma, pid, [start + k * PAGE for k in idx]):
            hist[s] = hist.get(s, 0) + 1
    tot = sum(hist.values())
    print("sample every %d pages (%d samples): %s" % (stride, tot, " ".join(
        "node%d=%.1f%%" % (k, 100.0 * v / tot) if k >= 0 else "err%d=%.1f%%" % (k, 100.0 * v / tot)
        for k, v in sorted(hist.items()))))


if __name__ == "__main__":
    main()
