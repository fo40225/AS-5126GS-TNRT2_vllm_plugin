"""Accuracy of the GEMV order of csrc/gemv.cuh against cuBLAS and the exact fp64 product, per batch size (single GPU).

The order of the kernel is the torch emulation of bench/test_gemv_push.py (``emulate_gemv``, bit-exact against the kernel there), with the number of
warps per row of the geometry variant that TopoComm picks by rows (V0 for T = 1, V3 for T >= 2: pooler 2, unpooler 1). cuBLAS is ``F.linear`` on the
same bf16 operands, as the model runs it. Data: the "mixed" kind of the test (outlier-heavy). Prints, per projection and T = 1 .. 8, the mean error in
ulps of the exact value, the largest error and the fraction of correctly rounded elements, averaged (max: largest) over the seeds. Results in
docs/r3b_gemv_push.md.

usage (server, under the GPU lock): GPU=0 scripts/py_topo.sh bench/acc_gemv_sweep.py [--seeds 3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import test_gemv_push as t
from common import err_stats

# warps per output row of the by-rows variants (pooler, unpooler): V0 for one row, V3 for two and more
WPR = {1: (2, 1), 2: (2, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    for kind, name in ((0, "pooler"), (1, "unpooler")):
        K = t.GEOM[kind]["K"]
        w = t.make_weight(0, kind, dev, 1)
        for T in range(1, 9):
            mine, cub = [], []
            for seed in range(args.seeds):
                x = t.make_x(T, K, dev, 400 + 10 * seed + T, "mixed")
                exact = x.double() @ w.double().t()
                mine.append(err_stats(t.emulate_gemv(x, w, WPR[min(T, 2)][kind]), exact))
                cub.append(err_stats(F.linear(x, w), exact))

            def fmt(rows):
                mean = sum(r["mean_ulp"] for r in rows) / len(rows)
                cr = sum(r["correctly_rounded_pct"] for r in rows) / len(rows)
                return f"mean {mean:.3f} ulp, max {max(r['max_ulp'] for r in rows):.1f} ulp, {cr:.3f} % correctly rounded"

            print(f"{name} T={T}: gemv.cuh {fmt(mine)} | cuBLAS {fmt(cub)}", flush=True)


if __name__ == "__main__":
    main()
