# H1a latency table (decode-size collectives)

> Note (review fix, 2026-10-24): the counter and flags are 64 bit now. Re-measured with the same protocol: all_reduce T <= 16 within +0.4 %, LL8 all_gather (pooler / unpooler shards, T <= 16) about +0.1 us b2b, everything else unchanged within noise; see `docs/h2_vllm_plugin.md`, "Review fixes".
>
> Note (H1b): from 2 MiB on the all_reduce dispatch uses the large algorithm (fused reduce-scatter + all-gather); the rows of this
> table at >= 2 MiB (alg2) are the H1a path, see `docs/h1b_latency.md` for the current dispatch numbers (e.g. 8 MiB: 454 us instead of 480 us).

Measured 2026-10-23 on the AS-5126GS-TNRT2 box (8x RTX PRO 6000 Blackwell), 8 processes, `bench/bench_h1a.py --public`
(the dispatch path `comm.all_reduce(x, out=out)` / `comm.all_gather(x, dim, out=out)`, results verified bitwise against
the arithmetic contract / `torch.cat` in the same run; `scripts/run_h1a.sh <tag> bench_public`). Protocol: CUDA graph of
N identical calls (N = 100, 20 for >= 1 MiB, 6 for >= 8 MiB), replayed; per-call time = replay time / N, median of 5
trials, max over ranks. `b2b` = back-to-back identical calls; `spacer` = (50 us sleep kernel, call) pairs minus the
sleep-only time = the time a collective exposes behind a compute kernel. A null kernel costs 0.69 us per node in the
same graphs. Baseline: vLLM's PyNcclCommunicator on the same gloo group (NCCL 2.29.7, `NCCL_P2P_LEVEL=SYS`, default
algorithm/protocol selection). Input: randn bf16. The `[algN x]` brackets list the best time of each algorithm of the
sweep (here only the dispatched one). all_reduce algorithms: 2 = 3-phase sentinel (1 = 2-phase sentinel and 3 = flag
never won). all_gather algorithms: 1 = LL8, 2 = flag protocol. The all_gather NCCL baseline gathers [T,w] -> [8T,w]
(rank-major); the dim=-1 column layout of vLLM needs an extra permute with NCCL that is not included in its time.

null kernel back-to-back in a graph: 0.69 us; 50 us sleep kernel measures 50.62 us

## all_reduce bf16 (us per call, CUDA graph; b2b = back-to-back, spacer = exposed time behind a 50 us kernel)

| shape | bytes | NCCL b2b | NCCL spacer | best topo algo (grid/block/unroll) | topo b2b | topo spacer | speedup (spacer) | bitwise contract |
|---|---|---|---|---|---|---|---|---|
| (1, 8192) | 16384 | 42.8 | 43.0 | alg2 (4/64/1) [alg2 5.4] | 5.42 | 5.45 | 7.9x | True |
| (2, 8192) | 32768 | 43.8 | 44.1 | alg2 (8/64/1) [alg2 6.3] | 6.25 | 6.18 | 7.1x | True |
| (3, 8192) | 49152 | 44.7 | 44.7 | alg2 (32/32/1) [alg2 7.0] | 7.00 | 6.96 | 6.4x | True |
| (4, 8192) | 65536 | 46.0 | 46.1 | alg2 (32/32/1) [alg2 7.8] | 7.83 | 7.92 | 5.8x | True |
| (6, 8192) | 98304 | 46.0 | 46.0 | alg2 (64/32/1) [alg2 9.6] | 9.59 | 9.52 | 4.8x | True |
| (8, 8192) | 131072 | 47.3 | 47.6 | alg2 (64/32/1) [alg2 11.5] | 11.50 | 11.32 | 4.2x | True |
| (12, 8192) | 196608 | 49.6 | 49.9 | alg2 (64/32/1) [alg2 16.9] | 16.93 | 16.59 | 3.0x | True |
| (16, 8192) | 262144 | 52.3 | 52.5 | alg2 (64/32/1) [alg2 18.5] | 18.52 | 18.32 | 2.9x | True |
| (24, 8192) | 393216 | 60.4 | 60.5 | alg2 (8/256/1) [alg2 25.7] | 25.71 | 25.46 | 2.4x | True |
| (32, 8192) | 524288 | 66.7 | 66.8 | alg2 (8/256/1) [alg2 33.0] | 33.03 | 32.73 | 2.0x | True |
| (48, 8192) | 786432 | 74.2 | 74.5 | alg2 (8/256/1) [alg2 46.9] | 46.89 | 46.64 | 1.6x | True |
| (64, 8192) | 1048576 | 85.9 | 86.1 | alg2 (8/256/1) [alg2 61.7] | 61.69 | 61.12 | 1.4x | True |
| (96, 8192) | 1572864 | 101.3 | 101.7 | alg2 (8/256/1) [alg2 91.7] | 91.73 | 91.07 | 1.1x | True |
| (128, 8192) | 2097152 | 134.5 | 134.8 | alg2 (8/256/1) [alg2 120.4] | 120.42 | 120.00 | 1.1x | True |
| (192, 8192) | 3145728 | 217.1 | 216.2 | alg2 (8/256/1) [alg2 180.9] | 180.89 | 180.55 | 1.2x | True |
| (256, 8192) | 4194304 | 287.8 | 287.8 | alg2 (8/256/1) [alg2 245.0] | 245.05 | 244.15 | 1.2x | True |
| (384, 8192) | 6291456 | 433.4 | 431.3 | alg2 (4/256/2) [alg2 359.5] | 359.45 | 357.61 | 1.2x | True |
| (512, 8192) | 8388608 | 580.4 | 573.0 | alg2 (4/256/2) [alg2 478.4] | 478.39 | 479.53 | 1.2x | True |
| (1, 4096) | 8192 | 41.2 | 41.6 | alg2 (4/32/1) [alg2 5.0] | 5.03 | 5.09 | 8.2x | True |
| (2, 4096) | 16384 | 42.7 | 42.8 | alg2 (4/64/1) [alg2 5.4] | 5.44 | 5.47 | 7.8x | True |
| (3, 4096) | 24576 | 43.4 | 43.8 | alg2 (8/64/1) [alg2 5.8] | 5.82 | 5.81 | 7.5x | True |
| (4, 4096) | 32768 | 44.0 | 44.3 | alg2 (8/64/1) [alg2 6.3] | 6.25 | 6.18 | 7.2x | True |
| (6, 4096) | 49152 | 44.7 | 44.8 | alg2 (32/32/1) [alg2 7.0] | 7.01 | 6.98 | 6.4x | True |
| (8, 4096) | 65536 | 45.8 | 46.1 | alg2 (32/32/1) [alg2 7.9] | 7.85 | 7.87 | 5.9x | True |
| (12, 4096) | 98304 | 46.0 | 46.2 | alg2 (64/32/1) [alg2 9.6] | 9.58 | 9.52 | 4.9x | True |
| (16, 4096) | 131072 | 47.2 | 47.5 | alg2 (64/32/1) [alg2 11.5] | 11.51 | 11.36 | 4.2x | True |
| (24, 4096) | 196608 | 49.7 | 50.0 | alg2 (64/32/1) [alg2 16.9] | 16.90 | 16.66 | 3.0x | True |
| (32, 4096) | 262144 | 52.4 | 52.4 | alg2 (64/32/1) [alg2 18.5] | 18.50 | 18.34 | 2.9x | True |
| (48, 4096) | 393216 | 60.4 | 60.5 | alg2 (8/256/1) [alg2 25.8] | 25.75 | 25.56 | 2.4x | True |
| (64, 4096) | 524288 | 66.7 | 66.8 | alg2 (8/256/1) [alg2 33.1] | 33.06 | 32.72 | 2.0x | True |
| (96, 4096) | 786432 | 74.2 | 74.5 | alg2 (8/256/1) [alg2 47.3] | 47.29 | 46.73 | 1.6x | True |
| (128, 4096) | 1048576 | 85.9 | 86.2 | alg2 (8/256/1) [alg2 61.6] | 61.56 | 61.18 | 1.4x | True |
| (192, 4096) | 1572864 | 101.9 | 101.7 | alg2 (8/256/1) [alg2 91.4] | 91.36 | 90.67 | 1.1x | True |
| (256, 4096) | 2097152 | 134.7 | 134.5 | alg2 (8/256/1) [alg2 119.7] | 119.72 | 119.70 | 1.1x | True |
| (384, 4096) | 3145728 | 216.9 | 217.1 | alg2 (8/256/1) [alg2 183.9] | 183.90 | 181.34 | 1.2x | True |
| (512, 4096) | 4194304 | 288.8 | 287.8 | alg2 (8/256/1) [alg2 244.2] | 244.15 | 239.82 | 1.2x | True |

## all_gather bf16 ([T,w] -> [T,8w] for dim-1, [8T,w] for dim0)

| shard shape | layout | shard bytes | NCCL b2b | NCCL spacer | best topo algo (grid/block/unroll) | topo b2b | topo spacer | speedup (spacer) |
|---|---|---|---|---|---|---|---|---|
| (1, 512) | dim-1 | 1024 | 24.6 | 25.0 | alg1 (2/128/1) [alg1 3.6] | 3.60 | 3.67 | 6.8x |
| (1, 512) | dim0 | 1024 | 24.6 | 25.0 | alg1 (2/128/1) [alg1 3.6] | 3.59 | 3.72 | 6.7x |
| (2, 512) | dim-1 | 2048 | 26.0 | 26.1 | alg1 (4/128/1) [alg1 3.8] | 3.79 | 3.83 | 6.8x |
| (2, 512) | dim0 | 2048 | 26.0 | 26.1 | alg1 (4/128/1) [alg1 3.8] | 3.79 | 3.87 | 6.7x |
| (4, 512) | dim-1 | 4096 | 26.6 | 26.5 | alg1 (8/128/1) [alg1 4.2] | 4.18 | 4.10 | 6.5x |
| (4, 512) | dim0 | 4096 | 26.6 | 26.5 | alg1 (8/128/1) [alg1 4.2] | 4.18 | 4.10 | 6.5x |
| (8, 512) | dim-1 | 8192 | 26.9 | 27.3 | alg1 (64/32/1) [alg1 5.0] | 5.01 | 4.86 | 5.6x |
| (8, 512) | dim0 | 8192 | 26.9 | 27.3 | alg1 (64/32/1) [alg1 5.0] | 5.01 | 4.87 | 5.6x |
| (16, 512) | dim-1 | 16384 | 28.2 | 28.3 | alg1 (128/32/1) [alg1 7.0] | 7.00 | 6.99 | 4.1x |
| (16, 512) | dim0 | 16384 | 28.2 | 28.3 | alg1 (128/32/1) [alg1 7.0] | 7.00 | 7.02 | 4.0x |
| (24, 512) | dim-1 | 24576 | 29.2 | 29.5 | alg2 (32/128/1) [alg2 9.0] | 8.98 | 8.85 | 3.3x |
| (24, 512) | dim0 | 24576 | 29.2 | 29.5 | alg2 (16/256/1) [alg2 9.0] | 9.00 | 8.88 | 3.3x |
| (32, 512) | dim-1 | 32768 | 30.8 | 31.2 | alg2 (32/128/1) [alg2 10.3] | 10.32 | 10.00 | 3.1x |
| (32, 512) | dim0 | 32768 | 30.8 | 31.2 | alg2 (16/256/1) [alg2 10.3] | 10.32 | 10.06 | 3.1x |
| (48, 512) | dim-1 | 49152 | 34.7 | 34.8 | alg2 (24/256/1) [alg2 13.0] | 13.02 | 12.70 | 2.7x |
| (48, 512) | dim0 | 49152 | 34.7 | 34.8 | alg2 (24/256/1) [alg2 13.0] | 12.99 | 12.61 | 2.8x |
| (64, 512) | dim-1 | 65536 | 37.6 | 38.0 | alg2 (16/256/1) [alg2 15.6] | 15.55 | 15.22 | 2.5x |
| (64, 512) | dim0 | 65536 | 37.6 | 38.0 | alg2 (16/256/1) [alg2 15.6] | 15.58 | 15.27 | 2.5x |
| (96, 512) | dim-1 | 98304 | 45.0 | 45.1 | alg2 (32/1024/1) [alg2 22.4] | 22.36 | 22.10 | 2.0x |
| (96, 512) | dim0 | 98304 | 45.0 | 45.1 | alg2 (32/1024/1) [alg2 22.2] | 22.17 | 21.94 | 2.1x |
| (128, 512) | dim-1 | 131072 | 51.2 | 51.4 | alg2 (32/1024/1) [alg2 27.2] | 27.18 | 26.89 | 1.9x |
| (128, 512) | dim0 | 131072 | 51.2 | 51.4 | alg2 (32/1024/1) [alg2 27.1] | 27.12 | 26.91 | 1.9x |
| (192, 512) | dim-1 | 196608 | 57.0 | 57.2 | alg2 (32/1024/1) [alg2 41.3] | 41.27 | 41.04 | 1.4x |
| (192, 512) | dim0 | 196608 | 57.0 | 57.2 | alg2 (32/1024/1) [alg2 41.2] | 41.19 | 41.09 | 1.4x |
| (256, 512) | dim-1 | 262144 | 71.5 | 71.8 | alg2 (32/1024/1) [alg2 55.7] | 55.71 | 55.48 | 1.3x |
| (256, 512) | dim0 | 262144 | 71.5 | 71.8 | alg2 (32/1024/1) [alg2 55.7] | 55.71 | 55.47 | 1.3x |
| (1, 1024) | dim-1 | 2048 | 26.1 | 26.2 | alg1 (4/128/1) [alg1 3.8] | 3.79 | 3.87 | 6.8x |
| (1, 1024) | dim0 | 2048 | 26.1 | 26.2 | alg1 (4/128/1) [alg1 3.8] | 3.80 | 3.90 | 6.7x |
| (2, 1024) | dim-1 | 4096 | 26.6 | 26.8 | alg1 (8/128/1) [alg1 4.2] | 4.20 | 4.19 | 6.4x |
| (2, 1024) | dim0 | 4096 | 26.6 | 26.8 | alg1 (8/128/1) [alg1 4.2] | 4.19 | 4.17 | 6.4x |
| (4, 1024) | dim-1 | 8192 | 26.9 | 27.4 | alg1 (64/32/1) [alg1 5.0] | 5.01 | 5.01 | 5.5x |
| (4, 1024) | dim0 | 8192 | 26.9 | 27.4 | alg1 (64/32/1) [alg1 5.0] | 5.01 | 4.96 | 5.5x |
| (8, 1024) | dim-1 | 16384 | 28.3 | 28.5 | alg1 (128/32/1) [alg1 7.0] | 7.01 | 7.03 | 4.1x |
| (8, 1024) | dim0 | 16384 | 28.3 | 28.5 | alg1 (128/32/1) [alg1 7.0] | 7.01 | 7.06 | 4.0x |
| (16, 1024) | dim-1 | 32768 | 30.8 | 31.3 | alg2 (32/128/1) [alg2 10.3] | 10.33 | 10.10 | 3.1x |
| (16, 1024) | dim0 | 32768 | 30.8 | 31.3 | alg2 (16/256/1) [alg2 10.3] | 10.33 | 10.16 | 3.1x |
| (24, 1024) | dim-1 | 49152 | 34.7 | 34.8 | alg2 (24/256/1) [alg2 13.0] | 12.99 | 12.71 | 2.7x |
| (24, 1024) | dim0 | 49152 | 34.7 | 34.8 | alg2 (24/256/1) [alg2 13.0] | 12.99 | 12.73 | 2.7x |
| (32, 1024) | dim-1 | 65536 | 37.9 | 38.1 | alg2 (16/256/1) [alg2 15.6] | 15.55 | 15.32 | 2.5x |
| (32, 1024) | dim0 | 65536 | 37.9 | 38.1 | alg2 (16/256/1) [alg2 15.6] | 15.58 | 15.31 | 2.5x |
| (48, 1024) | dim-1 | 98304 | 45.0 | 45.3 | alg2 (32/1024/1) [alg2 22.4] | 22.37 | 22.22 | 2.0x |
| (48, 1024) | dim0 | 98304 | 45.0 | 45.3 | alg2 (32/1024/1) [alg2 22.2] | 22.17 | 22.07 | 2.1x |
| (64, 1024) | dim-1 | 131072 | 51.3 | 51.4 | alg2 (32/1024/1) [alg2 27.2] | 27.22 | 27.06 | 1.9x |
| (64, 1024) | dim0 | 131072 | 51.3 | 51.4 | alg2 (32/1024/1) [alg2 27.2] | 27.17 | 26.95 | 1.9x |
| (96, 1024) | dim-1 | 196608 | 57.0 | 57.3 | alg2 (32/1024/1) [alg2 41.3] | 41.32 | 41.14 | 1.4x |
| (96, 1024) | dim0 | 196608 | 57.0 | 57.3 | alg2 (32/1024/1) [alg2 41.3] | 41.29 | 41.11 | 1.4x |
| (128, 1024) | dim-1 | 262144 | 71.6 | 71.9 | alg2 (32/1024/1) [alg2 56.0] | 56.02 | 55.54 | 1.3x |
| (128, 1024) | dim0 | 262144 | 71.6 | 71.9 | alg2 (32/1024/1) [alg2 55.8] | 55.79 | 55.63 | 1.3x |
| (192, 1024) | dim-1 | 393216 | 110.4 | 110.3 | alg2 (32/256/1) [alg2 87.2] | 87.22 | 86.98 | 1.3x |
| (192, 1024) | dim0 | 393216 | 110.4 | 110.3 | alg2 (32/256/1) [alg2 87.0] | 86.99 | 87.01 | 1.3x |
| (256, 1024) | dim-1 | 524288 | 147.2 | 146.8 | alg2 (32/256/1) [alg2 118.3] | 118.27 | 118.56 | 1.2x |
| (256, 1024) | dim0 | 524288 | 147.2 | 146.8 | alg2 (32/256/1) [alg2 118.5] | 118.51 | 118.42 | 1.2x |
| (1, 25256) | dim-1 | 50512 | 35.2 | 35.7 | alg2 (24/256/1) [alg2 13.3] | 13.28 | 13.02 | 2.7x |
| (1, 25256) | dim0 | 50512 | 35.2 | 35.7 | alg2 (24/256/1) [alg2 13.3] | 13.29 | 12.99 | 2.7x |


## Comparison with FlashInfer's PCIe IPC all-reduce and vLLM's CustomAllreduce (same session, spacer mode, us)

`scripts/run_h1a.sh <tag> fi_spacer` runs a copy of `/raid/vllm-runtime/musetest/customar/bench_spacer.py` (50 us
spacer, graph, FlashInfer with its per-shape tuned best configuration from `fi_small.jsonl`). FlashInfer's kernel is
faster for T = 2..32 because it exchanges bf16 partials over the socket cut and adds in bf16 (class
deterministic_different, not the fp32 contract of this package); at T = 1 the two are within 4%, at T = 64 muse_topo
is faster.

| shape | NCCL (spacer) | FlashInfer PCIe IPC, best tuned config (spacer) | vLLM CustomAllreduce 2-stage (spacer) | muse_topo (spacer) |
|---|---|---|---|---|
| [1, 8192] | 43.1 | 5.2 (UNSTAGED/b32/t128) | 33.6 | 5.4 |
| [2, 8192] | 44.1 | 5.6 (UNSTAGED/b32/t128) | 55.1 | 6.2 |
| [4, 8192] | 46.0 | 6.5 (UNSTAGED/b64/t128) | 105.7 | 7.9 |
| [8, 8192] | 47.6 | 8.7 (UNSTAGED/b32/t256) | 192.2 | 11.3 |
| [16, 8192] | 52.8 | 14.4 (UNSTAGED/b64/t512) | 359.9 | 18.3 |
| [32, 8192] | 67.0 | 24.4 (UNSTAGED/b64/t1024) | 811.3 | 32.7 |
| [64, 8192] | 86.2 | 69.9 (STAGED/b16/t512) | 1899.9 | 61.1 |
