# H1b latency tables: large-message collectives (all_reduce, reduce_scatter, all_gather) vs NCCL

> Note (review fix, 2026-10-24): the counter and flags are 64 bit now; the large collectives were re-measured with the same protocol and are unchanged within +-0.5 % (see `docs/h2_vllm_plugin.md`, "Review fixes").

Measured 2026-10-23 on the AS-5126GS-TNRT2 box (8x RTX PRO 6000 Blackwell, PCIe, 2 sockets), 8 processes, the dispatch
path the plugin calls (`comm.all_reduce(x, out=out)`, `comm.reduce_scatter(x, out=out)`, `comm.all_gather(x, dim, out=out)`),
every result verified bitwise against the arithmetic contract / `torch.cat` in the same run
(`bench/run_h1b.sh <tag> bench_ar bench_rs bench_ag`, raw rows in the server work dir, summarized by `bench/summarize_h1b.py`).
Protocol (same as `docs/h1a_latency.md`): CUDA graph of N identical calls (N = 20 below 8 MiB, 6 from 8 MiB), replayed; per-call
time = replay time / N, median of 5 trials, max over ranks. `b2b` = back-to-back identical calls, `spacer` = (50 us sleep
kernel, call) pairs minus the sleep-only time (up to 16 MiB; above that the call time dwarfs any spacer). Baseline: vLLM's
`PyNcclCommunicator` (NCCL 2.29.7, `NCCL_P2P_LEVEL=SYS`, default algorithm/protocol selection) on the same gloo group.
Input randn bf16. Sizes are the message bytes of the all_reduce / reduce_scatter input ([T, 8192], T = 64 x MiB) and the OUTPUT
bytes of the all_gather (shard [T/8, w] -> out; the pooler / unpooler gathers of an 8192-token prefill chunk are 64 / 128 MiB;
both layouts, dim -1 = column concatenation and dim 0 = rank-major). `h1a_table` is the H1a decode/medium algorithm (3-phase
sentinel, still the dispatch below 2 MiB); the dispatch column is the faster of the two at every size (the crossover is
2 MiB). The bf16-exchange column is the opt-in `MUSE_TOPO_LARGE_BF16_EXCHANGE=1` path (deterministic_different, see
`docs/h1b_collectives.md`); all other columns are the default exact path (bitwise identical to the small algorithms,
correctly rounded fp32 sum for activation-like data, see the accuracy note in `docs/h1b_collectives.md`).

Medium sizes of the spec ([64..512, 8192] bf16 = 1..8 MiB): the dispatch is never slower than the H1a algorithms (61.4 vs 61.3 us at
1 MiB, 117.6 vs 120.4 us at 2 MiB, 454 vs 480 us at 8 MiB), so the H1a tables of `docs/h1a_latency.md` stay valid below 2 MiB.


### all_reduce bf16 [T, 8192] (CUDA graph, us per call; T = 64 x MiB)

| size (MiB) | NCCL b2b us | NCCL spacer us | topo b2b us | topo spacer us | speedup b2b | speedup spacer | h1a_table us | bf16 exchange us (opt-in) | bitwise ok |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 86.2 | 86.2 | 61.4 | 61.0 | 1.40x | 1.41x | 61.3 | 60.3 | True |
| 1.5 | 101.6 | 102.1 | 90.5 | 90.5 | 1.12x | 1.13x | 91.2 | 82.1 | True |
| 2 | 133.8 | 134.0 | 117.6 | 118.0 | 1.14x | 1.14x | 120.4 | 105.4 | True |
| 2.5 | 175.8 | 176.9 | 144.6 | 143.2 | 1.22x | 1.24x | 150.9 | 128.5 | True |
| 3 | 215.8 | 216.7 | 168.9 | 169.8 | 1.28x | 1.28x | 179.3 | 146.9 | True |
| 4 | 287.4 | 287.1 | 222.6 | 223.0 | 1.29x | 1.29x | 244.1 | 199.8 | True |
| 6 | 432.9 | 429.2 | 331.0 | 332.4 | 1.31x | 1.29x | 358.2 | 293.2 | True |
| 8 | 577.0 | 573.0 | 454.1 | 448.3 | 1.27x | 1.28x | 480.4 | 388.3 | True |
| 16 | 1158.5 | 1149.3 | 917.9 | 911.1 | 1.26x | 1.26x | - | 745.6 | True |
| 32 | 2305.4 | - | 1795.6 | - | 1.28x | - | - | 1460.4 | True |
| 64 | 4597.7 | - | 3582.2 | - | 1.28x | - | - | 2939.3 | True |
| 128 | 9203.7 | - | 7097.1 | - | 1.30x | - | - | 5839.6 | True |

### reduce_scatter bf16 [T, 8192] -> [T/8, 8192] (CUDA graph, us per call)

| size (MiB) | NCCL b2b us | NCCL spacer us | topo b2b us | topo spacer us | speedup b2b | speedup spacer | bf16 exchange us (opt-in) | bitwise ok |
|---|---|---|---|---|---|---|---|---|
| 0.125 | 27.7 | 28.0 | 7.7 | 7.5 | 3.61x | 3.71x | 6.5 | True |
| 0.25 | 30.6 | 30.6 | 11.6 | 11.5 | 2.63x | 2.67x | 9.1 | True |
| 0.5 | 38.1 | 38.1 | 20.4 | 20.4 | 1.87x | 1.87x | 15.7 | True |
| 1 | 49.1 | 49.3 | 36.6 | 36.3 | 1.34x | 1.36x | 26.7 | True |
| 2 | 72.7 | 72.6 | 66.3 | 66.3 | 1.10x | 1.09x | 47.1 | True |
| 4 | 150.5 | 150.3 | 126.5 | 127.6 | 1.19x | 1.18x | 88.0 | True |
| 8 | 300.7 | 297.4 | 250.4 | 250.3 | 1.20x | 1.19x | 170.0 | True |
| 16 | 598.3 | 593.3 | 501.1 | 497.4 | 1.19x | 1.19x | 341.8 | True |
| 32 | 1192.9 | - | 1001.2 | - | 1.19x | - | 681.4 | True |
| 64 | 2378.5 | - | 1998.4 | - | 1.19x | - | 1360.4 | True |
| 128 | 4790.8 | - | 4022.2 | - | 1.19x | - | 2721.6 | True |

### all_gather bf16, size = OUTPUT MiB (CUDA graph, us per call)

| size (MiB) | layout | NCCL b2b us | NCCL spacer us | topo b2b us | topo spacer us | speedup b2b | speedup spacer | bf16 exchange us (opt-in) | bitwise ok |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | dim-1 | 30.9 | 31.1 | 10.4 | 10.1 | 2.98x | 3.09x | - | True |
| 0.25 | dim0 | 30.9 | 31.1 | 10.5 | 10.2 | 2.96x | 3.05x | - | True |
| 0.5 | dim-1 | 38.0 | 38.0 | 15.8 | 15.4 | 2.41x | 2.47x | - | True |
| 0.5 | dim0 | 38.0 | 38.0 | 15.8 | 15.4 | 2.41x | 2.46x | - | True |
| 1 | dim-1 | 51.3 | 51.4 | 27.3 | 27.7 | 1.88x | 1.86x | - | True |
| 1 | dim0 | 51.3 | 51.4 | 27.2 | 26.9 | 1.89x | 1.91x | - | True |
| 2 | dim-1 | 71.6 | 71.2 | 51.5 | 51.3 | 1.39x | 1.39x | - | True |
| 2 | dim0 | 71.6 | 71.2 | 51.7 | 51.6 | 1.38x | 1.38x | - | True |
| 4 | dim-1 | 147.9 | 146.1 | 103.8 | 103.6 | 1.42x | 1.41x | - | True |
| 4 | dim0 | 147.9 | 146.1 | 104.2 | 103.7 | 1.42x | 1.41x | - | True |
| 8 | dim-1 | 294.4 | 288.2 | 207.8 | 207.6 | 1.42x | 1.39x | - | True |
| 8 | dim0 | 294.4 | 288.2 | 208.1 | 208.3 | 1.41x | 1.38x | - | True |
| 16 | dim-1 | 586.6 | 578.2 | 414.5 | 417.0 | 1.42x | 1.39x | - | True |
| 16 | dim0 | 586.6 | 578.2 | 416.8 | 414.9 | 1.41x | 1.39x | - | True |
| 32 | dim-1 | 1171.2 | - | 830.6 | - | 1.41x | - | - | True |
| 32 | dim0 | 1171.2 | - | 831.2 | - | 1.41x | - | - | True |
| 64 | dim-1 | 2326.2 | - | 1662.8 | - | 1.40x | - | - | True |
| 64 | dim0 | 2326.2 | - | 1667.4 | - | 1.40x | - | - | True |
| 128 | dim-1 | 4652.1 | - | 3309.7 | - | 1.41x | - | - | True |
| 128 | dim0 | 4652.1 | - | 3317.8 | - | 1.40x | - | - | True |


## Baselines of the design iterations (128 MiB all_reduce, us)

| variant | us |
|---|---|
| NCCL | 9170 |
| H1a algorithm 2 on 8 MiB pieces (loop of existing kernels) | 8030-8220 |
| reduce_scatter + all_gather per chunk, two launches, default grid | 8770 |
| fused step, default grid (8 + 24 blocks) | 8440 |
| fused step + dedicated cut block for the all-gather, 4 + 8 blocks, unroll 2 (shipped) | 7100-7130 |
| same, bf16 exchange (opt-in) | 5840 |

## Transport measurements behind the design (bench/ce_cut_probe.py, bench/a2a_probe.py)

All numbers are GB/s per GPU and direction with all 8 GPUs active at the same time (eager launches, event timed).

| pattern | SM push | copy engine | SM pull |
|---|---|---|---|
| cut: partner exchange r <-> r^4 (both directions at once), 1 block x 512 threads | 9.4-9.9 | 9.9-10.0 (K = 1..8 pieces, >= 1 MiB) | 6.7 |
| cut, 2 / 4 / 8 blocks | 8.0 / 6.8 / 6.8 | - | - |
| island all-to-all, 3 flows (the 3 island peers), 1 block per flow | 42-46 (aggregate over the 3 flows) | 29-34 (3 streams), 36-47 (one stream, sequential) | 35 |
| island all-to-all, 2 / 4 / 8 blocks per flow | 35-40 | - | - |
| one island peer | 45-47 | 50.7-52.3 | 36-39 |
| byte mix of one all-reduce step per GPU (12 MiB island + 3 MiB cut), 1 block per flow | 401 us (39 GB/s total) | - | - |

Consequences: (1) the socket cut is capped at ~9.8 GB/s per GPU (39 GB/s aggregate per direction) for every transport, the
copy engine included; (2) SM push over the island saturates at ~45 GB/s per GPU, the copy engine is not better for the
all-to-all patterns of the collectives (it only wins for a single flow), so no copy-engine path was built; (3) fewer, fatter
blocks win everywhere (a single block per flow beats 2-8 blocks by 10-30%), which is why the tuned configurations use 4 reduce-scatter
and 8 all-gather blocks; (4) a pure-copy kernel with the byte mix of one all-reduce step needs 401 us, the real fused step takes
~420 us (trace below), so the shipped kernel is within ~5% of what SM push can do for the exact (fp32-partial) traffic.

In-kernel timeline of the fused step (bench/trace_h1b.py, MUSE_TOPO_TRACE=1 build, 128 MiB, rank 0): steady-state step 417-428 us
(reduce-scatter role 417-428 us, all-gather role 283 us, its cut block 170-180 us), launch gap between steps 3-20 us, first step (reduce-scatter alone)
248 us, last step (all-gather alone) 208 us. The two roles do not overlap perfectly because they share the island egress
(reduce-scatter alone 244 us, all-gather alone 245 us per 8 MiB chunk; fused 420 us instead of 490 us sequential).
