# A1: retune of the muse_topo dispatch tables for the bf16 exchange (S1-9 L9)

Host branch `s19-a1` (from main a7e7a03). Machine-specific tables of `muse_topo/collectives.py` for the production setting (bf16 exchange ON, medium
BX ON, the `env/host.env` defaults). **Bitwise identical outputs**: no kernel and no arithmetic changed, only which kernel geometry / launch
sequence serves a size; per-size proof in section 4.

## 1. What changed

| cell | message (bytes) | where it occurs | change | per call (warm / cold, 8 GPUs, graph) |
|---|---|---|---|---|
| reduce_scatter, bf16 exchange | 1.5 .. 4 MiB | sequence-parallel MoE partial `[T, 4096]` at T = 384 .. 512 (c48 / c64), and every smaller SP batch | new `_RS_BX_TABLE`: 8 blocks x 256 threads at every size (the exact-exchange row had 4 blocks for 1 .. 4 MiB; the rows up to 1 MiB and from 4 MiB up already had 8) | 1.5 MiB -9.4 / -11.6 us, 2 MiB -13.3 / -16.3, 3 MiB -20.2 / -25.1, 4 MiB -27.2 / -34.3 (-20 .. -24 %) |
| all_reduce 2bx, bf16 exchange | 1.5 MiB, 2 MiB | replicated attn-out `[128, 8192]` (c16), MoE-out `[256, 4096]` (c32), draft all-reduces | rows 8 x 512 instead of 16 x 128 | 1.5 MiB -1.1 / -3.4 us, 2 MiB -1.3 .. -1.5 / -4.5 (spacer equal) |
| all_reduce, bf16 exchange, above `max_bytes` (8 MiB) up to 64 MiB | 12 .. 64 MiB | c96 / c128 draft all-reduces (10 + 1 per step) `[768 / 1024, 8192]` | row slabs of at most 4 MiB, each through the 2bx table, instead of one call of the large algorithm | 12 MiB -72 / -66 us, 16 MiB -85 / -82, 24 MiB -108 / -102, 32 MiB -138 / -138, 64 MiB -298 / -266 (-12 %) |

Not changed (measured, no robust gain): all_gather (every table row), reduce_scatter at 6 MiB and above (the exact-exchange rows already had 8
blocks), all_reduce 2bx at every other size (one 3 .. 4 MiB candidate wins cold and loses spacer: see 5.4), the 128 MiB prefill embedding
all-reduce (stays on the large algorithm), the exact-exchange paths (`MUSE_TOPO_LARGE_BF16_EXCHANGE=0`, fp16 / fp32 inputs).

## 2. Why the old rows were not optimal for production

* `_RS_TABLE` and `_AR_LARGE_TABLE` were searched on 2026-10-23 (H1b) with the **exact fp32 exchange**, which sends twice the bytes across the socket cut
  (4 blocks were enough below 4 MiB). Production runs the bf16 exchange since 2026-10-24, so the reduce-scatter row for 1 .. 4 MiB and the large
  all-reduce row sat on the wrong side of the knee (docs/h1b_latency.md lists 88.0 us for the bf16 exchange at 4 MiB against 126.5 us for the table row).
* `_AR_BX_TABLE` (2026-10-26) was searched with the warm back-to-back protocol and the fewest threads within 1.5 %; with the exchange on, algorithm 2 also
  beats the large algorithm per MiB at every size, but the large algorithm took everything above `max_bytes` (8 MiB, the workspace limit).
* The S1-9 L8 spike (SPIKE.md) measured 126 -> 88 us at 4 MiB with 8 blocks (confirmed here) and claimed four 1 MiB all-reduces (161 us) beat one 4 MiB one
  (188 us) at c32 (reproduced only under the L8 memset-cold protocol: measured directly, four 1 MiB calls cost 171.0 us warm / 171.2 spacer against 164.0 / 163.8 us for the single 4 MiB call (+7 us), and
  161.9 us against 176.2 us in the memset-cold protocol (-14 us): the L8 gap is a cold-start effect, see 5.2).

## 3. Message sizes of the production steps

Production speculative profile: K = 7, verify batch T = 8 x requests, graphs up to 256 tokens (c1 .. c32: replicated stream, CUDA graph), c40 (T = 320)
eager replicated, sequence-parallel from T = 384 (c48 .. c128, eager), prefill chunks of 8192 rows (two r2e microbatches of 4096). Sources: the
round-5 boot log (`MUSE_TOPO_LOG_OPS=1`: every distinct (op, shape) of the boot, no counts), the model code (`vllm/models/muse_spark/nvidia/model.py`) and the
round-5 per-step kernel counts (c32: 125 all-reduce launches = 62 attn-out 4 MiB + 62 MoE-out 2 MiB + embedding; c64: 62 + 62 reduce-scatter launches, 124
all-gather chunk launches (1 MiB shard = 2 chunks of 512 KiB) + 60 flag all-gathers (144 KiB packed latent); c128: 186 reduce-scatter launches (16 MiB = 2,
8 MiB = 1 per layer), 316 all-gather chunk launches). The log predates L6 / L7; neither changes a collective shape. Draft rows are assumed equal to T
(c64 draft all-reduce = `[512, 8192]` 8 MiB at grid 16 in the round-5 trace; c128 = 16 MiB on the large algorithm). `bench/a1_sizes.json` is the machine-readable
form (calls per step per site).

| band | site | message | calls/step | row before | change |
|---|---|---|---|---|---|
| c1 | moe-out AR | (8, 4096) 64 KiB | 62 | 2bx (65536, 2, 8, 128, 1) | unchanged |
| c1 | embedding AR | (8, 8192) 128 KiB | 1 | 2bx (131072, 2, 8, 256, 1) | unchanged |
| c1 | draft AR (5 layers x 2 + emb) | (8, 8192) 128 KiB | 11 | 2bx (131072, 2, 8, 256, 1) | unchanged |
| c2 | attn-out AR | (16, 8192) 256 KiB | 62 | 2bx (262144, 2, 64, 64, 1) | unchanged |
| c2 | moe-out AR | (16, 4096) 128 KiB | 62 | 2bx (131072, 2, 8, 256, 1) | unchanged |
| c2 | embedding AR | (16, 8192) 256 KiB | 1 | 2bx (262144, 2, 64, 64, 1) | unchanged |
| c2 | unpooler col AG (dim -1) | (16, 1024) 32 KiB | 62 | AG (32768, 2, 32, 128, 1) | unchanged |
| c2 | draft AR (5 layers x 2 + emb) | (16, 8192) 256 KiB | 11 | 2bx (262144, 2, 64, 64, 1) | unchanged |
| c4 | attn-out AR | (32, 8192) 512 KiB | 62 | 2bx (524288, 2, 16, 256, 1) | unchanged |
| c4 | moe-out AR | (32, 4096) 256 KiB | 62 | 2bx (262144, 2, 64, 64, 1) | unchanged |
| c4 | embedding AR | (32, 8192) 512 KiB | 1 | 2bx (524288, 2, 16, 256, 1) | unchanged |
| c4 | pooler col AG (dim -1) | (32, 512) 32 KiB | 62 | AG (32768, 2, 32, 128, 1) | unchanged |
| c4 | unpooler col AG (dim -1) | (32, 1024) 64 KiB | 62 | AG (65536, 2, 16, 256, 1) | unchanged |
| c4 | draft AR (5 layers x 2 + emb) | (32, 8192) 512 KiB | 11 | 2bx (524288, 2, 16, 256, 1) | unchanged |
| c8 | attn-out AR | (64, 8192) 1 MiB | 62 | 2bx (1048576, 2, 8, 512, 1) | unchanged |
| c8 | moe-out AR | (64, 4096) 512 KiB | 62 | 2bx (524288, 2, 16, 256, 1) | unchanged |
| c8 | embedding AR | (64, 8192) 1 MiB | 1 | 2bx (1048576, 2, 8, 512, 1) | unchanged |
| c8 | pooler col AG (dim -1) | (64, 512) 64 KiB | 62 | AG (65536, 2, 16, 256, 1) | unchanged |
| c8 | unpooler col AG (dim -1) | (64, 1024) 128 KiB | 62 | AG (131072, 2, 32, 1024, 1) | unchanged |
| c8 | draft AR (5 layers x 2 + emb) | (64, 8192) 1 MiB | 11 | 2bx (1048576, 2, 8, 512, 1) | unchanged |
| c16 | attn-out AR | (128, 8192) 2 MiB | 62 | 2bx (2097152, 2, 16, 128, 1) | 2bx 8 x 512 (was 16 x 128) |
| c16 | moe-out AR | (128, 4096) 1 MiB | 62 | 2bx (1048576, 2, 8, 512, 1) | unchanged |
| c16 | embedding AR | (128, 8192) 2 MiB | 1 | 2bx (2097152, 2, 16, 128, 1) | 2bx 8 x 512 (was 16 x 128) |
| c16 | pooler col AG (dim -1) | (128, 512) 128 KiB | 62 | AG (131072, 2, 32, 1024, 1) | unchanged |
| c16 | unpooler col AG (dim -1) | (128, 1024) 256 KiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c16 | draft AR (5 layers x 2 + emb) | (128, 8192) 2 MiB | 11 | 2bx (2097152, 2, 16, 128, 1) | 2bx 8 x 512 (was 16 x 128) |
| c32 | attn-out AR | (256, 8192) 4 MiB | 62 | 2bx (4194304, 2, 16, 128, 1) | unchanged |
| c32 | moe-out AR | (256, 4096) 2 MiB | 62 | 2bx (2097152, 2, 16, 128, 1) | 2bx 8 x 512 (was 16 x 128) |
| c32 | embedding AR | (256, 8192) 4 MiB | 1 | 2bx (4194304, 2, 16, 128, 1) | unchanged |
| c32 | pooler col AG (dim -1) | (256, 512) 256 KiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c32 | unpooler col AG (dim -1) | (256, 1024) 512 KiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c32 | draft AR (5 layers x 2 + emb) | (256, 8192) 4 MiB | 11 | 2bx (4194304, 2, 16, 128, 1) | unchanged |
| c40 | attn-out AR | (320, 8192) 5 MiB | 62 | 2bx (6291456, 2, 16, 128, 1) | unchanged |
| c40 | moe-out AR | (320, 4096) 2.5 MiB | 62 | 2bx (3145728, 2, 8, 256, 1) | unchanged |
| c40 | embedding AR | (320, 8192) 5 MiB | 1 | 2bx (6291456, 2, 16, 128, 1) | unchanged |
| c40 | pooler col AG (dim -1) | (320, 512) 320 KiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c40 | unpooler col AG (dim -1) | (320, 1024) 640 KiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c40 | draft AR (5 layers x 2 + emb) | (320, 8192) 5 MiB | 11 | 2bx (6291456, 2, 16, 128, 1) | unchanged |
| c48 | attn-out RS | (384, 8192) 6 MiB | 62 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| c48 | moe-partial RS | (384, 4096) 3 MiB | 62 | RS (4194304, 1048576, 4, 256, 1, 0) | RS bx table: 8 x 256 (was 4 x 256) |
| c48 | QKV-in AG (dim0) | (48, 8192) 768 KiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c48 | packed-latent AG (dim0, u8) | (48, 2304) 108 KiB | 60 | AG (131072, 2, 32, 1024, 1) | unchanged |
| c48 | bf16-latent AG (dim0, layers 0/61) | (48, 4096) 384 KiB | 2 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c48 | routes AG (dim0, i32) | (48, 16) 3 KiB | 62 | AG (4096, 1, 8, 128, 1) | unchanged |
| c48 | embedding AR | (384, 8192) 6 MiB | 1 | 2bx (6291456, 2, 16, 128, 1) | unchanged |
| c48 | draft AR (5 layers x 2 + emb) | (384, 8192) 6 MiB | 11 | 2bx (6291456, 2, 16, 128, 1) | unchanged |
| c64 | attn-out RS | (512, 8192) 8 MiB | 62 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| c64 | moe-partial RS | (512, 4096) 4 MiB | 62 | RS (4194304, 1048576, 4, 256, 1, 0) | RS bx table: 8 x 256 (was 4 x 256) |
| c64 | QKV-in AG (dim0) | (64, 8192) 1 MiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c64 | packed-latent AG (dim0, u8) | (64, 2304) 144 KiB | 60 | AG (262144, 2, 32, 1024, 1) | unchanged |
| c64 | bf16-latent AG (dim0, layers 0/61) | (64, 4096) 512 KiB | 2 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c64 | routes AG (dim0, i32) | (64, 16) 4 KiB | 62 | AG (4096, 1, 8, 128, 1) | unchanged |
| c64 | embedding AR | (512, 8192) 8 MiB | 1 | 2bx (8388608, 2, 16, 128, 1) | unchanged |
| c64 | draft AR (5 layers x 2 + emb) | (512, 8192) 8 MiB | 11 | 2bx (8388608, 2, 16, 128, 1) | unchanged |
| c128 | attn-out RS | (1024, 8192) 16 MiB | 62 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| c128 | moe-partial RS | (1024, 4096) 8 MiB | 62 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| c128 | QKV-in AG (dim0) | (128, 8192) 2 MiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c128 | packed-latent AG (dim0, u8) | (128, 2304) 288 KiB | 60 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c128 | bf16-latent AG (dim0, layers 0/61) | (128, 4096) 1 MiB | 2 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| c128 | routes AG (dim0, i32) | (128, 16) 8 KiB | 62 | AG (8192, 1, 64, 32, 1) | unchanged |
| c128 | embedding AR | (1024, 8192) 16 MiB | 1 | AR_LARGE (4611686018427387904, 1, 1048576, 4, 8, 512, 2, 0, 1) | 2bx slabs of <= 4 MiB (was the large algorithm) |
| c128 | draft AR (5 layers x 2 + emb) | (1024, 8192) 16 MiB | 11 | AR_LARGE (4611686018427387904, 1, 1048576, 4, 8, 512, 2, 0, 1) | 2bx slabs of <= 4 MiB (was the large algorithm) |
| prefill8192 | attn-out RS | (8192, 8192) 128 MiB | 62 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| prefill8192 | moe-partial RS | (8192, 4096) 64 MiB | 62 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| prefill8192 | QKV-in AG | (1024, 8192) 16 MiB | 62 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| prefill8192 | packed-latent AG | (1024, 2304) 2.25 MiB | 60 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| prefill8192 | routes AG | (1024, 16) 64 KiB | 62 | AG (65536, 2, 16, 256, 1) | unchanged |
| prefill8192 | embedding AR | (8192, 8192) 128 MiB | 1 | AR_LARGE (4611686018427387904, 1, 1048576, 4, 8, 512, 2, 0, 1) | unchanged |
| prefill4096 (r2e half) | attn-out RS | (4096, 8192) 64 MiB | 124 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| prefill4096 (r2e half) | moe-partial RS | (4096, 4096) 32 MiB | 124 | RS (4611686018427387904, 1048576, 8, 256, 1, 0) | unchanged |
| prefill4096 (r2e half) | QKV-in AG | (512, 8192) 8 MiB | 124 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |
| prefill4096 (r2e half) | packed-latent AG | (512, 2304) 1.12 MiB | 120 | AGL (4611686018427387904, 524288, 8, 1024, 1) | unchanged |

### 3.1 Slab-path shapes beyond the band table

The slab path (8 MiB < bytes <= 64 MiB) is reached by far more shapes than the c96 / c128 draft all-reduce of the table above. The round-5 op log (distinct shapes only) holds about 405 distinct
`[T, 8192]` bf16 all-reduce shapes in that range, at odd T from 513 to 4096: the remainder chunks of chunked prefill and the mixed prefill + decode steps (embedding all-reduce and the draft rows
of the same step). Examples: `[519, 8192]` 8.1 MiB (3 slabs of 173 rows), `[1001, 8192]` 15.6 MiB (4 slabs of 251 / 251 / 251 / 248), `[2049, 8192]` 32 MiB (9 slabs of 228 / 225); `[4096, 8192]` = exactly 64 MiB sits at
the cap (16 slabs of 256 rows; one row more, `[4097, 8192]`, stays on the large algorithm). The rows of 16 KiB make every row count a whole number of 64-byte packs, so slab boundaries are free
and the arithmetic per element does not depend on them (rows are independent); the review asked for the uneven splits to be checked on 8 GPUs, and `bench/test_a1_tables.py` now does (section 7).
The calls per step of these shapes are not counted in the band tables of this section.

## 4. Method and the bitwise proof

* `bench/bench_a1_tables.py` (8 processes, one per GPU, TopoComm with the bf16 exchange, both islands run; the headline is the max over ranks, island means are
  reported). Three timing protocols, all CUDA graphs (the c16 / c32 bands replay graphs; the eager bands launch the same kernels):
  `warm` = N identical calls back to back (the protocol of the existing tables), `spacer` = a 50 us sleep kernel in front of each call, minus the sleep,
  `cold` = a 256 MiB read (L2 clean-cold), a copy that re-produces the input (hot and dirty in L2 like a GEMM output), the aligning tiny all-reduce, the call,
  minus the same graph without the call. The memset-cold protocol of the L8 spike (`coldw`) is reported too.
* Candidates per cell: all_reduce 2bx (grid x block x unroll, 61 .. 108 per size), large bx variants, row slabs (n sub-calls through the table); reduce_scatter
  (chunk bytes x blocks x threads x unroll x cut, 72 .. 252 per size); all_gather (large engine: chunk x blocks x threads x cut; flag and LL geometries).
* Every candidate is compared bitwise on all 8 ranks with the torch emulation of the contract (fp32 chain per island ascending, both island partials rounded to
  bf16, one final rounding; all_gather: the concatenation) before it is timed; finalists are re-checked on four data kinds (randn, mixed outliers, mostly -0.0,
  +-0 mixes) and timed in three interleaved rounds against the dispatch baseline.
* **Adoption rule (written down before the first sweep)**: a candidate replaces a cell only if (a) it is bitwise equal on every kind, (b) the model-like cold
  time improves in each of the 3 rounds by more than max(1.5 us, 3 %, 2x the baseline's own round spread) and on both islands, (c) spacer is not slower by more
  than max(1 us, 1.5 %), warm not by more than max(1 us, 2 %), memset-cold not by more than max(1.5 us, 3 %). Cold and spacer disagreeing beyond that means no change.
* Sizes between measured cells: the 2bx rows for 1 MiB, 1.5 MiB and 2 MiB are all 8 x 512, so 1.25 / 1.75 MiB sit between two measured cells of the same geometry; the reduce-scatter table is one
  geometry (8 x 256) for every size, measured at 1, 1.5, 2, 3, 4, 6, 8, 12, 16 MiB and the prefill sizes, so a size between two measured ones gets the geometry of both neighbours.
* **Confirmation through the dispatch path** (`--mode dispatch`, the code the plugin calls): host main a7e7a03 and the new tree measured in interleaved launches
  (before, after, before, after; 3 rounds each), with a SHA-256 of the output of every rank for the four data kinds. Result: every size of the table below
  has **identical digests before and after on all ranks and all kinds** (bitwise identical outputs) and equals the torch emulation of the contract.
  One more confirmation pair on the final tree (after reverting the two rows of 5.4) is in section 7; the sizes just above `max_bytes` (target group `arm`: 520, 600, 700, 1001, 2049 rows) were added after the review (5.3).

Both islands ({0..3}, {4..7}) are timed: the collectives run in lock step across the cut, and the island means of every changed cell agree within 0.1 us
(e.g. 4 MiB reduce-scatter warm 115.3 -> 88.1 us on both islands, 16 MiB all-reduce 740.8 / 740.9 -> 655.8 / 655.8), so the tables below use the max over ranks.

Changed cells (median of 2 launches x 3 rounds per side, max over ranks, us per call; negative delta = faster):

```
target                    KiB |    warm   spacer     cold    coldw (before) | delta after-before warm spacer cold coldw | digests
ar_T128x8192             2048 |     83.3     83.3     94.0     89.6 |     -1.3     -0.8     -4.5    -12.6 | digests IDENTICAL contract ok | GAIN
ar_T192x4096             1536 |     63.4     62.8     70.3     64.6 |     -1.1     -0.4     -3.4     -7.8 | digests IDENTICAL contract ok | GAIN
ar_T256x4096             2048 |     83.6     82.5     94.2     88.5 |     -1.5     -0.2     -4.5    -11.7 | digests IDENTICAL contract ok | GAIN
ar_T768x8192            12288 |    566.0    564.7    592.2    560.7 |    -72.1    -78.4    -66.2    -43.5 | digests IDENTICAL contract ok | GAIN
ar_T1024x8192           16384 |    741.3    740.5    771.7    741.8 |    -85.0    -89.9    -81.7    -64.4 | digests IDENTICAL contract ok | GAIN
ar_T1536x8192           24576 |   1096.5   1094.1   1118.5   1099.7 |   -108.4   -121.2   -101.6    -99.2 | digests IDENTICAL contract ok | GAIN
ar_T2048x8192           32768 |   1451.5   1450.8   1470.9   1459.2 |   -137.5   -151.6   -138.2   -139.1 | digests IDENTICAL contract ok | GAIN
ar_T4096x8192           65536 |   2914.9   2906.0   2891.9   2895.0 |   -298.4   -309.2   -266.3   -276.6 | digests IDENTICAL contract ok | GAIN
rs_T128x8192             2048 |     60.4     59.9     66.2     64.7 |    -13.3    -13.9    -16.3    -20.0 | digests IDENTICAL contract ok | GAIN
rs_T192x4096             1536 |     46.1     46.1     50.7     49.5 |     -9.4    -10.3    -11.6    -14.1 | digests IDENTICAL contract ok | GAIN
rs_T192x8192             3072 |     87.8     87.5     96.6     95.9 |    -20.1    -21.1    -25.4    -31.4 | digests IDENTICAL contract ok | GAIN
rs_T256x4096             2048 |     60.3     60.0     66.2     65.0 |    -13.2    -14.2    -16.1    -20.4 | digests IDENTICAL contract ok | GAIN
rs_T256x8192             4096 |    115.3    114.6    127.3    125.4 |    -27.1    -28.2    -34.3    -41.7 | digests IDENTICAL contract ok | GAIN
rs_T384x4096             3072 |     87.8     87.4     96.3     95.7 |    -20.2    -21.4    -25.1    -31.4 | digests IDENTICAL contract ok | GAIN
rs_T512x4096             4096 |    115.3    114.0    127.1    125.4 |    -27.2    -27.7    -34.3    -42.0 | digests IDENTICAL contract ok | GAIN
```

Controls, unchanged cells (the deltas are the noise floor of the comparison) and the two rows that were tried and reverted (5.4); the 128 MiB all-reduce is above
the 64 MiB slab cap and shows the noise of a 5.8 ms call:

```
ar_T256x8192             4096 |    164.2    162.0    184.7    184.3 |     +0.5     +0.0     +0.5     +1.0 | digests IDENTICAL contract ok | same
ar_T320x8192             5120 |    203.7    202.1    225.1    221.4 |     -0.1     +5.7     -5.1    -25.8 | digests IDENTICAL contract ok | REGRESS(spacer)
ar_T512x8192             8192 |    327.7    321.9    363.8    363.1 |     +0.6     -0.4     -0.3     +0.4 | digests IDENTICAL contract ok | same
ar_T320x4096             2560 |    102.3    101.5    113.7    108.3 |     -0.3     +2.4     -3.2    -10.9 | digests IDENTICAL contract ok | REGRESS(spacer)
ar_T8192x8192          131072 |   5795.6   5793.2   5782.3   5777.3 |     -7.6    -23.7    -17.7    -11.1 | digests IDENTICAL contract ok | same
rs_T512x8192             8192 |    171.0    168.8    178.1    161.8 |     +0.4     -0.4     +0.6     -0.5 | digests IDENTICAL contract ok | same
rs_T1024x4096            8192 |    171.1    168.7    178.3    161.9 |     -0.2     +0.9     -0.2     +0.6 | digests IDENTICAL contract ok | same
rs_T8192x8192          131072 |   2729.1   2724.1   2702.8   2697.0 |     +4.8     -0.7     +2.4     +0.3 | digests IDENTICAL contract ok | same
```

## 5. Findings per operation

### 5.1 reduce_scatter (bf16 exchange)

Every candidate that wins is 8 workers: at 3 MiB 8 x 256, 12 x 256, 16 x 128 and 8 x 512 are within 1 us of each other (67.5 .. 68.4 us against 88.6 for the old
row), at 4 MiB 8 x 256 / 16 x 128 / 8 x 512 within 1.5 us (87.9 .. 89.3 against 116.7); chunk size is irrelevant below 8 MiB (one block of the message fits one
chunk). From 6 MiB up nothing beats the 8 x 256 row beyond noise (6, 8, 12, 16 MiB, prefill 32 / 64 / 128 MiB: |delta| <= 1 % in warm / spacer / cold; 4 workers or the
cut block lose up to 30 us at 64 MiB). The new table is one row: 8 blocks x 256 threads, chunk 1 MiB, no cut, from 128 KiB to the end. Geometry limit found on the way:
a reduce-scatter with CUT blocks needs at least 4 workers (the cut block polls 4 packs per thread ahead of the workers; with 2 or 3 workers the first call ends in a
spin timeout, rank 0 blocks 0 .. 2 code 52 / 53). No table row uses cut; `bench/test_a1_tables.py` checks the rule for every table.

### 5.2 all_reduce 2bx (64 KiB .. 8 MiB)

One geometry, 8 x 512 x 1, wins by 1.1 .. 1.5 us warm and 3.4 .. 4.5 us cold (4 .. 5 %) with equal spacer at 1.5 MiB and 2 MiB; at 64 KiB .. 1 MiB nothing beats the existing
rows (|delta| <= 0.5 us), at 3, 4, 6, 8 MiB 8 x 512 wins cold by 5 .. 13 us and loses spacer by 3 .. 9 us (the old 16 x 128 / 8 x 256 rows stay). 2.5 MiB and 5 MiB (the c40
MoE-out and attn-out all-reduces) passed the sweep rule (cold -4.4 / -9.0 us) but the dispatch confirmation showed spacer +2.4 / +5.7 us (2 launches), so their two
trial rows were reverted: the cold gain is a cold-start effect that costs steady-state spacing (cold: -3.2 / -5.1 us, memset-cold -10.9 / -25.8 us, spacer +2.4 / +5.7 us).
Row chunking of a medium all-reduce (n sub-calls) never won below 8 MiB under the pre-registered rule: measured with `--force`, 4 MiB as 4 x 1 MiB is warm +7.0 / spacer +7.4 / cold -3.4 /
memset-cold -14.3 us against the single call (2 x 2 MiB: +3.1 / +3.2 / +3.9 / -0.4), 2 MiB as 4 x 512 KiB +6.1 / +6.7 / +0.8 / -3.2 (2 x 1 MiB +1.4 / +1.8 / -2.1 / -11.0): it only wins when the
whole call is cold (the L8 protocol); a collective in the middle of a layer chain is not cold in that sense. The harness-to-harness gap on the same 4 MiB call (memset-cold 176.2 us in section 2,
184.3 us in the dispatch table above) is about 8 us, larger than the effects at 1.5 / 2 MiB, and which protocol matches an all-reduce behind an O-projection GEMM that left its output hot in L2 is
undecided: the four-1-MiB lever of L8 is therefore not adopted here and not gated as an arm (it would need its own forced 4 x 1 MiB A/B at c32; it belongs to the R2 work, not to a table row).

The 1.5 / 2 MiB rows (the c16 attn-out and c32 MoE-out all-reduces, and the draft all-reduces) are worth at most 0.1 ms per step (0.1 .. 0.2 %): warm -1.1 .. -1.7 us and spacer -0.2 .. -0.8 us in the first pair, inside the
pair-to-pair drift of the unchanged controls (-1.5 .. -2.1 us warm at 5 .. 8 MiB, -1.0 .. -1.8 us spacer at 64 .. 512 KiB); only the cold protocols (-3.4 .. -4.9 us, about 5 %) exceed it. In the graph bands (T <= 256) the host
plugin's L2 weight prefetch (`AFTER_EXPERTS_CTAS=8`) runs on a side stream beside the MoE-out all-reduce, which the microbench does not model. The rows are kept only if the model A/B shows no regression at c16 / c32 (section 8: STEADY c16 and
c32 must not be worse than +0.3 % beyond noise); the revert is two table rows.

### 5.3 all_reduce above `max_bytes`: 2bx slabs

With the bf16 exchange algorithm 2 (island partials exchanged once) beats the large algorithm (which also all-gathers the result over the cut) per MiB: 4 MiB = 161 us
(40.3 us / MiB) against 744 us for 16 MiB on the large algorithm (46.5 us / MiB). A message of 8 MiB .. 64 MiB is therefore served as `ceil(bytes / 4 MiB)` row
slabs, each through `_AR_BX_TABLE` at its own size; the slabs are bitwise equal to the large algorithm with the exchange (same arithmetic per element, rows are
independent; the digests above). Conditions (all functions of shape / dtype / settings, so identical on every rank): bf16 with the bf16 exchange and the medium exchange
on, `max_bytes < bytes <= 64 MiB`, 2-D or more with rows of whole 64-byte packs, `bytes < ar_large_min_bx` (a tuning file that lowers it keeps its large algorithm), slabs within
`max_bytes`. The plugin always passes a fresh output buffer (never aliasing the input); in the test an `out=` that is aligned, at a 64-byte offset or misaligned by 2 bytes (staged through a copy) gives the same bits, and the
words around it stay untouched. Each slab is an ordinary 2bx call. CPU: the large pipeline is one C++ call (`ceil(bytes / 8 MiB) + 1` kernel launches inside it), the slabs are a Python loop with one iteration per slab
(two slice views, a table pick, the pybind launch): an estimate of 15 .. 25 us per slab (not measured: no torch on the machine that wrote this), i.e. about 0.3 .. 0.6 ms per eager c128 step (11 draft all-reduces of 4 slabs, plus 1 for
the embedding) and about 0.3 .. 0.4 ms for a 64 MiB prefill all-reduce (16 slabs). The R2 gate probe measured the CPU slack of the eager verify steps (S1-9 L9 R2_GATE, pure verify steps, no profiler): c64 about 12 ms,
c80 about 20 ms, c96 about 35 ms, c112 about 45 ms, c128 about 57 ms, so the slab loop fits where it runs (c65 and above); the model A/B of section 8 measures it directly (STEADY c128). If the CPU ever binds, the loop moves into a
C++ binding next to `all_reduce_large` (one call, atomic under the GIL). The cap is the measured range (64 MiB); the 128 MiB prefill embedding all-reduce (once per step) stays on the large algorithm.

Sizes just above `max_bytes` were measured after the review (dispatch pair, B A B A, 3 protocols, digests identical on all ranks and kinds): 520 rows (8.1 MiB, 3 slabs of 2.7 MiB) -66 us warm / -61 cold (402.7 against
469 us), 600 rows (9.4 MiB) -64 / -60, 700 rows (10.9 MiB) -73 / -62, 1001 rows (15.6 MiB, 4 uneven slabs) -86 / -85, 2049 rows (32 MiB, 9 slabs) -149 / -146: the slab path wins by 12 .. 14 % over the whole range, the 2.7 .. 3.9 MiB
slabs of the c65 .. c95 band included.

**Deviation from the pre-registered rule.** The rule written before the sweeps said that a row-chunked all-reduce is adoptable "only inside CUDA-graph bands (T <= 256)". The slab path
is taken in the eager c96 / c128 bands anyway: the baseline it replaces is already a multi-launch pipeline (`chunks + 1` = 3 launches at 16 MiB), so the net cost is one launch
(about 10 us of CPU) against 85 us of GPU time saved per call, and the eager c128 band has CPU slack (S1-9 L8: 12 .. 30 ms per step). It is a judgement call about CPU, not covered by
the microbench (which measures GPU time in graphs).

### 5.4 Tried, not adopted

* reduce_scatter grid / chunk / cut / unroll variants above 4 MiB, all-reduce large-bx variants (mode 0 / 1, chunk 512 K / 1 M, rs / ag grids, cut) at 12 .. 64 MiB: within noise
  of the baseline or worse (up to +70 us at 128 MiB).
* all_reduce 2bx at 3 / 4 / 6 / 8 MiB and 2.5 / 5 MiB: see 5.2 (cold wins, spacer loses).
* all_gather: the large engine (`_AGL_TABLE`, 8 blocks x 1024 threads, chunk 512 KiB, cut) is on a plateau for the shards that matter (c64 1 MiB, c128 2 MiB QKV-input gathers:
  best candidate within 1.5 %). Isolated wins (a 12-block geometry at 768 KiB -8.8 us, at 216 KiB packed latent -4.9 us, at 384 KiB column shard -5.2 us) do not extend to the
  neighbouring sizes (640 / 896 / 1152 / 1280 / 1792 KiB shards: nothing), they depend on how the shard splits into whole 1024-thread rounds, and the in-band sizes of c64 / c128
  gain 1.6 / 3.4 us per call on 60 calls (0.1 / 0.2 ms per step). The flag / LL rows below 128 KiB are at their optimum (|delta| <= 0.5 us). No all_gather row was changed.
* candidates with `g32` blocks in the large all-gather: -9 .. -12 us cold, +9 .. +23 us warm / spacer; rejected by the rule.

## 6. Estimated gain per band (microbench, not a model measurement)

Sum over the changed cells of (calls per step from section 3) x (dispatch delta). Negative = faster. Columns: ms per step with the warm / spacer / cold / memset-cold
protocol, then (after the bar) the cold time of the same changed calls before the change, ms per step:

```
c16                      -0.093 / -0.060 / -0.331 / -0.934 | 6.96
c32                      -0.095 / -0.013 / -0.282 / -0.726 | 5.84
c48                      -1.253 / -1.329 / -1.558 / -1.949 | 5.97
c64                      -1.686 / -1.714 / -2.126 / -2.604 | 7.88
c128                     -1.020 / -1.079 / -0.981 / -0.773 | 9.26
```

The STEADY step times of the round-5 profile: c16 55.24 ms, c32 82.46, c40 96.18, c64 106.05, c128 152.60 (c48 ~100 ms, not profiled). So the expected gain is c16 0.2 .. 0.6 %,
c32 0.1 .. 0.3 %, c48 1.3 .. 1.6 %, c64 1.6 .. 2.0 %, c128 0.65 % (c40 and c1 .. c8: no change). These are GPU-time deltas of the collective kernels on the critical
path: the microbench aligns the ranks, so it has no expert-parallel straggler skew (a collective that waits for a slow rank in the model does not shorten that wait: the
c64 MoE-partial reduce-scatter is the call that waits for the experts: the round-5 trace attributes 29.6 ms per step (about 477 us per call) to it against 127 us in isolation, so
roughly 350 us of each call is wait for the slowest rank's experts, and how much of the -1.7 ms survives that wait is what a model A/B decides), and the c96 / c128 slabs add CPU launch time
on eager steps. The c128 draft all-reduce gain is the one more likely to be realized in full (the draft GEMMs are rank-symmetric, no expert wait in front of it). A model-level A/B (STEADY
c48 / c64 / c128, spec profile) is the check; it was run afterwards and is reported in section 8.

## 7. Tests and reproduction

All on 8 GPUs under the GPU lock, server mirror copy of the new host tree (`work/s1-9/L9/a1/host_a1x`, never the production `host_topo`), vLLM tree 6da60b3062 (deployed `sm120-dev/src`):

| test | result | what it covers for A1 |
|---|---|---|
| `bench/test_a1_tables.py` (groups tables, ag, graph, ar, rs; bf16 exchange + medium exchange on) | 430 checks x 8 ranks, 0 failures (tables+ag 75, graph 6, ar 252, rs 97) before the review; 709 checks x 8 ranks, 0 failures after it (slab path at odd T 513 .. 4097, `out=` variants, slab == large algorithm, graphs c96 / odd-T slabs; section 8) | dispatch == torch emulation of the contract on four data kinds at every production size and every table bound, rank-identical, batch invariant, RS == rows of AR, all_gather == concatenation, CUDA-graph replays == eager, table well-formedness |
| `bench/test_medium_bx.py` (all groups; the `disp` expectation for `max_bytes + 128 B` now names the slab path) | 192 checks x 8 ranks, 0 failed | medium bf16 exchange: emulation, bits, determinism, dispatch, mixed sequences in graphs, RS == AR rows |
| `bench/tests_h1b.py` (exact exchange pinned: `MUSE_TOPO_LARGE_BF16_EXCHANGE=0`) | 284 checks x 8 ranks, 0 failed | large algorithms and exact-exchange tables (regression of the paths that did not change) |
| `bench/test_plugin.py --quick` (pins the exact exchange itself) | 175 checks x 8 ranks, 0 failures | the vLLM wrapper path of the unchanged exact exchange (regression, not coverage of the new bf16 tables) |
| vLLM SP file `tests/kernels/moe/test_muse_spark_sm120_sequence_parallel.py`, new host tree on PYTHONPATH with `env/host.env` | 75 passed, 2 failed | the 2 failures are `test_sharded_branch_matches_the_replicated_one[2]` and `..._on_an_expert_placement_matches_the_linear_one[2]` (world 2: `PyNcclCommunicator` init, "NCCL error: unhandled system error" under the host.env NCCL settings); the **same two fail identically on the unchanged main tree** (main copy with the shim: 2 failed, 75 passed; main copy with host.env and only these two tests: 2 failed), and the earlier S1-9 runs without host.env passed 77 / 77 (not re-run here) |
| the same SP file with the muse_topo plugin force-registered (a scratch `sitecustomize` shim, server only) | 75 passed, 2 failed (same two) | `TopoComm active` on 8 ranks in the world-8 cases; reduce_scatter `[T, 512]` (T = 264 .. 1024, up to 1 MiB), all_gather `[R, 512]` / `[R, 16]` int32 and all_reduce `[700, 512]` ran through the topo path (1296 op lines); these are small shapes, so the new 1.5 .. 4 MiB reduce-scatter row is not exercised by this file |
| vLLM r2e file `tests/kernels/moe/test_muse_spark_sm120_prefill_overlap.py` (`MUSE_R2E_REAL_CONFIG=1`) | 61 passed (also with the shim) | no collective of the plugin runs in this file (0 op lines): it passed on the production vLLM tree with the new host dir on PYTHONPATH |
| dispatch before / after on the **final tree** (main copy vs new tree, one launch each, 3 rounds, 39 all-reduce / reduce-scatter sizes x 4 data kinds) | digests IDENTICAL on all ranks and kinds for all 39 sizes, contract ok, verdict 15 GAIN / 24 same / 0 REGRESS | the two reverted rows (2.5 MiB, 5 MiB all-reduce) read "same" again |

The host-path coverage of the new tables is therefore `test_a1_tables.py`, `test_medium_bx.py` and the per-size digest comparison above; the SP / r2e files confirm that the vLLM tree is unaffected by the new host tree
(they do not reach the retuned rows). The first run of `test_a1_tables.py` stopped after 6 s on a missing `free` import in the test itself (fixed, rerun above).


Reproduce (server, one GPU lock hold; `HOST_DIR` = a host folder copy, never the production `host_topo`):

```
GPU=0,1,2,3,4,5,6,7 HOST_DIR=<host copy> <host copy>/scripts/runsrc_topo.sh <host copy>/bench/bench_a1_tables.py --mode sweep --what ar,arl,rs,ag --out <jsonl> [--only <substr,...>] [--force 'chunked x4']
GPU=0,1,2,3,4,5,6,7 HOST_DIR=<host copy> <host copy>/scripts/runsrc_topo.sh <host copy>/bench/bench_a1_tables.py --mode dispatch --tag <label> --what ar,arl,rs --out <jsonl>
python3 bench/summ_a1.py sweep <jsonl>                                  # adoption rule per finalist
python3 bench/summ_a1.py cmp <jsonl> <jsonl> --tag-before a,b --tag-after c,d   # before / after per size, digests, verdict
python3 bench/ms_step_a1.py bench/a1_sizes.json <jsonl> <jsonl> --tag-before a,b --tag-after c,d --targets <cells>
```

## 8. Model-level gate (S1-9 L9 A1 gate, 2026-11-04 / 05): bitwise and hang-free, NOT adopted by the pre-registered rule

Candidate = this branch at 58e9b9459d63 (`muse_topo/collectives.py` md5 43e85025db32, unchanged since 5a02275); OFF = production host tree a7e7a03 (server `host_topo`, identical to git main); both arms on the production vLLM tree
6da60b3062 (all 2688 tracked `vllm/*.py` files of the commit identical on the server), the production speculative launch line (RadixArk DFlash2 K = 7, FP8 draft level 2, cap 256, MNBT 8199) at the same exact bytes
11,573,411,840 (1,629,269 KV tokens per rank, idle 94,165 MiB in all six boots, 0 OOM, 0 preemptions; no S1-7 re-derivation: a slab call writes the same one `out` tensor the large algorithm allocates and the workspace is unchanged).
The rule was written down before the first gate boot (ADOPTION_RULE.txt, md5 bcaf82a890e1; `work/s1-9/L9/a1g/g` on the server, tags `s19a1g_{ON,OFF}_{1,2,3}`, order ON OFF OFF ON ON OFF, one GPU-lock hold).

| gate | result |
|---|---|
| tests on the candidate mirror | `test_a1_tables.py` 709 checks x 8, `test_medium_bx.py` 192 x 8, `test_plugin.py --quick` 175 x 8 (PASSED), `tests_h1b.py` (exact exchange, 20000 replays) 284 x 8: 0 failures |
| bitwise equality with production and run-to-run determinism | offline LLM TP8 + EP8, production speculative profile, 131 prompts of 9 .. 23,000 tokens (110 of 90 tokens so that 128 requests run at once: verify batches of 1024 rows, draft all-reduce 16 MiB), 128 new tokens, 12 passes (greedy, seeded t1k64 / top_p / mixed, penalties, repeats): three cold processes (candidate a, candidate b, production): **all 36 comparisons identical** (token ids, top-5 logprobs of every generated token, prompt logprobs): a == b, a == production, pass 1 == repeat in each. Path proof from the DEBUG lines of `vllm.muse_topo`: the candidate runs 688 "2bx slabs" lines (T = 520 .. 1253 and 3819 .. 4079 among others), production 0 slab lines and 712 "large" all-reduce lines |
| 30-minute hang soak (`test_a1_tables.py --groups soak`) | 188,600 graph replays of one graph with 16 collectives (RS 1.5 .. 4 MiB, AR 1.5 / 2 MiB rows, slabs T = 519 .. 4096, the first size above the cap) + 943 eager rounds in 1801 s, every result == the torch emulation, exit 0, no spin timeout |
| speculative A/B, 3 sessions per arm, all six boots healthy (armOK, cfgcheck OK) | **pre-registered primary NOT met: pooled primary score (STEADY c16 / c32 / c40 + real greedy / default tok/s c16 / c32 / c40 / c48 / c64) gain -0.04 % against the required +0.25 %** (no primary cell tripped its limit, every guard passed) |

STEADY step ms (engine counters, constant batch; OFF -> ON, 3 sessions each; "SIG" = beyond noise, Welch t = 2.78):

```
c1     16.934 -> 16.921  +0.08 % noise      c16   55.233 -> 54.769  -0.464 ms  +0.85 % SIG    c32   81.612 -> 81.556  +0.07 % noise
c40    95.034 -> 95.036   0.00 % noise      c48   93.079 -> 93.561  +0.482 ms  -0.51 % noise   c64  105.939 -> 103.804  -2.135 ms  +2.06 % SIG
c128  153.272 -> 151.985  -1.287 ms  +0.85 % SIG
TTFT 8k / 16k / 32k  +-0.1 % noise;  churn c64 +0.06 %;  natural 8192 c16 +0.13 %;  real-prompt tok/s (14 cells): every |gain| <= 0.76 %, none beyond noise
```

Reading: the microbench predictions hold where they were predicted to hold on the critical path: c64 -2.135 ms (predicted -1.7 .. -2.1), c128 -1.287 ms (predicted -1.0, the draft all-reduce slabs with no CPU penalty visible),
c16 -0.464 ms (predicted -0.09 .. -0.33). Nothing reached the real-prompt throughput cells (their session sd is 0.3 .. 1 %), which carried 10 of the 13 primary cells, and c48 (T = 384, the sequence-parallel floor, reduce-scatter 3 MiB) went the
other way within noise (STEADY +0.48 ms, real greedy -0.76 %, default -0.50 %; predicted -1.3 .. -1.6 ms): look at c48 before any re-gate. The non-speculative guard was started (ON_1 of 6 boots) and stopped after the speculative
primary had fixed the verdict; its ADOPT=no is "fewer than 3 sessions", not a guard trip. Not merged: the branch stays at 58e9b9459d63 plus this section for a decision on a re-gate (for example a STEADY-based primary written down in advance,
or more sessions on the real-prompt cells), the revert of the 1.5 / 2 MiB rows is not needed (STEADY c16 improved, c32 flat).
