# Medium-size all-reduce (64 KiB .. 8 MiB) with the bf16 exchange (track r3c-medium-ar)

At decode batch sizes of 16 and more the two all-reduces of a layer are no longer latency-bound: the attention output
`[T, 8192]` bf16 is 256 KiB (T = 16) .. 2 MiB (T = 128) and the MoE output `[T, 4096]` 128 KiB .. 1 MiB. In the real-weight c64 decode
profile (`profile_r2fin_real/c64dec_report.txt`) the attn-out all-reduce of 1 MiB took 62.8 us per layer (3.9 ms per step) and the
algorithm was bound by the socket cut: the 3-phase algorithm 2 (island reduce-scatter, partner exchange of the fp32 island
partials, island all-gather) sends 2 MiB per direction over the cut for a 1 MiB message, and the cut carries ~39 GB/s per direction
for every transport (`docs/h1b_latency.md`).

## What changed

* **Algorithm 2 gets the bf16 exchange** (`csrc/ar_sll.cu`, kernel variant `ar2_kernel<D, U, BX>`): both island partials are rounded
  to bf16 before the final add, so the partner exchange is one 16 B pack per input pack instead of two and the cut carries half the
  bytes (1 MiB per direction for a 1 MiB message). Per element: `out = round_bf16(bf16(P_0) + bf16(P_1))`, `P_I` the fp32 chain of the
  island in ascending rank order, exactly the BX path of the large algorithm (`large.cu`, `rs_body`, fp32 NaN canonicalization
  included), so **the medium and the large algorithm give the same bits** (checked on random bit patterns, NaN/inf included).
* **Dispatch** (`collectives.py`): with the large-message flag on (`MUSE_TOPO_LARGE_BF16_EXCHANGE=1`, the `env/host.env` default) and
  `MUSE_TOPO_MEDIUM_BX` not 0, a bf16 `all_reduce` of at least `MUSE_TOPO_MEDIUM_BX_MIN_BYTES` (default 64 KiB) takes algorithm 2 with
  the bf16 exchange, geometry from `_AR_BX_TABLE` (one row per measured size). fp16 / fp32 keep the exact fp32 exchange. The large
  algorithm now only takes the messages above `max_bytes` (8 MiB): with the bf16 exchange algorithm 2 is faster at every size up to
  there (table below).
* `bench/bench_medium.py` (sweep of grid x block x unroll per size, verified bitwise against the torch emulation, plus the
  dispatch path with the medium exchange off / on), `bench/make_table_medium.py` (rows -> table + thresholds),
  `bench/test_medium_bx.py` (correctness), `scripts/r3c_chain.sh` / `scripts/conc_sweep.sh` (the one-lock measurement sequence).
* Unchanged: the fused all-reduce + boundary kernel (`ar_bnd.cu`, T <= 8, attn-out) keeps the exact fp32 exchange and the dispatch
  limits of r2d; the model-side hook is untouched (no vLLM change). Consequence: the attn-out reduction at T = 4 .. 8 (64 .. 128 KiB) is
  exact only while the fused kernel serves it. With the fused kernels off (`MUSE_TOPO_FUSED=0`, `MUSE_TOPO_OPS` without `boundary`,
  or `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=0`) it goes through `comm.all_reduce` and takes the bf16 exchange, so fused on vs off is
  no longer bit-identical for T = 4 .. 8 (`docs/r2d_fused_epilogue.md`). A bit-identical fused on / off A/B needs
  `MUSE_TOPO_MEDIUM_BX=0` in both runs.
* `MUSE_TOPO_AR_TUNING_FILE=<json>` is an experiment hook: `bx_table` (rows of `_AR_BX_TABLE`), `bx_min`, `large_min_bx` override the
  built-in values (this is how the measured table was exercised end to end before it was compiled in). Like `MUSE_TOPO_MEDIUM_BX` and
  `MUSE_TOPO_MEDIUM_BX_MIN_BYTES` it must be identical on every rank; a bad integer or an unreadable file on some rank is reported
  through the stage-1 agreement of `TopoComm.__init__`, so every rank raises `TopoUnavailable` (`bench/test_init_failure.py`, case `tuning`).

## Accuracy class: deterministic_different

Reproducible run to run, identical on all ranks, equal to a torch emulation bit for bit; rounding-level difference to the exact
path, no systematic bias (round to nearest even on both partials). Evidence (`bench/test_medium_bx.py`, 8 ranks, real GPUs):

* ar2 BX == torch emulation (`bf16(P_0)`, `bf16(P_1)` added in fp32, rounded once) for 64 KiB .. 4 MiB, randn / mixed-exponent / signed-zero
  data, mostly -0.0 data (`negzeros`: 43 % of the results are -0.0 at every size, so the sign-of-zero path is exercised from 64 KiB on)
  and 7 launch geometries (grid 1 .. 188, block 32 .. 512, unroll 1 .. 4); the BX result differs from the exact path (not a no-op).
* dispatch (`comm.all_reduce`) from 32 KiB to 8 MiB and at the `max_bytes` edge: exactly 8 MiB still takes algorithm 2 with BX, 8 MiB +
  128 B takes the large algorithm; every size is checked for its bits and for the algorithm that ran (medium BX on / off / exchange off).
* random bit patterns (NaN, inf, subnormals, all-ones words): ar2 BX == large BX, fused and two-launch, bitwise, for two geometries.
* rank-identical (sha256 over ranks), run-to-run (5 runs), eager == CUDA graph replay, batch invariant (the rows of the first half alone
  equal the rows of the full batch), 4 mixed sequences of 40 all-reduce sizes + all-gathers in one graph (counter alignment, 3 replays),
  and decode-layer sequences that interleave the BX all-reduce with the fused `all_reduce_boundary` / `all_gather_rmsnorm` /
  `all_gather_boundary` kernels on the shared regions (the real decode order: embedding all-reduce, 6 layers of attn-out + boundary,
  pooler gather + RMSNorm, MoE-out all-reduce, unpooler gather + boundary, logits gather; T = 8 and T = 4 and 3 random mixes with
  layers of T in {1, 2, 3, 4, 5, 8} between bank all-reduces up to 16 MiB), eager and after 3 CUDA-graph replays, against the Triton
  kernels applied to the exact contract (fused kernels) and against the BX emulation (all-reduces). 124 hard checks pass (83 before the
  follow-up); `tests_h1a.py --quick` (119 checks) and `tests_h1b.py --quick` (190 checks) pass unchanged.
* error against the fp64 sum, 1 MiB, mean ulp (exact path / BX / NCCL ring): randn 0.24 / 1.10 / 1.98, mixed-exponent 0.25 / 0.79 / 1.45;
  mean signed error -3e-6 and +2e-6 of the mean |value| (no bias). Maximum error is large when the two island partials cancel
  (1920 / 3856 ulp, NCCL's ring: 7168 / 24448), the same property as the large path (`docs/h1b_collectives.md`).
* GSM8K smoke, real weights, production configuration (SP, bf16 exchange for large and medium messages, c64, 200 items x 2 passes
  forward / reverse): **97.5 % / 96.5 %** (`r3c_c1_acc`); the same harness without the medium exchange (`r2integ_acc`): 96.0 % / 97.0 %,
  NCCL 96.2 %. A vs B same final answer 192 / 200 (without: 191 / 200).

Batch-size consequence (same class as the 2 MiB threshold of the large path before): in the decode range the attn-out all-reduce of
T <= 8 is exact while the fused kernel serves it (16 .. 128 KiB; with the fused kernels off, T = 4 .. 8 carries the exchange as well), from T = 9 on it carries the bf16 exchange (in the CUDA-graph decode the padded sizes are 1, 2, 4, 8, 16, 24, ..., so
from T = 16); the MoE-out all-reduce `[T, 4096]` carries it from padded T = 8 (64 KiB). An element's bits therefore depend on the padded batch size at those boundaries, which the dummy end-to-end
bitwise tool cannot be used for; the GSM8K smoke above is the evidence.

## Latency (bf16 `[T, 8192]`, CUDA graph of back-to-back calls, us per call; `bench/bench_medium.py`, 2026-10-26)

b2b = back-to-back, spacer = exposed behind a 50 us kernel (both listed as b2b / spacer). `exact` = algorithm 2 with the best geometry of the
sweep (the previous behaviour), `BX` = algorithm 2 with the bf16 exchange, `large BX` = the large algorithm with the bf16 exchange.
Geometry (grid / block / unroll) of the BX column is `_AR_BX_TABLE`.

| KiB | shape | NCCL b2b | exact | BX | large BX | BX vs exact |
|---|---|---|---|---|---|---|
| 64 | [4, 8192] | 46.1 | 8.1 / 8.0 | 6.7 / 6.6 (8/128/1) | 34.7 / 34.4 | 1.21x |
| 96 | [6, 8192] | 46.2 | 9.9 / 9.8 | 7.7 / 7.7 (6/256/1) | 35.5 / 35.3 | 1.28x |
| 128 | [8, 8192] | 47.3 | 11.9 / 11.7 | 9.0 / 8.9 (8/256/1) | 36.2 / 36.1 | 1.32x |
| 192 | [12, 8192] | 49.6 | 16.1 / 16.1 | 11.5 / 11.3 (6/512/1) | 38.0 / 37.7 | 1.42x |
| 256 | [16, 8192] | 52.5 | 18.9 / 18.8 | 13.8 / 13.6 (64/64/1) | 39.8 / 39.6 | 1.38x |
| 384 | [24, 8192] | 60.5 | 25.8 / 25.7 | 18.6 / 18.7 (12/256/1) | 45.5 / 45.4 | 1.38x |
| 512 | [32, 8192] | 66.8 | 33.3 / 33.0 | 22.5 / 22.7 (16/256/1) | 48.3 / 48.2 | 1.47x |
| 768 | [48, 8192] | 74.1 | 47.7 / 47.6 | 32.8 / 32.6 (8/512/1) | 56.5 / 56.4 | 1.46x |
| 1024 | [64, 8192] | 86.1 | 62.5 / 62.0 | 43.1 / 42.8 (8/512/1) | 64.3 / 64.2 | 1.45x |
| 1536 | [96, 8192] | 101.3 | 94.6 / 94.0 | 63.3 / 63.6 (16/128/1) | 80.7 / 80.7 | 1.49x |
| 2048 | [128, 8192] | 134.8 | 123.7 / 123.4 | 82.8 / 82.9 (16/128/1) | 97.4 / 97.5 | 1.49x |
| 3072 | [192, 8192] | 216.9 | 183.0 / 182.9 | 122.2 / 122.1 (8/256/1) | 161.4 / 161.4 | 1.50x |
| 4096 | [256, 8192] | 290.7 | 243.8 / 240.9 | 161.4 / 160.6 (16/128/1) | 210.8 / 210.9 | 1.51x |
| 6144 | [384, 8192] | 433.1 | 357.9 / 355.1 | 241.0 / 240.4 (16/128/1) | 312.9 / 312.6 | 1.48x |
| 8192 | [512, 8192] | 577.4 | 480.3 / 471.4 | 321.3 / 321.0 (16/128/1) | 364.7 / 364.2 | 1.48x |

The dispatch path (`comm.all_reduce(x, out=out)`, `bench_medium.py --no-sweep`, exact / large-only BX / medium BX, b2b): 1 MiB
62.5 / 62.8 / 43.1, 512 KiB 33.3 / 33.2 / 22.6, 256 KiB 18.5 / 18.5 / 13.7, 2 MiB 118.3 / 100.7 / 82.1, 8 MiB 452 / 375 / 321, every
result bitwise equal to the contract it claims (exact path: arithmetic contract, BX: emulation).

Why more than the bytes suggest: the exact exchange moved 2 MiB per direction over the cut and 1.5 MiB over the islands per GPU for a
1 MiB message; the bf16 exchange moves 1 MiB and 1.5 MiB. 43 us for 1.75 MiB of egress per GPU is ~42 GB/s, the SM-store rate of one GPU
(`docs/h1b_latency.md`, transport table), so algorithm 2 is now at the egress roofline of the byte-optimal decomposition; the large
algorithm is slower because its all-gather half also crosses the cut (1.5 x the bytes).

## Real-weight concurrency (random 512 / 128, `scripts/conc_sweep.sh`, 2026-10-26, one server session per row, 2 repeats, out tok/s / TPOT ms)

| concurrency | medium exchange off (`MUSE_TOPO_MEDIUM_BX=0`, = r2fin) | on (default) | gain | `r2fin_conc_r_topo_512` |
|---|---|---|---|---|
| 1 | 96.3 / 9.63 | 96.7 / 9.58 | +0.5 % (noise: sizes below 64 KiB are untouched) | 96 |
| 16 | 481.1 / 27.41 | 491.5 / 26.43 | **+2.2 %**, TPOT -0.98 ms | 488 |
| 64 | 905.7 / 56.69 | 931.8 / 54.70 | **+2.9 %**, TPOT -1.99 ms | 906 |
| 128 | 1156.5 / 91.79 | 1178.9 / 89.66 | **+1.9 %**, TPOT -2.13 ms | 1159 |

The "off" session reproduces the r2fin rows (c64 906, c128 1159), the plugin summary line shows `medium bf16 exchange off` / `from 65536`
on all 8 ranks of the two sessions, and the serve logs have no spin timeout or fallback line.

## Caveats and follow-ups

* The messages below 64 KiB (c1 .. c4: MoE-out `[T, 4096]` up to 32 KiB, attn-out fused) keep the exact exchange; the gain there is
  small (6.7 vs 8.1 us at 64 KiB, less below) and not measured. Sizes 16 .. 48 KiB are a cheap extension of the sweep.
* The fused all-reduce + boundary kernel (T <= 8) could use the same exchange (cut bytes halved: ~-1.4 us at T = 4, ~-2.9 us at
  T = 8 per layer from the plain-kernel rows); not built here. Estimate (not measured): the plain bf16-exchange all-reduce at T = 8
  (9.0 us) plus the separate ~5.7 us boundary kernel may now be on par with or faster than the fused exact kernel (18.8 us).
* Geometry per size comes from one sweep (3 trials per candidate, top 2 retimed with the full protocol, fewest threads within 1.5 %);
  neighbouring geometries are within 1-3 %. The kernels spin-wait on peers with all blocks co-resident (<= 64 blocks of <= 512 threads),
  same rule as H1a.
* Prefill all-reduces above 8 MiB still use the large algorithm; per MiB it is slower than algorithm 2 with the bf16 exchange at 8 MiB
  (46.6 us/MiB at 16 MiB from `docs/h1b_latency.md` vs 40.1 us/MiB measured here), so a loop of 8 MiB pieces of algorithm 2 might beat it
  by ~14 % (extrapolation, not built or measured). The forced sequence-parallel prefill uses reduce_scatter + all_gather and does not
  call the all-reduce.
