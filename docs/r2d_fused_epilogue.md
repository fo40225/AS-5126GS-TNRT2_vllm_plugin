# Residual boundary and pre-expert norm as the epilogue of the decode collectives (track r2d-fused-epilogue)

A Muse Spark decode layer has three "collective, then one element-wise Triton kernel" pairs on its critical path (rank 0 of the real-weight
profile `profile_v3_real`, 62 layers):

| pair | collective | kernel after it |
|------|------------|-----------------|
| attention output | all-reduce `[T, 8192]` bf16 (`ar2_kernel`) | `_norm_residual_norm_fwd_kernel` (post-attn norm, gated residual, pre-FFN norm) |
| unpooler | all-gather `[T, 1024]` -> `[T, 8192]` (`ag_ll_kernel`) | `_norm_residual_norm_fwd_kernel` (post-FFN norm, gated residual, next pre-attn norm) |
| pooler | all-gather `[T, 512]` -> `[T, 4096]` (`ag_ll_kernel`) | `_rmsnorm_fwd_kernel` (pre-expert norm) |

Each pair costs a launch and a round trip of the collective's result through global memory, and the kernel after the collective loads
the fp32 stream and the per-channel parameters only once the collective is done. This track runs the element-wise kernel as the
epilogue of the collective (`muse_topo` kernels `ar2_bnd_kernel`, `ag_bnd_kernel`), through a generic hook of the vLLM branch
(`vllm/models/muse_spark/nvidia/ops/fused_boundary.py`: optional methods on the TP device communicator; the plugin implements them,
NCCL does not, and the model then keeps the two-step path).

## Accuracy class: identical (bit for bit), by construction and by test

The fused kernels reproduce the Triton kernels bit for bit, so the model's outputs do not change. "Reproduce" is defined against the
**SASS** of the Triton kernels on sm_120a (Triton 3.7.1, ptxas of CUDA 13.3), not against the Python source: ptxas contracts some of the
multiplies of the sum of squares into FMAs, and the reduction tree is the one Triton generates for the block layout. What the proof
established (`csrc/boundary.cuh` has the same description next to the code):

* `_norm_residual_norm_fwd_kernel`, D = 8192, `bT = 1`, `num_warps = 32`, layout `sizePerThread [1, 8]`, `threadsPerWarp [1, 32]`,
  `warpsPerCTA [1, 32]`: thread `t` of 1024 owns the 16-byte pack `t` (elements `8t .. 8t + 7`).
  * thread chain of the sum of squares: `acc = round(x1 * x1); acc = fma(x0, x0, acc); acc = fma(xk, xk, acc)` for `k = 2 .. 7`
    (the PTX looks different: it uses packed `f32x2` ops, and ptxas turns the `mul` + `add` pairs of the odd lanes into FMAs);
  * tree: `xor` butterfly over 32 lanes (offsets 16, 8, 4, 2, 1), the 32 warp partials through shared memory, a second butterfly of the
    same shape (`a + b` is commutative, so every lane ends with the same value);
  * `variance = fma(sum_sq, 2^-13, eps)`, `rsqrt.approx.ftz.f32`, `y = x * rs` (post-norm weight first: `x = x * (f32(w) + gain)`),
    `new = fma(alpha, stream, round(beta * y))`, second norm on `new` the same way, `z = (new * rs2) * (f32(w) + gain)`, `bf16_rn`.
* `_rmsnorm_fwd_kernel`, D = 4096, `bT = 2`, `num_warps = 8` (2 rows per program, 256 threads, thread `t` owns the packs `t` and `256 + t`
  of each row): chain over 16 elements (`round(x1*x1)`, then `fma` for `x0, x2 .. x7` of the first pack and `x0 .. x7` of the second),
  lane butterfly 16..1, 8 warp partials per row, butterfly 4, 2, 1; `fma(sum_sq, 2^-12, eps)`, `rsqrt.approx.ftz.f32`,
  `(x * rs) * (f32(w) + gain)`. ptxas fuses the bf16 widening into `FHFMA.BF16` / `FHADD.BF16` (mixed-precision ops with exactly the
  fp32 semantics).

Everything in `boundary.cuh` is written with explicit `__fmul_rn` / `__fmaf_rn` / `__fadd_rn`, `rsqrt.approx.ftz.f32` as inline PTX and
`cvt.rn.bf16x2`, so nvcc / ptxas cannot contract or reorder it.

The element-wise result of the all-reduce part is the arithmetic contract of `ar2_kernel` (`P_I = ((x_m0 + x_m1) + x_m2) + x_m3` in fp32,
`out = round(P_0 + P_1)`, one final rounding): the fused all-reduce slices a row differently (slice `j` of a row = packs
`256 j .. 256 j + 255`, one CTA per row) but every element goes through the same operations in the same order.

Evidence (server, GPU lock, vLLM tree `src_r2d-fused-epilogue`):

* `bench/test_boundary.py`: the CUDA epilogue math standalone against `fused_norm_residual_norm_fwd` and `fused_rmsnorm_fwd` (the real
  Triton kernels): 150 checks per repetition, 0 mismatches over `--reps 4` (600 checks; the default `--reps 20` is 3000) (T = 1, 2, 3, 8, 24, 64; both boundary variants, with and without the fp32 copy;
  D = 4096; normal, heavy-tailed, tiny, zero and huge data; three eps / gain configurations).
* `bench/test_ar_boundary.py` (8 processes): the fused collectives against `TopoComm.all_reduce` / `all_gather` followed by the Triton
  kernels, bitwise, identical on all ranks, eager and in CUDA graphs next to other collectives (see the results section).
* the model-level glue and the real-weight end-to-end A/B: see the results section.

## What was built

| file | content |
|------|---------|
| `muse_topo/csrc/boundary.cuh` | the bit-exact CUDA mirror of the two Triton kernels (device functions) |
| `muse_topo/csrc/ar_bnd.cu` | `ar2_bnd_kernel`: all-reduce (algorithm 2 protocol, sentinel LL) + boundary, one 256-thread CTA per row |
| `muse_topo/csrc/ag_bnd.cu` | `ag_bnd_kernel`: LL8 all-gather (column concatenation) + boundary (unpooler) or + RMSNorm (pooler); the gathered row is staged in shared memory |
| `muse_topo/collectives.py` | `TopoComm.all_reduce_boundary`, `all_gather_boundary`, `all_gather_rmsnorm`, their `supports_*` gates, `MUSE_TOPO_FUSED` |
| `muse_topo/vllm_plugin.py` | `CudaCommunicator.supports_fused_residual_boundary / all_reduce_residual_boundary / all_gather_residual_boundary / all_gather_rmsnorm` (op name `boundary` in `MUSE_TOPO_OPS`) |
| `bench/test_boundary.py`, `bench/boundary_test.cu` | standalone bitwise proof against Triton |
| `bench/test_ar_boundary.py` | 8-rank bitwise test of the fused collectives |
| `bench/bench_boundary.py` | latency table, fused vs two-step |

Decomposition of the fused all-reduce: the rows of the message are independent, so one CTA of 256 threads handles one row. Slice `j` of a
row is its packs `256 j .. 256 j + 255`; thread `i` handles pack `i` of every slice, which after phase c is exactly the pack layout
(`256 j + i`) of the Triton boundary kernel's 1024 virtual threads (the tree runs over registers: four independent warp butterflies per
thread, one per `j`, then the 32 partials through shared memory). The loads that do not depend on the collective (fp32 stream, alpha,
beta, weights) are issued behind the pushes of phase a.

Gating (`TopoComm.supports_boundary_rows`, rank-uniform, depends on T and tensor properties only): see the results section for the measured
crossover; larger batches keep the two-step path (they are bandwidth bound, and one CTA per row is slower there).

## Results (server, 8x RTX PRO 6000, 2026-10-24; scratch `/raid/vllm-runtime/work/r2d-fused-epilogue/`)

### Correctness (accuracy class identical)

| check | result |
|-------|--------|
| `bench/test_boundary.py` (CUDA epilogue math vs the real Triton kernels, the default `--reps 20`; 150 checks per repetition) | 3000 checks, 0 mismatches (600 at `--reps 4`, the run of the merged tree 2026-10-26) |
| `bench/test_ar_boundary.py` (8 ranks, fused collective vs `TopoComm` collective + Triton kernel; one default run, the count is per rank) | 416 checks per rank (x 8 ranks), 0 failures: T = 1 .. 64, both boundary variants, fp32 copy, eager and CUDA graph next to other collectives, random mixes of collectives, operands misaligned on odd ranks only; also with `MUSE_TOPO_PDL=1` and every `MUSE_TOPO_BND_LOADS` placement |
| regression of the existing suites | `tests_h1a --quick` 119 checks, `test_plugin --quick` 159 checks, `test_plugin_unit` PASSED |
| 6-layer dummy e2e (`nccl/e2e_bitwise.py`, fused on vs off, plugin on in both) | 20/20 requests, every logprob and prompt logprob bitwise equal |
| real weights, 62 layers, **base trees** (vLLM `aef9851844` + host `37b2d96`, no `boundary` op, none of the new code) vs the committed trees (fused on, 2026-10-26, same 270-request script) | 270/270 requests: token ids, top-5 logprobs and prompt logprobs bitwise equal (chosen-token and prompt |dlogprob| max 0, mean NLL delta 0); 118 fused-op log lines in the run with the fusion, none in the base run. This is the check that the whole change (model refactor included) leaves the outputs of the current default untouched |
| real weights, 62 layers (`e2e_ab.py` of the final-measure track + decode batches T = 1, 2, 3, 4, 5, 9, 12, 16, 17, 33, 48, 64) | 270/270 requests: token ids, top-5 logprobs and prompt logprobs bitwise equal fused on vs off; a second fused process equals the first; in-process repeats equal; the log shows `all_reduce_boundary` / `all_gather_boundary` (T = 1, 2, 4, 8) and `all_gather_rmsnorm` (T <= 16) inside the 62-layer CUDA graphs |

NaN payloads: a NaN stays a NaN everywhere; the all-reduce replaces words equal to `0xFFFFFFFF` (as before), so a NaN payload may differ
between runs of different batch compositions only in the case that was already true for the plain all-reduce.

### Latency of the fused kernels (`bench/bench_boundary.py`, `bench/bench_boundary_cold.py`)

Hot: graph of 50 back-to-back calls, operands in L2, max over ranks (us; two-step = collective + Triton kernel):

| T | all-reduce + boundary: two-step / fused | unpooler gather + boundary | pooler gather + RMSNorm |
|---|------------------------------------------|----------------------------|-------------------------|
| 1 | 9.8 / 8.8 | 8.3 / 7.2 | 5.1 / 4.1 |
| 4 | 12.7 / 11.9 | 10.0 / 9.0 | 6.2 / 4.7 |
| 8 | 16.4 / 15.5 | 12.0 / 11.0 | 7.0 / 5.6 |
| 16 | 23.4 / 26.4 | 15.1 / 15.6 | 9.0 / 7.7 |
| 32 | 37.8 / 49.2 | 20.5 / 26.5 | 12.1 / 12.6 |

Cold (a 256 MiB copy before every call, so alpha / beta / weights come from HBM, as in a decode step; profiler timestamps from the
start of the collective to the end of the epilogue; us, two-step / fused):

| T | all-reduce + boundary | unpooler gather + boundary | pooler gather + RMSNorm |
|---|------------------------|----------------------------|-------------------------|
| 1 | 13.4 / 11.8 | 12.2 / 10.2 | 9.0 / 6.8 |
| 2 | 14.7 / 13.1 | 13.0 / 11.2 | 9.3 / 7.1 |
| 4 | 16.9 / 14.4 | 13.8 / 11.5 | 10.1 / 7.1 |
| 8 | 20.9 / 18.6 | 16.0 / 12.9 | 10.7 / 8.5 |
| 16 | 29.3 / 29.3 | n/a | 12.8 / 10.5 |
| 24 / 32 / 64 | 35.7 / 43.6 / 77.3 vs 42.1 / 50.5 / 106.5 (slower) | | |

One CTA per row has no more than a handful of rows to serve: from 16 rows on the tuned many-CTA kernels of the plain collectives win
(the kernels themselves run up to 64 / 32 / 64 rows and are tested there). Deliberate deviation from "T <= 64": the dispatch
(`TopoComm.DISPATCH_MAX_ROWS`) is 8 rows for the all-reduce and the unpooler gather, 16 for the pooler gather.

Operand loads: the operands that do not depend on the collective (about 112 KiB per row) are loaded behind the first push stage
(`MUSE_TOPO_BND_LOADS=0`): that placement was the fastest of five (loads after the collective, L2 prefetch at the start plus loads after
the collective or after the first stage, loads after the partner exchange) in the cold benchmark at T = 1 / 4 / 8.

In-kernel timeline (`bench/trace_boundary.py`, hot, T = 1, all-reduce + boundary, us): start to pushes sent 1.15, peers 1.99, partial sums 0.10,
partner exchange 1.86, island push 0.10 + 1.43, then the epilogue: first tree 0.73, element-wise + stream store 0.83, second tree 0.29,
final stores 0.22 (total 8.9). The epilogue is ~2.1 us of it.

### Full model (real weights, EP8, host plugin, `vllm bench serve` random 512/128, 16..64 prompts, seed 9000)

Mean TPOT in ms; every row is a separate server start; "on" = default (`VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=1`, `MUSE_TOPO_BND_LOADS=0`,
4 runs), "off" = the same tree and host mirror with the flag at 0 (3 runs), runs interleaved (`r2d_conc_*`, `r2d_c2_*` in
`musetest/results/`):

| concurrency (graph T) | on, runs | on mean | off, runs | off mean | on - off |
|-----------------------|----------|---------|-----------|----------|----------|
| 1 | 10.49 10.44 10.45 10.43 | 10.45 | 10.57 10.59 10.60 | 10.59 | -0.14 (-1.3 %) |
| 2 | 13.33 13.33 13.33 13.33 | 13.33 | 13.48 13.48 13.47 | 13.48 | -0.15 (-1.1 %) |
| 4 | 17.04 16.88 16.96 16.92 | 16.95 | 16.91 16.60 16.55 | 16.69 | +0.26 (scheduler mode of the benchmark, not the kernels: see the note under the final rows) |
| 8 | 21.25 20.96 21.08 21.27 | 21.14 | 21.42 21.53 21.44 | 21.46 | -0.32 (-1.5 %) |
| 16 | 32.08 31.25 | 31.7 | 31.75 | 31.75 | within noise (the all-reduce / unpooler kernels do not run at 16 rows, the pooler norm does) |
| 32 | 44.94 45.20 | 45.07 | 45.02 | 45.02 | 0 (no fused kernel runs) |

Reference row `fin_r_topo` c1: 10.56 ms. So at the headline concurrency the gain is about 0.14 ms (1.3 %), well below the 0.4 .. 0.5 ms
that the cold microbenchmark suggests (3 fused ops per layer, 1.6 .. 2.2 us each at T = 1): in the real graph the Triton boundary kernel
is a programmatic dependent that already hides its launch and its parameter loads behind the tail of the collective.

### Final rows (standard protocol: `locked_run_topo.sh`, real weights, seed 9000, mean TPOT in ms)

Same serve arguments as `fin_r_topo` (EP8, fp8 KV, ctx 131072, max-num-seqs 128), c1 = random 512/128 x 16, ctx10 / 50 / 90 = 13 107 / 65 536 /
117 964 input tokens x 3, 256 output tokens, `MUSE_TOPO_ENABLE=1`. "on" = committed trees, defaults; "off" = same trees with
`VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=0`; "base" = vLLM `aef9851844` + host `37b2d96` (the trees of `fin_r_topo`), run in the same session as on / off.
Order of the server starts: on, off (round 1), off, on (round 2), base, on (round 3).

| row | on (3 runs) | on mean | off / base (3 runs: off, off, base) | off / base mean | on - off | `fin_r_topo` (2026-10-24) |
|-----|-------------|---------|--------------------------------------|-----------------|----------|---------------------------|
| c1 512/128 | 10.593 10.571 10.585 | 10.583 (94.5 t/s) | 10.715 10.687 10.696 | 10.699 (93.5 t/s) | -0.116 (-1.1 %) | 10.557 |
| ctx10 | 10.445 10.388 10.377 | 10.403 (96.1 t/s) | 10.553 10.533 10.524 | 10.537 (94.9 t/s) | -0.133 (-1.3 %) | 10.495 |
| ctx50 | 10.417 10.422 10.430 | 10.423 (95.9 t/s) | 10.571 10.577 10.573 | 10.574 (94.6 t/s) | -0.151 (-1.4 %) | 10.537 |
| ctx90 | 10.550 10.595 10.557 | 10.567 (94.6 t/s) | 10.683 10.695 10.687 | 10.688 (93.6 t/s) | -0.121 (-1.1 %) | 10.594 |

The fusion saves 0.12 .. 0.15 ms per decode step (1.1 .. 1.4 %) at every context length, consistently in all three pairs. Prefill is untouched
(prompt throughput 3991 .. 4061 / 6127 .. 6167 / 5935 .. 5942 / 5652 .. 5659 t/s for c1 / ctx10 / ctx50 / ctx90 in all eight runs; prefill batches are far above 8 rows and take the old path). The flag-off path costs nothing: "off" and "base"
agree to 0.01 ms. The sessions differ among each other: the session of 2026-10-26 is ~0.1 ms slower than the one of `fin_r_topo` for the same trees
(c1 base 10.696 vs 10.557), so compare on / off / base inside this table, not against the `fin_r_topo` column.

c4 in the concurrency table: the 512/128 sweep at c4 shows two modes of the scheduler, independent of the flag (mean TTFT 317 ms with TPOT 16.9 .. 17.0,
or 337 ms with TPOT 16.4 .. 16.6; 317-mode: fused on 17.04 / 16.88 / 16.96 / 16.92, off 16.91; 337-mode: off 16.60 / 16.55, fused with the bulk-copy variant
16.41). All default-on runs of the table landed in the 317 mode, so the +0.26 ms in the c4 row is the batch composition of the benchmark, not the kernels.

In-situ profile of a decode step at c1 (torch profiler, rank 0 windows, same tree, fused off / on): step wall 10.79 -> 10.63 ms, per layer
(layers 2..59) 162.2 -> 159.3 us. `_norm_residual_norm_fwd_kernel` (124 launches) and `_rmsnorm_fwd_kernel` (62) disappear (0.69 ms of
kernel time per step); the all-reduce / all-gather families grow by 0.41 ms (the epilogues): net -0.16 ms per step. The fused all-reduce
kernel takes 17.1 us on average in situ (plain all-reduce 13.4 us + boundary 4.8 us before), the fused unpooler gather 9.1 us, the fused
pooler gather 8.6 us. The epilogue's own cost is ~2.1 us at T = 1 (timeline above); about half of what is left of it is the two sums of
squares and the stores of `new_stream` / `out`, the rest the single-CTA load of 112 .. 128 KiB of operands per row.

Evaluated and not adopted: staging alpha / beta / the stream by bulk copy (`cp.async.bulk` into shared memory, completion on an mbarrier,
`ld_mode` 4; patch kept outside the tree). Bitwise identical (416 checks per rank, 270/270 real-weight requests), cold microbenchmark -0.6 us per
op at T = 1 and 0 at T >= 4, no TPOT difference in two interleaved rounds (c1 10.44 vs 10.43, c2 13.30 vs 13.33, c8 21.16 vs 21.18 ms): the
extra 96 KiB of dynamic shared memory per CTA is not worth it.

## Model side (vLLM branch)

`MuseSpark1xDecoderLayer` leaves the attention-output all-reduce to the boundary (`MuseSpark1xPartialSumStream`) where the communicator
fuses it, `MuseSpark1xColumnShardedStream` carries the unpooler shard to the next boundary, `MuseSpark1xMoE._pool_and_norm` asks for the
fused pooler gather + norm. Everything falls back to the two steps when the communicator declines (returns `None`), and the fused path
is only taken where the model would have run exactly the Triton kernel that the communicator reproduces (bf16 output, per-channel fp32
gate, weight-less or weight-first post-norm, weight-last pre-norm, no unrounded fp32 router input). Flag
`VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY` (default 1) on the model side, `MUSE_TOPO_FUSED=0` / `MUSE_TOPO_OPS` without `boundary` on the host
side.
