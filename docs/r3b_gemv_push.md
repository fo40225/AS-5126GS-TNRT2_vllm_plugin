# Pooler / unpooler GEMV that pushes into the peers (track r3b-gemv-push)

A Muse Spark decode layer has two sharded projections whose columns are gathered right after the GEMV:

| projection | GEMV of one rank | gather | consumer of the gather |
|------------|------------------|--------|------------------------|
| pooler (`[T, 8192] -> [T, 4096]`) | `x [T, 8192] . W[512 rows, 8192]^T` | `[T, 512] -> [T, 4096]` | pre-expert RMSNorm (r2d) |
| unpooler (`[T, 4096] -> [T, 8192]`) | `x [T, 4096] . W[1024 rows, 4096]^T` | `[T, 1024] -> [T, 8192]` | residual boundary of the next layer (r2d) |

Before this track the chain was `cuBLAS GEMV -> ag_bnd_kernel` (r2d: gather + epilogue in one kernel). On the real-weight c1 profile
(`profile_r2fin_real`) it cost 6.2 + 5.7 us (pooler) and 4.8 + 8.7 us (unpooler) per layer.

## What was built

Two kernels that are the two halves of one collective call (`TopoComm.gemv_all_gather_rmsnorm` / `gemv_all_gather_boundary`):

| kernel | file | job |
|--------|------|-----|
| `gemv_push_kernel<Geom, T>` | `csrc/gemv_push.cu`, `csrc/gemv.cuh` | the GEMV of this rank's columns (fixed fp32 order, one bf16 rounding), stored in plain form and, depending on the push mode, sent to the partner and the three island peers in the LL8 format of the all-gather |
| `ag_bnd_kernel<..., PUSHED>` | `csrc/ag_bnd.cu` | the existing gather + epilogue kernel of r2d without its first (push) stage: forward the partner's words, collect, run the RMSNorm / boundary |

Hooks: vLLM branch `vllm/models/muse_spark/nvidia/ops/fused_boundary.py` (`gemv_all_gather_rmsnorm`, `gemv_all_gather_residual_boundary`, kinds
`gemv_all_gather_rmsnorm` / `gemv_all_gather_boundary`; the model keeps cuBLAS + the r2d gather when the communicator declines), plugin methods in
`muse_topo/vllm_plugin.py`.

### Arithmetic (class deterministic_different against cuBLAS; more accurate for the pooler at T >= 2)

Order of one output element (`csrc/gemv.cuh`): 4 accumulators per thread by element position inside a 16-byte chunk (`fma`), chunks in ascending order, then
`(a0 + a1) + (a2 + a3)`, an xor butterfly over the 32 lanes, the `WPR` warps of the row in ascending order, one round-to-nearest-even to bf16. It depends
on the geometry variant (`WPR`), never on T, the rank or the run: a token row gets the same bits at every batch size, on every rank, in every run.
`bench/test_gemv_push.py` checks the shard bit for bit against a torch emulation of this order.

Accuracy against the exact fp64 product (`bench/test_gemv_push.py`, mixed data, 8 rows): this order is correctly rounded for 99.95 .. 100 % of the
elements (mean error 0.25 ulp, max 0.5 .. 1.2 ulp). cuBLAS gives the same at T = 1 (gemv) and for the unpooler, but for the **pooler at T = 2 .. 8
(cutlass wmma + split-K reduction through a bf16 workspace) only 58 .. 61 % correctly rounded, mean error 1.3 .. 3.5 ulp, max 260 .. 16800 ulp**.
So the pooler GEMV is `more_accurate` than the path it replaces at T >= 2 and equal in quality at T = 1; the unpooler is equal in quality.

Single-GPU sweep with the helpers of the test (`bench/acc_gemv_sweep.py`: `emulate_gemv`, `err_stats`; the by-rows variant; mixed data, mean of 3 seeds,
the max is the largest of the 3; 2026-10-27):

| projection | T | this order: mean ulp / max ulp / correctly rounded | cuBLAS: mean ulp / max ulp / correctly rounded |
|------------|---|------|------|
| pooler | 1 | 0.246 / 0.5 / 100 % | 0.247 / 0.9 / 99.94 % |
| pooler | 2 | 0.250 / 0.5 / 100 % | 1.351 / 386 / 59.8 % |
| pooler | 4 | 0.253 / 0.5 / 100 % | 1.402 / 668 / 59.1 % |
| pooler | 6 | 0.250 / 1.2 / 99.98 % | 3.483 / 16826 / 59.4 % |
| pooler | 8 | 0.248 / 0.5 / 99.98 % | 1.582 / 872 / 59.8 % |
| unpooler | 1 | 0.250 / 0.5 / 100 % | 0.250 / 0.5 / 99.97 % |
| unpooler | 4 | 0.248 / 0.5 / 99.99 % | 0.248 / 0.5 / 100 % |
| unpooler | 6 | 0.251 / 14.3 / 99.98 % | 0.253 / 36.3 / 99.97 % |
| unpooler | 8 | 0.249 / 0.5 / 100 % | 0.249 / 0.5 / 99.98 % |

(The unpooler maxima at T = 6 are outlier elements of this data set that both orders show; the means and the fraction of correctly rounded elements are the comparison.)

The router GEMV of the vLLM branch (`_muse_C::router_gemv`) is a different case: the cuBLAS router is already an fp32 product, so the class is
`deterministic_different` (a fixed order that is the same for every T, rank and run; cuBLAS switches algorithm between T = 1 and T >= 2), bounded by
the tests to the error of cuBLAS, not claimed to be more accurate.

## Protocol

* The pair uses one call epoch: the GEMV kernel reads the call counter (after `griddepcontrol.wait`), the gather kernel (`epoch_begin_pushed`)
  reads the same value and advances it. LL8 slots, double buffering and the reverse-dependency argument of the all-gather are unchanged.
* PDL: the GEMV kernel is launched as a programmatic dependent of the kernel in front of it. Its weights (8 MB, independent of everything) are loaded
  into registers *before* `griddepcontrol.wait` (`ld.global.L1::no_allocate` as a volatile asm: ptxas moves `.nc` loads below the wait, which would
  turn the early loads into late ones and hoist `.nc` activation loads above it, reading the producer's output too early). It triggers its successor at
  its start; the gather kernel loads its never-changing operands (RMS weights, boundary gate coefficients and weights) before its own wait.
* `Ctx::trigger`: the all-reduce kernels in front (`ar2_kernel` of the MoE output, `ar2_bnd_kernel` of the attention output) let the GEMV
  kernel start at a configurable point (`MUSE_TOPO_GEMV_TRIGGER`). Triggering at their start (stage 1) was measured to delay the all-reduce by
  4 .. 5 us (the 8 MB burst competes with its latency-critical first phases on the last-arriving rank), so the default is stage 3 (the last phase
  has ended); stage 2 (the last phase begins) measured the same within the noise. The other all-reduce kernels never fire stages 2 and 3, and the
  GEMV kernel then starts at their completion. A trigger only moves when the dependent grid may become resident: both kernels still wait for the
  completion and memory flush of the grid in front, so no result depends on it.

## Knobs

All of them are read once per process (host plugin: when `TopoComm` is built; vLLM branch: when the model is built) and must be identical on every
rank.

| variable | default | meaning |
|----------|---------|---------|
| `VLLM_MUSE_SPARK_SM120_FUSED_GEMV` (vLLM branch) | 1 | the model offers the sharded pooler / unpooler GEMV to the TP communicator (needs `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=1` and a communicator that implements the hook); 0 keeps the cuBLAS GEMV + the r2d gather |
| `MUSE_TOPO_GEMV` | 1 | the plugin accepts the offer (needs `MUSE_TOPO_FUSED` != 0); 0 declines every call, so the model runs cuBLAS + the r2d gather. For an A/B set both switches (the vLLM one alone only stops the offer) |
| `MUSE_TOPO_GEMV_TRIGGER` | 3 (provisional) | stage at which the all-reduce in front lets the GEMV kernel start: 0 = never, 1 = at the start of the all-reduce, 2 = when its last phase begins, 3 = when it ends. With `MUSE_TOPO_PDL=1` the all-reduce kernels trigger at their start whatever the stage |
| `MUSE_TOPO_GEMV_PUSH` | 1 | bits 0-1: push mode, 0 = the GEMV kernel only stores and the gather kernel pushes, 1 = every GEMV CTA pushes its words as soon as they are done, 2 = the last CTA to finish pushes the block in coalesced stores; bit 2 (+4): the weights are loaded behind the wait for the predecessor instead of before it. Valid values are 0, 1, 2, 4, 5, 6. **Do not use 3 or 7**: mode 3 pushes nothing but tells the gather kernel that the words were pushed, so the peers spin until the watchdog traps |
| `MUSE_TOPO_GEMV_VARIANT` | unset (by rows) | pin a geometry variant (0 .. 3) for all calls; unset = V0 for one row, V3 for 2 .. 8 rows. V1 / V2 sum in another order (another number of warps per row) |
| `MUSE_TOPO_GEMV_MAX_ROWS` | 8 | the largest padded decode batch the kernels serve (1 .. 8; the kernel rejects more); above it the model keeps cuBLAS + the r2d gather |
| `VLLM_MUSE_SPARK_SM120_ROUTER_GEMV` (vLLM branch) | 1 | the fp32 router GEMV of the decode layer (`_muse_C::router_gemv`), companion of this track: one kernel with a fixed summation order instead of cuBLAS dot / sgemm + reduction. Its input is the fp32 router input that the boundary kernel writes in the decode graphs only with the expert prefetch; without it the router stays on cuBLAS and the vLLM log says so once |
| `VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES` (vLLM branch) | 12582912 (12 MiB) | bytes of the routing-aware expert L2 prefetch (cache hint, no numeric effect); the router GEMV moves the routing ids earlier, which is what a larger budget would need |

## Results

### Where the time went (profile, 2026-10-26, real weights, c1, rank 0, build with the trigger at the start of the all-reduce)

* The chain from the end of the MoE-out all-reduce to the start of the QKV GEMM (the unpooler chain) dropped from 13.9 to 8.5 us (-5.4 us); the
  pooler side let the expert kernel start 2.1 us earlier.
* The all-reduce kernels in front got slower (`ar2_kernel` of the MoE output 32.3 -> 36.6 us, `ar2_bnd_kernel` 17.8 -> 22.9 us: +4.3 .. 5.1 us): the 8 MB
  burst of weight loads of the GEMV CTAs, which are resident from the start of the all-reduce, competes with its latency-critical first phases on the
  last-arriving rank. Layer period 147.6 us (off) vs 147.0 us (on); end-to-end TPOT c1 9.617 vs 9.651 ms, c4 15.595 vs 15.478, c16 27.058 vs 26.998: no gain.
* The router chain (cuBLAS dot 5.9 + reduce 1.5 + routing 3.1 = 10.5 us) became the long pole of the layer; at T = 4 the router sgemm even slowed down
  (10.8 + 3.9 -> 17.6 + 3.2 us) while the GEMV CTAs were resident. Fixes: the trigger stages 2 / 3 of the all-reduce (this folder) and the router GEMV
  of the vLLM branch (`VLLM_MUSE_SPARK_SM120_ROUTER_GEMV`).

### Real-weight TPOT with both fixes (S1-4a, 2026-10-27)

Protocol: random 512/128, EP8 + TP8, fp8 KV, max-model-len 131072, memory 0.95, max-num-seqs 128, production `env/host.env`, SEED 9000 + i, one lock hold with
one server session per arm, 2 repeats per concurrency (32 / 32 / 64 / 256 prompts at c1 / c4 / c16 / c64), TPOT in ms (mean of the 2 repeats). The base arm
`A0` is the production behaviour (`VLLM_MUSE_SPARK_SM120_FUSED_GEMV=0 MUSE_TOPO_GEMV=0 VLLM_MUSE_SPARK_SM120_ROUTER_GEMV=0`), measured at the start and at the end
of the session; the percentages are against the mean of the two (9.593 / 15.256 / 26.849 / 53.941 ms), and a session moves c16 and c64, which these
changes cannot reach (batches above 8 tokens), by about 1 %.

| arm | configuration | c1 | c4 | c16 | c64 |
|-----|---------------|----|----|-----|-----|
| A0 start / end | off | 9.606 / 9.580 | 15.301 / 15.210 | 26.709 / 26.988 | 54.086 / 53.797 |
| A1 | GEMV, trigger 2 | 9.385 (-2.2 %) | 15.094 (-1.1 %) | 26.447 | 53.608 |
| A2 | GEMV, trigger 3 | 9.379 (-2.2 %) | 15.241 (-0.1 %) | 26.632 | 53.604 |
| A3 | A2 + late weights (push 5) | 9.456 (-1.4 %) | 15.480 (+1.5 %) | 26.155 | 54.044 |
| A3b | A2 + last-CTA push + late weights (push 6) | 9.495 (-1.0 %) | 15.611 (+2.3 %) | 26.696 | 53.819 |
| A3c | A2 + last-CTA push (push 2) | 9.424 (-1.8 %) | 15.485 (+1.5 %) | 26.257 | 53.867 |
| A4 | router GEMV only | 9.495 (-1.0 %) | 15.175 (-0.5 %) | 26.727 | 53.787 |
| **A5** | A2 + router GEMV | **9.187 (-4.2 %)** | **14.763 (-3.2 %)** | 26.159 | 53.989 |
| BEST2 | A5, second session | 9.178 (-4.3 %) | 14.737 (-3.4 %) | 26.929 | 54.013 |
| A6 | A5 + weight prefetch CTAs 4 / 12 (host.env: 8) | 10.525 / 9.631 | 16.037 / 15.666 | 27.649 / 27.113 | 53.813 / 54.370 |

A5 is the shipped configuration (`VLLM_MUSE_SPARK_SM120_FUSED_GEMV=1`, `MUSE_TOPO_GEMV=1`, `VLLM_MUSE_SPARK_SM120_ROUTER_GEMV=1`, push 1, trigger 3): c1 -4.2 %,
c4 -3.2 % (reproduced in the second session), c16 and c64 flat. The two changes add up to more than the sum (GEMV -2.2 %, router GEMV -1.0 %, both -4.2 % at c1):
presumably because the router chain was the long pole once the pooler chain got shorter (profile above). Late weights and the coalesced push do not beat the default. Not measured:
trigger 2 together with the router GEMV, and the padded T = 8 graph (c5 .. c8; an early bench had the GEMV chain slower than cuBLAS at T = 8, hence
`MUSE_TOPO_GEMV_MAX_ROWS`).

### Microbenchmarks (single call, graph b2b, `bench/bench_gemv_push.py`)

cuBLAS pooler `[T, 8192] x [512, 8192]^T`: T = 1 gemv 5.3 us hot L2 / 24.3 us cold; T = 2 .. 8 cutlass wmma 4.1 + split-K reduce 2.0 = 6.1 / 19 us.
Unpooler `[T, 4096] x [1024, 4096]^T`: T = 1 4.3 / 18.5 us; T = 2 .. 8 3.9 + 1.8 = 5.7 / 19 us. Layer chain (all-reduce + boundary, pooler, MoE all-reduce,
unpooler + boundary, trigger at the start, hot L2, two-step -> fused): T = 1 34.6 -> 27.9 us, T = 2 40.4 -> 33.1 (V3), T = 4 43.9 -> 41.5 (V3); T = 8 52.1 -> 53.7
(GEMV without the pushes) / 58 .. 74 (with them), i.e. slower; cold L2: T = 1 36.8 -> 24.9, T = 2 43.0 -> 27.4, T = 4 44.9 -> 33.2.

### Tests

`bench/test_gemv_push.py` (8 ranks, under the GPU lock): the shard bit for bit against the torch emulation of the order (`emulate_gemv`), the fused calls against
`all_gather` of the emulated shards + the model's Triton kernels (bitwise, equal on every rank, all push modes, all variants, the by-rows selection), accuracy next
to cuBLAS, the decode-layer chain eager and in CUDA graphs with every trigger stage, misaligned operands, a random mix with other collectives.
`--quick` is the short run, no flag the full one. The shared all-reduce kernels are covered by `bench/test_ar_boundary.py`, the plugin hook by
`bench/test_plugin.py`, the medium all-reduce by `bench/test_medium_bx.py`.
