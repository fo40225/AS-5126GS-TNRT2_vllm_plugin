# muse_topo large-message collectives: all-reduce, reduce-scatter, all-gather (track H1b)

Scope: algorithms 4 and 5 of the host-track spec, built on the H1a foundation (`docs/h1a_collectives.md`): chunked
all-reduce (prefill: attn-out 128 MiB, moe-out 64 MiB), reduce-scatter and all-gather of any size for sequence-parallel
prefill. Same package, same workspace (no new allocation: 61.1 MiB per GPU as before), same arithmetic contract. The
public API is only extended (`TopoComm.reduce_scatter`, larger `supports()` envelope, two optional constructor
arguments).

## What was added

| item | where |
|------|-------|
| `rs_kernel` / `rs_body`: sentinel-protocol reduce-scatter chunk (algorithm 5), optional bf16 exchange (`BX`) | `csrc/large.cu` |
| `step_kernel`: fused pipeline step = reduce-scatter of chunk c + all-gather of chunk c-1 in one launch | `csrc/large.cu` |
| `ag_flag_body`: the H1a flag all-gather as a device function over a virtual grid, with chunk offset / total size | `csrc/ag_body.cuh` (used by `flag.cu`, `large.cu`) |
| host chunk loops `reduce_scatter`, `all_reduce_large`, `all_gather_large` | `csrc/bindings.cpp` |
| dispatch tables, `supports()` / `all_reduce` / `reduce_scatter` / `all_gather` large paths | `collectives.py` |
| tests `tests_h1b.py`, `test_timeout_large.py`; runner `run_h1b.sh` | `bench/` |

## Dataflow and arithmetic

The message is viewed as 8 equal blocks (block r = rows block r of a `[T, H]` tensor with `T % 8 == 0`; for an
all-reduce of any shape the flat buffer is split in 8 equal parts, `nbytes % 128 == 0`). It is processed in chunks of
C (1 MiB) taken from every block, so each launch fits the existing sentinel/flag regions of the small algorithms
(phase-a slots hold 2 sub-blocks, the partial region the fp32 partial, an all-gather slot one chunk).

Reduce-scatter chunk (`rs_body`): the island rank at partner-order position j owns the blocks of ranks R0 = owners[j]
(itself) and R1 = partner(R0).
1. (phase a) every island rank pushes its chunk of block R0 and of block R1 to the owner (sentinel slots, S-LL);
2. (phase b) the owner forms the island partial P_I of both blocks, fp32, ascending island rank order, sends P_I[R1]
   (fp32, 2 packs per input pack) to its partner R1 and receives the partner's partial of block R0;
3. `out = round_to_dtype(P_0 + P_1)`, island 0 first. Rank r ends with chunk of block r.

All-gather chunk (`ag_flag_body`): the H1a two-hop flag all-gather (push to the 3 island peers and the partner, the
partner forwards) over `out[origin * E + p0 + i]`.

All-reduce = reduce-scatter(chunk) + all-gather(chunk), the output of the first being the input of the second
(`out` doubles as staging, no extra buffer). The fused step runs `rs(chunk c)` and `ag(chunk c-1)` in one launch
(blocks [0, nb_rs) and the following nb_ag blocks), so the socket cut carries the fp32 partials of one chunk while
the finished chunk fans out. Cut traffic per GPU and direction is S/4 (partials, fp32) + S/8 (results, bf16) = 3S/8, versus S/2
for the 3-phase small algorithm; intra-island traffic is unchanged.

Every element is computed by exactly one rank with the expression of the small algorithms, including the 0xFFFFFFFF word
canonicalization (inputs, partials, result), and distributed bit for bit:
`all_gather(reduce_scatter(x), dim 0) == all_reduce(x)` bitwise, and large == H1a algorithm 2 bitwise for every input
(tests: random bit patterns, NaN/inf/-0). Results are rank-identical, run-to-run deterministic, independent of the
chunking and, on the default exact path, of the message size (batch invariant: an element's value does not depend on
which rank finalizes it, and the large algorithm computes the same expression as the small ones). `out` of all_reduce may
alias `x`.

## Opt-in: bf16 exchange (`MUSE_TOPO_LARGE_BF16_EXCHANGE=1`)

Both island partials are rounded to bf16 before the final add (`out = RN(RN16(P_0) + RN16(P_1))`); the
partial crossing the socket cut is 16 bits instead of 32, so the reduce-scatter cut bytes halve (S/8 instead of S/4 per
GPU). Accuracy class `deterministic_different`: deterministic, identical on all ranks, batch invariant INSIDE the large
algorithm (both partials are rounded, so the result does not depend on the block a row falls into), but no longer the
correctly rounded sum. Default OFF; must be set identically on all ranks; applies to bf16 all_reduce / reduce_scatter calls
served by the large algorithm. fp32 and fp16 tensors always use the exact exchange (review fix: with fp16 inputs a partial
above 65504 became inf and the final add NaN where the exact path is finite; the opt-in is bf16-only, bf16 has the fp32
exponent range).

Limit of the opt-in (review finding, measured): the small algorithms (all_reduce below 2 MiB) never round the partials, so
with the flag on an element's bits depend on whether its message is served by the large algorithm. The same rows
all-reduced as a 127-row message (small algorithm, exact partials) and as the first 127 rows of a 128-row message (large
algorithm, rounded partials) differ in 337,648 of 1,040,384 elements (120 vs 128 rows: 318,908 of 983,040; 255 vs 256 rows,
both large: 0; with the flag off: 0 everywhere). reduce_scatter (always large, from 128 KiB) vs all_reduce between 128 KiB
and 2 MiB differ in the same way. The results are still identical on all ranks and run-to-run deterministic, but with the
flag on the numerics of a token row depend on the padded batch size across the 2 MiB threshold: a further reason for the
class `deterministic_different` and the default OFF.

**Since r3c the 2 MiB boundary statements in this document (above and in the dispatch table below) are the state of H1b.** With the
flag on and `MUSE_TOPO_MEDIUM_BX` not 0 (the `env/host.env` defaults) the medium all-reduce (algorithm 2) also rounds the partials from
64 KiB (`ar_bx_min`), so the exact / bf16-exchange boundary of the bf16 all-reduce is 64 KiB, not 2 MiB, and reduce_scatter and
all_reduce agree bitwise from 128 KiB; the 2 MiB switch applies with `MUSE_TOPO_MEDIUM_BX=0` only (README section 8).

## Dispatch (collectives.py)

| call | condition | path |
|---|---|---|
| all_reduce | nbytes < 2 MiB (and <= `max_bytes`, multiple of 64 B) | H1a algorithm 2 (unchanged) |
| all_reduce | nbytes >= 2 MiB, multiple of 128 B, <= 2 GiB | large: fused reduce-scatter + all-gather pipeline, `_AR_LARGE_TABLE` |
| reduce_scatter | T % 8 == 0, bf16/fp16/fp32, message >= 128 KiB, multiple of 128 B | large reduce-scatter, `_RS_TABLE` (`_RS_BX_TABLE` for bf16 while the bf16 exchange is on, S1-9 L9 A1; otherwise `supports` is False -> NCCL) |
| all_gather | shard < 256 KiB (<= 1 MiB) | H1a LL8 / flag kernels (unchanged) |
| all_gather | shard >= 256 KiB, 16 B multiples (row bytes too for dim -1) | large all-gather chunk engine, `_AGL_TABLE` (any size, chunked) |

(With the bf16 exchange on, bf16 all-reduces between 64 KiB and 64 MiB are served by algorithm 2 (messages above `max_bytes` as row slabs of at most 4 MiB) instead of this
row, see README section 8; the large pipeline remains the path of the exact exchange and of everything above 64 MiB.)

Everything else (other dtypes, non-contiguous, odd sizes, world != 8) is rejected by `supports()` and falls back to NCCL.
The tables were produced with a (not kept) benchmark search (coordinate descent over mode, chunk, blocks, threads, unroll, cut
blocks, ramp, PDL per size, every candidate verified bitwise) and are in `collectives.py` with the measured numbers in comments.

## Results

CUDA graph, back-to-back, bf16, dispatch path, NCCL = vLLM PyNcclCommunicator with `NCCL_P2P_LEVEL=SYS`:

| op | size | NCCL us | topo us | speedup |
|---|---|---|---|---|
| all_reduce | 16 MiB | 1158 | 918 | 1.26x |
| all_reduce | 64 MiB (moe-out of an 8192-token chunk) | 4598 | 3582 | 1.28x |
| all_reduce | 128 MiB (attn-out) | 9204 | 7097 | 1.30x |
| all_reduce, bf16 exchange (opt-in) | 128 MiB | 9204 | 5840 | 1.58x |
| reduce_scatter | 64 / 128 MiB | 2379 / 4791 | 1998 / 4022 | 1.19x |
| reduce_scatter, bf16 exchange (opt-in) | 128 MiB | 4791 | 2722 | 1.76x |
| all_gather (out) | 64 / 128 MiB (pooler / unpooler of a chunk) | 2326 / 4652 | 1663 / 3310 | 1.40x |

Medium sizes: all_reduce 2-8 MiB 1.14-1.31x (H1a algorithm below 2 MiB), reduce_scatter 128 KiB-1 MiB 3.6-1.3x, all_gather 256 KiB-8 MiB
out 3.0-1.4x. Workspace: no new allocation (the large algorithms reuse the H1a regions: 61.1 MiB per GPU in total, as before).

Why not more: all_reduce moves per GPU and 1 MiB-per-block chunk 12 MiB over the island plus 3 MiB over the socket cut (fp32 partials 2 MiB,
bf16 results 1 MiB); the cut carries ~9.8 GB/s per GPU for any transport (SM push from one block, copy engine; measured) and
the SM stores saturate at ~45 GB/s per GPU, so a pure-copy kernel with this byte mix already needs 401 us per step; the shipped
fused step runs at ~420 us. The H1a-style exact exchange cannot go below S/4 bytes of fp32 partials per GPU over the cut (the
bf16 exchange halves that part). NCCL's ring moves 1.75 S over one cut link per direction at ~25 GB/s busbw; the 4 parallel
partner links, the 1.5 S cut volume and one-hop island traffic give the 1.3x.

## Accuracy

Default (exact fp32 partials): `more_accurate` than NCCL's bf16 ring and bitwise identical to the H1a small algorithms:
fp64 reference on 16 MiB [1024, 8192] bf16, randn: 100% correctly rounded, mean 0.236 ulp, max 0.5 ulp (NCCL ring: 47.8% correctly
rounded, mean 1.96 ulp, max 32992 ulp); mixed-exponent activations: 100% / 0.250 / 0.5 ulp (NCCL 55.6% / 1.55 / 131264 ulp)
(percentages to three decimals; a second mixed-exponent sample of 8.4 M elements, seed 800, gives max 0.531 ulp: the
elements whose fp32 partial sum was inexact, see below).
"Correctly rounded" is measured against a single-rounding reference (`bench/common.py: round_single`; the former reference
`s64.to(bf16)` rounds twice, float64 -> float32 -> bfloat16, like the algorithm, so it could not see a deviation). The
algorithm (fp32 chain, one final rounding) equals the correctly rounded sum whenever the fp32 partial sums are exact, i.e.
for activation-like data (exponent spread up to ~16 bits: randn 0 deviations, per-column-scale data 1 in 4.2 M elements);
for data with a very wide per-element exponent range (+-24 binary orders) an inexact fp32 partial can flip the final
rounding in 0.13% of the elements. The guarantee that always holds is an error of at most 0.5 ulp + 2^-16 relative,
far below NCCL's bf16 ring (mean ~2 ulp).
Opt-in bf16 exchange: 67.5% correctly rounded, mean 0.78 ulp on the same mixed data, but the max error is large (32640 ulp) when the two
island partials cancel (P_0 ~ -P_1: the rounded partials have lost the bits that decide the result), which is why it is
`deterministic_different` and OFF by default.

## Tests (bench/, all under the GPU lock, 8 processes, no vLLM model)

`tests_h1b.py` (289 checks x 8 ranks, all PASS; both H1a suites re-run after the changes: `tests_h1a.py --quick` 116 checks and `test_h1a.py --quick`, PASS):
large all_reduce (both modes, fused/two-launch, cut blocks, ramp, PDL, many chunk/block configurations) == arithmetic contract at 16/64/128
MiB for bf16/fp16/fp32 and randn / mixed-exponent / signed-zero inputs, odd sizes with a partial last chunk, and == H1a algorithm 2 bitwise
for 128 KiB..8 MiB; random bit patterns (NaN, inf, subnormals, all-ones words): large == H1a algorithm 2 and RS + AG == AR; reduce_scatter ==
rows block of the contract; reduce_scatter + all_gather(dim 0) == all_reduce bitwise (also on random bits); large all_gather exact in both layouts
for odd chunk counts and arbitrary bit patterns; rank-identical bits (sha256 over ranks), run-to-run, eager == CUDA graph replay
(all_reduce + reduce_scatter + all_gather in one graph), in-place all_reduce, batch invariance (rows alone == rows inside a bigger
batch); bf16 exchange == torch emulation bitwise, rank-identical, batch invariant (inside the large algorithm), deterministic, fp16 / fp32 keep the
exact exchange (+ the fp16 +-40000 overflow regression); accuracy vs fp64 and NCCL; 70 random mixed
sequences of small/medium/large ops (call-counter alignment); soak: 100000 back-to-back graph replays of a graph with 7 collectives (large all_reduce
in both modes, reduce_scatter, medium all_reduce, small all_reduce, both all_gathers) without hang or corruption (308 s).
`test_timeout_large.py`: one rank skips a large all_reduce / reduce_scatter / all_gather -> the other ranks trap within the 3 s
timeout (error word set), no hang (3 x PASS). Run: `bash scripts/locked_job.sh <tag> <timeout_s> bash bench/run_h1b.sh <tag> tests timeout`.

## Findings and design notes

* Design: the first plan (a persistent chunk-pipelined kernel with rings and credits) was dropped for host-driven chunk loops over
  the proven sentinel/flag kernels: all_reduce = reduce_scatter + all_gather per chunk gives RS + AG == AR by construction, needs no
  extra memory, and a fused step (RS of chunk c + AG of chunk c-1 in one launch) overlaps the two directions of the cut.
* The first untuned fused kernel (8 + 24 blocks) was slower than a loop of H1a algorithm 2 on 8 MiB pieces (8.4 vs 8.0 ms): too many blocks pushing
  across the cut (9.4 -> 6.8 GB/s per GPU from 1 -> 8 blocks). With 4 + 8 blocks, unroll 2 and the all-gather cut pushed by one dedicated
  block it is 7.1 ms. A dedicated block for the reduce-scatter partials (staged through the c2 region) is functional but slower
  (the workers pushing directly are better), merged cut blocks, ramped chunk sizes at both ends and PDL gave nothing; all stay as options
  (`cut_rs`, `cut_ag = 2`, `ramp`, `pdl` of `_all_reduce_large`), covered by tests, off in the tables.
* The copy engine does not help (probe): same ~9.9 GB/s on the cut, and slower than SM push for the island all-to-all.
* Fp32 inputs use the same kernels (partial = 1 pack per input pack), fp16 too.

## Caveats and limits

* The large kernels spin-wait on peers with all their blocks co-resident (4 + 8 + 1 blocks of 512 threads at >= 16 MiB, up to 4 + 32 + 1 at 2-8 MiB): do not overlap them with a kernel
  that can occupy every SM of the GPU (the spin timeout traps instead of hanging, `MUSE_TOPO_TIMEOUT_S`, default 10 s). Same rule as H1a.
* All collectives of one TopoComm must be issued on one stream (unchanged), same order and shapes on all ranks.
* `all_reduce` of an odd number of bytes (not a multiple of 128 B) above `max_bytes`, reduce_scatter with T % 8 != 0 or below 128 KiB and other dtypes
  go to NCCL through `supports()`.
* `supports("all_reduce")` is True for multiples of 128 B up to 2 GiB; the plugin's `_why_not` fallback reasons were updated for this in H2b
  (`muse_topo/vllm_plugin.py`).

## Follow-ups

* A faster cut would need a different transport: the earlier host-memory relay probe (docs/h1a_collectives.md) showed 13.2 GB/s per GPU, but the island
  egress (45 GB/s per GPU of SM stores) is then the next limit (15 MiB per step), so the gain would be about 15% (estimate, not built).
* bf16 exchange only changes the all_reduce/reduce_scatter partials; a fp8 exchange or error-feedback schemes are out of scope (deterministic_different).
* Sequence-parallel prefill (B5) should call `comm.reduce_scatter` / `comm.all_gather(dim 0)`: RS + AG costs 4.02 + 3.31 ms at 128 MiB, the same as the
  fused all_reduce (7.1 ms) and bitwise identical to it.
