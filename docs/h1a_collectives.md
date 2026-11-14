# muse_topo collectives: foundation and decode-size algorithms (track H1a)

> Update: the large-message algorithms, reduce_scatter and the large all_gather (H1b) are implemented, see `docs/h1b_collectives.md`;
> `reduce_scatter` below is no longer `NotImplementedError` and `supports()` also accepts all_reduce above `max_bytes`.

Scope: the `muse_topo` package skeleton (topology signature, JIT build, IPC workspace, `TopoComm`) and the
collectives needed for decode: all-reduce up to 8 MiB (algorithms 1, 2) and all-gather (algorithm 3, both layouts).
The large/chunked algorithms (4, 5), reduce-scatter and the vLLM plugin build on this API.

## API (stable)

```python
from muse_topo import TopoComm, TopoUnavailable
comm = TopoComm(cpu_group, device, rank, world_size, *, max_bytes=8 << 20, check_env=True)
comm.supports(op, tensor, dim=None) -> bool     # op in {"all_reduce", "all_gather", "reduce_scatter"}
comm.all_reduce(x, out=None)       -> tensor, same shape and dtype (new unless `out` is given)
comm.all_gather(x, dim, out=None)  -> new tensor; dim=-1: column concatenation [..., 8w]; dim=0: rank-major [8T, ...]
comm.reduce_scatter(x, out=None) -> rows block r of the reduced [T, ...] tensor (T % 8 == 0), added in H1b
comm.close()
```

* Construction is collective. It raises `TopoUnavailable` if `MUSE_TOPO_ENABLE != 1` (unless `check_env=False`),
  the world is not 8 GPUs on one host, a GPU is not an RTX PRO 6000 Blackwell (cc 12.0), a peer pair has no P2P
  access, or the PCI/NUMA layout is not 2 sockets x 2 virtual switches x 2 GPUs.
* `supports()` is the only eligibility check: CUDA tensor on the communicator's device, contiguous;
  all_reduce: bf16/fp16/fp32, size a multiple of 64 B and at most `max_bytes` (8 MiB default);
  all_gather: any dtype, dim = last or 0 (`all_gather` raises `ValueError` for any other dim), shard bytes a multiple of
  4, for dim = last the row bytes too (16 for shards above 64 KiB), shard at most 1 MiB. Callers fall back to NCCL when it
  is False. It depends on dtype / shape / contiguity only, never on rank-local state: the kernels need 16-byte aligned
  operands, but a misaligned view (or a misaligned caller-provided `out`) is staged through an aligned copy inside the
  op instead of being rejected, because a rank that fell back to NCCL while the others run topo kernels would deadlock
  the group (review fix).
* Construction is staged (topology and tuning check, JIT build + allocation, IPC exchange) and every stage ends with an all-rank
  agreement, so a failure on one rank (missing nvcc, IPC error) makes every rank raise `TopoUnavailable` instead of
  blocking the others (`bench/test_init_failure.py`).
* All calls are asynchronous on the current stream, CUDA-graph safe and must be issued in the same order with the
  same shapes on every rank. Collectives of one communicator must run in stream order on one stream (the
  communicator serializes a second stream behind the first, or raises with `MUSE_TOPO_STRICT_STREAM=1`).
* Environment: `MUSE_TOPO_ENABLE=1`, `MUSE_TOPO_TIMEOUT_S` (spin timeout, default 10 s), `MUSE_TOPO_PDL=1`
  (programmatic dependent launch, experimental), `MUSE_TOPO_VERBOSE=1` (JIT build log).

## Topology

`topology.derive()` maps each rank's GPU to its PCI address, `numa_node` -> socket (`physical_package_id` of the
node's CPUs) and the upstream bridge two levels above the GPU (the virtual switch). Island = socket (2 islands x 4
ranks, island 0 holds rank 0); partner(r) = the rank at the same position in the other island when each island is
ordered by (virtual switch PCI address, GPU PCI address), which sends each virtual switch to exactly one far switch
(collapse-free) and pairs the same physical GPUs for any rank assignment. On this host: islands {0,1,2,3} and
{4,5,6,7}, partner(r) = r ^ 4. It works for any CUDA_VISIBLE_DEVICES order (`bench/test_topology.py`, 200 random
rank permutations). The summation order is by rank, independent of the partner pairing.

## Arithmetic contract (every size, every algorithm)

```
P_I = ((x_m0 + x_m1) + x_m2) + x_m3      fp32, m0..m3 = ranks of island I in ascending order
out = round_to_dtype(P_0 + P_1)          fp32 add, island 0 first, single final rounding
```

Inputs (bf16/fp16/fp32) are widened exactly. All ranks compute the same expression from the same bits, so results
are bitwise identical on every rank (including the sign of zero), across runs, across sizes and across algorithms.
The only deviation from raw IEEE behavior: a 32-bit payload word equal to `0xFFFFFFFF` (a NaN pattern, two bf16/fp16
NaNs or one fp32 NaN) is replaced by `0xFFFFFFFE` before it is used, so NaN payloads may differ from NCCL's.

Opt-in variant (r3c, `docs/r3c_medium_ar.md`, bf16 inputs only, `MUSE_TOPO_LARGE_BF16_EXCHANGE=1`): algorithm 2 has a `BX` kernel variant
that rounds both island partials to bf16 before the final add, `out = round_bf16(bf16(P_0) + bf16(P_1))`; class
`deterministic_different`. It halves the bytes of the partner exchange over the socket cut and is what the dispatch uses for bf16
messages from 64 KiB on when the flag is set. The statement above ("across sizes and across algorithms") then holds among the bf16
messages of 64 KiB and more (medium and large algorithm give the same bits), not between those and the exact messages below 64 KiB.

## Algorithms and protocols

| id | op | scheme | protocol |
|----|----|--------|----------|
| 1 | all_reduce | 2-phase: push the input to the 3 island peers, every rank forms P_I of the whole message, exchanges it (fp32) with its partner | sentinel |
| 2 | all_reduce | 3-phase: slice j is owned by island rank j; push slices to owners, owner forms P_I[slice], exchanges it (fp32) with the partner, forms the result and pushes it to the island | sentinel |
| 3 | all_reduce | same as 2 with per-block release flags (`flag.cu`) | flag |
| 1 | all_gather | push the shard to the 3 island peers and the partner, the partner forwards it to its island | LL8 (tagged 8 B units) |
| 2 | all_gather | same with per-block release flags | flag |

Transport: SM push stores into the peer's IPC receive buffer. Island traffic is all-to-all (equal and scalable);
cross-island traffic only goes rank <-> partner (never one virtual switch to two far switches). Remote atomics are
not used (unsupported on this box).

Sentinel protocol (all-reduce): receive slots hold `0xFFFFFFFF` while empty; senders store plain 16 B packs, receivers
poll the payload itself (a pack is complete when none of its four words is the sentinel), consume it and reset the
slot. No fence, no flag, no wire overhead (an LL8 protocol would double the bytes). Reduction payloads are
canonicalized (see above). Flag protocol: payload stores, `__syncthreads`, `st.release.sys` of the call tag into the
receiver's per-block flag, receiver `ld.acquire.sys` + `__syncthreads`. LL8 (all-gather, bit-exact payloads): 8 B
units {data, call tag}.

Call counter and tags are 64 bit (review fix): the per-GPU counter `ctl[0..1]` and every flag word are 64-bit, the tag of
call e is e + 1 and a waiter spins until `flag >= tag` (plain signed 64-bit compare, never wraps, and the all-ones initial
value of a flag is -1, i.e. below every tag). The first version used 32-bit words and a signed compare: at the 2^31-th
launch of a communicator (about 28 h of continuous batch-1 decode at ~21k launches/s) the flag kernels waited for a flag
above INT_MAX (spin timeout, trap) and above it accepted stale flags (silent wrong data); an unsigned-difference patch
would still have accepted the all-ones initial flags once the tag has its high bit set. The 8 B LL8 units only have a
32-bit tag field: tag = (e >> 1) % 0xFFFFFFFE + 1 (a slot of parity p is reused by every second call, so its tag advances
by one per reuse; the value 0xFFFFFFFF of a never-written slot is never produced). The LL8 tag therefore repeats after
about 2^33 launches; a slot that stays unwritten for exactly that long could be mistaken for fresh data (residual, needs
a never-rewritten word and a late sender, not exercised). `bench/test_epoch.py` forces the counter (test hook
`Comm.set_epoch`, forward only) around 2^31 / 2^32 / 2^33 / 2^62 and checks every protocol with fresh and used flag
columns and rank skew.

Buffer reuse and graph safety: every receive region exists twice (parity of a per-GPU call counter `ctl[0]`). The
counter is read by every warp at kernel start; each warp registers its arrival with one atomic on a local word and
the last warp advances the counter, so all blocks of a launch agree on the parity and no host state is needed (graph
replays advance it). Reuse is safe because each collective has a reverse dependency for every sender -> receiver
direction (a sender can only start call k+2 after the receiver has finished call k). Waits are bounded: after the
timeout the kernel writes an error word (mapped host memory) and executes `__trap()`.

Workspace (per GPU): 61.1 MiB for `max_bytes=8 MiB` (sentinel regions for algorithms 1/2, LL8 and flag regions
for all-gather; the 64-bit flag words grow the flag region from 64 to 128 KiB, 64,094,208 B in total). Allocated once at
construction.


## Results

Full latency tables (all_reduce [T,8192] and [T,4096] for T = 1..512, all_gather pooler/unpooler shards for T = 1..256
and the logits shard, NCCL baseline, FlashInfer comparison): `docs/h1a_latency.md`. Headline numbers (CUDA graph,
back-to-back / behind a 50 us kernel, bf16): all_reduce [1,8192] 5.4 / 5.4 us (NCCL 42.8 / 43.0), [1,4096] 5.0 / 5.1
(NCCL 41.2 / 41.5), [8,8192] 11.5 / 11.3 (NCCL 47.3 / 47.6), [64,8192] 61 / 61 (NCCL 86), [512,8192] 477 / 480 (NCCL
580 / 573); all_gather [1,512]->[1,4096] 3.6 us (NCCL 24.6), [1,1024]->[1,8192] 3.8 us (NCCL 25.9), logits
[1,25256]->[1,202048] 13.3 us (NCCL 35.3) (these tables are the pre-64-bit-counter measurements; the re-measured decode
numbers after the review fix are in `docs/h2_vllm_plugin.md`, section "Review fixes"). Accuracy (bf16, fp64 reference,
16x8192): correctly rounded in 100% of the elements against a single-rounding reference (`bench/common.py: round_single`;
the fp32 chain followed by one bf16 rounding is the correctly rounded sum whenever the fp32 partial sums are exact, i.e. for
activation-like data whose exponents differ by at most ~16 bits; for pathological data with a very wide exponent range an
fp32 partial can be inexact and the result is then within 0.5 ulp + 2^-16 relative, e.g. 0.13% of the elements deviate from
the correctly rounded value for +-24 binary orders of per-element exponent spread), mean error 0.236 ulp (randn) / 0.250 ulp
(mixed exponents), versus NCCL's ring 47.6% / 55.4% correctly rounded, mean 1.88 / 2.08 ulp (max 4240 / 76976 ulp). The
earlier "100%" figures used `s64.to(bf16)` as reference, which rounds twice (float64 -> float32 -> bfloat16) just like the
algorithm.

## Dispatch and tuning

`muse_topo/collectives.py` holds the size dispatch tables (`_AR_TABLE`, `_AG_TABLES`): for every measured message size
the winning (algorithm, grid, block, unroll) of `bench/bench_h1a.py --sweep` (`bench/make_table.py` prints the
literals). Findings that shaped them (2026-10-23, protocol: CUDA graph, back-to-back, bf16):

* The sentinel protocol (all-reduce algorithms 1, 2) beats the flag protocol (algorithm 3) at every size: 5.5 vs 8.0 us
  at [1,8192] (three release fences cost ~0.9 us each) and 119 vs 219 us at [128,8192] (block-wide barriers
  between phases prevent the fine-grained pipelining the per-thread polling gives). It also avoids the 2x wire bytes of
  LL8. Algorithm 1 (2-phase, whole message per rank) never wins: it ships 4x more fp32 partials over the cut than
  algorithm 3-phase (6.6 vs 5.5 us at [1,8192], 17 vs 8 us at [4,8192]).
* All-gather: LL8 up to 16 KiB shards, the flag protocol from 32 KiB (LL8 doubles the wire bytes: 22 vs 16 us at 64 KiB).
* Small blocks win: the best configurations use 32-64 threads per block spread over 4-64 SMs for messages up to
  256 KiB (a single SM cannot drive enough remote stores), 8 blocks x 256 threads above 512 KiB (SM push over the
  socket cut gets slower with more CTAs: 29 vs 36 GB/s).
* Beyond [4,8192] the cut between the two sockets is the bottleneck, by construction of the arithmetic contract: the
  fp32 partials make every rank send 2S bytes per direction (S = bf16 message bytes) across the cut. [128,8192]
  moves 2 MiB per rank in 119 us (~35 GB/s aggregate over the four partner links), consistent with the measured SM
  push ceiling (30-37 GB/s). Beating this needs a faster cut transport (copy engine, or the opt-in bf16 exchange of the
  large-message algorithm), not a better decode kernel.
* Programmatic dependent launch (`MUSE_TOPO_PDL=1`, kernels start with `griddepcontrol.wait`): -0.3 us in back-to-back
  graphs (the previous kernel is another collective), +0.1..0.2 us behind a normal kernel (spacer mode), so it stays off.
* In-kernel timeline of all_reduce [1,8192] (`bench/trace_h1a.py`, MUSE_TOPO_TRACE=1 build, block 0 thread 0, mean over
  ranks): start to phase-a sent 0.58 us (counter + input loads), island hop 1.10, reduction 0.14, partner hop 1.85
  (1.35 latency + serialization + rank skew), island hop 1.40, total 5.2 us; the graph node adds ~0.7 us of which most
  overlaps. all_gather [1,1024] (LL8, before the store-on-arrival change): push 0.45, partner hop + forward 1.4, forwarded copies from the island 1.0, output stores and exit 0.6 = 3.5 us.

## Tests (bench/)

| script | what it checks |
|--------|----------------|
| `smoke.py` | forced algorithms 1-3 and all-gather algorithms 1-2 on a few shapes, 3 dtypes, 3 input kinds |
| `test_epoch.py` | 64-bit call counter / flags: counter forced to 0x7fffffff, 0x80000000, 0x90000000, 0xffffffff, 0x1fffffffa, 2^33, 2^62 on fresh communicators (all flag columns and LL8 slots pristine) with rank 0 joining late, and one communicator moved forward across 2^31 / 2^32 / the LL8 tag wrap with mixed flag / LL8 / sentinel / large ops and random per-rank delays; bitwise checks (a regression traps the context, spin timeout 20 s) |
| `tests_h1a.py` | arithmetic contract bitwise (bf16/fp16/fp32, T = 1..512, hidden 8192/4096, randn / mixed-exponent / +-0 inputs), rank-identical bits (sha256 over ranks), run-to-run determinism, eager == CUDA graph replay, algorithm 1 == 2 == 3 bitwise, accuracy vs fp64 and vs NCCL, NaN/inf, all_gather exactness (both layouts, flag and LL8, arbitrary bit patterns incl. 0xFFFF words), supports()-vs-kernel sweep of small all_gather shapes (1/2/3-D, extents 1..9, int8/bf16/fp32, dims 0 and last, odd rows included: everything `supports()` accepts must run and be exact), middle-dim `ValueError`, misaligned operands with a different alignment on every rank, 120-op mixed random sequences, 10^5 back-to-back graph replays of 5 collectives |
| `test_timeout.py` | one rank skips a collective: the others trap with an error word within the timeout (3 s) instead of hanging |
| `test_init_failure.py` | a build failure on one rank makes every rank raise `TopoUnavailable` (no hang) |
| `check_ctx.py` | TopoComm creates no CUDA context on the other 7 GPUs |
| `bench_h1a.py` | latency sweep vs NCCL (`--sweep` per-shape tuning, `--public` dispatch path), `make_table.py`, `summarize_h1a.py` |
| `trace_h1a.py` | in-kernel phase timeline |
| `host_relay_probe.py` | research probe of the cut transport through pinned host memory (see below) |

Run (server): `scripts/deploy_host.sh`, then
`bash scripts/locked_job.sh <tag> <timeout_s> bash scripts/run_h1a.sh <tag> smoke tests timeout ctx bench_public`
(takes the shared GPU lock; logs in `/raid/vllm-runtime/work/host/logs/<tag>.log`).

## Cut transport probe (negative result, input for the large-message track)

Above [4,8192] the all-reduce is limited by the partner exchange across the socket cut (2S bytes per direction with
the fp32 contract). `bench/host_relay_probe.py` measures the partner exchange alone (pairs r <-> r^4 exchange V bytes in
both directions at once, 8 GPUs, eager launches, verified data) for four transports; GB/s are per GPU and per
direction (x4 = aggregate over the cut), 2026-10-23, V = 8 MiB:

| transport | us | GB/s per GPU | aggregate over the cut |
|-----------|----|--------------|------------------------|
| SM push into the partner's IPC buffer (best: 1 CTA x 1024 threads) | 898-909 | 9.3 | 37 |
| SM stores/loads through pinned host shared memory on the writer's NUMA node | 636 | 13.2 | 52.8 |
| same, buffer on the reader's NUMA node | 706-712 | 11.8 | 47.3 |
| copy engine D2H + flag kernel + H2D through the same pinned memory (2 chunks) | 637 | 13.2 | 52.7 |

(256 KiB: push 27.2 us vs host relay 23.7 us; 64 KiB: 8.7 vs 8.9 us; the copy-engine variant has a ~58 us floor.)
The host path was expected to scale with the per-switch uplinks (57.7 GB/s each way, measured for host DMA alone) but
it saturates at ~53 GB/s aggregate, the same order as the copy engine P2P ceiling (43-48 GB/s): the socket cut itself is
the limit for every transport, and the best SM push (1 CTA, 37 GB/s) is already within 30% of it. A2 at [128,8192] and
[512,8192] runs at ~35 GB/s over the cut (92% of the SM push probe). So a faster transport can give at most ~1.3-1.4x
on the cut hop; the lever that matters is the byte count (opt-in bf16 partial exchange halves it, class
deterministic_different).
