# NUMA-local KV CPU offload (S1-10): SimpleCPUOffloadConnector + `--numa-bind`, exact per-rank tier

> **Status 2026-11-11 (S1-10R): the numbers in this document are 0.32.0 numbers.** The gate of this study ran on vLLM 0.32.0 (`268c64b0e2` + `9388ed1133`) with host `6f47eb1790`;
> production is now the 0.33.0 line (`sm120-muse-spark-open` `0261903b9e`, host `3192c6dd77`: spec KV bytes without a tier 11,720,212,480, `--max-num-scheduled-tokens 8192`, `NCCL_IB_DISABLE=1`, `VLLM_PLUGINS=muse_topo`).
> Every flag-keyed or bytes value (the offload-on GPU bytes 11,525,693,440 / 16,957,243,392 / 17,357,799,424, the bytes per block-id 20,480 / 65,536, the 45 GiB per-rank tier, the per-node A_N and the margin M of section 4,
> the S_live figures) is **0.32.0-derived and not re-derived on 0.33.0**; `env/offload.env` and the checks carry the same marks. S1-10R re-derives them (phase 3) and re-runs the gate chain (phase 4);
> until then treat them as study facts, not as launch values for 0.33.0. The offload branch is rebased as `s110r-offload` onto host main `3192c6dd77` without changing the design or the frozen gate tools.

Launch-scoped plus two generic vLLM commits (branch `muse/s110-offload` on `sm120-muse-spark-open` 268c64b0e2): 9388ed1133 "Keep a local KV cache connector from turning on the multimodal embedding path of a text-only launch" (needed) and
359cdc74aa "Add a NUMA memory policy option (bind | preferred)" (optional, section 8). Host side: `env/offload.env` (sourced after `env/host.env`, before `env/spec_k7.env`), `scripts/check_offload_launch.sh` (the boot check), `bench/s110/` (the tools).
Measured on `sm120-muse-spark-open` 268c64b0e2 + 9388ed1133 and host 6f47eb1790, real weights, EP8 + TP8, fp8 KV, max-model-len 131072. The pre-registered adoption rule and its amendments are in `bench/s110/rules/`.

**Result in one paragraph.** Offload gives a large benefit on revisited long prefixes and costs nothing measurable without reuse (the reuse numbers below are the BEST CASE of a tier: whole stored prompts re-sent cyclically over a working set 1.6x the GPU pool, 4 clients, no background decode, an upper bound of what mixed
traffic gains; the final gate of section 10 adds Zipf-popular documents with fresh question suffixes next to 64 background decoders). With the tier on, the 64k / 118k re-admission TTFT falls from 6.30 / 12.44 s to 0.29 / 0.47 s
(speculative profile; non-speculative 6.20 / 12.27 s -> 0.20 / 0.32 s), a working set 1.6x the GPU pool that cycles through the pool is served at 217k instead of 10.9k prompt tokens/s (non-speculative 286k instead of 11.3k; TTFT p50 0.24 s against 5.9 s,
p99 0.50 s against 12.0 s), a multi-turn chat whose histories overflow the pool finishes in 322 s instead of 760 s. Decode without reuse stays within -0.5 % (speculative: pooled gmean over c1 / c16 / c64 / c128, three boots against three; non-speculative -0.1 %).
The tier is one `torch.zeros` + `cudaHostRegister` per rank (SimpleCPUOffloadConnector), 45 GiB per rank (48,318,382,080 B; node 7 binds it), placed 100 % on the GPU's own node by `--numa-bind`; the exact GPU KV bytes move to 11,525,693,440 (speculative, -0.41 %;
the page tables of the tier take 90 MiB) and stay 16,957,243,392 (ns256). The native OffloadingConnector was measured on the same workloads and is not used: it pins the whole region in every worker (+100 s .. +21 min boot and 8x the GPU page tables at a 24 GiB tier),
its decode gmean is 0.979 .. 0.988, and under speculation a CPU hit does not reproduce the GPU hit (the draft group is not restored faithfully). A vLLM bug found on the way (a `kv_both` connector switched the text-only launch to the multimodal embedding path: +128 MiB
and a model on input embeddings) is fixed in the branch.

## 1. Facts of the box (step 1)

| fact | value (evidence) |
|---|---|
| GPU -> NUMA node | GPU 0,1 -> 3, GPU 2,3 -> 0, GPU 4,5 -> 7, GPU 6,7 -> 4 (vLLM auto-detect "Binding worker subprocess (local_rank=i, gpu_index=i) to NUMA node n", `/proc/<pid>/status` Cpus_allowed_list and numa_maps `bind:n` of every worker); nodes 1, 2, 5, 6 have no GPU |
| prefix caching | on in every production line (engine default; `enable_prefix_caching=True` in the boot log, `Prefix cache hit rate` in the stat lines); both connectors need it |
| `expandable_segments` | not set anywhere: not in `env/host.env`, not in `run_src_topo.sh` / the harness `lib.sh`, and `PYTORCH_CUDA_ALLOC_CONF` is absent from the environment of all 11 processes of every boot (`facts_worker_env.txt`, `expandable_segments_set=0`); vLLM rejects a KV connector together with it |
| `ulimit -l` | unlimited (pinned memory is not limited by RLIMIT_MEMLOCK); `cudaHostRegister` pages are NOT in `Mlocked` / `Unevictable` / `VmPin` (all 0): count them as anon (Simple) or shmem (native) of the node |
| `/dev/shm` | 504 GiB tmpfs, ~270 MB used by leftovers; used only by the native connector |
| NUMA policy | `numa_balancing=1`, THP madvise, `zone_reclaim_mode=0`, watermarks per node min 28 / low 157 / high 285 MiB. Without `--numa-bind` the workers already run on their GPU node's CPUs (Cpus_allowed_list 12-15,44-47 for GPU 0, 1 ...) but with memory policy `default` (first touch): 63-88 % of a worker's non-tier anon / heap pages are local; with `--numa-bind` the policy is `bind:N` and the pages are 100 % local (numa_maps of every worker, OFF arms `numa_after_reuse.txt`) |
| memory per GPU node | 125.7 .. 126.0 GiB each; with the server up (OFF, bound): workers 3.7 GiB anon + 0.3 GiB shmem each (2 per node), EngineCore 1.2 GiB anon (lands on node 3), API server 1.3 GiB anon (lands on node 7, unbound); worker growth over the whole workload < 0.1 GiB |
| node 7 | **13.3 GiB of memory that is neither page cache, anon, shmem nor slab at idle** (nodes 0 / 3 / 4: 0.9 .. 1.1 GiB), present before any server of ours (a foreign driver allocation, probably the Socket-Direct ConnectX-7): node 7 binds the tier size of every rank (every rank must hold the same number of CPU blocks) |

## 2. What the connector costs on the GPU, and the vLLM fix

The first connector boots used 340 MiB (ns256) and 504 MiB (speculative profile) more GPU memory at READY than the same launch without a connector, the same for both connectors and independent of the tier.
A CUDA allocator history diff (probe tree with memory-history recording: `bench/s110/sitecustomize_s110.py`, `bench/s110/snap_diff.py`) showed the whole difference of the live blocks to be one 128.0 MiB allocation,
`EncoderRunner.inputs_embeds` (`[max_num_batched_tokens, hidden]` bf16): `VllmConfig._resolve_mm_embedding_inputs` took `kv_role=kv_both` for a disaggregated KV consumer and forced
`enable_mm_embeds=True`, which switches a text-only launch of the multimodal Muse model (`--limit-mm-per-prompt` all 0) to the multimodal-input path (encoder cache, encoder runner, model on input embeddings; the boot log says
"enable_mm_embeds is True; modalities handled as embedding-only"). The fix (vLLM branch, `KVConnectorBase_V1.receives_remote_kv`, False for both CPU offload connectors) removes it:
GPU memory at READY +48 MiB at the 24 GiB tier (spec; the cudaHostRegister page tables, 2 MiB per GiB registered per process), +56 MiB at 28 GiB (ns256); cold prompts produce bitwise the same output as without a connector again.

## 3. Connector comparison (step 2): 2x2 {Simple, native} x {`--numa-bind` on / off} at an equal 24 GiB tier, speculative profile

Same tree, same host, one boot per arm unless stated; decode = real-prompt greedy closed loop, n = max(16, 4c) requests per cell with a `cache_salt` per request (no request can hit the GPU cache or the tier), 2 repetitions,
output tok/s; baseline = the OFF arm of the same bind policy (2 boots for OFF_B). REU = 80 documents of 32,768 random tokens (1.6x the GPU pool), 4 clients, passes 2 and 3 (reuse passes) pooled.

| arm | decode gmean (min cell) | REU TTFT p50 / p99 (s) | REU prompt tok/s | re-admission 64k / 118k (s) | READY (s) | GPU used at READY (MiB) |
|---|---|---|---|---|---|---|
| OFF, bound (reference; 3 boots decode, 1 reuse) | 1 | 5.36 / 11.98 | 10.87 k | 6.30 / 12.44 | 80-100 | 94,141 |
| **Simple, bound** (3 boots decode, 1 reuse) | **0.9949 (0.9886)** | **0.244 / 0.498** | **217.1 k** | **0.287 / 0.466** | 90 | 94,189 |
| Simple, not bound | 0.9985 (0.9869) | 0.239 / 0.520 | 216.2 k | 0.300 / 0.519 | 100 | 94,189 |
| native, bound, blocks_per_chunk 1 | 0.9795 (0.9663) | 0.333 / 1.483 | 160.6 k | 0.253 / 0.421 | 210 .. 1,271 | 94,527 |
| native, not bound, blocks_per_chunk 1 | 0.9819 (0.9742) | 0.326 / 1.297 | 165.7 k | 0.245 / 0.408 | 181 | 94,527 |
| native, bound, blocks_per_chunk 16 | 0.9880 (0.9724) | 0.270 / 0.415 | 220.7 k | 0.192 / 0.270 | 240 | 94,527 |
| Simple, bound, `lazy_offload` (2 boots decode, 1 reuse) | 0.9933 (0.9876) | 0.237 / 6.673 | 58.9 k | 0.309 / 12.4 (not stored) | 120 | 94,189 |

Reading the table (the native bound rows are recomputed here against the same three pooled OFF_B boots as the Simple rows, `compare_s110.py`; the running notes had used the first OFF_B boot only, 0.9796 (0.9714) and 0.9881 (0.9775), and the first version of this table
carried two rounding slips; the not-bound native row is against its single OFF_N boot):


* **Benefit.** Every connector arm turns the re-admission of a 64k / 118k prompt from a 6.3 / 12.4 s recompute into a 0.2 .. 0.5 s load, and the cycling working set (80 x 32,768 tokens, 1.6x the GPU pool, the
  GPU LRU hits nothing) into an almost pure load workload: 99.95 % of the prompt tokens of passes 2 and 3 come from the tier (`external_prefix_cache_hits_total`). The multi-turn chat (96 conversations, 5 turns of 6,000 new tokens,
  histories up to 30,640 tokens, 2.94 M tokens of final working set) finishes in 322 s (Simple, bound; 324 s not bound) against 760 s without the tier; turns 3-5 have TTFT p50 1.27 / 1.33 / 1.39 s against 3.25 / 3.67 / 4.85 s and
  p99 2.7-3.0 against 6.5-11.0 s (before turn 3 the pool still holds every history, so the arms are equal there).
* **Decode without reuse** (salted real prompts, no request can hit anything): Simple -0.5 % pooled over three boots against three OFF boots (cells c1 -0.2 %, c16 -0.1 %, c64 -0.6 %, c128 -1.1 %; the OFF-to-OFF spread of the c128 cell is 1.2 %).
  `--numa-bind` itself costs nothing: OFF not bound / OFF bound gmean 0.995 .. 1.005. Under speculation the native connector costs 1.2 .. 2.1 % (stores and 12-SM Triton loads next to the draft step; none without speculation, section 3b).
* **Boot and GPU memory.** The native connector registers the whole region in every worker: eight times the page tables (386 MiB against 48 MiB of GPU memory at a 24 GiB tier, 2 MiB per GiB registered per process) and, because the driver serializes the
  registrations (workers wait in `os_acquire_rwlock_write` while one of them runs for 20 CPU-minutes), +100 .. 150 s READY in three boots and 1,271 s in the fourth. Simple pins only the tier of the rank itself.
* **`lazy_offload` loses**: the blocks are stored when they come close to eviction, so the re-admission right after the prompt finishes finds nothing (118k: no gain), p99 of the cycling pass is 6.7 s and the reused fraction 0.86; decode is not cheaper.
* **Binding is not about TTFT.** On this profile bound and not bound give the same TTFT, throughput and chat time (the loads move 197 MB per rank per 32k document: a remote page costs nothing measurable; on ns256 the bound arm is 9-12 % slower on a 0.25 s TTFT, section 3b). What the bind changes is where the tier lives:
  100.0 % of every rank's tier pages on the GPU's own node against 85 / 86 / 85 / 86 % (socket 0) and 73 / 67 / 74 / 74 % (socket 1) without it, the rest on neighbouring nodes (peaks of 5-9 GiB on the GPU-less nodes 1, 2, 5, 6 at a 24 GiB tier).
  That is what makes the per-node arithmetic of section 4 valid (a rank can only take what its own node gives).


### 3b. The same measurements, non-speculative ns256 (`--max-num-seqs 256 --max-cudagraph-capture-size 256`, exact bytes 16,957,243,392, tier 28 GiB per rank = 458,752 blocks x 65,536 B)

REU = 143 documents of 32,768 random tokens (1.6x the pool of 2.91 M tokens), 2 passes (the second is the reuse pass). Decode = the same salted protocol, 2 repetitions, one boot per arm.

| arm | decode gmean (min cell) | REU TTFT p50 / p99 (s) | REU prompt tok/s | re-admission 64k / 118k (s) | READY (s) | GPU used at READY (MiB) |
|---|---|---|---|---|---|---|
| OFF, bound (c1 114.1 / c16 538.2 / c64 989.8 / c128 1,234.3 tok/s) | 1 | 5.06 / 9.36 | 11.28 k | 6.20 / 12.27 | 80 | 95,421 |
| OFF, not bound (decode vs OFF bound) | 0.9995 (0.9931) | 5.08 / 9.41 | 11.22 k | 6.23 / 12.31 | 90 | 95,421 |
| **Simple, bound** | **0.9993 (0.9915)** | **0.270 / 0.317** | **286.3 k** | **0.204 / 0.321** | 80 | 95,477 |
| Simple, not bound | 0.9988 (0.9959) | 0.218 / 0.298 | 321.6 k | 0.206 / 0.306 | 90 | 95,477 |
| native, bound, blocks_per_chunk 1 (28 GiB per rank = 224 GiB region) | 1.0036 (0.9975) | 0.239 / 0.601 | 306.9 k | 0.149 / 0.233 | 170 | 95,871 (+450) |

Same picture as the speculative profile: no decode cost (-0.1 %), a 25-29x faster cycling working set, re-admission 6.2 / 12.3 s -> 0.20 / 0.32 s, READY unchanged, +56 MiB of GPU memory (28 GiB of tier).
Without speculation the native connector is competitive on throughput and re-admission (0.15 / 0.23 s, no decode cost, outputs equal to the GPU hit in the pass-1 smoke), but it still needs READY 170 s against 80 s, and 450 MiB of GPU memory (8 x the page tables of the 28 GiB tier) against 56:
at the ns256 headroom (S_live 508.9 against the floor 500 with the Simple tier) that is a step-down of the KV bytes by about 350 MiB (-2 % of the KV tokens), charged to the arm by G4; with the speculative profile (the production line) it fails G1, G2 and G5 as shown above. One connector for both profiles: Simple. Repeat pair (one more boot each, 4 passes; reuse passes TTFT p50 / prompt tok/s): Simple not bound 0.218 | 0.224, 0.245, 0.220 s and 321 k | 318 k, 314 k, 319 k; Simple bound 0.270 | 0.271, 0.229, 0.250 s and 286 k | 281 k, 309 k, 311 k.
**Named deviation from the pre-registered rule** (`bench/s110/rules`: "B not worse than N by more than 3 % on REU p50"): on the speculative profile (the production line) the bound / unbound twins are within 2.1 % (0.244 / 0.239 s), on ns256 the bound arm is 9-12 % slower in the median (25-30 ms of a 0.22-0.27 s TTFT; the first reuse pass of both bound boots is 0.27 s,
the later passes 0.23-0.25 s). The 3 % clause was calibrated for multi-second TTFTs; at 0.2 s absolute a gap of this size is a few client scheduling slots of a 4-client closed loop, but it is consistently on the bound side, so it is reported, not explained away. The bind stays adopted
because the production profile passes the clause, it keeps the per-node arithmetic of section 4 valid (100 % local, no spill onto the GPU-less nodes) and it costs nothing in decode; a non-speculative deployment that values the last 25 ms of a reused prefix over placement can run `MUSE_OFFLOAD_NUMA_BIND=0`
(the tier is then 15-33 % off-node and spreads over all nodes).
**Rule conformance, stated plainly.** (a) On ns256 the pre-registered clause "B not worse than N by more than 3 % on REU p50" FAILS (9-12 %, confirmed by the repeat pair), so the rule-conformant ns256 default is the unbound twin (`MUSE_OFFLOAD_NUMA_BIND=0`); keeping the bind there
is a judgement override recorded above (on the speculative profile the clause holds). (b) The G1 clause grants a failing arm one retry boot; the native arms did not use it: native blocks_per_chunk 16 reads 0.9880 against the 0.990 line, inside the 1.2 % OFF-to-OFF spread of the c128 cell, with
G5 evidence from the bpc 1 boot only, so "native fails G1 / G5" is established for bpc 1 (0.9795, min cell 0.9663) and for the speculative CPU-hit output of the 65k smoke (bpc 1), not for bpc 16. The choice does not rest on it: Simple's reuse TTFT p50 is 10.7 % better than native bpc 16 (0.244 against 0.270 s,
above the 5 % tie band), its READY is 90 s against 240 s and its GPU cost 48 MiB against 386 MiB. (c) The pre-registered rule was amended three times before the boots it governs (md5 of each file in `bench/s110/rules`); the final gate of section 10 is a separate, fourth rule.

### 3c. Cold prefill with eager stores (study measurement, single boots; the final gate repeats it in 3 sessions, guard G2)

Eager mode stores every completed block while the prompt is still being prefilled (about 6 KB per token per rank, 197 MB per 32k prompt per rank) next to the sequence-parallel collectives; the cold passes of the study boots show what that costs. Pass 1 of the reuse cycle (80 / 143 documents of 32,768 tokens, 4 clients, nothing to hit) and the cold 64k / 118k singles:

| profile / arm | pass 1 prompt tok/s | cold 64k TTFT (s) | cold 118k TTFT (s) |
|---|---|---|---|
| spec, OFF bound | 10,876 | 6.277 | 12.431 |
| spec, Simple bound | 10,827 (-0.45 %) | 6.299 (+0.35 %) | 12.466 (+0.28 %) |
| spec, Simple not bound | 10,812 (-0.35 % against OFF not bound 10,850) | 6.369 | 12.499 |
| spec, native bound bpc 1 / bpc 16 | 10,821 / 10,851 | 6.334 / 6.301 | 12.504 / 12.456 |
| ns256, OFF bound | 11,271 | 6.185 | 12.260 |
| ns256, Simple bound | 11,212 (-0.53 %) | 6.196 (+0.18 %) | 12.308 (+0.39 %) |
| ns256, Simple not bound | 11,201 (-0.17 % against OFF not bound 11,220) | 6.206 | 12.329 |

A consistent 0.2-0.5 % on cold long prompts with the tier on; Chat turns 1-2 (cold) were equal within noise in the study (turn tables `chat_table_s110.py`).

## 4. Sizing (step 3)

Pass 1 (log of any boot with a small tier): `SimpleCPUOffloadScheduler: Allocating N offload blocks` and the eight `SimpleCPUOffloadWorker [CPU]: K tensors, N CPU blocks` lines give the bytes per block-id,
`bpb = per_rank_bytes / N` (floor): **20,480 B** in the speculative profile (565,108 GPU blocks = 11,573,411,840 B; K = 5 tensors of 4 KiB layer pages) and **65,536 B** in the non-speculative profiles
(258,747 blocks = 16,957,243,392 B; K = 16 tensors). The scheduler's CPU pool is derived from the GPU config (`num_gpu_blocks * cap // gpu_total_bytes`), each worker allocates `cap // total_bytes_per_block`: the two agree when
`cap` is a multiple of `bpb` (the check script fails a log where they differ). A per-rank value that is a multiple of **L = 327,680 B** (lcm of both `bpb`) therefore gives an exact block count in every profile.

Pass 2 (the formula). Host memory is not specific to a profile, so one formula serves all of them (and SG2 at 1M context, where only `bpb` and the bytes per token change):

    A_N   = MemFree_N + Active(file)_N + Inactive(file)_N         (MiB of GPU node N with the server up, tier off, bound, after the workloads; slab and shmem not counted)
    P*    = L * floor( (min_N A_N - M) / (2 * L) )                  per rank, the same for every rank (every rank must hold the same number of CPU blocks)
    CPU blocks per rank = P* / bpb                                  tokens of one prefix of T tokens cost 4 * ceil(T / 16) + 11 * 128 blocks (spec), ceil(T / 16) + 3 * 128 (ns)

Measured A_N (MiB, OFF bound, after 3 reuse passes and the chat run; `numa_end.txt`): node 0 117,738, node 3 117,205, node 4 118,063, **node 7 98,892** (node 7 carries 13.3 GiB of foreign memory, section 1).
The margin M is what the node must keep for the processes that are not the tier: EngineCore 1.2 GiB anon (1.6 GiB RSS) and the API server 1.3 GiB (1.7 GiB RSS) are already inside A (both sit on nodes 3 / 7), worker growth over the whole
workload +0.03 .. 0.09 GiB per node, the kernel's high watermark 0.28 GiB, the mapped library pages of the workers (0.9 GiB per node: evicting them makes the workers fault their code back in), page cache of the tools and logs (about 1 GiB),
growth of the foreign allocation and of the `/dev/shm` leftovers (0.3 GiB seen); together about 4.5 GiB, taken as **M = 6.6 GiB** (not itemised: the scheduler-side CPU pool of the connector, 2.36 M `KVCacheBlock` objects, about 0.3 GiB of EngineCore memory on node 3 plus a hash map that grows as the tier fills, sits inside the slack) so that node 7 never drops below 6.6 GiB of reclaimable memory:

| node | A_N (MiB) | largest rank size with M = 6.6 GiB | left unused at P* = 45 GiB (both ranks) |
|---|---|---|---|
| 0 | 117,738 | 54.2 GiB | 18.4 GiB |
| 3 | 117,205 | 54.0 GiB | 18.0 GiB |
| 4 | 118,063 | 54.4 GiB | 18.8 GiB |
| 7 | 98,892 | **45.0 GiB** | 0 |

The tier is therefore limited by node 7: **P* = 45 GiB = 48,318,382,080 B per rank** (the literal formula with A_7 = 98,892 MiB and M = 6.6 GiB gives 46,066.8 MiB = 44.99 GiB, i.e. 43 blocks of L fewer; the shipped 45 GiB is the largest clean boot of section 4 rounded to
the GiB, which leaves M = 6.57 GiB on node 7 by the formula and 10.4 GiB measured at READY) (a multiple of L; 2,359,296 CPU blocks in the speculative profile, 737,280 in the non-speculative ones; 360 GiB, 35.7 % of the RAM, in total).
(The 18 GiB per node that nodes 0 / 3 / 4 could still give is the price of equal block counts per rank and of the foreign memory on node 7; `--numa-membind-policy preferred`, section 8, is the option that uses it.)

Verification by boot (SIM bound, speculative profile, strict membind, watchdog at 1,024 MiB of MemFree + file cache per node, oom_score_adj 1000 on every process of the session):

| tier per rank | READY (s) | node 7 MemFree + file cache at READY (MiB) | direct reclaim scanned : stolen | major faults during the boot | smoke: CPU hit TTFT of a 65k prompt |
|---|---|---|---|---|---|
| 24 GiB | 90 | 48,548 | 1.0 | 17 | 0.28 s |
| 44 GiB | 110 | 7,919 | 1.0 (1.96 M pages) | 32,648 | 0.30 s |
| **45 GiB** | **100** | **10,415** | **1.01 (1.19 M pages)** | **336** | 0.30 s |
| 46 GiB | 170 | 9,519 | 4.5 (14.3 M pages) | 261,426 | 0.27 s |
| 47 GiB | 150 | 6,477 | 5.0 (47.1 M pages, 145 k movable allocation stalls) | 449,417 | 0.28 s |

No OOM, no watchdog trip, no populate stall up to 47 GiB; the tier is `torch.zeros` (touches every page: strict membind evicts the node's page cache for it) followed by `cudaHostRegister`. 45 GiB is the largest size whose populate is clean; from 46 GiB the kernel scans the LRU
12-40 times more pages than at 45 GiB and the workers refault 0.26-0.45 M pages (their mapped libraries). The 44 GiB boot shows that this is state dependent (32,648 faults after a boot that had left a different page-cache state): the knee is 45-46 GiB,
and 45 GiB keeps its predicted margin of 6.6 GiB on node 7 (measured 10.4 GiB at READY).
A boot right after the 47 GiB run (the weight page cache of the nodes was evicted): READY 130 s against 80-90 s warm (G2 holds: the cold-weight penalty of the maximum tier is +40 s).
**Limits of the margin (what the sizing does not prove).** The 45 GiB boot is one boot; the knee is state dependent (the 44 GiB boot had 32.6 k major faults, the 45 GiB boot 336, READY avail 7.9 against 10.4 GiB, non-monotonic) and node 7's 13.3 GiB of foreign memory and the `/dev/shm` tenants
can change at any time. A populate that runs out of memory under strict membind ends in a constrained OOM kill whose victim is the largest eligible RSS task: one of our workers (about 49 GiB), or a bigger foreign process. Therefore (i) `scripts/preflight_offload.sh` checks MemFree + file cache of the four GPU nodes
against 2 x per-rank bytes + what the server itself puts on the node (fail below 4 GiB left, warn below M = 6.6 GiB; the final gate runs it before every ON boot), and (ii) the server session should run with `oom_score_adj` 1000 (`echo 1000 > /proc/self/oom_score_adj` in the launching shell: children inherit it, an OOM kill then
hits this server and never another user's process; the study harness did that for every process of its sessions). The final gate boots the 45 GiB tier on every ON boot (3 + 3 sessions plus the stress and smoke boots, in different page-cache states).
Not bound, the same 46 GiB boot (SIM_N) takes 100 s with almost no reclaim, because the tier spreads over all eight nodes (peaks n0 88, n3 50, n4 53, n7 83, n1 33, n2 38, n5 29, n6 37 GiB): an unbound tier could be 2-3x larger but is 15-30 % off-node.

CPU bytes per token (speculative profile, per rank; SG2 reuses them): 15 KV groups = 4 full-attention groups (16 target layers + 4 padding layers = 4 x 5 layers) + 11 sliding-window groups (46 + 4 padding target layers = 10 groups, the DFlash2 draft = 1 group), each GPU block-id = 20,480 B (5 layers x 4 KiB):
**5,120 B per token** of prompt (4 x 20,480 B per 16 tokens) plus a fixed **28.8 MB per stored prefix** (11 groups x 128 window blocks x 20,480 B); measured: a 32,832-token document loads 9,612 blocks per rank = 196.9 MB = 6.0 KB per token, a 65,584-token prompt 17,804 blocks (4 x 4,099 + 11 x 128).
Non-speculative profiles: 4,096 B per token (1 full group x 65,536 B per 16 tokens) plus 25.2 MB (3 x 128 x 65,536 B); a 65,584-token prompt loads 4,483 blocks. The 45 GiB tier holds about 245 documents of 32k tokens (7.9 M tokens), 76 prompts of 118k, 8 prefixes of 2^20 tokens (5.40 GB each, fp8; 9 only with 10^6-token prefixes).


## 5. GPU bytes with the tier on (S1-7 procedure)

The tier costs GPU memory only through the page tables of `cudaHostRegister` (2 MiB per GiB registered per process, probe `gpu_overhead_probe.py`; 90 MiB at the 45 GiB tier), so the exact `--kv-cache-memory-bytes` of S1-7 / S1-9 is re-derived with the S1-7 procedure
(probe tree = the fixed tree + `sitecustomize.py` allocator probe, NVML 10 ms sampler, stress A1 B C1 C2 D H I L M X (speculative) / A1 B C1 C2 D H I X, REPS = 2, tier 45 GiB, bound; floor 1,256 MiB speculative / 500 MiB non-speculative; the stress also exercises the tier with
its prefix-hit phases H and I). S_live = idle free memory - exact growth of torch's allocated peak:

| profile | KV bytes | idle free (MiB; production without tier) | live growth (MiB) | S_live (MiB) | result |
|---|---|---|---|---|---|
| spec, production bytes | 11,573,411,840 | 3,019.4 (3,111.4) | 1,808.9 | **1,210.5** < 1,256 | clean (0 failures, 0 OOM), but under the floor: step down by 45.5 MiB |
| spec, stepped (rule) | **11,525,693,440** (562,778 blocks = 1,622,551 KV tokens per rank, -2,330 blocks = -6,718 tokens = -0.41 %) | 3,065.4 | 1,808.9 | **1,256.5** >= 1,256 | clean (0 failures, 0 OOM, health 200, 5 allocator retries): adopted |
| ns256, production bytes | **16,957,243,392** (unchanged; 258,747 blocks = 2,910,864 tokens) | 1,741.4 (1,831.4) | 1,232.5 | **508.9** >= 500 | clean; no step (headroom +8.9 MiB) |
| ns128 | 17,357,799,424 (S1-9) | not booted | | predicted 510 (S1-9 S_live 600.1 - 90) | re-confirm with the same stress before adopting |

The bytes are now a property of "offload on": `env/offload.env` exports them (`MUSE_OFFLOAD_KV_BYTES`, and `MUSE_SPEC_KV_BYTES_OVERRIDE` for the speculative profile, which `env/spec_k7.env` reads), and `scripts/check_offload_launch.sh` fails a launch line whose
`--kv-cache-memory-bytes` is above it. `scripts/check_spec_launch.sh` keeps its maxima (the stepped value is below them). Valid without `prompt_logprobs` / `echo` (S1-7 section 4). Re-derive after any change of the tier size (2 MiB per GiB), the sampler, the verify path, the graph set or the chunk size.


## 6. Determinism (G5)

Instrument: `bench/s110/reuse_driver.py --scenario smoke`, every request alone (no batching noise), greedy, `logprobs=1`, per length (13,107 / 65,536 / 117,964 random-token prompts): cold -> the same prompt + 64 new tokens twice from the GPU cache
(`gpuhit1`, `gpuhit2`) -> `POST /reset_prefix_cache` (GPU pool only) -> the same prompt twice (`after_reset1`, `after_reset2`: connector hits with a tier, recomputes without). Pairs with the SAME hit length are compared token by token and logprob value by logprob value
(`pairs_s110.py`, `cross_s110.py`). `gpuhit1` has a shorter hit than `gpuhit2` (the first hit stores 4 more blocks), and a recompute is a different computation from a hit: those pairs differ even without a connector and are not evidence.

| profile / arm | GPU hit vs connector hit (`gpuhit2` vs `after_reset1`) | repeated connector hit | across boots: OFF `gpuhit2` vs connector hit, cold vs cold |
|---|---|---|---|
| speculative, Simple, bound | 3 / 3 lengths bitwise equal (tokens and logprob values) | 3 / 3 equal | equal at all lengths; the CPU hit's speculation counters equal the GPU hit's (65k: 11 accepted of 70 drafted in 10 drafts) |
| non-speculative ns256, Simple, bound | 3 / 3 equal | 3 / 3 equal | equal at all lengths |
| non-speculative ns256, native | equal (pass-1 smoke at 65k; text and logprob hash) | equal | not repeated |
| speculative, native (bpc 1, bound) | 13k and 118k equal, **65,536 differs from token 8 (max abs dlogprob 0.093)** | 3 / 3 equal | the CPU hit of the 65k prompt has other counters (4 accepted of 98 drafted in 14 drafts against 11 of 70 in 10): the draft group's KV is not restored faithfully (`no KV-cache group is annotated as a drafter group; treating all groups as non-draft`), acceptance drops |

Without the vLLM fix of section 2 the connector arms also differed from the OFF arm at the first cold prompt (ns256 OFF `c2529b95101c` against `64f5e384f87f` in both connector arms; the model ran from input embeddings): with the fix, cold prompts are bitwise the same in every arm.
This is the online equivalent of the e2e protocol (hit source and repeat), not the offline `e2e_bitwise.py` run (the offline `LLM(...)` scripts use their own pool; S1-11 can pass the tier and the exact bytes to them). The 80-document cycle passes (4 clients) are NOT a usable pair:
two recomputes of the same prompt without any connector are text-equal in only 15-20 of 80 documents (random-token prompts are chaotic under greedy decoding and the bf16 exchange is not batch invariant), the same for the connector arms (33 of 80).


## 7. Launch lines and the check

Speculative production line with the tier (source order matters: `offload.env` sets `MUSE_SPEC_KV_BYTES_OVERRIDE`, which `spec_k7.env` reads when it is sourced):

    source $HOST_DIR/env/host.env; source $HOST_DIR/env/offload.env; source $HOST_DIR/env/spec_k7.env
    vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel \
      --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice \
      --speculative-config '{"draft_sample_method":"probabilistic","method":"dflash","model":"<RadixArk DFlash2 snapshot>","num_speculative_tokens":7,"rejection_sample_method":"block"}' \
      --per-request-spec-decode-metrics detailed $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD_SERVE_ARGS
    # = ... --max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --kv-cache-memory-bytes 11525693440 --numa-bind \
    #     --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":386547056640,"cpu_bytes_to_use_per_rank":48318382080}}'

Non-speculative (`MUSE_OFFLOAD_PROFILE=ns256 source env/offload.env` after `env/host.env`): `--max-num-seqs 256 --max-cudagraph-capture-size 256 --kv-cache-memory-bytes 16957243392 $MUSE_OFFLOAD_SERVE_ARGS` (ns128: `--max-num-seqs 128`, bytes 17,357,799,424, to be re-confirmed).
Through the harness: `SERVE_EXTRA="$MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD_SERVE_ARGS" SPEC_CONFIG=dflash2_radixark ... serve_sd.sh <tag> ...` (the JSON has no spaces: `SERVE_EXTRA` is split on whitespace).

After every boot: `scripts/check_spec_launch.sh <serve.log>` (speculative) and `scripts/check_offload_launch.sh <serve.log>` (with `env/offload.env` sourced in the checking shell). The second one checks from the log and `serve_cmd.txt`: the eight bindings
(`Binding worker subprocess ... to NUMA node 3 3 0 0 7 7 4 4`) and the EngineCore line, one scheduler allocation line and eight worker lines with the same block count (= per-rank bytes / bpb), the same tensor count on every worker, `Using LBNHC KV cache layout`, no `Pinned memory not available`,
no `cudaHostRegister failed`, no `enable_mm_embeds is True` (a vLLM tree without the fix), `--numa-bind` exactly when expected, the tier bytes and `--kv-cache-memory-bytes` not above the offload-on maximum. `bench/test_check_offload_launch.sh` runs 26 synthetic cases of it (no GPU).

The restart cost of the maximum tier: the tier evicts the weight page cache of the four nodes, the next boot reads part of the weights cold (130 s READY against 80-90 s; the stress boot with its debug logging 140-160 s READY).


## 8. The `preferred` memory policy (option, `--numa-membind-policy preferred`; NOT merged)

Decision of the S1-10 gate: the commit 359cdc74aa stays on the branch `muse/s110-offload` and is NOT merged into `sm120-muse-spark-open` (its own trigger, a strict-membind populate that stalls, never fired; it adds a CLI and config option that is silently ignored without `--numa-bind` and it logs its policy line once per worker call).
Only the `receives_remote_kv` commit 9388ed1133 is merged. If the option is ever merged it should warn when `--numa-membind-policy` is set without `--numa-bind` and log once.

`--numa-membind-policy preferred` (vLLM branch, `numa_utils`: `numactl --preferred=N` instead of `--membind=N`; default `bind`) was built to use the 18 GiB per node that nodes 0 / 3 / 4 could still give (section 4) and the memory of the GPU-less nodes. It works as designed
and is **not recommended**: one boot at 64 GiB per rank (SIM, bound CPUs, speculative profile; 512 GiB of tier in total) reached READY after 170 s with 559,173 major faults and 1.10 M pages scanned by direct reclaim, and only **35.6 %** of the pinned tier landed on its GPU's own node
(GPU 0-3: 45 %, GPU 4, 5: 39 %, GPU 6, 7: 13 %; the rest on the same-socket nodes 1, 2 / 5, 6 and, for the ranks of node 4, on node 5), with every node full (65-84 GiB of anon on all eight nodes). The CPU hit itself is not slower (65k: 0.276 s against 0.30 s) because loads are
bandwidth-trivial, so the policy trades locality for capacity: it is what to try when more than 45 GiB per rank is worth more than locality, and then the stress / bytes of section 5 and a 1M-prefix reuse run must be redone (the tier size also moves the page-table cost, 2 MiB per GiB).
The boots at 80 GiB and the OFF boot after them were not run (decision after the 64 GiB result, to keep the lock for the non-speculative A/B); tests of the option: `tests/utils_/test_numa_utils.py`, `tests/engine/test_arg_utils.py` in the vLLM branch.


## 9. For S1-11 and SG2

* **THE TIER SURVIVES `POST /reset_prefix_cache`.** That endpoint (and `llm.reset_prefix_cache()`) empties the GPU pool only; every prompt that was prefilled with the tier on stays in the tier. A benchmark that re-sends a prompt after such a reset therefore HITS THE TIER and reads as a speed-up: the first unsalted
  decode sweep of this study (`bench_sd.sh` rows resend the same 160 real prompts across cells and repetitions and reset only the GPU cache) read 2,142 tok/s at c128 against 1,340 without the tier, which was reuse, not overhead. S1-11's final matrix (real prompts re-sent across cells and repetitions, repeated TTFT
  8k / 32k / 118k prompts, random 512 / 128 rows with fixed seeds) must therefore either give every request its own `cache_salt` (the S1-9 L7 driver `--cycle`, `conc_driver_s110.py`) or reset the tier too: `POST /reset_prefix_cache?reset_external=true` (offline: `llm.reset_prefix_cache(reset_connector=True)`; a no-op without a connector).
  The sd harness is guarded since S1-10 (`/raid/vllm-runtime/work/sd/harness`: `bench_sd.sh`, `acceptance.py` and the mock server send `reset_external=true`, originals kept as `*.before_s110`, patch in `bench/s110/harness_reset_external.patch`); `scripts/conc_sweep.sh` does not reset at all (its repeated fixed-seed prompts hit the GPU cache
  with and without the tier alike, so its rows are symmetric). Any other row that resets only the GPU cache is invalid with the tier on. Tier-on numbers next to tier-off numbers of the same matrix must say which rows were salted.
* S1-11: the launch line of section 7 is the production speculative line plus `--numa-bind` and the connector; the exact GPU bytes are 11,525,693,440 (speculative, stepped by the rule; the stress of the stepped value is in section 5) and 16,957,243,392 (ns256, unchanged).
  Boots to check: `check_spec_launch.sh` and `check_offload_launch.sh`. A restart after a tier run reads part of the weights cold (+40 s).
* SG2 (1M context, NVFP4 KV): reuse the formula of section 4 unchanged (host memory is NOT independent of the context length: the model runner keeps a pinned UVA `all_token_ids` of [max_num_reqs, max_model_len] int32 per worker, 64 MiB at 128k with mns 128, 512 MiB at 1M, about 1 GiB at mns 256 and 1M,
  i.e. +0.9 .. 1.8 GiB per node at 1M against the 2.1 GiB of slack of M: P* = 45 GiB at 1M is unproven, re-derive A_N with the 1M launch), recompute `bpb` from the new `Allocating N offload blocks` line and the CPU bytes per token from the new group layout
  (token cost = full groups x bpb / block tokens, fixed cost = sliding-window groups x window blocks x bpb); a prefix of 2^20 tokens costs about 5.4 GB per rank at fp8 in the current layout, so the tier holds 8 of them, and the GPU side must carry the page-table cost of
  2 MiB per GiB of tier in its S1-7 bytes. The tier only serves the waiting (admission) path: a running 1M request keeps its full-attention KV on the GPU. If the NVFP4 block scales live in a separate tensor, the Simple connector (which copies the per-layer regions of the KV allocation) must be checked to copy them too.
* Open: the native connector would need three fixes to compete (pin only the rank's own region, e.g. a rank-major layout; annotate the Muse draft group as a drafter group; one pinned registration per node instead of eight), none is needed for the chosen configuration.
