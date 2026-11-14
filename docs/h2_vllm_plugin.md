# muse_topo vLLM plugin (tracks H2 and H2b)

Routes the tensor-parallel collectives of vLLM (v0.32, branch `sm120-muse-spark-open`) through `muse_topo.TopoComm`
without touching the vLLM tree or the venv. Accuracy class of the all-reduce: `more_accurate` (fp32 sum in a fixed
order, single final rounding; see `docs/h1a_collectives.md` and `docs/h1b_collectives.md`), all-gather is bit exact data
movement. H2 built the plugin on the small-message algorithms; H2b (this revision) integrated the large-message
algorithms of H1b: since then EVERY collective of the Muse model, decode and prefill, takes the topo path (the per-op
summary of the e2e run shows 188 keys on topo and none on NCCL, up to the 256 MiB all-reduce of a 16k-token chunk).

## Enable

```bash
export PYTHONPATH=<host folder>:<vllm tree>      # the dist-info directory muse_topo-0.1.dist-info is found on PYTHONPATH
export MUSE_TOPO_ENABLE=1                        # the plugin is a no-op otherwise
vllm serve ...                                   # or scripts/run_src_topo.sh / scripts/locked_run_topo.sh on the server
```

`muse_topo-0.1.dist-info/entry_points.txt` registers `muse_topo.vllm_plugin:register` in the `vllm.general_plugins` group
(no pip install). vLLM calls it in every process from `load_general_plugins()`; in the workers this happens in
`WorkerWrapperBase.init_worker`, before `init_device` creates the device communicators. If `VLLM_PLUGINS` is set it must
contain `muse_topo`. `scripts/run_src_topo.sh` and `scripts/locked_run_topo.sh` are copies of
`/raid/vllm-runtime/musetest/{run_src.sh,locked_run.sh}` (the originals are untouched) that add the host folder of the
script to `PYTHONPATH` and source `env/host.env`; `scripts/runsrc_topo.sh` does the same for plain python runs.
`MUSE_TOPO_ENABLE=0` with the same scripts gives the NCCL-only baseline.

## What it patches

`register()` patches `vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator` once per process:

| method | behavior |
|--------|----------|
| `__init__` | after the original constructor, the group named `tp*` with 8 ranks creates the collective `TopoComm` over its gloo `cpu_group` (JIT build + IPC workspace, 61.1 MiB per GPU, allocated during `init_worker_distributed_environment`, i.e. before vLLM's `init_snapshot`: see the note on KV sizing below). Result in `communicator.topo_comm` (`None` when the host does not match, the build fails, or `MUSE_TOPO_ENABLE != 1`; all ranks agree) |
| `all_reduce` | `topo_comm.supports("all_reduce", x)` -> topo kernel, else the original chain (NCCL) |
| `all_gather(x, dim)` | `supports("all_gather", x, dim)` -> topo kernel producing the requested layout directly (dim 0: rank-major, last dim: column concatenation; shards above 1 MiB take the chunked large engine), else original |
| `reduce_scatter(x, dim)` | dim 0 and `supports("reduce_scatter", x, 0)` (T % 8 == 0, bf16/fp16/fp32, message >= 128 KiB) -> topo large reduce-scatter, else original (dim != 0 stays on NCCL) |
| `destroy` | logs the per-key call counters, closes the topo communicator |
| `reduce_scatter_all_gather_matches_all_reduce` (property, added by r2c) | `True` when the communicator owns a `TopoComm`, `MUSE_TOPO_OPS` routes all three collectives (`all_reduce`, `all_gather`, `reduce_scatter`; the fused `boundary` op does not matter), the large algorithms are on and the bf16 exchange is off; vLLM's base class says `False` (NCCL's reduce-scatter + all-gather rounds differently from its ring all-reduce) |

The property is the plugin's side of one contract: every topo reduction is one fp32 sum in a fixed order with a single final
rounding, whatever the algorithm and the message size, so `all_gather(reduce_scatter(x, dim 0), dim 0) == all_reduce(x)` bit
for bit (`bench/tests_h1b.py`, and `bench/test_plugin.py` on the shapes of the Muse sequence-parallel stream) for every tensor the
topo path takes: bf16 / fp16 / fp32, a row count that is a multiple of 8, a size that is a multiple of 128 B between 128 KiB and
2 GiB. Callers that shard a reduction along the rows read the property to know that sharding leaves the reduced values
unchanged; the Muse model's `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL=auto` is the first user. The three switches that turn the
property off are exactly the ones under which that equality does not hold: one of the three collectives that is not routed (it falls back to NCCL; the `boundary` op is not one of them, so
`MUSE_TOPO_OPS=all_reduce,all_gather,reduce_scatter` keeps the property on),
`MUSE_TOPO_LARGE=0` (reduce_scatter and the large all-reduce go to NCCL) and `MUSE_TOPO_LARGE_BF16_EXCHANGE=1` (the large
algorithms round the island partials to bf16, the all-reduce below 2 MiB does not). `env/host.env` turns that last one ON since
2026-10-24 (throughput decision), so under the host launch environment the property is False and `SEQUENCE_PARALLEL=auto` never
engages; the branch's default is the forced `SEQUENCE_PARALLEL=1`. The collective is only one of the ops of a
sharded stream: whether the stream as a whole equals the replicated one also depends on the GEMMs that run on the local rows
(see the branch's `_SequenceParallelRowGate`).

`supports()` is the only eligibility gate. Nothing else changes: not the EP group (same ranks, would cost a second
workspace), not `gather`, `reduce_scatterv*`, `all_gatherv`, send/recv.

CUDA graphs: the kernels use device-side epoch counters only, so they are captured by `FULL` and by the breakable /
piecewise capture of the branch (the collectives sit inside the captured segments; only attention breaks) and replay
without Python. Eager calls (prefill, warm-up, logits) run the same kernels. Streams: `graph_capture` runs warm-up and
capture on a side stream, replays and eager prefill run on the main stream; `TopoComm` serializes a stream change behind
the previous stream (one "collective issued on a second stream" warning per rank at start-up is expected and benign).

## Environment

| variable | default | meaning |
|----------|---------|---------|
| `MUSE_TOPO_ENABLE` | 0 (`env/host.env`: 1) | master switch (plugin and `TopoComm`) |
| `MUSE_TOPO_OPS` | `all_reduce,all_gather,reduce_scatter,boundary` | ops routed to the topo path (A/B switch); `boundary` = the fused collective + epilogue kernels (`docs/r2d_fused_epilogue.md`) |
| `MUSE_TOPO_TIMEOUT_S` | 600 (plugin) | spin timeout of the kernels, then `__trap()`; `TopoComm` alone defaults to 10 s, which is too short for start-up skew (per-rank JIT/Triton compiles run while the other ranks already wait inside a collective) |
| `MUSE_TOPO_MAX_BYTES` | `TopoComm` default (8 MiB) | largest all-reduce of the decode/medium algorithms (the workspace grows with it); larger all-reduces (up to 2 GiB), reduce_scatter and big all-gather shards use the chunked large algorithms through the same workspace, so there is no reason to raise it |
| `MUSE_TOPO_LARGE` | 1 | 0 disables the large algorithms (A/B switch: all-reduce above 8 MiB, reduce_scatter and all-gather shards above 1 MiB then go to NCCL, as in H2) |
| `MUSE_TOPO_LARGE_BF16_EXCHANGE` | 0 in the plugin (`env/host.env`: 1, the throughput default of 2026-10-24) | class `deterministic_different`: the large all-reduce / reduce_scatter exchange the two island partials rounded to bf16 (bf16 inputs only, half the bytes over the socket cut; batch invariant only inside the large algorithm, so with the flag on a row can get different bits depending on the padded batch size across the 2 MiB threshold). See the benchmark below and `docs/h1b_collectives.md` |
| `MUSE_TOPO_LOG_OPS` | 0 | one-time per-op summary at info level: 1 = group rank 0, 2 = every rank (0: debug level) |
| `VLLM_MUSE_SPARK_SM120_L2_PREFETCH_AFTER_EXPERTS_CTAS` | 4 in vLLM (`env/host.env`: 8 with `MUSE_TOPO_ENABLE=1`) | vLLM branch knob, not a plugin variable: CTAs of the decode L2 weight prefetch behind the experts of each layer. The plugin's layer (160 us) is shorter than the NCCL one (296 us), so the 4-CTA launch (113 us) is still running when the next pooler and router weights are due; 8 CTAs: TPOT 10.69 -> 9.80 ms at batch 1 (real weights, c1 512/128), whereas on NCCL 8 CTAs cost +1.1% (17.09 against 16.90 ms). Cache hint, no numeric effect |

Per-op summary: the first call of every `(op, dim, shape, dtype, stream, capturing, path)` logs
`muse_topo op[rank 0]: all_reduce shape=(1, 8192) bfloat16 stream=0x.. capturing=True -> topo` (or `-> nccl (reason)`),
and `destroy` logs the call count of every key (calls replayed from a CUDA graph run no Python, so a count is the number
of eager calls plus graph captures). `bench/summarize_ops.py <log>` tabulates the lines.

## Collectives of the Muse model and their path (e2e run of 6 layers, 2026-10-24, H1a + H1b algorithms)

Per forward of an L-layer model: 1 embedding all-reduce `[T,8192]`, per layer an attention-out all-reduce `[T,8192]`, a
MoE-out all-reduce `[T,4096]`, a pooler all-gather `[T,512]` (dim 0, rank-major) and an unpooler all-gather `[T,1024]`
(dim 0), plus the logits all-gather `[T,25256]` (dim -1, eager). The per-key call counters of the e2e run confirm the
structure at L=6, T=1: 7 / 6 all-reduces `(1,8192)` / `(1,4096)`, 6 / 6 all-gathers `(1,512)` / `(1,1024)`, 1 logits
gather per forward, on every one of the 8 ranks, in eager mode and in the captured graphs (`capturing=True`).

| collective | shapes seen | path |
|------------|-------------|------|
| embedding / attn-out AR `[T,8192]`, MoE-out AR `[T,4096]`, T < 256 | up to 2 MiB (all decode graph sizes 1..128) | topo, H1a 3-phase sentinel algorithm |
| the same ARs from 2 MiB (`[256,8192]` / `[512,4096]`) up to `[16384,8192]` (prefill chunks of 256 .. 16k tokens, 2 .. 256 MiB) | 2 MiB .. 256 MiB | topo, H1b large all-reduce (fused reduce-scatter + all-gather pipeline) |
| pooler AG `[T,512]`, unpooler AG `[T,1024]` (dim 0) | T = 1 .. 256 (the model uses replicated pooler / unpooler projections for larger T, no gather) | topo (LL8 up to 16 KiB shards, flag protocol above; the large engine takes shards >= 256 KiB, `bench/test_plugin.py` covers `[8192,512]` / `[4096,1024]` shards) |
| logits AG `[T,25256]` (dim -1) | T = 1 .. 1024 (the 1024-row case is the prompt-logprob run) | topo (H1a flag kernels up to a 1 MiB shard, the large engine above: the NCCL fallback of H2 for T > 20 is gone) |
| reduce_scatter (sequence-parallel helpers `sp_reduce_scatter`, none in the benchmarked configuration) | dim 0, T % 8 == 0, >= 128 KiB | topo large reduce-scatter |

No rank ever disagreed on a path (`summarize_ops.py`: 188 keys, 188 topo, 0 NCCL, 0 problems). Sequence parallelism
(`is_sequence_parallel_enabled`) needs `moe_backend == "deep_gemm_mega_moe"`, which is off on this box, so the e2e runs do
not issue a reduce_scatter; the wrapper is covered by `bench/test_plugin.py` (large reduce-scatter, RS + AG == AR bitwise,
inside a CUDA graph). The sequence-parallel helpers ask `custom_all_gather` / `custom_reduce_scatter` of the device
communicator first: vLLM's CustomAllreduce is disabled here (the serve logs of all runs show "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs" on all 8 ranks), those return `None` and the
helpers continue into `tensor_model_parallel_all_gather` / `_reduce_scatter`, i.e. into the patched methods.

## Validation (server, GPU lock, SRC=/raid/vllm-runtime/sm120-dev/src)

### H2b (2026-10-24, small + large algorithms, deployed folder `/raid/vllm-runtime/host_topo`)

One lock acquisition: `scripts/run_h1a.sh h2b tests other`, `bench/run_h1b.sh h2b tests timeout`,
`scripts/run_h2.sh h2b plugin e2e_topo1 e2e_topo2 e2e_nccl compare ops lp_nccl lp_topo lp_compare` (logs
`/raid/vllm-runtime/work/host/logs/h2b_a.log`). All PASS.

* `bench/tests_h1a.py` (full): 253 checks x 8 ranks, 0 failures; `bench/test_h1a.py --quick`: ALL TESTS PASSED.
* `bench/tests_h1b.py` (20000-replay soak): 289 checks, 0 failed; `bench/test_timeout_large.py`: 3/3 PASS (one rank skips a
  large all_reduce / reduce_scatter / all_gather, the others trap within the 3 s timeout).
* `bench/test_plugin.py`: 411 checks x 8 ranks (H2: 352), 0 failures. New since H2: prefill-sized all_reduce through the vLLM
  wrappers (bf16 16 / 64 / 128 MiB + a 1234-row tail chunk, fp16 32 MiB, fp32 16 MiB) bitwise == the arithmetic contract and
  identical on all ranks; `tp.reduce_scatter` bitwise == rows block of the contract (1 MiB .. 128 MiB); reduce_scatter +
  all_gather(dim 0) == all_reduce bitwise; big all-gather shards in both layouts (`[8192,512]`, `[4096,1024]` dim 0,
  `[256,25256]`, `[2048,1024]` dim -1) exact; a CUDA graph holding a large all_reduce, reduce_scatter and all_gather
  (replayed next to eager large calls); random mixed sequences of small / medium / large ops; every one of these counted
  on the topo path (no NCCL fallback; the deliberate fallbacks left are 200 B, int32, 8 MiB + 64 B (not a multiple of
  128 B), non-contiguous, middle-dim all-gather, reduce_scatter dim 1).
  `bench/test_plugin_unit.py` (mock `TopoComm`, CPU): PASSED, including the new fallback reasons of `_why_not`.
* End-to-end determinism (`nccl/e2e_bitwise.py`, 6 layers, dummy weights, `VLLM_ENABLE_V1_MULTIPROCESSING=0`): two separate
  processes with the topo path give identical greedy token ids and bitwise identical logprobs AND prompt logprobs, 20/20
  requests (incl. the 3000-token prompt with prompt logprobs and the 6000-token chunked prompt, whose all-reduces
  (47 / 94 MiB) now run on the large algorithm), max |dlogprob| = 0; the in-process repeat is identical too (8/8).
  topo vs NCCL (class `more_accurate`, differences expected): token ids diverge within the first 10 tokens (dummy-weight
  logits are nearly flat, ties decide), max |dlogprob| 1.9e-6 over the common prefix, 3000-token prompt logprobs max 1.9e-5.
  Teacher-forced comparison (`bench/e2e_prompt_lp.py`, 12 prompts of 16..512 tokens): 6150 values, mean |d| 2.2e-6, median
  9.5e-7, p99 8.6e-6, max 1.3e-5 (same as H2: the large algorithm and the small ones are bitwise identical by design).
* Per-op summary of the e2e run (`bench/summarize_ops.py`): 188 keys, 188 on topo, 0 NCCL fallbacks, 0 rank disagreements;
  the shapes reach `[16384,8192]` (256 MiB all-reduce) and the `[1024,25256]` logits gather.

### r2c (2026-10-24): sequence parallelism by default, `reduce_scatter_all_gather_matches_all_reduce`

Branch side (`muse/r2c-sp-auto`): `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL=auto` shards a prefill batch only
when this property holds and the sharded stream is bit for bit the replicated one at that batch size (since the round-2
integration the branch default is the forced `1`, and `auto` stays selectable; the rows below are the r2c measurements of `auto`). Validation, real
checkpoint, plugin on, SRC = the r2c tree:

* `bench/test_plugin.py --quick`: 173 checks x 8 ranks, 0 failures, with the new ones: the property is `True` with the
  `TopoComm` and `False` without it, and `all_gather(reduce_scatter(x, 0), 0) == all_reduce(x)` bitwise (and identical on all
  ranks) for the sequence-parallel shapes `[520, 8192]`, `[1024, 4096]`, `[3000, 8192]`, `[4104, 4096]`, `[8192, 8192]`,
  `[8192, 4096]`; all on the topo path, none on NCCL. `bench/test_plugin_unit.py`: the property's three switches.
* Shadow run of the SP stream (62 layers, 8 ranks, every row-sliced op also run replicated on the gathered batch, rows compared
  bitwise): `reduce_scatter` rows == `all_reduce` rows in 992 / 992 / 992 / 3968 calls at T = 1308 / 3000 / 4308 / 8192, and the
  expert path (partial + RS + AG) == the replicated runner forward (isolated from the router) in every call. The only ops that
  differ at some sizes are the cuBLAS pooler / unpooler / fp32 router on ceil(T / 8) rows (different split-K than on T rows); at
  T = 8192 they agree and the whole stream is identical, which is what the branch's per-size gate admits.
* End to end (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, greedy, 56 requests with a mixed batch and 12000 / 20000-token chunked prompts,
  ids + top-5 logprobs + prompt logprobs): `auto` vs `0`: 56 / 56 bitwise equal (12 sharded 8192-token forwards, 744
  `reduce_scatter` calls per shape on rank 0, all topo). With NCCL (`MUSE_TOPO_ENABLE=0`) `auto` keeps the stream replicated
  (property `False`): 56 / 56 equal to flag `0`.
* Real-weight benchmark, plugin, `SEED=9000 LOAD_FORMAT=auto`, pp t/s flag `0` -> `auto` (flag `1`): c1 512/128 4058 -> 3979
  (4470), ctx 13,107 6181 -> 7051 (7709), ctx 65,536 5937 -> 7411 (7424), ctx 117,964 5649 -> 6937 (6986); output tok/s random
  512/128 at c16 / c64 / c128: 392 -> 393 / 712 -> 748 / 886 -> 979 (flag `1`: 412 / 787 / 1009).

### H2 (2026-10-23, small-message algorithms; superseded by the H2b numbers above where they overlap)

* `bench/test_plugin_unit.py` (CPU only, mock `TopoComm`): routing of the wrappers, dim normalization, dim != 0
  reduce_scatter fallback, lazy fallback reasons, `MUSE_TOPO_OPS`, `register()` gate. PASSED.
* `bench/test_plugin.py` (8 processes, vLLM TP group built through the plugin entry point): 352 checks x 8 ranks, 0
  failures. Eager and CUDA-graph (vLLM `graph_capture`, replays on the main stream, eager calls between replays)
  all_reduce is bitwise equal to the arithmetic contract and identical on all ranks (bf16 `[T,8192]` / `[T,4096]`,
  T = 1..512, fp16), all_gather exact for both layouts, fallbacks (200 B, int32, 32 MiB, non-contiguous, middle-dim, big
  shard) correct, 400 mixed topo/NCCL ops without hang, a collective on a side stream, `reduce_scatter` wrapper (NCCL
  fallback today), `destroy`. Max error of the topo all-reduce relative to the L1 norm 0.0039 = bf16 unit roundoff
  (single rounding), NCCL ring 0.012 .. 0.018.
  Dispatch-level graph timing, 20 chained calls, max over ranks: all_reduce `[1,8192]` 5.67 us (NCCL 43.0 us),
  all_gather `[1,512]` dim 0 3.85 us (NCCL 24.7 us).
* End-to-end determinism (`nccl/e2e_bitwise.py`, 6 layers, dummy weights, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, greedy,
  8 short prompts x 64 tokens, mixed 17 / 300 / 3000-token prompts with prompt logprobs, one 6000-token chunked prompt,
  in-process repeat): two separate processes with the topo path give identical token ids and bitwise identical logprobs
  and prompt logprobs (`nccl/compare.py`: 20/20 requests identical, max |dlogprob| = 0), also the in-process repeat.
  A 128-request decode batch run twice in-process (`bench/e2e_prompt_lp.py`) repeats exactly (128/128), with and without
  the topo path.
* vs the NCCL baseline of the same tree (class `more_accurate`, so the numbers are expected to differ). Caveat: with
  dummy weights the logits are almost constant (the top-1 .. top-5 logprobs of every prompt position are identical, ~-12,
  one fp32 ulp there is 9.5e-7), so greedy decoding is decided by ties and any perturbation flips them; the end-to-end
  numbers bound the size of the perturbation but do not measure model quality (that needs real weights, see the
  follow-ups). Greedy token ids (`nccl/compare.py`, same tree, NCCL SYS vs topo): diverge within the first 10 tokens in
  all 20 requests (first divergence at token 0 .. 9), max |dlogprob| over the common prefixes including the
  divergent position 4.8e-6; for scale, NCCL SYS vs the NCCL graph-file variant (2026-10-21 dumps) diverges in 3/20
  requests with max 3.8e-6. Teacher-forced comparison (`bench/e2e_prompt_lp.py`, 12 prompts of 16..512 tokens, every
  collective of the forward on the topo path): 6150 compared logprobs, mean |d| 2.2e-6, median 9.5e-7, p99 8.6e-6, max
  1.3e-5; 30.5 % of the values are bitwise equal. Prompt logprobs of the 3000-token prompt of the greedy test are bitwise
  equal because those all-reduces (> 8 MiB) are NCCL in both runs.

## Benchmark (full model, EP8, dummy weights; `SEED=9000 scripts/locked_run_topo.sh <tag>`, 2026-10-24)

Rows are pp / tg in t/s (pp = 512 / mean TTFT for c1, input tokens / TTFT for the ctx rows; tg = 1000 / mean TPOT in ms),
concurrency 1, random prompts, same method and seeds as `c_muse_ep8` (`musetest/results/<tag>/`).

| row | c_muse_ep8 (2026-10-21) | nccl_h2b_ep8 (same tree, `MUSE_TOPO_ENABLE=0`) | topo_h2_ep8 (H2: small algorithms only) | topo_h2b_ep8 (default, exact) | topo_h2b_bf16x_ep8 (opt-in bf16 exchange) |
|-----|---|---|---|---|---|
| c1 512/128: pp / tg, TTFT, TPOT | 4354 / 51.2, 118 ms, 19.5 ms | 4324 / 51.2, 118 ms, 19.5 ms | 4476 / 85.4, 114 ms, 11.7 ms | 4732 / 85.4, 108 ms, 11.7 ms | 4989 / 85.5, 103 ms, 11.7 ms |
| ctx10 (13,107 tokens): pp / tg | 5342 / 51.6 | 5413 / 51.7 | 5370 / 86.9 | 6059 / 87.2 | 6814 / 88.1 |
| ctx50 (65,536): pp / tg | 5208 / 51.3 | 5237 / 51.4 | 5236 / 85.8 | 5884 / 85.8 | 6473 / 86.2 |
| ctx90 (117,964): pp / tg, TTFT | 5021 / 51.6, 23,495 ms | 5025 / 51.7, 23,473 ms | 5026 / 85.4, 23,469 ms | 5671 / 86.7, 20,801 ms | 6186 / 87.1, 19,069 ms |

Exact path vs `c_muse_ep8`: decode tg +67 .. +69 % (TPOT 19.5 -> 11.5 .. 11.7 ms, from the small algorithms, unchanged by H2b);
prefill pp +8.7 % (c1, 512 tokens), +13.4 % (ctx10), +13.0 % (ctx50), +12.9 % (ctx90). Against the NCCL run of the same tree
the prefill gain is +9.4 / +11.9 / +12.4 / +12.9 %. At ctx90 the prefill takes 2.67 s less (14.4 chunks of 8192 tokens: about
186 ms per chunk, the isolated per-call numbers predicted 194 ms: 62 x (9.20 + 4.60 - 7.10 - 3.58 ms)). With the opt-in bf16
exchange (`MUSE_TOPO_LARGE_BF16_EXCHANGE=1`, class `deterministic_different`; the plugin default is OFF, `env/host.env` turns it ON) the ctx90 prefill is 4.40 s shorter
(about 306 ms per chunk; predicted 312 ms), pp +15 % (c1) and +23 .. +28 % (ctx rows) vs `c_muse_ep8`; decode is unchanged (the decode all-reduces are
below the large threshold). The H2 prefill rows (128 MiB all-reduces on NCCL) moved from +0.1 % to +12.9 %.

Memory is as in H2: workspace 61.1 MiB per GPU (the large algorithms add nothing; the 64-bit flags of the review fix add 64 KiB); at READY GPU 0 holds 91,796 MiB vs 91,786 MiB
for NCCL because the captured graphs shrink (NCCL allocates per-graph buffers, the topo kernels do not), the KV pool grows from
8.21 to 8.48 GiB per GPU (1,563,909 vs 1,513,149 tokens). One "collective issued on a second stream" warning per rank at
start-up (graph capture side stream), as before.

### Real checkpoint (`unsloth/Muse-Spark-Open-816B-A42B-NVFP4`, 497 GB in the HF cache, no `--load-format dummy`)

Same launch, `LOAD_FORMAT=auto CHAT_CHECK=1` (`scripts/run_h2.sh h2b bench_real bench_real_nccl`; weights load in 90 s from the
page cache; 69 GiB per GPU). pp / tg in t/s:

| row | nccl_real_ep8 (`MUSE_TOPO_ENABLE=0`) | topo_h2b_real_ep8 | change |
|-----|---|---|---|
| c1 512/128 | 3666 / 52.0, TTFT 140 ms, TPOT 19.2 ms | 3968 / 88.4, TTFT 129 ms, TPOT 11.3 ms | pp +8.2 %, tg +70 % |
| ctx10 (13,107) | 5342 / 52.2 | 5990 / 88.8 | pp +12.1 %, tg +70 % |
| ctx50 (65,536) | 5168 / 52.7 | 5795 / 89.4 | pp +12.1 %, tg +70 % |
| ctx90 (117,964) | 4955 / 53.0, TTFT 23,806 ms | 5522 / 89.6, TTFT 21,362 ms | pp +11.4 %, tg +69 % |

Real-weight end-to-end determinism of the FULL 62-layer model (`bench/e2e_real.py`, `run_h2.sh h2b real_topo1 real_topo2
real_nccl real_compare`, offline engine, TP8 + EP8, greedy, 51 requests: short prompts, three passage prompts with prompt
logprobs, a batch of 32, the short prompts again): two separate topo processes give identical token ids and bitwise
identical logprobs and prompt logprobs (51/51, max |dlogprob| = 0). topo vs NCCL: 33/51 requests have identical 48-token
greedy continuations, the others diverge at a near-tie (first divergence at token 1 .. 47; `more_accurate` vs the bf16 ring);
teacher-forced NLL of the three passages is the same within noise (NCCL 1.7321 / 1.3184 / 1.1796, topo 1.7241 / 1.3172 /
1.1801 nats per token; ppl 5.652 / 3.738 / 3.253 vs 5.608 / 3.733 / 3.255), and the sampled continuations are fluent and
correct ("Paris. The capital of Germany is Berlin ...", a fibonacci function, passage continuations).

The real-weight gains match the dummy-weight ones (the collectives do not depend on the weights; the absolute prefill speed is
lower with real weights because every token activates real expert mixes). Greedy chat check (`bench/chat_check.py`, five
temperature-0 chat requests including a 4,531-token document question whose prefill uses the 35 MiB / 71 MiB large
all-reduces): 5/5 coherent and correct on both paths (Paris, 391, Rayleigh scattering, a haiku request, "history of the
lighthouse"); the model starts its reply with its native `to=self` analysis channel on NCCL and on topo alike. The two paths
word the analysis differently after a few tokens (the topo all-reduce is the fp32 sum with a single final rounding, NCCL's bf16 ring rounds at every hop,
so near-tied logits flip), which is the expected effect of the `more_accurate` class, not a defect.

## Microbatched prefill (vLLM `VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP`, default ON, `=0` turns it off; track r2e, 2026-10-24)

Prefill of one 8192-token chunk (real weights, plugin, no SP) is 1323 ms of which 735 ms are collectives that run serially
with 584 ms of compute. The branch can split an eager batch of >= 2050 tokens into two microbatches (V2 runner, one thread
per microbatch, attention metadata per microbatch, the second one starts mid-request) that alternate at the TP collectives:
the collective of one microbatch runs on a high-priority comm stream while the other one computes. The flag is ON by
default (accuracy class `deterministic_different`: the halves change M of the cuBLAS GEMMs); decode and CUDA-graph batches are untouched.

When it applies (review round, 2026-10-26): the branch's config hook turns it on only for a launch whose V2 runner can microbatch:
SM120, TP > 1 with expert parallelism, PP = DP = 1, no speculative decoding, LoRA, PIECEWISE CUDA graph mode, context parallelism or
`VLLM_BATCH_INVARIANT`, and no multimodal inputs, i.e. `--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'` (every modality of
the model capped at zero: `{"image":0}` alone leaves audio and video enabled and keeps the flag off). On any other launch the flag
does nothing, and it never changes the model runner the launch gets (before the fix a multimodal launch without the limits fell
back from the V2 to the V1 runner because of it; `config probe` over 12 launches now shows the same runner with the flag on and
off). Around the microbatches the runner (a) sizes the workspace slot of every microbatch with one dummy step of the largest batch
before the CUDA graphs are captured (the workspace is locked afterwards; a launch with `--no-enable-flashinfer-autotune` used to die
on the first long prompt), (b) profiles the split batch as well as the whole one, so the KV cache is sized for it (see the memory
note below), and (c) lets a microbatch run on alone when its sibling issues fewer collectives (the halves of a batch can take
different model paths); the model also raises the split threshold to twice its sequence-parallel minimum, which depends on the CUDA
graph sizes (`--compilation-config '{"max_cudagraph_capture_size":2048}'` with `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL=1`: 4098 tokens
instead of 2050; a 4097-token request used to hang there). With that threshold switched off by a scratch plugin, so that the halves
(2048 / 2049 tokens) really straddle the sequence-parallel minimum, the real model completes 4097 and 4099-token requests in 0.5 s:
the hand-off alone is enough, the threshold only keeps the halves on the path of the whole batch.

What the plugin has to know about it:

* One `TopoComm` serves both streams. A spec idea was a second workspace for the comm stream; it is not needed: every collective
  of an overlapped step goes on the one comm stream in the order the two threads hand off (the same on every rank), the call
  counter is device state, and the hand-off events keep the collectives of consecutive steps (decode graph replays on the
  main stream, overlapped prefill on the comm stream) totally ordered. `TopoComm._order_stream` only sees a stream change
  between such steps (one warning per rank, benign). Two communicators could not add bandwidth either: the socket cut is the
  bottleneck, and the large kernels are comm-bound (860 ms busy of 959 ms per 8192-token step, 66 % of it hidden).
* The result of a collective that runs on the comm stream is allocated on the COMPUTE stream (`_output_stream`, `_alloc` in
  `vllm_plugin.py`; the branch names the stream through `vllm.v1.worker.ubatching.dbo_collective_output_stream`). Allocated
  on the comm stream it lives in a second allocator pool that the compute stream cannot reuse; on a GPU that is nearly full
  (`--gpu-memory-utilization 0.95`) the caching allocator then frees and re-allocates its blocks every step (profile of an
  8192-token step on rank 0: 202 cudaMalloc, 186 cudaFree, 365 cudaEventSynchronize, stalls of 17-29 ms on one rank after the
  other, the others wait in the collective): 1292 ms per step against 1316 without overlap. With the result on the compute stream:
  23 cudaMalloc, 959 ms.
* The large kernels spin-wait on their peers, and now run beside GEMM / attention kernels: the comm stream has the highest
  priority (default priority loses 14 %, -1 and -5 are the same), so that their blocks get an SM as soon as a compute block
  retires. Kernels of the compute stream never wait for a collective of the other microbatch, so a delayed block only delays.

Measured, real weights (SEED 9000 protocol of `scripts/locked_run_topo.sh`, pp t/s ctx10 / ctx50 / ctx90; branch `r2e-prefill-overlap`):

| stream | without (fin_r_topo / fin_r_topo_sp) | with microbatching (`r2e_r_ov2` / `r2e_r_ov_sp`) | gain |
|---|---|---|---|
| replicated (no SP) | 6174 / 5934 / 5651 | 8240 / 7735 / 7155 | +33 / +30 / +27 % |
| sequence parallel (`VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL=1`) | 7709 / 7424 / 6986 | 8803 / 8406 / 7778 | +14 / +13 / +11 % |

c1 512/128 stays 4057 (4072 without; 4452 vs 4470 with SP: below the threshold) and the decode TPOT stays 10.5-10.7 ms: a decode or
CUDA-graph batch is never split.

After the review fixes (branch commit f0e6b1299a, rows `r2e_g_r_ov` / `r2e_g_r_ov_sp`, same protocol, pp t/s c1 / ctx10 / ctx50 / ctx90):
replicated 4027 / 8232 / 7737 / 7151 (before the fixes 4057 / 8240 / 7735 / 7155), SP 4433 / 8823 / 8421 / 7784 (4452 / 8803 / 8406 / 7778),
TPOT 10.5 .. 10.7 ms, the five-request chat check 5/5 on both: the same within +-0.2 % (c1 -0.7 %, inside its noise).

Single-request prefill latency (`bench/prefill_probe.py`, `scripts/prefill_chain.sh`, 1-token completions of random ids, mean of
3 (2 at 64k), one server per row, dummy weights, replicated stream / sequence-parallel stream):

| prompt tokens | 2100 | 4300 | 8192 | 16384 | 65536 |
|---|---|---|---|---|---|
| replicated, off | 0.349 s | 0.687 | 1.296 | 2.565 | 10.595 |
| replicated, microbatched | 0.266 | 0.499 | 0.928 | 1.829 | 7.850 |
| SP, off | 0.281 | 0.558 | 1.029 | 2.044 | 8.619 |
| SP, microbatched | 0.253 | 0.467 | 0.877 | 1.729 | 7.333 |

(`VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_MIN_TOKENS` below the floor of 2050 is raised to it: both halves stay above the
sequence-parallel minimum of the launch, so that they take the same model path as the whole batch.)

Without the plugin (`MUSE_TOPO_ENABLE=0`, plain NCCL, same probe, 2100 / 8192 / 16384 tokens): replicated 0.393 / 1.473 / 2.952 s off,
0.269 / 0.955 / 1.900 s microbatched; SP 0.333 / 1.203 / 2.418 s off, 0.291 / 1.003 / 2.003 s microbatched. The overlap is a branch feature
and works on NCCL (a slower collective gives it more to hide); the plugin starts from the faster collective, so the replicated
gain on NCCL is larger (-35 % against -28 % latency at 8192) while the absolute prefill speed with the plugin stays the best.

Concurrency, real weights, plugin (`work/final/conc` protocol: random 512/128 and 2048/256 prompts, 4 x concurrency prompts, output tok/s;
rows `fin_conc_r_topo`, `fin_conc_r_topo_sp`, `r2e_conc_r_ov`, `r2e_conc_r_ov_sp`); mixed batches (chunked prefill beside decodes)
of at least 2050 tokens are split too:

| prompt/output, concurrency | 1 | 4 | 16 | 64 | 128 |
|---|---|---|---|---|---|
| 512/128 replicated, off | 87.0 | 206.9 | 391.7 | 711.5 | 886.2 |
| 512/128 replicated, microbatched | 86.6 | 208.3 | 415.5 (+6 %) | 794.2 (+12 %) | 1021.6 (+15 %) |
| 512/128 SP, off | 87.1 | 211.5 | 412.1 | 786.5 | 1009.0 |
| 512/128 SP, microbatched | 86.9 | 215.6 | 421.1 (+2 %) | 817.4 (+4 %) | 1059.8 (+5 %) |
| 2048/256 replicated, off | 84.5 | 188.4 | 315.4 | 489.5 | 568.5 |
| 2048/256 replicated, microbatched | 84.5 | 196.5 (+4 %) | 346.0 (+10 %) | 577.9 (+18 %) | 689.9 (+21 %) |
| 2048/256 SP, off | 86.3 | 198.1 | 341.3 | 564.3 | 672.9 |
| 2048/256 SP, microbatched | 86.4 | 202.2 (+2 %) | 356.1 (+4 %) | 602.0 (+7 %) | 725.3 (+8 %) |

Tried and dropped (all dummy weights, 16384 tokens, replicated stream unless noted): block counts of the large kernels x0.5
2.12 s (x1 1.86, x2 1.90; SP x2 1.86 vs 1.79); chunk 2 / 4 MiB per block 1.87 / 1.88 s (1 MiB 1.86); 48 KB of dynamic shared
memory per block (to keep GEMM blocks off the SMs of the collective) 1.87 s, 64 and 96 KB do not launch; comm stream priority 0
2.11 s; three / four microbatches 1.825 / 1.821 s at 16384 and 7.76 s at 65536 (two: 1.829 / 7.85; SP 1.722 / 1.703 and 7.15 against 1.729 and
7.33), at most 2.5 % better beyond 16k tokens and 28 % slower at 2100 tokens (0.341 s against 0.266), so two stay the default; hand-off before the enqueue of the collective (the all2all users' order, one thread
switch earlier) 2.66 s = no gain at all, because the collective then starts only after the CPU has enqueued the sibling's
whole segment.

Accuracy (real weights, `bench/e2e_overlap.py`: 21 prompts of 9..23000 tokens, mixed batches and chunked prefill, greedy,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`; two passes per process, `nccl/compare.py`):

* Run-to-run: with the flag on, pass 1 and pass 2 of one process are bitwise identical (21/21 requests, tokens, logprobs, prompt
  logprobs) on both streams, and two processes with the flag on are bitwise identical on the replicated stream. After the review
  fixes (`gfe2e_*`, tree f0e6b1299a) two separate processes are bitwise identical on BOTH streams (21/21, all logprobs; the
  sequence-parallel pair had not been compared before), and the fixed tree reproduces the reviewed tree bit for bit with the flag on
  and with it off on both streams (21/21, all logprobs): the fixes changed no serving-path arithmetic.
* Against the unsplit batch (`deterministic_different`, rounding level): replicated 16/21 greedy continuations identical, the
  sampled-token logprob differs by a median of 1.1e-4 (p90 0.048, p99 0.31) and the prompt-token logprobs by a median of 4e-6 (p99 0.28;
  mean 0.0031 on tokens with logprob > -1); sequence parallel 16/21 identical, sampled-token median 5e-6 (p99 0.13). The same
  comparison between the two unsplit streams (replicated against SP) gives 17/21. The replicated split deviates about three times
  more than that (mean |dlogprob| 0.021 against 0.007; the signed mean is 7e-6, so no bias): bf16-rounding scale, but not "the same
  kind and size of difference as the sequence-parallel stream", and the replicated stream is the one with the larger change.
  Rank identity is measured now: a passive check (a scratch vLLM plugin that is not shipped) compared a checksum of the final hidden
  states of every model step (the sum of the bf16 values viewed as int16 and an index-weighted sum) over the eight TP ranks with the
  flag on, replicated and SP, during the 21-prompt run (88 steps per process, the microbatched prefill steps among them): 0
  mismatches, and the outputs of those runs are bitwise identical to the runs without the check.
* GSM8K smoke (`/raid/vllm-runtime/work/final/acc` harness, 200 items, two passes A / B, effort high, real weights): replicated 97.5 / 96.5 %
  with the flag against 96.5 / 94.5 % without; sequence parallel 96.0 / 98.0 % with against 96.5 / 98.0 % without. No degradation.

Reproduce: `flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock bash scripts/prefill_chain.sh <configs.txt>` with lines like
`overlap|VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP=1`; `scripts/run_src_topo.sh` rows with the same variable in the environment;
`runsrc_topo.sh bench/e2e_overlap.py out.json` (twice, the flag off and on) and `nccl/compare.py a.json b.json`.

## Limits and notes

* Memory of the microbatched prefill: two microbatches in flight keep the activations of both halves alive, hold a second workspace
  slot (768 MB at 8192 tokens) and allocate their collective results on the compute stream's pool. The branch's memory profile runs
  the batch split as well as whole (a second pass after the whole one, without attention metadata, with a throw-away microbatch
  runner), so the KV cache is sized for the extra footprint instead of taking it from the headroom. Dummy weights, TP8 EP8,
  `--gpu-memory-utilization 0.90`, 8192-token chunks: KV pool 4.15 -> 3.48 GiB per GPU with the flag (replicated stream; 5.13 -> 4.49
  GiB with SP), `nvidia-smi` used after six prefills of 8192 .. 16384 tokens 85421 MiB without and 85377 MiB with it, i.e. the flag
  costs about 0.65 GiB of KV cache per GPU and no headroom. (The version before this profile pass measured +1.4 GiB reserved at
  8192 tokens on a ready server, 90.7 against 89.2 GiB/GPU with real weights at 0.95, and one benign allocator retry in the 2048/256
  concurrency-128 run, because only the whole batch was profiled.)
  Real weights, `--gpu-memory-utilization 0.95` (fixed tree, `r2e_g_r_ov` / `r2e_g_r_ov_sp`): KV pool 8.23 GiB per GPU (1,517,064
  tokens; SP 9.24 GiB, 1,703,396) against 8.90 GiB (1,640,273; SP 9.88) of the tree before the profile pass, and a ready server holds
  88,179 MiB per GPU (SP 89,209) against 88,923 (89,893): the extra footprint of the split is in the budget, no allocator retry
  in the benchmark rows.

* KV sizing and the workspace (review correction): the 61 MiB workspace is allocated in `CudaCommunicator.__init__`, which runs inside
  `init_worker_distributed_environment` (`gpu_worker.py` line 509), BEFORE the worker takes `init_snapshot` (line 534). The
  workspace is therefore part of the baseline and is NOT subtracted from the KV budget: `available KV = gpu_memory_utilization x total
  - non_kv_cache_memory`, and `non_kv_cache_memory` only counts what is allocated after the snapshot. The workspace instead
  shrinks the `1 - gpu_memory_utilization` headroom of the GPU by 61 MiB (as other allocations of the distributed initialization
  do). Irrelevant at 0.95 (4.8 GiB of headroom per GPU), it matters only at a utilization close to 1.0. (An earlier version of
  this document and of the plugin docstring claimed the KV sizing accounts for it.)
* Eligibility is rank-uniform by construction: `supports()` looks at dtype / shape / contiguity / device only (never at the data pointer
  or other rank-local state), and misaligned operands are staged through an aligned copy inside `TopoComm`, so all ranks of a collective
  take the same path (a mix of topo and NCCL ranks would deadlock until the spin timeout). Teardown must be symmetric: a rank whose
  communicator was destroyed falls back to NCCL while the others still issue topo collectives.

* Envelope (H1a + H1b algorithms): all-reduce multiples of 128 B up to 2 GiB (64 B multiples up to `max_bytes` = 8 MiB), all
  bf16 / fp16 / fp32; reduce_scatter dim 0, T % 8 == 0, >= 128 KiB; all-gather shards of any size (16 B multiples above 64 KiB,
  row bytes too for dim -1). `supports()` decides; the plugin only routes. Everything else (other dtypes, non-contiguous,
  sizes that are not 128 B multiples above 8 MiB, reduce_scatter dim != 0, other groups such as the EP group, world size != 8)
  keeps the original NCCL path (`MUSE_TOPO_LOG_OPS=1` shows which and why).
* The large kernels spin-wait on their peers with 13 .. 37 co-resident blocks per launch: a kernel on another stream that can
  occupy every SM of the GPU for a long time would stall them until `MUSE_TOPO_TIMEOUT_S`. vLLM issues the collectives of one
  forward on the main stream, so this does not happen in the benchmarked configuration (no hang in any run).
* The pooler/unpooler gathers are requested with dim 0 by the model (`sp_all_gather`), so `gather_columns` still does
  its own transpose copy after the gather; a model change to request dim -1 would remove it (topo produces that layout
  directly).
* Checkpoint / sleep (`suspend`, `resume`, `checkpoint_prepare`, `checkpoint_restore`) are not handled: the IPC
  workspace is not recreated.
* A kernel that waits longer than `MUSE_TOPO_TIMEOUT_S` for a peer traps and kills that worker's CUDA context. The
  plugin therefore passes 600 s instead of `TopoComm`'s own 10 s default (spec: "~10 s"): under vLLM the ranks can be
  tens of seconds apart at start-up (per-rank JIT / Triton / CuteDSL compiles while the other ranks already wait inside a
  collective), so 10 s would kill workers at random. Callers that construct `TopoComm` directly inside an engine should
  pass a large `timeout_s` too.
* Files outside the plugin that H2 touched: `scripts/deploy_host.sh` (`FROM=` source override), `scripts/locked_job.sh`
  (`LOCK_WAIT_S`, default 43200), and the copies `nccl/{bitwise,e2e_bitwise,compare}.py`, `nccl/graph_r4.xml` of
  `/raid/vllm-runtime/musetest/nccl/`. H2b added: `scripts/run_src_topo.sh` (`LOAD_FORMAT`, `CHAT_CHECK`),
  `scripts/run_h2.sh` (steps `bench_bf16x`, `bench_real`, `bench_real_nccl`, tags derived from the run tag, overwrite guard),
  `bench/chat_check.py`, `bench/e2e_real.py` (memory setting that actually fits next to the weights), the extended
  `bench/test_plugin*.py`, `env/host.env` (comments), and in `muse_topo/`: `_why_not` and the activation log line of
  `vllm_plugin.py`, a comment block in `collectives.py` (no code change, the JIT digest of the extension is unchanged).
* H2b did not change `muse_topo/ops.py`: the JIT hashes every file of `csrc`, so the H1b sources were picked up without
  edits (the first run of the deployed folder built `_muse_topo_C` once under `sm120-dev/vllm_cache/muse_topo/`).

## Review fixes (HOSTCHAIN2 H-fix, 2026-10-24)

An independent review of the host package found one blocker, one major and six minor issues (plus the open items of an earlier
review). Every finding was reproduced or confirmed in the code before it was fixed; the fixes are in the files named below and
each has a regression test that fails on the old code.

| # | finding | verdict | fix | regression test |
|---|---------|---------|-----|-----------------|
| 1 | blocker: 32-bit call counter and flags with a signed compare: at the 2^31-th launch of a communicator (about 28 h of batch-1 decode) the flag kernels (all-gather shards >= 16 KiB, prefill all-reduce / reduce-scatter / all-gather) time out and trap, or read stale data | confirmed: the former kernels trap on all 8 ranks (`CSRC_DIR=csrc_epoch`), the unsigned-difference patch (`csrc_epoch_fix`) reads unsynchronised data on fresh flag columns | `csrc/*`: counter `ctl[0..1]` and every flag word are 64 bit (`ld.acquire.sys.b64` / `st.release.sys.b64`, plain `<` compare), flag region 64 -> 128 KiB, LL8 tag = (e >> 1) % 0xFFFFFFFE + 1, which never takes the pristine all-ones value (`ll8_tag`) | `bench/test_epoch.py` (counter forced to 0x7fffffff .. 2^62 on fresh communicators with a late rank, and moved across 2^31 / 2^32 / 2^33 with mixed protocols and random skew); negative controls: former kernels trap, `csrc_epoch_fix` and the reviewer's 64-bit prototype (old LL8 tag) fail |
| 2 | major: `supports()` accepted a dim-0 all-gather of a narrow odd row (e.g. bf16 [4,3]) that the LL8 kernel rejected (`row % 4`), so the plugin crashed instead of using NCCL | confirmed (5 reviewer shapes raised; 112 of 1485 sweep ops) | `bindings.cpp`: the LL8 check only requires the row to be a multiple of 4 bytes for layout 0 (rank-major output ignores the row) | `tests_h1a.py` supports()-vs-kernel sweep (401 gathers run, none raises or is wrong) |
| 3 | minor: opt-in bf16 exchange rounds fp16 partials above 65504 to inf (NaN output) | confirmed | the opt-in is bf16-only; fp16 and fp32 keep the exact exchange (`collectives.py`, `bindings.cpp`, `large.cu`) | `tests_h1b.py` bx group: fp16 / fp32 == exact path, fp16 +-40000 stays finite and exact |
| 4 | minor: documentation claimed the bf16 exchange is independent of the message size | confirmed (337,648 of 1,040,384 elements differ between a 127-row and a 128-row message) | wording fixed in `collectives.py`, `large.cu`, `docs/h1b_collectives.md`, `env/host.env`, the plugin docstring; the small algorithms stay exact | doc only |
| 5 | minor: "100 % correctly rounded" measured against a reference that double-rounds | confirmed (counterexample 1.0 + 2^-8 + 2^-30: contract gives 1.0, correct 1.0078125) | `bench/common.py` / `bench/hlib.py`: `round_single` (true single rounding of the float64 sum); statements reworded (exact whenever the fp32 partial sums are exact, otherwise within 0.5 ulp + 2^-16 relative); the measured figures did not change (100 %, 0.236 / 0.250 ulp) | `tests_h1a.py` / `tests_h1b.py` accuracy checks use the new reference |
| 6 | minor: the workspace does not count against the KV budget as claimed | confirmed in `gpu_worker.py` (init_worker_distributed_environment at line 509, `init_snapshot` at 534) | wording fixed (plugin docstring, this document) | doc only |
| 7 | minor: `all_gather` silently returned a wrong layout for a middle dim | confirmed | one `_ag_layout` helper for `supports()` and `all_gather`; `ValueError` for other dims | `tests_h1a.py` |
| 8 | minor: eligibility depended on the rank-local data pointer alignment, so ranks could diverge between topo and NCCL (deadlock) | confirmed by reading (no trigger in the model) | `supports()` no longer looks at the pointer; misaligned inputs / outputs are staged through aligned copies inside the ops | `tests_h1a.py` (a different alignment on every rank), `test_plugin.py` (through the vLLM wrappers, none on NCCL) |
| 9 | minor (earlier review): an exception other than `TopoUnavailable` in the topology derivation / `_attach` escaped | confirmed | converted into an all-rank agreement reason (stage 1) and a logged fallback (`_attach`) | `test_plugin_unit.py` |

Not done: applying the bf16 rounding in the small all-reduce algorithms when the opt-in is on (it would make the opt-in
batch invariant across the 2 MiB threshold but costs accuracy in the decode path; documented instead, default stays OFF); a
cross-rank agreement for asymmetric `destroy()` (documented); `Comm.set_epoch` / `get_epoch` exist in the compiled module as
test hooks (forward only, idle only) but `TopoComm` and the plugin never call them.

Validation of the fixed tree (one GPU-lock acquisition per step list, server `/raid/vllm-runtime/host_topo`, SRC=`sm120-dev/src`, all PASS):

* `bench/tests_h1a.py` (full, 100,000-replay soak): 256 checks x 8 ranks, 0 failures (253 before + the supports() sweep, the middle-dim
  `ValueError` and the misaligned-operand checks); `bench/test_h1a.py` (the independent second suite, full): ALL TESTS PASSED;
  `bench/tests_h1b.py` (20,000-replay soak): 284 checks x 8 ranks, 0 failed; `bench/test_epoch.py`: PASS (13 phases, see above;
  the former kernels, the unsigned-difference patch and the reviewer's 64-bit prototype with the old LL8 tag each fail it);
  `bench/test_plugin.py`: 416 checks x 8 ranks (411 + 5 misaligned-view checks through the vLLM wrappers, none on NCCL),
  `bench/test_plugin_unit.py`, `bench/test_topology.py`, `bench/test_timeout.py` and `test_timeout_large.py` (ranks trap instead of
  hanging), `bench/test_init_failure.py`, `bench/check_ctx.py`: PASS. The reviewer's probes re-run on the deployed tree
  (not the prototype): `ag_dim0_odd` 8/8 shapes exact, `ag_small_sweep` 1485 ops: 1022 on topo and exact, 463 NCCL fallbacks,
  0 raising, 0 wrong (before: 910 / 463 / 112 raising), `bx_fp16` 0 NaN, `epoch_hang` at 0x7fffffff / 0xffffffff / 0x1ffffffff,
  `fix_hole` at 0x90000000 / 0x180000000 / 0xffffffff and `epoch_wrap64` all pass on 8 ranks.
* End-to-end determinism (`run_h2.sh fin ... e2e_topo1 e2e_topo2 e2e_nccl compare ops`, 6 layers, dummy weights): topo vs topo in
  two separate processes 20/20 requests identical token ids and bitwise identical logprobs and prompt logprobs (max |dlogprob| = 0),
  in-process repeat 8/8; topo vs NCCL differs as before (first divergence within 10 tokens, max |dlogprob| 1.9e-6, 3000-token
  prompt logprobs max 1.9e-5); teacher-forced comparison (`lp_compare`, 12 prompts): 6150 values, mean |d| 2.2e-6, median 9.5e-7, p99
  8.6e-6, max 1.3e-5 (H2b: the same); per-op summary 188 keys, 188 topo, 0 NCCL, 0 problems.
* Full-model benchmark `scripts/locked_run_topo.sh topo_final_ep8` (SEED=9000, dummy weights, same method as above), with an NCCL
  run of the same tree (`nccl_final_ep8`, c1 row):

| row | nccl_final_ep8 (`MUSE_TOPO_ENABLE=0`) | topo_h2b_ep8 (before the fixes) | topo_final_ep8 (after the fixes) |
|-----|---|---|---|
| c1 512/128: pp / tg, TTFT, TPOT | 4303 / 51.2, 119 ms, 19.5 ms | 4732 / 85.4, 108 ms, 11.7 ms | 4674 / 85.4, 110 ms, 11.7 ms |
| ctx10 (13,107 tokens): pp / tg | - | 6059 / 87.2 | 6097 / 87.3 |
| ctx50 (65,536): pp / tg | - | 5884 / 85.8 | 5903 / 85.8 |
| ctx90 (117,964): pp / tg, TTFT | - | 5671 / 86.7, 20,801 ms | 5682 / 86.6, 20,763 ms |

  Decode TPOT is unchanged (11.7 ms); the prefill rows agree with the pre-fix run within the run-to-run noise (c1 pp -1.2 %, ctx
  rows +0.2 .. +0.6 %). READY memory is unchanged (GPU 0: 91,796 MiB, KV pool 8.48 GiB per GPU = 1,563,909 tokens); the workspace is
  64,094,208 B = 61.13 MiB per GPU (+64 KiB for the 64-bit flags).
* Micro-benchmarks after the change (`run_h1a.sh fix bench_public`, `run_h1b.sh fix bench_ar bench_rs bench_ag`; same protocol as
  `docs/h1a_latency.md` / `docs/h1b_latency.md`, all results bitwise exact): decode all-reduce [T,8192] / [T,4096] T <= 16 is unchanged
  within +0.4 % (median; [1,8192] 5.42 -> 5.43 us, [8,8192] 11.50 -> 11.55 us, NCCL 42.9 / 47.3 us), the LL8 all-gathers of the
  pooler / unpooler shards (T <= 16) are about 0.1 us slower (+2 .. +3 % b2b: [1,512] dim 0 3.59 -> 3.70 us, [1,1024] 3.80 -> 3.91 us;
  behind a 50 us kernel the difference is within noise; the 64-bit counter tag costs a few dependent instructions in front of the
  first store; per decode step 124 such gathers = ~12 us of 11.7 ms), the logits gather is unchanged (13.28 -> 13.29 us). An A/B on
  the same day (former 32-bit kernels, the shipped kernels, and a variant whose LL8 tag is a compare/select instead of the 64-bit
  modulo; 3 interleaved runs each, median, decode shapes T <= 16) attributes most of the all-gather difference to the modulo:
  shipped +0.075 us (+2.0 %) vs the 32-bit kernels, compare/select variant +0.022 us (+0.6 %); all_reduce +0.031 us (+0.4 %) vs
  +0.019 us. The variant is not shipped (0.05 us per gather, ~6 us per decode step, would need another validation cycle); it is
  recorded as a follow-up. The
  large collectives are unchanged within +-0.5 % (all_reduce 16 / 64 / 128 MiB 918 / 3582 / 7097 -> 912 / 3572 / 7121 us,
  reduce_scatter 128 MiB 4022 -> 4033 us, all_gather 128 MiB output 3310 -> 3305 us).
* Real-weight full-model determinism with the fixed tree (`run_h2.sh fin real_topo1 real_topo2 real_nccl real_compare`, full 62-layer
  NVFP4 model, TP8 + EP8, 51 requests as in H2b): two separate topo processes give identical token ids and bitwise identical
  logprobs and prompt logprobs (51/51, max |dlogprob| = 0); the outputs are also bitwise identical to the topo run of H2b
  BEFORE the fixes (`nccl/compare.py real_h2b_topo1.json real_fin_topo1.json`: 51/51, max |dlogprob| 0; the dummy-weight e2e dumps
  agree the same way, 20/20), i.e. the fixes are accuracy class `identical` to the shipped topo path. topo vs NCCL: 33/51 identical
  48-token continuations, NLL of the three passages 1.7241 / 1.3172 / 1.1801 (topo) vs 1.7321 / 1.3184 / 1.1796 (NCCL), the same as H2b.

## Re-run

```bash
scripts/deploy_host.sh                    # DEST=/raid/vllm-runtime/host_topo (default); one mirror, host_topo_h2 is gone
ssh server 'cd /raid/vllm-runtime/host_topo && HOST_DIR=$PWD SRC=/raid/vllm-runtime/sm120-dev/src bash scripts/locked_job.sh h2b 7200 \
  bash scripts/run_h2.sh h2b plugin e2e_topo1 e2e_topo2 e2e_nccl compare ops lp_nccl lp_topo lp_compare bench bench_bf16x bench_nccl'
# real checkpoint (needs /raid/vllm-runtime/work/download/ALL_DONE): steps bench_real bench_real_nccl (LOAD_FORMAT=auto, CHAT_CHECK=1)
```

`run_h2.sh` derives the benchmark tags from the run tag (`topo_<tag>_ep8`, `nccl_<tag>_ep8`, `topo_<tag>_bf16x_ep8`,
`topo_<tag>_real_ep8`, `nccl_real_ep8`) and skips a step whose `musetest/results/<tag>/status.txt` already ends in `DONE`
(`FORCE=1` overwrites). `run_src_topo.sh` accepts `LOAD_FORMAT` (default `dummy`, `auto` = real weights) and `CHAT_CHECK=1`
(`bench/chat_check.py`: greedy chat requests right after READY, replies in `<results>/chat.json`).

Logs and dumps land in `/raid/vllm-runtime/work/host/h2/` and `/raid/vllm-runtime/work/host/logs/`, benchmark results in
`/raid/vllm-runtime/musetest/results/<tag>/` (`musetest/summarize.py`).
