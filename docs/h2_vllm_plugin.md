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
vllm serve ...                                   # or scripts/runsrc_topo.sh <python args> for a plain python run
```

`muse_topo-0.1.dist-info/entry_points.txt` registers `muse_topo.vllm_plugin:register` in the `vllm.general_plugins` group
(no pip install). vLLM calls it in every process from `load_general_plugins()`; in the workers this happens in
`WorkerWrapperBase.init_worker`, before `init_device` creates the device communicators. If `VLLM_PLUGINS` is set it must
contain `muse_topo`. `scripts/runsrc_topo.sh` adds the host folder of the script to `PYTHONPATH` and sources `env/host.env` for plain python runs.
`MUSE_TOPO_ENABLE=0` with the same script gives the NCCL-only baseline.

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
algorithms round the island partials to bf16; the all-reduce below 64 KiB and the fused T <= 8 kernels do not, and since r3c the medium all-reduce does unless `MUSE_TOPO_MEDIUM_BX=0`). `env/host.env` turns that last one ON since
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
| `MUSE_TOPO_OPS` | `all_reduce,all_gather,reduce_scatter,boundary` | ops routed to the topo path (A/B switch); `boundary` = the fused collective + epilogue kernels (README section 8) |
| `MUSE_TOPO_TIMEOUT_S` | 600 (plugin) | spin timeout of the kernels, then `__trap()`; `TopoComm` alone defaults to 10 s, which is too short for start-up skew (per-rank JIT/Triton compiles run while the other ranks already wait inside a collective) |
| `MUSE_TOPO_MAX_BYTES` | `TopoComm` default (8 MiB) | largest all-reduce of the decode/medium algorithms (the workspace grows with it); larger all-reduces (up to 2 GiB), reduce_scatter and big all-gather shards use the chunked large algorithms through the same workspace, so there is no reason to raise it |
| `MUSE_TOPO_LARGE` | 1 | 0 disables the large algorithms (A/B switch: all-reduce above 8 MiB, reduce_scatter and all-gather shards above 1 MiB then go to NCCL, as in H2) |
| `MUSE_TOPO_LARGE_BF16_EXCHANGE` | 0 in the plugin (`env/host.env`: 1, the throughput default of 2026-10-24) | class `deterministic_different`: the large all-reduce / reduce_scatter exchange the two island partials rounded to bf16 (bf16 inputs only, half the bytes over the socket cut; batch invariant only inside the bf16-exchange range: every bf16 all-reduce from `ar_bx_min` (64 KiB with `MUSE_TOPO_MEDIUM_BX` on, else the 2 MiB switch) and every reduce_scatter (at least 128 KiB, always the large algorithm with the exchange when the flag is on) give a row the same bits at any size, and reduce_scatter equals all-reduce bitwise from 128 KiB on; smaller all-reduces and the fused T <= 8 kernels stay exact, so with the flag on a row can get different bits depending on the padded batch size across the 64 KiB threshold: in decode the embedding all-reduce at T 3 -> 4 eager / padded 2 -> 4 and the MoE-out all-reduce at T 7 -> 8 eager / padded 4 -> 8; the attn-out at T <= 8 is exact while the fused kernel serves it). See the benchmark below and `docs/h1b_collectives.md` |
| `MUSE_TOPO_MEDIUM_BX` | 1 (only active with `MUSE_TOPO_LARGE_BF16_EXCHANGE=1`) | class `deterministic_different`: the bf16 all-reduces from `MUSE_TOPO_MEDIUM_BX_MIN_BYTES` (64 KiB) up to `max_bytes` take algorithm 2 with the bf16 exchange of the island partials (same bits as the large algorithm); 0 keeps the exact fp32 exchange below `ar_large_min` (2 MiB). With it on the fused kernels are the only exact T = 4 .. 8 attn-out path: `MUSE_TOPO_FUSED=0` makes that reduction take the exchange too, so a bit-identical fused on / off A/B needs `MUSE_TOPO_MEDIUM_BX=0` in both runs (README section 8). The summary line of the plugin prints `medium bf16 exchange from <N> B` or `off` |
| `MUSE_TOPO_AR_TUNING_FILE` | unset | experiment hook: JSON with `bx_table`, `bx_min`, `large_min_bx` overriding the built-in medium all-reduce table and thresholds (written by an offline table generator that is not part of this folder); must be the same on all ranks (an unreadable file or a bad `MUSE_TOPO_MEDIUM_BX_MIN_BYTES` on some rank makes every rank fall back through the stage-1 agreement) |
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
| embedding / attn-out AR `[T,8192]`, MoE-out AR `[T,4096]`, T < 256 | up to 2 MiB (all decode graph sizes 1..128) | topo, H1a 3-phase sentinel algorithm (with the medium bf16 exchange on, from 64 KiB the same algorithm with rounded island partials, up to 8 MiB) |
| the same ARs from 2 MiB (`[256,8192]` / `[512,4096]`) up to `[16384,8192]` (prefill chunks of 256 .. 16k tokens, 2 .. 256 MiB) | 2 MiB .. 256 MiB | topo, H1b large all-reduce (fused reduce-scatter + all-gather pipeline); with the medium bf16 exchange on only above 8 MiB |
| pooler AG `[T,512]`, unpooler AG `[T,1024]` (dim 0) | T = 1 .. 256 (the model uses replicated pooler / unpooler projections for larger T, no gather) | topo (LL8 up to 16 KiB shards, flag protocol above; the large engine takes shards >= 256 KiB, `bench/test_plugin.py` covers `[8192,512]` / `[4096,1024]` shards) |
| logits AG `[T,25256]` (dim -1) | T = 1 .. 1024 (the 1024-row case is the prompt-logprob run) | topo (H1a flag kernels up to a 1 MiB shard, the large engine above: the NCCL fallback of H2 for T > 20 is gone) |
| reduce_scatter (sequence-parallel helpers `sp_reduce_scatter`, none in the benchmarked configuration) | dim 0, T % 8 == 0, >= 128 KiB | topo large reduce-scatter |

No rank ever disagreed on a path (`summarize_ops.py`: 188 keys, 188 topo, 0 NCCL, 0 problems). Sequence parallelism
(`is_sequence_parallel_enabled`) needs `moe_backend == "deep_gemm_mega_moe"`, which is off on this box, so the e2e runs do
not issue a reduce_scatter; the wrapper is covered by `bench/test_plugin.py` (large reduce-scatter, RS + AG == AR bitwise,
inside a CUDA graph). The sequence-parallel helpers ask `custom_all_gather` / `custom_reduce_scatter` of the device
communicator first: vLLM's CustomAllreduce is disabled here (the serve logs of all runs show "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs" on all 8 ranks), those return `None` and the
helpers continue into `tensor_model_parallel_all_gather` / `_reduce_scatter`, i.e. into the patched methods.

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
  Real weights, `--gpu-memory-utilization 0.95` (fixed tree): KV pool 8.23 GiB per GPU (1,517,064
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
