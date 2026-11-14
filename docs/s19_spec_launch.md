# Speculative launch profile: graph cap 256 + sequence-parallel floor 384 (S1-9 lever 1)

Launch-scoped, no vLLM code change: `env/spec_k7.env` (sourced after `env/host.env`, speculative launches only) and
`scripts/check_spec_launch.sh` (the boot check). Branch under test: `sm120-muse-spark-open` 0ffa57b714; this folder at main b207e54.

## What it changes

Speculative decoding with RadixArk DFlash2, K = 7, `--max-num-seqs 128` verifies 8 tokens per request, so a batch of c requests is
T = 8c tokens. The vLLM branch shards the residual stream along the sequence (SP: row-sharded pooler / unpooler / router, reduce-scatter +
all-gather instead of all-reduce) only from the floor

    floor = max(VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS, 257, max_cudagraph_capture_size + 1)     SP iff T >= floor

(`sm120_sequence_parallel_min_tokens`, models/muse_spark/nvidia/model.py; the floor keeps SP out of every batch a CUDA graph can serve and
out of the decode-sized sharded projections, `_SHARDED_PROJECTION_MAX_TOKENS` = 256, a module constant without an env). With the default capture
list (graphs up to 512 tokens) the floor is 513 whatever the env says: an env value below 513 is silently raised, it is accepted only together
with a smaller `--max-cudagraph-capture-size`. So the verify batches of 33..64 requests (T = 264..512) run with the pooler / unpooler replicated
on every rank (8x redundant GEMM FLOPs; dense GEMMs 22.6 ms per step at c64 against 10.9 at the T = 256 class in the round-4 profile).

| requests | T | production (default list, env unset) | profile (`--max-cudagraph-capture-size 256`, MIN_TOKENS=384) |
|---|---|---|---|
| 1..32 | 8..256 | FULL graph, replicated, fused decode kernels | the same |
| 33..47 | 264..376 | FULL graph, replicated | eager, replicated (small eager tax, not measured: no arm covered it) |
| 48..64 | 384..512 | FULL graph, replicated | eager, sequence-parallel |
| 65..128 | 520..1024 | eager, sequence-parallel (floor 513) | the same |

The table covers the pure verify batches (T = 8c). Under the cap every batch of 257..512 tokens leaves the PIECEWISE graphs, not only the verify ones:
mixed prefill + decode batches and prefill-only batches too (a c1..c8 request with a 300..500-token prompt used to replay a replicated PIECEWISE graph).
With the profile they run eagerly: replicated for 257..383 tokens, sequence-parallel for 384..512 (a T that is not a multiple of 8 pads the rows,
`ceil(T / 8)` per rank, the last rank carries zero rows that are sliced off after the all-gather; same code path as the production SP prefill at T >= 513,
tested at 257 / 387 / 700 / 1503 and at 384 / 512). The gate rows `ttft_in300` / `ttft_in450` (c1, random 300- and 450-token prompts) measure that
low-concurrency TTFT, and `real_*` c40 measures the replicated eager band (c33..c47, T = 264..376) that no step-1 arm covered.

The graph sets shrink with the cap: target FULL 48 -> 32 and PIECEWISE 51 -> 35, drafter FULL 48 -> 32 (the draft step runs eagerly above 32
requests), graph pool 2.11 -> 1.22 GiB per rank. Non-speculative launches, `envs.py` defaults and `model.py` are not touched.

## Launch line

The stage C production line (`/tmp/muse_wf/s1-8/stageC/RESULTS.md` Part 5.3) plus the two settings:

    source $HOST_DIR/env/host.env; source $HOST_DIR/env/spec_k7.env     # MIN_TOKENS=384, MUSE_SPEC_SERVE_ARGS="--max-cudagraph-capture-size 256"
    vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 \
      --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 ... --speculative-config '<RadixArk K=7, block, probabilistic>' \
      --per-request-spec-decode-metrics detailed $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes 9676390400

Harness form (as the A/B ran, `master_l1.sh`): `SERVE_EXTRA="$MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes 9676390400" SPEC_CONFIG=dflash2_radixark ... serve_sd.sh <tag>`.

The two settings are one property: MIN_TOKENS=384 without the cap changes nothing (floor 513). The cap without the env (floor max(512, 257, 257) = 512, A/B arm B512)
runs 257..511-token batches eagerly and replicated, starts SP only at 512 (c64) and had NVML min free 340 / 624 MiB against 920 / 772 MiB with floor 384 and 400 / 484 on the
default list. The control arm B520 (cap 256, env 520: 257..519 eager and replicated) reached NVML min free 116 / 30 MiB at these KV bytes (no OOM, no margin). The fragment is
deliberately not in `host.env`: host.env is sourced by every launch, and the non-speculative `--max-num-seqs 256 --max-cudagraph-capture-size 256` launch would get floor 384
instead of 512 (its 384..511-token prefill batches would move onto SP, unmeasured).

After every boot `scripts/check_spec_launch.sh <serve.log>` must print OK (it reads the vLLM line "sequence-parallel residual stream for batches of at least 384
tokens (...MIN_TOKENS=384, CUDA graphs up to 256 tokens)" and, next to a harness `serve_cmd.txt`, the capture flag). A gate run whose log fails the check tested the old
configuration. `check_spec_launch.sh <log> 520 256` and `<log> 513 512` check the control arm and production.

## Measured (S1-9 L1 step 1, 8 boots, two sessions per arm, real prompts, tree 0ffa57b714, host b207e54, KV bytes 9,676,390,400)

Session spread of one arm 0.1-0.5 %. A = production, B384 = this profile, B520 = eager no-SP control.

| row (c48 / c64) | A | B384 | B384 vs A | SP alone (B384 vs B520) | eager tax (B520 vs A) |
|---|---|---|---|---|---|
| real 512-token outputs (r512 greedy), tok/s | 1547 / 1748 | 1603 / 1847 | +3.6 % / +5.7 % | +4.3 % / +6.5 % | -0.6 % / -0.8 % |
| r512 TPOT ms (AL) | 27.1 / 32.1 (4.44 / 4.53) | 26.1 / 30.3 (4.44 / 4.54) | | | |
| 128-token mix greedy, tok/s | 1092 / 1136 | 1097 / 1152 | +0.4 % / +1.4 % | | |
| mix t1k64, tok/s | 1063 / 1111 | 1063 / 1122 | 0.0 % / +1.0 % | | |

Guards c16 / c32 / c128 flat within +-1.3 % (t1k64 c16 noise +-2.5 % across arms with identical code); TTFT p50 flat (c128 3079 -> 2871 ms); 0 preemptions in every boot.
The realised saving is about 5 % of the step (6-7 ms at c64), not the 9 % (11.7 ms) the profile predicted. Random-token rows (rnd512) are not a valid comparator: the
acceptance length on random tokens varies run to run (4.10 against 4.32) while TPOT is better (24.8 / 28.8 against 25.4 / 29.4 ms).
Memory: idle 92,817 MiB per GPU (-912 MiB against production), NVML min free 920 / 772 MiB (two boots) against 400 / 484 MiB.
Accuracy class: rounding-level (row-sharded GEMM, reduce-scatter + all-gather instead of all-reduce; the class of the production SP prefill), deterministic.

## Gates owed before this is the production launch (not run in the implementation step)

1. Exact KV bytes (S1-7 procedure, `/tmp/muse_wf/s1-7/exact/`): the graph set changed, so at least a confirm boot at 9,676,390,400 plus the stress; a value about 900 MiB
   higher (about +130k KV tokens) only with its own boot and stress. The 1,256 MiB S_live floor of the drafter rows applies.
2. GSM8K 200x2 with speculation at c64 (`/raid/vllm-runtime/work/final/acc/`, stage C `s18c_g1c128_rx_B0` invocation with concurrency 64) on the profile. T = 384 and 512
   are new M values of the sharded path under speculation; the stage C c128 gate covered SP only at T >= 520. Band 95.5-98.0 %.
3. Determinism, two cold processes, with the batch pass inside the band: `greedy_check.py gen` (20 prompts, `max_num_seqs=128`) gives T = 160 and passes no
   `max_cudagraph_capture_size`, so it does not exercise the change. Needs a copy with a capture-size argument and >= 48 (better 64) prompts (gsm80 / mtbench80 have 80).
4. Every boot: `scripts/check_spec_launch.sh`.

Follow-up guard row, not a blocker: c33..c47 (T = 264..376) is eager and replicated under floor 384 (graphs 257+ are gone); a MIN_TOKENS=264 / 320 arm would
cover it and equals MIN_TOKENS=384 on every measured concurrency.
