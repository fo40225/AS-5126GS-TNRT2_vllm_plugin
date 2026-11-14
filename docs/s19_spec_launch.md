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
| 33..47 | 264..376 | FULL graph, replicated | eager, replicated (the eager tax: c40 mix greedy -0.3 % noise, t1k64 -1.3 %, gate hold) |
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

## Launch line (the production speculative launch line since the S1-9 lever 1 merge)

The stage C line (`/tmp/muse_wf/s1-8/stageC/RESULTS.md` Part 5.3) plus the two settings and the KV bytes re-derived for this graph set:

    source $HOST_DIR/env/host.env; source $HOST_DIR/env/spec_k7.env     # MIN_TOKENS=384, MUSE_SPEC_SERVE_ARGS, MUSE_SPEC_KV_BYTES=10526351360
    vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 \
      --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 ... --speculative-config '<RadixArk K=7, block, probabilistic>' \
      --per-request-spec-decode-metrics detailed $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES

i.e. `--max-cudagraph-capture-size 256 --kv-cache-memory-bytes 10526351360` (513,982 blocks = 1,482,225 KV tokens per rank, 11.31x "maximum concurrency for
131,072 tokens per request") and the env `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=384`. The bytes are valid without `prompt_logprobs` / `echo` (S1-7 section 4: a
chunked prompt_logprobs request can kill the engine at the gate tier); a deployment that must accept them drops `--kv-cache-memory-bytes` (util-0.95 default pool, not PLL-tested for
this profile). Without this profile (default capture list) the S1-7 value stays 9,676,390,400.

Harness form (as the gates ran): `SERVE_EXTRA="$MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES" SPEC_CONFIG=dflash2_radixark ... serve_sd.sh <tag>`.

The two settings are one property: MIN_TOKENS=384 without the cap changes nothing (floor 513). The cap without the env (floor max(512, 257, 257) = 512, A/B arm B512)
runs 257..511-token batches eagerly and replicated, starts SP only at 512 (c64) and had NVML min free 340 / 624 MiB against 920 / 772 MiB with floor 384 and 400 / 484 on the
default list. The control arm B520 (cap 256, env 520: 257..519 eager and replicated) reached NVML min free 116 / 30 MiB at these KV bytes (no OOM, no margin). The fragment is
deliberately not in `host.env`: host.env is sourced by every launch, and the non-speculative `--max-num-seqs 256 --max-cudagraph-capture-size 256` launch would get floor 384
instead of 512 (its 384..511-token prefill batches would move onto SP, unmeasured).

After every boot `scripts/check_spec_launch.sh <serve.log>` must print OK (it reads the vLLM line "sequence-parallel residual stream for batches of at least 384
tokens (...MIN_TOKENS=384, CUDA graphs up to 256 tokens)" and, next to a harness `serve_cmd.txt`, the capture flag). A gate run whose log fails the check tested the old
configuration. `check_spec_launch.sh <log> 520 256` and `<log> 513 512` check the control arm and production.

## Measured, step 1 (8 boots, two sessions per arm, real prompts, tree 0ffa57b714, host b207e54, KV bytes 9,676,390,400)

Session spread of one arm 0.1-0.5 %. A = production, B384 = this profile, B520 = eager no-SP control.

| row (c48 / c64) | A | B384 | B384 vs A | SP alone (B384 vs B520) | eager tax (B520 vs A) |
|---|---|---|---|---|---|
| real 512-token outputs (r512 greedy), tok/s | 1547 / 1748 | 1603 / 1847 | +3.6 % / +5.7 % | +4.3 % / +6.5 % | -0.6 % / -0.8 % |
| r512 TPOT ms (AL) | 27.1 / 32.1 (4.44 / 4.53) | 26.1 / 30.3 (4.44 / 4.54) | | | |
| 128-token mix greedy, tok/s | 1092 / 1136 | 1097 / 1152 | +0.4 % / +1.4 % | | |
| mix t1k64, tok/s | 1063 / 1111 | 1063 / 1122 | 0.0 % / +1.0 % | | |

The realised saving is about 5 % of the step (6-7 ms at c64), not the 9 % (11.7 ms) the profile predicted. Random-token rows (rnd512) are not a valid comparator: the
acceptance length on random tokens varies run to run (4.10 against 4.32) while TPOT is better (24.8 / 28.8 against 25.4 / 29.4 ms).
Memory: idle 92,817 MiB per GPU (-912 MiB against production), NVML min free 920 / 772 MiB (two boots) against 400 / 484 MiB.
Accuracy class: rounding-level (row-sharded GEMM, reduce-scatter + all-gather instead of all-reduce; the class of the production SP prefill), deterministic.

## Gates, step 4 (one GPU-lock hold 2026-10-31 17:31-20:22; tree = 0ffa57b714 + the test file of muse/s19-t512, host mirror b207e54, this profile sourced from env/spec_k7.env)

0. Tests: SP file 77 passed on 8 GPUs (91 s; includes the new T = 384 / 387 / 512 cases).
1. A/B, six boots in the order A B A B A B (three sessions per arm, KV bytes 9,676,390,400 for both arms, every boot passes `check_spec_launch.sh`: A 513 / 512, B 384 / 256, 0 preemptions,
   0 OOM). out tok/s, mean of three session means, B vs A (**bold** = beyond noise: Welch, 2.78 x se):

   | row | c16 | c32 | c40 | c48 | c64 | c128 |
   |---|---|---|---|---|---|---|
   | real 512-token outputs (r512) | | | | **+3.8 %** (1549 -> 1608) | **+5.7 %** (1746 -> 1845) | |
   | mix greedy (128-token cap) | +0.3 % | +0.3 % | -0.3 % | **+1.0 %** | **+1.3 %** | -0.3 % |
   | mix t1k64 | +1.2 % | **+0.7 %** | **-1.3 %** | **+0.7 %** | **+1.4 %** | +0.3 % |

   Session sd of one arm 0.1-0.5 % on r512 and up to 1.4 % on the c16 mix rows; the five-session pool with the two step-1 sessions per arm gives r512 +3.75 % / +5.65 %, mix greedy c64 +1.32 %, t1k64 c64 +1.22 %, c40 t1k64 -1.29 %.
   c40 is the replicated eager band (c33..c47): the eager tax with no SP gain, inside the 1.5 % adoption limit. TTFT p50 of 300- and 450-token prompts at c1 (random, n = 48 x 2 per session):
   300 tokens 96.2 -> 95.8 ms (eager replicated against the replicated PIECEWISE graph), 450 tokens 122.9 -> 105.0 ms (**-14.6 %**, eager SP prefill against the graph). c128 TTFT p50 3.24 -> 3.31 s (+2 %, noise;
   the single B512 outlier of step 1 did not recur). Non-speculative guard: production ns128 line, c1 / c64 x 3 reps, mix greedy 116.5 / 956.1 and t1k64 116.0 / 951.9 tok/s against the stage C
   sessions 116.5 / 953.8 / 956.9 and 116.5 / 949.1 / 952.2 (+-0.4 %, 0 preemptions): nothing outside the profile moved. Full tables: `/tmp/muse_wf/s1-9/L1/gate/verdict_p1.txt`.
2. Exact KV bytes (S1-7 procedure: probe tree, NVML 10 ms sampler, stress A1 / B / C1 / C2 / D / X, REPS = 2, drafter floor S_live >= 1,256 MiB): at 9,676,390,400 clean, idle free 4,435 MiB
   (S1-7 default list 3,525), exact live growth 2,268.8 MiB (the same as S1-7: the eager transients of the 257+ token batches are below the verify-logits peak), **S_live 2,167 MiB**, 2 allocator retries.
   Stepped up by floor((S_live - 1,256 - 100) MiB / 20,480 B) blocks to **10,526,351,360** (513,982 blocks, 1,482,225 tokens, +8.8 %): clean (0 client errors / mismatches / preemptions / OOM, health 200),
   **S_live 1,356.6 MiB** (floor 1,256), 4 retries, NVML min free 3.4 MiB (the allocator wall is touched, the normal regime of the drafter rows in S1-7: it releases its cache, no OOM). The NVML
   min free of the step-1 sessions (920 / 772 MiB, and 128 / 6 / 96 / 964 / 910 in the gate sessions) is not the gate: it varies by session with the allocator cache.
3. GSM8K 200x2 at c64 on the profile at the adopted bytes (pure tree, no probe; `s19g_gsm64`): A 194/200 (97.0 %), B 194/200 (97.0 %), pooled 388/400 = **97.0 %** (band 95.5-98.0 %), 0 errors,
   5 of 400 answers at the 16384-token cap, same final answer A vs B 192/200 (identical text 0/200, batch-composition rounding as in every earlier run). Boot: idle 93,627 MiB, KV 1,482,225 tokens.
4. Determinism, two cold processes of the profile (`greedy_check_g.py gen`: offline TP8 + EP8 engine, `max_cudagraph_capture_size` 256 + floor 384, 64 prompts = 20 GSM8K few-shot + 44 MT-bench, 128 tokens,
   batch pass = 64 concurrent sequences = verify batches of T = 512 down to 384 (eager SP), serial pass): **STRICT PASS 128/128 rows, 0 differing fields** (token ids, top-5 logprobs, text, finish reason; batch AL 4.553,
   serial 4.501 in both). Production process on the same prompts for the record: serial pass 61/64 identical (the 3 that differ are exactly 3 of the 5 prompts whose prefill is 257..383 tokens, eager replicated
   against a PIECEWISE graph; the 49 prompts of up to 256 and the 10 above 512 tokens are bit-identical), batch pass 2/64 identical (62 diverge by token 128, 8 of them within 0.1 nat): the expected effect of a
   rounding-level change at T = 512 (row-sharded GEMM, reduce-scatter + all-gather) on greedy continuations of reasoning text under NVFP4 / fp8-KV quantization; `greedy_check.py` is a diagnostic, the accuracy gate is
   the GSM8K run above (batch AL 4.553 against 4.594 in production, serial 4.501 against 4.527).

Open follow-up (not part of this profile): c33..c47 stays eager and replicated; `MIN_TOKENS=264` / `320` (floor 264 / 320 with the same cap) would put it on SP too. It equals floor 384 on every concurrency measured above except c40 and was not run.
