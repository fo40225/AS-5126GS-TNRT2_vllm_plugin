# Speculative launch profile: graph cap 256 + sequence-parallel floor 384 (S1-9 lever 1)

Launch-scoped, no vLLM code change: `env/spec_k7.env` (sourced after `env/host.env`, speculative launches only) and
`scripts/check_spec_launch.sh` (the boot check). Measured on `sm120-muse-spark-open` 0ffa57b714 + host b207e54; production pair after the merge: vLLM
5b827b215a (0ffa57b714 plus the SP test file, the runtime code is identical) + this folder.

**S1-9 lever 2 (the microbatched prefill beside the drafter, "r2e") is part of this profile since 2026-11-01**: `env/spec_k7.env` also sets
`VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE=1` and the KV bytes were **10,373,877,760** (1,460,755 tokens per rank). See the section "S1-9 lever 2" at the end of this file.

**S1-9 L4 (the DFlash2 aux projection on each rank's rows, `VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD=2`, the vLLM default) moves the bytes to 11,134,914,560** (543,697 blocks = 1,567,918 tokens per rank, +7.3 %):
`env/spec_k7.env` exports the flag and sets `MUSE_SPEC_KV_BYTES` from it (11,134,914,560 with the flag at 2, 10,373,877,760 otherwise). See the section "S1-9 L4" at the end of this file.

**S1-9 L5 (2026-11-02) adds `--max-num-batched-tokens 8199` to the profile** (`MUSE_SPEC_SERVE_ARGS`, aligned prefill chunks; the KV bytes stay 11,134,914,560). The SP floor stays 384 (floors 264 / 320 lost the A/B) and the router aux-stream overlap in the SP stream was not adopted. See the section "S1-9 L5" at the end of this file.

**S1-9 L7 (2026-11-04) puts the DFlash2 draft's weights in FP8 (`VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2`, kernel cutlass, the vLLM default) and moves the bytes to 11,573,411,840** (1,629,269 KV tokens per rank, +61,731 = +3.94 %):
`env/spec_k7.env` exports the flag and the kernel and derives `MUSE_SPEC_KV_BYTES` from (aux projection, FP8 flag); `scripts/check_spec_launch.sh` reads the FP8 state from the boot log and keys the bytes on it. See the section "S1-9 L7" at the end of this file.

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

    source $HOST_DIR/env/host.env; source $HOST_DIR/env/spec_k7.env     # MIN_TOKENS=384, r2e opt-in, FC_SHARD=2, MUSE_SPEC_SERVE_ARGS, MUSE_SPEC_KV_BYTES=11134914560 (10373877760 before S1-9 L4)
    vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 \
      --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 ... --speculative-config '<RadixArk K=7, block, probabilistic>' \
      --per-request-spec-decode-metrics detailed $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES

i.e. `--max-cudagraph-capture-size 256 --kv-cache-memory-bytes 11134914560` since S1-9 L4 (543,697 blocks = 1,567,918 KV tokens per rank, 11.96x "maximum concurrency for 131,072 tokens per request";
10373877760 = 506,537 blocks = 1,460,755 tokens, 11.14x, before L4 and for any launch whose log lacks "the aux projection runs in the target's forward") and the env `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=384` + `VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE=1` (lever 2; with the
lever-1-only profile the bytes were 10,526,351,360 = 513,982 blocks = 1,482,225 tokens, the value of every launch that does NOT set the opt-in). The bytes are valid without `prompt_logprobs` / `echo` (S1-7 section 4: a
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
   serial 4.501 in both). Production process on the same prompts for the record: serial pass 61/64 identical (the 3 that differ are exactly 3 of the 5 prompts whose prefill is 257..383 tokens; most likely the graph dispatcher of
   production pads such a prefill to the next capture size (280 -> 288, 345 -> 352, 16-token steps above 256) while the profile runs the exact shape eagerly, so the GEMM M differs;
   the 49 prompts of up to 256 and the 10 above 512 tokens are bit-identical), batch pass 2/64 identical (62 diverge by token 128, 8 of them within 0.1 nat): the expected effect of a
   rounding-level change at T = 512 (row-sharded GEMM, reduce-scatter + all-gather) on greedy continuations of reasoning text under NVFP4 / fp8-KV quantization; `greedy_check.py` is a diagnostic, the accuracy gate is
   the GSM8K run above (batch AL 4.553 against 4.594 in production, serial 4.501 against 4.527).

Open follow-up (not part of this profile): c33..c47 stays eager and replicated; `MIN_TOKENS=264` / `320` (floor 264 / 320 with the same cap) would put it on SP too. It equals floor 384 on every concurrency measured above except c40 and was not run.

## S1-9 lever 2: the microbatched prefill (r2e) beside the drafter (opt-in, part of the production profile)

Merged: vLLM `sm120-muse-spark-open` fast-forwarded from 5b827b215a to the head of `muse/s19-r2e` (10e09221d3; the gates ran on the identical tree of 7555260c06, a commit-message amend apart); this folder as of the merge.

**What.** The prefill overlap of S1-6 (`VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP`, default on) splits every eager batch of at least 2050 tokens into two microbatches whose tensor-parallel
collectives overlap each other's compute. It was switched off under speculative decoding. The branch now runs it beside the DFlash drafter (method `dflash`): the drafter builds its own attention
metadata and takes the merged auxiliary hidden states (5 taps, merged tap by tap in row order), a verify batch (8 tokens per request) is never split off its step, and CUDA graphs are untouched
(decode and verify batches are never microbatched: the decode threshold is unreachable). Effective threshold under the profile: 2050 tokens (2 x (max(SP floor 384, 1024) + 1); the capture ceiling 256 does not raise it), so prefill-only chunks
and mixed chunk + verify steps of at least 2050 tokens are split; below that nothing changes (TTFT of 2048-token prompts +0.1 %, flat).

**Why it is an opt-in under speculation.** The overlap adds a persistent footprint (a second set of FlashInfer metadata builders and the activations of two halves, 145 MiB per rank here) and so
moves the exact `--kv-cache-memory-bytes` of a launch line: the S1-7 bytes of every other speculative row (nvidia DFlash2, DSpark, the mns 64 rows, the default capture list 9,676,390,400) were derived with
it off and would be left under the S1-7 slack by a silent switch-on. The accuracy policy would make it default-on (rounding-level, deterministic, faster); the memory coupling is the reason it is not: the flag and the
bytes are ONE property of a launch line and travel together in `env/spec_k7.env`. `check_spec_launch.sh` fails a boot whose log shows the overlap (line "prefill batches of at least N tokens run as 2 microbatches")
with bytes above 10,373,877,760, or without it with bytes above 10,526,351,360 (`SPEC_EXPECT_R2E=on|off` demands the state). DSpark (no memory / speed run beside microbatches) and adaptive verification stay excluded.

**Memory (exact bytes, S1-7 procedure on the probe tree, REPS = 2, stress A1 B C1 C2 D H I X).** Phases H and I are new: H = 128 decoders in verify plus requests admitted on a prefix-cache hit (a cached 8192-token
prompt, 1k..6k tokens shared, no salt) with a 2k..7k-token remainder, I = split / unsplit interleave in one process (two prefix-sharing prompts admitted together run whole, a lone fresh 8k prompt runs split), 7,104 and 10,128
prefix-hit tokens per repetition (spec), 0 client errors. At **10,373,877,760** (506,537 blocks = 1,460,755 KV tokens, **-21,470 tokens = -1.45 %** against 10,526,351,360 / 1,482,225): clean, 0 errors / mismatches / preemptions / OOM,
idle free 3,623.4 MiB, exact live growth 2,269.0 MiB (the verify-step transient sets the global peak, identical with r2e off), **S_live 1,354.4 MiB** (floor 1,256), 7 allocator retries, NVML min free serving 169 MiB.
The reclaim that made this possible (vs the first version: -4.6 % tokens at the default pool, S_live 35 at the production bytes): one workspace slot for both microbatches (their scratch is never live across a hand-off, -768 MiB),
and the output merge frees the microbatch pieces tap by tap (the merged 114 MiB tensors had pinned the 790 MiB verify-logits segments: fragmentation, not live bytes). The default pool (util 0.95) loses 0.17 % (1,376,421 vs 1,378,783 tokens).
The 10,526,351,360 of lever 1 with r2e on left S_live 1,210 MiB (< 1,256): do not combine them.

**A/B (one lock hold, boots OFF ON OFF ON OFF ON, 3 sessions per arm, same code tree, OFF = lever-1 profile at 10,526,351,360, ON = this profile at 10,373,877,760, every boot passes cfgcheck and the boot check, 0 preemptions).**
Primary rows use natural-text 8192-token prompts (windows of the E2E corpus, the first token id differs per request so no prefix hit exists), 512 output tokens, closed loop, ignore_eos, greedy; random-token prompts are not used
(their acceptance length is chaotic). out tok/s, mean of the three session means (sd), ON vs OFF, **bold** = beyond noise (Welch, 2.78 x se):

| row | OFF | ON | ON vs OFF | TTFT p50 OFF -> ON | TPOT OFF -> ON |
|---|---|---|---|---|---|
| 8k prompt / 512 out, c1 | 153.3 (1.24) | 155.4 (0.07) | **+1.3 %** | 894 -> 784 ms (**-12.3 %**) | 4.79 -> 4.91 ms (AL 3.742 -> 3.654 at c1) |
| 8k prompt / 512 out, c16 | 372.0 (2.76) | 400.8 (0.88) | **+7.7 %** | 1,563 -> 1,069 ms (**-31.6 %**) | 35.77 -> 33.31 ms (**-6.9 %**) |
| 8k prompt / 512 out, c64 | 470.8 (1.04) | 522.2 (0.20) | **+10.9 %** | 3,142 -> 2,873 ms (**-8.6 %**) | 109.97 -> 99.18 ms (**-9.8 %**) |
| mixed churn c64 (288 requests, cold prefill, staggered) | 1,446.0 | 1,439.6 | -0.4 % (noise) | | |
| real 512-token outputs (r512) c64 | 1,840.1 | 1,839.6 | -0.0 % | | |
| mix greedy c16 / c64 | 753.4 / 1,152.2 | 747.0 / 1,153.2 | -0.9 % (noise) / +0.1 % | | |
| mix t1k64 c16 / c64 | 733.6 / 1,117.0 | 737.9 / 1,117.1 | +0.6 % / +0.0 % | | |

The gain is smaller than the profile-based estimate (+13 % / +16 % at c16 / c64) because natural text accepts 3.6 tokens per step and decode takes the larger share of the closed loop; it comes from the prefill chunk
(forward time of an 8185-token chunk -14.6 % in the round-4 style profile; 8185-token TTFT 0.903 -> 0.783 s, -13.2 %; 16k / 32k -13.2 %) and, in a loaded server, from shorter decode stalls (TPOT -7 % / -10 %).
Prompts below 2050 tokens are unaffected (guard rows flat). Acceptance length is unchanged at c16 / c64 (3.617 -> 3.614, 3.621 -> 3.656) and in every guard row (the aux taps behind a microbatched prefill are intact);
the c1 acceptance of the 12 distinct prompts is -2.3 % (3.742 -> 3.654, the earlier c1 checks on 32 + 8 natural prompts gave -2.8 / +2.2 %: prompt-specific rounding noise).

**Prefix-sharing traffic.** The first version counted every verify row (8 tokens) as a prefill "writer" in the guard that keeps a prefix-cache reader from being split off its writer, so any step that admitted a request on a prefix hit
beside a single verify row ran whole (the gain lost on exactly the mixed steps it is for, for traffic with a shared system prompt / RAG context). Fixed: a row is a prefill only if its tokens minus its draft tokens exceed one; the split point is kept
beyond all decode-like rows (`max_num_seqs * (1 + K)` tokens, the threshold floor: 2050 here). The stress phases H and I above exercise it; the A/B itself used no prefix hits (natural prompts with distinct first tokens, cold-salted churn).

**Accuracy gates** (ON profile at the adopted bytes). GSM8K 200x2 at c64 (pure tree, `s19l2g_gsm64`): A 193/200 + B 193/200 = **386/400 = 96.5 %** (band 95.5-98.0 %), 0 errors, 4 + 4 answers at the 16384-token cap, same final answer A vs B 189/200.
Determinism (offline TP8 + EP8 engine, profile env + cap 256, default pool, 72 prompts = the 64 of lever 1 + 8 natural prompts of 4,458 .. 20,158 tokens, 128 tokens, batch and serial passes):
two cold processes of the ON profile **STRICT PASS 144/144 rows, 0 differing fields** (token ids, top-5 logprobs, text; AL 4.354 batch / 4.331 serial in both); diagnostic OFF vs ON: serial pass 65/72 identical (the 7 that differ are 7 of the 8
prompts of at least 4.3k tokens, where the microbatched prefill engages; the 64 short prompts and the 8,158-token prompt are bit-identical), batch pass 2/72 identical (batch-composition rounding, as in lever 1: median first divergence at token 26).
Accuracy class: rounding-level (two half batches change GEMM tilings and the attention split), run-to-run deterministic, identical across ranks.

**Non-speculative production (the shared workspace slot changes it too).** Candidate tree vs production tree (same prompts, non-speculative, r2e on in both): two cold candidate processes STRICT PASS 144/144 and the production
tree's outputs are IDENTICAL to the candidate's (144/144 rows): the shared slot does not change a bit. Exact bytes of the candidate tree, S1-7 stress with H and I, REPS = 2: ns128 at 16,688,807,936 clean, S_live **1,238.1 MiB** (S1-7: 696; the shared slot frees 768 MiB of
persistent memory; floor 500 MiB), 0 retries; ns256 + `--max-cudagraph-capture-size 256` at 16,363,749,376 clean, S_live **1,165.9 MiB** (S1-7: 512), 2 retries (the stress phases H and I hit the prefix cache as in the speculative run). Guard A/B (production tree vs candidate tree, ns128, 2 sessions per arm): mix greedy c1 117.4 vs 117.1, c64 954.0 vs 954.9; mix t1k64 c1 116.8 vs 116.5, c64 950.6 vs 951.2; natural 8k prompts c16 364.5 vs 364.7 tok/s (all within -0.25 .. +0.10 %, noise; 0 preemptions;
both trees log the 2050-token overlap line). The non-speculative exact bytes of S1-7 stay valid and now carry about 540 MiB more headroom (S_live 1,238 / 1,166 against 696 / 512 for ns128 / ns256); a re-derived larger value is a follow-up, not part of this change.

**1M x 1.** Boot-only probe of this profile with `--max-model-len 1048576` (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, RadixArk DFlash2 drafter, r2e on, bytes 10,373,877,760): the log reads "GPU KV cache size: 1,932,643 tokens, Maximum concurrency
for 1,048,576 tokens per request: **1.84x**", so a 1M-token request still fits at fp8 KV with the drafter (SG2 needs 1M x 1). It is a boot and KV-sizing measurement only: no 1M-token request was served and the S1-7 stress was not run at that
max-model-len. The 1,460,755 tokens of the 131072 profile are not the 1M budget (the hybrid allocator bounds the sliding-window groups, not the global layers). The cost of the overlap, 1.45 % of the KV tokens, is paid by every workload,
also those without a prompt of 2050 tokens or more (none of whose rows move).

**Notes.** (1) The F.pad staging copies of `sp_reduce_scatter` (T % 8 != 0) are not removed work: the "16.4 -> 5.5 ms" in the profile is overlap-aware attribution; the copy bytes (about 25 GB read + write per 8185-token chunk, 16.2 ms at the measured
1.537 TB/s) remain and contend with the overlapped GEMMs. Aligning the chunk (`--max-num-batched-tokens 8199` under K = 7, 8192-row chunks, no tail step) measured a further -2.7 .. -3.9 % at 16k / 32k and -20 % at exactly 8192 tokens, one session each; it is a separate decision
that needs its own S1-7 derivation. (2) The drafter's attention groups build one metadata builder set instead of two (no visible change of the idle footprint: 93,629 against 93,627 MiB). (3) The microbatches share the experts' scratch because, under expert parallelism, no collective reads scratch rows and the prepare/finalize does not yield inside the expert op; both are asserted.
(4) Launch lines of the other speculative rows are unchanged: do not source `env/spec_k7.env` for them.

## S1-9 L4: the DFlash2 aux projection on each rank's rows (VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD=2)

Status: the A/B passed every performance criterion of the pre-registered rule and the pairing below is implemented, but the rule's per-cell acceptance-length guard failed on two default-sampling cells
(a zero-variance replay artifact, see "Gate"); the merge of the vLLM branch `muse/s19-draftshard` (head 0b6bf921fe) and of this folder's branch `s19-l4` waits for that decision. Until then the production launch stays at 10,373,877,760.

**What.** The RadixArk DFlash2 draft's aux fc ([5 x 8192 -> 8192], 671 MB, replicated) ran in the propose step on all rows of the verify batch or prefill chunk, after the target had all-gathered its 5 aux taps under the
sequence-parallel stream (5 gathers of 3.3 ms per 8185-token chunk, the fc 19.6 ms). With `FC_SHARD=2` the target's forward runs it where the taps are made: each rank projects its own T / 8 rows with the full weight and ONE
[T / 8, 8192] gather replaces the 5 tap gathers; below the SP floor (decode, inside the FULL graph) every rank projects all rows with its 1/8 slice of the output rows of the weight and gathers. Draft numerics only (another GEMM tiling,
identical on all ranks); the rejection sampler keeps the output distribution. `FC_SHARD=1` (column-sharded weight, -563 MiB) is an opt-in memory alternative, unmeasured in the A/B. The kernel-projection shard and the early seq-lens fetch of the
implementer's profile sessions are not in the tree (no measured gain).

**Bytes (S1-7 procedure, probe tree, REPS = 2, stress A1 B C1 C2 D H I L X with the eligible mix).** At 10,373,877,760: clean, idle free 3,893.4 MiB (3,623.4 before: +270), live growth 1,813.2 MiB (2,269.0 before: the prefill transient
shrinks, 456 MiB), **S_live 2,080.2 MiB**. Rule: step up by 725.8 MiB to **11,134,914,560** (543,697 blocks, 1,567,918 tokens per rank, +107,163 = +7.3 %): clean, **S_live 1,354.2 MiB** (floor 1,256), 0 errors / mismatches / preemptions / OOM,
1 retry. The flag and the bytes are ONE property: with the flag at 0 (or an older tree, or a fallback) the larger value leaves S_live of ~630 MiB. `env/spec_k7.env` derives the bytes from the flag and `scripts/check_spec_launch.sh` fails a boot
whose log lacks the success line "the aux projection runs in the target's forward (...DRAFT_FC_SHARD=2)" while serve_cmd.txt carries more than 10,373,877,760 (a launch that sets the flag but falls back logs "DRAFT_FC_SHARD=2 does not apply").

**Gate (one lock hold, boots OFF ON CTL OFF ON OFF ON; OFF = flag 0 at 10,373,877,760, ON = default 2 at 11,134,914,560, CTL = the production tree, same job; all boots cfgcheck OK, 0 preemptions).**
out tok/s, ON vs OFF (3 sessions per arm, Welch 2.78 x se): real-prompt greedy c64 / c128 +3.20 % / +3.81 %, default sampling c64 / c128 +3.32 % / +3.82 % (pooled primary score **+3.54 %**), c1 +1.75 % (greedy) / +0.09 % (default),
c16 +1.74 % / +3.68 %, c32 +1.30 % / +3.13 %, greedy c40 (the eager non-SP band) +2.57 %, t1k64 c16 / c64 +2.36 % / +2.21 %. Natural 8192-token prompts (prefill side): c16 +4.08 %, c64 +3.58 %, c1 +2.89 % (pooled c16 + c64 **+3.83 %**), TTFT p50 of one
8192 / 16384 / 32768-token request 787 -> 755 ms (**-4.0 %**), 1,556 -> 1,491 (-4.2 %), 3,154 -> 3,025 (-4.1 %), TPOT at c16 -3.5 %. The control: OFF vs the production tree -0.03 % (and bit-identical AL at c1): the flags-off code is production.
Acceptance length: draft-weighted mean ON - OFF over 16 cells **+0.000** (z 0.0). The two cells that failed the pre-registered per-cell AL guard ("<= -0.05 AND beyond noise", Welch with se floored at 0.15 %) are default sampling c1 (-0.128) and c32 (-0.055):
the three sessions of an arm replay one realization (unseeded requests take the same np.random seed stream in every boot: at c1 every OFF and CTL session has AL 4.7791 / 4.6930 for the two reps and every ON session 4.5014 / 4.7154, sd 0), so the test
compares two single draws of 24 / 128 requests; with a sampling error from the accept counts they are z = -1.29 and -1.23, and no cell is at z <= -2.9 (post hoc, `AL_ANALYSIS.txt` in the L4 scratch).

**Accuracy / determinism.** GSM8K 200x2 at c64 with speculation: ON greedy 191 + 198 = 389/400 = 97.25 % (band 95.5-98.0), ON default sampling 197 + 197 = 394/400 = 98.5 %; OFF greedy 389/400 = 97.25 %, OFF default 392/400 = 98.0 %; 0 errors.
Determinism (offline TP8 + EP8, production profile, 85 prompts, 128 tokens): two cold ON processes identical 85/85 for greedy and for seeded t1k64 in both passes, pass 1 vs pass 2 identical 85/85 in every process; OFF vs ON is diagnostic only
(draft numerics differ at rounding level): greedy 35/85 identical, seeded 20/85.

**Non-speculative guard** (production tree vs the L4 tree, ns128 at 16,688,807,936, 2 sessions per arm, no speculator so the aux projection is inert): mix greedy c1 / c64 -0.71 % / +0.14 %, t1k64 c1 / c64 -0.46 % / +0.15 %,
natural 8192-token c16 -0.05 %, all within noise, 0 preemptions.
**Tests** (tree 0b6bf921fe, the gated tree): SP 8-GPU file 77, r2e 8-GPU file 61, the DFlash2 shard file 26 with the host plugin and 26 with NCCL (it includes the real `MuseSpark1xModel.forward` with the combiner under SP batches of 512 / 1024 / 8185 tokens and
r2e microbatch splits, checked row by row against the projection of the taps), Muse suite 3645 passed / 198 skipped, tests/v1/spec_decode failing set identical to production (100).

## S1-9 L5 bundle: aligned prefill chunks adopted; SP floor 264 / 320, SP router overlap and level-4 verification not adopted

Production under test: vLLM `sm120-muse-spark-open` 1eb639f807 + this folder ebd579e2aa, speculative profile at 11,134,914,560 bytes. Two GPU-lock holds (2026-11-02 16:45-22:02 and 22:03-23:43), scripts and the frozen
rule in the L5 scratch (`ADOPTION_RULE.txt`, md5 fdf3857774614f9715fbecadbe620a6e before the first gate boot).

**Design of the A/B.** A 2^3 full factorial over B1 (VLLM_MUSE_SPARK_SM120_SP_ROUTER_OVERLAP_MAX_ROWS=128), B2 (SEQUENCE_PARALLEL_MIN_TOKENS 264 against 384) and B3 (`--max-num-batched-tokens` 8199 against 8192): eight boots, each factor
compared ON against OFF over 4 + 4 boots (the other factors average out; Welch over the boots, t 2.45, se floored at 0.15 % of the OFF mean), plus one production-tree control boot (CTL) and one informational level-4
boot. All 10 boots finished, 0 preemptions, arm checks OK (B1 line "the routing chain of the sequence-parallel stream runs on the aux stream ... for up to 128 rows per rank", the SP floor line, the scheduler line
"Chunked prefill is enabled with max_num_batched_tokens=N"). Real mix320 prompts, closed loop, 128 out tokens; TTFT = median of 5 single random-id requests per size. Control: the all-off corner against the production
tree -0.08 % (greedy c48 / c64 / c128, default c64 / c128; one boot each).

| item | primary cells (ON / OFF) | result | verdict |
|---|---|---|---|
| B3 aligned chunks, 8199 | median TTFT 16,384 / 32,768 tokens | 1.5247 -> 1.4707 s (**-3.54 %**), 3.0956 -> 3.0122 s (**-2.69 %**), pooled score +3.22 % (t 23); 8,192 tokens 0.7646 -> 0.7317 s (-4.31 %), 8,185 flat | **adopted** (threshold +1.0 % beyond noise) |
| B2 SP floor 264 | greedy c36 / c40 / c44 | -1.64 % (SIG) / -0.73 % (SIG) / +0.59 %; pooled -0.59 %; floor 320 (c40 / c44): -0.73 % / +0.59 %, pooled -0.07 % | not adopted: eager SP at T 288 .. 320 is slower than the replicated eager batch |
| B1 SP router overlap, 128 rows | greedy c48 / c64 / c128 | -0.66 % / -0.01 % / +0.18 %, pooled -0.16 % (noise) | not adopted (predicted +0.45 .. 0.76 %: the real all-gather takes the SMs the microbench left idle) |
| B5 level-4 verification (informational, 1 boot vs CTL) | toppall c64 / c128 | +2.6 % / +4.6 % tok/s | not adoptable: its kept-set audit found a ~1e-4 per-row mismatch rate (rule criterion (b) is zero) |

B3 guards (all passed): natural 8192-token prompts c1 +2.4 %, c16 -0.9 % (SIG, inside the -1.5 % bar), c64 -0.5 %; mixed churn c64 +0.26 %; greedy c1 / c16 / c36 / c40 / c44 / c48 / c64 / c128
-0.2 / -0.7 / -0.2 / +0.1 / +0.1 / -0.2 / +0.4 / +0.0 %, default c1 / c16 / c40 / c64 / c128 -0.2 / +0.3 / -0.2 / -0.6 / +0.4 % (all noise); acceptance length: draft-weighted mean ON - OFF +0.0000 over 13 cells, no trip.
The cfgcheck of the B3 boots flags the KV token count (1,567,538 tokens per rank at the same bytes against 1,567,918 without the flag: the sliding-window groups reserve per scheduled token); the verdict waives exactly that flag.

**Bytes.** S1-7 stress (probe tree of the production tree, REPS = 2, phases A1 B C1 C2 D H I L M X, the eligible mix) with the flag at 11,134,914,560: clean, **S_live 1,278.4 MiB** (floor 1,256; production without the flag 1,356.7) ->
unchanged bytes, 1,567,538 KV tokens per rank. `check_spec_launch.sh` now checks the chunk (log line and serve_cmd.txt; `SPEC_EXPECT_MNBT`).
**Accuracy / determinism** (production tree + the flag, bytes 11,134,914,560; rounding-level change: the chunk boundaries of long prefills move). GSM8K 200x2 at c64 with speculation, forward + reverse pass, 0 request errors:
greedy 192 + 196 = **388/400 = 97.00 %** (band 95.5-98.0 %; production boot of the same hold 195 + 193 = 388), default sampling 197 + 196 = 393/400 = 98.25 % (production 394), top_p 0.95 on every request 394 (production 393),
top_p 0.95 on every 4th item 393 (production 393); 6 + 2 answers hit the 16,384-token cap in the final greedy passes (production boot 2 + 2: the known repetition-loop tail). Determinism (offline TP8 + EP8, production profile + the flag,
85 prompts of 9 .. 23,000 tokens, 128 new tokens, passes greedy / seeded t1k64 / seeded top_p 0.95 / seeded T 0.7 top_p 0.9 / mixed, then mixed, top_p 0.95 and greedy again): two cold processes identical 85/85 in all 8 passes,
pass 1 against pass 2 identical 85/85 in both processes (E2E_GATE PASS, 0 OOM). Accuracy class: rounding-level, run-to-run deterministic, identical across ranks.

**Non-speculative exact bytes (B4, S1-7 procedure re-run on the production tree, REPS = 2, stress A1 B C1 C2 D H I X, floor 500 MiB).** The S1-7 values left ~540 MiB of headroom on the current tree (S1-9 L2: S_live 1,238 / 1,166 MiB against
696 / 512). Stepped up to a target S_live of ~600 MiB (one stress boot each, clean, 0 errors / mismatches / preemptions / OOM):
ns128 (`--max-num-seqs 128`) **17,357,799,424** (264,859 blocks = **2,979,623 KV tokens** per rank, +114,838 = +4.0 % against 16,688,807,936 / 2,864,785), S_live 600.1 MiB, idle free 1,785.4 MiB, live growth 1,185.3 MiB, 1 allocator retry;
ns256 (`--max-num-seqs 256 --max-cudagraph-capture-size 256`) **16,957,243,392** (258,747 blocks = **2,910,864 tokens**, +101,878 = +3.6 % against 16,363,749,376 / 2,808,986), S_live 599.9 MiB, idle free 1,831.4 MiB, live growth 1,231.5 MiB, 4 retries.
Gate tier (valid without `prompt_logprobs` / `echo`; the prompt_logprobs-safe tier of S1-7 section 4 was not re-derived and stays a valid lower bound). Non-speculative launches do not carry the aligned-chunk flag (K = 0: the chunk is 8192 already).

Not merged: vLLM branches `muse/s19-bundle` (B1: the SP router overlap knob, review fixes folded in, a fault-injected bitwise test whose two mutants without the join / the fork fail it) and `muse/s19-l4guard` (B5: the level-4 plan guard), both kept in
`/home/user/vllm`; the production vLLM tree is unchanged (1eb639f807).

## S1-9 L7: the DFlash2 draft in FP8 (VLLM_MUSE_SPARK_SM120_DRAFT_FP8 = 2, kernel cutlass)

**What.** The RadixArk DFlash2 draft's GEMMs are weight-streaming bound at 8 .. 128 rows (gate_up, down, QKV, O, the ten kernel projections and the candidate head: 1.40 ms of a 17.2 ms c1 step at 1.17-1.40 TB/s). The vLLM branch quantizes them at load, from the
bf16 checkpoint, to e4m3 with one fp32 scale per output channel of each rank's shard (a row-parallel layer scales its own partial, the all-reduce sums them) and runs them W8A8 (dynamic per-token e4m3 activations + `cutlass_scaled_mm`; graph-capturable, no split-K,
bitwise repeatable). Flag 1 = the 30 draft linears (5 layers x QKV / O / gate_up / down + 2 kernel projections); flag 2 (the default) = also an FP8 copy of the target's lm_head shard for the draft's candidate top-k (+197 MiB per rank: the draft shares the target's bf16
head, which the target keeps). The context K/V precompute (a fused bf16 weight, prefill-sized GEMM) and the aux fc (FC_SHARD=2) stay bf16. The verification is exact: the draft is a proposal, the rejection sampler keeps the output distribution; the price is
acceptance length. `VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL=marlin` (weight-only FP8, lower error, no faster) is not gated.

**The flag, the kernel and the bytes are ONE property.** The idle memory at identical bytes differs per state (against the bf16 draft: cutlass flag 2 -418 MiB, cutlass flag 1 -616, marlin flag 2 -542, marlin flag 1 -740), and the hook's "670 MiB freed" is numel arithmetic, not the
allocator delta (the probe of the flag-2 cutlass stress boot: torch allocated -443.5 MiB = -641 weights - (-197) head, reserved -420, non-torch +2 MiB, so no persistent workspace; the CUDA graph pool is 0.95 GiB in every boot, flag 0 included, and the loaded weights are the
same for both kernels: the Marlin / cutlass gap is not in the captured allocations and does not move with the capture list). `scripts/check_spec_launch.sh` reads the state from the success lines of the vLLM branch, "DFlash2 draft weights in FP8 (...DRAFT_FP8=<n>, kernel <k>):
30 linear layers quantized" (one per worker) and, for flag 2, "DFlash2 candidate head in FP8": a boot that carries more than the maximum of its state (`FP8_F2_MAX_BYTES` 11,573,411,840, `FP8_F1_MAX_BYTES` 11,781,017,600, flag 0: 11,134,914,560; Marlin is checked against the cutlass value
of its flag), a flag set in the shell with no success line, a flag-2 boot without the head line (a silent fall back to flag 1), or any "DRAFT_FP8=<n> does not apply" warning FAILS. `SPEC_EXPECT_FP8=off|1|2` and `SPEC_EXPECT_FP8_KERNEL` demand a state (the flag exported by
`env/spec_k7.env` is the default expectation).

**Bytes (S1-7 procedure, probe tree, REPS = 2, stress A1 B C1 C2 D H I L M X, MNBT 8199).** Flag 2: at 11,134,914,560 clean, idle free 3,505.4 MiB (production 3,087.4), live growth 1,808.8 (1,809.0), **S_live 1,696.6 MiB**; rule: step up by (S_live - 1,278.4) MiB to
**11,573,411,840** (565,108 blocks) clean, **S_live 1,278.5 MiB** (floor 1,256 = production's own S_live), **1,629,269 KV tokens per rank** (+61,731, +3.94 %), idle 94,165 MiB (= production). Flag 1 (one stress boot at the flag-2 bytes + 198 MiB): 11,781,017,600 (575,245 blocks), clean,
S_live 1,278.4 MiB, 1,658,495 KV tokens per rank (+90,957, +5.80 %).

**Gate (two lock holds 2026-11-04: exact bytes + determinism, then the A/B; rule frozen before the first gate boot, `ADOPTION_RULE.txt` md5 2db75a8141f3 in the L7 scratch).** Ten speculative boots OFF x3 (flag 0 at 11,134,914,560), F1 x3, F2 x3, one production-tree control, ~31 min
each, all healthy (cfgcheck OK, 0 preemptions, launch check + arm check OK, KV tokens per arm identical), real mix320 prompts, closed loop; non-speculative guard 3 + 3 boots. Pooled primary score (real greedy and default c1 / c16 / c32, Welch 3 v 3, t 2.78): **F2 +1.50 % SIG**
(greedy c1 +2.02 % SIG, default c1 +2.94 % SIG, c16 +0.69 / +1.64 %, c32 +0.64 / +1.08 %); STEADY step ms OFF -> F2 c1 17.467 -> 16.934 (**-3.15 %**), c16 -1.03 %, c32 -1.03 %, c40 -1.06 %, c48 +0.66 % (noise), c64 -0.74 %, c128 -1.51 %; guards all passed (greedy c40 / c48 /
c64 / c128 +0.50 / -0.59 / +0.37 / +1.30 %, default c64 / c128 +1.06 / +0.15 %, churn c64 +0.51 %, natural 8192-token c16 +0.21 % and c1 +1.68 % SIG, TTFT 8k / 16k / 32k within +-0.3 %); control F2 / CTL +1.77 %, OFF / CTL +0.27 %. Flag 1 (layers only): pooled +1.36 % but not beyond
noise (se 0.0053 needs 1.47 %), so it is not adopted; F2 against F1 is +0.14 % pooled (noise; default c1 ratio 0.9994): the head copy is a wash at c1 on the point estimates and the rule chose F2 because F1 did not clear it. Acceptance length (request-level paired resampling,
Bonferroni over 15 cells, z_crit 2.94): every cell inside it (worst: greedy c128 -0.029, z -2.86, with its tok/s +1.30 %), natural 8192 / 32768-token decode cells -0.006 / +0.006 / -0.022, draft-weighted mean dAL **-0.010**.
Non-speculative guard (ON = the feature tree, no draft model, vs the production tree, 3 + 3 boots at 17,357,799,424): greedy / default c1 +0.21 / +0.25 %, c64 +0.15 / +0.09 %, STEADY c1 / c64 -0.04 % / +0.22 % (all noise).

**Accuracy / determinism.** GSM8K 200x2 at c64 with speculation (forward + reverse passes, 0 request errors; ON = flag 2 at 11,573,411,840, CTL = the production tree at 11,134,914,560, one boot each in the same hold): greedy 196 + 195 = **391/400 = 97.75 %** (band 95.5-98.0 %; CTL
193 + 192 = 385 = 96.25 %; 2 + 2 answers hit the 16,384-token cap against 4 + 4: the repetition-loop tail), default sampling 198 + 198 = **396/400 = 99.0 %** (CTL 393), top_p 0.95 on every request 392 (392), top_p 0.95 on every 4th item 394 (392); GSM8K_GATE PASS. Determinism (offline
TP8 + EP8, production profile at 11,573,411,840, 85 prompts of 9 .. 23,000 tokens, 128 new tokens, the 12-pass protocol: greedy, seeded t1k64 / top_p 0.95 / T 0.7 top_p 0.9 / T 1.5 top_p 0.95 / top_p 0.99, mixed and mixed-G (seeded t1k64, top_p and penalty requests interleaved), then mixed, mixed-G, top_p 0.95 and
greedy again): two cold processes identical 85/85 in all 12 passes (token ids, top-5 logprobs, prompt logprobs, exact), pass 1 = pass 2 inside both processes, 0 OOM (E2E_GATE PASS). Diagnostic only: the production tree against the FP8 draft is identical in 37 / 85 requests
(greedy), 15 (t1k64), 18 (top_p 0.95), 27 (mixed): the draft numerics differ at rounding level, so accept lengths, verify-batch composition and the target's roundings differ. Rank identity: an 8-rank test (NCCL and the host plugin's muse_topo collectives) on the real per-rank
shapes: the row-parallel o_proj / down (per-rank weight scale, per-token activation scale, all-reduce) stay within 4.7 % max error of the fp32 reference (a single-rank bound of 6 %), the replicated kernel projection and every output are bitwise identical on all ranks, a CUDA graph replays
the eager bits, the FP8 head's cross-rank top-k returns the aimed token of every one of the 8 vocabulary shards (value error 0.7 %, top-16 overlap 0.94). Accuracy class: rounding level, draft only (the rejection sampler keeps the output distribution), deterministic, rank-identical.

**Not done / limits.** The eager rows above 256 draft rows (c33 and up) gain a fraction of the microbench prediction (two Python-wrapped ops per FP8 linear, ~40-55 us of CPU per layer; measured +0.7 .. 1.5 % STEADY at c40 .. c128 against -1.5 % predicted at c64);
a fused act-quant or one composite custom op is the follow-up. Marlin and the per-M torch rowwise GEMM switching were not built into the default.

## S1-9 L11: sequence parallel inside the FULL decode graphs (A1a adopted, ON in the profile; A1b not adopted)

**Status: A1a ADOPTED by the pre-registered gate of 2026-11-06 / 07 (one lock hold of 13.5 h, 4 counterbalanced boot groups, exact bytes, determinism, GSM8K): `env/spec_k7.env` sets `MUSE_SPEC_GRAPH_SP_MIN_TOKENS=136` by default
(exports `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=136`), capture list and KV bytes unchanged (11,573,411,840, 1,629,269 KV tokens per rank). A1b (`MUSE_SPEC_GRAPH_CAP=384`) failed its rule and stays opt-in.
The profile needs a vLLM tree with the feature (6da60b3062 + the L11 commits, `268c64b0e2`): `check_spec_launch.sh` fails a boot that logs no graph-floor line or no flagged FULL decode graph at or above the floor. Set
`MUSE_SPEC_GRAPH_SP_MIN_TOKENS=0` to restore the previous profile (the vLLM default of the variable itself stays 0: the non-speculative launches never see it).** Gate results: section "Gate" below.

### What it changes

The vLLM branch (muse/s19-graphsp, 6 commits on 6da60b3062, tip 268c64b0e2) shards the residual stream inside the FULL decode CUDA graphs from a floor of its own,

    VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=<n>   (default 0 = off; raised to 129, the muse decode engine's 128 tokens + 1)

for the warm-up and the capture pass of a decode-only FULL graph (`ForwardContext.full_decode_graph`, set by `ModelCudaGraphManager`); eager batches, mixed batches and the PIECEWISE
graphs keep `sm120_sequence_parallel_min_tokens` (max(MIN_TOKENS 384, 257, cap + 1), unchanged). The model, every decoder layer and every attention output projection take their decision
in one function, so they cannot disagree on a batch. Below 129 tokens the stream stays replicated (the muse decode kernel serves T <= 128 and cannot take the sharded path:
`sm120_nvfp4_moe.py:816` assert). A verify batch is T = 8c: n = 136 shards c17..c32 inside the graphs the profile captures anyway (A1a, cap 256, no new graph size: pool 0.95 -> 0.92 GiB), and a larger
capture list (`--max-cudagraph-capture-size 384`: the default grid adds 272..384 in steps of 16) puts c33..c48 into FULL graphs too (A1b).

| requests | T | CTL (production) | A1a (`MUSE_SPEC_GRAPH_SP_MIN_TOKENS=136`) | A1b (A1a + `MUSE_SPEC_GRAPH_CAP=384`) |
|---|---|---|---|---|
| 1..16 | 8..128 | FULL graph, replicated, fused decode kernels | the same | the same |
| 17..32 | 136..256 | FULL graph, replicated | FULL graph, sequence-parallel | the same as A1a |
| 33..47 | 264..376 | eager, replicated (launch-bound, ~93 ms floor) | the same | FULL graph (padded to the step-16 size), sequence-parallel |
| 48 | 384 | eager, sequence-parallel | the same | FULL graph, sequence-parallel |
| 49..64 | 392..512 | eager, sequence-parallel | the same | the same (eager above the cap) |
| mixed / prefill batches | 257..384 | eager, replicated (257..383), SP (384) | the same | PIECEWISE graph, replicated (the cap moves the eager floor to cap + 1 = 385) |

Accuracy class: rounding level, deterministic (row-sharded GEMMs, reduce-scatter + all-gather instead of an all-reduce: the class of the stream above 384 tokens that production already runs eagerly);
with the host plugin's collectives the reduce-scatter + gather equals the all-reduce bitwise (tested below).

### Launch lines

    A1a (adopted, the default of the profile):  source env/spec_k7.env     -> VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=136, graphs <= 256, bytes 11,573,411,840 (exact: S_live 1,302.5 MiB, the pool is 24 MiB smaller)
    off:                                        MUSE_SPEC_GRAPH_SP_MIN_TOKENS=0 source env/spec_k7.env   (the profile of S1-9 L7)
    A1b (not adopted):  MUSE_SPEC_GRAPH_SP_MIN_TOKENS=136 MUSE_SPEC_GRAPH_CAP=384 MUSE_SPEC_KV_BYTES_OVERRIDE=11346923520 source env/spec_k7.env     (a cap above 256 without the override logs a warning: the bytes are the cap-256 ones)
    check: scripts/check_spec_launch.sh serve.log        (a shell that sourced the profile is the expectation; SPEC_EXPECT_GRAPH_SP=off|<n> [SPEC_CAP_MAX_BYTES=<bytes>] ... [385 384] for another state)

`check_spec_launch.sh` reads the graph-floor line of the boot log ("sequence parallel inside FULL decode CUDA graphs for batches of at least <g> tokens (...MIN_TOKENS=<env>; decode MoE engines serve up to <E> tokens; eager, mixed and PIECEWISE
batches keep the floor of <f> tokens)"), not the env: the floor on it must be the eager floor of the first line (cap + 1 above a cap of 383) and max(env, E + 1) with E the engine limit named on the line (129 in production; the
review fix: the rule is no longer hard-coded). Since the review it also reads the graph manager's line "CUDA graphs: <N> FULL decode graphs are flagged full_decode_graph (token counts <lo>..<hi>), ..." from every worker and FAILS an armed boot
unless N > 0 and hi >= the floor (a capture list without FULL decode graphs, e.g. PIECEWISE only, printed the floor line and sharded nothing). A set vLLM env without the floor line, a cap above 256 without `SPEC_CAP_MAX_BYTES`, and
workers that disagree FAIL (`bench/test_check_spec_launch.sh`: 40 synthetic-log / profile cases, no GPU).

### Graph safety (host plugin collectives inside the FULL graphs)

The muse_topo kernels keep their state on the device only (a 64-bit per-GPU call counter selects the double-buffered receive slots and the flag tags; no host state per call), so the
reduce-scatter / all-gather of the sharded stream are captured and replayed like the all-reduces before them. Evidence of this item (all with the host plugin's collectives, MUSE_TOPO_ENABLE=1, asserted):
- `test_sharded_stream_replays_from_a_full_decode_graph` (8 GPUs): the attention-output reduce-scatter + gather captured at 136, 137, 200, 256, 263, 264, 272, 376, 384, 400 and 512 tokens (multiples of 8 and padded rows):
  the graph replays the bits of the warm-up pass (single call and a chain of three layers' worth), 20,000 replays at 136 and 272 tokens next to eager calls of other sizes (flagged and not, so the counter parity
  alternates between graph and eager launches) give the same bits again, every rank holds the same bits, reduce-scatter + gather == all-reduce bitwise (NCCL: rel. L2 4.6e-3, graph == eager as well), an eager forward of
  the same projection stays replicated below 384 tokens.
- `test_sharded_branch_*` (real router, pooler, unpooler, NVFP4 / BF16 / Marlin experts, packed latent, EP): now also 136, 137, 200, 264, 272, 400 tokens, captured and replayed == eager.
- Real boots: the op summary of a graph-floor boot lists `reduce_scatter(dim=0) shape=(136..256, 4096 / 8192) bfloat16 capturing=True -> topo` (A1a: 16 sizes x 2 shapes; cap 384: 24 sizes) and the matching
  `all_gather` shapes (17..., 8192 bf16 etc.), 0 `-> nccl`, 0 errors / tracebacks / OOM in every boot, and the per-size op log reads `capturing=False` (the FULL warm-up) before `capturing=True` at 136, 272 and 384 although
  the PIECEWISE graphs are captured first: the PIECEWISE passes (mixed batches) are never sharded.

### Tests (feature tree 5b65462cbf = 268c64b0e2 after the message rewrite, same tree 730da5ec2e29; server tree src_s19-graphsp, host mirror host_topo_s19-graphsp; chain t2 of the fix round)

| suite | feature tree | production tree (6da60b3062) |
|---|---|---|
| Muse suite (tests/kernels/{moe,core}/test_muse_spark*.py, 1 GPU) | 3690 passed / 201 skipped / 0 failed | 3645 / 199 (L7): +45 new tests (21 CPU of the implementation, 24 of the review round: flagged-graph line, replay self-check, prefetch lists, engine limit), +2 skips = the 8-GPU tests on 1 GPU |
| SP file, 8 GPUs, NCCL | 104 passed | 77 (L10) + 27 new |
| SP file, 8 GPUs, host plugin | 102 passed, 2 failed (world-2 variants of test_sharded_branch_*: `NCCL error: unhandled system error` under host.env) | the same 2 fail on the production tree |
| r2e file (test_muse_spark_sm120_prefill_overlap.py), 8 GPUs | 53 passed / 8 skipped (NCCL and host plugin); with MUSE_R2E_REAL_CONFIG=1 61 passed | - |
| tests/v1/spec_decode + target-dependent AR speculator | 100 failed / 490 passed / 11 skipped, failing set identical to production's | 100 |
| tests/v1/cudagraph + worker cudagraph / ubatch / attn_utils files | 12 failed / 284 passed / 2 skipped | 12 failed / 270 passed / 2 skipped, failing set identical |
| bench/test_check_spec_launch.sh (host, no GPU) | 40 cases pass | - |

New in the review round: `test_sharded_moe_collectives_replay_from_a_full_decode_graph` (8 GPUs, NCCL and host plugin): the gathers of the routes ([rows, 16] int32) and the packed latent ([rows, 2304] uint8), the reduce-scatter of the experts' [T, 4096] partial
sums and the gather of the aux combiner's rows in `forward_sequence_parallel`'s order, captured at 136..512 tokens, bits equal to the eager pass, 20,000 replays at 136 and 272 tokens next to eager calls of other sizes, every rank the same rows,
reduce-scatter == all-reduce (host plugin: rel 1e-6 bound, NCCL 1e-2); the opt-in full-model replay self-check (`VLLM_CUDAGRAPH_REPLAY_SELFCHECK=1`, `..._MIN_TOKENS=136`); the L2 prefetch hooks warm no pooler / unpooler slice on the sharded stream.

### Microbench of the implementer in real boots, before the gate (STEADY step ms, c requests with long outputs, 3 x 10 s windows per cell, engine counters, GPU 0 clocks sampled; round-6 driver `drive_r6.py`)

Protocol: production speculative line (host.env + env/spec_k7.env of the host mirror, FP8 draft 2, FC shard 2, r2e, MNBT 8199, real weights, TP8 + EP8), 3 boot groups, every group boots CTL / A1a / A1b384 back to back
(order rotated: CTL-A1a-A1b, A1b-CTL-A1a, A1a-A1b-CTL), cells c1 (cold), c16, 20, 24, 28, 32, 33, 36, 40, 44, 47, 48, 49, 50, 56, 64, c1 again (after the sequence-parallel band); differences are paired inside the boot group
(boot-to-boot level shifts of 1.3 % exist: c1 is 17.25 or 17.02 ms per boot state). CTL = the production tree (sm120-dev/src = 6da60b3062) at 11,573,411,840 bytes; A1a = feature tree, graph floor 136, cap 256, the same bytes;
A1b384 = feature tree, graph floor 136, `--max-cudagraph-capture-size 384` at **10,774,691,840 bytes (-762 MiB: the pool grows 216 MiB; exact bytes are the gate's S1-7 job)**. All 9 boots clean (0 errors / OOM, check_spec_launch OK,
FP8 draft 8/8, level 4 8/8).

| cell | CTL ms (mean of 3 boots) | A1a - CTL ms (%) | A1b384 - CTL ms (%) |
|---|---|---|---|
| c1 cold | 17.25 | -0.07 (-0.4%) | -0.07 (-0.4%) |
| c1 after the SP band | 17.25 | -0.07 (-0.4%) | -0.23 (-1.3%) |
| c16 | 55.59 | -0.24 (-0.4%) | -0.38 (-0.7%) |
| c20 | 65.87 | -1.03 (-1.6%) | -1.40 (-2.1%) |
| c24 | 70.96 | -2.71 (-3.8%) | -2.86 (-4.0%) |
| c28 | 76.99 | -4.49 (-5.8%) | -4.83 (-6.3%) |
| **c32** | 82.29 | **-6.08 (-7.4%)** [-5.82 -6.41 -6.01] | -6.27 (-7.6%) [-6.20 -6.22 -6.40] |
| c33 | 88.81 | +0.08 (+0.1%) | **-9.48 (-10.7%)** [-9.73 -9.16 -9.54] |
| c36 | 92.52 | -0.56 (-0.6%) | **-12.34 (-13.3%)** |
| c40 | 95.55 | -0.41 (-0.4%) | **-12.28 (-12.9%)** [-12.34 -11.92 -12.58] |
| c44 | 100.51 | -0.33 (-0.3%) | **-14.07 (-14.0%)** |
| c47 | 103.84 | -0.33 (-0.3%) | **-13.26 (-12.8%)** |
| c48 | 94.16 | -0.37 (-0.4%) [+1.48 -1.46 -1.14: noise of the launch-bound eager cell] | **-4.20 (-4.5%)** [-3.91 -4.77 -3.92] |
| c49 / c50 | 94.54 / 94.73 | -0.29 / +0.00 | -0.29 / -0.09 (eager above the cap, like CTL) |
| c56 / c64 | 98.33 / 104.44 | -0.21 / -0.23 | -0.52 / -0.47 (inside the window noise) |

The round-6 probe numbers reproduce (probe: A1a c32 -6.0, c28 -4.1, c24 -2.4, c20 -1.4; A1b c33 -9.3, c36 -11.7, c40 -12.2, c44 -13.7, c47 -12.9, c48 -4.2, c56 -1.3, c64 -1.6 ms). c1 / c16 guards: no regression (c1 never above CTL by more than 0.01 ms in any boot; the boot
state shows as 17.25 or 17.02 / 17.03 ms: CTL 17.25 in 6 of 6 c1 cells, A1a 17.25 in 4 of 6 (17.02 / 17.03 in the second group), A1b384 17.02 / 17.03 in 4 of 6 (and 17.02 in both group-4 cells), A1b400 17.25 / 17.26; the state is per boot and was not attributed to the arm). Acceptance length in the STEADY cells (tokens per draft, greedy, informational):
c32 CTL 4.43 / A1a 4.63 / A1b 4.55, c40 4.32 / 4.31 / 4.50, c48 4.39 / 4.37 / 4.40, c64 4.38 / 4.40 / 4.42 (rounding-level numerics move the greedy continuations; no loss).
Flags-off feature tree (`off_OFF`, one boot): c1 17.25 / 17.25, c16 55.31, c32 82.21, c40 95.73, c48 94.56, c64 104.33 ms = CTL within noise; idle 94,165 MiB, pool 0.95 GiB, no graph-floor line: default off is production.

### Capture list for A1b: memory (pool boots at 10,774,691,840 bytes; idle MiB per GPU, normalised to the production bytes by +762 MiB) and the choice of 384

| arm | capture list | graph pool | idle at the boot bytes | idle at production bytes | vs CTL |
|---|---|---|---|---|---|
| CTL | 1..256 | 0.95 GiB | 93,403 | 94,165 (= production's own) | - |
| A1a | 1..256 | 0.92 GiB | 93,379 | 94,141 | -24 MiB |
| cap 384 | default grid, 272..384 step 16 (8 new sizes) | 1.16 GiB | 93,619 | 94,381 | +216 MiB (about -2.0 % KV at equal headroom: -31.9k of 1,629k tokens per rank) |
| cap 400 | 272..400 (9 new) | 1.18 GiB | 93,639 | 94,401 | +236 MiB |
| cap 512 | 272..512 (17 new) | 1.43 GiB | 93,901 | 94,663 | +498 MiB (about -4.5 % KV) |
| explicit step-8 list 264..384 | 16 new sizes | 1.34 GiB | 93,807 | 94,569 | +404 MiB |

Cap 384 is the set to gate: the gain sits in c33..c48 (-9.5 .. -14 ms, c48 -4.2), group 4 (CTL / A1b384 / A1b400, STEADY c40 .. c56) shows nothing for the extra sizes (A1b400 c49 +0.30, c50 -0.88, c52 +0.02 ms against CTL; the launch-bound eager SP cells
c49 .. c64 already sit at the GPU line, 94 .. 104 ms), and the pool grows with every size (a step-8 list costs 0.18 GiB more than the step-16 grid for the odd request counts' padding, ~0.4 ms, not measured as a gain here). The cap also moves the eager floor to cap + 1 = 385 and puts
mixed batches of 257 .. 384 tokens into PIECEWISE graphs (replicated): measured only through the real-prompt rows below. The exact A1b bytes need the S1-7 procedure (stress REPS = 2); the estimate above is the pool delta only.

### Real-prompt smoke (informational: ONE boot group g5, closed loop, n = 4c requests, 256 tokens, natural EOS, cold prefill by salts, greedy, 3 repeats per cell; boot-state offset up to 1.3 %; not the gate)

| c | CTL out tok/s | A1a vs CTL | A1b384 vs CTL | TTFT ms CTL / A1a / A1b384 |
|---|---|---|---|---|
| 24 | 1060 | +3.4 % | +4.2 % | 790 / 771 / 777 |
| 28 | 1153 | +4.2 % | +5.3 % | 852 / 803 / 790 |
| 32 | 1201 | +2.8 % | +4.0 % | 927 / 927 / 906 |
| 36 | 1215 | +1.8 % | +7.2 % | 1106 / 1036 / 996 |
| 40 | 1280 | +0.9 % | +7.9 % | 1135 / 1092 / 1071 |
| 44 | 1341 | +0.4 % | +6.3 % | 1176 / 1154 / 1133 |
| 48 | 1407 | +0.7 % | +4.4 % | 1209 / 1182 / 1150 |

(probe, SP also in the PIECEWISE passes: c24 +0.8, c28 +3.4, c32 +4.0 with A1a; c36 +5.9, c40 +5.5, c44 +6.7, c48 +2.9 with A1b). No failed requests, TTFT not worse in any cell, acceptance length 4.45 .. 4.58 for CTL and 4.46 .. 4.60 for A1a / A1b (never more than 0.02 below CTL in a cell).
KV trade of A1b384 (estimate): -2.0 % KV tokens per rank against +4.4 .. +7.9 % real-prompt tok/s at c36 .. c48 (c33..c47 band) in this one session.

### Gate (SG_COMMON template, pre-registered `ADOPTION_RULE.txt` frozen before the first A/B boot; one lock hold 2026-11-06 19:59 .. 2026-11-07 09:25, the exact bytes in an earlier hold 18:42 .. 19:59)

**Arms**: CTL = production tree + production host folder; OFF = the feature tree with the graph floor unset (tree / host-path control); A1a = graph floor 136, cap 256; A1b = graph floor 136 + `--max-cudagraph-capture-size 384`; each at its exact bytes,
4 boot groups in a counterbalanced order (every arm once per position), plus two c1 layout-control boots each of OFFB (OFF at A1b's bytes) and CAPB (cap 384, no graph floor, A1b's bytes). 20 speculative boots of ~40 min (STEADY c1 / c8 / c16 / c17..c48 / c56..c128
with 2 windows of 12 s per ramp, real greedy / default c1..c128, churn, TTFT 8k / 16k / 32k, low-c TTFT of 200..3000-token prompts at c1 / c4 / c16 and next to 8 / 16 / 24 decoders), 6 non-speculative boots, 3 GSM8K boots (A1a c64 and c32, production c64: the A1b arm was not taken to GSM8K), 8 determinism processes; 0 retries,
every boot healthy (cfgcheck OK, 0 preemptions, launch check and arm check OK). Scoring paired inside the boot groups (t crit 3.18 at 4 groups); the c1 / c16 guards compare the worst state of each boot.

**Exact bytes** (S1-7 procedure, probe tree, REPS = 2, stress A1 B C1 C2 D H I L M X, MNBT 8199): A1a at the production bytes 11,573,411,840: clean, S_live 1,302.5 MiB (production 1,278.5), idle free 3,111.4 MiB (production 3,087.4), 1,629,269 KV tokens per rank:
**no KV cost, bytes unchanged**. A1b at production - 216 MiB = 11,346,923,520 (554,049 blocks, 1,597,384 KV tokens per rank): clean, S_live 1,278.5 MiB = production's: A1b costs 1.957 % of the KV tokens (31,885 per rank).

**A1a against the flag-off tree** (OFF; paired, 4 groups; gain = faster / more tok/s):

| cell | gain | beyond noise |
|---|---|---|
| real greedy c24 / c28 / c32 out tok/s | +0.92 / +2.28 / +2.29 %, pooled **+1.83 %** (production delta +1.64 %) | yes (rule: >= +1.0 %) |
| STEADY step c20 / c24 / c28 / c32 | +1.86 / +3.03 / +5.28 / +6.99 %, pooled **+4.29 %** | yes (rule: >= +2.0 %) |
| STEADY step c17 / c18 / c19 / c21 (band start) | +0.77 / +1.12 / +1.83 / +2.97 % | yes: no cell slower, the floor stays 136 |
| STEADY c56 / c64 / c96 / c128 | +0.10 / +0.21 / +0.14 / -0.31 % | noise (rule: not worse than -1.0 %) |
| c1 / c16 worst state | +0.28 % / -0.11 % (production +0.98 % / +0.01 %) | noise |
| real greedy c1 / c16 / c56 / c64 / c96 / c128 | 0.00 / +0.89 / -0.05 / -0.13 / +0.55 / +0.26 % | noise |
| real default c1 / c16 / c64 / c128, churn c64 | +0.01 / -0.12 / +0.40 / +0.44 %, +0.29 % | noise |
| TTFT 8k / 16k / 32k | +0.13 / +0.02 / -0.18 % | noise |
| low-c TTFT, pooled over the 10 lengths: c1, c16, c1 + 8 / 16 / 24 decoders | +0.19 / +0.45 / +1.56 / +0.09 / **+7.1 %** | only the last (next to 24 decoders: the decode steps are faster) beyond noise |
| acceptance length | draft-weighted dAL -0.0001 over 19 cells; worst cell greedy c48 -0.037 (z -2.79 < 3.01, tok/s guard passes) | no trip |

Non-speculative guard (ON = feature tree defaults vs production, 3 groups): real greedy c1 / c64 -0.02 / +0.26 %, default c1 / c64 +0.02 / -0.28 %, STEADY c1 / c64 +0.01 / -0.11 %: noise. Determinism: two cold offline processes (TP8 + EP8, 85 prompts of 9..23,000 tokens, 128 new tokens,
12 passes: greedy, t1k64, top_p variants, mixed, repeats) identical in every pass at mns 128 / cap 256 and with the batches pinned into the band (mns 24: T = 136..192 almost every step), and for A1b at mns 128 / cap 384 and mns 40; the full-model replay self-check
(`VLLM_CUDAGRAPH_REPLAY_SELFCHECK=1`, graphs >= 136 tokens: 16 / 8 / 24 / 20 graphs, all 8 workers) found every graph bit for bit equal to its flagged eager pass, twice. GSM8K 200x2 with speculation: A1a at c64 greedy 389/400 (97.25 %, band 95.5-98.0; production
388 = 97.00 %), default sampling 395 (production 395), top_p 0.95 394 (392), mixed top_p 395 (394); at c32 greedy 389, default 393 (the c32 boots run greedy and default only: the top_p lines read MISSING by design); 0 request errors.

**A1b (capture list 384) failed its rule**: STEADY c33 / c36 / c40 / c44 / c47 / c48 are +11.6 / +14.6 / +14.8 / +15.2 / +13.7 / +4.1 % faster than A1a, but the real-prompt cells c36 / c40 / c44 / c48 are only +2.67 / +2.63 / +3.25 / +1.66 % (pooled
**+2.55 %** against the pre-registered threshold 1.96 % KV lost + 2.0 points = 3.96 %: the real rows are bursts of 128-token outputs, probably dominated by prefill and ramp, which dilutes the decode gain; not separated further), and the cap moves the mixed-step edge: the TTFT of a 256-token prompt next to 16 decoders (mixed step of
384 tokens: eager SP in production, a replicated PIECEWISE graph with the capture list) is 7.5 % worse than the flag-off tree (6.7 % vs production, 6.8 % vs A1a), a 257-token prompt next to 8 decoders (321 tokens) 6.5 % worse than production, and bare 257 / 384-token prompts 5.1 % worse than
A1a: the predicted side effect of a capture list above 256, measured. c56 / c64 / c96 / c128 and churn of A1b are fine (churn +0.99 %). The c4 low-c TTFT row is bimodal (the TTFT of four simultaneous probes comes in ~240 / ~480 ms steps with the arrival order): informational in practice, it did not trip.

c1 two-state variable (the review's item): in this protocol (12 s windows, 2 per ramp, c1 after c8 and after c40) the worst-state c1 ms of a boot are CTL 17.33 / 17.68 / 17.39 / 17.55, OFF 17.26 / 17.30 / 17.29 / 17.61, A1a 17.24 / 17.33 / 17.44 / 17.26, A1b 17.32 / 17.69 / 17.28 / 17.39, the
layout controls OFFB 17.29 / 17.22 and CAPB 17.31 / 17.99: the bumps (17.5 .. 18.0 in the cell after c8, or after c128) appear in every arm including production and the layout controls: not attributable to the arm.

Evidence: server `/raid/vllm-runtime/work/s1-9/L11/g/` (ADOPTION_RULE.txt, verdict.txt, master_l11.log, scripts, determinism outputs), `musetest/results/s19l11g_*` (A/B), `s19l11gd_*` (c1 controls), `s19l11n_*` (non-speculative), `s19l11x_*` (stress boots),
`s19l11g_gsm_*` (GSM8K); local `/tmp/muse_wf/s1-9/L11/` (PROGRESS.md, gate/, verdict_final.txt, master_l11.log).
