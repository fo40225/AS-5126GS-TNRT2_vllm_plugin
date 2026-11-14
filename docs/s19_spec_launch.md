# Speculative launch profile: graph cap 256 + sequence-parallel floor 384 (S1-9 lever 1)

Launch-scoped, no vLLM code change: `env/spec_k7.env` (sourced after `env/host.env`, speculative launches only) and
`scripts/check_spec_launch.sh` (the boot check). Measured on `sm120-muse-spark-open` 0ffa57b714 + host b207e54; production pair after the merge: vLLM
5b827b215a (0ffa57b714 plus the SP test file, the runtime code is identical) + this folder.

**S1-9 lever 2 (the microbatched prefill beside the drafter, "r2e") is part of this profile since 2026-11-01**: `env/spec_k7.env` also sets
`VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE=1` and the KV bytes were **10,373,877,760** (1,460,755 tokens per rank). See the section "S1-9 lever 2" at the end of this file.

**S1-9 L4 (the DFlash2 aux projection on each rank's rows, `VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD=2`, the vLLM default) moves the bytes to 11,134,914,560** (543,697 blocks = 1,567,918 tokens per rank, +7.3 %):
`env/spec_k7.env` exports the flag and sets `MUSE_SPEC_KV_BYTES` from it (11,134,914,560 with the flag at 2, 10,373,877,760 otherwise). See the section "S1-9 L4" at the end of this file.

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
