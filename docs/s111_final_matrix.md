# S1-11 SG1 integration: final matrix, exact bytes, gates and known limits

Evidence document of the production launch profiles (`launch/serve_spec.sh`, `launch/serve_nonspec.sh <ns128|ns256>`, README section 4.7 and 4.8).
Every number of the matrix and gate tables below is copied from `muse_wf/s1-11/PHASE2_RESULTS.md` (the analysis of the phase 2 run of 2026-11-13,
independently re-verified: `muse_wf/s1-11/phase2_verify.json`, verdict AGREE, 216 cells and 4237 value comparisons, 0 differences); nothing in this file is recomputed.
Section numbers that appear inside copied paragraphs (for example "section 4" or "section 10") refer to `PHASE2_RESULTS.md`, not to this file. The table rows are extracted by script (`muse_wf/s1-11/tools/docs_extract.py`, `docs_build.py`) and checked line by line against the source.

## 1. Provenance and scope

- Matrix and gates ran on the production trees host main `f540ce1b37` (205 files, git-cross-checked against the deployed `host_topo`) and vLLM `sm120-muse-spark-open` `f6a60ab9d7`
  (`sm120-dev/src`, 10,757 files), pins re-verified in every hold. Chain 2026-11-13 02:36:49 -> 07:11:47 (4 h 35 min, 13 holds, all rc 0), 7 boots, 216 cells.
- Every boot went through the launch scripts and passed every post-boot check of `f540ce1b37` (without the F1 stop-path fallback and the F3 / `kv_connector_extra_config` line-check additions of this branch; live environment, live command line = `serve_cmd.txt` = clean-shell derivation, KV, offload, spec).
- The commits made after the run (branch `s111-docs`, cut from `f540ce1b37`) change only the stop path of the launcher (`launch/lib.sh`, `launch/session.sh`: the accepted session tuple is the
  fallback when `server.sess` is gone; `NOT_STOPPED` instead of a false `STOPPED`), the serve-line check (`scripts/check_serve_line.py`: the log's `kv_connector` / `kv_role` / non-empty `kv_connector_extra_config` must
  also be on the line), tests and documentation. No exported variable, serve line or default changes; the measured configurations are the ones of `f540ce1b37`.
- A measurement, not an adoption decision. The gate criteria were fixed before the run (S1-11 Amendment 3 and the phase 2 rulings); the 9 gates are 9 / 9 PASS.

### Protocol labels

- **STEADY**: `drive_steady_g11.py`, constant batch (spec `--win 12 --windows 3`; ns128 `--win 10 --windows 3`; ns256 `--win 12 --windows 2`, each equal to its baseline's protocol); value = upper
  median of the window step times; tok/s and TPOT are DERIVED (c x 1000 / step x tokens per step); TTFT is not measured.
- **DEC**: `conc_driver_s110.py`, gsm80 + mtbench real prompts, n = max(16, 4c), 128 output tokens, effort xhigh, per-request `cache_salt`; greedy has 2 repetitions (the grids show the mean),
  default sampling and top_p 0.95 one repetition each.
- **rand 512/128**: `drv_rand_g11.py` random 512-token prompts, 128 output tokens. **natural 8k**: natural 8,192-token prompts. **mixed churn**: S1-9 L2 mixed churn at c4..c128 (+ c256 for
  ns256), 256 tokens, jitter 24, stagger 0.15 s (c1 is not a churn cell).
- **Cold walls**: client wall with `max_tokens 1`, fresh salt, median of 5 (= cold TTFT at 8k / 32k / 64k / 118k prompt tokens).
- **Prefix-reuse workload**: the frozen reuse driver, cycle, 32k-token documents, working set about 1.61-1.62 x the GPU pool (spec ON was driven with the OFF pool size 1,649,935 tokens = 1.619 x its
  own pool of 1,639,394; spec OFF 1.608 x; ns256 ON used its own pool and 142 documents = 1.608 x), 4 clients.
- **Percentiles**: the conc / rand / nat8k drivers use the index `round(p * (n - 1))` (the p50 of an even n is the upper median); the reuse driver uses `ceil(p * n) - 1` (the p99 of 81 requests is the
  maximum). Every reported value is correct under its own driver's definition; the definitions differ.
- Every non-reuse cell is salted (per-request `cache_salt` or the driver's reset), and the delta of `prompt_tokens_cached_by_source` is recorded per cell: 216 of 216 cells valid, 0 invalid, 0 failed
  requests, 0 preemptions, disk / p2p / external_unspecified deltas 0 in every cell. Reuse, goff and long cells carry tier hits by design and are labelled so.

## 2. Boots, launch commands and the swap record

```bash
cd /raid/vllm-runtime/host_topo
launch/serve_spec.sh --out DIR                       # spec, tier ON (default)       -> boot spec_on
MUSE_OFFLOAD=0 launch/serve_spec.sh --out DIR        # spec reference, no tier       -> boot spec_off
launch/serve_nonspec.sh ns256 --out DIR              # ns256, tier ON (default)      -> boots ns256_on, ns256_goff, ns256_long
MUSE_OFFLOAD=0 launch/serve_nonspec.sh ns256 --out DIR   # ns256 no tier             -> boot ns256_off
launch/serve_nonspec.sh ns128 --out DIR              # ns128, no tier (default)      -> boot ns128_off
```

| boot | launch | --kv-cache-memory-bytes | KV tokens/rank | READY MiB/GPU | load s | cells | invalid | swap before | swap READY | swap peak | swap end of cells | swap after stop | boot pswpin | boot pswpout | cells pswpin | cells pswpout | cells pgmajfault |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec_on | spec ON (serve_spec.sh default, tier ON) | 11645337600 | 1,639,394 | 94159 | 105 | 54 | 0 | 48.8 | 49.1 | 49.1 | 49.1 | 49.1 | 0 | 2 | 0 | 0 | 50 |
| spec_off | spec MUSE_OFFLOAD=0 (reference) | 11720212480 | 1,649,935 | 94133 | 86 | 51 | 0 | 49.1 | 50.3 | 50.3 | 50.3 | 50.3 | 0 | 258 | 0 | 0 | 4 |
| ns256_on | ns256 ON (default, tier ON) | 16854810624 | 2,893,281 | 95519 | 120 | 59 | 0 | 50.3 | 4095.3 | 5429.6 | 4094.6 | 50.6 | 441,479 | 1,379,760 | 21,187 | 0 | 14,096 |
| ns128_off | ns128 OFF (default, no tier) | 17357799424 | 2,979,623 | 95565 | 81 | 49 | 0 | 50.6 | 51.6 | 51.6 | 51.6 | 51.6 | 0 | 258 | 0 | 0 | 0 |
| ns256_off | ns256 MUSE_OFFLOAD=0 (G-off OFF boot) | 16957243392 | 2,910,864 | 95519 | 75 | 1 | 0 | 51.6 | 54.3 | 55.3 | 54.3 | 54.3 | 125 | 884 | 0 | 0 | 466 |
| ns256_goff | ns256 ON, dedicated G-off boot | 16854810624 | 2,893,281 | 95519 | 116 | 1 | 0 | 54.3 | 4570.6 | 6677.8 | 4536.3 | 64.1 | 564,669 | 1,703,387 | 8,889 | 0 | 4,888 |
| ns256_long | ns256 ON, dedicated G-long boot | 16854810624 | 2,893,281 | 95519 | 95 | 1 | 0 | 63.3 | 63.8 | 63.8 | 63.8 | 63.8 | 0 | 81 | 2 | 3 | 7 |

- spec ON (the adoption-type boot) did not swap; ns256 ON swapped at READY in 2 of 3 boots: `ns256_on` 4,095.3 MiB used (peak 5,429.6 MiB) and `ns256_goff` 4,570.6 MiB used (peak 6,677.8 MiB)
  of the 8,192 MiB swap file; the third (`ns256_long`, started right after `ns256_goff` stopped) did not (63.8 MiB). Swap was fully released at each stop. Cells are NOT voided: every cell
  row carries the swap state of its boot and the `pswpin` / `pswpout` / `pgmajfault` deltas of the cell. In the cells of `ns256_on` the largest swap-in is 4,206 pages (the warm cell), the STEADY
  cells 513-2,763 pages, the reuse cells 2,187 pages, the GSM cells 477 pages, `pswpout` is 0 everywhere (the page-in of the first cells after READY; no swap-out under load).
- ON pools are 0.64 % (spec) / 0.6 % (ns256) smaller than the OFF pools by design (exact bytes per profile, section 3).
- The 404 answer of `POST /reset_prefix_cache` on the ns profiles (no `VLLM_SERVER_DEV_MODE`) is expected.

## 3. Exact bytes, tokens and the tier formula

| profile | tier | `--kv-cache-memory-bytes` | blocks per rank (bytes per block-id) | KV tokens per rank | S_live (floor) | source of S_live |
|---|---|---:|---:|---:|---:|---|
| spec | OFF (`MUSE_OFFLOAD=0`) | 11,720,212,480 | 572,276 (20,480) | 1,649,935 | 1,278.4 MiB (1,256) | S1-13 boot on 0.33.0 |
| spec | ON (default) | 11,645,337,600 | 568,620 (20,480) | 1,639,394 | 1,256.6 MiB (1,256) | S1-10R phase 3 confirmed boot |
| spec | ON (default) | 11,645,337,600 | 568,620 (20,480) | 1,639,394 | 1,256.4 MiB (1,256; margin 0.4 MiB) | S1-10R gate X boot |
| ns256 | OFF (`MUSE_OFFLOAD=0`) | 16,957,243,392 | 258,747 (65,536) | 2,910,864 | 500.8 MiB (500) | 0.33.0 validation boot |
| ns256 | ON (default) | 16,854,810,624 | 257,184 (65,536) | 2,893,281 | 501.8 MiB (500) | S1-10R phase 3 / gate X boot |
| ns128 | OFF (default) | 17,357,799,424 | 264,859 (65,536) | 2,979,623 | 503.1 MiB (500) | 0.33.0 validation boot |
| ns128 | ON (`MUSE_OFFLOAD=1`) | 17,257,267,200 | 263,325 (65,536) | 2,962,366 | 501.1 MiB (500) | S1-10R phase 3 / gate X boot |

The numbers live in `env/spec_k7.env`, `env/offload.env` and `env/launch.env` (the only source of numbers for the launch scripts). The tier ON bytes are the no-tier bytes stepped down by
the S_live shortfall that the GPU page tables of the registered tier cause (2 MiB per GiB registered, 96.5 MiB at P*; steps -74,874,880 / -102,432,768 / -100,532,224 B for spec / ns256 / ns128, in
2 MiB granules), found with the S1-7 stress procedure (`docs/s110_numa_offload.md` section 5a). The two spec ON S_live values come from two different runs and are not merged: 1,256.6 MiB is the
S1-10R phase 3 confirmed boot, 1,256.4 MiB the gate X boot that sets the 0.4 MiB margin. The margins are thin (spec ON 0.4 MiB, ns ON 1.1-1.8 MiB): a serve line or an environment that is not one of
the six launches is a new configuration (S1-17: per-configuration offsets), not covered by these numbers.

**Per-rank tier** (`env/offload.env`, `docs/s110_numa_offload.md` section 4a):

```
P* = L x floor((min_N A_N - M) / (2 L))          L = 327,680 B (lcm of the bytes per block-id 20,480 and 65,536), M = 6.6 GiB
A_N = MemFree_N + Active(file)_N + Inactive(file)_N per GPU node, measured on 0.33.0 with the server up (MiB):
   node 3 119,262.8    node 0 118,644.1    node 7 105,586.3 (the limiter)    node 4 119,641.2
P* = 327,680 x floor((105,586.3 MiB - 6.6 GiB) / 655,360 B) = 327,680 x 158,124 = 51,814,072,320 B per rank (48.2556 GiB)
cpu_bytes_to_use_per_rank = P* = 51,814,072,320        cpu_bytes_to_use = 8 x P* = 414,512,578,560 B (386 GiB for the eight ranks)
tier blocks per rank: 2,529,984 (spec, 20,480 B per block-id), 790,620 (ns256 / ns128, 65,536 B per block-id)
```

P* is valid only while the reclaimable slab of node 7 stays below about 0.3 GiB (it was 302 MiB when A_7 was measured; an older boot carried 6.7 GiB of it): run
`scripts/preflight_offload.sh` before every ON boot (the launch scripts do).

## 4. Matrix

Tables in the order spec ON, spec OFF (reference), ns256 ON, ns128 OFF. Positive "worse %" in section 5 means worse than the baseline. The per-cell appendix (every cache source, swap, `pswpin`,
`pgmajfault` column) is in `muse_wf/s1-11/PHASE2_RESULTS.md` appendix A.

### 4.1 spec ON (serve_spec.sh default, tier ON)

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 |
|---|---:|---:|---:|---:|---:|---:|
| STEADY step ms | 17.579 | 30.394 | 56.367 | 77.456 | 104.396 | 153.926 |
| STEADY tok/s (derived) | 402.9 | 737.2 | 1395.8 | 1935.4 | 2813.8 | 3752.0 |
| DEC greedy out tok/s | 221.4 | 416.5 | 756.6 | 961.9 | 1252.2 | 1479.1 |
| DEC greedy TPOT ms | 3.49 | 7.30 | 15.19 | 23.24 | 36.64 | 61.86 |
| DEC greedy TTFT p50 ms | 154 | 216 | 462 | 644 | 1033 | 1746 |
| DEC greedy TTFT p99 ms | 179 | 462 | 1419 | 2544 | 4361 | 8875 |
| DEC greedy AL | 4.671 | 4.639 | 4.872 | 4.880 | 4.830 | 4.841 |
| DEC default out tok/s | 226.3 | 414.7 | 745.0 | 973.0 | 1232.0 | 1444.9 |
| DEC default TPOT ms | 3.38 | 7.20 | 15.48 | 22.75 | 37.48 | 64.04 |
| DEC default TTFT p50 ms | 157 | 256 | 458 | 737 | 1045 | 1714 |
| DEC default TTFT p99 ms | 177 | 484 | 1451 | 2587 | 4386 | 8843 |
| DEC default AL | 4.802 | 4.560 | 4.780 | 4.902 | 4.693 | 4.715 |
| DEC top_p 0.95 out tok/s | 229.3 | 432.8 | 739.6 | 963.7 | 1238.8 | 1472.6 |
| DEC top_p 0.95 TPOT ms | 3.32 | 6.81 | 15.75 | 23.04 | 36.99 | 62.91 |
| DEC top_p 0.95 TTFT p50 ms | 158 | 258 | 471 | 718 | 1094 | 1763 |
| DEC top_p 0.95 TTFT p99 ms | 180 | 488 | 1432 | 2543 | 4425 | 8842 |
| DEC top_p 0.95 AL | 4.895 | 4.923 | 4.683 | 4.859 | 4.751 | 4.735 |
| rand 512/128 out tok/s | 148.5 | 283.1 | 578.5 | 768.3 | 1008.6 | 1272.2 |
| rand 512/128 TPOT ms | 5.90 | 11.86 | 22.65 | 34.58 | 52.75 | 83.26 |
| rand 512/128 TTFT p50 ms | 116 | 156 | 276 | 405 | 572 | 875 |
| rand 512/128 TTFT p99 ms | 117 | 298 | 777 | 1251 | 2084 | 3911 |
| rand 512/128 AL | 2.845 | 2.744 | 3.050 | 2.929 | 2.905 | 3.015 |
| natural 8k out tok/s | 92.0 | 130.2 | 156.1 | 168.9 | 175.2 | 177.2 |
| natural 8k TPOT ms | 5.29 | 21.21 | 66.16 | 126.34 | 184.97 | 200.43 |
| natural 8k TTFT p50 ms | 717 | 779 | 3323 | 4780 | 18519 | 63153 |
| natural 8k TTFT p99 ms | 736 | 2692 | 10354 | 20404 | 41466 | 82973 |
| natural 8k AL | 3.140 | 3.209 | 3.207 | 3.212 | 3.187 | 3.205 |
| mixed churn out tok/s | - | 498.5 | 918.3 | 1201.0 | 1567.4 | 1944.9 |
| mixed churn TTFT p50 ms | - | 212 | 324 | 461 | 652 | 1053 |
| mixed churn TTFT p99 ms | - | 278 | 448 | 655 | 935 | 1352 |
| mixed churn AL | - | 4.519 | 4.534 | 4.549 | 4.533 | 4.517 |
| preemptions (all cells) | 0 | 0 | 0 | 0 | 0 | 0 |

Cold client wall (max_tokens 1, fresh salt, median of 5; = cold TTFT 8k / 32k / 64k / 118k): 8192: 0.7430 s, 32768: 2.9749 s, 65536: 6.2794 s, 117964: 12.4243 s.

Prefix-reuse workload (frozen reuse driver, cycle, 32k docs, working set 1.6 x pool, 4 clients; prompt tok/s, TTFT p50 / p99 s, host-source tokens): cycle_p1 11,059.6 tok/s, 5.247 / 11.830 s, host 0 (81/81 ok); cycle_p2 204,306.1 tok/s, 0.256 / 0.512 s, host 2,658,096 (81/81 ok); cycle_p3 202,163.9 tok/s, 0.256 / 0.608 s, host 2,658,096 (81/81 ok). AL during the cell 1.529, preemptions 0.

### 4.2 spec MUSE_OFFLOAD=0 (reference)

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 |
|---|---:|---:|---:|---:|---:|---:|
| STEADY step ms | 17.301 | 30.783 | 55.583 | 76.474 | 105.314 | 153.916 |
| STEADY tok/s (derived) | 410.2 | 724.8 | 1433.4 | 1966.4 | 2801.9 | 3757.0 |
| DEC greedy out tok/s | 224.5 | 427.0 | 752.0 | 975.8 | 1254.3 | 1481.8 |
| DEC greedy TPOT ms | 3.43 | 6.94 | 15.23 | 22.86 | 36.63 | 61.93 |
| DEC greedy TTFT p50 ms | 154 | 233 | 467 | 646 | 1063 | 1748 |
| DEC greedy TTFT p99 ms | 178 | 461 | 1392 | 2586 | 4354 | 8804 |
| DEC greedy AL | 4.671 | 4.710 | 4.791 | 4.946 | 4.838 | 4.839 |
| DEC default out tok/s | 229.4 | 409.9 | 736.8 | 967.8 | 1234.1 | 1482.9 |
| DEC default TPOT ms | 3.33 | 7.27 | 15.67 | 23.11 | 37.27 | 63.47 |
| DEC default TTFT p50 ms | 155 | 254 | 458 | 695 | 962 | 1683 |
| DEC default TTFT p99 ms | 176 | 460 | 1423 | 2579 | 4325 | 8278 |
| DEC default AL | 4.802 | 4.529 | 4.646 | 4.854 | 4.730 | 4.712 |
| DEC top_p 0.95 out tok/s | 232.2 | 439.9 | 741.2 | 972.1 | 1238.9 | 1475.7 |
| DEC top_p 0.95 TPOT ms | 3.28 | 6.79 | 15.58 | 22.81 | 37.31 | 62.69 |
| DEC top_p 0.95 TTFT p50 ms | 159 | 253 | 467 | 780 | 1050 | 1764 |
| DEC top_p 0.95 TTFT p99 ms | 178 | 476 | 1407 | 2529 | 4326 | 8635 |
| DEC top_p 0.95 AL | 4.895 | 4.988 | 4.724 | 4.821 | 4.725 | 4.774 |
| rand 512/128 out tok/s | 150.3 | 284.6 | 581.5 | 772.7 | 997.8 | 1269.5 |
| rand 512/128 TPOT ms | 5.82 | 11.79 | 22.52 | 34.37 | 50.64 | 83.95 |
| rand 512/128 TTFT p50 ms | 115 | 155 | 274 | 405 | 617 | 904 |
| rand 512/128 TTFT p99 ms | 119 | 297 | 777 | 1242 | 2704 | 3881 |
| rand 512/128 AL | 2.845 | 2.744 | 3.050 | 2.929 | 3.044 | 2.970 |
| natural 8k out tok/s | 93.0 | 130.4 | 156.4 | 169.0 | 175.4 | 177.3 |
| natural 8k TPOT ms | 5.22 | 21.18 | 66.05 | 126.08 | 184.19 | 199.93 |
| natural 8k TTFT p50 ms | 714 | 779 | 3309 | 4783 | 18514 | 63152 |
| natural 8k TTFT p99 ms | 723 | 2679 | 10333 | 20389 | 41437 | 82968 |
| natural 8k AL | 3.140 | 3.209 | 3.207 | 3.236 | 3.219 | 3.218 |
| mixed churn out tok/s | - | 501.6 | 924.0 | 1206.3 | 1572.7 | 1922.1 |
| mixed churn TTFT p50 ms | - | 211 | 319 | 466 | 657 | 1037 |
| mixed churn TTFT p99 ms | - | 277 | 449 | 733 | 1052 | 1858 |
| mixed churn AL | - | 4.519 | 4.534 | 4.524 | 4.540 | 4.495 |
| preemptions (all cells) | 0 | 0 | 0 | 0 | 0 | 0 |

Cold client wall (max_tokens 1, fresh salt, median of 5; = cold TTFT 8k / 32k / 64k / 118k): 8192: 0.7286 s, 32768: 2.9633 s, 65536: 6.2657 s, 117964: 12.4090 s.

Prefix-reuse workload (frozen reuse driver, cycle, 32k docs, working set 1.6 x pool, 4 clients; prompt tok/s, TTFT p50 / p99 s, host-source tokens): cycle_p1 11,048.6 tok/s, 5.151 / 11.840 s, host 0 (81/81 ok); cycle_p2 11,036.4 tok/s, 5.146 / 12.457 s, host 0 (81/81 ok); cycle_p3 11,049.6 tok/s, 5.217 / 11.834 s, host 0 (81/81 ok). AL during the cell 1.461, preemptions 0.

### 4.3 ns256 ON (default, tier ON)

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 | c256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEADY step ms | 8.189 | 13.565 | 23.176 | 31.264 | 40.018 | 56.101 | 83.373 |
| STEADY tok/s (derived) | 122.1 | 294.9 | 690.4 | 1023.5 | 1599.3 | 2281.6 | 3070.6 |
| DEC greedy out tok/s | 115.2 | 263.5 | 535.5 | 722.6 | 980.9 | 1212.2 | 1432.2 |
| DEC greedy TPOT ms | 7.73 | 12.28 | 21.95 | 30.54 | 41.86 | 72.89 | 133.05 |
| DEC greedy TTFT p50 ms | 158 | 366 | 935 | 1640 | 2980 | 3824 | 4072 |
| DEC greedy TTFT p99 ms | 169 | 447 | 1372 | 2503 | 4215 | 8316 | 15842 |
| DEC default out tok/s | 114.4 | 265.5 | 531.6 | 721.6 | 978.9 | 1219.6 | 1444.0 |
| DEC default TPOT ms | 7.79 | 12.31 | 22.11 | 29.94 | 42.40 | 72.87 | 133.23 |
| DEC default TTFT p50 ms | 158 | 345 | 935 | 1709 | 2888 | 3864 | 4033 |
| DEC default TTFT p99 ms | 169 | 431 | 1390 | 2491 | 4172 | 7980 | 15170 |
| DEC top_p 0.95 out tok/s | 113.7 | 262.9 | 528.0 | 717.6 | 963.6 | 1190.5 | 1394.1 |
| DEC top_p 0.95 TPOT ms | 7.84 | 12.48 | 22.28 | 30.47 | 43.59 | 75.49 | 138.13 |
| DEC top_p 0.95 TTFT p50 ms | 158 | 345 | 937 | 1686 | 2952 | 3742 | 3993 |
| DEC top_p 0.95 TTFT p99 ms | 170 | 432 | 1390 | 2443 | 4172 | 8191 | 16196 |
| rand 512/128 out tok/s | 113.3 | 263.3 | 534.3 | 736.3 | 1011.1 | 1295.0 | 1543.8 |
| rand 512/128 TPOT ms | 8.12 | 13.37 | 25.22 | 36.08 | 52.20 | 82.50 | 142.48 |
| rand 512/128 TTFT p50 ms | 97 | 281 | 745 | 926 | 1335 | 2043 | 2099 |
| rand 512/128 TTFT p99 ms | 112 | 283 | 755 | 1226 | 1971 | 3413 | 6487 |
| natural 8k out tok/s | 73.8 | 117.0 | 150.1 | 163.4 | 173.7 | 181.4 | 181.5 |
| natural 8k TPOT ms | 8.22 | 22.85 | 72.44 | 139.80 | 268.84 | 516.50 | 521.16 |
| natural 8k TTFT p50 ms | 690 | 1384 | 2692 | 2697 | 2721 | 3421 | 82062 |
| natural 8k TTFT p99 ms | 706 | 2628 | 10099 | 20053 | 39983 | 78887 | 160507 |
| mixed churn out tok/s | - | 278.1 | 557.4 | 740.6 | 1013.5 | 1353.6 | 1745.5 |
| mixed churn TTFT p50 ms | - | 192 | 269 | 346 | 412 | 579 | 850 |
| mixed churn TTFT p99 ms | - | 310 | 438 | 538 | 791 | 831 | 1222 |
| preemptions (all cells) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Cold client wall (max_tokens 1, fresh salt, median of 5; = cold TTFT 8k / 32k / 64k / 118k): 8192: 0.7143 s, 32768: 2.9222 s, 65536: 6.2002 s, 117964: 12.2857 s.

Prefix-reuse workload (frozen reuse driver, cycle, 32k docs, working set 1.6 x pool, 4 clients; prompt tok/s, TTFT p50 / p99 s, host-source tokens): cycle_p1 11,216.3 tok/s, 5.160 / 9.418 s, host 0 (142/142 ok); cycle_p2 282,816.9 tok/s, 0.278 / 0.415 s, host 4,659,872 (142/142 ok); cycle_p3 295,762.5 tok/s, 0.246 / 0.375 s, host 4,659,872 (142/142 ok). AL during the cell -, preemptions 0.

STEADY rows of this boot and of ns128 OFF were recomputed by this analysis (chain parse_error, section 4).

### 4.4 ns128 OFF (default, no tier)

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 |
|---|---:|---:|---:|---:|---:|---:|
| STEADY step ms | 8.122 | 13.218 | 22.843 | 30.888 | 40.182 | 55.599 |
| STEADY tok/s (derived) | 123.1 | 302.6 | 700.4 | 1036.0 | 1592.7 | 2302.2 |
| DEC greedy out tok/s | 116.1 | 264.7 | 540.5 | 726.9 | 980.4 | 1232.8 |
| DEC greedy TPOT ms | 7.67 | 12.20 | 21.67 | 30.28 | 42.67 | 72.18 |
| DEC greedy TTFT p50 ms | 157 | 366 | 928 | 1679 | 2762 | 3822 |
| DEC greedy TTFT p99 ms | 168 | 456 | 1384 | 2479 | 4151 | 7797 |
| DEC default out tok/s | 115.5 | 266.1 | 545.7 | 731.5 | 982.7 | 1229.2 |
| DEC default TPOT ms | 7.71 | 12.31 | 21.80 | 29.82 | 42.80 | 71.90 |
| DEC default TTFT p50 ms | 156 | 343 | 931 | 1702 | 2801 | 3854 |
| DEC default TTFT p99 ms | 168 | 434 | 1164 | 2216 | 3905 | 7930 |
| DEC top_p 0.95 out tok/s | 114.8 | 264.6 | 534.1 | 715.3 | 965.3 | 1190.3 |
| DEC top_p 0.95 TPOT ms | 7.77 | 12.42 | 22.08 | 30.06 | 43.43 | 76.16 |
| DEC top_p 0.95 TTFT p50 ms | 157 | 344 | 933 | 1735 | 2878 | 3580 |
| DEC top_p 0.95 TTFT p99 ms | 168 | 419 | 1344 | 2539 | 4168 | 8335 |
| rand 512/128 out tok/s | 114.2 | 265.0 | 538.7 | 745.1 | 1007.7 | 1303.5 |
| rand 512/128 TPOT ms | 8.06 | 13.27 | 25.01 | 35.44 | 52.45 | 81.94 |
| rand 512/128 TTFT p50 ms | 96 | 280 | 729 | 967 | 1335 | 2035 |
| rand 512/128 TTFT p99 ms | 114 | 283 | 751 | 1196 | 1958 | 3359 |
| natural 8k out tok/s | 74.3 | 117.4 | 150.7 | 163.7 | 174.0 | 181.8 |
| natural 8k TPOT ms | 8.11 | 22.73 | 72.10 | 139.48 | 268.50 | 515.41 |
| natural 8k TTFT p50 ms | 691 | 1380 | 2687 | 2693 | 2716 | 3411 |
| natural 8k TTFT p99 ms | 707 | 2625 | 10079 | 20010 | 39907 | 78720 |
| mixed churn out tok/s | - | 280.2 | 563.9 | 744.6 | 1015.5 | 1354.2 |
| mixed churn TTFT p50 ms | - | 190 | 266 | 342 | 411 | 580 |
| mixed churn TTFT p99 ms | - | 308 | 432 | 535 | 790 | 821 |
| preemptions (all cells) | 0 | 0 | 0 | 0 | 0 | 0 |

Cold client wall (max_tokens 1, fresh salt, median of 5; = cold TTFT 8k / 32k / 64k / 118k): 8192: 0.7082 s, 32768: 2.9143 s, 65536: 6.1880 s, 117964: 12.2625 s.

### 4.5 Prefix-reuse workload, tier ON vs OFF

| launch | pass | prompt tok/s | TTFT p50 s | TTFT p99 s | host-source tokens | ok/n |
|---|---:|---:|---:|---:|---:|---:|
| spec_on | cycle_p1 | 11,059.6 | 5.247 | 11.830 | 0 | 81/81 |
| spec_on | cycle_p2 | 204,306.1 | 0.256 | 0.512 | 2,658,096 | 81/81 |
| spec_on | cycle_p3 | 202,163.9 | 0.256 | 0.608 | 2,658,096 | 81/81 |
| spec_off | cycle_p1 | 11,048.6 | 5.151 | 11.840 | 0 | 81/81 |
| spec_off | cycle_p2 | 11,036.4 | 5.146 | 12.457 | 0 | 81/81 |
| spec_off | cycle_p3 | 11,049.6 | 5.217 | 11.834 | 0 | 81/81 |
| ns256_on | cycle_p1 | 11,216.3 | 5.160 | 9.418 | 0 | 142/142 |
| ns256_on | cycle_p2 | 282,816.9 | 0.278 | 0.415 | 4,659,872 | 142/142 |
| ns256_on | cycle_p3 | 295,762.5 | 0.246 | 0.375 | 4,659,872 | 142/142 |

spec ON vs OFF: pass 1 (cold prefill, 32k docs) tok/s x1.001, TTFT p50 x1.019, p99 x0.999 (the tier's store path costs nothing measurable on cold prefill; baseline 11,055 -> OFF 11,048.6, -0.06 %); pass 2 tok/s x18.51 (TTFT p50 x0.050, p99 x0.041), pass 3 x18.30 (p50 x0.049, p99 x0.051). With the OFF launch the working set (1.6 x the GPU pool) cycles through the LRU and every pass recomputes (11.04-11.05k tok/s, host 0). Host-source hit tokens in passes 2 / 3: 2,658,096 of 2,659,392 prompt tokens (99.95 %). ns256 ON (142 docs, pool basis 2,893,281): pass 1 11,216.3 tok/s (ns256 validation baseline 11,203, +0.1 %), pass 2 / 3 282,816.9 / 295,762.5 tok/s (TTFT p50 0.278 / 0.246 s), host 4,659,872 tokens per pass; no same-chain ns256 OFF reuse run exists. Cell totals (spec ON): device 183,472, host 5,499,792, other 0 cached tokens; spec OFF: device 183,472 (the shared prefix inside pass 1), host 0. Preemptions 0, swap none (ns256 ON reuse: 2187 pages swapped in, 0 out).

### 4.6 Cold walls (client wall, `max_tokens 1`)

| prompt tokens | spec ON s | spec OFF s | ns256 ON s | ns128 OFF s | spec ON - OFF ms | spec OFF - ns128 OFF ms | ns256 ON - ns128 OFF ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 0.7430 | 0.7286 | 0.7143 | 0.7082 | +14.5 | +20.3 | +6.1 |
| 32768 | 2.9749 | 2.9633 | 2.9222 | 2.9143 | +11.6 | +49.1 | +8.0 |
| 65536 | 6.2794 | 6.2657 | 6.2002 | 6.1880 | +13.7 | +77.7 | +12.2 |
| 117964 | 12.4243 | 12.4090 | 12.2857 | 12.2625 | +15.3 | +146.5 | +23.2 |

Baselines (spec_off, protocol-matched, primary): 8192 0.724 s -> 0.7286 (+0.63 %), 32768 2.951 -> 2.9633 (+0.42 %), 65536 6.251 -> 6.2657 (+0.24 %); 118k: no baseline. The stage C 'spec prompt tax' ladder (+147 / +696 / +2330 ms at 8k / 32k / 118k) was measured with another probe (chat on corpus text, drafts running); these walls use max_tokens 1 (0 drafts), so the two are not comparable and no change of the prompt tax is claimed.

## 5. Sanity against the 0.33.0 baselines and the same-chain comparisons

Rule (Amendment 3 + ruling Q10): a cell more than 3 % worse than its 0.33.0 baseline of the same launch is investigated before the README claims anything. Result: **0 PRIMARY protocol-matched
rows above 3 % (worst +1.16 %)**, 3 protocol-matched SECONDARY rows above 3 %, 2 rows that are not protocol-matched (printed beside, never a flag). The comparison text of the analysis, verbatim:

**Flags (> 3 % worse, any role):**
- MATCHED(secondary, chain flagged only primary rows): spec_off dec.c1 224.51 tok/s is 3.44 % below the 0.33.0 validation value 232.5 (v033_VALIDATION stage 2B, protocol_matched=true in baselines.json); the S1-14 NEW-arm value 224.3 (primary) is met (-0.09 %). AL explains the gap: 4.6712 (identical to the S1-14 arm's 4.671) vs 4.856 at validation (-3.8 %); 232.5 x 4.671/4.856 = 223.6 vs 224.5 measured (+0.4 %); tok/s per AL 48.06 vs 47.88. Which of 224.3 / 232.5 is the primary baseline stays the open orchestrator reading.
- MATCHED(secondary, derived): spec_on dec.c1 221.45 tok/s is 3.44 % below the row derived from the 232.5 validation value (229.34 = 232.5 x 0.9864); the derived row from 224.3 is met (221.25, -0.09 %) and S1-10R phase-4 ON absolute 221.32 is met (-0.06 %). Same AL explanation (AL-normalised baseline 220.6 vs 221.4 measured).
- MATCHED(secondary): spec_off al.c1 4.6712 is 3.81 % below the validation AL 4.856 (primary 4.671 met, -0.00 %): AL realization of the 16 replayed prompts, see the two flags above.
- NOT PROTOCOL-MATCHED (printed beside, never a flag by the rulings): dec.c1 vs the Phase 2b FB1 value 242.7 (gsm80 + mtbench80, 2 x 80 requests) is 7.5 % (spec_off) / 7.5 % (spec_on derived 239.4) worse; different prompt count / realization.

**Chain defect and coverage:**
- CHAIN DEFECT (recovered here, no data lost): 13 non-speculative STEADY cells (ns256_on c1/4/16/32/64/128/256, ns128_off c1/4/16/32/64/128) are valid=True but the chain's p2.parse_result raised TypeError (the all-null tok_per_draft list is sorted) so step_ms / out_tps were missing from matrix.txt / matrix.json and from the 58-row baseline comparison; 'FLAGS: 0' therefore never saw them. Recomputed here from the raw steady_s1_c*.json values with the chain's own estimator (upper median): 15 additional baseline rows, none above +0.67 % worse (the stale c1 ns baselines are better by 4.2-4.9 %, see below).
- COVERAGE: 29 primary protocol-matched baseline rows (+44 secondary / beside rows) exist in total. rand512, nat8k, churn, default sampling, top_p, c4 / c32 / c256 DEC, the 65k / 118k walls and every TTFT / TPOT metric have NO 0.33.0 baseline; '0 primary flags' does not mean every cell is within 3 %. For these cells the only references are the same-chain OFF boot (spec) and the cross-profile ns128 boot (ns256), tabulated below.
- STALE BASELINES (check lenient, not a regression): ns256_on dec.c1 115.19 is 4.99 % better than the derived 109.7 (ns256_off 110.6 validation stage 3 x 0.9920) and ns128_off dec.c1 116.11 is 4.89 % better than 110.7; ns256_on steady.c1 8.189 is 4.5 % better than 8.575. Stage C (116), S1-10R ON absolute (115.13 / 8.197) and the ns128 S1-14 steady (8.104) all agree with the measurement, so the effective c1 check is the beside_abs row (-0.06 % dec, -0.09 % steady).

Spec ON vs the same-chain OFF boot (reference = the OFF boot, NOT a 0.33.0 baseline; positive = ON worse):

| family | c | ON vs OFF (tok/s or step) | tok/s worse % | TPOT % | TTFT p50 % | TTFT p99 % |
|---|---:|---|---:|---:|---:|---:|
| STEADY | 1 | step ms 17.579 vs 17.301 | +1.6 |  |  |  |
| STEADY | 4 | step ms 30.394 vs 30.783 | -1.3 |  |  |  |
| STEADY | 16 | step ms 56.367 vs 55.583 | +1.4 |  |  |  |
| STEADY | 32 | step ms 77.456 vs 76.474 | +1.3 |  |  |  |
| STEADY | 64 | step ms 104.396 vs 105.314 | -0.9 |  |  |  |
| STEADY | 128 | step ms 153.926 vs 153.916 | +0.0 |  |  |  |
| dec_greedy | 1 | 221.4 vs 224.5 | +1.4 | +1.5 | -0.1 | +0.9 |
| dec_greedy | 4 | 416.5 vs 427.0 | +2.5 | +5.2 | -7.2 | +0.2 |
| dec_greedy | 16 | 756.6 vs 752.0 | -0.6 | -0.3 | -1.1 | +1.9 |
| dec_greedy | 32 | 961.9 vs 975.8 | +1.4 | +1.7 | -0.3 | -1.6 |
| dec_greedy | 64 | 1252.2 vs 1254.3 | +0.2 | +0.0 | -2.8 | +0.2 |
| dec_greedy | 128 | 1479.1 vs 1481.8 | +0.2 | -0.1 | -0.1 | +0.8 |
| dec_default | 1 | 226.3 vs 229.4 | +1.4 | +1.4 | +1.6 | +0.8 |
| dec_default | 4 | 414.7 vs 409.9 | -1.2 | -1.0 | +0.8 | +5.4 |
| dec_default | 16 | 745.0 vs 736.8 | -1.1 | -1.2 | +0.1 | +2.0 |
| dec_default | 32 | 973.0 vs 967.8 | -0.6 | -1.6 | +6.1 | +0.3 |
| dec_default | 64 | 1232.0 vs 1234.1 | +0.2 | +0.6 | +8.6 | +1.4 |
| dec_default | 128 | 1444.9 vs 1482.9 | +2.6 | +0.9 | +1.8 | +6.8 |
| dec_top_p | 1 | 229.3 vs 232.2 | +1.3 | +1.4 | -0.3 | +0.7 |
| dec_top_p | 4 | 432.8 vs 439.9 | +1.6 | +0.4 | +1.8 | +2.5 |
| dec_top_p | 16 | 739.6 vs 741.2 | +0.2 | +1.1 | +0.8 | +1.8 |
| dec_top_p | 32 | 963.7 vs 972.1 | +0.9 | +1.0 | -8.0 | +0.6 |
| dec_top_p | 64 | 1238.8 vs 1238.9 | +0.0 | -0.8 | +4.2 | +2.3 |
| dec_top_p | 128 | 1472.6 vs 1475.7 | +0.2 | +0.3 | -0.1 | +2.4 |
| rand512 | 1 | 148.5 vs 150.3 | +1.2 | +1.3 | +0.7 | -1.5 |
| rand512 | 4 | 283.1 vs 284.6 | +0.5 | +0.5 | +0.2 | +0.5 |
| rand512 | 16 | 578.5 vs 581.5 | +0.5 | +0.6 | +0.8 | +0.0 |
| rand512 | 32 | 768.3 vs 772.7 | +0.6 | +0.6 | +0.2 | +0.7 |
| rand512 | 64 | 1008.6 vs 997.8 | -1.1 | +4.2 | -7.4 | -22.9 |
| rand512 | 128 | 1272.2 vs 1269.5 | -0.2 | -0.8 | -3.2 | +0.8 |
| nat8k | 1 | 92.0 vs 93.0 | +1.0 | +1.4 | +0.5 | +1.9 |
| nat8k | 4 | 130.2 vs 130.4 | +0.2 | +0.1 | -0.0 | +0.5 |
| nat8k | 16 | 156.1 vs 156.4 | +0.2 | +0.2 | +0.4 | +0.2 |
| nat8k | 32 | 168.9 vs 169.0 | +0.1 | +0.2 | -0.1 | +0.1 |
| nat8k | 64 | 175.2 vs 175.4 | +0.1 | +0.4 | +0.0 | +0.1 |
| nat8k | 128 | 177.2 vs 177.3 | +0.1 | +0.2 | +0.0 | +0.0 |
| churn | 4 | 498.5 vs 501.6 | +0.6 |  | +0.3 | +0.6 |
| churn | 16 | 918.3 vs 924.0 | +0.6 |  | +1.3 | -0.1 |
| churn | 32 | 1201.0 vs 1206.3 | +0.4 |  | -1.0 | -10.6 |
| churn | 64 | 1567.4 vs 1572.7 | +0.3 |  | -0.7 | -11.1 |
| churn | 128 | 1944.9 vs 1922.1 | -1.2 |  | +1.6 | -27.2 |

Largest throughput deficits of ON: dec_default c128 2.6 % (single rep; greedy c128 0.2 %), dec_greedy c4 2.5 % (the known c4 mode flip), nothing above 3 %. rand512 c64 TTFT p99 -22.9 % is an outlier of the OFF boot (OFF 2704 ms vs ON 2084 ms; the OFF value is out of line with its own c32 1242 / c128 3881 ms).

Cross-profile (ns256 ON vs ns128 OFF: max-num-seqs 256 + cap 256 + tier + a swap-affected boot; there is no same-chain ns256 OFF matrix):

| family | c | ns256 ON vs ns128 OFF | tok/s worse % | TPOT % | TTFT p50 % | TTFT p99 % |
|---|---:|---|---:|---:|---:|---:|
| STEADY | 1 | step ms 8.189 vs 8.122 | +0.8 |  |  |  |
| STEADY | 4 | step ms 13.565 vs 13.218 | +2.6 |  |  |  |
| STEADY | 16 | step ms 23.176 vs 22.843 | +1.5 |  |  |  |
| STEADY | 32 | step ms 31.264 vs 30.888 | +1.2 |  |  |  |
| STEADY | 64 | step ms 40.018 vs 40.182 | -0.4 |  |  |  |
| STEADY | 128 | step ms 56.101 vs 55.599 | +0.9 |  |  |  |
| dec_greedy | 1 | 115.2 vs 116.1 | +0.8 | +0.8 | +0.3 | +0.3 |
| dec_greedy | 4 | 263.5 vs 264.7 | +0.4 | +0.6 | +0.2 | -1.9 |
| dec_greedy | 16 | 535.5 vs 540.5 | +0.9 | +1.3 | +0.7 | -0.8 |
| dec_greedy | 32 | 722.6 vs 726.9 | +0.6 | +0.9 | -2.3 | +0.9 |
| dec_greedy | 64 | 980.9 vs 980.4 | -0.1 | -1.9 | +7.9 | +1.5 |
| dec_greedy | 128 | 1212.2 vs 1232.8 | +1.7 | +1.0 | +0.1 | +6.7 |
| dec_default | 1 | 114.4 vs 115.5 | +1.0 | +1.1 | +1.3 | +0.3 |
| dec_default | 4 | 265.5 vs 266.1 | +0.2 | -0.0 | +0.5 | -0.7 |
| dec_default | 16 | 531.6 vs 545.7 | +2.6 | +1.4 | +0.4 | +19.3 |
| dec_default | 32 | 721.6 vs 731.5 | +1.4 | +0.4 | +0.4 | +12.4 |
| dec_default | 64 | 978.9 vs 982.7 | +0.4 | -0.9 | +3.1 | +6.8 |
| dec_default | 128 | 1219.6 vs 1229.2 | +0.8 | +1.4 | +0.3 | +0.6 |
| dec_top_p | 1 | 113.7 vs 114.8 | +1.0 | +1.0 | +0.7 | +0.8 |
| dec_top_p | 4 | 262.9 vs 264.6 | +0.7 | +0.5 | +0.4 | +3.0 |
| dec_top_p | 16 | 528.0 vs 534.1 | +1.1 | +0.9 | +0.4 | +3.4 |
| dec_top_p | 32 | 717.6 vs 715.3 | -0.3 | +1.4 | -2.8 | -3.8 |
| dec_top_p | 64 | 963.6 vs 965.3 | +0.2 | +0.4 | +2.6 | +0.1 |
| dec_top_p | 128 | 1190.5 vs 1190.3 | -0.0 | -0.9 | +4.5 | -1.7 |
| rand512 | 1 | 113.3 vs 114.2 | +0.8 | +0.8 | +0.5 | -1.5 |
| rand512 | 4 | 263.3 vs 265.0 | +0.7 | +0.7 | +0.3 | +0.2 |
| rand512 | 16 | 534.3 vs 538.7 | +0.8 | +0.9 | +2.1 | +0.4 |
| rand512 | 32 | 736.3 vs 745.1 | +1.2 | +1.8 | -4.3 | +2.5 |
| rand512 | 64 | 1011.1 vs 1007.7 | -0.3 | -0.5 | +0.0 | +0.7 |
| rand512 | 128 | 1295.0 vs 1303.5 | +0.7 | +0.7 | +0.4 | +1.6 |
| nat8k | 1 | 73.8 vs 74.3 | +0.7 | +1.2 | -0.2 | -0.1 |
| nat8k | 4 | 117.0 vs 117.4 | +0.4 | +0.5 | +0.3 | +0.1 |
| nat8k | 16 | 150.1 vs 150.7 | +0.4 | +0.5 | +0.2 | +0.2 |
| nat8k | 32 | 163.4 vs 163.7 | +0.2 | +0.2 | +0.1 | +0.2 |
| nat8k | 64 | 173.7 vs 174.0 | +0.1 | +0.1 | +0.2 | +0.2 |
| nat8k | 128 | 181.4 vs 181.8 | +0.2 | +0.2 | +0.3 | +0.2 |
| churn | 4 | 278.1 vs 280.2 | +0.8 |  | +0.9 | +0.7 |
| churn | 16 | 557.4 vs 563.9 | +1.2 |  | +1.0 | +1.6 |
| churn | 32 | 740.6 vs 744.6 | +0.5 |  | +1.1 | +0.6 |
| churn | 64 | 1013.5 vs 1015.5 | +0.2 |  | +0.4 | +0.1 |
| churn | 128 | 1353.6 vs 1354.2 | +0.1 |  | -0.3 | +1.2 |

The ns256 bound-TTFT deviation of the 0.32.0 study (9-12 %, never re-gated; stated known limit) is not visible as a general shift here: median TTFT of ns256 ON vs ns128 OFF is within -4.3..+7.9 % and p99 mostly within +-4 % (see the table; the exceptions are single-run dec_default / dec_greedy cells); this chain does not re-gate it (cross-profile, single runs, different quantity).

### 5.1 Against the campaign start (r2fin) and stage C

Neither reference is protocol-matched (older tree, other prompts); the rows are context, never flags.

| launch | c | ours | r2fin_r_topo | vs r2fin % | r2fin NCCL-only | r2fin mns256 cap256 | vs % | S1-6 end s16R2E2 | vs S1-6 end % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ns128_off | 1 | 114.2 | 96.0 | +18.9 | 75.0 | - | - | 111.0 | +2.9 |
| ns128_off | 4 | 265.0 | 234.0 | +13.3 | 190.0 | - | - | 254.9 | +4.0 |
| ns128_off | 16 | 538.7 | 487.6 | +10.5 | 417.0 | - | - | 528.1 | +2.0 |
| ns128_off | 64 | 1007.7 | 906.3 | +11.2 | 753.0 | - | - | 986.1 | +2.2 |
| ns128_off | 128 | 1303.5 | 1159.3 | +12.4 | 957.0 | - | - | 1273.3 | +2.4 |
| ns256_on | 1 | 113.3 | 96.0 | +18.0 | 75.0 | - | - | 111.0 | +2.1 |
| ns256_on | 4 | 263.3 | 234.0 | +12.5 | 190.0 | - | - | 254.9 | +3.3 |
| ns256_on | 16 | 534.3 | 487.6 | +9.6 | 417.0 | - | - | 528.1 | +1.2 |
| ns256_on | 64 | 1011.1 | 906.3 | +11.6 | 753.0 | - | - | 986.1 | +2.5 |
| ns256_on | 128 | 1295.0 | 1159.3 | +11.7 | 957.0 | - | - | 1273.3 | +1.7 |
| ns256_on | 256 | 1543.8 | - | - | - | 1404.0 | +10.0 | - | - |

- ns128 / ns256 are +9.6..+18.9 % above the r2fin sweep at c1-c128 (+10.0 % at c256 vs the r2fin mns256 row) and +1.2..+4.0 % above the S1-6 end production row (111.0 / 254.9 / 528.1 / 986.1 / 1273.3), i.e. nothing of the campaign gain was lost by the 0.33.0 line, the launch scripts or the tier.
- Natural 8192-token prefill (nat8k c1 TTFT p50): ns128 OFF 691 ms, ns256 ON 690 ms, spec OFF 714 ms, spec ON 717 ms vs the r2fin profile chunk 793.8 ms (-12.9 / -13.1 / -10.1 / -9.6 %) and the S1-6 round-3 profile 704.4 ms (-1.9 / -2.1 / +1.4 / +1.8 %; a TTFT includes tokenization / scheduling, the references are the profiled chunk, approximate).
- Non-spec decode step (STEADY, constant batch): ns128 OFF c1 / c64 / c128 = 8.122 / 40.182 / 55.599 ms, ns256 ON 8.189 / 40.018 / 56.101 ms; profile references r2fin c1 9.571 ms, S1-6 round 3 c1 8.325 (random) / 8.086 (real gsm), c64 real plateau 39.15, c64 random 41.37, c128 random 57.18 ms (profiles, other protocol).

| comparison | c | tok/s stage C / ours | % | TPOT ms stage C / ours | % | TTFT p50 ms stage C / ours | % | AL stage C / ours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spec ON vs stage C RadixArk K=7 | 1 | 219 / 221.4 | +1.1 | 3.63 / 3.49 | -3.9 | 152 / 154 | +1.5 | 4.738 / 4.671 |
| spec ON vs stage C RadixArk K=7 | 4 | 412 / 416.5 | +1.1 | 7.66 / 7.30 | -4.7 | 228 / 216 | -5.1 | 4.556 / 4.639 |
| spec ON vs stage C RadixArk K=7 | 16 | 758 / 756.6 | -0.2 | 15.34 / 15.19 | -1.0 | 444 / 462 | +4.1 | 4.766 / 4.872 |
| spec ON vs stage C RadixArk K=7 | 32 | 958 / 961.9 | +0.4 | 23.24 / 23.24 | +0.0 | 679 / 644 | -5.1 | 4.938 / 4.880 |
| spec ON vs stage C RadixArk K=7 | 64 | 1138 / 1252.2 | +10.0 | 40.93 / 36.63 | -10.5 | 1311 / 1033 | -21.2 | 5.115 / 4.830 |
| spec ON vs stage C RadixArk K=7 | 128 | 1212 / 1479.1 | +22.0 | 69.75 / 61.87 | -11.3 | 3154 / 1746 | -44.6 | 5.188 / 4.841 |
| spec OFF vs stage C RadixArk K=7 | 1 | 219 / 224.5 | +2.5 | 3.63 / 3.43 | -5.4 | 152 / 154 | +1.6 | 4.738 / 4.671 |
| spec OFF vs stage C RadixArk K=7 | 4 | 412 / 427.0 | +3.6 | 7.66 / 6.94 | -9.4 | 228 / 233 | +2.3 | 4.556 / 4.710 |
| spec OFF vs stage C RadixArk K=7 | 16 | 758 / 752.0 | -0.8 | 15.34 / 15.23 | -0.7 | 444 / 467 | +5.3 | 4.766 / 4.791 |
| spec OFF vs stage C RadixArk K=7 | 32 | 958 / 975.8 | +1.9 | 23.24 / 22.86 | -1.6 | 679 / 646 | -4.8 | 4.938 / 4.946 |
| spec OFF vs stage C RadixArk K=7 | 64 | 1138 / 1254.3 | +10.2 | 40.93 / 36.62 | -10.5 | 1311 / 1063 | -18.9 | 5.115 / 4.838 |
| spec OFF vs stage C RadixArk K=7 | 128 | 1212 / 1481.8 | +22.3 | 69.75 / 61.93 | -11.2 | 3154 / 1748 | -44.6 | 5.188 / 4.839 |
| ns128 OFF vs stage C ns128 | 1 | 116 / 116.1 | +0.1 | 7.77 / 7.67 | -1.3 | 140 / 157 | +12.3 | - |
| ns128 OFF vs stage C ns128 | 4 | 271 / 264.7 | -2.3 | 12.38 / 12.20 | -1.4 | 304 / 366 | +20.3 | - |
| ns128 OFF vs stage C ns128 | 16 | 559 / 540.5 | -3.3 | 21.70 / 21.67 | -0.1 | 794 / 928 | +16.8 | - |
| ns128 OFF vs stage C ns128 | 32 | 761 / 726.9 | -4.5 | 29.18 / 30.28 | +3.8 | 1458 / 1679 | +15.1 | - |
| ns128 OFF vs stage C ns128 | 64 | 954 / 980.4 | +2.8 | 41.43 / 42.67 | +3.0 | 3344 / 2762 | -17.4 | - |
| ns128 OFF vs stage C ns128 | 128 | 1064 / 1232.8 | +15.9 | 70.82 / 72.18 | +1.9 | 5617 / 3822 | -32.0 | - |
| ns256 ON vs stage C ns256 | 1 | 116 / 115.2 | -0.7 | 7.79 / 7.73 | -0.8 | 141 / 158 | +11.8 | - |
| ns256 ON vs stage C ns256 | 16 | 559 / 535.5 | -4.2 | 21.68 / 21.95 | +1.2 | 795 / 935 | +17.6 | - |
| ns256 ON vs stage C ns256 | 64 | 955 / 980.9 | +2.7 | 42.00 / 41.86 | -0.3 | 3157 / 2980 | -5.6 | - |
| ns256 ON vs stage C ns256 | 128 | 1061 / 1212.2 | +14.3 | 71.37 / 72.89 | +2.1 | 5416 / 3824 | -29.4 | - |
| ns256 ON vs stage C ns256 | 256 | 1119 / 1432.2 | +28.0 | 120.65 / 133.05 | +10.3 | 10921 / 4072 | -62.7 | - |
| ns256 ON vs stage C ns128 (c4 / c32 have no ns256 row) | 1 | 116 / 115.2 | -0.7 | 7.77 / 7.73 | -0.5 | 140 / 158 | +12.6 | - |
| ns256 ON vs stage C ns128 (c4 / c32 have no ns256 row) | 4 | 271 / 263.5 | -2.8 | 12.38 / 12.28 | -0.8 | 304 / 366 | +20.5 | - |
| ns256 ON vs stage C ns128 (c4 / c32 have no ns256 row) | 16 | 559 / 535.5 | -4.2 | 21.70 / 21.95 | +1.1 | 794 / 935 | +17.7 | - |
| ns256 ON vs stage C ns128 (c4 / c32 have no ns256 row) | 32 | 761 / 722.6 | -5.0 | 29.18 / 30.54 | +4.7 | 1458 / 1640 | +12.5 | - |
| ns256 ON vs stage C ns128 (c4 / c32 have no ns256 row) | 64 | 954 / 980.9 | +2.8 | 41.43 / 41.86 | +1.0 | 3344 / 2980 | -10.9 | - |
| ns256 ON vs stage C ns128 (c4 / c32 have no ns256 row) | 128 | 1064 / 1212.2 | +13.9 | 70.82 / 72.89 | +2.9 | 5617 / 3824 | -31.9 | - |

Reading: speculative c1-c32 reproduces stage C at tok/s within -0.8..+3.6 % (c1 +1.1 / +2.5 %, AL 4.671 vs 4.738), c64 / c128 are +10 % / +22 % (tree and host changes since 0ffa57b714; AL is 5-7 % lower than in stage C there, so the gain is in the step). Non-spec ns128 / ns256 are 2.3-5.0 % below stage C at c4 / c16 / c32 and 13.9-28 % above at c128 / c256; the protocol-matched 0.33.0 DEC baselines are met (no worse than +0.56 %, c1 aside; section 4), so the c16 / c32 gap is the prompt mix. (ns256 c256: tok/s +28 % but TPOT +10 %: TPOT is a per-request mean in a different request mix.) TTFT (median) is 12-20 % higher than stage C at c1-c32 for non-spec and 5-32 % lower at c64-c128 (62.7 % at ns256 c256): different request count and prompts.

## 6. Gates

All nine gates PASS (FAIL 0, ARTIFACT 0, NA 0). `spec_off` and `ns128_off` have no gate of their own (`spec_off` is the reference boot of G-off/spec, `ns256_off` the OFF boot of G-off/ns256).

- G-det/spec_on: PASS - two cold processes, 12 request files + 12 per-rank hash files identical (789 requests, 3072 logprob values; 8 ranks one hash)
- G-det/ns256_on: PASS - same, 24 files identical (789 requests, 3072 logprob values; 8 ranks one hash)
- G-gsm/spec_on greedy: PASS - 389/400 = 97.25 % (passes 194 + 195; band 382..392)
- G-gsm/spec_on default sampling: PASS - 391/400 = 97.75 % (195 + 196; 1 below the band top 392)
- G-gsm/ns256_on greedy: PASS - 386/400 = 96.50 % (194 + 192)
- G-long/spec_on: PASS - 6/6 needles exact cold and 6/6 exact with a tier hit (32k x3, 118k x3; tier host fraction >= 0.9995)
- G-long/ns256_on: PASS (NR1: dedicated fresh boot ns256_long) - 6/6 exact cold, 6/6 exact with a tier hit (host fraction >= 0.9995)
- G-off/spec_on: PASS - D4' leg PASS (KV blocks tier == GPU bitwise: 59,100 blocks, xc_bad 0, hx_bad 0; 170/170 (b) checks) AND server leg PASS (6 lengths 4k..118k: tier-hit == GPU-hit == MUSE_OFFLOAD=0, restore after natural eviction == ON GPU-hit == OFF direct; tokens + 5 top-logprob values bitwise, 32 tokens)
- G-off/ns256_on: PASS (NR1: dedicated fresh boots ns256_goff / ns256_off, 404 start reset accepted with counters-at-0 evidence) - D4' leg PASS (15,356 blocks, xc_bad 0, hx_bad 0; 170/170) AND server leg PASS (6 lengths; restore after eviction == GPU-hit == OFF)
- spec_off and ns128_off: no gate defined (spec_off is the reference boot of G-off/spec; ns256_off is the OFF boot of G-off/ns256).

**Stated deviations of the gates (the claims are exactly as strong as this list allows):**

- **Q2: offline in-process G-det and D4'.** G-det and the D4' leg ran as offline in-process `LLM` objects bound to the serve line (environment equal to the launch dry run, the `E2E_*` settings from the
  serve line), not through the launch scripts. G-det compares two cold processes (each its own session, weight load and pid, run one after the other) pass by pass.
- **Q5: natural flood eviction only; the full-tier regime is not re-covered.** "Restore after eviction" is the natural GPU eviction by a flood of 1.3 x the pool (spec 66 documents / 192 s, ns256 115
  documents / 330 s); the regime in which the TIER itself evicts (tier bytes against the original KV) is not re-covered. The S1-10R gate T note stays: it checks that blocks restored from a full
  tier serve the same output as the GPU copy right after the restore, not tier bytes against the originally computed KV under eviction; D4' covers byte equality in the non-full regime only.
- **M3: the OFF / eviction leg is new, unfrozen code** (`goff_driver.py`); the frozen D4' tools are used unchanged for the D4' leg.
- **NR1: the ns256 gates on dedicated fresh boots.** G-off/ns256 and G-long/ns256 were decided on the dedicated fresh boots `ns256_goff`, `ns256_off` and `ns256_long`; the start reset answered 404
  (no dev endpoint on the ns lines, which must stay byte-equal to the gated lines) and the cached-token counters and `prompt_tokens_total` were 0 before the first request.
- The G-off/ns256 verdict was obtained on a swapped boot (`ns256_goff`, 4,570.6 MiB used at READY).

### 6.1 G-det

| profile | files compared | requests (token ids + num_cached_tokens) | logprob values | rank-hash files | result | OOM lines | bind / connector workers |
|---|---:|---:|---:|---:|---:|---:|---:|
| spec | 24 | 789 | 3072 | 12 | identical | 0 | 8 / 8 |
| ns256 | 24 | 789 | 3072 | 12 | identical | 0 | 8 / 8 |

Each process runs 4 workloads (greedy, seeded t1k64, mixed seeded/greedy/top_p/penalty, lp with logprobs=5) x 3 states (C cold, H GPU hit, X tier hit after a GPU-only reset) over the 85-prompt set; the 8 per-rank sampled-token hashes are identical across the ranks and across the two processes. Information, not a criterion (inside ONE process, token ids equal / requests): spec: greedy C=H 15/85, H=X 72/85, lp C=H 7/8, H=X 8/8, mixed C=H 14/85, H=X 68/85, t1k64 C=H 15/85, H=X 37/85; ns256: greedy C=H 15/85, H=X 81/85, lp C=H 7/8, H=X 8/8, mixed C=H 14/85, H=X 80/85, t1k64 C=H 13/85, H=X 81/85. C vs H differ by design (driver docstring of e2e_g11.py: greedy output depends on the cached length and the chunk shapes; a recompute is a different computation from a hit). H and X have the SAME hit length in all 85 requests (spec greedy / t1k64 checked) and still differ in part of the batched passes; this analysis did not find the cause (batched scheduling / chunk shapes of the 85-request passes are the candidate). It is not a gate criterion (G-det compares process a with process b pass by pass); KV-block bit-exactness of the tier restore is covered by D4'(a) below and the single-stream server leg; batched tier-hit outputs are therefore NOT claimed bitwise equal to GPU-hit outputs.

### 6.2 G-off

| profile | D4' leg | blocks tier==GPU | block x rank comparisons | xc_bad / hx_bad | D4'(b) checks | negative control | server leg | lengths | tokens | logprob values compared | restore after eviction (ON E) | flood (1.3 x pool) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec | PASS | 59,100 | 472,800 | 0 / 0 | 170 of 170 | OK | PASS | 6 | 32 | 6,762 | all 6 from tier, device 0 | 66 docs / 192 s |
| ns256 | PASS | 15,356 | 122,848 | 0 / 0 | 170 of 170 | OK | PASS | 6 | 32 | 6,726 | all 6 from tier, device 0 | 115 docs / 330 s |

Server leg checks per length (4096, 8192, 16384, 32768, 65536, 117964; 32 greedy tokens + token logprob + top-5 alternatives): ON H == OFF H, ON E == ON H, ON E == OFF H (direct), ON C == OFF C, OFF E == OFF C (the OFF launch recomputes, cached 0), and on spec ON X == ON H (GPU-only reset, tier hit). ON C/H/E cached-token counts: C 0, H prompt minus the last block remainder (12-16 tokens) from the device, E (after the flood) the same count from the HOST source with device 0 = the GPU copy was evicted and the tier restored it. All checks equal at all 6 lengths on both profiles; no response without logprobs (no NOT_COMPARABLE). 

Notes from the independent verification (not shown in the table above): the block counts (59,100 / 15,356 blocks, `xc_bad` / `hx_bad`) are the frozen checker's verdict detail, re-read and summed,
not re-derived from KV hashes; in D4' (b) 4 of the 170 items (index 15 and 59, greedy and `t1k64`) have `cached=0` in C, H and X on both profiles, so "170 / 170 equal" holds but 166 of 170 are
hit-state comparisons; the spec D4' verdict file carries one warning (`(a) draft KV groups [14]: restored blocks per group {14: 2324}`), no fail or artifact reason.

### 6.3 G-long


**spec** (reset at start: HTTP 200 {"success":true}; flood 66 docs / 192 s)

| # | prompt tokens | needle offset | answer | cold reply | cold | cold wall s | tier reply | tier hit | tier wall s | host tokens | host fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31216 | 8100 | 2442873 | 2442873 | exact | 3.3 | 2442873 | exact | 0.8 | 31,344 | 0.9995 |
| 1 | 31628 | 16300 | 1921202 | 1921202 | exact | 3.4 | 1921202 | exact | 0.7 | 31,760 | 0.9996 |
| 2 | 32040 | 24700 | 5445337 | 5445337 | exact | 3.3 | 5445337 | exact | 0.9 | 32,176 | 0.9998 |
| 3 | 116414 | 8100 | 9125934 | 9125934 | exact | 12.7 | 9125934 | exact | 0.9 | 116,544 | 0.9999 |
| 4 | 117421 | 60000 | 8607467 | 8607467 | exact | 12.8 | 8607467 | exact | 1.2 | 117,552 | 0.9999 |
| 5 | 117836 | 100000 | 1709533 | 1709533 | exact | 12.7 | 1709533 | exact | 0.9 | 117,968 | 0.9999 |

**ns256** (reset at start: HTTP 404 {"detail":"Not Found"}; flood 115 docs / 330 s)

| # | prompt tokens | needle offset | answer | cold reply | cold | cold wall s | tier reply | tier hit | tier wall s | host tokens | host fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31216 | 8100 | 2442873 | 2442873 | exact | 4.1 | 2442873 | exact | 1.2 | 31,344 | 0.9995 |
| 1 | 31628 | 16300 | 1921202 | 1921202 | exact | 4.0 | 1921202 | exact | 1.4 | 31,760 | 0.9996 |
| 2 | 32040 | 24700 | 5445337 | 5445337 | exact | 3.7 | 5445337 | exact | 1.2 | 32,176 | 0.9998 |
| 3 | 116414 | 8100 | 9125934 | 9125934 | exact | 12.6 | 9125934 | exact | 1.0 | 116,544 | 0.9999 |
| 4 | 117421 | 60000 | 8607467 | 8607467 | exact | 12.6 | 8607467 | exact | 1.7 | 117,552 | 0.9999 |
| 5 | 117836 | 100000 | 1709533 | 1709533 | exact | 12.9 | 1709533 | exact | 1.2 | 117,968 | 0.9999 |


### 6.4 G-gsm

| run | pass A | pass B | run total | verdict | truncated at 16384 | mean out tokens A / B | AL (cell) | A vs B agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spec/gsm_greedy | 194/200 (97.0 %) | 195/200 (97.5 %) | 389/400 = 97.25 % | PASS | 1 + 3 | 1063 / 1241 | 4.916 | AGREE A vs B: same_final_answer 191/200 same_correctness 191/200 ident |
| spec/gsm_default | 195/200 (97.5 %) | 196/200 (98.0 %) | 391/400 = 97.75 % | PASS | 0 + 0 | 841 / 886 | 4.468 | AGREE A vs B: same_final_answer 197/200 same_correctness 197/200 ident |
| ns256/gsm_greedy | 194/200 (97.0 %) | 192/200 (96.0 %) | 386/400 = 96.50 % | PASS | 2 + 4 | 1173 / 1334 | - | AGREE A vs B: same_final_answer 190/200 same_correctness 190/200 ident |

GSM8K counts a request that hits the 16,384-token output limit as wrong (no final answer): the "truncated at 16384" column (spec greedy 1 + 3, ns256 greedy 2 + 4 requests, default sampling 0 + 0) is
included in the totals above.

## 7. Known limits

- **DRAM read ceiling.** The measured cold DRAM read ceiling is 1.537 TB/s = 96 % of ncu's 1.596 TB/s peak at the only supported memory clock (12481 MHz), not the 1.79 TB/s nominal figure; S1-6 L3 was closed on
  this ceiling. L2-resident data can exceed it.
- **Link-bound PCIe collectives and the IB-lane PARK.** There is no NVLink. With the bf16 exchange every large collective is bound by each GPU's own x16 link, not by the cross-socket cut:
  isolated all-reduce and reduce-scatter kernels measure 0.91-1.04 x their link floor (S1-12; the in-prefill reduce-scatter runs at 1.12-1.19 x T_link, skew / contention), so a NIC lane (ConnectX-7) would move the same bytes over the same link: the IB-lane lever is PARKED FINAL
  (S1-12, S1-16: redirecting the cut stores to same-socket peers makes the large AG / RS slower at every size). The large all-gather runs at 1.17-1.21 x its link floor; the excess is internal to the
  AG kernel (two-hop relay, per-range flags, chunk pipelining), the reduce-scatter with the same cut share runs at 0.96-0.98 x. That AG relay is parked below.
- **EP straggler.** The expert-parallel straggler levers (R5 / R11) were ranked below 1.6 % ("not worth it or no mechanism") and are not worked on; the EP8 layout replicates tokens and has no all2all.
- **Swap with the tier ON.** At P* the pinned-tier allocation can swap unpinned worker / EngineCore / API-server anonymous memory to the 8 GiB swap file while the server starts. ns256 ON swapped at READY
  in 2 of 3 S1-11 boots (4,095.3 and 4,570.6 MiB used; peaks 5,429.6 and 6,677.8 MiB), spec ON in 0 of 1 (boots only: of the 6 offline G-det / D4' processes one, G-det spec ON process b, swapped up to 2,997 MiB and released it at exit, the other five stayed at 64-71 MiB; PHASE2_RESULTS does not list the offline processes' swap). In the S1-10R gate 14 of 18 ON-like boots / processes swapped (growth 3.8-7.4 GiB from launch
  to READY; peak 7,444.8 of 8,192 MiB in `s110r_spec_ON_1`), none of the 16 OFF / CTL-like boots did. Over all 22 large-tier ON boots of 2026-11-11 / 12 (gate chain, det / D4' boots and the adoption series)
  14 swapped, 8 did not; none of 17 no-tier boots swapped. **No predictor was found** (`muse_wf/s1-11/swap_predictor.json`, no GPU, 39 boots; the 4 ON boots of the S1-11 matrix are not among the 39 and the predictor was not re-run with them; together 16 of 26 large-tier ON boots swapped): the preflight margins A_N / LEFT, the node memory before
  the launch (free, file, slab), the tier profile and the previous boot do not separate swappers from non-swappers; all 3 boots that started 50 minutes or more after the previous teardown did not swap (3 of the 8 non-swappers), not significant after correction (Bonferroni 0.23), and 5 of the 8 non-swappers followed a boot by 3-41 s. Treat swap at READY as likely, check it per boot (`status.txt` has the swap
  record); the cells are not voided. Swap onset is 56-79 s after the launch, when the tier is allocated.
- **c128 p99 TTFT tail with the tier.** S1-10R gate: dec c128 p99 TTFT is 3.5-7.7 % higher ON than OFF in all six groups (spec mean x1.052, ns256 mean x1.051; systematic, informational). This matrix, single
  runs: spec ON vs OFF c128 p99 TTFT +0.8 % (greedy) / +6.8 % (default) / +2.4 % (top_p), nat8k 0.0 %, mixed churn c128 -27 %.
  The tier-cost statement elsewhere covers throughput only. Same-chain single-run TTFT deltas of spec ON vs OFF that are not attributed (a single run cannot separate the tier from run-to-run noise):
  dec_default p50 up to +8.6 % (c64; c32 +6.1 %), p99 +5.4 % (c4) and +6.8 % (c128); top_p c64 p50 +4.2 %.
- **ns256 bound-TTFT deviation, never re-gated.** The 0.32.0 study measured a 9-12 % slower median reuse TTFT for the NUMA-bound ns256 against the unbound one; it was not re-gated. In this matrix the
  median TTFT of ns256 ON against ns128 OFF is within -4.3..+7.9 % (cross-profile, single runs, a different quantity). `MUSE_OFFLOAD_NUMA_BIND=0` switches the binding off.
- **Batched tier-hit outputs are not bitwise equal to GPU-hit outputs for every request.** Inside one process the H (GPU hit) and X (tier hit after a GPU-only reset) passes have the same hit length in all
  85 requests of the workloads checked (spec greedy / t1k64) and still differ for part of the batched passes (requests with H = X: spec greedy 72 / 85, spec t1k64 37 / 85, spec mixed 68 / 85; ns256 greedy 81 / 85, mixed 80 / 85, t1k64 81 / 85; the cause was not found, batched scheduling / chunk shapes are the candidate). The evidence that
  holds: the single-stream tier restore is bitwise (D4' KV block bit-exactness, tokens and logprob values of the server leg) and two cold processes are identical (G-det). A recompute (C) differs from
  a hit (H) by design (the output depends on the cached length and the chunk shapes).
- **Per-configuration offset floor of about 1.5-2.5 %.** Each configuration (serve line, environment, layout) carries a deterministic offset of 0.4-0.6 % SD (max 1.3 %) at dispatch-identical cells; a
  single-configuration A/B effect below about 1.5-2.5 % is not attributable to the mechanism under test without layout randomization. The floor is inferred from two cells (ns c1 and c4) and is not a
  measured constant. The S1-10R STEADY c1 pass (0.9859 against the 0.985 threshold, margin 0.0009) sits inside it.
- **GSM8K counts 16,384-token truncations as wrong** (section 6.4), so a long-generation run scores lower than its answers alone.
- **Reproducibility of the cells.** Three protocol-matched secondary rows are more than 3 % below a 0.33.0 value (spec dec.c1 3.44 %, twice, and the spec_off acceptance length at c1 3.81 %); the AL of the
  16 replayed prompts explains them (4.671 against 4.856 at validation; 232.5 x 4.671 / 4.856 = 223.6 against 224.5 measured; which of 224.3 / 232.5 is the primary spec c1 baseline stays an open orchestrator reading, with 232.5 the spec c1 DEC rows are 3.44 % worse). 29 primary protocol-matched baseline rows exist; rand512, nat8k,
  churn, default sampling, top_p, c4 / c32 / c256 DEC, the 65k / 118k walls and every TTFT / TPOT metric have NO 0.33.0 baseline, so "0 primary flags" does not mean every cell is within 3 %.
  13 non-speculative STEADY cells were unparsed by the chain (`TypeError` in `parse_result`) and recomputed from the raw per-window values.

## 8. Parked levers and their triggers

Only triggers that a source states are given; nothing is invented.

| lever | state | trigger or restart condition | where |
|---|---|---|---|
| S1-17 fused-boundary dispatch rows per kind (open levers: (1) both T = 8 kinds two-step in ns, Q-like, ns c8 -1.86 % at the floor; (2) RMS 24 at T = 17..24, spec raw c3 -1.4..-2.1 %; (3) AG 16, out of class; (4) AR 3 at T = 4, not resolvable) | PARKED 2026-11-12, production keeps 8 / 8 / 16 | each needs a protocol that beats the per-configuration floor (several layout draws per configuration) or an effect above about 2.5 %; lever (1) is the best supported (S1-14 F1 turned both T = 8 kinds ON and saw ns c8 +1.20 %, stage 1 turned them OFF and saw -1.86 %) | host branch `s117-bndcap` `e7491668cb`, unmerged (per-kind knobs, init agreement, tests); `muse_wf/s1-17/`, spec Addendum E |
| S1-16 all-gather relay (the AG's 1.17-1.21 x excess over its link floor, internal to the kernel) | ranked, not started; ceiling C about 3.9 % prefill / 2.1 % c64 / 3.0 % c128, expected gain E about 1.9 / 1.05 / 1.5 % (below the 2 % line) | none stated | `muse_wf/s1-16/VERDICT.md`; bench-only branch `s116-redirect` `e9cc9926ee`, unmerged |
| S1-15 more cut CTAs for the large all-gather | PARKED (more cut CTAs are slower in isolation at every production shard) | none stated | branch `s115-agcut` `a043642133`, unmerged |
| R1 out-of-class verification without the full gather | PARKED | first item if more than about 25 % of production requests are out of class (top_p > 0.95, T > 1, penalties, top_k outside 40..64); the handover gives 7.4 % (c64) / 10.3 % (c128) for all-out-of-class traffic | handover "Parked" |
| R3 BF16 layers 0 and 61 -> FP8 / NVFP4 | PARKED (changes the model) | needs a USER policy decision; c64 -4.6 %, c128 -4.2 %, +4.6 GiB KV per rank; a drift study only on approval | handover "Parked" |
| R2a | REJECTED (S1-9 L10: real +0.67 %, KV -3.64 %, c1 -1.79 % unexplained) | none (rejected); the branches are kept as evidence | handover; vLLM `muse/s19-r2a` `58b0a5fde9` and `muse/s19-r2a-squashed` `a513b77b6f`, host branch `s19-r2a` |
| A1b capture list 384 (cap 384, sizes 272..384 step 16) | PARKED (STEADY c33..c47 +11.6..+15.2 % but real c36..c48 only +1.7..+3.3 %, pooled +2.55 % below the pre-registered 3.96 %; KV -1.96 %) | restart if KV is abundant | handover "Parked"; `docs/s19_spec_launch.md` |

The IB lane (ConnectX-7 cross-socket link) is PARKED FINAL by S1-12 and S1-16 (section 7): not a parked lever with a trigger.

## 9. Notes on this file

- The independent verification (`phase2_verify.json`) found no substantive discrepancy and these minor notes, all applied above: the swap values are MiB (the analysis text divided MiB by 1000 in one
  place); the percentile definitions differ per driver; 166 of 170 D4' (b) items are hit-state comparisons; one D4' verdict warning (draft KV group 14); the reuse working set is 1.61-1.62 x the own pool;
  the D4' block counts are the frozen checker's verdict detail.
- The previous README claims "tested only on v0.32.0" and "throughput of the final defaults not measured" were replaced by this file and README section 4.8.
