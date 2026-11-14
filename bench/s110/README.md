# S1-10 measurement tools (NUMA-local KV CPU offload)

The scripts of the S1-10 study (`docs/s110_numa_offload.md`), copied from the server scratch `/raid/vllm-runtime/work/s1-10/tools` (they use those paths and the harness `/raid/vllm-runtime/work/sd/harness`):

| file | what |
|---|---|
| `master_s110.sh` | one GPU-lock hold over a RUNS file: boots (profile spec / ns256, arm OFF / SIM / NAT x bound / not), node-memory sampler, watchdog (memory floor per node, oom_score_adj 1000), post checks; `xboot=1` runs an S1-7 stress boot |
| `job_s110.sh` | the job inside a boot: facts (binding, layout, worker env, node snapshot), decode sweep (salted real prompts), reuse cycle, chat, smoke (determinism pairs) |
| `reuse_driver.py` | token-id prompt driver: cycle (documents + 64k / 118k re-admission), chat, smoke; metric deltas per pass, logprob arrays per request |
| `numa_snap.sh`, `numa_maps_summary.py`, `region_nodes.py`, `sample_nodes.sh` | per-node memory, per-worker numa_maps, owner check of the native region, 2 s node / vmstat sampler |
| `analyze_s110.py`, `compare_s110.py`, `pairs_s110.py`, `cross_s110.py`, `chat_table_s110.py`, `sizing_table_s110.py` | result tables (adoption-rule metrics, determinism pairs, per-turn chat, sizing) |
| `exact_boot_s110.sh`, `sitecustomize_s110.py`, `snap_diff.py`, `gpu_overhead_probe.py` | S1-7 stress boot with extra serve args, allocator probe + memory-history snapshot, snapshot diff, host-register / stream GPU-memory probe |
| `pinned_bw.py`, `microbench.sh` | pinned-memory bandwidth microbench (local / remote node) |
| `chk_driver.py`, `chk_worker.py` | the final-gate KV byte / single-stream checks of amendment 3 (D4'), `CHK=d4`: verdict JSON PASS / FAIL / ARTIFACT incl. the mandatory negative control; the S1-10R Phase 2 investigation modes `CHK=spec|ns256` are kept (see `docs/s110_numa_offload.md` section 10; logic test `bench/test_chk_d4_verdict.py`) |
| `rules/` | the pre-registered adoption rule and its three amendments (md5 recorded before the boots they govern), the final rule `ADOPTION_RULE_FINAL.txt` with `_AMENDMENT1..4` (amendment 3 = D4', amendment 4 = swap record, both POST-HOC, 2026-11-11) |

Usage: `setsid nohup flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock env TREE=<vllm tree> bash master_s110.sh <runs_file> &` (never take the lock twice; the header of `master_s110.sh` documents the RUNS line).
