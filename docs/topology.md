# Topology of the 8x RTX PRO 6000 host (Supermicro AS-5126GS-TNRT2, 10.19.209.74)

Measured 2026-10-23 with the scripts in `/raid/vllm-runtime/musetest/topo/` on the server (kept there, not in this folder:
`p2p_ext.cu`, `topo_bench.py` with the stages `info link matrix sweep latency conc conc2 multi multi2 rule uplink host cechunk
hier a2a a2a_rounds rd`, `locked.sh`, `stages_body.sh`, and the nccl-tests wrappers `nccl_body.sh`, `nccl_ll_body.sh`,
`nccl_a2a_body.sh`; results in `.../topo/results/` and `.../topo/logs/`). Units: GB/s = 1e9 B/s, latencies in microseconds,
256 MiB buffers, median of 3 unless noted. No clock or power-limit change was made (GPUs ran at ~2.3 GHz, 140-150 W during traffic).
Everything here is what the code in `muse_topo/` is built on: `muse_topo/topology.py` derives the same layout from sysfs at run
time and refuses to activate when it does not match.

## 1. Hardware and layout

* Host: Supermicro H14DSG-O-CPU (AS-5126GS-TNRT2), 2x AMD EPYC 9174F (Genoa, 16 cores each), **NPS4 = 8 NUMA nodes**.
  Socket 0 = NUMA 0-3 (CPUs 0-15,32-47), socket 1 = NUMA 4-7 (CPUs 16-31,48-63). NUMA distance 10 local / 12 same socket /
  32 cross-socket. Never group GPUs by `numa_node` (it gives 4 groups of 2, not the 2 sockets); use
  `physical_package_id` of the first CPU of the node.
* 8x NVIDIA RTX PRO 6000 Blackwell Server (cc 12.0, 96 GB, PCIe Gen5 x16, no NVLink). P2P access works for all 56 pairs;
  native P2P atomics are **not** supported (use store + flag, or the LL protocol, never remote atomics). BAR1 128 GiB per GPU,
  kernel cmdline `iommu=pt`, ACS redirect cleared on the Broadcom ports, MaxPayload 256 B. Idle ASPM shows 2.5 GT/s
  ("downgraded"); under load all links run Gen5 x16.
* Fabric: four PCIe tiers. Each tier = AMD root port (Gen5 x16) -> one Broadcom/LSI **PEX890xx "virtual switch"** (x16 uplink)
  -> two x16 Gen5 ports to two GPUs, one x4 port to a Micron 7450 NVMe, two unused ports and a management endpoint.
  `lspci -vvv` reports identical Device Serial Numbers for the switches of one socket, so there are only **two physical
  switch chips (one per socket), each exposing two virtual switches** with separate uplinks to different root ports.

| GPU | PCI address | NUMA | socket | virtual switch (upstream port) | root port | partner (r ^ 4) |
|-----|-------------|------|--------|-------------------------------|-----------|-----------------|
| 0 | 03:00.0 | 3 | 0 | 01:00.0 | 00:01.1 | 4 |
| 1 | 04:00.0 | 3 | 0 | 01:00.0 | 00:01.1 | 5 |
| 2 | 44:00.0 | 0 | 0 | 41:00.0 | 40:01.1 | 6 |
| 3 | 45:00.0 | 0 | 0 | 41:00.0 | 40:01.1 | 7 |
| 4 | 83:00.0 | 7 | 1 | 81:00.0 | 80:01.1 | 0 |
| 5 | 84:00.0 | 7 | 1 | 81:00.0 | 80:01.1 | 1 |
| 6 | c3:00.0 | 4 | 1 | c1:00.0 | c0:01.1 | 2 |
| 7 | c4:00.0 | 4 | 1 | c1:00.0 | c0:01.1 | 3 |

CPU affinity (`nvidia-smi topo -m`): GPU0,1 -> CPUs 12-15,44-47 (NUMA 3); GPU2,3 -> 0-3,32-35 (NUMA 0); GPU4,5 -> 28-31,60-63
(NUMA 7); GPU6,7 -> 16-19,48-51 (NUMA 4). `nvidia-smi topo -m` shows GPU0-1, 2-3, 4-5, 6-7 as `PIX`, every other pair as `SYS`.
ConnectX-7 NICs: `mlx5_1`, `mlx5_4` sit on the switch of GPU2,3, `mlx5_3`, `mlx5_5` on the switch of GPU4,5; GPU0,1 and
GPU6,7 have no NIC.

**Consequence: there are exactly two P2P tiers, not three.** Inside a socket (4 GPUs) every pair is equal, whether it is the
`PIX` pair or a pair across the two virtual switches (same bandwidth, same latency, no interference). The only other tier is
cross-socket. A pair-level tier has no measured basis. The "islands" of `muse_topo` are the two sockets, {0,1,2,3} and {4,5,6,7}.

## 2. Bandwidth

### 2.1 One flow (row = source, one class per cell group; the measured matrices are flat inside each class)

| transport | same socket (any pair) | cross socket |
|-----------|-----------------------|--------------|
| copy engine (`cudaMemcpyPeerAsync`) | 53.7 (peak 55.5-55.6 with 4-16 MiB chunks, 52.3 at 256 KiB) | 38.7 (peak 42.0 with 4-8 MiB chunks) |
| SM push, 188 CTAs x 512 threads | 41.8 | 29.4-29.5 |
| SM push, 1 CTA x 512-1024 threads | 44.8-45.3 | 35.3-37.4 |
| SM pull (loads from the peer), >= 4 CTAs x 512 threads (1 CTA x 512 gives only 24.6 / 16.1) | 50.6 | 44.1 (flows ending on GPUs 2,3,6,7 are 8-9 % slower: 40.2-40.6) |
| host DMA, pinned memory, per GPU alone | H2D 57.7 / D2H 56.6 | far-socket pinned memory: H2D 57.7, D2H 53.8-56.5 |

SM push peaks with a single CTA and decays to the 41.8 / 29.4 plateau from about 8 CTAs on; pull needs several CTAs. The copy
engine beats SM stores by 28-31 % and is the best single cross-socket transport.

### 2.2 Both directions of a pair at once (per direction)

| transport | same socket | cross socket |
|-----------|-------------|--------------|
| copy engine | 51.4 (sum 102.8) | 32.1-37.3 (GPU{0,1}<->{4,5} 37.3, {0,1}<->{6,7} and {2,3}<->{4,5} 32.2-32.3, {2,3}<->{6,7} 32.1) |
| SM push | 41.1 (sum 82.3) | 24.7-29.5 (sum 50.0-58.9) |
| SM pull | 39.9 | 24.8-29.3 (sum 49.6-58.5) |

Cross-socket paths through switches 1 and 3 (NUMA 0 and 4) are 8-15 % slower than through switches 0 and 2.

### 2.3 Concurrency (what scales and what does not)

* **Same socket scales perfectly.** 4 same-socket flows: CE 205.6 GB/s total, SM push 164.5; 8 same-socket flows on both sockets:
  CE 411 (51.4 per flow), SM push 329. Both GPUs of a virtual switch can send to the other switch of the socket at full
  rate (CE 107.3, SM push 83.7 from one switch). Same-socket P2P does not touch the uplink: it runs at full rate next to a
  saturating host DMA of the same GPU.
* **Cross-socket aggregate does not scale with the number of flows.** 1 / 2 / 3 / 4 concurrent flows from distinct GPUs
  (0>4, 1>5, 2>6, 3>7): SM push 29.5 / 29.5 / 30.7 / 30.5, copy engine 38.4 / 42.6 / 45.6 / 47.4 (48.1 with 4-8 MiB chunks), SM
  pull 44.1 -> 43.1, SM push with 1 CTA 35.3 -> 36.2. Both directions with 4+4 flows: CE 43.5 per direction (88.3 total with 1-16 MiB
  chunks), SM push 28.4 per direction (65.7 with 1 CTA), SM pull 27.5. The cross-socket fabric is the shared bottleneck.
* **The copy engine serializes P2P copies per source GPU** (0>4 at 38.3 then 0>2 averaged 22.3 GB/s = the sequential harmonic mix).
  CE fan-out and overlap with other CE copies are impossible; use SM kernels for concurrent intra-socket traffic and at most one
  CE flow per GPU for the cut.
* Receiving is cheap: several sources can push into one GPU ([4,6]>0: 37.2; [2,4]>0: 42.8; [1,2,3]>0: 41.7 = the 41.8 SM-write cap of
  the destination link). Two sources on the same switch into one GPU are limited by the destination link (push 21.0 + 20.9, CE 54.2).
* Host DMA: two GPUs of one switch share its uplink (28.8 each, 57.7 total); GPUs on different switches get 57.7 each (all 8 GPUs:
  230.8 H2D / 226.4 D2H). Host DMA of GPU0 concurrent with cross-socket P2P into the same switch shares the uplink (H2D 32.5 + P2P
  26.6 = 59 of 57.7 cap).

#### Cell (a'): 8 concurrent SM-push cut flows on r^4 (S1-12, 2026-11-10)

Eight cross-socket SM-push flows at once on the r^4 partner pairs, both directions (0<->4, 1<->5, 2<->6, 3<->7), one kernel per GPU (256 MiB message
per flow, 150 ms window, per-flow median of 5 iterations, 2 repeats; P2P only, NIC untouched; measured on kernel 6.8.0-147, driver 595.91.07):

* The per-flow pattern is the same at 4 / 8 / 16 CTAs per GPU: pairs 0<->4 and 1<->5 run ~8.3 GB/s each, pairs 2<->6 and 3<->7 ~5.5 GB/s each, sum 27.6 GB/s per direction. With 1 CTA per GPU the pairs run 9.7 / 6.4 GB/s, sum ~32 GB/s per direction. The 4 x min
  score is 22.1 GB/s at 4-16 CTAs and 25.3-25.5 with 1 CTA. Plain `st.global` v4 and `st.volatile.global` v4 give
  the same rates, so the store flavour is not the cause; CTA count does not matter from 4 up.
* Beside intra-socket pushes (one 8-CTA push per GPU) a 1-CTA cut flow collapses to ~10-18 GB/s (4 x min; single flows 2.5-5.2 GB/s), while an
  8-CTA cut flow keeps 22.1.
* Source: `muse_wf/s1-12/acell_results.md` (raw files under `/raid/vllm-runtime/work/s1-12/acell/`).

### 2.4 The COLLAPSE RULE (the design constraint everything else follows from)

A virtual switch must never have **concurrent outbound P2P (push or pull) to two different far-socket switches**, and must not mix
**host D2H with outbound cross-socket P2P**. Receiving from several far switches is fine.

| pattern | aggregate |
|---------|-----------|
| one GPU sends (or pulls) to far GPUs on two different switches at once (0>[4,6], 0>[4,7], 0>[5,6], pull into 0 from [4,6]) | 1.0-1.4 GB/s total (0.65 per flow, **40x below the 29.5 cap**), for SM push, SM pull and CE, 16 KiB-256 MiB, any CTA count |
| three far destinations / all 7 peers | 0.5 / 0.32 per flow |
| far peers on the SAME switch ([4,5] or [6,7]) | 29.5 total, fine |
| near + far mixes ([2,4], [1,2,4], [2,3,4,5]) | 41.8 total, fine |
| the collapse is switch-level, not GPU-level: CE 0>4 + 1>6 | 1.1 + 1.1 GB/s (vs 0>4 + 1>5: 21.0 + 21.1; 0>4 + 2>6, i.e. switch0>switch2 and switch1>switch3: 25.2 + 21.3, fine) |
| receiving from two far switches ([4,6]>0; pushes 4>0 + 6>1) | 37.2 / 18.3 + 18.8, fine |
| host D2H of GPU0 with CE 1>4 | 12.5 + 12.5 |

The mechanism was not identified (it is not an xGMI bandwidth limit: host DMA to far-socket memory runs at full speed). It is
the pattern of NCCL all-to-all (1.2-1.5 GB/s busbw), of all-pairs one-shot kernels (16 KiB: 33.9 us push, 74.9 us pull) and of vLLM's
CustomAllreduce (every rank reads all peers), which is why that one is disabled by vLLM on this box and is 3-30x slower than
NCCL when forced. The partner mapping **i <-> i ^ 4** (0-4, 1-5, 2-6, 3-7) sends switch 0 only to switch 2 and switch 1 only to
switch 3 (and the reverse), so it is collapse-free; `topology.py` derives the same mapping from sysfs for any rank order.

## 3. Latency

| measurement | same socket | cross socket |
|-------------|-------------|--------------|
| SM one-way, LL protocol (8 B units {4 B data, 4 B flag}, no fences), 4 B | 1.03-1.08 | 1.34-1.40 |
| SM one-way, flag protocol (payload stores, CTA barrier, `st.release.sys`), 1 CTA, 4 B | 1.63-1.67 | 1.85-2.02 |
| the same, 16 KiB | 1.74-1.85 | 2.09-2.16 |
| dependent remote load round trip (local memory 0.15) | 0.93-0.96 | 1.63-1.67 (+0.7) |
| copy engine, CUDA-graph memcpy node, 4 B-16 KiB | 1.01-1.04 (source GPUs 0-3), 1.27-1.38 (source GPUs 4-7) | 1.03-1.05 from GPU0 |
| copy engine, eager `cudaMemcpyPeerAsync` + sync | 5.8-6.8 | 6.7-8.1 (source GPUs 4-7) |

Payload size has no effect up to ~1 KiB. LL needs no fence but doubles the wire bytes (13.4 us at 64 KiB with 1 CTA vs 4.06 us for
the flag protocol).

End-to-end exchange skeletons (in-kernel loop, results verified each iteration, 0 errors): pair exchange 16 KiB LL 1.09 us / flag
2.38 us; three-round recursive-doubling all-reduce among 8 GPUs (pair, other switch, other socket): 4 B-1 KiB 4.41-5.09 us, 16 KiB 4.45 us
(LL, 64 CTAs) / 5.74 us (flag, 16 CTAs), 64 KiB 12.5 us, 128 KiB 31 us; one socket (4 GPUs) 16 KiB 2.7-2.8 us. The floor is
the sum of the one-way latencies of three rounds: 1.05 (pair) + 1.06 (other switch) + 1.35 (other socket) = 3.5 us. All-pairs one-shot
among 8 GPUs is ruled out beyond a socket (4 KiB 10.2 us, 16 KiB 33.9 us flag / 78.5 LL, 1 MiB 2380 us = 3 GB/s per GPU; the subset
{0,2,4,6}, where each switch hits two far switches, costs 6.56 us at 16 KiB and 220 us at 1 MiB).

## 4. Bottleneck analysis

**What limits a large all-reduce (NCCL, 128 MiB = 9.14 ms, busbw 25.7 GB/s).** Not the GPU link (CE 55.6, SM push 41.8-45), not
the switch uplink (same-socket P2P bypasses it), not the CPU root complex (411 GB/s of same-socket P2P without loss). It is the
cross-socket fabric: in a ring only the hops 3>4 and 7>0 cross the socket boundary and each carries 2(n-1)/n * S = 1.75 S = 235 MB per
direction at 24.9 GB/s (SM push, both directions, 188 CTAs): 235 / 24.9 = 9.43 ms predicted vs 9.14 measured. More NCCL channels or
rings do not help (fixed 1/4/8/16 channels: 25.4-26.2 busbw), and a ring order inside a socket is irrelevant, only the two crossing
hops matter. With copy-engine hops the ring model gives 7.3 ms (`NCCL_P2P_USE_CUDA_MEMCPY=1` measured 7.48 ms, but 5x worse at 8 KiB);
NCCL's Tree reaches 7.4 ms because it moves less over the cut.

**Volume bounds.** Any bandwidth-optimal all-reduce moves 1.75 S per GPU on its own link; the cut needs at least S per direction (not
1.75 S as in a ring), spread over the 4 parallel partner flows i <-> i^4 at the fabric cap. Hierarchy: reduce-scatter inside each
socket, exchange the partial with the cross-socket partner, all-gather inside the socket. Floors for S = 128 MiB (134.2 MB): GPU link
1.75 S / 53.7 = 4.37 ms; intra-socket 1.5 S / 53.7 = 3.75 ms (measured 3.84 ms); cut S / 44 GB/s = 3.05 ms (measured 3.39 ms).
Without pipelining the hierarchy is NOT better (copy-engine phases summed: 7.23 ms, flat CE ring 6.94 ms); it only pays when the
phases overlap. `muse_topo` implements that as a chunk-pipelined reduce-scatter + all-gather and, because its arithmetic contract
exchanges fp32 partials, carries 3S/8 per GPU and direction across the cut (S/4 fp32 partials + S/8 bf16 results; the opt-in bf16
exchange halves the partial part). Measured result: 7.1 ms at 128 MiB (1.30x), see `docs/h1b_latency.md`.

**Small messages are a latency problem.** For 16 KiB the floor is the 3.5 us of three dependent hops; NCCL needs 44.5 us (default
LL128/Simple) or 19.6 us with forced LL. `muse_topo` reaches 5.4 us for `[1,8192]` bf16 (CUDA graph), see `docs/h1a_latency.md`.

**Pathologies to design around**: (1) the collapse rule above; (2) CE serializes per source GPU; (3) SM stores waste link bandwidth
(41.8-45 vs CE 55.6) and decay with CTA count; SM pull needs >= 4 CTAs x 512 threads; (4) paths via switches 1 and 3 are 8-15 % slower;
(5) EP all-to-all would have to be scheduled in XOR rounds (i <-> i^k, 7 rounds, 15.1 GB/s at 32 MiB per pair, 12.6x faster than NCCL
all-to-all at 195.6 ms) or aggregated through the partner GPU; XOR rounds never let a switch address two far switches at once.

## 5. NCCL baselines on this box (NCCL 2.29.7 of the dev venv, nccl-tests 2.20.0 threaded `-g 8`, bf16, `NCCL_P2P_LEVEL=SYS`)

| op | message | eager us | CUDA graph us (`-G 1`) | busbw GB/s (eager) |
|----|---------|----------|------------------------|--------------------|
| all_reduce | 8 KiB / 16 KiB / 64 KiB | 43.9 / 44.5 / 47.4 | 48.6 / 48.7 / 52.6 | 0.33 / 0.64 / 2.42 |
| all_reduce | 1 MiB / 16 MiB | 89.6 / 1147 | 83.3 / 1140 | 20.5 / 25.6 |
| all_reduce | 128 MiB / 256 MiB | 9137 / 18131 | - | 25.7 / 25.9 |
| all_reduce, forced `NCCL_PROTO=LL` | 8 KiB / 16 KiB / 64 KiB / 1 MiB | - | 20.2 / 19.6 / 20.8 / 131 | - |
| all_gather | 8 KiB / 1 MiB / 128 MiB | 28.0 / 52.1 / 4643 | 32.7 / 48.4 / - | 0.26 / 17.6 / 25.3 |
| reduce_scatter | 8 KiB / 1 MiB / 128 MiB | 28.0 / 52.7 / 4742 | 28.3 / 50.2 / - | 0.26 / 17.4 / 24.8 |
| alltoall | 16 KiB / 1 MiB / 256 MiB | 36.4 / 640 / 195632 | 23.2 / 519 / - | 0.39 / 1.43 / 1.2 |

The graph numbers are back-to-back per-op times inside one captured graph (not single-launch latency). An 8-process MPI cross-check
agrees within noise (16 KiB 44.3 us, 128 MiB 9141 us / 25.7 GB/s). NCCL 2.29.7's graph search fails on this box ("Could not find a path
for pattern 4/1"): without `NCCL_P2P_LEVEL=SYS` it falls back to a ring 0..7 on 2 channels, Ring/Simple at every size, and sends the four
cross-pair hops through host shared memory (128 MiB 12.04 ms instead of 9.11 ms). The system NCCL is 2.31.2; it behaves like the opt-in
graph file `nccl/graph_r4.xml` (valid 4-channel ring, LL up to 512 KiB). Default protocol selection is LL128/Simple for <= 128 KiB;
forced LL is faster for decode sizes (19.6 vs 48.7 us at 16 KiB in graph mode) and slower from ~1 MiB.

Other implementations on the same box (2026-10-23, CUDA graph): FlashInfer PCIe IPC all-reduce (push, island-decomposed, bf16 tree
adds) 5.0 us at `[1,8192]`, 8.6 us at `[8,8192]`, 52.7 us at `[64,8192]`; vLLM CustomAllreduce (pull-based) 3-30x slower than NCCL.

## 6. Design rules derived from the measurements (as implemented in `muse_topo`)

1. Islands = sockets; inside an island everything is all-to-all with SM push stores into the peer's IPC receive buffer (flat, scalable).
2. Across the cut only the fixed partner r <-> r^4 talks (collapse-free), with one block (or few) pushing, since SM push peaks at 1 CTA
   and the copy engine gave no gain in the end-to-end kernels (`bench/ce_cut_probe.py`, `bench/a2a_probe.py`, `bench/host_relay_probe.py`).
3. Latency-bound sizes use the LL/sentinel protocols (no fences), bandwidth-bound sizes the flag protocol; the dispatch tables in
   `muse_topo/collectives.py` hold the measured winner per size.
4. No remote atomics (not supported); no host-side per-call state (CUDA graphs); no collective may overlap a kernel that can occupy
   every SM (the large kernels spin on peers with co-resident blocks).

## 7. Caveats and unknowns

* The internal crossbar of the switch chip and the cause of the 40x multi-far-switch collapse were inferred from measurements only.
* Copy-engine tests with more than one flow per source GPU (`conc` stages A6, A6b, B2c, D3 CE) are invalid as concurrency measurements
  because the CE serializes them; use the SM numbers there. Copy-engine flows use fixed iteration counts, so in mixed-speed tests a fast
  flow's rate includes time after slower flows ended. CE eager latencies are host-dominated and inflated for source GPUs 4-7 because the
  Python process runs on socket 0.
* Not measured: concurrent CE + SM traffic on one GPU link, and the numbers of an NCCL setup other than the ones listed above.
* The layout can change with a BIOS/NPS setting or a different card order; `muse_topo` therefore re-derives islands and partners from
  sysfs and refuses to activate (NCCL fallback, one log line per rank) unless the box shows 2 sockets x 2 virtual switches x 2 GPUs of
  cc 12.0 with P2P access on all 56 pairs.

## 8. NCCL 2.30.7 (vllm 0.33.0 venv, 2026-11-09)

Checked on the rebuilt dev venv (nccl-tests 2.20.0 threaded `-g 8` and the 8-process vLLM `PyNcclCommunicator` harness, same box, the
2.29.7 wheel of `/raid/vllm-runtime/.venv` as same-day control):

* The graph search bug is NOT fixed: "Could not find a path for pattern 4/1" with and without `NCCL_P2P_LEVEL=SYS`.
* New failure mode: with IB enabled the cross-socket hop 3>4 is connected `via NET/IB/5` (RoCE mlx5_2:2, GDRDMA) instead of P2P (SYS)
  or SHM (default), and `ibv_modify_qp` fails with "Operation not supported", so the first collective fails (`unhandled system error`;
  threaded nccl-tests too). The NICs now report 200 Gb/s (2.29.7: 40 Gb/s), which is presumably why the search prefers NET.
  `NCCL_IB_DISABLE=1` fixes it (`env/host.env` and `scripts/py_topo.sh` set it by default): all hops `P2P/CUMEM` (multi-process) or
  `P2P/direct pointer` (threaded). `NCCL_NET_DISABLE_INTRA=1` also initialises (not run through the full matrix).
* With `NCCL_IB_DISABLE=1`: `NCCL_P2P_LEVEL=SYS` and the graph file behave as on 2.29.7 (default without SYS: 3>4 / 7>0 via SHM again;
  `graph_r4.xml`: 4 channels), the outputs are bitwise identical to 2.29.7 (27 shapes eager + graph, all ranks, every config) and the
  bus bandwidths match 2.29.7 (128 MiB 9173 vs 9135 us, +0.4 %; 8-64 KiB with SYS only +1.2-1.5 us, +3 %; with the graph file equal). The graph file needs IB disabled ("Could not find NET with id 0" otherwise).
* Since 2026-11-10 the ConnectX-7 is in IB mode (the failure above was seen with RoCE); `NCCL_IB_DISABLE=1` stays required (`env/host.env`, `scripts/py_topo.sh`).
