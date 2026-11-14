# AS-5126GS-TNRT2_vllm_plugin

針對 **10.19.209.74（Supermicro AS-5126GS-TNRT2，8x NVIDIA RTX PRO 6000 Blackwell Server）** 這台主機的 PCIe / NUMA 拓樸所做的最佳化，
用來加速 `unsloth/Muse-Spark-Open-816B-A42B-NVFP4`（vLLM branch `sm120-muse-spark-open`，目前 production 為 v0.33.0 + Muse Spark backport（`f6a60ab9d7`），
TP8 + EP8）的推論。

這個資料夾只放「只對這台機器的拓樸才有意義」的東西：topology-specialized 的 P2P collectives（`muse_topo` 套件與 vLLM plugin）、NCCL 環境變數與
graph file、啟動 / 驗證 script 與量測文件。與機器無關的通用 SM120 最佳化則是 vLLM branch 裡的 git commits，不在這裡。

本資料夾是 deliverable，**本機的 `/home/user/trivial/AS-5126GS-TNRT2_vllm_plugin/` 是 source of truth**；伺服器上的
`/raid/vllm-runtime/host_topo/` 只是用 `scripts/deploy_host.sh` rsync 出去的 mirror。

**本檔是這個資料夾的最終說明，自成一體**：為什麼需要（第 1 節）、資料夾內容（第 2 節）、部署與前置條件（第 3 節）、**怎麼啟動**（第 4 節：三個 launch script、選項、結束碼、開機後檢查、停止）、
exact bytes 與 tier 公式（第 5 節）、最終 matrix 與 9 / 9 gates（第 6 節）、fp8 KV 1M context 的量測（第 7 節）、設定與 accuracy class（第 8 節）、
plugin 的 microbenchmark（第 9 節）、驗證方式（第 10 節）、已知限制與 noise floor（第 11 節）、parked 的 lever（第 12 節）、出處（第 13 節）。

## 摘要

| 項目 | 結果（EP8、concurrency 1、2026-10-24，詳見第 9 節） |
|------|------|
| decode（tg） | 51.2 -> 85.4 t/s（TPOT 19.5 -> 11.7 ms，+67 %）；real weights 52.0 -> 88.4 t/s（+70 %） |
| prefill（pp） | c1（512 tokens）+7 %、13k / 65k / 118k tokens 的 context 約 +13 .. +14 %；real weights +8 .. +12 % |
| 額外記憶體 | 每張 GPU 61.1 MiB（large-message 演算法不再增加） |
| accuracy class（預設） | `more_accurate`：fp32 加總、單次最終 rounding，所有 size / 演算法 / rank / run 之間 bitwise 相同 |
| production 最終結果（S1-11，vLLM 0.33.0、host `f540ce1b37`，2026-11-13） | spec ON c1 / c16 / c64 / c128 的 DEC greedy 221.4 / 756.6 / 1252.2 / 1479.1 tok/s（AL 4.64 .. 4.88）；ns256 ON 115.2 / 535.5 / 980.9 / 1212.2（c256 1432.2）；ns128 OFF 116.1 / 540.5 / 980.4 / 1232.8；9 / 9 gates PASS；表格、exact bytes、已知限制見第 5、6、11 節 |
| 啟用條件 | `MUSE_TOPO_ENABLE=1`，且拓樸簽章（2 sockets x 2 virtual switches x 2 GPUs，cc 12.0，56 組 P2P 全通）吻合，否則自動退回 NCCL |

## 1. 為什麼需要（背景與 topology）

這台機器沒有 NVLink。8 張 GPU 分在兩顆 CPU socket（GPU 0-3 與 4-7），每個 socket 是一顆 Broadcom PEX890xx，對外呈現兩個 virtual switch，
所以一共 4 個 virtual switch：{0,1}、{2,3}、{4,5}、{6,7}。實測重點（2026-10-23，完整資料見 `docs/topology.md`）：

* socket 內任兩張 GPU 的頻寬 / latency 完全相同（copy engine 53.7 GB/s、SM push 41.8 GB/s，one-way LL latency 約 1.05 us），而且多條 flow 可完美擴展
  （8 條 same-socket flow 共 411 GB/s）。
* 跨 socket 只有一個共享瓶頸：單 flow 約 29.5（SM push）/ 38.7（copy engine），**flow 變多總頻寬也不增加**（SM push 約 30 GB/s、copy engine 約 44-48 GB/s）。
* **COLLAPSE RULE**：一個 virtual switch 不可同時對兩個不同的「對岸 socket 的 switch」送出 P2P（push 或 pull 皆然），否則總頻寬掉到約 1.3 GB/s（慢約 40 倍）；
  也不可把 host D2H 與跨 socket 的 outbound P2P 混在一起。partner 對應 `r <-> r ^ 4`（0-4、1-5、2-6、3-7）不會踩到這條規則。
* NCCL 在這台機器上的 128 MiB all-reduce 要 9.14 ms（busbw 25.7 GB/s）：ring 只有兩個 hop 跨 socket，每個 hop 要送 1.75 S；
  小訊息（8-16 KiB）則受 latency 支配，NCCL 約 43-48 us，而理論下限約 3.5 us。vLLM 自己的 CustomAllreduce 在 8 張 PCIe GPU 上被關閉
  （它是 pull-based、會踩到 collapse rule，強制啟用時比 NCCL 慢 3-30 倍）。

`muse_topo` 的做法：以 socket 為 island（{0,1,2,3}、{4,5,6,7}），island 內用 SM push store 做 all-to-all，跨 island 只走固定的 partner（r <-> r^4），
小訊息用 LL / sentinel protocol（無 fence），大訊息用 chunked、pipelined 的 reduce-scatter + all-gather。

## 2. 資料夾內容

```
AS-5126GS-TNRT2_vllm_plugin/
├── README.md                    本檔
├── muse_topo/                   Python 套件（JIT 編譯的 CUDA extension + vLLM plugin）
│   ├── __init__.py              對外 API：TopoComm、TopoUnavailable
│   ├── topology.py              由 sysfs 推導 islands / partners，拓樸簽章檢查（不吻合就 raise TopoUnavailable）
│   ├── collectives.py           TopoComm：supports() 資格判斷、size dispatch 表（附量測日期與 protocol）、stream 規則
│   ├── ops.py                   extension 的 JIT build（digest 目錄、flock 保護）與 IPC workspace
│   ├── vllm_plugin.py           register()：patch CudaCommunicator 的 all_reduce / all_gather / reduce_scatter
│   └── csrc/                    CUDA / C++ 原始碼（ar_sll.cu、flag.cu、ag.cu、large.cu、bindings.cpp ...；
│                                boundary.cuh、ar_bnd.cu、ag_bnd.cu：collective + residual boundary / RMSNorm epilogue 的 fused kernel；
│                                gemv.cuh、gemv_push.cu：pooler / unpooler 的 GEMV 與其後的 all-gather 融合）
├── muse_topo-0.1.dist-info/     entry_points.txt：[vllm.general_plugins] muse_topo = muse_topo.vllm_plugin:register
│                                （放在 PYTHONPATH 上就會被 vLLM 找到，不需要 pip install）
├── env/host.env                 主機環境變數：NCCL_P2P_LEVEL=SYS、MUSE_TOPO_ENABLE=1、bf16 exchange 與 NCCL graph file（預設 ON），其他選項以註解列出
├── env/spec_k7.env              speculative decoding（RadixArk DFlash2 K=7）的 launch profile：capture 上限 256、`--max-num-batched-tokens 8199`、
│                                `--max-num-scheduled-tokens 8192`、exact KV bytes；source `env/offload.env`（tier 預設 ON，`MUSE_OFFLOAD=0` 退出）
├── env/offload.env              NUMA-local KV CPU offload：`--numa-bind` + SimpleCPUOffloadConnector，每個 rank P* = 51,814,072,320 B（48.26 GiB）的 pinned tier；
│                                三個 profile 的 ON bytes；在 `env/host.env` 之後 source
├── env/launch.env               launch script 用的共用資料：共有的 flag、speculative 的參數、non-speculative 的 no-tier bytes、各 profile 的 tier 預設、
│                                各狀態預期的 KV tokens / blocks；只有 `LAUNCH_*` shell 變數，不 export 任何東西
├── launch/                      production launch scripts（第 4 節）
│   ├── serve_spec.sh            speculative profile（tier 預設 ON，`MUSE_OFFLOAD=0` = 無 tier 的 launch）
│   ├── serve_nonspec.sh         `<ns128|ns256>`（ns256 tier 預設 ON，ns128 預設 OFF，`MUSE_OFFLOAD=1` / `0` 覆寫）
│   ├── stop_server.sh           以 `server.sess`（server 自己在 setsid 之後記下的 pid / session / start time）停止 launch 出來的 server
│   ├── session.sh / lc_child.sh session 的記錄與保護（lc_child.sh 在 setsid 之後記下自己的 session，再 exec server）
│   └── lib.sh                   共用流程（清環境、source env 檔、serve line、preflight、等 READY、開機後檢查、錯誤時停 server）
├── scripts/
│   ├── deploy_host.sh           本機 -> 伺服器 mirror 的 rsync（只同步本資料夾）
│   ├── preflight_offload.sh     tier ON 開機前的記憶體檢查（node 7 的 reclaimable slab、每個 node 的 LEFT_N）
│   ├── check_spec_launch.sh     由 serve log 確認 speculative profile 有生效（SP floor、graph 上限、chunk、draft FP8、fused boundary ...）
│   ├── check_offload_launch.sh  由 serve log / serve_cmd.txt 確認 offload 有生效或確實沒有（NUMA binding、CPU block 數、layout、bytes 配對）
│   ├── check_kv_launch.sh       由 serve log / serve_cmd.txt / cache_config_info.txt 確認三個 profile 的 exact KV bytes、KV tokens / blocks、`--max-num-seqs`、capture cap
│   ├── check_serve_line.py      live command line = serve_cmd.txt = clean shell 推導的 line，vLLM 解析出的參數與 connector / role 一致
│   ├── runsrc_topo.sh           以 dev venv 執行 python（PYTHONPATH 含本資料夾與 vLLM tree，source env/host.env）
│   ├── py_topo.sh               跑單機測試的 python wrapper（scratch 與 JIT cache 在 `$MUSE_TEST_SCRATCH`）
│   ├── locked_job.sh            在 GPU lock 下 detached 執行任意指令，log 與 DONE marker 寫到 `$MUSE_TEST_SCRATCH/logs`
│   ├── run_h1a.sh               小訊息演算法的驗證序列（steps：smoke tests tests_quick ctx initfail other other_full topology timeout epoch unit）
│   └── run_h2.sh                vLLM plugin 的驗證序列（steps：plugin e2e_* lp_* ops real_* compare）
├── nccl/
│   ├── graph_r4.xml             NCCL graph file（deterministic_different；`env/host.env` 預設使用，見第 8 節）
│   ├── bitwise.py               PyNccl all-reduce 的 bitwise identity 測試（8 ranks，eager + CUDA graph）
│   ├── e2e_bitwise.py           6 層 dummy-weight 模型的 end-to-end dump（token ids、logprobs、prompt logprobs）
│   └── compare.py               比較兩份 e2e dump（token / logprob 是否 bitwise 相同）
├── bench/                       correctness / launch 測試與少數輔助 script（見下）
├── docs/
│   ├── topology.md              實測拓樸：矩陣、collapse rule、bottleneck 分析、NCCL baseline
│   ├── h1a_collectives.md       基礎設施與 decode-size 演算法（API、arithmetic contract、protocol、dispatch）
│   ├── h1b_collectives.md       large-message all-reduce / reduce-scatter / all-gather
│   └── h2_vllm_plugin.md        plugin 的 enable / patch 範圍 / 環境變數 / 每個 collective 的路徑 / 限制
└── tools/deploy.sh              把 `/home/user/vllm`（vLLM branch `sm120-muse-spark-open`）的 source 同步到伺服器的 `sm120-dev/src/`（第 3 節）
```

`bench/` 的內容：

| 類別 | 檔案 |
|------|------|
| launch / 檢查 script 的測試（CPU、bash，不需要 GPU 與 vLLM） | `test_launch_equiv.sh`、`test_launch_flow.sh`、`test_check_kv_launch.sh`、`test_check_spec_launch.sh`、`test_check_offload_launch.sh`、`test_check_serve_line.sh`、`test_offload_env.sh`、`test_preflight_offload.sh`；fixture 在 `bench/fixtures/`（`launch_ref`：gated boot 的 serve line / env 紀錄與其 PROVENANCE，`launch`：真實 boot log 的相關行，`launch_stub`：stub 的 vllm / nvidia-smi / preflight） |
| plugin 的 CPU 單元測試 | `test_plugin_unit.py`（mock TopoComm）、`test_topology.py`（需要 `nvidia-smi`） |
| correctness（8 processes、GPU lock 下執行） | `tests_h1a.py`、`tests_h1b.py`、`test_epoch.py`、`test_timeout.py`、`test_timeout_large.py`、`test_init_failure.py`、`check_ctx.py`、`smoke.py`、`test_plugin.py`（8 ranks、經 vLLM wrapper）、`test_medium_bx.py`（medium all-reduce 的 bf16 exchange）、`test_a1_tables.py`（各 dispatch table 的逐 size 檢查）、`test_boundary.py`、`test_ar_boundary.py`（fused boundary kernel 對 Triton kernel 的 bit-exact 檢查，`boundary_test.cu` 由它 JIT build）、`test_gemv_push.py`（GEMV + all-gather 融合） |
| 第二套獨立的 H1a 測試 | `test_h1a.py` + `hlib.py` + `test_topology.py` |
| 測試輔助 | `common.py`、`hlib.py`（各自的 helper）、`bench_h1a.py` 與 `bench_medium.py`（只保留作測試支援：`test_medium_bx.py` 與 `test_a1_tables.py` import 其中的 `contract_ref_bx`，不是 benchmark 入口）、`build_ext.py`（預先 JIT build）、`summarize_ops.py`（整理 `MUSE_TOPO_LOG_OPS` 的 log） |
| end-to-end | `e2e_prompt_lp.py`（teacher-forced logprob）、`e2e_real.py`（real weights、62 層）、`chat_check.py`（greedy chat 檢查） |
| runner | `run_h1b.sh`（steps：tests_quick tests tests_soak timeout epoch） |

**注意：H1a 有兩套彼此獨立的測試，是刻意保留的 cross-check，請勿刪除其中一套**：`tests_h1a.py` + `common.py`，以及 `test_h1a.py` + `hlib.py` + `test_topology.py`。

## 3. 部署與前置條件

### 3.1 同步本資料夾到伺服器

```bash
cd /home/user/trivial/AS-5126GS-TNRT2_vllm_plugin
scripts/deploy_host.sh -n -i                        # 先看會動到哪些檔案（額外的參數交給 rsync：-n = dry run，-i = 列出差異；最後的 "synced" 訊息在 dry run 時仍會印）
scripts/deploy_host.sh                               # rsync -a --delete 到 user@10.19.209.74:/raid/vllm-runtime/host_topo/（排除 __pycache__、*.pyc、.git、build）
# 試驗用：DEST=<server dir> FROM=<local dir> scripts/deploy_host.sh
```

`--delete` 會把本機已刪除的檔案也從 mirror 移除；production launch 從 mirror 讀 `launch/`、`env/`、`scripts/check_*`，所以只在沒有我們自己的 server 佔著 GPU lock 時才同步，
並在同步前用 `-n` 看過清單。launch script 的行為變更（`launch/`、`env/`、`scripts/check_*`）要同步之後才在伺服器上生效。

### 3.2 部署 vLLM tree（branch `sm120-muse-spark-open`）

伺服器的 vLLM tree `/raid/vllm-runtime/sm120-dev/src/` 是 `/home/user/vllm` 的 branch `sm120-muse-spark-open`（production：v0.33.0 + Muse Spark backport，`f6a60ab9d7`）的 source，
疊在 venv 內 pip 安裝的 vllm 0.33.0 之上（`PYTHONPATH` = 本資料夾 + 這個 tree；tree 裡預先編譯的 artifacts 留在伺服器上，deploy 不動它們）：

```bash
cd /home/user/vllm && git checkout sm120-muse-spark-open      # 確認 working tree 乾淨、HEAD = f6a60ab9d7
/home/user/trivial/AS-5126GS-TNRT2_vllm_plugin/tools/deploy.sh      # 印出 "deployed <short sha>"；有未 commit 的修改會加 "+dirty"
```

`tools/deploy.sh` 把 `/home/user/vllm` 下 `vllm/`、`tests/`、`benchmarks/` 中 tracked 與 untracked（未被 ignore）的檔案 rsync 到伺服器的 `sm120-dev/src/`，
排除 `vllm/vllm_flash_attn` 與 `vllm/third_party`（伺服器上預先編譯的版本，保持不動）。它會覆蓋共用的部署樹，只有負責整合的人才應該執行，部署前確認沒有 server 在跑。

### 3.3 伺服器 runtime（前置條件）

* **venv**：`/raid/vllm-runtime/sm120-dev/.venv`（CPython 3.12，用 uv 建立）。安裝配方（逐字取自當時的安裝 script；aliyun mirror）：

  ```bash
  export PATH=$HOME/.local/bin:$PATH
  cd /raid/vllm-runtime/sm120-dev
  uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ vllm==0.33.0 b12x==1.5.1 cuda-tile==1.6.0 pytest
  ```

  （production 上實際的版本：torch 2.14.0+cu130、flashinfer 0.7.2.post1、NCCL 2.30.7、triton 3.8.0。）
* **JIT build**：`muse_topo` 的 CUDA extension（`_muse_topo_C`）在第一次啟動時 JIT build，結果在 `$VLLM_CACHE_ROOT/muse_topo/<digest>`（server：`/raid/vllm-runtime/sm120-dev/vllm_cache/muse_topo/`）。
  digest 由 `csrc` 所有檔案、編譯旗標、torch 與 nvcc 版本決定（Python 檔的修改不影響）；改了 `csrc` 之後第一次啟動會重新 build 一次（flock 保護，8 個 rank 只 build 一次）。需要 nvcc >= 12.8（伺服器：`/usr/local/cuda` 13.3）。
  可先用 `scripts/runsrc_topo.sh $HOST_DIR/bench/build_ext.py` 預先 build。
* **cache 與 scratch**：`/raid/vllm-runtime/sm120-dev/` 下的 `vllm_cache`、`triton_cache`、`inductor_cache`、`cute_cache`、`fi_workspace`（launch script 與 `scripts/runsrc_topo.sh` 使用）；
  測試 runner 的 scratch（log、JIT、dump）在 `${MUSE_TEST_SCRATCH:-/raid/vllm-runtime/work/plugin-tests}`（各 script 自己 `mkdir -p`；不要寫進本資料夾）。
* **模型**：`unsloth/Muse-Spark-Open-816B-A42B-NVFP4`（HF cache 離線載入，`HF_HUB_OFFLINE=1`）；speculative profile 的 drafter 是 `RadixArk/Muse-Spark-Open-DFlash2` 的 snapshot
  `46c90f412549c472941a68382d91a99e5e6372a0`（路徑在 `env/launch.env`）。
* **共用 GPU lock**：`/raid/vllm-runtime/sm120-dev/gpu.lock`，所有 GPU 工作一律在 `flock -w 43200` 之下執行；launch 出來的 server 會繼承 lock 的 fd，所以 lock 直到 server 結束才釋放。
  **不要在一個自己會拿 lock 的 script 外面再包同一個 lock 的 flock（會 deadlock）**；`scripts/locked_job.sh` 已經拿 lock。
* **開始前**：確認 GPU 閒置（`nvidia-smi`、`sudo docker ps -a`；`nemotron-ultra` 必須維持 stopped）。tier ON 的 boot 可能把部分 anonymous memory 換出到 swap（伺服器有 8 GiB 的 swap file，見第 11.1 節）；`--numa-bind` 需要 `numactl`。

## 4. 怎麼啟動（how to launch）

三個 production profile 各有一支 script，位於 `launch/`（伺服器上是 `/raid/vllm-runtime/host_topo/launch/`）。**這三支 script 是唯一支援的啟動方式**：
每一條 serve line 與 export 的環境都和 gated boot 逐 byte 相同（`bench/test_launch_equiv.sh` 靜態證明），開機後再做硬檢查。

### 4.1 三個指令與 `MUSE_OFFLOAD`

| 指令 | tier 預設 | `MUSE_OFFLOAD=0` | `MUSE_OFFLOAD=1` |
|------|-----------|------------------|------------------|
| `launch/serve_spec.sh` | ON（adopted） | 無 tier 的 launch：11,720,212,480 B、無 connector、無 `--numa-bind` | tier ON，11,645,337,600 B + `$MUSE_OFFLOAD_SERVE_ARGS` |
| `launch/serve_nonspec.sh ns256` | ON（RECOMMENDED） | 0.33.0 validation 的 line：`--max-num-seqs 256 --max-cudagraph-capture-size 256 --kv-cache-memory-bytes 16957243392` | `... 16854810624 $MUSE_OFFLOAD_SERVE_ARGS` |
| `launch/serve_nonspec.sh ns128` | **OFF**（shipped） | production 的 line：`--max-num-seqs 128 --kv-cache-memory-bytes 17357799424` | `... 17257267200 $MUSE_OFFLOAD_SERVE_ARGS`（opt-in） |

```bash
ssh user@10.19.209.74
cd /raid/vllm-runtime/host_topo
flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock launch/serve_spec.sh --out <out dir>              # tier ON（預設）
flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock env MUSE_OFFLOAD=0 launch/serve_spec.sh --out <out dir>    # 沒有 tier
flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock launch/serve_nonspec.sh ns256 --out <out dir>     # tier ON（預設）
flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock launch/serve_nonspec.sh ns128 --out <out dir>     # 沒有 tier（預設）；MUSE_OFFLOAD=1 打開
launch/stop_server.sh <out dir>                                                                          # 停止
```

只在不共用的機器上才可以省掉 `flock`。`launch/` 的 script 不會自己拿 lock；server 繼承 flock 的 fd，所以 lock 直到 server 結束才釋放。
`MUSE_OFFLOAD` 是唯一的 operator 輸入（0 或 1，空字串 = 該 profile 的預設，其他值是錯誤）。

### 4.2 一次做完的事

清掉 calling shell 殘留的 `MUSE_*` / `VLLM_*` / `NCCL_*` / `SPEC_*` export 與 `CUDA_VISIBLE_DEVICES`（`--keep-env` 才保留；scrub 之外的變數，例如 `LD_PRELOAD`、`OMP_*`，不會被清也不在比對範圍）、
匯出 gated boot 的 runtime（venv、cache、`PYTHONPATH` = 本資料夾 + vLLM tree）、source `env/host.env` 與 profile 的 env 檔、組出 serve line、tier ON 時先跑 `scripts/preflight_offload.sh`（rc != 0 就拒絕啟動）、
server 在自己的 session 啟動、等 `/health`、存下 READY 當下的資料、跑開機後的檢查。**所有數字只來自 env 檔**：`env/spec_k7.env`（speculative，並 source `env/offload.env`）、`env/offload.env`（tier、ON bytes）、`env/launch.env`（共用 flag、non-speculative 的 no-tier bytes、預期 KV tokens / blocks）。

* 選項：`--out DIR`（輸出目錄，預設 `./launch_<profile>_tier<0|1>_<時間>`）、`--keep-on-fail`（檢查失敗時不停 server）、`--keep-env`（不清 calling shell 的 export，做實驗用）、`--dry-run`（只寫 `serve_cmd.txt` 與 `env.txt`，不碰 GPU / venv / preflight）。
* 結束碼：0 = READY 且所有檢查通過（server 留著繼續跑）；1 = 開機後檢查失敗（除非 `--keep-on-fail`，server 已停止）；2 = 拒絕啟動（輸入錯誤、GPU 忙、preflight 失敗、env 檔錯誤，什麼都沒啟動）；3 = server 一直沒有 READY（死掉、逾時或被中斷，已停止；若同時記 `NOT_STOPPED`，server 可能還在跑，見停止路徑）；64 = 用法錯誤。
* 輸出目錄：`serve_cmd.txt`、`env.txt`（與 gated boot 同樣的寫法與 filter）、`serve.log`、`serve.pid`、`server.sess`（`<pid> <sid> <pgrp> <start time> <boot_id>`，由 server 自己的 child process 在 setsid 之後寫入；leader 已消失時，`stop_server.sh` 只會處理 boot_id 相同、cwd 為此輸出目錄、start time 不早於 leader 的成員，其餘只列出不送 signal，exit 3）、`status.txt`（最後一行 `RESULT ...`，另有 swap 紀錄：launch / READY 的 swap used、`pswpin` / `pswpout` 差）、`preflight.txt`、`cache_config_info.txt`（`/metrics` 的 `vllm:cache_config_info`，block 數只在這裡）、`pid_cmdline.txt`、`pid_environ.txt`、`pid_environ_expected.txt`、`clean_ref/`、`clean_check/`（clean shell 的 dry run：啟動前與開機後各一次）、`gpu_mem_ready.txt`、`check_*.txt`。
* 等價性：每條路徑的 serve line 與 export 的環境和 gated / adoption boot 逐 byte 相同，由 `bench/test_launch_equiv.sh` 靜態證明（spec ON / OFF、ns256 ON、ns128 ON 對各自的 adoption boot 紀錄；ns128 / ns256 OFF 對 production 與 0.33.0 validation 的 line）。
  已知差異：0.33.0 validation 的 ns256 OFF boot 沒有 env 紀錄，環境只能對 ns128 的 production record 與它的 `flags` 行證明；script 的 ns256 OFF launch 會 export `VLLM_PLUGINS=muse_topo`，而那次 validation boot 沒有（host `a26c2be` 在它之後才加；之後的 gate boot 有，證據只有 0 tracebacks）；
  ns256 OFF 的 `cache_config_info` fixture（blocks 258,747）是由 validation 紀錄合成的（該 boot 沒存 metrics），smoke boot 會記錄真正的 `cache_config_info` 與 pid environ；speculative 的 script 會 export `VLLM_SERVER_DEV_MODE=1`（adoption boot 也有），non-speculative 沒有。
* 停止（`launch/stop_server.sh <out dir>`）：TERM 記錄下來的 session，再 KILL；leader 的 pid + start time 必須與記錄相同，包含 launcher 或其 parent 的 session 永遠不送 signal。
  launcher 在收到 TERM / INT / HUP 或開機檢查失敗時停止 server，使用 `server.sess`；該檔案消失或無法讀取時，改用 launcher 接受 session 當下記在記憶體裡的 tuple（pid、session、process group、start time、boot id），並通過同樣的 guard。
  只有停止以結束碼 0 結束才記 `STOPPED gpu_used_mib=...`；被拒絕、或 leader 已不在而 member 無法證明屬於這次 launch 時記 `NOT_STOPPED rc=...`（server 可能還在跑，要手動停；EXIT trap 不會重複停止）。

### 4.3 開機後檢查（任何一個失敗都是失敗）

* `env`：server process 的 live environ 經過與 gated boot 的 `pid_environ.txt` 相同的 filter，必須與 clean shell（`env -i`，只帶 `MUSE_OFFLOAD`）dry run 推導出的 `clean_ref/pid_environ_expected.txt` 逐行相同（`--keep-env` 時改比 calling shell 自己的紀錄，`status.txt` 有 `KEEP_ENV` 行）。
* `line`（`scripts/check_serve_line.py`）：live API-server 的 `/proc/<pid>/cmdline`（NUL 為界）= `serve_cmd.txt` = 開機後在 clean shell 重新推導的 serve line，且 vLLM 自己解析出的 `non-default args` 行帶有完全相同的選項與值、沒有多的選項，
  且 log 的 KVTransferConfig 不得有 command line 沒有的 `kv_connector` / `kv_role` / 非空的 `kv_connector_extra_config`（tier 大小）；啟動前 launcher 也先拒絕與 clean shell 推導不同的 serve line；`--keep-env` 時不比 clean shell 的推導。
* speculative：`SPEC_EXPECT_FB=1 SPEC_EXPECT_OFFLOAD=$MUSE_OFFLOAD scripts/check_spec_launch.sh <serve.log>`、`scripts/check_offload_launch.sh`、`scripts/check_kv_launch.sh spec <1|0>`；
  non-speculative：`scripts/check_offload_launch.sh`（tier OFF 時 `MUSE_OFFLOAD_EXPECT=off`）與 `scripts/check_kv_launch.sh <ns128|ns256> <1|0>`（`--kv-cache-memory-bytes`、`--max-num-seqs`、ns256 的 capture cap 256 / ns128 無 cap、KV tokens、`num_gpu_blocks`、FULL decode graph 的上限）。數字見第 5 節與 `env/launch.env` 的表。
* 測試用的 `MUSE_LAUNCH_*` hook（取代 vllm / preflight / nvidia-smi / venv）只要有設就會在 `status.txt` 記一行 `HOOK ...`；production 啟動前確認 calling shell 沒有殘留這些 export。
* 手動確認 plugin 有生效：serve log 應該出現（每個 process / 每個 worker 各一行）

```
muse_topo plugin registered in pid <pid> (ops=all_gather,all_reduce,reduce_scatter)
muse_topo active on rank <r> (island <I>, partner <P>, workspace 61.1 MiB): islands=[[0, 1, 2, 3], [4, 5, 6, 7]] partner_of=[4, 5, 6, 7, 0, 1, 2, 3] ...
```

  拓樸不吻合、build 失敗或 `MUSE_TOPO_ENABLE != 1` 時改為 `muse_topo disabled on rank <r>: <原因>` / `muse_topo not used for group ...`，所有 rank 的決定一致，vLLM 繼續使用 NCCL（不會 crash）。
  看每個 collective 實際走哪條路徑：`MUSE_TOPO_LOG_OPS=1`（group rank 0）或 `2`（每個 rank），再用 `python3 bench/summarize_ops.py <serve.log>`（若有任何 key 走 NCCL，加 `--paths` 會列出並附原因）。
  A/B 開關：`MUSE_TOPO_ENABLE=0`（純 NCCL）、`MUSE_TOPO_LARGE=0`（只關 large-message 演算法）、`MUSE_TOPO_OPS=all_reduce`（只讓部分 op 走 topo）；這些要在 `--keep-env` 的實驗 launch 裡設，production launch 會清掉它們。

### 4.4 每個狀態的完整 serve line

下面六行逐字取自 `--dry-run` 的 `serve_cmd.txt`（與 `bench/fixtures/launch_ref/*.serve_cmd.txt` 逐 byte 相同）。**不要手動貼這些 line 啟動**：環境、preflight 與檢查都在 launch script 裡。

spec，tier ON（`serve_spec.sh` 預設）：

```
vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt \{\"image\":0\,\"audio\":0\,\"video\":0\} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice --speculative-config \{\"draft_sample_method\":\"probabilistic\"\,\"method\":\"dflash\"\,\"model\":\"/raid/.cache/huggingface/hub/models--RadixArk--Muse-Spark-Open-DFlash2/snapshots/46c90f412549c472941a68382d91a99e5e6372a0\"\,\"num_speculative_tokens\":7\,\"rejection_sample_method\":\"block\"\} --per-request-spec-decode-metrics detailed --max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 11645337600 --numa-bind --kv-transfer-config \{\"kv_connector\":\"SimpleCPUOffloadConnector\"\,\"kv_role\":\"kv_both\"\,\"kv_connector_extra_config\":\{\"cpu_bytes_to_use\":414512578560\,\"cpu_bytes_to_use_per_rank\":51814072320\}\}
```

spec，`MUSE_OFFLOAD=0`：

```
vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt \{\"image\":0\,\"audio\":0\,\"video\":0\} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice --speculative-config \{\"draft_sample_method\":\"probabilistic\"\,\"method\":\"dflash\"\,\"model\":\"/raid/.cache/huggingface/hub/models--RadixArk--Muse-Spark-Open-DFlash2/snapshots/46c90f412549c472941a68382d91a99e5e6372a0\"\,\"num_speculative_tokens\":7\,\"rejection_sample_method\":\"block\"\} --per-request-spec-decode-metrics detailed --max-cudagraph-capture-size 256 --max-num-batched-tokens 8199 --max-num-scheduled-tokens 8192 --kv-cache-memory-bytes 11720212480
```

ns256，tier ON（預設）：

```
vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel --max-num-seqs 256 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt \{\"image\":0\,\"audio\":0\,\"video\":0\} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice --max-cudagraph-capture-size 256 --kv-cache-memory-bytes 16854810624 --numa-bind --kv-transfer-config \{\"kv_connector\":\"SimpleCPUOffloadConnector\"\,\"kv_role\":\"kv_both\"\,\"kv_connector_extra_config\":\{\"cpu_bytes_to_use\":414512578560\,\"cpu_bytes_to_use_per_rank\":51814072320\}\}
```

ns256，`MUSE_OFFLOAD=0`：

```
vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel --max-num-seqs 256 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt \{\"image\":0\,\"audio\":0\,\"video\":0\} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice --max-cudagraph-capture-size 256 --kv-cache-memory-bytes 16957243392
```

ns128，無 tier（預設）：

```
vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt \{\"image\":0\,\"audio\":0\,\"video\":0\} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice --kv-cache-memory-bytes 17357799424
```

ns128，`MUSE_OFFLOAD=1`：

```
vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --host 127.0.0.1 --port 8100 --enable-expert-parallel --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt \{\"image\":0\,\"audio\":0\,\"video\":0\} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice --kv-cache-memory-bytes 17257267200 --numa-bind --kv-transfer-config \{\"kv_connector\":\"SimpleCPUOffloadConnector\"\,\"kv_role\":\"kv_both\"\,\"kv_connector_extra_config\":\{\"cpu_bytes_to_use\":414512578560\,\"cpu_bytes_to_use_per_rank\":51814072320\}\}
```

每個 flag 的作用見第 5 節。不是這六條路徑的 line 或環境就是新的 configuration（第 11b 節：每個 configuration 有 0.4-0.6 % SD 的固定偏移，margin 只有 spec ON 0.4 MiB、ns ON 1.1-1.8 MiB）。

### 4.5 tier ON 的條件與 swap（必讀）

**tier ON 之前必須先通過 `scripts/preflight_offload.sh`**（launch script 自動跑；rc != 0 不啟動）：P* = 51,814,072,320 B per rank 只在 node 7 的 reclaimable slab 低於約 0.3 GiB 時有效。
bytes 與 offload 參數是同一個 launch 屬性：`$MUSE_OFFLOAD_SERVE_ARGS` 必須接在 `--kv-cache-memory-bytes` 之後；帶 offload-on bytes 卻沒有 tier，或 connector 配 no-tier bytes，兩個檢查腳本都會 FAIL。
`offload.env` 的輸入（`MUSE_OFFLOAD_KV_BYTES_OVERRIDE`、`MUSE_OFFLOAD_PER_RANK_BYTES`、`MUSE_OFFLOAD_NUMA_BIND`）要在 profile 的 env 檔之前 export（只有 `--keep-env` 的實驗 launch 能用）。
手動組 line 只是理解用的推導（支援的啟動方式是第 4.1 節的 script，它們多做了 scrub、preflight、檢查與記錄）；env 檔給的值如下，spec（tier ON）與 ns256：

```bash
# speculative
source env/host.env; source env/spec_k7.env              # offload 預設 ON（先 export MUSE_OFFLOAD=0 則為無 tier 的 line）
scripts/preflight_offload.sh                              # ON boot 之前，rc 0
vllm serve ... $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD_SERVE_ARGS
SPEC_EXPECT_FB=1 scripts/check_spec_launch.sh <serve.log>; scripts/check_offload_launch.sh <serve.log>      # 開機後兩個檢查都要 rc 0
# ns256
source env/host.env; export MUSE_OFFLOAD_PROFILE=ns256; source env/offload.env
scripts/preflight_offload.sh
vllm serve ... --max-num-seqs 256 --max-cudagraph-capture-size 256 --kv-cache-memory-bytes $MUSE_OFFLOAD_KV_BYTES $MUSE_OFFLOAD_SERVE_ARGS
scripts/check_offload_launch.sh <serve.log>
```

`$MUSE_OFFLOAD_SERVE_ARGS` 必須接在 `--kv-cache-memory-bytes` 之後（與 gated boot 的 serve line 逐 byte 相同；`MUSE_OFFLOAD=0` 時它是空字串）。
`MUSE_OFFLOAD=0` 的 speculative launch 得到不含 tier 的 line（KV bytes 11,720,212,480、無 connector、無 `--numa-bind`）；手動組 line 時不要 source `offload.env` 來取 no-tier bytes：ns256 / ns128 的 `offload.env` 即使 `MUSE_OFFLOAD=0` 仍 export ON bytes（只有 `spec_k7.env` 會切換 bytes）。

**ns256 RECOMMENDED**（非 speculative，`--max-num-seqs 256`）：`launch/serve_nonspec.sh ns256` 做完所有步驟（ON bytes 16,854,810,624，2,893,281 KV tokens per rank）。
gate 數字（ON vs OFF）：reuse TTFT p50 x0.0481、prompt tok/s x26.19、TTFT p99 x0.0376、118k re-admission x0.026、hit 0.9995；STEADY gmean 0.9956、DEC tok/s gmean 0.992；stress X S_live 501.8 MiB（floor 500）。
**具名偏離（0.32.0 研究）**：這條 line 含 `--numa-bind`，但該研究中 bound 的 ns256 比 unbound 的 median reuse TTFT 慢 9-12 %（25-30 ms），未重新 gate；`MUSE_OFFLOAD_NUMA_BIND=0` 可關。
**ns128 offload profile**（shipped，**預設 OFF**：S1-10R 的規則只憑 X stress 就讓 ns128 作為 profile 出貨，沒有 A/B，是 orchestrator 的決定；`launch/serve_nonspec.sh ns128` 預設不帶 tier，`MUSE_OFFLOAD=1` 才開）：`MUSE_OFFLOAD_PROFILE=ns128`，`--max-num-seqs 128`，ON bytes 17,257,267,200（2,962,366 KV tokens per rank），S_live 501.1 MiB（floor 500）。舊 launch（無 tier）：用 `launch/serve_spec.sh` / `launch/serve_nonspec.sh <ns128|ns256>` 加 `MUSE_OFFLOAD=0`，no-tier bytes 在 `env/launch.env`（ns256 16,957,243,392、ns128 17,357,799,424）。手動組 line 時不要 source `offload.env` 來取 no-tier bytes：ns256 / ns128 的 `offload.env` 即使 `MUSE_OFFLOAD=0` 仍 export ON bytes（只有 `spec_k7.env` 會切換 bytes）。

**Swap（必讀）**：gate 中 14 / 18 個 ON 類 boot 發生 swap（launch 到 READY 增加 3.8-7.4 GiB），16 / 16 的 OFF / CTL 類 boot 沒有；swap 峰值 7,444.8 MiB / 8,192 MiB；adoption 的 gate-T boot 只有 45.6 MiB。
所以 ON launch 不是 swap-free，也不保證每次 boot 都不 swap；c128 p99 TTFT 在六組中都比 OFF 高 3.5-7.7 %；spec 的 stress S_live 餘裕只有 1,256.4 vs 1,256 MiB。需要 vLLM branch 的 `receives_remote_kv` 修正。S1-11 的 matrix 中 ns256 ON 在 3 次 boot 有 2 次在 READY 時已 swap（spec ON 1 次沒有）；沒有找到可預測 swap 的量（第 11 節）。

以下效能結果是 0.32.0 實測（45 GiB tier），0.33.0 的 gate 數字見上；結果（real weights、spec profile）：64k / 118k prompt 的 re-admission TTFT 6.30 / 12.44 s -> 0.29 / 0.47 s；比 GPU pool 大 1.6 倍的 working set 循環讀取時 TTFT p50 5.9 -> 0.24 s、throughput 10.9k -> 217k prompt tokens/s；
沒有 reuse 時 decode -0.5 %（三次 boot 的 gmean）。native `OffloadingConnector` 已量測、不採用（每個 worker pin 整個 region：boot +100 s .. +21 分鐘、GPU page table 8 倍、decode -1.2 .. -2.1 %）。

## 5. Exact bytes、tier 公式與每個 flag 的作用

**Exact bytes 與 tier 公式**（數字只存在 `env/spec_k7.env`、`env/offload.env`、`env/launch.env`）：

| profile | tier | `--kv-cache-memory-bytes` | blocks per rank（bytes per block-id） | KV tokens per rank | S_live（floor）MiB | S_live 的來源 |
|---|---|---:|---:|---:|---:|---|
| spec | OFF（`MUSE_OFFLOAD=0`） | 11,720,212,480 | 572,276（20,480） | 1,649,935 | 1,278.4（1,256） | S1-13 boot |
| spec | ON（預設） | 11,645,337,600 | 568,620（20,480） | 1,639,394 | 1,256.6（1,256） | S1-10R phase 3 確認 boot |
| spec | ON（預設） | 11,645,337,600 | 568,620（20,480） | 1,639,394 | 1,256.4（1,256；餘裕 0.4） | S1-10R gate 的 X boot |
| ns256 | OFF（`MUSE_OFFLOAD=0`） | 16,957,243,392 | 258,747（65,536） | 2,910,864 | 500.8（500） | 0.33.0 validation boot |
| ns256 | ON（預設） | 16,854,810,624 | 257,184（65,536） | 2,893,281 | 501.8（500） | S1-10R phase 3 / gate X boot |
| ns128 | OFF（預設） | 17,357,799,424 | 264,859（65,536） | 2,979,623 | 503.1（500） | 0.33.0 validation boot |
| ns128 | ON（`MUSE_OFFLOAD=1`） | 17,257,267,200 | 263,325（65,536） | 2,962,366 | 501.1（500） | S1-10R phase 3 / gate X boot |

spec ON 的兩個 S_live 來自兩次不同的 run，不合併。餘裕很薄（spec ON 0.4 MiB、ns ON 1.1-1.8 MiB）：不是這六個 launch 的 serve line 或環境就是新的 configuration（第 11b 節）。ON bytes = no-tier bytes 依 tier 的 GPU page table（每註冊 1 GiB 為 2 MiB，P* 時 96.5 MiB）造成的 S_live 不足量往下調（2 MiB 為單位），
用 S1-7 stress 程序確認（stress 會讓 server 在 S_live 的邊界上跑，量到的 S_live 必須不低於 floor）。每個 rank 的 tier：

```
P* = L x floor((min_N A_N - M) / (2 L))      L = 327,680 B（兩個 profile 的 bytes per block-id 20,480 與 65,536 的最小公倍數），M = 6.6 GiB
A_N = MemFree_N + Active(file)_N + Inactive(file)_N（每個 GPU node，0.33.0、server 運作中量測，MiB）：node 3 119,262.8、node 0 118,644.1、node 7 105,586.3（限制者）、node 4 119,641.2
P* = 327,680 x 158,124 = 51,814,072,320 B per rank（48.2556 GiB）；cpu_bytes_to_use_per_rank = P*；cpu_bytes_to_use = 8 x P* = 414,512,578,560 B
tier blocks per rank：2,529,984（spec）、790,620（ns256 / ns128）
```

P* 只在 node 7 的 reclaimable slab 低於約 0.3 GiB 時有效（量測時為 302 MiB），所以每次 ON boot 之前跑 `scripts/preflight_offload.sh`（launch script 會跑）。

**每個 flag 的作用**（serve line 由 `env/*.env` 與 `launch/lib.sh` 組出，見第 4 節）：

| flag | 作用 |
|------|------|
| `--tensor-parallel-size 8 --enable-expert-parallel` | TP8 attention + EP8 MoE（tokens replicated，沒有 all2all） |
| `--max-model-len 131072` | context 長度 |
| `--gpu-memory-utilization 0.95` | 一般情況下決定 KV pool；給了 `--kv-cache-memory-bytes` 之後 KV 大小由該值決定（vLLM 的 `CacheConfig` 在 `kv_cache_memory_bytes` 不是 None 時忽略它），此旗標仍留在 line 上以與 gated boot 逐 byte 相同 |
| `--kv-cache-memory-bytes N` | 每張 GPU 的 exact KV bytes（上表；與 offload 參數是同一個 launch 屬性，`check_kv_launch.sh` / `check_offload_launch.sh` 檢查配對） |
| `--max-num-seqs 128 / 256` | 同時 running request 上限：spec 與 ns128 為 128，ns256 為 256 |
| `--kv-cache-dtype fp8`、`--quantization modelopt_fp4` | fp8 KV、NVFP4 權重 |
| `--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'` | text-only launch（所有 modality 設 0；microbatched prefill overlap 需要它） |
| `--reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice` | chat API 的 reasoning / tool-call 解析 |
| `--host 127.0.0.1 --port 8100` | 只聽 loopback |
| `--speculative-config {...}`（spec only） | RadixArk DFlash2 drafter：method `dflash`、`num_speculative_tokens` 7、`draft_sample_method` probabilistic、`rejection_sample_method` block |
| `--per-request-spec-decode-metrics detailed`（spec only） | 每個 request 的 acceptance 統計（AL 的來源） |
| `--max-cudagraph-capture-size 256`（spec 與 ns256） | FULL decode CUDA graph 的 capture 上限 256 tokens；ns256 必須明確給（否則 SP floor 變 513）；ns128 沒有此旗標（vLLM 預設 list） |
| `--max-num-batched-tokens 8199`、`--max-num-scheduled-tokens 8192`（spec only） | K=7 時 prefill chunk 對齊成 8192 tokens（S1-9 L5）；0.33.0 的 DFlash draft slots 為 0，MNBT 8199 單獨會排出 8199-token 的 chunk，所以 MNST 把 chunk 壓回 8192（S1-14 F2；少了它 32k 冷 prefill 約慢 2.7 %（S1-14 A/B 實測：32k 冷 wall 3.0325 s 對 2.951 s；64k 約 2.6 %、reuse pass 1 的冷 prefill tok/s 10,747.3 對 11,055.1，+2.9 %）） |
| `--numa-bind` | 每個 worker 在 `numactl --cpunodebind=N --membind=N` 下執行（N = 該 GPU 的 node），使 tier 100 % 在自己 GPU 的 node 上 |
| `--kv-transfer-config '{...}'` | `SimpleCPUOffloadConnector`、`kv_role` `kv_both`；`cpu_bytes_to_use_per_rank` = P*，`cpu_bytes_to_use` = 8 x P*（tier ON 才有） |

## 6. 最終結果（final matrix、tier 成本、gates）

**範圍與出處。** matrix 與 gates 跑在 production tree 上：host main `f540ce1b37` + vLLM `sm120-muse-spark-open` `f6a60ab9d7`，2026-11-13 02:36:49 -> 07:11:47（4 h 35 min、13 個 hold、7 次 boot、216 個 cell）。
每次 boot 都經 `launch/serve_*.sh` 啟動，並通過 `f540ce1b37` 的開機後檢查（第 4.3 節；不含之後才加入的停止路徑 fallback 與 `kv_connector` / `kv_role` / `kv_connector_extra_config` 的 serve-line 檢查）。數字由分析 script 逐行複製、經獨立重算驗證（216 個 cell、4237 個數值比較、0 個差異）。
這是量測，不是 adoption 決定。matrix 跑完之後的 commit 只改 launcher 的停止路徑、serve line 檢查、測試與文件，不改任何 export、serve line 或預設值。
每個 cell 的完整 grid（default sampling、top_p 0.95、rand / nat8k 的 TTFT 與 TPOT、churn 的 TTFT、各 cell 的 AL、cache source、swap）不在 repo 裡保留；這裡只有下面四張主表、冷 TTFT、reuse 與 gates。

**量測 protocol（表中每個 metric 的意義）。**

* STEADY：固定 batch 的 driver，值 = 各 window step time 的 upper median；tok/s 是由 step 推得（c x 1000 / step x tokens per step），沒有 TTFT。spec `--win 12 --windows 3`、ns128 `--win 10 --windows 3`、ns256 `--win 12 --windows 2`，各與其 baseline 的 protocol 相同。
* DEC：併發 driver，gsm80 + mtbench 的真實 prompt，n = max(16, 4c)，128 output tokens，每個 request 一個 `cache_salt`；greedy 重複 2 次（表中為平均），default sampling 與 top_p 0.95 各 1 次。rand 512/128：隨機 prompt。natural 8k：自然文 8,192 tokens。mixed churn：S1-9 L2，256 tokens，jitter 24，stagger 0.15 s（c1 不是 churn cell）。
* TTFT / TPOT 的 percentile 定義依 driver 而不同：conc / rand / nat8k 用 index `round(p x (n - 1))`（偶數 n 的 p50 = upper median），reuse driver 用 `ceil(p x n) - 1`（81 個 request 的 p99 = 最大值）。
* 每個 non-reuse cell 都有 per-request salt，並記錄 `prompt_tokens_cached_by_source` 的差值：216 個 cell 全部有效，0 個 invalid、0 個失敗 request、0 次 preemption。reuse / goff / long cell 依設計有 tier hit，標為 reuse。
* swap 不使 cell 無效：每個 cell 都帶它那次 boot 的 swap 狀態（見下方「已知限制」）。

**最終 matrix（output tok/s、ms；欄位 = concurrency；數字由分析 script 逐行複製，沒有重算）。**

spec ON（`launch/serve_spec.sh` 預設，tier ON）：

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 |
|---|---:|---:|---:|---:|---:|---:|
| STEADY tok/s (derived) | 402.9 | 737.2 | 1395.8 | 1935.4 | 2813.8 | 3752.0 |
| DEC greedy out tok/s | 221.4 | 416.5 | 756.6 | 961.9 | 1252.2 | 1479.1 |
| DEC greedy TPOT ms | 3.49 | 7.30 | 15.19 | 23.24 | 36.64 | 61.86 |
| DEC greedy TTFT p50 ms | 154 | 216 | 462 | 644 | 1033 | 1746 |
| DEC greedy TTFT p99 ms | 179 | 462 | 1419 | 2544 | 4361 | 8875 |
| DEC greedy AL | 4.671 | 4.639 | 4.872 | 4.880 | 4.830 | 4.841 |
| rand 512/128 out tok/s | 148.5 | 283.1 | 578.5 | 768.3 | 1008.6 | 1272.2 |
| natural 8k out tok/s | 92.0 | 130.2 | 156.1 | 168.9 | 175.2 | 177.2 |
| mixed churn out tok/s | - | 498.5 | 918.3 | 1201.0 | 1567.4 | 1944.9 |

spec `MUSE_OFFLOAD=0`（reference，同一個 chain 的 OFF boot）：

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 |
|---|---:|---:|---:|---:|---:|---:|
| STEADY tok/s (derived) | 410.2 | 724.8 | 1433.4 | 1966.4 | 2801.9 | 3757.0 |
| DEC greedy out tok/s | 224.5 | 427.0 | 752.0 | 975.8 | 1254.3 | 1481.8 |
| DEC greedy TPOT ms | 3.43 | 6.94 | 15.23 | 22.86 | 36.63 | 61.93 |
| DEC greedy TTFT p50 ms | 154 | 233 | 467 | 646 | 1063 | 1748 |
| DEC greedy TTFT p99 ms | 178 | 461 | 1392 | 2586 | 4354 | 8804 |
| DEC greedy AL | 4.671 | 4.710 | 4.791 | 4.946 | 4.838 | 4.839 |
| rand 512/128 out tok/s | 150.3 | 284.6 | 581.5 | 772.7 | 997.8 | 1269.5 |
| natural 8k out tok/s | 93.0 | 130.4 | 156.4 | 169.0 | 175.4 | 177.3 |
| mixed churn out tok/s | - | 501.6 | 924.0 | 1206.3 | 1572.7 | 1922.1 |

ns256 ON（預設，tier ON；c256 欄只有 ns256）：

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 | c256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEADY tok/s (derived) | 122.1 | 294.9 | 690.4 | 1023.5 | 1599.3 | 2281.6 | 3070.6 |
| DEC greedy out tok/s | 115.2 | 263.5 | 535.5 | 722.6 | 980.9 | 1212.2 | 1432.2 |
| DEC greedy TPOT ms | 7.73 | 12.28 | 21.95 | 30.54 | 41.86 | 72.89 | 133.05 |
| DEC greedy TTFT p50 ms | 158 | 366 | 935 | 1640 | 2980 | 3824 | 4072 |
| DEC greedy TTFT p99 ms | 169 | 447 | 1372 | 2503 | 4215 | 8316 | 15842 |
| rand 512/128 out tok/s | 113.3 | 263.3 | 534.3 | 736.3 | 1011.1 | 1295.0 | 1543.8 |
| natural 8k out tok/s | 73.8 | 117.0 | 150.1 | 163.4 | 173.7 | 181.4 | 181.5 |
| mixed churn out tok/s | - | 278.1 | 557.4 | 740.6 | 1013.5 | 1353.6 | 1745.5 |

ns128 OFF（預設，無 tier）：

| workload / metric | c1 | c4 | c16 | c32 | c64 | c128 |
|---|---:|---:|---:|---:|---:|---:|
| STEADY tok/s (derived) | 123.1 | 302.6 | 700.4 | 1036.0 | 1592.7 | 2302.2 |
| DEC greedy out tok/s | 116.1 | 264.7 | 540.5 | 726.9 | 980.4 | 1232.8 |
| DEC greedy TPOT ms | 7.67 | 12.20 | 21.67 | 30.28 | 42.67 | 72.18 |
| DEC greedy TTFT p50 ms | 157 | 366 | 928 | 1679 | 2762 | 3822 |
| DEC greedy TTFT p99 ms | 168 | 456 | 1384 | 2479 | 4151 | 7797 |
| rand 512/128 out tok/s | 114.2 | 265.0 | 538.7 | 745.1 | 1007.7 | 1303.5 |
| natural 8k out tok/s | 74.3 | 117.4 | 150.7 | 163.7 | 174.0 | 181.8 |
| mixed churn out tok/s | - | 280.2 | 563.9 | 744.6 | 1015.5 | 1354.2 |

冷 TTFT（client wall，`max_tokens 1`、新的 salt、5 次取中位數；= 8k / 32k / 64k / 118k tokens 的冷 prefill TTFT，秒）：

| prompt tokens | spec ON s | spec OFF s | ns256 ON s | ns128 OFF s | spec ON - OFF ms | spec OFF - ns128 OFF ms | ns256 ON - ns128 OFF ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 0.7430 | 0.7286 | 0.7143 | 0.7082 | +14.5 | +20.3 | +6.1 |
| 32768 | 2.9749 | 2.9633 | 2.9222 | 2.9143 | +11.6 | +49.1 | +8.0 |
| 65536 | 6.2794 | 6.2657 | 6.2002 | 6.1880 | +13.7 | +77.7 | +12.2 |
| 117964 | 12.4243 | 12.4090 | 12.2857 | 12.2625 | +15.3 | +146.5 | +23.2 |

prefix-reuse workload（frozen reuse driver，cycle，32k 文件，working set 約 1.61-1.62 倍 GPU pool，4 個 client）：

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

**與 baseline 的比較。** 規則：比同一 launch 的 0.33.0 baseline 差超過 3 % 的 cell 要先調查。結果：**0 個 PRIMARY（protocol 相符）的列超過 3 %（最差 +1.16 %）**；3 個相符的 SECONDARY 列超過 3 %（spec dec.c1 兩列 3.44 %、spec_off AL c1 3.81 %），
AL 解釋了差距：16 個 replay prompt 的 AL 為 4.671 對 validation 的 4.856（232.5 x 4.671 / 4.856 = 223.6，量到 224.5）；spec c1 的 primary baseline 是 224.3 還是 232.5 仍是 orchestrator 未決的讀法（以 232.5 為準時 spec c1 DEC 兩列差 3.44 %）；2 列 protocol 不符（7.5 %，只列在旁邊、不算 flag）。
protocol 相符的 primary baseline 只有 29 列：rand512、nat8k、churn、default sampling、top_p、c4 / c32 / c256 的 DEC、65k / 118k 的 wall 與所有 TTFT / TPOT 都**沒有** 0.33.0 baseline，所以「0 個 primary flag」不代表每個 cell 都在 3 % 內。
chain 的 13 個 non-spec STEADY cell 沒有被 parse（`parse_result` 的 `TypeError`），由分析重新從原始 window 值算出（15 個額外的 baseline 列，最差 +0.67 %）。
tier 的成本（spec ON 對同一 chain 的 OFF boot）：沒有任何 throughput cell 比 OFF 差超過 2.6 %（STEADY step 最多 +1.6 %）、冷 wall +11.6..+15.3 ms（+0.1..+2.0 %）、冷 prefill pass 1 11,059.6 對 11,048.6 tok/s；reuse 的 pass 2 / 3 為 18.5 倍 / 18.3 倍 prompt tok/s。
ns256 ON 對 ns128 OFF（跨 profile，非同一 chain 的 OFF）最多低 2.6 %。與 campaign 開頭與較早 production 的對照（protocol 不同，只當背景）：non-spec 的 random 512/128 比 campaign 開頭的 sweep 高 9.6..18.9 %、比 S1-6 結束時的 production 高 1.2..4.0 %；speculative c1-c32 與先前的 RadixArk K=7 結果在 -0.8..+3.6 % 內。

**Gates：9 / 9 PASS（FAIL 0、ARTIFACT 0、NA 0）。** 判定標準在 chain 開始前固定。逐行：

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

gates 的**具名偏離**（結論只有這份清單允許的強度）：

* Q2：G-det 與 D4' 是以綁定到 serve line 的 offline in-process `LLM` 執行（環境與 launch dry run 相同，`E2E_*` 取自 serve line），不是經 launch script；G-det 比較兩個 cold process（各有自己的 session 與 weight load，先後執行）。
* Q5：「eviction 之後 restore」是 1.3 倍 pool 的 flood 造成的自然 GPU eviction（spec 66 份文件 / 192 s、ns256 115 份 / 330 s）；**tier 自己也在 evict 的 full-tier regime 沒有重新涵蓋**（S1-10R gate T 的範圍說明不變：D4' 只涵蓋非滿載 regime 的 byte 相等）。
* M3：G-off 的 OFF / eviction leg的 driver 是新寫、未凍結的程式；D4' leg 用凍結的工具。
* NR1：ns256 的 G-off / G-long 在專用的全新 boot（`ns256_goff`、`ns256_off`、`ns256_long`）上判定：ns 線沒有 dev endpoint，start reset 回 404，改以 cached-token counters 與 `prompt_tokens_total` 在第一個 request 前為 0 作證據；G-off/ns256 是在 swap 的 boot 上取得的。
* GSM8K 把撞到 16,384 output tokens 上限的 request 算錯（沒有最後答案），已含在上面的總數中（spec greedy 1 + 3、ns256 greedy 2 + 4、default sampling 0 + 0 個）。
* D4'（b）的 170 項中 4 項（index 15 與 59，greedy 與 `t1k64`）在 C、H、X 三種狀態都是 cached = 0（沒有 tier hit），所以「170/170 相等」成立，但只有 166 項是 hit 狀態的比較；spec 的 D4' verdict 有一則 warning（draft KV group 14 restored blocks 2324），沒有 fail 或 artifact 原因。block 數（59,100 / 15,356）是凍結 checker 的 verdict 明細，不是重新由 KV hash 算出。

## 7. fp8 KV 在 max-model-len 1,048,576（S2-0，量測；production 維持 131072）

**範圍。** 這是量測，不是 production 設定：`env/launch.env` 固定 `LAUNCH_MAX_MODEL_LEN=131072`，`launch/lib.sh` 沒有 max-model-len 的 hook，`scripts/check_kv_launch.sh` 也會拒絕 1M 的 bytes，所以**沒有 1M 的啟動指令**。
1M 的 boot 是 production 的 dry-run serve line，**只換三個欄位**：`--max-model-len 1048576`、`--kv-cache-memory-bytes`（下表）與 tier 大小（`cpu_bytes_to_use`、`cpu_bytes_to_use_per_rank`，第 7.3 節）。
haystack = 1,044,480 tokens（1M 的 99.6 %）、fp8 KV、spec 與 ns256 兩個 profile 都是 tier ON。

### 7.1 判定（依 spec 的規則）

規則：FEASIBLE 要求 (a) 在 exact bytes 開機且每個開機後檢查通過、(b) M4（三根 needle）PASS 且 invalid 清單為空（cold q0、calibration OK）、(c) M3 tier round-trip bitwise 相同。

| profile | 開機（exact bytes + 檢查） | M4 | M3 | 判定 |
|---|---|---|---|---|
| spec（DFlash2 K=7，tier ON） | 是 | PASS 3/3，invalid 為空 | bitwise 相同（64 tokens） | **FEASIBLE** |
| ns256（tier ON） | 是 | needle 3/3 exact，但 cell **invalid**（calibration 的 chat call，見下） | bitwise 相同（64 tokens） | **依規則不可行**；失敗步驟只有 M4 的 invalid |

ns256 依規則不可行的**唯一原因**是 M4 的 calibration：driver 在送出 209-token 的 short-needle chat 之後，伺服器回 HTTP 200（`usage.prompt_tokens` 209），但 `message.content` 為 null，driver 在
`content[:80]` 當掉並丟掉回應（finish_reason、reasoning 與 completion_tokens 沒有被記錄）；同一個呼叫在 spec 回 "1234567"。從 serve log 推估 ns256 這次生成約 1,000 tokens（碰到 1,024 的上限而沒有 final channel），spec 約 150 tokens；
tokenization、parser flag 與 1M KV 都已排除。**ns256 的 1M retrieval 證據是正的**：10 / 50 / 90 % 深度的三根 needle（9037972、4624119、8963259）都 exact、與 spec 的答案相同，
cold prefill 與 GPU-cached decode 都正確，host tier round trip bitwise 相同；沒有 1M model / KV 失敗的跡象。短 prompt 為何在 non-speculative launch 花約 1,000 tokens 仍未解（要重跑這一個 call 並記錄回應才能決定）。判定維持規則的結果。

### 7.2 Exact 1M GPU bytes（每個 rank）

S_live = READY 後最小的 idle free - stress 期間 allocator 的 live growth（S1-7 程序 + 1M 階段）。每個 stress boot：0 errors、0 mismatches、0 preemptions、0 OOM、health 200、所有 1M request 都是 cold。

| profile | tier | `--kv-cache-memory-bytes` | blocks | KV tokens（pool 是 1M 的幾倍） | S_live MiB（floor） |
|---|---|---:|---:|---:|---:|
| spec | OFF | 9,489,448,960 | 463,352 | 1,767,804（1.69x） | 1,320.8（1,256） |
| spec | ON | **9,388,769,280** | 458,436 | 1,749,048（1.67x） | 1,320.5（1,256） |
| ns256 | OFF | 15,833,628,672 | 241,602 | 3,671,832（3.50x） | 661.8（500） |
| ns256 | ON | **15,732,965,376** | 240,066 | 3,648,488（3.48x） | 661.8（500） |

ON bytes = OFF bytes 減掉 tier 的 GPU page table（spec 100,679,680 B、ns256 100,663,296 B，皆為 96 MiB 的整 block 數）。每個 tier 只需一次 stress boot（第一次就 S_live >= floor）。
spec 在 allocator 邊界的 min free 為 11.4 MiB（tier ON）、ns256 17.4 MiB，沒有 OOM；ns256 的 S_live 比 floor 高 161.8 MiB，bytes 比最小值高約 160 MiB per rank（沒有啟用 step-up boot）。

### 7.3 Tier 大小（local RAM）

公式同第 5 節：P*_1M = L x floor((min_N A_N - M_1M) / (2 L))，L = 327,680 B。1M server 的 A_N（MiB per node，1M server 在 workload 之後，限制者都是 node 7）：spec 118,455.4 / 117,979.2 / **104,187.9** / 118,801.0（node 3 / 0 / 7 / 4），
ns256 118,391.9 / 117,488.1 / **105,061.9** / 118,781.0。M_1M = 6.65 GiB（spec）/ 6.80 GiB（ns256）。1M 才有的 host 項（pinned 的 `all_token_ids`，每個 worker 512 MiB spec / 1,024 MiB ns256）在 worker 初始化時配置，所以已在 A_N 裡、不另加到 M。

| profile | P*_1M per rank | cpu_bytes_to_use（x8） | CPU blocks per rank | tier 內放得下的 1M prefix |
|---|---:|---:|---:|---:|
| spec | **51,054,182,400**（47.5479 GiB） | 408,433,459,200 | 2,492,880 | 9 |
| ns256（計算值） | 51,431,669,760（47.8995 GiB） | 411,453,358,080 | 784,785 | 11 |
| ns256（實際開機的 tier） | **50,798,592,000**（47.31 GiB） | 406,388,736,000 | 775,125 | |

ns256 在計算值 P*_1M 上的 strict preflight 被拒絕：node 7 的 LEFT 7,805 MiB - 1M host 項 1,792 MiB = 6,013 MiB < M_1M 6,964 MiB，**不足 951 MiB**（node 7 在 A_N 快照之後又掉了約 0.95 GiB，P*_1M 本來就是貼著邊界算的）。
step-down：(951 / 2 + 128) = 603.5 MiB per rank -> 1,932 L = 633,077,760 B，實際 tier 為 50,798,592,000 B（margin：b2b 249 MiB、B4 51 MiB；spec B4 的 margin 是 19 MiB）。ns256 所有的 GPU bytes、S_live 與量測都是在這個 tier 上取得。
**限制**：ns256 計算值 51,431,669,760 取決於 A_N 的 dip 視窗（整個 run 的最低點在 start-up 期間；用 READY 之後的視窗，dip 為 496.9 MiB、M = 6.7 GiB、P*_1M = 51,485,409,280，+53,739,520 B）；實際開機的 tier 與 spec 不受影響。

### 7.4 可同時容納的 1M 並行數

* spec：**1**。由 probe 與 block 算術得出（每個 1M request 需要 262,795 blocks，pool 458,436 blocks）；N=2 的 probe 沒有 preemption，第二個 request 排隊（wall 591.4 s，TTFT 591.4 / 296.1 s）。stress 的原始欄位在沒有 ladder 的 spec 上讀到 0。
* ns256：**3**（ladder 的上限，不是量到的極限；rung 1 / 2 / 3 都乾淨，wall 296.4 / 589.8 / 885.3 s；N=4 的 probe 沒有 preemption、peak running 2、wall 1,177.2 s）。三個 1M context 同時在 ns256 常駐時 KV usage 82.1 %。

1M 的 prefill 是串行的（每個 request 佔滿整個 8,192-token budget），所以並行的 1M request 依序完成（295 / 589 / 883 s）。

### 7.5 速度

| 項目 | spec | ns256 |
|---|---|---|
| **cold 1M prefill TTFT**（2 次） | **296.1 / 295.6 s**（3,528 / 3,533 tok/s） | **294.9 / 294.9 s**（3,542 tok/s） |
| 每個 8,192-token chunk 的時間 | 0.735 s（深度 0）-> 3.631 s（約 1M 深度），約 3 ms per 1k 深度（總成本是深度的平方），兩個 profile 在每個深度都相同 | 同左 |
| **decode，約 1M 深度，單一 stream** | TPOT 3.00 ms = **333.4 tok/s，AL 7.125**；每個 verify step 21.3 ms | TPOT 12.06 ms = **82.9 tok/s** |
| 相對深度 0 的 step | 21.3 ms 對約 17.2 ms：每個 step 約多 4 ms（+24 %） | 12.06 ms 對 8.19 ms：每個 step 約多 4 ms |
| 最大並行的 decode | n/a（admitted 1） | N=3：aggregate 133.8 tok/s（每個 stream 22.4-22.7 ms；單 stream 的 1.61 倍） |
| GPU-hit TTFT / **tier restore TTFT** 的 1M prefix | 1.52 s / **3.28 s**（x2.16） | 0.89 s / **2.35 s**（x2.63） |
| tier restore 的內容 | 與 GPU-hit bitwise 相同（64 tokens） | 同左（flood 153 份 x 32,768 = 5,013,504 tokens 造成 eviction） |
| 8k 的 short（c4）在 cold 1M prefill 3 s 後送入：TTFT | 約 294-295 s（control 無 prefill 時 0.777 s） | 約 293 s（control 2.032 s） |

* **333.4 tok/s 是 acceptance 被抬高的數字**：它是在 pile 文章的續寫上量到的 AL 7.125（接近 K+1 = 8 的上限）；同一個 server 上其他 cell 的 AL 是 4.85（M4 needle 文字）、5.62（M3）、2.66（M5 control），production matrix 的 AL 是 4.64-4.88。
  以 production 的 AL 約 4.7 與同樣的 21.3 ms step 推算，1M 深度的 spec decode 約 **220 tok/s**（4.7 / 21.3 ms，**推算值，不是量測**）。
* 深度的成本是 attention 讀 KV：decode step 在 1M 深度每個 step 多約 4 ms（spec 21.3 vs 約 17.2 ms、ns256 12.06 vs 8.19 ms；約 17.2 ms 是 c1 的 production step，8.19 ms 是 final matrix 的 ns256 c1 STEADY step）。
  事前的估計是 17.2 ms + 2.79 ms 的 KV 讀取（+16 %）；實測 +24 %。cold prefill 的事前估計是 267 s（fit），實測 296 s，超過整個 95-267 s 的預測範圍；non-spec 只快約 1 s，draft prompt 的額外成本看不出來。
* **head-of-line blocking**：1M 的 prefill 每個 step 用滿整個 8,192 / 8,199-token budget，排在它後面的 short 要等到整個 prefill 結束才有第一個 token（TTFT 約 294-295 s，約 control 的 380 倍（spec）/ 145 倍（ns256））。
  n = 4 的 p50 取的是上中位數（294.8 / 293.5 s）；兩個中間值的平均為 294.43 s（spec）/ 293.16 s（ns256）。short 的 TPOT 是 prefill 結束後的 c4 decode，不是 prefill 期間的 decode 速度。
* chunk 時間曲線是由 4 Hz 的 GPU KV-usage gauge 事後推導的（vLLM 的 prompt-token counter 與 iteration histogram 只在 request 結束時才動，sampler 直接讀到的 tok/s 與 ms per chunk 是 artifact）。
* M3 的 bitwise 相同是 driver 的 verdict（driver 比較 token ids 與 logprob 值；原始向量沒有保留，獨立驗證也沒有重算）；cold 對 GPU-hit 的 logprob 在兩個 profile 都不相同（top_logprobs index 0 不同，僅供參考；規則比的是 tier-hit 對 GPU-hit）。

### 7.6 資源與 swap

GPU 總共 97,887 MiB；READY 時 spec 94,095 MiB、ns256 95,359 MiB（idle free 3,158 / 1,894 MiB）；開機（load）約 100 s / 95 s。node 7 是最緊的 node（B4 結束時 free 5.3 GiB spec / 6.0 GiB ns256）。
swap（報告、不作廢）：spec 的 stress boot（tier ON）在配置 tier 時換出 5.4 GiB（peak 5,462.6 MiB，pswpin 439,679 / pswpout 1,427,213 pages），stress 期間約 4.1 GiB 留在 swap，結束時 58.8 MiB；同樣 bytes 與 tier 的 B4 boot 只有 pswpin 2 / pswpout 37（ns256 B4：3 / 0）。原因沒有調查。
沒有任何 boot 出現 OOM。

### 7.7 與事前估計的差異

事前估計 spec 的 KV bytes 9,730,621,440（實測 OFF 9,489,448,960：1M 的額外成本比估計大約 230 MiB per rank）、ns256 15,833,637,411（實測 OFF 15,833,628,672）；事前的 P* 估計是 spec 51,344,506,880、ns256 50,874,613,760，實測 P*_1M 為 51,054,182,400（-0.57 %）/ 51,431,669,760（+1.10 %，實際開機 50,798,592,000）。

## 8. 設定一覽與 accuracy class

accuracy class 的定義（對照目前 branch 的輸出）：

* `identical`：bitwise 相同。預設 ON。
* `more_accurate`：每個元素都可證明更接近精確數學結果（例如 fp32 固定順序加總、單次最終 rounding，取代一串 bf16 rounding）。預設 ON。
* `deterministic_different`：run-to-run 可重現、各 TP rank 相同，但 rounding 不同且不能證明更好。原先 opt-in、預設 OFF；使用者 2026-10-24 的吞吐量決策（「rounding 差異只要不影響最終 sampling 就接受」）之後，只要更快、確定性、rank 相同、沒有系統性偏差且 GSM8K 沒有退步，就可以預設 ON：本資料夾的 bf16 exchange 與 NCCL graph file 由 `env/host.env` 預設開啟（plugin 本身的預設仍是 OFF）。
* `nondeterministic`：不出貨。

| 設定 | 預設 | accuracy class | 說明 / 量測 |
|------|------|----------------|-------------|
| `NCCL_P2P_LEVEL=SYS`（`env/host.env`） | ON | identical | NCCL 2.29.7 在本機 graph search 失敗，4 個跨 socket hop 走 host SHM；`SYS` 讓所有 hop 走 P2P，ring / channels / protocol 不變。129 個 shape（eager + graph）與 end-to-end 20/20 requests bitwise 相同；128 MiB all-reduce 12.04 -> 9.11 ms（2026-10-21）。EP8 dummy（同一棵 tree，只差這個變數）：c1 3700 / 42.2 -> 4318 / 47.0 t/s（pp +16.7 %、tg +11.4 %），ctx 列 pp +15 .. +17 % |
| `MUSE_TOPO_ENABLE=1`（`env/host.env`），all_reduce 與 reduce_scatter | ON | more_accurate | 見第 9 節。算術：`P_I = ((x_a + x_b) + x_c) + x_d`（fp32，island 內 rank 升冪），`out = round(P_0 + P_1)`；每個 size、演算法、rank、run 之間 bitwise 相同（batch invariant） |
| `MUSE_TOPO_ENABLE=1`，all_gather | ON | identical（純資料搬移） | 直接產生要求的 layout（dim 0 = rank-major；最後一維 = column concatenation），與 NCCL 的結果逐 bit 相同 |
| `MUSE_TOPO_LARGE=0` | 1（large 演算法 ON） | 不影響數值 | A/B 開關：all-reduce > 8 MiB、reduce_scatter、all-gather shard > 1 MiB 改走 NCCL（即 H2 版行為） |
| `MUSE_TOPO_OPS` | `all_reduce,all_gather,reduce_scatter,boundary` | 視 op 而定 | 指定哪些 op 走 topo；`boundary` = 下一列的 fused kernel |
| `MUSE_TOPO_FUSED=0`、`MUSE_TOPO_BND_LOADS` | fused ON、loads 0 | identical（前提：`MUSE_TOPO_MEDIUM_BX=0`；medium BX 開啟時見下） | decode（T <= 8，pooler gather T <= 16）時把 attention-out all-reduce / unpooler all-gather 之後的 residual boundary、pooler all-gather 之後的 pre-expert RMSNorm 當作 collective kernel 的 epilogue（`csrc/boundary.cuh` 是 vLLM Triton kernel 的 bit-exact CUDA 版本，已對 Triton SASS 驗證）。vLLM 端的開關是 `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY`（預設 1）。`MUSE_TOPO_BND_LOADS` 選擇 epilogue 運算元（stream / alpha / beta / weight）的載入位置（0..3，預設 0 最快），不影響數值。**注意**：fused kernel 恆用 exact 的 fp32 exchange；`env/host.env` 預設的 medium BX 開啟時，關掉 fused（`MUSE_TOPO_FUSED=0`、`MUSE_TOPO_OPS` 不含 `boundary`、或 vLLM 端 `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=0`）會讓 T = 4..8 的 attn-out all-reduce（64..128 KiB）改走 two-step 的 `comm.all_reduce`，即 medium BX（bf16 exchange），bits 就不同（deterministic_different，不再 identical）。要做 bit-identical 的 fused 開 / 關 A/B，兩次都要設 `MUSE_TOPO_MEDIUM_BX=0`（fused 開 + `MEDIUM_BX=0` 對 fused 關 + `MEDIUM_BX=0`） |
| `MUSE_TOPO_MAX_BYTES` | 8 MiB（`TopoComm` 預設） | 不影響數值 | 只限制 decode / medium 演算法的訊息大小並決定 workspace 大小；更大的 all-reduce（到 2 GiB）本來就走 large 演算法，沒有理由調高 |
| `MUSE_TOPO_TIMEOUT_S` | 600（plugin） | - | kernel 等不到 peer 時 `__trap()` 的秒數（`TopoComm` 單獨使用預設 10 s；vLLM 啟動時各 rank 可能差數十秒，所以 plugin 用 600） |
| `MUSE_TOPO_LOG_OPS` | 0 | - | 每個 (op, dim, shape, dtype, stream, capturing, path) 第一次呼叫印一行；0 = debug level、1 = rank 0、2 = 每個 rank |
| `MUSE_TOPO_LARGE_BF16_EXCHANGE=1` | plugin 預設 OFF；**`env/host.env` 預設 ON**（2026-10-24 決策；`=0` 回到 exact 的 fp32 exchange） | **deterministic_different** | 大訊息 all-reduce / reduce_scatter 的跨 socket partial 先 round 成 bf16（跨 socket bytes 減半）。只對 bf16；fp16 / fp32 維持 exact。確定性、rank 相同，但不是 fp32 加總單次 rounding 的結果；batch invariant 只在 bf16 exchange 的範圍內：每個 >= `ar_bx_min`（64 KiB，`MUSE_TOPO_MEDIUM_BX=1` 時；`=0` 則是 2 MiB 的 large 門檻）的 bf16 all-reduce，以及所有 reduce_scatter（>= 128 KiB，flag 開啟時一律走 large 演算法加 bf16 exchange），同一 row 在任何 size 下 bits 相同，reduce_scatter 與 all-reduce 從 128 KiB 起 bitwise 相同；較小的 all-reduce 與 T <= 8 的 fused kernel 維持 exact，所以 bits 會在 64 KiB 門檻改變：embedding all-reduce `[T, 8192]`（從不 fused）eager 在 T 3 -> 4、graph 的 padded T 2 -> 4；MoE-out `[T, 4096]` eager 在 T 7 -> 8、padded T 4 -> 8；fused 開啟時 attn-out 在 T <= 8 一律 exact（`all_reduce_boundary` 在 T = 3 與 4 都 exact），T >= 9 起才走 bf16 exchange。final defaults 的驗證（real weights，2026-10-26，其中大訊息 all-reduce / reduce_scatter 走 bf16 exchange）：GSM8K 200 x 2 為 96.5 %（192 / 200、194 / 200；對照 NCCL 96.2 %、plugin 95.5 %、+SP+bf16x 96.2 %），兩個 cold process 的 determinism A/B（56 個 requests 加 in-process 重複，token ids / logprobs / prompt logprobs）bitwise 相同。128 MiB all-reduce 7.10 -> 5.84 ms |
| `MUSE_TOPO_MEDIUM_BX`、`MUSE_TOPO_MEDIUM_BX_MIN_BYTES` | **預設 ON**（1；只在 `MUSE_TOPO_LARGE_BF16_EXCHANGE=1` 時生效）、64 KiB | **deterministic_different** | 同一個 bf16 exchange 用在 medium 的 bf16 all-reduce（64 KiB .. 8 MiB：decode 的 embedding `[T, 8192]`（從不 fused）在 T >= 4（graph 的 padded T >= 4）、attn-out `[T, 8192]` 在 T >= 9（CUDA graph 的 padded T = 16 起；T <= 8 是 exact 的 fused kernel）與 MoE-out `[T, 4096]` 在 T >= 8（padded T >= 8）；演算法 2 的 `BX` 變體，與 large 演算法逐 bit 相同）。跨 socket bytes 減半：1 MiB 62.5 -> 43.1 us、512 KiB 33.3 -> 22.6 us；real weights c16 / c64 / c128 out tok/s +2.2 / +2.9 / +1.9 %；GSM8K 97.5 / 96.5 %。`=0` 回到 exact 的 fp32 exchange（2 MiB 以下）。T <= 8 的 fused attn-out kernel 維持 exact，但只在 fused 開啟時；fused 關閉時 T = 4..8 的 attn-out 會走 BX（見 `MUSE_TOPO_FUSED` 一列，fused A/B 要兩次都設 `MUSE_TOPO_MEDIUM_BX=0`）。`MUSE_TOPO_AR_TUNING_FILE` 是實驗用 hook（這幾個設定在所有 rank 必須相同） |
| `MUSE_TOPO_GEMV`（vLLM 端的對應開關：`VLLM_MUSE_SPARK_SM120_FUSED_GEMV`） | **預設 ON**（1；需要 `MUSE_TOPO_FUSED` != 0 與 vLLM 端 `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=1`；兩邊都要是 1 才會啟用） | **deterministic_different**（對 cuBLAS GEMV）；pooler 在 T >= 2 是 `more_accurate`、T = 1 與 unpooler 相當 | decode（padded T <= `MUSE_TOPO_GEMV_MAX_ROWS`，預設 4，最多 8）時，pooler（`[T, 8192]` -> 512 columns）與 unpooler（`[T, 4096]` -> 1024 columns）的 sharded GEMV 和其後的 all-gather + epilogue 由兩個 kernel 組成一次 collective 呼叫（`csrc/gemv_push.cu`、`ag_bnd.cu` 的 `pushed` 版本）：GEMV kernel 在前一個 all-reduce 還在跑時就把 8 MB 權重載入 register，每個 CTA 算完就把 columns push 給 partner 與 island peers。算術：fp32 固定順序加總、單次 bf16 rounding（`csrc/gemv.cuh`），與 T、rank、run 無關（同一個 token row 在任何 batch size 下 bits 相同），但不是 cuBLAS 的順序。對 fp64 精確值（mixed data）：此順序 pooler 為 99.97 .. 100 % correctly rounded、mean 0.25 ulp、max 0.5 .. 1.2 ulp；cuBLAS 在 T = 2..8（bf16 split-K workspace）只有 58 .. 61 %、mean 1.3 .. 3.5 ulp、max 260 .. 16800 ulp，T = 1 與 unpooler 兩者相當（`bench/test_gemv_push.py` 第 3 段）。與 router GEMV 一起量測見第 9 節。關閉（回到 cuBLAS GEMV + r2d gather）：`MUSE_TOPO_GEMV=0` 與 `VLLM_MUSE_SPARK_SM120_FUSED_GEMV=0` 一起設（只設 vLLM 端只是不提供給 plugin） |
| `MUSE_TOPO_GEMV_TRIGGER` | 3 | 不影響數值 | 前面的 all-reduce kernel 何時讓 GEMV kernel 開始：0 = 不觸發（all-reduce 完成才開始）、1 = all-reduce 一開始（8 MB 權重載入與 all-reduce 第一個 phase 搶頻寬，all-reduce 慢 4 .. 5 us，e2e 沒有增益）、2 = 最後一個 communication phase 開始、3 = 最後一個 communication phase 結束（此時 all-reduce 的 output，以及 `ar2_bnd_kernel` 的 boundary epilogue，還沒寫完）。3 是 S1-4a 實測 arm 的設定；單用 GEMV 時 2 與 3 在雜訊內相同（S1-4a：c1 -2.2 % 兩者相同；c4 -1.1 % 與 -0.1 %，在 repeat 的 1.6 % spread 內）。S1-4b 決定性量測（含 router GEMV，規則：c1 與 c4 TPOT 平均較低者、差距 0.5 % 內選 2）：trigger 3 的 mean(c1, c4) = 12.001 ms，trigger 2 = 12.073 ms，3 低 0.60 %，所以是 3；c1 相同（9.214 / 9.209 ms），差異全在 c4（14.788 / 14.937 ms），而 c4 的 repeat spread（1.4 %）與開頭到結尾的 drift（+1.1 %）一樣大，用開頭與結尾兩次的平均來比只差 -0.34 %（落在 tie 範圍內，事後分析、未用於決定），所以這個預設**不穩健**，在 c1 沒有可量測的效果。只有 `ar2_kernel` / `ar2_bnd_kernel` 會觸發 2、3；`MUSE_TOPO_PDL=1` 時所有 kernel 一開始就觸發（stage 2、3 無效，init 時 log 一次）。**stage 是單次 launch 的屬性，不是 communicator 的**：只有接在 GEMV 前面的兩個 all-reduce 會帶 stage（`all_reduce_boundary` 且 T <= `MAX_ROWS`，後面是 pooler GEMV；bf16 `[T, 4096]` 的 `all_reduce` 且 T <= `MAX_ROWS`，即 MoE latent，後面是 unpooler GEMV），其餘（embedding all-reduce、T > `MAX_ROWS`、prefill、其他模型）維持 kernel 結束才 trigger 的行為。帶 stage 的 all-reduce 後面若有 programmatic launch 的 kernel，必須在讀 output 之前呼叫 `griddepcontrol.wait`（見 `muse_topo/csrc/topo_types.h` 中 `trigger` 欄位的註解） |
| `MUSE_TOPO_GEMV_PUSH`、`MUSE_TOPO_GEMV_MAX_ROWS`、`MUSE_TOPO_GEMV_VARIANT` | 1、4、未設（依 row 數選） | PUSH、MAX_ROWS 不影響數值；VARIANT 0 / 3 同一個加總順序，1 / 2 是另一個順序（deterministic，rounding 不同） | `PUSH` 的 bits 0-1 = push 模式（0 = 只存、由 gather kernel push；1 = 每個 CTA 算完就 push；2 = 最後一個 CTA 用 coalesced store push），+4 = 權重在等前一個 kernel 之後才載入。合法值 0、1、2、4、5、6；**3、7 與範圍外的值會被拒絕**（不 push 卻告訴 gather kernel 已 push，peers 會 spin 到 watchdog trap）：`TopoComm` 初始化時由 stage-1 agreement 檢查（任何一個 rank 的值不合法，所有 rank 都退回 NCCL，不會卡住），`gemv_push` 屬性與三個 C++ 入口也各檢查一次。S1-4a：late weights（5）與 2 / 6 都沒有比 1 好。`MAX_ROWS` = GEMV kernel 服務的最大 padded decode batch（env 值 clamp 到 0 .. 8，0 = 不使用）；超過就維持 cuBLAS + r2d gather。**預設 4**：S1-4b 在 padded T = 8（c5 .. c8，也是 K = 7 speculative decoding 的 verify batch）量測，c8 的 TPOT 在 `MAX_ROWS` 8 / 4 / 只有 router GEMV / 全部關閉 = 19.992 / 19.897 / 19.793 / 19.738 ms（c6：22.972 / 23.005 / 22.841 / 23.071 ms）；規則（8 不比 4 差 0.5 % 以上才用 8）只以 0.02 % 之差成立，GEMV 在 padded T = 8 沒有可量測的增益，三組比較都偏慢，早期 microbenchmark 的 T = 8 layer chain 也比 cuBLAS 慢（53.7 vs 52.1 us），V3 在 T = 8 還有 register spill（`ptxas -v`：unpooler 64 bytes、pooler 96 bytes spill stores，T <= 7 沒有），所以 orchestrator 決定預設為 4（padded T = 1、2、4 使用 GEMV）。`VARIANT` 未設時 T = 1 用 V0（128 CTAs）、T >= 2 用 V3（64 CTAs x 512 threads）；以上都要在所有 rank 相同 |
| `VLLM_MUSE_SPARK_SM120_ROUTER_GEMV`（vLLM branch 的 flag，不是 plugin 的） | **預設 ON**（1） | **deterministic_different**（對 cuBLAS router） | decode（T <= 8）的 fp32 router `[T, 8192] x [256, 8192]^T` 改用 `_muse_C::router_gemv`：一個 kernel 讀一次 8 MiB 權重，加總順序固定、與 T / rank / run 無關（cuBLAS 在 T = 1 與 T >= 2 用不同演算法，不是 batch invariant），也讓後面的 routing kernel 提早啟動。與 cuBLAS 的差異在最後幾個 bit，測試限定誤差不超過 cuBLAS 的（最大 relative error 8 倍、mean absolute error 2 倍）。輸入的 fp32 router input 在 decode graph 中只有 expert prefetch 開啟時才由 boundary kernel 寫出，否則 router 維持 cuBLAS，vLLM log 會印一行 warning。`=0` 回到 cuBLAS router。這是 vLLM branch 的 flag：沒有 host plugin 的 NCCL-only 執行也會用 router GEMV（routing rounding 與舊的正式版不同）。單用 router GEMV：c1 -1.0 %、c4 -0.5 %；與 GEMV 融合一起：c1 -4.2 %、c4 -3.2 %（第 9 節） |
| `VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES`（vLLM branch 的 flag） | 12582912（12 MiB） | identical（cache hint，不寫任何 tensor） | routing-aware expert L2 prefetch 每次拉進 L2 的 bytes（第一個 expert 的開頭，GEMM1 再 GEMM2）。預設與原本的常數相同，行為不變；> 0。更大的 budget 要有更早的 routing ids 才划算（router GEMV 讓 ids 提早），太大會與 kernel 自己的載入搶 HBM（整個 rank 的 96 MiB 比不 prefetch 慢 1.2 %）。S1-4b 的 sweep（c1 TPOT）：12 MiB 9.180 ms（兩次平均）、20 MiB 9.202（+0.24 %）、28 MiB 9.230（+0.55 %），沒有任何值勝過 12 MiB 達 1 %，維持 12 MiB |
| `NCCL_GRAPH_FILE=<host folder>/nccl/graph_r4.xml` | **`env/host.env` 預設 ON**（檔案存在時 export；呼叫端已設定 `NCCL_GRAPH_FILE` 時以呼叫端的值為準） | **deterministic_different** | 合法的 4-channel ring，NCCL 在 <= 512 KiB 改用 LL。2026-10-21 實測 decode 51.2 -> 70.2 t/s（prefill 不變），但**不是 bitwise identical**（小而奇數 row 的 shape 有 1 bf16 ulp 差；end-to-end 3/20 mixed-batch requests 在 decode 發散，|dlogprob| <= 3.8e-6）。與 topo plugin 同時開啟時，TP collectives 都已繞過 NCCL，graph file 只影響 fallback（非 TP group 與不在 envelope 的呼叫）；這個組合已隨 final defaults 的 GSM8K 與 determinism A/B 驗證過（見上一列），throughput 另行量測（第 9 節） |
| `VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP`（vLLM branch 的 flag，不是 plugin 的；plugin 只負責把 comm stream 上 collective 的結果配置在 compute stream） | ON | deterministic_different | 把 eager batch（>= 2050 tokens）切成兩個 microbatch，一個的 TP collective 在 comm stream 上與另一個的 compute 重疊。real weights prefill（ctx10 / 50 / 90）：replicated stream +33 / +30 / +27 %，sequence-parallel stream +14 / +13 / +11 %；decode 不變。run-to-run bitwise 相同（兩個 process、replicated 與 SP 兩條 stream 都驗證過）；與不切分的輸出有 rounding 等級的差異。只在 V2 runner 能 microbatch 的 launch 生效：需 `--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'`（所有 modality 都要設 0），且沒有 PIECEWISE CUDA graph / context parallel / spec decode / LoRA / `VLLM_BATCH_INVARIANT`；其他 launch 上 flag 無作用，也不會改變 runner。memory profile 會多跑一次切分後的 batch，KV pool 約少 0.65 GiB / GPU |
| `MUSE_TOPO_PDL=1`、`MUSE_TOPO_STRICT_STREAM=1`、`MUSE_TOPO_VERBOSE=1`、`MUSE_TOPO_TRACE=1` | OFF | - | 實驗 / 除錯用：PDL（back-to-back -0.3 us，kernel 之後 +0.1 .. 0.2 us，所以關閉）、第二條 stream 直接報錯、JIT build log、in-kernel timeline（需另一份 JIT build） |

## 9. Plugin 的 microbenchmark 與 end-to-end 效果（vLLM 0.32.0 時期；沒有在 0.33.0 重跑）

下面是 plugin 本身的量測：所有 pp / tg 皆為 concurrency 1、隨機 prompt、同一組 seed（`SEED=9000`），pp = input tokens / 平均 TTFT，tg = 1000 / 平均 TPOT(ms)；EP8、`--gpu-memory-utilization 0.95`、fp8 KV、`--max-num-seqs 128`、max-model-len 131072。這些數字是 vLLM v0.32.0 + branch `sm120-muse-spark-open`、torch 2.13 cu130、NCCL 2.29.7 時期量的（量測用的 benchmark script 已不在這個資料夾），**沒有在 0.33.0 重跑**；production 在 0.33.0 的結果是第 6 節。

### 9.1 Microbenchmark（CUDA graph，bf16，8 processes，max over ranks）

2026-10-23 量測；2026-10-24 的 review fix（64-bit counter / flags）之後重測，decode 大小的 all-reduce 差在 +0.4 % 內，LL8 all-gather 約慢 0.1 us，large 差在 +-0.5 % 內。
NCCL 為 vLLM 的 PyNcclCommunicator（`NCCL_P2P_LEVEL=SYS`）。

| op / shape | NCCL | muse_topo | 倍率 |
|------------|------|-----------|------|
| all_reduce `[1,8192]`（decode） | 42.8 us | 5.4 us | 7.9x |
| all_reduce `[8,8192]` | 47.3 us | 11.5 us | 4.1x |
| all_reduce `[64,8192]`（1 MiB） | 86.2 us | 61.4 us | 1.40x |
| all_reduce 8 MiB | 577 us | 454 us | 1.27x |
| all_reduce 16 MiB / 64 MiB（MoE-out） / 128 MiB（attn-out）| 1158 / 4598 / 9204 us | 918 / 3582 / 7097 us | 1.26x / 1.28x / 1.30x |
| reduce_scatter 64 / 128 MiB | 2379 / 4791 us | 1998 / 4022 us | 1.19x |
| all_gather `[1,512]` -> `[1,4096]`（pooler） | 24.6 us | 3.6 us | 6.8x |
| all_gather `[1,1024]` -> `[1,8192]`（unpooler） | 25.9 us | 3.8 us | 6.8x |
| all_gather logits `[1,25256]` -> `[1,202048]` | 35.3 us | 13.3 us | 2.7x |
| all_gather 64 / 128 MiB（輸出） | 2326 / 4652 us | 1663 / 3310 us | 1.40x |
| bf16 exchange（現為 host.env 預設）：all_reduce 128 MiB / reduce_scatter 128 MiB | 9204 / 4791 us | 5840 / 2722 us | 1.58x / 1.76x |

經 vLLM wrapper 的 plugin-level 量測（`bench/test_plugin.py`，CUDA graph、20 次串接呼叫，2026-10-24）：all_reduce `[1,8192]` 5.68 us（NCCL 43.06 us）、
all_gather `[1,512]` 3.86 us（NCCL 24.74 us）。

### 9.2 End-to-end 的 headline（concurrency 1，隨機 prompt，EP8）

| 項目 | 結果 |
|------|------|
| decode，dummy weights | NCCL 基準 51.2 -> topo 85.4 t/s（TPOT 19.5 -> 11.7 ms，+66.8 %）；ctx10 / ctx50 / ctx90（13,107 / 65,536 / 117,964 tokens）+69.2 / +67.3 / +67.8 % |
| decode，real weights（62 層） | NCCL 52.0 -> topo 88.4 t/s（TPOT 19.2 -> 11.3 ms，+70 %） |
| prefill（pp），dummy weights | c1（512 tokens）+7.3 %；ctx10 / ctx50 / ctx90 +14.1 / +13.3 / +13.2 %（ctx90 TTFT 23,495 -> 20,763 ms；每個 8192-token chunk 省約 186 ms） |
| prefill（pp），real weights | c1 +8.2 %；ctx10 / ctx50 / ctx90 +12.1 / +12.1 / +11.4 % |
| 每個 collective 都走 topo | 6 層 e2e 的 per-op 摘要：188 個 key、188 走 topo、0 走 NCCL、0 個 rank 不一致（最大到 `[16384,8192]` 的 256 MiB all-reduce 與 `[1024,25256]` 的 logits gather） |
| bf16 exchange（`env/host.env` 預設，dummy weights） | 對 exact 的 topo：pp +6.7 / +11.8 / +9.7 / +8.9 %（c1 / ctx10 / ctx50 / ctx90），decode 不變；128 MiB all-reduce 7.10 -> 5.84 ms |
| medium all-reduce 的 bf16 exchange（`MUSE_TOPO_MEDIUM_BX`，real weights） | c16 / c64 / c128 out tok/s +2.2 / +2.9 / +1.9 %；1 MiB 的 all-reduce 62.5 -> 43.1 us；GSM8K 97.5 / 96.5 % |
| pooler / unpooler GEMV 融合 + router GEMV（real weights，S1-4a） | decode TPOT c1 -4.2 %、c4 -3.2 %（對全關的 9.593 / 15.256 ms）；c16 / c64 在 noise 內 |

基準的定義：dummy weights 為 NCCL（`NCCL_P2P_LEVEL=SYS`）的 2026-10-21 run，real weights 為同一棵 tree 的 `MUSE_TOPO_ENABLE=0`；兩者各自對自己的 topo run 比較，不要跨列比較。
這些數字都是 vLLM 0.32.0 時期、concurrency 1 的量測，**沒有在 0.33.0 重跑**；0.33.0 的 production 結果是第 6 節。

### 9.3 記憶體

* workspace 每張 GPU 64,094,208 B = 61.13 MiB（large 演算法重用同一塊）。它在 `CudaCommunicator.__init__` 配置，早於 vLLM 的 `init_snapshot`，所以不會從 KV budget 扣除，
  而是縮減 `1 - gpu_memory_utilization` 的 headroom（0.95 時有約 4.8 GiB，沒有影響；只有 utilization 接近 1.0 才需要注意）。
* READY 時 GPU 0 為 91,796 MiB（NCCL 為 91,786 MiB），因為 captured graph 變小（NCCL 為每張 graph 配置 buffer，topo kernel 不用）；KV pool 每張 GPU 8.48 GiB（1,563,909 tokens）對 NCCL 的 8.21 GiB（1,513,149 tokens）。


## 10. 驗證方式

### 10.1 不需要 GPU 的檢查（CPU）

launch / 檢查 script 的 8 個測試（bash，不需要 GPU 與 vLLM；server 用 `bench/fixtures/launch_stub` 的 stub，log 是真實 boot 的相關行 `bench/fixtures/launch`）。在 repo 的根目錄執行，每一個的最後一行是結論（`0 case(s) failed` / `ALL CASES PASSED`）：

```bash
for t in test_launch_equiv test_launch_flow test_check_kv_launch test_check_spec_launch test_check_offload_launch test_check_serve_line test_offload_env test_preflight_offload; do
  echo "== $t"; bash bench/$t.sh | tail -n 1; done
```

* `test_launch_equiv.sh`：六個 launch 狀態（含 dirty shell）的 `--dry-run` 與 `bench/fixtures/launch_ref` 逐 byte 相同（serve_cmd.txt、env.txt、server 預期的環境），並與 base commit 的 env 檔 export 比較（`da1bdad519`，需要 git history）。
* `test_launch_flow.sh`：用 stub server 跑完整流程（READY、各種失敗與停止路徑、exit code）。
* `test_check_*.sh`、`test_offload_env.sh`、`test_preflight_offload.sh`：檢查 script 與 env 檔的行為（`test_offload_env.sh` 以 `942acddc48` 比較 `MUSE_OFFLOAD=0` 的 export）。
* **這兩個測試需要 repo 的 git history**（`da1bdad519`、`942acddc48` 要留在 main 的歷史裡；沒有時它們印 `skip:`，不算通過）。

plugin 的 CPU 單元測試（mock TopoComm；需要 torch，不需要 GPU）：

```bash
PYTHONPATH=$PWD python3 bench/test_plugin_unit.py        # 結尾印 test_plugin_unit PASSED
scripts/runsrc_topo.sh $HOST_DIR/bench/test_topology.py  # 拓樸推導：任意 rank 排列都是 collapse-free（200 組隨機排列），需要 nvidia-smi（在伺服器上）
scripts/runsrc_topo.sh -c "from vllm.plugins import load_general_plugins; load_general_plugins()"   # plugin 被 dist-info 找到並註冊
```

### 10.2 GPU 上的 correctness 測試（8 processes，不需要完整模型）

下列指令在伺服器上執行（`cd /raid/vllm-runtime/host_topo`，先 `scripts/deploy_host.sh`）。GPU 測試要在 GPU lock 下跑：`scripts/locked_job.sh <tag> <timeout_s> <command...>` 會等 lock、等 GPU 記憶體釋放、detached 執行，
log 在 `$MUSE_TEST_SCRATCH/logs/<tag>.log`（預設 `${MUSE_TEST_SCRATCH:-/raid/vllm-runtime/work/plugin-tests}`），完成後寫 `<tag>.DONE`（內容為 return code）。scratch 與結果不要寫進本資料夾。

```bash
cd /raid/vllm-runtime/host_topo; export HOST_DIR=$PWD SRC=/raid/vllm-runtime/sm120-dev/src
bash scripts/locked_job.sh v1 7200 bash -c 'bash scripts/run_h1a.sh v1 tests other; bash bench/run_h1b.sh v1 tests timeout epoch;
  bash scripts/run_h2.sh v1 plugin'
```

涵蓋：與 arithmetic contract 逐 bit 相同、所有 rank bits 相同（含 -0 的符號）、run-to-run determinism、eager == CUDA graph replay、all_reduce == reduce_scatter + all_gather（bitwise）、
10^5 次 back-to-back graph replay、不同大小混合序列（counter 對齊）、64-bit epoch 在 2^31 / 2^32 / 2^33 / 2^62 附近、一個 rank 少呼叫時其他 rank 會 trap 而不是 hang、
extension build 失敗時所有 rank 一致 fallback、plugin 經 vLLM wrapper 的 eager / graph / fallback。最近一次完整結果（2026-10-24，review fix 之後，全部 PASS）：
`tests_h1a.py` 256 checks x 8 ranks、`test_h1a.py`（第二套）ALL TESTS PASSED、`tests_h1b.py` 284 checks、`test_epoch.py` PASS、`test_plugin.py` 416 checks x 8 ranks（0 failures）、
`test_timeout.py`、`test_timeout_large.py`（3/3）、`test_init_failure.py`、`check_ctx.py`、`test_plugin_unit.py`、`test_topology.py`。

runner 的 step：`scripts/run_h1a.sh <tag> smoke tests tests_quick ctx initfail other other_full topology timeout epoch unit`、`bench/run_h1b.sh <tag> tests_quick tests tests_soak timeout epoch`、
`scripts/run_h2.sh <tag> plugin e2e_topo1 e2e_topo2 e2e_nccl lp_topo lp_nccl lp_compare ops real_topo1 real_topo2 real_nccl real_compare compare`（`locked_job.sh` 之下執行）。

沒有 runner 的測試，直接在 GPU lock 下用 `scripts/py_topo.sh` 執行（`SRC` = vLLM tree，`GPU` = 使用的 GPU；加 `BENCH_WATCHDOG_S=<秒>` 設 watchdog）：

```bash
SRC=/raid/vllm-runtime/sm120-dev/src bash scripts/py_topo.sh bench/test_medium_bx.py --groups emu,bits,det,disp,seq,rsar [--quick]        # BENCH_WATCHDOG_S=900
SRC=... bash scripts/py_topo.sh bench/test_a1_tables.py [--quick] [--groups tables,ar,rs,ag,graph,soak] [--soak-seconds N]
GPU=0 SRC=... bash scripts/py_topo.sh bench/test_boundary.py [--rows N] [--ckpt-weights]                                                 # 一張 GPU
GPU=0,1,2,3,4,5,6,7 SRC=... bash scripts/py_topo.sh bench/test_ar_boundary.py [--quick]
GPU=0,1,2,3,4,5,6,7 SRC=... bash scripts/py_topo.sh bench/test_gemv_push.py [--quick]     # T = 1..8、兩個 projection、所有 variant 與合法的 push 模式、每個 trigger stage、eager 與 CUDA graph、3 / 7 的拒絕
python3 bench/chat_check.py --port 8100 --model unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --out chat.json                                 # 對 launch/serve_*.sh 啟動的 server 做 greedy chat 檢查
```

### 10.3 End-to-end determinism 與 NCCL 對照

```bash
bash scripts/locked_job.sh v2 7200 bash scripts/run_h2.sh v2 e2e_topo1 e2e_topo2 e2e_nccl compare ops lp_nccl lp_topo lp_compare   # 6 層 dummy weights
bash scripts/locked_job.sh v3 7200 bash scripts/run_h2.sh v3 real_topo1 real_topo2 real_nccl real_compare                        # 完整 62 層、real weights
```

`run_h2.sh` 內部用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（multiprocess engine 的 request admission 會改變 batch 組成，無法做 bitwise A/B）。結果在 `$MUSE_TEST_SCRATCH/h2/`。預期與已觀察到的結果：

* topo vs topo（兩個獨立 process）：token ids、logprobs、prompt logprobs 全部 bitwise 相同。dummy weights 20/20（in-process 重跑 8/8），real weights 51/51，max |dlogprob| = 0。
* topo vs NCCL：**預期不會 bitwise 相同**（`more_accurate`）。real weights：33/51 requests 的 48-token greedy continuation 相同，其餘在近似平手的 token 分歧；teacher-forced NLL 在 noise 內相同
  （NCCL 1.7321 / 1.3184 / 1.1796，topo 1.7241 / 1.3172 / 1.1801 nats/token）；`bench/chat_check.py` 的 5 個 greedy chat 在 NCCL 與 topo 上都連貫且正確。
  dummy weights 的 logits 幾乎是平的，greedy 由 tie 決定，所以那邊只能當作擾動大小的上限（teacher-forced：6150 個值 mean |d| 2.2e-6、max 1.3e-5）。
* 每個 collective 都走 topo：`ops` step（`summarize_ops.py`）顯示 188 keys、188 topo、0 NCCL。

accuracy 的量測（`tests_h1a.py` / `tests_h1b.py` / `test_plugin.py`，fp64 reference，單次 rounding 的 `round_single` 為標準答案）：activation-like 資料 100 % correctly rounded，
平均誤差 0.236 ulp（randn）/ 0.250 ulp（mixed exponent），NCCL bf16 ring 為 47.6-47.8 % / 55.4-55.6 % correctly rounded、平均約 1.6-2.1 ulp；相對 L1 norm 的最大誤差 0.0039（= bf16 unit roundoff）對 NCCL 的 0.012-0.018。
指數範圍極寬的病態資料（約 +-24 個二進位數量級）可能有 0.13 % 的元素不是 correctly rounded，保證的上界是 0.5 ulp + 2^-16 相對誤差。

### 10.4 NCCL 工具

```bash
# PyNccl all-reduce 的 bitwise 測試（用法見檔頭；NCCL_* 環境變數由呼叫端決定；要用 GPU，所以走 lock）
bash scripts/locked_job.sh bw 1800 bash -c 'SRC=/raid/vllm-runtime/sm120-dev/src scripts/py_topo.sh nccl/bitwise.py --out "$MUSE_TEST_SCRATCH/bw.json"'
python3 nccl/compare.py REF.json OTHER.json [...]          # 比較 e2e_bitwise.py / e2e_real.py 產生的 dump（不需要 GPU）
```

## 11. 已知限制

### 11.1 final matrix 的已知限制

* **DRAM 讀取上限**：實測冷 DRAM 讀取上限為 1.537 TB/s = ncu 1.596 TB/s 峰值的 96 %（唯一支援的 memory clock 12481 MHz，不是 1.79 TB/s 的標稱值），S1-6 L3 因此關閉；L2 常駐的資料可以超過。
* **link-bound 的 PCIe collectives 與 IB lane 的 PARK**：沒有 NVLink；bf16 exchange 下每個大型 collective 都受每張 GPU 自己的 x16 link 限制，不是跨 socket 的 cut：isolated kernel 量到的 all-reduce / reduce-scatter 是其 link floor 的 0.91-1.04 倍（S1-12；prefill 內的 reduce-scatter 為 T_link 的 1.12-1.19 倍，有 skew / contention），所以經 ConnectX-7 的 lane 搬的是同樣的 bytes、同一條 link，IB lane 為 PARK FINAL（S1-12、S1-16：把 cut 的 store 改送同 socket 的 peer，大型 AG / RS 在每個 size 都更慢）。
  大型 all-gather 為其 link floor 的 1.17-1.21 倍，超出部分在 AG kernel 內部（two-hop relay、per-range flag、chunk pipelining），同樣 cut 比例的 reduce-scatter 為 0.96-0.98 倍；這個 AG relay 在下方 parked。
* **EP straggler**（R5 / R11）：排名低於 1.6 %（「不值得或找不到機制」），沒有處理；EP8 把 tokens replicate，沒有 all2all。
* **tier ON 時的 swap**：P* 的 pinned tier 配置時，worker / EngineCore / API server 沒有 pin 的 anonymous memory 可能被換出到 8 GiB 的 swap file。本次 matrix：**ns256 ON 在 3 次 boot 中有 2 次在 READY 時已 swap**（`ns256_on` 使用 4,095.3 MiB、峰值 5,429.6 MiB；`ns256_goff` 使用 4,570.6 MiB、峰值 6,677.8 MiB；`ns256_long` 63.8 MiB 沒有 swap），spec ON 1 次 boot 沒有 swap；離線的 G-det / D4' process 不在此統計：其中 1 個（G-det spec ON，process b）swap 到 2,997 MiB 並在結束時釋放（matrix 的結果表沒有列出離線 process 的 swap），其餘 5 個離線 process 為 64-71 MiB。S1-10R gate 為 14 / 18 個 ON 類 boot swap（3.8-7.4 GiB，峰值 7,444.8 / 8,192 MiB），16 個 OFF / CTL 類 boot 都沒有。
  包含 det / D4' boot 與 adoption 系列的 22 個 large-tier ON boot 中有 14 個 swap、8 個沒有，17 個沒有 tier 的 boot 都沒有。**沒有找到預測因子**（39 個 boot 的 predictor 分析，不用 GPU；本次 matrix 的 4 個 ON boot 不在這 39 個之內，predictor 沒有用它們重跑）：preflight 的 A_N / LEFT、launch 前的 node 記憶體（free / file / slab）、tier profile、前一次 boot 都無法區分 swap 與否；
  launch 時距前一次 teardown 50 分鐘以上的 3 個 boot 都沒有 swap（8 個沒有 swap 的 boot 中的 3 個），校正後不顯著（Bonferroni 0.23），且 8 個沒有 swap 的 boot 中 5 個緊接在 3-41 秒前的 boot 之後。所以把 READY 時的 swap 視為很可能發生（likely；S1-11 matrix 的 4 個 ON boot 中 2 個 swap，原 22 個中 14 個，合計 26 個中 16 個），每次 boot 檢查（`status.txt` 有 swap 紀錄）；cell 不因 swap 作廢。
* **c128 p99 TTFT 的 tail**：S1-10R gate 中 dec c128 的 p99 TTFT 在六組都比 OFF 高 3.5-7.7 %（有 tier，系統性、僅供參考）；本次 matrix（單次 run）spec ON 對 OFF 的 c128 p99 TTFT 為 +0.8 %（greedy）/ +6.8 %（default）/ +2.4 %（top_p）、nat8k 0.0 %、churn c128 -27 %。
  同一 chain、單次 run 的 spec ON 對 OFF 的 TTFT 也有較大的差，沒有歸因（單次 run，無法分辨 tier 的成本與 run 間 noise）：dec_default 的 p50 最多 +8.6 %（c64；c32 +6.1 %）、p99 +5.4 %（c4）與 +6.8 %（c128），top_p c64 的 p50 +4.2 %。上面「tier 的成本」只涵蓋 throughput。
* **ns256 bound-TTFT 偏離從未重新 gate**（0.32.0 研究：bound 的 ns256 比 unbound 慢 9-12 %）；本次 matrix 中 ns256 ON 對 ns128 OFF 的 median TTFT 在 -4.3..+7.9 %（跨 profile、單次 run、不同的量），不構成 gate。
* **batched 的 tier-hit 輸出不是每個 request 都與 GPU-hit 逐位元相同**：同一個 process 中 H（GPU hit）與 X（GPU-only reset 之後的 tier hit）在檢查過的 workload（spec greedy / t1k64）的 85 個 request 都有相同的 hit 長度，但 batched pass 中仍有一部分輸出不同（H = X 的 request 數：spec greedy 72 / 85、spec t1k64 37 / 85、spec mixed 68 / 85；ns256 greedy 81 / 85、mixed 80 / 85、t1k64 81 / 85；hit 長度相同只檢查過 spec greedy / t1k64；原因沒找到，候選是 batched 排程 / chunk shape）。
  成立的證據：single-stream 的 tier restore 逐位元相同（D4' 的 KV block 與 server leg 的 tokens + logprob 值），兩個 cold process 完全相同（G-det）。
* GSM8K 把 16,384-token 截斷算錯（見上）。

### 11.2 plugin 與環境的限制


* **只適用這台主機**：拓樸簽章要求 TP world size 8、同一個 node、全是 RTX PRO 6000 Blackwell（cc 12.0）、56 組 P2P 全通、PCI layout 為 2 sockets x 2 virtual switches x 2 GPUs，不吻合就用 NCCL。
  script 內的路徑寫死為 `/raid/vllm-runtime/...`。
* **版本**：第 8、9 節的 H1 / H2 / r3 microbenchmark 與 end-to-end 數字是 vLLM v0.32.0 + branch `sm120-muse-spark-open`、torch 2.13 cu130、NCCL 2.29.7 時期量的，沒有重跑。目前 production 是 v0.33.0 線（vLLM `f6a60ab9d7`、torch 2.14.0+cu130、flashinfer 0.7.2.post1、NCCL 2.30.7、triton 3.8.0）+ host `f540ce1b37`；它的最終 matrix 與 gates 見第 6 節（2026-11-13）。plugin 是在 runtime patch `CudaCommunicator`，換 vLLM 版本要重新驗證。
* 只涵蓋 TP group（名稱以 `tp` 開頭、8 ranks）的 `all_reduce` / `all_gather` / `reduce_scatter`；EP group、`gather`、`reduce_scatterv`、`all_gatherv`、send / recv 仍走原本的路徑。
  不在 envelope 的呼叫（其他 dtype、non-contiguous、不是 128 B 倍數的大訊息、reduce_scatter dim != 0、all-gather 的中間維度）由 `supports()` 退回 NCCL。
* 本節的 H2 效能數字都是 **concurrency 1**；decode graph size 1..128 都有走 topo（正確性已驗證，128 requests 的 decode batch 重跑 128/128 相同）。高 concurrency（c4 .. c128，
  real weights）的整體 throughput（c1 .. c256、四個 launch）已在第 6 節量測；已知限制（DRAM 上限、link-bound collectives、EP straggler、tier 時的 swap 等）見第 11.1 節。
* vLLM branch 的 SM120 sequence-parallel stream（`VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL`，預設 1，不需要 `deep_gemm_mega_moe`）在 >= 512 tokens 的 eager prefill 會走 `reduce_scatter` 與 `all_gather`，
  end-to-end 的 op log 有 `reduce_scatter(dim=0) shape=(8192, 8192) -> topo`；`test_plugin.py` 另外覆蓋 CUDA graph 與 RS + AG == AR。
  只有 `deep_gemm_mega_moe` 的原版 SP 路徑在這台機器上是關的。
* 同一個 `TopoComm` 的所有 collective 必須在同一條 stream 上依序發出，且所有 rank 的呼叫順序與 shape 相同；大訊息 kernel 會與 peer 互相 spin-wait（co-resident blocks），
  不可與會佔滿所有 SM 的其他 stream kernel 重疊。vLLM 預設在 main stream 發出 collectives；microbatched prefill（`VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP`，預設 ON）改在一條 comm stream 上，
  依各 rank 一致的順序發出，並以 hand-off event 滿足這條規則。
* kernel 等不到 peer 超過 `MUSE_TOPO_TIMEOUT_S` 會 `__trap()`，該 worker 的 CUDA context 會失效（刻意如此，避免靜默 hang）。checkpoint / sleep（`suspend`、`resume` ...）沒有處理：IPC workspace 不會重建。
  `destroy()` 需要對稱（一個 rank 先 destroy，其他 rank 仍發 topo collective 會一直等到 timeout）。
* 數值：exchange payload 中等於 `0xFFFFFFFF` 的 32-bit word（NaN pattern）會被換成 `0xFFFFFFFE`，所以 NaN payload 可能與 NCCL 不同（仍是 NaN）。LL8 的 tag 約每 2^33 次 launch 重複一次（殘餘風險，沒有實測到）。
* bf16 exchange 與 NCCL graph file 都是 `deterministic_different`，現為 `env/host.env` 的預設（使用者 2026-10-24 決策）；final defaults 的 GSM8K 與 determinism A/B 已通過（第 8 節）。
  bf16 exchange 在兩個 partial 互相抵銷時最大誤差可到約 32640 ulp，且對 padded batch size 不是 batch invariant；需要 exact 結果或批次不變性的場合設 `MUSE_TOPO_LARGE_BF16_EXCHANGE=0`。
  medium all-reduce 的 bf16 exchange（r3c）讓 decode 範圍內出現新的 batch size 邊界：embedding all-reduce 在 eager T 3 與 4 之間（graph 的 padded T = 2 與 4 之間）、attn-out 在 T <= 8（fused kernel，exact；T = 3 與 4 都 exact）與 T >= 9（graph 的 padded T = 16 起，bf16 exchange）之間、MoE-out 在 eager T 7 與 8 之間（padded T = 4 與 8 之間）；只想關掉這一項設 `MUSE_TOPO_MEDIUM_BX=0`。fused 關閉時（`MUSE_TOPO_FUSED=0` 等）T = 4..8 的 attn-out 也走 bf16 exchange，所以 fused 開 / 關的 bitwise A/B 兩次都要設 `MUSE_TOPO_MEDIUM_BX=0`。
* r3b 的 GEMV + all-gather 融合（`MUSE_TOPO_GEMV`）與 router GEMV 都是 `deterministic_different`（對 cuBLAS；見第 8 節）：同一個 token row 在任何 batch size、rank、run 下 bits 相同，但與 cuBLAS 路徑的輸出
  在最後一個 bit 可能不同（router 的 sigmoid + top-8 只會在兩個 logits 差距小於那幾個 bit 時選到另一個 expert）。GEMV kernel 的環境變數（`MUSE_TOPO_GEMV*`）必須在所有 rank 相同，`MUSE_TOPO_GEMV_PUSH` 不可設成 3 或 7（會被拒絕）。
  padded T = 8（c5 .. c8）的 GEMV chain 在整個模型上沒有可量測的增益，所以預設 `MUSE_TOPO_GEMV_MAX_ROWS=4` 保留 cuBLAS；router GEMV 需要 fp32 router input（decode graph 中只有 expert prefetch 開啟時才有），沒有時維持 cuBLAS。
* 研究 probe（all-to-all、copy-engine cut、host relay，程式已不在這個資料夾）顯示：cross-socket 的 copy engine 或 host relay 做 cut transport 沒有實質增益（host relay 約 13 GB/s per GPU，island egress 隨後成為瓶頸，估計最多約 15 %，未實作）。
* 已知的後續項目：把 LL8 tag 的 64-bit modulo 換成 compare / select 可省約 0.05 us per all-gather（每個 decode step 約 6 us，未出貨，需再做一輪驗證）。


## 11b. S1-17 的 per-configuration noise floor

每個 configuration（serve line + 環境 + 記憶體 layout）在 dispatch 相同的 cell 上有固定的偏移，**0.4-0.6 % SD（最大 1.3 %）**。所以單一 configuration 內的 A/B 差異低於約 **1.5-2.5 %**，
不做 layout 隨機化（每個 configuration 多次 layout 抽樣）就不能歸因於被測機制。這個 floor 是由**兩個 cell（ns c1、c4）推論**出來的，不是量到的常數；
S1-10R 的 STEADY c1 pass（0.9859 對門檻 0.985，餘裕 0.0009）落在它之內。對 bytes 的後果：S_live 的餘裕很薄（spec ON 0.4 MiB、ns ON 1.1-1.8 MiB），
任何不是第 4.4 節六條 serve line 與環境的 launch 都是新的 configuration，需要重新推導 bytes（第 5 節）。

## 12. Parked 的 lever 與重啟條件

只列來源寫明的條件；沒有寫的標「無」。這些 lever 的程式碼不在這個資料夾裡（沒有 merge）。

* **S1-17 fused-boundary dispatch rows**：PARKED，production 維持 8 / 8 / 16。四個 open lever：(1) ns 的兩種 T = 8 kind 改 two-step（ns c8 -1.86 %，在 floor 上；最有支持的一個：打開後 c8 +1.20 %，stage 1 關閉後 -1.86 %）、
  (2) T = 17..24 的 RMS 24（spec raw c3 -1.4..-2.1 %）、(3) AG 16（out of class）、(4) T = 4 的 AR 3（無法解析）。重啟條件：需要能壓過 per-configuration floor 的 protocol（每個 configuration 多次 layout 抽樣），或效果大於約 2.5 %（第 11b 節）。
* **S1-16 all-gather relay**（AG 對其 link floor 的 1.17-1.21 倍超出）：已排名、未開始；ceiling 約 3.9 %（prefill）/ 2.1 %（c64）/ 3.0 %（c128），預期 gain 約 1.9 / 1.05 / 1.5 %（低於 2 % 的線）；重啟條件：無。
* **S1-15 更多 cut CTA 的大型 all-gather**：PARKED（在每個 production shard 的 isolated 量測都更慢）；重啟條件：無。
* **R1 out-of-class verification（不做完整 gather 的驗證）**：當超過約 25 % 的 production request 是 out of class（top_p > 0.95、T > 1、penalty、top_k 在 40..64 之外）時，這是第一個要做的項目；全部 out-of-class 流量的成本是 c64 7.4 % / c128 10.3 %。
* **R3 layers 0 / 61 的 BF16 改 FP8 / NVFP4**：會改變模型，需要使用者的 policy 決定（c64 -4.6 %、c128 -4.2 %、每 rank +4.6 GiB KV）；經同意才做 drift study。
* **R2a**：REJECTED（S1-9 L10：real +0.67 %、KV -3.64 %、c1 -1.79 % 原因不明）；重啟條件：無。
* **A1b capture list 384**（cap 384，sizes 272..384 step 16）：PARKED（STEADY c33..c47 +11.6..+15.2 %，但 real c36..c48 只 +1.7..+3.3 %，pooled +2.55 %，低於事先登記的 3.96 %；KV -1.96 %）；重啟條件：KV 充裕時。
* **LL8 tag 的 64-bit modulo 改 compare / select**：約省 0.05 us per all-gather（每個 decode step 約 6 us）；未出貨，需再做一輪驗證。
* **IB lane（ConnectX-7 的 cross-socket link）**：PARK FINAL（把 cut 的 store 改送同 socket 的 peer，大型 AG / RS 在每個 size 都更慢；每張 GPU 自己的 x16 link 才是瓶頸）。
* **cut transport 的研究**：cross-socket 的 copy engine 或 host relay 做 cut transport 沒有實質增益（host relay 約 13 GB/s per GPU，island egress 隨後成為瓶頸，估計最多約 15 %，未實作）。

## 13. 出處

* final matrix 與 gates 的量測：vLLM `sm120-muse-spark-open` `f6a60ab9d7` + 本 repo `main` 的 `f540ce1b37`（2026-11-13 02:36:49 -> 07:11:47；13 個 hold、7 次 boot、216 個 cell）。
* `f540ce1b37` 之後的 commit 只改 launcher 的停止路徑（server.sess 消失時改用記憶體裡的 tuple）、serve line 檢查（log 的 connector / role 要與 line 一致）、測試與文件，以及這次整理（移除實驗與歷史檔案、重寫本檔）：
  **沒有改任何 export、serve line 或預設值**（`--dry-run` 的 serve_cmd.txt、env.txt、pid_environ_expected.txt 在整理前後逐 byte 相同）。
* 伺服器的 mirror 要重新用 `scripts/deploy_host.sh` 同步才會有這些變更（先 `-n`，且只在沒有 server 佔著 GPU lock 時）。
* 歷史要求：`bench/test_launch_equiv.sh` 需要 `da1bdad519`、`bench/test_offload_env.sh` 需要 `942acddc48` 留在 `main` 的 git history 裡（不要 squash、不要改寫歷史）。
