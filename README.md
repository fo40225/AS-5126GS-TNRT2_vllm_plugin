# AS-5126GS-TNRT2_vllm_plugin

針對 **10.19.209.74（Supermicro AS-5126GS-TNRT2，8x NVIDIA RTX PRO 6000 Blackwell Server）** 這台主機的 PCIe / NUMA 拓樸所做的最佳化，
用來加速 `unsloth/Muse-Spark-Open-816B-A42B-NVFP4`（vLLM branch `sm120-muse-spark-open`，v0.32.0 + Muse Spark backport，
TP8 + EP8）的推論。

這個資料夾只放「只對這台機器的拓樸才有意義」的東西：topology-specialized 的 P2P collectives（`muse_topo` 套件與 vLLM plugin）、NCCL 環境變數與
graph file、啟動 / 驗證 script 與量測文件。與機器無關的通用 SM120 最佳化則是 vLLM branch 裡的 git commits，不在這裡。

本資料夾是 deliverable，**本機的 `/home/user/trivial/AS-5126GS-TNRT2_vllm_plugin/` 是 source of truth**；伺服器上的
`/raid/vllm-runtime/host_topo/` 只是用 `scripts/deploy_host.sh` rsync 出去的 mirror。

## 1. 摘要

| 項目 | 結果（EP8、concurrency 1、2026-10-24，詳見第 6 節） |
|------|------|
| decode（tg） | 51.2 -> 85.4 t/s（TPOT 19.5 -> 11.7 ms，+67 %）；real weights 52.0 -> 88.4 t/s（+70 %） |
| prefill（pp） | c1（512 tokens）+7 %、13k / 65k / 118k tokens 的 context 約 +13 .. +14 %；real weights +8 .. +12 % |
| 額外記憶體 | 每張 GPU 61.1 MiB（large-message 演算法不再增加） |
| accuracy class（預設） | `more_accurate`：fp32 加總、單次最終 rounding，所有 size / 演算法 / rank / run 之間 bitwise 相同 |
| 啟用條件 | `MUSE_TOPO_ENABLE=1`，且拓樸簽章（2 sockets x 2 virtual switches x 2 GPUs，cc 12.0，56 組 P2P 全通）吻合，否則自動退回 NCCL |

## 2. 為什麼需要（背景）

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

## 3. 資料夾內容

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
│                                gemv.cuh、gemv_push.cu：pooler / unpooler 的 GEMV 與其後的 all-gather 融合（r3b））
├── muse_topo-0.1.dist-info/     entry_points.txt：[vllm.general_plugins] muse_topo = muse_topo.vllm_plugin:register
│                                （放在 PYTHONPATH 上就會被 vLLM 找到，不需要 pip install）
├── env/host.env                 主機環境變數：NCCL_P2P_LEVEL=SYS、MUSE_TOPO_ENABLE=1、bf16 exchange 與 NCCL graph file（2026-10-24 起預設 ON），其他選項以註解列出
├── env/spec_k7.env              speculative decoding（RadixArk DFlash2 K=7）專用的 launch profile：`--max-cudagraph-capture-size 256` 搭配 `VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=384`，並提供該設定的 exact KV bytes `MUSE_SPEC_KV_BYTES=10526351360`（1,482,225 tokens；只在 host.env 之後 source，見 `docs/s19_spec_launch.md`）
├── scripts/
│   ├── deploy_host.sh           本機 -> 伺服器 mirror 的 rsync（只同步本資料夾）
│   ├── run_src_topo.sh          /raid/vllm-runtime/musetest/run_src.sh 的副本：把本資料夾加進 PYTHONPATH、source env/host.env，
│   │                            另加 LOAD_FORMAT、CHAT_CHECK
│   ├── locked_run_topo.sh       musetest/locked_run.sh 的副本：拿 GPU lock、等 GPU 記憶體釋放，再跑標準 EP8 benchmark
│   ├── runsrc_topo.sh           sm120-dev/runsrc.sh 的副本：以 dev venv 執行 python（PYTHONPATH 含本資料夾與 vLLM tree）
│   ├── py_topo.sh               跑單機測試 / benchmark 用的 python wrapper（scratch 與 JIT cache 在 work/host）
│   ├── locked_job.sh            在 GPU lock 下 detached 執行任意指令，log 與 DONE marker 寫到 work/host/logs
│   ├── run_h1a.sh               小訊息演算法的驗證序列（steps：smoke tests bench_* topology unit ...）
│   ├── run_h2.sh                vLLM plugin 的驗證序列（steps：plugin e2e_* compare ops bench* real_* ...）
│   ├── prefill_chain.sh         一次 lock 內依序啟動多組 server 設定、量單一 request 的 prefill latency（microbatched prefill 的 A/B；搭配 bench/prefill_probe.py）
│   ├── check_spec_launch.sh     由 serve log 確認 speculative launch profile 有生效（SP floor 384、graph 上限 256；不符就 exit 1）
│   ├── run_r3c.sh               medium all-reduce（r3c）的驗證 / 量測步驟（steps：medium_test h1a_quick h1b_quick bench_* ...）
│   ├── r3c_chain.sh             r3c 在一次 GPU lock 內的完整序列：測試、sweep、產生 table、dispatch 檢查、real-weight concurrency A/B、GSM8K smoke
│   └── conc_sweep.sh            real weights、單一 server session 的 concurrency sweep（random 512/128，可重複；r2 sweep script 的參數化副本）
├── nccl/
│   ├── graph_r4.xml             NCCL graph file（deterministic_different；`env/host.env` 預設使用，見第 5、6 節）
│   ├── bitwise.py               PyNccl all-reduce 的 bitwise identity 測試（8 ranks，eager + CUDA graph）
│   ├── e2e_bitwise.py           6 層 dummy-weight 模型的 end-to-end dump（token ids、logprobs、prompt logprobs）
│   └── compare.py               比較兩份 e2e dump（token / logprob 是否 bitwise 相同）
├── bench/                       小型 correctness / benchmark / probe script（見下）
├── docs/
│   ├── topology.md              實測拓樸：矩陣、collapse rule、bottleneck 分析、NCCL baseline
│   ├── h1a_collectives.md       基礎設施與 decode-size 演算法（API、arithmetic contract、protocol、dispatch）
│   ├── h1a_latency.md           decode-size latency 表（vs NCCL、vs FlashInfer）
│   ├── h1b_collectives.md       large-message all-reduce / reduce-scatter / all-gather
│   ├── h1b_latency.md           large-message latency 表
│   ├── h2_vllm_plugin.md        plugin 的 patch 範圍、環境變數、驗證紀錄、benchmark、review fixes、限制
│   ├── r2d_fused_epilogue.md    decode 的 collective + residual boundary / pre-expert norm 融合（bit-exact 的 CUDA 版 Triton kernel）
│   ├── r3b_gemv_push.md         decode 的 pooler / unpooler GEMV 與 all-gather 融合（r3b）：算術與精度、protocol、環境變數、S1-4a 量測
│   └── r3c_medium_ar.md         medium all-reduce（64 KiB .. 8 MiB）的 bf16 exchange：算術、dispatch table、latency、real-weight concurrency、GSM8K
└── tools/                       本機端工具（/tmp/muse_tools 的副本）：deploy.sh、deploy_to.sh
```

`tools/deploy.sh` 與 `tools/deploy_to.sh` 同步的是 **vLLM source tree**（`/home/user/vllm` 或某個 worktree -> 伺服器的 `sm120-dev/src*`），
不是這個資料夾；同步本資料夾請用 `scripts/deploy_host.sh`。注意 `tools/deploy.sh` 固定把 `/home/user/vllm`（main checkout）同步到共用的部署樹
`/raid/vllm-runtime/sm120-dev/src/`，會覆蓋其他人正在使用的那份，只有負責整合的人才應該執行；一般情況請用 `tools/deploy_to.sh <local_git_tree> <remote_src_dir>`
（目標必須是已存在的獨立 tree，例如 `cp -al` 出來的 `sm120-dev/src_<track>`）。

`bench/` 分類：

| 類別 | 檔案 |
|------|------|
| correctness（8 processes、GPU lock 下執行） | `tests_h1a.py`、`tests_h1b.py`、`test_epoch.py`、`test_timeout.py`、`test_timeout_large.py`、`test_init_failure.py`、`check_ctx.py`、`smoke.py`、`test_plugin.py`（8 ranks、經 vLLM wrapper）、`test_medium_bx.py`（medium all-reduce 的 bf16 exchange）、`test_gemv_push.py`（GEMV + all-gather 融合，8 ranks）；CPU-only：`test_plugin_unit.py`、`test_topology.py` |
| 第二套獨立的 H1a 測試 | `test_h1a.py` + `hlib.py` + `test_topology.py` |
| benchmark / 表格 | `bench_h1a.py`、`bench_h1b.py`、`bench_medium.py`（medium all-reduce sweep）、`bench_gemv_push.py`（GEMV + all-gather 融合的 layer chain，`--variants` / `--triggers` / `--cold`）、`acc_gemv_sweep.py`（GEMV 加總順序對 cuBLAS 與 fp64 的精度，單 GPU）、`make_table*.py`（含 `make_table_medium.py`）、`summarize_h1a.py`、`summarize_h1b.py`、`summarize_ops.py`、`trace_h1a.py`、`trace_h1b.py`、`build_ext.py` |
| 研究 probe（不屬於 runtime） | `a2a_probe.py`、`ce_cut_probe.py`、`host_relay_probe.py` |
| end-to-end | `e2e_prompt_lp.py`（teacher-forced logprob）、`e2e_real.py`（real weights、62 層）、`e2e_overlap.py`（real weights、長 prompt / chunked prefill / mixed batch，microbatched prefill 的 A/B）、`chat_check.py`（greedy chat 檢查） |
| runner | `run_h1b.sh` |

**注意：H1a 有兩套彼此獨立的測試，是刻意保留的 cross-check，請勿刪除其中一套**：`tests_h1a.py` + `common.py`，以及 `test_h1a.py` + `hlib.py` + `test_topology.py`
（`common.py` 與 `hlib.py` 是各自的 helper）。

## 4. 部署與啟用

### 4.1 部署到伺服器

```bash
cd /home/user/trivial/AS-5126GS-TNRT2_vllm_plugin
scripts/deploy_host.sh        # rsync -a --delete 到 user@10.19.209.74:/raid/vllm-runtime/host_topo/（排除 __pycache__、*.pyc、.git、build）
# 試驗用：DEST=<server dir> FROM=<local dir> scripts/deploy_host.sh
```

JIT build 的結果（`_muse_topo_C`）不在本資料夾，而是在 `$VLLM_CACHE_ROOT/muse_topo/<digest>`（server：`/raid/vllm-runtime/sm120-dev/vllm_cache/muse_topo/`）。
digest 由 `csrc` 所有檔案、編譯旗標、torch 與 nvcc 版本決定；改了 `csrc` 之後第一次啟動會重新 build 一次（flock 保護，8 個 rank 只 build 一次；需要 nvcc >= 12.8，
伺服器上是 `/usr/local/cuda` 13.3）。可先用 `scripts/runsrc_topo.sh $HOST_DIR/bench/build_ext.py` 預先 build。

### 4.2 建議做法：用 script 啟動（完整環境、有驗證過）

在伺服器上（`ssh user@10.19.209.74`）。開始前先確認 GPU 閒置（`nvidia-smi`、`sudo docker ps -a`；`nemotron-ultra` 必須維持 stopped）。
所有 GPU 工作一律在共用的 GPU lock 下執行；`locked_run_topo.sh` 會自己拿 lock，**不要再在外面包同一個 lock 的 flock（會 deadlock）**。

```bash
cd /raid/vllm-runtime/host_topo

# 標準 EP8 dummy-weight benchmark（c1 512/128 + ctx 13,107 / 65,536 / 117,964；約 15-25 分鐘），結果在 musetest/results/<tag>/
SEED=9000 scripts/locked_run_topo.sh topo_myrun_ep8

# real weights（HF cache 離線載入）+ greedy chat 檢查，結果另含 results/<tag>/chat.json
LOAD_FORMAT=auto CHAT_CHECK=1 SEED=9000 scripts/locked_run_topo.sh topo_myrun_real_ep8

# 同一條啟動路徑的 NCCL-only baseline（A/B 用）
MUSE_TOPO_ENABLE=0 SEED=9000 scripts/locked_run_topo.sh nccl_myrun_ep8

# 摘要
python3 /raid/vllm-runtime/musetest/summarize.py /raid/vllm-runtime/musetest/results topo_myrun
```

`locked_run_topo.sh` / `run_src_topo.sh` 的常用變數：`SRC`（vLLM source tree，預設 `sm120-dev/src`）、`SEED`、`TP`（8）、`MAXLEN`（131072）、`GPU_UTIL`、`PORT`（8100）、
`READY_TIMEOUT_S`、`SKIP_CTX=1`（只跑 c1）、`LOAD_FORMAT`（預設 `dummy`；`auto` = real weights）、`CHAT_CHECK=1`、`HOST_DIR`（預設為 script 所在的資料夾）。
原本的 `musetest/run_src.sh`、`locked_run.sh`、`sm120-dev/runsrc.sh` 都沒有被修改。

### 4.3 手動啟動 `vllm serve`（從上面的 script 推導，尚未單獨驗證）

只要滿足三件事：`PYTHONPATH` 含本資料夾（讓 `muse_topo-0.1.dist-info` 被找到）、`MUSE_TOPO_ENABLE=1`、JIT 用得到 nvcc 與可寫的 `VLLM_CACHE_ROOT`。

```bash
source /raid/vllm-runtime/sm120-dev/.venv/bin/activate
HOST=/raid/vllm-runtime/host_topo
export PYTHONPATH=$HOST:/raid/vllm-runtime/sm120-dev/src
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export VLLM_CACHE_ROOT=/raid/vllm-runtime/sm120-dev/vllm_cache
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
source $HOST/env/host.env          # NCCL_P2P_LEVEL=SYS、MUSE_TOPO_ENABLE=1（可用同名環境變數預先覆寫）
flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock \
  vllm serve unsloth/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 8 --enable-expert-parallel \
    --max-num-seqs 128 --kv-cache-dtype fp8 --quantization modelopt_fp4 --max-model-len 131072 \
    --gpu-memory-utilization 0.95 --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
```

上面的 flock 在 server 存活期間會一直佔著共用 GPU lock；常駐服務請先與其他使用者協調。若設定了 `VLLM_PLUGINS`，其中必須包含 `muse_topo`。

### 4.4 確認有生效

serve log 應該出現（每個 process / 每個 worker 各一行）：

```
muse_topo plugin registered in pid <pid> (ops=all_gather,all_reduce,reduce_scatter)
muse_topo active on rank <r> (island <I>, partner <P>, workspace 61.1 MiB): islands=[[0, 1, 2, 3], [4, 5, 6, 7]] partner_of=[4, 5, 6, 7, 0, 1, 2, 3] ...
```

拓樸不吻合、build 失敗或 `MUSE_TOPO_ENABLE != 1` 時改為 `muse_topo disabled on rank <r>: <原因>` / `muse_topo not used for group ...`，所有 rank 的決定一致，
vLLM 繼續使用 NCCL（不會 crash）。看每個 collective 實際走哪條路徑：`MUSE_TOPO_LOG_OPS=1`（group rank 0）或 `2`（每個 rank），再用
`python3 bench/summarize_ops.py <serve.log>`（若有任何 key 走 NCCL，加 `--paths` 會列出並附原因）。

### 4.5 關閉與 A/B

`MUSE_TOPO_ENABLE=0` 即回到純 NCCL（同一條啟動路徑）。只關 large-message 演算法：`MUSE_TOPO_LARGE=0`；只讓部分 op 走 topo：`MUSE_TOPO_OPS=all_reduce`。

## 5. 設定一覽與 accuracy class

accuracy class 的定義（對照目前 branch 的輸出）：

* `identical`：bitwise 相同。預設 ON。
* `more_accurate`：每個元素都可證明更接近精確數學結果（例如 fp32 固定順序加總、單次最終 rounding，取代一串 bf16 rounding）。預設 ON。
* `deterministic_different`：run-to-run 可重現、各 TP rank 相同，但 rounding 不同且不能證明更好。原先 opt-in、預設 OFF；使用者 2026-10-24 的吞吐量決策（「rounding 差異只要不影響最終 sampling 就接受」）之後，只要更快、確定性、rank 相同、沒有系統性偏差且 GSM8K 沒有退步，就可以預設 ON：本資料夾的 bf16 exchange 與 NCCL graph file 由 `env/host.env` 預設開啟（plugin 本身的預設仍是 OFF）。
* `nondeterministic`：不出貨。

| 設定 | 預設 | accuracy class | 說明 / 量測 |
|------|------|----------------|-------------|
| `NCCL_P2P_LEVEL=SYS`（`env/host.env`） | ON | identical | NCCL 2.29.7 在本機 graph search 失敗，4 個跨 socket hop 走 host SHM；`SYS` 讓所有 hop 走 P2P，ring / channels / protocol 不變。129 個 shape（eager + graph）與 end-to-end 20/20 requests bitwise 相同；128 MiB all-reduce 12.04 -> 9.11 ms（2026-10-21）。EP8 dummy（同一棵 tree，只差這個變數）：c1 3700 / 42.2 -> 4318 / 47.0 t/s（pp +16.7 %、tg +11.4 %），ctx 列 pp +15 .. +17 % |
| `MUSE_TOPO_ENABLE=1`（`env/host.env`），all_reduce 與 reduce_scatter | ON | more_accurate | 見第 6 節。算術：`P_I = ((x_a + x_b) + x_c) + x_d`（fp32，island 內 rank 升冪），`out = round(P_0 + P_1)`；每個 size、演算法、rank、run 之間 bitwise 相同（batch invariant） |
| `MUSE_TOPO_ENABLE=1`，all_gather | ON | identical（純資料搬移） | 直接產生要求的 layout（dim 0 = rank-major；最後一維 = column concatenation），與 NCCL 的結果逐 bit 相同 |
| `MUSE_TOPO_LARGE=0` | 1（large 演算法 ON） | 不影響數值 | A/B 開關：all-reduce > 8 MiB、reduce_scatter、all-gather shard > 1 MiB 改走 NCCL（即 H2 版行為） |
| `MUSE_TOPO_OPS` | `all_reduce,all_gather,reduce_scatter,boundary` | 視 op 而定 | 指定哪些 op 走 topo；`boundary` = 下一列的 fused kernel |
| `MUSE_TOPO_FUSED=0`、`MUSE_TOPO_BND_LOADS` | fused ON、loads 0 | identical（前提：`MUSE_TOPO_MEDIUM_BX=0`；medium BX 開啟時見下） | decode（T <= 8，pooler gather T <= 16）時把 attention-out all-reduce / unpooler all-gather 之後的 residual boundary、pooler all-gather 之後的 pre-expert RMSNorm 當作 collective kernel 的 epilogue（`csrc/boundary.cuh` 是 vLLM Triton kernel 的 bit-exact CUDA 版本，已對 Triton SASS 驗證，見 `docs/r2d_fused_epilogue.md`）。vLLM 端的開關是 `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY`（預設 1）。`MUSE_TOPO_BND_LOADS` 選擇 epilogue 運算元（stream / alpha / beta / weight）的載入位置（0..3，預設 0 最快），不影響數值。**注意**：fused kernel 恆用 exact 的 fp32 exchange；`env/host.env` 預設的 medium BX 開啟時，關掉 fused（`MUSE_TOPO_FUSED=0`、`MUSE_TOPO_OPS` 不含 `boundary`、或 vLLM 端 `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=0`）會讓 T = 4..8 的 attn-out all-reduce（64..128 KiB）改走 two-step 的 `comm.all_reduce`，即 medium BX（bf16 exchange），bits 就不同（deterministic_different，不再 identical）。要做 bit-identical 的 fused 開 / 關 A/B，兩次都要設 `MUSE_TOPO_MEDIUM_BX=0`（fused 開 + `MEDIUM_BX=0` 對 fused 關 + `MEDIUM_BX=0`） |
| `MUSE_TOPO_MAX_BYTES` | 8 MiB（`TopoComm` 預設） | 不影響數值 | 只限制 decode / medium 演算法的訊息大小並決定 workspace 大小；更大的 all-reduce（到 2 GiB）本來就走 large 演算法，沒有理由調高 |
| `MUSE_TOPO_TIMEOUT_S` | 600（plugin） | - | kernel 等不到 peer 時 `__trap()` 的秒數（`TopoComm` 單獨使用預設 10 s；vLLM 啟動時各 rank 可能差數十秒，所以 plugin 用 600） |
| `MUSE_TOPO_LOG_OPS` | 0 | - | 每個 (op, dim, shape, dtype, stream, capturing, path) 第一次呼叫印一行；0 = debug level、1 = rank 0、2 = 每個 rank |
| `MUSE_TOPO_LARGE_BF16_EXCHANGE=1` | plugin 預設 OFF；**`env/host.env` 預設 ON**（2026-10-24 決策；`=0` 回到 exact 的 fp32 exchange） | **deterministic_different** | 大訊息 all-reduce / reduce_scatter 的跨 socket partial 先 round 成 bf16（跨 socket bytes 減半）。只對 bf16；fp16 / fp32 維持 exact。確定性、rank 相同，但不是 fp32 加總單次 rounding 的結果；batch invariant 只在 bf16 exchange 的範圍內：每個 >= `ar_bx_min`（64 KiB，`MUSE_TOPO_MEDIUM_BX=1` 時；`=0` 則是 2 MiB 的 large 門檻）的 bf16 all-reduce，以及所有 reduce_scatter（>= 128 KiB，flag 開啟時一律走 large 演算法加 bf16 exchange），同一 row 在任何 size 下 bits 相同，reduce_scatter 與 all-reduce 從 128 KiB 起 bitwise 相同；較小的 all-reduce 與 T <= 8 的 fused kernel 維持 exact，所以 bits 會在 64 KiB 門檻改變：embedding all-reduce `[T, 8192]`（從不 fused）eager 在 T 3 -> 4、graph 的 padded T 2 -> 4；MoE-out `[T, 4096]` eager 在 T 7 -> 8、padded T 4 -> 8；fused 開啟時 attn-out 在 T <= 8 一律 exact（`all_reduce_boundary` 在 T = 3 與 4 都 exact），T >= 9 起才走 bf16 exchange。final defaults 的驗證（real weights，2026-10-26，vLLM `muse/integ` 468ea2ed86 + 本資料夾，其中大訊息 all-reduce / reduce_scatter 走 bf16 exchange）：GSM8K 200 x 2 為 96.5 %（192 / 200、194 / 200；對照 NCCL 96.2 %、plugin 95.5 %、+SP+bf16x 96.2 %），兩個 cold process 的 determinism A/B（56 個 requests 加 in-process 重複，token ids / logprobs / prompt logprobs）bitwise 相同。128 MiB all-reduce 7.10 -> 5.84 ms |
| `MUSE_TOPO_MEDIUM_BX`、`MUSE_TOPO_MEDIUM_BX_MIN_BYTES` | **預設 ON**（1；只在 `MUSE_TOPO_LARGE_BF16_EXCHANGE=1` 時生效）、64 KiB | **deterministic_different** | 同一個 bf16 exchange 用在 medium 的 bf16 all-reduce（64 KiB .. 8 MiB：decode 的 embedding `[T, 8192]`（從不 fused）在 T >= 4（graph 的 padded T >= 4）、attn-out `[T, 8192]` 在 T >= 9（CUDA graph 的 padded T = 16 起；T <= 8 是 exact 的 fused kernel）與 MoE-out `[T, 4096]` 在 T >= 8（padded T >= 8）；演算法 2 的 `BX` 變體，與 large 演算法逐 bit 相同）。跨 socket bytes 減半：1 MiB 62.5 -> 43.1 us、512 KiB 33.3 -> 22.6 us；real weights c16 / c64 / c128 out tok/s +2.2 / +2.9 / +1.9 %；GSM8K 97.5 / 96.5 %（見 `docs/r3c_medium_ar.md`）。`=0` 回到 exact 的 fp32 exchange（2 MiB 以下）。T <= 8 的 fused attn-out kernel 維持 exact，但只在 fused 開啟時；fused 關閉時 T = 4..8 的 attn-out 會走 BX（見 `MUSE_TOPO_FUSED` 一列，fused A/B 要兩次都設 `MUSE_TOPO_MEDIUM_BX=0`）。`MUSE_TOPO_AR_TUNING_FILE` 是實驗用 hook（這幾個設定在所有 rank 必須相同） |
| `MUSE_TOPO_GEMV`（vLLM 端的對應開關：`VLLM_MUSE_SPARK_SM120_FUSED_GEMV`） | **預設 ON**（1；需要 `MUSE_TOPO_FUSED` != 0 與 vLLM 端 `VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=1`；兩邊都要是 1 才會啟用） | **deterministic_different**（對 cuBLAS GEMV）；pooler 在 T >= 2 是 `more_accurate`、T = 1 與 unpooler 相當 | decode（padded T <= `MUSE_TOPO_GEMV_MAX_ROWS`，預設 4，最多 8）時，pooler（`[T, 8192]` -> 512 columns）與 unpooler（`[T, 4096]` -> 1024 columns）的 sharded GEMV 和其後的 all-gather + epilogue 由兩個 kernel 組成一次 collective 呼叫（`csrc/gemv_push.cu`、`ag_bnd.cu` 的 `pushed` 版本，細節見 `docs/r3b_gemv_push.md`）：GEMV kernel 在前一個 all-reduce 還在跑時就把 8 MB 權重載入 register，每個 CTA 算完就把 columns push 給 partner 與 island peers。算術：fp32 固定順序加總、單次 bf16 rounding（`csrc/gemv.cuh`），與 T、rank、run 無關（同一個 token row 在任何 batch size 下 bits 相同），但不是 cuBLAS 的順序。對 fp64 精確值（mixed data）：此順序 pooler 為 99.97 .. 100 % correctly rounded、mean 0.25 ulp、max 0.5 .. 1.2 ulp；cuBLAS 在 T = 2..8（bf16 split-K workspace）只有 58 .. 61 %、mean 1.3 .. 3.5 ulp、max 260 .. 16800 ulp，T = 1 與 unpooler 兩者相當（`bench/test_gemv_push.py` 第 3 段、`docs/r3b_gemv_push.md`）。與 router GEMV 一起量測見第 6.4c 節。關閉（回到 cuBLAS GEMV + r2d gather）：`MUSE_TOPO_GEMV=0` 與 `VLLM_MUSE_SPARK_SM120_FUSED_GEMV=0` 一起設（只設 vLLM 端只是不提供給 plugin） |
| `MUSE_TOPO_GEMV_TRIGGER` | 3 | 不影響數值 | 前面的 all-reduce kernel 何時讓 GEMV kernel 開始：0 = 不觸發（all-reduce 完成才開始）、1 = all-reduce 一開始（8 MB 權重載入與 all-reduce 第一個 phase 搶頻寬，all-reduce 慢 4 .. 5 us，e2e 沒有增益）、2 = 最後一個 communication phase 開始、3 = 最後一個 communication phase 結束（此時 all-reduce 的 output，以及 `ar2_bnd_kernel` 的 boundary epilogue，還沒寫完）。3 是 S1-4a 實測 arm 的設定；單用 GEMV 時 2 與 3 在雜訊內相同（S1-4a：c1 -2.2 % 兩者相同；c4 -1.1 % 與 -0.1 %，在 repeat 的 1.6 % spread 內）。S1-4b 決定性量測（含 router GEMV，規則：c1 與 c4 TPOT 平均較低者、差距 0.5 % 內選 2）：trigger 3 的 mean(c1, c4) = 12.001 ms，trigger 2 = 12.073 ms，3 低 0.60 %，所以是 3；c1 相同（9.214 / 9.209 ms），差異全在 c4（14.788 / 14.937 ms），而 c4 的 repeat spread（1.4 %）與開頭到結尾的 drift（+1.1 %）一樣大，用開頭與結尾兩次的平均來比只差 -0.34 %（落在 tie 範圍內，事後分析、未用於決定），所以這個預設**不穩健**，在 c1 沒有可量測的效果。只有 `ar2_kernel` / `ar2_bnd_kernel` 會觸發 2、3；`MUSE_TOPO_PDL=1` 時所有 kernel 一開始就觸發（stage 2、3 無效，init 時 log 一次）。**stage 是單次 launch 的屬性，不是 communicator 的**：只有接在 GEMV 前面的兩個 all-reduce 會帶 stage（`all_reduce_boundary` 且 T <= `MAX_ROWS`，後面是 pooler GEMV；bf16 `[T, 4096]` 的 `all_reduce` 且 T <= `MAX_ROWS`，即 MoE latent，後面是 unpooler GEMV），其餘（embedding all-reduce、T > `MAX_ROWS`、prefill、其他模型）維持 kernel 結束才 trigger 的行為。帶 stage 的 all-reduce 後面若有 programmatic launch 的 kernel，必須在讀 output 之前呼叫 `griddepcontrol.wait`（見 `docs/r3b_gemv_push.md` Protocol） |
| `MUSE_TOPO_GEMV_PUSH`、`MUSE_TOPO_GEMV_MAX_ROWS`、`MUSE_TOPO_GEMV_VARIANT` | 1、4、未設（依 row 數選） | PUSH、MAX_ROWS 不影響數值；VARIANT 0 / 3 同一個加總順序，1 / 2 是另一個順序（deterministic，rounding 不同） | `PUSH` 的 bits 0-1 = push 模式（0 = 只存、由 gather kernel push；1 = 每個 CTA 算完就 push；2 = 最後一個 CTA 用 coalesced store push），+4 = 權重在等前一個 kernel 之後才載入。合法值 0、1、2、4、5、6；**3、7 與範圍外的值會被拒絕**（不 push 卻告訴 gather kernel 已 push，peers 會 spin 到 watchdog trap）：`TopoComm` 初始化時由 stage-1 agreement 檢查（任何一個 rank 的值不合法，所有 rank 都退回 NCCL，不會卡住），`gemv_push` 屬性與三個 C++ 入口也各檢查一次。S1-4a：late weights（5）與 2 / 6 都沒有比 1 好。`MAX_ROWS` = GEMV kernel 服務的最大 padded decode batch（env 值 clamp 到 0 .. 8，0 = 不使用）；超過就維持 cuBLAS + r2d gather。**預設 4**：S1-4b 在 padded T = 8（c5 .. c8，也是 K = 7 speculative decoding 的 verify batch）量測，c8 的 TPOT 在 `MAX_ROWS` 8 / 4 / 只有 router GEMV / 全部關閉 = 19.992 / 19.897 / 19.793 / 19.738 ms（c6：22.972 / 23.005 / 22.841 / 23.071 ms）；規則（8 不比 4 差 0.5 % 以上才用 8）只以 0.02 % 之差成立，GEMV 在 padded T = 8 沒有可量測的增益，三組比較都偏慢，早期 microbenchmark 的 T = 8 layer chain 也比 cuBLAS 慢（53.7 vs 52.1 us），V3 在 T = 8 還有 register spill（`ptxas -v`：unpooler 64 bytes、pooler 96 bytes spill stores，T <= 7 沒有），所以 orchestrator 決定預設為 4（padded T = 1、2、4 使用 GEMV）。`VARIANT` 未設時 T = 1 用 V0（128 CTAs）、T >= 2 用 V3（64 CTAs x 512 threads）；以上都要在所有 rank 相同 |
| `VLLM_MUSE_SPARK_SM120_ROUTER_GEMV`（vLLM branch 的 flag，不是 plugin 的） | **預設 ON**（1） | **deterministic_different**（對 cuBLAS router） | decode（T <= 8）的 fp32 router `[T, 8192] x [256, 8192]^T` 改用 `_muse_C::router_gemv`：一個 kernel 讀一次 8 MiB 權重，加總順序固定、與 T / rank / run 無關（cuBLAS 在 T = 1 與 T >= 2 用不同演算法，不是 batch invariant），也讓後面的 routing kernel 提早啟動。與 cuBLAS 的差異在最後幾個 bit，測試限定誤差不超過 cuBLAS 的（最大 relative error 8 倍、mean absolute error 2 倍）。輸入的 fp32 router input 在 decode graph 中只有 expert prefetch 開啟時才由 boundary kernel 寫出，否則 router 維持 cuBLAS，vLLM log 會印一行 warning。`=0` 回到 cuBLAS router。這是 vLLM branch 的 flag：沒有 host plugin 的 NCCL-only 執行也會用 router GEMV（routing rounding 與舊的正式版不同）。單用 router GEMV：c1 -1.0 %、c4 -0.5 %；與 GEMV 融合一起：c1 -4.2 %、c4 -3.2 %（第 6.4c 節） |
| `VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES`（vLLM branch 的 flag） | 12582912（12 MiB） | identical（cache hint，不寫任何 tensor） | routing-aware expert L2 prefetch 每次拉進 L2 的 bytes（第一個 expert 的開頭，GEMM1 再 GEMM2）。預設與原本的常數相同，行為不變；> 0。更大的 budget 要有更早的 routing ids 才划算（router GEMV 讓 ids 提早），太大會與 kernel 自己的載入搶 HBM（整個 rank 的 96 MiB 比不 prefetch 慢 1.2 %）。S1-4b 的 sweep（c1 TPOT）：12 MiB 9.180 ms（兩次平均）、20 MiB 9.202（+0.24 %）、28 MiB 9.230（+0.55 %），沒有任何值勝過 12 MiB 達 1 %，維持 12 MiB |
| `NCCL_GRAPH_FILE=<host folder>/nccl/graph_r4.xml` | **`env/host.env` 預設 ON**（檔案存在時 export；呼叫端已設定 `NCCL_GRAPH_FILE` 時以呼叫端的值為準） | **deterministic_different** | 合法的 4-channel ring，NCCL 在 <= 512 KiB 改用 LL。2026-10-21 實測 decode 51.2 -> 70.2 t/s（prefill 不變），但**不是 bitwise identical**（小而奇數 row 的 shape 有 1 bf16 ulp 差；end-to-end 3/20 mixed-batch requests 在 decode 發散，|dlogprob| <= 3.8e-6）。與 topo plugin 同時開啟時，TP collectives 都已繞過 NCCL，graph file 只影響 fallback（非 TP group 與不在 envelope 的呼叫）；這個組合已隨 final defaults 的 GSM8K 與 determinism A/B 驗證過（見上一列），throughput 列（`r2fin_*`）另行量測 |
| `VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP`（vLLM branch 的 flag，不是 plugin 的；plugin 只負責把 comm stream 上 collective 的結果配置在 compute stream，見 `docs/h2_vllm_plugin.md`「Microbatched prefill」） | ON | deterministic_different | 把 eager batch（>= 2050 tokens）切成兩個 microbatch，一個的 TP collective 在 comm stream 上與另一個的 compute 重疊。real weights prefill（ctx10 / 50 / 90）：replicated stream +33 / +30 / +27 %，sequence-parallel stream +14 / +13 / +11 %；decode 不變。run-to-run bitwise 相同（兩個 process、replicated 與 SP 兩條 stream 都驗證過）；與不切分的輸出有 rounding 等級的差異。只在 V2 runner 能 microbatch 的 launch 生效：需 `--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'`（所有 modality 都要設 0），且沒有 PIECEWISE CUDA graph / context parallel / spec decode / LoRA / `VLLM_BATCH_INVARIANT`；其他 launch 上 flag 無作用，也不會改變 runner。memory profile 會多跑一次切分後的 batch，KV pool 約少 0.65 GiB / GPU |
| `MUSE_TOPO_PDL=1`、`MUSE_TOPO_STRICT_STREAM=1`、`MUSE_TOPO_VERBOSE=1`、`MUSE_TOPO_TRACE=1` | OFF | - | 實驗 / 除錯用：PDL（back-to-back -0.3 us，kernel 之後 +0.1 .. 0.2 us，所以關閉）、第二條 stream 直接報錯、JIT build log、in-kernel timeline（需另一份 JIT build，只給 `trace_*.py`） |

## 6. 實測效果

量測日期與方法一併列出。所有 pp / tg 皆為 concurrency 1、隨機 prompt、同一組 seed（`SEED=9000`），pp = input tokens / 平均 TTFT，tg = 1000 / 平均 TPOT(ms)；
EP8、`--gpu-memory-utilization 0.95`、fp8 KV、`--max-num-seqs 128`、max-model-len 131072。結果在伺服器 `/raid/vllm-runtime/musetest/results/<tag>/`。

### 6.1 Microbenchmark（CUDA graph，bf16，8 processes，max over ranks）

2026-10-23 量測；2026-10-24 的 review fix（64-bit counter / flags）之後重測，decode 大小的 all-reduce 差在 +0.4 % 內，LL8 all-gather 約慢 0.1 us，large 差在 +-0.5 % 內。
NCCL 為 vLLM 的 PyNcclCommunicator（`NCCL_P2P_LEVEL=SYS`）。完整表格：`docs/h1a_latency.md`、`docs/h1b_latency.md`。

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

### 6.2 End-to-end，dummy weights（pp / tg，t/s）

| 列 | `c_muse_ep8`（基準，2026-10-21，NCCL SYS） | `topo_final_ep8`（**出貨版**，2026-10-24） | 變化 |
|----|------|------|------|
| c1 512/128 | 4354 / 51.2（TTFT 118 ms、TPOT 19.5 ms） | 4674 / 85.4（TTFT 110 ms、TPOT 11.7 ms） | pp +7.3 %、tg +66.8 % |
| ctx10（13,107 tokens） | 5342 / 51.6 | 6097 / 87.3 | pp +14.1 %、tg +69.2 % |
| ctx50（65,536 tokens） | 5208 / 51.3 | 5903 / 85.8 | pp +13.3 %、tg +67.3 % |
| ctx90（117,964 tokens） | 5021 / 51.6（TTFT 23,495 ms） | 5682 / 86.6（TTFT 20,763 ms） | pp +13.2 %、tg +67.8 % |

* 同一棵 source tree 的 NCCL 對照（`MUSE_TOPO_ENABLE=0`）：`nccl_final_ep8`（c1）4303 / 51.2；`nccl_h2b_ep8` 4324 / 51.2、5413 / 51.7、5237 / 51.4、5025 / 51.7（TTFT 23,473 ms）。
  以 `nccl_h2b_ep8` 為對照時 prefill 增益為 +9.4 / +11.9 / +12.4 / +12.9 %（`topo_h2b_ep8`）。
* `topo_final_ep8` 是 review fix 之後的出貨 kernel；review fix 之前的 `topo_h2b_ep8` 為 4732 / 85.4、6059 / 87.2、5884 / 85.8、5671 / 86.7，兩者在 run-to-run noise 內
  （c1 pp -1.2 %、ctx 列 +0.2 .. +0.6 %）。
* decode 增益來自小訊息演算法（TPOT 19.5 -> 11.7 ms）；prefill 增益來自 large 演算法：ctx90 每個 8192-token chunk 省約 186 ms（單獨量測預測 194 ms）。
* 對照：H2 版（只有小訊息演算法，`topo_h2_ep8`）的 decode 相同（85.4 .. 86.9 t/s），但 prefill 幾乎沒有增益（c1 +2.8 %、ctx 列 +0.1 .. +0.5 %），因為 128 MiB all-reduce 仍在 NCCL。
* 每個 collective 都走 topo：6 層 e2e 的 per-op 摘要為 188 個 key、188 走 topo、0 走 NCCL、0 個 rank 不一致（最大到 `[16384,8192]` 的 256 MiB all-reduce 與 `[1024,25256]` 的 logits gather）。

### 6.3 End-to-end，real weights（62 層、`unsloth/Muse-Spark-Open-816B-A42B-NVFP4`，`LOAD_FORMAT=auto CHAT_CHECK=1`，2026-10-24）

| 列 | `nccl_real_ep8`（`MUSE_TOPO_ENABLE=0`） | `topo_h2b_real_ep8` | 變化 |
|----|------|------|------|
| c1 512/128 | 3666 / 52.0（TTFT 140 ms、TPOT 19.2 ms） | 3968 / 88.4（TTFT 129 ms、TPOT 11.3 ms） | pp +8.2 %、tg +70 % |
| ctx10 | 5342 / 52.2 | 5990 / 88.8 | pp +12.1 %、tg +70 % |
| ctx50 | 5168 / 52.7 | 5795 / 89.4 | pp +12.1 %、tg +70 % |
| ctx90 | 4955 / 53.0（TTFT 23,806 ms） | 5522 / 89.6（TTFT 21,362 ms） | pp +11.4 %、tg +69 % |

`topo_h2b_real_ep8` 是 review fix 之前的 kernel；修正後的 tree 在 real weights 上的輸出與修正前 bitwise 相同（51/51 requests，max |dlogprob| = 0），所以這些列仍適用
（fix 屬於 `identical`，dummy-weight 出貨版的效能差異在 noise 內，見 6.2）。

### 6.4 bf16 exchange（`MUSE_TOPO_LARGE_BF16_EXCHANGE=1`，現為 `env/host.env` 的預設；dummy weights，2026-10-24，`topo_h2b_bf16x_ep8`）

pp / tg：c1 4989 / 85.5、ctx10 6814 / 88.1、ctx50 6473 / 86.2、ctx90 6186 / 87.1（TTFT 19,069 ms）。對 `c_muse_ep8`：pp +14.6 / +27.6 / +24.3 / +23.2 %；對 exact 的 `topo_h2b_ep8`：
pp +6.7 / +11.8 / +9.7 / +8.9 %（ctx90 每個 chunk 再省約 120 ms）。decode 不變（decode 的 all-reduce 低於 large 門檻）。accuracy class 為 `deterministic_different`；plugin 本身預設 OFF，`env/host.env` 預設 ON（見第 5 節）。

### 6.4b medium all-reduce 的 bf16 exchange（r3c，`MUSE_TOPO_MEDIUM_BX`，預設 ON；real weights，2026-10-26，`r3c_c1_conc_bx{0,1}`、`r3c_c1_acc`）

同一個 server 設定、每列一個 session、每個 concurrency 兩次重複（random 512/128，out tok/s / TPOT ms；`bx0` = `MUSE_TOPO_MEDIUM_BX=0` = r2fin 行為）：

| concurrency | bx0 | bx1（預設） | 差異 |
|---|---|---|---|
| 1 | 96.3 / 9.63 | 96.7 / 9.58 | +0.5 %（noise） |
| 16 | 481.1 / 27.41 | 491.5 / 26.43 | +2.2 % |
| 64 | 905.7 / 56.69 | 931.8 / 54.70 | +2.9 %（TPOT -1.99 ms） |
| 128 | 1156.5 / 91.79 | 1178.9 / 89.66 | +1.9 % |

microbenchmark（CUDA graph b2b，us）：256 KiB 18.9 -> 13.8、512 KiB 33.3 -> 22.5、1 MiB 62.5 -> 43.1、2 MiB 123.7 -> 82.8、8 MiB 480 -> 321；在 8 MiB 以內都比 large 演算法快。GSM8K（real weights、c64、200 題 x 2 pass）97.5 / 96.5 %（沒有 medium exchange 的 `r2integ_acc`：96.0 / 97.0 %）。完整表格、測試與限制見 `docs/r3c_medium_ar.md`。

### 6.4c pooler / unpooler GEMV 與 all-gather 融合 + router GEMV（r3b，`MUSE_TOPO_GEMV`、`VLLM_MUSE_SPARK_SM120_ROUTER_GEMV`；real weights，S1-4a，2026-10-27）

協定：random 512/128、EP8 + TP8、fp8 KV、max-model-len 131072、memory 0.95、max-num-seqs 128、正式的 `env/host.env`；一次 GPU lock 內每個 arm 一個 server session，每個 concurrency 兩次重複
（SEED 9000 + i，c1 / c4 / c16 / c64 的 prompts 數 32 / 32 / 64 / 256），表中是 TPOT ms（兩次的平均）。`A0` = 正式版行為（`MUSE_TOPO_GEMV=0`、`VLLM_MUSE_SPARK_SM120_FUSED_GEMV=0`、
`VLLM_MUSE_SPARK_SM120_ROUTER_GEMV=0`），session 開頭與結尾各量一次，百分比對兩者的平均（9.593 / 15.256 / 26.849 / 53.941 ms）。c16 / c64（batch > 8 tokens）不會被這些改動影響，
它們的變動（約 1 %）就是 session 雜訊。

| arm | 設定 | c1 | c4 | c16 | c64 |
|-----|------|----|----|-----|-----|
| A0 開頭 / 結尾 | 全關 | 9.606 / 9.580 | 15.301 / 15.210 | 26.709 / 26.988 | 54.086 / 53.797 |
| A1 | GEMV 融合，trigger 2 | 9.385（-2.2 %） | 15.094（-1.1 %） | 26.447 | 53.608 |
| A2 | GEMV 融合，trigger 3 | 9.379（-2.2 %） | 15.241（-0.1 %） | 26.632 | 53.604 |
| A4 | 只有 router GEMV | 9.495（-1.0 %） | 15.175（-0.5 %） | 26.727 | 53.787 |
| **A5** | A2 + router GEMV（**預設組合**） | **9.187（-4.2 %）** | **14.763（-3.2 %）** | 26.159 | 53.989 |
| BEST2 | A5 在第二個 session | 9.178（-4.3 %） | 14.737（-3.4 %） | 26.929 | 54.013 |

late weights（push 5）、last-CTA push（2）與兩者合用（6）都沒有勝過預設的 push 1（c1 9.456 / 9.424 / 9.495，c4 15.480 / 15.485 / 15.611）。兩項改動合起來比個別相加更多
（GEMV -2.2 %、router GEMV -1.0 %、兩者 -4.2 %）；profile 顯示只融合 GEMV 時 router chain（cuBLAS dot + reduce + routing = 10.5 us）成為 layer 的長桿。
S1-4b 決定性量測（2026-10-27，一次 lock、11 個 session，c1 / c2 / c4 / c6 / c8）補上 S1-4a 沒量的部分：trigger 2 對 3（加 router GEMV）、padded T = 8、expert prefetch budget 20 / 28 MiB；結果與規則見上面各環境變數的說明，完整的表格見 `docs/r3b_gemv_push.md`。詳細的 profile、microbenchmark 與精度表同樣在那裡。

### 6.5 NCCL graph file（2026-10-21，dummy weights，`c_muse_gf_ep8`，當時未採用；2026-10-24 起 `env/host.env` 預設使用，只影響 plugin 之外的 NCCL collectives）

c1 4366 / 70.2（TPOT 14.3 ms）、ctx10 5331 / 71.0、ctx50 5201 / 70.4、ctx90 5010 / 70.5：decode +37 %、prefill 不變，但不是 bitwise identical（第 5 節）。topo plugin
（tg 85.4 .. 87.3 t/s，且為 `more_accurate`）在 decode 上比它再快約 22 .. 23 %。

### 6.6 記憶體

* workspace 每張 GPU 64,094,208 B = 61.13 MiB（large 演算法重用同一塊）。它在 `CudaCommunicator.__init__` 配置，早於 vLLM 的 `init_snapshot`，所以不會從 KV budget 扣除，
  而是縮減 `1 - gpu_memory_utilization` 的 headroom（0.95 時有約 4.8 GiB，沒有影響；只有 utilization 接近 1.0 才需要注意）。
* READY 時 GPU 0 為 91,796 MiB（NCCL 為 91,786 MiB），因為 captured graph 變小（NCCL 為每張 graph 配置 buffer，topo kernel 不用）；KV pool 每張 GPU 8.48 GiB（1,563,909 tokens）對 NCCL 的 8.21 GiB（1,513,149 tokens）。

## 7. 驗證方式

下列指令皆在伺服器上執行（`cd /raid/vllm-runtime/host_topo`，先 `scripts/deploy_host.sh`）。GPU 測試要在 GPU lock 下跑：`scripts/locked_job.sh <tag> <timeout_s> <command...>`
會等 lock、等 GPU 記憶體釋放、detached 執行，log 在 `/raid/vllm-runtime/work/host/logs/<tag>.log`，完成後寫 `<tag>.DONE`（內容為 return code）。scratch 與結果不要寫進本資料夾。

### 7.1 不需要 GPU 的快速檢查

```bash
export HOST_DIR=/raid/vllm-runtime/host_topo
scripts/runsrc_topo.sh $HOST_DIR/bench/test_plugin_unit.py        # plugin 的 routing / fallback 原因（mock TopoComm，CPU）
scripts/runsrc_topo.sh $HOST_DIR/bench/test_topology.py           # 拓樸推導：任意 rank 排列都是 collapse-free（200 組隨機排列）
scripts/runsrc_topo.sh -c "from vllm.plugins import load_general_plugins; load_general_plugins()"   # plugin 被 dist-info 找到並註冊
```

### 7.2 GPU 上的 correctness 測試（8 processes，不需要完整模型）

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
r3b 的 GEMV + all-gather 融合另有 `bench/test_gemv_push.py`（8 ranks，T = 1..8、兩個 projection、所有 variant 與合法的 push 模式、每個 trigger stage（exact 與 bf16 exchange 的 all-reduce）、eager 與 CUDA graph、3 / 7 的拒絕、PDL=1、未對齊的權重、programmatic sentinel kernel；
對 torch 的 emulation 與「`all_gather` + 模型的 Triton kernel」逐 bit 比較，並與 fp64 精確值及 cuBLAS 比精度）；用法：`GPU=0,1,2,3,4,5,6,7 SRC=<vLLM tree> scripts/py_topo.sh bench/test_gemv_push.py [--quick]`（GPU lock 下）。

### 7.3 End-to-end determinism 與 NCCL 對照

```bash
bash scripts/locked_job.sh v2 7200 bash scripts/run_h2.sh v2 e2e_topo1 e2e_topo2 e2e_nccl compare ops lp_nccl lp_topo lp_compare   # 6 層 dummy weights
bash scripts/locked_job.sh v3 7200 bash scripts/run_h2.sh v3 real_topo1 real_topo2 real_nccl real_compare                        # 完整 62 層、real weights
```

`run_h2.sh` 內部用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（multiprocess engine 的 request admission 會改變 batch 組成，無法做 bitwise A/B）。結果在
`/raid/vllm-runtime/work/host/h2/`。預期與已觀察到的結果：

* topo vs topo（兩個獨立 process）：token ids、logprobs、prompt logprobs 全部 bitwise 相同。dummy weights 20/20（in-process 重跑 8/8），real weights 51/51，max |dlogprob| = 0。
* topo vs NCCL：**預期不會 bitwise 相同**（`more_accurate`）。real weights：33/51 requests 的 48-token greedy continuation 相同，其餘在近似平手的 token 分歧；teacher-forced NLL 在 noise 內相同
  （NCCL 1.7321 / 1.3184 / 1.1796，topo 1.7241 / 1.3172 / 1.1801 nats/token）；`bench/chat_check.py` 的 5 個 greedy chat 在 NCCL 與 topo 上都連貫且正確。
  dummy weights 的 logits 幾乎是平的，greedy 由 tie 決定，所以那邊只能當作擾動大小的上限（teacher-forced：6150 個值 mean |d| 2.2e-6、max 1.3e-5）。
* 每個 collective 都走 topo：`ops` step（`summarize_ops.py`）顯示 188 keys、188 topo、0 NCCL。

accuracy 的量測（`tests_h1a.py` / `tests_h1b.py` / `test_plugin.py`，fp64 reference，單次 rounding 的 `round_single` 為標準答案）：activation-like 資料 100 % correctly rounded，
平均誤差 0.236 ulp（randn）/ 0.250 ulp（mixed exponent），NCCL bf16 ring 為 47.6-47.8 % / 55.4-55.6 % correctly rounded、平均約 1.6-2.1 ulp；相對 L1 norm 的最大誤差 0.0039（= bf16 unit roundoff）對 NCCL 的 0.012-0.018。
指數範圍極寬的病態資料（約 +-24 個二進位數量級）可能有 0.13 % 的元素不是 correctly rounded，保證的上界是 0.5 ulp + 2^-16 相對誤差。

### 7.4 效能 A/B

```bash
SEED=9000 scripts/locked_run_topo.sh topo_x_ep8                          # 本資料夾預設（topo plugin + env/host.env 的 bf16 exchange 與 NCCL graph file）
MUSE_TOPO_ENABLE=0 SEED=9000 scripts/locked_run_topo.sh nccl_x_ep8       # 同一棵 tree、同一條啟動路徑的 NCCL 基準（host.env 的 graph file 仍然有效）
MUSE_TOPO_LARGE_BF16_EXCHANGE=0 SEED=9000 scripts/locked_run_topo.sh topo_x_exact_ep8             # exact（fp32）exchange
# 或在同一次 lock 內串起來；tag 由 run tag 衍生（topo_<tag>_ep8、nccl_<tag>_ep8、topo_<tag>_bf16x_ep8），
# 已完成（status.txt 以 DONE 結尾）的結果目錄不會被覆寫（FORCE=1 可覆寫）；bench 需要同一 tag 先跑過 e2e_topo1，bench_nccl 只跑 c1
bash scripts/locked_job.sh y 14400 bash scripts/run_h2.sh y e2e_topo1 bench bench_nccl bench_bf16x
```

微基準：`bash scripts/locked_job.sh <tag> 3600 bash scripts/run_h1a.sh <tag> bench_public`、`bash bench/run_h1b.sh <tag> bench_ar bench_rs bench_ag`（結果 jsonl 在 `work/host/`，
用 `bench/summarize_h1a.py` / `summarize_h1b.py` 轉成表格）。

### 7.5 NCCL 工具

```bash
# PyNccl all-reduce 的 bitwise 測試（用法見檔頭；NCCL_* 環境變數由呼叫端決定；要用 GPU，所以走 lock）
bash scripts/locked_job.sh bw 1800 bash -c 'SRC=/raid/vllm-runtime/sm120-dev/src scripts/py_topo.sh nccl/bitwise.py --out /raid/vllm-runtime/work/host/bw.json'
python3 nccl/compare.py REF.json OTHER.json [...]          # 比較 e2e_bitwise.py / e2e_real.py 產生的 dump（不需要 GPU）
```

## 8. 限制與注意事項

* **只適用這台主機**：拓樸簽章要求 TP world size 8、同一個 node、全是 RTX PRO 6000 Blackwell（cc 12.0）、56 組 P2P 全通、PCI layout 為 2 sockets x 2 virtual switches x 2 GPUs，不吻合就用 NCCL。
  script 內的路徑寫死為 `/raid/vllm-runtime/...`。
* **只測過這個版本**：vLLM v0.32.0 + branch `sm120-muse-spark-open`（HEAD a5eda7714f）、torch 2.13 cu130、NCCL 2.29.7。plugin 是在 runtime patch `CudaCommunicator`，換 vLLM 版本要重新驗證。
* 只涵蓋 TP group（名稱以 `tp` 開頭、8 ranks）的 `all_reduce` / `all_gather` / `reduce_scatter`；EP group、`gather`、`reduce_scatterv`、`all_gatherv`、send / recv 仍走原本的路徑。
  不在 envelope 的呼叫（其他 dtype、non-contiguous、不是 128 B 倍數的大訊息、reduce_scatter dim != 0、all-gather 的中間維度）由 `supports()` 退回 NCCL。
* 本節的 H2 效能數字都是 **concurrency 1**；decode graph size 1..128 都有走 topo（正確性已驗證，128 requests 的 decode batch 重跑 128/128 相同）。高 concurrency（c4 .. c128，
  real weights）的 throughput 見 `docs/h2_vllm_plugin.md` 的 Microbatched prefill 一節；final defaults 的整體 throughput 列（`r2fin_*`）尚未量測。
* vLLM branch 的 SM120 sequence-parallel stream（`VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL`，預設 1，不需要 `deep_gemm_mega_moe`）在 >= 512 tokens 的 eager prefill 會走 `reduce_scatter` 與 `all_gather`，
  end-to-end 的 op log 有 `reduce_scatter(dim=0) shape=(8192, 8192) -> topo`；`test_plugin.py` 另外覆蓋 CUDA graph 與 RS + AG == AR。
  只有 `deep_gemm_mega_moe` 的原版 SP 路徑在這台機器上是關的。
* 同一個 `TopoComm` 的所有 collective 必須在同一條 stream 上依序發出，且所有 rank 的呼叫順序與 shape 相同；大訊息 kernel 會與 peer 互相 spin-wait（co-resident blocks），
  不可與會佔滿所有 SM 的其他 stream kernel 重疊。vLLM 預設在 main stream 發出 collectives；microbatched prefill（`VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP`，預設 ON）改在一條 comm stream 上，
  依各 rank 一致的順序發出，並以 hand-off event 滿足這條規則（`docs/h2_vllm_plugin.md` 的 Microbatched prefill 一節）。
* kernel 等不到 peer 超過 `MUSE_TOPO_TIMEOUT_S` 會 `__trap()`，該 worker 的 CUDA context 會失效（刻意如此，避免靜默 hang）。checkpoint / sleep（`suspend`、`resume` ...）沒有處理：IPC workspace 不會重建。
  `destroy()` 需要對稱（一個 rank 先 destroy，其他 rank 仍發 topo collective 會一直等到 timeout）。
* 數值：exchange payload 中等於 `0xFFFFFFFF` 的 32-bit word（NaN pattern）會被換成 `0xFFFFFFFE`，所以 NaN payload 可能與 NCCL 不同（仍是 NaN）。LL8 的 tag 約每 2^33 次 launch 重複一次（殘餘風險，沒有實測到）。
* bf16 exchange 與 NCCL graph file 都是 `deterministic_different`，現為 `env/host.env` 的預設（使用者 2026-10-24 決策）；final defaults 的 GSM8K 與 determinism A/B 已通過（第 5 節）。
  bf16 exchange 在兩個 partial 互相抵銷時最大誤差可到約 32640 ulp，且對 padded batch size 不是 batch invariant；需要 exact 結果或批次不變性的場合設 `MUSE_TOPO_LARGE_BF16_EXCHANGE=0`。
  medium all-reduce 的 bf16 exchange（r3c）讓 decode 範圍內出現新的 batch size 邊界：embedding all-reduce 在 eager T 3 與 4 之間（graph 的 padded T = 2 與 4 之間）、attn-out 在 T <= 8（fused kernel，exact；T = 3 與 4 都 exact）與 T >= 9（graph 的 padded T = 16 起，bf16 exchange）之間、MoE-out 在 eager T 7 與 8 之間（padded T = 4 與 8 之間）；只想關掉這一項設 `MUSE_TOPO_MEDIUM_BX=0`。fused 關閉時（`MUSE_TOPO_FUSED=0` 等）T = 4..8 的 attn-out 也走 bf16 exchange，所以 fused 開 / 關的 bitwise A/B 兩次都要設 `MUSE_TOPO_MEDIUM_BX=0`。
* r3b 的 GEMV + all-gather 融合（`MUSE_TOPO_GEMV`）與 router GEMV 都是 `deterministic_different`（對 cuBLAS；見第 5 節）：同一個 token row 在任何 batch size、rank、run 下 bits 相同，但與 cuBLAS 路徑的輸出
  在最後一個 bit 可能不同（router 的 sigmoid + top-8 只會在兩個 logits 差距小於那幾個 bit 時選到另一個 expert）。GEMV kernel 的環境變數（`MUSE_TOPO_GEMV*`）必須在所有 rank 相同，`MUSE_TOPO_GEMV_PUSH` 不可設成 3 或 7（會被拒絕）。
  padded T = 8（c5 .. c8）的 GEMV chain 在整個模型上沒有可量測的增益，所以預設 `MUSE_TOPO_GEMV_MAX_ROWS=4` 保留 cuBLAS；router GEMV 需要 fp32 router input（decode graph 中只有 expert prefetch 開啟時才有），沒有時維持 cuBLAS。
* 研究 probe（`a2a_probe.py`、`ce_cut_probe.py`、`host_relay_probe.py`）顯示：cross-socket 的 copy engine 或 host relay 做 cut transport 沒有實質增益（host relay 約 13 GB/s per GPU，island egress 隨後成為瓶頸，估計最多約 15 %，未實作）。
* 已知的後續項目：把 LL8 tag 的 64-bit modulo 換成 compare / select 可省約 0.05 us per all-gather（每個 decode step 約 6 us，未出貨，需再做一輪驗證）。

## 9. 文件索引與資料位置

* 實測拓樸與瓶頸：`docs/topology.md`；演算法與 API：`docs/h1a_collectives.md`、`docs/h1b_collectives.md`；latency 表：`docs/h1a_latency.md`、`docs/h1b_latency.md`；
  plugin、驗證紀錄、review fixes：`docs/h2_vllm_plugin.md`；pooler / unpooler GEMV 與 all-gather 融合：`docs/r3b_gemv_push.md`；medium all-reduce 的 bf16 exchange：`docs/r3c_medium_ar.md`。
  speculative launch 的 graph 上限 / sequence-parallel floor 組合（S1-9）：`docs/s19_spec_launch.md`。
* 伺服器上的資料（不在本資料夾）：benchmark 結果 `/raid/vllm-runtime/musetest/results/<tag>/`；測試 / benchmark 的 log 與 jsonl `/raid/vllm-runtime/work/host/`；
  JIT cache `/raid/vllm-runtime/sm120-dev/vllm_cache/muse_topo/`；拓樸量測 script 與原始結果 `/raid/vllm-runtime/musetest/topo/`；NCCL 量測 `/raid/vllm-runtime/musetest/nccl/`。
* 歷史：H1a（小訊息演算法）、H1b（large-message 演算法）、H2 / H2b（vLLM plugin）、review fixes（64-bit counter / flags、supports() 與 kernel 一致、fp16 的 bf16 exchange 溢位、對齊與 stream 規則，2026-10-24）、H3（本資料夾的定稿）。
