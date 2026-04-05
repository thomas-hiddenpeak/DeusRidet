# DeusRidet 开发日志

## 2026-04-02 — Phase 1：GPTQ-Int4 内核

### 背景

LLM（Qwen3.5-27B-GPTQ-Int4）仅 MLP 层量化为 INT4（GPTQ），所有注意力层
（GQA 和 DeltaNet SSM）保持 BF16。没有正确的 GPTQ 反量化+矩阵乘法内核，
量化层无法完成前向传播。

### 权重格式分析

- **qweight**: `I32 [K/8, N]` — 每个 uint32 打包 8 个 INT4 nibble，LSB-first 沿 K 维
- **qzeros**: `I32 [K/128, N/8]` — 常量 `0x88888888`（零点=8，对称量化）
- **scales**: `F16 [K/128, N]` — 每组每列 FP16 缩放因子
- **g_idx**: `I32 [K]` — 简单顺序(`g_idx[i] = i/128`），忽略
- 反量化公式：`W[k,n] = scales[k/128, n] * (qw_4bit - 8)`
- 仅 MLP gate/up/down_proj 被量化（全部 64 层）

### 实现

**新建文件：**
- `src/machina/gptq.h` — GPTQ-Int4 接口：`GptqWeight` 描述符，
  `gptq_gemv()`、`gptq_gemm()`、`gptq_linear()` 自动分发，基准测试 API
- `src/machina/gptq.cu` — CUDA 内核：
  - GEMV (解码, M=1)：N 方向分块 64 列/block，K 维 8 线程分割，FP32 累加，共享内存归约
  - GEMM (预填充, M>1)：分块 [32×64×128]，BK=128 对齐 group_size，
    共享内存反量化，每线程 [4×2] FP32 累加
  - 基准测试工具（CPU 参考使用 FP64 累加）

**修改文件：**
- `src/machina/allocator.h` — 新增 `INT32` DataType
- `src/machina/safetensors.cpp` — "I32" 映射到 `INT32`
- `src/main.cpp` — 新增 `test-gptq` 和 `bench-gptq` 命令

### Tegra 内存发现

`cudaHostRegister` 在 `PROT_READ` 只读 mmap 内存上失败（返回"invalid argument"）。
权重数据必须通过 `cudaMemcpy` 显式复制到设备内存。这实际上是更优选择：设备内存
避免了 Tegra iGPU 上频繁读取的一致性开销。

### 结果

**正确性**（真实模型权重，layer 0 gate_proj K=5120→N=17408）：
- GEMV 最大绝对误差：0.000038（对比 FP64 CPU 参考）
- GEMV 最大相对误差：1.86%（仅在绝对值极小的列上）
- GEMM 最大相对误差：1.68%
- **均通过 ✓**

**基准测试**（合成数据，SM87 Orin）：

| 用例 | 耗时 (µs) | 指标 | 值 |
|------|-----------|------|-----|
| gate_proj GEMV (5120→17408) | 882 | 带宽 | 52.1 GB/s |
| down_proj GEMV (17408→5120) | 902 | 带宽 | 51.0 GB/s |
| gate_proj GEMM M=32 | 3887 | TFLOPS | 1.47 |
| gate_proj GEMM M=128 | 15216 | TFLOPS | 1.50 |
| gate_proj GEMM M=512 | 60407 | TFLOPS | 1.51 |
| down_proj GEMM M=128 | 15492 | TFLOPS | 1.47 |

GEMV 达到 192 GB/s 理论带宽的 ~27%。GEMM 达到 ~1.5 TFLOPS（FP16 Tensor Core
理论 ~5.2 TFLOPS 的 ~29%）。这是首个正确实现的基线数据，优化留待后续阶段。

---

## 2026-04-02 — Phase 0 完成：基础设施

### 结果

Phase 0 实现并验证完毕，所有里程碑指标达成。

**构建组件：**
- CMake 骨架：SM87 目标架构，CUTLASS 子模块，`machina` + `communis` 静态库
- `SafetensorsLoader`：零拷贝 mmap，多分片支持（通过 `model.safetensors.index.json`）
- `Tokenizer`：GPT2 字节级 BPE，248070 词表，247504 合并规则，26 个特殊标记
- `Allocator`：`DeviceAllocator`（cudaMalloc）、`UnifiedAllocator`（cudaMallocManaged）、
  `MmapAllocator`（根据可用内存自适应 populate/willneed 策略）
- `Tensor`：支持拥有/非拥有两种模式的轻量级描述符
- `Config`：key=value 解析器（`machina.conf`），带类型化访问器
- `Log`：带时间戳的结构化日志

**AGX Orin 64GB 验证：**
- `test-tokenizer "Hello, world! 你好世界"` → 7 tokens，往返测试通过
- `test-tokenizer "人类一思考，上帝就发笑；AI一思考，人类就不笑了。"` → 往返测试通过
- `test-weights` → 1775 个张量，28.16 GB，11 个分片全部加载成功
- `version` → Orin SM8.7，16 SMs，61.4 GB，CUDA 12.6

**文件（19 个新文件，约 10800 行）：**
`CMakeLists.txt`, `LICENSE`, `.gitignore`, `.gitmodules`, `configs/machina.conf`,
`docs/ACKNOWLEDGMENTS.md`, `src/main.cpp`, `src/communis/{config.h,config.cpp,log.h}`,
`src/machina/{allocator.h,allocator.cpp,tensor.h,safetensors.h,safetensors.cpp,
tokenizer.h,tokenizer.cpp}`, `third_party/{cutlass,stb}`

**下一步：** Phase 1 — GPTQ-Int4 反量化内核（解码用 GEMV，预填充用 GEMM）

---

## 2026-04-02 — 开发计划

### 背景

环境确认：CUDA 12.6、CMake 4.1.0、GCC 11.4.0、Jetson AGX Orin 64 GB
（ARM v8l, SM87）。参考项目 qwen35-orin 提供了完整的推理引擎（safetensors
加载器、BPE 分词器、DeltaNet SSM、GQA 注意力、分页 KV 缓存 + SSD 溢出、
ViT 视觉、ASR、TTS、说话人识别、VAD、WS/HTTP 服务器），但缺少 GPTQ-Int4
内核和 GPU 采样。

LLM 配置（Qwen3.5-27B-GPTQ-Int4）：词表 248320，隐藏 5120，64 层（48
linear_attention + 16 full_attention，每 4 层一个），中间维度 17408，24 注意力
头 / 4 KV 头，head_dim 256，SSM: 16 key heads / 48 value heads / conv kernel 4 /
key/value_head_dim 128，MTP 1 层，rope_theta 10M，partial_rotary 0.25，mRoPE
交错 [11,11,10]。GPTQ bits=4, group_size=128, sym=true, desc_act=false。

### 策略

关键路径：LLM 必须先能正确生成 token，意识才有可能存在。
自下而上构建：内核 → 单次前向 → 持续循环 → 感知 → 表达。
参考代码只做适配，不做复制——每个内核都针对 Tegra 统一内存模型验证。

---

### Phase 0 — 基础设施（构建系统 + 权重加载 + 分词器）

目标：CMake 项目能编译，加载权重，分词文本，分配 GPU 内存。

| # | 任务 | 输入 | 输出 | 验证 |
|---|------|------|------|------|
| 0.1 | CMake 脚手架 | — | 编译空 `deusridet` 二进制，SM87，CUTLASS 子模块链接 | `cmake --build` 成功 |
| 0.2 | SafetensorsLoader | 11 个分片文件 | Tensor 映射：名称 → {dtype, shape, mmap ptr} | 打印张量名+形状，与 index.json 比对 |
| 0.3 | GPU 内存池（Allocator）| — | 设备内存上的 Arena 分配器 | 分配/释放往返，碎片测试 |
| 0.4 | BPE 分词器 | `vocab.json` + `merges.txt` | `encode(text) → token_ids`, `decode(ids) → text` | 往返测试100条字符串，特殊 token 正确 |
| 0.5 | 配置解析器 | `machina.conf` | 含模型路径、量化参数、内存预算的结构体 | 加载+打印无崩溃 |

里程碑：`./deusridet --test-tokenizer "Hello world"` 打印 token ID 和解码文本。

---

### Phase 1 — GPTQ-Int4 内核（最难的部分）

目标：Decode (GEMV) 和 Prefill (GEMM) 的 GPTQ 反量化 + 矩阵乘法正确。

| # | 任务 | 输入 | 输出 | 验证 |
|---|------|------|------|------|
| 1.1 | GPTQ 权重解包 | 打包 INT4 + scales + zeros | 按组反量化到 FP16 缓冲区 | 与 Python GPTQ 参考比特精确匹配 |
| 1.2 | GPTQ GEMV 内核（Decode）| x[1,K] × W_q[K,N] (INT4) | y[1,N] (FP16) | 100 随机输入最大绝对误差 < 1e-2 |
| 1.3 | GPTQ GEMM 内核（Prefill）| X[M,K] × W_q[K,N] (INT4), M=32..512 | Y[M,N] (FP16) | 同上容差，M=32,64,128,256,512 |
| 1.4 | GEMV/GEMM 基准测试 | 多种 M,K,N | tok/s, TFLOPS, 带宽利用率 | 内存带宽利用率必须超过 cuBLAS FP16（数据量减少4×）|
| 1.5 | 集成：替换线性层 | 模型前向框架 | 量化层走 GPTQ 路径，非量化层走 FP16 | 单 MLP 块输出与 Python 参考匹配 |

关键维度：K=5120→N=17408（gate/up），K=17408→N=5120（down），K=5120→N=5120（QKV）。

里程碑：`./deusridet --bench-gptq` 显示正确输出 + 吞吐数据。

---

### Phase 2 — 单层前向传播

目标：单个完整 Transformer 层（SSM 或 Full Attention）输出正确。

| # | 任务 | 输入 | 输出 | 验证 |
|---|------|------|------|------|
| 2.1 | RMSNorm 内核 | x[B,L,5120] | 归一化后的 x | 与 Python 最大误差 < 1e-5 |
| 2.2 | RoPE 内核（mRoPE）| Q/K 张量 + 位置 ID | 旋转后的 Q/K (partial_rotary=0.25, interleaved [11,11,10]) | 前 4 个位置与 Python transformers 对比 |
| 2.3 | SiLU + 逐元素门控 | gate[B,L,17408], up[B,L,17408] | gate_output[B,L,17408] | 精确匹配 |
| 2.4 | DeltaNet SSM 前向 | 输入 + 循环状态 + 卷积状态 | 输出 + 更新后的状态 | 与 HuggingFace DeltaNet 参考单层对比 |
| 2.5 | GQA Full Attention (Prefill) | Q[B,L,24,256], K[B,L,4,256], V | 注意力输出，KV 存入分页块 | 与 `F.scaled_dot_product_attention` 对比 |
| 2.6 | Paged Attention (Decode) | Q[B,1,24,256] + block table | 从分页 KV 解码输出 | 对缓存序列对比 |
| 2.7 | 完整层组装 | 一层的 token 嵌入 | 层输出 | 端到端输出与 Python 匹配（SSM 和 FA 两种类型）|

里程碑：`./deusridet --test-layer 0` 和 `--test-layer 3` 分别匹配 Python 参考输出。

---

### Phase 3 — 完整模型前向 + 采样 + 文本生成

目标：模型能从提示词生成连贯文本。

| # | 任务 | 输入 | 输出 | 验证 |
|---|------|------|------|------|
| 3.1 | 嵌入 + 64 层前向（Prefill）| Token IDs [1,L] | 隐藏状态 → logits [1,L,248320] | 最后 token logits top-5 与 Python 匹配 |
| 3.2 | KV 缓存管理器 | — | 块池分配/释放、逐序列块表 | 压力测试：分配→填充→淘汰→重分配 |
| 3.3 | Decode 循环 | Prefill 状态 + 新 token | 下一个 token logits | 自回归 10 token 与 Python 贪心匹配 |
| 3.4 | GPU 采样（top-k, top-p, temperature）| Logits [1,V] | 采样 token ID | 统计分布测试（1000 次采样, 卡方检验）|
| 3.5 | MTP 推测解码 | Draft 层 + 验证 | 接受的 token | 标准提示词接受率 > 50% |
| 3.6 | 聊天模板 + 停止 token | ChatML 格式输入 | 正确生成含 EOS 处理 | 多轮对话产生合理输出 |
| 3.7 | 端到端文本生成 | "什么是意识？" | 连贯的多句回复 | 人工评估：有意义 |

里程碑：`./deusridet --chat` 交互式文本生成正常工作。

---

### Phase 4 — 意识流（核心创新）

目标：持续 Prefill 循环 + 多轨 Decode。

| # | 任务 | 输入 | 输出 | 验证 |
|---|------|------|------|------|
| 4.1 | 意识帧抽象 | 合并的输入 | 帧结构：tokens + 元数据 | 单元测试：合并多个输入源 |
| 4.2 | 脉冲 Prefill 循环 | 输入队列中的帧 | 持续增长的 KV 状态 | 运行 1000 帧无 OOM 或崩溃 |
| 4.3 | SSM 状态连续性 | 连续的帧 | 循环状态跨帧传递 | 状态发散测试：连续 vs 重置 |
| 4.4 | 多轨 Decode 分支 | 共享 Prefill 前缀 | Thinking、Speech、Action、Daydream 输出 | 4 条分支独立产生 token |
| 4.5 | 时分复用器 | 分支优先级 | 在单一 GPU 流上调度执行 | 负载下优先级抢占正常工作 |
| 4.6 | Arbiter（分支合并）| 多条分支输出 | 最终决策：说/做/想/梦 | 决策反映最高优先级可操作输出 |
| 4.7 | 觉醒监测器 | 输入密度指标 | 觉醒等级 [0.0–1.0] | 有输入时上升，空闲时衰减 |
| 4.8 | P/D 时间预算调度器 | 觉醒等级 | Prefill/Decode GPU 时间分配 | 指标中可观察到平滑比例调整 |

里程碑：系统持续运行 10 分钟，处理注入的文本，分支为思考 + 语音输出，
可观察到觉醒状态转换。

---

### Phase 5 — 缓存管理器（持续淘汰）

目标：KV 缓存通过 SSD 溢出 + 基于重要性的淘汰无限增长。

| # | 任务 | 输入 | 输出 | 验证 |
|---|------|------|------|------|
| 5.1 | 块池 + 块追踪器 | — | GPU 块分配 + 位置追踪 | 14 GB 池，分配/释放/查询 |
| 5.2 | KV 交换器（GPU ↔ SSD）| 满的块 | 换出写入 SSD，换入恢复 | 比特精确往返 |
| 5.3 | SSM/Conv 状态快照 | 循环状态 | KV 块旁的 `.ssm`/`.conv` 文件 | 快照恢复后产生相同输出 |
| 5.4 | 重要性评分器 | 每块的注意力权重 | 累积重要性分数 | 低注意力块排名最低 |
| 5.5 | 持续淘汰循环 | 预算压力 | 最不重要的块换到 SSD | 系统支撑 256K+ token 无 OOM |
| 5.6 | 淘汰触发整合钩子 | 即将被淘汰的块 | 发送事件到 SomniumConsolidator（桩）| 钩子触发，元数据被捕获 |
| 5.7 | 前缀缓存（基于哈希）| 重复的前缀 | 缓存命中跳过重计算 | 重复提示词命中率 > 0 |

里程碑：意识流运行 1 小时，KV 缓存溢出到 SSD 并优雅恢复，无内存泄漏。

---

### Phase 6 — 多模态感知（ASR + 视觉 + 文本）

目标：系统能听、能看、能读——全部汇入意识流。

（从参考项目适配 ASR/Vision/Speaker ID/VAD，集成到意识流输入队列。）

### Phase 7 — 语音输出（TTS）

目标：系统以一致的人格语音说话。

（从参考项目适配 TTS，集成流式 WebSocket 音频输出。需先下载 TTS Tokenizer。）

### Phase 8 — 长期记忆（Memoria Longa）

目标：情景回忆（HNSW）+ 语义知识图（CSR），在做梦时整合。

（全新开发 HNSW 索引 + CSR 图 + 混合检索器 + Prefill 注入。）

### Phase 9 — 梦境与整合（Somnium）

目标：低觉醒状态触发记忆整合和创造性探索。

### Phase 10 — 人格与表达（Persona + Arbiter）

目标：实体拥有丰富的内心世界，自主塑造外在表达。

### Phase 11 — 工具使用（Instrumenta）

目标：实体能发现、调用和创造工具。

### Phase 12 — WebUI 与 Nexus（外部接口）

目标：完整的可观测面板，实时音视频，意识可视化。

---

### 阶段总览

| 阶段 | 名称 | 依赖 | 复杂度 |
|------|------|------|--------|
| 0 | 基础设施 | — | 低 |
| 1 | GPTQ-Int4 内核 | Phase 0 | **高**（全新内核开发）|
| 2 | 单层前向 | Phase 0, 1 | 中 |
| 3 | 完整模型 + 生成 | Phase 2 | 中 |
| 4 | 意识流 | Phase 3 | **高**（核心创新）|
| 5 | 缓存管理器 | Phase 3 | 中高 |
| 6 | 多模态感知 | Phase 4 | 中（适配参考）|
| 7 | 语音输出 | Phase 4 | 中（适配参考）|
| 8 | 长期记忆 | Phase 4, 5 | **高**（从零开发 HNSW + 图）|
| 9 | 梦境与整合 | Phase 4, 5, 8 | 高 |
| 10 | 人格与表达 | Phase 4 | 中 |
| 11 | 工具使用 | Phase 4 | 中 |
| 12 | WebUI | Phase 4 | 中 |

Phase 4 稳定后，Phase 6–12 部分可并行开发。

### 决策

从 Phase 0 开始。首次提交目标：CMake 脚手架 + SafetensorsLoader +
Tokenizer + Allocator。

## 2026-04-04 — Phase 2.1: 解码速度优化

### 背景

前向推理已能正确输出文本，报告速度 ~480 ms/token（实际每次前向约 295 ms，报
告数值包含了 prefill 阶段）。目标：识别并消除瓶颈。

### 优化 1: 自定义 FP16 GEMV 内核

`linear_forward()` 使用 `cublasGemmEx(CUBLAS_OP_T)` 做 FP16 矩阵向量乘。对
M=1 解码，转置读取导致非合并访存。

替换为自定义 `fp16_gemv_kernel`：每 warp 处理一个输出行，float4 向量化加载
（每线程每步 8 个 half），沿 K 维合并读取，warp shuffle 归约。

结果：投影接近带宽极限（q_proj 0.72ms, o_proj 0.37ms, lm_head 14.29ms）。

### 优化 2: 融合 GQA 解码注意力

GQA 注意力使用 4 次 `cublasGemmEx` 做 QK^T + 4 次做 V@scores，对极小矩阵
（如 pos=1 时 [2,6]）开销灾难性：**21.41 ms**。

替换为单一 `gqa_decode_attention_kernel`：24 个 block（每 query head 一个），
256 线程维度并行，融合 QK^T + softmax + V@scores。

结果：FullAttn 每层 24.54ms → **1.46ms**，融合核 **0.01ms**。

### 优化 3: GPU argmax 采样

原实现：复制 487KB logits 到 CPU + 线性扫描。替换为 GPU `argmax_kernel`（单
block 1024 线程树归约），仅回传 4 字节结果。1.05ms → **0.21ms**。

### 计时修正

修复 test-forward 计时，分别报告 prefill 和 decode 阶段。

### 结果

| 组件 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| FullAttn attn（每层，隔离计时） | 24.54 ms | 1.46 ms | 94% |
| Greedy sample | 1.05 ms | 0.21 ms | 80% |
| **实际解码** | **~295 ms** | **~283 ms** | **4.1%** |

MLP（GPTQ GEMV）占 172ms（61%）为当前主要瓶颈。

## 2026-04-04 — 阶段 2.2：CUDA Graph + 核优化

### 背景

每次 forward 约 1,476 次核启动，在 ARM Tegra 上每次启动开销约 5-10 µs，总计
~5-15 ms。此外注意力核在共享内存中实体化 O(seq_len) 的分数数组，限制了最大
seq_len 并浪费内存。仅支持贪心采样，无法生成多样性文本。

### 方法

#### CUDA Graph 解码图捕获

将整个解码前向传播预编译为 CUDA Graph（1,351 节点）。首次前向调用时捕获，后续
所有 token 通过 `cudaGraphLaunch` 回放，消除逐核启动的主机端驱动开销。

**关键设计：**
- **非默认流**：`cudaStreamBeginCapture` 无法在遗留流 0 上使用，新增
  `compute_stream`。
- **设备间接参数**：依赖 `pos` 的核（RoPE、KV 写入、注意力）从设备指针
  `int* d_pos` 读取，使图拓扑完全静态——仅 `*d_pos` 的值变化。
- **固定主机暂存**：`cudaHostAlloc` 分配 `h_token_pinned` 和 `h_pos_pinned`，
  图回放前写入新值，图内 H2D memcpy 节点自动读取最新值。
- **图内 argmax**：`argmax_kernel` 被捕获入图，仅最终 D2H + 同步在图外执行。

#### Flash-Decoding 风格在线 softmax 注意力

参考 Flash Attention v2 / FlashInfer FlashDecoding 思想重写注意力核：

- **O(1) 共享内存**：仅 36 字节（8 warp sums + 1 broadcast float），此前为
  `O(8 + seq_len)` float。解除 SM87 48KB 共享内存对 max_kv_len 的限制。
- **消除串行 softmax**：旧核在 thread 0 串行计算 softmax，新核使用运行中
  `(m, l, o)` 累加器——全并行。
- **单趟 K+V 加载**：每个位置的 K 和 V 在同一循环迭代加载（缓存友好），
  此前为两趟（QK^T + V@scores）。
- **数值稳定**：使用 `exp(s - m_running)` 重缩放，与 Flash Attention v2 公式
  等价。

#### 核融合

- **`residual_rms_norm`**：融合 `elementwise_add + rms_norm`，省 64 次核启动，
  消除每层一次 5120 元素的冗余 load/store。
- **`silu_mul`**：融合 MLP 激活中的 `silu_inplace + elementwise_mul`，省 64 次
  核启动，消除每层一次 17408 元素的冗余 load/store。

#### Top-k / Top-p GPU 采样

参考 FlashInfer 思想实现 GPU 端采样：

1. **温度缩放 + 在线 softmax**（1 block, 1024 线程）：并行 max reduction、
   exp/sum、归一化至 FP32 概率。
2. **二分搜索阈值**（32 轮迭代）：找到概率阈值 T 使
   `count(prob >= T) <= top_k` 且 `sum(prob >= T) >= top_p`。每轮扫描词表
   248K 元素（~243/线程），全并行。
3. **多项式采样**：thread 0 按累积和对比 PRNG 随机数（LCG）顺序扫描候选词。

### 结果

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| 解码（贪心, CUDA Graph） | 283 ms/token | 279 ms/token | -1.4% |
| 解码（采样, 无图） | N/A | 375 ms/token | （新能力） |
| 图节点数 | N/A | 1,351 | 仅捕获一次 |
| 注意力 smem/block | `(8+seq_len)*4` B | 36 B | O(1) vs O(N) |
| 节省核启动数 | 0 | ~128（融合） | 减少 8.7% |
| 输出质量 | 仅贪心 | Top-k/top-p 采样 | （新能力） |

CUDA Graph 带来约 4ms 改善，与 ~3µs/launch × 1351 节点的估算吻合。前向传播
仍为显存带宽瓶颈（GPTQ GEMV 占 61%），核启动开销优化的边际收益已递减。

### 正确性验证

- **贪心路径**：优化前后 token ID 完全一致：
  `90700 8340 25 271 16 13 220 2972 2014 53983 279 5952 64700 198 262 348`
- **采样路径**：temp=0.7, top_k=50, top_p=0.9 下生成连贯思维结构输出
- **在线 softmax**：数学等价于实体化 softmax（通过贪心输出匹配验证）

---

## 2026-04-05 — Phase 2.3：带宽优化 — SMEM、向量化、核融合

### 目标

缩小与 qwen35-orin 参考项目的带宽利用率差距。强制原则：不以"硬件限制"为借口
跳过优化，每个 CUDA kernel 必须追求最大显存带宽利用率。

### 分析（qwen35-orin 参考项目研究）

| 技术 | qwen35-orin | DeusRidet 优化前 |
|------|-------------|-----------------|
| GEMV x 向量加载至 SMEM | ✓ | ✗（重复全局内存读取） |
| Dual GEMV gate+up 共享 | ✓ | ✗（分开启动） |
| GEMV + 残差融合 | ✓ | ✗（独立 elementwise_add） |
| float4 向量化加载 | ✓ | ✗（标量 __half 逐个加载） |
| 寄存器缓存 RMSNorm | ✓ | ✗（两遍全局读取） |
| 寄存器 conv1d | ✓ | ✗（基于 SMEM） |
| GEMV scale 提升 | ✓ | ✗（每行读取 scale） |
| 循环展开（≥4路） | ✓ | ✗ |

### 实施的优化

**1. GPTQ GEMV 核心重写** (`gptq.cu`)：
- x 向量通过 256 线程协作 float4 加载至共享内存（K=5120 → 640 个 float4）
- Scale 提升：每 16 个 packed row 读取一次 `__half2float`（原为每行一次）
- 4 路循环展开 + 4 个独立 FP32 累加器提升 ILP
- Block 256（原为 512）提升占用率
- `template<bool ADD_RES>` 实现可选融合残差加法 — 消除 MLP 后独立 `elementwise_add`
- 新增 `gptq_gemv_add()`：GEMV + 残差加法单核完成
- 新增 `gptq_dual_gemv()`：gate_proj + up_proj 共享 SMEM 中的 x —
  2 次权重读取 + 1 次 x SMEM 加载，取代 2 次权重 + 2 次全局 x 读取

**2. 逐元素操作向量化** (`layer.cu`)：
- `silu_mul`：float4 加载/存储 + `__expf` 快速指数
- `sigmoid_gate`：float4 加载/存储 + `__expf`
- `embedding`：float4 向量化拷贝（原为标量 __half 循环）
- 门控 RMSNorm：`__expf` 替代 `expf`

**3. 寄存器缓存 RMSNorm** (`layer.cu`)：
- 每线程 `float cache[40]` — x 单次全局读取，用于方差计算和归一化输出两遍
- `residual_rms_norm_kernel` 同样处理
- 完全消除第二遍全局内存读取

**4. 寄存器优化 conv1d** (`layer.cu`)：
- 预加载 `state[3]` 和 `weight[4]` 至寄存器
- 全程在寄存器中计算和更新 — 零 SMEM 需求

**5. DeltaNet 单遍递推** (`forward.cu`)：
- 融合两遍（key-value 外积 + state-query 矩阵乘）为单遍
- SMEM 从 `k+v+v` 降至 `k+v`
- 消除不必要的 `__syncthreads`

**6. 前向传播组装** (`forward.cu`)：
- `mlp_forward` 使用 `gptq_dual_gemv` 处理 gate+up
- `mlp_forward` 接受 residual 指针，使用 `gptq_gemv_add` 融合 down_proj + 残差
- `forward_body` 消除 MLP 后独立 `elementwise_add`
- CUDA Graph 节点：1351 → 1223（减少 128 次核启动）

### Bug：GPTQ 反量化中的无符号整数下溢

初始测试显示所有输出 token ID = 0（时序正确但输出错误）。`test-gptq` 显示 GEMV
产生 inf/-inf 而 GEMM 通过。

**根因**：新 GEMV 核直接操作 `uint32_t`：
```cpp
(float)(((p0 >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT)
```
`((uint32_t) & 0xF)` 产生 `uint32_t`。减去 `int 8` 触发无符号提升：当 4-bit
值为 0–7 时，结果回绕至 ~42.9 亿而非 -8 到 -1。原代码使用 `int extract_int4()`
返回类型避免了此问题。

**修复**：在单 GEMV 和双 GEMV 核的所有 8 处减法前添加 `(int)` 强制转换：
```cpp
(float)((int)((p0 >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT)
```

**教训**：混用 uint32_t 位操作与有符号算术时，必须显式指定符号性。C++ 无符号
提升规则是沉默的杀手。

### 结果

| 指标 | Phase 2.2 | Phase 2.3 | 提升 |
|------|-----------|-----------|------|
| 解码延迟（贪心） | 279 ms/tok | 183 ms/tok | **-34.4%** |
| CUDA Graph 节点 | 1,351 | 1,223 | -128 次启动 |
| GPTQ GEMV 带宽 | ~50 GB/s | 117 GB/s | **+134%** |
| **MLP（64层）** | 119.3 ms | 60.4 ms | **-49.4%** |
| Norms | 24.2 ms | 16.6 ms | **-31.4%** |
| DeltaNet attn（48层） | 75.5 ms | 72.2 ms | -4.4% |
| FullAttn（16层） | 24.8 ms | 21.9 ms | -11.7% |
| LM Head | 14.3 ms | 14.3 ms | 不变 |

### 正确性验证

- **GPTQ GEMV**：最大相对误差 0.000472（对比 CPU 参考，256 列）
- **贪心路径**：Token ID 完全匹配：
  `90700 8340 25 271 16 13 220 2972 2014 53983 279 5952 64700 198 262 348`
- **采样路径**：temp=0.7, top_k=50, top_p=0.9 下生成连贯输出
- **GPTQ GEMM**：不变，最大相对误差 0.017

### 剩余瓶颈

| 组件 | 耗时 | 占总比 | 备注 |
|------|------|--------|------|
| DeltaNet attn | 72.2 ms | 39% | 递推扫描 — 本质串行 |
| MLP | 60.4 ms | 33% | 已减半；进一步需 GEMV BW > 117 GB/s |
| FullAttn | 21.9 ms | 12% | FP16 GEMV (q_proj) + flash-decode |
| Norms | 16.6 ms | 9% | 64 pre-norm + 64 post-norm = 128 次启动 |
| LM Head | 14.3 ms | 8% | FP16 cuBLAS GEMV — 厂商优化 |

---

## 2026-04-04 — Phase 2.4：FP16 投影层 INT8 量化

### 背景

Phase 2.3 基线：183 ms/token。目标：< 140 ms/token。

nsys 分析揭示解码时间分布：
- **fp16_gemv_kernel：98.3 ms (53%)** — 主要瓶颈
- gptq_dual_gemv_kernel：44.5 ms (24%)
- gptq_gemv_kernel：30.7 ms (17%)
- deltanet_recurrent：5.6 ms (3%)
- rms_norm：1.3 ms (< 1%)

FP16 GEMV 已达到约 90% 带宽利用率（173.9 GB/s / 192 GB/s）。
唯一进一步降低时间的方法是减少数据量。
INT8 量化将这些投影层的权重读取量减半。

### K_THREADS=8 优化 (183 → 176 ms)

GPTQ GEMV 从 K_THREADS=4（256 线程/块）提升到 K_THREADS=8（512 线程/块），
warp 数翻倍（8 → 16），显著提升 `down_proj` 占用率（1 块/SM → 16 warps/SM = 33%）。

基准测试：
- gate_proj：116.7 → 118.5 GB/s
- down_proj：104.6 → 117.2 GB/s (+12%)
- 解码：183.0 → 175.9 ms

### 失败尝试（已回退）

1. **Tiled SMEM x**：分块加载 x 到共享内存，syncthreads 开销杀死了流水线。
2. **L2-based x（无 SMEM）**：Orin 上 SMEM 读取（~30 周期）远优于 L2（~100-200 周期）。
3. **4-way dual GEMV 展开**：寄存器压力过大，+5.6 ms 回退。
4. **融合 silu+down 内核**：需从 L2 读取 gate+up，+27 ms 回退。
5. **cudaFuncSetAttribute**：无效果。

### INT8 量化实现 (176 → 135.5 ms)

所有 DeltaNet 和 FullAttention 投影层（磁盘上为 BF16）在模型加载时量化为 INT8：

**量化方案**：逐通道对称 INT8
- `scale[n] = max(|W[n,:]|) / 127`
- `W_int8[n,k] = round(W[n,k] / scale[n])`，截断到 [-127, 127]
- 推理时：`result = sum(W_int8 * x_fp16) * scale`

**新 INT8 GEMV 内核**（`int8_gemv_kernel`）：
- 4 warps/块（128 线程），1 warp/行
- 向量化加载：float4 读取 16 个 INT8 + 2 个 float4 读取 16 个 FP16 x
- FP32 累加，通道级 scale 在累加完成后应用
- Warp shuffle 归约

**受影响的权重**（每层 9 种投影）：
- DeltaNet（48 层）：in_proj_qkv、in_proj_z、in_proj_a、in_proj_b、out_proj
- FullAttention（16 层）：q_proj、k_proj、v_proj、o_proj

**未量化**（保持 FP16）：lm_head、embed_tokens、MLP（已经是 GPTQ-INT4）

### 结果

- **解码延迟：135.5 ms/token**（原 176 ms，**-23%**）
- **预填充：136.9 ms/token**
- **精度：完美** — Token ID 完全匹配：
  `90700 8340 25 271 16 13 220 2972 2014 53983 279 5952 64700 198 262 348`
- **模型加载时间：31.7s**（含 GPU 上 INT8 量化）

### 性能历程

| 阶段 | 解码 (ms/token) | 关键变更 |
|------|----------------|----------|
| 2.1（cuBLAS 基线）| 302 | 起点 |
| 2.1（自定义 GEMV）| 216 | 自定义 FP16/GPTQ GEMV 内核 |
| 2.2（dual GEMV）| 183 | 融合 gate+up GPTQ GEMV |
| 2.3（K_THREADS=8）| 176 | GPTQ GEMV 线程数翻倍 |
| **2.4（INT8 量化）**| **135.5** | **FP16 投影层 INT8 量化** |

### 剩余瓶颈估计

理论最小值：~128 ms（24.5 GB 权重 / 192 GB/s）。当前 135.5 ms 仅比理论值
高 6% — 带宽利用率优秀。

---

## 2026-04-04 — Phase 2.5：GPTQ Scale 延迟乘法

### 背景

INT8 量化后 135.5 ms/token，nsys 分析（--cuda-graph-trace=node）新内核分布：

| 内核 | 时间 (ms) | 带宽 (GB/s) | 利用率 |
|------|-----------|-------------|--------|
| gptq_dual_gemv (gate+up) | 43.6 | 135 | 70% |
| int8_gemv (attn) | 43.3 | 169 | 88% |
| gptq_gemv (down) | 25.0 | 118 | 61% |
| fp16_gemv (lm_head) | 14.3 | 166 | 86% |
| deltanet_recurrent | 5.6 | — | — |
| norms + 其他 | 4.7 | — | — |

**关键发现**：GPTQ INT4 内核每字节需 ~12 次运算，接近 Orin 的计算/带宽
平衡点（13.1 FLOPS/byte）。INT8（3 ops/byte）和 FP16（2 ops/byte）
是纯内存受限的，而 GPTQ 是边缘计算受限的。

### Scale 延迟乘法优化

将 scale 乘法从内层循环每元素移到 group 边界（每 128 元素一次），
减少约 15% 的 FP32 运算。

### 失败尝试

1. **L2 x 替代 SMEM x**：down_proj 跳过 SMEM → 回退（130 → 103 GB/s）
2. **SMEM carveout**：增大 SMEM 分区 → 在 Orin 上无效
3. **4-way dual 展开 + scale 延迟**：仍回退（131.5 → 142.2 ms）

### 结果

| 投影 | 优化前 (GB/s) | 优化后 (GB/s) | 提升 |
|------|---------------|---------------|------|
| gate_proj | 118.5 | 131.4 | +10.9% |
| down_proj | 117.2 | 130.1 | +11.0% |

- **解码延迟：131.5 ms/token**（原 135.5 ms，**-3.0%**）
- **精度：完美**

### 性能历程

| 阶段 | 解码 (ms/token) | 关键变更 |
|------|----------------|----------|
| 2.1（cuBLAS 基线）| 302 | 起点 |
| 2.1（自定义 GEMV）| 216 | 自定义 FP16/GPTQ GEMV 内核 |
| 2.2（dual GEMV）| 183 | 融合 gate+up GPTQ GEMV |
| 2.3（K_THREADS=8）| 176 | GPTQ GEMV 线程数翻倍 |
| 2.4（INT8 量化）| 135.5 | FP16 投影层 INT8 量化 |
| **2.5（scale 延迟）**| **131.5** | **GPTQ scale 延迟乘法** |

### 剩余差距分析

修正后的理论最小值（含 INT8）：
- 每步权重读取：~17.9 GB → 93.2 ms 纯 DRAM 读取
- 非 GEMV 计算：~10.3 ms
- **绝对最小：~103.5 ms**

当前差距 28 ms 主要来自 GPTQ 仅 70% 带宽（21.6 ms 差距）。进一步优化
需结构性改变（如 qweight 转置 [K/8,N] → [N,K/8] 实现顺序 DRAM 访问），
预估可达 ~122 ms。

---

## 2026-04-02 — Phase 2.6：内核融合 + INT8 lm_head

### 背景

深度 nsys 分析揭示各内核带宽利用率差异巨大：
- 大 INT8 GEMV（qkv, z, out）：192 GB/s 的 88-91%，接近峰值
- GPTQ dual/single：68-70%，受 INT4 反量化计算限制
- lm_head（FP16）：93% 带宽，但 FP16 意味着每步 2.54 GB
- 小投影（a/b N=48）：14% 带宽，启动开销主导
- k/v 投影（N=1024）：71% 带宽

关键洞察：**lm_head 以 14.3 ms 占总时间 11%**，且未量化。INT8 量化可将
权重数据从 2.54 GB 减半至 1.27 GB。

### 已实施优化

#### 1. INT8 lm_head 量化（节省 7.1 ms）

在模型加载时将 lm_head 从 FP16 量化为 INT8，使用与注意力投影相同的
per-channel 对称量化。解码时使用 int8_gemv 替代 fp16_gemv。

内存开销：+1.28 GB。精度影响：零 — argmax(logits) 对 INT8 量化噪声鲁棒，
生成的 token ID 完全一致。

#### 2. 双 INT8 GEMV 内核（节省 0.6 ms）

新 `int8_dual_gemv_kernel` 在单次启动中计算两个矩阵乘法，通过 SMEM
共享 x 向量：
- DeltaNet in_proj_a + in_proj_b：18.6 → 12 μs/层
- FullAttn k_proj + v_proj：77.4 → 60.5 μs/层

#### 3. 融合 conv1d_step + SiLU（节省 0.08 ms）

新 `conv1d_step_silu_kernel` 在寄存器中完成 conv1d 和 SiLU，消除中间
全局写入。

#### 4. 融合 l2norm + scale（节省 0.06 ms）

新 `l2norm_scaled_kernel` 将 1/√128 缩放合并到 l2norm 归一化写入中。

### 失败尝试：单 int8_gemv 的 SMEM x

为 int8_gemv_kernel 添加协作 SMEM x 加载。结果：**大投影回退 +0.46 ms**。

原因：`__syncthreads()` 屏障阻止 warp 重叠计算和 x 加载。在 Orin 上 x
向量（10 KB）始终驻留 L2（4 MB），直接 L2 读取无需 SMEM 开销。

关键结论：**SMEM x 仅在 warp 共享 x 且减少启动次数来摊销屏障成本时有效**
（如 dual GEMV），对单 GEMV 无效。已回退。

### 结果

| 优化 | 节省时间 (ms) |
|------|--------------|
| INT8 lm_head | -7.15 |
| 双 GEMV (a+b, k+v) | -0.59 |
| 融合 conv1d+silu | -0.08 |
| 融合 l2norm+scale | -0.06 |
| **合计** | **-7.88 → 回退后 -7.42** |

**解码延迟：123.8 ms/token**（原 131.5 ms，**-5.9%**）
**图节点：1063**（内核融合后从 ~1400 减少）
**精度：完美**

### 性能历程

| 阶段 | 解码 (ms/token) | 关键变更 |
|------|----------------|----------|
| 2.1（cuBLAS 基线）| 302 | 起点 |
| 2.1（自定义 GEMV）| 216 | 自定义 FP16/GPTQ GEMV 内核 |
| 2.2（dual GEMV）| 183 | 融合 gate+up GPTQ GEMV |
| 2.3（K_THREADS=8）| 176 | GPTQ GEMV 线程数翻倍 |
| 2.4（INT8 量化）| 135.5 | FP16 投影层 INT8 量化 |
| 2.5（scale 延迟）| 131.5 | GPTQ scale 延迟乘法 |
| **2.6（INT8 lm_head + 融合）**| **123.8** | **INT8 lm_head + dual GEMV + conv+silu + l2norm+scale** |

### 修正差距分析

每步权重数据：~16.7 GB（GPTQ 8.2 + INT8 7.3 + lm_head 1.27）
192 GB/s 下：86.9 ms 纯 DRAM 读取
非 GEMV 开销：~10.3 ms（recurrent 5.6 + norm 1.9 + helper 2.8）
GPTQ 计算开销：~18 ms（68% → 100% 带宽差距）

**修正理论最小值：~115 ms**

当前差距：123.8 - 115 = 8.8 ms，分布在 GPTQ 结构性 BW 限制（18 ms）、
INT8 近峰值（~5 ms）、CUDA 图内核调度开销（~4 ms）。

---

## 2026-04-03 — 阶段 3.0：批量 Prefill

### 背景

解码优化已达 123.8 ms/token（接近 ~115 ms GEMV 理论极限），重点转向 **prefill** ——
意识流的核心持续运行部分。此前 prefill 通过 M 次串行 `forward_one_token` 调用实现
（M=1 CUDA Graph 回放），每次 ~132.6 ms。11 token 提示词：总计 1459 ms。

Prefill 与 decode 本质不同：M>1 token 允许使用 GEMM（矩阵-矩阵乘法）替代 GEMV，
将所有 token 批量通过每个投影层。核心挑战：GPTQ GEMM 内核（已存在）和 INT8 路径
均未端到端支持 M>1。

### 实现内容

**forward.cu 新函数（约 450 行）：**
- `forward_prefill()` — 主编排器：embed → 64 层 → final_norm → lm_head → argmax
- `mlp_forward_prefill()` — 批量 MLP，使用 `gptq_linear` 自动分发
- `deltanet_prefill()` — 批量 INT8 投影 + 逐 token 串行 conv1d/recurrent
- `full_attention_prefill()` — 批量 QKV 投影 + 因果自注意力
- `prefill_attention_kernel` — 在线 softmax 因果注意力（grid = num_heads × M）
- `rope_batch_kernel` — M token 批量 RoPE（连续位置）
- `kv_cache_write_batch_kernel` — 并行写入 M 位置到 KV cache
- `split_q_gate_batch_kernel` — 批量 Q/Gate 解交织

**layer.cu 新内核：**
- `int8_batch_gemv_kernel` — 每行权重只加载一次，计算 M 个点积（X 驻留 L2）。
  4 warp/block（128 线程），warp 对应一个输出行，向量化 float4 加载。

**gptq.cu 新内核：**
- `gptq_batch_gemv_kernel` — GPTQ-Int4 同架构：一个 warp 对应一个输出列，
  M 个累加器。仅对小 N（≤1024）有效。

**分发逻辑（gptq.h）：**
- M=1 → GEMV（decode 路径，不变）
- M>1，N≤1024 → batch GEMV（避免 BM tile 填充浪费）
- M>1，N>1024 → GEMM（避免 L2 放大效应）

### 关键技术决策

**INT8 batch GEMV vs tiled GEMM：**
INT8 tiled GEMM（BM=32）对 M=11 仅 7.7% DRAM 带宽利用率（tile 利用率 34.4%）。
batch GEMV 方法——每行权重加载一次，从 L2 缓存的 X 计算 M 个点积——快 3.6 倍
（422 → 116 ms）。关键洞察：小 M 时 Orin L2（4 MB）轻松容纳整个 X 矩阵
（11 × 5120 × 2 = 112 KB），权重带宽成为唯一瓶颈。

**GPTQ batch GEMV 的 L2 放大失败：**
将同样方法应用于 GPTQ MLP（N=17408）反而慢 2 倍（1293 vs 648 ms）。根因：
大 N 时 L2 总读取 = N × M × K × 2 字节（每个输出列重新读取完整 X）。
N=17408 时每次调用约 1.87 GB，超出 L2 带宽能力。tiled GEMM 尽管 M 利用率低，
通过 SMEM tile 实现了更好的数据复用。

处理：GPTQ batch GEMV 限制为 N≤1024（有效未使用——无 GPTQ 层 N 如此小）。

**DeltaNet 串行瓶颈：**
DeltaNet 层具有固有串行组件：conv1d 状态和 recurrent 状态更新
（S ← diag(g)·S + β·k^T·v）必须逐 token 处理。48 层 × 11 token 约 44 ms，
目前可接受但是个根本性串行化点。

### 性能结果

| 组件 | 优化前（串行）| 优化后（批量）| 加速比 |
|------|-------------|-------------|--------|
| DN 投影（48 层）| ~422 ms | 116 ms | 3.6× |
| DN 串行（48 层）| ~45 ms | 45 ms | 1.0×（固有串行）|
| DN 后处理（48 层）| ~128 ms | 46 ms | 2.8× |
| Full Attention（16 层）| ~156 ms | 49 ms | 3.2× |
| MLP（64 层）| ~648 ms | 648 ms | 1.0×（同 GPTQ GEMM）|
| Norms（64 层）| ~114 ms | ~114 ms | 1.0× |
| **总 prefill（11 token）** | **1459 ms** | **1012 ms** | **1.44×** |
| **每 token** | **132.6 ms** | **92.0 ms** | **1.44×** |

**瓶颈分析：**
MLP GPTQ GEMM 以 648 ms 占总 prefill 的 64%。GPTQ GEMM 在 M=11 时仅 ~7.7%
DRAM 带宽利用率（BM=32 tile 利用率 34.4%，加上 SMEM 反量化开销）。这是明确的
下一步优化目标。

**正确性：** 输出 token ID 在串行解码、tiled GEMM prefill、batch GEMV prefill
间完全一致。

### 性能历程（更新）

| 阶段 | 解码 (ms/token) | Prefill (ms/token) | 关键变更 |
|------|----------------|-------------------|----------|
| 2.1（cuBLAS 基线）| 302 | — | 起点 |
| 2.1（自定义 GEMV）| 216 | — | 自定义 FP16/GPTQ GEMV |
| 2.2（dual GEMV）| 183 | — | 融合 gate+up GPTQ GEMV |
| 2.3（K_THREADS=8）| 176 | — | GPTQ GEMV 线程数翻倍 |
| 2.4（INT8 量化）| 135.5 | — | FP16 投影层 INT8 量化 |
| 2.5（scale 延迟）| 131.5 | — | GPTQ scale 延迟乘法 |
| 2.6（融合）| 123.8 | 132.6 | INT8 lm_head + 内核融合 |
| **3.0（批量 prefill）** | **123.8** | **92.0** | **INT8 batch GEMV + 批量注意力** |

### 下一步

- **MLP GPTQ GEMM 优化**（648 ms = prefill 64%）：BM=16 tile、tensor core WMMA、
  或反量化至 FP16 + cuBLAS
- **扩展测试**：M=32, 64, 128 下的 prefill 吞吐量，适配意识流批量大小

---

## 2026-04-04 — 阶段 3.1：Tensor Core WMMA GEMM

### 背景

阶段 3.0 达到 92 ms/token prefill，但 MLP GPTQ GEMM 以 648 ms 占 64%。
分析发现根因：所有矩阵乘法使用 CUDA core FMA（~5.3 TFLOPS），而 SM87 tensor
core 提供 ~107 TFLOPS FP16。GPTQ GEMM 在 M=11 时是**计算受限**的
（arithmetic intensity 46 FLOP/byte > 平衡点 27.6）。Tensor core 是唯一
突破路径。

### 实现

**GPTQ WMMA 内核**（gptq.cu `gptq_wmma_gemm_kernel`）：
- Tile: BM=16, BN=64, BK=128（对齐 group_size）
- 128 线程 = 4 warps，每个 warp 通过 WMMA m16n16k16 计算 16×16 输出
- SMEM: X [16,128] (4 KB) + W^T [128,64] (16 KB) = 20 KB
- INT4 反量化：协作加载 qweight，解包 8 个 nibble，乘 per-group scale，写 FP16 到 SMEM

**INT8 WMMA 内核**（layer.cu `int8_wmma_gemm_kernel`）：
- 关键设计：W 是 [N, K] 行主序 INT8。不做转置，以 [BN, BK_PAD] 布局存入 SMEM，
  使用 `wmma::col_major` matrix_b 隐式转置
- BK_PAD = 136（BK+8）避免 SMEM bank 冲突
- 协作读取：每个 warp 每次迭代读一完整 W 行（32 lane × 4 字节 = 128 字节 = BK）

### 失败尝试：转置 INT8 SMEM 布局

首版 INT8 WMMA 将 W^T 存为 [BK, BN] 并使用 `row_major` matrix_b。写入
`smem_w[k * BN + n]` 时所有线程写入同一 n_local → **32 路 SMEM bank 冲突**。
结果：902 ms prefill，比 batch GEMV 回退 47%。

修复：改为 [BN, BK_PAD] 布局（W 原始行主序）+ `col_major` WMMA matrix_b。
SMEM 写入沿 K 快维度 → bank 冲突从 32 路降至 2 路。

### 性能结果

| 组件 | 阶段 3.0 | 阶段 3.1 | 加速比 |
|------|---------|---------|--------|
| MLP GPTQ（64 层）| 648 ms | ~248 ms | **2.6×** |
| INT8 投影 | 198 ms | ~99 ms | **2.0×** |
| DN 串行 | 45 ms | 45 ms | 1.0× |
| Norms + 其他 | ~121 ms | ~121 ms | 1.0× |
| **总 prefill（11 token）** | **1012 ms** | **516 ms** | **1.96×** |
| **每 token** | **92.0 ms** | **46.9 ms** | **1.96×** |

### 性能历程（更新）

| 阶段 | 解码 (ms/token) | Prefill (ms/token) | 关键变更 |
|------|----------------|-------------------|----------|
| 2.1（cuBLAS 基线）| 302 | — | 起点 |
| 2.1（自定义 GEMV）| 216 | — | 自定义 FP16/GPTQ GEMV |
| 2.2（dual GEMV）| 183 | — | 融合 gate+up GPTQ GEMV |
| 2.3（K_THREADS=8）| 176 | — | GPTQ GEMV 线程数翻倍 |
| 2.4（INT8 量化）| 135.5 | — | FP16 投影层 INT8 量化 |
| 2.5（scale 延迟）| 131.5 | — | GPTQ scale 延迟乘法 |
| 2.6（融合）| 123.8 | 132.6 | INT8 lm_head + 内核融合 |
| 3.0（批量 prefill）| 123.8 | 92.0 | INT8 batch GEMV + 批量注意力 |
| **3.1（tensor core WMMA）** | **123.8** | **46.9** | **WMMA GPTQ + INT8 tensor core** |

### 差距分析

理论最小值（带宽受限）：
- GPTQ MLP INT4 数据：8.22 GB / 192 GB/s = 42.8 ms
- INT8 投影：8.51 GB / 192 GB/s = 44.3 ms
- DN 串行：~18 ms
- Norms/misc：~3 ms
- **最小值：~108 ms（9.8 ms/token）**

当前 516 ms。MLP WMMA ~248 ms（18% BW 利用率），INT8 WMMA ~99 ms（45% BW 利用率）。
改进空间：SMEM 双缓冲、更大 tile、或 CUTLASS 混合输入 GEMM。

### 下一步

- **GPTQ WMMA 占用率调优**：SMEM 双缓冲重叠加载与计算
- **Norm 融合**：RMSNorm 融入 WMMA 输出写入
- **CUTLASS 混合输入 GEMM**：INT4×FP16 逼近带宽极限
- **DeltaNet 并行扫描**：大 M 下的 chunk-parallel FLA

## 2026-04-04 — 阶段 3.2：SMEM Bank Conflict 消除

### 背景

阶段 3.1 达到 516 ms（46.9 ms/token）。通过在 forward\_prefill 中插入 cudaEvent
计时获得实际组件分解：
- MLP/GPTQ: 358.6 ms (69%)
- DeltaNet: 117.0 ms (23%)
- Full Attention: 26.3 ms (5%)
- Norms/misc: 2.9 ms (0.6%)

GPTQ WMMA 内核仅实现可达 DRAM 带宽（171 GB/s）的 ~13%。隔离基准测试确认
gate\_proj 每次调用 1.575 ms，29.2 GB/s。

### 根因：ncu 分析

`ncu --set full` 揭示了关键问题：

> **32.1 路 SMEM bank conflict**，预估加速 **77%**

原因：SMEM leading dimension 恰好为 64 或 128 halfs（128 或 256 字节），
使每行映射到**相同的 32 个 bank**。WMMA `load_matrix_sync` 读取 16×16 tile 时，
同一列的 16 行全部命中同一个 bank → 32 路序列化。

修复前 ncu 指标：
- Memory Throughput: 88%（被 bank conflict 的 SMEM 读取饱和）
- Compute Throughput: 31%
- IPC: 1.04

### 修复：SMEM Padding (+8 halfs)

在 GPTQ 和 INT8 WMMA 内核的 SMEM leading dimension 添加 +8 half 填充：
- **GPTQ**: `smem_x[16, 72]` + `smem_w[64, 72]` = 11.25 KB/block
- **INT8**: `smem_x[16, 136]` + `smem_w[64, 136]` = 21.25 KB/block

每行偏移 4 个 bank。16 行 → 8 个唯一 bank 偏移 → 最多 2 路冲突（原 32 路）。

*对齐约束*：PAD 必须是 8 halfs 的倍数以保证 float4 对齐。PAD=4 曾导致
float4 未对齐崩溃（SIGSEGV）。

同时将 GPTQ BK 从 128 降到 64：降低 SMEM（10 KB → 4 blocks/SM）提高占用率。

### ncu 修复后

- Memory Throughput: 51.6%
- **Compute Throughput: 66.8%**（现在的瓶颈 — ALU 反量化）
- **IPC: 2.20**（2.1× 提升！）
- Achieved Occupancy: 42.33/48 warps (88%)

内核从 **内存受限（SMEM）** 变为 **计算受限（ALU）**。

### 结果

| 组件 | 阶段 3.1 | 阶段 3.2 | 改进 |
|------|---------|---------|------|
| MLP/GPTQ | 358.6 ms | ~198 ms | **1.81×** |
| DeltaNet | 117.0 ms | ~98 ms | **1.19×** |
| Full Attention | 26.3 ms | ~20 ms | **1.32×** |
| Norms/misc | 2.9 ms | ~3 ms | 1.0× |
| **总 Prefill** | **516 ms** | **328 ms** | **1.57×** |
| **每 token** | **46.9 ms** | **29.8 ms** | **1.57×** |

隔离 GPTQ WMMA 基准测试（gate\_proj 维度）：
- 修复前：1.575 ms, 29.2 GB/s (17.1%)
- 修复后：0.749 ms, 61.3 GB/s (35.9%)
- **仅 padding 一项即 2.10× 加速**

### 性能历程（更新）

| 阶段 | Decode (ms/tok) | Prefill (ms/tok) | 关键改动 |
|------|----------------|-----------------|---------|
| 3.0（批量 prefill）| 123.8 | 92.0 | INT8 batch GEMV |
| 3.1（tensor core WMMA）| 123.8 | 46.9 | WMMA tensor core |
| **3.2（SMEM bank fix）** | **123.8** | **29.8** | **+8 padding, BK=64** |

**总计：1012 → 328 ms，Phase 3.0 起 3.08× 加速。**

### 经验总结

1. **必须用 ncu 分析再做优化**——13% BW 利用率看似带宽瓶颈，实际根因是 SMEM
   bank 冲突。没有硬件 profiler 数据完全不可见。
2. **SMEM 步长 = 32 bank 的倍数是灾难**——stride 64, 128, 256... halfs
   的矩阵每行都映射到同组 bank。适用于所有 WMMA 内核。
3. **PAD 对齐要求**：float4 需要 16 字节对齐 → PAD 必须是 8 halfs 的倍数。
4. **BK 减半提升占用率**：BK 128→64 使 SMEM 减半，4 blocks/SM 而非 2，
   更好的延迟隐藏补偿了 2× 的 K-tile 迭代。

### 下一步

- **反量化 ALU 优化**：内核现在 66.8% ALU 受限，向量化加载、减少 FP 运算
- **DeltaNet 并行扫描**：消除 token 串行循环（估计 ~38 ms 开销）
- **CUTLASS 集成**：混合输入 GEMM + GPTQ 自定义反量化
- **理论最小值**：~108 ms (9.8 ms/tok)，当前 328 ms 仍高 3.04×

## 2026-04-05 — Phase 3.3：DeltaNet 融合内核 + 寄存器缓存状态

### 背景

DeltaNet prefill（48 层，11 token）是第二大耗时组件，~98 ms（总耗时的 30%）。
瓶颈是大量内核启动开销：每个 token 触发 ~7 次内核启动/层（conv1d、2× repeat_interleave、
compute_g_beta、2× l2norm、recurrent）= 总计 ~3,696 次启动。

### 方案

**两个融合内核替代逐 token 串行循环：**

1. **`conv1d_batch_silu_kernel`**：替代 M 次独立 `causal_conv1d_step_silu` 调用。
   每个线程处理一个通道的所有 M 个 token（conv 状态是逐通道串行的）。1 次启动
   （⌈10240/256⌉ = 40 blocks × 256 threads）替代每层 11 次启动。

2. **`deltanet_fused_head_kernel`**：融合 repeat_interleave + compute_g_beta +
   l2norm_q + l2norm_k + recurrent。Grid = 48 blocks（每个 value head 一个），
   Block = 128 threads。每个 block 内部循环所有 M 个 token。Q/K 源 head 映射
   （`head / 3`）替代显式 repeat_interleave。L2 norm 使用 warp shuffle 归约 +
   cross-warp 归约（通过共享内存）。

**寄存器缓存递归状态**：核心创新。每个线程拥有 S[128,128] float 状态的一列。
不再每个 token 从全局内存读写 S（2 遍 × 128 次加载+存储 = 64KB/head/token），
整列（128 float = 512 bytes）在内核启动时加载到寄存器，结束时写回一次。

内存流量减少：
- 之前：48 heads × 11 tokens × 2 passes × 128KB = ~132 MB 全局状态流量
- 之后：48 heads × 2 × 64KB = ~6 MB（加载一次 + 存储一次）
- **22× 递归状态内存流量减少**

### 寄存器压力分析

| launch_bounds | 寄存器/线程 | 溢出字节 | blocks/SM | 波次 | Prefill（中位数）|
|---------------|-----------|---------|-----------|------|-----------------|
| (128, 3)      | 168       | 1,184   | 3         | 1    | ~306 ms         |
| **(128, 2)**  | **255**   | **168** | **2**     | **2**| **~299 ms**     |

尽管需要 2 个执行波次（48 blocks / 32 active = 1.5 波次），寄存器溢出的
近乎消除（1184→168 bytes）完全补偿了占用率损失。

### GPTQ K-Inner 布局实验（已放弃）

同时测试了 K-inner 内存布局 + col_major WMMA B + float4 向量化存储。
结果：gate/up_proj +3.1%，down_proj -0.8%。整体 MLP 净收益 ~6 ms——
不值得增加代码复杂度。

### 结果

| 组件 | Phase 3.2 | Phase 3.3 | 提升 |
|------|-----------|-----------|------|
| DeltaNet 启动次数/层 | ~77 | 2 | **38× 更少** |
| DeltaNet 状态流量 | ~132 MB | ~6 MB | **22× 更少** |
| **总 prefill** | **328 ms** | **299 ms** | **1.10×** |
| **每 token** | **29.8 ms** | **27.1 ms** | **1.10×** |

### 性能历程（更新）

| 阶段 | Decode (ms/tok) | Prefill (ms/tok) | 关键变更 |
|------|-----------------|------------------|---------|
| 3.0（batched prefill） | 123.8 | 92.0 | INT8 batch GEMV + batched attn |
| 3.1（tensor core WMMA） | 123.8 | 46.9 | WMMA GPTQ + INT8 tensor core |
| 3.2（SMEM bank fix） | 123.8 | 29.8 | +8 padding, BK=64 |
| **3.3（DeltaNet 融合）** | **123.8** | **27.1** | **寄存器缓存状态，2 个融合内核** |

**总计：1012 → 299 ms，Phase 3.0 起 3.38× 加速。**

### 经验总结

1. **寄存器缓存对递归模型是变革性的**——SSM 状态 S[128,128] 每个 head 每个
   token 都要反复读写全局内存。把每线程列缓存到寄存器消除了所有中间流量。
2. **`__launch_bounds__` 占用率 vs 溢出的权衡非显而易见**——高占用率（3 blocks/SM）
   配合严重溢出（1184 bytes）比低占用率（2 blocks/SM）配合极少溢出更慢。
3. **融合收益叠加：启动开销 + 内存流量 + 寄存器复用**——融合内核省去 ~3500 次
   启动，消除中间缓冲区，并使跨内核不可能实现的寄存器缓存成为可能。

### 下一步

- **INT8 GEMV 带宽**：DeltaNet INT8 投影每层 ~110 MB × 48 层 = 5.3 GB。
  理论 171 GB/s → 31 ms 最小值。当前 ~68 ms 表明 INT8 内核有 2× 开销。
- **GPTQ WMMA 计算优化**：ALU 66.8%——探索 uint2 加载、FP16 反量化、LUT 转换。
- **理论最小值**：~108 ms (9.8 ms/tok)，当前 299 ms 仍高 2.75×。

## 2026-04-05 — Phase 3.4：GPTQ + INT8 内核深度优化

### 背景

通过分组 CUDA Event 性能分析获得精确分解（M=11, 64 层）：
MLP GPTQ = 198 ms (68%)，DeltaNet SSM = 61 ms (21%)，Full Attention = 20 ms (7%)，
Norms = 3 ms (1%)。GPTQ 内核受限于 ALU（ncu 测量 66.8% ALU-bound），原因是昂贵的
反量化链：每个 nibble `INT4→INT32→FP32 乘法→FP16` 消耗过多时钟周期。

### 四阶段优化

**阶段 1：GPTQ 反量化重构**（base_values 表 + hmul2 + K-inner + float4）
- 32 字节 SMEM 查找表 `base_values[16] = {-8,...,7}` 消除了 CVT.F16.S32 指令
- `__hmul2` FP16 乘法 2× 吞吐量 vs `FMUL.F32`
- K-inner SMEM 布局 + `col_major` WMMA B → 1 float4 写入替代 8 次单独写入
- 结果：**253 ms**（299→253），+15% 提升

**阶段 2：qweight 预加载**（内存级并行）
- 将 4 个 uint32 qweight 全部预载到寄存器，再做反量化
- 4 个同时发出的 DRAM 请求 vs 之前 1 个
- 结果：**200 ms**（253→200），+21% 提升

**阶段 3：INT8 WMMA FP16 反量化**
- `__int2half_rn()` + `__hmul2()` 替代 FP32 乘法链
- half2 SMEM 写入（每 uint32 仅 2 次 vs 4 次）
- 4 行一组预加载 + `__launch_bounds__(128, 2)`
- 结果：DN 60→53 ms (-12%)，FA 20→15 ms (-28%)

**阶段 4：GPTQ 寄存器级下一 tile 预取**
- WMMA 计算阶段仅读 SMEM → DRAM 空闲。在 WMMA 执行期间发出下一 tile 的全局加载
- 需要两组寄存器：`cur`（当前 tile）和 `nxt`（下一 tile）
- `__launch_bounds__(128, 2)` 提供 64 寄存器/线程
- 结果：MLP 101→93 ms (-8%)

### 失败实验
1. **BN=128 tile 扩展**：200→221 ms 退化，占用率从 4→2 blocks/SM 不足以补偿
2. **仅改 launch_bounds(128,2)**：无变化，编译器已优化寄存器分配

### 结果

| 组件       | Phase 3.3 | Phase 3.4 | 提升 |
|------------|-----------|-----------|------|
| MLP GPTQ   | ~198 ms   | 92.7 ms   | **2.14×** |
| DeltaNet   | ~61 ms    | 52.2 ms   | **1.17×** |
| Full Attn  | ~20 ms    | 14.4 ms   | **1.39×** |
| Norms      | ~3 ms     | 2.3 ms    | 1.30× |
| **总计**   | **299 ms**| **162 ms**| **1.85×** |
| **每 token** | **27.1** | **14.7**  | **1.84×** |

### 性能历程（更新）

| 阶段 | Prefill (ms/tok) | 关键变更 |
|------|------------------|---------|
| 3.0 | 92.0 | INT8 batch GEMV + batched attn |
| 3.1 | 46.9 | WMMA GPTQ + INT8 tensor core |
| 3.2 | 29.8 | +8 padding, BK=64, 4 blocks/SM |
| 3.3 | 27.1 | 寄存器缓存状态，融合内核 |
| **3.4** | **14.7** | **FP16 反量化，预加载，寄存器预取** |

**总计：1012 → 162 ms（层），Phase 3.0 起 6.25× 加速。**

### 带宽分析

| 内核 | 数据 (GB) | 时间 (ms) | 带宽 (GB/s) | 占理论 171 GB/s |
|------|-----------|-----------|-------------|-----------------|
| MLP GPTQ | 8.53 | 92.7 | 92 | 54% |
| DN INT8 proj | ~5.57 | ~42 | ~133 | 78% |
| FA INT8 proj | ~1.68 | ~10 | ~168 | 98% |

### 下一步

- MLP 仍占 57%。理论最小值 ~50 ms。需从 93→65 ms 范围。
- 可能：融合 silu_mul/add、合并 gate+up 投影、CUDA Graph（~3 ms 启动开销）。
- 目标：<10 ms/tok = 110 ms 总计。当前 162 ms 仍高 1.47×。

---

## 2026-04-05 — Phase 3.5：SM87 结构性分析 & INT8 寄存器预取

### 背景

从 Phase 3.4（162 ms，14.7 ms/tok）继续优化。目标：110 ms（10 ms/tok）。
系统性探索 GPTQ 内核优化空间，并将已验证的寄存器预取模式应用于 INT8 WMMA 内核。

### 实验总结

**6 个实验，2 个成功，4 个失败：**

#### ❌ GPTQ SMEM 双缓冲（92→102 ms，-10% 回退）
- 理念：双 SMEM buffer（乒乓），反量化写入 B 时 WMMA 读取 A。
- SMEM 翻倍：11.5→23 KB/block。L1：105→82 KB。
- 结果：L1 缩减导致 X 张量（160 KB）缓存命中率下降，损失大于重叠收益。

#### ❌ GPTQ BK=128（92→97 ms，回退）
- 理念：匹配 INT8 内核的 BK=128，tile 数减半（80→40），sync 减半。
- SMEM：21.3 KB/block。L1：85 KB。64 寄存器，0 溢出。
- 结果：L1 压力导致回退。INT8 在 BK=128 下正常是因反量化更简单，不是因 BK 本身。

#### ❌ GPTQ 4 blocks/SM（92→98 ms，回退）
- 理念：更多 warp（8→16）以隐藏 DRAM 延迟。
- SMEM：46 KB。L1：82 KB。同样问题。

#### ❌ 并发 gate+up 双 CUDA Stream（158→163 ms，回退）
- GPU 调度器将两个内核的 block 放在同一 SM 上，产生 4 blocks/SM，触发相同 L1 退化。
- 另外需要 fork-join 事件同步（首次没有 fork 导致数据竞争，输出错误 token）。

#### ✅ INT8 WMMA 寄存器预取 + Scale 提升（-3.6 ms）
- 将 GPTQ 的寄存器预取模式应用到 INT8 内核：预加载 16 个 uint32 权重 + 2 个 float4 X。
- 将 per-channel scales 提出循环（不依赖 K-tile），缓存为 `half2 cached_s2[16]`。
- DN：52.1→49.6 ms（-4.7%），FA：14.5→13.3 ms（-8.3%）。

#### ✅ 向量化 add_kernel（忽略不计）
- 标量→float4 路径 + `__hadd2`。对 56K 元素影响 <0.4 ms。

### DRAM 带宽实测

流式读取内核实测 Orin DRAM 带宽：**峰值可达 175 GB/s**（128-256 MB 工作集）。
cudaMemcpy D2D：153 GB/s（读写合计）。

### SM87 结构性瓶颈

**所有增加 SMEM 的优化都因同一原因失败：**

SM87 L1+SMEM 统一预算 128 KB/SM。GPTQ 最优：2 blocks × 11.5 KB = 23 KB → L1=105 KB。
X 张量 160 KB > 105 KB → 缓存未命中。任何 SMEM 增加（双缓冲、BK=128、4 blocks、
并发流）进一步压缩 L1 → 更多未命中 → 回退。

GPTQ 55% 带宽利用率（97/175 GB/s）的根因：反量化阶段 DRAM 空闲，占每 tile 约 50%
时间。SMEM 双缓冲可解决但需 2× SMEM → L1 退化。这是 SM87 的结构性限。

### 结果

| 组件       | Phase 3.4 | Phase 3.5 | 提升 |
|------------|-----------|-----------|------|
| MLP GPTQ   | 92.7 ms   | 92.4 ms   | ~不变 |
| DeltaNet   | 52.2 ms   | 49.6 ms   | **-5.0%** |
| Full Attn  | 14.4 ms   | 13.3 ms   | **-7.6%** |
| Norms      | 2.3 ms    | 2.3 ms    | 不变 |
| **总计**   | **162 ms**| **158 ms**| **-2.5%** |
| **每 token** | **14.7** | **14.3**  | **-2.7%** |

### 理论极限

以 175 GB/s 峰值带宽：
- MLP 理论最小：51 ms（当前 92，54% 利用率）
- DN 理论最小：32 ms（当前 50，64%）
- FA 理论最小：10 ms（当前 13，73%）
- **理论总最小：~95 ms**（当前 158，60% 整体利用率）

110 ms 目标需 86% 平均带宽利用率。在 SM87 上通过内核级调优已无法达到，
需要架构级变更（不同量化方式、持久内核或下一代硬件）。

---

## 2026-04-06 — Phase 3.6：子层剖析、内核融合、内联反量化

### 背景

从 158 ms（14.3 ms/tok）继续优化 prefill。目标：110 ms（10 ms/tok）。
本阶段聚焦子层剖析定位瓶颈、内核融合消除启动开销、内联反量化消除 SMEM 银行冲突。

### 主要变更

1. **GPTQ 内联 `__int2half_rn`**（-5.9 ms）：替代 SMEM base_values 查表，
   消除 4-way 银行冲突。MLP 92.4 → 86.5 ms。
2. **子层剖析**：逐操作定时揭示 94% 时间在线性投影，ab_proj 启动主导。
3. **DN 双核 batch GEMV**（-1.0 ms）：ab_proj N=48×2 融合为单次启动。
4. **Silu 融合失败**（+11 ms，已回退）：数据依赖的 silu ALU 序列化预取流水线。
5. **残差加法融合**（-0.3 ms）：down_proj 输出直接写入残差，消除 add 内核。

### 结果

| 组件 | Phase 3.5 | Phase 3.6 | 提升 |
|------|-----------|-----------|------|
| MLP | 92.4 ms | 86.1 ms | **-6.8%** |
| DN | 49.6 ms | 48.3 ms | **-2.6%** |
| FA | 13.3 ms | 13.6 ms | 约持平 |
| **总计** | **158 ms** | **150 ms** | **-5.1%** |
| **每token** | **14.3** | **13.7** | **-4.2%** |

**第 3.0 阶段至今：1012 → 150 ms，6.7 倍加速**。

---

## 2026-04-06 — Phase 3.7：合并投影权重

### 背景

Phase 3.6 的子层剖析揭示了两个优化机会：
- DeltaNet ab_proj（in_proj_a + in_proj_b，各 5120→48）消耗 0.065 ms/层 × 48 = 3.12 ms
  ——batch GEMV 中 X 的冗余读取（每个输出行读一次 X，96 行 = 96 次）
- Full Attention 的 k_proj 和 v_proj（各 5120→1024）执行两次独立 WMMA 调用，
  每次仅填充半数 wave

### 方法：加载后权重合并

创建 `merge_projection_weights()` 函数，在模型加载后运行：

1. **DN qkv+ab 合并**：拼接 qkv[10240,5120] + a[48,5120] + b[48,5120] → INT8 [10368,5120]
   （10336 对齐到 WMMA 的 64 倍数）。一次 WMMA 调用替代 qkv + dual batch GEMV。

2. **FA k+v 合并**：拼接 k[1024,5120] + v[1024,5120] → INT8 [2048,5120]。
   一次 WMMA（32 tile = 2 full wave）替代两次独立 WMMA（各 16 tile = 1 wave）。

额外显存：2592 MB（合并权重；原始权重无法从 pool 中单独释放）。

### 结果

| 组件 | Phase 3.6 | Phase 3.7 | 提升 |
|------|-----------|-----------|------|
| DeltaNet SSM | 48.3 ms | 46.0 ms | **-2.3 ms (-4.7%)** |
| Full Attention | 13.6 ms | 12.5 ms | **-1.1 ms (-8.1%)** |
| MLP GPTQ | 86.1 ms | 86.1 ms | 不变 |
| 总计 | 150.3 ms | 147.0 ms | **-3.3 ms (-2.2%)** |
| 每 token | 13.66 | 13.36 | **-0.30 ms/tok** |

**第 3.0 阶段至今：1012 → 147 ms，6.9 倍加速**。

---

## 2026-04-07 — Phase 3.8（实验）：PTX MMA 替代 WMMA（已回退）

### 背景

Phase 3.7 之后，5 个 GPTQ 内核变体（warp-spec syncthreads/named-barrier、BK=128/BN=32、
3-blocks/SM、INT8 WMMA）全部无改善——均与 V1 基线相同 ~15260 µs。假设 WMMA API 的
结构性 SMEM bank 冲突（数学证明不可避免：FP16 padding 的 GCD 至少 4-way）是 ~61%
BW 效率上限的根因。

### 实验

**PTX mma.m16n8k16 + ldmatrix.x4** 替代 WMMA，消除结构性 bank 冲突：

1. 确认 SM87 支持 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 和
   `ldmatrix.sync.aligned.m8n8.x4.shared.b16` ✅
2. **经验性发现 A fragment 寄存器顺序**（与朴素假设不同）：
   - a0={A[g,2l], A[g,2l+1]}（顶行，左 k-半）
   - **a1={A[g+8,2l], A[g+8,2l+1]}**（**底行**，左 k-半）　← 注意与 a2 交换
   - **a2={A[g,2l+8], A[g,2l+9]}**（顶行，**右 k-半**）
   - a3={A[g+8,2l+8], A[g+8,2l+9]}（底行，右 k-半）
   - 通过 WMMA fragment dump 对比手动 PTX 寄存器发现
3. 完整 ldmatrix.x4 + ldmatrix.x2.trans + MMA 随机 GEMM 测试：max\_err=0.0015，通过
4. 写入 `gptq_ptx_mma_gemm_kernel` 和 `gptq_ptx_mma_gemm_add_kernel`，替换分发

### 结果

| 指标 | WMMA V1 | PTX MMA | 变化 |
|------|---------|---------|------|
| bench-gptq M=128 | 15260 µs | 15262 µs | **无变化** |
| test-forward Prefill | 147-168 ms | **193.4 ms** | **+31% 回退** |

### 分析

- **隔离基准完全相同**：SMEM bank 冲突在计算阶段 **不是瓶颈**
- **全前向回退 31%**：手动标量 W 加载 + FP32 scatter store 的指令调度比 WMMA API
  差。WMMA 允许 nvcc 内部排列指令以重叠内存访问和计算；内联 PTX 破坏了这种优化
- **结论**：55% BW 效率的根因是反量化 ALU 占时（DRAM 空闲 ~50% tile 时间），
  而非计算阶段的 SMEM bank 冲突

### 已回退

完整回退至 V1 WMMA 内核。PTX MMA 内核代码已删除。

### 经验总结

1. **微基准结果相同时，不要假设全系统也相同**——指令调度、寄存器压力、编译器优化
   在全前向中产生显著差异
2. **WMMA API 比手工 PTX 更好**（在 SM87 上）——编译器对 WMMA 有特殊优化路径，
   手工 PTX 丧失这些优化
3. **PTX MMA fragment 顺序**（SM87 实测）：行交错在 k 推进之前
   （a0=顶左, a1=**底**左, a2=顶**右**, a3=底右），与部分文档描述不同

---

## 2026-07-16 — Marlin 优化：SMEM 精确分配 + 融合残差加 + Tile 配置调优

### 背景

Marlin GPTQ INT4 移植（Phase 4.0, commit f7b5983）及内存开销审计（commit dd9bc57）后，
MLP 层耗时 55.46 ms（64 层），总计 119.21 ms（10.84 ms/tok，M=11）。发现三个优化点：

1. SMEM 过分配：所有内核配置统一 96 KB，实际需求仅 42–66 KB。SM87 共 128 KB 统一
   L1/SMEM，96 KB SMEM 仅留 32 KB L1
2. down_proj 后单独的 elementwise_add 内核：每 pass 64 次多余内核启动
3. M≤16 的 tile 配置 (1,8,8,8) [thread_k=128, thread_n=128] 可能非最优

### 修改

**1. 按内核配置精确计算动态 SMEM**

用 `MARLIN_SMEM_BYTES(M,N,K)` 编译期宏替代 `96*1024`：
```
(1,16,4,8): 42 KB SMEM → 86 KB L1  (原 96→32)
(2,16,4,8): 50 KB → 78 KB L1
(3,16,4,8): 58 KB → 70 KB L1
(4,16,4,8): 66 KB → 62 KB L1
```

**2. Marlin write_result 融合残差加**

内核新增 `const int4* residual` 参数。非空时 write_result 执行读-改-写：
`C[i] = residual[i] + result`。仅 M ≤ 64 安全（单 slice，无 global reduce）。
新 API：`marlin_gemm_add()` 原地累加。

**3. 统一 tile 配置：所有 M 使用 thread_k=64, thread_n=256**

M≤16 从 (1,8,8,8) 改为 (1,16,4,8)：相同总迭代次数（5440/SM），但更宽 N tile
改善写局部性，更少 SMEM（42KB vs 49KB）释放更多 L1。

### 结果

| 组件 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| MLP gate_proj | 0.300 ms | 0.292 ms | **-2.7%** |
| MLP up_proj | 0.292 ms | 0.285 ms | **-2.4%** |
| MLP down_proj+add | 0.287 ms | 0.272 ms | **-5.2%** |
| **MLP 总计** | **0.886 ms** | **0.856 ms** | **-3.4%** |
| MLP ×64 | 55.46 ms | 53.80 ms | **-1.66 ms** |
| **总计（64层）** | **119.21 ms** | **115.89 ms** | **-2.8%** |
| **每 token** | **10.84 ms** | **10.54 ms** | **-0.30 ms/tok** |

带宽利用率（实测峰值 175 GB/s）：gate 85%→87%，up 87%→89%，down+add 90%→**94.5%**

### 性能历程（更新）

| 阶段 | Prefill (ms/tok) | 关键变更 |
|------|------------------|----------|
| 3.7（合并权重）| 13.4 | qkv+ab 合并, k+v 合并 |
| 4.0（Marlin 移植）| 10.6 | Marlin GPTQ INT4 GEMM |
| **4.1（Marlin 调优）** | **10.54** | SMEM 精确分配, 融合残差加, tile 配置调优 |
