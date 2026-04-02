# DeusRidet 开发日志

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
