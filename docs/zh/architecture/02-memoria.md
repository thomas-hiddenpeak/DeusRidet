# 02 — Memoria（缓存管理器 + 长期记忆）

Memoria 承载实体所记住的一切，从最近 100 ms 的工作 KV Cache 到终身的
情节记录。

参考：qwen35-thor 的 `cache_engine`、`cache_manager`、`kv_swapper`、
`block_tracker`。以意识中心的需求为准进行改编——**不可照搬**。

## 缓存管理器——工作记忆

三层架构：GPU 块 → SSD 溢出 → 丢弃。

| 组件 | 职责 |
|------|------|
| **BlockTracker** | 按请求跟踪块位置（GPU vs SSD） |
| **CacheManager** | 统一接口：KV 块分配、BlockTracker、SSD swapper |
| **KVSwapper** | 换出（GPU → staging → fwrite → SSD）、换入、预取 |
| **CacheEngine** | SSD 前缀缓存，LRU 驱逐，基于哈希查找 |

### 对 DeusRidet 的适配

- 意识是单一的持久"请求"——其 KV Cache 无限增长，**必须**优雅地
  溢出到 SSD。
- 多轨 Decode 分支共享 Prefill 前缀——块引用计数防止共享块被过早驱逐。
- SSM 递归态 + Conv 态必须与 KV 块一同快照（每个检查点单独的
  `.ssm` / `.conv` 文件）。
- SSD 写入后的 `FADV_DONTNEED` 在统一内存上至关重要，用于将物理页
  释放回 GPU 分配器。
- 预算：~20 GB。默认划分：KV 14 GB、SSM/Conv 2 GB、scratch 4 GB。

## 连续驱逐模型

Runner 风格服务器在 KV 预算耗尽时驱逐*整个请求*。DeusRidet 只有单一
无限生命周期的意识流——驱逐必须发生在流**内部**，在流继续运行的
同时有选择地丢弃个别 KV 块。

- **基于注意力分的重要性评分**：每次 Prefill 帧后，记录每个 KV 块
  在所有 Full Attention 层上获得的累积注意力权重。持续被忽略的块成为
  驱逐候选。由 `MemoriaImportanceScorer` 实现——在 Prefill 完成后
  异步运行于独立 CUDA stream。
- **驱逐触发的整合钩子**：KV 块被驱逐前，向 `SomniumConsolidator` 触发
  事件。它提取压缩摘要并写入情节存储。驱逐成为*留痕的遗忘*——没有
  记忆被默默丢失。
- **稀疏块表**：连续驱逐在 KV 序列中留下洞。Paged Attention 已经处理
  非连续块表；块表管理器需要高效跟踪空闲槽位。
- **DeltaNet SSM 作为潜意识连续性**：SSM 递归态**不受** KV 驱逐影响
  ——它们携带所有历史的压缩编码（带自然信息衰减）。即使 Full Attention
  失去了对被驱逐 KV 块的访问，SSM 态仍保留"潜意识印象"。这是混合
  模型在连续意识上的架构优势。

## Memoria Longa——长期记忆

除工作 KV 外，DeusRidet 在 SSD 上维护持久的长期记忆，按需加载到 GPU。

### 设计原则

- 零外部依赖——所有数据结构用 C++/CUDA 实现。
- 使用 LLM 隐藏态作为嵌入向量（零额外内存开销）。
- **始终保存原始文本**连同嵌入，以便模型升级安全：替换 LLM 后所有
  嵌入都失效，但原始文本允许完整重嵌入作为初始化步骤。
- 记忆整合由 `SomniumConsolidator` 负责（见 `04-vigilia.md`）
  ——Memoria 只提供存储与检索。

### 情节存储（"发生了什么"的向量检索）

| 属性 | 值 |
|------|-----|
| 索引 | HNSW |
| 向量 | LLM 最后一层隐藏态，dim=5120 |
| 存储 | SSD 支持，顶层常驻 GPU 以快速搜索 |
| 记录 | `{embedding, original_text, timestamp_t0, speaker, emotion, importance}` |
| 容量 | ~500 K–1 M 条（~2.5–5 GB SSD，HNSW 顶层 ~200 MB GPU）|

每条原始文本 50–500 字节。

### 语义图（实体-关系网络）

| 属性 | 值 |
|------|-----|
| 结构 | CSR 邻接 |
| 节点 | 实体：人、地点、概念、事件 |
| 边 | 加权，类型化（因果/时序/关联/情感）|
| 边衰减 | 基于时间的权重衰减，被重访加强 |
| 遍历 | 每跳 Top-K 剪枝 BFS |

### 图遍历约束（与人类认知极限对齐）

| 场景 | 最大跳数 | 时间预算 | 理由 |
|------|----------|----------|------|
| 对话（警觉/聚焦）| 1–2 | < 10 ms | 需适合一个 Prefill 帧 |
| 白日梦/空闲 | 3–4 | < 100 ms | 背景联想 |
| 深度梦境整合 | ≤ 6 | 无界 | 完整网络探索 |

每跳仅扩展按 `边权重 × 近度 × 情感突出度` 排名的 Top-K 邻居。防止
组合爆炸。

### 混合检索（MemoriaRetriever）

1. 用当前上下文嵌入查询 HNSW → Top-N 情节匹配。
2. 从匹配中提取实体 → 图遍历的种子节点。
3. 遍历语义图（对话 1–2 跳，梦境最多 6 跳）。
4. 合并结果 → 作为上下文注入下一个 Prefill 帧。

### 模型升级策略

LLM 升级后，所有基于嵌入的索引失效：

1. 加载新权重。
2. 遍历所有情节记录，用新模型重嵌入原始文本。
3. 重建 HNSW 索引。
4. 语义图（基于文本的节点/边）仍然有效——无需重建。

一次性初始化开销，概念上类似"醒来变成另一个人之后，用新眼睛重温
所有记忆"。

## 实现面

```
src/memoria/
├── cache_manager.{h,cpp}
├── cache_engine.{h,cpp}
├── block_tracker.h
├── kv_swapper.{h,cpp}
├── importance_scorer.{h,cu}
├── episodic_store.{h,cpp}
├── semantic_graph.{h,cpp}
└── retriever.{h,cpp}
```
