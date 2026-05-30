# 12 — DiariZen Reclusterer（会话边界说话人重归因）

> *拉丁名*（占位）：**Orator Recapitulator** ——"重新思考的说话人识别者"。
> 落在 [`src/orator/`](../../../src/orator/) 下，是会话边界 reclusterer，
> 不新增顶层子系统。

## 为何需要

2026-05-29 的编码器验证总结
（[`tools/verification_2026/PLAN.md`](../../../tools/verification_2026/PLAN.md)）
给出一条硬数据：

```
accuracy(tests/test.mp3, speaker-id 4-way):
  baseline（live, 双编码器）                  31.0 %
  DiariZen-WavLM-large-s80-md-v2（离线）       93.5 %   (Δ = +62.5 pp)
```

`+62.5 pp` 是项目尚未兑现的最大单点收益。宪法规则
（`philosophy.instructions.md` §"准确率是唯一指标"）要求我们必须
落地——但**必须经过 live `awaken` 验证**，且**不破坏现有流式行为**。

## 待移植的算法栈

DiariZen-v2 是三段式流水线，每一段对应一个待新增或扩展的 C++/CUDA 组件：

| 段 | 算法 | 现有 C++/CUDA | 新增工作量 |
|----|------|----------------|-------------|
| **S — 分段** | WavLM-Large 编码器（25 个 hidden state，**BUT-FIT 做了结构化剪枝**——逐层 dim 不一致，例如 conv layer 1 是 153 ch 而非 512），16 s 块 / 0.1 s 步进 | `src/orator/wavlm_ecapa_encoder.*` 导出的是 vanilla 512-ch WavLM-Large 24 层；无法直接加载剪枝后权重。新 loader 必须根据 checkpoint 中实际张量形状驱动每层 dim | 中 |
| **C — Conformer EEND 头** | 4 层 Conformer (d=256, ffn=1024, 4 heads), MHA + conv module + macaron FFN + classifier 头**输出 16-d powerset logits**（= C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 1+4+6+4+1，覆盖最多 4 并发说话人），中值滤波 | 无 | **大** |
| **E — 嵌入** | WeSpeaker ResNet34-LM，256-d cosine | 无（现有 WavLM-ECAPA 是 192-d，维度不兼容） | 中 |
| **K — 聚类** | VBx（变分贝叶斯 HMM）+ PLDA 先验，`xvec_transform.npz` + `plda.npz` | 已有 `spectral_cluster_gpu.cu`，但是不同算法 | 中 |
| **O — 编排** | Pyannote `SpeakerDiarization.apply`（16 s 块循环、每块嵌入、AHC 初始化、VBx 收敛、标签拼接） | `auditus` 是逐帧流式，结构不匹配 | 中 |

硬数据锚：3615 s 录音的离线运行在 Orin GPU 上 740 s wall-clock（RTF 0.20），
其中 S 段占 ~85 % 时间。

## 集成形态——Hybrid（Reclusterer 模式）

考虑过三种集成形态，最终选 **C — Hybrid**：

1. **流式层保持不动。** Live `awaken` 继续从现有 WavLM-ECAPA + DualDb
   通道发出 `speaker_event`。延迟保持今天水平；31.0 % live baseline 是
   系统永远不会跌破的最坏情况。
2. **DiariZen 作为会话边界 reclusterer 运行。** 当 Vigilia 检测到会话
   边界（idle → active 转换、睡眠、长静默阈值、或 Nexus 显式用户请求）
   时，最近捕获的 PCM ring 被送入 DiariZen 流水线，top-4 cluster id
   通过重叠映射到现有 live speaker id，发出一串 `speaker_amend` 事件
   （Step 17b-A 已实现，见 [`/memories/auditus-tuning.md`](../../../)
   "17b-A PASSED" 条目），事后改写 transcript 的 `speaker_id` 字段。
3. **未经 live 验证不切默认。** 按宪法规则，reclusterer 出厂时
   `DEUSRIDET_DIARIZEN_RECLUSTER=0`；只有当 live `awaken` 跑出
   `accuracy(tests/test.mp3, speaker-id 4-way): 31.0% → X%` 那条线
   之后，才允许翻为 `1`。

### 为什么是这个形态

- **反熵。** 增加能力但不删除/重写任何工作中的代码。流式
  SpeakerVectorStore 仍是即时决策的真值来源；DiariZen 给出基于完整
  录音的更优后验。
- **哲学契合。** "Continuity over request-response"——流式大脑仍以
  20 W 运行；reclusterer 是空闲时触发的整理者，正如 Somnium 在睡眠
  时触发。
- **风险可控。** 若 C++/CUDA DiariZen 移植与 Python 参考偏离，症状
  就是错误的 `speaker_amend` 事件，可在 WebUI 观测到，环境变量
  一翻即可回到流式身份。
- **复用 Step 17b-A 基础设施。** `speaker_amend` 信封类型、广播
  线路、WebUI 消费端均已存在且通过验证；本工作仅改变 amend 事件的
  **来源**——从 RetroFullRing（peek_best 单次）改为 DiariZen
  （整段重分段 + 重聚类）。

## 文件布局（计划）

新代码全部进 `src/orator/`——本质是说话人归因算法；不新增顶层子系统。

```
src/orator/
├── diarizen_pipeline.{h,cpp}          # 门面：PCM → speaker_amend 列表
├── diarizen_conformer_head.{h,cu}     # 4 层 Conformer EEND 头
├── diarizen_segmentation.{h,cu}       # 16 s 块循环、中值滤波
├── diarizen_resnet34_embed.{h,cu}     # WeSpeaker ResNet34-LM 256-d
├── diarizen_vbx_cluster.{h,cu}        # VBx VB-HMM + PLDA 评分
└── diarizen_weights.{h,cpp}           # 四份权重的 safetensors 加载器
```

加一个一次性 Python 工具（可接受：装机时跑一次，输出确定性 safetensors）：

```
tools/convert_diarizen_to_safetensors.py
  → ~/models/dev/diarizen_v2/
       wavlm_large_s80_md.safetensors          (~1.2 GB FP16)
       conformer_eend_head.safetensors         (~30 MB FP16)
       wespeaker_resnet34_lm.safetensors       (~26 MB FP16)
       xvec_transform.bin                       (LDA 矩阵的原始打包)
       plda.bin                                 (PLDA 先验的原始打包)
```

`.npz` 先验（`xvec_transform`、`plda`）转为打包二进制：体积小
（~MB 级），仅 VBx 内核消费，safetensors 是 overkill。

## 显存预算影响

四份 DiariZen 权重合计 **~1.3 GB FP16**，全部**懒加载**——首次
reclusterer 触发时加载，会话结束后**释放**（不常驻）。这一点很重要——
`11-machina.md` 中常驻预算已经紧（~64 GB DRAM）。

- 瞬时峰值：~1.3 GB 权重 + ~600 MB 激活 scratch + ~200 MB 每会话嵌入
  缓存 ≤ **~2.1 GB 瞬时**。
- 计算路径：S 段（WavLM-large）占 ~85 % wall-clock，C/E/K 三段各 ≤ 5 %。
- Stream：reclusterer 跑在专用低优先级 CUDA 流，不能饿死 Conscientia
  prefill / decode。

Loader 落地时更新 `11-machina.md` Machina 显存预算表，标注为
`env-gated, lazy, session-scoped`。

## 分阶段计划（每阶段独立可验证）

按 [`workflow.instructions.md`](../../../.github/instructions/workflow.instructions.md)，
任何一阶段都不超过软上限，每阶段以绿色构建收尾。

> **2026-05-29 更新。** 下表中原生 CUDA 移植（P1–P3）**延后**。
> 我们通过 IPC 捷径调外部 Python DiariZen-v2 进程，已交付等价能力。
> 见 `docs/{en,zh}/devlog/2026-05-29.md`。表格保留以备未来原生移植；
> 表尾的 *Hybrid IPC* 行记录今天实际发布的内容。

| 阶段 | 交付 | 验证 | 状态 |
|------|------|------|------|
| **P0** | 本 RFC（en/zh）+ `00-overview.md` TOC 更新 + 权重转换脚本（Python，在现有 `py310_diarizen` env 跑） | RFC 可读，脚本产出 4 个 safetensors 文件 | **done 2026-05-29** |
| **P1a** | WavLM-Large 25-hidden tap + s80-md safetensors loader 扩展到 `wavlm_ecapa_encoder` | `test_wavlm_s80md` 与 Python 参考的 cosine ≥ 0.999（一个 16 s 块） | 延后（由 IPC 捷径替代） |
| **P1b** | `diarizen_conformer_head.cu` 前向 + 权重加载器 + 中值滤波 | 在固定输入张量上 dry-run，匹配 Python `model.head(x)` 至 ≤ 1e-3 abs | 延后 |
| **P1c** | `diarizen_segmentation.cu` 编排器（16 s × 0.1 s 滑动 + 拼接），跑在 P1a + P1b 之上 | 在 `tests/test.mp3` 前 60 s 上，端到端分段 logits 与 Python 参考 cosine ≥ 0.99 | 延后 |
| **P2a** | `diarizen_resnet34_embed.cu` + safetensors loader | 10 个参考片段上嵌入与 Python cosine ≥ 0.999 | 延后 |
| **P2b** | `diarizen_vbx_cluster.cu`（`VBx.py` 的 NumPy → CUDA 移植） | 固定嵌入序列上标签序列与 Python 位等价 | 延后 |
| **P3a** | `diarizen_pipeline.cpp` 门面把 S→C→E→K 接起来 | 在 `tests/test.mp3` 上离线运行通过 `tools/verification_2026/offline_score.py` 重现 93.5 % ± 0.5 pp | **已经 IPC 完成**（`e96255b`） |
| **P3b** | `awaken` 集成：会话边界触发器 + `speaker_amend` 广播，受 `DEUSRIDET_DIARIZEN_RECLUSTER=1` 控制 | 由 `tools/replay_to_transcript.py` 捕获的 live `awaken` 跑出 `accuracy(tests/test.mp3, speaker-id 4-way): 31.0% → X%` | **已经 IPC 完成**（`b0e3a8f` + `0cc9d0d`） |
| **P3c** | 默认翻为 `=1`——**当且仅当** P3b 实测 live 准确率 ≥ 80 % | commit message 中带宪法 accuracy 行 | 待 LLM 加载复测 |
| **Hybrid IPC P0** | `DiarizenFacade` C++/Python 行 JSON 桥，使用 `tools/diarizen_worker.py` | `tests/test.mp3` 上 round-trip diarize 调用返回 1658 段 | **done 2026-05-29**（`e96255b`） |
| **Hybrid IPC P1** | `AudioPipeline` session 捕获 tap + WS `diarizen_finalize` | 通过 `tools/diarizen_live_score.py` 得 `accuracy(tests/test.mp3, diarization): — → 93.6%` | **done 2026-05-29**（`b0e3a8f`） |
| **Hybrid IPC P2** | `TranscriptHoldback` + `DiarizenPeriodicWorker`；WS `diarizen_trigger` / `diarizen_finalize`；LLM 注入前重写 `speaker_id` | `accuracy(tests/test.mp3, diarization): 93.5% → 93.6%` 无回归 | **done 2026-05-29**（`0cc9d0d`） |
| **Hybrid IPC P2-verify** | LLM 加载（`DEUSRIDET_TEST_WS_ENABLE_LLM=1`）端到端复测 | holdback 激活下 accuracy 保持 ≥ 93.5% | **engine 稳定、gate 阻塞** 2026-05-29（`c294ebf` + `6249481`）—— 27B Qwen3.6-uncensored-heretic GPTQ-Int4 LLM 在 live `awaken` + diarizen 回放期间整 50 分钟运行无任何 CUDA 错误，但 diarizen worker 卡在 `facade.diarize returned empty: diarize: no opening brace` 的死循环，超过 client 1500 s 预算，未返回 `speaker_diarize_final`，因此没有 accuracy 行。阻塞从 "27B prefill kernel mismatch"（由 `c294ebf` 解决）重新分类为 "worker 再 extract 死循环" |

按 `workflow.instructions.md` git 纪律，任一阶段失败不阻塞汇报；
每次尝试都提交，即使最后回滚。

### 架构锚定 —— 原生 CUDA P1–P3 是强制，不是可选

IPC 捷径（Hybrid IPC P0/P1/P2，表尾三行）是过渡桥，不是终点。
项目的硬约束是**纯 C++/CUDA**，覆盖所有常驻子系统（见
[philosophy.instructions.md](../../../.github/instructions/philosophy.instructions.md)
§"Compute Belongs on the GPU" 与项目一句话定义"自洽的多模态 LLM
应用"）。在推理回路上挂 Python 子进程，仅当作临时兼容垫片可
接受，永远不能作为发布默认。所以：

- **原生 P1a/P1b/P1c**（WavLM s80-md tap + Conformer EEND head +
  segmentation 编排器）**必须落地**，DiariZen 才能被翻为
  `DEUSRIDET_DIARIZEN_ENABLE=1` 默认。当前状态：
  `延后（由 IPC 捷径替代）`。
- **原生 P2a/P2b**（ResNet34-LM 嵌入 + VBx 聚类）出于同一理由
  **必须落地**。当前状态：`延后`。
- **原生 P3a（`diarizen_pipeline.cpp` C++ 门面）** 替代 Python
  worker 全部职责。**P3b-3（已完成）：** `tools/diarizen_worker.py`
  子进程与 `src/orator/diarizen_facade.{h,cpp}` 中的行 JSON 桥已
  **删除**；原生 pipeline 是唯一路径（`DEUSRIDET_DIARIZEN_ENABLE=1`
  在启动时加载它）。`DiarizenSegment` 现位于 `diarizen_pipeline.h`。
- IPC 制品（`diarizen_worker.py`、`DiarizenFacade` JSON 桥、
  `test_diarizen_facade`）——**已在 P3b-3 删除**；部署说明里的
  `py310_diarizen` conda env 仅作历史遗存，不再参与任何运行时路径。
  活跃 DiariZen 代码库中已无遗留哲学违例。

**默认翻转 gate（P3c）** 因此受 *两个* 独立前提同时约束：
1. 原生 P1–P3 已落地（架构约束）。
2. live `awaken` + LLM 加载复测产出
   `accuracy(tests/test.mp3, diarization): <baseline>% → ≥ 93.5%`
   （宪法规则，philosophy §"Accuracy Is the Sole Metric"）。

任一单独前提都不够。IPC 捷径可以给出数字，但通过 Python 子进程
得到的数字不能作为默认值发布。

## 计划期识别到的风险

1. **VBx 与数据集绑死。** `xvec_transform.npz` 与 `plda.npz` 是和
   ResNet34-LM 256-d 嵌入联合拟合的。替换任何其他编码器（如
   ReDimNet）都需要**重训两个先验**——本工作范围外，参见
   `tools/verification_2026/PLAN.md` "Deferred candidates"。
2. **WavLM-Large 权重变体。** `wavlm_large_s80_md` 是 BUT-FIT 自蒸馏
   变体，不是 Microsoft 原版 WavLM。25-hidden tap 顺序和它们 config
   中的 `selected_channel=0` 必须精确复刻；P1a 受 Python 位等价检查
   闸口。
3. **VBx 是 CPU-friendly 的。** VB-HMM 迭代是 O(K²T)，K（最终
   说话人数）≤ 4，T ≈ 36 000（3600 s × 10 fps）。这真的在 "小到
   适合 CPU" 的边缘；对照 `philosophy.instructions.md` §"Compute
   Belongs on the GPU"。决策：第一刀放 CPU（确定性、可追踪、按
   Python 参考 ≤ 5 % wall-clock）；profiling 显示瓶颈再上 GPU。
4. **Conformer conv module。** Kernel-15 depthwise 1-D 卷积 + GLU
   激活——形状小但不常见。P1b 包含**逐 block 位等价检查**，不仅
   end-to-end。
5. **0.1 s 步进的中值滤波。** Pyannote 用 `scipy.signal` 中值
   滤波；对应 CUDA 内核 trivial，但**窗口大小必须匹配 Python
   默认**（9 帧 × 0.1 s = 0.9 s）。

## 本工作不做的事

- **不**替换 `speaker_db` 或 `speaker_vector_store`。它们仍是流式
  真值来源。
- **不**移除 `wavlm_ecapa_encoder`。它是流式编码器；DiariZen S 段
  并行运行（不同权重集、不同输出 tap、不同生命周期）。
- **不**影响 ASR。`speaker_amend` 信封只改写已 finalize 的 transcript
  条目的 `speaker_id` 字段。
- **不**改 Conscientia、Memoria、Vox。

## 参考

- 流水线源码：<https://github.com/BUTSpeechFIT/DiariZen>
- WavLM-large-s80-md(-v2) 权重：HuggingFace `BUT-FIT/diarizen-wavlm-large-s80-md-v2`
- WeSpeaker ResNet34-LM 权重：HuggingFace `pyannote/wespeaker-voxceleb-resnet34-LM`
- 离线准确率结果：`tools/verification_2026/PLAN.md` 行 "#7-v2"
- 候选选拔期的 GPU 驱动：`tools/verification_2026/diar_diarizen_gpu.py`
- Live-evidence 宪法规则：
  [`philosophy.instructions.md`](../../../.github/instructions/philosophy.instructions.md)
