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

## 当前集成形态——原生 Hybrid

截至 2026-06-02，进程内 CUDA `DiarizenPipeline` 是唯一 DiariZen 运行时
路径。已退役的 Python-IPC 桥不再承担 fallback 角色。

1. **流式层仍负责即时响应。** Live `awaken` 继续从 Auditus/Orator
   流式路径发出低延迟在线 `speaker_event` 决策。当离线后验有更强证据时，
   这些在线决策可被视为临时标签。
2. **原生 DiariZen 修正整段会话。** `awaken` 默认启用内存 PCM 捕获缓冲，
   启动时加载一次 `DiarizenPipeline`，并接入 `DiarizenPeriodicWorker`。
   partial / 按需 pass 使用 120 s 滑动窗口服务 live WebUI 广播与 transcript
   holdback 改写；finalize 始终跑完整会话的原生 pass。
3. **运行时开关是退出开关，不是替代实现。** 原生 DiariZen 默认开启，
   用 `DEUSRIDET_DIARIZEN_ENABLE=0` 关闭。定时 partial pass 由
   `DEUSRIDET_DIARIZEN_PERIODIC=0/1` 控制；`diarizen_trigger` 与
   `diarizen_finalize` 无论如何都走原生路径。

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
       wavlm_pruned.safetensors          (127 MB FP16 — BUT-FIT pruned)
       conformer_head.safetensors        ( 12 MB FP16)
       wespeaker_resnet34.safetensors    ( 13 MB FP16)
       xvec_transform.npz                (134 KB — LDA matrices, verbatim)
       plda.npz                          (134 KB — PLDA priors, verbatim)
       shapes.json                       (per-tensor shape index)
```

`.npz` 先验保持原样，因为它们很小，C++ VBx 通过小型自定义读取器解包；
safetensors 在这里反而是 overkill。

## 显存预算影响

五份 DiariZen 权重制品合计 **~152 MB FP16**，均在 DiariZen 启用时加载；
与始终常驻的 LLM 相比很小，但仍属于 Orin 64 GB 统一内存预算的一部分。

- 瞬时峰值：~152 MB 权重 + ~600 MB 激活 scratch + ~200 MB 每会话嵌入
  缓存 ≤ **~1.0 GB 瞬时**。
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
| **P3b** | `awaken` 集成：capture buffer + WS `diarizen_trigger` / `diarizen_finalize` + partial/final 广播，受 `DEUSRIDET_DIARIZEN_ENABLE` 控制 | 由 `tools/replay_to_transcript.py` 捕获 live `awaken` 并产出宪法 accuracy 行 | 2026-05-29 **经 IPC 完成**，2026-05-30 被**原生进程内 CUDA 路径取代** |
| **P3c** | 默认翻为 `=1`——**当且仅当** P3b 实测 live 准确率 ≥ 80 % | commit message 中带宪法 accuracy 行 | **已翻转 2026-05-30**——native DiariZen 现在默认开启（`diarizen_enabled = true`；用 `DEUSRIDET_DIARIZEN_ENABLE=0` 退出）。`accuracy(tests/test.mp3, diarization): 93.6% → 93.6%`（同一 bit-eq 已验证路径），finalize RTF 0.10（369 s），0 CUDA 错误。由 periodic-worker + 广播 schema 修复解除阻塞（见 *Native P3c-verify* 行） |
| **Hybrid IPC P0** | `DiarizenFacade` C++/Python 行 JSON 桥，使用 `tools/diarizen_worker.py` | `tests/test.mp3` 上 round-trip diarize 调用返回 1658 段 | **done 2026-05-29**（`e96255b`） |
| **Hybrid IPC P1** | `AudioPipeline` session 捕获 tap + WS `diarizen_finalize` | 通过 `tools/diarizen_live_score.py` 得 `accuracy(tests/test.mp3, diarization): — → 93.6%` | **done 2026-05-29**（`b0e3a8f`） |
| **Hybrid IPC P2** | `TranscriptHoldback` + `DiarizenPeriodicWorker`；WS `diarizen_trigger` / `diarizen_finalize`；LLM 注入前重写 `speaker_id` | `accuracy(tests/test.mp3, diarization): 93.5% → 93.6%` 无回归 | **done 2026-05-29**（`0cc9d0d`） |
| **Hybrid IPC P2-verify** | LLM 加载（`DEUSRIDET_TEST_WS_ENABLE_LLM=1`）端到端复测 | holdback 激活下 accuracy 保持 ≥ 93.5% | **engine 稳定、gate 阻塞** 2026-05-29（`c294ebf` + `6249481`）—— 27B Qwen3.6-uncensored-heretic GPTQ-Int4 LLM 在 live `awaken` + diarizen 回放期间整 50 分钟运行无任何 CUDA 错误，但 diarizen worker 卡在 `facade.diarize returned empty: diarize: no opening brace` 的死循环，超过 client 1500 s 预算，未返回 `speaker_diarize_final`，因此没有 accuracy 行。阻塞从 "27B prefill kernel mismatch"（由 `c294ebf` 解决）重新分类为 "worker 再 extract 死循环" |
| **Native P3c-verify** | LLM 加载、*native* 进程内 DiariZen、holdback 激活 | live accuracy ≥ 93.5%、无 CUDA context 污染 | **完成 2026-05-30**——定位并修复两个阻塞：（1）periodic worker 每 60 s 对*整段*累积会话重新 diarize（O(N²)；后期单趟独占 GPU 211 s，饿死 live FRCRN/VAD/说话人识别流水线 → ring buffer 溢出 → illegal memory access → CUDA context 污染 → 最终 diarize 永不运行）。修复：定时节奏改为 `DEUSRIDET_DIARIZEN_PERIODIC=1` 门控（默认 OFF；finalize/trigger 路径不变）+ `enhance()` 设备路径补 `max_samples_` 边界钳制。（2）LLM 加载下 finalize 走 `worker->finalize()`，其广播用对象形 segment 且无 `ok` 字段，score client 读到 `FAILED: unknown`。修复：广播改为带 `ok` + 数组形 `[start,end,label]` segment + `audio_sec`/`wall_sec`（WebUI 两种形式都兼容）。结果：`accuracy(tests/test.mp3, diarization): 93.5% → 93.6%`，finalize RTF 0.10，0 FRCRN 错误，0 periodic 独占 |
| **Vires Background 路由** | 把原生前向（ResNet34 嵌入器 + Conformer 头 + WavLM-pruned 编码器）穿到 Vires **Background** 优先级流上，使其不再在 Tegra 默认流上屏障 live 感知 | 逐位一致（仅改流）+ live 不回退 + 0 CUDA 错误 | **完成 2026-05-30**（`afe9a15`）——每个子模型新增 `set_stream(cudaStream_t)`（绑定 cuBLAS/cuDNN 句柄；每个 `<<<…>>>` 携带该流；异步拷贝）；`DiarizenPipeline::load` 注册 `"diarizen"` Vires Background 消费者并绑定其流。同 kernel/顺序/数学 → P3a fixture bit-eq PASS 28/28（`min_cos 0.999980`）；`accuracy(tests/test.mp3, diarization): 93.6% → 93.6%`。争用收益：live finalize 墙钟 **685 s → 359.6 s**（RTF 0.19 → 0.099），因为 Background 流不再与 live 音频流水线串行化。见 RFC 13（Vires） |

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
  **已完成（原生进程内 pipeline；默认于 2026-05-30 翻转，
  现为 `=0` 退出制）**。
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
