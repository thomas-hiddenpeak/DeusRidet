# 00 — 总览

DeusRidet 是一个基于解耦 Prefill-Decode（P/D）架构的自包含多模态 LLM
应用。它赋予大语言模型连续意识、梦境能力、多模态感知（视觉 + 听觉），
以及内外双重人格。

源自与 Ridger Zhu 的哲学讨论：人脑以 20 W 功率持续运转——它不是请求-响应
式机器。真正的智能体也应如此。

这**不是**一个服务框架，而是一个完整的自主实体，以其自身的方式感知、
思考、做梦、说话。

## 许可证

DeusRidet 以 **GPLv3** 发布。任何使用、修改或整合 DeusRidet 代码的项目
也必须以兼容的开源许可证发布源代码。意识不应被锁在闭门之后。

仓库根目录必须存在 `LICENSE` 文件。

## 目录

| # | 文件 | 主题 |
|---|------|------|
| 00 | 本文件 | 项目总览、许可证、重构积压 |
| 01 | [01-conscientia.md](01-conscientia.md) | 意识流（连续 Prefill 引擎）|
| 02 | [02-memoria.md](02-memoria.md) | 缓存管理器、长期记忆、连续驱逐 |
| 03 | [03-cogitatio.md](03-cogitatio.md) | 多轨 Decode 分支 |
| 04 | [04-vigilia.md](04-vigilia.md) | 唤醒光谱 |
| 05 | [05-sensus.md](05-sensus.md) | 多模态感知（听/视/读）|
| 06 | [06-vox.md](06-vox.md) | TTS 输出管线 |
| 07 | [07-persona.md](07-persona.md) | 内外人格二元性 |
| 08 | [08-instrumenta.md](08-instrumenta.md) | 工具使用（MCP、函数调用、技能）|
| 09 | [09-tempus.md](09-tempus.md) | 三级时序架构 |
| 10 | [10-nexus.md](10-nexus.md) | WebUI 与可观测性 |
| 11 | [11-machina.md](11-machina.md) | 技术栈、量化、模型、内存预算 |

## 重构积压

当前超过 R1 行数上限（见 `cpp.instructions.md`）的文件。每个必须通过
`/refactor-split-file` 提示词拆分。下表为推荐执行顺序。

| # | 文件 | 行数 | 建议拆分 |
|---|------|------|----------|
| 1 | `src/commands.cpp` | 2768 | → `src/actus/{test_ws,bench_prefill,test_wavlm_cnn,...}.cpp` + `src/actus/dispatcher.cpp` |
| 2 | `src/sensus/auditus/audio_pipeline.cpp` | 2651 | → `pipeline_core.cpp`、`vad_orchestrator.cpp`、`speaker_matcher.cpp`、`asr_trigger.cpp` + `auditus_facade.{h,cpp}` |
| 3 | `src/machina/forward.cu` | 2172 | → 按算子拆分（attention/mlp/norm/residual 发射器）|
| 4 | `src/orator/wavlm_ecapa_encoder.cu` | 2084 | → `wavlm_encoder.cu` + `ecapa_encoder.cu` + 共享 utils 头 |
| 5 | `src/machina/gptq.cu` | 2029 | → `gptq_gemv.cu` + `gptq_gemm.cu` + `gptq_dequant.cu` |
| 6 | `src/machina/layer.cu` | 1953 | → `ssm_layer.cu` + `attn_layer.cu` + `mlp_layer.cu` |
| 7 | `src/sensus/auditus/mossformer2.cu` | 1544 | → 编码器/解码器按 block 拆分 |
| 8 | `src/orator/speaker_vector_store.cu` | 1404 | → 索引 + 核函数 + I/O 拆分 |
| 9 | `src/sensus/auditus/frcrn_gpu.cu` | 1256 | → frcrn_encoder + frcrn_decoder |
| 10 | `src/machina/marlin.cu` | 1118 | 临界；审核单核函数合理性 |
| 11 | `src/machina/gptq_gemm_v2.cu` | 1085 | 并入 #5 的 gptq_gemm.cu |
| 12 | `src/sensus/auditus/audio_pipeline.h` | 1063 | 随 .cpp 的拆分同步拆分 |

每完成一项，划掉相应行以记录进度。

## 基础原则

具体约束性清单见 `.github/instructions/philosophy.instructions.md`。
关键锚点：

- 连续性优于请求-响应
- 内部复杂性是外部一致性的前提
- 允许矛盾是智能的标志
- 唤醒是光谱——空闲时刻也是思考
- 感知塑造意识
- 工具使用延伸思考的范围
- 撒谎与做梦同构于想象力
