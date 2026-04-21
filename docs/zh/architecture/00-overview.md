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

### 进度日志（2026-04-21）

- [x] **第 1 步 —— 测试资产清理**：`tests/` 下仅保留 `test.mp3` 与
      `test.txt` 作为评估基线，旧 benchmark 文本全部删除（提交
      `32763b6`）。
- [x] **第 2 步 —— DEVLOG 按日归档**：`docs/{en,zh}/DEVLOG.md` 拆为
      `docs/{en,zh}/devlog/YYYY-MM-DD.md` 每日文件；顶层 DEVLOG.md
      变成 24 行倒序索引（提交 `6f37fbc`）。
      `PLAN_AUDIO_ENHANCEMENT.md` 迁移到 `docs/{en,zh}/archive/`
      并附已超越标记。
- [x] **第 3 步 —— PDD 结构化**：927 行单体
      `.github/copilot-instructions.md` 按 GitHub Prompt-Driven
      Development 约定拆为导航中枢 + 作用域指令 + 可复用提示词 +
      架构 RFC（提交 `8e6c052`）。
- [x] **第 4 步 —— Actus 重命名**：`src/commands.{h,cpp}` →
      `src/actus/actus.{h,cpp}`，加入哲学锚点（提交 `0eea0e6`）。
- [x] **第 5 步 —— Actus 拆分**：2768 行 `src/actus/actus.cpp` 拆为
      14 个单命令翻译单元 + 100 行注册文件（提交 `2525450`）。除
      `cmd_test_ws.cpp`（1543 行，将作为 Auditus 外观的自然目标）外
      均已满足 R1。
- [x] **第 6 步 —— 外观评估**：见下方清单（提交 `851df90`）。
- [x] **第 7 步 —— Auditus 外观 + Actus 路由**（2026-04-21）：
      `cmd_test_ws.cpp` 从 1543 行 降至 458 行（R1 以内）。
      - 7a（`d96a503`）新建 `auditus_facade.{h,cpp}`；迁移
        vad / asr_partial / drop 三个回调。
      - 7b（`dbd9f9e`）transcript / asr_log / stats / speaker 迁移。
      - 7c（`cd3224d`）WS 二进制 PCM 入口迁移。范围收窄：
        connect/disconnect 广播的是 Conscientia 状态而非 Auditus
        事件，未纳入此次迁移。
      - 7d（`7e1c9ef`）545 行 `set_on_text` 命令路由器抽到平级
        Actus 翻译单元 `cmd_test_ws_router.{h,cpp}`——该路由器以
        单个 switch 跨越 Auditus + Conscientia + Persona，正是
        Actus 的职责，因此它与 `cmd_test_ws.cpp` 同层，
        而不入任何单子系统外观。
      - 7e（`628dd69`）91 行 `set_on_connect` hello 封装抽到
        `cmd_test_ws_hello.{h,cpp}`。
- [ ] **第 8 步+ —— 其余子系统外观**：`cmd_test_ws` 当前直接访问
      私有头文件的 Nexus、Memoria、Persona、Orator 等子系统。
- [ ] **第 9 步 —— CUDA/音频 R1 拆分大行动**：下表中剩余 11 个
      超限文件。

### 第 6 步 —— 外观评估（耦合清单）

对每个 `src/actus/cmd_*.cpp` 扫描其实际使用（非单纯 include）的子系统
类型：

| 命令 | 行数 | 实际使用的外部类型 |
|------|------|---------------------|
| `cmd_test_ws` | 1543 | AudioPipeline、WsServer、CacheManager、FRCRN、MossFormer、Persona、Orator 说话人库、TimelineLogger |
| `cmd_test_wavlm_cnn` | 262 | Orator 说话人库 |
| `cmd_test_gptq` | 252 | GPTQ（machina 公共 API）|
| `cmd_bench_gptq` | 90 | GPTQ（machina 公共 API）|
| 其余 `cmd_*` | ≤ 174 | 仅 machina 公共 API（model / forward / allocator）|

**结论**：耦合负债几乎全部集中在长驻意识服务 `cmd_test_ws` 上。其余
所有 Actus 入口已经遵守目标子系统的公共 API 边界。因此外观化行动有
一个清晰的单点起步：Auditus。将 WS 装配抽到 `auditus_facade.{h,cpp}`
既能让 `cmd_test_ws.cpp` 回到 R1 范围，也为后续 Nexus / Memoria /
Orator / Persona 外观树立模板。

### 超限文件（R1 违规——Actus 于 2026-04-21 完结）

| # | 文件 | 行数 | 建议拆分 |
|---|------|------|----------|
| 1 | `src/sensus/auditus/audio_pipeline.cpp` | 2651 | → `pipeline_core.cpp`、`vad_orchestrator.cpp`、`speaker_matcher.cpp`、`asr_trigger.cpp` |
| 2 | `src/machina/forward.cu` | 2172 | → 按算子拆分（attention/mlp/norm/residual 发射器）|
| 3 | `src/orator/wavlm_ecapa_encoder.cu` | 2084 | → `wavlm_encoder.cu` + `ecapa_encoder.cu` + 共享 utils 头 |
| 4 | `src/machina/gptq.cu` | 2029 | → `gptq_gemv.cu` + `gptq_gemm.cu` + `gptq_dequant.cu` |
| 5 | `src/machina/layer.cu` | 1953 | → `ssm_layer.cu` + `attn_layer.cu` + `mlp_layer.cu` |
| ~~6~~ | ~~`src/actus/cmd_test_ws.cpp`~~ | ~~1543~~ → **458** | **第 7 步已解决（2026-04-21）**：router + hello + auditus_facade |
| 7 | `src/sensus/auditus/mossformer2.cu` | 1544 | → 编码器/解码器按 block 拆分 |
| 8 | `src/orator/speaker_vector_store.cu` | 1404 | → 索引 + 核函数 + I/O 拆分 |
| 9 | `src/sensus/auditus/frcrn_gpu.cu` | 1256 | → frcrn_encoder + frcrn_decoder |
| 10 | `src/machina/marlin.cu` | 1118 | 临界；审核单核函数合理性 |
| 11 | `src/machina/gptq_gemm_v2.cu` | 1085 | 并入 #4 的 gptq_gemm.cu |
| 12 | `src/sensus/auditus/audio_pipeline.h` | 1063 | 随 .cpp 的拆分同步拆分 |

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
