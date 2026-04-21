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
- [x] **第 8a 步 —— Conscientia 外观**（2026-04-22，提交 `b4ddea6`）：
      三个意识流 → WS 广播 lambda（decode / speech_token / state，
      83 行内联）抽到 `src/conscientia/conscientia_facade.{h,cpp}`，
      复用 `auditus_facade` 的 `install_*` 模板。共用 JSON 工具
      上提到 `src/communis/json_util.h`；`auditus_facade.h` 现以
      `using communis::{sanitize_utf8,json_escape}` 再导出，
      所有既有调用点零改动。`cmd_test_ws.cpp` 458 → 392 行。
- [ ] **第 8b+ 步 —— 其余子系统外观**：后续可能浮现的 Nexus /
      Memoria / Persona / Orator 反向渗透；当前 `awaken.cpp`
      中 126 行的 LLM+意识流引导块是下一个 Actus 层抽取候选
      （横跨 machina + memoria + conscientia + persona，
      应落入平级 Actus TU，而非子系统外观）。
- [x] **第 9 步 —— CUDA/音频 R1 拆分大行动**（2026-04-21，
      20 个提交 `57ecd1a` → `172b264`）：下表 12 个超限文件全部解决。
      20 个原子拆分依序推进；每次拆分独立通过 `cmake + make` 与
      觉醒仪式（HTTP=200 WS=101）验证。概要：
      - 头/.cpp：`auditus_facade.cpp` 525→130+421、`awaken_router.cpp`
        577→437+185、`ws_server.cpp` 607→359+277、`asr_engine.cpp`
        611→488+149、`spectral_cluster.h` 626→66+590（实体下沉到平级
        .cpp TU）、`tokenizer.cpp` 665→443+268、`stream.cpp`
        836→464+400、`model.cpp` 982→227+426+381、`audio_pipeline.h`
        1068→481+81+365+199、`audio_pipeline.cpp` 2656→327+1574+260+558。
      - .cu / .cuh：`asr_ops.cu` 898→535+378、`gptq_gemm_v2.cu`
        1092→656+461、`marlin.cu` 1125→379+555(.cuh)+238、
        `frcrn_gpu.cu` 1263→787+512、`speaker_vector_store.cu`
        1411→498+684+276、`mossformer2.cu`
        1551→35+590+518+425(.cuh)+70(.h)、`layer.cu`
        1960→552+495+436+555、`gptq.cu`
        2036→276+419+182+380+346+342+207+26(.cuh)、
        `wavlm_ecapa_encoder.cu` 2091→448+416+361+238+743(.cuh)、
        `forward.cu` 2179→639+247+339+336+431+320(.cuh)。
      - 技术要点：平级 `.cuh` 中的共享核函数声明为 `static __global__`
        （RDC 关闭 → 每个平级 TU 拿到独立实例化）；`sensus` / `orator`
        / `machina` 使用 `GLOB_RECURSE`，因此拆分无需改 CMake。
      - 遗留：三个 TU 在 .cpp 500 行上限之上，原因是每个都只包含
        一个不可再拆的方法/函数。第 11 步跟踪函数级分解。
- [x] **第 10 步 —— Actus 宪章复位**（2026-04-23，提交
      `d5fffd8`、`95ac9d3`、`728e39e`、`f530573`、`887a32e`）：
      诊断命名回归——`src/actus/` 下每个 TU 都背着 `cmd_` 前缀，
      且其中许多根本不是 Actus 动词（引擎探针、内核计时基准、
      集成测试）。五个原子子提交修复宪章：
      - 10b（`d5fffd8`）`bench_*` → `tools/`，独立可执行文件
        （开发者度量引擎的工具，并非在世间行动的实体）。
      - 10c（`95ac9d3`）`profile_*` → `tools/`（同上）。
      - 10d（`728e39e`）六个 `cmd_test_*` 引擎探针 →
        `tests/integration/`，独立可执行文件，仅链接所需库；
        函数体逐字保留。
      - 10e（`f530573`）硬切换 `test-ws` → `awaken`（无 CLI 别名）：
        DeusRidet 主 Actus 动词（让实体觉醒的那个动作）一直藏在
        开发味的标签 `test-ws` 之下。文件、符号、日志标签、
        以及标准验证仪式全部更新。
      - 10f（`887a32e`）去掉最后两个 Actus 动词的 `cmd_` 前缀：
        `cmd_load_model` → `load_model`，
        `cmd_load_weights` → `load_weights`。CLI 动词不变。
      结果：`src/actus/` 现仅剩六个 TU——`actus.{h,cpp}`、
      `awaken.cpp`、`awaken_router.{h,cpp}`、`awaken_hello.{h,cpp}`、
      `load_model.cpp`、`load_weights.cpp`——每一个名字
      都诚实地描述一个在世间行动的实体。

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

### 超限文件（R1 违规——第 9 步于 2026-04-21 完结）

下表 12 个文件均在第 9 步处理完毕，保留以便历史溯源。

| # | 文件 | 原行数 → 结果 | 解决提交 |
|---|------|--------------|----------|
| 1 | `src/sensus/auditus/audio_pipeline.cpp` | 2656 → 327 + 1574*¹ + 260 + 558*¹ | `172b264` |
| 2 | `src/machina/forward.cu` | 2179 → 639 + 247 + 339 + 336 + 431 + 320(.cuh) | `132f529` |
| 3 | `src/orator/wavlm_ecapa_encoder.cu` | 2091 → 448 + 416 + 361 + 238 + 743(.cuh) | `9cebc7e` |
| 4 | `src/machina/gptq.cu` | 2036 → 276 + 419 + 182 + 380 + 346 + 342 + 207 + 26(.cuh) | `6406777` |
| 5 | `src/machina/layer.cu` | 1960 → 552 + 495 + 436 + 555 | `cf71b11` |
| ~~6~~ | ~~`src/actus/cmd_test_ws.cpp`~~ | ~~1543~~ → **458** | **第 7 步**（`d96a503`..`628dd69`）|
| 7 | `src/sensus/auditus/mossformer2.cu` | 1551 → 35 + 590 + 518 + 425(.cuh) + 70(.h) | `0b3349c` |
| 8 | `src/orator/speaker_vector_store.cu` | 1411 → 498 + 684 + 276 | `856392d` |
| 9 | `src/sensus/auditus/frcrn_gpu.cu` | 1263 → 787 + 512 | `7ccd43a` |
| 10 | `src/machina/marlin.cu` | 1125 → 379 + 555(.cuh) + 238 | `5023a33` |
| 11 | `src/machina/gptq_gemm_v2.cu` | 1092 → 656 + 461 | `1fe0aa2` |
| 12 | `src/sensus/auditus/audio_pipeline.h` | 1068 → 481 + 81 + 365 + 199 | `5a4f295` |

*¹ 两个遗留（`audio_pipeline_process.cpp` 1574、`speaker_tracker_check.cpp`
558）均是单个不可再拆的方法——详见第 11 步。

### 第 11 步 —— 函数级分解（2026-04-21 开启）

第 9 步完成后，最基础的"拆文件"工作告一段落。当前仍有 3 个 TU 超出
.cpp 500 行上限，但每个只包含**恰好一个方法/函数**。进一步下降需要
深入函数体，将子步骤抽成私有 helper。这是下一轮精细手术。

| # | 文件 | 行数 | 唯一符号 | 预计子步骤 |
|---|------|------|----------|------------|
| A1 | `src/sensus/auditus/audio_pipeline_process.cpp` | 1574 | `AudioPipeline::process_loop` | VAD 状态分支 / SAAS 短段继承 / 重叠跟踪 / CAM++ 早/全提取 / 段内变更检测 / 段结束 + 谱聚类热身 / WL-ECAPA 原生通路 / SpeakerTracker 并行管道 / ASR 连续累积 / 尾部统计 |
| A2 | `src/sensus/auditus/speaker_tracker_check.cpp` | 558 | `SpeakerTracker::check` | 重叠检测 / MossFormer2 分离 / embedding 评分分支 |
| A3 | `src/orator/spectral_cluster.cpp` | 590 | `spectral_cluster()` | PCA / 余弦相似 / 时序混合 / p-剪枝 / 对称化 + 孤立点修复 / 归一化 Laplacian / 特征间隙 K 选择 / K-means++ / 时序平滑 / 中心合并 |

推荐执行顺序：**先 A3**（纯算法、无副作用；函数级抽取最干净的模板），
然后 A2，最后 A1（最大、跨子系统最多——等前二者把套路走熟再做）。

### 第 12 步 —— 其余外观/Actus 边界工作

- **第 8b+ 步**：`awaken.cpp` 内 126 行的 LLM + 意识流引导块
  横跨 machina + memoria + conscientia + persona —— 适合作为
  Actus 层平级 TU 候选（不是外观；子系统太多，不适合单子系统外观）。
- A1-A3 过程中若浮现新的 Nexus / Memoria / Persona / Orator
  反向渗透，则自然触发下一个外观。

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
