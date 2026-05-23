# 05 — Sensus（多模态感知）

感知塑造意识。你看到和听到的东西成为你是谁。

## 5.1 Auditus——听觉

```
[麦克风] → 环形缓冲 → VAD → 语音段 → ASR 编码 → 文本 token
                                                    ↓
                                         Prefill 输入队列
                                         （与内念合并）
```

- **连续感知模式**：VAD 控制注入节奏，仅在检测到有效语音时消耗
  Prefill 预算。
- **关键词触发的优先级提升**：唤醒词或名字检测立即提升唤醒等级和
  Decode 优先级。
- **说话人分离**：CAM++ 说话人嵌入 + 聚类以识别"谁在说话"。参考：
  qwen35-orin `speaker_encoder_gpu.cu` 的 GPU 实现；FunCineForge 的
  聚类策略。

## 5.2 Visus——视觉

```
[摄像头 / WS 视频] → 帧采样器 → ViT 编码器 → 视觉 token → Prefill 输入队列
```

- **双输入源**：本地 V4L2/GStreamer 摄像头捕获 AND 来自浏览器的
  WebSocket 视频上行（MediaStream API）。两者喂同一个帧采样器。
- Qwen3.5-27B 有原生视觉（27 层 ViT，patch 16，temporal_patch 2）。
- **帧采样**：自适应——场景变化检测或周期性间隔（如空闲 1–2 fps，
  运动/事件时爆发）。
- **视频理解**：Temporal patch 支持多帧推理。

## 5.3 Lectio——阅读

- 来自 WebUI 的 WebSocket 文本消息。
- 直接注入 Prefill 输入队列。

## 实现面

```
src/sensus/
├── auditus/
│   ├── asr_engine.{h,cpp}
│   ├── asr_encoder.{h,cu}
│   ├── asr_decoder.{h,cu}
│   ├── mel_gpu.{h,cu}
│   ├── vad.{h,cu}
│   └── audio_utils.{h,cpp}
├── visus/
│   ├── camera.{h,cpp}
│   └── frame_sampler.h
└── lectio/
    └── text_input.{h,cpp}
```

## 说话人识别——Orator

独立成模块，因为说话人身份是横切的：

```
src/orator/
├── speaker_encoder.{h,cu}
├── diarizer.{h,cpp}
└── speaker_db.h
```

90% 说话人归属验收标准见
`.github/instructions/benchmarks.instructions.md`——Orator 的输出质量
是一级验收门槛。

### Step 19 —— 短段救援与质心卫生（2026-05-22 / 23）

背景：在 4 说话人普通话语料上的 10 分钟同源回放评测长期卡在
coverage ≈ 0.25 / decided_macro ≈ 0.85。共存三条正交失败路径：

1. **混说话人 VAD 污染质心** —— 一个 VAD 跨越两位说话人时，FULL
   identify 路径会把混合 embedding 作为 exemplar 接纳到余弦胜出的
   簇上，长期蚕食可分性。
2. **短 VAD 无法回收** —— fbank ∈ [50, 150) 的段帧数过少，FULL
   提取路径放弃打标，悄无声息地汇入 `no_segment`。
3. **INHERIT-BROADCAST 级联放大** —— 一个短段或弃判段一旦从
   `prev_seg_speaker_id_` 继承到错误标签，链条会沿着随后约 10 个
   短段持续传播。

解决方案（`src/sensus/auditus/audio_pipeline_process_saas_full.cpp`）：

- **Step 19c —— MULTI-GATE。** FULL identify 之前，先在
  `seg_fbank_buf_` 上滑动 CAM++ 余弦探测（1.5 s 窗口、0.5 s 步长）。
  若相邻窗口最小余弦 < `speaker_multi_gate_threshold`（默认 0.58），
  判定为多说话人 VAD，识别路径切换到 `peek_best` —— 只读、不接纳
  exemplar、不自动注册、不做 EMA，retro-push 环也不写入。47 个长
  VAD 评测：AUC = 0.819；thr = 0.58 时 precision = 1.000 /
  recall = 0.375（对单说话人 VAD 零误判）。
- **Step 19d —— SHORT-IDENTIFY 并解耦级联。** 重新启用短段段
  （fbank ∈ [50, 150)），双编码器 peek_best，
  `speaker_short_identify_threshold = 0.40`、margin = 0.05。关键：
  SHORT-IDENTIFY 命中**不**更新 `prev_seg_speaker_id_`。短段标签只
  影响它自己；后续段必须自行命中或继承自真正的 FULL identify。
  这把原本二元的阈值地形变成单调：阈值越低，决策越多，0.40 之前
  dec_macro 几乎不动。

同源 10 分钟回放结果（MULTI-GATE 锁定）：

| 阶段 | coverage | decided_macro | 识别到的说话人 |
|------|---------:|--------------:|---------------:|
| 19c 前基线 | 0.253 | 0.854 | 3 / 4 |
| 仅 19c MULTI-GATE | 0.293 | 0.949 | 3 / 4 |
| **19c + 19d（已上线）** | **0.439 – 0.475** | **0.934 – 0.956** | **4 / 4** |

净改进：coverage **+18.6 pts**，decided_macro **+9.2 pts**。四位
说话人（包括此前完全不可见的石一）全部被识别，per-spk decided
准确率均 ≥ 0.88。

剩余 coverage 缺口已不在说话人识别本身：约 36% 的 GT 段没有对应的
runtime VAD（VAD 层瓶颈），约 18% 因 WL-ECAPA 硬要求 ≥ 1 s 真实语音
而弃判（短段 encoder 选型瓶颈，零填充会因 stat-pool 被静音主导而
造成 dec_macro 崩塌，已证伪）。后续工作归入 Vox/VAD 层与说话人
encoder 选型 RFC。

### Step 20 —— INHERIT-BROADCAST recency 扩窗（2026-05-23）

19d 收尾后对 74 个 `no_segment` GT 做分桶诊断：74 个全部与一个
runtime VAD 重叠，瓶颈是 inherit 分支中 `inh_id < 0` 拦截了广播。
137 个未配对 VAD 中，只有 47 个落在最近一次 FULL 事件的 2.0 s
（现行 recency）窗口内，84 个在 4.0 s 内。

修复仅一行常量：`audio_pipeline_process_saas_full.cpp` 中
`prev_full` recency 从 2.0 s 改到 4.0 s。SI hit 仍 **不** 刷新
`prev_seg`/`prev_full`（19d 解耦保留），级联抑制依靠 prev_full
更新点上已有的 `!multi_speaker_suspect` 过滤。窗口扫描
（10 分钟回放，19d SI 配置锁定）：

| 窗口 | coverage | decided_macro | n_no_seg | speakers |
|-----:|---------:|--------------:|---------:|---------:|
| 2.0 s (19d) | 0.439 | 0.934 | 74 | 4 |
| 3.0 s | 0.535 | 0.920 | 62 | 4 |
| **4.0 s（已上线）** | **0.571** | **0.921 – 0.929** | **52** | **4** |
| 5.0 s | 0.530 | 0.901 | 56 | 4 |

4.0 s 是单调最优峰；5.0 s 在两条指标上同时回归（标签过旧）。
两次独立运行均验证。

**pre-19c → 20 累积改进：** coverage **+31.8 pts**（0.253 → 0.571），
decided_macro **+6.7 pts**（0.854 → 0.921），四位说话人全程保留。
剩余 52 个 `no_segment` 已不在说话人子系统范围内（VAD 层 + 短段
encoder）。

### Step 21 —— SHORT-IDENTIFY → prev_full 时效刷新（旋钮，默认关闭，2026-05-23）

后续探索：一次强 SI 命中
（`peek.similarity >= speaker_si_refresh_prev_full_threshold`）是否
应该刷新 `prev_full_time_`，使后续短反馈段继承 SI 标注身份。配置旋钮
`speaker_si_refresh_prev_full_threshold`（默认 **0.0 = 关闭**）与环境
变量 `DEUSRIDET_SI_REFRESH_PREVFULL_THR` 覆盖均位于
`audio_pipeline_process_saas_full.cpp`。

tests/test.mp3 扫描（600 s，198 GT 段）：

| thr | n_decided | coverage | dec_macro |
|----:|---------:|---------:|---------:|
| 0.0（Step 20 基线） | 113 | 0.571 | 0.921 |
| 0.55（1 次） | 117 | 0.591 | 0.894 |
| 0.60 run 1 | 121 | 0.611 | 0.903 |
| 0.60 run 2（验证） | 110 | 0.556 | 0.899 |

同 fixture 同配置两次 thr=0.60 在 coverage 上为 0.611 vs 0.556
（Δ=0.055）。运行间方差超过提议信号本身，因此旋钮可配置但默认关闭，
等待更大规模 fixture 可以区分该效果再启用。流程教训：本 10 分钟
fixture 上的单次"改善"已不再作为决策依据。

### Step 22 —— Step 21 在 30 分钟 fixture 上提升为默认开启 @0.60（2026-05-23）

Step 21 "旋钮默认关闭"的结论是测量假象：
`tests/fixtures/test_ground_truth_v1.jsonl` 覆盖 `tests/test.mp3`
完整 60 分钟（1169 GT 段），但扫描脚本被 `--max-sec 600` 卡死，
只用了 17% 的可用证据。600 s 切片的 coverage 噪声（~0.055）
超过了拟议效应。

`tools/run_short_identify_sweep.sh` 扩展了可选 `max_sec` 参数；
1800 s 切片（571 GT 段）把噪声压缩约 √3，效应清晰：

| 配置 | 运行 | n_decided | cov | dec_macro | macro |
|------|----:|----------:|----:|----------:|------:|
| baseline (thr=0.0) | r1 | 307 | 0.5377 | 0.7898 | 0.4144 |
| baseline (thr=0.0) | r2 | 319 | 0.5587 | 0.8215 | 0.4506 |
| **thr=0.60** | r1 | 358 | 0.6270 | 0.7832 | 0.4632 |
| **thr=0.60** | r2 | 360 | 0.6305 | 0.7848 | 0.4704 |

同配置噪声：Δcov(baseline)=0.021，Δcov(thr=0.60)=0.0035。
跨配置差距：**Δcov=+0.081（~4 σ）**、Δmacro=+0.034、Δdec_macro=−0.022。
coverage 增益主导整体 macro 提升。上线：
`speaker_si_refresh_prev_full_threshold` 默认 0.0 → 0.60，
同步更新 `audio_pipeline.h` 与 `configs/auditus.conf`。1800 s harness
成为说话人侧消融的新参考 fixture。

**流程不变量：** "方差 > 信号"描述的是测量本身、不是改动 ——
宣判假设死亡之前，先验证 fixture 是否具备足够统计功效。
