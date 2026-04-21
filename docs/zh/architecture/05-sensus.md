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
