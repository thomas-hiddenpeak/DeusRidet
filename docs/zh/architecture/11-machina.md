# 11 — Machina（技术栈、量化、模型、内存预算）

## 技术栈

| 层 | 技术 |
|----|------|
| 语言 | **纯 C++17 / CUDA** —— 运行时零 Python 依赖 |
| 构建 | CMake 3.24+, nvcc (SM87), C++17 + CUDA 12 |
| GEMM | CUTLASS (submodule) + cuBLAS 运行时选择 |
| 权重格式 | safetensors（零拷贝 mmap）|
| 量化 | GPTQ-Int4（dequant + GEMV/GEMM）、FP16（cuBLAS/CUTLASS）|
| 存储 | SSD KV Cache via mmap，NVMe 卸载 |
| 内部通信 | 引擎线程间无锁环形缓冲 + eventfd |
| 外部 API | WebSocket（实时）+ HTTP（配置/查询）|
| 前端 | WebUI（HTML/JS），WS 连接，与后端完全解耦 |
| 第三方 | CUTLASS、stb_image、uWebSockets（或自研 WS）|

> **无 Python，无 gRPC，无 ZMQ**。所有组件间通信通过共享内存或无锁
> 队列进程内完成。网络协议仅用于 WebUI 外部接口。

## 单设备上的 P/D 分离

在单颗 Orin 上，P/D 分离是**逻辑的，而非物理的**：

- Prefill 和 Decode 在同一进程中通过独立 CUDA Stream 运行。
- 共享 KV Cache 池，零拷贝交接（Prefill 写 → Decode 读）。
- 抽象 `PrefillNode` / `DecodeNode` 接口以备未来多设备扩展。
- 通过唤醒驱动的时间预算分配器调度（见 `04-vigilia.md`）。

## 量化核函数需求

| 格式 | Decode (B=1~few) | Prefill (B≥17) | 状态 |
|------|-------------------|-----------------|------|
| GPTQ-Int4 | Dequant + GEMV (group_size=128, symmetric) | Dequant + GEMM | **P0——必建** |
| FP16 (BF16) | cuBLAS GEMV | CUTLASS/cuBLAS GEMM | **P1——参考实现**|
| INT8 | Dequant + GEMV | Dequant + GEMM | **P2——未来** |

GPTQ 核函数是全新工作——两个参考项目都不支持 GPTQ。

## 模型

所有模型使用 safetensors 格式和零拷贝 mmap 加载。下列模型用于初始
开发和测试阶段——引擎架构与模型无关。

### LLM —— Qwen3.5-27B-GPTQ-Int4（主测试模型）

| 属性 | 值 |
|------|-----|
| 路径 | `~/models/dev/llm/Qwen3.5-27b-GPTQ-Int4` |
| 架构 | DeltaNet SSM (48 层) + GQA Full Attention (16 层) = 64 层 |
| 隐藏维度 | 5120 |
| KV 头 | 4 (GQA), head_dim 256 |
| SSM | linear_key_heads 16, linear_value_heads 48, conv_kernel 4 |
| MTP | 1 层草稿（投机解码）|
| 量化 | GPTQ: bits=4, group_size=128, sym=true, desc_act=false |
| 未量化 | lm_head, embed_tokens, 所有 attn 层, shared_expert, mtp, visual |
| 视觉 | ViT 27 层 (1152 hidden, patch 16, spatial_merge 2, temporal_patch 2) |
| 模态 | 文本 + 图像 + 视频 |
| 上下文 | 262 144 tokens 最大 |
| 权重大小 | ~30.2 GB 磁盘 |

未来模型目标：Qwen3.5-9B BF16、Qwen3.5-9B INT8。

### ASR —— Qwen3-ASR-1.7B

| 属性 | 值 |
|------|-----|
| 路径 | `~/models/dev/asr/Qwen/Qwen3-ASR-1.7B` |
| 音频编码器 | Whisper 风格，24 Transformer 层，d_model=1024，16 头 |
| 文本解码器 | Qwen3，28 层，hidden_size=2048，16 头，8 KV 头 |
| 音频输入 | 128 mel bins, hop=160, n_fft=400, 16 kHz 采样率 |
| 语言 | 30 语言 + 22 汉语方言 |
| 权重大小 | ~4.7 GB (BF16) |

### TTS —— Qwen3-TTS-12Hz-1.7B-CustomVoice

| 属性 | 值 |
|------|-----|
| 路径 | `~/models/dev/tts/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| 架构 | 离散多码本 LM（Qwen3-TTS-Tokenizer-12Hz codec）|
| 流式 | Dual-Track 混合，首包延迟 ~97 ms |
| 语音 | 9 预设说话人，instruct 控制情感/语调 |
| 语言 | 10 (zh, en, ja, ko, de, fr, ru, pt, es, it) |
| 权重大小 | ~4.3 GB (BF16) |

还需 **Qwen3-TTS-Tokenizer-12Hz** codec。
路径：`~/models/dev/tts/Qwen/Qwen3-TTS-Tokenizer-12Hz`。

## 内存预算（64 GB Orin）

**所有生产模型都按常驻内存预算**。按需换入模型权重不可接受——它
引入与连续意识不兼容的延迟。benchmark-only gate 可以在某次运行中
关闭某个模型，但不能把它从生产预算里删除。

| 组件 | 估计 | 常驻性 |
|------|------|--------|
| LLM 权重 | ~30.2 GB | 生产常驻 |
| ASR 权重 | ~4.7 GB | 生产常驻；benchmark 可 env-gate |
| TTS 模型 + tokenizer | ~5.5 GB | 生产常驻 |
| 说话人编码器 (CAM++ + WavLM-ECAPA) | ~6.5 GB | dual speaker 开启时常驻 |
| VAD / enhancement sidecars (Silero, pyannote, FRCRN) | ~0.1 GB | 常驻 |
| MossFormer2 separator | ~0.2 GB | lazy-loaded；非启动常驻 |
| **常驻权重总计，不含 lazy separator** | **~47.0 GB** | |
| OS + CUDA 运行时开销 | ~3.5 GB | |
| **KV Cache + 激活可用** | **~13-14 GB** | |

> **关键约束**：旧的 ~20 GB headroom 只适用于 CAM++-only speaker stack。
> 当前 dual speaker 路径默认配置 WavLM-ECAPA，生产 headroom 会降到约
> 13-14 GB。这部分必须覆盖 KV Cache（paged 块）、SSM 递归态、Conv 态、
> 激活暂存空间、长期记忆索引（HNSW 顶层 ~200 MB，图热集 ~100 MB）、
> ASR/TTS 中间缓冲，以及任何 shadow probe。每次分配都必须有账。

检查磁盘占用时，只计算被选中的模型路径，不要把整个
`~/models/dev/llm` 目录当成运行时常驻：该目录可能包含多个互斥的 LLM
候选和 engine artifacts。在 Tegra 上，`cudaMemGetInfo` 只能作为 telemetry；
预算决策使用 `/proc/meminfo` 的 `MemAvailable`、进程 `VmRSS` 和
`NvMapMemUsed`。

## 实现面

```
src/machina/
├── model.{h,cpp}
├── gptq.{h,cu}
├── gemm.{h,cu}
├── layer.{h,cu}
├── paged_attention.{h,cu}
├── tokenizer.{h,cpp}
├── safetensors.{h,cpp}
├── vision.{h,cu}
├── sampling.{h,cu}
└── allocator.{h,cpp}
```

## 参考项目

| 项目 | 角色 |
|------|------|
| [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor) | C++/CUDA 推理引擎参考（SM110a Blackwell）|
| qwen35-orin (`~/qwen35-orin`) | C++/CUDA 引擎 + ASR/TTS 插件（SM87 Orin）|
| [FunCineForge speaker_diarization](https://github.com/FunAudioLLM/FunCineForge/tree/main/speaker_diarization) | 说话人分离（CAM++ + 聚类）|

仅作参考——不可照搬。归属规则见
`.github/instructions/docs.instructions.md`。
