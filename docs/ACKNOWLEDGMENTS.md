# Acknowledgments / 致谢

> *Standing on the shoulders of giants — and remembering to look up.*
> *站在巨人的肩膀上——且不忘仰望。*

---

## Reference Projects / 参考项目

### qwen35-thor

- **Author**: thomas-hiddenpeak
- **Repository**: https://github.com/thomas-hiddenpeak/qwen35-thor
- **License**: (check repository)
- **What was referenced**: C++/CUDA inference engine architecture for SM110a
  Blackwell, including forward pass structure, CUTLASS integration patterns,
  and kernel design approaches.
- **参考内容**: SM110a Blackwell 平台的 C++/CUDA 推理引擎架构，包括前向传播结构、
  CUTLASS 集成模式和内核设计思路。

### qwen35-orin

- **Author**: thomas-hiddenpeak
- **Location**: Local project (`~/qwen35-orin`)
- **What was referenced**: Complete inference engine for SM87 (Orin), including:
  SafetensorsLoader (zero-copy mmap), BPE Tokenizer, DeltaNet SSM layer
  implementation, GQA Full Attention, Paged Attention (Split-K), KV Cache
  Manager with SSD swap, ViT vision encoder, ASR pipeline (Whisper-style
  encoder + Qwen3 decoder), TTS pipeline (multi-codebook LM + codec
  tokenizer), Speaker identification (CAM++), VAD, WebSocket/HTTP server.
  Architecture ideas and implementation strategies adapted to fit DeusRidet's
  consciousness-centric design. No code copied verbatim.
- **参考内容**: SM87 (Orin) 平台的完整推理引擎，包括：SafetensorsLoader（零拷贝
  mmap）、BPE 分词器、DeltaNet SSM 层实现、GQA 全注意力、分页注意力（Split-K）、
  KV Cache 管理器（含 SSD 交换）、ViT 视觉编码器、ASR 流水线（Whisper 风格编码器
  + Qwen3 解码器）、TTS 流水线（多码本 LM + 编解码器分词器）、说话人识别
  （CAM++）、VAD、WebSocket/HTTP 服务器。架构思路和实现策略均根据 DeusRidet 的
  意识中心设计进行适配，未逐字复制代码。

### FunCineForge Speaker Diarization

- **Authors**: FunAudioLLM team
- **Repository**: https://github.com/FunAudioLLM/FunCineForge/tree/main/speaker_diarization
- **License**: (check repository)
- **What was referenced**: Multimodal speaker diarization approach using CAM++
  speaker embeddings with improved clustering strategy.
- **参考内容**: 使用 CAM++ 说话人嵌入和改进聚类策略的多模态说话人分离方法。

---

## Libraries / 依赖库

### NVIDIA CUTLASS

- **Repository**: https://github.com/NVIDIA/cutlass
- **License**: BSD-3-Clause
- **Usage**: GEMM/GEMV kernel templates for FP16/BF16 matrix operations.
- **用途**: FP16/BF16 矩阵运算的 GEMM/GEMV 内核模板。

### stb_image

- **Author**: Sean Barrett
- **Repository**: https://github.com/nothings/stb
- **License**: Public domain / MIT
- **Usage**: Single-header image loading for vision pipeline test inputs.
- **用途**: 单头文件图像加载，用于视觉流水线测试输入。

---

## Gratitude / 感谢

To Ridger Zhu — for the conversation that started it all: "The human brain
runs continuously at 20 watts. It is not a request-response machine."

致 Ridger Zhu——感谢那场开启一切的对话："人脑以 20 瓦功率持续运行，
它不是一个请求-响应机器。"

---

*This file is maintained as new references are incorporated. Every adaptation
is attributed both here and at the point of use in source code.*

*本文件随新参考内容的引入持续更新。每一处适配均在此处及源代码中的使用点进行标注。*
