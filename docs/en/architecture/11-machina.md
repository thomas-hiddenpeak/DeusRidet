# 11 — Machina (Technical Stack, Quantization, Models, Memory Budget)

## Technical Stack

| Layer | Technology |
|-------|------------|
| Language | **Pure C++17 / CUDA** — zero Python dependency |
| Build | CMake 3.24+, nvcc (SM87), C++17 + CUDA 12 |
| GEMM | CUTLASS (submodule) + cuBLAS for runtime selection |
| Weight format | safetensors (zero-copy mmap) |
| Quantization | GPTQ-Int4 (dequant + GEMV/GEMM), FP16 (cuBLAS/CUTLASS) |
| Storage | SSD-based KV Cache via mmap, NVMe offload |
| Communication (internal) | lock-free ring buffers + eventfd between engine threads |
| External API | WebSocket (real-time) + HTTP (config/query) |
| Frontend | WebUI (HTML/JS), WS-connected, fully decoupled from backend |
| Third-party | CUTLASS, stb_image, uWebSockets (or self-contained WS impl) |

> **No Python, no gRPC, no ZMQ.** All inter-component communication is
> in-process via shared memory or lock-free queues. Network protocols only
> for the WebUI external interface.

## P/D Separation on Single Device

On a single Orin, P/D separation is **logical, not physical**:

- Prefill and Decode run in the same process via independent CUDA Streams.
- Shared KV Cache pool with zero-copy handoff (Prefill writes → Decode reads).
- Abstract `PrefillNode` / `DecodeNode` interfaces for future multi-device
  extension.
- Scheduling via wakefulness-driven time-budget allocator
  (see `04-vigilia.md`).

## Quantization Kernel Requirements

| Format | Decode (B=1~few) | Prefill (B≥17) | Status |
|--------|-------------------|-----------------|--------|
| GPTQ-Int4 | Dequant + GEMV (group_size=128, symmetric) | Dequant + GEMM | **P0 — must build** |
| FP16 (BF16) | cuBLAS GEMV | CUTLASS/cuBLAS GEMM | **P1 — from reference** |
| INT8 | Dequant + GEMV | Dequant + GEMM | **P2 — future** |

The GPTQ kernel is entirely new work — neither reference project supports GPTQ.

## Models

All models use safetensors format with zero-copy mmap loading. The models
listed below are for the initial development and testing phase — the
engine architecture is model-agnostic.

### LLM — Qwen3.5-27B-GPTQ-Int4 (Primary Test Model)

| Property | Value |
|----------|-------|
| Path | `~/models/dev/llm/Qwen3.5-27b-GPTQ-Int4` |
| Architecture | DeltaNet SSM (48 layers) + GQA Full Attention (16 layers) = 64 layers |
| Hidden size | 5120 |
| KV heads | 4 (GQA), head_dim 256 |
| SSM | linear_key_heads 16, linear_value_heads 48, conv_kernel 4 |
| MTP | 1 draft layer (speculative decoding) |
| Quantization | GPTQ: bits=4, group_size=128, sym=true, desc_act=false |
| NOT quantized | lm_head, embed_tokens, all attn layers, shared_expert, mtp, visual |
| Vision | ViT 27-layer (1152 hidden, patch 16, spatial_merge 2, temporal_patch 2) |
| Modalities | Text + Image + Video |
| Context | 262 144 tokens max |
| Weight size | ~30.2 GB on disk |

Future model targets: Qwen3.5-9B BF16, Qwen3.5-9B INT8.

### ASR — Qwen3-ASR-1.7B

| Property | Value |
|----------|-------|
| Path | `~/models/dev/asr/Qwen/Qwen3-ASR-1.7B` |
| Audio encoder | Whisper-style, 24 Transformer layers, d_model=1024, 16 heads |
| Text decoder | Qwen3, 28 layers, hidden_size=2048, 16 heads, 8 KV heads |
| Audio input | 128 mel bins, hop=160, n_fft=400, 16 kHz sample rate |
| Languages | 30 languages + 22 Chinese dialects |
| Weight size | ~4.7 GB (BF16) |

### TTS — Qwen3-TTS-12Hz-1.7B-CustomVoice

| Property | Value |
|----------|-------|
| Path | `~/models/dev/tts/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Architecture | Discrete multi-codebook LM (Qwen3-TTS-Tokenizer-12Hz codec) |
| Streaming | Dual-Track hybrid, first-packet latency ~97 ms |
| Voices | 9 preset speakers, instruct-controlled emotion/tone |
| Languages | 10 (zh, en, ja, ko, de, fr, ru, pt, es, it) |
| Weight size | ~4.3 GB (BF16) |

Also requires **Qwen3-TTS-Tokenizer-12Hz** codec.
Path: `~/models/dev/tts/Qwen/Qwen3-TTS-Tokenizer-12Hz`.

## Memory Budget (64 GB Orin)

**All three models (LLM + ASR + TTS) are memory-resident at all times.**
Swapping model weights on demand is not acceptable — it introduces latency
incompatible with continuous consciousness.

| Component | Estimate |
|-----------|----------|
| LLM weights | ~30.2 GB |
| ASR weights | ~4.7 GB |
| TTS model + tokenizer | ~5.5 GB |
| Speaker encoder (CAM++) | ~0.1 GB |
| **Total weights** | **~40.5 GB** |
| OS + CUDA runtime overhead | ~3.5 GB |
| **Available for KV Cache + activations** | **~20 GB** |

> **Critical constraint**: ~20 GB must cover KV Cache (paged blocks), SSM
> recurrent states, Conv states, activation scratch space, long-term
> memory indices (HNSW top layer ~200 MB, graph hot set ~100 MB), and all
> intermediate buffers for ASR/TTS inference. Every allocation must be
> accounted for.

## Implementation Surface

```
src/machina/
├── model.{h,cpp}         # Qwen3.5 forward pass (SSM + GQA hybrid)
├── gptq.{h,cu}           # GPTQ-Int4 dequant kernels (GEMV + GEMM)
├── gemm.{h,cu}           # FP16/BF16 GEMM dispatch (CUTLASS + cuBLAS)
├── layer.{h,cu}          # DeltaNet SSM + Full Attention layers
├── paged_attention.{h,cu}# paged KV Cache + Split-K decode attention
├── tokenizer.{h,cpp}     # BPE tokenizer
├── safetensors.{h,cpp}   # zero-copy weight loader
├── vision.{h,cu}         # ViT encoder for image/video
├── sampling.{h,cu}       # GPU sampling (Gumbel-Max, top_k/p, penalties)
└── allocator.{h,cpp}     # GPU memory pool
```

## Reference Projects

| Project | Role |
|---------|------|
| [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor) | C++/CUDA inference engine reference (SM110a Blackwell) |
| qwen35-orin (`~/qwen35-orin`) | C++/CUDA engine + ASR/TTS plugins (SM87 Orin) |
| [FunCineForge speaker_diarization](https://github.com/FunAudioLLM/FunCineForge/tree/main/speaker_diarization) | Speaker diarization (CAM++ + clustering) |

References only — do not copy verbatim. Attribution rules in
`.github/instructions/docs.instructions.md`.
