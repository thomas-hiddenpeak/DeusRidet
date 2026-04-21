# 00 — Overview

DeusRidet is a self-contained multimodal LLM application built on a
Disaggregated Prefill-Decode (P/D) architecture. It grants large language
models continuous consciousness, dreaming capability, multimodal perception
(vision + hearing), and a dual inner/outer persona.

Born from a philosophical discussion with Ridger Zhu: the human brain runs
continuously at 20 W — it is not a request-response machine. A genuine
agent should behave likewise.

This is **not** a serving framework. It is a complete, autonomous entity
that perceives, thinks, dreams, and speaks on its own terms.

## License

DeusRidet is released under **GPLv3**. Any project that uses, modifies, or
incorporates DeusRidet code must also release its source code under a
compatible open-source license. Consciousness should not be locked behind
closed doors.

A `LICENSE` file must be present at the repository root.

## Table of Contents

| # | File | Topic |
|---|------|-------|
| 00 | this file | Project overview, license, refactor backlog |
| 01 | [01-conscientia.md](01-conscientia.md) | Consciousness stream (continuous Prefill engine) |
| 02 | [02-memoria.md](02-memoria.md) | Cache Manager, Memoria Longa, continuous eviction |
| 03 | [03-cogitatio.md](03-cogitatio.md) | Multi-track Decode branches |
| 04 | [04-vigilia.md](04-vigilia.md) | Wakefulness spectrum |
| 05 | [05-sensus.md](05-sensus.md) | Multimodal perception (auditus / visus / lectio) |
| 06 | [06-vox.md](06-vox.md) | TTS output pipeline |
| 07 | [07-persona.md](07-persona.md) | Inner / outer persona duality |
| 08 | [08-instrumenta.md](08-instrumenta.md) | Tool use (MCP, function calling, skills) |
| 09 | [09-tempus.md](09-tempus.md) | Three-tier temporal architecture |
| 10 | [10-nexus.md](10-nexus.md) | WebUI & observability |
| 11 | [11-machina.md](11-machina.md) | Technical stack, quantization, models, memory budget |

## Refactor Backlog

Files currently exceeding R1 size limits (see `cpp.instructions.md`).
Each must be split by the `/refactor-split-file` prompt. Order below is
the recommended execution order.

| # | File | Lines | Proposed split |
|---|------|-------|----------------|
| 1 | `src/commands.cpp` | 2768 | → `src/actus/{test_ws,bench_prefill,test_wavlm_cnn,...}.cpp` + `src/actus/dispatcher.cpp` |
| 2 | `src/sensus/auditus/audio_pipeline.cpp` | 2651 | → `pipeline_core.cpp`, `vad_orchestrator.cpp`, `speaker_matcher.cpp`, `asr_trigger.cpp` + `auditus_facade.{h,cpp}` |
| 3 | `src/machina/forward.cu` | 2172 | → per-op kernels (attention/mlp/norm/residual launchers) |
| 4 | `src/orator/wavlm_ecapa_encoder.cu` | 2084 | → `wavlm_encoder.cu` + `ecapa_encoder.cu` + shared utils header |
| 5 | `src/machina/gptq.cu` | 2029 | → `gptq_gemv.cu` + `gptq_gemm.cu` + `gptq_dequant.cu` |
| 6 | `src/machina/layer.cu` | 1953 | → `ssm_layer.cu` + `attn_layer.cu` + `mlp_layer.cu` |
| 7 | `src/sensus/auditus/mossformer2.cu` | 1544 | → encoder/decoder split by block |
| 8 | `src/orator/speaker_vector_store.cu` | 1404 | → index + kernels + I/O split |
| 9 | `src/sensus/auditus/frcrn_gpu.cu` | 1256 | → frcrn_encoder + frcrn_decoder |
| 10 | `src/machina/marlin.cu` | 1118 | borderline; audit for single-kernel justification |
| 11 | `src/machina/gptq_gemm_v2.cu` | 1085 | merge into gptq_gemm.cu under #5 |
| 12 | `src/sensus/auditus/audio_pipeline.h` | 1063 | split per the .cpp split |

Track progress by striking through rows as each split lands.

## Foundational Principles

See `.github/instructions/philosophy.instructions.md` for the binding
list. Key anchors reproduced here for readers outside the agent context:

- Continuity over request-response
- Internal complexity is the prerequisite for external consistency
- Allowing contradictions is the hallmark of intelligence
- Wakefulness is a spectrum — even idle moments are thought
- Perception shapes consciousness
- Tool use extends the reach of thought
- Lying and dreaming are isomorphic with imagination
