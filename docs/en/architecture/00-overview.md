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

### Progress log (2026-04-21)

- [x] **Step 1 — Housekeeping**: stale benchmark artifacts pruned from
      `tests/`; only `test.mp3` + `test.txt` remain as the evaluation
      baseline (commit `32763b6`).
- [x] **Step 2 — DEVLOG daily archival**: `docs/{en,zh}/DEVLOG.md` split
      into `docs/{en,zh}/devlog/YYYY-MM-DD.md` daily files; top-level
      DEVLOG.md is now a 24-line reverse-chronological index (commit
      `6f37fbc`). `PLAN_AUDIO_ENHANCEMENT.md` archived to
      `docs/{en,zh}/archive/` with superseded-by header.
- [x] **Step 3 — PDD restructure**: monolithic `.github/copilot-instructions.md`
      (927 lines) split into navigation hub + scoped instructions +
      prompts + architecture RFCs per GitHub Prompt-Driven Development
      convention (commit `8e6c052`).
- [x] **Step 4 — Actus rename**: `src/commands.{h,cpp}` renamed to
      `src/actus/actus.{h,cpp}` with philosophical anchors (commit
      `0eea0e6`).
- [x] **Step 5 — Actus split**: `src/actus/actus.cpp` (2768 lines) split
      into 14 per-command translation units + a 100-line registry file
      (commit `2525450`). All under R1 except `cmd_test_ws.cpp`
      (1543 lines), which is the natural Auditus-facade target.
- [x] **Step 6 — Facade evaluation**: see inventory below (commit `851df90`).
- [x] **Step 7 — Auditus facade + Actus routing** (2026-04-21):
      `cmd_test_ws.cpp` brought from 1543 → 458 lines (under R1).
      - 7a (`d96a503`) `auditus_facade.{h,cpp}` introduced; vad /
        asr_partial / drop callbacks migrated.
      - 7b (`dbd9f9e`) transcript / asr_log / stats / speaker migrated.
      - 7c (`cd3224d`) WS binary PCM ingress migrated. Scope narrowed:
        connect/disconnect stayed behind as they broadcast Conscientia
        state, not Auditus events.
      - 7d (`7e1c9ef`) 545-line `set_on_text` command router extracted
        to peer Actus TU `cmd_test_ws_router.{h,cpp}` — the router
        bridges Auditus + Conscientia + Persona, which is precisely
        the Actus charter, so it sits beside `cmd_test_ws.cpp` rather
        than inside any single subsystem facade.
      - 7e (`628dd69`) 91-line `set_on_connect` hello envelope
        extracted to `cmd_test_ws_hello.{h,cpp}`.
- [ ] **Step 8+ — Remaining subsystem facades**: Nexus, Memoria, Persona,
      Orator where `cmd_test_ws` currently reaches into private headers.
- [ ] **Step 9 — CUDA/audio R1 split campaign**: the 11 remaining
      oversized files in the table below.

### Step 6 — Facade evaluation (coupling inventory)

Scan of each `src/actus/cmd_*.cpp` for subsystem-type usage (not mere
header inclusion):

| Command | Lines | External types actually used |
|---------|-------|------------------------------|
| `cmd_test_ws` | 1543 | AudioPipeline, WsServer, CacheManager, FRCRN, MossFormer, Persona, Orator speaker store, TimelineLogger |
| `cmd_test_wavlm_cnn` | 262 | Orator speaker store |
| `cmd_test_gptq` | 252 | GPTQ (machina public) |
| `cmd_bench_gptq` | 90 | GPTQ (machina public) |
| All other `cmd_*` | ≤ 174 | machina public API only (model / forward / allocator) |

**Finding.** The coupling debt is almost entirely concentrated in
`cmd_test_ws`, the long-running consciousness server. All other Actus
entry points already respect the public-API boundary of their target
subsystem. Therefore the facade campaign has a clear, single starting
point: Auditus. Extracting the WS wiring into `auditus_facade.{h,cpp}`
should bring `cmd_test_ws.cpp` under R1 and simultaneously establish the
template for subsequent Nexus / Memoria / Orator / Persona facades.

### Oversized files (R1 violations — Actus resolved 2026-04-21)

| # | File | Lines | Proposed split |
|---|------|-------|----------------|
| 1 | `src/sensus/auditus/audio_pipeline.cpp` | 2651 | → `pipeline_core.cpp`, `vad_orchestrator.cpp`, `speaker_matcher.cpp`, `asr_trigger.cpp` |
| 2 | `src/machina/forward.cu` | 2172 | → per-op kernels (attention/mlp/norm/residual launchers) |
| 3 | `src/orator/wavlm_ecapa_encoder.cu` | 2084 | → `wavlm_encoder.cu` + `ecapa_encoder.cu` + shared utils header |
| 4 | `src/machina/gptq.cu` | 2029 | → `gptq_gemv.cu` + `gptq_gemm.cu` + `gptq_dequant.cu` |
| 5 | `src/machina/layer.cu` | 1953 | → `ssm_layer.cu` + `attn_layer.cu` + `mlp_layer.cu` |
| ~~6~~ | ~~`src/actus/cmd_test_ws.cpp`~~ | ~~1543~~ → **458** | **Resolved Step 7 (2026-04-21)**: router + hello + auditus_facade |
| 7 | `src/sensus/auditus/mossformer2.cu` | 1544 | → encoder/decoder split by block |
| 8 | `src/orator/speaker_vector_store.cu` | 1404 | → index + kernels + I/O split |
| 9 | `src/sensus/auditus/frcrn_gpu.cu` | 1256 | → frcrn_encoder + frcrn_decoder |
| 10 | `src/machina/marlin.cu` | 1118 | borderline; audit for single-kernel justification |
| 11 | `src/machina/gptq_gemm_v2.cu` | 1085 | merge into gptq_gemm.cu under #4 |
| 12 | `src/sensus/auditus/audio_pipeline.h` | 1063 | split per the .cpp split |

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
