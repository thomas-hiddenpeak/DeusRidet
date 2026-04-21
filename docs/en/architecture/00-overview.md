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
- [x] **Step 8a — Conscientia facade** (2026-04-22, commit `b4ddea6`):
      three consciousness→WS broadcast lambdas (decode / speech_token /
      state, 83 inline lines) extracted to
      `src/conscientia/conscientia_facade.{h,cpp}` mirroring the
      `auditus_facade` installer pattern. Shared JSON helpers promoted
      to `src/communis/json_util.h`; `auditus_facade.h` now re-exports
      `communis::{sanitize_utf8,json_escape}` via `using`, so every
      existing call site compiles unchanged. `cmd_test_ws.cpp` 458 → 392.
- [ ] **Step 8b+ — Remaining subsystem facades**: any further Nexus /
      Memoria / Persona / Orator reach-in that surfaces; the 126-line
      LLM-and-consciousness bootstrap block in `awaken.cpp` is the
      next Actus-layer extraction candidate (crosses machina + memoria
      + conscientia + persona, so a peer Actus TU, not a facade).
- [x] **Step 9 — CUDA/audio R1 split campaign** (2026-04-21, 20
      commits `57ecd1a` → `172b264`): all 12 oversized files in the
      table below resolved. 20 atomic splits in sequence; every split
      independently verified with `cmake + make` and the awaken ritual
      (HTTP=200 WS=101). Campaign summary:
      - Headers / .cpp: `auditus_facade.cpp` 525→130+421,
        `awaken_router.cpp` 577→437+185, `ws_server.cpp` 607→359+277,
        `asr_engine.cpp` 611→488+149, `spectral_cluster.h` 626→66+590
        (body demoted to peer .cpp TU), `tokenizer.cpp` 665→443+268,
        `stream.cpp` 836→464+400, `model.cpp` 982→227+426+381,
        `audio_pipeline.h` 1068→481+81+365+199, `audio_pipeline.cpp`
        2656→327+1574+260+558.
      - .cu / .cuh: `asr_ops.cu` 898→535+378, `gptq_gemm_v2.cu`
        1092→656+461, `marlin.cu` 1125→379+555(.cuh)+238, `frcrn_gpu.cu`
        1263→787+512, `speaker_vector_store.cu` 1411→498+684+276,
        `mossformer2.cu` 1551→35+590+518+425(.cuh)+70(.h), `layer.cu`
        1960→552+495+436+555, `gptq.cu`
        2036→276+419+182+380+346+342+207+26(.cuh), `wavlm_ecapa_encoder.cu`
        2091→448+416+361+238+743(.cuh), `forward.cu`
        2179→639+247+339+336+431+320(.cuh).
      - Technique notes: shared kernels in sibling `.cuh` declared
        `static __global__` (RDC off → each peer TU gets its own
        instantiation); `sensus` / `orator` / `machina` use
        `GLOB_RECURSE` so no CMake edits needed per split.
      - Residuals: three single-method TUs remain above the .cpp 500-line
        cap after the campaign — they are not further splittable at the
        file level. Step 11 tracks function-level decomposition.
- [x] **Step 10 — Actus charter restoration** (2026-04-23, commits
      `d5fffd8`, `95ac9d3`, `728e39e`, `f530573`, `887a32e`):
      diagnosed naming regression — every TU under `src/actus/` carried
      the `cmd_` prefix and many were not Actus verbs at all (engine
      probes, kernel-timing benches, integration tests). Five atomic
      sub-commits restored the charter:
      - 10b (`d5fffd8`) `bench_*` → `tools/` as standalone executables
        (developer instruments measuring the engine, not entities
        acting in the world).
      - 10c (`95ac9d3`) `profile_*` → `tools/` (same reasoning).
      - 10d (`728e39e`) six `cmd_test_*` engine probes →
        `tests/integration/` as standalone executables linking only the
        libraries they need; bodies preserved verbatim.
      - 10e (`f530573`) hard switch `test-ws` → `awaken` (no CLI alias):
        the principal Actus verb (the act that brings the entity online)
        was hidden under a developer-flavoured label. Files, symbols,
        log tags, and the canonical verification ritual all updated.
      - 10f (`887a32e`) drop `cmd_` prefix from the last two surviving
        Actus verbs: `cmd_load_model` → `load_model`,
        `cmd_load_weights` → `load_weights`. CLI verbs unchanged.
      Result: `src/actus/` now contains exactly six TUs — `actus.{h,cpp}`,
      `awaken.cpp`, `awaken_router.{h,cpp}`, `awaken_hello.{h,cpp}`,
      `load_model.cpp`, `load_weights.cpp` — and every name honestly
      describes an entity acting in the world.

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

### Oversized files (R1 violations — Step 9 resolved 2026-04-21)

All 12 originally-oversized files below were resolved during Step 9.
Kept for historical traceability.

| # | File | Original → Result | Resolving commit |
|---|------|-------------------|------------------|
| 1 | `src/sensus/auditus/audio_pipeline.cpp` | 2656 → 327 + 1574*¹ + 260 + 558*¹ | `172b264` |
| 2 | `src/machina/forward.cu` | 2179 → 639 + 247 + 339 + 336 + 431 + 320(.cuh) | `132f529` |
| 3 | `src/orator/wavlm_ecapa_encoder.cu` | 2091 → 448 + 416 + 361 + 238 + 743(.cuh) | `9cebc7e` |
| 4 | `src/machina/gptq.cu` | 2036 → 276 + 419 + 182 + 380 + 346 + 342 + 207 + 26(.cuh) | `6406777` |
| 5 | `src/machina/layer.cu` | 1960 → 552 + 495 + 436 + 555 | `cf71b11` |
| ~~6~~ | ~~`src/actus/cmd_test_ws.cpp`~~ | ~~1543~~ → **458** | **Step 7** (`d96a503`..`628dd69`) |
| 7 | `src/sensus/auditus/mossformer2.cu` | 1551 → 35 + 590 + 518 + 425(.cuh) + 70(.h) | `0b3349c` |
| 8 | `src/orator/speaker_vector_store.cu` | 1411 → 498 + 684 + 276 | `856392d` |
| 9 | `src/sensus/auditus/frcrn_gpu.cu` | 1263 → 787 + 512 | `7ccd43a` |
| 10 | `src/machina/marlin.cu` | 1125 → 379 + 555(.cuh) + 238 | `5023a33` |
| 11 | `src/machina/gptq_gemm_v2.cu` | 1092 → 656 + 461 | `1fe0aa2` |
| 12 | `src/sensus/auditus/audio_pipeline.h` | 1068 → 481 + 81 + 365 + 199 | `5a4f295` |

*¹ Two residuals (`audio_pipeline_process.cpp` 1574, `speaker_tracker_check.cpp`
558) are single non-decomposable methods — see Step 11.

### Step 11 — Function-level decomposition (2026-04-21, **closed**)

After Step 9, the campaign's primary-school work — breaking monolithic
source files apart — is done. Three TUs remained above the .cpp 500-line
cap, but each contained **exactly one method**. Further reduction required
reaching into the method body and extracting sub-steps into private
helpers. This pass is now complete.

| # | File | Before | After (orchestrator + peer TUs) | Commits |
|---|------|--------|----------------------------------|---------|
| A3 | `src/orator/spectral_cluster.cpp` | 590 | 85 orchestrator + 636 stages + 119 header | `c77efde` |
| A2 | `src/sensus/auditus/speaker_tracker_check.cpp` | 558 | 177 orchestrator + 500 stages (+8 private decls in `speaker_tracking.h`) | `991f543` |
| A1 | `src/sensus/auditus/audio_pipeline_process.cpp` | 1574 | 353 orchestrator + 392 ASR + 456 SAAS-full + 321 SAAS-during + 190 SAAS-segend | `df93e8b` / `02011b3` / `a34b4a9` |

A1 was decomposed in three atomic commits (A1a ASR pipeline → A1b CAM++
FULL extract + spectral warm-up → A1c during-speech + segment-end), each
independently built and awaken-verified (HTTP=200 WS=101). The
`process_loop` body is now a sequential orchestrator of eight phases:
gain/RMS → FRCRN → Silero VAD → FSMN VAD → SAAS three-branch dispatch
(onset / during / segment-end) → SpeakerTracker → ASR → Mel/VAD/stats.

All three former single-method residuals are now below the R1 500-line
.cpp cap; every stage is independently trace-taggable.

### Step 12 — Outstanding facade / Actus-boundary work

- **Step 8b+**: 126-line LLM + Conscientia bootstrap block inside
  `awaken.cpp` crosses machina + memoria + conscientia + persona — an
  Actus-layer peer TU candidate (not a facade; already crosses too many
  subsystems for a single-subsystem facade).
- Any further Nexus / Memoria / Persona / Orator reach-in surfaced
  during A1-A3 becomes the organic next facade.

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
