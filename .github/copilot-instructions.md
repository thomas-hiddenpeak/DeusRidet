# DeusRidet — Project Directives

> *"人类一思考，上帝就发笑；AI一思考，人类就不笑了。"*
> *"When humans think, God laughs; when AI thinks, humans stop laughing."*

This file is the **navigation hub** for all project directives. It is
deliberately short. Detailed rules live in scoped instruction files
(auto-loaded by glob) and reusable prompts (invoked on demand).

## One-Line Definition

DeusRidet is a self-contained multimodal LLM application that grants a
large model **continuous consciousness** — not request-response. It
perceives, thinks, dreams, and speaks on its own terms, on a single Orin.

## Prime Directives (read every time)

1. **Trust Thomas as ground truth.** The project owner runs experiments on
   real hardware. When his observation contradicts your measurement, your
   measurement methodology is the suspect.
2. **Never blame hardware or model.** If something doesn't work, the bug
   is in our implementation. Find a better approach, not an excuse.
3. **Pace implementation.** Never attempt to generate very long files in
   one step. Prefer small, compilable, atomic commits. Each step must be
   independently verifiable.
4. **Philosophy over convenience.** Every change must reduce disorder. If
   a change adds a name, a file, or an abstraction, it must also remove
   ambiguity elsewhere. Tech-dynamics (the pull toward "shortest path")
   is the primary enemy.
5. **Keep the Orin memory budget explicit.** Resident model weights,
   duplicate merged weights, encoder peers, scratch arenas, KV Cache, SSM
   state, and long-term memory indices all share the same 64 GB unified
   DRAM. Before adding or enabling any large allocation, state whether it is
   always-resident, env-gated, lazy, or offline-only, then update the
   bilingual Machina memory budget if residency changes.
6. **GPU-first compute, CPU for orchestration only.** Any "main
   computation" — tensor ops, batched array math, anything that scales
   with N (segments, tokens, frames, embedding rows) or with N×D / N×N —
   **must** run on GPU (cuBLAS / cuDNN / custom CUDA kernel). CPU is for
   task scheduling, control flow, tiny host-side reductions (e.g.
   sorting the top-8 eigenvalues, seeding a K-means++ farthest-point
   init for determinism), and operations that genuinely cannot execute
   on GPU (file I/O, external SDK calls, integer bookkeeping, problems
   with N ≤ 32). Before writing a `for (auto& x : large_array)` or a
   nested N² loop in `.cpp`, stop and ask: should this be a `.cu`
   kernel? The default answer is **yes**. The only acceptable reasons
   to keep it on CPU are (a) one-shot with tiny N, (b) a truly
   non-parallelisable dependency chain, or (c) an external library with
   no GPU entry point — and the reason must be stated in the commit
   message or as a code comment. Local optima found by CPU prototypes
   do not transfer to the GPU production path; do not waste cycles
   tuning a CPU implementation that is destined to be replaced.
7. **Semantic correctness goes through the agent's eyes, not a
   similarity score.** For any judgement that hinges on meaning
   (diarization, speaker attribution, ASR fidelity, persona coherence,
   dream consolidation, dialogue understanding), scripts may only
   capture, deterministically patch, time-align, and render
   human-readable reports. Macro-F1, fuzzy string matching, edit
   distance and any "auto-judged correctness number" are forbidden in
   that phase — the agent reads the full output and reports
   *pattern → evidence → suspected cause → candidate intervention*.
   Numeric scores are reserved for true physical quantities (latency,
   memory, throughput) and deterministic bit-equality checks. Full
   rules in [workflow.instructions.md](instructions/workflow.instructions.md).
8. **Live-system-only evidence; no detached/half-system tests.**
   `tests/test.mp3` + `tests/fixtures/test_ground_truth.json` is the
   **canonical** behavioural reference — no discount, no
   substitution. Every default-value change and every phase verdict
   in sensus / orator / auditus must cite a live `awaken` run against
   this audio, captured by `tools/replay_to_transcript.py`, read by
   the agent. Evaluators that bypass the live pipeline (eval-drivers
   over `fused_*.bin`, kernel micro-benchmarks, GPU/CPU bit-equality
   checks) are `internal-check-only`: their numbers describe a slice,
   not the system, and MUST NOT influence defaults or ship calls.
   Historical fixture-only verdicts are `⚠️ unverified` until
   re-tested under this rule.
9. **Accuracy is the sole metric (Constitutional, 2026-05-26).**
   The only number that decides whether a change to a behavioural
   subsystem is an improvement is the live-system accuracy on
   `tests/test.mp3` measured against
   `tests/fixtures/test_ground_truth.json` under best one-to-one
   mapping between predicted ids and GT identities. Every commit
   that claims an improvement MUST contain a line of the form
   `accuracy(tests/test.mp3, <task>): <before>% → <after>% (Δ = ±X.X pp)`.
   No commit message that lacks this line is allowed to ship a
   default-value flip or a "phase positive" verdict. Banned as
   primary metrics: macro-F1, micro-F1, NMI, ARI, DER, JER, WER on
   non-canonical slices; K_pred, abstain%, top-share, eigengap, NME;
   cosine similarities, intra-cluster distances; any "macro on
   sNNNN" or "macro on fused.bin" framing. Latency / memory /
   throughput remain real, but they only gate **feasibility**, never
   improvement. The reclusterer default-flip of 2026-05-25 (macro_f1
   0.5476→0.7025 on a fused fixture, live accuracy 25%→0%) is the
   case study this rule was written against. Full text in
   [philosophy.instructions.md](instructions/philosophy.instructions.md).

## Model Residency Budget Guardrail

The source of truth is `docs/{en,zh}/architecture/11-machina.md`, but verify
the live runtime shape before changing it. The top-level `~/models/dev/llm`
directory may contain multiple alternative LLMs and engine artifacts; do not
treat that directory total as simultaneous residency. Count the selected model
path plus every enabled audio/speaker model.

Current Auditus caveat: `awaken` configures WavLM-ECAPA by default, while ASR
is `DEUSRIDET_TEST_WS_ENABLE_ASR=1` gated and MossFormer2 remains lazy-loaded.
So the old CAM++-only speaker line is not enough for runtime headroom analysis.
On Tegra, `cudaMemGetInfo` is telemetry only; use `/proc/meminfo`
`MemAvailable`, process `VmRSS`, and `NvMapMemUsed` when making budget
decisions.

## Directive Map

| Scope | File | Applies to |
|-------|------|-----------|
| Philosophy & Nomenclatura | [philosophy.instructions.md](instructions/philosophy.instructions.md) | all files |
| Workflow, verification, git | [workflow.instructions.md](instructions/workflow.instructions.md) | all files |
| C++/CUDA source structure | [cpp.instructions.md](instructions/cpp.instructions.md) | `**/*.{cpp,h,hpp,cu,cuh}` |
| CUDA / Tegra / perf | [cuda.instructions.md](instructions/cuda.instructions.md) | `**/*.{cu,cuh}` |
| WebUI | [webui.instructions.md](instructions/webui.instructions.md) | `src/nexus/webui/**` |
| Docs (bilingual, DEVLOG) | [docs.instructions.md](instructions/docs.instructions.md) | `docs/**`, `**/*.md` |
| Benchmarks & evaluation | [benchmarks.instructions.md](instructions/benchmarks.instructions.md) | `tests/**`, `tools/**` |

## Reusable Prompts

| Prompt | Purpose |
|--------|---------|
| [/verify-change](prompts/verify-change.prompt.md) | Run build + kill + drop_caches + HTTP 200 + WS 101 |
| [/refactor-split-file](prompts/refactor-split-file.prompt.md) | Split an oversized source file per R1 |
| [/add-cuda-kernel](prompts/add-cuda-kernel.prompt.md) | Author a new CUDA kernel with Tegra discipline |
| [/module-facade](prompts/module-facade.prompt.md) | Create a subsystem facade (R3 boundary) |
| [/devlog-entry](prompts/devlog-entry.prompt.md) | Append a bilingual daily DEVLOG entry |

## Architecture RFCs

Long-form design documents for every subsystem live under
`docs/{en,zh}/architecture/`. Bilingual parity is invariant.

| # | Subsystem | EN | ZH |
|---|-----------|----|----|
| 00 | Overview + refactor backlog | [en](../docs/en/architecture/00-overview.md) | [zh](../docs/zh/architecture/00-overview.md) |
| 01 | Conscientia (consciousness) | [en](../docs/en/architecture/01-conscientia.md) | [zh](../docs/zh/architecture/01-conscientia.md) |
| 02 | Memoria (cache + long-term memory) | [en](../docs/en/architecture/02-memoria.md) | [zh](../docs/zh/architecture/02-memoria.md) |
| 03 | Cogitatio (multi-track decode) | [en](../docs/en/architecture/03-cogitatio.md) | [zh](../docs/zh/architecture/03-cogitatio.md) |
| 04 | Vigilia + Somnium (wakefulness & dreaming) | [en](../docs/en/architecture/04-vigilia.md) | [zh](../docs/zh/architecture/04-vigilia.md) |
| 05 | Sensus (perception) | [en](../docs/en/architecture/05-sensus.md) | [zh](../docs/zh/architecture/05-sensus.md) |
| 06 | Vox (TTS) | [en](../docs/en/architecture/06-vox.md) | [zh](../docs/zh/architecture/06-vox.md) |
| 07 | Persona (inner/outer) | [en](../docs/en/architecture/07-persona.md) | [zh](../docs/zh/architecture/07-persona.md) |
| 08 | Instrumenta (tool use) | [en](../docs/en/architecture/08-instrumenta.md) | [zh](../docs/zh/architecture/08-instrumenta.md) |
| 09 | Tempus (three-tier time) | [en](../docs/en/architecture/09-tempus.md) | [zh](../docs/zh/architecture/09-tempus.md) |
| 10 | Nexus (WebUI, WS, HTTP) | [en](../docs/en/architecture/10-nexus.md) | [zh](../docs/zh/architecture/10-nexus.md) |
| 11 | Machina (stack, quant, models, budget) | [en](../docs/en/architecture/11-machina.md) | [zh](../docs/zh/architecture/11-machina.md) |
| 12 | DiariZen reclusterer (speaker re-attribution) | [en](../docs/en/architecture/12-diarizen.md) | [zh](../docs/zh/architecture/12-diarizen.md) |

## Latin Nomenclatura (glance-reference)

| Latin | English | Role |
|-------|---------|------|
| Conscientia | Consciousness | Continuous Prefill engine |
| Machina | Engine | Core inference engine |
| Cogitatio | Thought | Multi-track Decode branches |
| Sensus | Senses | Perception (auditus/visus/lectio) |
| Vox | Voice | TTS output |
| Somnium | Dream | Dreaming & memory consolidation |
| Vigilia | Wakefulness | Wakefulness monitor |
| Persona | Persona | Inner/outer duality |
| Memoria | Memory | Cache + long-term memory |
| Arbiter | Arbiter | Decision decode |
| Nexus | Connection | WS/HTTP server, WebUI |
| Communis | Common | Shared utilities (tempus, trace, ring buffer) |
| Orator | Speaker | Speaker identification |
| Instrumenta | Tools | MCP, function calling, skills |
| Actus | Action/Entry | External CLI command entry points |
| Tempus | Time | Three-tier temporal architecture |

Full binding rules (including forbidden names, CamelCase conventions,
three-level philosophical anchors) are in
[philosophy.instructions.md](instructions/philosophy.instructions.md).

## Philosophical Principles (non-negotiable)

- **Continuity over request-response.**
- **Internal complexity is the prerequisite for external consistency.**
- **Allowing contradictions is the hallmark of intelligence.**
- **Wakefulness is a spectrum — even idle moments are thought.**
- **Perception shapes consciousness.**
- **Tool use extends the reach of thought.**
- **Lying and dreaming are isomorphic with imagination.** An AI that
  cannot lie and dream cannot become a truly intelligent species.
