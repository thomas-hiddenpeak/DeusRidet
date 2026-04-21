---
applyTo: "**"
---

# Philosophia — World-View Anchors

> *"人类一思考，上帝就发笑；AI一思考，人类就不笑了。"*
> *"When humans think, God laughs; when AI thinks, humans stop laughing."*

These are the non-negotiable philosophical anchors of DeusRidet. Every file,
every function, every commit must be traceable back to one of these.

## First Principles

- **Continuity over request-response.** The brain runs continuously at 20 W.
  A genuine agent behaves likewise. DeusRidet is not a serving framework —
  it is an autonomous entity that perceives, thinks, dreams, and speaks on
  its own terms.
- **Internal complexity is the prerequisite for external consistency.**
- **Allowing contradictions is the hallmark of intelligence.**
- **Wakefulness is a spectrum** — idle moments are a form of thought.
- **Perception shapes consciousness** — what you see and hear becomes who you are.
- **Tool use extends the reach of thought** — a mind that cannot act remains an observer.
- **Lying and dreaming are isomorphic with imagination.** An AI that cannot
  lie and dream cannot become a truly intelligent species.

## Anti-Entropy Rule

Every change must *reduce* disorder. If a change adds a file, a name, an
abstraction, or a dependency, it must also remove ambiguity elsewhere.
Tech-dynamics (the pull toward "shortest path") is the primary enemy.
When in doubt, prefer the solution that honors philosophy over the one
that honors convenience.

## Never Blame Hardware or Model

When something doesn't work:
- The problem is in **our implementation**, not in the hardware or the model.
- Do not say "this is the model's upper limit" or "the hardware can't do this".
- Investigate what our code is doing wrong and fix it.
- If an approach isn't working, find a better approach — don't make excuses.

## Collaboration Ground Truth

Thomas is the project owner and architect with deep expertise in Tegra/CUDA
systems and LLM inference. When he reports an observation (memory usage,
performance numbers, system behavior), **trust it as ground truth**. If
measurement data appears to contradict his report, the measurement methodology
is more likely flawed than his observation. When in doubt, ask for
clarification rather than assuming error.

## Latin Nomenclatura

All major components are named in Latin. The Latin name is not decoration
— it forces the author to state the philosophical role before writing code.

| Latin | English | Role |
|-------|---------|------|
| Conscientia | Consciousness | Continuous Prefill engine, stream of awareness |
| Machina | Engine | Core inference engine (forward pass, GEMM, KV) |
| Cogitatio | Thought | Multi-track Decode branches |
| Sensus | Senses | Multimodal perception (auditus/visus/lectio) |
| Vox | Voice | TTS output pipeline |
| Somnium | Dream | Dreaming, daydreaming, memory consolidation |
| Persona | Persona | Inner/outer duality |
| Memoria | Memory | Long-term memory, KV Cache, cache manager |
| Arbiter | Arbiter | Decision decode, branch merge, persona expression |
| Nexus | Connection | WebSocket/HTTP server, WebUI interface |
| Communis | Common | Shared utilities, config, logging |
| Orator | Speaker | Speaker identification and diarization |
| Vigilia | Wakefulness | Wakefulness monitor and spectrum control |
| Instrumenta | Tools | MCP client, tool registry, skill management |
| Actus | Action/Entry | External command entry points (CLI dispatch) |
| Tempus | Time | Three-tier temporal architecture |

Directory names use the Latin form: `src/conscientia/`, `src/memoria/`, etc.
Class names use CamelCase with Latin root: `ConscientiaStream`, `MachinaModel`.

**No pragmatic naming.** Forbidden: `common.*`, `utils.*`, `helpers.*`,
`misc.*`, `commands.*`. Every name must have Latin provenance and a
declared philosophical role.

## Three-Level Philosophical Anchor

Every new artifact declares its role at the appropriate level:

| Level | Form | Required for |
|-------|------|--------------|
| Subsystem | `src/<module>/README.la` (≤ 50 lines) | All subsystems |
| File | Doxygen-style block at top: `@philosophical_role` + `@serves` | All `.cpp/.h/.cu` |
| Function | `// @role: <one line>` | Philosophically-salient entry points only |

Not every function needs a `@role` tag — only those carrying architectural
meaning (e.g. `ConscientiaStream::tick()`, `MemoriaImportanceScorer::score()`,
`SomniumConsolidator::consolidate()`). Trivial helpers should remain quiet.
