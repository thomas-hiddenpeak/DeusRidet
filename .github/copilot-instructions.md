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
