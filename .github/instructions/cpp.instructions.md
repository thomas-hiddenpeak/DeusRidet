---
applyTo: "**/*.{cpp,h,hpp,cu,cuh}"
---

# C++ / CUDA Source — Structural Rules

## Language Baseline

- **Pure C++17 / CUDA.** Zero Python dependency at runtime.
- **Comments in English only.**
- **No Python, no gRPC, no ZMQ.** Internal communication via lock-free ring
  buffers + `eventfd`. Network only for WebUI (WS + HTTP).

## R1 — File-Size Limits

| Kind | Hard limit |
|------|------------|
| `.cpp` / `.h` / `.hpp` | **500 lines** |
| `.cu` / `.cuh` | **800 lines** |

Exceeding the limit triggers mandatory split. See `/refactor-split-file` prompt.

## R2 — Directory = Philosophy

Each top-level `src/` directory corresponds to one Latin module (see
`philosophy.instructions.md`). No utility-dump directories. No mixed-concern
directories.

## R3 — Subsystem Facade

External callers (Actus CLI dispatch, Nexus WS server) **must not** directly
wire callbacks to subsystem internals. Each subsystem provides a facade:

```
src/<module>/<module>_facade.{h,cpp}
    ↳ class ModuleFacade {
        void wire_to_ws(WsServer&, TimelineLogger&);
        void wire_to_consciousness(ConscientiaStream&);
        ...
      }
```

The facade is the *philosophical boundary* of the subsystem — if you're
crossing it, you're declaring a coupling. Make it explicit.

## R4 — Three-Level Philosophical Anchor

| Level | Form |
|-------|------|
| Subsystem | `src/<module>/README.la` ≤ 50 lines |
| File | Doxygen block at top with `@philosophical_role` + `@serves` |
| Key function | `// @role: <one-line>` on architecturally-salient entry points |

### File-level template

```cpp
/**
 * @file memoria/importance_scorer.cu
 * @philosophical_role
 *   Attention-score-based KV block importance tracking — the substrate
 *   of "forgetting with a trace". Blocks consistently ignored become
 *   eviction candidates.
 * @serves
 *   - Continuous Eviction Model (see docs/en/architecture/02-memoria.md)
 *   - SomniumConsolidator hook (dream-state memory maintenance)
 */
```

### Function-level template

```cpp
// @role: consciousness frame tick — merges sensory inputs with internal
//        thought output, advances T1 consciousness clock by one frame.
void ConscientiaStream::tick();
```

Trivial helpers (`static inline size_t round_up(size_t x, size_t a)`) do NOT
require a `@role` tag. Only architecturally-salient entry points do.

## R7 — Naming Discipline

Forbidden filenames: `common.*`, `utils.*`, `helpers.*`, `misc.*`,
`commands.*`, `stuff.*`, `core.*` (too generic).

Every name must be a specific Latin word with a declared philosophical role
(see nomenclatura table in `philosophy.instructions.md`).

Class names use CamelCase with Latin root:
`ConscientiaStream`, `MemoriaRetriever`, `SomniumDreamer`, `VigiliaMonitor`.

## Observability First

Every internal process must be inspectable from the WebUI. Every Decode
branch carries a unique trace ID. Every event/memory record/log entry
carries a `tempus::TimeStamp` (see `docs/en/architecture/09-tempus.md`).

## Error Handling

Validate at system boundaries only. No defensive coding in hot inner loops.
