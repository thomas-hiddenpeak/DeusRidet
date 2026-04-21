---
mode: agent
description: Create a facade layer for a subsystem (R3 boundary formalization).
---

# Module Facade

You are formalizing the coupling surface between a subsystem and its external
callers (Actus CLI, Nexus WS server, Conscientia stream). Per R3 in
`cpp.instructions.md`, external code must never wire directly to subsystem
internals.

## Design

Given a subsystem `src/<module>/`:

1. **List all current external coupling points** — grep for callbacks,
   `set_on_*`, direct member access, construction calls made from outside
   `src/<module>/`.
2. **Group them by coupling surface**:
   - Wiring to the WS server (broadcast events, binary streams)
   - Wiring to the consciousness stream (frame injection, state queries)
   - Wiring to the timeline logger (JSONL emission)
   - Wiring to other subsystems (peer-to-peer)
3. **Design the facade interface**:
   ```cpp
   // src/<module>/<module>_facade.h
   class ModuleFacade {
   public:
     explicit ModuleFacade(ModuleCore& core);

     void wire_to_ws(WsServer& server, TimelineLogger& log);
     void wire_to_consciousness(ConscientiaStream& stream);
     // ... one method per coupling surface
   };
   ```
4. **Keep `ModuleCore` (the real subsystem class) free of WS/Stream/Logger
   knowledge.** The facade is the translator; the core stays pure.

## Implement

1. Create `<module>_facade.{h,cpp}` with the philosophical anchor header.
2. Move the existing wiring code (currently in `commands.cpp` /
   `main.cpp` / wherever) into facade methods.
3. In the caller (e.g. `actus/test_ws.cpp`), replace the inline wiring with
   a single `facade.wire_to_ws(server, log)` call.
4. Build and run `/verify-change`.
5. Commit as `refactor(<module>): introduce facade, extract wiring from <caller>`.

## Do not

- Do not let the facade become a god-object. If it grows past 200 lines,
  the subsystem has too many coupling surfaces — that's a design smell, not
  a facade problem.
- Do not leak facade-only types into the core. The core should still be
  buildable and testable without the facade.
