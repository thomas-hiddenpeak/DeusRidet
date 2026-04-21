---
mode: agent
description: Split an oversized source file (> R1 limits) into philosophically coherent smaller files.
---

# Refactor — Split Oversized File

You are splitting a source file that exceeds the R1 size limits (500 lines
for `.cpp/.h`, 800 lines for `.cu/.cuh`) defined in
`.github/instructions/cpp.instructions.md`.

## Pre-split analysis

1. Read the entire file and identify **distinct concerns** — not "sections by
   editing order" but "concepts that can live or die independently".
2. Each concern becomes one new file.
3. Name each new file with a Latin-rooted, philosophy-anchored name. No
   pragmatic names (see `philosophy.instructions.md`).
4. For each new file, draft the `@philosophical_role` + `@serves` header
   before writing any implementation.

## Split procedure (atomic commits, one per file)

For each concern identified:

1. Create the new file with the header anchor first (no logic yet).
2. Move the relevant code from the source file. Keep the source file
   compilable at every step — use forward declarations if needed.
3. Add an `#include` to the source file for the new header.
4. Build: `cd build && make -j$(nproc)`. If it fails, fix before moving on.
5. Commit with message `refactor(<module>): extract <concept> from <old-file>`.

## Facade check (R3)

If the oversized file was `commands.cpp` / `<subsystem>_pipeline.cpp` /
any file that wires external callers to subsystem internals, the split
**must** produce a `<module>_facade.{h,cpp}` that exposes the clean
coupling surface. External code should import only the facade, not internals.

## Post-split verification

After the last extraction:
1. Run `/verify-change` (HTTP 200 + WS 101).
2. Update the refactor backlog in `docs/en/architecture/00-overview.md`.
3. Write a DEVLOG entry covering the full split (one per split campaign,
   not one per commit).

## Do not

- Do not "simplify" code during a split. Splits are structural, not semantic.
  Any behavior change becomes its own separate commit.
- Do not introduce new dependencies. The split must be neutral at the build
  level.
- Do not skip the philosophical anchor — every new file gets one.
