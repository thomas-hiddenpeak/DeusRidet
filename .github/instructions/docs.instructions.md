---
applyTo: "docs/**,**/*.md"
---

# Documentation — Bilingual Parity & Archival Rules

## Bilingual Parity (invariant)

Development documentation (architecture, internals, experiments, APIs) and
user-facing documentation (setup, usage, configuration, tutorials) are
maintained in **both** `docs/en/` and `docs/zh/`. Both versions must be
kept in sync on every update — same number of entries, same dates,
no drift.

README is **trilingual**:
- `README.md` — Latin (default, repo root)
- `docs/zh/README.md` — Chinese
- `docs/en/README.md` — English

In-code comments, commit messages, and issue/PR descriptions are English only.

## DEVLOG Rules

- **Format**: `## YYYY-MM-DD — Title` with context, approach, result, metrics.
- **Daily granularity**: `docs/en/devlog/YYYY-MM-DD.md` — one file per active
  day. The top-level `docs/en/DEVLOG.md` is an index linking to daily files,
  organized reverse-chronologically by month.
- **Every experiment, optimization attempt, and architectural decision gets
  an entry** — including failures. Recording what didn't work is as valuable
  as recording what did.
- **Bilingual parity is mandatory** — `docs/zh/devlog/YYYY-MM-DD.md` tracks
  the same days. Do not allow divergence.

## Architecture Docs

Long-form architecture RFCs live under `docs/{en,zh}/architecture/`.
Each file ≤ 400 lines, one philosophical concept per file. See
`docs/en/architecture/00-overview.md` for the table of contents.

## Archival

Superseded planning docs move to `docs/{en,zh}/archive/` with a header block:

```markdown
> **Archived YYYY-MM-DD.** Status: implemented / abandoned / superseded.
> Reason: <one paragraph>. Replacement: <link to current doc or commit>.
```

Never delete — archive. History must be retrievable.

## Attribution & Acknowledgments

When code, architecture ideas, algorithms, or implementation strategies are
adapted from external projects:

1. **Inline comment attribution** at the point of use:
   ```cpp
   // Adapted from <project> (<file>): <brief description>
   // Original: <URL or path>
   ```
2. **`docs/ACKNOWLEDGMENTS.md`** — centralized record of all referenced
   projects, their licenses, what was adapted, and gratitude notes.
   Bilingual (English + Chinese in one file).
3. **No verbatim copying.** Adapt ideas. If a substantial portion mirrors
   the source, attribution must be a prominent block comment, not inline.
4. **License compliance** — verify permissions, record license type.

## File-Size Budget

- Single `.md` file ≤ 400 lines except `ACKNOWLEDGMENTS.md` and auto-generated files.
- `DEVLOG.md` (index) ≤ 200 lines — daily files carry the detail.

## No Premature Markdown

Do not create new markdown files to document code changes unless explicitly
requested by the user or mandated by the rules above (DEVLOG entry,
architecture RFC, facade README.la).
