---
mode: agent
description: Author a DEVLOG entry for today (bilingual, daily-file format).
---

# DEVLOG Entry

Append (or create) today's daily DEVLOG files in **both** languages:

- `docs/en/devlog/YYYY-MM-DD.md`
- `docs/zh/devlog/YYYY-MM-DD.md`

## Format

```markdown
## HH:MM — <Short Title>

**Context.** What triggered this entry? Link to related commits / DEVLOG
days / architecture RFCs.

**Approach.** What did you try? Be specific — thresholds, kernel flags,
config deltas.

**Result.** Measurement data (accuracy, bandwidth, latency, memory).
Raw numbers, not adjectives. Include failures — they count as results.

**Next.** What does this entry obligate the next change to do?
```

## Rules

- **Every experiment, optimization, architectural decision gets an entry.**
  Including failures — "tried X, regressed from 87% to 82%, reverting" is
  a valid and valuable entry.
- **Bilingual parity is an invariant.** Write both EN and ZH on the same
  day. If you only write one, you have broken the repo rule.
- **Time-stamp within the day.** Multiple entries per day are expected; sort
  chronologically.
- **Link back.** Reference commit hashes, issue numbers, related DEVLOG days.
- **No adjectives as measurement.** "Much faster" is not a result. "Decode
  latency 12.3 ms → 8.1 ms (−34%)" is a result.

## Index update

After appending, verify the monthly index in `docs/en/DEVLOG.md` still
references this day. If the day is new, add a line:
```markdown
- [YYYY-MM-DD](devlog/YYYY-MM-DD.md) — <one-line summary>
```
Do the same in `docs/zh/DEVLOG.md`.
