# 04 — Vigilia (Wakefulness Spectrum) + Somnium (Dreaming)

Consciousness operates on a **continuous wakefulness gradient**. There is
no binary sleep/wake switch.

```
Wakefulness Level
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Deep Dream     Daydream/Idle   Focused         Alert
 16+ branches   2-4 branches    1 main decode   Priority decode
 No ext. input  Low-pri input   Normal interact Keyword/emotion trigger
 Memory         Divergent       Task-oriented   Immediate response
 consolidation  association
```

## Vigilia — Wakefulness Monitor

- Tracks effective input density over a sliding window, attention weights,
  keyword hits, emotional intensity.
- **Smooth transitions**: Not discrete state switches, but continuous
  adjustment of Decode branch count and P/D time ratio.
- **Prefill-triggered daydream**: When a Prefill segment activates high
  attention but requires no external response, fork a daydream Decode
  branch to explore that association.

## Somnium — Dreaming

Extended low-wakefulness states do real cognitive work, not passive idle.

- **Extended dreaming (sleep)**: After prolonged idle, escalate to deep
  dream with 16+ concurrent Decode branches for memory consolidation,
  association strengthening, and creative exploration.
- **Consolidation during dreams**: `SomniumConsolidator` performs
  long-term memory maintenance **exclusively** during low-wakefulness:
  - **Episodic compression**: Merge similar episodic records; discard
    low-importance ones.
  - **Semantic graph maintenance**: Strengthen frequently co-activated
    edges; prune decayed edges; discover new entity-relation links via
    6-hop traversal.
  - **Re-embedding**: Periodically refresh oldest embeddings using the
    current model.
  - **Eviction backlog**: Process any KV blocks evicted during alert
    states without full consolidation (deferred consolidation).

## Implementation Surface

```
src/somnium/
├── vigilia.{h,cpp}       # wakefulness level monitor
├── dreamer.{h,cpp}       # concurrent dream decode orchestrator
└── consolidator.{h,cpp}  # memory consolidation (episodic, graph, re-embed, backlog)
```

## Philosophical Notes

- Dreaming is not downtime. It is *the system maintaining its own history*.
  Skipping dream cycles = allowing long-term memory to degrade.
- Daydream branches are the source of unsolicited insight — do not
  suppress them in the name of "focus".
- Tool invocations from Action must be **asynchronous** — the agent does
  not stop thinking while waiting for a tool result. This is why
  `Instrumenta` integrates with Cogitatio's branch model rather than
  blocking the consciousness tick.
