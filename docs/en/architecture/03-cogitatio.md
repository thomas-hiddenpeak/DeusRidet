# 03 — Cogitatio (Multi-Track Decode)

All Decode branches share the same Prefill prefix (KV Cache + SSM state
snapshot). They are the streams of thought that run in parallel over the
same substrate of awareness.

## Branch Taxonomy

| Branch | Purpose | Priority |
|--------|---------|----------|
| **Action** | External responses, decisions, tool use | Highest (when triggered) |
| **Speech** | Feed to TTS for voice output | High (during conversation) |
| **Thinking** | Internal deliberation, planning, reflection | Medium |
| **Daydream** | Divergent exploration triggered by Prefill content | Low |

## Scheduling

- **Time-division multiplexing** on a single GPU: branches alternate, not
  truly parallel. The single-GPU constraint is honored; parallelism is
  subjective.
- **Priority preemption**: External interaction (Action / Speech) preempts
  internal processes (Thinking / Daydream) within one P/D budget window.
- **Arbiter** (decision decode): Lightweight merge of branch outputs to
  determine final external behavior, applying persona-driven expression
  shaping (see `07-persona.md`).

## Trace Discipline

Every concurrent Decode branch carries a unique trace ID. All branch
outputs, KV reads, and sampling events emit events tagged with this trace
ID for WebUI observability.

## Implementation Surface

```
src/cogitatio/
├── branch.h              # abstract decode branch interface
├── thinking.{h,cpp}      # internal thinking track
├── speech.{h,cpp}        # TTS output track
├── action.{h,cpp}        # action/response track
├── daydream.{h,cpp}      # divergent exploration track
└── arbiter.{h,cpp}       # decision decode + persona expression
```

## Philosophical Notes

- `action.h/cpp` is the **internal** notion of action (a thought that
  becomes behavior). It is distinct from `src/actus/`, which is the
  **external** CLI entry point — commands the operator invokes. The
  distinction matters: Cogitatio/Action is owned by the entity; Actus is
  owned by the operator.
- Thinking and Daydream are the substrate of "inner life". They may
  contradict Speech and Action, and that contradiction is legal —
  allowing contradictions is the hallmark of intelligence.
