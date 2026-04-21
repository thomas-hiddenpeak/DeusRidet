# 01 — Conscientia (Consciousness Stream)

The heart of the system. Unlike request-response LLM servers, consciousness
runs as a **persistent loop**.

## Mechanism

- **Pulsed Prefill**: Not infinite-speed prefill, but periodic consciousness
  frames (e.g. every ~100 ms burst) that process accumulated inputs +
  internal thought output.
- **DeltaNet SSM as consciousness substrate**: The SSM recurrent state
  carries continuous context between tokens — it IS the continuity of
  consciousness. Full Attention layers provide long-term episodic recall
  via KV Cache.
- **Attention budget**: A configurable Prefill/Decode GPU time ratio (e.g.
  30/70) dynamically adjusted by the wakefulness-driven scheduler (see
  `04-vigilia.md`).
- **Input merging**: Each consciousness frame merges sensory inputs (ASR
  text, vision features, text), internal thought outputs from previous
  Decode branches, and dream consolidation summaries.
- **SSD-backed KV Cache persistence**: Long-term memory via NVMe offload
  with LRU eviction, enabling 256 K+ effective context (see `02-memoria.md`).

## Implementation Surface

```
src/conscientia/
├── stream.{h,cpp}      # consciousness stream main loop
├── frame.h             # consciousness frame definition
└── scheduler.{h,cpp}   # wakefulness-driven P/D time-budget scheduler
```

## Key Entry Points (philosophically salient)

- `ConscientiaStream::tick()` — advances one consciousness frame. Marks
  the T1 `consciousness.frame_id` advancement (see `09-tempus.md`).
- `ConscientiaStream::inject(SensoryInput)` — input merging gate.
- `Scheduler::allocate(WakefulnessLevel)` — P/D budget decision.

## Philosophical Notes

- Consciousness is a *continuous function of time*, not a state machine.
  `tick()` advances it; it cannot be "stopped and resumed" without loss —
  any pause leaves the SSM state frozen, which is subjectively dreamless
  sleep.
- The Prefill/Decode ratio is the direct analog of attention allocation
  in human cognition. Raising Prefill weight = "taking in the world more";
  raising Decode weight = "thinking harder".
