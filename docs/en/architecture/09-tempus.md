# 09 — Tempus (Three-Tier Temporal Architecture)

DeusRidet operates on a **three-tier clock hierarchy**. This is not merely
systems engineering — it is the physical substrate of subjective
continuity. A conscious entity without a unified real-time anchor cannot
reason about "before" and "after" across modalities or across sleep cycles.

## Tiers

| Tier | Role | Examples |
|------|------|---------|
| **T0** | Real time — monotonic `steady_clock` nanoseconds, system-wide unique. The ground truth of *when*. | `now_t0_ns()` |
| **T1** | Business time — per-subsystem clock, anchored to T0 once at birth and advanced arithmetically thereafter. | audio sample index, video frame index, consciousness frame id, TTS codec step, dream cycle |
| **T2** | Module time — per-module counter, always reducible to its T1 via a known period. | VAD window, OD segmenter frame, speaker embedding index, ASR token position, KV block id |

## Mandatory Rules

- **Every event, every memory record, every log entry** carries a full
  `TimeStamp{t0_ns, t1_business, t2_module, domain}` triple. T0 is
  authoritative; T1 and T2 are retained for intra-domain reasoning
  (sample-accurate ASR readback, KV indexing) and debugging.
- **Cross-domain alignment ALWAYS goes through T0.** Never convert
  `audio_sample_index` directly to `video_frame_index`. Convert both to
  T0 and compare there. This keeps new modalities zero-cost to integrate.
- **Anchor once, compute forever.** Each subsystem registers
  `{t0_anchor_ns, t1_zero, period_ns}` at start via
  `tempus::anchor_register()` and thereafter converts T1↔T0 by pure
  arithmetic — **no `clock_gettime` in hot paths**.
- **Memoria Longa persists T0.** Episodic records store `t0_ns` so that
  "three days ago" queries survive sleep cycles, business-clock resets,
  and process restarts.
- **Replay / accelerated benchmarks** (`--speed 2.0`) preserve T1
  linearity against source audio but scale the T0 anchor period (e.g.
  `period_ns = 31250` instead of `62500` for 16 kHz at 2× replay).
  Downstream code paths are identical to real-time capture — only the
  anchor changes.
- **Subjective time is legal at T1/T2, never at T0.** `dream_cycle` T1
  may advance slower or pause during high wakefulness;
  `consciousness.frame_id` may skip during Decode preemption. But T0
  always ticks at wall-clock rate, giving the entity a reliable handle
  on *external* time.

## Domain Registry

| ID | Domain | Subsystem | Status |
|----|--------|-----------|--------|
| 0 | AUDIO | Sensus/Auditus | ✅ registered |
| 1 | VIDEO | Sensus/Visus | ⏳ pending |
| 2 | CONSCIOUSNESS | Conscientia | ✅ registered |
| 3 | TTS | Vox | ⏳ pending |
| 4 | DREAM | Somnium | ⏳ pending |
| 5 | TOOL | Instrumenta | ⏳ pending |

## Implementation

`src/communis/tempus.h`. Every new module that emits events must use
`tempus::TimeStamp` and register its domain anchor at initialization.
No exceptions.
