# 05 — Sensus (Multimodal Perception)

Perception shapes consciousness. What you see and hear becomes who you are.

## 5.1 Auditus — Hearing

```
[Mic] → ring buffer → VAD → speech segment → ASR Encode → text tokens
                                                              ↓
                                                    Prefill input queue
                                                    (merged with internal thought)
```

- **Continuous perception mode**: VAD controls injection rhythm, consuming
  Prefill budget only when valid speech detected.
- **Keyword-triggered priority boost**: Wake word or name detection raises
  wakefulness and Decode priority immediately.
- **Speaker diarization**: CAM++ speaker embeddings + clustering to
  identify who is speaking. Reference: qwen35-orin
  `speaker_encoder_gpu.cu` for GPU implementation; FunCineForge for
  clustering strategy.

## 5.2 Visus — Seeing

```
[Camera / WS Video] → frame sampler → ViT encoder → vision tokens → Prefill input queue
```

- **Dual input sources**: Local V4L2/GStreamer camera capture AND WebSocket
  video upstream from browser (MediaStream API). Both feed the same frame
  sampler.
- Qwen3.5-27B has native vision (27-layer ViT, patch 16, temporal_patch 2).
- **Frame sampling**: Adaptive — scene change detection or periodic
  intervals (e.g. 1–2 fps idle, burst on motion/event).
- **Video understanding**: Temporal patches enable multi-frame reasoning.

## 5.3 Lectio — Reading

- WebSocket text messages from WebUI.
- Injected directly into Prefill input queue.

## Implementation Surface

```
src/sensus/
├── auditus/                        # Hearing
│   ├── asr_engine.{h,cpp}          # Qwen3-ASR forward pass orchestrator
│   ├── asr_encoder.{h,cu}          # Whisper-style audio encoder
│   ├── asr_decoder.{h,cu}          # Qwen3 text decoder
│   ├── mel_gpu.{h,cu}              # Mel-spectrogram kernels
│   ├── vad.{h,cu}                  # Voice Activity Detection
│   └── audio_utils.{h,cpp}         # ring buffer, resampling, PCM I/O
├── visus/                          # Sight
│   ├── camera.{h,cpp}              # V4L2 / GStreamer frame capture
│   └── frame_sampler.h             # adaptive frame sampling
└── lectio/                         # Reading
    └── text_input.{h,cpp}
```

## Speaker Identification — Orator

Split out into its own module since speaker identity is cross-cutting:

```
src/orator/
├── speaker_encoder.{h,cu}   # CAM++ / WavLM-ECAPA GPU implementation
├── diarizer.{h,cpp}         # clustering + assignment
└── speaker_db.h             # known speaker database
```

See `.github/instructions/benchmarks.instructions.md` for the 90%
speaker-attribution acceptance criterion — Orator's output quality is a
first-class acceptance gate.

### Step 19 — Short-segment rescue & centroid hygiene (2026-05-22 / 23)

Motivation: the 10-minute apples-to-apples replay on the 4-speaker
Mandarin corpus stalled at coverage ≈ 0.25 / decided_macro ≈ 0.85. Three
orthogonal failure modes co-existed:

1. **Multi-speaker VADs polluted centroids** — when a VAD bridged two
   speakers, the FULL identify path admitted a mixed-speaker exemplar
   onto whichever cluster won cosine, eroding discriminability.
2. **Short VADs were unrecoverable** — segments with fbank ∈ [50, 150)
   had too few frames for the FULL extraction path; they abstained
   silently and joined `no_segment` in the score.
3. **Cascade amplification via INHERIT-BROADCAST** — once a short or
   abstained VAD inherited a wrong label from `prev_seg_speaker_id_`,
   the chain propagated through ~10 subsequent short segments.

Resolution (`src/sensus/auditus/audio_pipeline_process_saas_full.cpp`):

- **Step 19c — MULTI-GATE.** Before any FULL identify, slide a CAM++
  cosine probe over `seg_fbank_buf_` (1.5 s window, 0.5 s hop). If the
  minimum adjacent-window cosine < `speaker_multi_gate_threshold`
  (default 0.58), the VAD is multi-speaker. We switch to `peek_best`
  for labeling — read-only, no exemplar admission, no auto-register,
  no EMA, and the retro-push ring entry is suppressed. Eval on 47
  long VADs: AUC = 0.819; at thr = 0.58, precision = 1.000 /
  recall = 0.375 (zero false positives on single-speaker VADs).
- **Step 19d — SHORT-IDENTIFY with cascade-decoupling.** Re-enable the
  short band (fbank ∈ [50, 150)) with the dual-encoder peek_best at
  `speaker_short_identify_threshold = 0.40`, margin = 0.05. Critically,
  SHORT-IDENTIFY hits **do not** update `prev_seg_speaker_id_`. A
  short-segment label applies only to its own segment; subsequent
  segments must earn their own hit or inherit from a real FULL
  identify. Threshold landscape, previously binary, becomes monotone:
  lower thr → more decisions, dec_macro flat until 0.40.

Result on the same 10-minute replay (MULTI-GATE locked):

| stage | coverage | decided_macro | speakers seen |
|-------|---------:|--------------:|--------------:|
| pre-19c baseline | 0.253 | 0.854 | 3 / 4 |
| 19c MULTI-GATE only | 0.293 | 0.949 | 3 / 4 |
| **19c + 19d (shipped)** | **0.439 – 0.475** | **0.934 – 0.956** | **4 / 4** |

Net delta: coverage **+18.6 pts**, decided_macro **+9.2 pts**. All four
speakers (including the previously-invisible 石一) are now identified
with per-speaker decided accuracy ≥ 0.88.

Remaining coverage gap is no longer in speaker ID — it sits in VAD
(~36 % of GT has no paired runtime VAD) and short-PCM encoder support
(WL-ECAPA hard-requires ≥ 1 s real speech; zero-padding silences the
stat-pool and was proven counter-productive). Future work is tracked
under the Vox/VAD layer and the speaker-encoder selection RFC.

### Step 20 — INHERIT-BROADCAST recency widening (2026-05-23)

Post-19d diagnostic of the 74 `no_segment` GTs revealed all 74 overlap
a runtime VAD; the bottleneck is `inh_id < 0` in the inherit branch.
Of 137 unpaired VADs, only 47 were within 2.0 s of a FULL event (the
existing recency window) while 84 were within 4.0 s.

The fix is a single constant in `audio_pipeline_process_saas_full.cpp`:
`prev_full` recency 2.0 → 4.0 s. SI hits still do not refresh
`prev_seg`/`prev_full` (19d decoupling preserved); cascade containment
relies on the existing `!multi_speaker_suspect` gate on prev_full
updates. Window-sweep (10-min replay, 19d-locked SI):

| window | coverage | decided_macro | n_no_seg | speakers |
|-------:|---------:|--------------:|---------:|---------:|
| 2.0 s (19d) | 0.439 | 0.934 | 74 | 4 |
| 3.0 s | 0.535 | 0.920 | 62 | 4 |
| **4.0 s (shipped)** | **0.571** | **0.921 – 0.929** | **52** | **4** |
| 5.0 s | 0.530 | 0.901 | 56 | 4 |

4.0 s is the monotone peak; 5.0 s regresses on both axes (staler
labels). Confirmed across two independent runs.

**Cumulative delta pre-19c → 20:** coverage **+31.8 pts**
(0.253 → 0.571), decided_macro **+6.7 pts** (0.854 → 0.921), all
four speakers retained. Remaining 52 `no_segment` cases sit outside
the speaker subsystem (VAD layer + short-PCM encoder).

### Step 21 — SHORT-IDENTIFY → prev_full recency refresh (knob, default-OFF, 2026-05-23)

Optional follow-up explored whether a strong SI hit
(`peek.similarity >= speaker_si_refresh_prev_full_threshold`) should
refresh `prev_full_time_` so subsequent short backchannels inherit
the SI-labelled identity. Config knob
`speaker_si_refresh_prev_full_threshold` (default **0.0 = disabled**)
plus env override `DEUSRIDET_SI_REFRESH_PREVFULL_THR` exists in
`audio_pipeline_process_saas_full.cpp`.

Sweep on tests/test.mp3 (600 s, 198 GT segs):

| thr | n_decided | coverage | dec_macro |
|----:|---------:|---------:|---------:|
| 0.0 (Step 20 baseline) | 113 | 0.571 | 0.921 |
| 0.55 (1 run) | 117 | 0.591 | 0.894 |
| 0.60 run 1 | 121 | 0.611 | 0.903 |
| 0.60 run 2 (verify) | 110 | 0.556 | 0.899 |

Two independent runs at the same thr=0.60 gave coverage 0.611 vs 0.556
(Δ=0.055) on the same fixture and config. Run-to-run variance exceeds
the proposed signal, so the knob is configurable but defaulted OFF
until a larger fixture can discriminate the effect. Process lesson:
single-run improvements on this 10-minute fixture are no longer
acceptable as decision evidence.

### Step 22 — Step 21 promoted to default-ON@0.60 on 30-min fixture (2026-05-23)

The Step 21 "knob defaults OFF" verdict was a measurement artefact:
`tests/fixtures/test_ground_truth_v1.jsonl` covers the full 60 minutes
of `tests/test.mp3` (1169 GT segs), but the sweep harness capped at
`--max-sec 600` and consumed only 17 % of the available evidence. The
600 s slice's coverage noise (~0.055) exceeded the proposed effect.

`tools/run_short_identify_sweep.sh` was extended with an optional
`max_sec` parameter; the 1800 s slice (571 GT segs) tightens noise
≈ √3 and resolves the effect cleanly:

| config | run | n_decided | cov | dec_macro | macro |
|--------|----:|----------:|----:|----------:|------:|
| baseline (thr=0.0) | r1 | 307 | 0.5377 | 0.7898 | 0.4144 |
| baseline (thr=0.0) | r2 | 319 | 0.5587 | 0.8215 | 0.4506 |
| **thr=0.60** | r1 | 358 | 0.6270 | 0.7832 | 0.4632 |
| **thr=0.60** | r2 | 360 | 0.6305 | 0.7848 | 0.4704 |

Within-config noise: Δcov(baseline)=0.021, Δcov(thr=0.60)=0.0035.
Between-config: **Δcov=+0.081 (~4 σ)**, Δmacro=+0.034, Δdec_macro=−0.022.
Coverage gain dominates; overall macro improves. Shipped:
`speaker_si_refresh_prev_full_threshold` default 0.0 → 0.60 in both
`audio_pipeline.h` and `configs/auditus.conf`. The 1800 s harness
becomes the new reference fixture for speaker-side ablations.

**Process invariant:** "variance > signal" describes the measurement,
not the change — verify fixture power before declaring a hypothesis
dead.
