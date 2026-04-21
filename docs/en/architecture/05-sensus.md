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
