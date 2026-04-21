# 06 — Vox (TTS Output)

```
Arbiter/Speech Decode output → text → TTS model → codec tokens → Tokenizer decode → PCM
                                                                                      ↓
                                                                              WebSocket → WebUI
```

## Mechanism

- **Streaming**: First-packet latency ~97 ms thanks to Dual-Track
  architecture (see `11-machina.md` model spec).
- **Persona voice**: Instruct-controlled emotion, tone, and speaking rate
  aligned with outer persona expression (see `07-persona.md`).
- **Voice continuity**: Maintain consistent speaker identity across all
  outputs. A change of voice without narrative reason breaks the illusion
  of a continuous self.

## Implementation Surface

```
src/vox/
├── tts_engine.{h,cpp}      # Qwen3-TTS forward pass orchestrator
├── tts_model.{h,cu}        # multi-codebook LM forward
├── tts_tokenizer.{h,cu}    # 12Hz codec encoder/decoder
└── tts_speaker.h           # speaker identity management
```

## Model

Qwen3-TTS-12Hz-1.7B-CustomVoice + Qwen3-TTS-Tokenizer-12Hz.
Full model spec in `11-machina.md`.
