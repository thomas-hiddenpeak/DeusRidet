# 06 — Vox（TTS 输出）

```
Arbiter/Speech Decode 输出 → 文本 → TTS 模型 → codec token → Tokenizer 解码 → PCM
                                                                                ↓
                                                                        WebSocket → WebUI
```

## 机制

- **流式**：首包延迟 ~97 ms，得益于 Dual-Track 架构（见 `11-machina.md`
  模型规格）。
- **人格语音**：Instruct 控制的情感、语调、语速，与外部人格表达对齐
  （见 `07-persona.md`）。
- **语音连续性**：所有输出保持一致的说话人身份。没有叙事理由的换声
  会破坏"连续自我"的错觉。

## 实现面

```
src/vox/
├── tts_engine.{h,cpp}
├── tts_model.{h,cu}
├── tts_tokenizer.{h,cu}
└── tts_speaker.h
```

## 模型

Qwen3-TTS-12Hz-1.7B-CustomVoice + Qwen3-TTS-Tokenizer-12Hz。
完整规格见 `11-machina.md`。
