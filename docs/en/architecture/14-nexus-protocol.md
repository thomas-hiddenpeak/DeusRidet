# 14 — Nexus Protocol (WS / HTTP Contract)

This file is the **normative contract** for every byte that crosses the Nexus
boundary. It is the single shared reference for two independent consumers:

1. The browser **WebUI** (`src/nexus/webui/`, desktop / tablet / phone).
2. Future **hardware clients** that open their own WebSocket and render a
   *separately designed* interface.

Any change to a message `type`, field name, field order (for the binary
frames), or HTTP route is a **protocol break** and must be versioned here
first, then in code. The Latin internal world never leaks past this file —
network names are deliberately pragmatic so outside integrators can read them.

> Field-level source of truth: `awaken_hello.cpp`, `awaken_router*.cpp`,
> `conscientia_facade.cpp`, `auditus_facade_broadcasts.cpp`,
> `diarizen_periodic_worker.cpp`, `awaken.cpp`, `ws_server*.cpp`.

---

## 1. Transport

| Property | Value |
|----------|-------|
| Protocol | WebSocket (RFC 6455) over HTTP/1.1 upgrade |
| Default endpoint | `ws://<host>:8080/` (any path upgrades) |
| Text frames | UTF-8 JSON, one object per frame, always carries `"type"` |
| Binary frames | Raw little-endian **int16 mono PCM @ 16 kHz** |
| HTTP (same port) | Static WebUI files, `GET` only |

The server is a single `WsServer` (IPv6 dual-stack, epoll, non-blocking). The
**dev / debug console runs as a second `WsServer` instance on its own port**
(separate `static_dir`); the protocol on both ports is identical — only the
served frontend differs. Hardware clients are expected to speak this same
contract on the production port and skip the HTTP/static surface entirely.

### Binary framing details

- **Upstream (client → entity)**: microphone capture. The WebUI AudioWorklet
  emits 512-sample int16 chunks (1024 bytes, ~32 ms). Any chunk size is
  accepted; the server reassembles a continuous stream. Mono, 16 kHz, LE.
- **Downstream (entity → client)**: Vox / TTS and loopback playback, same
  int16 mono 16 kHz LE format. Clients convert to float32 by `s / 32768`.

There is **no metadata header** on audio binary frames today — the format is
fixed by contract. (Video upstream is reserved and not yet on the wire.)

---

## 2. Connection lifecycle

On every new WS connection the server immediately pushes a hello pair:

1. `consciousness_state` — full snapshot (see §4.4). If the LLM is not loaded
   the snapshot is the minimal `{"type":"consciousness_state","llm_loaded":false}`.
2. `consciousness_prompts` — the four current prompt strings (see §4.4).

No client handshake message is required. A client may begin sending audio or
commands as soon as the socket is open. Reconnect is the client's
responsibility (the WebUI auto-reconnects after 2 s).

---

## 3. Upstream — client → entity

### 3.1 Binary

Send raw int16 mono 16 kHz PCM frames to stream microphone audio into the
Auditus pipeline. No envelope, no header.

### 3.2 Text commands

Commands are **colon-delimited plain strings**, NOT JSON. Every command is
acknowledged by a JSON reply (echo envelope naming the same key). Unknown
commands are logged and ignored.

| Command | Effect | Ack `type` |
|---------|--------|-----------|
| `loopback:on` / `loopback:off` | Echo mic PCM back downstream | `loopback` |
| `gain:<f>` | Input gain 0.1–20.0 | `gain` |
| `silero_enable:on\|off` | Toggle Silero VAD | `silero_enable` |
| `silero_threshold:<f>` | VAD threshold 0–1 | `silero_threshold` |
| `frcrn_enable:on\|off` | Toggle FRCRN denoise | `frcrn_enable` |
| `vad_source:silero\|any` | Select VAD source | `vad_source` |
| `speaker_enable:on\|off` | Toggle CAM++ speaker ID | `speaker_enable` |
| `speaker_threshold:<f>` | CAM++ match threshold 0–1 | `speaker_threshold` |
| `speaker_clear` | Clear CAM++ roster | `speaker_clear` |
| `speaker_name:<id>:<name>` | Name a CAM++ speaker | `speaker_name` |
| `wlecapa_enable:on\|off` | Toggle WL-ECAPA speaker ID | `wlecapa_enable` |
| `wlecapa_threshold:<f>` | WL-ECAPA match threshold | `wlecapa_threshold` |
| `wlecapa_margin:<f>` | Abstain margin 0–0.5 | `wlecapa_margin` |
| `wlecapa_clear` | Clear WL-ECAPA roster | `wlecapa_clear` |
| `wlecapa_name:<id>:<name>` | Name a WL-ECAPA speaker | `wlecapa_name` |
| `wlecapa_delete:<id>` | Remove a WL-ECAPA speaker | `wlecapa_delete` |
| `wlecapa_merge:<dst>:<src>` | Merge two WL-ECAPA speakers | `wlecapa_merge` |
| `early_enable:on\|off` | Toggle early-trigger ID | `early_enable` |
| `early_trigger:<f>` | Early-trigger seconds 0.5–5 | `early_trigger` |
| `min_speech:<f>` | Min speech seconds 0.5–10 | `min_speech` |
| `asr_enable:on\|off` | Toggle ASR | `asr_enable` |
| `asr_vad_source:silero\|any\|direct` | ASR VAD source | `asr_vad_source` |
| `asr_param:<key>:<value>` | Set one ASR tunable (see §3.3) | `asr_param` |
| `consciousness_enable:<mode>:<on\|off>` | Toggle a decode pipeline (§3.4) | `consciousness_enable` |
| `consciousness_param:<key>:<value>` | Set sampling param (§3.4) | `consciousness_param` |
| `consciousness_prompt:<pipeline>:<text>` | Set a prompt (§3.4) | `consciousness_prompt` |
| `text_input:<text>` | Inject typed text into the stream | `text_input_ack` |
| `diarizen_trigger` | Request an extra reclustering pass | `speaker_diarize_progress` |
| `diarizen_finalize` | Finalise diarization for the session | `speaker_diarize_progress` → `_final` |

### 3.3 ASR tunables (`asr_param:<key>:<value>`)

`post_silence_ms`, `max_buf_sec`, `min_dur_sec`, `pre_roll_sec`, `max_tokens`,
`rep_penalty`, `min_energy`, `partial_sec`, `speech_ratio`,
`adaptive_silence` (bool), `adaptive_short_ms`, `adaptive_long_ms`,
`adaptive_vlong_ms`. The ack echoes the clamped value the engine adopted.

### 3.4 Consciousness commands

- **`consciousness_enable:<mode>:<on|off>`** — modes: `response`, `daydream`,
  `dreaming`, `llm`, `speech`, `thinking`, `action`.
- **`consciousness_param:<key>:<value>`** — `key` may be global
  (`temperature`, `top_k`, `top_p`) or pipeline-scoped
  (`speech.temperature`, `thinking.top_k`, `action.max_tokens`, …).
- **`consciousness_prompt:<pipeline>:<text>`** — `pipeline` ∈ `identity`
  (system prompt), `speech`, `thinking`, `action`. Legacy form without a
  pipeline prefix maps to `identity`.

---

## 4. Downstream — entity → client

All downstream text frames are JSON objects with a `"type"` discriminator.
Grouped below by subsystem. Float fields are plain JSON numbers.

### 4.1 Perception (Auditus)

**`pipeline_stats`** — high-frequency telemetry tick. Carries the full audio
pipeline state: `rms`, `is_speech`, `gain`; VAD (`silero_prob`,
`silero_speech`, `silero_threshold`, `vad_source`); FRCRN
(`frcrn_active/enabled/loaded`, `frcrn_lat_ms`); CAM++ (`speaker_id`,
`speaker_sim`, `speaker_new`, `speaker_count`, `speaker_name`,
`speaker_enabled`, `speaker_threshold`, `speaker_active`); WL-ECAPA
(`wlecapa_id/sim/new/count/exemplars/hits_above/name/enabled/threshold/active/margin`);
overlap detection (`od_*`); separation (`sep_*`); ASR (`asr_*` enable/loaded/
busy/latency/buffer/tunables); plus `multi_speaker`, `multi_score`,
`multi_source`, and `speaker_lists` (array of `{model, speakers:[{id, name,
count, exemplars, min_diversity}]}` for CAM++, CAM++Legacy, WL-ECAPA).

**`audio_stats`** — lightweight server-side audio counters (throughput / bytes).

**`vad`** — `{type, event:"start"|"end"}`, speech boundary edges.

### 4.2 Speaker identity (Orator)

| `type` | Key fields |
|--------|-----------|
| `speaker` | `id`, `sim`, `new` (bool), `name` |
| `speaker_amend` | `target_t_close_sec`, `prior_id`, `prior_sim`, `id`, `sim`, `name` — retroactive relabel of an earlier utterance |
| `speaker_relabel` | `segment_id`, `old_id`, `new_id`, `confidence` — reclusterer global-merge / K-cap |
| `speaker_diarize_progress` | `status` (`triggered`/`running`/`finalizing`) or `ok:false`+`error`; optional `samples`, `sec` |
| `speaker_diarize_partial` | `pass`, `origin_sec`, `audio_sec`, `wall_sec`, `segment_count`, `n_segments`, `changed_pending`, `segments:[[start,end,label],…]` |
| `speaker_diarize_final` | Same shape as partial (terminal), or `ok:false`+`error` |

`segments` use absolute stream-relative seconds; `label` is a global
voiceprint-anchored identity string (e.g. `S3`).

### 4.3 Hearing / ASR (Auditus)

| `type` | Key fields |
|--------|-----------|
| `asr_transcript` | `text`, `latency_ms`, `audio_sec`, `stream_start_sec`, `stream_end_sec`, `mel_ms`, `encoder_ms`, `decode_ms`, `tokens`, `mel_frames`, `speaker_id`, `speaker_name`, `speaker_sim`, `speaker_confidence`, `speaker_source`, `trigger` |
| `asr_transcript_amend` | `text`, `stream_start_sec`, `stream_end_sec`, `speaker_id`, `speaker_name` — FINAL speaker the LLM consumed |
| `asr_partial` | Streaming partial hypothesis |
| `asr_log` | Stage-tagged diagnostic envelope (`stage`: trigger/skipped/partial/result/transcript/fusion_shadow) |
| `asr_enable` | `enabled` (bool) — echo / state |
| `asr_param` | `key`, `value` — echo / state |

### 4.4 Thinking & reply (Conscientia)

**`consciousness_state`** — emitted on connect and on every state callback.
Fields: `state` (`active`/`daydream`/`dreaming`), `wakefulness` (0–1),
`kv_used`, `kv_free`, `pos`, `llm_loaded`, `entity` (persona name),
prefill/decode metrics (`prefill_ms`, `prefill_tps`, `prefill_tokens`,
`decode_ms_per_tok`, `decode_tokens`, `total_*`), memory
(`cuda_free_mb`, `cuda_total_mb`, `mem_avail_mb`, `rss_mb`), and the full
enable/sampling block (`enable_response/daydream/dreaming/llm/speech/thinking/
action`, global `temperature/top_k/top_p`, and per-pipeline `speech`,
`thinking`, `action` objects with `{temperature, top_k, top_p, max_tokens}`).

**`consciousness_prompts`** — `{identity, speech, thinking, action}` prompt
strings (sent separately because prompts may contain JSON-hostile characters).

| `type` | Key fields |
|--------|-----------|
| `consciousness_decode` | `text`, `tokens`, `time_ms`, `state` — a completed decode burst (the thinking / speech / action output) |
| `speech_token` | `text`, `token_id` — per-token streaming of spoken output |
| `consciousness_enable` | `mode`, `enabled` — echo |
| `consciousness_param` | `key`, `value` (or `error:"unknown"`) — echo |
| `consciousness_prompt` | `pipeline`, `ok` — echo |
| `text_input_ack` | `ok` — typed input accepted |

### 4.5 System (Vires)

**`vires_compute_snapshot`** — GPU substrate ledger: `greatest_priority`,
`least_priority`, `background_yielding` (bool), `foreground_idle_us`
(int or `null`), and `consumers:[{id, name, priority, submitted, reclaimed}]`.

---

## 5. HTTP surface

The same port serves static WebUI assets via `GET`:

- `GET /` → `index.html`
- `GET /<path>` → file under `static_dir`, mime-typed by extension
- Path traversal (`..`) and out-of-root `realpath` are rejected (403/404)
- Non-`GET` methods are unsupported on the static surface

The REST endpoints described in earlier drafts (`/health`, `/api/state`,
`/api/memory`, `/api/config`, `/api/input`) are **aspirational** — the live
control plane today is the WS text-command channel (§3.2). Hardware clients
should drive everything over WS and treat HTTP as WebUI-asset-only.

---

## 6. Guidance for hardware clients

A non-browser device that wants to be a first-class peer needs only:

1. A WebSocket client to the production port.
2. Int16/16 kHz/mono PCM up (mic) and down (TTS) — no codec, no header.
3. A JSON parser dispatching on `type` for the §4 messages it cares about.
4. The §3.2 text commands for any control it exposes.

A minimal "listening + identity + reply" device consumes `pipeline_stats`
(or just `vad` + `speaker`), `asr_transcript`, `consciousness_decode` /
`speech_token`, and `consciousness_state`; it emits PCM frames and optionally
`text_input:` / `asr_enable:on`. Everything else (diarization, tuning knobs,
vires telemetry) is optional and ignorable.

---

## 7. Versioning rules

- Adding a new `type` or a new optional field is **backward-compatible**;
  clients MUST ignore unknown `type`s and unknown fields.
- Renaming/removing a field, reordering binary PCM semantics, or changing a
  command grammar is a **breaking change** — bump a `protocol_version` (to be
  introduced in the hello envelope) and record it in this file's changelog.
- This document and its `docs/zh/` mirror move together. Drift is a bug.
