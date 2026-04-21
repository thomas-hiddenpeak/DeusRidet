# 10 — Nexus (WebUI & External Interface)

Nexus is the **only** external interface. Internal subsystems never expose
themselves to the network — Nexus translates.

## WebSocket Channels

- **Audio upstream**: Raw PCM from browser mic → Auditus pipeline.
- **Audio downstream**: Vox PCM output → browser playback.
- **Video upstream**: Camera frames (MJPEG or raw via MediaStream API) →
  Visus pipeline. Binary frames with metadata header (timestamp,
  resolution, format). Also supports local V4L2 capture bypassing WS.
- **State stream**: JSON frames — consciousness state, Decode branches,
  wakefulness, active speaker, attention heatmap.
- **Control**: Start/stop, config changes, persona switches.

## HTTP Endpoints

- `GET  /health`     — readiness probe
- `GET  /api/state`  — snapshot of consciousness state
- `GET  /api/memory` — query long-term memory
- `POST /api/config` — update runtime configuration
- `POST /api/input`  — inject text input

## Observable Dashboard

The WebUI visualizes:
- Current consciousness stream content
- All Decode branch states (thinking / speech / action / daydream)
- Wakefulness level gauge
- Speaker identification panel
- GPU utilization, memory, KV Cache occupancy, eviction rate
- KV block importance heatmap (hot vs eviction candidates)
- Vision: current camera frame + detected features
- Long-term memory: episodic store size, semantic graph stats, retrievals
- Consolidation activity: dream-state memory maintenance progress

## Frontend Rules

See `.github/instructions/webui.instructions.md` for semantic HTML,
style-function decoupling, component model, and file structure rules.

## Implementation Surface

```
src/nexus/
├── ws_server.{h,cpp}   # WebSocket server (audio/state/control)
├── http_server.{h,cpp} # HTTP REST (config, snapshots, memory query)
└── webui/              # static HTML/JS/CSS frontend
    ├── index.html
    ├── css/
    ├── js/
    └── assets/
```

## Philosophical Notes

- Nexus is the *only* place the internal Latin world meets the external
  world. Names on the network side (WS message types, HTTP routes) are
  allowed to be pragmatic — they must be comprehensible to outside
  integrators. But that translation happens only in `src/nexus/`.
