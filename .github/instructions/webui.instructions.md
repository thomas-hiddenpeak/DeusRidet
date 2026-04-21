---
applyTo: "src/nexus/webui/**"
---

# WebUI — Nexus Frontend Rules

The WebUI is a **professional-grade single-page application** connected via
WebSocket for real-time bidirectional streams.

## Frontend Architecture Principles

- **Semantic HTML**: Use `<article>`, `<section>`, `<nav>`, `<aside>`,
  `<figure>`, etc. — never `<div>` soup.
- **Style–function decoupling**: All visual styling via CSS (BEM or utility
  classes). Zero inline styles. JS handles logic only, never DOM appearance
  directly.
- **Feature–feature decoupling**: Each panel/widget is an independent module
  with its own state, WS subscription, and lifecycle. Panels communicate via
  a lightweight event bus — never direct DOM manipulation across modules.
- **Component model**: Each UI component is a self-contained ES module with
  interface `{mount, unmount, onMessage, render}`. No framework dependency
  (no React/Vue) — vanilla JS with a thin component abstraction.
- **Responsive layout**: CSS Grid / Flexbox, works on desktop and tablet.
- **Accessibility**: ARIA labels, keyboard navigation, high-contrast support.

## File Structure

```
src/nexus/webui/
├── index.html          # semantic shell, module entry
├── css/
│   ├── tokens.css      # design tokens (colors, spacing, typography)
│   ├── layout.css      # grid, responsive breakpoints
│   └── components/     # per-component styles
├── js/
│   ├── app.js          # bootstrap, WS connection, event bus
│   ├── ws-client.js    # WebSocket protocol handler
│   ├── components/     # UI modules
│   └── utils/          # shared helpers (formatters, audio worklet)
└── assets/             # icons, fonts
```

## WebSocket Channels

- **Audio upstream**: Raw PCM from browser mic → ASR pipeline.
- **Audio downstream**: TTS PCM output → browser playback.
- **Video upstream**: Camera frames (MJPEG or raw via MediaStream API) →
  Vision pipeline. Binary frames with metadata header (timestamp,
  resolution, format). Also supports local V4L2 capture bypassing WS.
- **State stream**: JSON frames — consciousness state, Decode branches,
  wakefulness level, active speaker, attention heatmap.
- **Control**: Start/stop, config changes, persona switches.

## HTTP Endpoints

- `GET  /health`       — readiness probe
- `GET  /api/state`    — snapshot of consciousness state
- `GET  /api/memory`   — query long-term memory
- `POST /api/config`   — update runtime configuration
- `POST /api/input`    — inject text input

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

## Verification After UI Changes

Every UI change must pass the full HTTP 200 + WS 101 check (see
`workflow.instructions.md`, step 5–7), plus newly added asset files must
each return 200.
