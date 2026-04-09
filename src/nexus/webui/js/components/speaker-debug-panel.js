// speaker-debug-panel.js — Speaker identification debug visualizations.
// Displays: confidence gauge, speaker timeline, change detection plot,
// latency breakdown, and per-speaker diversity metrics.

import { spkColor } from '../utils/speaker-colors.js';

export class SpeakerDebugPanel {
    constructor(ws) {
        this.el = document.getElementById('speaker-debug');
        this._ws = ws;  // WebSocket reference for sending commands
        if (!this.el) return;

        this._timeline = [];           // [{ts, id, name, sim}]
        this._maxTimeline = 200;
        this._simHistory = [];          // change detection similarity values
        this._maxSimHistory = 100;
        this._lastSim = 0;
        this._lastId = -1;
        this._threshold = 0.55;
        this._changeThreshold = 0.50;
        this._earlyTriggerSec = 1.7;

        this._build();
        this._resizeCanvas();
        window.addEventListener('resize', () => this._resizeCanvas());
    }

    // --- Public API ---

    // Called on each `speaker` event from WS.
    onSpeakerEvent(data) {
        if (!this.el) return;
        this._lastSim = data.sim || 0;
        this._lastId = data.id ?? -1;
        const name = data.name || `#${data.id}`;

        // Backfill gray entries before adding the resolved result.
        if (data.id >= 0) {
            this._backfillTimeline(data.id, name);
        }

        this._timeline.push({
            ts: Date.now(),
            id: data.id,
            name: name,
            sim: data.sim
        });
        if (this._timeline.length > this._maxTimeline)
            this._timeline.shift();

        this._renderGauge();
        this._renderTimeline();
    }

    // Called on `pipeline_stats` with speaker info.
    onPipelineStats(stats) {
        if (!this.el) return;

        // Update threshold from server.
        if (stats.wlecapa_threshold !== undefined)
            this._threshold = stats.wlecapa_threshold;

        // When WL-ECAPA is active this tick, use it to feed gauge + timeline.
        if (stats.wlecapa_active) {
            this._lastSim = stats.wlecapa_sim || 0;
            this._lastId = stats.wlecapa_id ?? -1;
            const name = stats.wlecapa_name || `#${stats.wlecapa_id}`;
            const isEarly = !!stats.wlecapa_is_early;

            this._timeline.push({
                ts: Date.now(),
                id: stats.wlecapa_id,
                name: name,
                sim: stats.wlecapa_sim,
                early: isEarly
            });
            if (this._timeline.length > this._maxTimeline)
                this._timeline.shift();

            // Retroactive backfill: when a full-segment result resolves,
            // recolor recent gray (id=-1) entries with the actual speaker.
            if (!isEarly && stats.wlecapa_id >= 0) {
                this._backfillTimeline(stats.wlecapa_id, name);
            }

            this._renderGauge();
            this._renderTimeline();

            // Change detection from pipeline_stats (end-of-segment only).
            if (stats.change_similarity !== undefined) {
                this._simHistory.push(stats.change_similarity);
                if (this._simHistory.length > this._maxSimHistory)
                    this._simHistory.shift();
                this._renderChangePlot();
            }
        }

        // Update early trigger value for display.
        if (stats.early_trigger_sec !== undefined)
            this._earlyTriggerSec = stats.early_trigger_sec;

        // Latency breakdown (populated when server sends these fields).
        this._renderLatency(stats);

        // Diversity metrics from speaker_lists.
        this._renderDiversity(stats);
    }

    // Called with speaker_debug data (future server extension).
    onDebugData(data) {
        if (!this.el) return;

        if (data.change_similarity !== undefined) {
            this._simHistory.push(data.change_similarity);
            if (this._simHistory.length > this._maxSimHistory)
                this._simHistory.shift();
            if (data.change_threshold !== undefined)
                this._changeThreshold = data.change_threshold;
            this._renderChangePlot();
        }

        if (data.latency) this._renderLatency(data.latency);
    }

    // --- Internal: Build DOM ---

    _build() {
        this.el.innerHTML = `
        <div class="dbg-row">
            <div class="dbg-card dbg-gauge-card">
                <h4 class="dbg-card__title">Confidence</h4>
                <canvas id="dbg-gauge-canvas" class="dbg-gauge-canvas"
                        width="180" height="24"></canvas>
                <div class="dbg-gauge-val">
                    <span id="dbg-gauge-sim">—</span>
                    <span class="dbg-gauge-thresh" id="dbg-gauge-thresh">
                        thr: <strong>${this._threshold.toFixed(2)}</strong>
                    </span>
                </div>
            </div>
            <div class="dbg-card dbg-latency-card">
                <h4 class="dbg-card__title">Latency</h4>
                <div class="dbg-latency-grid" id="dbg-latency">
                    <span>CNN</span><span id="lat-cnn">—</span>
                    <span>Encoder</span><span id="lat-enc">—</span>
                    <span>ECAPA</span><span id="lat-ecapa">—</span>
                    <span>Total</span><span id="lat-total" class="dbg-lat-total">—</span>
                </div>
            </div>
        </div>
        <div class="dbg-card">
            <h4 class="dbg-card__title">Speaker Timeline</h4>
            <canvas id="dbg-timeline-canvas" class="dbg-timeline-canvas"
                    width="600" height="32"></canvas>
        </div>
        <div class="dbg-card">
            <h4 class="dbg-card__title">Change Detection</h4>
            <canvas id="dbg-change-canvas" class="dbg-change-canvas"
                    width="600" height="48"></canvas>
        </div>
        <div class="dbg-card">
            <h4 class="dbg-card__title">Diversity</h4>
            <div id="dbg-diversity" class="dbg-diversity"></div>
        </div>
        <div class="dbg-card">
            <h4 class="dbg-card__title">Diversity</h4>
            <div id="dbg-diversity" class="dbg-diversity"></div>
        </div>`;

        this._gaugeCanvas = document.getElementById('dbg-gauge-canvas');
        this._gaugeCtx = this._gaugeCanvas.getContext('2d');
        this._timelineCanvas = document.getElementById('dbg-timeline-canvas');
        this._timelineCtx = this._timelineCanvas.getContext('2d');
        this._changeCanvas = document.getElementById('dbg-change-canvas');
        this._changeCtx = this._changeCanvas.getContext('2d');
        this._simEl = document.getElementById('dbg-gauge-sim');
        this._threshEl = document.getElementById('dbg-gauge-thresh');
        this._diversityEl = document.getElementById('dbg-diversity');
    }

    _resizeCanvas() {
        if (!this.el) return;
        const w = this.el.clientWidth - 32;
        [this._timelineCanvas, this._changeCanvas].forEach(c => {
            if (c && w > 100) c.width = w;
        });
        this._renderTimeline();
        this._renderChangePlot();
    }

    // --- Confidence gauge ---

    _renderGauge() {
        const ctx = this._gaugeCtx;
        if (!ctx) return;
        const W = this._gaugeCanvas.width, H = this._gaugeCanvas.height;
        ctx.clearRect(0, 0, W, H);

        // Background track.
        ctx.fillStyle = '#21262d';
        ctx.fillRect(0, 4, W, H - 8);

        // Fill bar.
        const frac = Math.max(0, Math.min(1, this._lastSim));
        const fillW = frac * W;
        const hue = frac < this._threshold ? 0 : 120 * ((frac - this._threshold) / (1 - this._threshold));
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(0, 4, fillW, H - 8);

        // Threshold line.
        const tx = this._threshold * W;
        ctx.strokeStyle = '#c9d1d9';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(tx, 0);
        ctx.lineTo(tx, H);
        ctx.stroke();
        ctx.setLineDash([]);

        this._simEl.textContent = this._lastSim.toFixed(3);
        this._threshEl.innerHTML = `thr: <strong>${this._threshold.toFixed(2)}</strong>`;
    }

    // --- Speaker timeline ---

    // Backfill recent gray (id=-1) timeline entries with resolved speaker.
    // Scans backwards within a time window (early trigger + buffer) and
    // updates all id=-1 entries, even if non-gray entries are interspersed.
    _backfillTimeline(resolvedId, resolvedName) {
        const cutoff = Date.now() - (this._earlyTriggerSec + 2) * 1000;
        for (let i = this._timeline.length - 1; i >= 0; i--) {
            const ev = this._timeline[i];
            if (ev.ts < cutoff) break;  // beyond time window
            if (ev.id === -1) {
                ev.id = resolvedId;
                ev.name = resolvedName;
            }
        }
    }

    _renderTimeline() {
        const ctx = this._timelineCtx;
        if (!ctx || this._timeline.length === 0) return;
        const W = this._timelineCanvas.width, H = this._timelineCanvas.height;
        ctx.clearRect(0, 0, W, H);

        const now = Date.now();
        const windowMs = 60000; // show last 60s
        const tStart = now - windowMs;

        ctx.fillStyle = '#161b22';
        ctx.fillRect(0, 0, W, H);

        let lastX = -1, lastId = null, lastColor = '';
        for (const ev of this._timeline) {
            if (ev.ts < tStart) continue;
            const x = ((ev.ts - tStart) / windowMs) * W;
            const color = spkColor(ev.id);

            if (lastId !== null && lastX >= 0) {
                ctx.fillStyle = lastColor;
                ctx.fillRect(lastX, 2, x - lastX, H - 4);
            }
            lastX = x;
            lastId = ev.id;
            lastColor = color;
        }
        // Draw last segment to now.
        if (lastId !== null && lastX >= 0) {
            ctx.fillStyle = lastColor;
            ctx.fillRect(lastX, 2, W - lastX, H - 4);
        }

        // Labels — use contrasting text color (dark on bright, light on gray).
        ctx.font = '10px var(--font-mono)';
        ctx.textBaseline = 'middle';
        let prevLabelX = -100;
        for (const ev of this._timeline) {
            if (ev.ts < tStart) continue;
            const x = ((ev.ts - tStart) / windowMs) * W;
            if (x - prevLabelX > 40) {
                ctx.fillStyle = ev.id < 0 ? '#8b949e' : '#0d1117';

                ctx.fillText(ev.name, x + 3, H / 2);
                prevLabelX = x;
            }
        }
    }

    // --- Change detection plot ---

    _renderChangePlot() {
        const ctx = this._changeCtx;
        if (!ctx || this._simHistory.length === 0) return;
        const W = this._changeCanvas.width, H = this._changeCanvas.height;
        ctx.clearRect(0, 0, W, H);

        ctx.fillStyle = '#161b22';
        ctx.fillRect(0, 0, W, H);

        const n = this._simHistory.length;
        const dx = W / Math.max(n - 1, 1);

        // Threshold line.
        const ty = (1 - this._changeThreshold) * (H - 4) + 2;
        ctx.strokeStyle = '#f8514966';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, ty);
        ctx.lineTo(W, ty);
        ctx.stroke();
        ctx.setLineDash([]);

        // Similarity curve.
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = i * dx;
            const y = (1 - this._simHistory[i]) * (H - 4) + 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Dots below threshold.
        for (let i = 0; i < n; i++) {
            if (this._simHistory[i] < this._changeThreshold) {
                const x = i * dx;
                const y = (1 - this._simHistory[i]) * (H - 4) + 2;
                ctx.fillStyle = '#f85149';
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    // --- Latency ---

    _renderLatency(data) {
        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el && val !== undefined && val !== null)
                el.textContent = typeof val === 'number' ? val.toFixed(1) + ' ms' : val;
        };
        set('lat-cnn', data.lat_cnn_ms);
        set('lat-enc', data.lat_encoder_ms);
        set('lat-ecapa', data.lat_ecapa_ms);
        set('lat-total', data.lat_total_ms);
    }

    // --- Diversity ---

    _renderDiversity(stats) {
        if (!this._diversityEl || !stats.speaker_lists) return;

        let html = '';
        for (const group of stats.speaker_lists) {
            if (group.model !== 'WL-ECAPA') continue;
            for (const spk of group.speakers) {
                const ex = spk.exemplars || 1;
                const div = spk.min_diversity;
                const divStr = (div !== undefined && div !== null)
                    ? div.toFixed(3) : '—';
                const divPct = div !== undefined ? Math.min(div / 0.3, 1) * 100 : 0;
                const label = spk.name || `#${spk.id}`;
                const dotColor = spkColor(spk.id);
                html += `<div class="dbg-div-row">
                    <span class="dbg-div-dot" style="background:${dotColor}"></span>
                    <span class="dbg-div-label">${this._esc(label)}</span>
                    <span class="dbg-div-ex">${ex}ex</span>
                    <span class="dbg-div-val">div=${divStr}</span>
                    <div class="dbg-div-bar-wrap">
                        <div class="dbg-div-bar" style="width:${divPct}%"></div>
                    </div>
                </div>`;
            }
        }
        if (!html) html = '<span class="dbg-div-empty">No WL-ECAPA speakers</span>';
        this._diversityEl.innerHTML = html;
    }

    _esc(s) {
        return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
}
