// tracker-panel.js — SpeakerTracker continuous pipeline visualization.
// Displays: current speaker, state, confidence, F0, similarity curve,
// speaker timeline, and parameter controls for A/B comparison with SAAS.

import { spkColor } from '../utils/speaker-colors.js';

const STATE_NAMES = ['Silence', 'Tracking', 'Transition', 'Overlap', 'Unknown'];
const CONF_NAMES  = ['—', 'Low', 'Med', 'High'];

export class TrackerPanel {
    constructor(ws) {
        this.el = document.getElementById('tracker-panel');
        this._ws = ws;
        if (!this.el) return;

        this._timeline = [];        // [{ts, id, name, sim, state, confidence}]
        this._maxTimeline = 200;
        this._simHistory = [];       // [{ts, sim}]
        this._maxSimHistory = 120;
        this._speakers = [];         // from tracker_speakers

        this._build();
        this._resizeCanvas();
        window.addEventListener('resize', () => this._resizeCanvas());
    }

    onPipelineStats(stats) {
        if (!this.el) return;
        if (!stats.tracker_enabled) {
            this._statusEl.textContent = 'Disabled';
            return;
        }

        const state = stats.tracker_state ?? 0;
        const id = stats.tracker_spk_id ?? -1;
        const sim = stats.tracker_spk_sim ?? 0;
        const name = stats.tracker_spk_name || (id >= 0 ? `#${id}` : '—');
        const conf = stats.tracker_confidence ?? 0;
        const f0 = stats.tracker_f0_hz ?? 0;
        const jitter = stats.tracker_f0_jitter ?? 0;
        const simRef = stats.tracker_sim_to_ref ?? 0;
        const simAvg = stats.tracker_sim_avg ?? 0;
        const switches = stats.tracker_switches ?? 0;
        const spkCount = stats.tracker_spk_count ?? 0;
        const latMs = stats.tracker_check_lat_ms ?? 0;
        const checkActive = stats.tracker_check_active;

        // Update text displays.
        this._statusEl.textContent = STATE_NAMES[state] || '?';
        this._statusEl.className = 'tracker-state state-' + (STATE_NAMES[state] || 'unknown').toLowerCase();
        this._spkEl.textContent = name;
        this._spkEl.style.color = id >= 0 ? spkColor(id) : '#888';
        this._confEl.textContent = CONF_NAMES[conf] || '?';
        this._simEl.textContent = sim.toFixed(3);
        this._f0El.textContent = f0 > 0 ? `${f0.toFixed(0)} Hz` : '—';
        this._jitterEl.textContent = jitter > 0 ? jitter.toFixed(3) : '—';
        this._switchEl.textContent = switches;
        this._countEl.textContent = spkCount;
        this._latEl.textContent = checkActive ? `${latMs.toFixed(0)} ms` : '—';

        // Sim history (only on check ticks).
        if (checkActive) {
            this._simHistory.push({ ts: Date.now(), sim: simRef });
            if (this._simHistory.length > this._maxSimHistory)
                this._simHistory.shift();
            this._renderSimCurve(stats.tracker_threshold ?? 0.55);
        }

        // Timeline (on new check or state change).
        if (checkActive) {
            this._timeline.push({
                ts: Date.now(), id, name, sim, state, confidence: conf
            });
            if (this._timeline.length > this._maxTimeline)
                this._timeline.shift();
            this._renderTimeline();
        }

        // Speaker list.
        if (stats.tracker_speakers) {
            this._speakers = stats.tracker_speakers;
            this._renderSpeakers();
        }
    }

    _build() {
        this.el.innerHTML = `
        <h3>Speaker Tracker <span class="tracker-badge">Independent Pipeline</span></h3>
        <div class="tracker-grid">
            <div class="tracker-info">
                <div class="tracker-row"><span class="lbl">State</span><span class="val" data-ref="status">—</span></div>
                <div class="tracker-row"><span class="lbl">Speaker</span><span class="val" data-ref="spk">—</span></div>
                <div class="tracker-row"><span class="lbl">Confidence</span><span class="val" data-ref="conf">—</span></div>
                <div class="tracker-row"><span class="lbl">Similarity</span><span class="val" data-ref="sim">—</span></div>
                <div class="tracker-row"><span class="lbl">F0</span><span class="val" data-ref="f0">—</span></div>
                <div class="tracker-row"><span class="lbl">F0 Jitter</span><span class="val" data-ref="jitter">—</span></div>
                <div class="tracker-row"><span class="lbl">Switches</span><span class="val" data-ref="switches">0</span></div>
                <div class="tracker-row"><span class="lbl">Speakers</span><span class="val" data-ref="count">0</span></div>
                <div class="tracker-row"><span class="lbl">Latency</span><span class="val" data-ref="lat">—</span></div>
            </div>
            <div class="tracker-canvas-wrap">
                <canvas class="tracker-sim-canvas" data-ref="simCanvas"></canvas>
            </div>
        </div>
        <div class="tracker-timeline-wrap">
            <canvas class="tracker-timeline-canvas" data-ref="timelineCanvas"></canvas>
        </div>
        <div class="tracker-speakers" data-ref="speakerList"></div>
        <div class="tracker-controls">
            <label>Enable <input type="checkbox" data-ref="enableCb" checked></label>
            <label>Threshold <input type="number" step="0.05" min="0" max="1" value="0.55" data-ref="threshIn" class="num-input"></label>
            <label>Change <input type="number" step="0.05" min="0" max="1" value="0.35" data-ref="changeIn" class="num-input"></label>
            <label>Interval <input type="number" step="50" min="250" max="5000" value="500" data-ref="intervalIn" class="num-input"> ms</label>
            <label>Window <input type="number" step="100" min="500" max="5000" value="1500" data-ref="windowIn" class="num-input"> ms</label>
            <button data-ref="clearBtn" class="btn-small">Clear DB</button>
        </div>`;

        // Bind refs.
        const ref = (n) => this.el.querySelector(`[data-ref="${n}"]`);
        this._statusEl = ref('status');
        this._spkEl    = ref('spk');
        this._confEl   = ref('conf');
        this._simEl    = ref('sim');
        this._f0El     = ref('f0');
        this._jitterEl = ref('jitter');
        this._switchEl = ref('switches');
        this._countEl  = ref('count');
        this._latEl    = ref('lat');
        this._simCanvas = ref('simCanvas');
        this._timelineCanvas = ref('timelineCanvas');
        this._speakerListEl = ref('speakerList');

        // Controls.
        const enableCb = ref('enableCb');
        enableCb.addEventListener('change', () => {
            this._ws.send(`tracker_enable:${enableCb.checked ? 'on' : 'off'}`);
        });
        ref('threshIn').addEventListener('change', (e) => {
            this._ws.send(`tracker_threshold:${e.target.value}`);
        });
        ref('changeIn').addEventListener('change', (e) => {
            this._ws.send(`tracker_change_threshold:${e.target.value}`);
        });
        ref('intervalIn').addEventListener('change', (e) => {
            this._ws.send(`tracker_interval:${e.target.value}`);
        });
        ref('windowIn').addEventListener('change', (e) => {
            this._ws.send(`tracker_window:${e.target.value}`);
        });
        ref('clearBtn').addEventListener('click', () => {
            this._ws.send('tracker_clear');
            this._timeline = [];
            this._simHistory = [];
            this._speakers = [];
            this._renderTimeline();
            this._renderSimCurve(0.55);
            this._renderSpeakers();
        });
    }

    _resizeCanvas() {
        if (!this._simCanvas) return;
        const dpr = window.devicePixelRatio || 1;

        const simWrap = this._simCanvas.parentElement;
        this._simCanvas.width = simWrap.clientWidth * dpr;
        this._simCanvas.height = simWrap.clientHeight * dpr;
        this._simCanvas.style.width = simWrap.clientWidth + 'px';
        this._simCanvas.style.height = simWrap.clientHeight + 'px';

        const tlWrap = this._timelineCanvas.parentElement;
        this._timelineCanvas.width = tlWrap.clientWidth * dpr;
        this._timelineCanvas.height = tlWrap.clientHeight * dpr;
        this._timelineCanvas.style.width = tlWrap.clientWidth + 'px';
        this._timelineCanvas.style.height = tlWrap.clientHeight + 'px';

        this._renderSimCurve(0.55);
        this._renderTimeline();
    }

    _renderSimCurve(threshold) {
        const cv = this._simCanvas;
        if (!cv) return;
        const ctx = cv.getContext('2d');
        const w = cv.width, h = cv.height;
        const dpr = window.devicePixelRatio || 1;
        ctx.clearRect(0, 0, w, h);

        // Background.
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, w, h);

        if (this._simHistory.length < 2) return;

        const n = this._simHistory.length;
        const xStep = w / (this._maxSimHistory - 1);

        // Threshold line.
        const thY = h - threshold * h;
        ctx.strokeStyle = 'rgba(255,200,0,0.4)';
        ctx.lineWidth = 1 * dpr;
        ctx.setLineDash([4 * dpr, 4 * dpr]);
        ctx.beginPath();
        ctx.moveTo(0, thY);
        ctx.lineTo(w, thY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Sim curve.
        ctx.strokeStyle = '#4fc3f7';
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath();
        const startX = (this._maxSimHistory - n) * xStep;
        for (let i = 0; i < n; i++) {
            const x = startX + i * xStep;
            const y = h - this._simHistory[i].sim * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Labels.
        ctx.fillStyle = '#aaa';
        ctx.font = `${10 * dpr}px monospace`;
        ctx.fillText('1.0', 2, 12 * dpr);
        ctx.fillText('0.0', 2, h - 2);
        ctx.fillText(`th=${threshold.toFixed(2)}`, w - 60 * dpr, thY - 4);
    }

    _renderTimeline() {
        const cv = this._timelineCanvas;
        if (!cv) return;
        const ctx = cv.getContext('2d');
        const w = cv.width, h = cv.height;
        const dpr = window.devicePixelRatio || 1;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, w, h);

        if (this._timeline.length === 0) return;

        const n = this._timeline.length;
        const barW = Math.max(2, w / this._maxTimeline);

        for (let i = 0; i < n; i++) {
            const entry = this._timeline[i];
            const x = (this._maxTimeline - n + i) * barW;
            const barH = entry.sim * h;

            if (entry.id >= 0) {
                ctx.fillStyle = spkColor(entry.id);
            } else if (entry.state === 3) {
                ctx.fillStyle = '#ff6b6b'; // overlap = red
            } else if (entry.state === 4) {
                ctx.fillStyle = '#666'; // unknown = gray
            } else {
                ctx.fillStyle = '#333'; // silence
            }
            ctx.fillRect(x, h - barH, barW - 1, barH);

            // Confidence indicator: small dot on top.
            if (entry.confidence >= 2) {
                ctx.fillStyle = entry.confidence >= 3 ? '#4caf50' : '#ffc107';
                ctx.fillRect(x, 0, barW - 1, 3 * dpr);
            }
        }
    }

    _renderSpeakers() {
        if (!this._speakerListEl) return;
        if (!this._speakers || this._speakers.length === 0) {
            this._speakerListEl.innerHTML = '<em>No speakers registered</em>';
            return;
        }

        let html = '<table class="tracker-spk-table"><tr><th>ID</th><th>Name</th><th>Exemplars</th><th></th></tr>';
        for (const spk of this._speakers) {
            const color = spkColor(spk.id);
            html += `<tr>
                <td style="color:${color}; font-weight:bold">#${spk.id}</td>
                <td><input type="text" value="${spk.name || ''}" 
                    class="spk-name-input" data-spk-id="${spk.id}" 
                    placeholder="unnamed"></td>
                <td>${spk.total_exemplars || spk.exemplars || '?'}</td>
                <td><button class="btn-tiny btn-rename" data-spk-id="${spk.id}">✓</button></td>
            </tr>`;
        }
        html += '</table>';
        this._speakerListEl.innerHTML = html;

        // Bind rename buttons.
        this._speakerListEl.querySelectorAll('.btn-rename').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.dataset.spkId;
                const input = this._speakerListEl.querySelector(`input[data-spk-id="${id}"]`);
                if (input) this._ws.send(`tracker_name:${id}:${input.value}`);
            });
        });
    }
}
