// asr-panel.js — ASR debug & evaluation panel.
// Displays transcript history, per-utterance latency, RTF, cumulative stats,
// and tunable parameter controls (post-silence, buffer size, pre-roll, etc).

export class AsrPanel {
    constructor(ws) {
        this.ws = ws;
        this.el = document.getElementById('asr-panel');
        this.totalCount = 0;
        this.totalLatency = 0;
        this.totalAudioSec = 0;
        this._latencyHistory = [];   // recent latency_ms values for chart
        this._latencyMax = 50;
        this._build();
        this._bindControls();
    }

    _build() {
        this.el.innerHTML = `
            <h2 class="panel__title">ASR — Speech Recognition</h2>
            <div class="asr-controls">
                <button id="asr-toggle" class="btn btn--vad btn--active"
                        aria-pressed="true">ASR</button>
                <span class="asr-status" id="asr-status">—</span>
                <span class="asr-buf-indicator" id="asr-buf-ind" title="Buffer state">
                    <span class="asr-buf-bar" id="asr-buf-bar"></span>
                    <span class="asr-buf-label" id="asr-buf-label">0.0s</span>
                </span>
            </div>
            <div class="vad-source-control">
                <label class="vad-threshold-label">
                    VAD → ASR:
                </label>
                <select id="asr-vad-select" class="vad-source-select">
                    <option value="silero" selected>Silero</option>
                    <option value="fsmn">FSMN</option>
                    <option value="ten">TEN</option>
                    <option value="any">Any (OR)</option>
                    <option value="direct">Direct (no VAD)</option>
                </select>
            </div>
            <div class="asr-stats-grid" id="asr-stats">
                <div class="asr-stat">
                    <span class="asr-stat__label">Utterances</span>
                    <strong class="asr-stat__value" id="asr-count">0</strong>
                </div>
                <div class="asr-stat">
                    <span class="asr-stat__label">Avg Latency</span>
                    <strong class="asr-stat__value" id="asr-avg-lat">—</strong>
                </div>
                <div class="asr-stat">
                    <span class="asr-stat__label">Avg RTF</span>
                    <strong class="asr-stat__value" id="asr-avg-rtf">—</strong>
                </div>
                <div class="asr-stat">
                    <span class="asr-stat__label">Last Latency</span>
                    <strong class="asr-stat__value" id="asr-last-lat">—</strong>
                </div>
            </div>
            <details class="asr-params-section" open>
                <summary class="asr-params-toggle">Parameters</summary>
                <div class="asr-params-grid">
                    <label class="asr-param">
                        <span class="asr-param__label">Post-silence (ms)</span>
                        <input type="range" id="asr-p-post-silence" min="100" max="2000" step="50" value="300">
                        <output id="asr-v-post-silence">300</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Max buffer (s)</span>
                        <input type="range" id="asr-p-max-buf" min="5" max="30" step="1" value="30">
                        <output id="asr-v-max-buf">30</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Min duration (s)</span>
                        <input type="range" id="asr-p-min-dur" min="0.1" max="2.0" step="0.1" value="0.3">
                        <output id="asr-v-min-dur">0.3</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Pre-roll (s)</span>
                        <input type="range" id="asr-p-pre-roll" min="0" max="3.0" step="0.1" value="1.0">
                        <output id="asr-v-pre-roll">1.0</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Max tokens</span>
                        <input type="range" id="asr-p-max-tokens" min="32" max="2048" step="32" value="448">
                        <output id="asr-v-max-tokens">448</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Rep penalty</span>
                        <input type="range" id="asr-p-rep-penalty" min="1.0" max="2.0" step="0.05" value="1.0">
                        <output id="asr-v-rep-penalty">1.00</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Min energy</span>
                        <input type="range" id="asr-p-min-energy" min="0" max="0.05" step="0.001" value="0.008">
                        <output id="asr-v-min-energy">0.008</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Partial interval (s)</span>
                        <input type="range" id="asr-p-partial-sec" min="0" max="5" step="0.5" value="2.0">
                        <output id="asr-v-partial-sec">2.0</output>
                    </label>
                    <label class="asr-param">
                        <span class="asr-param__label">Min speech ratio</span>
                        <input type="range" id="asr-p-speech-ratio" min="0" max="0.5" step="0.01" value="0.15">
                        <output id="asr-v-speech-ratio">0.15</output>
                    </label>
                </div>
            </details>
            <canvas id="asr-latency-chart" class="asr-latency-chart" width="400" height="80"
                    aria-label="ASR latency history"></canvas>
            <div class="asr-partial" id="asr-partial" aria-live="polite">
                <span class="asr-partial__label">Partial:</span>
                <span class="asr-partial__text" id="asr-partial-text">—</span>
            </div>
        `;
        this.toggleBtn = this.el.querySelector('#asr-toggle');
        this.statusEl = this.el.querySelector('#asr-status');
        this.countEl = this.el.querySelector('#asr-count');
        this.avgLatEl = this.el.querySelector('#asr-avg-lat');
        this.avgRtfEl = this.el.querySelector('#asr-avg-rtf');
        this.lastLatEl = this.el.querySelector('#asr-last-lat');
        this.canvas = this.el.querySelector('#asr-latency-chart');
        this.ctx = this.canvas.getContext('2d');
        this.bufBar = this.el.querySelector('#asr-buf-bar');
        this.bufLabel = this.el.querySelector('#asr-buf-label');
        this.asrVadSelect = this.el.querySelector('#asr-vad-select');
        this.partialTextEl = this.el.querySelector('#asr-partial-text');
        this._partialClearTimer = null;

        // Parameter sliders.
        this._paramDefs = [
            { id: 'post-silence', key: 'post_silence_ms', fmt: v => `${v}` },
            { id: 'max-buf',      key: 'max_buf_sec',     fmt: v => `${v}` },
            { id: 'min-dur',      key: 'min_dur_sec',     fmt: v => parseFloat(v).toFixed(1) },
            { id: 'pre-roll',     key: 'pre_roll_sec',    fmt: v => parseFloat(v).toFixed(1) },
            { id: 'max-tokens',   key: 'max_tokens',      fmt: v => `${v}` },
            { id: 'rep-penalty',  key: 'rep_penalty',     fmt: v => parseFloat(v).toFixed(2) },
            { id: 'min-energy',   key: 'min_energy',      fmt: v => parseFloat(v).toFixed(3) },
            { id: 'partial-sec',  key: 'partial_sec',     fmt: v => parseFloat(v).toFixed(1) },
            { id: 'speech-ratio', key: 'speech_ratio',    fmt: v => parseFloat(v).toFixed(2) },
        ];
        this._sliders = {};
        this._outputs = {};
        for (const def of this._paramDefs) {
            this._sliders[def.key] = this.el.querySelector(`#asr-p-${def.id}`);
            this._outputs[def.key] = this.el.querySelector(`#asr-v-${def.id}`);
        }
    }

    _bindControls() {
        this.toggleBtn.addEventListener('click', () => {
            const on = this.toggleBtn.getAttribute('aria-pressed') === 'true';
            this.ws.sendText(`asr_enable:${on ? 'off' : 'on'}`);
        });

        // ASR VAD source selector.
        if (this.asrVadSelect) {
            this.asrVadSelect.addEventListener('change', () => {
                this.ws.sendText(`asr_vad_source:${this.asrVadSelect.value}`);
            });
        }


        // Bind parameter sliders — send on change (mouseup / touchend).
        for (const def of this._paramDefs) {
            const slider = this._sliders[def.key];
            const output = this._outputs[def.key];
            slider.addEventListener('input', () => {
                output.textContent = def.fmt(slider.value);
            });
            slider.addEventListener('change', () => {
                output.textContent = def.fmt(slider.value);
                this.ws.sendText(`asr_param:${def.key}:${slider.value}`);
            });
        }
    }

    // Called when backend sends asr_transcript.
    onTranscript(obj) {
        this.totalCount++;
        this.totalLatency += obj.latency_ms || 0;
        this.totalAudioSec += obj.audio_sec || 0;

        this._latencyHistory.push(obj.latency_ms || 0);
        if (this._latencyHistory.length > this._latencyMax) {
            this._latencyHistory.shift();
        }

        this._updateStats();
        this._drawChart();

        // Clear partial on final transcript.
        if (this.partialTextEl) this.partialTextEl.textContent = '—';
    }

    // Called when backend sends asr_partial (streaming partial transcription).
    onPartial(obj) {
        if (this.partialTextEl && obj.text) {
            this.partialTextEl.textContent = obj.text;
            // Auto-clear after 5s if no update.
            if (this._partialClearTimer) clearTimeout(this._partialClearTimer);
            this._partialClearTimer = setTimeout(() => {
                this.partialTextEl.textContent = '—';
            }, 5000);
        }
    }

    // Called on pipeline_stats for live enable/active state and params sync.
    onPipelineStats(obj) {
        if (obj.asr_enabled !== undefined) {
            const on = obj.asr_enabled;
            this.toggleBtn.classList.toggle('btn--active', on);
            this.toggleBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
        }
        if (obj.asr_loaded !== undefined) {
            const loaded = obj.asr_loaded;
            const enabled = obj.asr_enabled;
            if (!loaded) {
                this.statusEl.textContent = 'Not loaded';
                this.statusEl.className = 'asr-status asr-status--off';
            } else if (!enabled) {
                this.statusEl.textContent = 'Disabled';
                this.statusEl.className = 'asr-status asr-status--off';
            } else {
                this.statusEl.textContent = 'Active';
                this.statusEl.className = 'asr-status asr-status--on';
            }
        }

        // Buffer indicator.
        if (obj.asr_buf_sec !== undefined) {
            const maxSec = obj.asr_max_buf_sec || 30;
            const pct = Math.min(100, (obj.asr_buf_sec / maxSec) * 100);
            this.bufBar.style.width = `${pct}%`;
            this.bufBar.classList.toggle('asr-buf-bar--speech', !!obj.asr_buf_has_speech);
            this.bufLabel.textContent = `${obj.asr_buf_sec.toFixed(1)}s`;
        }

        // Sync slider values from server (first sync or after reconnect).
        this._syncParam('post_silence_ms', obj.asr_post_silence_ms);
        this._syncParam('max_buf_sec', obj.asr_max_buf_sec);
        this._syncParam('min_dur_sec', obj.asr_min_dur_sec);
        this._syncParam('pre_roll_sec', obj.asr_pre_roll_sec);
        this._syncParam('max_tokens', obj.asr_max_tokens);
        this._syncParam('rep_penalty', obj.asr_rep_penalty);
        this._syncParam('min_energy', obj.asr_min_energy);
        this._syncParam('partial_sec', obj.asr_partial_sec);
        this._syncParam('speech_ratio', obj.asr_min_speech_ratio);

        // Sync ASR VAD source selector.
        if (obj.asr_vad_source !== undefined && this.asrVadSelect &&
            document.activeElement !== this.asrVadSelect) {
            const map = {0:'silero', 1:'fsmn', 2:'ten', 3:'any', 4:'direct'};
            this.asrVadSelect.value = map[obj.asr_vad_source] || 'silero';
        }
    }

    _syncParam(key, value) {
        if (value === undefined) return;
        const slider = this._sliders[key];
        const output = this._outputs[key];
        if (!slider || !output) return;
        // Only sync if user is not actively dragging.
        if (document.activeElement === slider) return;
        slider.value = value;
        const def = this._paramDefs.find(d => d.key === key);
        output.textContent = def ? def.fmt(value) : value;
    }

    // Called when backend confirms asr_enable toggle.
    onAsrEnable(obj) {
        const on = obj.enabled;
        this.toggleBtn.classList.toggle('btn--active', on);
        this.toggleBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
        this.statusEl.textContent = on ? 'Active' : 'Disabled';
        this.statusEl.className = on ? 'asr-status asr-status--on' : 'asr-status asr-status--off';
    }

    // Called when backend confirms asr_param set.
    onAsrParam(obj) {
        if (obj.key && obj.value !== undefined) {
            this._syncParam(obj.key, obj.value);
        }
    }

    _updateStats() {
        this.countEl.textContent = this.totalCount;
        if (this.totalCount > 0) {
            const avgLat = this.totalLatency / this.totalCount;
            const avgRtf = this.totalAudioSec > 0 ?
                (this.totalLatency / 1000) / this.totalAudioSec : 0;
            this.avgLatEl.textContent = `${avgLat.toFixed(0)} ms`;
            this.avgRtfEl.textContent = avgRtf.toFixed(3);
        } else {
            this.avgLatEl.textContent = '—';
            this.avgRtfEl.textContent = '—';
        }
        if (this._latencyHistory.length > 0) {
            const last = this._latencyHistory[this._latencyHistory.length - 1];
            this.lastLatEl.textContent = `${last.toFixed(0)} ms`;
        } else {
            this.lastLatEl.textContent = '—';
        }
    }

    _drawChart() {
        const c = this.canvas;
        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        const w = c.clientWidth;
        const h = c.clientHeight;
        if (c.width !== w * dpr || c.height !== h * dpr) {
            c.width = w * dpr;
            c.height = h * dpr;
            ctx.scale(dpr, dpr);
        }
        ctx.clearRect(0, 0, w, h);

        const values = this._latencyHistory;
        if (values.length < 2) return;

        // Find max latency for Y scale.
        let maxLat = 0;
        for (const v of values) if (v > maxLat) maxLat = v;
        maxLat = Math.max(maxLat * 1.1, 100);

        const pad = { l: 40, r: 8, t: 8, b: 16 };
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;

        // Grid lines.
        ctx.strokeStyle = '#30363d';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + plotH * (1 - i / 4);
            ctx.beginPath();
            ctx.moveTo(pad.l, y);
            ctx.lineTo(pad.l + plotW, y);
            ctx.stroke();
        }

        // Y axis labels.
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + plotH * (1 - i / 4);
            const val = (maxLat * i / 4);
            ctx.fillText(val >= 1000 ? `${(val/1000).toFixed(1)}s` : `${val.toFixed(0)}`, pad.l - 4, y);
        }

        // Latency bars.
        const barW = Math.max(2, (plotW / values.length) - 1);
        for (let i = 0; i < values.length; i++) {
            const lat = values[i];
            const x = pad.l + (i / values.length) * plotW;
            const barH = (lat / maxLat) * plotH;
            const y = pad.t + plotH - barH;
            ctx.fillStyle = lat > 5000 ? '#f85149' :
                            lat > 2000 ? '#d29922' : '#58a6ff';
            ctx.fillRect(x, y, barW, barH);
        }

        // RTF=1.0 reference line (latency == avg_audio_sec * 1000).
        if (this.totalCount > 0 && this.totalAudioSec > 0) {
            const avgAudioMs = (this.totalAudioSec / this.totalCount) * 1000;
            ctx.strokeStyle = '#3fb950';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            const refY = pad.t + plotH * (1 - avgAudioMs / maxLat);
            if (refY > pad.t && refY < pad.t + plotH) {
                ctx.moveTo(pad.l, refY);
                ctx.lineTo(pad.l + plotW, refY);
                ctx.stroke();
                ctx.fillStyle = '#3fb950';
                ctx.textAlign = 'left';
                ctx.fillText('RTF=1', pad.l + 2, refY - 4);
            }
            ctx.setLineDash([]);
        }
    }

    _escapeHtml(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
                .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
}
