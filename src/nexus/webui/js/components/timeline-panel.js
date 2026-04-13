// timeline-panel.js — Wrapping time-aligned visualization.
// 4 independent lanes per time row: VAD, ASR, SAAS, Tracker.
// Time wraps to next row at panel edge — no horizontal scrolling.
// Container scrolls vertically, auto-follows latest data.

const SPEAKER_COLORS = [
    '#5caaef', '#ef6c5c', '#5cef8a', '#efcf5c', '#c77dff',
    '#ff8fab', '#4ecdc4', '#ffa62f', '#9d65c9', '#00d2ff',
];
const TRACKER_COLORS = [
    '#3d8abf', '#bf4d3d', '#3dbf6a', '#bf9f3d', '#9d5ddf',
    '#df6f8f', '#2eada4', '#df861f', '#7d45a9', '#00b2df',
];
const UNKNOWN_COLOR = '#444';

const SOURCE_COLORS = {
    SAAS_FULL:    '#3fb950',
    SAAS_EARLY:   '#58a6ff',
    SAAS_CHANGE:  '#d29922',
    SAAS_INHERIT: '#8b949e',
    TRACKER:      '#b48ead',
    SNAPSHOT:     '#555',
};

// Lane heights (px)
const LH = {
    time:    14,
    vad:     12,
    asr:     24,
    saas:    16,
    tracker: 16,
    gap:      4,
};
const ROW_H = LH.time + LH.vad + LH.asr + LH.saas + LH.tracker + LH.gap;
const LABEL_W = 30;            // left margin for lane labels
const DEFAULT_PX_SEC = 40;     // pixels per second

export class TimelinePanel {
    constructor() {
        this.el = document.getElementById('timeline-panel');
        // Data
        this.vadIntervals  = [];  // {start, end}
        this.asrSegments   = [];  // {start, end, text, trigger, latency, ...}
        this.saasSegments  = [];  // {start, end, spkId, spkName, spkSim, spkConf, spkSrc}
        this.trackerSpans  = [];  // {start, end, trkId, trkName, state, sim}
        this.trackerEvents = [];  // {time, type, oldId, newId, name, sim, state}
        this.maxEntries    = 800;
        // VAD state tracking from pipeline_stats
        this._vadSpeech    = false;
        this._vadStart     = 0;
        this._lastStreamSec = 0;
        // Tracker state tracking from pipeline_stats
        this._trkPrevId    = -2;  // -2 = no data yet
        this._trkPrevState = -1;
        this._trkPrevSwitches = 0;
        this._trkSpanStart = 0;
        // View
        this.pxPerSec      = DEFAULT_PX_SEC;
        this.autoScroll    = true;
        this._dirty        = true;
        this._build();
        this._bindEvents();
        this._scheduleRender();
    }

    // ─── DOM ───

    _build() {
        this.el.innerHTML = `
            <h2 class="panel__title">Timeline</h2>
            <div class="tl-controls">
                <button id="tl-follow" class="btn btn--vad btn--active btn--sm" aria-pressed="true">Follow</button>
                <button id="tl-zoom-in" class="btn btn--vad btn--sm">+</button>
                <button id="tl-zoom-out" class="btn btn--vad btn--sm">−</button>
                <span class="tl-info" id="tl-info">0 seg | 40 px/s</span>
            </div>
            <div class="tl-scroll" id="tl-scroll">
                <canvas id="tl-canvas" class="tl-canvas"></canvas>
            </div>
            <div class="tl-tooltip" id="tl-tooltip" hidden></div>
            <div class="tl-legend">
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#e5534b"></span>VAD Speech</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#3fb950"></span>Full</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#58a6ff"></span>Early</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#d29922"></span>Change</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#8b949e"></span>Inherit</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#b48ead"></span>Tracker</span>
                <span class="tl-legend-item"><span style="color:#e5534b;font-weight:bold">⬥</span> Spk Change</span>
                <span class="tl-legend-item"><span style="color:#3fb950;font-weight:bold">▲</span> Register</span>
                <span class="tl-legend-item"><span style="color:#d29922;font-weight:bold">✕</span> Overlap</span>
            </div>
        `;
        this.scrollEl  = this.el.querySelector('#tl-scroll');
        this.canvas    = this.el.querySelector('#tl-canvas');
        this.ctx       = this.canvas.getContext('2d');
        this.tooltipEl = this.el.querySelector('#tl-tooltip');
        this.infoEl    = this.el.querySelector('#tl-info');
        this.followBtn = this.el.querySelector('#tl-follow');
        this.zoomInBtn = this.el.querySelector('#tl-zoom-in');
        this.zoomOutBtn= this.el.querySelector('#tl-zoom-out');
    }

    _bindEvents() {
        this.followBtn.addEventListener('click', () => {
            this.autoScroll = !this.autoScroll;
            this.followBtn.classList.toggle('btn--active', this.autoScroll);
            this.followBtn.setAttribute('aria-pressed', this.autoScroll);
        });
        this.zoomInBtn.addEventListener('click', () => {
            this.pxPerSec = Math.min(this.pxPerSec * 1.5, 200);
            this._dirty = true;
        });
        this.zoomOutBtn.addEventListener('click', () => {
            this.pxPerSec = Math.max(this.pxPerSec / 1.5, 8);
            this._dirty = true;
        });

        // Scroll → disable auto-follow on manual scroll up
        this.scrollEl.addEventListener('scroll', () => {
            const el = this.scrollEl;
            const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
            if (!atBottom && this.autoScroll) {
                this.autoScroll = false;
                this.followBtn.classList.remove('btn--active');
                this.followBtn.setAttribute('aria-pressed', false);
            }
        });

        // Hover for tooltip
        this.canvas.addEventListener('mousemove', (e) => this._onHover(e));
        this.canvas.addEventListener('mouseleave', () => { this.tooltipEl.hidden = true; });

        // Resize
        this._resizeObs = new ResizeObserver(() => { this._dirty = true; });
        this._resizeObs.observe(this.el);
    }

    // ─── Data Input ───

    onVad(obj) {
        // From 'vad' WS messages — no absolute time, so we use _lastStreamSec
        // as approximate time. This is supplementary to pipeline_stats tracking.
    }

    onPipelineStats(obj) {
        const streamSec = (obj.pcm_samples || 0) / 16000;
        if (streamSec <= 0) return;
        this._lastStreamSec = streamSec;

        // Track VAD speech intervals from is_speech transitions
        const speech = !!obj.is_speech;
        if (speech && !this._vadSpeech) {
            // Speech start
            this._vadStart = streamSec;
            this._vadSpeech = true;
        } else if (!speech && this._vadSpeech) {
            // Speech end
            this.vadIntervals.push({ start: this._vadStart, end: streamSec });
            if (this.vadIntervals.length > this.maxEntries) this.vadIntervals.shift();
            this._vadSpeech = false;
            this._dirty = true;
        }
        // Extend current speech interval visually
        if (this._vadSpeech) this._dirty = true;

        // Track SpeakerTracker from pipeline_stats
        if (obj.tracker_enabled && obj.tracker_check_active) {
            const trkId    = obj.tracker_spk_id ?? -1;
            const trkName  = obj.tracker_spk_name || '';
            const trkState = obj.tracker_state ?? 0;
            const trkSim   = obj.tracker_sim_to_ref || 0;
            const switches = obj.tracker_switches || 0;

            // Detect speaker change (switches increment or ID change)
            if (this._trkPrevId !== -2) {
                if (switches > this._trkPrevSwitches) {
                    // Speaker change event
                    this.trackerEvents.push({
                        time: streamSec, type: 'change',
                        oldId: this._trkPrevId, newId: trkId,
                        name: trkName, sim: trkSim, state: trkState,
                    });
                    if (this.trackerEvents.length > this.maxEntries) this.trackerEvents.shift();
                }
                if (trkState === 3 && this._trkPrevState !== 3) {
                    // OVERLAP start
                    this.trackerEvents.push({
                        time: streamSec, type: 'overlap',
                        oldId: this._trkPrevId, newId: trkId,
                        name: '', sim: trkSim, state: 3,
                    });
                    if (this.trackerEvents.length > this.maxEntries) this.trackerEvents.shift();
                }
            }

            // Registration event
            if (obj.tracker_reg_event) {
                this.trackerEvents.push({
                    time: streamSec, type: 'register',
                    oldId: -1, newId: obj.tracker_reg_id ?? trkId,
                    name: obj.tracker_reg_name || trkName, sim: trkSim, state: trkState,
                });
                if (this.trackerEvents.length > this.maxEntries) this.trackerEvents.shift();
            }

            // Build continuous spans — close old span on ID or state change
            if (trkId !== this._trkPrevId || trkState !== this._trkPrevState) {
                if (this._trkPrevId !== -2) {
                    this.trackerSpans.push({
                        start: this._trkSpanStart, end: streamSec,
                        trkId: this._trkPrevId, trkName: this._trkPrevName || '',
                        state: this._trkPrevState, sim: this._trkPrevSim || 0,
                    });
                    if (this.trackerSpans.length > this.maxEntries) this.trackerSpans.shift();
                }
                this._trkSpanStart = streamSec;
            }

            this._trkPrevId = trkId;
            this._trkPrevName = trkName;
            this._trkPrevState = trkState;
            this._trkPrevSim = trkSim;
            this._trkPrevSwitches = switches;
            this._dirty = true;
        }
    }

    onTranscript(obj) {
        const start = obj.stream_start_sec || 0;
        const end   = obj.stream_end_sec || 0;
        if (end <= 0) return;

        // ASR segment
        const seg = {
            start, end,
            text:    obj.text || '',
            trigger: obj.trigger || '',
            latency: obj.latency_ms || 0,
            audio:   obj.audio_sec || 0,
        };
        // Merge buffer_full continuations
        if (seg.trigger === 'buffer_full' && this.asrSegments.length > 0) {
            const prev = this.asrSegments[this.asrSegments.length - 1];
            if (prev.end >= seg.start - 0.5) {
                prev.end = seg.end;
                prev.text += seg.text;
                this._dirty = true;
                return; // don't duplicate SAAS/tracker for merged segs
            }
        }
        this.asrSegments.push(seg);
        if (this.asrSegments.length > this.maxEntries) this.asrSegments.shift();

        // SAAS segment
        this.saasSegments.push({
            start, end,
            spkId:   obj.speaker_id ?? -1,
            spkName: obj.speaker_name || '',
            spkSim:  obj.speaker_sim || 0,
            spkConf: obj.speaker_confidence || 0,
            spkSrc:  obj.speaker_source || '',
        });
        if (this.saasSegments.length > this.maxEntries) this.saasSegments.shift();

        // (Tracker segments now come from pipeline_stats continuous tracking,
        //  not from asr_transcript snapshots)

        this._dirty = true;
    }

    // ─── Rendering ───

    _scheduleRender() {
        requestAnimationFrame(() => {
            if (this._dirty) this._render();
            this._scheduleRender();
        });
    }

    _render() {
        this._dirty = false;
        const dpr = window.devicePixelRatio || 1;
        const containerW = this.scrollEl.clientWidth;
        if (containerW <= 0) return;

        const W = containerW;                // logical width
        const usableW = W - LABEL_W;         // width for time data
        const secsPerRow = usableW / this.pxPerSec;

        // Determine total time span
        let maxT = this._lastStreamSec;
        if (this.asrSegments.length) maxT = Math.max(maxT, this.asrSegments[this.asrSegments.length - 1].end);
        if (this._vadSpeech) maxT = Math.max(maxT, this._lastStreamSec);
        if (maxT <= 0) maxT = 1;

        const numRows = Math.max(1, Math.ceil(maxT / secsPerRow));
        const H = numRows * ROW_H;

        // Resize canvas
        this.canvas.width  = W * dpr;
        this.canvas.height = H * dpr;
        this.canvas.style.width  = W + 'px';
        this.canvas.style.height = H + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const c = this.ctx;

        // Store layout info for hover
        this._layoutW = W;
        this._secsPerRow = secsPerRow;
        this._numRows = numRows;

        // Clear
        c.fillStyle = '#0d1117';
        c.fillRect(0, 0, W, H);

        // Draw each row
        for (let r = 0; r < numRows; r++) {
            const rowY  = r * ROW_H;
            const tStart = r * secsPerRow;
            const tEnd   = tStart + secsPerRow;

            this._drawRow(c, W, rowY, tStart, tEnd, secsPerRow);
        }

        // Info
        this.infoEl.textContent = `${this.asrSegments.length} seg | ${this.pxPerSec.toFixed(0)} px/s | ${maxT.toFixed(0)}s`;

        // Auto-scroll to bottom
        if (this.autoScroll) {
            this.scrollEl.scrollTop = this.scrollEl.scrollHeight;
        }
    }

    _drawRow(c, W, y, tStart, tEnd, secsPerRow) {
        const usableW = W - LABEL_W;
        const x0 = LABEL_W;

        // Helper: time → x coordinate within this row
        const tx = (t) => x0 + ((t - tStart) / secsPerRow) * usableW;
        const tw = (dur) => (dur / secsPerRow) * usableW;

        // ── Time axis ──
        const tyTop = y;
        c.fillStyle = '#111518';
        c.fillRect(0, tyTop, W, LH.time);

        // Tick marks
        let tickInt = 1;
        if (secsPerRow > 120) tickInt = 10;
        else if (secsPerRow > 60) tickInt = 5;
        else if (secsPerRow > 20) tickInt = 2;

        const firstTick = Math.ceil(tStart / tickInt) * tickInt;
        c.strokeStyle = '#30363d';
        c.fillStyle = '#8b949e';
        c.font = '9px monospace';
        c.textBaseline = 'bottom';
        c.lineWidth = 1;
        for (let t = firstTick; t <= tEnd; t += tickInt) {
            const px = tx(t);
            c.beginPath();
            c.moveTo(px, tyTop + LH.time - 3);
            c.lineTo(px, tyTop + LH.time);
            c.stroke();
            const min = Math.floor(t / 60);
            const sec = Math.floor(t % 60);
            const label = min > 0 ? `${min}:${sec.toString().padStart(2, '0')}` : `${sec}s`;
            c.fillText(label, px + 1, tyTop + LH.time - 1);
        }

        // ── Lane backgrounds + labels ──
        const lanes = [
            { name: 'VAD', y: tyTop + LH.time,                       h: LH.vad },
            { name: 'ASR', y: tyTop + LH.time + LH.vad,              h: LH.asr },
            { name: 'SPK', y: tyTop + LH.time + LH.vad + LH.asr,    h: LH.saas },
            { name: 'TRK', y: tyTop + LH.time + LH.vad + LH.asr + LH.saas, h: LH.tracker },
        ];
        for (const lane of lanes) {
            c.fillStyle = '#161b22';
            c.fillRect(x0, lane.y, usableW, lane.h);
            c.fillStyle = '#484f58';
            c.font = '8px monospace';
            c.textBaseline = 'middle';
            c.fillText(lane.name, 2, lane.y + lane.h / 2);
        }

        // ── Row border ──
        c.strokeStyle = '#21262d';
        c.beginPath();
        c.moveTo(0, y + ROW_H - 1);
        c.lineTo(W, y + ROW_H - 1);
        c.stroke();

        // ── VAD lane ──
        const vadY = lanes[0].y;
        const vadH = lanes[0].h;
        // Completed intervals
        for (const iv of this.vadIntervals) {
            if (iv.end < tStart || iv.start > tEnd) continue;
            const px1 = Math.max(x0, tx(Math.max(iv.start, tStart)));
            const px2 = Math.min(W, tx(Math.min(iv.end, tEnd)));
            if (px2 - px1 < 0.5) continue;
            c.fillStyle = '#e5534b';
            c.globalAlpha = 0.65;
            c.fillRect(px1, vadY + 1, px2 - px1, vadH - 2);
            c.globalAlpha = 1;
        }
        // In-progress speech
        if (this._vadSpeech && this._lastStreamSec > tStart && this._vadStart < tEnd) {
            const px1 = Math.max(x0, tx(Math.max(this._vadStart, tStart)));
            const px2 = Math.min(W, tx(Math.min(this._lastStreamSec, tEnd)));
            if (px2 - px1 >= 0.5) {
                c.fillStyle = '#e5534b';
                c.globalAlpha = 0.45;
                c.fillRect(px1, vadY + 1, px2 - px1, vadH - 2);
                c.globalAlpha = 1;
            }
        }

        // ── ASR lane ──
        const asrY = lanes[1].y;
        const asrH = lanes[1].h;
        for (const seg of this.asrSegments) {
            if (seg.end < tStart || seg.start > tEnd) continue;
            const px1 = Math.max(x0, tx(Math.max(seg.start, tStart)));
            const px2 = Math.min(W, tx(Math.min(seg.end, tEnd)));
            if (px2 - px1 < 1) continue;
            c.fillStyle = '#3a7dd8';
            c.globalAlpha = 0.6;
            c.fillRect(px1, asrY + 1, px2 - px1, asrH - 2);
            c.globalAlpha = 1;

            // Trigger indicator — thin colored line at bottom
            const trigColor = seg.trigger === 'silence' ? '#3fb950'
                            : seg.trigger === 'spk_change' ? '#d29922'
                            : seg.trigger === 'buffer_full' ? '#e5534b' : '#484f58';
            c.fillStyle = trigColor;
            c.fillRect(px1, asrY + asrH - 3, px2 - px1, 2);

            // Text
            if (px2 - px1 > 20) {
                c.save();
                c.beginPath();
                c.rect(px1 + 1, asrY, px2 - px1 - 2, asrH);
                c.clip();
                c.fillStyle = '#fff';
                c.font = '10px sans-serif';
                c.textBaseline = 'middle';
                const maxChars = Math.floor((px2 - px1 - 4) / 5.5);
                const txt = seg.text.length > maxChars ? seg.text.slice(0, maxChars) + '…' : seg.text;
                c.fillText(txt, px1 + 2, asrY + asrH / 2);
                c.restore();
            }
        }

        // ── SAAS lane ──
        const spkY = lanes[2].y;
        const spkH = lanes[2].h;
        for (const seg of this.saasSegments) {
            if (seg.end < tStart || seg.start > tEnd) continue;
            const px1 = Math.max(x0, tx(Math.max(seg.start, tStart)));
            const px2 = Math.min(W, tx(Math.min(seg.end, tEnd)));
            if (px2 - px1 < 1) continue;

            const color = seg.spkId >= 0 ? SPEAKER_COLORS[seg.spkId % SPEAKER_COLORS.length] : UNKNOWN_COLOR;
            c.fillStyle = color;
            c.globalAlpha = 0.7;
            c.fillRect(px1, spkY + 1, px2 - px1, spkH - 2);
            c.globalAlpha = 1;

            // Source stripe at top
            const srcColor = SOURCE_COLORS[seg.spkSrc] || '#555';
            c.fillStyle = srcColor;
            c.fillRect(px1, spkY + 1, px2 - px1, 2);

            // Name
            if (px2 - px1 > 24) {
                c.save();
                c.beginPath();
                c.rect(px1, spkY, px2 - px1, spkH);
                c.clip();
                c.fillStyle = '#fff';
                c.font = '9px monospace';
                c.textBaseline = 'middle';
                const label = seg.spkName || (seg.spkId >= 0 ? `S${seg.spkId}` : '?');
                c.fillText(label, px1 + 2, spkY + spkH / 2 + 1);
                c.restore();
            }
        }

        // ── Tracker lane (continuous spans from pipeline_stats) ──
        const trkY = lanes[3].y;
        const trkH = lanes[3].h;

        const STATE_NAMES = ['SIL', 'TRK', 'TRANS', 'OVLP', 'UNK'];
        const STATE_ALPHA = [0.2, 0.55, 0.4, 0.5, 0.35];

        // Draw completed spans
        const allTrkSpans = [...this.trackerSpans];
        // Add in-progress span
        if (this._trkPrevId !== -2 && this._lastStreamSec > this._trkSpanStart) {
            allTrkSpans.push({
                start: this._trkSpanStart, end: this._lastStreamSec,
                trkId: this._trkPrevId, trkName: this._trkPrevName || '',
                state: this._trkPrevState, sim: this._trkPrevSim || 0,
            });
        }
        for (const seg of allTrkSpans) {
            if (seg.end < tStart || seg.start > tEnd) continue;
            const px1 = Math.max(x0, tx(Math.max(seg.start, tStart)));
            const px2 = Math.min(W, tx(Math.min(seg.end, tEnd)));
            if (px2 - px1 < 1) continue;

            const st = seg.state || 0;
            const color = seg.trkId >= 0 ? TRACKER_COLORS[seg.trkId % TRACKER_COLORS.length] : UNKNOWN_COLOR;
            c.fillStyle = color;
            c.globalAlpha = STATE_ALPHA[st] || 0.4;
            c.fillRect(px1, trkY + 1, px2 - px1, trkH - 2);
            c.globalAlpha = 1;

            // OVERLAP: hatched overlay
            if (st === 3) {
                c.strokeStyle = '#e5534b';
                c.lineWidth = 1;
                c.globalAlpha = 0.6;
                for (let hx = px1; hx < px2; hx += 6) {
                    c.beginPath();
                    c.moveTo(hx, trkY + 1);
                    c.lineTo(hx + 3, trkY + trkH - 1);
                    c.stroke();
                }
                c.globalAlpha = 1;
            }
            // UNKNOWN: dotted top border
            if (st === 4) {
                c.strokeStyle = '#d29922';
                c.lineWidth = 1;
                c.setLineDash([3, 3]);
                c.beginPath();
                c.moveTo(px1, trkY + 1);
                c.lineTo(px2, trkY + 1);
                c.stroke();
                c.setLineDash([]);
            }

            // Label
            if (px2 - px1 > 24) {
                c.save();
                c.beginPath();
                c.rect(px1, trkY, px2 - px1, trkH);
                c.clip();
                c.fillStyle = '#fff';
                c.font = '9px monospace';
                c.textBaseline = 'middle';
                const name = seg.trkName || (seg.trkId >= 0 ? `T${seg.trkId}` : STATE_NAMES[st] || '?');
                c.fillText(name, px1 + 2, trkY + trkH / 2);
                c.restore();
            }
        }

        // ── Tracker events (change markers, registration markers) ──
        for (const ev of this.trackerEvents) {
            if (ev.time < tStart || ev.time > tEnd) continue;
            const px = tx(ev.time);

            if (ev.type === 'change') {
                // Vertical red line spanning all lanes for visibility
                c.strokeStyle = '#e5534b';
                c.lineWidth = 2;
                c.globalAlpha = 0.8;
                c.beginPath();
                c.moveTo(px, trkY);
                c.lineTo(px, trkY + trkH);
                c.stroke();
                c.globalAlpha = 1;
                // Diamond marker
                c.fillStyle = '#e5534b';
                c.beginPath();
                c.moveTo(px, trkY);
                c.lineTo(px + 4, trkY + trkH / 2);
                c.lineTo(px, trkY + trkH);
                c.lineTo(px - 4, trkY + trkH / 2);
                c.closePath();
                c.fill();
            } else if (ev.type === 'register') {
                // Green triangle-up marker
                c.fillStyle = '#3fb950';
                c.beginPath();
                c.moveTo(px, trkY + 1);
                c.lineTo(px + 4, trkY + trkH - 1);
                c.lineTo(px - 4, trkY + trkH - 1);
                c.closePath();
                c.fill();
            } else if (ev.type === 'overlap') {
                // Orange X marker
                c.strokeStyle = '#d29922';
                c.lineWidth = 2;
                c.beginPath();
                c.moveTo(px - 3, trkY + 2);
                c.lineTo(px + 3, trkY + trkH - 2);
                c.moveTo(px + 3, trkY + 2);
                c.lineTo(px - 3, trkY + trkH - 2);
                c.stroke();
            }
        }
    }

    // ─── Hover tooltip ───

    _onHover(e) {
        if (!this._secsPerRow || this._secsPerRow <= 0) return;
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const mx = (e.clientX - rect.left);
        const my = (e.clientY - rect.top);

        // Which row?
        const row = Math.floor(my / ROW_H);
        if (row < 0 || row >= this._numRows) { this.tooltipEl.hidden = true; return; }

        const rowY = row * ROW_H;
        const localY = my - rowY;
        const tStart = row * this._secsPerRow;

        // Which lane?
        let lane = '';
        const vadTop  = LH.time;
        const asrTop  = vadTop + LH.vad;
        const spkTop  = asrTop + LH.asr;
        const trkTop  = spkTop + LH.saas;
        const trkEnd  = trkTop + LH.tracker;
        if (localY >= vadTop && localY < asrTop) lane = 'vad';
        else if (localY >= asrTop && localY < spkTop) lane = 'asr';
        else if (localY >= spkTop && localY < trkTop) lane = 'saas';
        else if (localY >= trkTop && localY < trkEnd) lane = 'tracker';
        else { this.tooltipEl.hidden = true; return; }

        // Time at mouse
        if (mx < LABEL_W) { this.tooltipEl.hidden = true; return; }
        const usableW = this._layoutW - LABEL_W;
        const timeSec = tStart + ((mx - LABEL_W) / usableW) * this._secsPerRow;

        let html = '';
        if (lane === 'vad') {
            // Check VAD intervals
            let found = this.vadIntervals.find(iv => iv.start <= timeSec && iv.end >= timeSec);
            if (!found && this._vadSpeech && timeSec >= this._vadStart && timeSec <= this._lastStreamSec) {
                found = { start: this._vadStart, end: this._lastStreamSec };
            }
            if (found) {
                html = `<strong>VAD Speech</strong><br>${found.start.toFixed(1)}s – ${found.end.toFixed(1)}s (${(found.end - found.start).toFixed(1)}s)`;
            }
        } else if (lane === 'asr') {
            const seg = this.asrSegments.find(s => s.start <= timeSec && s.end >= timeSec);
            if (seg) {
                html = `<strong>ASR</strong> ${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s (${(seg.end - seg.start).toFixed(1)}s)<br>`
                     + `"${this._esc(seg.text.slice(0, 120))}"<br>`
                     + `Trigger: ${seg.trigger} | Latency: ${seg.latency.toFixed(0)}ms`;
            }
        } else if (lane === 'saas') {
            const seg = this.saasSegments.find(s => s.start <= timeSec && s.end >= timeSec);
            if (seg) {
                const name = seg.spkName || (seg.spkId >= 0 ? `S${seg.spkId}` : '?');
                html = `<strong>SAAS</strong> ${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s<br>`
                     + `${this._esc(name)} (sim=${seg.spkSim.toFixed(3)}, conf=${(seg.spkConf * 100).toFixed(0)}%)<br>`
                     + `Source: ${seg.spkSrc}`;
            }
        } else if (lane === 'tracker') {
            // Check continuous tracker spans
            const allSpans = [...this.trackerSpans];
            if (this._trkPrevId !== -2 && this._lastStreamSec > this._trkSpanStart) {
                allSpans.push({
                    start: this._trkSpanStart, end: this._lastStreamSec,
                    trkId: this._trkPrevId, trkName: this._trkPrevName || '',
                    state: this._trkPrevState, sim: this._trkPrevSim || 0,
                });
            }
            const span = allSpans.find(s => s.start <= timeSec && s.end >= timeSec);
            const STATE_LABELS = ['SILENCE', 'TRACKING', 'TRANSITION', 'OVERLAP', 'UNKNOWN'];
            if (span) {
                const name = span.trkName || (span.trkId >= 0 ? `T${span.trkId}` : '?');
                html = `<strong>Tracker</strong> ${span.start.toFixed(1)}s – ${span.end.toFixed(1)}s<br>`
                     + `${this._esc(name)} (sim=${span.sim.toFixed(3)})<br>`
                     + `State: ${STATE_LABELS[span.state] || span.state}`;
            }
            // Check for nearby events
            const nearEv = this.trackerEvents.find(e => Math.abs(e.time - timeSec) < 0.5);
            if (nearEv) {
                const evLabels = { change: '⬥ SPK CHANGE', register: '▲ NEW SPEAKER', overlap: '✕ OVERLAP' };
                html += `<br><span style="color:#e5534b">${evLabels[nearEv.type] || nearEv.type}</span>`;
                if (nearEv.type === 'change') html += ` (${nearEv.oldId}→${nearEv.newId})`;
                if (nearEv.name) html += ` ${this._esc(nearEv.name)}`;
            }
        }

        if (!html) { this.tooltipEl.hidden = true; return; }

        this.tooltipEl.innerHTML = html;
        this.tooltipEl.hidden = false;
        const parentRect = this.el.getBoundingClientRect();
        this.tooltipEl.style.left = (e.clientX - parentRect.left + 12) + 'px';
        this.tooltipEl.style.top  = (e.clientY - parentRect.top + 12) + 'px';
    }

    _esc(s) {
        if (!s) return '';
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
}
