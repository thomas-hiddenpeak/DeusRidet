// timeline-panel.js — Time-aligned visualization of ASR segments + speaker identification.
// X-axis = stream time (seconds). Two swim lanes:
//   1. ASR lane: colored blocks per transcript segment (color = SAAS speaker)
//   2. Speaker lane: colored blocks from pipeline_stats speaker events
// Hovering a block shows details. The view auto-scrolls to follow live data,
// but can be paused by clicking/dragging (manual pan).

const SPEAKER_COLORS = [
    '#5caaef', '#ef6c5c', '#5cef8a', '#efcf5c', '#c77dff',
    '#ff8fab', '#4ecdc4', '#ffa62f', '#9d65c9', '#00d2ff',
];
const TRACKER_COLORS = [
    '#3d8abf', '#bf4d3d', '#3dbf6a', '#bf9f3d', '#9d5ddf',
    '#df6f8f', '#2eada4', '#df861f', '#7d45a9', '#00b2df',
];
const UNKNOWN_COLOR = '#555';

// Source badge colors for the speaker lane
const SOURCE_COLORS = {
    SAAS_FULL:    '#3fb950',
    SAAS_EARLY:   '#58a6ff',
    SAAS_CHANGE:  '#d29922',
    SAAS_INHERIT: '#8b949e',
    TRACKER:      '#b48ead',
    SNAPSHOT:     '#555',
};

export class TimelinePanel {
    constructor() {
        this.el = document.getElementById('timeline-panel');
        // Data stores
        this.asrSegments = [];      // {start, end, text, spkId, spkName, spkSim, spkConf, spkSrc, trkId, trkName}
        this.spkEvents = [];        // {time, spkId, spkName, sim, source} from pipeline_stats
        this.maxSegments = 500;
        // View state
        this.viewStart = 0;         // visible start (seconds)
        this.viewEnd = 60;          // visible end (seconds)
        this.autoScroll = true;     // follow live data
        this.pixelsPerSec = 0;      // computed from canvas width
        this.dragState = null;
        this.hoverInfo = null;
        // Speaker name map (updated from pipeline_stats speaker_lists)
        this.speakerNames = {};     // id → name
        this._build();
        this._bindEvents();
        this._raf = null;
        this._scheduleRender();
    }

    _build() {
        this.el.innerHTML = `
            <h2 class="panel__title">Timeline</h2>
            <div class="tl-controls">
                <button id="tl-follow" class="btn btn--vad btn--active btn--sm" aria-pressed="true">Follow</button>
                <button id="tl-zoom-in" class="btn btn--vad btn--sm">+</button>
                <button id="tl-zoom-out" class="btn btn--vad btn--sm">−</button>
                <span class="tl-range" id="tl-range">0 – 60 s</span>
                <span class="tl-count" id="tl-count">0 segments</span>
            </div>
            <div class="tl-canvas-wrap">
                <canvas id="tl-canvas" class="tl-canvas"></canvas>
            </div>
            <div class="tl-tooltip" id="tl-tooltip" hidden></div>
            <div class="tl-legend">
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#3fb950"></span>SAAS Full</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#58a6ff"></span>SAAS Early</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#d29922"></span>SPK Change</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#8b949e"></span>Inherit</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#b48ead"></span>Tracker</span>
                <span class="tl-legend-item"><span class="tl-legend-dot" style="background:#555"></span>Snapshot/?</span>
            </div>
        `;
        this.canvas = this.el.querySelector('#tl-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.tooltipEl = this.el.querySelector('#tl-tooltip');
        this.rangeEl = this.el.querySelector('#tl-range');
        this.countEl = this.el.querySelector('#tl-count');
        this.followBtn = this.el.querySelector('#tl-follow');
        this.zoomInBtn = this.el.querySelector('#tl-zoom-in');
        this.zoomOutBtn = this.el.querySelector('#tl-zoom-out');
    }

    _bindEvents() {
        this.followBtn.addEventListener('click', () => {
            this.autoScroll = !this.autoScroll;
            this.followBtn.classList.toggle('btn--active', this.autoScroll);
            this.followBtn.setAttribute('aria-pressed', this.autoScroll);
        });
        this.zoomInBtn.addEventListener('click', () => this._zoom(0.5));
        this.zoomOutBtn.addEventListener('click', () => this._zoom(2.0));

        // Drag to pan
        this.canvas.addEventListener('mousedown', (e) => {
            this.dragState = { startX: e.offsetX, viewStart: this.viewStart, viewEnd: this.viewEnd };
            this.autoScroll = false;
            this.followBtn.classList.remove('btn--active');
            this.followBtn.setAttribute('aria-pressed', false);
        });
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.dragState) {
                const dx = e.offsetX - this.dragState.startX;
                const dtSec = -dx / this.pixelsPerSec;
                const span = this.dragState.viewEnd - this.dragState.viewStart;
                this.viewStart = this.dragState.viewStart + dtSec;
                this.viewEnd = this.viewStart + span;
                if (this.viewStart < 0) { this.viewStart = 0; this.viewEnd = span; }
            } else {
                this._updateHover(e.offsetX, e.offsetY);
            }
        });
        this.canvas.addEventListener('mouseup', () => { this.dragState = null; });
        this.canvas.addEventListener('mouseleave', () => {
            this.dragState = null;
            this.hoverInfo = null;
            this.tooltipEl.hidden = true;
        });

        // Wheel to zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 1.3 : 0.77;
            const rect = this.canvas.getBoundingClientRect();
            const frac = (e.clientX - rect.left) / rect.width;
            this._zoomAt(factor, frac);
        }, { passive: false });

        // Resize observer
        this._resizeObs = new ResizeObserver(() => this._resizeCanvas());
        this._resizeObs.observe(this.el);
        setTimeout(() => this._resizeCanvas(), 50);
    }

    _resizeCanvas() {
        const wrap = this.canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;
        const w = wrap.clientWidth;
        const h = 120;  // fixed height: 3 lanes (time axis + ASR + speaker)
        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    _zoom(factor) {
        this._zoomAt(factor, 0.5);
    }

    _zoomAt(factor, frac) {
        const span = this.viewEnd - this.viewStart;
        const newSpan = Math.min(Math.max(span * factor, 5), 3600);
        const pivot = this.viewStart + span * frac;
        this.viewStart = pivot - newSpan * frac;
        this.viewEnd = this.viewStart + newSpan;
        if (this.viewStart < 0) { this.viewStart = 0; this.viewEnd = newSpan; }
        this.autoScroll = false;
        this.followBtn.classList.remove('btn--active');
        this.followBtn.setAttribute('aria-pressed', false);
    }

    // --- Data input ---

    onTranscript(obj) {
        const seg = {
            start: obj.stream_start_sec || 0,
            end: obj.stream_end_sec || 0,
            text: obj.text || '',
            spkId: obj.speaker_id ?? -1,
            spkName: obj.speaker_name || '',
            spkSim: obj.speaker_sim || 0,
            spkConf: obj.speaker_confidence || 0,
            spkSrc: obj.speaker_source || '',
            trkId: obj.tracker_id ?? -1,
            trkName: obj.tracker_name || '',
            trigger: obj.trigger || '',
            latency: obj.latency_ms || 0,
        };
        // Merge buffer_full continuations
        if (seg.trigger === 'buffer_full' && this.asrSegments.length > 0) {
            const prev = this.asrSegments[this.asrSegments.length - 1];
            if (prev.spkId === seg.spkId && seg.start <= prev.end + 0.5) {
                prev.end = seg.end;
                prev.text += seg.text;
                return;
            }
        }
        this.asrSegments.push(seg);
        if (this.asrSegments.length > this.maxSegments) this.asrSegments.shift();

        // Auto-scroll
        if (this.autoScroll) {
            const span = this.viewEnd - this.viewStart;
            this.viewEnd = Math.max(seg.end + span * 0.1, this.viewEnd);
            this.viewStart = this.viewEnd - span;
            if (this.viewStart < 0) { this.viewStart = 0; this.viewEnd = span; }
        }
        this.countEl.textContent = `${this.asrSegments.length} segments`;
    }

    onPipelineStats(obj) {
        // Record speaker identification events from the pipeline
        const streamSec = (obj.pcm_samples || 0) / 16000;
        if (streamSec <= 0) return;

        // Update speaker names from speaker_lists
        if (obj.speaker_lists) {
            for (const group of obj.speaker_lists) {
                for (const spk of group.speakers) {
                    if (spk.name) this.speakerNames[spk.id] = spk.name;
                }
            }
        }

        // Only record WL-ECAPA events (the active system)
        if (obj.wlecapa_active && obj.wlecapa_id >= 0) {
            const last = this.spkEvents.length > 0 ? this.spkEvents[this.spkEvents.length - 1] : null;
            // Deduplicate: skip if same speaker and very close in time
            if (!last || last.spkId !== obj.wlecapa_id || streamSec - last.time > 0.3) {
                this.spkEvents.push({
                    time: streamSec,
                    spkId: obj.wlecapa_id,
                    spkName: obj.wlecapa_name || '',
                    sim: obj.wlecapa_sim || 0,
                    isEarly: obj.wlecapa_is_early || false,
                });
                if (this.spkEvents.length > 2000) this.spkEvents.shift();
            }
        }

        // Also record tracker events
        if (obj.tracker_check_active && obj.tracker_spk_id >= 0) {
            const last = this.spkEvents.length > 0 ? this.spkEvents[this.spkEvents.length - 1] : null;
            if (!last || last.trkId !== obj.tracker_spk_id || streamSec - last.time > 0.3) {
                // Store tracker separately by tagging
                // (We'll draw these in a different sub-lane)
            }
        }
    }

    // --- Rendering ---

    _scheduleRender() {
        this._raf = requestAnimationFrame(() => {
            this._render();
            this._scheduleRender();
        });
    }

    _render() {
        const c = this.ctx;
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const H = this.canvas.height / (window.devicePixelRatio || 1);
        if (W <= 0 || H <= 0) return;

        const span = this.viewEnd - this.viewStart;
        this.pixelsPerSec = W / span;

        // Clear
        c.fillStyle = '#0d1117';
        c.fillRect(0, 0, W, H);

        // Layout: 3 rows
        const timeH = 18;        // time axis
        const asrY = timeH;
        const asrH = 40;         // ASR transcript lane
        const spkY = asrY + asrH + 2;
        const spkH = 30;         // Speaker identification lane
        const trkY = spkY + spkH + 2;
        const trkH = H - trkY;   // Tracker lane

        // Draw time axis
        this._drawTimeAxis(c, W, timeH, span);

        // Lane backgrounds
        c.fillStyle = '#161b22';
        c.fillRect(0, asrY, W, asrH);
        c.fillRect(0, spkY, W, spkH);
        c.fillRect(0, trkY, W, trkH);

        // Lane labels
        c.fillStyle = '#8b949e';
        c.font = '10px monospace';
        c.textBaseline = 'middle';
        c.fillText('ASR', 2, asrY + asrH / 2);
        c.fillText('SPK', 2, spkY + spkH / 2);
        c.fillText('TRK', 2, trkY + trkH / 2);

        const labelW = 28;  // offset for lane labels

        // Draw ASR segments
        for (const seg of this.asrSegments) {
            if (seg.end < this.viewStart || seg.start > this.viewEnd) continue;
            const x0 = Math.max(labelW, (seg.start - this.viewStart) * this.pixelsPerSec);
            const x1 = Math.min(W, (seg.end - this.viewStart) * this.pixelsPerSec);
            if (x1 - x0 < 1) continue;

            // Color by SAAS speaker
            const color = seg.spkId >= 0 ? SPEAKER_COLORS[seg.spkId % SPEAKER_COLORS.length] : UNKNOWN_COLOR;
            c.fillStyle = color;
            c.globalAlpha = 0.7;
            c.fillRect(x0, asrY + 2, x1 - x0, asrH - 4);
            c.globalAlpha = 1.0;

            // Source indicator stripe at bottom of block
            const srcColor = SOURCE_COLORS[seg.spkSrc] || '#555';
            c.fillStyle = srcColor;
            c.fillRect(x0, asrY + asrH - 6, x1 - x0, 4);

            // Text label if wide enough
            if (x1 - x0 > 40) {
                c.fillStyle = '#fff';
                c.font = '10px sans-serif';
                c.textBaseline = 'middle';
                const label = seg.spkName || (seg.spkId >= 0 ? `S${seg.spkId}` : '?');
                c.save();
                c.beginPath();
                c.rect(x0 + 2, asrY, x1 - x0 - 4, asrH);
                c.clip();
                c.fillText(label, x0 + 3, asrY + 12);
                // Truncated transcript text
                if (x1 - x0 > 80) {
                    c.font = '9px sans-serif';
                    c.fillStyle = 'rgba(255,255,255,0.7)';
                    const maxChars = Math.floor((x1 - x0 - 6) / 5.5);
                    const txt = seg.text.length > maxChars ? seg.text.slice(0, maxChars) + '…' : seg.text;
                    c.fillText(txt, x0 + 3, asrY + 26);
                }
                c.restore();
            }
        }

        // Draw speaker detection events as colored spans
        // Group consecutive same-speaker events into blocks
        this._drawSpkLane(c, labelW, W, spkY, spkH, false);
        this._drawSpkLane(c, labelW, W, trkY, trkH, true);

        // Update range display
        this.rangeEl.textContent = `${this.viewStart.toFixed(0)} – ${this.viewEnd.toFixed(0)} s`;
    }

    _drawTimeAxis(c, W, H, span) {
        c.fillStyle = '#161b22';
        c.fillRect(0, 0, W, H);

        // Determine tick interval based on visible span
        let tickInterval;
        if (span <= 30) tickInterval = 1;
        else if (span <= 60) tickInterval = 5;
        else if (span <= 300) tickInterval = 10;
        else if (span <= 600) tickInterval = 30;
        else tickInterval = 60;

        const firstTick = Math.ceil(this.viewStart / tickInterval) * tickInterval;
        c.strokeStyle = '#30363d';
        c.fillStyle = '#8b949e';
        c.font = '9px monospace';
        c.textBaseline = 'bottom';
        c.lineWidth = 1;

        for (let t = firstTick; t <= this.viewEnd; t += tickInterval) {
            const x = (t - this.viewStart) * this.pixelsPerSec;
            // Tick line
            c.beginPath();
            c.moveTo(x, H - 4);
            c.lineTo(x, H);
            c.stroke();
            // Label
            const min = Math.floor(t / 60);
            const sec = Math.floor(t % 60);
            const label = min > 0 ? `${min}:${sec.toString().padStart(2, '0')}` : `${sec}s`;
            c.fillText(label, x + 2, H - 1);
        }
    }

    _drawSpkLane(c, labelW, W, y, h, isTracker) {
        // For the SPK lane: use ASR segments to draw speaker identity blocks
        // For the TRK lane: use tracker_id from ASR segments
        for (const seg of this.asrSegments) {
            if (seg.end < this.viewStart || seg.start > this.viewEnd) continue;
            const x0 = Math.max(labelW, (seg.start - this.viewStart) * this.pixelsPerSec);
            const x1 = Math.min(W, (seg.end - this.viewStart) * this.pixelsPerSec);
            if (x1 - x0 < 1) continue;

            const id = isTracker ? seg.trkId : seg.spkId;
            const name = isTracker
                ? (seg.trkName || (id >= 0 ? `T${id}` : '?'))
                : (seg.spkName || (id >= 0 ? `S${id}` : '?'));
            const palette = isTracker ? TRACKER_COLORS : SPEAKER_COLORS;
            const color = id >= 0 ? palette[id % palette.length] : UNKNOWN_COLOR;

            c.fillStyle = color;
            c.globalAlpha = isTracker ? 0.5 : 0.65;
            c.fillRect(x0, y + 2, x1 - x0, h - 4);
            c.globalAlpha = 1.0;

            if (!isTracker) {
                // Source color stripe at top
                const srcColor = SOURCE_COLORS[seg.spkSrc] || '#555';
                c.fillStyle = srcColor;
                c.fillRect(x0, y + 2, x1 - x0, 3);
            }

            // Label
            if (x1 - x0 > 30) {
                c.fillStyle = '#fff';
                c.font = '9px monospace';
                c.textBaseline = 'middle';
                c.save();
                c.beginPath();
                c.rect(x0 + 1, y, x1 - x0 - 2, h);
                c.clip();
                c.fillText(name, x0 + 3, y + h / 2);
                c.restore();
            }
        }
    }

    _updateHover(mx, my) {
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const timeH = 18;
        const asrY = timeH;
        const asrH = 40;

        // Only check ASR lane hover
        if (my < asrY || my > asrY + asrH) {
            this.tooltipEl.hidden = true;
            return;
        }

        const timeSec = this.viewStart + mx / this.pixelsPerSec;
        const seg = this.asrSegments.find(s => s.start <= timeSec && s.end >= timeSec);
        if (!seg) {
            this.tooltipEl.hidden = true;
            return;
        }

        const spkLabel = seg.spkId >= 0 ? `${seg.spkName || 'S' + seg.spkId} (sim=${seg.spkSim.toFixed(3)}, conf=${(seg.spkConf * 100).toFixed(0)}%, src=${seg.spkSrc})` : '?';
        const trkLabel = seg.trkId >= 0 ? `${seg.trkName || 'T' + seg.trkId}` : '?';
        this.tooltipEl.innerHTML =
            `<strong>${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s</strong> (${(seg.end - seg.start).toFixed(1)}s)<br>` +
            `SPK: ${this._esc(spkLabel)} | TRK: ${this._esc(trkLabel)}<br>` +
            `"${this._esc(seg.text.slice(0, 100))}"<br>` +
            `Latency: ${seg.latency.toFixed(0)}ms | Trigger: ${seg.trigger}`;
        this.tooltipEl.hidden = false;

        // Position tooltip
        const rect = this.canvas.getBoundingClientRect();
        const parentRect = this.el.getBoundingClientRect();
        this.tooltipEl.style.left = (mx + rect.left - parentRect.left + 10) + 'px';
        this.tooltipEl.style.top = (my + rect.top - parentRect.top + 10) + 'px';
    }

    _esc(s) {
        if (!s) return '';
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
}
