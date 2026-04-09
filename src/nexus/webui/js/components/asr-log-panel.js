// asr-log-panel.js — ASR detailed log panel.
// Shows trigger events, transcription timing breakdown, raw vs processed text,
// and per-stage latency for debugging ASR efficiency and accuracy.

export class AsrLogPanel {
    constructor(ws) {
        this.ws = ws;
        this.el = document.getElementById('asr-log-panel');
        this.logs = [];
        this.maxLogs = 200;
        this._build();
        this._bindControls();
    }

    _build() {
        this.el.innerHTML = `
            <h2 class="panel__title">ASR Logs</h2>
            <div class="asr-log-controls">
                <label class="asr-log-filter">
                    <input type="checkbox" id="asr-log-show-trigger" checked> Triggers
                </label>
                <label class="asr-log-filter">
                    <input type="checkbox" id="asr-log-show-result" checked> Results
                </label>
                <label class="asr-log-filter">
                    <input type="checkbox" id="asr-log-show-partial" checked> Partials
                </label>
                <label class="asr-log-filter">
                    <input type="checkbox" id="asr-log-show-empty" checked> Empty
                </label>
                <button id="asr-log-clear" class="btn btn--vad btn--sm">Clear</button>
                <span class="asr-log-count" id="asr-log-count">0 entries</span>
            </div>
            <div class="asr-log-list" id="asr-log-list" role="log" aria-live="polite"></div>
        `;
        this.listEl = this.el.querySelector('#asr-log-list');
        this.countEl = this.el.querySelector('#asr-log-count');
        this.showTrigger = this.el.querySelector('#asr-log-show-trigger');
        this.showResult = this.el.querySelector('#asr-log-show-result');
        this.showPartial = this.el.querySelector('#asr-log-show-partial');
        this.showEmpty = this.el.querySelector('#asr-log-show-empty');
        this.clearBtn = this.el.querySelector('#asr-log-clear');
    }

    _bindControls() {
        this.clearBtn.addEventListener('click', () => {
            this.logs = [];
            this.listEl.innerHTML = '';
            this._updateCount();
        });
        // Filter toggles re-render visible entries.
        for (const cb of [this.showTrigger, this.showResult, this.showPartial, this.showEmpty]) {
            cb.addEventListener('change', () => this._rerender());
        }
    }

    // Called when backend sends asr_log.
    onAsrLog(obj) {
        const ts = new Date().toLocaleTimeString('en-GB', { hour12: false, fractionalSecondDigits: 1 });
        const entry = { ts, ...obj };
        this.logs.push(entry);
        if (this.logs.length > this.maxLogs) this.logs.shift();
        this._appendRow(entry);
        this._updateCount();
    }

    // Also handle asr_transcript for enriched result display.
    onTranscript(obj) {
        const ts = new Date().toLocaleTimeString('en-GB', { hour12: false, fractionalSecondDigits: 1 });
        const entry = {
            ts,
            stage: 'transcript',
            text: obj.text,
            latency_ms: obj.latency_ms,
            audio_sec: obj.audio_sec,
            mel_ms: obj.mel_ms,
            encoder_ms: obj.encoder_ms,
            decode_ms: obj.decode_ms,
            tokens: obj.tokens,
            mel_frames: obj.mel_frames
        };
        this.logs.push(entry);
        if (this.logs.length > this.maxLogs) this.logs.shift();
        this._appendRow(entry);
        this._updateCount();
    }

    _shouldShow(entry) {
        if (entry.stage === 'trigger') return this.showTrigger.checked;
        if (entry.stage === 'skipped') return this.showEmpty.checked;
        if (entry.stage === 'partial') return this.showPartial.checked;
        if (entry.stage === 'result' && !entry.text && !entry.hallucinated) return this.showEmpty.checked;
        if (entry.stage === 'result' && entry.hallucinated) return this.showResult.checked;
        if (entry.stage === 'result' || entry.stage === 'transcript') return this.showResult.checked;
        return true;
    }

    _rerender() {
        this.listEl.innerHTML = '';
        for (const entry of this.logs) {
            if (this._shouldShow(entry)) {
                this._appendRow(entry, false);
            }
        }
    }

    _appendRow(entry, checkFilter = true) {
        if (checkFilter && !this._shouldShow(entry)) return;

        const row = document.createElement('div');
        row.className = `asr-log-row asr-log-row--${entry.stage}`;

        if (entry.stage === 'skipped') {
            row.className += ' asr-log-row--skipped';
            row.innerHTML = `
                <span class="asr-log-ts">${entry.ts}</span>
                <span class="asr-log-badge asr-log-badge--skipped">SKIP</span>
                <span class="asr-log-detail">
                    ${this._esc(entry.reason)} ·
                    buf=${entry.buf_sec != null ? entry.buf_sec.toFixed(2) + 's' : '?'}
                    speech=${entry.speech_sec != null ? entry.speech_sec.toFixed(2) + 's' : '?'}
                    (${entry.speech_ratio != null ? (entry.speech_ratio * 100).toFixed(0) : '?'}%)
                </span>
            `;
        } else if (entry.stage === 'trigger') {
            row.innerHTML = `
                <span class="asr-log-ts">${entry.ts}</span>
                <span class="asr-log-badge asr-log-badge--trigger">TRIGGER</span>
                <span class="asr-log-detail">
                    ${this._esc(entry.reason)} ·
                    buf=${entry.buf_sec != null ? entry.buf_sec.toFixed(2) + 's' : ''}
                    speech=${entry.speech_sec != null ? entry.speech_sec.toFixed(2) + 's' : ''}
                    (${entry.speech_ratio != null ? (entry.speech_ratio * 100).toFixed(0) + '%' : ''})
                    ${entry.samples != null ? entry.samples + ' samples' : ''}
                </span>
            `;
        } else if (entry.stage === 'partial') {
            row.className += ' asr-log-row--partial';
            const hasText = entry.text && entry.text.length > 0;
            row.innerHTML = `
                <span class="asr-log-ts">${entry.ts}</span>
                <span class="asr-log-badge asr-log-badge--partial">PARTIAL</span>
                <span class="asr-log-detail">
                    ${this._fmtMs(entry.total_ms)} · ${entry.tokens || 0}tok · ${entry.audio_sec != null ? entry.audio_sec.toFixed(2) + 's' : ''}
                    ${hasText ? `<span class="asr-log-text">${this._esc(entry.text)}</span>` : '<em>empty</em>'}
                </span>
            `;
        } else if (entry.stage === 'result') {
            const hasText = entry.text && entry.text.length > 0;
            const isHallucination = entry.hallucinated;
            const textChanged = entry.raw_text && entry.text && entry.raw_text !== entry.text;
            if (isHallucination) row.className += ' asr-log-row--hallucination';
            row.innerHTML = `
                <span class="asr-log-ts">${entry.ts}</span>
                <span class="asr-log-badge asr-log-badge--result">${isHallucination ? 'HALLUC' : hasText ? 'RESULT' : 'EMPTY'}</span>
                <span class="asr-log-detail">
                    <span class="asr-log-timing">
                        mel=${this._fmtMs(entry.mel_ms)} ·
                        enc=${this._fmtMs(entry.encoder_ms)} ·
                        dec=${this._fmtMs(entry.decode_ms)}
                        (${entry.tokens || 0}tok) ·
                        pp=${this._fmtMs(entry.postprocess_ms)} ·
                        total=${this._fmtMs(entry.total_ms)}
                    </span>
                    <span class="asr-log-meta">
                        ${entry.mel_frames || 0} mel · ${entry.encoder_out || 0} enc ·
                        ${entry.audio_sec != null ? entry.audio_sec.toFixed(2) + 's' : ''} audio ·
                        trigger=${this._esc(entry.trigger || '?')}
                    </span>
                    ${hasText ? `<span class="asr-log-text">${this._esc(entry.text)}</span>` : ''}
                    ${isHallucination ? `<span class="asr-log-raw">hallucination: "${this._esc(entry.raw_text)}"</span>` :
                      textChanged ? `<span class="asr-log-raw">raw: ${this._esc(entry.raw_text)}</span>` : ''}
                </span>
            `;
        } else if (entry.stage === 'transcript') {
            // Enriched transcript from asr_transcript WS message.
            row.innerHTML = `
                <span class="asr-log-ts">${entry.ts}</span>
                <span class="asr-log-badge asr-log-badge--transcript">ASR</span>
                <span class="asr-log-detail">
                    <span class="asr-log-timing">
                        ${this._fmtMs(entry.latency_ms)} total ·
                        mel=${this._fmtMs(entry.mel_ms)} enc=${this._fmtMs(entry.encoder_ms)}
                        dec=${this._fmtMs(entry.decode_ms)} (${entry.tokens || 0}tok) ·
                        ${entry.audio_sec != null ? entry.audio_sec.toFixed(2) + 's audio' : ''}
                    </span>
                    <span class="asr-log-text">${this._esc(entry.text)}</span>
                </span>
            `;
        }

        this.listEl.appendChild(row);
        this.listEl.scrollTop = this.listEl.scrollHeight;
        // Trim DOM.
        while (this.listEl.children.length > this.maxLogs) {
            this.listEl.removeChild(this.listEl.firstChild);
        }
    }

    _updateCount() {
        this.countEl.textContent = `${this.logs.length} entries`;
    }

    _fmtMs(v) {
        if (v == null || v === undefined) return '—';
        return v < 1 ? v.toFixed(2) + 'ms' : v.toFixed(0) + 'ms';
    }

    _esc(s) {
        if (!s) return '';
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }
}
