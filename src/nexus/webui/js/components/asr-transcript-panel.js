// asr-transcript-panel.js — Dedicated ASR transcript display panel.
// Shows clean transcription results with speaker identification labels.
// Same-speaker buffer_full continuations merge into one row with text dedup.

// Deterministic color palette for speaker badges (high-contrast on dark bg).
const SPEAKER_COLORS = [
    '#5caaef', '#ef6c5c', '#5cef8a', '#efcf5c', '#c77dff',
    '#ff8fab', '#4ecdc4', '#ffa62f', '#9d65c9', '#00d2ff',
];

export class AsrTranscriptPanel {
    constructor() {
        this.el = document.getElementById('asr-transcript-panel');
        this.transcripts = [];
        this.maxHistory = 200;
        this._speakerColors = {};  // speaker_id → color
        this._build();
    }

    _build() {
        this.el.innerHTML = `
            <h2 class="panel__title">ASR Transcripts</h2>
            <div class="asr-tx-controls">
                <button id="asr-tx-clear" class="btn btn--vad btn--sm">Clear</button>
                <span class="asr-tx-count" id="asr-tx-count">0</span>
            </div>
            <div class="asr-tx-list" id="asr-tx-list" role="log" aria-live="polite"></div>
        `;
        this.listEl = this.el.querySelector('#asr-tx-list');
        this.countEl = this.el.querySelector('#asr-tx-count');
        this.clearBtn = this.el.querySelector('#asr-tx-clear');
        this.clearBtn.addEventListener('click', () => {
            this.transcripts = [];
            this.listEl.innerHTML = '';
            this.countEl.textContent = '0';
            this._speakerColors = {};
        });
    }

    onTranscript(obj) {
        if (!obj.text) return;
        const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
        const entry = {
            text: obj.text,
            latency_ms: obj.latency_ms || 0,
            audio_sec: obj.audio_sec || 0,
            speaker_id: obj.speaker_id ?? -1,
            speaker_name: obj.speaker_name || '',
            speaker_sim: obj.speaker_sim || 0,
            trigger: obj.trigger || '',
            ts
        };

        // Merge logic: when buffer_full triggers split continuous same-speaker speech,
        // append to the previous row instead of creating a new one.
        const prev = this.transcripts.length > 0
            ? this.transcripts[this.transcripts.length - 1] : null;
        const shouldMerge = prev
            && entry.trigger === 'buffer_full'
            && entry.speaker_id >= 0
            && entry.speaker_id === prev.speaker_id;

        if (shouldMerge) {
            // Deduplicate pre-roll overlap: find longest suffix of prev.text
            // that matches a prefix of entry.text, then skip that overlap.
            const newText = this._dedup(prev.text, entry.text);
            prev.text += newText;
            prev.audio_sec += entry.audio_sec;
            prev.latency_ms = Math.max(prev.latency_ms, entry.latency_ms);
            this._updateLastRow(prev);
        } else {
            this.transcripts.push(entry);
            if (this.transcripts.length > this.maxHistory) this.transcripts.shift();
            this._addRow(entry);
            this.countEl.textContent = this.transcripts.length;
        }
    }

    // Remove overlapping text caused by pre-roll audio re-transcription.
    // Finds longest suffix of `prev` (up to 40 chars) that matches a prefix of `next`.
    _dedup(prev, next) {
        const maxCheck = Math.min(prev.length, next.length, 40);
        let overlapLen = 0;
        for (let len = 1; len <= maxCheck; len++) {
            const suffix = prev.slice(-len);
            const prefix = next.slice(0, len);
            if (suffix === prefix) overlapLen = len;
        }
        return next.slice(overlapLen);
    }

    _getSpeakerColor(id) {
        if (id < 0) return '#888';
        if (!(id in this._speakerColors)) {
            this._speakerColors[id] = SPEAKER_COLORS[id % SPEAKER_COLORS.length];
        }
        return this._speakerColors[id];
    }

    _addRow(entry) {
        const row = document.createElement('div');
        row.className = 'asr-tx-row';
        this._renderRow(row, entry);
        this.listEl.appendChild(row);
        this.listEl.scrollTop = this.listEl.scrollHeight;
        while (this.listEl.children.length > this.maxHistory) {
            this.listEl.removeChild(this.listEl.firstChild);
        }
    }

    _updateLastRow(entry) {
        const row = this.listEl.lastElementChild;
        if (!row) return;
        this._renderRow(row, entry);
        this.listEl.scrollTop = this.listEl.scrollHeight;
    }

    _renderRow(row, entry) {
        const rtf = entry.audio_sec > 0 ? (entry.latency_ms / 1000 / entry.audio_sec) : 0;
        const latClass = entry.latency_ms > 5000 ? 'asr-tx-val--danger' :
                         entry.latency_ms > 2000 ? 'asr-tx-val--warn' : '';

        // Speaker badge
        const spkColor = this._getSpeakerColor(entry.speaker_id);
        const spkLabel = entry.speaker_id >= 0
            ? (entry.speaker_name || `S${entry.speaker_id}`)
            : '?';
        const spkTitle = entry.speaker_id >= 0
            ? `Speaker ${entry.speaker_id}: ${entry.speaker_name || '(unnamed)'} (sim=${entry.speaker_sim.toFixed(3)})`
            : 'Unknown speaker';

        row.innerHTML = `
            <span class="asr-tx-ts">${entry.ts}</span>
            <span class="asr-tx-speaker" style="--spk-color:${spkColor}" title="${spkTitle}">${this._esc(spkLabel)}</span>
            <span class="asr-tx-text">${this._esc(entry.text)}</span>
            <span class="asr-tx-meta">
                <span class="asr-tx-metric ${latClass}">${entry.latency_ms.toFixed(0)}ms</span>
                <span class="asr-tx-metric">${entry.audio_sec.toFixed(1)}s</span>
                <span class="asr-tx-metric">×${rtf.toFixed(2)}</span>
            </span>
        `;
    }

    _esc(s) {
        if (!s) return '';
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
                .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
}
