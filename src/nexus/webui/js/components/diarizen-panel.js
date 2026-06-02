// diarizen-panel.js — DiariZen recluster control + status panel.
//
// Surfaces the Hybrid P2 (TranscriptHoldback + DiarizenPeriodicWorker)
// behaviour to the operator: shows last pass timestamp, segment count,
// per-label durations, and provides a manual "Trigger Now" button
// (sends WS text command `diarizen_trigger`).

export class DiarizenPanel {
    constructor(ws) {
        this._ws = ws;
        const root = document.getElementById('diarizen-panel');
        if (!root) return;
        root.innerHTML = `
            <h2 class="panel__title">DiariZen Recluster</h2>
            <div class="diarizen-controls">
                <button id="diarizen-trigger-btn" class="btn btn--vad">
                    Trigger Now
                </button>
                <button id="diarizen-finalize-btn" class="btn btn--vad">
                    Finalize
                </button>
                <span class="stat">Pass: <strong id="diarizen-pass">0</strong></span>
                <span class="stat">Status: <strong id="diarizen-status">idle</strong></span>
            </div>
            <div class="diarizen-stats">
                <span class="stat">Segments: <strong id="diarizen-seg-count">—</strong></span>
                <span class="stat">Origin: <strong id="diarizen-origin-sec">—</strong>s</span>
                <span class="stat">Changed pending: <strong id="diarizen-changed">—</strong></span>
            </div>
            <h3 class="panel__subtitle">Per-label duration (s)</h3>
            <pre id="diarizen-label-durs" class="log-output diarizen-log"></pre>
            <h3 class="panel__subtitle">Recent passes</h3>
            <pre id="diarizen-history" class="log-output diarizen-log"></pre>
        `;
        this._passEl     = document.getElementById('diarizen-pass');
        this._statusEl   = document.getElementById('diarizen-status');
        this._segCountEl = document.getElementById('diarizen-seg-count');
        this._originEl   = document.getElementById('diarizen-origin-sec');
        this._changedEl  = document.getElementById('diarizen-changed');
        this._durEl      = document.getElementById('diarizen-label-durs');
        this._historyEl  = document.getElementById('diarizen-history');
        this._history    = [];

        document.getElementById('diarizen-trigger-btn')
            ?.addEventListener('click', () => this._ws.sendText('diarizen_trigger'));
        document.getElementById('diarizen-finalize-btn')
            ?.addEventListener('click', () => this._ws.sendText('diarizen_finalize'));
    }

    onProgress(obj) {
        this._statusEl.textContent = obj.status || '?';
    }

    onPartial(obj) {
        this._render(obj, false);
    }

    onFinal(obj) {
        this._render(obj, true);
    }

    _render(obj, isFinal) {
        if (this._passEl)     this._passEl.textContent = obj.pass ?? '—';
        if (this._statusEl)   this._statusEl.textContent = isFinal ? 'final' : 'partial';
        if (this._segCountEl) this._segCountEl.textContent = obj.segment_count ?? '—';
        if (this._originEl)   this._originEl.textContent = (obj.origin_sec ?? 0).toFixed(2);
        if (this._changedEl)  this._changedEl.textContent = obj.changed_pending ?? '—';

        // Aggregate per-label duration.
        const durs = {};
        for (const s of obj.segments || []) {
            const [start, end, label] = Array.isArray(s)
                ? s
                : [s.start, s.end, s.label];
            const k = String(label);
            durs[k] = (durs[k] || 0) + (end - start);
        }
        const rows = Object.entries(durs)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => `  label=${k.padEnd(4)} ${v.toFixed(1).padStart(7)}s`);
        if (this._durEl) this._durEl.textContent = rows.join('\n') || '—';

        // History.
        const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
        const tag = isFinal ? 'FINAL' : 'partial';
        this._history.push(
            `[${ts}] ${tag} pass=${obj.pass} segs=${obj.segment_count}`
            + ` origin=${(obj.origin_sec ?? 0).toFixed(1)}s`
            + ` changed=${obj.changed_pending ?? '—'}`);
        if (this._history.length > 20) this._history.shift();
        if (this._historyEl) this._historyEl.textContent = this._history.join('\n');
    }
}
