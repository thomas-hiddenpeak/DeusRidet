// vires-panel.js — Vires compute-substrate telemetry panel.
//
// Renders the single inspectable view of "who is computing what" (RFC 13,
// observability rule): the live consumer registry with each consumer's
// metabolic class + submitted-pass counter + V3 glymphatic-clearance count,
// plus the V2 back-pressure state
// (whether background consumers are currently yielding to a busy foreground).
// Fed by the periodic `vires_compute_snapshot` WS message broadcast from the
// awaken main thread every 2 s.

export class ViresPanel {
    constructor() {
        const root = document.getElementById('vires-panel');
        if (!root) return;
        root.innerHTML = `
            <h2 class="panel__title">Vires — GPU Compute Substrate</h2>
            <div class="vires-status">
                <span class="stat">Priority range:
                    <strong id="vires-prio-range">—</strong></span>
                <span class="stat">Back-pressure:
                    <strong id="vires-yielding">—</strong></span>
                <span class="stat">Foreground idle:
                    <strong id="vires-idle">—</strong></span>
            </div>
            <table class="vires-table">
                <thead>
                    <tr>
                        <th>#</th><th>Consumer</th>
                        <th>Class</th><th>Submitted</th><th>Reclaimed</th>
                    </tr>
                </thead>
                <tbody id="vires-rows"></tbody>
            </table>
        `;
        this._rangeEl    = document.getElementById('vires-prio-range');
        this._yieldingEl = document.getElementById('vires-yielding');
        this._idleEl     = document.getElementById('vires-idle');
        this._rowsEl     = document.getElementById('vires-rows');
    }

    onSnapshot(obj) {
        if (!this._rowsEl) return;

        if (this._rangeEl) {
            this._rangeEl.textContent =
                `greatest=${obj.greatest_priority} least=${obj.least_priority}`;
        }
        if (this._yieldingEl) {
            const y = !!obj.background_yielding;
            this._yieldingEl.textContent = y ? 'yielding' : 'idle';
            this._yieldingEl.dataset.state = y ? 'yielding' : 'idle';
        }
        if (this._idleEl) {
            this._idleEl.textContent = (obj.foreground_idle_us == null)
                ? 'never'
                : `${(obj.foreground_idle_us / 1000).toFixed(1)} ms`;
        }

        const consumers = Array.isArray(obj.consumers) ? obj.consumers : [];
        consumers.sort((a, b) => (a.id || 0) - (b.id || 0));
        this._rowsEl.replaceChildren();
        for (const c of consumers) {
            const tr = document.createElement('tr');
            tr.dataset.class = c.priority || '';
            for (const v of [c.id, c.name, c.priority, c.submitted,
                             c.reclaimed]) {
                const td = document.createElement('td');
                td.textContent = (v == null) ? '—' : String(v);
                tr.appendChild(td);
            }
            this._rowsEl.appendChild(tr);
        }
    }
}
