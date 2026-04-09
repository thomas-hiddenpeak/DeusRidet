// consciousness-panel.js — Consciousness state + system metrics panel
//
// Displays: wakefulness gauge, state indicator, KV cache stats,
// sequence position, prefill/decode metrics, system memory bars,
// mode toggles, and state history.

export class ConsciousnessPanel {
    constructor(ws) {
        this.ws = ws;
        this.el = document.getElementById('consciousness-panel');
        if (!this.el) return;

        this.llmLoaded = false;
        this.entityName = '';
        this.state = 'active';
        this.wakefulness = 0;
        this.kvUsed = 0;
        this.kvFree = 0;
        this.pos = 0;
        this.stateHistory = [];
        this.maxHistory = 100;

        this._render();
        this._bindEvents();
    }

    _render() {
        this.el.innerHTML = `
            <h2 class="panel__title">Consciousness</h2>
            <div class="cs-status" id="cs-status">
                <span class="cs-entity" id="cs-entity">—</span>
                <span class="cs-state-badge" id="cs-state-badge">—</span>
                <span class="cs-llm-status" id="cs-llm-status">LLM: not loaded</span>
            </div>
            <div class="cs-wakefulness">
                <label class="cs-label">Wakefulness</label>
                <div class="cs-gauge-wrap">
                    <div class="cs-gauge-bar" id="cs-gauge-bar"></div>
                    <span class="cs-gauge-val" id="cs-gauge-val">0.000</span>
                </div>
            </div>
            <div class="cs-metrics">
                <span class="stat">Prefill: <strong id="cs-prefill-tps">—</strong> tok/s</span>
                <span class="stat">(<strong id="cs-prefill-ms">—</strong> ms)</span>
                <span class="stat">Decode: <strong id="cs-decode-ms">—</strong> ms/tok</span>
            </div>
            <div class="cs-metrics">
                <span class="stat">Total P: <strong id="cs-total-prefill">0</strong></span>
                <span class="stat">Total D: <strong id="cs-total-decode">0</strong></span>
                <span class="stat">Pos: <strong id="cs-pos">0</strong></span>
            </div>
            <div class="cs-sys-bars">
                <div class="sm-row">
                    <span class="sm-label">CUDA</span>
                    <div class="sm-bar-wrap"><div class="sm-bar sm-bar--cuda" id="cs-cuda-bar"></div></div>
                    <span class="sm-val" id="cs-cuda-val">—</span>
                </div>
                <div class="sm-row">
                    <span class="sm-label">Avail</span>
                    <div class="sm-bar-wrap"><div class="sm-bar sm-bar--mem" id="cs-mem-bar"></div></div>
                    <span class="sm-val" id="cs-mem-val">—</span>
                </div>
                <div class="sm-row">
                    <span class="sm-label">KV</span>
                    <div class="sm-bar-wrap"><div class="sm-bar sm-bar--kv" id="cs-kv-bar"></div></div>
                    <span class="sm-val" id="cs-kv-val">0/0</span>
                </div>
            </div>
            <div class="cs-toggles">
                <button class="btn btn--vad btn--active" id="cs-toggle-llm"
                        aria-pressed="true">LLM</button>
                <button class="btn btn--vad btn--active" id="cs-toggle-response"
                        aria-pressed="true">Response</button>
                <button class="btn btn--vad" id="cs-toggle-daydream"
                        aria-pressed="false">Daydream</button>
                <button class="btn btn--vad" id="cs-toggle-dreaming"
                        aria-pressed="false">Dreaming</button>
            </div>
            <div class="cs-history" id="cs-history"></div>
        `;

        this.entityEl = this.el.querySelector('#cs-entity');
        this.badgeEl = this.el.querySelector('#cs-state-badge');
        this.llmStatusEl = this.el.querySelector('#cs-llm-status');
        this.gaugeBar = this.el.querySelector('#cs-gauge-bar');
        this.gaugeVal = this.el.querySelector('#cs-gauge-val');
        this.posEl = this.el.querySelector('#cs-pos');
        this.historyEl = this.el.querySelector('#cs-history');

        this.prefillTpsEl = this.el.querySelector('#cs-prefill-tps');
        this.prefillMsEl = this.el.querySelector('#cs-prefill-ms');
        this.decodeMsEl = this.el.querySelector('#cs-decode-ms');
        this.totalPrefillEl = this.el.querySelector('#cs-total-prefill');
        this.totalDecodeEl = this.el.querySelector('#cs-total-decode');

        // System bars
        this.cudaBar = this.el.querySelector('#cs-cuda-bar');
        this.cudaVal = this.el.querySelector('#cs-cuda-val');
        this.memBar = this.el.querySelector('#cs-mem-bar');
        this.memVal = this.el.querySelector('#cs-mem-val');
        this.kvBar = this.el.querySelector('#cs-kv-bar');
        this.kvVal = this.el.querySelector('#cs-kv-val');

        this.toggleResponse = this.el.querySelector('#cs-toggle-response');
        this.toggleDaydream = this.el.querySelector('#cs-toggle-daydream');
        this.toggleDreaming = this.el.querySelector('#cs-toggle-dreaming');
        this.toggleLlm = this.el.querySelector('#cs-toggle-llm');
    }

    _bindEvents() {
        const bind = (btn, mode) => {
            if (!btn) return;
            btn.addEventListener('click', () => {
                const on = btn.getAttribute('aria-pressed') === 'true';
                const next = !on;
                btn.classList.toggle('btn--active', next);
                btn.setAttribute('aria-pressed', String(next));
                this.ws.sendText(`consciousness_enable:${mode}:${next ? 'on' : 'off'}`);
            });
        };
        bind(this.toggleResponse, 'response');
        bind(this.toggleDaydream, 'daydream');
        bind(this.toggleDreaming, 'dreaming');
        bind(this.toggleLlm, 'llm');
    }

    onConsciousnessState(data) {
        if (!this.el) return;

        if (data.llm_loaded !== undefined) {
            this.llmLoaded = data.llm_loaded;
            this.llmStatusEl.textContent = data.llm_loaded ? 'LLM: loaded' : 'LLM: not loaded';
            this.llmStatusEl.classList.toggle('cs-llm-active', data.llm_loaded);
        }
        if (data.entity) {
            this.entityName = data.entity;
            this.entityEl.textContent = data.entity;
        }
        if (data.state) {
            this.state = data.state;
            this.badgeEl.textContent = data.state.toUpperCase();
            this.badgeEl.className = 'cs-state-badge cs-state--' + data.state;
        }
        if (data.wakefulness !== undefined) {
            this.wakefulness = data.wakefulness;
            const pct = Math.min(100, Math.max(0, data.wakefulness * 100));
            this.gaugeBar.style.width = pct + '%';
            this.gaugeVal.textContent = data.wakefulness.toFixed(3);
            if (data.wakefulness > 0.6) this.gaugeBar.className = 'cs-gauge-bar cs-gauge--high';
            else if (data.wakefulness > 0.3) this.gaugeBar.className = 'cs-gauge-bar cs-gauge--mid';
            else this.gaugeBar.className = 'cs-gauge-bar cs-gauge--low';
        }
        if (data.pos !== undefined) this.posEl.textContent = data.pos;

        // Metrics
        if (data.prefill_tps !== undefined) {
            this.prefillTpsEl.textContent = data.prefill_tps.toFixed(1);
            this.prefillMsEl.textContent = data.prefill_ms.toFixed(0);
        }
        if (data.decode_ms_per_tok !== undefined) this.decodeMsEl.textContent = data.decode_ms_per_tok.toFixed(1);
        if (data.total_prefill_tok !== undefined) this.totalPrefillEl.textContent = data.total_prefill_tok;
        if (data.total_decode_tok !== undefined) this.totalDecodeEl.textContent = data.total_decode_tok;

        // System memory bars
        if (data.cuda_free_mb !== undefined && data.cuda_total_mb !== undefined) {
            const used = data.cuda_total_mb - data.cuda_free_mb;
            const pct = data.cuda_total_mb > 0 ? (used / data.cuda_total_mb) * 100 : 0;
            this.cudaBar.style.width = Math.min(100, pct) + '%';
            this.cudaBar.className = 'sm-bar sm-bar--cuda' + (pct > 90 ? ' sm-bar--danger' : pct > 75 ? ' sm-bar--warn' : '');
            this.cudaVal.textContent = `${(used/1024).toFixed(1)}/${(data.cuda_total_mb/1024).toFixed(1)}G`;
        }
        if (data.mem_avail_mb !== undefined) {
            const totalMb = 65536;
            const usedPct = ((totalMb - data.mem_avail_mb) / totalMb) * 100;
            this.memBar.style.width = Math.min(100, usedPct) + '%';
            this.memBar.className = 'sm-bar sm-bar--mem' + (usedPct > 90 ? ' sm-bar--danger' : usedPct > 75 ? ' sm-bar--warn' : '');
            this.memVal.textContent = `${(data.mem_avail_mb/1024).toFixed(1)}G free`;
        }
        if (data.kv_used !== undefined && data.kv_free !== undefined) {
            this.kvUsed = data.kv_used;
            this.kvFree = data.kv_free;
            const total = data.kv_used + data.kv_free;
            const pct = total > 0 ? (data.kv_used / total) * 100 : 0;
            this.kvBar.style.width = Math.min(100, pct) + '%';
            this.kvBar.className = 'sm-bar sm-bar--kv' + (pct > 90 ? ' sm-bar--danger' : pct > 75 ? ' sm-bar--warn' : '');
            this.kvVal.textContent = `${data.kv_used}/${total}`;
        }

        // Sync toggles on initial state
        if (data.enable_response !== undefined) this._syncToggle(this.toggleResponse, data.enable_response);
        if (data.enable_daydream !== undefined) this._syncToggle(this.toggleDaydream, data.enable_daydream);
        if (data.enable_dreaming !== undefined) this._syncToggle(this.toggleDreaming, data.enable_dreaming);
        if (data.enable_llm !== undefined) this._syncToggle(this.toggleLlm, data.enable_llm);

        // History
        if (data.state && data.wakefulness !== undefined) {
            const now = new Date().toLocaleTimeString('en-GB', { hour12: false });
            this.stateHistory.push({ time: now, state: data.state, wakefulness: data.wakefulness });
            if (this.stateHistory.length > this.maxHistory) this.stateHistory.shift();
            this._renderHistory();
        }
    }

    onConsciousnessEnable(data) {
        if (!this.el) return;
        const map = { response: this.toggleResponse, daydream: this.toggleDaydream, dreaming: this.toggleDreaming, llm: this.toggleLlm };
        const btn = map[data.mode];
        if (btn) this._syncToggle(btn, data.enabled);
    }

    _syncToggle(btn, on) {
        if (!btn) return;
        btn.classList.toggle('btn--active', on);
        btn.setAttribute('aria-pressed', String(on));
    }

    _renderHistory() {
        if (!this.historyEl) return;
        const recent = this.stateHistory.slice(-8);
        this.historyEl.innerHTML = recent.map(h =>
            `<span class="cs-hist-entry cs-state--${h.state}">${h.time} ${h.state} w=${h.wakefulness.toFixed(2)}</span>`
        ).join('');
    }
}
