// config-panel.js — Per-pipeline configuration panel
//
// Sections: Identity (system prompt), Speech, Thinking, Action.
// Each pipeline has its own system prompt and sampling parameters
// (temperature, top_k, top_p, max_tokens).

const PIPELINES = [
    { id: 'speech',   label: 'Speech',   maxTok: 80,   temp: 0.7, topk: 50, topp: 0.9, tokMax: 512,  defaultOn: true },
    { id: 'thinking', label: 'Thinking', maxTok: 256,  temp: 0.7, topk: 50, topp: 0.9, tokMax: 4096, defaultOn: false },
    { id: 'action',   label: 'Action',   maxTok: 512,  temp: 0.7, topk: 50, topp: 0.9, tokMax: 4096, defaultOn: false },
];

export class ConfigPanel {
    constructor(ws) {
        this.ws = ws;
        this.el = document.getElementById('config-panel');
        if (!this.el) return;
        this.sliders = {};  // { 'speech.temperature': { slider, valEl } }
        this.toggles = {};  // { 'speech': buttonEl }
        this._render();
        this._bindEvents();
    }

    _render() {
        const pipelineHTML = PIPELINES.map(p => `
            <details class="cfg-pipeline" id="cfg-pipe-${p.id}">
                <summary class="cfg-pipe-title">${p.label}
                    <button class="btn btn--vad cfg-enable-toggle${p.defaultOn ? ' btn--active' : ''}"
                            id="cfg-enable-${p.id}" data-pipeline="${p.id}"
                            aria-pressed="${p.defaultOn}">${p.defaultOn ? 'ON' : 'OFF'}</button>
                </summary>
                <div class="cfg-pipe-body">
                    <label class="cfg-label" for="cfg-prompt-${p.id}">Prompt</label>
                    <textarea id="cfg-prompt-${p.id}" class="cfg-textarea" rows="2"
                        placeholder="${p.label} pipeline prompt"></textarea>
                    <button class="btn btn--vad cfg-prompt-apply"
                            data-pipeline="${p.id}">Apply</button>
                    <div class="cfg-slider-row">
                        <label class="cfg-label">Temp
                            <strong id="cfg-${p.id}-temp-val">${p.temp.toFixed(2)}</strong></label>
                        <input type="range" id="cfg-${p.id}-temp" class="cfg-slider"
                               min="0" max="2" step="0.05" value="${p.temp}"
                               data-pipe="${p.id}" data-key="temperature">
                    </div>
                    <div class="cfg-slider-row">
                        <label class="cfg-label">Top-K
                            <strong id="cfg-${p.id}-topk-val">${p.topk}</strong></label>
                        <input type="range" id="cfg-${p.id}-topk" class="cfg-slider"
                               min="1" max="200" step="1" value="${p.topk}"
                               data-pipe="${p.id}" data-key="top_k">
                    </div>
                    <div class="cfg-slider-row">
                        <label class="cfg-label">Top-P
                            <strong id="cfg-${p.id}-topp-val">${p.topp.toFixed(2)}</strong></label>
                        <input type="range" id="cfg-${p.id}-topp" class="cfg-slider"
                               min="0" max="1" step="0.05" value="${p.topp}"
                               data-pipe="${p.id}" data-key="top_p">
                    </div>
                    <div class="cfg-slider-row">
                        <label class="cfg-label">Max Tokens
                            <strong id="cfg-${p.id}-maxtok-val">${p.maxTok}</strong></label>
                        <input type="range" id="cfg-${p.id}-maxtok" class="cfg-slider"
                               min="10" max="${p.tokMax}" step="10" value="${p.maxTok}"
                               data-pipe="${p.id}" data-key="max_tokens">
                    </div>
                </div>
            </details>
        `).join('');

        this.el.innerHTML = `
            <h2 class="panel__title">Configuration</h2>
            <details class="cfg-pipeline" open>
                <summary class="cfg-pipe-title">Identity</summary>
                <div class="cfg-pipe-body">
                    <label class="cfg-label" for="cfg-prompt-identity">System Prompt</label>
                    <textarea id="cfg-prompt-identity" class="cfg-textarea" rows="3"
                        placeholder="Identity system prompt (applied on next restart)"></textarea>
                    <button id="cfg-prompt-identity-btn" class="btn btn--vad cfg-prompt-apply"
                            data-pipeline="identity">Apply</button>
                </div>
            </details>
            ${pipelineHTML}
        `;

        // Cache slider references
        for (const p of PIPELINES) {
            for (const [suffix, key] of [['temp','temperature'],['topk','top_k'],['topp','top_p'],['maxtok','max_tokens']]) {
                const slider = this.el.querySelector(`#cfg-${p.id}-${suffix}`);
                const valEl = this.el.querySelector(`#cfg-${p.id}-${suffix}-val`);
                if (slider && valEl) {
                    this.sliders[`${p.id}.${key}`] = { slider, valEl };
                }
            }
        }

        // Cache enable toggle references
        for (const p of PIPELINES) {
            const btn = this.el.querySelector(`#cfg-enable-${p.id}`);
            if (btn) this.toggles[p.id] = btn;
        }
    }

    _bindEvents() {
        // Enable toggle buttons
        for (const [pipeId, btn] of Object.entries(this.toggles)) {
            btn.addEventListener('click', (e) => {
                e.stopPropagation(); // prevent <details> toggle
                const on = btn.getAttribute('aria-pressed') === 'true';
                const next = !on;
                btn.classList.toggle('btn--active', next);
                btn.setAttribute('aria-pressed', String(next));
                btn.textContent = next ? 'ON' : 'OFF';
                this.ws.sendText(`consciousness_enable:${pipeId}:${next ? 'on' : 'off'}`);
            });
        }

        // Prompt apply buttons
        this.el.querySelectorAll('.cfg-prompt-apply').forEach(btn => {
            btn.addEventListener('click', () => {
                const pipeline = btn.dataset.pipeline;
                const textarea = this.el.querySelector(`#cfg-prompt-${pipeline}`);
                const text = textarea?.value.trim();
                if (text) {
                    this.ws.sendText(`consciousness_prompt:${pipeline}:${text}`);
                }
            });
        });

        // Slider bindings — delegated
        const fmtMap = {
            temperature: v => parseFloat(v).toFixed(2),
            top_k: v => String(parseInt(v)),
            top_p: v => parseFloat(v).toFixed(2),
            max_tokens: v => String(parseInt(v)),
        };

        for (const [fullKey, ref] of Object.entries(this.sliders)) {
            const key = fullKey.split('.')[1];
            const fmt = fmtMap[key] || (v => v);

            ref.slider.addEventListener('input', () => {
                ref.valEl.textContent = fmt(ref.slider.value);
            });
            ref.slider.addEventListener('change', () => {
                ref.valEl.textContent = fmt(ref.slider.value);
                this.ws.sendText(`consciousness_param:${fullKey}:${ref.slider.value}`);
            });
        }
    }

    onConsciousnessState(data) {
        if (!this.el) return;
        // Sync per-pipeline enable toggles from initial state
        for (const p of PIPELINES) {
            const key = `enable_${p.id}`;
            if (data[key] !== undefined) this._syncEnableToggle(p.id, data[key]);
        }
        // Sync per-pipeline params from initial state
        for (const p of PIPELINES) {
            const pData = data[p.id];
            if (!pData) continue;
            this._syncSlider(`${p.id}.temperature`, pData.temperature, v => parseFloat(v).toFixed(2));
            this._syncSlider(`${p.id}.top_k`, pData.top_k, v => String(parseInt(v)));
            this._syncSlider(`${p.id}.top_p`, pData.top_p, v => parseFloat(v).toFixed(2));
            this._syncSlider(`${p.id}.max_tokens`, pData.max_tokens, v => String(parseInt(v)));
        }
    }

    onConsciousnessPrompts(data) {
        if (!this.el) return;
        // Populate textareas with server-side default prompts
        const idEl = this.el.querySelector('#cfg-prompt-identity');
        if (idEl && data.identity) idEl.value = data.identity;
        for (const p of PIPELINES) {
            const el = this.el.querySelector(`#cfg-prompt-${p.id}`);
            if (el && data[p.id]) el.value = data[p.id];
        }
    }

    onConsciousnessParam(data) {
        if (!this.el) return;
        const fmtMap = {
            temperature: v => parseFloat(v).toFixed(2),
            top_k: v => String(parseInt(v)),
            top_p: v => parseFloat(v).toFixed(2),
            max_tokens: v => String(parseInt(v)),
        };
        const key = data.key; // e.g. "speech.temperature"
        const paramName = key.includes('.') ? key.split('.')[1] : key;
        const fmt = fmtMap[paramName] || (v => v);
        this._syncSlider(key, data.value, fmt);
    }

    onConsciousnessEnable(data) {
        if (!this.el) return;
        this._syncEnableToggle(data.mode, data.enabled);
    }

    _syncEnableToggle(pipeId, on) {
        const btn = this.toggles[pipeId];
        if (!btn) return;
        btn.classList.toggle('btn--active', on);
        btn.setAttribute('aria-pressed', String(on));
        btn.textContent = on ? 'ON' : 'OFF';
    }

    _syncSlider(fullKey, value, fmt) {
        const ref = this.sliders[fullKey];
        if (ref && value !== undefined) {
            ref.slider.value = value;
            ref.valEl.textContent = fmt(value);
        }
    }
}
