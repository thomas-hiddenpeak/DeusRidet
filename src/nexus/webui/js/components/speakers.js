// speakers.js — band of recognised voices.
// Consumes `speaker` events (id, name, new) and pipeline_stats speaker_count.
// Each distinct id becomes a colour-coded chip; the most recent is marked active.

import { i18n } from '../i18n.js';
import { spkColor } from '../utils/speaker-colors.js';

export class Speakers {
    constructor() {
        this._el = null;
        this._list = null;
        this._seen = new Map();   // id -> { name }
        this._activeId = null;
    }

    mount(parent) {
        const el = document.createElement('section');
        el.className = 'speakers';
        el.setAttribute('aria-label', 'Recognised voices');
        el.innerHTML = `
            <span class="speakers__label" data-role="label"></span>
            <div class="speakers__list" data-role="list"></div>`;
        parent.appendChild(el);
        this._el = el;
        this._list = el.querySelector('[data-role=list]');
        this.render();
    }

    onMessage(msg) {
        if (msg.type !== 'speaker') return;
        if (typeof msg.id !== 'number' || msg.id < 0) return;
        this._seen.set(msg.id, { name: msg.name || null });
        this._activeId = msg.id;
        this.render();
    }

    render() {
        if (!this._el) return;
        this._el.querySelector('[data-role=label]').textContent = i18n.t('speakers.label');
        if (this._seen.size === 0) {
            this._list.innerHTML = `<span class="speakers__label">${i18n.t('speakers.none')}</span>`;
            return;
        }
        this._list.replaceChildren(...[...this._seen.entries()].map(([id, info]) => {
            const chip = document.createElement('span');
            chip.className = 'speaker-chip' + (id === this._activeId ? ' speaker-chip--active' : '');
            const dot = document.createElement('span');
            dot.className = 'speaker-chip__dot';
            dot.style.background = spkColor(id);
            chip.appendChild(dot);
            chip.appendChild(document.createTextNode(info.name || `#${id}`));
            return chip;
        }));
    }

    unmount() { this._el?.remove(); this._el = null; }
}
