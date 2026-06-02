// presence.js — the entity's wakefulness orb and state label.
// Consumes consciousness_state. Reflects active/daydream/dreaming + wakefulness.

import { i18n } from '../i18n.js';

const STATE_KEY = {
    active: 'presence.active',
    daydream: 'presence.daydream',
    dreaming: 'presence.dreaming',
    idle: 'presence.idle',
};

export class Presence {
    constructor() {
        this._el = null;
        this._state = 'idle';
        this._wake = 0;
        this._entity = '';
        this._online = false;
    }

    mount(parent) {
        const el = document.createElement('section');
        el.className = 'presence presence--idle';
        el.setAttribute('aria-label', 'Entity presence');
        el.innerHTML = `
            <div class="presence__orb" aria-hidden="true"></div>
            <div class="presence__body">
                <span class="presence__state" data-role="state"></span>
                <span class="presence__meta" data-role="meta"></span>
                <div class="presence__wake" role="progressbar" aria-valuemin="0" aria-valuemax="100">
                    <div class="presence__wake-fill" data-role="wake"></div>
                </div>
            </div>`;
        parent.appendChild(el);
        this._el = el;
        this.render();
    }

    setOnline(on) { this._online = on; this.render(); }

    onMessage(msg) {
        if (msg.type !== 'consciousness_state') return;
        if (typeof msg.state === 'string') this._state = msg.state;
        if (typeof msg.wakefulness === 'number') this._wake = msg.wakefulness;
        if (typeof msg.entity === 'string') this._entity = msg.entity;
        this.render();
    }

    render() {
        if (!this._el) return;
        const cls = this._online ? this._state : 'idle';
        this._el.className = `presence presence--${cls}`;
        const stateKey = this._online ? (STATE_KEY[this._state] || 'presence.idle')
                                      : 'presence.offline';
        this._el.querySelector('[data-role=state]').textContent = i18n.t(stateKey);
        this._el.querySelector('[data-role=meta]').textContent = this._entity;
        const pct = Math.round(Math.max(0, Math.min(1, this._wake)) * 100);
        const wake = this._el.querySelector('[data-role=wake]');
        wake.style.width = pct + '%';
        this._el.querySelector('.presence__wake').setAttribute('aria-valuenow', String(pct));
    }

    unmount() { this._el?.remove(); this._el = null; }
}
