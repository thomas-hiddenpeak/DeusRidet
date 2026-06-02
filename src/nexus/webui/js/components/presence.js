// presence.js — compact top-bar status chip.
// Consumes consciousness_state and surfaces state + entity label without the
// large orb block.

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
        this._entity = '';
        this._online = false;
    }

    mount(parent) {
        const el = document.createElement('button');
        el.type = 'button';
        el.className = 'presence-chip presence-chip--idle';
        el.setAttribute('aria-label', 'Entity presence');
        el.innerHTML = `
            <span class="presence-chip__lamp" aria-hidden="true"></span>
            <span class="presence-chip__label" data-role="state"></span>`;
        parent.appendChild(el);
        this._el = el;
        this.render();
    }

    setOnline(on) { this._online = on; this.render(); }

    onMessage(msg) {
        if (msg.type !== 'consciousness_state') return;
        if (typeof msg.state === 'string') this._state = msg.state;
        if (typeof msg.entity === 'string') this._entity = msg.entity;
        this.render();
    }

    entityName() { return this._entity || ''; }

    render() {
        if (!this._el) return;
        const cls = this._online ? this._state : 'idle';
        this._el.className = `presence-chip presence-chip--${cls}`;
        const stateKey = this._online ? (STATE_KEY[this._state] || 'presence.idle')
                                      : 'presence.offline';
        this._el.querySelector('[data-role=state]').textContent = i18n.t(stateKey);
    }

    unmount() { this._el?.remove(); this._el = null; }
}
