// listening.js — compact top mic control.
// Consumes pipeline_stats/vad and encodes listening+volume into button state.

import { i18n } from '../i18n.js';

export class Listening {
    constructor() {
        this._el = null;
        this._hot = false;
        this._micOn = false;
        this._level = 0;
        this.onMic = null;     // set by app: fn(enabled:boolean)
    }

    mount(parent) {
        const el = document.createElement('button');
        el.type = 'button';
        el.className = 'mic-chip';
        el.setAttribute('aria-label', 'Hearing status');
        el.innerHTML = `
            <span class="mic-chip__lamp" aria-hidden="true"></span>
            <span class="mic-chip__label" data-role="label"></span>`;
        parent.appendChild(el);
        this._el = el;
        el.addEventListener('click', () => {
            this._micOn = !this._micOn;
            this.onMic?.(this._micOn);
            this.render();
        });
        this.render();
    }

    onMessage(msg) {
        if (msg.type === 'pipeline_stats') {
            this._hot = !!msg.is_speech;
            const rms = Math.max(0, Math.min(1, (msg.rms ?? 0) * 8));
            this._level = Math.max(0, Math.min(5, Math.floor(rms * 6)));
            this.render();
        } else if (msg.type === 'vad') {
            this._hot = msg.event === 'start';
            this.render();
        }
    }

    render() {
        if (!this._el) return;
        this._el.className = `mic-chip mic-chip--lvl${this._level}`
            + (this._hot ? ' mic-chip--hot' : '')
            + (this._micOn ? ' mic-chip--on' : '');
        this._el.querySelector('[data-role=label]').textContent =
            i18n.t(this._micOn ? 'listen.mic_on' : 'listen.mic_off');
    }

    unmount() { this._el?.remove(); this._el = null; }
}
