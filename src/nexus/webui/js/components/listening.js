// listening.js — hearing status strip.
// Consumes pipeline_stats (rms, is_speech) and vad edges. Hosts the mic toggle
// button, which it surfaces to the app via an onMic callback.

import { i18n } from '../i18n.js';

export class Listening {
    constructor() {
        this._el = null;
        this._hot = false;
        this._micOn = false;
        this.onMic = null;     // set by app: fn(enabled:boolean)
    }

    mount(parent) {
        const el = document.createElement('section');
        el.className = 'listen';
        el.setAttribute('aria-label', 'Hearing status');
        el.innerHTML = `
            <span class="listen__icon" aria-hidden="true"></span>
            <span class="listen__label" data-role="label"></span>
            <div class="listen__meter"><div class="listen__meter-fill" data-role="meter"></div></div>
            <button type="button" class="listen__mic" data-role="mic"></button>`;
        parent.appendChild(el);
        this._el = el;
        el.querySelector('[data-role=mic]').addEventListener('click', () => {
            this._micOn = !this._micOn;
            this.onMic?.(this._micOn);
            this.render();
        });
        this.render();
    }

    onMessage(msg) {
        if (msg.type === 'pipeline_stats') {
            this._hot = !!msg.is_speech;
            const rms = Math.max(0, Math.min(1, (msg.rms ?? 0) * 6));
            this._setMeter(rms);
            this.render();
        } else if (msg.type === 'vad') {
            this._hot = msg.event === 'start';
            this.render();
        }
    }

    _setMeter(v) {
        this._el?.querySelector('[data-role=meter]').style.setProperty('width', (v * 100) + '%');
    }

    render() {
        if (!this._el) return;
        this._el.classList.toggle('listen--hot', this._hot);
        this._el.querySelector('[data-role=label]').textContent =
            i18n.t(this._hot ? 'listen.hot' : 'listen.idle');
        const mic = this._el.querySelector('[data-role=mic]');
        mic.classList.toggle('listen__mic--on', this._micOn);
        mic.textContent = i18n.t(this._micOn ? 'listen.mic_on' : 'listen.mic_off');
    }

    unmount() { this._el?.remove(); this._el = null; }
}
