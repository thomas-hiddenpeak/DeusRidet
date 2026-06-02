// composer.js — typed-input box. Sends `text_input:<text>` upstream.
// The send action is surfaced to the app via onSend(text); the app owns the
// WS client so the composer stays transport-agnostic.

import { i18n } from '../i18n.js';

export class Composer {
    constructor() {
        this._el = null;
        this._input = null;
        this._btn = null;
        this.onSend = null;   // set by app: fn(text:string)
    }

    mount(parent) {
        const el = document.createElement('form');
        el.className = 'composer';
        el.innerHTML = `
            <textarea class="composer__input" data-role="input" rows="1"></textarea>
            <button type="submit" class="composer__send" data-role="send"></button>`;
        parent.appendChild(el);
        this._el = el;
        this._input = el.querySelector('[data-role=input]');
        this._btn = el.querySelector('[data-role=send]');

        el.addEventListener('submit', (e) => { e.preventDefault(); this._send(); });
        this._input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._send(); }
        });
        this._input.addEventListener('input', () => this._autosize());
        i18n.onChange(() => this.render());
        this.render();
    }

    _autosize() {
        this._input.style.height = 'auto';
        this._input.style.height = Math.min(120, this._input.scrollHeight) + 'px';
    }

    _send() {
        const text = this._input.value.trim();
        if (!text) return;
        this.onSend?.(text);
        this._input.value = '';
        this._autosize();
        this._input.focus();
    }

    setEnabled(on) {
        if (this._btn) this._btn.disabled = !on;
        if (this._input) this._input.disabled = !on;
    }

    render() {
        if (!this._el) return;
        this._input.placeholder = i18n.t('composer.placeholder');
        this._btn.textContent = i18n.t('composer.send');
    }

    unmount() { this._el?.remove(); this._el = null; }
}
