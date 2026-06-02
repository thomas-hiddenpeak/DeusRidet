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
        this._window = {
            audioSec: 0,
            lastEndSec: 0,
            lastPass: 0,
            progress: 0,
            running: 'idle',
        };
        this.onRename = null;
    }

    mount(parent) {
        const el = document.createElement('section');
        el.className = 'speakers';
        el.setAttribute('aria-label', 'Recognised voices');
        el.innerHTML = `
            <span class="speakers__label" data-role="label"></span>
            <div class="speakers__window" data-role="window" aria-hidden="true">
                <span class="speakers__window-label" data-role="window-label"></span>
                <span class="speakers__window-bar"><span class="speakers__window-fill" data-role="window-fill"></span></span>
            </div>
            <div class="speakers__list" data-role="list"></div>`;
        parent.appendChild(el);
        this._el = el;
        this._list = el.querySelector('[data-role=list]');
        this.render();
    }

    onMessage(msg) {
        if (msg.type === 'speaker') {
            if (typeof msg.id !== 'number' || msg.id < 0) return;
            this._seen.set(msg.id, { name: msg.name || null });
            this._activeId = msg.id;
        } else if (msg.type === 'speaker_name') {
            if (typeof msg.id !== 'number' || msg.id < 0) return;
            this._seen.set(msg.id, { name: msg.name || null });
        } else if (msg.type === 'pipeline_stats') {
            this._window.audioSec = (msg.audio_t1 ?? 0) / 16000.0;
            if (Array.isArray(msg.speaker_lists)) {
                for (const model of msg.speaker_lists) {
                    if (!Array.isArray(model?.speakers)) continue;
                    for (const spk of model.speakers) {
                        if (typeof spk.id === 'number' && spk.id >= 0) {
                            this._seen.set(spk.id, { name: spk.name || null });
                        }
                    }
                }
            }
        } else if (msg.type === 'speaker_diarize_status') {
            this._window.progress = Number(msg.cycle_progress || 0);
            this._window.running = msg.phase || 'idle';
        } else if (msg.type === 'speaker_diarize_partial' || msg.type === 'speaker_diarize_final') {
            const origin = Number(msg.origin_sec || 0);
            const audio = Number(msg.audio_sec || 0);
            this._window.lastEndSec = origin + audio;
            this._window.lastPass = Number(msg.pass || this._window.lastPass || 0);
            this._window.running = 'idle';
            this._window.progress = 0;
        } else {
            return;
        }
        this.render();
    }

    render() {
        if (!this._el) return;
        this._el.querySelector('[data-role=label]').textContent = i18n.t('speakers.label');
        const wLabel = this._el.querySelector('[data-role=window-label]');
        const wFill = this._el.querySelector('[data-role=window-fill]');
        const phase = (this._window.running === 'running' || this._window.running === 'periodic' || this._window.running === 'triggered')
            ? 'speakers.window_running'
            : (this._window.running === 'finalizing'
                ? 'speakers.window_finalizing'
                : 'speakers.window_idle');
        wLabel.textContent = `${i18n.t('speakers.window_label')} · ${i18n.t(phase)}`;
        wFill.style.width = `${Math.round(Math.max(0, Math.min(1, this._window.progress)) * 100)}%`;
        wFill.dataset.state = this._window.running;
        if (this._seen.size === 0) {
            this._list.innerHTML = `<span class="speakers__label">${i18n.t('speakers.none')}</span>`;
            return;
        }
        this._list.replaceChildren(...[...this._seen.entries()].map(([id, info]) => {
            const chip = document.createElement('button');
            chip.className = 'speaker-chip' + (id === this._activeId ? ' speaker-chip--active' : '');
            chip.type = 'button';
            chip.title = i18n.t('speakers.rename_prompt');
            const dot = document.createElement('span');
            dot.className = 'speaker-chip__dot';
            dot.style.background = spkColor(id);
            chip.appendChild(dot);
            chip.appendChild(document.createTextNode(info.name || `${i18n.t('turn.speaker_prefix')} ${id}`));
            chip.addEventListener('click', () => {
                const next = window.prompt(i18n.t('speakers.rename_prompt'), info.name || '');
                if (typeof next !== 'string') return;
                const trimmed = next.trim();
                if (!trimmed || trimmed === info.name) return;
                this.onRename?.(id, trimmed);
            });
            return chip;
        }));
    }

    unmount() { this._el?.remove(); this._el = null; }
}
