// conversation.js — the unified dialogue stream.
// Heard speech (asr_transcript) and the entity's own output
// (speech_token streaming + consciousness_decode) interleave here as turns.
// Thinking-state decodes render as a collapsible thinking layer, not a reply.

import { i18n } from '../i18n.js';
import { spkColor } from '../utils/speaker-colors.js';

function clockNow() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export class Conversation {
    constructor() {
        this._el = null;
        this._hint = null;
        this._laneLive = null;
        this._lanePrefill = null;
        this._liveEntity = null;   // DOM node accumulating speech_token text
        this._turnsBySpan = new Map();
        this._aliasKey = 'dr_speaker_aliases';
    }

    mount(parent) {
        const el = document.createElement('div');
        el.className = 'conversation';
        el.setAttribute('role', 'log');
        el.setAttribute('aria-live', 'polite');
        el.innerHTML = `
            <section class="conversation__lane conversation__lane--prefill" data-role="lane-prefill">
                <h2 class="conversation__lane-title" data-role="title-prefill"></h2>
            </section>
            <section class="conversation__lane conversation__lane--live" data-role="lane-live">
                <h2 class="conversation__lane-title" data-role="title-live"></h2>
            </section>`;
        parent.appendChild(el);
        this._el = el;
        this._laneLive = el.querySelector('[data-role=lane-live]');
        this._lanePrefill = el.querySelector('[data-role=lane-prefill]');
        this._renderTitles();
        this._renderHint();
        i18n.onChange(() => {
            this._renderTitles();
            this._renderHint();
        });
    }

    _renderTitles() {
        this._el?.querySelector('[data-role=title-live]')?.replaceChildren(i18n.t('turn.lane_live'));
        this._el?.querySelector('[data-role=title-prefill]')?.replaceChildren(i18n.t('turn.lane_prefill'));
    }

    _renderHint() {
        const hasTurn = this._laneLive?.querySelector('.turn') || this._lanePrefill?.querySelector('.turn');
        if (hasTurn) {
            this._hint?.remove();
            this._hint = null;
            return;
        }
        if (!this._hint) {
            this._hint = document.createElement('p');
            this._hint.className = 'hint';
            this._laneLive.appendChild(this._hint);
        }
        this._hint.textContent = i18n.t('hint.empty');
    }

    _clearHint() { this._hint?.remove(); this._hint = null; }

    onMessage(msg) {
        switch (msg.type) {
            case 'asr_transcript':      return this._heard(msg);
            case 'asr_transcript_amend': return this._amend(msg);
            case 'speech_token':        return this._streamToken(msg);
            case 'consciousness_decode': return this._decode(msg);
            case 'text_input_ack':      return;   // echo handled by composer
        }
    }

    _speakerLabel(id, name) {
        let alias = '';
        if (id >= 0) {
            try {
                const map = JSON.parse(localStorage.getItem(this._aliasKey) || '{}');
                alias = (typeof map[String(id)] === 'string') ? map[String(id)].trim() : '';
            } catch {}
        }
        if (alias) return alias;
        const hasSpeakerName = (typeof name === 'string') && name.trim();
        return hasSpeakerName
            ? name
            : (id >= 0
                ? `${i18n.t('turn.speaker_prefix')} ${id}`
                : i18n.t('turn.unknown'));
    }

    _turnKey(msg) {
        const start = Number(msg.stream_start_sec ?? -1).toFixed(2);
        const end = Number(msg.stream_end_sec ?? -1).toFixed(2);
        return `${start}:${end}:${msg.text || ''}`;
    }

    // A human utterance was transcribed.
    _heard(msg) {
        if (!msg.text || !msg.text.trim()) return;
        this._clearHint();
        const id = (typeof msg.speaker_id === 'number') ? msg.speaker_id : -1;
        const name = this._speakerLabel(id, msg.speaker_name);
        const turn = this._appendTurn({ who: name, color: spkColor(id), text: msg.text, entity: false, lane: 'live' });
        const note = document.createElement('div');
        note.className = 'turn__amend';
        note.innerHTML = `
            <span class="turn__amend-tag turn__amend-tag--online">${i18n.t('turn.online')}</span>
            <span>${name}</span>`;
        turn.appendChild(note);
        this._turnsBySpan.set(this._turnKey(msg), {
            turn,
            onlineLabel: name,
            finalLabel: null,
        });
    }

    _amend(msg) {
        const rec = this._turnsBySpan.get(this._turnKey(msg));
        if (!rec) return;
        const id = (typeof msg.speaker_id === 'number') ? msg.speaker_id : -1;
        const finalLabel = this._speakerLabel(id, msg.speaker_name);
        rec.finalLabel = finalLabel;
        rec.turn.querySelector('.turn__name').textContent = finalLabel;
        rec.turn.querySelector('.turn__dot').style.background = spkColor(id);
        const note = rec.turn.querySelector('.turn__amend');
        note.innerHTML = `
            <span class="turn__amend-tag turn__amend-tag--online">${i18n.t('turn.online')}</span>
            <span>${rec.onlineLabel}</span>
            <span>-></span>
            <span class="turn__amend-tag turn__amend-tag--prefill">${i18n.t('turn.prefill')}</span>
            <span>${finalLabel}</span>`;
        rec.turn.classList.add('turn--amended');
        this._lanePrefill.appendChild(rec.turn);
        this._scroll();
    }

    // Streaming spoken output token-by-token.
    _streamToken(msg) {
        if (!msg.text) return;
        this._clearHint();
        if (!this._liveEntity) {
            this._liveEntity = this._appendTurn({
                who: '', color: 'var(--c-accent)', text: '', entity: true, lane: 'live',
            }).querySelector('.turn__text');
        }
        this._liveEntity.textContent += msg.text;
        this._scroll();
    }

    // A completed decode burst.
    _decode(msg) {
        this._clearHint();
        if (msg.state === 'thinking') {
            this._appendThinking(msg.text || '');
            return;
        }
        // Speech / action: finalise the streaming bubble, or create one.
        if (this._liveEntity) {
            if (msg.text) this._liveEntity.textContent = msg.text;
            this._liveEntity = null;
        } else if (msg.text && msg.text.trim()) {
            this._appendTurn({ who: '', color: 'var(--c-accent)', text: msg.text, entity: true, lane: 'live' });
        }
    }

    _appendTurn({ who, color, text, entity, lane = 'live' }) {
        const turn = document.createElement('article');
        turn.className = 'turn' + (entity ? ' turn--entity' : '');
        turn.innerHTML = `
            <header class="turn__who">
                <span class="turn__dot" aria-hidden="true"></span>
                <span class="turn__name"></span>
                <span class="turn__time"></span>
            </header>
            <div class="turn__text"></div>`;
        turn.querySelector('.turn__dot').style.background = color;
        turn.querySelector('.turn__name').textContent = who;
        turn.querySelector('.turn__time').textContent = clockNow();
        turn.querySelector('.turn__text').textContent = text;
        (lane === 'prefill' ? this._lanePrefill : this._laneLive).appendChild(turn);
        this._scroll();
        return turn;
    }

    _appendThinking(text) {
        if (!text.trim()) return;
        const box = document.createElement('section');
        box.className = 'thinking';
        box.innerHTML = `
            <button type="button" class="thinking__toggle" aria-expanded="false">
                <span>▸</span><span class="thinking__caption"></span>
            </button>
            <div class="thinking__body" hidden></div>`;
        box.querySelector('.thinking__caption').textContent = i18n.t('turn.thinking');
        const body = box.querySelector('.thinking__body');
        body.textContent = text;
        const btn = box.querySelector('.thinking__toggle');
        btn.addEventListener('click', () => {
            const open = body.hidden;
            body.hidden = !open;
            btn.setAttribute('aria-expanded', String(open));
            btn.firstElementChild.textContent = open ? '▾' : '▸';
        });
        this._laneLive.appendChild(box);
        this._scroll();
    }

    _scroll() {
        if (this._laneLive) this._laneLive.scrollTop = this._laneLive.scrollHeight;
        if (this._lanePrefill) this._lanePrefill.scrollTop = this._lanePrefill.scrollHeight;
    }

    unmount() { this._el?.remove(); this._el = null; }
}
