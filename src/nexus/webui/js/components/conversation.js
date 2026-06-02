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
        this._liveEntity = null;   // DOM node accumulating speech_token text
    }

    mount(parent) {
        const el = document.createElement('div');
        el.className = 'conversation';
        el.setAttribute('role', 'log');
        el.setAttribute('aria-live', 'polite');
        parent.appendChild(el);
        this._el = el;
        this._renderHint();
        i18n.onChange(() => this._renderHint());
    }

    _renderHint() {
        if (this._el.children.length > (this._hint ? 1 : 0)) return;
        if (!this._hint) {
            this._hint = document.createElement('p');
            this._hint.className = 'hint';
            this._el.appendChild(this._hint);
        }
        this._hint.textContent = i18n.t('hint.empty');
    }

    _clearHint() { this._hint?.remove(); this._hint = null; }

    onMessage(msg) {
        switch (msg.type) {
            case 'asr_transcript':      return this._heard(msg);
            case 'speech_token':        return this._streamToken(msg);
            case 'consciousness_decode': return this._decode(msg);
            case 'text_input_ack':      return;   // echo handled by composer
        }
    }

    // A human utterance was transcribed.
    _heard(msg) {
        if (!msg.text || !msg.text.trim()) return;
        this._clearHint();
        const id = (typeof msg.speaker_id === 'number') ? msg.speaker_id : -1;
        const name = msg.speaker_name || (id >= 0 ? `#${id}` : i18n.t('turn.you'));
        this._appendTurn({ who: name, color: spkColor(id), text: msg.text, entity: false });
    }

    // Streaming spoken output token-by-token.
    _streamToken(msg) {
        if (!msg.text) return;
        this._clearHint();
        if (!this._liveEntity) {
            this._liveEntity = this._appendTurn({
                who: '', color: 'var(--c-accent)', text: '', entity: true,
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
            this._appendTurn({ who: '', color: 'var(--c-accent)', text: msg.text, entity: true });
        }
    }

    _appendTurn({ who, color, text, entity }) {
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
        this._el.appendChild(turn);
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
        this._el.appendChild(box);
        this._scroll();
    }

    _scroll() {
        const stream = this._el.parentElement;
        if (stream) stream.scrollTop = stream.scrollHeight;
    }

    unmount() { this._el?.remove(); this._el = null; }
}
