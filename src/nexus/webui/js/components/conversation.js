// conversation.js — the unified dialogue stream.
// Heard speech (asr_transcript) and the entity's own output
// (speech_token streaming + consciousness_decode) interleave here as turns.
// Thinking-state decodes render as a collapsible thinking layer, not a reply.

import { i18n } from '../i18n.js';
import { spkColor } from '../utils/speaker-colors.js';

function clockNow() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    const mmm = String(d.getMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss}.${mmm}`;
}

export class Conversation {
    constructor() {
        this._el = null;
        this._hint = null;
        this._laneLive = null;
        this._lanePrefill = null;
        this._liveEntity = null;   // DOM node accumulating speech_token text
        this._partialTurn = null;
        this._turnsBySpan = new Map();
        this._aliasKey = 'dr_speaker_aliases';
        this._diarizeProgress = 0;
        this._diarizePhase = 'idle';
        this._laneProgressFill = null;
        // Sticky-bottom intent per lane: auto-follow newest text by default,
        // but release the lane the moment the reader scrolls up to revisit
        // earlier dialogue so incoming events stop yanking them to the foot.
        this._stickLive = true;
        this._stickPrefill = true;
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
        this._trackStick(this._laneLive, v => { this._stickLive = v; });
        this._trackStick(this._lanePrefill, v => { this._stickPrefill = v; });
        this._renderTitles();
        this._renderHint();
        i18n.onChange(() => {
            this._renderTitles();
            this._renderHint();
        });
    }

    _renderTitles() {
        const liveTitle = this._el?.querySelector('[data-role=title-live]');
        if (liveTitle) {
            const label = document.createElement('span');
            label.textContent = i18n.t('turn.lane_live');
            const bar = document.createElement('span');
            bar.className = 'lane-progress';
            bar.setAttribute('aria-hidden', 'true');
            const fill = document.createElement('span');
            fill.className = 'lane-progress__fill';
            fill.dataset.state = this._diarizePhase || 'idle';
            fill.style.width = `${Math.round((this._diarizeProgress || 0) * 100)}%`;
            bar.appendChild(fill);
            liveTitle.replaceChildren(label, bar);
            this._laneProgressFill = fill;
        }
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

    // A lane is "stuck" when the reader is parked within a small margin of the
    // foot. Re-evaluate on every user scroll so releasing/re-engaging follow is
    // automatic: scroll up to read history (release), scroll back down (re-engage).
    _trackStick(lane, set) {
        if (!lane) return;
        const NEAR_BOTTOM_PX = 80;
        lane.addEventListener('scroll', () => {
            const gap = lane.scrollHeight - lane.scrollTop - lane.clientHeight;
            set(gap <= NEAR_BOTTOM_PX);
        }, { passive: true });
    }

    onMessage(msg) {
        switch (msg.type) {
            case 'asr_partial':           return this._partial(msg);
            case 'asr_transcript':        return this._heard(msg);
            case 'asr_transcript_amend':  return this._amend(msg);
            case 'speaker_relabel':       return this._relabel(msg);
            case 'speaker_name':          return this._updateSpeakerName(msg);
            case 'speaker_diarize_status': return this._updateProgressBar(msg);
            case 'speech_token':          return this._streamToken(msg);
            case 'consciousness_decode':  return this._decode(msg);
            case 'text_input_ack':        return;   // echo handled by composer
        }
    }

    // `useAlias` selects whether the localStorage alias map (keyed by the
    // DiariZen S<gid> identity space, written by the contacts roster) may be
    // consulted. The live `asr_transcript` line carries the provisional ONLINE
    // dual_db_ id, a DIFFERENT identity space that overlaps numerically near 0,
    // so applying a gid-keyed alias there mislabels speakers (online-id-3 would
    // borrow gid-3's name). Only the authoritative `asr_transcript_amend` (gid
    // space) is allowed to resolve aliases.
    _speakerLabel(id, name, useAlias = true) {
        let alias = '';
        if (useAlias && id >= 0) {
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

    _partial(msg) {
        if (!msg.text || !msg.text.trim()) return;
        this._clearHint();
        const text = msg.text.trim();
        // Recover if the live bubble was consumed or migrated to the prefill
        // lane: a stale pointer would silently swallow every new partial.
        if (this._partialTurn && this._partialTurn.parentNode !== this._laneLive) {
            this._partialTurn = null;
        }
        if (!this._partialTurn) {
            this._partialTurn = this._appendTurn({
                who: i18n.t('turn.asr_live'),
                color: 'var(--c-speak)',
                text,
                entity: false,
                lane: 'live',
            });
            this._partialTurn.classList.add('turn--partial');
        } else {
            this._partialTurn.querySelector('.turn__text').textContent = text;
        }
        this._scroll();
    }

    // A human utterance was transcribed.
    _heard(msg) {
        if (!msg.text || !msg.text.trim()) return;
        this._clearHint();
        const id = (typeof msg.speaker_id === 'number') ? msg.speaker_id : -1;
        // Provisional ONLINE identity — no gid-alias lookup (wrong id space).
        const name = this._speakerLabel(id, msg.speaker_name, false);

        let turn = this._partialTurn;
        if (turn) {
            turn.classList.remove('turn--partial');
            turn.querySelector('.turn__dot').style.background = spkColor(id);
            turn.querySelector('.turn__name').textContent = name;
            turn.querySelector('.turn__time').textContent = clockNow();
            turn.querySelector('.turn__text').textContent = msg.text;
        } else {
            turn = this._appendTurn({ who: name, color: spkColor(id), text: msg.text, entity: false, lane: 'live' });
        }
        this._partialTurn = null;

        const note = document.createElement('div');
        note.className = 'turn__amend';
        note.innerHTML = `
            <span class="turn__amend-tag turn__amend-tag--online">${i18n.t('turn.online')}</span>
            <span>${name}</span>`;
        turn.appendChild(note);
        this._turnsBySpan.set(this._turnKey(msg), {
            turn,
            speakerId: id,
            onlineLabel: name,
            finalLabel: null,
        });
        this._scroll();
    }

    _amend(msg) {
        const rec = this._turnsBySpan.get(this._turnKey(msg));
        if (!rec) return;
        const id = (typeof msg.speaker_id === 'number') ? msg.speaker_id : -1;
        const finalLabel = this._speakerLabel(id, msg.speaker_name);
        // Adopt the authoritative DiariZen gid as this turn's tracked identity
        // so later renames (speaker_name, gid space) and relabels match it. The
        // turn started life tagged with the provisional online id; from the
        // amend onward DiariZen is the single authority for this row.
        rec.speakerId = id;
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
        if (rec.turn === this._partialTurn) this._partialTurn = null;
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
        // Only follow the foot for lanes the reader has left parked at the
        // bottom; lanes scrolled up to review earlier dialogue stay put.
        if (this._laneLive && this._stickLive)
            this._laneLive.scrollTop = this._laneLive.scrollHeight;
        if (this._lanePrefill && this._stickPrefill)
            this._lanePrefill.scrollTop = this._lanePrefill.scrollHeight;
    }

    // Periodic reclusterer reassigned old_id → new_id; update all unconfirmed turns.
    _relabel(msg) {
        const oldId = Number(msg.old_id);
        const newId = Number(msg.new_id);
        if (!Number.isFinite(oldId) || !Number.isFinite(newId)) return;
        for (const rec of this._turnsBySpan.values()) {
            if (rec.speakerId !== oldId) continue;
            rec.speakerId = newId;
            const newName = this._speakerLabel(newId, '');
            rec.turn.querySelector('.turn__name').textContent = newName;
            rec.turn.querySelector('.turn__dot').style.background = spkColor(newId);
            const note = rec.turn.querySelector('.turn__amend');
            if (note && rec.finalLabel === null) {
                // was online-only, now amend it
                note.innerHTML = `
                    <span class="turn__amend-tag turn__amend-tag--online">${i18n.t('turn.online')}</span>
                    <span>${rec.onlineLabel}</span>
                    <span>-></span>
                    <span class="turn__amend-tag turn__amend-tag--prefill">${i18n.t('turn.prefill')}</span>
                    <span>${newName}</span>`;
                rec.finalLabel = newName;
                rec.turn.classList.add('turn--amended');
                if (rec.turn === this._partialTurn) this._partialTurn = null;
                this._lanePrefill.appendChild(rec.turn);
            } else if (note && rec.finalLabel !== null) {
                // already amended (by holdback); update the prefill span only
                const spans = note.querySelectorAll('span:not(.turn__amend-tag)');
                if (spans.length >= 1) spans[spans.length - 1].textContent = newName;
                rec.finalLabel = newName;
            }
        }
        this._scroll();
    }

    // User renamed a speaker; update any rendered turn that has that speaker ID.
    _updateSpeakerName(msg) {
        const id = Number(msg.id ?? msg.speaker_id);
        const name = String(msg.name ?? msg.speaker_name ?? '').trim();
        if (!Number.isFinite(id) || !name) return;
        for (const rec of this._turnsBySpan.values()) {
            if (rec.speakerId !== id) continue;
            const label = name || this._speakerLabel(id, '');
            rec.turn.querySelector('.turn__name').textContent = label;
            // Update whichever amend slot is newest
            if (rec.finalLabel !== null) rec.finalLabel = label;
            else rec.onlineLabel = label;
        }
    }

    // Periodic re-ID status: update the progress bar fill in the live lane title.
    _updateProgressBar(msg) {
        this._diarizeProgress = Number(msg.cycle_progress ?? 0);
        this._diarizePhase = msg.phase || 'idle';
        if (this._laneProgressFill) {
            this._laneProgressFill.style.width = `${Math.round(this._diarizeProgress * 100)}%`;
            this._laneProgressFill.dataset.state = this._diarizePhase;
        }
    }

    unmount() { this._el?.remove(); this._el = null; }
}
