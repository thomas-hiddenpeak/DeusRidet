// contacts.js — IM-style contact roster.
// Groups recognized speakers into named and unnamed lists and supports rename.

import { i18n } from '../i18n.js';
import { spkColor } from '../utils/speaker-colors.js';

export class Contacts {
    constructor() {
        this._el = null;
        this._activeCard = null;
        this._knownList = null;
        this._unknownList = null;
        this._knownTitle = null;
        this._unknownTitle = null;
        this._seen = new Map(); // id -> { name, updatedAt }
        this._links = new Map(); // live id -> { stableId, score, committed }
        this._merged = new Map(); // src id -> dst id (folded contacts)
        this._dismissed = new Set();
        this._activeId = -1;
        this._tick = 0;
        this._aliasKey = 'dr_speaker_aliases';
        this._aliases = this._loadAliases();
        this._uuidKey = 'dr_speaker_uuids';
        this._uuids = this._loadUuids();
        this._renamingId = -1;
        this._renameDraft = '';
        this._mergingId = -1;
        this._mergeTargetId = -1;
        this._unknownActionId = -1;
        this._copiedId = -1;
        this._copyTimer = null;
        this.onRename = null;
        this.onMerge = null;
        this.onRemove = null;
    }

    _loadUuids() {
        try {
            const raw = localStorage.getItem(this._uuidKey);
            const obj = raw ? JSON.parse(raw) : {};
            return (obj && typeof obj === 'object') ? obj : {};
        } catch {
            return {};
        }
    }

    _saveUuids() {
        localStorage.setItem(this._uuidKey, JSON.stringify(this._uuids));
    }

    /** Stable per-speaker UUID; the identity key duplicate names can hang off. */
    _uuidFor(id) {
        const key = String(id);
        let uuid = this._uuids[key];
        if (!uuid) {
            uuid = (crypto?.randomUUID)
                ? crypto.randomUUID()
                : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
                    const r = (Math.random() * 16) | 0;
                    const v = c === 'x' ? r : (r & 0x3) | 0x8;
                    return v.toString(16);
                });
            this._uuids[key] = uuid;
            this._saveUuids();
        }
        return uuid;
    }

    _loadAliases() {
        try {
            const raw = localStorage.getItem(this._aliasKey);
            const obj = raw ? JSON.parse(raw) : {};
            return (obj && typeof obj === 'object') ? obj : {};
        } catch {
            return {};
        }
    }

    _saveAliases() {
        localStorage.setItem(this._aliasKey, JSON.stringify(this._aliases));
    }

    /** Re-send all stored aliases to the server after reconnect. */
    reapplyAliases(sendText) {
        if (typeof sendText !== 'function') return;
        for (const [idStr, alias] of Object.entries(this._aliases)) {
            const id = Number(idStr);
            if (!Number.isInteger(id) || id < 0 || !alias) continue;
            sendText(`speaker_name:${id}:${alias}`);
        }
    }

    mount(parent) {
        const el = document.createElement('section');
        el.className = 'contacts';
        el.setAttribute('aria-label', 'Recognized contacts');
        el.innerHTML = `
            <header class="contacts__header">
                <h2 class="contacts__title" data-role="title"></h2>
            </header>
            <section class="contacts__active" data-role="active"></section>
            <section class="contacts__group">
                <h3 class="contacts__group-title" data-role="known-title"></h3>
                <ul class="contacts__list" data-role="known-list"></ul>
            </section>
            <section class="contacts__group">
                <h3 class="contacts__group-title" data-role="unknown-title"></h3>
                <ul class="contacts__list" data-role="unknown-list"></ul>
            </section>`;
        parent.appendChild(el);
        this._el = el;
        this._knownList = el.querySelector('[data-role=known-list]');
        this._unknownList = el.querySelector('[data-role=unknown-list]');
        this._knownTitle = el.querySelector('[data-role=known-title]');
        this._unknownTitle = el.querySelector('[data-role=unknown-title]');
        this._activeCard = el.querySelector('[data-role=active]');
        this.render();
        i18n.onChange(() => this.render());
    }

    /** Parse a DiariZen global stitched label "S<gid>" → numeric gid (-1 if not). */
    _parseGid(label) {
        if (typeof label !== 'string') return -1;
        const m = label.match(/^S(\d+)$/);
        return m ? Number(m[1]) : -1;
    }

    onMessage(msg) {
        if (msg.type === 'speaker_diarize_partial' || msg.type === 'speaker_diarize_final') {
            // DiariZen's identity registry is the SINGLE speaker authority.
            // Every windowed pass broadcasts segments whose label is the
            // global stitched id "S<gid>"; we mint each gid we hear and mark
            // the latest as the active speaker. This replaces the old online
            // dual_db_ id space (`speaker` / `speaker_amend`) which minted
            // numerically-overlapping "ghost" contacts.
            const segs = Array.isArray(msg.segments) ? msg.segments : [];
            let lastGid = -1;
            for (const seg of segs) {
                const label = Array.isArray(seg) ? seg[2] : (seg && seg.label);
                const gid = this._parseGid(label);
                if (gid < 0) continue;
                this._touch(gid, '', false);
                lastGid = gid;
            }
            if (lastGid >= 0) this._activeId = lastGid;
        } else if (msg.type === 'asr_transcript_amend') {
            // Post-holdback utterance carries the DiariZen gid — the SAME
            // space as the diarize segments above. The plain `asr_transcript`
            // carries the online dual_db_ id (a different, ghost-prone space)
            // and is intentionally ignored so the roster has one authority.
            const id = Number(msg.speaker_id);
            if (!Number.isInteger(id) || id < 0) return;
            this._touch(id, msg.speaker_name || '', false);
        } else if (msg.type === 'speaker_name') {
            // Rename echo: id is the gid the user just named. Refresh display.
            const id = Number(msg.id);
            if (!Number.isInteger(id) || id < 0) return;
            this._touch(id, msg.name || '', false);
        } else {
            // Everything else (`speaker`, `speaker_amend`, `asr_transcript`,
            // `speaker_relabel`, `speaker_id_link`, `pipeline_stats`) belongs
            // to the online / legacy / shadow id spaces. None of them mints
            // roster identity any more — DiariZen is the sole authority. This
            // is the structural fix for "post-processing conjures a new
            // speaker": there is now exactly one id space on screen.
            return;
        }
        // While a rename/merge editor is open, defer message-driven DOM
        // rebuilds. The full re-render recreates the <input>, destroying
        // the focus AND any in-flight IME composition (Chinese names),
        // which is exactly why the box "loses focus before you can type".
        // State is already updated via _touch; the editor's own
        // commit/cancel path triggers a fresh render afterwards.
        if (this._renamingId < 0 && this._mergingId < 0) {
            this.render();
        }
    }

    _touch(id, rawName, fromRoster) {
        // Redirect any activity of a folded speaker onto its merge target.
        let guard = 0;
        while (this._merged.has(id) && guard++ < 8) id = this._merged.get(id);
        if (fromRoster && this._dismissed.has(id)) return;
        if (!fromRoster) this._dismissed.delete(id);
        this._tick += 1;
        const alias = (this._aliases[String(id)] || '').trim();
        const backendName = (typeof rawName === 'string') ? rawName.trim() : '';
        const name = alias || backendName;
        this._seen.set(id, { name, updatedAt: this._tick });
    }

    _isKnown(info) {
        return !!(info.name && info.name.trim());
    }

    _displayName(id, info) {
        if (this._isKnown(info)) return info.name;
        return `${i18n.t('turn.speaker_prefix')} ${id}`;
    }

    _copyUuid(id) {
        const uuid = this._uuidFor(id);
        const markCopied = () => {
            this._copiedId = id;
            if (this._copyTimer) clearTimeout(this._copyTimer);
            this._copyTimer = setTimeout(() => {
                this._copiedId = -1;
                this.render();
            }, 1200);
        };
        if (navigator?.clipboard?.writeText) {
            navigator.clipboard.writeText(uuid)
                .then(() => {
                    markCopied();
                    this.render();
                })
                .catch(() => {
                    this._copiedId = -1;
                    this.render();
                });
            return;
        }
        this._copiedId = -1;
    }

    _startRename(id) {
        const info = this._seen.get(id);
        this._renamingId = id;
        this._renameDraft = this._isKnown(info || {}) ? info.name : '';
        this._mergingId = -1;
        this._unknownActionId = -1;
        this.render();
    }

    _cancelRename() {
        this._renamingId = -1;
        this._renameDraft = '';
        this.render();
    }

    _commitRename(id) {
        const trimmed = (this._renameDraft || '').trim();
        if (!trimmed) return;
        this._aliases[String(id)] = trimmed;
        this._saveAliases();
        this._seen.set(id, { name: trimmed, updatedAt: this._tick + 1 });
        this._renamingId = -1;
        this._renameDraft = '';
        this.onRename?.(id, trimmed);
        this.render();
    }

    _startMerge(id) {
        const targets = [...this._seen.keys()].filter((tid) => tid !== id);
        this._mergingId = id;
        this._mergeTargetId = targets.length > 0 ? targets[0] : -1;
        this._renamingId = -1;
        this._unknownActionId = -1;
        this.render();
    }

    _cancelMerge() {
        this._mergingId = -1;
        this._mergeTargetId = -1;
        this.render();
    }

    _mergeInto(srcId, dstId) {
        const dstInfo = this._seen.get(dstId) || { name: '' };
        const dstName = (dstInfo.name || '').trim();

        this._merged.set(srcId, dstId);
        this._seen.delete(srcId);
        this._links.delete(srcId);
        delete this._aliases[String(srcId)];
        this._saveAliases();
        // Fold identity: the source adopts the target's UUID so a duplicate
        // cluster of the same person resolves to one stable identity.
        this._uuids[String(srcId)] = this._uuidFor(dstId);
        this._saveUuids();
        if (this._activeId === srcId) this._activeId = dstId;
        const di = this._seen.get(dstId);
        if (di) this._seen.set(dstId, { ...di, updatedAt: this._tick + 1 });

        this._mergingId = -1;
        this._mergeTargetId = -1;
        this.onMerge?.(srcId, dstId, dstName);
        this.render();
    }

    _commitMerge(srcId) {
        const dstId = Number(this._mergeTargetId);
        if (!Number.isInteger(dstId) || dstId < 0 || dstId === srcId) return;
        this._mergeInto(srcId, dstId);
    }

    _renderList(target, rows) {
        target.replaceChildren(...rows.map(([id, info]) => {
            const li = document.createElement('li');
            li.className = 'contacts__item';

            const row = document.createElement('div');
            row.className = 'contacts__row';

            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'contacts__chip' + (id === this._activeId ? ' contacts__chip--active' : '');
            btn.title = i18n.t('contacts.copy_id');

            const dot = document.createElement('span');
            dot.className = 'contacts__dot';
            dot.style.background = spkColor(id);

            const name = document.createElement('span');
            name.className = 'contacts__name';
            name.textContent = this._displayName(id, info);

            const meta = document.createElement('span');
            meta.className = 'contacts__meta';
            const baseMeta = this._isKnown(info)
                ? i18n.t('contacts.meta_named')
                : i18n.t('contacts.meta_unknown');
            const link = this._links.get(id);
            if (link && Number.isInteger(link.stableId) && link.stableId >= 0) {
                const pct = Math.round(Math.max(0, Math.min(1, Number(link.score || 0))) * 100);
                const lead = link.committed ? i18n.t('contacts.link_stable') : i18n.t('contacts.link_candidate');
                meta.textContent = `${baseMeta} · ${lead} S${link.stableId} ${pct}%`;
            } else {
                meta.textContent = baseMeta;
            }
            if (this._copiedId === id) {
                meta.textContent = `${meta.textContent} · ${i18n.t('contacts.copied')}`;
            }

            btn.append(dot, name, meta);
            btn.addEventListener('click', () => {
                this._copyUuid(id);
                if (!this._isKnown(info)) {
                    this._unknownActionId = this._unknownActionId === id ? -1 : id;
                    this._renamingId = -1;
                    this._mergingId = -1;
                }
                this.render();
            });

            const rename = document.createElement('button');
            rename.type = 'button';
            rename.className = 'contacts__rename';
            rename.title = i18n.t('contacts.rename');
            rename.setAttribute('aria-label', i18n.t('contacts.rename'));
            rename.textContent = '✎';
            rename.addEventListener('click', () => this._startRename(id));

            const merge = document.createElement('button');
            merge.type = 'button';
            merge.className = 'contacts__merge';
            merge.title = i18n.t('contacts.merge');
            merge.setAttribute('aria-label', i18n.t('contacts.merge'));
            merge.textContent = '⥃';
            merge.addEventListener('click', () => this._startMerge(id));

            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'contacts__remove';
            remove.title = i18n.t('contacts.remove');
            remove.setAttribute('aria-label', i18n.t('contacts.remove'));
            remove.textContent = '×';
            remove.addEventListener('click', () => {
                delete this._aliases[String(id)];
                this._saveAliases();
                this._dismissed.add(id);
                this._seen.delete(id);
                this._links.delete(id);
                if (this._activeId === id) this._activeId = -1;
                if (this._renamingId === id) this._renamingId = -1;
                if (this._mergingId === id) this._mergingId = -1;
                if (this._unknownActionId === id) this._unknownActionId = -1;
                // Forget the prototype on the backend live matcher too, else a
                // stranger keeps matching the deleted speaker's stale voiceprint.
                this.onRemove?.(id);
                this.render();
            });

            row.append(btn, rename, merge, remove);
            li.appendChild(row);

            if (!this._isKnown(info) && this._unknownActionId === id &&
                this._renamingId !== id && this._mergingId !== id) {
                const actions = document.createElement('div');
                actions.className = 'contacts__inline';

                const nameBtn = document.createElement('button');
                nameBtn.type = 'button';
                nameBtn.className = 'contacts__action';
                nameBtn.textContent = i18n.t('contacts.action_name');
                nameBtn.addEventListener('click', () => this._startRename(id));

                const mergeBtn = document.createElement('button');
                mergeBtn.type = 'button';
                mergeBtn.className = 'contacts__action';
                mergeBtn.textContent = i18n.t('contacts.action_merge');
                mergeBtn.addEventListener('click', () => this._startMerge(id));

                actions.append(nameBtn, mergeBtn);
                li.appendChild(actions);
            }

            if (this._renamingId === id) {
                const box = document.createElement('div');
                box.className = 'contacts__inline';

                const input = document.createElement('input');
                input.className = 'contacts__input';
                input.type = 'text';
                input.placeholder = i18n.t('contacts.name_placeholder');
                input.value = this._renameDraft;
                input.addEventListener('input', (e) => {
                    this._renameDraft = e.target.value;
                });
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') this._commitRename(id);
                    if (e.key === 'Escape') this._cancelRename();
                });

                const save = document.createElement('button');
                save.type = 'button';
                save.className = 'contacts__action';
                save.textContent = i18n.t('contacts.save');
                save.addEventListener('click', () => this._commitRename(id));

                const cancel = document.createElement('button');
                cancel.type = 'button';
                cancel.className = 'contacts__action';
                cancel.textContent = i18n.t('contacts.cancel');
                cancel.addEventListener('click', () => this._cancelRename());

                box.append(input, save, cancel);
                li.appendChild(box);
            }

            if (this._mergingId === id) {
                const box = document.createElement('div');
                box.className = 'contacts__inline';
                const targetRows = [...this._seen.entries()].filter(([tid]) => tid !== id);
                if (targetRows.length === 0) {
                    const empty = document.createElement('span');
                    empty.className = 'contacts__empty';
                    empty.textContent = i18n.t('contacts.merge_none');
                    box.appendChild(empty);
                } else {
                    const select = document.createElement('select');
                    select.className = 'contacts__input';
                    for (const [tid, tinfo] of targetRows) {
                        const opt = document.createElement('option');
                        opt.value = String(tid);
                        opt.textContent = this._displayName(tid, tinfo);
                        if (tid === this._mergeTargetId) opt.selected = true;
                        select.appendChild(opt);
                    }
                    select.addEventListener('change', (e) => {
                        this._mergeTargetId = Number(e.target.value);
                    });

                    const save = document.createElement('button');
                    save.type = 'button';
                    save.className = 'contacts__action';
                    save.textContent = i18n.t('contacts.save');
                    save.addEventListener('click', () => this._commitMerge(id));

                    const cancel = document.createElement('button');
                    cancel.type = 'button';
                    cancel.className = 'contacts__action';
                    cancel.textContent = i18n.t('contacts.cancel');
                    cancel.addEventListener('click', () => this._cancelMerge());

                    box.append(select, save, cancel);
                }
                li.appendChild(box);
            }

            return li;
        }));
    }

    render() {
        if (!this._el) return;
        // Capture editor focus + caret before the lists are rebuilt, so a
        // render that slips through while an editor is open (e.g. an i18n
        // language change) does not silently steal focus from the input.
        const active = document.activeElement;
        const wasEditing = active &&
            active.classList?.contains('contacts__input') &&
            this._el.contains(active);
        const caretStart = wasEditing && typeof active.selectionStart === 'number'
            ? active.selectionStart : null;
        const caretEnd = wasEditing && typeof active.selectionEnd === 'number'
            ? active.selectionEnd : null;
        this._el.querySelector('[data-role=title]').textContent = i18n.t('contacts.title');

        const activeInfo = this._activeId >= 0 ? this._seen.get(this._activeId) : null;
        if (this._activeCard) {
            if (activeInfo) {
                this._activeCard.innerHTML = '';
                const card = document.createElement('article');
                card.className = 'contacts__active-card';
                card.innerHTML = `
                    <span class="contacts__active-kicker">${i18n.t('contacts.now_speaking')}</span>
                    <span class="contacts__active-name"></span>
                    <span class="contacts__active-meta"></span>`;
                card.querySelector('.contacts__active-name').textContent = this._displayName(this._activeId, activeInfo);
                card.querySelector('.contacts__active-meta').textContent = this._isKnown(activeInfo)
                    ? i18n.t('contacts.meta_named')
                    : i18n.t('contacts.meta_unknown');
                this._activeCard.appendChild(card);
            } else {
                this._activeCard.innerHTML = `<div class="contacts__empty">${i18n.t('contacts.no_active')}</div>`;
            }
        }

        const rows = [...this._seen.entries()]
            .sort((a, b) => {
                if (a[0] === this._activeId) return -1;
                if (b[0] === this._activeId) return 1;
                return (b[1].updatedAt || 0) - (a[1].updatedAt || 0);
            });
        const known = rows.filter(([, info]) => this._isKnown(info));
        const unknown = rows.filter(([, info]) => !this._isKnown(info));

        this._knownTitle.textContent = `${i18n.t('contacts.known')} (${known.length})`;
        this._unknownTitle.textContent = `${i18n.t('contacts.unknown')} (${unknown.length})`;

        if (known.length === 0) {
            this._knownList.innerHTML = `<li class="contacts__empty">${i18n.t('contacts.empty_known')}</li>`;
        } else {
            this._renderList(this._knownList, known);
        }

        if (unknown.length === 0) {
            this._unknownList.innerHTML = `<li class="contacts__empty">${i18n.t('contacts.empty_unknown')}</li>`;
        } else {
            this._renderList(this._unknownList, unknown);
        }

        // Restore editor focus + caret after the rebuild.
        if (wasEditing) {
            const input = this._el.querySelector('.contacts__input');
            if (input) {
                input.focus();
                if (caretStart !== null && typeof input.setSelectionRange === 'function') {
                    try { input.setSelectionRange(caretStart, caretEnd); } catch { /* select elements have no range */ }
                }
            }
        }
    }

    unmount() {
        this._el?.remove();
        this._el = null;
    }
}
