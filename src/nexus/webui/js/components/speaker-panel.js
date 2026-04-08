// speaker-panel.js — Speaker identification panel component.
// Displays current speaker, similarity bar, threshold control, and speaker DB management.

export class SpeakerPanel {
    constructor(ws) {
        this._ws = ws;

        // DOM refs.
        this._toggleBtn      = document.getElementById('speaker-toggle');
        this._clearBtn       = document.getElementById('speaker-clear');
        this._idVal           = document.getElementById('speaker-id-val');
        this._nameVal         = document.getElementById('speaker-name-val');
        this._simVal          = document.getElementById('speaker-sim-val');
        this._simBar          = document.getElementById('speaker-sim-bar');
        this._thresholdSlider = document.getElementById('speaker-threshold');
        this._thresholdVal    = document.getElementById('speaker-threshold-val');
        this._countVal        = document.getElementById('speaker-count');
        this._nameIdInput     = document.getElementById('speaker-name-id');
        this._nameTextInput   = document.getElementById('speaker-name-text');
        this._nameSetBtn      = document.getElementById('speaker-name-set');

        this._enabled = true;

        // Events.
        this._toggleBtn?.addEventListener('click', () => this._toggleEnabled());
        this._clearBtn?.addEventListener('click', () => this._clearDb());
        this._thresholdSlider?.addEventListener('input', () => this._onThresholdChange());
        this._nameSetBtn?.addEventListener('click', () => this._setName());
    }

    enable() {
        if (this._toggleBtn) this._toggleBtn.disabled = false;
        if (this._clearBtn)  this._clearBtn.disabled = false;
        if (this._nameSetBtn) this._nameSetBtn.disabled = false;
    }

    disable() {
        if (this._toggleBtn) this._toggleBtn.disabled = true;
        if (this._clearBtn)  this._clearBtn.disabled = true;
        if (this._nameSetBtn) this._nameSetBtn.disabled = true;
    }

    // Called on every pipeline_stats message.
    updateFromStats(stats) {
        const isActive = stats.speaker_active === true;

        // Only update speaker ID / sim / name when there's a fresh extraction.
        if (isActive && stats.speaker_id !== undefined) {
            const id = stats.speaker_id;
            this._idVal.textContent = id >= 0 ? `#${id}` : '—';
            this._nameVal.textContent = stats.speaker_name || '';
            this._simVal.textContent = stats.speaker_sim >= 0 ? stats.speaker_sim.toFixed(3) : '—';

            const pct = Math.min(stats.speaker_sim * 100, 100);
            this._simBar.style.width = `${pct}%`;

            // Flash the speaker-current container.
            const currentEl = document.getElementById('speaker-current');
            if (currentEl) {
                currentEl.classList.add('speaker-current--active');
                setTimeout(() => currentEl.classList.remove('speaker-current--active'), 1000);
            }
        }
        // Always update count.
        if (stats.speaker_count !== undefined) {
            this._countVal.textContent = stats.speaker_count;
        }
        // Sync enable state from server.
        if (stats.speaker_enabled !== undefined) {
            this._enabled = stats.speaker_enabled;
            this._toggleBtn.classList.toggle('btn--active', this._enabled);
            this._toggleBtn.textContent = this._enabled ? 'Enabled' : 'Disabled';
            this._toggleBtn.setAttribute('aria-pressed', this._enabled);
        }
        // Sync threshold from server.
        if (stats.speaker_threshold !== undefined && !this._thresholdSlider.matches(':active')) {
            this._thresholdSlider.value = stats.speaker_threshold;
            this._thresholdVal.textContent = stats.speaker_threshold.toFixed(2);
        }
    }

    // Called on dedicated speaker event.
    updateSpeaker(ev) {
        this._idVal.textContent = ev.id >= 0 ? `#${ev.id}` : '—';
        this._nameVal.textContent = ev.name || '';
        this._simVal.textContent = ev.sim.toFixed(3);
        const pct = Math.min(ev.sim * 100, 100);
        this._simBar.style.width = `${pct}%`;
    }

    _toggleEnabled() {
        this._enabled = !this._enabled;
        this._toggleBtn.classList.toggle('btn--active', this._enabled);
        this._toggleBtn.textContent = this._enabled ? 'Enabled' : 'Disabled';
        this._toggleBtn.setAttribute('aria-pressed', this._enabled);
        this._ws.sendText(`speaker_enable:${this._enabled ? 'on' : 'off'}`);
    }

    _clearDb() {
        this._ws.sendText('speaker_clear');
        this._idVal.textContent = '—';
        this._nameVal.textContent = '';
        this._simVal.textContent = '—';
        this._simBar.style.width = '0%';
        this._countVal.textContent = '0';
    }

    _onThresholdChange() {
        const v = parseFloat(this._thresholdSlider.value);
        this._thresholdVal.textContent = v.toFixed(2);
        this._ws.sendText(`speaker_threshold:${v.toFixed(2)}`);
    }

    _setName() {
        const id = parseInt(this._nameIdInput.value, 10);
        const name = this._nameTextInput.value.trim();
        if (isNaN(id) || !name) return;
        this._ws.sendText(`speaker_name:${id}:${name}`);
        this._nameTextInput.value = '';
    }
}
