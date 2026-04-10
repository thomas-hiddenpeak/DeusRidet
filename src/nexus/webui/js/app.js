// app.js — DeusRidet WebUI bootstrap.
// Connects to the backend WebSocket server, initialises components,
// and provides a lightweight log display.

import { WsClient } from './ws-client.js';
import { AudioPanel } from './components/audio-panel.js';
import { SpeakerDebugPanel } from './components/speaker-debug-panel.js';
import { AsrPanel } from './components/asr-panel.js';
import { AsrTranscriptPanel } from './components/asr-transcript-panel.js';
import { AsrLogPanel } from './components/asr-log-panel.js';
import { ConsciousnessPanel } from './components/consciousness-panel.js';
import { TextOutputPanel } from './components/text-output-panel.js';
import { ConfigPanel } from './components/config-panel.js';
import { TrackerPanel } from './components/tracker-panel.js';
import { spkColor } from './utils/speaker-colors.js';

// --- Log utility ---
const logEl = document.getElementById('log-output');
function log(msg) {
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    logEl.textContent += `[${ts}] ${msg}\n`;
    const lines = logEl.textContent.split('\n');
    if (lines.length > 200) logEl.textContent = lines.slice(-200).join('\n');
    logEl.scrollTop = logEl.scrollHeight;
}

// --- WebSocket connection ---
const ws = new WsClient();
const statusEl = document.getElementById('conn-status');

ws.onOpen = () => {
    statusEl.textContent = 'Connected';
    statusEl.classList.add('connected');
    audioPanel.enable();
    log('WebSocket connected');
};

ws.onClose = () => {
    statusEl.textContent = 'Disconnected';
    statusEl.classList.remove('connected');
    audioPanel.disable();
    log('WebSocket disconnected — reconnecting...');
};

ws.onText = (msg) => {
    try {
        const obj = JSON.parse(msg);
        if (obj.type === 'audio_stats') {
            audioPanel.updateServerStats(obj);
            return;
        }
        if (obj.type === 'pipeline_stats') {
            audioPanel.updatePipelineStats(obj);
            updateSpeakerPanel(obj);
            speakerDebug.onPipelineStats(obj);
            asrPanel.onPipelineStats(obj);
            trackerPanel.onPipelineStats(obj);
            return;
        }
        if (obj.type === 'vad') {
            audioPanel.updateVad(obj);
            if (obj.event === 'start') log('VAD: speech started');
            if (obj.event === 'end') log('VAD: speech ended');
            return;
        }
        if (obj.type === 'speaker') {
            log(`Speaker: id=${obj.id} sim=${obj.sim.toFixed(3)} ${obj.new ? 'NEW' : ''} ${obj.name || '(unnamed)'}`);
            speakerDebug.onSpeakerEvent(obj);
            return;
        }
        if (obj.type === 'speaker_debug') {
            speakerDebug.onDebugData(obj);
            return;
        }
        if (obj.type === 'asr_transcript') {
            asrPanel.onTranscript(obj);
            asrTranscriptPanel.onTranscript(obj);
            asrLogPanel.onTranscript(obj);
            textOutputPanel.onAsrTranscript(obj);
            log(`ASR: "${obj.text}" (${obj.latency_ms.toFixed(0)}ms, ${obj.audio_sec.toFixed(1)}s)`);
            return;
        }
        if (obj.type === 'asr_partial') {
            asrPanel.onPartial(obj);
            return;
        }
        if (obj.type === 'asr_log') {
            asrLogPanel.onAsrLog(obj);
            return;
        }
        if (obj.type === 'asr_enable') {
            asrPanel.onAsrEnable(obj);
            log(`ASR ${obj.enabled ? 'ON' : 'OFF'}`);
            return;
        }
        if (obj.type === 'asr_param') {
            asrPanel.onAsrParam(obj);
            log(`ASR param ${obj.key}=${obj.value}`);
            return;
        }
        if (obj.type === 'consciousness_state') {
            consciousnessPanel.onConsciousnessState(obj);
            configPanel.onConsciousnessState(obj);
            return;
        }
        if (obj.type === 'consciousness_enable') {
            consciousnessPanel.onConsciousnessEnable(obj);
            configPanel.onConsciousnessEnable(obj);
            log(`Consciousness ${obj.mode} ${obj.enabled ? 'ON' : 'OFF'}`);
            return;
        }
        if (obj.type === 'consciousness_param') {
            configPanel.onConsciousnessParam(obj);
            log(`Consciousness ${obj.key}=${obj.value}`);
            return;
        }
        if (obj.type === 'consciousness_prompt') {
            log(`${obj.pipeline || 'system'} prompt ${obj.ok ? 'updated' : 'failed'}`);
            return;
        }
        if (obj.type === 'consciousness_prompts') {
            configPanel.onConsciousnessPrompts(obj);
            return;
        }
        if (obj.type === 'consciousness_decode') {
            textOutputPanel.onDecode(obj);
            log(`[${obj.state}] ${obj.text} (${obj.tokens}tok ${obj.time_ms?.toFixed(0)}ms)`);
            return;
        }
        if (obj.type === 'speech_token') {
            textOutputPanel.onSpeechToken(obj);
            return;
        }
        if (obj.type === 'text_input_ack') {
            return;  // silently acknowledge
        }
        if (obj.type === 'asr_vad_source') {
            const map = {0:'silero', 1:'fsmn', 2:'ten', 3:'any', 4:'direct'};
            log(`ASR VAD source → ${map[obj.value] || obj.value}`);
            return;
        }
        if (obj.type === 'loopback') {
            log(`Loopback ${obj.enabled ? 'ON' : 'OFF'}`);
            return;
        }
        if (obj.type === 'vad_threshold') {
            log(`VAD threshold → ${obj.value}`);
            return;
        }
        if (obj.type === 'gain') {
            log(`Gain → ${obj.value}x`);
            return;
        }
        if (obj.type === 'silero_threshold') {
            log(`Silero threshold → ${obj.value}`);
            return;
        }
    } catch (_) { /* not JSON, show as text */ }
    log(`← ${msg}`);
};

ws.onBinary = (buf) => {
    audioPanel.playLoopback(buf);
};

// --- Components ---
const audioPanel = new AudioPanel(ws);
const speakerDebug = new SpeakerDebugPanel(ws);
const asrPanel = new AsrPanel(ws);
const asrTranscriptPanel = new AsrTranscriptPanel();
const asrLogPanel = new AsrLogPanel(ws);
const consciousnessPanel = new ConsciousnessPanel(ws);
const textOutputPanel = new TextOutputPanel(ws);
const configPanel = new ConfigPanel(ws);
const trackerPanel = new TrackerPanel(ws);

// --- VAD source selector ---
const vadSourceSelect = document.getElementById('vad-source-select');
if (vadSourceSelect) {
    vadSourceSelect.addEventListener('change', () => {
        ws.sendText(`vad_source:${vadSourceSelect.value}`);
    });
}

// ====================================================================
// Speaker ID panel — controls + roster
// ====================================================================
const MODELS = [
    { prefix: 'speaker',    enableCmd: 'speaker_enable',    thresholdCmd: 'speaker_threshold',
      clearCmd: 'speaker_clear',  nameCmd: 'speaker_name',  btnId: 'spk-en-campp',  label: 'CAM++',
      deleteCmd: null, mergeCmd: null },
    { prefix: 'wavlm',     enableCmd: 'wavlm_enable',      thresholdCmd: 'wavlm_threshold',
      clearCmd: 'wavlm_clear',   nameCmd: 'wavlm_name',    btnId: 'spk-en-wavlm',  label: 'WavLM',
      deleteCmd: null, mergeCmd: null },
    { prefix: 'unispeech',  enableCmd: 'unispeech_enable',  thresholdCmd: 'unispeech_threshold',
      clearCmd: 'unispeech_clear', nameCmd: 'unispeech_name', btnId: 'spk-en-ecapa', label: 'ECAPA',
      deleteCmd: null, mergeCmd: null },
    { prefix: 'wlecapa',   enableCmd: 'wlecapa_enable',    thresholdCmd: 'wlecapa_threshold',
      clearCmd: 'wlecapa_clear',  nameCmd: 'wlecapa_name',  btnId: 'spk-en-wlecapa', label: 'WL-ECAPA',
      deleteCmd: 'wlecapa_delete', mergeCmd: 'wlecapa_merge' },
];
const MODEL_BY_PREFIX = {};
MODELS.forEach(m => MODEL_BY_PREFIX[m.prefix] = m);

// Enable buttons — toggle per model.
MODELS.forEach(m => {
    const btn = document.getElementById(m.btnId);
    if (!btn) return;
    btn.addEventListener('click', () => {
        const on = btn.getAttribute('aria-pressed') === 'true';
        const next = !on;
        btn.classList.toggle('btn--active', next);
        btn.setAttribute('aria-pressed', next);
        ws.sendText(`${m.enableCmd}:${next ? 'on' : 'off'}`);
    });
});

// Clear All.
document.getElementById('spk-clear-all')?.addEventListener('click', () => {
    MODELS.forEach(m => ws.sendText(m.clearCmd));
});

// Settings toggle.
const settingsEl = document.getElementById('spk-settings');
document.getElementById('spk-settings-toggle')?.addEventListener('click', () => {
    const show = settingsEl.hidden;
    settingsEl.hidden = !show;
});

// Threshold sliders.
MODELS.forEach(m => {
    const slider = document.getElementById(`${m.prefix}-threshold`);
    const valEl = document.getElementById(`${m.prefix}-threshold-val`);
    if (!slider || !valEl) return;
    slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        valEl.textContent = v.toFixed(2);
        ws.sendText(`${m.thresholdCmd}:${v.toFixed(2)}`);
    });
});

// WL-ECAPA margin guard slider.
{
    const marginSlider = document.getElementById('wlecapa-margin');
    const marginVal = document.getElementById('wlecapa-margin-val');
    if (marginSlider && marginVal) {
        marginSlider.addEventListener('input', () => {
            const v = parseFloat(marginSlider.value);
            marginVal.textContent = v.toFixed(2);
            ws.sendText(`wlecapa_margin:${v.toFixed(2)}`);
        });
    }
}

// Early trigger controls.
{
    const earlyToggle = document.getElementById('spk-early-toggle');
    const earlySlider = document.getElementById('spk-early-slider');
    const earlyVal = document.getElementById('spk-early-val');
    let earlyEnabled = true;
    if (earlyToggle) {
        earlyToggle.addEventListener('click', () => {
            earlyEnabled = !earlyEnabled;
            earlyToggle.classList.toggle('btn--active', earlyEnabled);
            earlyToggle.setAttribute('aria-pressed', earlyEnabled);
            earlySlider.disabled = !earlyEnabled;
            ws.sendText(`early_enable:${earlyEnabled ? 'on' : 'off'}`);
        });
    }
    if (earlySlider && earlyVal) {
        earlySlider.addEventListener('input', () => {
            earlyVal.textContent = parseFloat(earlySlider.value).toFixed(1);
        });
        earlySlider.addEventListener('change', () => {
            ws.sendText(`early_trigger:${parseFloat(earlySlider.value).toFixed(2)}`);
        });
    }
    // Sync from server in updateSpeakerPanel.
    window._syncEarly = (stats) => {
        if (stats.early_trigger_sec !== undefined && earlySlider && !earlySlider.matches(':active')) {
            earlySlider.value = stats.early_trigger_sec;
            earlyVal.textContent = stats.early_trigger_sec.toFixed(1);
        }
        if (stats.early_enabled !== undefined && earlyToggle) {
            earlyEnabled = stats.early_enabled;
            earlyToggle.classList.toggle('btn--active', earlyEnabled);
            earlyToggle.setAttribute('aria-pressed', earlyEnabled);
            earlySlider.disabled = !earlyEnabled;
        }
    };
}

// Min speech duration control.
{
    const minSlider = document.getElementById('spk-minspeech-slider');
    const minVal = document.getElementById('spk-minspeech-val');
    if (minSlider && minVal) {
        minSlider.addEventListener('input', () => {
            minVal.textContent = parseFloat(minSlider.value).toFixed(1);
        });
        minSlider.addEventListener('change', () => {
            ws.sendText(`min_speech:${parseFloat(minSlider.value).toFixed(2)}`);
        });
    }
    window._syncMinSpeech = (stats) => {
        if (stats.min_speech_sec !== undefined && minSlider && !minSlider.matches(':active')) {
            minSlider.value = stats.min_speech_sec;
            minVal.textContent = stats.min_speech_sec.toFixed(1);
        }
    };
}

// --- Roster rendering ---
const rosterEl = document.getElementById('speaker-roster');
let lastRosterKey = '';   // serialized speaker_lists for change detection
let activeTimers = {};    // prefix → timeout id, for decaying highlight

function updateSpeakerPanel(stats) {
    // Sync enable buttons from server.
    MODELS.forEach(m => {
        const key = `${m.prefix}_enabled`;
        if (stats[key] !== undefined) {
            const btn = document.getElementById(m.btnId);
            if (btn) {
                btn.classList.toggle('btn--active', stats[key]);
                btn.setAttribute('aria-pressed', stats[key]);
            }
        }
        // Sync threshold.
        const tKey = `${m.prefix}_threshold`;
        if (stats[tKey] !== undefined) {
            const slider = document.getElementById(`${m.prefix}-threshold`);
            const valEl = document.getElementById(`${m.prefix}-threshold-val`);
            if (slider && valEl && !slider.matches(':active')) {
                slider.value = stats[tKey];
                valEl.textContent = stats[tKey].toFixed(2);
            }
        }
    });
    // Sync margin slider.
    if (stats.wlecapa_margin !== undefined) {
        const s = document.getElementById('wlecapa-margin');
        const v = document.getElementById('wlecapa-margin-val');
        if (s && v && !s.matches(':active')) {
            s.value = stats.wlecapa_margin;
            v.textContent = stats.wlecapa_margin.toFixed(2);
        }
    }
    // Sync VAD source.
    if (stats.vad_source !== undefined && vadSourceSelect && !vadSourceSelect.matches(':focus')) {
        const map = {0:'silero', 1:'fsmn', 2:'ten', 3:'any'};
        vadSourceSelect.value = map[stats.vad_source] || 'any';
    }

    // Sync early trigger and min speech from server.
    if (window._syncEarly) window._syncEarly(stats);
    if (window._syncMinSpeech) window._syncMinSpeech(stats);

    if (!rosterEl || !stats.speaker_lists) return;

    // Structural key: IDs + names + exemplar counts (not match counts or hits).
    const curKey = JSON.stringify(stats.speaker_lists.map(g => ({
        model: g.model,
        speakers: g.speakers.map(s => ({ id: s.id, name: s.name, ex: s.exemplars || 1 }))
    })));
    if (curKey !== lastRosterKey) {
        lastRosterKey = curKey;
        rebuildRoster(stats);
    }
    // Always update counts/hits in-place and active highlighting.
    updateRosterCounts(stats);
    updateRosterActive(stats);
    updateMergeButton();
}

function rebuildRoster(stats) {
    let html = '';
    for (const group of stats.speaker_lists) {
        const m = findModelByLabel(group.model);
        if (!m) continue;
        for (const spk of group.speakers) {
            const ex = spk.exemplars || 1;
            const canManage = !!m.deleteCmd;
            html += `<div class="roster-row" data-prefix="${m.prefix}" data-id="${spk.id}">`;
            if (canManage) html += `<input type="checkbox" class="roster-merge-cb" title="Select for merge">`;
            html += `<span class="roster-dot" style="background:${spkColor(spk.id)}"></span>` +
                `<span class="roster-model">${m.label}</span>` +
                `<span class="roster-id">#${spk.id}</span>` +
                `<input class="roster-name-input" type="text" value="${esc(spk.name)}" placeholder="unnamed">` +
                `<button class="btn btn--vad roster-set-btn">Set</button>` +
                `<span class="roster-exemplars" title="Exemplars stored">${ex}ex</span>` +
                `<span class="roster-hits" title="Exemplars above threshold"></span>` +
                `<span class="roster-count">\u00d7${spk.count}</span>`;
            if (canManage) html += `<button class="btn btn--danger roster-del-btn" title="Delete speaker">🗑️</button>`;
            html += `</div>`;
        }
    }
    if (!html) {
        html = '<div class="roster-empty">No speakers identified yet</div>';
    }
    rosterEl.innerHTML = html;
    // Bind buttons and events.
    rosterEl.querySelectorAll('.roster-row').forEach(row => {
        const prefix = row.dataset.prefix;
        const id = row.dataset.id;
        const input = row.querySelector('.roster-name-input');
        const setBtn = row.querySelector('.roster-set-btn');
        const delBtn = row.querySelector('.roster-del-btn');
        const cb = row.querySelector('.roster-merge-cb');
        const m = MODEL_BY_PREFIX[prefix];
        if (!m || !setBtn || !input) return;
        setBtn.addEventListener('click', () => {
            const name = input.value.trim();
            if (!name) return;
            ws.sendText(`${m.nameCmd}:${id}:${name}`);
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') setBtn.click();
        });
        if (delBtn && m.deleteCmd) {
            delBtn.addEventListener('click', () => {
                ws.sendText(`${m.deleteCmd}:${id}`);
            });
        }
        if (cb) {
            cb.addEventListener('change', updateMergeButton);
        }
    });
}

function updateRosterCounts(stats) {
    for (const group of stats.speaker_lists) {
        const m = findModelByLabel(group.model);
        if (!m) continue;
        for (const spk of group.speakers) {
            const row = rosterEl.querySelector(`[data-prefix="${m.prefix}"][data-id="${spk.id}"]`);
            if (!row) continue;
            const countEl = row.querySelector('.roster-count');
            if (countEl) countEl.textContent = `\u00d7${spk.count}`;
            const exEl = row.querySelector('.roster-exemplars');
            if (exEl) exEl.textContent = `${spk.exemplars || 1}ex`;
        }
    }
    // Update hits for WL-ECAPA active match (from top-level stats).
    if (stats.wlecapa_active && stats.wlecapa_id >= 0) {
        const row = rosterEl.querySelector(`[data-prefix="wlecapa"][data-id="${stats.wlecapa_id}"]`);
        if (row) {
            const hitsEl = row.querySelector('.roster-hits');
            if (hitsEl) {
                const hits = stats.wlecapa_hits_above || 0;
                const ex = stats.wlecapa_exemplars || 0;
                hitsEl.textContent = ex > 0 ? `${hits}/${ex}` : '';
            }
        }
    }
}

function updateRosterActive(stats) {
    MODELS.forEach(m => {
        const isActive = stats[`${m.prefix}_active`] === true;
        const activeId = stats[`${m.prefix}_id`];
        if (isActive) {
            // Clear previous timer for this model.
            if (activeTimers[m.prefix]) clearTimeout(activeTimers[m.prefix]);
            // Highlight the matching row, un-highlight others of same model.
            rosterEl.querySelectorAll(`[data-prefix="${m.prefix}"]`).forEach(row => {
                const id = parseInt(row.dataset.id, 10);
                const match = id === activeId;
                row.classList.toggle('roster-row--active', match);
                row.querySelector('.roster-dot')?.classList.toggle('roster-dot--active', match);
            });
            // Decay after 1.5s.
            activeTimers[m.prefix] = setTimeout(() => {
                rosterEl.querySelectorAll(`[data-prefix="${m.prefix}"]`).forEach(row => {
                    row.classList.remove('roster-row--active');
                    row.querySelector('.roster-dot')?.classList.remove('roster-dot--active');
                });
            }, 1500);
        }
    });
}

// --- Merge button management ---
let mergeBtn = null;
function updateMergeButton() {
    // Collect checked rows grouped by prefix.
    const checked = rosterEl.querySelectorAll('.roster-merge-cb:checked');
    if (checked.length < 2) {
        if (mergeBtn) { mergeBtn.remove(); mergeBtn = null; }
        return;
    }
    // All checked must share same prefix (can only merge within one model).
    const rows = Array.from(checked).map(cb => cb.closest('.roster-row'));
    const prefix = rows[0]?.dataset.prefix;
    const allSame = rows.every(r => r.dataset.prefix === prefix);
    const m = MODEL_BY_PREFIX[prefix];
    if (!allSame || !m || !m.mergeCmd) {
        if (mergeBtn) { mergeBtn.remove(); mergeBtn = null; }
        return;
    }
    if (!mergeBtn) {
        mergeBtn = document.createElement('button');
        mergeBtn.className = 'btn btn--merge roster-merge-btn';
        rosterEl.parentNode.insertBefore(mergeBtn, rosterEl);
    }
    const ids = rows.map(r => r.dataset.id);
    mergeBtn.textContent = `Merge ${ids.map(id => '#' + id).join(' + ')} → #${ids[0]}`;
    mergeBtn.onclick = () => {
        const dst = ids[0];
        for (let i = 1; i < ids.length; i++) {
            ws.sendText(`${m.mergeCmd}:${dst}:${ids[i]}`);
        }
        // Uncheck all.
        checked.forEach(cb => { cb.checked = false; });
        if (mergeBtn) { mergeBtn.remove(); mergeBtn = null; }
    };
}

function findModelByLabel(label) {
    // Map backend label → MODELS entry.
    const map = { 'CAM++': 'speaker', 'WavLM': 'wavlm', 'ECAPA-TDNN': 'unispeech', 'WL-ECAPA': 'wlecapa' };
    const prefix = map[label];
    return prefix ? MODEL_BY_PREFIX[prefix] : null;
}

function esc(s) {
    return (s || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// --- Connect ---
const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProto}//${location.host}/ws`;
ws.connect(wsUrl);
log(`Connecting to ${wsUrl} ...`);
