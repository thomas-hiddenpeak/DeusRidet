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
import { TimelinePanel } from './components/timeline-panel.js';
import { DiarizenPanel } from './components/diarizen-panel.js';
import { ViresPanel } from './components/vires-panel.js';
import { EvalRunPanel } from './components/eval-run-panel.js';
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

// --- Mission workflow (user-path driven interaction layer) ---
const missionState = {
    connected: false,
    micOn: false,
    speakerSeen: false,
    asrEnabled: false,
    asrCount: 0,
    finalized: false,
};

const missionEls = {
    steps: Array.from(document.querySelectorAll('.mission-step')),
    nextText: document.getElementById('mission-next-text'),
    actCapture: document.getElementById('mission-act-capture'),
    actAsr: document.getElementById('mission-act-asr'),
    actFinalize: document.getElementById('mission-act-finalize'),
    actReset: document.getElementById('mission-act-reset'),
};

function updateMissionFlow() {
    const activeStep = getActiveMissionStep();
    const recommendation = getMissionRecommendation(activeStep);

    missionEls.steps.forEach((btn) => {
        const step = Number(btn.dataset.step || 0);
        btn.classList.toggle('is-active', step === activeStep);
        btn.classList.toggle('is-done', step < activeStep);
    });
    if (missionEls.nextText) missionEls.nextText.textContent = recommendation;
}

function getActiveMissionStep() {
    if (!missionState.connected) return 1;
    if (!missionState.micOn) return 2;
    if (!missionState.speakerSeen) return 3;
    if (!missionState.asrEnabled || missionState.asrCount === 0) return 4;
    if (!missionState.finalized) return 5;
    return 5;
}

function getMissionRecommendation(step) {
    if (step === 1) return '等待 WS 连接稳定；连接后直接进入采集步骤。';
    if (step === 2) return '点击“切换麦克风”开始采集，先确认输入波形与 RMS 在变化。';
    if (step === 3) return '说一段完整句子触发首个说话人注册，再观察 Speaker 面板是否出现新身份。';
    if (step === 4) return '开启 ASR 并等待至少 1 条 transcript，确认识别链路通畅。';
    if (!missionState.finalized) return '执行 Finalize 进入复盘，检查 DiariZen 最终段和导入评估结果。';
    return '本轮已完成，可重置后开始下一轮采样。';
}

function bindMissionFlowActions() {
    missionEls.steps.forEach((btn) => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.target;
            const target = targetId ? document.getElementById(targetId) : null;
            target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    missionEls.actCapture?.addEventListener('click', () => {
        document.getElementById('mic-toggle')?.click();
    });

    missionEls.actAsr?.addEventListener('click', () => {
        ws.sendText(`asr_enable:${missionState.asrEnabled ? 'off' : 'on'}`);
    });

    missionEls.actFinalize?.addEventListener('click', () => {
        ws.sendText('diarizen_finalize');
        missionState.finalized = true;
        updateMissionFlow();
    });

    missionEls.actReset?.addEventListener('click', () => {
        document.getElementById('eval-reset-btn')?.click();
        missionState.finalized = false;
        missionState.speakerSeen = false;
        missionState.asrCount = 0;
        updateMissionFlow();
    });
}

function bindMissionRuntimeSync() {
    const micBtn = document.getElementById('mic-toggle');
    if (micBtn) {
        const observer = new MutationObserver(() => {
            missionState.micOn = micBtn.getAttribute('aria-pressed') === 'true';
            updateMissionFlow();
        });
        observer.observe(micBtn, { attributes: true, attributeFilter: ['aria-pressed'] });
    }
}

function initAnalysisMode() {
    const liveBtn = document.getElementById('analysis-tab-live');
    const reviewBtn = document.getElementById('analysis-tab-review');
    const panels = Array.from(document.querySelectorAll('[data-analysis-view]'));
    if (!liveBtn || !reviewBtn || !panels.length) return;

    const setMode = (mode) => {
        const isLive = mode === 'live';
        liveBtn.classList.toggle('is-active', isLive);
        reviewBtn.classList.toggle('is-active', !isLive);
        liveBtn.setAttribute('aria-selected', isLive ? 'true' : 'false');
        reviewBtn.setAttribute('aria-selected', isLive ? 'false' : 'true');

        panels.forEach((panel) => {
            const view = panel.getAttribute('data-analysis-view');
            panel.classList.toggle('analysis-hidden', view !== mode);
        });
    };

    window.__setAnalysisMode = setMode;

    liveBtn.addEventListener('click', () => setMode('live'));
    reviewBtn.addEventListener('click', () => setMode('review'));
    setMode('live');
}

function initStudioActions() {
    const modal = document.getElementById('assistant-studio-modal');
    const openStudio = document.getElementById('open-assistant-studio');
    const closeStudio = document.getElementById('close-assistant-studio');
    const openDiagLive = document.getElementById('open-diagnostics-live');
    const openDiagReview = document.getElementById('open-diagnostics-review');
    const analysisLane = document.querySelector('.lane--analysis');

    openStudio?.addEventListener('click', () => modal?.showModal());
    closeStudio?.addEventListener('click', () => modal?.close());

    openDiagLive?.addEventListener('click', () => {
        window.__setAnalysisMode?.('live');
        analysisLane?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    openDiagReview?.addEventListener('click', () => {
        window.__setAnalysisMode?.('review');
        analysisLane?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
}

// --- WebSocket connection ---
const ws = new WsClient();
const statusEl = document.getElementById('conn-status');

ws.onOpen = () => {
    missionState.connected = true;
    missionState.finalized = false;
    updateMissionFlow();
    statusEl.textContent = 'Connected';
    statusEl.classList.add('connected');
    audioPanel.enable();
    evalRunPanel.onConnection(true);
    log('WebSocket connected');
};

ws.onClose = () => {
    missionState.connected = false;
    missionState.micOn = false;
    updateMissionFlow();
    statusEl.textContent = 'Disconnected';
    statusEl.classList.remove('connected');
    audioPanel.disable();
    evalRunPanel.onConnection(false);
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
            timelinePanel.onPipelineStats(obj);
            evalRunPanel.onPipelineStats(obj);
            return;
        }
        if (obj.type === 'vad') {
            audioPanel.updateVad(obj);
            if (obj.event === 'start') log('VAD: speech started');
            if (obj.event === 'end') log('VAD: speech ended');
            return;
        }
        if (obj.type === 'speaker') {
            missionState.speakerSeen = true;
            missionState.finalized = false;
            updateMissionFlow();
            log(`Speaker: id=${obj.id} sim=${obj.sim.toFixed(3)} ${obj.new ? 'NEW' : ''} ${obj.name || '(unnamed)'}`);
            speakerDebug.onSpeakerEvent(obj);
            timelinePanel.onSpeakerEvent(obj);
            setRosterActive('speaker', obj.id, 1800);
            return;
        }
        if (obj.type === 'speaker_debug') {
            speakerDebug.onDebugData(obj);
            return;
        }
        if (obj.type === 'speaker_relabel') {
            // OratorReclusterer global-merge / K-cap event. Patch the
            // timeline forward-map so already-rendered speaker chips fold
            // into the surviving identity.
            timelinePanel.onSpeakerRelabel(obj);
            log(`Speaker relabel: seg=${obj.segment_id} ${obj.old_id} → ${obj.new_id} (conf=${(obj.confidence ?? 0).toFixed(3)})`);
            return;
        }
        if (obj.type === 'speaker_diarize_progress') {
            diarizenPanel.onProgress(obj);
            evalRunPanel.onDiarizeProgress(obj);
            const progressState = obj.status
                || (obj.error ? `error: ${obj.error}` : (obj.ok === false ? 'failed' : 'progress'));
            const sampleText = obj.samples !== undefined ? ` samples=${obj.samples}` : '';
            log(`DiariZen ${progressState}${sampleText}`);
            return;
        }
        if (obj.type === 'speaker_diarize_status') {
            diarizenPanel.onStatus(obj);
            if (obj.ok === false) {
                log(`DiariZen status error: ${obj.error || 'unknown'}`);
            }
            return;
        }
        if (obj.type === 'speaker_diarize_partial') {
            diarizenPanel.onPartial(obj);
            evalRunPanel.onDiarizePartial(obj);
            timelinePanel.onDiarize(obj);
            log(`DiariZen partial pass=${obj.pass} segs=${obj.segment_count} changed=${obj.changed_pending ?? '?'}`);
            return;
        }
        if (obj.type === 'speaker_diarize_final') {
            missionState.finalized = true;
            updateMissionFlow();
            diarizenPanel.onFinal(obj);
            evalRunPanel.onDiarizeFinal(obj);
            timelinePanel.onDiarize(obj);
            log(`DiariZen FINAL pass=${obj.pass} segs=${obj.segment_count}`);
            return;
        }
        if (obj.type === 'vires_compute_snapshot') {
            viresPanel.onSnapshot(obj);
            return;
        }
        if (obj.type === 'asr_transcript') {
            missionState.asrCount += 1;
            missionState.finalized = false;
            updateMissionFlow();
            asrPanel.onTranscript(obj);
            asrTranscriptPanel.onTranscript(obj);
            asrLogPanel.onTranscript(obj);
            textOutputPanel.onAsrTranscript(obj);
            timelinePanel.onTranscript(obj);
            evalRunPanel.onAsrTranscript(obj);
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
            missionState.asrEnabled = !!obj.enabled;
            missionState.finalized = false;
            updateMissionFlow();
            asrPanel.onAsrEnable(obj);
            evalRunPanel.onAsrEnable(obj);
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
            const map = {0:'silero', 2:'any', 3:'direct'};
            log(`ASR VAD source → ${map[obj.value] || obj.value}`);
            return;
        }
        if (obj.type === 'loopback') {
            log(`Loopback ${obj.enabled ? 'ON' : 'OFF'}`);
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
const timelinePanel = new TimelinePanel();
const diarizenPanel = new DiarizenPanel(ws);
const viresPanel = new ViresPanel();
const evalRunPanel = new EvalRunPanel(ws);

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
    clearCmd: 'speaker_clear',  nameCmd: 'speaker_name',  btnId: 'spk-en-campp',  label: 'CAM++' },
];

const ROSTER_MODELS = {
    'CAM++': {
        prefix: 'speaker',
        label: 'CAM++',
        nameCmd: 'speaker_name',
        editable: true,
    },
};

// Clear All.
document.getElementById('spk-clear-all')?.addEventListener('click', () => {
    ws.sendText('speaker_clear');
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

// Ambiguity guard (margin_abstain) — min (top1-top2) margin to trust a match.
{
    const slider = document.getElementById('speaker-margin');
    const valEl = document.getElementById('speaker-margin-val');
    if (slider && valEl) {
        slider.addEventListener('input', () => {
            const v = parseFloat(slider.value);
            valEl.textContent = v.toFixed(2);
            ws.sendText(`speaker_margin_abstain:${v.toFixed(2)}`);
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
    // Sync threshold controls from server.
    MODELS.forEach(m => {
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
    // Sync VAD source.
    if (stats.vad_source !== undefined && vadSourceSelect && !vadSourceSelect.matches(':focus')) {
        const map = {0:'silero', 2:'any'};
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
    // Always update counts in-place and active highlighting.
    updateRosterCounts(stats);
    updateRosterActive(stats);
}

function rebuildRoster(stats) {
    let html = '';
    for (const group of stats.speaker_lists) {
        const m = findRosterModel(group.model);
        if (!m) continue;
        for (const spk of group.speakers) {
            const ex = spk.exemplars || 1;
            html += `<div class="roster-row" data-prefix="${m.prefix}" data-id="${spk.id}">`;
            html += `<span class="roster-dot" style="background:${spkColor(spk.id)}"></span>`;
            html += `<span class="roster-model">${m.label}</span>`;
            html += `<span class="roster-id">#${spk.id}</span>`;
            if (m.editable) {
                html += `<input class="roster-name-input" type="text" value="${esc(spk.name)}" placeholder="unnamed">`;
                html += `<button class="btn btn--vad roster-set-btn">Set</button>`;
            } else {
                html += `<span class="roster-name-static">${esc(spk.name || 'unnamed')}</span>`;
            }
            html += `<span class="roster-exemplars" title="Exemplars stored">${ex}ex</span>`;
            html += `<span class="roster-count">\u00d7${spk.count}</span>`;
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
        const m = findRosterModelByPrefix(prefix);
        if (!m) return;
        if (setBtn && input && m.nameCmd) {
            setBtn.addEventListener('click', () => {
                const name = input.value.trim();
                if (!name) return;
                ws.sendText(`${m.nameCmd}:${id}:${name}`);
            });
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') setBtn.click();
            });
        }
    });
}

function updateRosterCounts(stats) {
    for (const group of stats.speaker_lists) {
        const m = findRosterModel(group.model);
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
}

function updateRosterActive(stats) {
    MODELS.forEach(m => {
        const isActive = stats[`${m.prefix}_active`] === true;
        const activeId = stats[`${m.prefix}_id`];
        if (isActive) {
            setRosterActive(m.prefix, activeId, 1500);
        }
    });
}

function setRosterActive(prefix, activeId, ttlMs) {
    if (!rosterEl) return;
    if (!Number.isFinite(activeId) || activeId < 0) return;
    if (activeTimers[prefix]) clearTimeout(activeTimers[prefix]);
    rosterEl.querySelectorAll(`[data-prefix="${prefix}"]`).forEach(row => {
        const id = parseInt(row.dataset.id, 10);
        const match = id === activeId;
        row.classList.toggle('roster-row--active', match);
        row.querySelector('.roster-dot')?.classList.toggle('roster-dot--active', match);
    });
    activeTimers[prefix] = setTimeout(() => {
        rosterEl.querySelectorAll(`[data-prefix="${prefix}"]`).forEach(row => {
            row.classList.remove('roster-row--active');
            row.querySelector('.roster-dot')?.classList.remove('roster-dot--active');
        });
    }, ttlMs);
}

function findRosterModel(label) {
    return ROSTER_MODELS[label] || null;
}

function findRosterModelByPrefix(prefix) {
    return Object.values(ROSTER_MODELS).find((m) => m.prefix === prefix) || null;
}

function esc(s) {
    return (s || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// --- Connect ---
const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProto}//${location.host}/ws`;
bindMissionFlowActions();
bindMissionRuntimeSync();
initAnalysisMode();
initStudioActions();
updateMissionFlow();
ws.connect(wsUrl);
log(`Connecting to ${wsUrl} ...`);
