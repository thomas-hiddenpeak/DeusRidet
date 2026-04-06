// app.js — DeusRidet WebUI bootstrap.
// Connects to the backend WebSocket server, initialises components,
// and provides a lightweight log display.

import { WsClient } from './ws-client.js';
import { AudioPanel } from './components/audio-panel.js';

// --- Log utility ---
const logEl = document.getElementById('log-output');
function log(msg) {
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    logEl.textContent += `[${ts}] ${msg}\n`;
    // Keep log to last 200 lines.
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
            return;
        }
        if (obj.type === 'vad') {
            audioPanel.updateVad(obj);
            if (obj.event === 'start') log('VAD: speech started');
            if (obj.event === 'end') log('VAD: speech ended');
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
    // Loopback audio from server — play it.
    audioPanel.playLoopback(buf);
};

// --- Components ---
const audioPanel = new AudioPanel(ws);

// --- Connect ---
const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProto}//${location.host}/ws`;
ws.connect(wsUrl);
log(`Connecting to ${wsUrl} ...`);
