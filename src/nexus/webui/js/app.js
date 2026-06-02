// app.js — user interface bootstrap.
// Owns the single WsClient, the header (connection pill + language toggle),
// mic capture, and fan-out of every WS message to the mounted components.
// Components never touch the socket directly — they receive parsed messages
// via onMessage and push user intent back through callbacks.

import { WsClient } from './ws-client.js';
import { i18n } from './i18n.js';
import { Presence } from './components/presence.js';
import { Listening } from './components/listening.js';
import { Speakers } from './components/speakers.js';
import { Conversation } from './components/conversation.js';
import { Composer } from './components/composer.js';

const ws = new WsClient();
const components = [];
let online = false;
let micWanted = false;

function syncMicDrivenRuntime() {
    if (!ws.connected) return;
    ws.sendText(micWanted ? 'asr_enable:on' : 'asr_enable:off');
}

// ── Header: connection pill + language toggle ──────────────────────────
const connEl = document.querySelector('[data-role=conn]');
const langBtn = document.querySelector('[data-role=lang]');
const titleEl = document.querySelector('[data-role=title]');

function renderHeader() {
    titleEl.textContent = i18n.t('app.title');
    langBtn.textContent = i18n.t('lang.toggle');
    connEl.className = 'conn ' + (online ? 'conn--on' : 'conn--off');
    connEl.querySelector('[data-role=conn-label]').textContent =
        i18n.t(online ? 'conn.online' : 'conn.offline');
}

langBtn.addEventListener('click', () => { i18n.toggle(); renderAll(); });
i18n.onChange(() => renderHeader());

function renderAll() {
    renderHeader();
    for (const c of components) c.render?.();
}

// ── Mount components ───────────────────────────────────────────────────
const presence = new Presence();
const listening = new Listening();
const speakers = new Speakers();
const conversation = new Conversation();
const composer = new Composer();

presence.mount(document.querySelector('[data-slot=presence]'));
listening.mount(document.querySelector('[data-slot=strips]'));
speakers.mount(document.querySelector('[data-slot=strips]'));
conversation.mount(document.querySelector('[data-slot=stream]'));
composer.mount(document.querySelector('[data-slot=composer]'));
components.push(presence, listening, speakers, conversation, composer);

// ── User intent → upstream ─────────────────────────────────────────────
composer.onSend = (text) => ws.sendText('text_input:' + text);
listening.onMic = (on) => {
    micWanted = on;
    syncMicDrivenRuntime();
    if (on) mic.start();
    else mic.stop();
};

// ── Mic capture (16 kHz mono int16 PCM, raw waveform preserved) ─────────
const mic = {
    ctx: null, stream: null, node: null,
    async start() {
        if (this.node) return;
        try {
            this.ctx = new AudioContext({ sampleRate: 16000 });
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1, sampleRate: 16000,
                    echoCancellation: false, noiseSuppression: false,
                    autoGainControl: false,
                },
            });
            await this.ctx.audioWorklet.addModule('js/utils/audio-processor.js');
            this.node = new AudioWorkletNode(this.ctx, 'audio-capture-processor');
            this.node.port.onmessage = (ev) => {
                if (ws.connected) ws.sendBinary(ev.data.pcm.buffer);
            };
            this.ctx.createMediaStreamSource(this.stream).connect(this.node);
        } catch (e) {
            console.error('[mic] capture failed', e);
        }
    },
    stop() {
        this.node?.disconnect();
        this.stream?.getTracks().forEach((t) => t.stop());
        this.ctx?.close();
        this.ctx = this.stream = this.node = null;
    },
};

// ── WS lifecycle + message dispatch ────────────────────────────────────
ws.onOpen = () => {
    online = true;
    presence.setOnline(true);
    composer.setEnabled(true);
    syncMicDrivenRuntime();
    renderHeader();
};
ws.onClose = () => {
    online = false;
    presence.setOnline(false);
    composer.setEnabled(false);
    renderHeader();
};
ws.onText = (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    if (!msg || typeof msg.type !== 'string') return;
    for (const c of components) c.onMessage?.(msg);
};

composer.setEnabled(false);
renderAll();

// Connect to the same host that served this page.
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
ws.connect(`${proto}://${location.host}/ws`);
