// audio-panel.js — Audio capture component.
// Manages browser mic, AudioWorklet, PCM streaming, server stats, and loopback.

export class AudioPanel {
    constructor(wsClient) {
        this.ws = wsClient;
        this.audioCtx = null;
        this.workletNode = null;
        this.stream = null;
        this.capturing = false;
        this.framesSent = 0;
        this.bytesSent = 0;
        this.loopbackOn = false;

        // Loopback playback context (separate from capture).
        this.playCtx = null;
        this.playNextTime = 0;

        this.btnMic      = document.getElementById('mic-toggle');
        this.btnLoopback = document.getElementById('loopback-toggle');
        this.statSent    = document.getElementById('stat-sent');
        this.statBytes   = document.getElementById('stat-bytes');
        this.canvas      = document.getElementById('audio-viz');
        this.canvasCtx   = this.canvas.getContext('2d');

        // Server stats elements.
        this.srvFrames  = document.getElementById('srv-frames');
        this.srvRms     = document.getElementById('srv-rms');
        this.srvPeak    = document.getElementById('srv-peak');
        this.srvRmsBar  = document.getElementById('srv-rms-bar');
        this.srvPeakBar = document.getElementById('srv-peak-bar');

        // Pipeline stats elements.
        this.pipeMel     = document.getElementById('pipe-mel');
        this.pipeSpeech  = document.getElementById('pipe-speech');
        this.pipeEnergy  = document.getElementById('pipe-energy');
        this.pipeNoise   = document.getElementById('pipe-noise');
        this.vadDot      = document.getElementById('vad-dot');
        this.vadLabel    = document.getElementById('vad-label');

        // VAD threshold slider.
        this.vadSlider   = document.getElementById('vad-threshold');
        this.vadSliderVal = document.getElementById('vad-threshold-val');

        // Input gain slider.
        this.gainSlider   = document.getElementById('input-gain');
        this.gainSliderVal = document.getElementById('input-gain-val');

        // Silero VAD elements.
        this.sileroProb      = document.getElementById('silero-prob');
        this.sileroProbBar   = document.getElementById('silero-prob-bar');
        this.sileroSlider    = document.getElementById('silero-threshold');
        this.sileroSliderVal = document.getElementById('silero-threshold-val');

        // FSMN VAD elements.
        this.fsmnProb      = document.getElementById('fsmn-prob');
        this.fsmnProbBar   = document.getElementById('fsmn-prob-bar');
        this.fsmnSlider    = document.getElementById('fsmn-threshold');
        this.fsmnSliderVal = document.getElementById('fsmn-threshold-val');

        // TEN VAD elements.
        this.tenProb      = document.getElementById('ten-prob');
        this.tenProbBar   = document.getElementById('ten-prob-bar');
        this.tenSlider    = document.getElementById('ten-threshold');
        this.tenSliderVal = document.getElementById('ten-threshold-val');

        // Per-VAD toggle buttons.
        this.sileroToggle = document.getElementById('silero-toggle');
        this.fsmnToggle   = document.getElementById('fsmn-toggle');
        this.tenToggle    = document.getElementById('ten-toggle');

        // Log output.
        this.logOutput = document.getElementById('log-output');

        this.btnMic.addEventListener('click', () => this.toggle());
        this.btnLoopback.addEventListener('click', () => this.toggleLoopback());
        this.vadSlider.addEventListener('input', () => this._onThresholdChange());
        this.gainSlider.addEventListener('input', () => this._onGainChange());
        this.sileroSlider.addEventListener('input', () => this._onSileroThresholdChange());
        this.fsmnSlider.addEventListener('input', () => this._onFsmnThresholdChange());
        this.tenSlider.addEventListener('input', () => this._onTenThresholdChange());
        this.sileroToggle.addEventListener('click', () => this._toggleVad('silero'));
        this.fsmnToggle.addEventListener('click', () => this._toggleVad('fsmn'));
        this.tenToggle.addEventListener('click', () => this._toggleVad('ten'));
    }

    enable() {
        this.btnMic.disabled = false;
        this.btnLoopback.disabled = false;
    }

    disable() {
        this.btnMic.disabled = true;
        this.btnLoopback.disabled = true;
        this.stopCapture();
    }

    async toggle() {
        if (this.capturing) {
            this.stopCapture();
        } else {
            await this.startCapture();
        }
    }

    toggleLoopback() {
        this.loopbackOn = !this.loopbackOn;
        this.btnLoopback.textContent = this.loopbackOn ? 'Loopback On' : 'Loopback Off';
        this.btnLoopback.classList.toggle('btn--active', this.loopbackOn);
        this.ws.sendText(this.loopbackOn ? 'loopback:on' : 'loopback:off');
        if (!this.loopbackOn && this.playCtx) {
            this.playCtx.close();
            this.playCtx = null;
        }
    }

    updateServerStats(stats) {
        this.srvFrames.textContent = stats.frames;
        this.srvRms.textContent    = stats.rms.toFixed(4);
        this.srvPeak.textContent   = stats.peak.toFixed(4);
        const rmsW = Math.min(stats.rms / 0.15, 1.0) * 100;
        const peakW = Math.min(stats.peak, 1.0) * 100;
        this.srvRmsBar.style.width  = rmsW + '%';
        this.srvPeakBar.style.width = peakW + '%';
        this.srvPeakBar.classList.toggle('level-bar--warn', stats.peak >= 0.5 && stats.peak < 0.8);
        this.srvPeakBar.classList.toggle('level-bar--clip', stats.peak >= 0.8);
    }

    updatePipelineStats(stats) {
        this.pipeMel.textContent    = stats.mel_frames;
        this.pipeSpeech.textContent = stats.speech_frames;
        this.pipeEnergy.textContent = stats.energy.toFixed(2);
        if (stats.noise_floor !== undefined) {
            this.pipeNoise.textContent = stats.noise_floor.toFixed(2);
        }
        // Silero probability display.
        if (stats.silero_prob !== undefined) {
            this.sileroProb.textContent = stats.silero_prob.toFixed(3);
            this.sileroProbBar.style.width = (stats.silero_prob * 100) + '%';
            this.sileroProbBar.classList.toggle('level-bar--speech',
                stats.silero_speech);
        }
        // FSMN probability display.
        if (stats.fsmn_prob !== undefined) {
            this.fsmnProb.textContent = stats.fsmn_prob.toFixed(3);
            this.fsmnProbBar.style.width = (stats.fsmn_prob * 100) + '%';
            this.fsmnProbBar.classList.toggle('level-bar--speech',
                stats.fsmn_speech);
        }
        // TEN probability display.
        if (stats.ten_prob !== undefined) {
            this.tenProb.textContent = stats.ten_prob.toFixed(3);
            this.tenProbBar.style.width = (stats.ten_prob * 100) + '%';
            this.tenProbBar.classList.toggle('level-bar--speech',
                stats.ten_speech);
        }

        // Sync enable toggle state from server.
        if (stats.silero_enabled !== undefined) {
            this.sileroToggle.classList.toggle('btn--active', stats.silero_enabled);
            this.sileroToggle.setAttribute('aria-pressed', String(stats.silero_enabled));
        }
        if (stats.fsmn_enabled !== undefined) {
            this.fsmnToggle.classList.toggle('btn--active', stats.fsmn_enabled);
            this.fsmnToggle.setAttribute('aria-pressed', String(stats.fsmn_enabled));
        }
        if (stats.ten_enabled !== undefined) {
            this.tenToggle.classList.toggle('btn--active', stats.ten_enabled);
            this.tenToggle.setAttribute('aria-pressed', String(stats.ten_enabled));
        }

        // Log VAD results (throttled — every ~500ms).
        if (!this._lastLogTime || Date.now() - this._lastLogTime > 500) {
            this._lastLogTime = Date.now();
            const parts = [];
            if (stats.silero_enabled) parts.push(`Silero=${stats.silero_prob.toFixed(3)}${stats.silero_speech ? '▲' : '▽'}`);
            if (stats.fsmn_enabled) parts.push(`FSMN=${stats.fsmn_prob.toFixed(3)}${stats.fsmn_speech ? '▲' : '▽'}`);
            if (stats.ten_enabled) parts.push(`TEN=${stats.ten_prob.toFixed(3)}${stats.ten_speech ? '▲' : '▽'}`);
            if (parts.length > 0) {
                this._appendLog(parts.join('  ') + `  rms=${stats.rms.toFixed(4)}`);
            }
        }
        // Update VAD indicator from pipeline stats too.
        this.vadDot.classList.toggle('speech', stats.is_speech);
        this.vadLabel.textContent = stats.is_speech ? 'Speech' : 'Silence';
    }

    updateVad(vad) {
        this.vadDot.classList.toggle('speech', vad.speech);
        this.vadLabel.textContent = vad.speech ? 'Speech' : 'Silence';
    }

    playLoopback(arrayBuffer) {
        if (!this.loopbackOn) return;
        if (!this.playCtx) {
            this.playCtx = new AudioContext({ sampleRate: 16000 });
            this.playNextTime = this.playCtx.currentTime;
        }
        // Convert int16 PCM to float32 for Web Audio.
        const int16 = new Int16Array(arrayBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }
        const buf = this.playCtx.createBuffer(1, float32.length, 16000);
        buf.getChannelData(0).set(float32);
        const src = this.playCtx.createBufferSource();
        src.buffer = buf;
        src.connect(this.playCtx.destination);
        // Schedule seamless playback.
        const now = this.playCtx.currentTime;
        if (this.playNextTime < now) this.playNextTime = now;
        src.start(this.playNextTime);
        this.playNextTime += buf.duration;
    }

    async startCapture() {
        try {
            this.audioCtx = new AudioContext({ sampleRate: 16000 });
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    // Disable browser audio processing to preserve speaker-specific
                    // waveform characteristics needed by WavLM/UniSpeech encoders.
                    // AGC normalizes amplitude, noise suppression strips spectral
                    // detail, echo cancellation modifies the signal — all degrade
                    // speaker verification accuracy.
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });

            await this.audioCtx.audioWorklet.addModule('js/utils/audio-processor.js');
            this.workletNode = new AudioWorkletNode(this.audioCtx,
                                                     'audio-capture-processor');

            this.workletNode.port.onmessage = (ev) => {
                if (!this.ws.connected) return;
                const pcm = ev.data.pcm;
                this.ws.sendBinary(pcm.buffer);
                this.framesSent++;
                this.bytesSent += pcm.byteLength;
                this.statSent.textContent  = this.framesSent;
                this.statBytes.textContent = this._fmtBytes(this.bytesSent);
            };

            const source = this.audioCtx.createMediaStreamSource(this.stream);

            this.analyser = this.audioCtx.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
            this.analyser.connect(this.workletNode);
            // Don't connect worklet to destination to avoid feedback.

            this.capturing = true;
            this.btnMic.textContent = 'Disable Mic';
            this.btnMic.setAttribute('aria-pressed', 'true');
            this._drawViz();
        } catch (err) {
            console.error('Mic start failed:', err);
        }
    }

    stopCapture() {
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        if (this.audioCtx) {
            this.audioCtx.close();
            this.audioCtx = null;
        }
        this.capturing = false;
        this.btnMic.textContent = 'Enable Mic';
        this.btnMic.setAttribute('aria-pressed', 'false');
        this._clearViz();
    }

    _onThresholdChange() {
        const val = parseFloat(this.vadSlider.value);
        this.vadSliderVal.textContent = val.toFixed(1);
        this.ws.sendText('vad_threshold:' + val.toFixed(1));
    }

    _onGainChange() {
        const val = parseFloat(this.gainSlider.value);
        this.gainSliderVal.textContent = val.toFixed(1);
        this.ws.sendText('gain:' + val.toFixed(1));
    }

    _onSileroThresholdChange() {
        const val = parseFloat(this.sileroSlider.value);
        this.sileroSliderVal.textContent = val.toFixed(2);
        this.ws.sendText('silero_threshold:' + val.toFixed(2));
    }

    _onFsmnThresholdChange() {
        const val = parseFloat(this.fsmnSlider.value);
        this.fsmnSliderVal.textContent = val.toFixed(2);
        this.ws.sendText('fsmn_threshold:' + val.toFixed(2));
    }

    _onTenThresholdChange() {
        const val = parseFloat(this.tenSlider.value);
        this.tenSliderVal.textContent = val.toFixed(2);
        this.ws.sendText('ten_threshold:' + val.toFixed(2));
    }

    _toggleVad(name) {
        const btnMap = { silero: this.sileroToggle, fsmn: this.fsmnToggle, ten: this.tenToggle };
        const btn = btnMap[name];
        const isActive = btn.classList.contains('btn--active');
        const newState = !isActive;
        btn.classList.toggle('btn--active', newState);
        btn.setAttribute('aria-pressed', String(newState));
        this.ws.sendText(name + '_enable:' + (newState ? 'on' : 'off'));
    }

    _appendLog(msg) {
        if (!this.logOutput) return;
        const ts = new Date().toLocaleTimeString('en', { hour12: false, fractionalSecondDigits: 1 });
        this.logOutput.textContent += `[${ts}] ${msg}\n`;
        // Auto-scroll and limit lines.
        const lines = this.logOutput.textContent.split('\n');
        if (lines.length > 200) {
            this.logOutput.textContent = lines.slice(-150).join('\n');
        }
        this.logOutput.scrollTop = this.logOutput.scrollHeight;
    }

    _drawViz() {
        if (!this.capturing || !this.analyser) return;
        requestAnimationFrame(() => this._drawViz());

        const bufLen = this.analyser.frequencyBinCount;
        const data = new Uint8Array(bufLen);
        this.analyser.getByteFrequencyData(data);

        const ctx = this.canvasCtx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.fillStyle = getComputedStyle(document.documentElement)
                        .getPropertyValue('--clr-bg').trim();
        ctx.fillRect(0, 0, w, h);

        const barW = w / bufLen;
        ctx.fillStyle = getComputedStyle(document.documentElement)
                        .getPropertyValue('--clr-accent').trim();
        for (let i = 0; i < bufLen; i++) {
            const barH = (data[i] / 255) * h;
            ctx.fillRect(i * barW, h - barH, barW - 1, barH);
        }
    }

    _clearViz() {
        const ctx = this.canvasCtx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    _fmtBytes(n) {
        if (n < 1024) return n + ' B';
        if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
        return (n / 1048576).toFixed(1) + ' MB';
    }
}
