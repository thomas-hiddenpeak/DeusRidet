// eval-run-panel.js — Evaluation-first run monitor for live diarization.
//
// This panel is designed for test/verification workflows: it tracks
// pipeline progress, diarization partial/final messages, and supports
// importing score/segment artifacts from runs/* for post-run review.

function fmtSec(v) {
    if (!Number.isFinite(v)) return '-';
    return `${v.toFixed(1)}s`;
}

function fmtPct(v) {
    if (!Number.isFinite(v)) return '-';
    return `${(v * 100).toFixed(1)}%`;
}

function parseSegments(obj) {
    const segs = Array.isArray(obj?.segments) ? obj.segments : [];
    return segs.map((s) => {
        if (Array.isArray(s)) {
            return {
                start: Number(s[0] || 0),
                end: Number(s[1] || 0),
                label: String(s[2] || 'unknown'),
            };
        }
        return {
            start: Number(s.start || 0),
            end: Number(s.end || 0),
            label: String(s.label || 'unknown'),
        };
    }).filter((s) => s.end > s.start);
}

function topLabelDurations(segments, topN = 6) {
    const map = new Map();
    for (const s of segments) {
        map.set(s.label, (map.get(s.label) || 0) + (s.end - s.start));
    }
    return Array.from(map.entries())
        .map(([label, dur]) => ({ label, dur }))
        .sort((a, b) => b.dur - a.dur)
        .slice(0, topN);
}

function parseJsonl(text) {
    const out = [];
    const lines = String(text || '').split('\n');
    for (const line of lines) {
        const t = line.trim();
        if (!t) continue;
        try {
            out.push(JSON.parse(t));
        } catch (_) {
            // Ignore malformed lines to keep the UI resilient.
        }
    }
    return out;
}

function summarizePairings(rows) {
    const s = {
        total: rows.length,
        decided: 0,
        noSegment: 0,
        bySpeaker: new Map(),
    };
    for (const r of rows) {
        const spk = String(r.gt_speaker || 'unknown');
        if (!s.bySpeaker.has(spk)) s.bySpeaker.set(spk, { total: 0, noSegment: 0, decided: 0 });
        const item = s.bySpeaker.get(spk);
        item.total += 1;
        const st = String(r.status || '');
        if (st === 'no_segment') {
            s.noSegment += 1;
            item.noSegment += 1;
        } else {
            s.decided += 1;
            item.decided += 1;
        }
    }
    return s;
}

function summarizeRawEvents(rows) {
    const counts = new Map();
    for (const r of rows) {
        const t = String(r.type || 'unknown');
        counts.set(t, (counts.get(t) || 0) + 1);
    }
    return Array.from(counts.entries())
        .map(([type, n]) => ({ type, n }))
        .sort((a, b) => b.n - a.n);
}

export class EvalRunPanel {
    constructor(ws) {
        this._ws = ws;
        this._root = document.getElementById('eval-run-panel');
        if (!this._root) return;

        this._connected = false;
        this._status = 'idle';
        this._streamSec = 0;
        this._streamInSec = 0;
        this._speechFrames = 0;
        this._asrCount = 0;
        this._lastPass = 0;
        this._segmentCount = 0;
        this._changedPending = 0;
        this._finalObj = null;
        this._importedScore = null;
        this._importedSegments = null;
        this._importedPairings = null;
        this._importedRawEvents = null;
        this._baselinePct = 93.5;
        this._asrEnabled = false;

        // Browser audio capture state (evaluation mode).
        this._audioCtx = null;
        this._workletNode = null;
        this._stream = null;
        this._analyser = null;
        this._capturing = false;
        this._micFrames = 0;
        this._micBytes = 0;
        this._micRaf = 0;

        this._renderShell();
        this._bind();
        this._renderStats();
    }

    _renderShell() {
        this._root.innerHTML = `
            <div class="eval-shell">
                <div class="eval-shell__hero">
                    <div class="eval-shell__heading">
                        <span class="eval-shell__eyebrow">Live evaluation stage</span>
                        <h2 class="panel__title">Evaluation Run</h2>
                        <p class="eval-shell__lede">Streaming state, capture, and artifact review live on one surface.</p>
                    </div>
                    <div class="eval-shell__actions">
                        <button id="eval-asr-toggle" class="btn btn--vad" aria-pressed="false">ASR Off</button>
                        <button id="eval-finalize-btn" class="btn btn--vad">Finalize</button>
                        <button id="eval-reset-btn" class="btn btn--vad">Reset</button>
                    </div>
                </div>

                <div class="eval-shell__grid">
                    <section class="eval-shell__main">
                        <div class="eval-run-status-row">
                            <span class="eval-chip" id="eval-chip-conn">WS: offline</span>
                            <span class="eval-chip" id="eval-chip-state">Run: idle</span>
                        </div>

                        <div class="eval-grid eval-grid--kpi">
                            <div class="eval-kpi"><span>audio_t1</span><strong id="eval-kpi-stream">-</strong></div>
                            <div class="eval-kpi"><span>audio_t1_in</span><strong id="eval-kpi-stream-in">-</strong></div>
                            <div class="eval-kpi"><span>Speech frames</span><strong id="eval-kpi-speech">0</strong></div>
                            <div class="eval-kpi"><span>ASR events</span><strong id="eval-kpi-asr">0</strong></div>
                            <div class="eval-kpi"><span>Diarize pass</span><strong id="eval-kpi-pass">0</strong></div>
                            <div class="eval-kpi"><span>Segments</span><strong id="eval-kpi-segs">0</strong></div>
                        </div>

                        <div class="eval-summary" id="eval-summary">
                            <div class="eval-summary-title">Final Summary</div>
                            <div class="eval-summary-body" id="eval-summary-body">No final result yet.</div>
                        </div>
                    </section>

                    <aside class="eval-shell__rail">
                        <div class="eval-audio-box">
                            <div class="eval-summary-title">Browser Audio Input</div>
                            <div class="eval-run-status-row">
                                <button id="eval-mic-toggle" class="btn btn--vad" aria-pressed="false" disabled>
                                    Enable Mic
                                </button>
                                <span class="eval-chip" id="eval-chip-mic">Mic: off</span>
                                <span class="eval-chip" id="eval-chip-mic-frames">frames: 0</span>
                                <span class="eval-chip" id="eval-chip-mic-bytes">bytes: 0 B</span>
                            </div>
                            <canvas id="eval-audio-viz" class="eval-audio-viz" width="640" height="56"
                                    aria-label="Browser mic level"></canvas>
                        </div>
                        <div class="eval-rail-actions">
                            <button id="eval-open-import" class="btn btn--vad" type="button">Open Artifact Import</button>
                            <div class="eval-import-inline" id="eval-import-inline">Artifacts not loaded yet.</div>
                        </div>
                    </aside>
                </div>

                <dialog id="eval-import-modal" class="app-modal app-modal--import" aria-label="Evaluation Artifact Import">
                    <div class="app-modal__card">
                        <div class="app-modal__head">
                            <h3 class="app-modal__title">Artifact Import (runs/*)</h3>
                            <button id="eval-close-import" class="btn btn--vad" type="button">Close</button>
                        </div>
                        <p class="app-modal__desc">低频操作放入模态：只在复盘时导入 score/segments/pairings/raw events。</p>
                        <div class="eval-import-row">
                            <label class="eval-import-label">score.json
                                <input id="eval-score-file" type="file" accept="application/json,.json">
                            </label>
                            <label class="eval-import-label">diarize_segments.json
                                <input id="eval-segs-file" type="file" accept="application/json,.json">
                            </label>
                            <label class="eval-import-label">pairings.jsonl
                                <input id="eval-pairings-file" type="file" accept="application/json,.jsonl,.txt">
                            </label>
                            <label class="eval-import-label">raw_events.jsonl
                                <input id="eval-raw-events-file" type="file" accept="application/json,.jsonl,.txt">
                            </label>
                            <label class="eval-import-label">Baseline (%)
                                <input id="eval-baseline" type="number" step="0.1" min="0" max="100" value="93.5">
                            </label>
                        </div>
                        <pre id="eval-import-out" class="log-output eval-import-out">Import artifacts to compare live result with scored result.</pre>
                    </div>
                </dialog>
            </div>
        `;

        this._connEl = document.getElementById('eval-chip-conn');
        this._stateEl = document.getElementById('eval-chip-state');
        this._kpiStream = document.getElementById('eval-kpi-stream');
        this._kpiStreamIn = document.getElementById('eval-kpi-stream-in');
        this._kpiSpeech = document.getElementById('eval-kpi-speech');
        this._kpiAsr = document.getElementById('eval-kpi-asr');
        this._kpiPass = document.getElementById('eval-kpi-pass');
        this._kpiSegs = document.getElementById('eval-kpi-segs');
        this._summaryBody = document.getElementById('eval-summary-body');
        this._importOut = document.getElementById('eval-import-out');
        this._importInline = document.getElementById('eval-import-inline');
        this._asrToggleBtn = document.getElementById('eval-asr-toggle');
        this._micBtn = document.getElementById('eval-mic-toggle');
        this._micStateEl = document.getElementById('eval-chip-mic');
        this._micFramesEl = document.getElementById('eval-chip-mic-frames');
        this._micBytesEl = document.getElementById('eval-chip-mic-bytes');
        this._micCanvas = document.getElementById('eval-audio-viz');
        this._micCanvasCtx = this._micCanvas?.getContext('2d') || null;
        this._scoreFile = document.getElementById('eval-score-file');
        this._segsFile = document.getElementById('eval-segs-file');
        this._pairingsFile = document.getElementById('eval-pairings-file');
        this._rawEventsFile = document.getElementById('eval-raw-events-file');
        this._baselineEl = document.getElementById('eval-baseline');
        this._openImportBtn = document.getElementById('eval-open-import');
        this._closeImportBtn = document.getElementById('eval-close-import');
        this._importModal = document.getElementById('eval-import-modal');
        this._finalizeBtn = document.getElementById('eval-finalize-btn');
        this._resetBtn = document.getElementById('eval-reset-btn');
    }

    _bind() {
        this._asrToggleBtn?.addEventListener('click', () => {
            const next = !this._asrEnabled;
            this._ws.sendText(`asr_enable:${next ? 'on' : 'off'}`);
        });

        this._micBtn?.addEventListener('click', async () => {
            if (this._capturing) {
                this._stopMicCapture();
            } else {
                await this._startMicCapture();
            }
        });

        this._finalizeBtn?.addEventListener('click', () => {
            this._status = 'finalizing';
            this._renderStats();
            this._ws.sendText('diarizen_finalize');
        });
        this._resetBtn?.addEventListener('click', () => this._reset());

        this._baselineEl?.addEventListener('change', () => {
            const v = Number(this._baselineEl.value);
            if (Number.isFinite(v) && v >= 0 && v <= 100) {
                this._baselinePct = v;
                this._renderImport();
            }
        });

        this._openImportBtn?.addEventListener('click', () => {
            this._importModal?.showModal();
        });
        this._closeImportBtn?.addEventListener('click', () => {
            this._importModal?.close();
        });

        this._scoreFile?.addEventListener('change', async () => {
            const f = this._scoreFile.files?.[0];
            if (!f) return;
            try {
                this._importedScore = JSON.parse(await f.text());
                this._renderImport();
            } catch (err) {
                this._importOut.textContent = `score.json parse error: ${err}`;
            }
        });

        this._segsFile?.addEventListener('change', async () => {
            const f = this._segsFile.files?.[0];
            if (!f) return;
            try {
                this._importedSegments = JSON.parse(await f.text());
                this._renderImport();
            } catch (err) {
                this._importOut.textContent = `diarize_segments.json parse error: ${err}`;
            }
        });

        this._pairingsFile?.addEventListener('change', async () => {
            const f = this._pairingsFile.files?.[0];
            if (!f) return;
            try {
                this._importedPairings = parseJsonl(await f.text());
                this._renderImport();
            } catch (err) {
                this._importOut.textContent = `pairings.jsonl parse error: ${err}`;
            }
        });

        this._rawEventsFile?.addEventListener('change', async () => {
            const f = this._rawEventsFile.files?.[0];
            if (!f) return;
            try {
                this._importedRawEvents = parseJsonl(await f.text());
                this._renderImport();
            } catch (err) {
                this._importOut.textContent = `raw_events.jsonl parse error: ${err}`;
            }
        });
    }

    _reset() {
        this._status = this._connected ? 'running' : 'idle';
        this._streamSec = 0;
        this._streamInSec = 0;
        this._speechFrames = 0;
        this._asrCount = 0;
        this._lastPass = 0;
        this._segmentCount = 0;
        this._changedPending = 0;
        this._finalObj = null;
        this._summaryBody.textContent = 'No final result yet.';
        this._renderStats();
    }

    _fmtBytes(n) {
        if (n < 1024) return `${n} B`;
        if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
        return `${(n / 1048576).toFixed(1)} MB`;
    }

    _drawMicIdle() {
        if (!this._micCanvasCtx || !this._micCanvas) return;
        this._micCanvasCtx.clearRect(0, 0, this._micCanvas.width, this._micCanvas.height);
    }

    _drawMicViz() {
        if (!this._capturing || !this._analyser || !this._micCanvasCtx || !this._micCanvas) return;
        const ctx = this._micCanvasCtx;
        const w = this._micCanvas.width;
        const h = this._micCanvas.height;
        const bufLen = this._analyser.frequencyBinCount;
        const data = new Uint8Array(bufLen);
        this._analyser.getByteFrequencyData(data);

        ctx.clearRect(0, 0, w, h);
        const barW = w / bufLen;
        ctx.fillStyle = '#58a6ff';
        for (let i = 0; i < bufLen; i += 1) {
            const barH = (data[i] / 255) * h;
            ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
        }

        this._micRaf = requestAnimationFrame(() => this._drawMicViz());
    }

    async _startMicCapture() {
        if (!this._connected || !this._ws?.connected) return;
        try {
            this._audioCtx = new AudioContext({ sampleRate: 16000 });
            this._stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });

            await this._audioCtx.audioWorklet.addModule('js/utils/audio-processor.js');
            this._workletNode = new AudioWorkletNode(this._audioCtx, 'audio-capture-processor');
            const src = this._audioCtx.createMediaStreamSource(this._stream);
            this._analyser = this._audioCtx.createAnalyser();
            this._analyser.fftSize = 256;

            src.connect(this._analyser);
            this._analyser.connect(this._workletNode);

            this._workletNode.port.onmessage = (ev) => {
                if (!this._ws?.connected) return;
                const pcm = ev?.data?.pcm;
                if (!pcm) return;
                this._ws.sendBinary(pcm.buffer);
                this._micFrames += 1;
                this._micBytes += pcm.byteLength;
                this._micFramesEl.textContent = `frames: ${this._micFrames}`;
                this._micBytesEl.textContent = `bytes: ${this._fmtBytes(this._micBytes)}`;
            };

            this._capturing = true;
            this._micBtn.textContent = 'Disable Mic';
            this._micBtn.setAttribute('aria-pressed', 'true');
            this._micBtn.classList.add('btn--active');
            this._micStateEl.textContent = 'Mic: on';
            this._micStateEl.dataset.state = 'online';
            this._drawMicViz();
        } catch (err) {
            this._micStateEl.textContent = `Mic error`;
            this._micStateEl.dataset.state = 'failed';
            this._stopMicCapture();
        }
    }

    _stopMicCapture() {
        if (this._micRaf) {
            cancelAnimationFrame(this._micRaf);
            this._micRaf = 0;
        }
        if (this._workletNode) {
            this._workletNode.disconnect();
            this._workletNode = null;
        }
        if (this._stream) {
            this._stream.getTracks().forEach((t) => t.stop());
            this._stream = null;
        }
        if (this._audioCtx) {
            this._audioCtx.close();
            this._audioCtx = null;
        }
        this._analyser = null;
        this._capturing = false;
        if (this._micBtn) {
            this._micBtn.textContent = 'Enable Mic';
            this._micBtn.setAttribute('aria-pressed', 'false');
            this._micBtn.classList.remove('btn--active');
        }
        if (this._micStateEl) {
            this._micStateEl.textContent = 'Mic: off';
            this._micStateEl.dataset.state = 'offline';
        }
        this._drawMicIdle();
    }

    _setStatus(s) {
        this._status = s;
        this._renderStats();
    }

    _renderStats() {
        if (!this._root) return;

        this._connEl.textContent = `WS: ${this._connected ? 'online' : 'offline'}`;
        this._connEl.dataset.state = this._connected ? 'online' : 'offline';

        this._stateEl.textContent = `Run: ${this._status}`;
        this._stateEl.dataset.state = this._status;

        this._kpiStream.textContent = fmtSec(this._streamSec);
        this._kpiStreamIn.textContent = fmtSec(this._streamInSec);
        this._kpiSpeech.textContent = String(this._speechFrames);
        this._kpiAsr.textContent = String(this._asrCount);
        this._kpiPass.textContent = String(this._lastPass);
        this._kpiSegs.textContent = `${this._segmentCount} (${this._changedPending} pending)`;
    }

    _renderFinalSummary(obj) {
        const segments = parseSegments(obj);
        const top = topLabelDurations(segments, 6);
        const rows = [];
        rows.push(`ok=${obj.ok ? 'true' : 'false'}  pass=${obj.pass ?? '-'}  segs=${obj.segment_count ?? obj.n_segments ?? segments.length}`);
        rows.push(`audio_sec=${Number(obj.audio_sec || 0).toFixed(2)}  wall_sec=${Number(obj.wall_sec || 0).toFixed(2)}  origin_sec=${Number(obj.origin_sec || 0).toFixed(2)}`);
        if (top.length) {
            rows.push('top labels by duration:');
            for (const t of top) rows.push(`  ${t.label}: ${t.dur.toFixed(1)}s`);
        }
        this._summaryBody.textContent = rows.join('\n');
    }

    _renderImport() {
        const score = this._importedScore;
        const segs = this._importedSegments;
        const pairings = this._importedPairings;
        const rawEvents = this._importedRawEvents;
        if (!score && !segs && !pairings && !rawEvents) return;

        const lines = [];
        if (score) {
            const livePct = Number(score.micro || score.accuracy || 0) * 100;
            const d = livePct - this._baselinePct;
            lines.push(`accuracy(tests/test.mp3, diarization): ${this._baselinePct.toFixed(1)}% -> ${livePct.toFixed(1)}% (d=${d >= 0 ? '+' : ''}${d.toFixed(1)} pp)`);
            lines.push(`coverage=${fmtPct(Number(score.coverage || 0))}  n_gt=${score.n_gt ?? '-'}  n_decided=${score.n_decided ?? '-'}  n_no_seg=${score.n_no_seg ?? '-'}`);
            if (score.per_spk) {
                lines.push('per speaker:');
                for (const [k, v] of Object.entries(score.per_spk)) {
                    lines.push(`  ${k}: ${fmtPct(Number(v || 0))}`);
                }
            }
        }

        if (segs) {
            const segments = parseSegments(segs);
            const top = topLabelDurations(segments, 6);
            lines.push(`segments file: ${segments.length} parsed`);
            for (const t of top) lines.push(`  ${t.label}: ${t.dur.toFixed(1)}s`);
        }

        if (Array.isArray(pairings)) {
            lines.push(`pairings rows: ${pairings.length}`);
            if (!pairings.length) {
                lines.push('  note: file is empty (current scorer may not emit pairings yet).');
            } else {
                const p = summarizePairings(pairings);
                lines.push(`  decided=${p.decided}  no_segment=${p.noSegment}`);
                lines.push('  per speaker no_segment risk:');
                const risk = Array.from(p.bySpeaker.entries())
                    .map(([speaker, v]) => ({ speaker, ratio: v.total ? v.noSegment / v.total : 0, total: v.total }))
                    .sort((a, b) => b.ratio - a.ratio)
                    .slice(0, 4);
                for (const r of risk) {
                    lines.push(`    ${r.speaker}: ${fmtPct(r.ratio)} (${r.total} rows)`);
                }
            }
        }

        if (Array.isArray(rawEvents)) {
            lines.push(`raw events rows: ${rawEvents.length}`);
            const topTypes = summarizeRawEvents(rawEvents).slice(0, 8);
            for (const t of topTypes) {
                lines.push(`  ${t.type}: ${t.n}`);
            }
            const finals = rawEvents.filter((r) => r.type === 'speaker_diarize_final');
            if (finals.length) {
                const last = finals[finals.length - 1];
                lines.push(`  last final: ok=${!!last.ok} segs=${last.segment_count ?? last.n_segments ?? '-'} wall_sec=${Number(last.wall_sec || 0).toFixed(2)}`);
            }
        }

        if (score && this._finalObj?.ok) {
            const importAcc = Number(score.micro || score.accuracy || 0);
            const finalSegs = Number(this._finalObj.segment_count || this._finalObj.n_segments || 0);
            const importSegs = Number((this._importedSegments?.segment_count || this._importedSegments?.n_segments || parseSegments(this._importedSegments || {}).length) || 0);
            lines.push('live vs imported quick check:');
            lines.push(`  final segs=${finalSegs}  imported segs=${importSegs}`);
            lines.push(`  imported micro=${fmtPct(importAcc)}`);
        }

        this._importOut.textContent = lines.join('\n');
        if (this._importInline) {
            this._importInline.textContent = lines[0] || 'Artifacts loaded.';
        }
    }

    onConnection(connected) {
        this._connected = !!connected;
        if (this._connected && this._status === 'idle') this._status = 'running';
        if (!this._connected) {
            this._status = 'disconnected';
            this._stopMicCapture();
        }
        if (this._micBtn) this._micBtn.disabled = !this._connected;
        this._renderStats();
    }

    onPipelineStats(obj) {
        const t = Number(obj.audio_t1 || 0) / 16000.0;
        const inT = Number(obj.audio_t1_in || 0) / 16000.0;
        if (Number.isFinite(t) && t >= this._streamSec) this._streamSec = t;
        if (Number.isFinite(inT) && inT >= this._streamInSec) this._streamInSec = inT;
        if (obj.is_speech) this._speechFrames += 1;
        if (this._status === 'idle' && this._connected) this._status = 'running';
        this._renderStats();
    }

    onAsrTranscript() {
        this._asrCount += 1;
        this._renderStats();
    }

    onAsrEnable(obj) {
        this._asrEnabled = !!obj?.enabled;
        if (!this._asrToggleBtn) return;
        this._asrToggleBtn.textContent = this._asrEnabled ? 'ASR On' : 'ASR Off';
        this._asrToggleBtn.setAttribute('aria-pressed', this._asrEnabled ? 'true' : 'false');
        this._asrToggleBtn.classList.toggle('btn--active', this._asrEnabled);
    }

    onDiarizeProgress(obj) {
        if (obj?.status) this._setStatus(String(obj.status));
    }

    onDiarizePartial(obj) {
        this._lastPass = Number(obj.pass || 0);
        this._segmentCount = Number(obj.segment_count || obj.n_segments || 0);
        this._changedPending = Number(obj.changed_pending || 0);
        this._setStatus('partial');
    }

    onDiarizeFinal(obj) {
        this._finalObj = obj || {};
        this._lastPass = Number(obj?.pass || this._lastPass);
        this._segmentCount = Number(obj?.segment_count || obj?.n_segments || 0);
        this._changedPending = Number(obj?.changed_pending || 0);
        this._setStatus(obj?.ok ? 'final' : 'failed');
        this._renderFinalSummary(this._finalObj);
    }
}
