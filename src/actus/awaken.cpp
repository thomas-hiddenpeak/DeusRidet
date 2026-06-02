/**
 * @file awaken.cpp
 * @philosophical_role External command `awaken`. An Actus function — one CLI verb, one finite
 *         act, one return code.
 * @serves main.cpp dispatch (declaration in actus.h).
 */


#include "actus.h"
#include "communis/config.h"
#include "communis/log.h"
#include "communis/tegra.h"
#include "machina/gptq.h"
#include "machina/gptq_gemm_v2.h"
#include "machina/model.h"
#include "machina/forward.h"
#include "machina/allocator.h"
#include "machina/safetensors.h"
#include "machina/tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <signal.h>
#include <cerrno>
#include "nexus/ws_server.h"
#include "sensus/auditus/audio_pipeline.h"
#include "sensus/auditus/auditus_facade.h"
#include "awaken_router.h"
#include "awaken_hello.h"
#include "awaken_consciousness.h"
#include "orator/wavlm_ecapa_encoder.h"
#include "orator/diarizen_pipeline.h"
#include "orator/diarizen_periodic_worker.h"
#include "orator/diarizen_pipeline.h"
#include "sensus/auditus/transcript_holdback.h"
#include "conscientia/stream.h"
#include "conscientia/conscientia_facade.h"
#include "memoria/cache_manager.h"
#include "communis/timeline_logger.h"
#include "vires/vires_facade.h"

namespace deusridet {

// Serialize the Vires compute ledger into a `vires_compute_snapshot` WS message
// and broadcast it to all WebUI clients. JSON lives at the Nexus boundary; Vires
// core stays compute-only and never learns about WS/JSON. Called on a fixed
// telemetry cadence from the awaken main thread.
static void broadcast_vires_snapshot(WsServer& server) {
    const vires::Snapshot snap = vires::Arbiter::instance().snapshot();
    std::string js;
    js.reserve(128 + snap.consumers.size() * 96);
    js += "{\"type\":\"vires_compute_snapshot\"";
    js += ",\"greatest_priority\":" + std::to_string(snap.greatest_priority);
    js += ",\"least_priority\":" + std::to_string(snap.least_priority);
    js += ",\"background_yielding\":";
    js += (snap.background_yielding ? "true" : "false");
    js += ",\"foreground_idle_us\":";
    js += (snap.foreground_idle_us == UINT64_MAX)
              ? "null"
              : std::to_string(snap.foreground_idle_us);
    js += ",\"consumers\":[";
    for (size_t i = 0; i < snap.consumers.size(); ++i) {
        const vires::ConsumerStat& c = snap.consumers[i];
        if (i) js += ",";
        js += "{\"id\":" + std::to_string(c.id);
        js += ",\"name\":\"" + c.name + "\"";
        js += ",\"priority\":\"";
        js += vires::priority_str(c.priority);
        js += "\",\"submitted\":" + std::to_string(c.submitted);
        js += ",\"reclaimed\":" + std::to_string(c.reclaimed) + "}";
    }
    js += "]}";
    server.broadcast_text(js);
}

int awaken(const std::string& webui_dir,
                const std::string& llm_model_dir,
                const std::string& persona_conf_path,
                float replay_speed) {
    printf("[awaken] Starting WebSocket + Audio Pipeline...\n");
    printf("[awaken] WebUI dir: %s\n", webui_dir.c_str());
    if (replay_speed != 1.0f) {
        printf("[awaken] Replay speed: %.2fx (AUDIO anchor period scaled; "
               "T0 tracks wall time, T1 tracks source-audio samples)\n",
               (double)replay_speed);
    }

    // ── LLM + Consciousness setup (optional — skip if no model dir) ──
    // Bootstrap bundle owns the six subsystems; see
    // awaken_consciousness.{h,cpp} for the installer. `awaken()` still
    // owns the matching destroy/free at shutdown (bottom of this file).
    ConscientiaBootstrap cb;
    if (int rc = bootstrap_consciousness(llm_model_dir, persona_conf_path, cb)) {
        return rc;
    }

    WsServer server;
    WsServerConfig ws_cfg;
    ws_cfg.port = 8080;
    ws_cfg.static_dir = webui_dir;

    // Audio pipeline.
    AudioPipeline audio;
    AudioPipelineConfig audio_cfg;
    // defaults: n_fft=400, hop=160, n_mels=128, sr=16000
    audio_cfg.replay_speed = replay_speed;

    // Model root (workspace-local by default; override with DEUSRIDET_MODEL_ROOT).
    std::string model_root = getenv("DEUSRIDET_MODEL_ROOT")
                             ? getenv("DEUSRIDET_MODEL_ROOT")
                             : "/home/rm01/DeusRidet/models/dev";

    // Configure Silero VAD model path.
    audio_cfg.silero.model_path = model_root + "/vad/silero_vad.safetensors";

    // Configure FRCRN speech enhancement (CUDA GPU, safetensors weights).
    audio_cfg.frcrn.weights_dir = model_root + "/vad/frcrn_weights";

    // Configure FSMN VAD model paths.
    // Configure P1: pyannote overlap detection (native CUDA).
    audio_cfg.overlap_det.model_path = model_root + "/vad/pyannote_seg3.safetensors";
    audio_cfg.overlap_det.enabled = true;
    // Override overlap confidence threshold (softmax) via env for S4 sweep.
    // Default (0.5) is tuned in overlap_detector.h; accepts [0.0, 1.0].
    if (const char* thr_env = std::getenv("DEUSRIDET_OVERLAP_THRESHOLD")) {
        float thr = std::atof(thr_env);
        if (thr > 0.0f && thr <= 1.0f) {
            audio_cfg.overlap_det.overlap_threshold = thr;
        }
    }

    // Configure P2: MossFormer2 speech separation (native CUDA, lazy loaded).
    audio_cfg.separator.model_path = model_root + "/vad/mossformer2_ss_16k.safetensors";
    audio_cfg.separator.lazy_load = true;
    if (const char* sep_overlap_env = std::getenv("DEUSRIDET_SEPARATOR_OVERLAP_SAMPLES")) {
        int overlap = std::atoi(sep_overlap_env);
        if (overlap >= 0 && overlap < audio_cfg.separator.max_chunk) {
            audio_cfg.separator.overlap_samples = overlap;
            printf("[awaken] Separator overlap override: %d samples\n", overlap);
        } else {
            fprintf(stderr, "[awaken] Ignoring invalid DEUSRIDET_SEPARATOR_OVERLAP_SAMPLES=%s\n",
                    sep_overlap_env);
        }
    }

    // Configure CAM++ speaker encoder model path.
    audio_cfg.speaker.model_path = model_root + "/speaker/campplus/campplus.safetensors";

    // Configure WavLM-Large + ECAPA-TDNN native GPU speaker encoder.
    audio_cfg.wavlm_ecapa_model = model_root + "/speaker/espnet_wavlm_ecapa/wavlm_ecapa.safetensors";
    audio_cfg.wavlm_ecapa_threshold = 0.55f;

    // ASR is part of the canonical perception path: speaker-id is never
    // validated in isolation from ASR — the two share the live pipeline and
    // the holdback speaker<->content boundary, so a speaker-only benchmark
    // would measure a configuration the system never actually runs in. Load
    // ASR by default; opt out only with DEUSRIDET_TEST_WS_ENABLE_ASR=0 for a
    // pure VAD/diarization micro-benchmark.
    const char* test_ws_enable_asr = std::getenv("DEUSRIDET_TEST_WS_ENABLE_ASR");
    bool enable_asr_in_test_ws =
        (test_ws_enable_asr == nullptr) || std::string(test_ws_enable_asr) != "0";
    if (enable_asr_in_test_ws) {
        audio_cfg.asr_model_path = model_root + "/asr/Qwen/Qwen3-ASR-1.7B";
        printf("[awaken] ASR load enabled (default; set "
               "DEUSRIDET_TEST_WS_ENABLE_ASR=0 to disable)\n");
    } else {
        audio_cfg.asr_model_path.clear();
        printf("[awaken] ASR load disabled by DEUSRIDET_TEST_WS_ENABLE_ASR=0\n");
    }

    // Track WS-level stats.
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<bool> loopback{false};

    // DiariZen-v2 in-process native pipeline. Loaded once at startup
    // whenever DiariZen capture is enabled; replaces the retired Python
    // worker/facade IPC bridge. Resident once loaded: WavLM-pruned +
    // Conformer head + ResNet34 embedder + clustering.
    std::shared_ptr<orator::DiarizenPipeline> diarizen_native;
    // Hybrid P2: ASR→Conscientia holdback + periodic recluster worker.
    // Constructed before install_transcript_callback so the lambda can
    // capture the holdback pointer; left null when DiariZen is disabled.
    std::unique_ptr<auditus::TranscriptHoldback> diarizen_holdback;
    std::unique_ptr<orator::DiarizenPeriodicWorker> diarizen_worker;
    double diarizen_cap_sec = 0.0;
    // Periodic recluster cadence. 30s is the sweet spot from the Jun 2 live
    // 1x replay: max per-pass wall on the 120s window was ~19s, so 30s keeps
    // ~37% headroom (no backlog), and 75s holdback / 30s = 2.5 correction
    // passes per pending item (enough redundancy to retroactively fix the
    // speaker cold-start lag). Override with DEUSRIDET_DIARIZEN_PERIOD_SEC.
    double diarizen_period_sec = 30.0;
    double diarizen_holdback_sec = 75.0;
    // P3c default flip (2026-05-30): native DiariZen is ON by default. The
    // LLM-loaded live gate cleared at 93.55% (≥93.5%) with finalize RTF 0.10
    // and zero CUDA errors, so the in-process pipeline is the default speaker
    // re-attribution path. Opt out with DEUSRIDET_DIARIZEN_ENABLE=0. Periodic
    // full-session re-diarise stays separately gated behind
    // DEUSRIDET_DIARIZEN_PERIODIC=1 (O(N²); off by default — see the worker).
    bool   diarizen_enabled = true;
    if (const char* en = std::getenv("DEUSRIDET_DIARIZEN_ENABLE")) {
        if (en[0] == '0') diarizen_enabled = false;
    }
    if (diarizen_enabled) {
        diarizen_cap_sec = 4000.0;
        if (const char* cap = std::getenv("DEUSRIDET_DIARIZEN_CAP_SEC")) {
            double v = std::atof(cap);
            if (v >= 60.0 && v <= 14400.0) diarizen_cap_sec = v;
        }
        if (const char* p = std::getenv("DEUSRIDET_DIARIZEN_PERIOD_SEC")) {
            double v = std::atof(p);
            if (v >= 5.0 && v <= 3600.0) diarizen_period_sec = v;
        }
        if (const char* h = std::getenv("DEUSRIDET_TRANSCRIPT_HOLDBACK_SEC")) {
            double v = std::atof(h);
            if (v >= 1.0 && v <= 3600.0) diarizen_holdback_sec = v;
        }
        if (cb.loaded) {
            diarizen_holdback = std::make_unique<auditus::TranscriptHoldback>(
                cb.stream, diarizen_holdback_sec,
                [&audio]() { return audio.audio_t1_in_sec(); });
        }
    }

    // Persistent timeline data logger (JSONL).
    TimelineLogger timeline;
    if (timeline.open()) {
        printf("[awaken] Timeline log: %s\n", timeline.path().c_str());
    } else {
        fprintf(stderr, "[awaken] WARNING: failed to open timeline log\n");
    }

    // Helper: strip trailing incomplete UTF-8 sequence from a byte string.
    // Aliases to the Auditus-facade helpers so remaining (non-migrated) callbacks
    // keep their short call-site form.
    using auditus::sanitize_utf8;
    using auditus::json_escape;

    // Audio pipeline callbacks — vad / asr_partial / drop migrated to Auditus facade.
    auditus::install_vad_callback(audio, server, timeline);

    // ASR full transcript — migrated to Auditus facade (wires ws "asr_transcript"
    // envelope + timeline log_asr + optional injection into consciousness stream).
    auditus::install_transcript_callback(audio, server, timeline, cb.stream, cb.loaded,
                                          diarizen_holdback.get());

    // ASR detail log — migrated to Auditus facade.
    auditus::install_asr_log_callback(audio, server, timeline);

    // ASR streaming partial callback — migrated to Auditus facade.
    auditus::install_asr_partial_callback(audio, server);

    // Audio-drop callback — migrated to Auditus facade.
    auditus::install_drop_callback(audio, server, timeline);

    // Per-tick pipeline stats (speaker lists, VAD, ASR, tracker, multi-speaker
    // fusion) — migrated to Auditus facade.
    auditus::install_stats_callback(audio, server, timeline);

    // One-shot speaker match (Legacy CAM++ path) — migrated to Auditus facade.
    auditus::install_speaker_match_callback(audio, server);

    // OratorReclusterer global-identity correction — emits ws "speaker_relabel"
    // whenever the rolling-window spectral pass disagrees with the online id.
    auditus::install_speaker_relabel_callback(audio, server);
    // ── Consciousness stream callbacks — migrated to Conscientia facade ──
    // decode / speech_token / state broadcasts were 83 inline lines; each is
    // now an `install_*` call wiring ConscientiStream → WsServer with byte-
    // identical JSON envelopes.
    if (cb.loaded) {
        conscientia::install_decode_callback(cb.stream, server);
        conscientia::install_speech_token_callback(cb.stream, server);
        conscientia::install_state_callback(cb.stream, server);
    }

    server.set_on_connect([&](int fd) {
        actus::send_consciousness_hello(fd, server, cb.stream, cb.cache, cb.persona_cfg, cb.loaded);
    });


    server.set_on_disconnect([&](int fd) {
        printf("[awaken] WS client disconnected (fd=%d)\n", fd);
    });

    // Text WS frames (runtime-control command router) — migrated to Actus helper.
    server.set_on_text([&](int fd, const std::string& msg) {
        actus::handle_ws_text_command(fd, msg, audio, server, cb.stream, loopback, cb.loaded,
                                       diarizen_worker.get(),
                                       diarizen_native.get());
    });


    // Binary WS frames (PCM ingress + audio_stats + loopback) — migrated to
    // Auditus facade.
    auditus::install_ws_binary_callback(server, audio, total_frames, total_bytes, loopback);

    // Default runtime policy: Silero is the sole VAD (FSMN removed April 2026,
    // lost to Silero at every tested threshold per Step 2 evaluation matrix).
    audio.set_vad_source(VadSource::SILERO);
    audio.set_asr_vad_source(VadSource::SILERO);
    audio.set_silero_enabled(true);
    audio.set_gain(4.0f);
    audio.set_silero_threshold(0.001f);
    printf("[awaken] Default VAD policy: source=silero, silero=ON, gain=4.0, silero_threshold=0.001\n");

    // Load configs/auditus.conf (diarization runtime knobs). Missing keys
    // fall back to AudioPipelineConfig defaults, so the file is optional.
    // Only the four gates the clean three-concern online facade consumes are
    // parsed; the pre-redesign discovery/recency/short-identify/multi-gate/
    // campp-shadow/inherit-peek knobs were removed (2026-06-02 anti-entropy).
    {
        Config aud_cfg;
        if (aud_cfg.load("configs/auditus.conf")) {
            audio_cfg.speaker_threshold          = (float)aud_cfg.get_double("speaker_threshold",          audio_cfg.speaker_threshold);
            audio_cfg.speaker_register_threshold = (float)aud_cfg.get_double("speaker_register_threshold", audio_cfg.speaker_register_threshold);
            audio_cfg.speaker_margin_abstain     = (float)aud_cfg.get_double("speaker_margin_abstain",     audio_cfg.speaker_margin_abstain);
            audio_cfg.speaker_min_fbank_frames   =        aud_cfg.get_int   ("speaker_min_fbank_frames",   audio_cfg.speaker_min_fbank_frames);
            printf("[awaken] Auditus diarization knobs loaded from configs/auditus.conf:\n"
                   "           match=%.3f reg=%.3f margin=%.3f min_fbank=%d\n",
                   audio_cfg.speaker_threshold, audio_cfg.speaker_register_threshold,
                   audio_cfg.speaker_margin_abstain, audio_cfg.speaker_min_fbank_frames);
        } else {
            printf("[awaken] configs/auditus.conf not found — using compiled defaults\n");
        }
    }

    // Start audio pipeline.
    if (!audio.start(audio_cfg)) {
        fprintf(stderr, "[awaken] Failed to start audio pipeline\n");
        return 1;
    }

    // Vires — autonomic GPU compute substrate. Construct the process-global
    // arbiter once here so it queries the device's cooperative stream-priority
    // range (the load-bearing feasibility assumption) and becomes observable.
    // V1 delivery core: consumers route their streams through Vires in a later
    // increment; this boot touch only lights up the substrate. Compute only —
    // GPU memory remains Memoria's charge.
    (void)vires::Arbiter::instance();

    // DiariZen-v2 capture + native pipeline. Off by default; enable with
    // DEUSRIDET_DIARIZEN_ENABLE=1. The in-process CUDA pipeline is loaded
    // once at startup (no Python subprocess). Hybrid P2 also spawns a
    // periodic worker that re-runs DiariZen every
    // DEUSRIDET_DIARIZEN_PERIOD_SEC seconds and rewrites the speaker_id of
    // still-pending transcripts before they reach Conscientia.
    if (diarizen_enabled) {
        audio.diarizen_capture_enable(true, diarizen_cap_sec);
        // Load the native pipeline once at startup. On failure the
        // diarizen subsystem stays disabled (no Python fallback exists).
        auto np = std::make_shared<orator::DiarizenPipeline>();
        orator::DiarizenPipelineConfig np_cfg;
        if (np->load(np_cfg)) {
            diarizen_native = std::move(np);
            printf("[awaken] DiariZen-v2 native pipeline LOADED "
                   "(in-process CUDA)\n");
        } else {
            printf("[awaken] DiariZen-v2 native load FAILED (%s); "
                   "diarizen disabled\n", np->last_error().c_str());
            diarizen_enabled = false;
        }
        if (diarizen_enabled) {
            // A: the periodic worker is created unconditionally so
            // diarizen_trigger / periodic re-diarise work in audio-only
            // sessions too. The holdback (LLM-only Conscientia drain) is
            // passed as a nullable pointer — when the LLM is not loaded the
            // worker still diarises and broadcasts speaker_diarize_* for the
            // live WebUI, it just has no transcript queue to rewrite.
            if (diarizen_holdback) diarizen_holdback->start();
            diarizen_worker = std::make_unique<orator::DiarizenPeriodicWorker>(
                audio, *diarizen_native, diarizen_holdback.get(), server,
                diarizen_period_sec);
            diarizen_worker->start();
            const char* periodic_env = std::getenv("DEUSRIDET_DIARIZEN_PERIODIC");
            const bool periodic_on =
                (periodic_env == nullptr) || std::string(periodic_env) != "0";
            std::string periodic_desc =
                periodic_on
                    ? ("ON period=" + std::to_string((long)diarizen_period_sec) + "s")
                    : std::string("OFF (DEUSRIDET_DIARIZEN_PERIODIC=0; trigger/finalize only)");
            // Direction C — sliding-window live diarise. Default 120s: covers
            // the 75s holdback with margin and bounds per-pass wall so the
            // shared GPU is not starved. Unset keeps this default; env=0 forces
            // full-session passes. Finalize is always full regardless.
            const char* win_env = std::getenv("DEUSRIDET_DIARIZEN_WINDOW_SEC");
            const double win_sec = win_env ? std::atof(win_env) : 120.0;
            std::string window_desc =
                (win_sec > 0.0)
                    ? ("window=" + std::to_string((long)win_sec) + "s (finalize=full)")
                    : std::string("window=OFF (full-session passes)");
            if (diarizen_holdback) {
                printf("[awaken] DiariZen-v2 Hybrid P2 ENABLED "
                       "(cap=%.0fs holdback=%.0fs; periodic=%s; %s); "
                       "WS commands: diarizen_trigger / diarizen_finalize\n",
                       diarizen_cap_sec, diarizen_holdback_sec,
                       periodic_desc.c_str(), window_desc.c_str());
            } else {
                printf("[awaken] DiariZen-v2 worker ENABLED "
                       "(cap=%.0fs, LLM not loaded so no holdback drain; "
                       "periodic=%s; %s); WS commands: diarizen_trigger / "
                       "diarizen_finalize\n",
                       diarizen_cap_sec, periodic_desc.c_str(),
                       window_desc.c_str());
            }
        }
    }

    // Start WS server.
    if (!server.start(ws_cfg)) {
        fprintf(stderr, "[awaken] Failed to start WS server\n");
        audio.stop();
        return 1;
    }

    printf("[awaken] Server running on http://localhost:%d\n", ws_cfg.port);
    printf("[awaken] Audio pipeline: Mel(n_fft=%d hop=%d mels=%d) + VAD\n",
           audio_cfg.mel.n_fft, audio_cfg.mel.hop_length, audio_cfg.mel.n_mels);

    // Start consciousness stream (after server is running so callbacks work)
    if (cb.loaded) {
        cb.stream.start();
        printf("[awaken] Consciousness stream running (entity=%s)\n",
               cb.persona_cfg.name.c_str());
    }

    printf("[awaken] Press Ctrl+C to stop...\n");

    // Block until SIGINT/SIGTERM, emitting a Vires compute telemetry snapshot to
    // the WebUI on a fixed cadence between signals. The awaken main thread is
    // otherwise idle here, so it doubles as the telemetry heartbeat — no extra
    // thread, matching the "CPU for orchestration only" rule.
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &mask, nullptr);
    int sig = 0;
    const struct timespec telem_period = {2, 0};  // 2 s snapshot cadence
    for (;;) {
        sig = sigtimedwait(&mask, nullptr, &telem_period);
        if (sig > 0) break;                 // SIGINT/SIGTERM received
        if (errno == EAGAIN) {              // cadence elapsed — emit telemetry
            broadcast_vires_snapshot(server);
            continue;
        }
        if (errno == EINTR) continue;       // interrupted by another signal
        break;                              // unexpected — fall through to exit
    }
    printf("\n[awaken] Caught signal %d, shutting down...\n", sig);

    // Hybrid P2: drain the DiariZen periodic worker + holdback before we
    // tear down Conscientia so any still-pending transcript is injected
    // with the freshest possible speaker_id.
    if (diarizen_worker) {
        printf("[awaken] DiariZen-v2 P2 finalize: triggering one last pass\n");
        diarizen_worker->finalize();
        diarizen_worker.reset();
    }
    if (diarizen_holdback) {
        diarizen_holdback->stop();
        diarizen_holdback.reset();
    }

    // Stop consciousness first (it depends on model/cache)
    if (cb.loaded) {
        cb.stream.stop();
        printf("[awaken] Consciousness stream stopped\n");
    }

    audio.stop();
    server.stop();
    timeline.close();
    printf("[awaken] Timeline log closed: %s\n", timeline.path().c_str());
    printf("[awaken] Total: %lu WS frames, %.1f KB\n",
           total_frames.load(), total_bytes.load() / 1024.0);

    // Cleanup LLM resources
    if (cb.loaded) {
        cb.cache.destroy();
        cb.state.free();
        free_model_weights(cb.weights);
        printf("[awaken] LLM resources released\n");
    }

    return 0;
}

} // namespace deusridet
