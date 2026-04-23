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
#include "nexus/ws_server.h"
#include "sensus/auditus/audio_pipeline.h"
#include "sensus/auditus/auditus_facade.h"
#include "awaken_router.h"
#include "awaken_hello.h"
#include "awaken_consciousness.h"
#include "orator/wavlm_ecapa_encoder.h"
#include "conscientia/stream.h"
#include "conscientia/conscientia_facade.h"
#include "memoria/cache_manager.h"
#include "communis/timeline_logger.h"

namespace deusridet {

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

    // Configure CAM++ speaker encoder model path.
    audio_cfg.speaker.model_path = model_root + "/speaker/campplus/campplus.safetensors";

    // Configure WavLM-Large + ECAPA-TDNN native GPU speaker encoder.
    audio_cfg.wavlm_ecapa_model = model_root + "/speaker/espnet_wavlm_ecapa/wavlm_ecapa.safetensors";
    audio_cfg.wavlm_ecapa_threshold = 0.55f;

    // Speaker-ID benchmark stage: keep ASR fully disabled by default to avoid
    // extra GPU/queue pressure that can distort diarization-focused metrics.
    const char* test_ws_enable_asr = std::getenv("DEUSRIDET_TEST_WS_ENABLE_ASR");
    bool enable_asr_in_test_ws =
        (test_ws_enable_asr != nullptr) && std::string(test_ws_enable_asr) == "1";
    if (enable_asr_in_test_ws) {
        audio_cfg.asr_model_path = model_root + "/asr/Qwen/Qwen3-ASR-1.7B";
        printf("[awaken] ASR load enabled by DEUSRIDET_TEST_WS_ENABLE_ASR=1\n");
    } else {
        audio_cfg.asr_model_path.clear();
        printf("[awaken] ASR load disabled for speaker-only benchmark stage\n");
    }

    // Track WS-level stats.
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<bool> loopback{false};

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
    auditus::install_transcript_callback(audio, server, timeline, cb.stream, cb.loaded);

    // ASR detail log — migrated to Auditus facade.
    auditus::install_asr_log_callback(audio, server);

    // ASR streaming partial callback — migrated to Auditus facade.
    auditus::install_asr_partial_callback(audio, server);

    // Audio-drop callback — migrated to Auditus facade.
    auditus::install_drop_callback(audio, server, timeline);

    // Per-tick pipeline stats (speaker lists, VAD, ASR, tracker, multi-speaker
    // fusion) — migrated to Auditus facade.
    auditus::install_stats_callback(audio, server, timeline);

    // One-shot speaker match (Legacy CAM++ path) — migrated to Auditus facade.
    auditus::install_speaker_match_callback(audio, server);
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
        actus::handle_ws_text_command(fd, msg, audio, server, cb.stream, loopback, cb.loaded);
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
    {
        Config aud_cfg;
        if (aud_cfg.load("configs/auditus.conf")) {
            audio_cfg.speaker_threshold          = (float)aud_cfg.get_double("speaker_threshold",          audio_cfg.speaker_threshold);
            audio_cfg.speaker_register_threshold = (float)aud_cfg.get_double("speaker_register_threshold", audio_cfg.speaker_register_threshold);
            audio_cfg.speaker_discovery_count    =        aud_cfg.get_int   ("speaker_discovery_count",    audio_cfg.speaker_discovery_count);
            audio_cfg.speaker_discovery_boost    = (float)aud_cfg.get_double("speaker_discovery_boost",    audio_cfg.speaker_discovery_boost);
            audio_cfg.speaker_recency_window_sec = (float)aud_cfg.get_double("speaker_recency_window_sec", audio_cfg.speaker_recency_window_sec);
            audio_cfg.speaker_recency_bonus      = (float)aud_cfg.get_double("speaker_recency_bonus",      audio_cfg.speaker_recency_bonus);
            audio_cfg.speaker_margin_abstain     = (float)aud_cfg.get_double("speaker_margin_abstain",     audio_cfg.speaker_margin_abstain);
            audio_cfg.speaker_max_auto_reg_count =        aud_cfg.get_int   ("speaker_max_auto_reg_count", audio_cfg.speaker_max_auto_reg_count);
            audio_cfg.speaker_min_fbank_frames   =        aud_cfg.get_int   ("speaker_min_fbank_frames",   audio_cfg.speaker_min_fbank_frames);
            printf("[awaken] Auditus diarization knobs loaded from configs/auditus.conf:\n"
                   "           match=%.3f reg=%.3f disc=[count=%d,boost=%.3f] recency=[win=%.1fs,bonus=%.3f] margin=%.3f max_autoreg=%d min_fbank=%d\n",
                   audio_cfg.speaker_threshold, audio_cfg.speaker_register_threshold,
                   audio_cfg.speaker_discovery_count, audio_cfg.speaker_discovery_boost,
                   audio_cfg.speaker_recency_window_sec, audio_cfg.speaker_recency_bonus,
                   audio_cfg.speaker_margin_abstain, audio_cfg.speaker_max_auto_reg_count,
                   audio_cfg.speaker_min_fbank_frames);
        } else {
            printf("[awaken] configs/auditus.conf not found — using compiled defaults\n");
        }
    }

    // Start audio pipeline.
    if (!audio.start(audio_cfg)) {
        fprintf(stderr, "[awaken] Failed to start audio pipeline\n");
        return 1;
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

    // Block until SIGINT/SIGTERM.
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &mask, nullptr);
    int sig = 0;
    sigwait(&mask, &sig);
    printf("\n[awaken] Caught signal %d, shutting down...\n", sig);

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
