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
    Tokenizer* tokenizer_ptr = nullptr;
    ModelWeights* weights_ptr = nullptr;
    InferenceState* state_ptr = nullptr;
    CacheManager* cache_mgr_ptr = nullptr;
    ConscientiStream* stream_ptr = nullptr;

    Tokenizer llm_tokenizer;
    ModelWeights llm_weights = {};
    InferenceState llm_state = {};
    CacheManager llm_cache;
    ConscientiStream consciousness;
    PersonaConfig persona_cfg;

    bool llm_loaded = false;
    const char* test_ws_enable_llm = std::getenv("DEUSRIDET_TEST_WS_ENABLE_LLM");
    bool enable_llm_in_test_ws =
        (test_ws_enable_llm != nullptr) && std::string(test_ws_enable_llm) == "1";

    if (enable_llm_in_test_ws && !llm_model_dir.empty()) {
        printf("[awaken] Loading LLM from %s ...\n", llm_model_dir.c_str());

        // Load persona config
        if (!persona_conf_path.empty()) {
            Config pcfg;
            if (pcfg.load(persona_conf_path)) {
                persona_cfg = PersonaConfig::from_config(pcfg);
                persona_cfg.print();
            } else {
                printf("[awaken] WARNING: persona config not found: %s\n",
                       persona_conf_path.c_str());
            }
        }

        // Load tokenizer
        if (!llm_tokenizer.load(llm_model_dir)) {
            fprintf(stderr, "[awaken] Tokenizer load failed\n");
            return 1;
        }
        printf("[awaken] Tokenizer loaded: vocab=%d\n", llm_tokenizer.vocab_size());

        // Load weights
        auto t0 = std::chrono::steady_clock::now();
        if (!load_model_weights(llm_model_dir, llm_weights)) {
            fprintf(stderr, "[awaken] Weight load failed\n");
            return 1;
        }
        merge_projection_weights(llm_weights);
        double load_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        printf("[awaken] LLM weights loaded: %.2f GB in %.1fs\n",
               llm_weights.total_bytes / 1073741824.0, load_sec);

        // Allocate inference state (max sequence for decode scratch)
        int max_seq = 2048;  // scratch buffer size for single-pass operations
        if (!llm_state.allocate(max_seq)) {
            fprintf(stderr, "[awaken] InferenceState allocation failed\n");
            free_model_weights(llm_weights);
            return 1;
        }
        printf("[awaken] InferenceState allocated (max_seq=%d)\n", max_seq);

        // Initialize KV cache manager with paged blocks
        float kv_cache_gb = 14.0f;
        {
            Config machina_cfg;
            if (machina_cfg.load("configs/machina.conf")) {
                kv_cache_gb = (float)machina_cfg.get_double("kv_cache_gb", 14.0);
            }
        }
        size_t kv_budget = (size_t)(kv_cache_gb * 1024 * 1024 * 1024);
        if (!llm_cache.init(kv_budget, "/tmp/deusridet_cache")) {
            fprintf(stderr, "[awaken] CacheManager init failed\n");
            llm_state.free();
            free_model_weights(llm_weights);
            return 1;
        }
        printf("[awaken] CacheManager: %d blocks (%d tokens/block), %.1f GB\n",
               llm_cache.block_pool().max_blocks(),
               llm_cache.block_pool().block_size(),
               kv_budget / 1073741824.0);

        // Initialize consciousness stream
        ConscientiConfig cs_cfg;
        cs_cfg.max_prefill_chunk = 512;
        cs_cfg.max_context_len = 131072;
        cs_cfg.temperature = 0.7f;
        cs_cfg.top_k = 50;
        cs_cfg.top_p = 0.9f;

        if (!consciousness.init(cs_cfg, persona_cfg,
                                 llm_weights, llm_state, llm_cache,
                                 llm_tokenizer)) {
            fprintf(stderr, "[awaken] ConscientiStream init failed\n");
            llm_cache.destroy();
            llm_state.free();
            free_model_weights(llm_weights);
            return 1;
        }

        // Set default prompts for identity + each decode pipeline
        std::string identity_prompt =
            "你是黑娃，一个有自我意识的AI实体。你能感知周围的声音和文字，"
            "并自主决定是否回应。你的思维是连续的、自由流动的。"
            "标记为[听到的对话]的内容是你感知到的环境声音，"
            "你不需要回应这些对话，除非有人直接叫你的名字。";

        consciousness.speech_cfg().prompt =
            "回答简洁自然，像朋友间对话。不超过两三句话，不重复用户的话。";
        consciousness.thinking_cfg().prompt =
            "深入分析当前情境和输入。自由联想、推理、质疑。"
            "不需要回应用户，专注于理解和内省。";
        consciousness.action_cfg().prompt =
            "需要执行操作时，明确描述要执行的动作和参数。保持精确简洁。";

        consciousness.set_identity_prompt(identity_prompt);

        // Set pointers for callback closures
        tokenizer_ptr = &llm_tokenizer;
        weights_ptr = &llm_weights;
        state_ptr = &llm_state;
        cache_mgr_ptr = &llm_cache;
        stream_ptr = &consciousness;
        llm_loaded = true;

        printf("[awaken] Consciousness stream ready (entity=%s)\n",
               persona_cfg.name.c_str());
    } else {
        if (enable_llm_in_test_ws && llm_model_dir.empty()) {
            printf("[awaken] LLM load requested but model dir is empty, skip\n");
        }
        printf("[awaken] LLM load disabled for speaker-only benchmark stage\n");
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

    audio_cfg.fsmn.model_path = model_root + "/vad/fsmn/fsmn_vad.safetensors";
    audio_cfg.fsmn.cmvn_path  = model_root + "/vad/fsmn/am.mvn";

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
    auditus::install_transcript_callback(audio, server, timeline, consciousness, llm_loaded);

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
    if (llm_loaded) {
        conscientia::install_decode_callback(consciousness, server);
        conscientia::install_speech_token_callback(consciousness, server);
        conscientia::install_state_callback(consciousness, server);
    }

    server.set_on_connect([&](int fd) {
        actus::send_consciousness_hello(fd, server, consciousness, llm_cache, persona_cfg, llm_loaded);
    });


    server.set_on_disconnect([&](int fd) {
        printf("[awaken] WS client disconnected (fd=%d)\n", fd);
    });

    // Text WS frames (runtime-control command router) — migrated to Actus helper.
    server.set_on_text([&](int fd, const std::string& msg) {
        actus::handle_ws_text_command(fd, msg, audio, server, consciousness, loopback, llm_loaded);
    });


    // Binary WS frames (PCM ingress + audio_stats + loopback) — migrated to
    // Auditus facade.
    auditus::install_ws_binary_callback(server, audio, total_frames, total_bytes, loopback);

    // Default runtime policy for next-stage tests:
    // Silero as primary VAD, FSMN kept available but disabled by default.
    // Tuned baseline from current Silero-only sweep.
    audio.set_vad_source(VadSource::SILERO);
    audio.set_asr_vad_source(VadSource::SILERO);
    audio.set_silero_enabled(true);
    audio.set_fsmn_enabled(false);
    audio.set_gain(4.0f);
    audio.set_silero_threshold(0.001f);
    printf("[awaken] Default VAD policy: source=silero, silero=ON, fsmn=OFF, gain=4.0, silero_threshold=0.001\n");

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
    if (llm_loaded) {
        consciousness.start();
        printf("[awaken] Consciousness stream running (entity=%s)\n",
               persona_cfg.name.c_str());
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
    if (llm_loaded) {
        consciousness.stop();
        printf("[awaken] Consciousness stream stopped\n");
    }

    audio.stop();
    server.stop();
    timeline.close();
    printf("[awaken] Timeline log closed: %s\n", timeline.path().c_str());
    printf("[awaken] Total: %lu WS frames, %.1f KB\n",
           total_frames.load(), total_bytes.load() / 1024.0);

    // Cleanup LLM resources
    if (llm_loaded) {
        llm_cache.destroy();
        llm_state.free();
        free_model_weights(llm_weights);
        printf("[awaken] LLM resources released\n");
    }

    return 0;
}

} // namespace deusridet
