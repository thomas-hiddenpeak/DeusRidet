/**
 * @file cmd_test_ws.cpp
 * @philosophical_role External command `cmd_test_ws`. An Actus function — one CLI verb, one finite
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
#include "orator/wavlm_ecapa_encoder.h"
#include "conscientia/stream.h"
#include "memoria/cache_manager.h"
#include "communis/timeline_logger.h"

namespace deusridet {

int cmd_test_ws(const std::string& webui_dir,
                const std::string& llm_model_dir,
                const std::string& persona_conf_path,
                float replay_speed) {
    printf("[test-ws] Starting WebSocket + Audio Pipeline...\n");
    printf("[test-ws] WebUI dir: %s\n", webui_dir.c_str());
    if (replay_speed != 1.0f) {
        printf("[test-ws] Replay speed: %.2fx (AUDIO anchor period scaled; "
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
        printf("[test-ws] Loading LLM from %s ...\n", llm_model_dir.c_str());

        // Load persona config
        if (!persona_conf_path.empty()) {
            Config pcfg;
            if (pcfg.load(persona_conf_path)) {
                persona_cfg = PersonaConfig::from_config(pcfg);
                persona_cfg.print();
            } else {
                printf("[test-ws] WARNING: persona config not found: %s\n",
                       persona_conf_path.c_str());
            }
        }

        // Load tokenizer
        if (!llm_tokenizer.load(llm_model_dir)) {
            fprintf(stderr, "[test-ws] Tokenizer load failed\n");
            return 1;
        }
        printf("[test-ws] Tokenizer loaded: vocab=%d\n", llm_tokenizer.vocab_size());

        // Load weights
        auto t0 = std::chrono::steady_clock::now();
        if (!load_model_weights(llm_model_dir, llm_weights)) {
            fprintf(stderr, "[test-ws] Weight load failed\n");
            return 1;
        }
        merge_projection_weights(llm_weights);
        double load_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        printf("[test-ws] LLM weights loaded: %.2f GB in %.1fs\n",
               llm_weights.total_bytes / 1073741824.0, load_sec);

        // Allocate inference state (max sequence for decode scratch)
        int max_seq = 2048;  // scratch buffer size for single-pass operations
        if (!llm_state.allocate(max_seq)) {
            fprintf(stderr, "[test-ws] InferenceState allocation failed\n");
            free_model_weights(llm_weights);
            return 1;
        }
        printf("[test-ws] InferenceState allocated (max_seq=%d)\n", max_seq);

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
            fprintf(stderr, "[test-ws] CacheManager init failed\n");
            llm_state.free();
            free_model_weights(llm_weights);
            return 1;
        }
        printf("[test-ws] CacheManager: %d blocks (%d tokens/block), %.1f GB\n",
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
            fprintf(stderr, "[test-ws] ConscientiStream init failed\n");
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

        printf("[test-ws] Consciousness stream ready (entity=%s)\n",
               persona_cfg.name.c_str());
    } else {
        if (enable_llm_in_test_ws && llm_model_dir.empty()) {
            printf("[test-ws] LLM load requested but model dir is empty, skip\n");
        }
        printf("[test-ws] LLM load disabled for speaker-only benchmark stage\n");
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
        printf("[test-ws] ASR load enabled by DEUSRIDET_TEST_WS_ENABLE_ASR=1\n");
    } else {
        audio_cfg.asr_model_path.clear();
        printf("[test-ws] ASR load disabled for speaker-only benchmark stage\n");
    }

    // Track WS-level stats.
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<bool> loopback{false};

    // Persistent timeline data logger (JSONL).
    TimelineLogger timeline;
    if (timeline.open()) {
        printf("[test-ws] Timeline log: %s\n", timeline.path().c_str());
    } else {
        fprintf(stderr, "[test-ws] WARNING: failed to open timeline log\n");
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
    // ── Consciousness stream callbacks ──────────────────────────────
    if (llm_loaded) {
        // Decode output → WS as JSON
        consciousness.set_on_decode([&](const DecodeResult& result) {
            std::string escaped = json_escape(result.text);
            const char* state_names[] = {"active", "daydream", "dreaming"};
            char json[4096];
            snprintf(json, sizeof(json),
                R"({"type":"consciousness_decode","text":"%s","tokens":%d,"time_ms":%.1f,"state":"%s"})",
                escaped.c_str(), result.tokens_generated, result.time_ms,
                state_names[static_cast<int>(result.state_during)]);
            server.broadcast_text(json);
            printf("[consciousness] %s: \"%s\" (%d tok, %.0fms)\n",
                   state_names[static_cast<int>(result.state_during)],
                   result.text.c_str(), result.tokens_generated, result.time_ms);
        });

        // Per-token speech streaming → WS as JSON
        // token_id == -1 signals start of new speech (reset accumulator on frontend)
        consciousness.set_on_speech_token([&](const std::string& token_text, int token_id) {
            std::string escaped = json_escape(token_text);
            char json[512];
            snprintf(json, sizeof(json),
                R"({"type":"speech_token","text":"%s","token_id":%d})",
                escaped.c_str(), token_id);
            server.broadcast_text(json);
        });

        // State update → WS as JSON (includes metrics + system stats)
        consciousness.set_on_state([&](WakefulnessState state,
                                        float wakefulness,
                                        int kv_blocks_used,
                                        int kv_blocks_free,
                                        int current_pos,
                                        const ConscientiMetrics& metrics) {
            // System memory stats
            size_t cuda_free = 0, cuda_total = 0;
            cudaMemGetInfo(&cuda_free, &cuda_total);
            size_t mem_avail_kb = read_memavail_kb();
            size_t rss_kb = 0;
            {
                FILE* fp = fopen("/proc/self/status", "r");
                if (fp) {
                    char line[256];
                    while (fgets(line, sizeof(line), fp)) {
                        if (strncmp(line, "VmRSS:", 6) == 0) {
                            rss_kb = strtoull(line + 6, nullptr, 10);
                            break;
                        }
                    }
                    fclose(fp);
                }
            }

            const char* state_names[] = {"active", "daydream", "dreaming"};
            char json[1024];
            snprintf(json, sizeof(json),
                R"({"type":"consciousness_state","state":"%s","wakefulness":%.3f,)"
                R"("kv_used":%d,"kv_free":%d,"pos":%d,)"
                R"("prefill_ms":%.1f,"prefill_tps":%.1f,"prefill_tokens":%d,)"
                R"("decode_ms_per_tok":%.1f,"decode_tokens":%d,)"
                R"("total_prefill_tok":%d,"total_decode_tok":%d,)"
                R"("total_prefill_ms":%.0f,"total_decode_ms":%.0f,)"
                R"("cuda_free_mb":%.0f,"cuda_total_mb":%.0f,)"
                R"("mem_avail_mb":%.0f,"rss_mb":%.0f})",
                state_names[static_cast<int>(state)], wakefulness,
                kv_blocks_used, kv_blocks_free, current_pos,
                metrics.last_prefill_ms, metrics.last_prefill_tps,
                metrics.last_prefill_tokens,
                metrics.last_decode_ms_per_tok, metrics.last_decode_tokens,
                metrics.total_prefill_tokens, metrics.total_decode_tokens,
                metrics.total_prefill_ms, metrics.total_decode_ms,
                cuda_free / 1048576.0, cuda_total / 1048576.0,
                mem_avail_kb / 1024.0, rss_kb / 1024.0);
            server.broadcast_text(json);
        });
    }

    server.set_on_connect([&](int fd) {
        printf("[test-ws] WS client connected  (fd=%d)\n", fd);
        // Send initial consciousness state with full context
        if (llm_loaded) {
            const char* state_names[] = {"active", "daydream", "dreaming"};
            auto m = consciousness.metrics();

            size_t cuda_free = 0, cuda_total = 0;
            cudaMemGetInfo(&cuda_free, &cuda_total);
            size_t mem_avail_kb = read_memavail_kb();
            size_t rss_kb = 0;
            {
                FILE* fp = fopen("/proc/self/status", "r");
                if (fp) {
                    char line[256];
                    while (fgets(line, sizeof(line), fp)) {
                        if (strncmp(line, "VmRSS:", 6) == 0) {
                            rss_kb = strtoull(line + 6, nullptr, 10);
                            break;
                        }
                    }
                    fclose(fp);
                }
            }

            char json[4096];
            snprintf(json, sizeof(json),
                R"({"type":"consciousness_state","state":"%s","wakefulness":%.3f,)"
                R"("kv_used":%d,"kv_free":%d,"pos":%d,)"
                R"("llm_loaded":true,"entity":"%s",)"
                R"("prefill_ms":%.1f,"prefill_tps":%.1f,"prefill_tokens":%d,)"
                R"("decode_ms_per_tok":%.1f,"decode_tokens":%d,)"
                R"("total_prefill_tok":%d,"total_decode_tok":%d,)"
                R"("total_prefill_ms":%.0f,"total_decode_ms":%.0f,)"
                R"("cuda_free_mb":%.0f,"cuda_total_mb":%.0f,)"
                R"("mem_avail_mb":%.0f,"rss_mb":%.0f,)"
                R"("enable_response":%s,"enable_daydream":%s,"enable_dreaming":%s,"enable_llm":%s,)"
                R"("enable_speech":%s,"enable_thinking":%s,"enable_action":%s,)"
                R"("temperature":%.2f,"top_k":%d,"top_p":%.2f,)"
                R"("speech_max_tokens":%d,"thinking_max_tokens":%d,)"
                R"("speech":{"temperature":%.2f,"top_k":%d,"top_p":%.2f,"max_tokens":%d},)"
                R"("thinking":{"temperature":%.2f,"top_k":%d,"top_p":%.2f,"max_tokens":%d},)"
                R"("action":{"temperature":%.2f,"top_k":%d,"top_p":%.2f,"max_tokens":%d}})",
                state_names[static_cast<int>(consciousness.current_state())],
                consciousness.scheduler().wakefulness(),
                llm_cache.gpu_blocks_used(), llm_cache.free_blocks(),
                consciousness.current_pos(),
                persona_cfg.name.c_str(),
                m.last_prefill_ms, m.last_prefill_tps, m.last_prefill_tokens,
                m.last_decode_ms_per_tok, m.last_decode_tokens,
                m.total_prefill_tokens, m.total_decode_tokens,
                m.total_prefill_ms, m.total_decode_ms,
                cuda_free / 1048576.0, cuda_total / 1048576.0,
                mem_avail_kb / 1024.0, rss_kb / 1024.0,
                consciousness.response_enabled() ? "true" : "false",
                consciousness.daydream_enabled() ? "true" : "false",
                consciousness.dreaming_enabled() ? "true" : "false",
                consciousness.llm_enabled() ? "true" : "false",
                consciousness.speech_enabled() ? "true" : "false",
                consciousness.thinking_enabled() ? "true" : "false",
                consciousness.action_enabled() ? "true" : "false",
                consciousness.temperature(), consciousness.top_k(),
                consciousness.top_p(),
                consciousness.speech_max_tokens(),
                consciousness.thinking_max_tokens(),
                consciousness.speech_cfg().sampling.temperature,
                consciousness.speech_cfg().sampling.top_k,
                consciousness.speech_cfg().sampling.top_p,
                consciousness.speech_cfg().max_tokens,
                consciousness.thinking_cfg().sampling.temperature,
                consciousness.thinking_cfg().sampling.top_k,
                consciousness.thinking_cfg().sampling.top_p,
                consciousness.thinking_cfg().max_tokens,
                consciousness.action_cfg().sampling.temperature,
                consciousness.action_cfg().sampling.top_k,
                consciousness.action_cfg().sampling.top_p,
                consciousness.action_cfg().max_tokens);
            server.send_text(fd, json);

            // Send prompt defaults as a separate message (prompts may
            // contain characters that break snprintf-assembled JSON)
            std::string pj = R"({"type":"consciousness_prompts",)"
                R"("identity":")" + json_escape(consciousness.identity_prompt()) + R"(",)"
                R"("speech":")" + json_escape(consciousness.speech_cfg().prompt) + R"(",)"
                R"("thinking":")" + json_escape(consciousness.thinking_cfg().prompt) + R"(",)"
                R"("action":")" + json_escape(consciousness.action_cfg().prompt) + R"("})";
            server.send_text(fd, pj);
        } else {
            server.send_text(fd, R"({"type":"consciousness_state","llm_loaded":false})");
        }
    });

    server.set_on_disconnect([&](int fd) {
        printf("[test-ws] WS client disconnected (fd=%d)\n", fd);
    });

    server.set_on_text([&](int fd, const std::string& msg) {
        if (msg == "loopback:on") {
            loopback.store(true, std::memory_order_relaxed);
            server.send_text(fd, R"({"type":"loopback","enabled":true})");
            printf("[test-ws] Loopback ON (fd=%d)\n", fd);
        } else if (msg == "loopback:off") {
            loopback.store(false, std::memory_order_relaxed);
            server.send_text(fd, R"({"type":"loopback","enabled":false})");
            printf("[test-ws] Loopback OFF (fd=%d)\n", fd);
        } else if (msg.rfind("vad_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 14, nullptr);
            audio.set_vad_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"vad_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] VAD threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("gain:", 0) == 0) {
            float g = std::strtof(msg.c_str() + 5, nullptr);
            if (g < 0.1f) g = 0.1f;
            if (g > 20.0f) g = 20.0f;
            audio.set_gain(g);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"gain","value":%.1f})", g);
            server.send_text(fd, json);
            printf("[test-ws] Gain = %.1fx (fd=%d)\n", g, fd);
        } else if (msg.rfind("silero_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 17, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_silero_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"silero_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] Silero threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("fsmn_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 15, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_fsmn_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"fsmn_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] FSMN threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg == "silero_enable:on" || msg == "silero_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_silero_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"silero_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] Silero %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg == "frcrn_enable:on" || msg == "frcrn_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_frcrn_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"frcrn_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] FRCRN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg == "fsmn_enable:on" || msg == "fsmn_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_fsmn_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"fsmn_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] FSMN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg.rfind("vad_source:", 0) == 0) {
            auto val = msg.substr(11);
            VadSource src = VadSource::ANY;
            if (val == "silero") src = VadSource::SILERO;
            else if (val == "fsmn") src = VadSource::FSMN;
            else src = VadSource::ANY;
            audio.set_vad_source(src);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"vad_source","value":%d})", static_cast<int>(src));
            server.send_text(fd, json);
            printf("[test-ws] VAD source = %s (%d) (fd=%d)\n", val.c_str(), static_cast<int>(src), fd);
        } else if (msg == "speaker_enable:on" || msg == "speaker_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_speaker_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"speaker_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] Speaker %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg.rfind("speaker_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 18, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_speaker_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"speaker_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] Speaker threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg == "speaker_clear") {
            audio.clear_speaker_db();
            server.send_text(fd, R"({"type":"speaker_clear"})");
            printf("[test-ws] Speaker (CAM++) DB cleared (fd=%d)\n", fd);
        } else if (msg.rfind("speaker_name:", 0) == 0) {
            // Format: speaker_name:ID:Name
            auto rest = msg.substr(13);
            auto colon = rest.find(':');
            if (colon != std::string::npos) {
                int id = std::stoi(rest.substr(0, colon));
                std::string name = rest.substr(colon + 1);
                audio.set_speaker_name(id, name);
                char json[256];
                snprintf(json, sizeof(json),
                    R"({"type":"speaker_name","id":%d,"name":"%s"})", id, name.c_str());
                server.send_text(fd, json);
                printf("[test-ws] CAM++ Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
            }
        } else if (msg == "wlecapa_enable:on" || msg == "wlecapa_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_wlecapa_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"wlecapa_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] WL-ECAPA %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg.rfind("wlecapa_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 18, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_wlecapa_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"wlecapa_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] WL-ECAPA threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("wlecapa_margin:", 0) == 0) {
            float m = std::strtof(msg.c_str() + 15, nullptr);
            if (m < 0.0f) m = 0.0f;
            if (m > 0.5f) m = 0.5f;
            audio.wlecapa_db().set_min_margin(m);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"wlecapa_margin","value":%.2f})", m);
            server.send_text(fd, json);
            printf("[test-ws] WL-ECAPA margin = %.2f (fd=%d)\n", m, fd);
        } else if (msg.rfind("early_trigger:", 0) == 0) {
            float s = std::strtof(msg.c_str() + 14, nullptr);
            if (s < 0.5f) s = 0.5f;
            if (s > 5.0f) s = 5.0f;
            audio.set_early_trigger_sec(s);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"early_trigger","value":%.2f})", s);
            server.send_text(fd, json);
            printf("[test-ws] Early trigger = %.2fs (fd=%d)\n", s, fd);
        } else if (msg == "early_enable:on" || msg == "early_enable:off") {
            bool en = (msg == "early_enable:on");
            audio.set_early_trigger_enabled(en);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"early_enable","value":%s})", en ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] Early trigger %s (fd=%d)\n", en ? "enabled" : "disabled", fd);
        } else if (msg.rfind("min_speech:", 0) == 0) {
            float s = std::strtof(msg.c_str() + 11, nullptr);
            if (s < 0.5f) s = 0.5f;
            if (s > 10.0f) s = 10.0f;
            audio.set_min_speech_sec(s);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"min_speech","value":%.2f})", s);
            server.send_text(fd, json);
            printf("[test-ws] Min speech = %.2fs (fd=%d)\n", s, fd);
        } else if (msg == "wlecapa_clear") {
            audio.clear_wlecapa_db();
            server.send_text(fd, R"({"type":"wlecapa_clear"})");
            printf("[test-ws] WL-ECAPA DB cleared (fd=%d)\n", fd);
        } else if (msg.rfind("wlecapa_name:", 0) == 0) {
            auto rest = msg.substr(13);
            auto colon = rest.find(':');
            if (colon != std::string::npos) {
                int id = std::stoi(rest.substr(0, colon));
                std::string name = rest.substr(colon + 1);
                audio.set_wlecapa_name(id, name);
                char json[256];
                snprintf(json, sizeof(json),
                    R"({"type":"wlecapa_name","id":%d,"name":"%s"})", id, name.c_str());
                server.send_text(fd, json);
                printf("[test-ws] WL-ECAPA Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
            }
        } else if (msg.rfind("wlecapa_delete:", 0) == 0) {
            // Format: wlecapa_delete:ID
            int id = std::stoi(msg.substr(15));
            bool ok = audio.remove_wlecapa_speaker(id);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"wlecapa_delete","id":%d,"ok":%s})", id, ok ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] WL-ECAPA delete #%d %s (fd=%d)\n", id, ok ? "OK" : "FAIL", fd);
        } else if (msg.rfind("wlecapa_merge:", 0) == 0) {
            // Format: wlecapa_merge:DST_ID:SRC_ID
            auto rest = msg.substr(14);
            auto colon = rest.find(':');
            if (colon != std::string::npos) {
                int dst = std::stoi(rest.substr(0, colon));
                int src = std::stoi(rest.substr(colon + 1));
                bool ok = audio.merge_wlecapa_speakers(dst, src);
                char json[128];
                snprintf(json, sizeof(json),
                    R"({"type":"wlecapa_merge","dst":%d,"src":%d,"ok":%s})",
                    dst, src, ok ? "true" : "false");
                server.send_text(fd, json);
                printf("[test-ws] WL-ECAPA merge #%d←#%d %s (fd=%d)\n", dst, src, ok ? "OK" : "FAIL", fd);
            }
        // ── SpeakerTracker controls ──
        } else if (msg == "tracker_enable:on" || msg == "tracker_enable:off") {
            bool on = msg.back() == 'n';
            audio.tracker().set_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"tracker_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] Tracker %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg.rfind("tracker_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 18, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.tracker().set_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"tracker_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] Tracker threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("tracker_change_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 24, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.tracker().set_change_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"tracker_change_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] Tracker change threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("tracker_interval:", 0) == 0) {
            int ms = std::stoi(msg.substr(17));
            if (ms < 250) ms = 250;
            if (ms > 5000) ms = 5000;
            audio.tracker().set_interval_ms(ms);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"tracker_interval","value":%d})", ms);
            server.send_text(fd, json);
            printf("[test-ws] Tracker interval = %d ms (fd=%d)\n", ms, fd);
        } else if (msg.rfind("tracker_window:", 0) == 0) {
            int ms = std::stoi(msg.substr(15));
            if (ms < 500) ms = 500;
            if (ms > 5000) ms = 5000;
            audio.tracker().set_window_ms(ms);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"tracker_window","value":%d})", ms);
            server.send_text(fd, json);
            printf("[test-ws] Tracker window = %d ms (fd=%d)\n", ms, fd);
        } else if (msg == "tracker_clear") {
            audio.tracker().clear();
            server.send_text(fd, R"({"type":"tracker_clear"})");
            printf("[test-ws] Tracker DB cleared (fd=%d)\n", fd);
        } else if (msg.rfind("tracker_name:", 0) == 0) {
            auto rest = msg.substr(13);
            auto colon = rest.find(':');
            if (colon != std::string::npos) {
                int id = std::stoi(rest.substr(0, colon));
                std::string name = rest.substr(colon + 1);
                audio.tracker().set_speaker_name(id, name);
                char json[256];
                snprintf(json, sizeof(json),
                    R"({"type":"tracker_name","id":%d,"name":"%s"})", id, name.c_str());
                server.send_text(fd, json);
                printf("[test-ws] Tracker Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
            }
        } else if (msg == "asr_enable:on" || msg == "asr_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_asr_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"asr_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] ASR %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg.rfind("asr_vad_source:", 0) == 0) {
            auto val = msg.substr(15);
            VadSource src = VadSource::ANY;
            if (val == "silero") src = VadSource::SILERO;
            else if (val == "fsmn") src = VadSource::FSMN;
            else if (val == "direct") src = VadSource::DIRECT;
            else src = VadSource::ANY;
            audio.set_asr_vad_source(src);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"asr_vad_source","value":%d})", static_cast<int>(src));
            server.send_text(fd, json);
            printf("[test-ws] ASR VAD source = %s (%d) (fd=%d)\n", val.c_str(), static_cast<int>(src), fd);
        } else if (msg.rfind("asr_param:", 0) == 0) {
            // Generic ASR parameter setter: "asr_param:<key>:<value>"
            auto rest = msg.substr(10);
            auto sep = rest.find(':');
            if (sep != std::string::npos) {
                auto key = rest.substr(0, sep);
                auto val = rest.substr(sep + 1);
                char json[256];
                if (key == "post_silence_ms") {
                    int v = std::stoi(val);
                    audio.set_asr_post_silence_ms(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"post_silence_ms","value":%d})",
                        audio.asr_post_silence_ms());
                } else if (key == "max_buf_sec") {
                    float v = std::stof(val);
                    audio.set_asr_max_buf_sec(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"max_buf_sec","value":%.1f})",
                        audio.asr_max_buf_sec());
                } else if (key == "min_dur_sec") {
                    float v = std::stof(val);
                    audio.set_asr_min_dur_sec(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"min_dur_sec","value":%.2f})",
                        audio.asr_min_dur_sec());
                } else if (key == "pre_roll_sec") {
                    float v = std::stof(val);
                    audio.set_asr_pre_roll_sec(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"pre_roll_sec","value":%.2f})",
                        audio.asr_pre_roll_sec());
                } else if (key == "max_tokens") {
                    int v = std::stoi(val);
                    audio.set_asr_max_tokens(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"max_tokens","value":%d})",
                        audio.asr_max_tokens());
                } else if (key == "rep_penalty") {
                    float v = std::stof(val);
                    audio.set_asr_rep_penalty(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"rep_penalty","value":%.2f})",
                        audio.asr_rep_penalty());
                } else if (key == "min_energy") {
                    float v = std::stof(val);
                    audio.set_asr_min_energy(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"min_energy","value":%.4f})",
                        audio.asr_min_energy());
                } else if (key == "partial_sec") {
                    float v = std::stof(val);
                    audio.set_asr_partial_sec(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"partial_sec","value":%.1f})",
                        audio.asr_partial_sec());
                } else if (key == "speech_ratio") {
                    float v = std::stof(val);
                    audio.set_asr_min_speech_ratio(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"speech_ratio","value":%.2f})",
                        audio.asr_min_speech_ratio());
                } else if (key == "adaptive_silence") {
                    bool v = (val == "true" || val == "1" || val == "on");
                    audio.set_asr_adaptive_silence(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"adaptive_silence","value":%s})",
                        audio.asr_adaptive_silence() ? "true" : "false");
                } else if (key == "adaptive_short_ms") {
                    int v = std::stoi(val);
                    audio.set_asr_adaptive_short_ms(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"adaptive_short_ms","value":%d})",
                        audio.asr_adaptive_short_ms());
                } else if (key == "adaptive_long_ms") {
                    int v = std::stoi(val);
                    audio.set_asr_adaptive_long_ms(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"adaptive_long_ms","value":%d})",
                        audio.asr_adaptive_long_ms());
                } else if (key == "adaptive_vlong_ms") {
                    int v = std::stoi(val);
                    audio.set_asr_adaptive_vlong_ms(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"adaptive_vlong_ms","value":%d})",
                        audio.asr_adaptive_vlong_ms());
                } else {
                    snprintf(json, sizeof(json),
                        R"({"type":"asr_param","key":"%s","error":"unknown"})",
                        key.c_str());
                }
                server.send_text(fd, json);
                printf("[test-ws] ASR param %s=%s (fd=%d)\n",
                       key.c_str(), val.c_str(), fd);
            }
        } else if (msg.rfind("consciousness_enable:", 0) == 0 && llm_loaded) {
            // Format: consciousness_enable:<mode>:<on|off>
            auto rest = msg.substr(21);
            auto sep = rest.find(':');
            if (sep != std::string::npos) {
                auto mode = rest.substr(0, sep);
                bool on = rest.substr(sep + 1) == "on";
                if (mode == "response") consciousness.set_response_enabled(on);
                else if (mode == "daydream") consciousness.set_daydream_enabled(on);
                else if (mode == "dreaming") consciousness.set_dreaming_enabled(on);
                else if (mode == "llm") consciousness.set_llm_enabled(on);
                else if (mode == "speech") consciousness.set_speech_enabled(on);
                else if (mode == "thinking") consciousness.set_thinking_enabled(on);
                else if (mode == "action") consciousness.set_action_enabled(on);
                char json[128];
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_enable","mode":"%s","enabled":%s})",
                    mode.c_str(), on ? "true" : "false");
                server.send_text(fd, json);
                printf("[test-ws] Consciousness %s %s (fd=%d)\n",
                       mode.c_str(), on ? "ON" : "OFF", fd);
            }
        } else if (msg.rfind("consciousness_param:", 0) == 0 && llm_loaded) {
            // Format: consciousness_param:<key>:<value>
            // Key can be global (e.g. "temperature") or per-pipeline
            // (e.g. "speech.temperature", "thinking.top_k")
            auto rest = msg.substr(20);
            auto sep = rest.find(':');
            if (sep != std::string::npos) {
                auto key = rest.substr(0, sep);
                auto val = rest.substr(sep + 1);
                char json[256];

                // Determine pipeline target
                std::string pipeline;
                std::string param_name = key;
                auto dot = key.find('.');
                if (dot != std::string::npos) {
                    pipeline = key.substr(0, dot);
                    param_name = key.substr(dot + 1);
                }

                // Helper to get target pipeline config
                PipelineConfig* pcfg = nullptr;
                if (pipeline == "speech") pcfg = &consciousness.speech_cfg();
                else if (pipeline == "thinking") pcfg = &consciousness.thinking_cfg();
                else if (pipeline == "action") pcfg = &consciousness.action_cfg();

                if (param_name == "temperature") {
                    float v = std::stof(val);
                    if (v < 0.0f) v = 0.0f;
                    if (v > 2.0f) v = 2.0f;
                    if (pcfg) pcfg->sampling.temperature = v;
                    else consciousness.set_temperature(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"consciousness_param","key":"%s","value":%.2f})",
                        key.c_str(), v);
                } else if (param_name == "top_k") {
                    int v = std::stoi(val);
                    if (v < 1) v = 1;
                    if (v > 200) v = 200;
                    if (pcfg) pcfg->sampling.top_k = v;
                    else consciousness.set_top_k(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"consciousness_param","key":"%s","value":%d})",
                        key.c_str(), v);
                } else if (param_name == "top_p") {
                    float v = std::stof(val);
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    if (pcfg) pcfg->sampling.top_p = v;
                    else consciousness.set_top_p(v);
                    snprintf(json, sizeof(json),
                        R"({"type":"consciousness_param","key":"%s","value":%.2f})",
                        key.c_str(), v);
                } else if (param_name == "max_tokens") {
                    int v = std::stoi(val);
                    if (v < 10) v = 10;
                    if (v > 4096) v = 4096;
                    if (pcfg) pcfg->max_tokens = v;
                    snprintf(json, sizeof(json),
                        R"({"type":"consciousness_param","key":"%s","value":%d})",
                        key.c_str(), v);
                } else {
                    snprintf(json, sizeof(json),
                        R"({"type":"consciousness_param","key":"%s","error":"unknown"})",
                        key.c_str());
                }
                server.send_text(fd, json);
                printf("[test-ws] Consciousness param %s=%s (fd=%d)\n",
                       key.c_str(), val.c_str(), fd);
            }
        } else if (msg.rfind("consciousness_prompt:", 0) == 0 && llm_loaded) {
            // Format: consciousness_prompt:<pipeline>:<text>
            // pipeline: "identity" (system prompt), "speech", "thinking", "action"
            auto rest = msg.substr(21);
            auto sep = rest.find(':');
            if (sep != std::string::npos) {
                auto pipeline = rest.substr(0, sep);
                auto text = rest.substr(sep + 1);
                if (pipeline == "identity") {
                    consciousness.set_identity_prompt(text);
                } else if (pipeline == "speech") {
                    consciousness.speech_cfg().prompt = text;
                    consciousness.recompose_system_prompt();
                } else if (pipeline == "thinking") {
                    consciousness.thinking_cfg().prompt = text;
                    consciousness.recompose_system_prompt();
                } else if (pipeline == "action") {
                    consciousness.action_cfg().prompt = text;
                    consciousness.recompose_system_prompt();
                }
                char json[256];
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_prompt","pipeline":"%s","ok":true})",
                    pipeline.c_str());
                server.send_text(fd, json);
                printf("[test-ws] %s prompt updated (%zu chars, fd=%d)\n",
                       pipeline.c_str(), text.size(), fd);
            } else {
                // Legacy: no pipeline prefix → identity prompt
                consciousness.set_identity_prompt(rest);
                server.send_text(fd, R"({"type":"consciousness_prompt","pipeline":"identity","ok":true})");
                printf("[test-ws] System prompt updated (%zu chars, fd=%d)\n",
                       rest.size(), fd);
            }
        } else if (msg.rfind("text_input:", 0) == 0 && llm_loaded) {
            // Inject text into consciousness stream
            std::string text = msg.substr(11);
            if (!text.empty()) {
                InputItem item;
                item.source = InputSource::TEXT;
                item.text = text;
                item.priority = 1.0f;
                consciousness.inject_input(std::move(item));
                // Echo back confirmation
                char json[256];
                snprintf(json, sizeof(json),
                    R"({"type":"text_input_ack","ok":true})");
                server.send_text(fd, json);
                printf("[test-ws] Text input injected: \"%s\" (fd=%d)\n",
                       text.c_str(), fd);
            }
        } else {
            printf("[test-ws] Text from fd=%d: %s\n", fd, msg.c_str());
        }
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
    printf("[test-ws] Default VAD policy: source=silero, silero=ON, fsmn=OFF, gain=4.0, silero_threshold=0.001\n");

    // Start audio pipeline.
    if (!audio.start(audio_cfg)) {
        fprintf(stderr, "[test-ws] Failed to start audio pipeline\n");
        return 1;
    }

    // Start WS server.
    if (!server.start(ws_cfg)) {
        fprintf(stderr, "[test-ws] Failed to start WS server\n");
        audio.stop();
        return 1;
    }

    printf("[test-ws] Server running on http://localhost:%d\n", ws_cfg.port);
    printf("[test-ws] Audio pipeline: Mel(n_fft=%d hop=%d mels=%d) + VAD\n",
           audio_cfg.mel.n_fft, audio_cfg.mel.hop_length, audio_cfg.mel.n_mels);

    // Start consciousness stream (after server is running so callbacks work)
    if (llm_loaded) {
        consciousness.start();
        printf("[test-ws] Consciousness stream running (entity=%s)\n",
               persona_cfg.name.c_str());
    }

    printf("[test-ws] Press Ctrl+C to stop...\n");

    // Block until SIGINT/SIGTERM.
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &mask, nullptr);
    int sig = 0;
    sigwait(&mask, &sig);
    printf("\n[test-ws] Caught signal %d, shutting down...\n", sig);

    // Stop consciousness first (it depends on model/cache)
    if (llm_loaded) {
        consciousness.stop();
        printf("[test-ws] Consciousness stream stopped\n");
    }

    audio.stop();
    server.stop();
    timeline.close();
    printf("[test-ws] Timeline log closed: %s\n", timeline.path().c_str());
    printf("[test-ws] Total: %lu WS frames, %.1f KB\n",
           total_frames.load(), total_bytes.load() / 1024.0);

    // Cleanup LLM resources
    if (llm_loaded) {
        llm_cache.destroy();
        llm_state.free();
        free_model_weights(llm_weights);
        printf("[test-ws] LLM resources released\n");
    }

    return 0;
}

} // namespace deusridet
