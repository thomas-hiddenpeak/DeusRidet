/**
 * @file audio_pipeline.cpp
 * @philosophical_role End-to-end auditory pipeline: PCM ingress -> VAD -> enhancement -> ASR -> diarisation -> publish. Perception shapes consciousness; this file is the perceptual spine.
 * @serves ConscientiaStream input queue, Nexus WS ingress, Orator, Somnium recall inputs.
 */
// audio_pipeline.cpp — Real-time audio processing pipeline implementation.
//
// Processing thread: pull PCM from ring buffer → push to MelSpectrogram (GPU)
// → VAD on computed frames → report stats/events via callbacks.

#include "audio_pipeline.h"
#include "separatio_orator_probe.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"
#include "../../orator/spectral_cluster.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace deusridet {

namespace {

bool env_truthy_local(const char* name) {
    const char* value = std::getenv(name);
    return value &&
        (std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "yes") == 0);
}

bool fusion_shadow_enabled() {
    return env_truthy_local("DEUSRIDET_AUDITUS_FUSION_SHADOW");
}

bool fusion_canary_enabled() {
    return env_truthy_local("DEUSRIDET_AUDITUS_FUSION_CANARY");
}

std::string escape_json_local(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char value : input) {
        if (value == '"') out += "\\\"";
        else if (value == '\\') out += "\\\\";
        else if (value == '\n') out += "\\n";
        else if (value == '\r') out += "\\r";
        else if (value == '\t') out += "\\t";
        else out += value;
    }
    return out;
}

float env_float_local(const char* name, float fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    float parsed = std::strtof(value, &end);
    return end != value ? parsed : fallback;
}

int env_int_local(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return end != value ? (int)parsed : fallback;
}

}  // namespace

AudioPipeline::AudioPipeline() = default;

AudioPipeline::~AudioPipeline() {
    stop();
    delete ring_;
}

bool AudioPipeline::start(const AudioPipelineConfig& cfg) {
    if (running_.load()) return false;
    cfg_ = cfg;

    // Initialize ring buffer.
    ring_ = new RingBuffer(cfg_.ring_buffer_bytes);

    // Initialize Mel spectrogram.
    if (!mel_.init(cfg_.mel)) {
        LOG_ERROR("AudioPipe", "Failed to init Mel spectrogram");
        return false;
    }

    // Initialize FRCRN speech enhancement (optional — non-fatal).
    if (!cfg_.frcrn.weights_dir.empty() && cfg_.frcrn.enabled) {
        if (!frcrn_.init(cfg_.frcrn)) {
            LOG_WARN("AudioPipe", "FRCRN speech enhancement init failed — running without denoising");
        }
    }

    // Initialize P1: overlap detector (optional — non-fatal).
    if (!cfg_.overlap_det.model_path.empty() && cfg_.overlap_det.enabled) {
        if (!overlap_det_.init(cfg_.overlap_det)) {
            LOG_WARN("AudioPipe", "Overlap detector init failed — using heuristic fallback");
        }
    }

    // Initialize P2: speech separator (optional — lazy loaded).
    if (!cfg_.separator.model_path.empty()) {
        if (!separator_.init(cfg_.separator)) {
            LOG_WARN("AudioPipe", "Speech separator init failed");
        }
    }

    // Initialize Silero VAD (optional — non-fatal if model not found).
    if (!cfg_.silero.model_path.empty()) {
        if (!silero_.init(cfg_.silero)) {
            LOG_WARN("AudioPipe", "Silero VAD init failed — speaker/ASR pipelines will lack VAD gating");
        }
    }

    // Initialize speaker encoder (optional — non-fatal).
    bool need_fbank = false;
    if (!cfg_.speaker.model_path.empty()) {
        if (!speaker_enc_.init(cfg_.speaker)) {
            LOG_WARN("AudioPipe", "Speaker encoder init failed");
        } else {
            need_fbank = true;
        }
    }

    if (need_fbank) {
        // Initialize 80-dim fbank (shared between CAM++ and ECAPA-TDNN).
        // Povey window + [-1,1] PCM normalization — matches WeSpeaker/Kaldi defaults.
        if (!speaker_fbank_.init(80, 400, 160, 512, 16000,
                                 FbankWindowType::POVEY, true)) {
            LOG_WARN("AudioPipe", "Speaker fbank init failed");
        }
    }

    // Initialize WavLM-Large + ECAPA-TDNN native GPU speaker encoder (optional).
    if (!cfg_.wavlm_ecapa_model.empty()) {
        if (!wlecapa_enc_.init(cfg_.wavlm_ecapa_model)) {
            LOG_WARN("AudioPipe", "WavLM-ECAPA native GPU init failed");
        } else {
            LOG_INFO("AudioPipe", "WavLM-ECAPA native GPU encoder ready (192-dim)");
            // Enable dual-encoder matching (384D = CAM++ 192D + WL-ECAPA 192D).
            use_dual_encoder_ = true;
            LOG_INFO("AudioPipe", "Dual-encoder matching enabled (384D)");
        }
    }

    speaker_threshold_.store(cfg_.speaker_threshold, std::memory_order_relaxed);
    // Pending-pool confirmation threshold — tunable via configs/auditus.conf
    // (key: speaker_register_threshold). See AudioPipelineConfig for rationale.
    speaker_register_threshold_.store(cfg_.speaker_register_threshold,
                                      std::memory_order_relaxed);

    // v24: threshold set from config (default 0.50 in header, 0.45 in machina.conf).
    // Recency bonus (-0.05) applies dynamically during FULL identification.
    // No threshold override needed — dual encoder mode uses cfg value directly.
    wlecapa_threshold_.store(cfg_.wavlm_ecapa_threshold, std::memory_order_relaxed);

    // Initialize ASR engine (optional — non-fatal, but heavy: ~4.7 GB weights).
    if (!cfg_.asr_model_path.empty()) {
        asr_engine_ = std::make_unique<asr::ASREngine>();
        asr_engine_->load_model(cfg_.asr_model_path);
        if (asr_engine_->is_loaded()) {
            LOG_INFO("AudioPipe", "Qwen3-ASR engine loaded from %s", cfg_.asr_model_path.c_str());
        } else {
            LOG_WARN("AudioPipe", "Qwen3-ASR engine failed to load — ASR disabled");
            asr_engine_.reset();
        }
    }

    // Reset stats.
    memset(&stats_, 0, sizeof(stats_));

    // Register the AUDIO business-clock anchor (T1 = sample index at this
    // pipeline's input, resolution 1/sample_rate). T0 advances wall-clock;
    // T1 advances one unit per PCM sample. period_ns = 1e9 / sample_rate
    // for real-time capture (62500 ns @ 16 kHz). Under accelerated replay
    // (cfg_.replay_speed > 1.0) we scale the period so that T0 tracks wall
    // time: 1 wall-second of elapsed time corresponds to replay_speed
    // seconds of source audio, i.e. replay_speed * sample_rate samples.
    double effective_rate = (double)cfg_.mel.sample_rate * (double)cfg_.replay_speed;
    const uint64_t audio_period_ns =
        effective_rate > 0.0
            ? (uint64_t)llround(1e9 / effective_rate)
            : 62500ULL;
    tempus::anchor_register(tempus::Domain::AUDIO,
                            tempus::now_t0_ns(),
                            /*t1_zero=*/0,
                            /*period_ns=*/audio_period_ns);
    LOG_INFO("AudioPipe",
             "Tempus anchor registered: domain=AUDIO period=%lu ns (sr=%d, replay_speed=%.2f)",
             audio_period_ns, cfg_.mel.sample_rate, (double)cfg_.replay_speed);

    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&AudioPipeline::process_loop, this);

    // Start ASR worker thread if engine loaded.
    if (asr_engine_ && asr_engine_->is_loaded()) {
        asr_thread_ = std::thread(&AudioPipeline::asr_loop, this);
    }

    LOG_INFO("AudioPipe", "Started (ring=%zu KB, chunk=%d ms)",
             cfg_.ring_buffer_bytes / 1024, cfg_.process_chunk_ms);
    return true;
}

void AudioPipeline::stop() {
    if (!running_.load()) return;
    running_.store(false, std::memory_order_release);
    // Wake ASR thread so it can exit.
    asr_cv_.notify_all();
    if (asr_thread_.joinable()) asr_thread_.join();
    if (thread_.joinable()) thread_.join();
    LOG_INFO("AudioPipe", "Stopped (total: %lu samples, %lu mel frames, %lu speech)",
             stats_.audio_t1_processed, stats_.mel_frames, stats_.speech_frames);
}

void AudioPipeline::set_asr_rep_penalty(float p) {
    asr_rep_penalty_.store(p, std::memory_order_relaxed);
    if (asr_engine_) asr_engine_->set_repetition_penalty(p);
}

// ASR worker thread — picks up jobs from queue, runs transcription off-main-loop.
void AudioPipeline::asr_loop() {
    LOG_INFO("AudioPipe", "ASR worker thread started");
    std::unique_ptr<SeparatioOratorProbe> shadow_speaker;
    while (true) {
        ASRJob job;
        {
            std::unique_lock<std::mutex> lock(asr_mutex_);
            asr_cv_.wait(lock, [this] {
                return !asr_queue_.empty() || !running_.load(std::memory_order_relaxed);
            });
            if (!running_.load(std::memory_order_relaxed) && asr_queue_.empty())
                break;
            if (asr_queue_.empty()) continue;
            job = std::move(asr_queue_.front());
            asr_queue_.pop();
        }

        asr_busy_.store(true, std::memory_order_relaxed);

        // Dynamic max_tokens: scale with audio duration (~8 tokens/sec).
        // Adapted from qwen35-thor (voice_session.cpp): min(configured, max(40, dur*8))
        int max_tok_cfg = asr_max_tokens_.load(std::memory_order_relaxed);
        int dynamic_tok = std::max(40, (int)(job.audio_duration_sec * 8.0f));
        int max_tok = std::min(max_tok_cfg, dynamic_tok);
        auto result = asr_engine_->transcribe(
            job.pcm_f32.data(), (int)job.pcm_f32.size(), 16000, max_tok);

        asr_busy_.store(false, std::memory_order_relaxed);

        // Handle streaming partial vs final transcript differently.
        if (job.is_partial) {
            if (!result.text.empty()) {
                LOG_INFO("AudioPipe", "ASR(partial): \"%s\" (%.1fms, %.2fs audio)",
                         result.text.c_str(), result.total_ms, job.audio_duration_sec);
                if (on_asr_partial_) on_asr_partial_(result.text, job.audio_duration_sec);
            }
            // Log partial to ASR log panel for observability.
            if (on_asr_log_) {
                char json[1024];
                snprintf(json, sizeof(json),
                    R"({"stage":"partial","audio_sec":%.2f,"total_ms":%.1f,"tokens":%d,"text":"%s"})",
                    job.audio_duration_sec, result.total_ms, result.token_count,
                    escape_json_local(result.text).c_str());
                on_asr_log_(json);
            }
            continue;
        }

        stats_.asr_active = true;
        stats_.asr_latency_ms = result.total_ms;
        stats_.asr_audio_duration_s = job.audio_duration_sec;

        if (!result.text.empty()) {
            LOG_INFO("AudioPipe", "ASR: \"%s\" (%.1fms, %.2fs audio, mel=%.0fms enc=%.0fms dec=%.0fms %dtok)",
                     result.text.c_str(), result.total_ms, job.audio_duration_sec,
                     result.mel_ms, result.encoder_ms, result.decode_ms, result.token_count);
            if (on_transcript_) on_transcript_(result, job.audio_duration_sec,
                                                 job.speaker_id, job.speaker_name,
                                                 job.speaker_sim, job.speaker_confidence,
                                                 job.speaker_source, job.trigger_reason,
                                                 job.stream_start_sec, job.stream_end_sec);
        } else {
            LOG_INFO("AudioPipe", "ASR: (empty) (%.1fms, %.2fs audio)",
                     result.total_ms, job.audio_duration_sec);
        }

        // Send detailed ASR log for WebUI debug panel.
        if (on_asr_log_) {
            char json[2048];
            snprintf(json, sizeof(json),
                R"({"stage":"result","trigger":"%s","audio_sec":%.2f,)"
                R"("mel_ms":%.1f,"mel_frames":%d,"encoder_ms":%.1f,"encoder_out":%d,)"
                R"("decode_ms":%.1f,"tokens":%d,"postprocess_ms":%.1f,"total_ms":%.1f,)"
                R"("raw_text":"%s","text":"%s"})",
                job.trigger_reason.c_str(), job.audio_duration_sec,
                result.mel_ms, result.mel_frames, result.encoder_ms, result.encoder_out_len,
                result.decode_ms, result.token_count, result.postprocess_ms, result.total_ms,
                escape_json_local(result.raw_text).c_str(), escape_json_local(result.text).c_str());
            on_asr_log_(json);
        }

        if (fusion_shadow_enabled() && on_asr_log_) {
            auto shadow_start = std::chrono::high_resolution_clock::now();
            std::string detail;
            if (!separator_enabled() || !separator_.initialized()) {
                char json[768];
                snprintf(json, sizeof(json),
                    R"({"stage":"fusion_shadow","enabled":true,"valid":false,)"
                    R"("reason":"separator_unavailable","trigger":"%s","audio_sec":%.2f,)"
                    R"("stream_start_sec":%.2f,"stream_end_sec":%.2f,"mix_text":"%s"})",
                    job.trigger_reason.c_str(), job.audio_duration_sec,
                    job.stream_start_sec, job.stream_end_sec,
                    escape_json_local(result.text).c_str());
                detail = json;
            } else {
                asr_busy_.store(true, std::memory_order_relaxed);
                auto sep_start = std::chrono::high_resolution_clock::now();
                SeparationResult separated = separator_.separate(
                    job.pcm_f32.data(), (int)job.pcm_f32.size());
                auto sep_end = std::chrono::high_resolution_clock::now();
                float sep_ms = std::chrono::duration<float, std::milli>(sep_end - sep_start).count();

                asr::ASRResult src1_result;
                asr::ASRResult src2_result;
                ShadowSpeakerEvidence src1_speaker;
                ShadowSpeakerEvidence src2_speaker;
                if (separated.valid) {
                    int source_max_tok = std::min(
                        asr_max_tokens_.load(std::memory_order_relaxed),
                        std::max(20, (int)(job.audio_duration_sec * 8.0f)));
                    src1_result = asr_engine_->transcribe(
                        separated.source1.data(), (int)separated.source1.size(), 16000, source_max_tok);
                    src2_result = asr_engine_->transcribe(
                        separated.source2.data(), (int)separated.source2.size(), 16000, source_max_tok);
                    if (!shadow_speaker) {
                        shadow_speaker = std::make_unique<SeparatioOratorProbe>();
                        shadow_speaker->init(cfg_, use_dual_encoder_);
                    }
                    float spk_threshold = env_float_local(
                        "DEUSRIDET_AUDITUS_FUSION_SPK_THRESHOLD", 0.35f);
                    float spk_min_margin = env_float_local(
                        "DEUSRIDET_AUDITUS_FUSION_SPK_MIN_MARGIN", 0.055f);
                    int stable_min_exemplars = env_int_local(
                        "DEUSRIDET_AUDITUS_FUSION_STABLE_MIN_EXEMPLARS", 1);
                    int stable_min_matches = env_int_local(
                        "DEUSRIDET_AUDITUS_FUSION_STABLE_MIN_MATCHES", 2);
                    SpeakerVectorStore& shadow_db = use_dual_encoder_ ? dual_db_ : campp_db_;
                    WavLMEcapaEncoder* shadow_wavlm = use_dual_encoder_ ? &wlecapa_enc_ : nullptr;
                    src1_speaker = shadow_speaker->score(
                        separated.source1.data(), (int)separated.source1.size(),
                        shadow_db, shadow_wavlm, spk_threshold, spk_min_margin,
                        stable_min_exemplars, stable_min_matches);
                    src2_speaker = shadow_speaker->score(
                        separated.source2.data(), (int)separated.source2.size(),
                        shadow_db, shadow_wavlm, spk_threshold, spk_min_margin,
                        stable_min_exemplars, stable_min_matches);
                }
                asr_busy_.store(false, std::memory_order_relaxed);

                auto shadow_end = std::chrono::high_resolution_clock::now();
                float total_ms = std::chrono::duration<float, std::milli>(shadow_end - shadow_start).count();
                char head[1536];
                snprintf(head, sizeof(head),
                    R"({"stage":"fusion_shadow","enabled":true,"valid":%s,)"
                    R"("trigger":"%s","audio_sec":%.2f,"stream_start_sec":%.2f,"stream_end_sec":%.2f,)"
                    R"("timeline_speaker_id":%d,"timeline_speaker_name":"%s",)"
                    R"("timeline_speaker_sim":%.3f,"timeline_speaker_confidence":%.3f,)"
                    R"("timeline_speaker_source":"%s","mix_text":"%s",)"
                    R"("sep_ms":%.1f,"shadow_total_ms":%.1f,"src1_rms":%.5f,"src2_rms":%.5f,)",
                    separated.valid ? "true" : "false",
                    job.trigger_reason.c_str(), job.audio_duration_sec,
                    job.stream_start_sec, job.stream_end_sec,
                    job.speaker_id,
                    escape_json_local(job.speaker_name).c_str(),
                    job.speaker_sim, job.speaker_confidence,
                    escape_json_local(job.speaker_source).c_str(),
                    escape_json_local(result.text).c_str(),
                    sep_ms, total_ms, separated.energy1, separated.energy2);
                detail = head;
                detail += "\"src1\":" + source_result_json(src1_result, src1_speaker) + ",";
                detail += "\"src2\":" + source_result_json(src2_result, src2_speaker) + ",";
                detail += "\"arbitrium\":" + fusion_arbitrium_json(
                    src1_result, src1_speaker, src2_result, src2_speaker,
                    job.speaker_id) + ",";
                detail += "\"ledger\":" + fusion_evidence_ledger_json(
                    src1_result, src1_speaker, src2_result, src2_speaker,
                    job.speaker_id, fusion_canary_enabled()) + "}";
            }
            on_asr_log_(detail);
        }
    }
    LOG_INFO("AudioPipe", "ASR worker thread exited");
}

void AudioPipeline::push_pcm(const int16_t* data, int n_samples) {
    if (!ring_ || n_samples <= 0) return;
    size_t bytes = n_samples * sizeof(int16_t);
    size_t written = ring_->push(reinterpret_cast<const uint8_t*>(data), bytes);
    int n_enqueued = (int)(written / sizeof(int16_t));
    uint64_t t1_end = 0;
    if (n_enqueued > 0) {
        // fetch_add returns the previous value; t1_end = prev + n_enqueued is
        // the exclusive end of the enqueued range.
        uint64_t t1_start = audio_t1_in_.fetch_add((uint64_t)n_enqueued,
                                                   std::memory_order_relaxed);
        t1_end = t1_start + (uint64_t)n_enqueued;
    } else {
        t1_end = audio_t1_in_.load(std::memory_order_relaxed);
    }
    if (written < bytes) {
        size_t dropped_bytes = bytes - written;
        size_t dropped_samples = dropped_bytes / sizeof(int16_t);
        // The dropped samples would have extended the T1 range from t1_end
        // to t1_end + dropped_samples, had there been room in the ring.
        // Record the gap at that range so the timeline can show it.
        LOG_WARN("AudioPipe", "Ring buffer overflow, dropped %zu bytes (%zu samples)",
                 dropped_bytes, dropped_samples);
        if (on_drop_) {
            on_drop_(t1_end, t1_end + (uint64_t)dropped_samples,
                     "ring_overflow", dropped_bytes);
        }
    }
}

} // namespace deusridet
