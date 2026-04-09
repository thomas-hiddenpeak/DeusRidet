// audio_pipeline.cpp — Real-time audio processing pipeline implementation.
//
// Processing thread: pull PCM from ring buffer → push to MelSpectrogram (GPU)
// → VAD on computed frames → report stats/events via callbacks.

#include "audio_pipeline.h"
#include "../../communis/log.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

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

    // Initialize VAD.
    vad_.init(cfg_.vad);

    // Initialize Silero VAD (optional — non-fatal if model not found).
    if (!cfg_.silero.model_path.empty()) {
        if (!silero_.init(cfg_.silero)) {
            LOG_WARN("AudioPipe", "Silero VAD init failed — running energy-only VAD");
        }
    }

    // Initialize FSMN VAD (optional — non-fatal).
    if (!cfg_.fsmn.model_path.empty()) {
        if (!fsmn_.init(cfg_.fsmn)) {
            LOG_WARN("AudioPipe", "FSMN VAD init failed");
        }
    }

    // Initialize TEN VAD (optional — non-fatal).
    if (!cfg_.ten.model_path.empty()) {
        if (!ten_.init(cfg_.ten)) {
            LOG_WARN("AudioPipe", "TEN VAD init failed");
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

    // ECAPA-TDNN also requires fbank features.
    if (!cfg_.unispeech.model_path.empty()) {
        need_fbank = true;
    }

    if (need_fbank) {
        // Initialize 80-dim fbank (shared between CAM++ and ECAPA-TDNN).
        // Povey window + [-1,1] PCM normalization — matches WeSpeaker/Kaldi defaults.
        if (!speaker_fbank_.init(80, 400, 160, 512, 16000,
                                 FbankWindowType::POVEY, true)) {
            LOG_WARN("AudioPipe", "Speaker fbank init failed");
        }
    }

    // Initialize WavLM ONNX speaker encoder (optional — non-fatal).
    if (!cfg_.wavlm.model_path.empty()) {
        if (!wavlm_enc_.init(cfg_.wavlm)) {
            LOG_WARN("AudioPipe", "WavLM speaker encoder init failed");
        }
    }

    // Initialize UniSpeech-SAT ONNX speaker encoder (optional — non-fatal).
    if (!cfg_.unispeech.model_path.empty()) {
        if (!unispeech_enc_.init(cfg_.unispeech)) {
            LOG_WARN("AudioPipe", "UniSpeech-SAT speaker encoder init failed");
        }
    }

    // Initialize WavLM-Large + ECAPA-TDNN native GPU speaker encoder (optional).
    if (!cfg_.wavlm_ecapa_model.empty()) {
        if (!wlecapa_enc_.init(cfg_.wavlm_ecapa_model)) {
            LOG_WARN("AudioPipe", "WavLM-ECAPA native GPU init failed");
        } else {
            LOG_INFO("AudioPipe", "WavLM-ECAPA native GPU encoder ready (192-dim)");
        }
    }

    speaker_threshold_.store(cfg_.speaker_threshold, std::memory_order_relaxed);
    wavlm_threshold_.store(cfg_.wavlm_threshold, std::memory_order_relaxed);
    unispeech_threshold_.store(cfg_.unispeech_threshold, std::memory_order_relaxed);
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
             stats_.pcm_samples_in, stats_.mel_frames, stats_.speech_frames);
}

void AudioPipeline::set_asr_rep_penalty(float p) {
    asr_rep_penalty_.store(p, std::memory_order_relaxed);
    if (asr_engine_) asr_engine_->set_repetition_penalty(p);
}

// ASR worker thread — picks up jobs from queue, runs transcription off-main-loop.
void AudioPipeline::asr_loop() {
    LOG_INFO("AudioPipe", "ASR worker thread started");
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
                auto esc = [](const std::string& s) -> std::string {
                    std::string out;
                    out.reserve(s.size() + 16);
                    for (char c : s) {
                        if (c == '"') out += "\\\"";
                        else if (c == '\\') out += "\\\\";
                        else if (c == '\n') out += "\\n";
                        else out += c;
                    }
                    return out;
                };
                char json[1024];
                snprintf(json, sizeof(json),
                    R"({"stage":"partial","audio_sec":%.2f,"total_ms":%.1f,"tokens":%d,"text":"%s"})",
                    job.audio_duration_sec, result.total_ms, result.token_count,
                    esc(result.text).c_str());
                on_asr_log_(json);
            }
            continue;
        }

        stats_.asr_active = true;
        stats_.asr_latency_ms = result.total_ms;
        stats_.asr_audio_duration_s = job.audio_duration_sec;

        if (!result.text.empty()) {
            // Check hallucination filter toggle — when OFF, pass all results through.
            bool suppress = result.hallucinated &&
                            asr_halluc_filter_.load(std::memory_order_relaxed);
            if (!suppress) {
                LOG_INFO("AudioPipe", "ASR: \"%s\" (%.1fms, %.2fs audio, mel=%.0fms enc=%.0fms dec=%.0fms %dtok%s)",
                         result.text.c_str(), result.total_ms, job.audio_duration_sec,
                         result.mel_ms, result.encoder_ms, result.decode_ms, result.token_count,
                         result.hallucinated ? " [artifact]" : "");
                if (on_transcript_) on_transcript_(result, job.audio_duration_sec,
                                                     job.speaker_id, job.speaker_name,
                                                     job.speaker_sim, job.trigger_reason);
            } else {
                LOG_INFO("AudioPipe", "ASR: artifact filtered \"%s\" (%.1fms, %.2fs audio)",
                         result.text.c_str(), result.total_ms, job.audio_duration_sec);
            }
        } else {
            LOG_INFO("AudioPipe", "ASR: (empty) (%.1fms, %.2fs audio)",
                     result.total_ms, job.audio_duration_sec);
        }

        // Send detailed ASR log for WebUI debug panel.
        if (on_asr_log_) {
            // Escape raw_text and text for JSON.
            auto esc = [](const std::string& s) -> std::string {
                std::string out;
                out.reserve(s.size() + 16);
                for (char c : s) {
                    if (c == '"') out += "\\\"";
                    else if (c == '\\') out += "\\\\";
                    else if (c == '\n') out += "\\n";
                    else if (c == '\r') out += "\\r";
                    else if (c == '\t') out += "\\t";
                    else out += c;
                }
                return out;
            };
            char json[2048];
            snprintf(json, sizeof(json),
                R"({"stage":"result","trigger":"%s","audio_sec":%.2f,)"
                R"("mel_ms":%.1f,"mel_frames":%d,"encoder_ms":%.1f,"encoder_out":%d,)"
                R"("decode_ms":%.1f,"tokens":%d,"postprocess_ms":%.1f,"total_ms":%.1f,)"
                R"("hallucinated":%s,"raw_text":"%s","text":"%s"})",
                job.trigger_reason.c_str(), job.audio_duration_sec,
                result.mel_ms, result.mel_frames, result.encoder_ms, result.encoder_out_len,
                result.decode_ms, result.token_count, result.postprocess_ms, result.total_ms,
                result.hallucinated ? "true" : "false",
                esc(result.raw_text).c_str(), esc(result.text).c_str());
            on_asr_log_(json);
        }
    }
    LOG_INFO("AudioPipe", "ASR worker thread exited");
}

void AudioPipeline::push_pcm(const int16_t* data, int n_samples) {
    if (!ring_ || n_samples <= 0) return;
    size_t bytes = n_samples * sizeof(int16_t);
    size_t written = ring_->push(reinterpret_cast<const uint8_t*>(data), bytes);
    if (written < bytes) {
        LOG_WARN("AudioPipe", "Ring buffer overflow, dropped %zu bytes",
                 bytes - written);
    }
}

void AudioPipeline::process_loop() {
    // Process chunk size in samples.
    int chunk_samples = cfg_.mel.sample_rate * cfg_.process_chunk_ms / 1000;
    size_t chunk_bytes = chunk_samples * sizeof(int16_t);
    std::vector<int16_t> pcm_buf(chunk_samples);

    // Host buffer for reading back Mel frames for VAD.
    int n_mels = cfg_.mel.n_mels;
    std::vector<float> mel_host(n_mels);

    // Silero VAD processes 512-sample windows from float PCM.
    int silero_window = silero_.initialized() ? cfg_.silero.window_samples : 0;
    std::vector<float> pcm_float;  // reused buffer for gain-applied float PCM
    std::vector<float> silero_buf; // carries remainder samples across chunks

    // TEN VAD processes 160-sample hops (10ms).
    int ten_hop = ten_.initialized() ? cfg_.ten.hop_size : 0;
    std::vector<int16_t> ten_buf;  // carries remainder for TEN VAD

    LOG_INFO("AudioPipe", "Process loop: chunk=%d samples (%d ms), silero=%s fsmn=%s ten=%s",
             chunk_samples, cfg_.process_chunk_ms,
             silero_.initialized() ? "ON" : "OFF",
             fsmn_.initialized() ? "ON" : "OFF",
             ten_.initialized() ? "ON" : "OFF");

    int diag_counter = 0;

    while (running_.load(std::memory_order_relaxed)) {
        size_t avail = ring_->available();

        if (avail < chunk_bytes) {
            // Not enough data yet — sleep briefly.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Pull PCM from ring.
        size_t got = ring_->pop(reinterpret_cast<uint8_t*>(pcm_buf.data()),
                                chunk_bytes);
        int n_samples = got / sizeof(int16_t);
        stats_.pcm_samples_in += n_samples;

        // Apply gain before processing.
        float g = gain_.load(std::memory_order_relaxed);
        if (g != 1.0f) {
            for (int i = 0; i < n_samples; i++) {
                int32_t s = (int32_t)(pcm_buf[i] * g);
                pcm_buf[i] = (int16_t)std::max(-32768, std::min(32767, s));
            }
        }

        // Compute RMS from PCM for stats (fast, on host).
        double sum_sq = 0;
        for (int i = 0; i < n_samples; i++) {
            float s = pcm_buf[i] / 32768.0f;
            sum_sq += s * s;
        }
        stats_.last_rms = n_samples > 0 ? sqrtf((float)(sum_sq / n_samples)) : 0;

        // Run Silero VAD on raw PCM (512-sample windows).
        if (silero_.initialized() && silero_window > 0 &&
            enable_silero_.load(std::memory_order_relaxed)) {
            // Convert int16 -> float for Silero (reuse buffer).
            pcm_float.resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                pcm_float[i] = pcm_buf[i] / 32768.0f;
            }
            // Append new samples to remainder from previous chunk.
            silero_buf.insert(silero_buf.end(), pcm_float.begin(),
                              pcm_float.begin() + n_samples);
            // Process in silero_window-sized chunks.
            int consumed = 0;
            while (consumed + silero_window <= (int)silero_buf.size()) {
                SileroVadResult svr = silero_.process(
                    silero_buf.data() + consumed, silero_window);
                consumed += silero_window;
                stats_.silero_prob = svr.probability;
                stats_.silero_speech = svr.is_speech;
                // Use Silero result as authoritative VAD if available.
                stats_.is_speech = svr.is_speech;
                if (on_vad_ && (svr.segment_start || svr.segment_end)) {
                    VadResult vr{};
                    vr.is_speech = svr.is_speech;
                    vr.segment_start = svr.segment_start;
                    vr.segment_end = svr.segment_end;
                    vr.energy = svr.probability;  // repurpose energy field for prob
                    on_vad_(vr, (int)stats_.mel_frames);
                }
            }
            // Keep remainder for next chunk.
            if (consumed > 0) {
                silero_buf.erase(silero_buf.begin(),
                                 silero_buf.begin() + consumed);
            }
        }

        // Run FSMN VAD on raw PCM chunk (accumulates fbank internally).
        if (fsmn_.initialized() &&
            enable_fsmn_.load(std::memory_order_relaxed)) {
            FsmnVadResult fvr = fsmn_.process(pcm_buf.data(), n_samples);
            stats_.fsmn_prob = fvr.probability;
            stats_.fsmn_speech = fvr.is_speech;
        }

        // Run TEN VAD on raw PCM (160-sample hops).
        if (ten_.initialized() && ten_hop > 0 &&
            enable_ten_.load(std::memory_order_relaxed)) {
            ten_buf.insert(ten_buf.end(), pcm_buf.data(),
                           pcm_buf.data() + n_samples);
            int consumed = 0;
            while (consumed + ten_hop <= (int)ten_buf.size()) {
                TenVadResult tvr = ten_.process(
                    ten_buf.data() + consumed, ten_hop);
                consumed += ten_hop;
                stats_.ten_prob = tvr.probability;
                stats_.ten_speech = tvr.is_speech;
            }
            if (consumed > 0) {
                ten_buf.erase(ten_buf.begin(),
                              ten_buf.begin() + consumed);
            }
        }

        // Buffer PCM for speaker identification during speech (VAD-gated).
        bool any_speaker_enabled =
            (speaker_enc_.initialized() && enable_speaker_.load(std::memory_order_relaxed)) ||
            (wavlm_enc_.initialized() && enable_wavlm_.load(std::memory_order_relaxed)) ||
            (unispeech_enc_.initialized() && enable_unispeech_.load(std::memory_order_relaxed)) ||
            (wlecapa_enc_.initialized() && enable_wlecapa_.load(std::memory_order_relaxed));
        bool need_segment_pcm = any_speaker_enabled;

        // Clear active flags each tick — only set true when extraction happens.
        stats_.speaker_active = false;
        stats_.wavlm_active = false;
        stats_.unispeech_active = false;
        stats_.wlecapa_active = false;
        stats_.wlecapa_change_valid = false;
        stats_.asr_active = false;

        if (need_segment_pcm) {
            // Determine speech state from selected VAD source.
            VadSource src = static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed));
            bool vad_speech = false;
            switch (src) {
                case VadSource::SILERO: vad_speech = stats_.silero_speech; break;
                case VadSource::FSMN:   vad_speech = stats_.fsmn_speech; break;
                case VadSource::TEN:    vad_speech = stats_.ten_speech; break;
                case VadSource::ANY:
                default:
                    vad_speech = stats_.is_speech || stats_.silero_speech ||
                                 stats_.fsmn_speech || stats_.ten_speech;
                    break;
            }
            if (vad_speech && !in_speech_segment_) {
                in_speech_segment_ = true;
                early_extracted_   = false;
                speech_pcm_buf_.clear();
                speaker_fbank_.reset();
                // Reset speaker ID for new segment — prevents stale ID from
                // previous speaker being captured by ASR trigger if this segment
                // is too short for WL-ECAPA extraction (< 1.0s full, < 1.7s early).
                stats_.wlecapa_id = -1;
                stats_.wlecapa_sim = 0.0f;
                stats_.wlecapa_new = false;
                strncpy(stats_.wlecapa_name, "", sizeof(stats_.wlecapa_name));
            }
            if (in_speech_segment_) {
                speech_pcm_buf_.insert(speech_pcm_buf_.end(),
                                       pcm_buf.data(), pcm_buf.data() + n_samples);
                if ((speaker_enc_.initialized() &&
                     enable_speaker_.load(std::memory_order_relaxed)) ||
                    (unispeech_enc_.initialized() &&
                     enable_unispeech_.load(std::memory_order_relaxed))) {
                    speaker_fbank_.push_pcm(pcm_buf.data(), n_samples);
                }
                // Limit to 10 seconds (160000 samples @ 16kHz).
                if (speech_pcm_buf_.size() > 160000) {
                    speech_pcm_buf_.erase(speech_pcm_buf_.begin(),
                                          speech_pcm_buf_.begin() + n_samples);
                }

                // Early extraction: run WL-ECAPA once we have enough speech,
                // without waiting for end-of-segment. This reduces "time to light-up".
                // IMPORTANT: auto_register=false — never create new speakers from
                // short early clips. If no match, report "identifying" state.
                int early_thresh = early_trigger_samples_.load(std::memory_order_relaxed);
                if (!early_extracted_ &&
                    enable_early_.load(std::memory_order_relaxed) &&
                    wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed) &&
                    (int)speech_pcm_buf_.size() >= early_thresh) {
                    early_extracted_ = true;
                    int early_samples = (int)speech_pcm_buf_.size();
                    std::vector<float> pcm_f32(early_samples);
                    for (int i = 0; i < early_samples; i++)
                        pcm_f32[i] = speech_pcm_buf_[i] / 32768.0f;
                    auto emb = wlecapa_enc_.extract(pcm_f32.data(), early_samples);
                    if (!emb.empty()) {
                        float thresh = wlecapa_threshold_.load(std::memory_order_relaxed);
                        // No auto-registration: match only against existing speakers.
                        SpeakerMatch match = wlecapa_db_.identify(emb, thresh, /*auto_register=*/false);
                        stats_.wlecapa_active = true;
                        stats_.wlecapa_is_early = true;
                        stats_.wlecapa_lat_cnn_ms     = wlecapa_enc_.last_lat_cnn_ms();
                        stats_.wlecapa_lat_encoder_ms = wlecapa_enc_.last_lat_encoder_ms();
                        stats_.wlecapa_lat_ecapa_ms   = wlecapa_enc_.last_lat_ecapa_ms();
                        stats_.wlecapa_lat_total_ms   = wlecapa_enc_.last_lat_total_ms();
                        if (match.speaker_id >= 0) {
                            // Matched an existing speaker — light up immediately.
                            stats_.wlecapa_id = match.speaker_id;
                            stats_.wlecapa_sim = match.similarity;
                            stats_.wlecapa_new = false;
                            stats_.wlecapa_count = wlecapa_db_.count();
                            stats_.wlecapa_exemplars = match.exemplar_count;
                            stats_.wlecapa_hits_above = match.hits_above;
                            strncpy(stats_.wlecapa_name, match.name.c_str(),
                                    sizeof(stats_.wlecapa_name) - 1);
                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "WL-ECAPA(early): id=%d sim=%.3f %s (%.2fs, %.1fms)",
                                     match.speaker_id, match.similarity,
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     early_samples / 16000.0f,
                                     wlecapa_enc_.last_lat_total_ms());
                            if (on_speaker_) on_speaker_(match);
                        } else {
                            // No match — signal "identifying" to UI.
                            stats_.wlecapa_id = -1;
                            stats_.wlecapa_sim = match.similarity;
                            stats_.wlecapa_new = false;
                            strncpy(stats_.wlecapa_name, "(identifying)",
                                    sizeof(stats_.wlecapa_name) - 1);
                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "WL-ECAPA(early): no match (best_sim=%.3f, %.2fs, %.1fms) — awaiting full segment",
                                     match.similarity, early_samples / 16000.0f,
                                     wlecapa_enc_.last_lat_total_ms());
                        }
                    }
                }
            }
            if (!vad_speech && in_speech_segment_) {
                in_speech_segment_ = false;
                int speech_samples = (int)speech_pcm_buf_.size();
                float speech_duration = speech_samples / 16000.0f;

                LOG_INFO("AudioPipe", "Speech segment ended: %.2fs (%d samples)",
                         speech_duration, speech_samples);

                // Debug: dump first 10 speech segments to WAV for analysis.
                {
                    static int dump_count = 0;
                    if (dump_count < 10 && speech_samples >= 16000) {
                        char path[128];
                        snprintf(path, sizeof(path), "/tmp/spk_seg_%d.wav", dump_count);
                        FILE* f = fopen(path, "wb");
                        if (f) {
                            // Write minimal WAV header (16-bit mono 16kHz).
                            uint32_t data_sz = speech_samples * 2;
                            uint32_t file_sz = 36 + data_sz;
                            uint16_t fmt = 1; // PCM
                            uint16_t ch = 1;
                            uint32_t sr = 16000;
                            uint32_t bps = 32000;
                            uint16_t ba = 2;
                            uint16_t bits = 16;
                            fwrite("RIFF", 1, 4, f);
                            fwrite(&file_sz, 4, 1, f);
                            fwrite("WAVEfmt ", 1, 8, f);
                            uint32_t fmt_sz = 16;
                            fwrite(&fmt_sz, 4, 1, f);
                            fwrite(&fmt, 2, 1, f);
                            fwrite(&ch, 2, 1, f);
                            fwrite(&sr, 4, 1, f);
                            fwrite(&bps, 4, 1, f);
                            fwrite(&ba, 2, 1, f);
                            fwrite(&bits, 2, 1, f);
                            fwrite("data", 1, 4, f);
                            fwrite(&data_sz, 4, 1, f);
                            fwrite(speech_pcm_buf_.data(), 2, speech_samples, f);
                            fclose(f);
                            LOG_INFO("AudioPipe", "Dumped segment %d: %s (%.2fs)",
                                     dump_count, path, speech_duration);
                        }
                        dump_count++;
                    }
                }

                // Read fbank features (shared between CAM++ and ECAPA-TDNN).
                // CAM++ needs >= 150 frames (~1.5s), ECAPA-TDNN >= 100 frames (~1.0s).
                int fbank_frames = speaker_fbank_.frames_ready();
                std::vector<float> fbank_host;
                if (fbank_frames >= 100) {
                    fbank_host.resize(fbank_frames * 80);
                    speaker_fbank_.read_fbank(fbank_host.data(), fbank_frames);
                }

                // CAM++ speaker encoder (uses fbank features).
                if (speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    fbank_frames >= 150) {
                        float thresh = speaker_threshold_.load(std::memory_order_relaxed);

                        // Diagnostic: fbank value statistics.
                        {
                            float fmin = 1e30f, fmax = -1e30f, fsum = 0;
                            int total = fbank_frames * 80;
                            for (int i = 0; i < total; i++) {
                                fmin = std::min(fmin, fbank_host[i]);
                                fmax = std::max(fmax, fbank_host[i]);
                                fsum += fbank_host[i];
                            }
                            LOG_INFO("AudioPipe", "CAM++ fbank: frames=%d min=%.3f max=%.3f mean=%.3f",
                                     fbank_frames, fmin, fmax, fsum / total);
                        }

                        auto emb = speaker_enc_.extract(fbank_host.data(), fbank_frames);
                        if (!emb.empty()) {
                            // Validate embedding norm (should be ~1.0 if L2-normalized).
                            float enorm = 0;
                            for (float v : emb) enorm += v * v;
                            enorm = sqrtf(enorm);
                            LOG_INFO("AudioPipe", "CAM++ emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                     enorm, emb[0], emb[1], emb[2], emb[3]);
                            SpeakerMatch match = speaker_db_.identify(emb, thresh);
                            stats_.speaker_id = match.speaker_id;
                            stats_.speaker_sim = match.similarity;
                            stats_.speaker_new = match.is_new;
                            stats_.speaker_count = speaker_db_.count();
                            stats_.speaker_active = true;
                            strncpy(stats_.speaker_name, match.name.c_str(),
                                    sizeof(stats_.speaker_name) - 1);
                            stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "CAM++: id=%d sim=%.3f %s%s (fbank=%d)",
                                     match.speaker_id, match.similarity,
                                     match.is_new ? "NEW " : "",
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     fbank_frames);
                            if (on_speaker_) on_speaker_(match);
                        }
                }

                // WavLM speaker encoder (uses raw PCM waveform).
                if (wavlm_enc_.initialized() &&
                    enable_wavlm_.load(std::memory_order_relaxed) &&
                    speech_samples >= 24000) {  // minimum ~1.5s for Gemm output
                    auto emb = wavlm_enc_.extract_int16(speech_pcm_buf_.data(), speech_samples);
                    if (!emb.empty()) {
                        float enorm = 0;
                        for (float v : emb) enorm += v * v;
                        enorm = sqrtf(enorm);
                        LOG_INFO("AudioPipe", "WavLM emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                 enorm, emb[0], emb[1], emb[2], emb[3]);
                        float thresh = wavlm_threshold_.load(std::memory_order_relaxed);
                        SpeakerMatch match = wavlm_db_.identify(emb, thresh);
                        stats_.wavlm_id = match.speaker_id;
                        stats_.wavlm_sim = match.similarity;
                        stats_.wavlm_new = match.is_new;
                        stats_.wavlm_count = wavlm_db_.count();
                        stats_.wavlm_active = true;
                        strncpy(stats_.wavlm_name, match.name.c_str(),
                                sizeof(stats_.wavlm_name) - 1);
                        stats_.wavlm_name[sizeof(stats_.wavlm_name) - 1] = '\0';
                        LOG_INFO("AudioPipe", "WavLM: id=%d sim=%.3f %s%s (%d samples)",
                                 match.speaker_id, match.similarity,
                                 match.is_new ? "NEW " : "",
                                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                 speech_samples);
                    }
                }

                // ECAPA-TDNN speaker encoder (uses fbank features, not raw PCM).
                // Adapted from WeSpeaker ECAPA-TDNN-1024-LM with ASTP attention pooling.
                if (unispeech_enc_.initialized() &&
                    enable_unispeech_.load(std::memory_order_relaxed) &&
                    fbank_frames >= 100) {  // minimum ~1.0s for ECAPA-TDNN
                    auto emb = unispeech_enc_.extract_fbank(fbank_host.data(), fbank_frames, 80);
                    if (!emb.empty()) {
                        float enorm = 0;
                        for (float v : emb) enorm += v * v;
                        enorm = sqrtf(enorm);
                        LOG_INFO("AudioPipe", "ECAPA emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                 enorm, emb[0], emb[1], emb[2], emb[3]);
                        float thresh = unispeech_threshold_.load(std::memory_order_relaxed);
                        SpeakerMatch match = unispeech_db_.identify(emb, thresh);
                        stats_.unispeech_id = match.speaker_id;
                        stats_.unispeech_sim = match.similarity;
                        stats_.unispeech_new = match.is_new;
                        stats_.unispeech_count = unispeech_db_.count();
                        stats_.unispeech_active = true;
                        strncpy(stats_.unispeech_name, match.name.c_str(),
                                sizeof(stats_.unispeech_name) - 1);
                        stats_.unispeech_name[sizeof(stats_.unispeech_name) - 1] = '\0';
                        LOG_INFO("AudioPipe", "ECAPA: id=%d sim=%.3f %s%s (fbank=%d)",
                                 match.speaker_id, match.similarity,
                                 match.is_new ? "NEW " : "",
                                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                 fbank_frames);
                    }
                }

                // WavLM-Large + ECAPA-TDNN native GPU speaker encoder (uses raw PCM).
                int min_spk_samples = min_speech_samples_.load(std::memory_order_relaxed);
                if (wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed) &&
                    speech_samples >= min_spk_samples) {
                    // Convert int16 PCM to float32 [-1, 1].
                    std::vector<float> pcm_f32(speech_samples);
                    for (int i = 0; i < speech_samples; i++)
                        pcm_f32[i] = speech_pcm_buf_[i] / 32768.0f;
                    auto emb = wlecapa_enc_.extract(pcm_f32.data(), speech_samples);
                    if (!emb.empty()) {
                        float enorm = 0;
                        for (float v : emb) enorm += v * v;
                        enorm = sqrtf(enorm);
                        LOG_INFO("AudioPipe", "WL-ECAPA emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                 enorm, emb[0], emb[1], emb[2], emb[3]);
                        float thresh = wlecapa_threshold_.load(std::memory_order_relaxed);
                        SpeakerMatch match = wlecapa_db_.identify(emb, thresh);

                        stats_.wlecapa_id = match.speaker_id;
                        stats_.wlecapa_sim = match.similarity;
                        stats_.wlecapa_new = match.is_new;
                        stats_.wlecapa_count = wlecapa_db_.count();
                        stats_.wlecapa_exemplars = match.exemplar_count;
                        stats_.wlecapa_hits_above = match.hits_above;
                        stats_.wlecapa_active = true;
                        stats_.wlecapa_is_early = false;
                        stats_.wlecapa_lat_cnn_ms     = wlecapa_enc_.last_lat_cnn_ms();
                        stats_.wlecapa_lat_encoder_ms = wlecapa_enc_.last_lat_encoder_ms();
                        stats_.wlecapa_lat_ecapa_ms   = wlecapa_enc_.last_lat_ecapa_ms();
                        stats_.wlecapa_lat_total_ms   = wlecapa_enc_.last_lat_total_ms();

                        // Change detection: cosine similarity with previous segment embedding.
                        if (!prev_wlecapa_emb_.empty() && prev_wlecapa_emb_.size() == emb.size()) {
                            float dot = 0;
                            for (size_t j = 0; j < emb.size(); j++)
                                dot += emb[j] * prev_wlecapa_emb_[j];
                            stats_.wlecapa_change_sim = dot;  // both L2-normed → dot = cosine
                            stats_.wlecapa_change_valid = true;
                        } else {
                            stats_.wlecapa_change_sim = -1.0f;
                            stats_.wlecapa_change_valid = false;
                        }
                        prev_wlecapa_emb_ = emb;

                        strncpy(stats_.wlecapa_name, match.name.c_str(),
                                sizeof(stats_.wlecapa_name) - 1);
                        stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                        LOG_INFO("AudioPipe", "WL-ECAPA: id=%d sim=%.3f %s%s (%d samples, %.1fms)",
                                 match.speaker_id, match.similarity,
                                 match.is_new ? "NEW " : "",
                                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                 speech_samples, wlecapa_enc_.last_lat_total_ms());
                        if (on_speaker_) on_speaker_(match);
                    }
                }

                speech_pcm_buf_.clear();
            }
        }

        // ASR: accumulate ALL audio continuously. Whisper encoder naturally handles
        // silence and mixed audio. VAD is used only as a trigger hint (WHEN to fire
        // transcription) and to track speech content ratio for filtering.
        if (asr_engine_ && asr_engine_->is_loaded() &&
            enable_asr_.load(std::memory_order_relaxed)) {

            // Always accumulate ALL audio (speech + silence).
            asr_pcm_buf_.insert(asr_pcm_buf_.end(),
                                pcm_buf.data(), pcm_buf.data() + n_samples);

            // Determine speech state from ASR-specific VAD source (independent from speaker pipeline).
            VadSource asr_src = static_cast<VadSource>(asr_vad_source_.load(std::memory_order_relaxed));
            bool asr_vad_speech = false;
            switch (asr_src) {
                case VadSource::SILERO: asr_vad_speech = stats_.silero_speech; break;
                case VadSource::FSMN:   asr_vad_speech = stats_.fsmn_speech; break;
                case VadSource::TEN:    asr_vad_speech = stats_.ten_speech; break;
                case VadSource::DIRECT: asr_vad_speech = true; break;  // always "speech" — trigger on buffer duration
                case VadSource::ANY:
                default:
                    asr_vad_speech = stats_.is_speech || stats_.silero_speech ||
                                     stats_.fsmn_speech || stats_.ten_speech;
                    break;
            }

            if (asr_vad_speech) {
                asr_saw_speech_ = true;
                asr_post_silence_ = 0;
                asr_speech_samples_ += n_samples;
            } else if (asr_saw_speech_) {
                asr_post_silence_++;
            }

            // Streaming ASR partial: during active speech, submit partial transcription
            // every N seconds for real-time display.
            // Adapted from qwen35-thor: STREAMING_ASR_CHUNK_S periodic partial.
            int partial_interval = asr_partial_samples_.load(std::memory_order_relaxed);
            if (partial_interval > 0 && asr_saw_speech_ && asr_post_silence_ == 0 &&
                !asr_busy_.load(std::memory_order_relaxed)) {
                int buf_samples = (int)asr_pcm_buf_.size();
                if (buf_samples - asr_partial_sent_at_ >= partial_interval &&
                    buf_samples >= partial_interval) {
                    // Copy current buffer for partial transcription.
                    std::vector<float> pcm_f32(buf_samples);
                    for (int i = 0; i < buf_samples; i++)
                        pcm_f32[i] = asr_pcm_buf_[i] / 32768.0f;
                    float dur = buf_samples / 16000.0f;
                    {
                        std::lock_guard<std::mutex> lock(asr_mutex_);
                        asr_queue_.push(ASRJob{std::move(pcm_f32), dur, "streaming_partial", true});
                    }
                    asr_cv_.notify_one();
                    asr_partial_sent_at_ = buf_samples;
                }
            }

            // Trigger transcription when:
            // (a) Post-speech silence reaches configured threshold, or
            // (b) Buffer exceeds max size during continuous speech.
            int asr_post_silence_chunks = asr_post_silence_ms_.load(std::memory_order_relaxed)
                                          / cfg_.process_chunk_ms;
            if (asr_post_silence_chunks < 1) asr_post_silence_chunks = 1;
            int ASR_MAX_BUF_SAMPLES = asr_max_buf_samples_.load(std::memory_order_relaxed);
            int ASR_MIN_SAMPLES = asr_min_samples_.load(std::memory_order_relaxed);
            int ASR_PRE_ROLL_SAMPLES = asr_pre_roll_samples_.load(std::memory_order_relaxed);

            bool asr_trigger = false;
            std::string trigger_reason;
            if (asr_saw_speech_ && asr_post_silence_ >= asr_post_silence_chunks) {
                asr_trigger = true;
                trigger_reason = "post_silence";
            } else if ((int)asr_pcm_buf_.size() >= ASR_MAX_BUF_SAMPLES) {
                asr_trigger = true;
                trigger_reason = "buffer_full";
            }

            if (asr_trigger && (int)asr_pcm_buf_.size() >= ASR_MIN_SAMPLES) {
                int asr_samples = (int)asr_pcm_buf_.size();
                float asr_duration = asr_samples / 16000.0f;
                float speech_sec = asr_speech_samples_ / 16000.0f;
                float speech_ratio = asr_duration > 0 ? speech_sec / asr_duration : 0;

                // Speech content filter: skip segments with too little detected speech.
                // Only applies to buffer_full triggers — post_silence triggers are already
                // VAD-confirmed (asr_saw_speech_=true), so short affirmative responses
                // like "好", "嗯", "ok" pass through correctly.
                bool has_enough_speech = true;
                if (trigger_reason == "buffer_full") {
                    has_enough_speech = (speech_sec >= 0.3f);

                    // Speech ratio filter: reject long buffers where speech is a tiny fraction.
                    float min_speech_ratio = asr_min_speech_ratio_.load(std::memory_order_relaxed);
                    if (has_enough_speech && asr_duration > 5.0f && min_speech_ratio > 0 &&
                        speech_ratio < min_speech_ratio) {
                        has_enough_speech = false;
                    }
                }

                // Energy filter: compute average RMS energy and reject low-energy segments.
                // Adapted from qwen35-thor (voice_session.cpp): min_avg_energy rejection.
                float min_energy = asr_min_energy_.load(std::memory_order_relaxed);
                bool has_enough_energy = true;
                float avg_energy = 0.0f;
                if (min_energy > 0.0f && asr_samples > 0) {
                    double energy_sum = 0;
                    for (int i = 0; i < asr_samples; i++) {
                        float s = asr_pcm_buf_[i] / 32768.0f;
                        energy_sum += s * s;
                    }
                    avg_energy = (float)std::sqrt(energy_sum / asr_samples);
                    has_enough_energy = (avg_energy >= min_energy);
                }

                if (has_enough_speech && has_enough_energy) {
                    // Trim trailing silence: scan backwards to find last
                    // energetic region, keep a small tail margin (100ms).
                    // This prevents feeding long silence tails to the model
                    // which can cause hallucinated filler outputs.
                    int trim_samples = asr_samples;
                    {
                        const int window = 1600; // 100ms windows
                        const float silence_rms = 0.005f;
                        const int tail_margin = 1600; // keep 100ms after last energy
                        int last_energy_pos = trim_samples;
                        for (int pos = trim_samples - window; pos >= 0; pos -= window) {
                            double w_sum = 0;
                            int w_end = std::min(pos + window, trim_samples);
                            for (int j = pos; j < w_end; j++) {
                                float s = asr_pcm_buf_[j] / 32768.0f;
                                w_sum += s * s;
                            }
                            float w_rms = (float)std::sqrt(w_sum / (w_end - pos));
                            if (w_rms > silence_rms) {
                                last_energy_pos = w_end;
                                break;
                            }
                        }
                        int trimmed = std::min(trim_samples, last_energy_pos + tail_margin);
                        // Don't trim too aggressively — keep at least 80% of original
                        if (trimmed >= asr_samples * 4 / 5) {
                            trim_samples = trimmed;
                        }
                    }
                    float trimmed_duration = trim_samples / 16000.0f;

                    // Convert int16 → float32 for ASR engine.
                    std::vector<float> pcm_f32(trim_samples);
                    for (int i = 0; i < trim_samples; i++)
                        pcm_f32[i] = asr_pcm_buf_[i] / 32768.0f;

                    // Send ASR log: trigger event.
                    if (on_asr_log_) {
                        char json[512];
                        snprintf(json, sizeof(json),
                            R"({"stage":"trigger","reason":"%s","buf_sec":%.2f,"trimmed_sec":%.2f,"speech_sec":%.2f,"speech_ratio":%.2f,"samples":%d})",
                            trigger_reason.c_str(), asr_duration, trimmed_duration, speech_sec, speech_ratio, asr_samples);
                        on_asr_log_(json);
                    }

                    // Push job to async ASR thread (non-blocking).
                    // Capture current speaker identification for transcript annotation.
                    int spk_id = stats_.wlecapa_id;
                    float spk_sim = stats_.wlecapa_sim;
                    std::string spk_name(stats_.wlecapa_name);
                    {
                        std::lock_guard<std::mutex> lock(asr_mutex_);
                        ASRJob job{std::move(pcm_f32), trimmed_duration, trigger_reason,
                                   /*is_partial=*/false, spk_id, std::move(spk_name), spk_sim};
                        asr_queue_.push(std::move(job));
                    }
                    asr_cv_.notify_one();
                } else {
                    // Not enough speech or energy — skip ASR, log for debug.
                    std::string skip_reason = !has_enough_energy ? "low_energy" :
                        (speech_sec < 0.3f ? "low_speech" : "low_speech_ratio");
                    if (on_asr_log_) {
                        char json[512];
                        snprintf(json, sizeof(json),
                            R"({"stage":"skipped","reason":"%s","buf_sec":%.2f,"speech_sec":%.2f,"speech_ratio":%.2f,"avg_energy":%.5f})",
                            skip_reason.c_str(), asr_duration, speech_sec, speech_ratio, avg_energy);
                        on_asr_log_(json);
                    }
                    LOG_INFO("AudioPipe", "ASR: skipped (%s, speech=%.2fs/%.2fs ratio=%.0f%% energy=%.5f)",
                             skip_reason.c_str(), speech_sec, asr_duration, speech_ratio * 100, avg_energy);
                }

                // Keep last pre_roll as context for next segment.
                if (asr_samples > ASR_PRE_ROLL_SAMPLES) {
                    asr_pcm_buf_.erase(asr_pcm_buf_.begin(),
                                       asr_pcm_buf_.end() - ASR_PRE_ROLL_SAMPLES);
                }
                asr_saw_speech_ = false;
                asr_post_silence_ = 0;
                asr_speech_samples_ = 0;
                asr_partial_sent_at_ = 0;
            }

            // When idle (no speech seen), trim buffer to ~2x pre-roll to avoid
            // unbounded growth during long silence periods.
            if (!asr_saw_speech_ && (int)asr_pcm_buf_.size() > ASR_PRE_ROLL_SAMPLES * 2) {
                asr_pcm_buf_.erase(asr_pcm_buf_.begin(),
                                   asr_pcm_buf_.end() - ASR_PRE_ROLL_SAMPLES);
                asr_speech_samples_ = 0;
            }
        }

        // Update ASR buffer stats.
        stats_.asr_buf_sec = asr_pcm_buf_.size() / 16000.0f;
        stats_.asr_buf_has_speech = asr_saw_speech_;
        stats_.asr_busy = asr_busy_.load(std::memory_order_relaxed);

        // Push to Mel spectrogram (GPU).
        int new_frames = mel_.push_pcm(pcm_buf.data(), n_samples);
        stats_.mel_frames += new_frames;

        if (new_frames <= 0) continue;

        // Run VAD on new frames.
        //   Copy new Mel frames back to host one at a time for state machine.
        int start_frame = mel_.frames_ready() - new_frames;
        for (int i = 0; i < new_frames; i++) {
            int frame_idx = start_frame + i;
            cudaMemcpy(mel_host.data(),
                       mel_.mel_buffer() + frame_idx * n_mels,
                       n_mels * sizeof(float),
                       cudaMemcpyDeviceToHost);

            VadResult vr = vad_.process_frame(mel_host.data(), n_mels);
            stats_.last_energy = vr.energy;
            if (vr.is_speech) stats_.speech_frames++;

            // Energy VAD drives is_speech only when Silero is not available.
            if (!silero_.initialized()) {
                stats_.is_speech = vr.is_speech;
                if (on_vad_ && (vr.segment_start || vr.segment_end)) {
                    on_vad_(vr, frame_idx);
                }
            }
        }

        // Report stats.
        if (on_stats_) {
            on_stats_(stats_);
        }

        // Periodic diagnostic log (~every 1s = 10 chunks at 100ms).
        if (++diag_counter % 10 == 0) {
            LOG_INFO("AudioPipe", "DIAG rms=%.4f silero=%.3f fsmn=%.3f ten=%.3f speech=%d gain=%.1f spk=%d(%.2f)",
                     stats_.last_rms, stats_.silero_prob,
                     stats_.fsmn_prob, stats_.ten_prob,
                     (int)stats_.is_speech,
                     gain_.load(std::memory_order_relaxed),
                     stats_.speaker_id, stats_.speaker_sim);
        }
    }
}

} // namespace deusridet
