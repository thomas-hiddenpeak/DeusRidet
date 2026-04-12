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
        }
    }

    speaker_threshold_.store(cfg_.speaker_threshold, std::memory_order_relaxed);
    wlecapa_threshold_.store(cfg_.wavlm_ecapa_threshold, std::memory_order_relaxed);

    // Initialize SpeakerStream (independent speaker identification with Bayesian tracking).
    if (wlecapa_enc_.initialized()) {
        bool dual_encoder = speaker_enc_.initialized();
        SpeakerStreamConfig spk_cfg;
        spk_cfg.stride_samples       = 20000;   // 1.25s stride
        spk_cfg.window_samples       = 24000;   // 1.5s window
        spk_cfg.min_samples          = 16000;   // 1.0s minimum
        spk_cfg.bayesian_alpha       = 6.0f;
        spk_cfg.transition_prob      = 0.12f;
        spk_cfg.change_threshold     = 0.65f;
        // Dual-encoder fusion: fused cosine sim ≈ avg(wl_sim, cam_sim), typically
        // 0.10-0.15 lower than single-encoder sims. Calibrated thresholds:
        // same-speaker fused sims ~0.50-0.65, cross-speaker ~0.30-0.45.
        spk_cfg.confirm_threshold    = dual_encoder ? 0.50f : cfg_.wavlm_ecapa_threshold;
        spk_stream_.init(&wlecapa_enc_, &wlecapa_db_, &spk_timeline_, spk_cfg,
                         dual_encoder ? &speaker_enc_ : nullptr);

        // Adjust SpeakerVectorStore thresholds for dual-encoder fusion.
        // Fused cosine sims are lower (averaged across two encoders), so
        // pending confirmation and margin guards need proportional reduction.
        if (dual_encoder) {
            wlecapa_db_.set_pending_threshold(0.42f);  // was 0.50 for single-encoder
            wlecapa_db_.set_min_margin(0.08f);         // was 0.12 for single-encoder
            wlecapa_db_.set_proximity_margin(0.08f);   // tighter than 0.20 for fused space
            wlecapa_db_.set_match_margin(0.0f);        // disabled — too crude for few speakers
        }

        // Wire SPK events for timeline logging.
        spk_stream_.set_on_spk_event([this](const SpeakerEvent& ev) {
            if (on_spk_event_) on_spk_event_(ev);
        });

        // Wire speaker match for stats reporting.
        spk_stream_.set_on_speaker([this](const SpeakerMatch& m) {
            stats_.wlecapa_id = m.speaker_id;
            stats_.wlecapa_sim = m.similarity;
            stats_.wlecapa_new = m.is_new;
            stats_.wlecapa_count = wlecapa_db_.count();
            stats_.wlecapa_exemplars = m.exemplar_count;
            stats_.wlecapa_hits_above = m.hits_above;
            stats_.wlecapa_active = true;
            // SpeakerStream now manages its own VAD state internally.
            // last_was_full() = true means end-of-speech extraction (SPK_FULL).
            stats_.wlecapa_is_early = !spk_stream_.last_was_full();
            stats_.wlecapa_lat_total_ms = wlecapa_enc_.last_lat_total_ms();
            stats_.wlecapa_lat_cnn_ms = wlecapa_enc_.last_lat_cnn_ms();
            stats_.wlecapa_lat_encoder_ms = wlecapa_enc_.last_lat_encoder_ms();
            stats_.wlecapa_lat_ecapa_ms = wlecapa_enc_.last_lat_ecapa_ms();
            strncpy(stats_.wlecapa_name, m.name.c_str(), sizeof(stats_.wlecapa_name) - 1);
            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
            if (on_speaker_) on_speaker_(m);
        });

        LOG_INFO("AudioPipe", "SpeakerStream initialized (Bayesian tracking, stride=%.2fs)",
                 spk_cfg.stride_samples / 16000.0f);
    }

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
                R"("raw_text":"%s","text":"%s"})",
                job.trigger_reason.c_str(), job.audio_duration_sec,
                result.mel_ms, result.mel_frames, result.encoder_ms, result.encoder_out_len,
                result.decode_ms, result.token_count, result.postprocess_ms, result.total_ms,
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
        total_samples_in_ += n_samples;

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
                    on_vad_(vr, stats_.pcm_samples_in / 16000.0f);
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

        // Speaker identification: two independent paths.
        //   1. CAM++: VAD-gated segments (needs fbank accumulation during speech)
        //   2. WL-ECAPA (SpeakerStream): receives ALL audio with VAD metadata,
        //      internally decides what to use. Produces timeline events independently.
        bool cam_enabled = speaker_enc_.initialized() && enable_speaker_.load(std::memory_order_relaxed);
        bool wlecapa_enabled = wlecapa_enc_.initialized() && enable_wlecapa_.load(std::memory_order_relaxed);

        // Clear active flags each tick — only set true when extraction happens.
        stats_.speaker_active = false;
        stats_.wlecapa_active = false;
        stats_.wlecapa_change_valid = false;
        stats_.asr_active = false;

        // Determine speech state from selected VAD source (shared by both paths).
        bool spk_vad_speech = false;
        {
            VadSource src = static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed));
            switch (src) {
                case VadSource::SILERO: spk_vad_speech = stats_.silero_speech; break;
                case VadSource::FSMN:   spk_vad_speech = stats_.fsmn_speech; break;
                case VadSource::TEN:    spk_vad_speech = stats_.ten_speech; break;
                case VadSource::ANY:
                default:
                    spk_vad_speech = stats_.is_speech || stats_.silero_speech ||
                                     stats_.fsmn_speech || stats_.ten_speech;
                    break;
            }
        }

        // ── WL-ECAPA / SpeakerStream: push every tick unconditionally ──
        // SpeakerStream uses is_speech internally to gate what PCM it accumulates
        // and when to trigger extraction. This is the independent speaker pipeline.
        if (wlecapa_enabled) {
            spk_stream_.push_audio(pcm_buf.data(), n_samples, total_samples_in_, spk_vad_speech);
        }

        // ── CAM++: VAD-gated segments (legacy path, kept separate) ──
        if (cam_enabled) {
            if (spk_vad_speech && !in_speech_segment_) {
                in_speech_segment_ = true;
                speech_pcm_buf_.clear();
                speaker_fbank_.reset();
            }
            if (in_speech_segment_) {
                speech_pcm_buf_.insert(speech_pcm_buf_.end(),
                                       pcm_buf.data(), pcm_buf.data() + n_samples);
                speaker_fbank_.push_pcm(pcm_buf.data(), n_samples);
                if (speech_pcm_buf_.size() > 160000) {
                    speech_pcm_buf_.erase(speech_pcm_buf_.begin(),
                                          speech_pcm_buf_.begin() + n_samples);
                }
            }
            if (!spk_vad_speech && in_speech_segment_) {
                in_speech_segment_ = false;
                int speech_samples = (int)speech_pcm_buf_.size();
                float speech_duration = speech_samples / 16000.0f;

                LOG_INFO("AudioPipe", "Speech segment ended: %.2fs (%d samples)",
                         speech_duration, speech_samples);

                int fbank_frames = speaker_fbank_.frames_ready();
                std::vector<float> fbank_host;
                if (fbank_frames >= 100) {
                    fbank_host.resize(fbank_frames * 80);
                    speaker_fbank_.read_fbank(fbank_host.data(), fbank_frames);
                }

                if (fbank_frames >= 150) {
                    float thresh = speaker_threshold_.load(std::memory_order_relaxed);
                    auto emb = speaker_enc_.extract(fbank_host.data(), fbank_frames);
                    if (!emb.empty()) {
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

                speech_pcm_buf_.clear();
            }
        }


        // ASR: accumulate ALL audio continuously. Whisper encoder naturally handles
        // silence and mixed audio. VAD is used only as a trigger hint (WHEN to fire
        // transcription) and to track speech content ratio for filtering.
        if (asr_engine_ && asr_engine_->is_loaded() &&
            enable_asr_.load(std::memory_order_relaxed)) {

            // Determine speech state FIRST (before accumulation) for VAD gap detection.
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

            // VAD gap detection: speech resuming after a non-trivial silence.
            // This is the key decoupling mechanism — natural pauses become ASR
            // segment boundaries regardless of whether speaker changed.
            // Without this, asr_post_silence_ resets when new speech arrives,
            // potentially merging two different speakers' utterances into one segment.
            int ASR_MIN_SAMPLES_EARLY = asr_min_samples_.load(std::memory_order_relaxed);
            if (asr_vad_speech && asr_saw_speech_ && asr_post_silence_ > 0) {
                int gap_ms = asr_post_silence_ * cfg_.process_chunk_ms;
                int gap_min = asr_vad_gap_ms_.load(std::memory_order_relaxed);
                if (gap_min > 0 && gap_ms >= gap_min &&
                    (int)asr_pcm_buf_.size() >= ASR_MIN_SAMPLES_EARLY) {
                    // Queue a split at the current buffer end (before new speech chunk).
                    asr_vad_gap_pending_ = true;
                    asr_vad_gap_split_at_ = (int)asr_pcm_buf_.size();
                    LOG_INFO("AudioPipe", "VAD gap: %dms silence detected, queuing ASR split at %d samples",
                             gap_ms, asr_vad_gap_split_at_);
                }
            }

            // Always accumulate ALL audio (speech + silence).
            asr_pcm_buf_.insert(asr_pcm_buf_.end(),
                                pcm_buf.data(), pcm_buf.data() + n_samples);

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
            // (a) Post-speech silence reaches adaptive threshold, or
            // (b) Buffer exceeds max size during continuous speech, or
            // (c) Speaker change detected (SAAS split).
            int base_silence_ms = asr_post_silence_ms_.load(std::memory_order_relaxed);
            int effective_silence_ms = base_silence_ms;

            // Speaker: adaptive post-silence based on current buffer length.
            if (asr_adaptive_silence_.load(std::memory_order_relaxed) && asr_saw_speech_) {
                float buf_sec = asr_pcm_buf_.size() / 16000.0f;
                if (buf_sec < 0.8f) {
                    effective_silence_ms = asr_adaptive_short_ms_.load(std::memory_order_relaxed);
                } else if (buf_sec > 15.0f) {
                    effective_silence_ms = asr_adaptive_vlong_ms_.load(std::memory_order_relaxed);
                } else if (buf_sec > 5.0f) {
                    effective_silence_ms = asr_adaptive_long_ms_.load(std::memory_order_relaxed);
                }
                // else: use base value (0.8-5s range)
            }

            int asr_post_silence_chunks = effective_silence_ms / cfg_.process_chunk_ms;
            if (asr_post_silence_chunks < 1) asr_post_silence_chunks = 1;
            int ASR_MAX_BUF_SAMPLES = asr_max_buf_samples_.load(std::memory_order_relaxed);
            int ASR_MIN_SAMPLES = asr_min_samples_.load(std::memory_order_relaxed);
            int ASR_PRE_ROLL_SAMPLES = asr_pre_roll_samples_.load(std::memory_order_relaxed);

            // Update stats with effective silence for frontend display.
            stats_.asr_effective_silence_ms = effective_silence_ms;
            stats_.asr_post_silence_ms = asr_post_silence_ * cfg_.process_chunk_ms;

            bool asr_trigger = false;
            std::string trigger_reason;

            // VAD gap split: speech resumed after a non-trivial silence.
            // Split the ASR buffer at the gap boundary so each speech segment
            // gets its own ASR transcription with correct speaker attribution.
            if (asr_vad_gap_pending_) {
                asr_vad_gap_pending_ = false;
                int split_at = asr_vad_gap_split_at_;
                int asr_buf_sz = (int)asr_pcm_buf_.size();
                if (split_at > ASR_MIN_SAMPLES && split_at < asr_buf_sz) {
                    int pre_samples = split_at;
                    float pre_duration = pre_samples / 16000.0f;
                    float speech_sec = asr_speech_samples_ / 16000.0f;

                    // Trim trailing silence from pre-gap portion.
                    int trim_samples = pre_samples;
                    {
                        const int window = 1600; // 100ms
                        const float silence_rms = 0.005f;
                        const int tail_margin = 1600;
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
                        if (trimmed >= pre_samples * 3 / 5) {
                            trim_samples = trimmed;
                        }
                    }
                    float trimmed_duration = trim_samples / 16000.0f;

                    // Energy check: skip if too quiet.
                    float min_energy = asr_min_energy_.load(std::memory_order_relaxed);
                    bool has_energy = true;
                    if (min_energy > 0.0f && trim_samples > 0) {
                        double energy_sum = 0;
                        for (int i = 0; i < trim_samples; i++) {
                            float s = asr_pcm_buf_[i] / 32768.0f;
                            energy_sum += s * s;
                        }
                        float avg_energy = (float)std::sqrt(energy_sum / trim_samples);
                        has_energy = (avg_energy >= min_energy);
                    }

                    if (has_energy && speech_sec >= 0.2f) {
                        std::vector<float> pcm_f32(trim_samples);
                        for (int i = 0; i < trim_samples; i++)
                            pcm_f32[i] = asr_pcm_buf_[i] / 32768.0f;

                        if (on_asr_log_) {
                            char json[512];
                            snprintf(json, sizeof(json),
                                R"({"stage":"trigger","reason":"vad_gap","buf_sec":%.2f,"trimmed_sec":%.2f,"speech_sec":%.2f,"split_at":%d})",
                                pre_duration, trimmed_duration, speech_sec, split_at);
                            on_asr_log_(json);
                        }

                        // Resolve speaker from timeline.
                        int64_t asr_audio_start = total_samples_in_ - (int64_t)asr_pcm_buf_.size();
                        int64_t asr_audio_end = asr_audio_start + split_at;
                        auto resolved = spk_timeline_.resolve(asr_audio_start, asr_audio_end);
                        int spk_id = resolved.speaker_id;
                        float spk_sim = resolved.similarity;
                        float spk_conf = resolved.confidence;
                        std::string spk_name(resolved.name);
                        std::string spk_source;
                        {
                            static const char* kSN[] = {"SPK_EARLY","SPK_FULL","SPK_CHANGE"};
                            if (spk_id < 0) {
                                spk_id = spk_stream_.current_speaker_id();
                                spk_sim = spk_stream_.current_confidence();
                                spk_conf = (spk_id >= 0) ? 0.10f : 0.0f;
                                spk_name = "";
                                spk_source = "SNAPSHOT";
                            } else {
                                spk_source = kSN[static_cast<int>(resolved.source)];
                                LOG_INFO("AudioPipe", "Timeline(vad_gap): resolved spk=%d %s (sim=%.3f, src=%s) for %.2f-%.2fs",
                                         spk_id, spk_name.c_str(), spk_sim,
                                         spk_source.c_str(),
                                         asr_audio_start / 16000.0f, asr_audio_end / 16000.0f);
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lock(asr_mutex_);
                            ASRJob job;
                            job.pcm_f32 = std::move(pcm_f32);
                            job.audio_duration_sec = trimmed_duration;
                            job.trigger_reason = "vad_gap";
                            job.is_partial = false;
                            job.stream_start_sec = asr_audio_start / 16000.0f;
                            job.stream_end_sec   = asr_audio_end / 16000.0f;
                            job.speaker_id = spk_id;
                            job.speaker_name = std::move(spk_name);
                            job.speaker_sim = spk_sim;
                            job.speaker_confidence = spk_conf;
                            job.speaker_source = std::move(spk_source);
                            asr_queue_.push(std::move(job));
                        }
                        asr_cv_.notify_one();

                        LOG_INFO("AudioPipe", "VAD gap: ASR split done, pre=%.2fs (trimmed=%.2fs), remaining=%d samples",
                                 pre_duration, trimmed_duration, asr_buf_sz - split_at);
                    } else {
                        LOG_INFO("AudioPipe", "VAD gap: ASR split skipped (energy=%d speech=%.2fs)", has_energy, speech_sec);
                    }

                    // Keep post-gap audio (new speech).
                    asr_pcm_buf_.erase(asr_pcm_buf_.begin(),
                                       asr_pcm_buf_.begin() + split_at);
                    asr_saw_speech_ = true;  // new segment starts with speech
                    asr_post_silence_ = 0;
                    asr_speech_samples_ = 0;
                    asr_partial_sent_at_ = 0;
                } else {
                    asr_vad_gap_pending_ = false;
                    if (split_at <= ASR_MIN_SAMPLES) {
                        LOG_INFO("AudioPipe", "VAD gap: skipped (pre-gap too short: %d samples)", split_at);
                    }
                }
            }

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
                    // Resolve speaker label via timeline.
                    int64_t asr_audio_start = total_samples_in_ - (int64_t)asr_pcm_buf_.size();
                    int64_t asr_audio_end = total_samples_in_;
                    auto resolved = spk_timeline_.resolve(asr_audio_start, asr_audio_end);
                    int spk_id = resolved.speaker_id;
                    float spk_sim = resolved.similarity;
                    float spk_conf = resolved.confidence;
                    std::string spk_name(resolved.name);
                    std::string spk_source;
                    static const char* kSourceNames[] = {"SPK_EARLY","SPK_FULL","SPK_CHANGE"};
                    // Fallback: if timeline has no result, use current SAAS snapshot.
                    if (spk_id < 0) {
                        spk_id = stats_.wlecapa_id;
                        spk_sim = stats_.wlecapa_sim;
                        spk_conf = (spk_id >= 0) ? 0.10f : 0.0f;
                        spk_name = std::string(stats_.wlecapa_name);
                        spk_source = "SNAPSHOT";
                    } else {
                        spk_source = kSourceNames[static_cast<int>(resolved.source)];
                        LOG_INFO("AudioPipe", "Timeline: resolved spk=%d %s (sim=%.3f, conf=%.3f, src=%s) for %.2f-%.2fs",
                                 spk_id, spk_name.c_str(), spk_sim, spk_conf,
                                 spk_source.c_str(),
                                 asr_audio_start / 16000.0f, asr_audio_end / 16000.0f);
                    }
                    {
                        std::lock_guard<std::mutex> lock(asr_mutex_);
                        ASRJob job;
                        job.pcm_f32 = std::move(pcm_f32);
                        job.audio_duration_sec = trimmed_duration;
                        job.trigger_reason = trigger_reason;
                        job.is_partial = false;
                        job.stream_start_sec = asr_audio_start / 16000.0f;
                        job.stream_end_sec   = asr_audio_end / 16000.0f;
                        job.speaker_id = spk_id;
                        job.speaker_name = std::move(spk_name);
                        job.speaker_sim = spk_sim;
                        job.speaker_confidence = spk_conf;
                        job.speaker_source = std::move(spk_source);
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
                    on_vad_(vr, stats_.pcm_samples_in / 16000.0f);
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
