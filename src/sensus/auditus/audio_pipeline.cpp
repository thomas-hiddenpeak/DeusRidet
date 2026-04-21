// audio_pipeline.cpp — Real-time audio processing pipeline implementation.
//
// Processing thread: pull PCM from ring buffer → push to MelSpectrogram (GPU)
// → VAD on computed frames → report stats/events via callbacks.

#include "audio_pipeline.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"
#include "../../orator/spectral_cluster.h"

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

    // Initialize FRCRN speech enhancement (optional — non-fatal).
    if (!cfg_.frcrn.weights_dir.empty() && cfg_.frcrn.enabled) {
        if (!frcrn_.init(cfg_.frcrn)) {
            LOG_WARN("AudioPipe", "FRCRN speech enhancement init failed — running without denoising");
        }
    }

    // Initialize P1: overlap detector (optional — non-fatal).
    if (!cfg_.overlap_det.model_path.empty() && cfg_.overlap_det.enabled) {
        if (!tracker_.init_overlap_det(cfg_.overlap_det)) {
            LOG_WARN("AudioPipe", "Overlap detector init failed — using heuristic fallback");
        }
    }

    // Initialize P2: speech separator (optional — lazy loaded).
    if (!cfg_.separator.model_path.empty()) {
        if (!tracker_.init_separator(cfg_.separator)) {
            LOG_WARN("AudioPipe", "Speech separator init failed");
        }
    }

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
            // Initialize SpeakerTracker (shares encoder, independent DB).
            tracker_.init(&wlecapa_enc_);
            // Enable dual-encoder matching (384D = CAM++ 192D + WL-ECAPA 192D).
            use_dual_encoder_ = true;
            LOG_INFO("AudioPipe", "Dual-encoder matching enabled (384D)");
        }
    }

    speaker_threshold_.store(cfg_.speaker_threshold, std::memory_order_relaxed);
    speaker_register_threshold_.store(0.60f, std::memory_order_relaxed);

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
                                                 job.tracker_id, job.tracker_name,
                                                 job.tracker_sim,
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

    LOG_INFO("AudioPipe", "Process loop: chunk=%d samples (%d ms), frcrn_loaded=%s frcrn_enabled=%s silero_loaded=%s silero_enabled=%s fsmn_loaded=%s fsmn_enabled=%s",
             chunk_samples, cfg_.process_chunk_ms,
             frcrn_.initialized() ? "ON" : "OFF",
             enable_frcrn_.load(std::memory_order_relaxed) ? "ON" : "OFF",
             silero_.initialized() ? "ON" : "OFF",
             enable_silero_.load(std::memory_order_relaxed) ? "ON" : "OFF",
             fsmn_.initialized() ? "ON" : "OFF",
             enable_fsmn_.load(std::memory_order_relaxed) ? "ON" : "OFF");

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
        // Advance AUDIO T1: the authoritative "now" on the processing side.
        audio_t1_processed_ += (uint64_t)n_samples;
        stats_.audio_t1_processed = audio_t1_processed_;
        stats_.audio_t1_in        = audio_t1_in_.load(std::memory_order_relaxed);

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

        // FRCRN speech enhancement: denoise PCM before all downstream processing.
        // Uses direct per-chunk enhancement (no accumulation latency).
        // FRCRN internally pads to valid STFT alignment.
        //
        // NOTE: the speaker tracker also needs the *pre-FRCRN* signal for
        // MossFormer2 separation in overlap regions (S3: FRCRN suppresses
        // the weaker speaker, which defeats separation). We stash a raw
        // copy here and hand it to tracker_.feed() alongside the denoised
        // buffer below. The copy is skipped entirely when FRCRN is off.
        stats_.frcrn_active = false;
        std::vector<int16_t> pcm_raw_buf;
        if (frcrn_.initialized() && enable_frcrn_.load(std::memory_order_relaxed)) {
            pcm_raw_buf.assign(pcm_buf.begin(), pcm_buf.begin() + n_samples);
            frcrn_.enhance_inplace(pcm_buf.data(), n_samples);
            stats_.frcrn_active = true;
            stats_.frcrn_lat_ms = frcrn_.last_latency_ms();
        }

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
                    on_vad_(vr, (int)stats_.mel_frames, audio_t1_processed_);
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

        // Buffer PCM for speaker identification during speech (VAD-gated).
        bool any_speaker_enabled =
            (speaker_enc_.initialized() && enable_speaker_.load(std::memory_order_relaxed)) ||
            (wlecapa_enc_.initialized() && enable_wlecapa_.load(std::memory_order_relaxed));
        bool need_segment_pcm = any_speaker_enabled;

        // Clear active flags each tick — only set true when extraction happens.
        stats_.speaker_active = false;
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
                case VadSource::ANY:
                default:
                    vad_speech = stats_.is_speech || stats_.silero_speech ||
                                 stats_.fsmn_speech;
                    break;
            }
            if (vad_speech && !in_speech_segment_) {
                in_speech_segment_ = true;
                early_extracted_   = false;
                campp_early_extracted_ = false;
                seg_has_ref_       = false;
                seg_last_recheck_at_ = 0;
                seg_ref_emb_.clear();
                seg_ref_speaker_id_ = -1;
                seg_ref_speaker_name_.clear();
                seg_ref_speaker_sim_ = 0.0f;
                speech_pcm_buf_.clear();
                speaker_fbank_.reset();
                seg_fbank_buf_.clear();
                seg_overlap_chunks_ = 0;
                seg_total_chunks_ = 0;

                // SAAS: short-segment speaker inheritance.
                // If this new segment starts within a short gap of the previous one,
                // pre-populate the speaker ID so that very short utterances (< 1.0s)
                // that can't extract their own embedding get a reasonable speaker label.
                int64_t gap_samples = audio_t1_processed_ - prev_seg_end_t1_;
                float gap_sec = gap_samples / 16000.0f;
                if (prev_seg_speaker_id_ >= 0 && gap_sec < 0.8f) {
                    // Populate both CAM++ and WL-ECAPA stats for inheritance.
                    stats_.speaker_id = prev_seg_speaker_id_;
                    stats_.speaker_sim = prev_seg_speaker_sim_;
                    stats_.speaker_new = false;
                    strncpy(stats_.speaker_name, prev_seg_speaker_name_.c_str(),
                            sizeof(stats_.speaker_name) - 1);
                    stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                    stats_.wlecapa_id = prev_seg_speaker_id_;
                    stats_.wlecapa_sim = prev_seg_speaker_sim_;
                    stats_.wlecapa_new = false;
                    strncpy(stats_.wlecapa_name, prev_seg_speaker_name_.c_str(),
                            sizeof(stats_.wlecapa_name) - 1);
                    stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                    seg_ref_speaker_id_ = prev_seg_speaker_id_;
                    seg_ref_speaker_name_ = prev_seg_speaker_name_;
                    seg_ref_speaker_sim_ = prev_seg_speaker_sim_;
                    LOG_INFO("AudioPipe", "SAAS: inherited spk=%d (%s) from prev segment (gap=%.3fs)",
                             prev_seg_speaker_id_, prev_seg_speaker_name_.c_str(), gap_sec);
                    // Timeline: SAAS inheritance event (covers ~2s from onset).
                    {
                        SpeakerEvent ev{};
                        ev.audio_start = audio_t1_processed_;
                        ev.audio_end   = audio_t1_processed_ + 32000;  // 2s look-ahead
                        ev.source      = SpkEventSource::SAAS_INHERIT;
                        ev.speaker_id  = prev_seg_speaker_id_;
                        ev.similarity  = prev_seg_speaker_sim_;
                        strncpy(ev.name, prev_seg_speaker_name_.c_str(), sizeof(ev.name) - 1);
                        spk_timeline_.push(ev);
                    }
                } else {
                    // Reset speaker ID for new segment.
                    stats_.speaker_id = -1;
                    stats_.speaker_sim = 0.0f;
                    stats_.speaker_new = false;
                    stats_.speaker_name[0] = '\0';
                    stats_.wlecapa_id = -1;
                    stats_.wlecapa_sim = 0.0f;
                    stats_.wlecapa_new = false;
                    strncpy(stats_.wlecapa_name, "", sizeof(stats_.wlecapa_name));
                }
            }
            if (in_speech_segment_) {
                speech_pcm_buf_.insert(speech_pcm_buf_.end(),
                                       pcm_buf.data(), pcm_buf.data() + n_samples);
                // Track overlap during this speech segment.
                seg_total_chunks_++;
                if (stats_.overlap_detected) seg_overlap_chunks_++;
                if ((speaker_enc_.initialized() &&
                     enable_speaker_.load(std::memory_order_relaxed))) {
                    speaker_fbank_.push_pcm(pcm_buf.data(), n_samples);
                    // Accumulate fbank frames for CAM++ EARLY/FULL extraction.
                    int avail = speaker_fbank_.frames_ready();
                    if (avail > 0) {
                        size_t old_sz = seg_fbank_buf_.size();
                        seg_fbank_buf_.resize(old_sz + avail * 80);
                        speaker_fbank_.read_fbank(seg_fbank_buf_.data() + old_sz, avail);
                    }
                }
                // Limit to 10 seconds (160000 samples @ 16kHz).
                if (speech_pcm_buf_.size() > 160000) {
                    speech_pcm_buf_.erase(speech_pcm_buf_.begin(),
                                          speech_pcm_buf_.begin() + n_samples);
                }

                // CAM++ EARLY extraction: when >= 150 fbank frames accumulated
                // during speech, extract embedding and match against existing speakers.
                // No auto-register — avoids spurious registrations from short clips.
                if (!campp_early_extracted_ &&
                    !use_dual_encoder_ &&  // skip EARLY speaker when dual-encoder active
                    enable_early_.load(std::memory_order_relaxed) &&
                    speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed)) {
                    int fbank_frames = (int)(seg_fbank_buf_.size() / 80);
                    if (fbank_frames >= 150) {
                        campp_early_extracted_ = true;
                        auto emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
                        if (!emb.empty()) {
                            float thresh = speaker_threshold_.load(std::memory_order_relaxed);
                            float reg_thresh = speaker_register_threshold_.load(std::memory_order_relaxed);
                            SpeakerMatch match = campp_db_.identify(emb, thresh, /*auto_register=*/false, reg_thresh);
                            if (match.speaker_id >= 0) {
                                stats_.speaker_id = match.speaker_id;
                                stats_.speaker_sim = match.similarity;
                                stats_.speaker_new = false;
                                stats_.speaker_count = campp_db_.count();
                                strncpy(stats_.speaker_name, match.name.c_str(),
                                        sizeof(stats_.speaker_name) - 1);
                                stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                                seg_ref_speaker_id_ = match.speaker_id;
                                seg_ref_speaker_name_ = match.name;
                                seg_ref_speaker_sim_ = match.similarity;
                                LOG_INFO("AudioPipe", "CAM++(early): id=%d sim=%.3f %s (fbank=%d)",
                                         match.speaker_id, match.similarity,
                                         match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                         fbank_frames);
                                if (on_speaker_) on_speaker_(match);
                                // Timeline: SAAS early event.
                                SpeakerEvent ev{};
                                ev.audio_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                ev.audio_end   = audio_t1_processed_;
                                ev.source      = SpkEventSource::SAAS_EARLY;
                                ev.speaker_id  = match.speaker_id;
                                ev.similarity  = match.similarity;
                                strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                                spk_timeline_.push(ev);
                            } else {
                                LOG_INFO("AudioPipe", "CAM++(early): no match (best_sim=%.3f, fbank=%d)",
                                         match.similarity, fbank_frames);
                            }
                        }
                    }
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
                        // Store as reference for intra-segment speaker change detection.
                        seg_ref_emb_ = emb;
                        seg_has_ref_ = true;
                        seg_last_recheck_at_ = (int)speech_pcm_buf_.size();

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
                            // SAAS: track speaker ref for ASR annotation and inheritance.
                            // When dual encoder active, EARLY uses wlecapa_db_ (different ID space
                            // from dual_db_). Don't contaminate seg_ref or timeline.
                            if (!use_dual_encoder_) {
                                seg_ref_speaker_id_ = match.speaker_id;
                                seg_ref_speaker_name_ = match.name;
                                seg_ref_speaker_sim_ = match.similarity;
                            }
                            strncpy(stats_.wlecapa_name, match.name.c_str(),
                                    sizeof(stats_.wlecapa_name) - 1);
                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "WL-ECAPA(early): id=%d sim=%.3f %s (%.2fs, %.1fms)%s",
                                     match.speaker_id, match.similarity,
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     early_samples / 16000.0f,
                                     wlecapa_enc_.last_lat_total_ms(),
                                     use_dual_encoder_ ? " [skip timeline: dual mode]" : "");
                            if (on_speaker_) on_speaker_(match);
                            // Timeline: SAAS early extraction event.
                            // Skip when dual encoder active — wlecapa_db_ IDs != dual_db_ IDs.
                            if (!use_dual_encoder_) {
                                SpeakerEvent ev{};
                                ev.audio_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                ev.audio_end   = audio_t1_processed_;
                                ev.source      = SpkEventSource::SAAS_EARLY;
                                ev.speaker_id  = match.speaker_id;
                                ev.similarity  = match.similarity;
                                strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                                spk_timeline_.push(ev);
                            }
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

                // Intra-segment speaker change detection: periodically re-extract
                // an embedding from the recent audio window and compare against the
                // segment's reference speaker. If similarity drops below threshold,
                // force a segment boundary — this catches speaker transitions that
                // VAD misses (rapid turn-taking without silence).
                if (early_extracted_ && seg_has_ref_ &&
                    enable_spk_recheck_.load(std::memory_order_relaxed) &&
                    wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed)) {
                    int recheck_interval = spk_recheck_samples_.load(std::memory_order_relaxed);
                    int buf_sz = (int)speech_pcm_buf_.size();
                    if (buf_sz - seg_last_recheck_at_ >= recheck_interval) {
                        seg_last_recheck_at_ = buf_sz;
                        int win_samples = spk_recheck_window_samples_.load(std::memory_order_relaxed);
                        int start = std::max(0, buf_sz - win_samples);
                        int len = buf_sz - start;
                        if (len >= 16000) { // at least 1s for meaningful embedding
                            std::vector<float> pcm_f32(len);
                            for (int i = 0; i < len; i++)
                                pcm_f32[i] = speech_pcm_buf_[start + i] / 32768.0f;
                            auto emb = wlecapa_enc_.extract(pcm_f32.data(), len);
                            if (!emb.empty() && emb.size() == seg_ref_emb_.size()) {
                                // Cosine similarity between reference and current window.
                                float dot = 0, na = 0, nb = 0;
                                for (size_t i = 0; i < emb.size(); i++) {
                                    dot += seg_ref_emb_[i] * emb[i];
                                    na  += seg_ref_emb_[i] * seg_ref_emb_[i];
                                    nb  += emb[i] * emb[i];
                                }
                                float sim = (na > 0 && nb > 0) ? dot / (sqrtf(na) * sqrtf(nb)) : 0.0f;
                                float change_thresh = spk_change_threshold_.load(std::memory_order_relaxed);
                                LOG_INFO("AudioPipe", "SPK-RECHECK: sim=%.3f (thresh=%.3f) at %.2fs in segment",
                                         sim, change_thresh, buf_sz / 16000.0f);
                                if (sim < change_thresh) {
                                    // Speaker changed mid-segment! Soft restart:
                                    // 1. Run full WL-ECAPA on pre-change audio (auto_register=true)
                                    // 2. Save speaker state for inheritance
                                    // 3. Trigger ASR split
                                    // 4. Keep in_speech_segment_=true, carry tail audio forward
                                    LOG_INFO("AudioPipe", "SPK-CHANGE detected (sim=%.3f < %.3f) — soft restart at %.2fs",
                                             sim, change_thresh, buf_sz / 16000.0f);

                                    // --- 1. Full WL-ECAPA on pre-change audio ---
                                    int pre_samples = start;
                                    int pre_min = min_speech_samples_.load(std::memory_order_relaxed);
                                    if (pre_samples >= pre_min) {
                                        std::vector<float> pre_f32(pre_samples);
                                        for (int i = 0; i < pre_samples; i++)
                                            pre_f32[i] = speech_pcm_buf_[i] / 32768.0f;
                                        auto pre_emb = wlecapa_enc_.extract(pre_f32.data(), pre_samples);
                                        if (!pre_emb.empty()) {
                                            float wt = wlecapa_threshold_.load(std::memory_order_relaxed);
                                            SpeakerMatch m = wlecapa_db_.identify(pre_emb, wt);
                                            // When dual encoder active, don't pollute seg_ref with
                                            // wlecapa_db_ IDs (different ID space from dual_db_).
                                            if (!use_dual_encoder_) {
                                                seg_ref_speaker_id_ = m.speaker_id;
                                                seg_ref_speaker_name_ = m.name;
                                                seg_ref_speaker_sim_ = m.similarity;
                                            }
                                            stats_.wlecapa_id = m.speaker_id;
                                            stats_.wlecapa_sim = m.similarity;
                                            stats_.wlecapa_new = m.is_new;
                                            stats_.wlecapa_count = wlecapa_db_.count();
                                            stats_.wlecapa_exemplars = m.exemplar_count;
                                            stats_.wlecapa_hits_above = m.hits_above;
                                            stats_.wlecapa_is_early = false;
                                            strncpy(stats_.wlecapa_name, m.name.c_str(),
                                                    sizeof(stats_.wlecapa_name) - 1);
                                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                                            LOG_INFO("AudioPipe", "SPK-CHANGE full-extract: id=%d sim=%.3f %s%s (%.2fs)",
                                                     m.speaker_id, m.similarity,
                                                     m.is_new ? "NEW " : "",
                                                     m.name.empty() ? "(unnamed)" : m.name.c_str(),
                                                     pre_samples / 16000.0f);
                                            if (on_speaker_) on_speaker_(m);
                                            prev_wlecapa_emb_ = pre_emb;
                                            // Timeline: SAAS speaker change event.
                                            // Skip when dual encoder active — wlecapa_db_ IDs != dual_db_ IDs.
                                            if (!use_dual_encoder_) {
                                                int64_t seg_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                                SpeakerEvent ev{};
                                                ev.audio_start = seg_start;
                                                ev.audio_end   = seg_start + pre_samples;
                                                ev.source      = SpkEventSource::SAAS_CHANGE;
                                                ev.speaker_id  = m.speaker_id;
                                                ev.similarity  = m.similarity;
                                                strncpy(ev.name, m.name.c_str(), sizeof(ev.name) - 1);
                                                spk_timeline_.push(ev);
                                            }
                                        }
                                    }

                                    // --- 2. Save speaker state for inheritance ---
                                    prev_seg_end_t1_ = audio_t1_processed_;
                                    if (seg_ref_speaker_id_ >= 0) {
                                        prev_seg_speaker_id_ = seg_ref_speaker_id_;
                                        prev_seg_speaker_name_ = seg_ref_speaker_name_;
                                        prev_seg_speaker_sim_ = seg_ref_speaker_sim_;
                                    }

                                    // --- 3. Trigger ASR split ---
                                    int tail_samples = buf_sz - start;
                                    int asr_buf_sz = (int)asr_pcm_buf_.size();
                                    int split_at = std::max(0, asr_buf_sz - tail_samples);
                                    if (asr_saw_speech_ && split_at > 0 && split_at < asr_buf_sz) {
                                        asr_spk_change_pending_ = true;
                                        asr_spk_change_split_at_ = split_at;
                                        LOG_INFO("AudioPipe", "SAAS: ASR split queued at sample %d/%d (%.2fs)",
                                                 split_at, asr_buf_sz, split_at / 16000.0f);
                                    }

                                    // --- 4. Soft restart: carry tail audio, reset for new speaker ---
                                    std::vector<int16_t> tail(speech_pcm_buf_.begin() + start,
                                                              speech_pcm_buf_.end());
                                    speech_pcm_buf_ = std::move(tail);
                                    // New speaker's embedding becomes ref
                                    seg_ref_emb_ = emb;
                                    seg_has_ref_ = true;
                                    early_extracted_ = true;
                                    seg_last_recheck_at_ = (int)speech_pcm_buf_.size();
                                    seg_ref_speaker_id_ = -1;
                                    seg_ref_speaker_name_.clear();
                                    seg_ref_speaker_sim_ = 0.0f;
                                    // Reset fbank for new sub-segment
                                    speaker_fbank_.reset();
                                    if (((speaker_enc_.initialized() &&
                                        enable_speaker_.load(std::memory_order_relaxed))) &&
                                        !speech_pcm_buf_.empty()) {
                                        speaker_fbank_.push_pcm(speech_pcm_buf_.data(),
                                                                (int)speech_pcm_buf_.size());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (!vad_speech && in_speech_segment_) {
                in_speech_segment_ = false;
                int speech_samples = (int)speech_pcm_buf_.size();
                float speech_duration = speech_samples / 16000.0f;

                // SAAS: save speaker state for short-segment inheritance.
                prev_seg_end_t1_ = audio_t1_processed_;
                if (seg_ref_speaker_id_ >= 0) {
                    prev_seg_speaker_id_ = seg_ref_speaker_id_;
                    prev_seg_speaker_name_ = seg_ref_speaker_name_;
                    prev_seg_speaker_sim_ = seg_ref_speaker_sim_;
                } else if (stats_.speaker_id >= 0) {
                    // Fallback: use CAM++ stats (primary encoder).
                    prev_seg_speaker_id_ = stats_.speaker_id;
                    prev_seg_speaker_name_ = stats_.speaker_name;
                    prev_seg_speaker_sim_ = stats_.speaker_sim;
                } else if (!use_dual_encoder_ && stats_.wlecapa_id >= 0) {
                    // Further fallback: use whatever WL-ECAPA ID was last reported.
                    // Skip when dual encoder active — wlecapa_db_ IDs != dual_db_ IDs.
                    prev_seg_speaker_id_ = stats_.wlecapa_id;
                    prev_seg_speaker_name_ = stats_.wlecapa_name;
                    prev_seg_speaker_sim_ = stats_.wlecapa_sim;
                }

                LOG_INFO("AudioPipe", "Speech segment ended: %.2fs (%d samples, spk=%d %s)",
                         speech_duration, speech_samples, prev_seg_speaker_id_,
                         prev_seg_speaker_name_.c_str());

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

                // Read remaining fbank features and append to segment accumulator.
                {
                    int avail = speaker_fbank_.frames_ready();
                    if (avail > 0) {
                        size_t old_sz = seg_fbank_buf_.size();
                        seg_fbank_buf_.resize(old_sz + avail * 80);
                        speaker_fbank_.read_fbank(seg_fbank_buf_.data() + old_sz, avail);
                    }
                }
                int fbank_frames = (int)(seg_fbank_buf_.size() / 80);

                // CAM++ speaker encoder — FULL extraction using accumulated fbank.
                // Warm-up spectral clustering: collect first N embeddings,
                // run spectral clustering to find K speakers, rebuild store.
                // Adapted from qwen35-orin spectral clustering (Phase 3b).
                if (speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    fbank_frames >= 150) {
                        float thresh = speaker_threshold_.load(std::memory_order_relaxed);
                        float reg_thresh = speaker_register_threshold_.load(std::memory_order_relaxed);

                        auto emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
                        if (!emb.empty()) {
                            // Overlap guard: when OD detects overlapping speech,
                            // suppress auto-registration to prevent mixed embeddings
                            // from polluting the speaker store.
                            bool overlap_noregister = stats_.overlap_detected;

                            // v25: disable auto-registration after warmup spectral
                            // clustering completes. The warmup already establishes
                            // the expected speaker set (4 speakers via spectral
                            // clustering); any further "new" speakers registered
                            // post-warmup are false-split fragments that destroy
                            // accuracy. Unmatched segments abstain as spk-1 instead.
                            bool auto_reg = !warmup_done_;

                            // v29→v30: Late registration cap — after 100 FULL
                            // identifications all speakers should be registered.
                            // Further registrations are drift clones (e.g. spk5
                            // at 1872s in v8) that create false attributions.
                            // v30: lowered from 200→100 because spk5 registered
                            // at count=181 in v9.
                            static constexpr int kMaxAutoRegCount = 1000;
                            if (campp_full_count_ >= kMaxAutoRegCount) {
                                auto_reg = false;
                            }
                            // Overlap guard: never register new speakers from
                            // overlapping segments — the mixed embedding would
                            // pollute the speaker store.
                            // Exception: during warmup, allow registration even
                            // when OD fires. The warmup must discover all speakers;
                            // blocking registration here caused 石一 (38% of GT)
                            // to never register because OD fires frequently during
                            // her segments. FULL embeddings are VAD-segmented
                            // (predominantly single-speaker), so mixed-embedding
                            // risk is low.
                            if (overlap_noregister && warmup_done_) {
                                auto_reg = false;
                            }
                            float match_thresh = thresh;

                            // v24d: Discovery phase — use higher threshold during
                            // early extractions to force speaker separation.
                            // Without this, similar speakers (e.g. 徐子景/朱杰)
                            // get absorbed into the first registered speaker.
                            static constexpr int kDiscoveryCount = 50;
                            static constexpr float kDiscoveryBoost = 0.07f;
                            if (campp_full_count_ < kDiscoveryCount) {
                                match_thresh += kDiscoveryBoost;  // 0.45 → 0.52
                            }                            // v24: Temporal recency bonus — lower threshold when recent
                            // speaker still active, reducing false negatives (fragmentation).
                            float seg_mid_time = (float)(audio_t1_processed_ - (int64_t)speech_pcm_buf_.size() / 2) / 16000.0f;
                            float time_since_prev = seg_mid_time - prev_full_time_;
                            static constexpr float kRecencyWindow = 15.0f;
                            // v32: reverted to 0.05 — v31's 0.03 hurt spk2
                            // (徐子景 28 vs GT 73). 0.05 was fine in v9.
                            static constexpr float kRecencyBonus  = 0.05f;
                            bool recency_active = (prev_full_speaker_id_ >= 0 &&
                                                   time_since_prev < kRecencyWindow);
                            if (recency_active) {
                                match_thresh -= kRecencyBonus;
                                // v32: restored from v30 — lowered threshold
                                // must NOT allow new-speaker registration.
                                // v12 showed spk1 registering at mt=0.47 and
                                // merging into spk0, scrambling all mappings.
                                auto_reg = false;
                            }

                            SpeakerMatch match;
                            std::vector<float> wl_emb;  // hoisted for warmup reuse
                            if (use_dual_encoder_) {
                                // Dual-encoder: concatenate CAM++ + WL-ECAPA → 384D.
                                int speech_samples = (int)speech_pcm_buf_.size();
                                if (speech_samples >= 16000) {
                                    std::vector<float> pcm_f32(speech_samples);
                                    for (int si = 0; si < speech_samples; si++)
                                        pcm_f32[si] = speech_pcm_buf_[si] / 32768.0f;
                                    wl_emb = wlecapa_enc_.extract(pcm_f32.data(), speech_samples);
                                }
                                if (!wl_emb.empty()) {
                                    // Build 384D vector: [CAM++ 192D | WL-ECAPA 192D], L2-normalized.
                                    std::vector<float> dual(384);
                                    std::copy(emb.begin(), emb.end(), dual.begin());
                                    std::copy(wl_emb.begin(), wl_emb.end(), dual.begin() + 192);
                                    float n2 = 0;
                                    for (float v : dual) n2 += v * v;
                                    float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                    for (float& v : dual) v *= inv;
                                    match = dual_db_.identify(dual, match_thresh,
                                                              auto_reg, reg_thresh);
                                } else {
                                    // WL-ECAPA extraction failed (segment too short).
                                    // Skip — don't fallback to different ID space.
                                    LOG_INFO("AudioPipe", "CAM++ FULL: WL-ECAPA failed, skip dual identify");
                                }
                            } else {
                                match = campp_db_.identify(emb, match_thresh,
                                                           auto_reg, reg_thresh);
                            }

                            // v24: Recency validation — if threshold was lowered and matched
                            // a DIFFERENT speaker than the recent one, discard the match and
                            // re-run at standard threshold to avoid false positives.
                            if (recency_active && match.speaker_id >= 0 &&
                                match.speaker_id != prev_full_speaker_id_ &&
                                match.similarity < thresh) {
                                LOG_INFO("AudioPipe", "Recency: matched #%d(%.3f) != prev #%d, re-check at %.2f",
                                         match.speaker_id, match.similarity, prev_full_speaker_id_, thresh);
                                // Re-identify at standard threshold (reuse wl_emb).
                                if (use_dual_encoder_ && !wl_emb.empty()) {
                                    std::vector<float> dual(384);
                                    std::copy(emb.begin(), emb.end(), dual.begin());
                                    std::copy(wl_emb.begin(), wl_emb.end(), dual.begin() + 192);
                                    float n2 = 0;
                                    for (float v : dual) n2 += v * v;
                                    float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                    for (float& v : dual) v *= inv;
                                    match = dual_db_.identify(dual, thresh, auto_reg, reg_thresh);
                                } else if (!use_dual_encoder_) {
                                    match = campp_db_.identify(emb, thresh, auto_reg, reg_thresh);
                                }
                            }

                            // Margin gate: abstain on ambiguous matches where
                            // top-1 and top-2 are too close to distinguish.
                            // Threshold 0.05 yields ~91% accuracy on test.mp3.
                            static constexpr float kMarginAbstainThresh = 0.05f;
                            if (match.speaker_id >= 0 && !match.is_new &&
                                match.second_best_id >= 0 &&
                                (match.similarity - match.second_best_sim) < kMarginAbstainThresh) {
                                LOG_INFO("AudioPipe", "FULL margin-abstain: id=%d sim=%.3f 2nd=#%d(%.3f) margin=%.3f < %.2f",
                                         match.speaker_id, match.similarity,
                                         match.second_best_id, match.second_best_sim,
                                         match.similarity - match.second_best_sim, kMarginAbstainThresh);
                                match.speaker_id = -1;
                                match.similarity = 0;
                                match.name.clear();
                            }

                            stats_.speaker_id = match.speaker_id;
                            stats_.speaker_sim = match.similarity;
                            stats_.speaker_new = match.is_new;
                            stats_.speaker_count = use_dual_encoder_ ? dual_db_.count() : campp_db_.count();
                            stats_.speaker_active = true;
                            stats_.speaker_exemplars = match.exemplar_count;
                            stats_.speaker_hits_above = match.hits_above;

                            campp_full_count_++;

                            // v24b: Collect embeddings for warmup spectral clustering.
                            // Only collect clean segments (no overlap detected).
                            if (!warmup_done_ && !overlap_noregister) {
                                warmup_embeddings_.push_back(emb);
                                warmup_timestamps_.push_back(seg_mid_time);
                                // Reuse already-extracted WL-ECAPA embedding.
                                if (use_dual_encoder_) {
                                    if (!wl_emb.empty()) {
                                        warmup_wlecapa_embs_.push_back(wl_emb);
                                    } else {
                                        warmup_wlecapa_embs_.push_back(std::vector<float>(192, 0.0f));
                                    }
                                }
                            }
                            strncpy(stats_.speaker_name, match.name.c_str(),
                                    sizeof(stats_.speaker_name) - 1);
                            stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "FULL: id=%d sim=%.3f 2nd=#%d(%.3f) m=%.3f %s%s (fbank=%d, ex=%d, recency=%s, mt=%.2f)",
                                     match.speaker_id, match.similarity,
                                     match.second_best_id, match.second_best_sim,
                                     match.similarity - match.second_best_sim,
                                     match.is_new ? "NEW " : "",
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     fbank_frames, match.exemplar_count,
                                     recency_active ? "ON" : "off", match_thresh);
                            if (on_speaker_) on_speaker_(match);

                            // DEBUG: dump embedding for offline clustering analysis.
                            // Format per record (1560 bytes):
                            //   float32 timestamp, int32 speaker_id, int32 fbank_frames,
                            //   float32 similarity, float32[192] campp, float32[192] wavlm
                            {
                                static FILE* emb_fp = nullptr;
                                if (!emb_fp) emb_fp = fopen("/tmp/spk_embeddings.bin", "ab");
                                if (emb_fp) {
                                    float ts_val = seg_mid_time;
                                    int32_t sid = match.speaker_id;
                                    int32_t fb = fbank_frames;
                                    float sim = match.similarity;
                                    fwrite(&ts_val, 4, 1, emb_fp);
                                    fwrite(&sid, 4, 1, emb_fp);
                                    fwrite(&fb, 4, 1, emb_fp);
                                    fwrite(&sim, 4, 1, emb_fp);
                                    // CAM++ 192D (already L2-normalized by encoder)
                                    if (emb.size() == 192) {
                                        fwrite(emb.data(), 4, 192, emb_fp);
                                    } else {
                                        float zeros[192] = {};
                                        fwrite(zeros, 4, 192, emb_fp);
                                    }
                                    // WavLM-ECAPA 192D
                                    if (wl_emb.size() == 192) {
                                        fwrite(wl_emb.data(), 4, 192, emb_fp);
                                    } else {
                                        float zeros[192] = {};
                                        fwrite(zeros, 4, 192, emb_fp);
                                    }
                                    fflush(emb_fp);
                                }
                            }

                            // Update recency tracking + run-length.
                            if (match.speaker_id >= 0) {
                                if (match.speaker_id == prev_full_speaker_id_) {
                                    speaker_run_length_++;
                                } else {
                                    speaker_run_length_ = 1;
                                }
                                prev_full_speaker_id_ = match.speaker_id;
                                prev_full_time_ = seg_mid_time;
                                prev_full_speaker_name_ = match.name;  // v29
                            }

                            // SAAS: feed result into speaker timeline.
                            if (match.speaker_id >= 0) {
                                seg_ref_speaker_id_ = match.speaker_id;
                                seg_ref_speaker_name_ = match.name;
                                seg_ref_speaker_sim_ = match.similarity;
                                int64_t seg_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                SpeakerEvent ev{};
                                ev.audio_start = seg_start;
                                ev.audio_end   = audio_t1_processed_;
                                ev.source      = SpkEventSource::SAAS_FULL;
                                ev.speaker_id  = match.speaker_id;
                                ev.similarity  = match.similarity;
                                strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                                spk_timeline_.push(ev);
                            }

                            // v24d: No absorb — threshold up to 0.73 between different
                            // speakers makes centroid-based merge unsafe.

                            // v24b: Warm-up spectral clustering with temporal fusion.
                            // After collecting kWarmupCount embeddings, run spectral
                            // clustering to find speaker count and centroids, then
                            // rebuild the speaker store. Temporal fusion (α=0.65)
                            // separates confusable speakers by WHEN they spoke.
                            // Adapted from qwen35-orin offline pipeline (88.7% accuracy).
                            if (!warmup_done_ &&
                                (int)warmup_embeddings_.size() >= kWarmupCount) {
                                int n_emb = (int)warmup_embeddings_.size();
                                bool use_dual_w = use_dual_encoder_ &&
                                                  (int)warmup_wlecapa_embs_.size() == n_emb;
                                int cluster_dim = use_dual_w ? 384 : 192;

                                LOG_INFO("AudioPipe", "=== v24b WARM-UP SPECTRAL CLUSTERING: "
                                         "%d embeddings, %s (%dD) ===",
                                         n_emb, use_dual_w ? "dual 384D" : "CAM++ 192D",
                                         cluster_dim);

                                // Build clustering input.
                                std::vector<std::vector<float>> cluster_input(n_emb);
                                for (int i = 0; i < n_emb; ++i) {
                                    cluster_input[i].resize(cluster_dim);
                                    std::copy(warmup_embeddings_[i].begin(),
                                              warmup_embeddings_[i].end(),
                                              cluster_input[i].begin());
                                    if (use_dual_w) {
                                        std::copy(warmup_wlecapa_embs_[i].begin(),
                                                  warmup_wlecapa_embs_[i].end(),
                                                  cluster_input[i].begin() + 192);
                                    }
                                    // L2-normalize.
                                    float n2 = 0;
                                    for (float v : cluster_input[i]) n2 += v * v;
                                    float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                    for (float& v : cluster_input[i]) v *= inv;
                                }

                                // Spectral clustering with PCA dimension reduction.
                                // Full 384D has noise dims that confuse clustering.
                                // PCA to 32D focuses on discriminative directions.
                                // No temporal fusion — conversation segments alternate
                                // speakers, so temporal proximity would merge different
                                // speakers who spoke close in time.
                                SpectralClusterConfig sc_cfg;
                                sc_cfg.temporal_alpha = 0.0f;   // pure embedding clustering
                                sc_cfg.pca_dim = cluster_dim;   // v15c: full dim (384D)
                                                                // PCA 32D changed cluster balance but didn't help
                                sc_cfg.merge_threshold = 1.0f;  // NO auto-merge
                                sc_cfg.max_k = 6;               // allow up to 6
                                auto cr = spectral_cluster(cluster_input,
                                                           warmup_timestamps_,
                                                           cluster_dim, sc_cfg);

                                LOG_INFO("AudioPipe", "Spectral: K=%d from %d embeddings (α=%.2f, PCA=%d→%dD)",
                                         cr.K, n_emb, sc_cfg.temporal_alpha,
                                         cluster_dim, sc_cfg.pca_dim);

                                // Log cluster sizes and avg timestamps.
                                std::vector<int> sizes(cr.K, 0);
                                std::vector<float> avg_t(cr.K, 0.0f);
                                for (int i = 0; i < n_emb; ++i) {
                                    sizes[cr.labels[i]]++;
                                    avg_t[cr.labels[i]] += warmup_timestamps_[i];
                                }
                                for (int c = 0; c < cr.K; ++c) {
                                    if (sizes[c] > 0) avg_t[c] /= sizes[c];
                                    LOG_INFO("AudioPipe", "  cluster[%d]: %d embeddings, avg_t=%.1fs",
                                             c, sizes[c], avg_t[c]);
                                }

                                // No forced merge — keep K from eigengap.

                                // Rebuild dual_db_ (or campp_db_) with cluster centroids.
                                if (use_dual_w) {
                                    dual_db_.clear();
                                    for (int c = 0; c < cr.K; ++c) {
                                        // Collect per-cluster embeddings.
                                        std::vector<int> members;
                                        for (int i = 0; i < n_emb; ++i)
                                            if (cr.labels[i] == c) members.push_back(i);

                                        if (members.empty()) continue;

                                        // v15d: First-member registration (same as v14).
                                        // Medoid was too generic (v15c: 41% vs v14: 54%).
                                        int anchor = members[0];
                                        std::vector<float> first_emb(384);
                                        for (int d = 0; d < 192; ++d) {
                                            first_emb[d] = warmup_embeddings_[anchor][d];
                                            first_emb[192 + d] = warmup_wlecapa_embs_[anchor][d];
                                        }
                                        float n2 = 0;
                                        for (float v : first_emb) n2 += v * v;
                                        float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                        for (float& v : first_emb) v *= inv;

                                        int id = dual_db_.register_speaker("", first_emb);

                                        // Add up to 14 more exemplars from cluster members.
                                        int added = 0;
                                        for (size_t m = 1; m < members.size() && added < 14; ++m) {
                                            int mi = members[m];
                                            std::vector<float> emb(384);
                                            for (int d = 0; d < 192; ++d) {
                                                emb[d] = warmup_embeddings_[mi][d];
                                                emb[192 + d] = warmup_wlecapa_embs_[mi][d];
                                            }
                                            float n2e = 0;
                                            for (float v : emb) n2e += v * v;
                                            float inve = 1.0f / sqrtf(n2e + 1e-12f);
                                            for (float& v : emb) v *= inve;
                                            dual_db_.add_exemplar(id, emb);
                                            added++;
                                        }

                                        LOG_INFO("AudioPipe", "  registered spk%d from cluster %d (%d members, %d exemplars)",
                                                 id, c, (int)members.size(), added);
                                    }
                                } else {
                                    campp_db_.clear();
                                    for (int c = 0; c < cr.K; ++c) {
                                        std::vector<int> members;
                                        for (int i = 0; i < n_emb; ++i)
                                            if (cr.labels[i] == c) members.push_back(i);

                                        if (members.empty()) continue;

                                        std::vector<float> first_emb(192);
                                        for (int d = 0; d < 192; ++d)
                                            first_emb[d] = warmup_embeddings_[members[0]][d];
                                        float n2 = 0;
                                        for (int d = 0; d < 192; ++d) n2 += first_emb[d] * first_emb[d];
                                        float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                        for (int d = 0; d < 192; ++d) first_emb[d] *= inv;

                                        int id = campp_db_.register_speaker("", first_emb);

                                        int added = 1;
                                        for (size_t m = 1; m < members.size() && added < 15; ++m) {
                                            std::vector<float> emb(192);
                                            for (int d = 0; d < 192; ++d)
                                                emb[d] = warmup_embeddings_[members[m]][d];
                                            float n2e = 0;
                                            for (int d = 0; d < 192; ++d) n2e += emb[d] * emb[d];
                                            float inve = 1.0f / sqrtf(n2e + 1e-12f);
                                            for (int d = 0; d < 192; ++d) emb[d] *= inve;
                                            campp_db_.add_exemplar(id, emb);
                                            added++;
                                        }

                                        LOG_INFO("AudioPipe", "  registered spk%d from cluster %d (%d members, %d exemplars)",
                                                 id, c, (int)members.size(), added);
                                    }
                                }

                                // Reset state for new ID space.
                                prev_full_speaker_id_ = -1;
                                prev_full_time_ = -100.0f;
                                speaker_run_length_ = 0;
                                seg_ref_speaker_id_ = -1;
                                spk_timeline_.clear();

                                warmup_done_ = true;
                                warmup_embeddings_.clear();
                                warmup_wlecapa_embs_.clear();
                                warmup_timestamps_.clear();

                                auto& db = use_dual_encoder_ ? dual_db_ : campp_db_;
                                LOG_INFO("AudioPipe", "=== v24b WARM-UP DONE: %d speakers ===",
                                         db.count());
                            }
                        }
                }
                skip_full_identify:;


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

                        // SAAS: track speaker ref for ASR annotation and inheritance.
                        // When dual encoder active, dual_db_ FULL path already sets seg_ref.
                        // Don't overwrite with wlecapa_db_ IDs.
                        if (!use_dual_encoder_) {
                            seg_ref_speaker_id_ = match.speaker_id;
                            seg_ref_speaker_name_ = match.name;
                            seg_ref_speaker_sim_ = match.similarity;
                        }

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
                        // Timeline: SAAS full end-of-segment event (highest authority).
                        // Skip when dual encoder active — dual_db_ FULL path already pushed.
                        if (!use_dual_encoder_) {
                            int64_t seg_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                            SpeakerEvent ev{};
                            ev.audio_start = seg_start;
                            ev.audio_end   = audio_t1_processed_;
                            ev.source      = SpkEventSource::SAAS_FULL;
                            ev.speaker_id  = match.speaker_id;
                            ev.similarity  = match.similarity;
                            strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                            spk_timeline_.push(ev);
                        }
                    }
                }

                speech_pcm_buf_.clear();
            }
        }

        // SpeakerTracker: independent continuous pipeline.
        // Uses same VAD source as SAAS for consistency but tracks independently.
        {
            VadSource src = static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed));
            bool tracker_vad = false;
            switch (src) {
                case VadSource::SILERO: tracker_vad = stats_.silero_speech; break;
                case VadSource::FSMN:   tracker_vad = stats_.fsmn_speech; break;
                case VadSource::ANY:
                default:
                    tracker_vad = stats_.is_speech || stats_.silero_speech ||
                                  stats_.fsmn_speech;
                    break;
            }
            tracker_.feed(pcm_buf.data(), n_samples, tracker_vad,
                          pcm_raw_buf.empty() ? nullptr : pcm_raw_buf.data());
            if (tracker_.check()) {
                // Timeline: Tracker identification event.
                // Skip when dual encoder active — tracker uses its own DB with
                // different speaker IDs from dual_db_.
                auto& ts = tracker_.stats();
                if (ts.speaker_id >= 0 && !use_dual_encoder_) {
                    int win = tracker_.window_ms() * 16;  // window in samples
                    SpeakerEvent ev{};
                    ev.audio_start = audio_t1_processed_ - win;
                    ev.audio_end   = audio_t1_processed_;
                    ev.source      = SpkEventSource::TRACKER;
                    ev.speaker_id  = ts.speaker_id;
                    ev.similarity  = ts.speaker_sim;
                    strncpy(ev.name, ts.speaker_name, sizeof(ev.name) - 1);
                    spk_timeline_.push(ev);
                }
            }
            // Copy P1/P2 stats from tracker to pipeline stats.
            {
                auto& ts = tracker_.stats();
                stats_.overlap_detected   = ts.overlap_detected;
                stats_.overlap_ratio      = ts.overlap_ratio;
                stats_.od_latency_ms      = ts.od_latency_ms;
                stats_.separation_active  = ts.separation_active;
                stats_.separation_lat_ms  = ts.separation_lat_ms;
                stats_.sep_source1_energy = ts.sep_source1_energy;
                stats_.sep_source2_energy = ts.sep_source2_energy;
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
                case VadSource::DIRECT: asr_vad_speech = true; break;  // always "speech" — trigger on buffer duration
                case VadSource::ANY:
                default:
                    asr_vad_speech = stats_.is_speech || stats_.silero_speech ||
                                     stats_.fsmn_speech;
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
            // (a) Post-speech silence reaches adaptive threshold, or
            // (b) Buffer exceeds max size during continuous speech, or
            // (c) Speaker change detected (SAAS split).
            int base_silence_ms = asr_post_silence_ms_.load(std::memory_order_relaxed);
            int effective_silence_ms = base_silence_ms;

            // SAAS: adaptive post-silence based on current buffer length.
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

            // SAAS: speaker-change-driven ASR split takes priority.
            if (asr_spk_change_pending_) {
                asr_spk_change_pending_ = false;
                int split_at = asr_spk_change_split_at_;
                int asr_buf_sz = (int)asr_pcm_buf_.size();
                if (split_at > ASR_MIN_SAMPLES && split_at < asr_buf_sz) {
                    // Submit the pre-change portion for ASR.
                    int pre_samples = split_at;
                    float pre_duration = pre_samples / 16000.0f;
                    float speech_sec = asr_speech_samples_ / 16000.0f;

                    std::vector<float> pcm_f32(pre_samples);
                    for (int i = 0; i < pre_samples; i++)
                        pcm_f32[i] = asr_pcm_buf_[i] / 32768.0f;

                    if (on_asr_log_) {
                        char json[512];
                        snprintf(json, sizeof(json),
                            R"({"stage":"trigger","reason":"spk_change","buf_sec":%.2f,"speech_sec":%.2f,"split_at":%d})",
                            pre_duration, speech_sec, split_at);
                        on_asr_log_(json);
                    }

                    // Resolve old speaker via timeline for pre-change audio.
                    int64_t asr_audio_start = audio_t1_processed_ - (int64_t)asr_pcm_buf_.size();
                    int64_t asr_audio_end = asr_audio_start + split_at;
                    auto resolved = spk_timeline_.resolve(asr_audio_start, asr_audio_end);
                    int spk_id = resolved.speaker_id;
                    float spk_sim = resolved.similarity;
                    float spk_conf = resolved.confidence;
                    std::string spk_name(resolved.name);
                    std::string spk_source;
                    {
                        static const char* kSN[] = {"SAAS_EARLY","SAAS_FULL","SAAS_CHANGE","SAAS_INHERIT","TRACKER"};
                        // Fallback: if timeline has no result, use seg_ref (old speaker snapshot).
                        if (spk_id < 0) {
                            spk_id = seg_ref_speaker_id_;
                            spk_sim = seg_ref_speaker_sim_;
                            spk_conf = (spk_id >= 0) ? 0.10f : 0.0f;
                            spk_name = seg_ref_speaker_name_;
                            spk_source = "SNAPSHOT";
                        } else {
                            spk_source = kSN[static_cast<int>(resolved.source)];
                            LOG_INFO("AudioPipe", "Timeline(spk_change): resolved spk=%d %s (sim=%.3f, src=%s) for %.2f-%.2fs",
                                     spk_id, spk_name.c_str(), spk_sim,
                                     spk_source.c_str(),
                                     asr_audio_start / 16000.0f, asr_audio_end / 16000.0f);
                        }
                    }
                    // Capture tracker pipeline speaker for A/B comparison.
                    auto& tst = tracker_.stats();
                    int trk_id = tst.speaker_id;
                    float trk_sim = tst.speaker_sim;
                    std::string trk_name(tst.speaker_name);
                    {
                        std::lock_guard<std::mutex> lock(asr_mutex_);
                        ASRJob job;
                        job.pcm_f32 = std::move(pcm_f32);
                        job.audio_duration_sec = pre_duration;
                        job.trigger_reason = "spk_change";
                        job.is_partial = false;
                        job.stream_start_sec = asr_audio_start / 16000.0f;
                        job.stream_end_sec   = asr_audio_end / 16000.0f;
                        job.speaker_id = spk_id;
                        job.speaker_name = std::move(spk_name);
                        job.speaker_sim = spk_sim;
                        job.speaker_confidence = spk_conf;
                        job.speaker_source = std::move(spk_source);
                        job.tracker_id = trk_id;
                        job.tracker_name = std::move(trk_name);
                        job.tracker_sim = trk_sim;
                        asr_queue_.push(std::move(job));
                    }
                    asr_cv_.notify_one();

                    // Keep post-change audio as the start of the new segment.
                    asr_pcm_buf_.erase(asr_pcm_buf_.begin(),
                                       asr_pcm_buf_.begin() + split_at);
                    asr_saw_speech_ = true;  // new segment starts with speech
                    asr_post_silence_ = 0;
                    asr_speech_samples_ = 0;
                    asr_partial_sent_at_ = 0;

                    LOG_INFO("AudioPipe", "SAAS: ASR split done, pre=%.2fs, remaining=%d samples",
                             pre_duration, (int)asr_pcm_buf_.size());
                } else {
                    LOG_INFO("AudioPipe", "SAAS: ASR split skipped (split_at=%d too small or invalid)", split_at);
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
                    // Resolve speaker label via timeline (fused SAAS + Tracker).
                    int64_t asr_audio_start = audio_t1_processed_ - (int64_t)asr_pcm_buf_.size();
                    int64_t asr_audio_end = audio_t1_processed_;
                    auto resolved = spk_timeline_.resolve(asr_audio_start, asr_audio_end);
                    int spk_id = resolved.speaker_id;
                    float spk_sim = resolved.similarity;
                    float spk_conf = resolved.confidence;
                    std::string spk_name(resolved.name);
                    std::string spk_source;
                    static const char* kSourceNames[] = {"SAAS_EARLY","SAAS_FULL","SAAS_CHANGE","SAAS_INHERIT","TRACKER"};
                    // Fallback: if timeline has no result, use current SAAS snapshot.
                    // Prefer CAM++ (speaker_id) over WL-ECAPA when both available.
                    if (spk_id < 0) {
                        if (stats_.speaker_id >= 0 && enable_speaker_.load(std::memory_order_relaxed)) {
                            spk_id = stats_.speaker_id;
                            spk_sim = stats_.speaker_sim;
                            spk_conf = 0.10f;
                            spk_name = std::string(stats_.speaker_name);
                        } else {
                            spk_id = stats_.wlecapa_id;
                            spk_sim = stats_.wlecapa_sim;
                            spk_conf = (spk_id >= 0) ? 0.10f : 0.0f;
                            spk_name = std::string(stats_.wlecapa_name);
                        }
                        spk_source = "SNAPSHOT";
                    } else {
                        spk_source = kSourceNames[static_cast<int>(resolved.source)];
                        LOG_INFO("AudioPipe", "Timeline: resolved spk=%d %s (sim=%.3f, conf=%.3f, src=%s) for %.2f-%.2fs",
                                 spk_id, spk_name.c_str(), spk_sim, spk_conf,
                                 spk_source.c_str(),
                                 asr_audio_start / 16000.0f, asr_audio_end / 16000.0f);
                    }
                    // Capture tracker pipeline speaker for A/B comparison.
                    auto& ts = tracker_.stats();
                    int trk_id = ts.speaker_id;
                    float trk_sim = ts.speaker_sim;
                    std::string trk_name(ts.speaker_name);
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
                        job.tracker_id = trk_id;
                        job.tracker_name = std::move(trk_name);
                        job.tracker_sim = trk_sim;
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
                    on_vad_(vr, frame_idx, audio_t1_processed_);
                }
            }
        }

        // Report stats.
        if (on_stats_) {
            on_stats_(stats_);
        }

        // Periodic diagnostic log (~every 1s = 10 chunks at 100ms).
        if (++diag_counter % 10 == 0) {
            LOG_INFO("AudioPipe", "DIAG rms=%.4f silero=%.3f fsmn=%.3f speech=%d gain=%.1f spk=%d(%.2f)",
                     stats_.last_rms, stats_.silero_prob,
                     stats_.fsmn_prob,
                     (int)stats_.is_speech,
                     gain_.load(std::memory_order_relaxed),
                     stats_.speaker_id, stats_.speaker_sim);
        }
    }
}

// ==================== SpeakerTracker Implementation ====================

SpeakerTracker::SpeakerTracker() = default;

bool SpeakerTracker::init_overlap_det(const OverlapDetectorConfig& cfg) {
    if (!overlap_det_.init(cfg)) return false;
    enable_overlap_det_.store(true, std::memory_order_relaxed);
    return true;
}

bool SpeakerTracker::init_separator(const SpeechSeparatorConfig& cfg) {
    if (!separator_.init(cfg)) return false;
    enable_separator_.store(true, std::memory_order_relaxed);
    return true;
}

void SpeakerTracker::init(WavLMEcapaEncoder* enc, int dim) {
    enc_ = enc;
    ring_capacity_ = window_samples_ * 2;
    ring_.resize(ring_capacity_, 0);
    ring_raw_.resize(ring_capacity_, 0);
    ring_write_ = 0;
    ring_count_ = 0;
    clear();
    LOG_INFO("Tracker", "Initialized (dim=%d, interval=%dms, window=%dms)",
             dim, interval_samples_ / 16, window_samples_ / 16);
}

void SpeakerTracker::clear() {
    db_.clear();
    ref_emb_.clear();
    ref_f0_ = 0.0f;
    current_spk_id_ = -1;
    current_spk_name_.clear();
    current_sim_ = 0.0f;
    confidence_ = TrackerConfidence::NONE;
    state_ = TrackerState::SILENCE;
    seg_start_sample_ = 0;
    low_sim_count_ = 0;
    sim_running_avg_ = 0.0f;
    declining_count_ = 0;
    unknown_count_ = 0;
    unknown_embs_.clear();
    unknown_f0s_.clear();
    f0_profiles_.clear();
    centroids_.clear();
    centroid_counts_.clear();
    timeline_.clear();
    samples_since_check_ = 0;
    total_samples_ = 0;
    vad_speech_ = false;
    prev_vad_speech_ = false;
    silence_samples_ = 0;
    speech_since_onset_ = 0;
    onset_pending_ = false;
    onset_at_sample_ = 0;
    memset(&stats_, 0, sizeof(stats_));
}

void SpeakerTracker::feed(const int16_t* pcm, int n_samples, bool vad_speech,
                          const int16_t* pcm_raw) {
    if (!enc_ || !enabled_.load(std::memory_order_relaxed)) return;

    // Push PCM to circular ring buffer. Also mirror to `ring_raw_` using
    // the pre-FRCRN copy when provided — otherwise mirror the primary
    // audio so separator falls back transparently when FRCRN is off.
    const bool have_raw = (pcm_raw != nullptr);
    for (int i = 0; i < n_samples; i++) {
        ring_[ring_write_]     = pcm[i];
        ring_raw_[ring_write_] = have_raw ? pcm_raw[i] : pcm[i];
        ring_write_ = (ring_write_ + 1) % ring_capacity_;
        if (ring_count_ < ring_capacity_) ring_count_++;
    }

    prev_vad_speech_ = vad_speech_;
    vad_speech_ = vad_speech;

    if (vad_speech) {
        speech_since_onset_ += n_samples;
        silence_samples_ = 0;

        // Detect silence→speech onset for fast path.
        if (!prev_vad_speech_) {
            onset_pending_ = true;
            onset_at_sample_ = (int)total_samples_;
            speech_since_onset_ = n_samples;
        }
    } else {
        silence_samples_ += n_samples;
        speech_since_onset_ = 0;

        // 1s silence → close current segment.
        if (silence_samples_ > 16000 && state_ != TrackerState::SILENCE) {
            if (!timeline_.empty() && timeline_.back().end_sample == 0)
                timeline_.back().end_sample = total_samples_;
            state_ = TrackerState::SILENCE;
            confidence_ = TrackerConfidence::NONE;
            current_spk_id_ = -1;
            current_spk_name_.clear();
            ref_emb_.clear();
            low_sim_count_ = 0;
            declining_count_ = 0;
            onset_pending_ = false;
        }
    }

    samples_since_check_ += n_samples;
    total_samples_ += n_samples;
}

bool SpeakerTracker::check() {
    if (!enc_ || !enc_->initialized() ||
        !enabled_.load(std::memory_order_relaxed))
        return false;

    stats_.check_active = false;
    stats_.reg_event = false;
    stats_.overlap_confirm_valid = false;
    stats_.overlap_spk1_id = -1;
    stats_.overlap_spk1_sim = 0.0f;
    stats_.overlap_spk1_name[0] = '\0';
    stats_.overlap_spk2_id = -1;
    stats_.overlap_spk2_sim = 0.0f;
    stats_.overlap_spk2_name[0] = '\0';

    // Determine if check is needed.
    bool do_check = false;
    bool is_fast_path = false;
    int extract_samples = window_samples_;

    // Fast path: 0.5s after silence→speech onset.
    if (onset_pending_ && vad_speech_ &&
        speech_since_onset_ >= fast_path_samples_) {
        do_check = true;
        is_fast_path = true;
        extract_samples = fast_path_samples_;
        onset_pending_ = false;
    }

    // Regular interval check during speech.
    if (!do_check && vad_speech_ && samples_since_check_ >= interval_samples_) {
        do_check = true;
    }

    if (!do_check) return false;
    samples_since_check_ = 0;

    // Extract PCM window from ring buffer.
    int n = std::min(extract_samples, ring_count_);
    if (n < 8000) return false;  // minimum 0.5s

    std::vector<float> pcm_f32(n);
    int read_pos = (ring_write_ - n + ring_capacity_) % ring_capacity_;
    for (int i = 0; i < n; i++)
        pcm_f32[i] = ring_[(read_pos + i) % ring_capacity_] / 32768.0f;

    // Parallel raw (pre-FRCRN) window. MossFormer2 separation runs on
    // this buffer when overlap is suspected so FRCRN does not suppress
    // the weaker speaker. Populated lazily below only if the separator
    // actually runs, to avoid the per-check copy when OD is negative.
    std::vector<float> pcm_raw_f32;

    auto t0 = std::chrono::steady_clock::now();

    auto emb = enc_->extract(pcm_f32.data(), n);
    if (emb.empty()) return false;

    float f0 = estimate_f0(pcm_f32.data(), n);
    float jitter = compute_f0_jitter(pcm_f32.data(), n);

    auto t1 = std::chrono::steady_clock::now();
    float lat_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // === No current speaker — identify from scratch ===
    if (current_spk_id_ < 0 || state_ == TrackerState::SILENCE) {
        ScoredMatch sm = score_best(emb, f0);

        if (sm.speaker_id >= 0) {
            current_spk_id_ = sm.speaker_id;
            current_spk_name_ = sm.name;
            current_sim_ = sm.score;
            ref_emb_ = emb;
            ref_f0_ = f0;
            state_ = TrackerState::TRACKING;
            confidence_ = TrackerConfidence::LOW;
            seg_start_sample_ = total_samples_ - n;
            low_sim_count_ = 0;
            declining_count_ = 0;
            sim_running_avg_ = sm.sim_emb;
            unknown_count_ = 0;
            unknown_embs_.clear();
            unknown_f0s_.clear();

            // Update centroid.
            auto cit = centroids_.find(sm.speaker_id);
            if (cit != centroids_.end()) {
                int cnt = ++centroid_counts_[sm.speaker_id];
                float a = 1.0f / cnt;
                for (size_t j = 0; j < emb.size(); j++)
                    cit->second[j] = cit->second[j] * (1.0f - a) + emb[j] * a;
            } else {
                centroids_[sm.speaker_id] = emb;
                centroid_counts_[sm.speaker_id] = 1;
            }

            // Update F0 profile.
            if (f0 > 0) {
                auto& fp = f0_profiles_[sm.speaker_id];
                fp.count++;
                float old_mean = fp.mean;
                fp.mean += (f0 - old_mean) / fp.count;
                fp.sum_sq += (f0 - old_mean) * (f0 - fp.mean);
            }

            // Start timeline entry.
            if ((int)timeline_.size() >= MAX_TIMELINE)
                timeline_.erase(timeline_.begin());
            timeline_.push_back({seg_start_sample_, 0, current_spk_id_,
                                current_spk_name_, current_sim_, confidence_, state_});
            LOG_INFO("Tracker", "Identified: id=%d sim=%.3f score=%.3f %s (%.2fs, fast=%d)",
                     sm.speaker_id, sm.sim_emb, sm.score,
                     sm.name.empty() ? "(unnamed)" : sm.name.c_str(),
                     n / 16000.0f, is_fast_path);
        } else {
            // Unknown speaker — registration gate.
            state_ = TrackerState::UNKNOWN;
            unknown_count_++;
            unknown_embs_.push_back(emb);
            unknown_f0s_.push_back(f0);

            if (unknown_count_ >= register_confirm_) {
                // Self-consistency check: average pairwise cosine sim.
                float pair_sum = 0;
                int pair_count = 0;
                for (size_t a = 0; a < unknown_embs_.size(); a++) {
                    for (size_t b = a + 1; b < unknown_embs_.size(); b++) {
                        float dot = 0;
                        for (size_t j = 0; j < unknown_embs_[a].size(); j++)
                            dot += unknown_embs_[a][j] * unknown_embs_[b][j];
                        pair_sum += dot;
                        pair_count++;
                    }
                }
                float avg_pair = pair_count > 0 ? pair_sum / pair_count : 0;
                bool consistent = avg_pair >= self_consistency_;

                // F0 stability check: CV < 0.20.
                bool f0_stable = true;
                if (unknown_f0s_.size() >= 2) {
                    float fsum = 0, fsum2 = 0;
                    int fvalid = 0;
                    for (float v : unknown_f0s_) {
                        if (v > 0) { fsum += v; fsum2 += v * v; fvalid++; }
                    }
                    if (fvalid >= 2) {
                        float fm = fsum / fvalid;
                        float fv = fsum2 / fvalid - fm * fm;
                        float cv = fm > 1e-6f ? sqrtf(std::max(0.0f, fv)) / fm : 0;
                        f0_stable = cv < 0.20f;
                    }
                }

                if (consistent && f0_stable) {
                    // Build average embedding.
                    int dim = (int)emb.size();
                    std::vector<float> avg_emb(dim, 0.0f);
                    for (auto& e : unknown_embs_)
                        for (int j = 0; j < dim; j++) avg_emb[j] += e[j];
                    float inv = 1.0f / unknown_embs_.size();
                    float norm = 0;
                    for (int j = 0; j < dim; j++) {
                        avg_emb[j] *= inv;
                        norm += avg_emb[j] * avg_emb[j];
                    }
                    norm = sqrtf(norm);
                    if (norm > 0)
                        for (int j = 0; j < dim; j++) avg_emb[j] /= norm;

                    int new_id = db_.register_speaker("", avg_emb);
                    current_spk_id_ = new_id;
                    current_spk_name_.clear();
                    current_sim_ = 1.0f;
                    ref_emb_ = avg_emb;
                    ref_f0_ = 0;
                    for (float v : unknown_f0s_) if (v > 0) ref_f0_ += v;
                    int fcnt = 0;
                    for (float v : unknown_f0s_) if (v > 0) fcnt++;
                    if (fcnt > 0) ref_f0_ /= fcnt;

                    state_ = TrackerState::TRACKING;
                    confidence_ = TrackerConfidence::LOW;
                    seg_start_sample_ = total_samples_ - n;
                    sim_running_avg_ = 1.0f;
                    centroids_[new_id] = avg_emb;
                    centroid_counts_[new_id] = (int)unknown_embs_.size();
                    if (ref_f0_ > 0)
                        f0_profiles_[new_id] = {ref_f0_, 0.0f, 1};

                    if ((int)timeline_.size() >= MAX_TIMELINE)
                        timeline_.erase(timeline_.begin());
                    timeline_.push_back({seg_start_sample_, 0, new_id, "",
                                        1.0f, confidence_, state_});

                    stats_.reg_event = true;
                    stats_.reg_id = new_id;
                    stats_.reg_name[0] = '\0';
                    LOG_INFO("Tracker", "Registered NEW speaker id=%d (pair_sim=%.3f, f0=%.0f Hz)",
                             new_id, avg_pair, ref_f0_);
                } else {
                    LOG_INFO("Tracker", "Registration gate failed (pair=%.3f stable=%d) — reset",
                             avg_pair, (int)f0_stable);
                }
                unknown_count_ = 0;
                unknown_embs_.clear();
                unknown_f0s_.clear();
            }
        }
    } else {
        // === Already tracking — check if still same speaker ===
        float sim_to_ref = 0;
        if (!ref_emb_.empty() && ref_emb_.size() == emb.size()) {
            for (size_t i = 0; i < emb.size(); i++)
                sim_to_ref += ref_emb_[i] * emb[i];
        }

        // Update EMA.
        const float ema_alpha = 0.3f;
        sim_running_avg_ = sim_running_avg_ * (1.0f - ema_alpha) +
                           sim_to_ref * ema_alpha;

        // Absolute change detection.
        if (sim_to_ref < change_threshold_) {
            low_sim_count_++;
        } else {
            low_sim_count_ = 0;
        }

        // Relative decline detection.
        if (sim_running_avg_ - sim_to_ref > 0.25f) {
            declining_count_++;
        } else {
            declining_count_ = 0;
        }

        bool speaker_changed = (low_sim_count_ >= change_confirm_count_) ||
                               (declining_count_ >= 3);

        // P1: Learned overlap detection (replaces heuristic).
        bool overlap_suspected = false;
        if (overlap_det_.initialized() && enable_overlap_det_.load(std::memory_order_relaxed)) {
            // Feed PCM (already float) to overlap detector.
            if (n > 0) {
                OverlapResult odr;
                auto od_t0 = std::chrono::steady_clock::now();
                if (overlap_det_.feed(pcm_f32.data(), n, odr)) {
                    auto od_t1 = std::chrono::steady_clock::now();
                    overlap_suspected = odr.is_overlap;
                    stats_.overlap_detected = odr.is_overlap;
                    stats_.overlap_ratio = odr.overlap_ratio;
                    stats_.od_latency_ms = std::chrono::duration<float, std::milli>(od_t1 - od_t0).count();
                    if (odr.overlap_ratio > 0.0f) {
                        LOG_INFO("Tracker", "OD: ratio=%.3f is_overlap=%d lat=%.1fms",
                                 odr.overlap_ratio, (int)odr.is_overlap, stats_.od_latency_ms);
                    }
                }
            }
        } else {
            // Heuristic fallback.
            overlap_suspected = (jitter > 0.15f && low_sim_count_ >= 1);
        }

        if (speaker_changed) {
            // Close current timeline entry.
            if (!timeline_.empty() && timeline_.back().end_sample == 0)
                timeline_.back().end_sample = total_samples_;
            stats_.switches++;

            // Identify new speaker.
            ScoredMatch sm = score_best(emb, f0);
            if (sm.speaker_id >= 0 && sm.speaker_id != current_spk_id_) {
                LOG_INFO("Tracker", "Speaker CHANGE: %d→%d (sim_ref=%.3f, score=%.3f)",
                         current_spk_id_, sm.speaker_id, sim_to_ref, sm.score);
                current_spk_id_ = sm.speaker_id;
                current_spk_name_ = sm.name;
                current_sim_ = sm.score;
                ref_emb_ = emb;
                ref_f0_ = f0;
                state_ = TrackerState::TRACKING;
                confidence_ = TrackerConfidence::LOW;
                seg_start_sample_ = total_samples_ - n;
                sim_running_avg_ = sm.sim_emb;

                // Update centroid and F0.
                auto cit = centroids_.find(sm.speaker_id);
                if (cit != centroids_.end()) {
                    int cnt = ++centroid_counts_[sm.speaker_id];
                    float a = 1.0f / cnt;
                    for (size_t j = 0; j < emb.size(); j++)
                        cit->second[j] = cit->second[j] * (1.0f - a) + emb[j] * a;
                } else {
                    centroids_[sm.speaker_id] = emb;
                    centroid_counts_[sm.speaker_id] = 1;
                }
                if (f0 > 0) {
                    auto& fp = f0_profiles_[sm.speaker_id];
                    fp.count++;
                    float old_mean = fp.mean;
                    fp.mean += (f0 - old_mean) / fp.count;
                    fp.sum_sq += (f0 - old_mean) * (f0 - fp.mean);
                }

                if ((int)timeline_.size() >= MAX_TIMELINE)
                    timeline_.erase(timeline_.begin());
                timeline_.push_back({seg_start_sample_, 0, current_spk_id_,
                                    current_spk_name_, current_sim_, confidence_, state_});
            } else if (sm.speaker_id < 0) {
                LOG_INFO("Tracker", "Speaker CHANGE: %d→unknown (sim_ref=%.3f)",
                         current_spk_id_, sim_to_ref);
                current_spk_id_ = -1;
                current_spk_name_.clear();
                ref_emb_.clear();
                state_ = TrackerState::UNKNOWN;
                confidence_ = TrackerConfidence::NONE;
                unknown_count_ = 1;
                unknown_embs_ = {emb};
                unknown_f0s_ = {f0};

                if ((int)timeline_.size() >= MAX_TIMELINE)
                    timeline_.erase(timeline_.begin());
                timeline_.push_back({total_samples_ - n, 0, -1, "",
                                    0, confidence_, state_});
            } else {
                // Same speaker re-identified — false alarm.
                LOG_INFO("Tracker", "Change false alarm: still id=%d", current_spk_id_);
            }
            low_sim_count_ = 0;
            declining_count_ = 0;
        } else if (overlap_suspected) {
            state_ = TrackerState::OVERLAP;

            // P2: Speech separation — separate overlapping speakers.
            if (separator_.initialized() && enable_separator_.load(std::memory_order_relaxed)) {
                if (n >= 3200) {  // minimum 200ms for meaningful separation
                    // Lazily extract the parallel raw (pre-FRCRN) window.
                    // Using pre-FRCRN audio for MossFormer2 preserves the
                    // weaker speaker that FRCRN tends to suppress. When
                    // FRCRN was not applied upstream, `ring_raw_` mirrors
                    // `ring_` so this is equivalent to the denoised path.
                    if (pcm_raw_f32.empty()) {
                        pcm_raw_f32.resize(n);
                        for (int i = 0; i < n; i++) {
                            pcm_raw_f32[i] =
                                ring_raw_[(read_pos + i) % ring_capacity_] / 32768.0f;
                        }
                    }
                    auto sep_t0 = std::chrono::steady_clock::now();
                    auto sep_result = separator_.separate(pcm_raw_f32.data(), n);
                    auto sep_t1 = std::chrono::steady_clock::now();

                    stats_.separation_active = true;
                    stats_.separation_lat_ms = std::chrono::duration<float, std::milli>(sep_t1 - sep_t0).count();

                    if (sep_result.valid) {
                        stats_.sep_source1_energy = sep_result.energy1;
                        stats_.sep_source2_energy = sep_result.energy2;

                        // Stage 1: Energy ratio gate — reject OD when the
                        // secondary source has negligible energy relative to
                        // primary. This indicates single-speaker audio where
                        // MossFormer2 produced an artifact in src2.
                        float max_e = std::max(sep_result.energy1, sep_result.energy2);
                        float min_e = std::min(sep_result.energy1, sep_result.energy2);
                        float energy_ratio = (max_e > 1e-6f) ? (min_e / max_e) : 0.0f;
                        stats_.sep_energy_ratio = energy_ratio;

                        static constexpr float kMinEnergyRatio = 0.10f;  // 10%
                        if (energy_ratio < kMinEnergyRatio) {
                            // Single-speaker — separator split is artifact.
                            stats_.overlap_detected = false;
                            stats_.od_reject_reason = TrackerStats::OdReject::ENERGY_RATIO;
                            stats_.sep_quality = 0.0f;
                            LOG_INFO("Tracker",
                                "OD rejected (energy): ratio=%.3f (%.4f/%.4f) < %.2f lat=%.1fms",
                                energy_ratio, min_e, max_e, kMinEnergyRatio,
                                stats_.separation_lat_ms);
                        } else {
                            // Energy balance OK — proceed with speaker identification
                            // on separated sources.
                            static constexpr float kMinSepEnergy = 0.005f;
                            std::vector<float> emb1_cache, emb2_cache;

                            if (sep_result.energy1 > kMinSepEnergy) {
                                emb1_cache = enc_->extract(sep_result.source1.data(), n);
                                if (!emb1_cache.empty()) {
                                    auto m1 = db_.identify(emb1_cache, identify_threshold_, false, identify_threshold_);
                                    if (m1.speaker_id >= 0) {
                                        stats_.overlap_spk1_id = m1.speaker_id;
                                        stats_.overlap_spk1_sim = m1.similarity;
                                        strncpy(stats_.overlap_spk1_name, m1.name.c_str(),
                                                sizeof(stats_.overlap_spk1_name) - 1);
                                        stats_.overlap_spk1_name[sizeof(stats_.overlap_spk1_name) - 1] = '\0';
                                    }
                                }
                            }

                            if (sep_result.energy2 > kMinSepEnergy) {
                                emb2_cache = enc_->extract(sep_result.source2.data(), n);
                                if (!emb2_cache.empty()) {
                                    auto m2 = db_.identify(emb2_cache, identify_threshold_, false, identify_threshold_);
                                    if (m2.speaker_id >= 0) {
                                        stats_.overlap_spk2_id = m2.speaker_id;
                                        stats_.overlap_spk2_sim = m2.similarity;
                                        strncpy(stats_.overlap_spk2_name, m2.name.c_str(),
                                                sizeof(stats_.overlap_spk2_name) - 1);
                                        stats_.overlap_spk2_name[sizeof(stats_.overlap_spk2_name) - 1] = '\0';
                                    }
                                }
                            }

                            // Cross-source embedding similarity check.
                            bool separation_valid = true;
                            float src_sim = 0.0f;
                            if (!emb1_cache.empty() && !emb2_cache.empty() &&
                                emb1_cache.size() == emb2_cache.size()) {
                                float dot = 0, na = 0, nb = 0;
                                for (size_t i = 0; i < emb1_cache.size(); i++) {
                                    dot += emb1_cache[i] * emb2_cache[i];
                                    na  += emb1_cache[i] * emb1_cache[i];
                                    nb  += emb2_cache[i] * emb2_cache[i];
                                }
                                src_sim = (na > 0 && nb > 0)
                                    ? dot / (sqrtf(na) * sqrtf(nb)) : 0.0f;
                                stats_.sep_cross_sim = src_sim;

                                if (src_sim > 0.55f) {
                                    separation_valid = false;
                                    stats_.overlap_detected = false;
                                    stats_.od_reject_reason = TrackerStats::OdReject::CROSS_SIM;
                                    stats_.sep_quality = 0.0f;
                                    LOG_INFO("Tracker",
                                        "OD rejected (cross_sim): sim=%.3f > 0.55 energy_ratio=%.3f lat=%.1fms",
                                        src_sim, energy_ratio, stats_.separation_lat_ms);
                                }
                            }

                            if (separation_valid) {
                                // Stage 2: Compute separation quality score.
                                // quality = energy_balance * speaker_match_confidence
                                // energy_balance: clamped ratio [0,1]
                                float e_score = std::min(energy_ratio / 0.5f, 1.0f);
                                // speaker_match: average of best similarities for identified speakers
                                float s_score = 0.0f;
                                int s_count = 0;
                                if (stats_.overlap_spk1_id >= 0) { s_score += stats_.overlap_spk1_sim; s_count++; }
                                if (stats_.overlap_spk2_id >= 0) { s_score += stats_.overlap_spk2_sim; s_count++; }
                                if (s_count > 0) s_score /= s_count;
                                // Bonus for identifying two DIFFERENT speakers
                                float diversity_bonus = 0.0f;
                                if (stats_.overlap_spk1_id >= 0 && stats_.overlap_spk2_id >= 0 &&
                                    stats_.overlap_spk1_id != stats_.overlap_spk2_id) {
                                    diversity_bonus = 0.2f;
                                }
                                stats_.sep_quality = std::min(e_score * (s_score + diversity_bonus), 1.0f);
                                stats_.od_reject_reason = TrackerStats::OdReject::NONE;

                                stats_.overlap_confirm_valid =
                                    (stats_.overlap_spk1_id >= 0 || stats_.overlap_spk2_id >= 0);

                                LOG_INFO("Tracker",
                                    "OD confirmed: energy_ratio=%.3f cross_sim=%.3f quality=%.3f "
                                    "s1=%d(%.3f) s2=%d(%.3f) lat=%.1fms",
                                    energy_ratio, src_sim, stats_.sep_quality,
                                    stats_.overlap_spk1_id, stats_.overlap_spk1_sim,
                                    stats_.overlap_spk2_id, stats_.overlap_spk2_sim,
                                    stats_.separation_lat_ms);
                            }
                        }

                        LOG_INFO("Tracker", "Overlap separation: src1_rms=%.4f src2_rms=%.4f lat=%.1fms",
                                 sep_result.energy1, sep_result.energy2, stats_.separation_lat_ms);
                    }
                }
            }
        } else {
            // Same speaker continues.
            current_sim_ = sim_to_ref;
            state_ = TrackerState::TRACKING;

            // Update centroid and F0.
            auto cit = centroids_.find(current_spk_id_);
            if (cit != centroids_.end()) {
                int cnt = ++centroid_counts_[current_spk_id_];
                float a = 1.0f / cnt;
                for (size_t j = 0; j < emb.size(); j++)
                    cit->second[j] = cit->second[j] * (1.0f - a) + emb[j] * a;
            } else {
                centroids_[current_spk_id_] = emb;
                centroid_counts_[current_spk_id_] = 1;
            }
            if (f0 > 0) {
                auto& fp = f0_profiles_[current_spk_id_];
                fp.count++;
                float old_mean = fp.mean;
                fp.mean += (f0 - old_mean) / fp.count;
                fp.sum_sq += (f0 - old_mean) * (f0 - fp.mean);
            }

            // Progressive refinement.
            int seg_duration = (int)(total_samples_ - seg_start_sample_);
            if (confidence_ == TrackerConfidence::LOW && seg_duration >= 48000) {
                confidence_ = TrackerConfidence::MED;
                try_refine();
            } else if (confidence_ == TrackerConfidence::MED && seg_duration >= 80000) {
                confidence_ = TrackerConfidence::HIGH;
                try_refine();
            }

            // Update active timeline entry.
            if (!timeline_.empty() && timeline_.back().end_sample == 0) {
                timeline_.back().avg_sim = sim_running_avg_;
                timeline_.back().confidence = confidence_;
            }
        }
    }

    // Update stats.
    stats_.enabled = true;
    stats_.state = state_;
    stats_.speaker_id = current_spk_id_;
    stats_.speaker_sim = current_sim_;
    strncpy(stats_.speaker_name, current_spk_name_.c_str(),
            sizeof(stats_.speaker_name) - 1);
    stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
    stats_.confidence = confidence_;
    stats_.speaker_count = db_.count();
    stats_.timeline_len = (int)timeline_.size();
    stats_.f0_hz = f0;
    stats_.f0_jitter = jitter;
    stats_.sim_to_ref = (state_ == TrackerState::TRACKING && !ref_emb_.empty()) ?
        sim_running_avg_ : 0.0f;
    stats_.sim_running_avg = sim_running_avg_;
    stats_.check_active = true;
    stats_.check_lat_ms = lat_ms;

    return true;
}

float SpeakerTracker::estimate_f0(const float* pcm, int n_samples) {
    // Autocorrelation-based F0 estimation.
    // Search range: 60-500 Hz → lags [32, 266] at 16 kHz.
    const int min_lag = 32;   // 500 Hz
    const int max_lag = 266;  // ~60 Hz
    if (n_samples < max_lag * 2) return 0.0f;

    // Use center 4000 samples (~250ms) for efficiency.
    int win = std::min(n_samples, 4000);
    int offset = (n_samples - win) / 2;
    const float* p = pcm + offset;

    float best_corr = 0.0f;
    int best_lag = 0;

    for (int lag = min_lag; lag <= max_lag && lag < win; lag++) {
        float sum = 0.0f;
        float norm_a = 0.0f, norm_b = 0.0f;
        int len = win - lag;
        for (int i = 0; i < len; i++) {
            sum += p[i] * p[i + lag];
            norm_a += p[i] * p[i];
            norm_b += p[i + lag] * p[i + lag];
        }
        float denom = sqrtf(norm_a * norm_b);
        float corr = denom > 1e-8f ? sum / denom : 0.0f;
        if (corr > best_corr) {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Require minimum correlation for valid F0.
    if (best_corr < 0.3f || best_lag == 0) return 0.0f;
    return 16000.0f / best_lag;
}

float SpeakerTracker::compute_f0_jitter(const float* pcm, int n_samples) {
    // Split into 30ms frames, estimate F0 per frame, compute CV (coeff of variation).
    const int frame_len = 480;  // 30ms at 16 kHz
    const int hop = 480;
    std::vector<float> f0s;
    for (int start = 0; start + frame_len <= n_samples; start += hop) {
        float f = estimate_f0(pcm + start, frame_len);
        if (f > 0.0f) f0s.push_back(f);
    }
    if (f0s.size() < 3) return 0.0f;

    float sum = 0, sum2 = 0;
    for (float v : f0s) { sum += v; sum2 += v * v; }
    float mean = sum / f0s.size();
    float var = sum2 / f0s.size() - mean * mean;
    if (mean < 1e-6f) return 0.0f;
    return sqrtf(std::max(0.0f, var)) / mean;  // CV
}

SpeakerTracker::ScoredMatch SpeakerTracker::score_best(
    const std::vector<float>& emb, float query_f0) {
    // Primary scoring: DB identify for embedding match.
    SpeakerMatch db_match = db_.identify(emb, identify_threshold_, /*auto_register=*/false);
    ScoredMatch result;

    if (db_match.speaker_id < 0) return result;

    result.speaker_id = db_match.speaker_id;
    result.sim_emb = db_match.similarity;
    result.name = db_match.name;

    // Centroid similarity bonus.
    auto cit = centroids_.find(db_match.speaker_id);
    if (cit != centroids_.end()) {
        float dot = 0;
        for (size_t i = 0; i < emb.size() && i < cit->second.size(); i++)
            dot += emb[i] * cit->second[i];
        result.sim_cen = dot;  // both L2-normalized
    } else {
        result.sim_cen = result.sim_emb;  // fallback to emb sim
    }

    // F0 compatibility.
    auto fit = f0_profiles_.find(db_match.speaker_id);
    if (fit != f0_profiles_.end() && fit->second.count >= 2 && query_f0 > 0) {
        float f0_mean = fit->second.mean;
        float f0_std = (fit->second.count > 1) ?
            sqrtf(std::max(0.0f, fit->second.sum_sq / (fit->second.count - 1))) : 50.0f;
        f0_std = std::max(f0_std, 20.0f);  // minimum 20 Hz spread
        float diff = fabsf(query_f0 - f0_mean);
        result.f0_compat = std::max(0.0f, 1.0f - diff / (3.0f * f0_std));
    } else {
        result.f0_compat = 0.5f;  // neutral when no profile
    }

    result.score = w_emb_ * result.sim_emb +
                   w_centroid_ * result.sim_cen +
                   w_f0_ * result.f0_compat;
    return result;
}

void SpeakerTracker::try_refine() {
    if (!enc_ || !enc_->initialized()) return;

    // Re-extract from a larger window for better accuracy.
    int n = std::min(ring_count_, window_samples_);
    if (n < 16000) return;

    std::vector<float> pcm_f32(n);
    int read_pos = (ring_write_ - n + ring_capacity_) % ring_capacity_;
    for (int i = 0; i < n; i++)
        pcm_f32[i] = ring_[(read_pos + i) % ring_capacity_] / 32768.0f;

    auto emb = enc_->extract(pcm_f32.data(), n);
    if (emb.empty()) return;

    // Re-identify with refined embedding.
    SpeakerMatch match = db_.identify(emb, identify_threshold_, /*auto_register=*/false);
    if (match.speaker_id >= 0) {
        // Update reference embedding to the refined one.
        ref_emb_ = emb;
        current_spk_id_ = match.speaker_id;
        current_spk_name_ = match.name;
        current_sim_ = match.similarity;
        sim_running_avg_ = match.similarity;
        LOG_INFO("Tracker", "Refined: id=%d sim=%.3f conf=%d %s (%.2fs)",
                 match.speaker_id, match.similarity, (int)confidence_,
                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                 n / 16000.0f);
    }
}

} // namespace deusridet
