/**
 * @file src/sensus/auditus/audio_pipeline_process_asr.cpp
 * @philosophical_role
 *   Stage-extract of AudioPipeline::process_loop (Step 11 A1a).
 *   Phase 8: continuous ASR accumulation.
 *
 *   The Whisper encoder naturally tolerates silence and mixed audio, so the
 *   ASR path accumulates every chunk unconditionally and only consults VAD
 *   as a trigger hint (WHEN to fire transcription) and as a content-ratio
 *   filter. SAAS speaker-change events take priority over post-silence
 *   triggers so that ASR segments align with speaker turns.
 * @serves
 *   Sensus auditus — transcription arm of the per-chunk pipeline.
 */
#include "audio_pipeline.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: one chunk of continuous ASR — accumulate, trigger, resolve speaker, submit.
void AudioPipeline::process_asr_pipeline_(const int16_t* pcm_buf, int n_samples) {
        // ASR: accumulate ALL audio continuously. Whisper encoder naturally handles
        // silence and mixed audio. VAD is used only as a trigger hint (WHEN to fire
        // transcription) and to track speech content ratio for filtering.
        if (asr_engine_ && asr_engine_->is_loaded() &&
            enable_asr_.load(std::memory_order_relaxed)) {

            // Always accumulate ALL audio (speech + silence).
            asr_pcm_buf_.insert(asr_pcm_buf_.end(),
                                pcm_buf, pcm_buf + n_samples);

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

}

} // namespace deusridet
