/**
 * @file src/sensus/auditus/audio_pipeline_process_saas_during.cpp
 * @philosophical_role
 *   Stage-extract of AudioPipeline::process_loop (Step 11 A1c-1).
 *   During-speech branch: everything that runs while in_speech_segment_ is true.
 *
 *   Appends pcm to speech_pcm_buf_ and fbank frames to seg_fbank_buf_, limits
 *   the buffer to 10s, fires CAM++ EARLY extraction once fbank crosses the
 *   threshold, runs WL-ECAPA early identification, and performs intra-segment
 *   speaker change detection (soft-restart on detected speaker turn).
 * @serves
 *   Sensus auditus — SAAS mid-segment identity arm.
 */
#include "audio_pipeline.h"
#include "separatio_orator_probe.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: during-speech per-chunk SAAS work — buffer, fbank, EARLY, WL-early, recheck/split.
void AudioPipeline::process_saas_during_speech_(const int16_t* pcm_buf, int n_samples) {
                speech_pcm_buf_.insert(speech_pcm_buf_.end(),
                                       pcm_buf, pcm_buf + n_samples);
                // Track overlap during this speech segment.
                seg_total_chunks_++;
                if (stats_.overlap_detected) seg_overlap_chunks_++;
                if ((speaker_enc_.initialized() &&
                     enable_speaker_.load(std::memory_order_relaxed))) {
                    speaker_fbank_.push_pcm(pcm_buf, n_samples);
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
                    std::vector<float> emb;
                    float lat_cnn_ms = 0.0f;
                    float lat_encoder_ms = 0.0f;
                    float lat_ecapa_ms = 0.0f;
                    float lat_total_ms = 0.0f;
                    {
                        std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
                        emb = wlecapa_enc_.extract(pcm_f32.data(), early_samples);
                        lat_cnn_ms = wlecapa_enc_.last_lat_cnn_ms();
                        lat_encoder_ms = wlecapa_enc_.last_lat_encoder_ms();
                        lat_ecapa_ms = wlecapa_enc_.last_lat_ecapa_ms();
                        lat_total_ms = wlecapa_enc_.last_lat_total_ms();
                    }
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
                        stats_.wlecapa_lat_cnn_ms     = lat_cnn_ms;
                        stats_.wlecapa_lat_encoder_ms = lat_encoder_ms;
                        stats_.wlecapa_lat_ecapa_ms   = lat_ecapa_ms;
                        stats_.wlecapa_lat_total_ms   = lat_total_ms;
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
                                     lat_total_ms,
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
                                     lat_total_ms);
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
                            std::vector<float> emb;
                            {
                                std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
                                emb = wlecapa_enc_.extract(pcm_f32.data(), len);
                            }
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
                                        std::vector<float> pre_emb;
                                        {
                                            std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
                                            pre_emb = wlecapa_enc_.extract(pre_f32.data(), pre_samples);
                                        }
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

} // namespace deusridet
