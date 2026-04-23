/**
 * @file src/sensus/auditus/audio_pipeline_process_saas_full.cpp
 * @philosophical_role
 *   Stage-extract of AudioPipeline::process_loop (Step 11 A1b).
 *   End-of-segment CAM++ FULL speaker extraction.
 *
 *   When a speech segment ends, fbank frames accumulated during the
 *   segment are handed to the CAM++ encoder to produce a single robust
 *   embedding. This embedding feeds (a) overlap-guarded match/register
 *   against the speaker DB, and (b) dual-encoder CAM++||WL-ECAPA fusion
 *   when enabled.
 *
 *   (Historical note: an online spectral-clustering warm-up lived here
 *   through Step 11 A1b. It was disabled by test results and removed in
 *   Step 14a — see docs/{en,zh}/devlog/ for the failure record.)
 * @serves
 *   Sensus auditus — SAAS end-of-segment identity arm.
 */
#include "audio_pipeline.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: run FULL CAM++ extraction + dual-encoder fuse + spectral warmup for an ended segment.
// @param fbank_frames  number of 80-dim fbank frames accumulated in seg_fbank_buf_.
void AudioPipeline::process_saas_full_extract_(int fbank_frames) {
                // CAM++ speaker encoder — FULL extraction using accumulated fbank.
                // Minimum-frames gate is configurable (speaker_min_fbank_frames);
                // default 50 (~500 ms) is the ECAPA-style stat-pooling floor,
                // not a test-recording-specific choice. Segments below this
                // threshold are dropped outright: no embedding, no identity.
                const int kMinFbankFrames = cfg_.speaker_min_fbank_frames;
                if (speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    fbank_frames >= kMinFbankFrames) {
                        float thresh = speaker_threshold_.load(std::memory_order_relaxed);
                        float reg_thresh = speaker_register_threshold_.load(std::memory_order_relaxed);

                        auto emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
                        if (!emb.empty()) {
                            bool auto_reg = true;

                            // Late registration cap — after N FULL identifications
                            // every legitimate speaker should already be registered.
                            // Further registrations have historically been drift
                            // clones (observed in v8/v9 test runs at count ≈ 180).
                            // Tunable: configs/auditus.conf:speaker_max_auto_reg_count
                            const int kMaxAutoRegCount = cfg_.speaker_max_auto_reg_count;
                            if (campp_full_count_ >= kMaxAutoRegCount) {
                                auto_reg = false;
                            }
                            float match_thresh = thresh;

                            // v24d: Discovery phase — use higher threshold during
                            // early extractions to force speaker separation.
                            // Without this, similar speakers (e.g. 徐子景/朱杰)
                            // get absorbed into the first registered speaker.
                            // Tunable: configs/auditus.conf:speaker_discovery_{count,boost}
                            const int   kDiscoveryCount = cfg_.speaker_discovery_count;
                            const float kDiscoveryBoost = cfg_.speaker_discovery_boost;
                            if (campp_full_count_ < kDiscoveryCount) {
                                match_thresh += kDiscoveryBoost;
                            }                            // v24: Temporal recency bonus — lower threshold when recent
                            // speaker still active, reducing false negatives (fragmentation).
                            float seg_mid_time = (float)(audio_t1_processed_ - (int64_t)speech_pcm_buf_.size() / 2) / 16000.0f;
                            float time_since_prev = seg_mid_time - prev_full_time_;
                            // Tunable: configs/auditus.conf:speaker_recency_{window_sec,bonus}
                            const float kRecencyWindow = cfg_.speaker_recency_window_sec;
                            const float kRecencyBonus  = cfg_.speaker_recency_bonus;
                            // Step 16 iter 1: recency is a post-discovery
                            // stabilizer only. During discovery (first
                            // kDiscoveryCount FULL extractions) its combined
                            // effect — lowered threshold + auto_reg=false —
                            // absorbs every cold-start speaker into spk0
                            // (baseline: 48.4% GT-side on seg0 0–600s). We
                            // therefore gate recency on post-discovery so
                            // newcomers within 15 s of another speaker still
                            // get a clean registration chance at reg_thresh.
                            bool recency_active =
                                (prev_full_speaker_id_ >= 0 &&
                                 time_since_prev < kRecencyWindow &&
                                 campp_full_count_ >= kDiscoveryCount);
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

                            // Step 16c: recency-absorb guard — when recency was
                            // active AND we accepted a match below the unmodified
                            // threshold, abstain even if the matched id equals
                            // prev. Manual review of 1x baseline on tests/test.mp3
                            // showed 徐子景 @05:21 being absorbed into the 唐云峰
                            // cluster at sim<0.50 because the recency bonus
                            // dropped match_thresh to 0.45 AND the v24 re-check
                            // above only fires when matched id != prev. That
                            // asymmetry lets cross-talk paint the wrong cluster.
                            // Abstaining here keeps the next FULL extraction free
                            // to register the true incoming speaker.
                            if (recency_active && match.speaker_id >= 0 && !match.is_new &&
                                match.similarity < thresh) {
                                LOG_INFO("AudioPipe", "Recency absorb-guard: sim=%.3f < thresh=%.2f (matched #%d=prev); abstain",
                                         match.similarity, thresh, match.speaker_id);
                                match.speaker_id = -1;
                                match.similarity = 0;
                                match.name.clear();
                            }

                            // Margin gate: abstain on ambiguous matches where
                            // top-1 and top-2 are too close to distinguish.
                            // Tunable: configs/auditus.conf:speaker_margin_abstain
                            const float kMarginAbstainThresh = cfg_.speaker_margin_abstain;
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
                        }
                }
                // Step 4c: short-segment INHERIT-BROADCAST.
                // prev_seg_speaker_id_ is reset to -1 whenever the previous
                // segend took this same short-skip path (no FULL → no
                // seg_ref / stats). To stay continuous across a run of
                // short segments we fall back to prev_full_speaker_id_,
                // which persists from the last successful FULL identify.
                //
                // RECENCY GATE: only inherit from prev_full when it is
                // temporally fresh (≤ 2.0 s since prev FULL midpoint).
                // Without this, every short segment for the rest of the
                // timeline would inherit whichever speaker happened to be
                // last identified by FULL, collapsing all clusters onto
                // that one identity (observed empirically: decided_macro
                // crashes from 0.92 to 0.32).
                int         inh_id = -1;
                std::string inh_name;
                float       inh_sim = 0.0f;
                const char* inh_src = "";
                if (prev_seg_speaker_id_ >= 0) {
                    inh_id   = prev_seg_speaker_id_;
                    inh_name = prev_seg_speaker_name_;
                    inh_sim  = prev_seg_speaker_sim_;
                    inh_src  = "seg";
                } else if (prev_full_speaker_id_ >= 0) {
                    float now_sec = (float)audio_t1_processed_ / 16000.0f;
                    float age     = now_sec - prev_full_time_;
                    if (age <= 2.0f) {
                        inh_id   = prev_full_speaker_id_;
                        inh_name = prev_full_speaker_name_;
                        inh_sim  = 0.0f;
                        inh_src  = "full";
                    }
                }
                if (!(speaker_enc_.initialized() &&
                      enable_speaker_.load(std::memory_order_relaxed) &&
                      fbank_frames >= kMinFbankFrames) &&
                    cfg_.speaker_short_inherit_enable &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    inh_id >= 0) {
                    // Short-segment inheritance broadcast.
                    //
                    // This segment is too short for CAM++ FULL to produce a
                    // trustworthy embedding. Rather than drop it silently
                    // (→ "no_segment" in the replay scorer), forward the
                    // last successfully-identified speaker as a best-effort
                    // label. prev_seg_speaker_id_ is updated at the START
                    // of every segend, so it holds whichever identity the
                    // most recent long-enough segment produced (or the
                    // VAD-start 0.8 s inheritance when that fired).
                    //
                    // Critically: we do NOT call campp_db_.identify(),
                    // dual_db_.identify(), or register_speaker() here. The
                    // speaker library is untouched. This is pure continuity
                    // propagation at the broadcast layer. Wrong labels on
                    // isolated short utterances by a new speaker will show
                    // up in macro(all) but CANNOT pollute centroids.
                    SpeakerMatch inh{};
                    inh.speaker_id     = inh_id;
                    inh.similarity     = inh_sim;
                    inh.is_new         = false;
                    inh.name           = inh_name;
                    inh.exemplar_count = 0;
                    inh.hits_above     = 0;
                    LOG_INFO("AudioPipe",
                             "FULL-skip INHERIT-BROADCAST: id=%d sim=%.3f %s (fbank=%d < %d, src=%s)",
                             inh.speaker_id, inh.similarity,
                             inh.name.empty() ? "(unnamed)" : inh.name.c_str(),
                             fbank_frames, kMinFbankFrames,
                             inh_src);
                    if (on_speaker_) on_speaker_(inh);
                }
}

} // namespace deusridet
