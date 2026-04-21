/**
 * @file src/sensus/auditus/audio_pipeline_process_saas_full.cpp
 * @philosophical_role
 *   Stage-extract of AudioPipeline::process_loop (Step 11 A1b).
 *   End-of-segment CAM++ FULL speaker extraction + spectral warm-up.
 *
 *   When a speech segment ends, fbank frames accumulated during the
 *   segment are handed to the CAM++ encoder to produce a single robust
 *   embedding. This embedding feeds (a) overlap-guarded match/register
 *   against the speaker DB, (b) dual-encoder CAM++||WL-ECAPA fusion when
 *   enabled, and (c) spectral-clustering warm-up that, after enough
 *   embeddings, rebuilds the DB with cluster-centroid exemplars.
 * @serves
 *   Sensus auditus — SAAS end-of-segment identity arm.
 */
#include "audio_pipeline.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"
#include "../../orator/spectral_cluster.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: run FULL CAM++ extraction + dual-encoder fuse + spectral warmup for an ended segment.
// @param fbank_frames  number of 80-dim fbank frames accumulated in seg_fbank_buf_.
void AudioPipeline::process_saas_full_extract_(int fbank_frames) {
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
}

} // namespace deusridet
