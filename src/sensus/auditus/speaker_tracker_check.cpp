/**
 * @file src/sensus/auditus/speaker_tracker_check.cpp
 * @philosophical_role
 *   Peer TU of audio_pipeline.cpp under R1 split — SpeakerTracker::check (single 537-line method, isolated TU).
 * @serves
 *   Sensus auditus pipeline.
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


} // namespace deusridet
