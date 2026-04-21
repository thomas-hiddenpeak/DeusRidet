/**
 * @file src/sensus/auditus/speaker_tracker.cpp
 * @philosophical_role
 *   Peer TU of audio_pipeline.cpp under R1 split — SpeakerTracker ctor/init/clear/feed/estimate_f0/compute_f0_jitter/score_best/try_refine.
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
