/**
 * @file src/sensus/auditus/speaker_tracker_check.cpp
 * @philosophical_role
 *   Step 11 A2 orchestrator — `SpeakerTracker::check()` reduced to a
 *   thin sequential dispatcher over named private stages. The heavy
 *   branch bodies live in `speaker_tracker_check_stages.cpp` (peer TU).
 *
 *   Stages (all private members on `SpeakerTracker`; each is one
 *   independently inspectable unit):
 *     1. check_should_run_          — gating (fast path / interval)
 *     2. check_extract_pcm_          — ring-buffer read → pcm_f32
 *     3.   (inline) enc_->extract + estimate_f0 + compute_f0_jitter
 *     4. check_identify_new_         — no-current-speaker branch
 *     5. check_track_existing_       — tracking branch (dispatches
 *        4 sub-stages internally)
 *     6. check_update_stats_         — tail: publish telemetry
 *
 *   No behavioural change — stage bodies are verbatim from the
 *   pre-split monolith (commit c77efde).
 * @serves
 *   Sensus auditus pipeline (SpeakerTracker).
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

// ============================================================
// Orchestrator
// ============================================================
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

    bool is_fast_path = false;
    int extract_samples = window_samples_;
    if (!check_should_run_(is_fast_path, extract_samples)) return false;
    samples_since_check_ = 0;

    std::vector<float> pcm_f32;
    int read_pos = 0;
    int n = 0;
    if (!check_extract_pcm_(extract_samples, n, pcm_f32, read_pos)) return false;

    // Parallel raw (pre-FRCRN) window. MossFormer2 separation runs on
    // this buffer when overlap is suspected so FRCRN does not suppress
    // the weaker speaker. Populated lazily inside the overlap sub-stage.
    std::vector<float> pcm_raw_f32;

    auto t0 = std::chrono::steady_clock::now();
    auto emb = enc_->extract(pcm_f32.data(), n);
    if (emb.empty()) return false;

    float f0 = estimate_f0(pcm_f32.data(), n);
    float jitter = compute_f0_jitter(pcm_f32.data(), n);

    auto t1 = std::chrono::steady_clock::now();
    float lat_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (current_spk_id_ < 0 || state_ == TrackerState::SILENCE) {
        check_identify_new_(emb, f0, n, is_fast_path);
    } else {
        check_track_existing_(emb, f0, n, pcm_f32, pcm_raw_f32, read_pos, jitter);
    }

    check_update_stats_(f0, jitter, lat_ms);
    return true;
}

// ============================================================
// Stage 1: gating — decide whether this call performs a check.
// ============================================================
bool SpeakerTracker::check_should_run_(bool& is_fast_path, int& extract_samples) {
    bool do_check = false;
    is_fast_path = false;
    extract_samples = window_samples_;

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

    return do_check;
}

// ============================================================
// Stage 2: PCM extraction from the ring buffer.
// ============================================================
bool SpeakerTracker::check_extract_pcm_(int extract_samples, int& n,
                                        std::vector<float>& pcm_f32,
                                        int& read_pos) {
    n = std::min(extract_samples, ring_count_);
    if (n < 8000) return false;  // minimum 0.5s

    pcm_f32.assign(n, 0.0f);
    read_pos = (ring_write_ - n + ring_capacity_) % ring_capacity_;
    for (int i = 0; i < n; i++)
        pcm_f32[i] = ring_[(read_pos + i) % ring_capacity_] / 32768.0f;
    return true;
}

// ============================================================
// Stage 6: telemetry publication (tail).
// ============================================================
void SpeakerTracker::check_update_stats_(float f0, float jitter, float lat_ms) {
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
}

// ============================================================
// Stage 5: tracking dispatch.
// ============================================================
void SpeakerTracker::check_track_existing_(const std::vector<float>& emb,
                                           float f0, int n,
                                           std::vector<float>& pcm_f32,
                                           std::vector<float>& pcm_raw_f32,
                                           int read_pos, float jitter) {
    float sim_to_ref = 0.0f;
    bool overlap_suspected = false;
    track_compute_signals_(emb, pcm_f32, n, jitter, sim_to_ref, overlap_suspected);

    bool speaker_changed = (low_sim_count_ >= change_confirm_count_) ||
                           (declining_count_ >= 3);

    if (speaker_changed) {
        track_handle_change_(emb, f0, n, sim_to_ref);
    } else if (overlap_suspected) {
        state_ = TrackerState::OVERLAP;
        track_handle_overlap_(pcm_f32, pcm_raw_f32, read_pos, n);
    } else {
        track_handle_continue_(emb, f0, sim_to_ref);
    }
}

} // namespace deusridet
