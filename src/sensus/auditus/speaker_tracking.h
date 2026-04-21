/**
 * @file speaker_tracking.h
 * @philosophical_role The entity's memory of who-spoke-when. SpeakerTimeline is the event-sourced ledger; SpeakerTracker is the independent second opinion. Split out of audio_pipeline.h so the pipeline facade stops dragging an entire identity-attribution subsystem into every TU that includes it.
 * @serves AudioPipeline (fused speaker resolution + A/B tracker comparison).
 */
#pragma once

#include "overlap_detector.h"
#include "speech_separator.h"
#include "../../orator/wavlm_ecapa_encoder.h"
#include "../../orator/speaker_vector_store.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace deusridet {

// ─── SpeakerTimeline: fused speaker resolution via event sourcing ───
//
// All speaker identification subsystems (SAAS early/full/change/inherit, Tracker)
// write timestamped events to a shared timeline. When an ASR segment needs a
// speaker label, resolve() queries the timeline for the best label covering the
// audio range via weighted majority voting.
//
// Authority hierarchy: SAAS_FULL > SAAS_CHANGE > SAAS_EARLY > SAAS_INHERIT > TRACKER
// This replaces the point-in-time snapshot approach that caused timing mismatches.

enum class SpkEventSource : uint8_t {
    SAAS_EARLY   = 0,  // SAAS early extraction (auto_register=false)
    SAAS_FULL    = 1,  // SAAS end-of-segment (auto_register=true)
    SAAS_CHANGE  = 2,  // SAAS intra-segment speaker change detection
    SAAS_INHERIT = 3,  // SAAS inheritance from previous segment
    TRACKER      = 4,  // SpeakerTracker
};

struct SpeakerEvent {
    int64_t  audio_start;   // start of audio range this event covers (absolute sample)
    int64_t  audio_end;     // end of audio range this event covers (absolute sample)
    SpkEventSource source;
    int      speaker_id;    // -1 = unknown/no match
    float    similarity;
    char     name[64];
};

struct ResolvedSpeaker {
    int      speaker_id = -1;
    float    confidence = 0.0f;   // total weighted vote
    float    similarity = 0.0f;
    char     name[64] = {};
    SpkEventSource source = SpkEventSource::SAAS_EARLY;
};

class SpeakerTimeline {
public:
    static constexpr int kMaxEvents = 2000;

    void push(const SpeakerEvent& ev) {
        events_[write_pos_] = ev;
        write_pos_ = (write_pos_ + 1) % kMaxEvents;
        if (count_ < kMaxEvents) ++count_;
    }

    // See speaker_tracking.cpp for the definition.
    ResolvedSpeaker resolve(int64_t start_sample, int64_t end_sample) const;


    int event_count() const { return count_; }
    void clear() { count_ = 0; write_pos_ = 0; }

private:
    SpeakerEvent events_[kMaxEvents];
    int write_pos_ = 0;
    int count_ = 0;
};

// SpeakerTracker: continuous sliding-window speaker identification pipeline.
//
// Independent from the SAAS pipeline. Extracts embeddings from a 1.5s sliding
// window every 0.5s, identifies speakers via its own SpeakerVectorStore DB,
// and outputs a speaker timeline for comparison with SAAS results.
//
// Features:
//   - Regular check (every interval_ms, 1.5s window)
//   - VAD-onset fast path (0.5s window, single-confirm after silence gap)
//   - Progressive refinement (re-identify at 3s, 5s from segment start)
//   - Change detection: absolute (2× confirm) + relative (sim drop)
//   - Multi-signal scoring: embedding + centroid + F0
//   - Registration gate: 3× unknown + self-consistency + F0 stability
//   - Overlap detection: F0 jitter + embedding instability

// Tracker confidence level for current speaker attribution.
enum class TrackerConfidence : int {
    NONE       = 0,  // no identification yet
    LOW        = 1,  // 1.5-3s of audio
    MED        = 2,  // 3-5s of audio
    HIGH       = 3,  // ≥ 5s of audio (locked)
};

// Tracker state per check cycle.
enum class TrackerState : int {
    SILENCE    = 0,  // VAD=false, no tracking
    TRACKING   = 1,  // normal speaker tracking
    TRANSITION = 2,  // speaker change pending confirmation
    OVERLAP    = 3,  // multiple speakers detected
    UNKNOWN    = 4,  // speaker not in DB
};

// Timeline entry for the speaker tracker.
struct TrackerTimelineEntry {
    int64_t start_sample  = 0;
    int64_t end_sample    = 0;    // 0 = still active
    int     speaker_id    = -1;   // -1 = unknown
    std::string name;
    float   avg_sim       = 0.0f;
    TrackerConfidence confidence = TrackerConfidence::NONE;
    TrackerState state    = TrackerState::SILENCE;
};

// Stats exposed to WebUI via pipeline_stats JSON.
struct TrackerStats {
    bool     enabled       = false;
    TrackerState state     = TrackerState::SILENCE;
    int      speaker_id    = -1;
    float    speaker_sim   = 0.0f;
    char     speaker_name[64] = {};
    TrackerConfidence confidence = TrackerConfidence::NONE;
    int      speaker_count = 0;
    int      timeline_len  = 0;
    int      switches      = 0;    // total speaker change count
    float    f0_hz         = 0.0f; // latest F0 estimate
    float    f0_jitter     = 0.0f; // frame-to-frame F0 variability
    float    sim_to_ref    = 0.0f; // cosine sim to current ref embedding
    float    sim_running_avg = 0.0f; // EMA of sim_to_ref
    bool     check_active  = false; // true on tick when a check ran
    float    check_lat_ms  = 0.0f;  // latency of last check (extraction + scoring)
    // Registration events.
    bool     reg_event     = false;  // true on tick when new speaker registered
    int      reg_id        = -1;
    char     reg_name[64]  = {};
    // P1: Overlap detection stats.
    bool     overlap_detected = false;
    float    overlap_ratio  = 0.0f;
    float    od_latency_ms  = 0.0f;
    // P2: Speech separation stats.
    bool     separation_active = false;
    float    separation_lat_ms = 0.0f;
    float    sep_source1_energy = 0.0f;
    float    sep_source2_energy = 0.0f;
    float    sep_energy_ratio = 0.0f;   // min(e1,e2)/max(e1,e2), 0=single-spk
    float    sep_cross_sim = 0.0f;      // cosine sim between separated sources
    // Separation quality score: combines energy balance + speaker match confidence.
    // Range [0,1]: 0=definitely false positive, 1=perfect separation.
    float    sep_quality = 0.0f;
    enum class OdReject : uint8_t {
        NONE = 0,          // not rejected
        ENERGY_RATIO,      // src2/src1 too low (single speaker)
        CROSS_SIM,         // separated sources are same speaker
        OVERLAP_RATIO,     // Seg3 overlap_ratio below threshold
    };
    OdReject od_reject_reason = OdReject::NONE;
    // Overlap speaker confirmation from separated sources.
    bool     overlap_confirm_valid = false;
    int      overlap_spk1_id = -1;
    float    overlap_spk1_sim = 0.0f;
    char     overlap_spk1_name[64] = {};
    int      overlap_spk2_id = -1;
    float    overlap_spk2_sim = 0.0f;
    char     overlap_spk2_name[64] = {};
};

class SpeakerTracker {
public:
    SpeakerTracker();
    ~SpeakerTracker() = default;

    // Initialize with encoder reference and parameters.
    // The tracker does NOT own the encoder — it shares the encoder instance
    // with the SAAS pipeline (serial calls from same thread, safe).
    void init(WavLMEcapaEncoder* enc, int dim = 192);

    // Feed PCM chunk (called every process loop iteration).
    // `pcm`     : primary (post-FRCRN) audio used for speaker embedding,
    //             F0 estimation, and overlap detection.
    // `pcm_raw` : optional pre-FRCRN audio. When provided, MossFormer2
    //             separation runs on this raw signal to avoid FRCRN
    //             suppressing the weaker speaker in overlap regions.
    //             Pass nullptr (or omit) when FRCRN is disabled — the
    //             primary buffer is then used for separation as well.
    void feed(const int16_t* pcm, int n_samples, bool vad_speech,
              const int16_t* pcm_raw = nullptr);

    // Perform check if interval reached. Returns true if a check was executed.
    // Must be called after feed() in the same loop iteration.
    bool check();

    // Access current stats (read by awaken for JSON).
    const TrackerStats& stats() const { return stats_; }

    // Access timeline (for detailed JSON dump).
    const std::vector<TrackerTimelineEntry>& timeline() const { return timeline_; }

    // Access the independent speaker DB.
    SpeakerVectorStore& db() { return db_; }
    const SpeakerVectorStore& db() const { return db_; }

    // Runtime parameter control.
    void set_enabled(bool e) { enabled_.store(e, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    void set_interval_ms(int ms) { interval_samples_ = std::max(4000, ms * 16); }
    int  interval_ms() const { return interval_samples_ / 16; }

    void set_window_ms(int ms) { window_samples_ = std::max(8000, ms * 16); }
    int  window_ms() const { return window_samples_ / 16; }

    void set_threshold(float t) { identify_threshold_ = t; }
    float threshold() const { return identify_threshold_; }

    void set_change_threshold(float t) { change_threshold_ = t; }
    float change_threshold() const { return change_threshold_; }

    // Clear DB and reset all state.
    void clear();

    void set_speaker_name(int id, const std::string& name) { db_.set_name(id, name); }

    // P1: Overlap detection init and control.
    bool init_overlap_det(const OverlapDetectorConfig& cfg);
    void set_overlap_det_enabled(bool e) { enable_overlap_det_.store(e, std::memory_order_relaxed); }
    bool overlap_det_enabled() const { return enable_overlap_det_.load(std::memory_order_relaxed); }
    bool overlap_det_loaded() const { return overlap_det_.initialized(); }

    // P2: Speech separator init and control.
    bool init_separator(const SpeechSeparatorConfig& cfg);
    void set_separator_enabled(bool e) { enable_separator_.store(e, std::memory_order_relaxed); }
    bool separator_enabled() const { return enable_separator_.load(std::memory_order_relaxed); }
    bool separator_loaded() const { return separator_.loaded(); }

    // Expose separator for FULL-extraction overlap recovery.
    SeparationResult separate(const float* pcm, int n) { return separator_.separate(pcm, n); }
    bool separator_ready() const { return separator_.loaded() && enable_separator_.load(std::memory_order_relaxed); }

private:
    // Estimate F0 (fundamental frequency) from PCM using autocorrelation.
    float estimate_f0(const float* pcm, int n_samples);

    // Compute F0 jitter: frame-by-frame F0 stability measure.
    float compute_f0_jitter(const float* pcm, int n_samples);

    // Multi-signal speaker scoring.
    struct ScoredMatch {
        int   speaker_id = -1;
        float score      = 0.0f;  // weighted combined score
        float sim_emb    = 0.0f;  // raw embedding similarity
        float sim_cen    = 0.0f;  // centroid similarity
        float f0_compat  = 0.0f;  // F0 compatibility [0, 1]
        std::string name;
    };

    // Score all speakers in DB against query embedding + F0.
    ScoredMatch score_best(const std::vector<float>& emb, float query_f0);

    // Attempt progressive refinement using accumulated segment audio.
    void try_refine();

    // ---- State ----
    WavLMEcapaEncoder* enc_ = nullptr;
    SpeakerVectorStore db_{"TrackerDb", 192, 0.15f};
    std::atomic<bool> enabled_{true};

    // Ring buffer for recent PCM (stores window_samples_ + margin).
    std::vector<int16_t> ring_;
    // Parallel ring buffer holding pre-FRCRN (raw) audio. Same write
    // cursor as `ring_`. Consumed by the MossFormer2 separator so
    // overlap regions can be separated from the un-denoised signal
    // (FRCRN tends to suppress the weaker speaker, defeating
    // separation). When feed() is called without a raw pointer, this
    // ring holds the same data as `ring_`.
    std::vector<int16_t> ring_raw_;
    int ring_capacity_ = 0;
    int ring_write_    = 0;   // next write position (circular)
    int ring_count_    = 0;   // valid samples in ring

    // Timing.
    int interval_samples_ = 8000;   // 0.5s
    int window_samples_   = 24000;  // 1.5s
    int samples_since_check_ = 0;
    int64_t total_samples_   = 0;   // monotonic counter

    // VAD state.
    bool vad_speech_      = false;
    bool prev_vad_speech_ = false;
    int  silence_samples_ = 0;    // consecutive silence samples
    int  speech_since_onset_ = 0; // speech samples since last silence→speech transition

    // VAD-onset fast path state.
    bool onset_pending_   = false;   // silence→speech detected, waiting 0.5s for fast check
    int  onset_at_sample_ = 0;      // sample count at onset

    // Current segment state.
    std::vector<float> ref_emb_;     // reference embedding for current speaker
    float ref_f0_         = 0.0f;    // reference F0 for current speaker
    int   current_spk_id_ = -1;
    std::string current_spk_name_;
    float current_sim_    = 0.0f;
    TrackerConfidence confidence_ = TrackerConfidence::NONE;
    TrackerState state_   = TrackerState::SILENCE;
    int64_t seg_start_sample_ = 0;  // start of current speaker segment

    // Change detection.
    int   low_sim_count_  = 0;       // consecutive checks with sim < change_threshold
    float sim_running_avg_ = 0.0f;   // EMA of sim_to_ref
    int   declining_count_ = 0;      // consecutive checks where sim dropped > 0.25 from avg

    // Registration gate.
    int   unknown_count_       = 0;  // consecutive unknown checks
    std::vector<std::vector<float>> unknown_embs_; // embeddings during unknown streak
    std::vector<float> unknown_f0s_; // F0 values during unknown streak

    // Per-speaker F0 profile (maintained on host).
    struct F0Profile {
        float mean    = 0.0f;
        float sum_sq  = 0.0f;  // for variance calculation
        int   count   = 0;
    };
    std::unordered_map<int, F0Profile> f0_profiles_;

    // Per-speaker centroid (maintained on host for multi-signal scoring).
    std::unordered_map<int, std::vector<float>> centroids_;  // spk_id → 192-dim mean
    std::unordered_map<int, int> centroid_counts_;           // number of embeddings in mean

    // Timeline.
    std::vector<TrackerTimelineEntry> timeline_;
    static constexpr int MAX_TIMELINE = 500;

    // Output stats.
    TrackerStats stats_{};

    // Parameters.
    float identify_threshold_   = 0.55f;
    float change_threshold_     = 0.35f;
    int   change_confirm_count_ = 2;
    int   register_confirm_     = 3;
    float self_consistency_     = 0.55f;
    float w_emb_                = 0.50f;
    float w_centroid_           = 0.30f;
    float w_f0_                 = 0.20f;
    int   fast_path_samples_    = 8000;  // 0.5s for VAD-onset fast path

    // P1: Learned overlap detection.
    OverlapDetector overlap_det_;
    std::atomic<bool> enable_overlap_det_{false};

    // P2: MossFormer2 speech separation.
    SpeechSeparator separator_;
    std::atomic<bool> enable_separator_{false};
};


} // namespace deusridet
