// audio_pipeline.h — Real-time audio processing pipeline.
//
// Wires: WS PCM input → Ring Buffer → Gain → Mel (GPU) → Energy VAD
//                                          └→ Silero VAD (ONNX, CPU)
//                                          └→ FSMN VAD (ONNX, GPU fbank)
//                                          └→ TEN VAD (ONNX, CPU)
//                                          └→ Speaker Encoder (CAM++ GPU)
//                                          └→ ASR (Qwen3-ASR, native CUDA)
// Runs a processing thread that pulls from the ring buffer, computes
// Mel frames on GPU, runs VAD engines, extracts speaker embeddings, and reports results.

#pragma once

#include "mel_gpu.h"
#include "silero_vad.h"
#include "fsmn_vad.h"
#include "fsmn_fbank_gpu.h"
#include "ten_vad_wrapper.h"
#include "vad.h"
#include "asr/asr_engine.h"
#include "../../communis/ring_buffer.h"
#include "../../orator/speaker_encoder.h"
#include "../../orator/onnx_speaker_encoder.h"
#include "../../orator/wavlm_ecapa_encoder.h"
#include "../../orator/speaker_db.h"
#include "../../orator/speaker_vector_store.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <string>

namespace deusridet {

// Which VAD engine drives the speech detection for speaker extraction.
enum class VadSource : int {
    SILERO = 0,
    FSMN   = 1,
    TEN    = 2,
    ANY    = 3,  // OR of all enabled VADs
    DIRECT = 4,  // bypass VAD — ASR triggers on buffer duration only
};

struct AudioPipelineConfig {
    MelConfig mel;
    VadConfig vad;
    SileroVadConfig silero;             // Silero VAD model config
    FsmnVadConfig fsmn;                 // FSMN VAD model config
    TenVadConfig  ten;                  // TEN VAD model config
    SpeakerEncoderConfig speaker;       // CAM++ speaker encoder config
    OnnxSpeakerConfig wavlm;              // WavLM speaker encoder config
    OnnxSpeakerConfig unispeech;          // ECAPA-TDNN speaker encoder config (uses fbank, not raw PCM)
    std::string wavlm_ecapa_model;         // WavLM-Large+ECAPA-TDNN safetensors path (native GPU)
    float wavlm_ecapa_threshold = 0.55f;   // default cosine sim threshold
    std::string asr_model_path;            // Qwen3-ASR model directory (empty = disabled)
    size_t ring_buffer_bytes = 1 << 20;  // 1 MB (~32 seconds of int16 mono 16kHz)
    int process_chunk_ms     = 100;      // process in 100ms chunks (10 mel frames)
    float speaker_threshold  = 0.50f;    // CAM++ cosine sim match threshold
    float wavlm_threshold    = 0.80f;    // WavLM Gemm threshold (same ~0.86-0.93, diff ~0.36-0.76)
    float unispeech_threshold= 0.55f;    // ECAPA-TDNN threshold (same ~0.57, diff ~0.03-0.45)
};

struct AudioPipelineStats {
    uint64_t pcm_samples_in;   // total PCM samples received
    uint64_t mel_frames;       // total mel frames computed
    uint64_t speech_frames;    // mel frames classified as speech (energy VAD)
    float    last_rms;         // latest frame RMS (linear)
    float    last_energy;      // latest frame mean log-energy
    bool     is_speech;        // current energy VAD state
    // Silero VAD.
    float    silero_prob;      // latest Silero speech probability [0,1]
    bool     silero_speech;    // Silero VAD speech state
    // FSMN VAD.
    float    fsmn_prob;        // latest FSMN speech probability [0,1]
    bool     fsmn_speech;      // FSMN VAD speech state
    // TEN VAD.
    float    ten_prob;         // latest TEN speech probability [0,1]
    bool     ten_speech;       // TEN VAD speech state
    // Speaker identification (CAM++).
    int      speaker_id;       // current speaker ID (-1 = unknown)
    float    speaker_sim;      // best cosine similarity
    bool     speaker_new;      // true if newly registered speaker
    int      speaker_count;    // number of known speakers
    int      speaker_exemplars;  // exemplar count for matched speaker
    int      speaker_hits_above; // exemplars above threshold in this match
    char     speaker_name[64]; // current speaker name (empty if unnamed)
    // Speaker identification (WavLM).
    int      wavlm_id;         // WavLM speaker ID
    float    wavlm_sim;        // WavLM best similarity
    bool     wavlm_new;
    int      wavlm_count;
    char     wavlm_name[64];
    // Speaker identification (UniSpeech-SAT).
    int      unispeech_id;
    float    unispeech_sim;
    bool     unispeech_new;
    int      unispeech_count;
    char     unispeech_name[64];
    // Speaker identification (WavLM-Large + ECAPA-TDNN, native GPU).
    int      wlecapa_id;
    float    wlecapa_sim;
    bool     wlecapa_new;
    int      wlecapa_count;
    int      wlecapa_exemplars;      // exemplar count for matched speaker
    int      wlecapa_hits_above;     // exemplars above threshold in this match
    char     wlecapa_name[64];
    // Active flags — true only on the tick when extraction happened.
    bool     speaker_active;
    bool     wavlm_active;
    bool     unispeech_active;
    bool     wlecapa_active;
    // WL-ECAPA latency breakdown (ms), set after each extraction.
    float    wlecapa_lat_cnn_ms;
    float    wlecapa_lat_encoder_ms;
    float    wlecapa_lat_ecapa_ms;
    float    wlecapa_lat_total_ms;
    bool     wlecapa_is_early;       // true if this was an early extraction (not end-of-segment)
    // Change detection: cosine similarity between consecutive segment embeddings.
    float    wlecapa_change_sim;     // -1 if no previous segment
    bool     wlecapa_change_valid;   // true when change_sim is meaningful
    // ASR (Qwen3-ASR).
    bool     asr_active;             // true on tick when new transcript is ready
    bool     asr_busy;               // true when ASR thread is processing
    float    asr_latency_ms;         // transcription latency
    float    asr_audio_duration_s;   // audio segment duration
    float    asr_buf_sec;            // current ASR buffer duration (seconds)
    bool     asr_buf_has_speech;     // buffer contains detected speech
    // SAAS: adaptive post-silence feedback.
    int      asr_effective_silence_ms; // currently effective post-silence threshold
    int      asr_post_silence_ms;    // current accumulated post-silence (ms)
};

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

    // Resolve the best speaker label for a given audio sample range.
    // Uses weighted majority voting with overlap-proportional weighting.
    // If no events overlap the query range, performs context fill:
    // finds the nearest event before and after — if they agree on speaker, fills.
    ResolvedSpeaker resolve(int64_t start_sample, int64_t end_sample) const {
        if (count_ == 0 || end_sample <= start_sample)
            return {};

        // Source authority weights.
        static constexpr float kWeights[] = {
            0.70f,  // SAAS_EARLY   — early extraction, might not match final
            1.00f,  // SAAS_FULL    — end-of-segment, highest authority
            0.90f,  // SAAS_CHANGE  — speaker change detection, high authority
            0.50f,  // SAAS_INHERIT — inherited from previous, could be wrong
            0.40f,  // TRACKER      — independent pipeline, lower authority
        };

        int64_t query_len = end_sample - start_sample;

        // Accumulate weighted votes per speaker_id.
        static constexpr int kMaxSpk = 64;
        float votes[kMaxSpk] = {};
        float best_sim[kMaxSpk] = {};
        char  best_name[kMaxSpk][64] = {};
        SpkEventSource best_source[kMaxSpk] = {};
        float best_source_weight[kMaxSpk] = {};
        bool any_overlap = false;

        // Also track nearest non-overlapping events for context fill.
        int64_t nearest_before_dist = INT64_MAX;
        int     nearest_before_spk  = -1;
        float   nearest_before_sim  = 0.0f;
        char    nearest_before_name[64] = {};
        SpkEventSource nearest_before_src = SpkEventSource::SAAS_EARLY;

        int64_t nearest_after_dist  = INT64_MAX;
        int     nearest_after_spk   = -1;
        float   nearest_after_sim   = 0.0f;
        char    nearest_after_name[64] = {};
        SpkEventSource nearest_after_src = SpkEventSource::SAAS_EARLY;

        int start_idx = (count_ < kMaxEvents) ? 0 : write_pos_;
        for (int i = 0; i < count_; ++i) {
            int idx = (start_idx + i) % kMaxEvents;
            const auto& ev = events_[idx];
            if (ev.speaker_id < 0 || ev.speaker_id >= kMaxSpk) continue;

            // Compute overlap between event range and query range.
            int64_t ov_start = std::max(ev.audio_start, start_sample);
            int64_t ov_end   = std::min(ev.audio_end, end_sample);
            if (ov_end > ov_start) {
                any_overlap = true;
                float overlap_frac = (float)(ov_end - ov_start) / (float)query_len;
                float w = kWeights[static_cast<int>(ev.source)] * overlap_frac;
                votes[ev.speaker_id] += w;

                // Track best similarity per speaker (prefer highest-authority source).
                float sw = kWeights[static_cast<int>(ev.source)];
                if (sw > best_source_weight[ev.speaker_id] ||
                    (sw == best_source_weight[ev.speaker_id] &&
                     ev.similarity > best_sim[ev.speaker_id])) {
                    best_sim[ev.speaker_id] = ev.similarity;
                    memcpy(best_name[ev.speaker_id], ev.name, 64);
                    best_source[ev.speaker_id] = ev.source;
                    best_source_weight[ev.speaker_id] = sw;
                }
            } else {
                // No overlap — track nearest events for context fill.
                if (ev.audio_end <= start_sample) {
                    int64_t dist = start_sample - ev.audio_end;
                    if (dist < nearest_before_dist) {
                        nearest_before_dist = dist;
                        nearest_before_spk  = ev.speaker_id;
                        nearest_before_sim  = ev.similarity;
                        memcpy(nearest_before_name, ev.name, 64);
                        nearest_before_src  = ev.source;
                    }
                } else if (ev.audio_start >= end_sample) {
                    int64_t dist = ev.audio_start - end_sample;
                    if (dist < nearest_after_dist) {
                        nearest_after_dist = dist;
                        nearest_after_spk  = ev.speaker_id;
                        nearest_after_sim  = ev.similarity;
                        memcpy(nearest_after_name, ev.name, 64);
                        nearest_after_src  = ev.source;
                    }
                }
            }
        }

        if (any_overlap) {
            // Find speaker with highest vote.
            int best_id = -1;
            float best_vote = 0.0f;
            for (int i = 0; i < kMaxSpk; ++i) {
                if (votes[i] > best_vote) {
                    best_vote = votes[i];
                    best_id = i;
                }
            }

            if (best_id < 0) return {};

            ResolvedSpeaker result;
            result.speaker_id = best_id;
            result.confidence = best_vote;
            result.similarity = best_sim[best_id];
            memcpy(result.name, best_name[best_id], 64);
            result.source = best_source[best_id];
            return result;
        }

        // Context fill: no overlapping events. Check nearest before/after.
        // If both agree on same speaker AND gap is < 5s (80000 samples), fill.
        // If only one side has a result within 3s (48000 samples), use it.
        static constexpr int64_t kFillBothMaxGap  = 80000;  // 5s
        static constexpr int64_t kFillSingleMaxGap = 48000;  // 3s

        if (nearest_before_spk >= 0 && nearest_after_spk >= 0 &&
            nearest_before_spk == nearest_after_spk &&
            nearest_before_dist + nearest_after_dist < kFillBothMaxGap) {
            // Both neighbors agree — high confidence context fill.
            ResolvedSpeaker result;
            result.speaker_id = nearest_before_spk;
            result.confidence = 0.35f;  // lower confidence for context fill
            // Use the higher-authority source's similarity.
            float sw_b = kWeights[static_cast<int>(nearest_before_src)];
            float sw_a = kWeights[static_cast<int>(nearest_after_src)];
            if (sw_b >= sw_a) {
                result.similarity = nearest_before_sim;
                memcpy(result.name, nearest_before_name, 64);
                result.source = nearest_before_src;
            } else {
                result.similarity = nearest_after_sim;
                memcpy(result.name, nearest_after_name, 64);
                result.source = nearest_after_src;
            }
            return result;
        }

        // Single-side fill: only one neighbor within 3s.
        if (nearest_before_spk >= 0 && nearest_before_dist < kFillSingleMaxGap) {
            ResolvedSpeaker result;
            result.speaker_id = nearest_before_spk;
            result.confidence = 0.20f;  // low confidence single-side fill
            result.similarity = nearest_before_sim;
            memcpy(result.name, nearest_before_name, 64);
            result.source = nearest_before_src;
            return result;
        }
        if (nearest_after_spk >= 0 && nearest_after_dist < kFillSingleMaxGap) {
            ResolvedSpeaker result;
            result.speaker_id = nearest_after_spk;
            result.confidence = 0.15f;  // lowest confidence
            result.similarity = nearest_after_sim;
            memcpy(result.name, nearest_after_name, 64);
            result.source = nearest_after_src;
            return result;
        }

        return {};
    }

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
    void feed(const int16_t* pcm, int n_samples, bool vad_speech);

    // Perform check if interval reached. Returns true if a check was executed.
    // Must be called after feed() in the same loop iteration.
    bool check();

    // Access current stats (read by commands.cpp for JSON).
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
    float self_consistency_     = 0.78f;
    float w_emb_                = 0.50f;
    float w_centroid_           = 0.30f;
    float w_f0_                 = 0.20f;
    int   fast_path_samples_    = 8000;  // 0.5s for VAD-onset fast path
};

class AudioPipeline {
public:
    using OnVadEvent  = std::function<void(const VadResult&, int frame_idx)>;
    using OnStats     = std::function<void(const AudioPipelineStats&)>;
    using OnSpeaker   = std::function<void(const SpeakerMatch&)>;
    using OnTranscript = std::function<void(const asr::ASRResult& result, float audio_sec,
                                             int speaker_id, const std::string& speaker_name,
                                             float speaker_sim, float speaker_confidence,
                                             const std::string& speaker_source,
                                             const std::string& trigger_reason,
                                             int tracker_id, const std::string& tracker_name,
                                             float tracker_sim,
                                             float stream_start_sec, float stream_end_sec)>;
    using OnAsrLog = std::function<void(const std::string& json)>;
    using OnAsrPartial = std::function<void(const std::string& text, float audio_sec)>;

    AudioPipeline();
    ~AudioPipeline();

    AudioPipeline(const AudioPipeline&) = delete;
    AudioPipeline& operator=(const AudioPipeline&) = delete;

    bool start(const AudioPipelineConfig& cfg);
    void stop();
    bool running() const { return running_.load(std::memory_order_relaxed); }

    // Push raw int16 PCM from WS callback (producer thread). Non-blocking.
    void push_pcm(const int16_t* data, int n_samples);

    // Register callbacks.
    void set_on_vad(OnVadEvent cb) { on_vad_ = std::move(cb); }
    void set_on_stats(OnStats cb)  { on_stats_ = std::move(cb); }
    void set_on_speaker(OnSpeaker cb) { on_speaker_ = std::move(cb); }
    void set_on_transcript(OnTranscript cb) { on_transcript_ = std::move(cb); }
    void set_on_asr_log(OnAsrLog cb) { on_asr_log_ = std::move(cb); }
    void set_on_asr_partial(OnAsrPartial cb) { on_asr_partial_ = std::move(cb); }

    const AudioPipelineStats& stats() const { return stats_; }

    // Runtime VAD threshold adjustment (thread-safe: atomic float write).
    void set_vad_threshold(float t) { vad_.set_threshold(t); }
    float vad_threshold() const { return vad_.config().energy_threshold; }
    float vad_noise_floor() const { return vad_.noise_floor(); }

    // Silero VAD threshold.
    void set_silero_threshold(float t) { silero_.set_threshold(t); }
    float silero_threshold() const { return silero_.threshold(); }
    float silero_prob() const { return stats_.silero_prob; }

    // FSMN VAD threshold.
    void set_fsmn_threshold(float t) { fsmn_.set_threshold(t); }
    float fsmn_threshold() const { return fsmn_.threshold(); }

    // TEN VAD threshold.
    void set_ten_threshold(float t) { ten_.set_threshold(t); }
    float ten_threshold() const { return ten_.threshold(); }

    // Per-VAD enable/disable (thread-safe).
    void set_silero_enabled(bool e) { enable_silero_.store(e, std::memory_order_relaxed); }
    bool silero_enabled() const { return enable_silero_.load(std::memory_order_relaxed); }
    void set_fsmn_enabled(bool e) { enable_fsmn_.store(e, std::memory_order_relaxed); }
    bool fsmn_enabled() const { return enable_fsmn_.load(std::memory_order_relaxed); }
    void set_ten_enabled(bool e) { enable_ten_.store(e, std::memory_order_relaxed); }
    bool ten_enabled() const { return enable_ten_.load(std::memory_order_relaxed); }

    // Speaker encoder enable/disable (thread-safe).
    void set_speaker_enabled(bool e) { enable_speaker_.store(e, std::memory_order_relaxed); }
    bool speaker_enabled() const { return enable_speaker_.load(std::memory_order_relaxed); }
    void set_wavlm_enabled(bool e) { enable_wavlm_.store(e, std::memory_order_relaxed); }
    bool wavlm_enabled() const { return enable_wavlm_.load(std::memory_order_relaxed); }
    void set_unispeech_enabled(bool e) { enable_unispeech_.store(e, std::memory_order_relaxed); }
    bool unispeech_enabled() const { return enable_unispeech_.load(std::memory_order_relaxed); }
    void set_wlecapa_enabled(bool e) { enable_wlecapa_.store(e, std::memory_order_relaxed); }
    bool wlecapa_enabled() const { return enable_wlecapa_.load(std::memory_order_relaxed); }

    // Per-backend threshold control.
    void set_speaker_threshold(float t) { speaker_threshold_.store(t, std::memory_order_relaxed); }
    float speaker_threshold() const { return speaker_threshold_.load(std::memory_order_relaxed); }
    void set_wavlm_threshold(float t) { wavlm_threshold_.store(t, std::memory_order_relaxed); }
    float wavlm_threshold() const { return wavlm_threshold_.load(std::memory_order_relaxed); }
    void set_unispeech_threshold(float t) { unispeech_threshold_.store(t, std::memory_order_relaxed); }
    float unispeech_threshold() const { return unispeech_threshold_.load(std::memory_order_relaxed); }
    void set_wlecapa_threshold(float t) { wlecapa_threshold_.store(t, std::memory_order_relaxed); }
    float wlecapa_threshold() const { return wlecapa_threshold_.load(std::memory_order_relaxed); }

    // Early extraction trigger (in seconds of speech).
    void set_early_trigger_sec(float s) { early_trigger_samples_.store((int)(s * 16000), std::memory_order_relaxed); }
    float early_trigger_sec() const { return early_trigger_samples_.load(std::memory_order_relaxed) / 16000.0f; }
    void set_early_trigger_enabled(bool e) { enable_early_.store(e, std::memory_order_relaxed); }
    bool early_trigger_enabled() const { return enable_early_.load(std::memory_order_relaxed); }

    // Minimum speech duration for full-segment speaker ID (in seconds).
    void set_min_speech_sec(float s) { min_speech_samples_.store(std::max(1, (int)(s * 16000)), std::memory_order_relaxed); }
    float min_speech_sec() const { return min_speech_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Intra-segment speaker change detection: re-check speaker identity
    // within long segments to detect speaker transitions missed by VAD.
    void set_spk_recheck_sec(float s) { spk_recheck_samples_.store(std::max(16000, (int)(s * 16000)), std::memory_order_relaxed); }
    float spk_recheck_sec() const { return spk_recheck_samples_.load(std::memory_order_relaxed) / 16000.0f; }
    void set_spk_recheck_enabled(bool e) { enable_spk_recheck_.store(e, std::memory_order_relaxed); }
    bool spk_recheck_enabled() const { return enable_spk_recheck_.load(std::memory_order_relaxed); }
    // Cosine similarity threshold below which we declare a speaker change.
    void set_spk_change_threshold(float t) { spk_change_threshold_.store(t, std::memory_order_relaxed); }
    float spk_change_threshold() const { return spk_change_threshold_.load(std::memory_order_relaxed); }
    // Window size (seconds) for the re-check embedding extraction.
    void set_spk_recheck_window_sec(float s) { spk_recheck_window_samples_.store(std::max(8000, (int)(s * 16000)), std::memory_order_relaxed); }
    float spk_recheck_window_sec() const { return spk_recheck_window_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // ASR (Qwen3-ASR) enable/disable and tunable parameters.
    void set_asr_enabled(bool e) { enable_asr_.store(e, std::memory_order_relaxed); }
    bool asr_enabled() const { return enable_asr_.load(std::memory_order_relaxed); }
    bool asr_loaded() const { return asr_engine_ && asr_engine_->is_loaded(); }

    // Post-silence trigger: ms of silence after speech before triggering ASR.
    void set_asr_post_silence_ms(int ms) { asr_post_silence_ms_.store(std::max(100, ms), std::memory_order_relaxed); }
    int  asr_post_silence_ms() const { return asr_post_silence_ms_.load(std::memory_order_relaxed); }

    // SAAS: adaptive post-silence — dynamically adjusts based on segment length.
    void set_asr_adaptive_silence(bool e) { asr_adaptive_silence_.store(e, std::memory_order_relaxed); }
    bool asr_adaptive_silence() const { return asr_adaptive_silence_.load(std::memory_order_relaxed); }
    void set_asr_adaptive_short_ms(int ms) { asr_adaptive_short_ms_.store(std::max(100, ms), std::memory_order_relaxed); }
    int  asr_adaptive_short_ms() const { return asr_adaptive_short_ms_.load(std::memory_order_relaxed); }
    void set_asr_adaptive_long_ms(int ms) { asr_adaptive_long_ms_.store(std::max(50, ms), std::memory_order_relaxed); }
    int  asr_adaptive_long_ms() const { return asr_adaptive_long_ms_.load(std::memory_order_relaxed); }
    void set_asr_adaptive_vlong_ms(int ms) { asr_adaptive_vlong_ms_.store(std::max(50, ms), std::memory_order_relaxed); }
    int  asr_adaptive_vlong_ms() const { return asr_adaptive_vlong_ms_.load(std::memory_order_relaxed); }

    // Max buffer duration before forced transcription (seconds).
    void set_asr_max_buf_sec(float s) { asr_max_buf_samples_.store(std::max(16000, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_max_buf_sec() const { return asr_max_buf_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Minimum audio duration to trigger ASR (seconds).
    void set_asr_min_dur_sec(float s) { asr_min_samples_.store(std::max(1600, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_min_dur_sec() const { return asr_min_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Pre-roll: seconds of audio retained after transcription as context.
    void set_asr_pre_roll_sec(float s) { asr_pre_roll_samples_.store(std::max(0, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_pre_roll_sec() const { return asr_pre_roll_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Max new tokens for ASR decoder.
    void set_asr_max_tokens(int t) { asr_max_tokens_.store(std::max(1, std::min(4096, t)), std::memory_order_relaxed); }
    int  asr_max_tokens() const { return asr_max_tokens_.load(std::memory_order_relaxed); }

    // Repetition penalty for ASR decoder.
    void set_asr_rep_penalty(float p);
    float asr_rep_penalty() const { return asr_rep_penalty_.load(std::memory_order_relaxed); }

    // Minimum average energy for ASR segment (reject silence/noise).
    // Adapted from qwen35-thor (voice_session.cpp): min_avg_energy rejection.
    void set_asr_min_energy(float e) { asr_min_energy_.store(std::max(0.0f, e), std::memory_order_relaxed); }
    float asr_min_energy() const { return asr_min_energy_.load(std::memory_order_relaxed); }

    // Streaming ASR partial interval (seconds). 0 = disabled.
    // Adapted from qwen35-thor: STREAMING_ASR_CHUNK_S (~2s partial transcriptions).
    void set_asr_partial_sec(float s) { asr_partial_samples_.store(std::max(0, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_partial_sec() const { return asr_partial_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Minimum speech ratio for ASR trigger (0.0–1.0). Segments with speech_sec / buf_sec
    // below this ratio are skipped (when buffer > 2s). Default 0.15 (15%).
    void set_asr_min_speech_ratio(float r) { asr_min_speech_ratio_.store(std::max(0.0f, std::min(1.0f, r)), std::memory_order_relaxed); }
    float asr_min_speech_ratio() const { return asr_min_speech_ratio_.load(std::memory_order_relaxed); }

    // Per-backend speaker database access.
    SpeakerDb& speaker_db() { return speaker_db_; }
    SpeakerVectorStore& campp_db() { return campp_db_; }
    SpeakerDb& wavlm_db() { return wavlm_db_; }
    SpeakerDb& unispeech_db() { return unispeech_db_; }
    SpeakerVectorStore& wlecapa_db() { return wlecapa_db_; }

    // Speaker Tracker (independent pipeline for comparison).
    SpeakerTracker& tracker() { return tracker_; }
    const SpeakerTracker& tracker() const { return tracker_; }

    // Per-backend clear and name.
    void clear_speaker_db() {
        speaker_db_.clear();
        campp_db_.clear();
        stats_.speaker_id = -1; stats_.speaker_sim = 0;
        stats_.speaker_new = false; stats_.speaker_count = 0;
        stats_.speaker_active = true;  // trigger UI refresh
        stats_.speaker_name[0] = '\0';
    }
    void clear_wavlm_db() {
        wavlm_db_.clear();
        stats_.wavlm_id = -1; stats_.wavlm_sim = 0;
        stats_.wavlm_new = false; stats_.wavlm_count = 0;
        stats_.wavlm_active = true;  // trigger UI refresh
        stats_.wavlm_name[0] = '\0';
    }
    void clear_unispeech_db() {
        unispeech_db_.clear();
        stats_.unispeech_id = -1; stats_.unispeech_sim = 0;
        stats_.unispeech_new = false; stats_.unispeech_count = 0;
        stats_.unispeech_active = true;  // trigger UI refresh
        stats_.unispeech_name[0] = '\0';
    }
    void clear_wlecapa_db() {
        wlecapa_db_.clear();
        stats_.wlecapa_id = -1; stats_.wlecapa_sim = 0;
        stats_.wlecapa_new = false; stats_.wlecapa_count = 0;
        stats_.wlecapa_exemplars = 0; stats_.wlecapa_hits_above = 0;
        stats_.wlecapa_active = true;
        stats_.wlecapa_name[0] = '\0';
    }
    void set_speaker_name(int id, const std::string& name) { speaker_db_.set_name(id, name); }
    void set_wavlm_name(int id, const std::string& name) { wavlm_db_.set_name(id, name); }
    void set_unispeech_name(int id, const std::string& name) { unispeech_db_.set_name(id, name); }
    void set_wlecapa_name(int id, const std::string& name) { wlecapa_db_.set_name(id, name); }
    bool remove_wlecapa_speaker(int id) { return wlecapa_db_.remove_speaker(id); }
    bool merge_wlecapa_speakers(int dst_id, int src_id) { return wlecapa_db_.merge_speakers(dst_id, src_id); }

    // Input gain (applied before Mel + VAD). 1.0 = unity.
    void set_gain(float g) { gain_.store(g, std::memory_order_relaxed); }
    float gain() const { return gain_.load(std::memory_order_relaxed); }

    // VAD source selection for speaker extraction pipeline routing.
    void set_vad_source(VadSource s) { vad_source_.store(static_cast<int>(s), std::memory_order_relaxed); }
    VadSource vad_source() const { return static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed)); }

    // VAD source selection for ASR pipeline (independent from speaker).
    void set_asr_vad_source(VadSource s) { asr_vad_source_.store(static_cast<int>(s), std::memory_order_relaxed); }
    VadSource asr_vad_source() const { return static_cast<VadSource>(asr_vad_source_.load(std::memory_order_relaxed)); }

private:
    void process_loop();
    void asr_loop();

    AudioPipelineConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    RingBuffer* ring_ = nullptr;
    MelSpectrogram mel_;
    VoiceActivityDetector vad_;
    SileroVad silero_;
    FsmnVad fsmn_;
    TenVadWrapper ten_;
    SpeakerEncoder speaker_enc_;
    SpeakerVectorStore campp_db_{"CamppDb", 192, 0.15f};
    SpeakerDb speaker_db_{"CAM++Db"};  // legacy — kept for UI/API backward compat
    FsmnFbankGpu speaker_fbank_;  // 80-dim fbank for CAM++
    std::vector<float> seg_fbank_buf_;   // accumulated fbank frames for current speech segment
    bool campp_early_extracted_ = false; // whether CAM++ EARLY has fired this segment

    // CAM++ temporal smoothing: majority voting over recent identifications.
    // Prevents rapid flip-backs between speakers.
    static constexpr int kSmoothWindowSize = 3;
    int smooth_ring_[kSmoothWindowSize] = {-1, -1, -1};  // recent speaker IDs
    int smooth_ring_pos_ = 0;
    int smoothed_speaker_id_ = -1;       // current smoothed speaker
    int campp_full_count_ = 0;           // count FULL extractions for periodic absorption

    // Warm-up spectral clustering: collect embeddings during warm-up,
    // then run one-shot spectral clustering to find speaker count and centroids.
    // After clustering, rebuild the speaker store and lock registration.
    static constexpr int kWarmupCount = 80;  // FULL extractions before clustering
    std::vector<std::vector<float>> warmup_embeddings_;
    std::vector<float> warmup_timestamps_;   // mid-time in seconds
    bool warmup_done_ = false;

    OnnxSpeakerEncoder wavlm_enc_;
    SpeakerDb wavlm_db_{"WavLMDb", 0.1f};       // low EMA to resist centroid contamination
    OnnxSpeakerEncoder unispeech_enc_;
    SpeakerDb unispeech_db_{"ECAPADb", 0.15f};  // ECAPA-TDNN: higher EMA for stable centroids
    WavLMEcapaEncoder wlecapa_enc_;
    SpeakerVectorStore wlecapa_db_{"WLEcapaDb", 192, 0.15f};

    AudioPipelineStats stats_{};
    std::atomic<float> gain_{1.0f};
    std::atomic<int> vad_source_{static_cast<int>(VadSource::SILERO)};
    std::atomic<int> asr_vad_source_{static_cast<int>(VadSource::SILERO)};  // ASR defaults to SILERO (same as speaker)
    std::atomic<bool> enable_silero_{true};
    std::atomic<bool> enable_fsmn_{false};
    std::atomic<bool> enable_ten_{false};
    std::atomic<bool> enable_speaker_{true};    // CAM++ — primary SAAS encoder
    std::atomic<bool> enable_wavlm_{false};
    std::atomic<bool> enable_unispeech_{false};
    std::atomic<bool> enable_wlecapa_{false};    // WL-ECAPA — disabled (CAM++ is primary)
    std::atomic<float> speaker_threshold_{0.50f}; // CAM++ matching threshold
    std::atomic<float> speaker_register_threshold_{0.55f}; // CAM++ registration threshold (pending pool)
    std::atomic<float> wavlm_threshold_{0.80f};
    std::atomic<float> unispeech_threshold_{0.55f};
    std::atomic<float> wlecapa_threshold_{0.55f};
    std::atomic<int>   early_trigger_samples_{27200};  // 1.7s default
    std::atomic<bool>  enable_early_{true};              // early trigger on/off
    std::atomic<int>   min_speech_samples_{16000};       // 1.0s default for full-segment ID
    // Intra-segment speaker change detection.
    std::atomic<int>   spk_recheck_samples_{48000};      // re-check every 3.0s of speech
    std::atomic<bool>  enable_spk_recheck_{true};         // on by default
    std::atomic<float> spk_change_threshold_{0.35f};      // cosine sim below this = different speaker
    std::atomic<int>   spk_recheck_window_samples_{24000}; // 1.5s window for re-check embedding
    std::atomic<bool>  enable_asr_{true};                // ASR on/off
    std::atomic<int>   asr_post_silence_ms_{300};         // post-silence trigger (ms) — base value
    std::atomic<int>   asr_max_buf_samples_{480000};      // max buffer (30s @ 16kHz)

    // SAAS: adaptive post-silence parameters.
    // Actual post-silence = base * multiplier, where multiplier depends on segment length.
    std::atomic<bool>  asr_adaptive_silence_{true};        // enable adaptive post-silence
    std::atomic<int>   asr_adaptive_short_ms_{700};        // post-silence for short segments (<0.8s)
    std::atomic<int>   asr_adaptive_long_ms_{200};         // post-silence for long segments (5-15s)
    std::atomic<int>   asr_adaptive_vlong_ms_{150};        // post-silence for very long segments (>15s)
    std::atomic<int>   asr_min_samples_{8000};            // min audio for ASR (0.5s)
    std::atomic<int>   asr_pre_roll_samples_{1600};       // pre-roll retention (0.1s)
    std::atomic<int>   asr_max_tokens_{448};              // decoder max new tokens
    std::atomic<float> asr_rep_penalty_{1.0f};            // repetition penalty
    std::atomic<float> asr_min_energy_{0.008f};           // min avg energy for ASR segment
    std::atomic<int>   asr_partial_samples_{32000};       // streaming partial interval (2s default)
    std::atomic<float> asr_min_speech_ratio_{0.15f};      // min speech / buffer ratio for trigger

    // ASR engine (Qwen3-ASR).
    std::unique_ptr<asr::ASREngine> asr_engine_;

    // PCM buffer for speech segments (accumulated for speaker embedding).
    std::vector<int16_t> speech_pcm_buf_;
    bool in_speech_segment_ = false;
    bool early_extracted_   = false;   // true after early extraction during speech
    // Intra-segment speaker change detection state.
    std::vector<float> seg_ref_emb_;     // reference embedding for current segment's speaker
    int seg_ref_speaker_id_ = -1;        // speaker ID from early/full extraction
    std::string seg_ref_speaker_name_;   // speaker name from early/full extraction
    float seg_ref_speaker_sim_ = 0.0f;   // speaker similarity from early/full extraction
    int seg_last_recheck_at_ = 0;        // sample position of last re-check
    bool seg_has_ref_ = false;           // true after initial speaker identified

    // SAAS: short-segment speaker inheritance state.
    int prev_seg_speaker_id_ = -1;       // speaker ID from the previous segment
    std::string prev_seg_speaker_name_;  // speaker name from the previous segment
    float prev_seg_speaker_sim_ = 0.0f;  // speaker similarity from the previous segment
    int64_t prev_seg_end_sample_ = 0;    // sample counter at end of previous segment
    int64_t total_samples_in_ = 0;       // monotonic sample counter for gap measurement

    // SAAS: speaker-change ASR split flag.
    bool asr_spk_change_pending_ = false;   // speaker change detected, trigger ASR split
    int  asr_spk_change_split_at_ = 0;      // sample position in asr_pcm_buf_ to split at

    // ASR buffer absolute sample tracking for timeline resolution.
    int64_t asr_buf_start_sample_ = 0;      // absolute sample position of asr_pcm_buf_[0]

    // Speaker timeline: fused speaker resolution across SAAS + Tracker events.
    SpeakerTimeline spk_timeline_;

    // ASR audio accumulation: ALL audio is accumulated (Whisper handles silence
    // naturally). VAD is used only to decide WHEN to trigger transcription and
    // to track speech content ratio for filtering mostly-silence segments.
    std::vector<int16_t> asr_pcm_buf_;
    bool asr_saw_speech_    = false;   // any speech detected in current accumulation window
    int  asr_post_silence_  = 0;       // silence chunks after last speech (for trigger)
    int  asr_speech_samples_ = 0;      // samples accumulated while VAD=speech (content quality metric)
    int  asr_partial_sent_at_ = 0;     // buffer size (samples) at last partial submission

    // ASR async thread — transcription runs off-process_loop to avoid blocking.
    struct ASRJob {
        std::vector<float> pcm_f32;     // speech audio, already int16→float32
        float audio_duration_sec;
        std::string trigger_reason;     // "post_silence" or "buffer_full" or "streaming_partial"
        bool is_partial = false;        // streaming partial — don't count as final transcript
        // Stream position (absolute time from start of audio stream, in seconds).
        float stream_start_sec = 0.0f;  // start of this segment in stream time
        float stream_end_sec   = 0.0f;  // end of this segment in stream time
        // Speaker identification from timeline fusion.
        int speaker_id = -1;            // resolved speaker ID (-1 = unknown)
        std::string speaker_name;       // resolved speaker name
        float speaker_sim = 0.0f;       // similarity from best-authority source
        float speaker_confidence = 0.0f; // timeline fusion confidence (weighted vote)
        std::string speaker_source;      // source name ("SAAS_FULL", "TRACKER", etc.)
        // SpeakerTracker snapshot (independent pipeline for A/B comparison).
        int tracker_id = -1;
        std::string tracker_name;
        float tracker_sim = 0.0f;
    };
    std::thread asr_thread_;
    std::mutex asr_mutex_;
    std::condition_variable asr_cv_;
    std::queue<ASRJob> asr_queue_;
    std::atomic<bool> asr_busy_{false};

    // Change detection: previous segment embedding for inter-segment cosine similarity.
    std::vector<float> prev_wlecapa_emb_;  // 192-dim, empty if first segment

    // Speaker Tracker (independent pipeline for A/B comparison with SAAS).
    SpeakerTracker tracker_;

    OnVadEvent on_vad_;
    OnStats    on_stats_;
    OnSpeaker  on_speaker_;
    OnTranscript on_transcript_;
    OnAsrLog   on_asr_log_;
    OnAsrPartial on_asr_partial_;
};

} // namespace deusridet
