// speaker_stream.h — Independent speaker identification stream with Bayesian tracking.
//
// Truly decoupled from ASR: receives ALL audio with VAD metadata, internally
// decides what to accumulate and when to extract. Produces speaker identification
// events on the timeline independently of ASR segmentation.
//
// Core loop (called every audio tick with is_speech from VAD):
//
//   1. If is_speech: accumulate PCM, advance stride counter
//   2. Every stride_samples of new speech, extract embedding
//   3. search_all() → per-speaker similarities
//   4. Bayesian update: P(speaker | observations) with temporal prior
//   5. If MAP speaker changes → emit SpeakerChanged event on timeline
//   6. Always emit SpeakerIdentified event on the Speaker Timeline
//   7. On speech→silence transition: force final extraction (auto_reg=true)
//
// Bayesian model:
//   - Likelihood: P(obs | spk=k) ∝ exp(α · sim_k)     (softmax over sims)
//   - Temporal prior: P(spk=k | prev=j) = (1-ε) if k==j, else ε/(N-1)
//     where ε is the transition probability (lower = stickier)
//   - Unknown class: fixed low prior for unregistered speakers
//   - Posterior: normalized product of likelihood × prior
//   - Change: MAP speaker ≠ current AND posterior confidence > change_threshold
//
// VAD is used INTERNALLY as metadata (only speech PCM goes into extraction buffer),
// but the caller never gates push_audio() — it is called unconditionally every tick.

#pragma once

#include "speaker_timeline.h"  // SpeakerEvent, SpkEventSource, SpeakerTimeline
#include "povey_fbank_gpu.h"   // PoveyFbankGpu for CAM++ FBank
#include "../../orator/speaker_db.h"  // SpeakerMatch
#include "../../orator/wavlm_ecapa_encoder.h"
#include "../../orator/speaker_encoder.h"  // CAM++ SpeakerEncoder
#include "../../orator/speaker_vector_store.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace deusridet {

struct SpeakerStreamConfig {
    int   stride_samples        = 20000;  // identify every 1.25s of new speech
    int   window_samples        = 24000;  // 1.5s window for embedding extraction
    int   min_samples           = 16000;  // minimum 1.0s for meaningful embedding

    // Bayesian HMM parameters.
    float bayesian_alpha        = 6.0f;   // softmax temperature for likelihood
    float transition_prob       = 0.12f;  // ε: probability of speaker change per step
    float unknown_margin        = 0.15f;  // γ: unknown class sim = max_sim - γ (adaptive)
    float unknown_floor         = 0.25f;  // minimum sim for unknown class (floor)
    float forgetting_factor     = 0.92f;  // β: posterior smoothing (prevents runaway posteriors)
    float change_threshold      = 0.65f;  // posterior confidence needed to switch speaker
    int   change_confirm_steps  = 2;      // consecutive steps new speaker must lead before change
    float confirm_threshold     = 0.50f;  // sim threshold below which identify() is unlikely right

    // Warmup clustering: collect embeddings before registering speakers.
    // Agglomerative clustering finds natural speaker groups, eliminating
    // sensitivity to speaker arrival order.
    // Set warmup_embeddings=0 to disable warmup and use greedy online registration.
    int   warmup_embeddings     = 0;      // disabled by default (0 = no warmup)
    float warmup_merge_thresh   = 0.55f;  // cosine sim threshold for merging clusters
    int   warmup_min_cluster    = 2;      // minimum embeddings per cluster to register
};

class SpeakerStream {
public:
    using OnSpeakerChanged = std::function<void(int old_id, int new_id,
                                                 float new_sim, int64_t change_pos)>;

    SpeakerStream() = default;

    void init(WavLMEcapaEncoder* encoder,
              SpeakerVectorStore* store,
              SpeakerTimeline* timeline,
              const SpeakerStreamConfig& cfg = {},
              SpeakerEncoder* cam_encoder = nullptr);

    // Called EVERY tick by audio pipeline, unconditionally.
    // pcm: raw PCM samples for this tick.
    // samples: number of samples.
    // abs_position: monotonic absolute sample count at end of this push.
    // is_speech: VAD metadata — SpeakerStream uses this internally to decide
    //   what to accumulate (only speech PCM), but the caller never gates this call.
    // Returns true if extraction happened this tick (for stats).
    bool push_audio(const int16_t* pcm, int samples, int64_t abs_position, bool is_speech);

    // Whether last push_audio triggered an end-of-speech extraction (SPK_FULL).
    bool last_was_full() const { return last_was_full_; }

    // Current speaker state.
    int   current_speaker_id() const { return current_speaker_id_; }
    float current_confidence() const { return current_confidence_; }

    // Configuration (runtime adjustable).
    void set_stride_sec(float s) { cfg_.stride_samples = std::max(8000, (int)(s * 16000)); }
    float stride_sec() const { return cfg_.stride_samples / 16000.0f; }
    void set_window_sec(float s) { cfg_.window_samples = std::max(8000, (int)(s * 16000)); }
    float window_sec() const { return cfg_.window_samples / 16000.0f; }
    void set_transition_prob(float p) { cfg_.transition_prob = std::max(0.01f, std::min(0.5f, p)); }
    float transition_prob() const { return cfg_.transition_prob; }
    void set_change_threshold(float t) { cfg_.change_threshold = t; }
    float change_threshold() const { return cfg_.change_threshold; }
    void set_bayesian_alpha(float a) { cfg_.bayesian_alpha = a; }
    float bayesian_alpha() const { return cfg_.bayesian_alpha; }

    // Callbacks.
    void set_on_speaker_changed(OnSpeakerChanged cb) { on_changed_ = std::move(cb); }
    void set_on_spk_event(std::function<void(const SpeakerEvent&)> cb) { on_spk_event_ = std::move(cb); }
    void set_on_speaker(std::function<void(const SpeakerMatch&)> cb) { on_speaker_ = std::move(cb); }

    // For stats reporting.
    struct BayesianState {
        int   map_speaker_id = -1;    // MAP estimate
        float map_posterior  = 0.0f;  // posterior of MAP speaker
        float unknown_posterior = 0.0f; // posterior of unknown class
        int   n_speakers = 0;
        int   steps_since_change = 0;
        int   pending_change_id = -1;   // speaker waiting for confirmation (-1 = none)
        int   pending_change_steps = 0; // consecutive steps pending speaker has led
    };
    BayesianState bayesian_state() const;

private:
    // Run one identification step: extract embedding, search, Bayesian update.
    // auto_reg: whether to allow new speaker registration (true at end-of-segment).
    // Returns true if speaker change detected.
    bool identify_step(int64_t seg_start_abs, int64_t seg_end_abs, bool auto_reg);

    // Recursive Bayesian HMM forward update.
    // Accumulates evidence over time via:
    //   1. Predict: apply transition model to previous posteriors
    //   2. Update: multiply by observation likelihood
    //   3. Forget: apply forgetting factor to prevent runaway posteriors
    //   4. Normalize: log-sum-exp normalization
    // Returns MAP speaker_id.
    int bayesian_update(const std::vector<SpeakerVectorStore::SearchResult>& results);

    // Apply forgetting factor: shrink posteriors toward uniform to prevent
    // evidence lock-in that makes speaker changes undetectable.
    void apply_forgetting();

    // Reset Bayesian posteriors (e.g., when speaker DB is cleared).
    void reset_posteriors();

    // Partial reset: shrink posteriors toward uniform by retain_fraction.
    // retain=0.0 → full reset to uniform, retain=1.0 → no change.
    // Used at turn boundaries to reduce bias from previous speaker.
    void partial_reset_posteriors(float retain_fraction);

    SpeakerStreamConfig cfg_;
    WavLMEcapaEncoder*  encoder_ = nullptr;
    SpeakerEncoder*     cam_encoder_ = nullptr;  // CAM++ encoder for dual-encoder fusion
    PoveyFbankGpu        cam_fbank_;              // dedicated FBank for CAM++ (80-dim Povey)
    SpeakerVectorStore* store_   = nullptr;
    SpeakerTimeline*    timeline_ = nullptr;

    // Speech buffer (only speech PCM, gated internally by is_speech).
    std::vector<int16_t> speech_buf_;
    int samples_since_last_id_ = 0;
    int64_t seg_start_abs_ = 0;  // absolute sample position at speech start
    bool was_speech_ = false;    // previous tick's is_speech state
    bool last_was_full_ = false; // true if last push_audio did end-of-speech extraction

    // Recursive Bayesian state: accumulated log-posteriors.
    // Updated via HMM forward pass at each identification step.
    static constexpr int kMaxTracked = 64;
    float log_posterior_[kMaxTracked] = {};  // indexed by external_id (0..63)
    float log_unknown_ = 0.0f;              // log-posterior for "unknown/new"
    bool  has_prior_ = false;               // true after first observation (posteriors are meaningful)
    int   current_speaker_id_ = -1;
    float current_confidence_ = 0.0f;
    int   steps_since_change_ = 0;

    // Change confirmation state: require consecutive evidence before switching.
    int   pending_change_id_ = -1;     // candidate speaker for change (-1 = none)
    int   pending_change_steps_ = 0;   // consecutive steps this candidate has led MAP
    float pending_change_conf_ = 0.0f; // best confidence during pending window

    // Warmup clustering state.
    bool  warmup_complete_ = false;
    std::vector<std::vector<float>> warmup_embs_;  // collected embeddings
    std::vector<int64_t> warmup_starts_;            // segment start positions
    std::vector<int64_t> warmup_ends_;              // segment end positions

    // Run agglomerative clustering on warmup_embs_ and register speakers.
    void run_warmup_clustering();

    bool  initialized_ = false;

    OnSpeakerChanged on_changed_;
    std::function<void(const SpeakerEvent&)> on_spk_event_;
    std::function<void(const SpeakerMatch&)> on_speaker_;
};

} // namespace deusridet
