// speaker_change_detector.h — Lightweight speaker change detection
//
// Reuses the WavLM CNN frontend (7 conv layers, ~1ms for 0.5s audio) to
// extract 512-dim pooled features from short audio windows. Compares
// consecutive windows via cosine similarity; a drop below threshold
// signals a speaker change point.
//
// Designed to run during speech accumulation (between full speaker ID
// extractions) to detect mid-segment speaker transitions. The pipeline
// can then split segments at change points for separate identification.

#pragma once

#include <vector>

namespace deusridet {

class WavLMEcapaEncoder;

class SpeakerChangeDetector {
public:
    SpeakerChangeDetector();
    ~SpeakerChangeDetector();

    SpeakerChangeDetector(const SpeakerChangeDetector&) = delete;
    SpeakerChangeDetector& operator=(const SpeakerChangeDetector&) = delete;

    // Initialize with a shared encoder (borrows CNN frontend, does not own).
    // Encoder must be initialized and outlive this object.
    bool init(WavLMEcapaEncoder* encoder);
    bool initialized() const { return initialized_; }

    // Feed a PCM window (host float32, 16kHz, normalized [-1,1]).
    // n_samples should be >= window_size (default 8000 = 0.5s).
    // Returns true if a speaker change is detected relative to the previous window.
    bool feed(const float* pcm, int n_samples);

    // Reset state (call after silence gap, segment boundary, etc.)
    void reset();

    // Accessors
    float last_similarity() const { return last_sim_; }
    const std::vector<float>& similarity_history() const { return sim_history_; }

    // Configuration
    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }
    void set_min_interval_samples(int s) { min_interval_ = s; }

    static constexpr int kCnnDim = 512;  // WavLM CNN output channels
    static constexpr int kDefaultWindowSamples = 8000;   // 0.5s at 16kHz
    static constexpr int kMinSamples = 3200;             // 0.2s minimum

private:
    bool initialized_ = false;
    WavLMEcapaEncoder* encoder_ = nullptr;

    // GPU buffer for audio window upload
    float* d_window_ = nullptr;
    size_t window_cap_ = 0;

    // GPU buffer for pooled features [512]
    float* d_pooled_ = nullptr;

    // Host-side state
    std::vector<float> prev_features_;
    float last_sim_ = 1.0f;
    std::vector<float> sim_history_;
    bool has_prev_ = false;
    int samples_since_change_ = 0;

    // Parameters
    float threshold_ = 0.50f;       // cosine sim drop threshold
    int   min_interval_ = 16000;    // 1s between detected changes
};

} // namespace deusridet
