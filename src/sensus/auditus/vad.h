// vad.h — Voice Activity Detection (energy-based, GPU-accelerated).
//
// Simple frame-level VAD operating on Mel spectrogram output.
// Computes mean energy per mel frame and applies threshold + hangover.
// This is a placeholder for pipeline testing — will be replaced by a
// learned VAD or the ASR encoder's own silence detection.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace deusridet {

struct VadConfig {
    float energy_threshold  = -6.0f;   // log-energy threshold (tune per environment)
    int   speech_pad_frames = 3;       // frames to pad before/after speech
    int   min_speech_frames = 5;       // minimum speech segment (50ms at 10ms/frame)
    int   min_silence_frames = 15;     // silence frames before ending segment (150ms)
};

struct VadResult {
    bool  is_speech;        // current frame is speech
    bool  segment_start;    // rising edge: silence → speech
    bool  segment_end;      // falling edge: speech → silence (after hangover)
    int   speech_frames;    // consecutive speech frames in current segment
    float energy;           // mean log-energy of current frame
};

class VoiceActivityDetector {
public:
    VoiceActivityDetector() = default;
    ~VoiceActivityDetector() = default;

    void init(const VadConfig& cfg) { cfg_ = cfg; reset(); }

    // Process one Mel frame (host pointer, n_mels floats of log-mel energy).
    // Returns VAD result for this frame.
    VadResult process_frame(const float* mel_frame, int n_mels);

    // Process a batch of frames (GPU pointer). Results written to host array.
    // mel_frames: [n_frames x n_mels] on device.
    void process_batch(const float* d_mel_frames, int n_frames, int n_mels,
                       VadResult* results);

    void reset();
    const VadConfig& config() const { return cfg_; }

    // Adaptive threshold: tracks noise floor and adjusts.
    void set_threshold(float t) { cfg_.energy_threshold = t; }
    float noise_floor() const { return noise_floor_; }

private:
    VadConfig cfg_;
    int speech_count_  = 0;
    int silence_count_ = 0;
    bool in_speech_    = false;

    // Noise floor estimation (running average of non-speech energy).
    float noise_floor_ = -10.0f;
    float noise_alpha_  = 0.02f;   // EMA smoothing
};

// GPU kernel: compute mean energy per frame for a batch.
// d_mel: [n_frames x n_mels] log-mel values.
// d_energy: [n_frames] output mean energy per frame.
void launch_frame_energy(
    const float* d_mel,
    float* d_energy,
    int n_frames,
    int n_mels,
    cudaStream_t stream = nullptr);

} // namespace deusridet
