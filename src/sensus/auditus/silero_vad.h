// silero_vad.h — Silero VAD v5 wrapper using ONNX Runtime C++ API.
//
// Model: silero_vad.onnx (~2.3 MB, Conv1D + LSTM)
// Input:  raw PCM float32, 512 new samples + 64 context = 576 total @ 16kHz
// Output: speech probability [0, 1]
// State:  LSTM hidden/cell [2, 1, 128] + 64-sample context carried across calls
//
// Reference: https://github.com/snakers4/silero-vad
//
// Note: Silero v5 uses LSTM + If/Loop control flow that TensorRT cannot parse.
// Using ONNX Runtime CPU EP. On Tegra unified memory, CPU and GPU share the
// same physical DRAM — no bandwidth penalty for this 2.3 MB model.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {

struct SileroVadConfig {
    std::string model_path;            // path to silero_vad.onnx
    int sample_rate       = 16000;     // must be 16000 or 8000
    int window_samples    = 512;       // 512 for 16kHz (32ms), 256 for 8kHz
    float threshold       = 0.5f;      // speech probability threshold
    int min_speech_ms     = 250;       // min speech duration to trigger
    int min_silence_ms    = 100;       // min silence duration to end speech
    int speech_pad_ms     = 30;        // padding around speech segments
};

struct SileroVadResult {
    float probability;     // raw speech probability [0, 1]
    bool  is_speech;       // probability > threshold
    bool  segment_start;   // rising edge
    bool  segment_end;     // falling edge
};

class SileroVad {
public:
    SileroVad();
    ~SileroVad();

    SileroVad(const SileroVad&) = delete;
    SileroVad& operator=(const SileroVad&) = delete;

    bool init(const SileroVadConfig& cfg);
    SileroVadResult process(const float* pcm, int n_samples);
    void reset_state();

    void set_threshold(float t) { cfg_.threshold = t; }
    float threshold() const { return cfg_.threshold; }
    bool initialized() const { return initialized_; }

private:
    SileroVadConfig cfg_;
    bool initialized_ = false;

    void* env_       = nullptr;  // Ort::Env*
    void* session_   = nullptr;  // Ort::Session*

    static constexpr int STATE_DIM = 128;
    std::vector<float> state_;    // [2, 1, 128]

    static constexpr int CONTEXT_SIZE_16K = 64;
    static constexpr int CONTEXT_SIZE_8K  = 32;
    std::vector<float> context_;

    bool in_speech_ = false;
    int  speech_samples_  = 0;
    int  silence_samples_ = 0;
};

} // namespace deusridet
