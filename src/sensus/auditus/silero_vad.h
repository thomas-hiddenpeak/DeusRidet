// silero_vad.h — Silero VAD v5 native C++ inference (safetensors weights).
//
// Model: ~310K params, STFT(Conv1d) + 4×Conv1d encoder + LSTM + linear decoder.
// Input:  raw PCM float32, 512 new samples + 64 context = 576 total @ 16kHz
// Output: speech probability [0, 1]
// State:  LSTM hidden/cell [1, 128] + 64-sample context carried across calls
//
// Reference: https://github.com/snakers4/silero-vad
//
// Pure CPU implementation — model is too small for GPU kernel launch overhead
// to be justified. On Tegra unified memory, CPU and GPU share the same DRAM.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {

struct SileroVadConfig {
    std::string model_path;            // path to silero_vad.safetensors
    int sample_rate       = 16000;     // must be 16000
    int window_samples    = 512;       // 512 for 16kHz (32ms)
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

    // Model constants (16kHz)
    static constexpr int kContextSize  = 64;
    static constexpr int kWindowSize   = 512;
    static constexpr int kTotalInput   = 576;   // context + window
    static constexpr int kNfft         = 256;
    static constexpr int kHopLength    = 128;
    static constexpr int kPadRight     = 64;    // reflect pad right only
    static constexpr int kPaddedLen    = 640;   // 576 + 64
    static constexpr int kStftBins     = 129;   // n_fft/2 + 1
    static constexpr int kStftFrames   = 4;     // (640-256)/128+1
    static constexpr int kEnc0Out      = 128;
    static constexpr int kEnc1Out      = 64;
    static constexpr int kEnc2Out      = 64;
    static constexpr int kEnc3Out      = 128;
    static constexpr int kEnc0Frames   = 4;     // same as STFT (s=1)
    static constexpr int kEnc1Frames   = 2;     // (4+2-3)/2+1
    static constexpr int kEnc2Frames   = 1;     // (2+2-3)/2+1
    static constexpr int kEnc3Frames   = 1;     // (1+2-3)/1+1
    static constexpr int kLstmHidden   = 128;

    // Weights (CPU, owned)
    std::vector<float> stft_basis_;     // (258, 256)
    std::vector<float> enc_w_[4];       // conv weights
    std::vector<float> enc_b_[4];       // conv biases
    std::vector<float> lstm_wih_;       // (512, 128)
    std::vector<float> lstm_whh_;       // (512, 128)
    std::vector<float> lstm_bih_;       // (512,)
    std::vector<float> lstm_bhh_;       // (512,)
    std::vector<float> dec_w_;          // (128,) — reshaped from (1,128,1)
    float dec_b_ = 0.0f;               // scalar bias

    // Persistent state
    std::vector<float> h_state_;        // (128,) LSTM hidden
    std::vector<float> c_state_;        // (128,) LSTM cell
    std::vector<float> context_;        // (64,)  audio context

    // State machine
    bool in_speech_ = false;
    int  speech_samples_  = 0;
    int  silence_samples_ = 0;
};

} // namespace deusridet
