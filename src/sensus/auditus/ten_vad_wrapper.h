// ten_vad_wrapper.h — TEN VAD (Agora) C++ wrapper.
//
// Wraps the C API of TEN VAD for integration into the audio pipeline.
// TEN VAD: STFT → biquad → pitch estimation → feature conversion → AED (ONNX).
// Input: int16 PCM, hop_size samples per call (160 or 256 @ 16kHz).
// Output: speech probability [0, 1].
//
// Reference: https://github.com/TEN-framework/ten-vad
// Licensed under Apache 2.0 with additional conditions.

#pragma once

#include <string>

namespace deusridet {

struct TenVadConfig {
    std::string model_path;       // path to ten-vad.onnx
    int hop_size      = 160;      // 160 samples = 10ms @ 16kHz
    float threshold   = 0.5f;     // speech probability threshold
};

struct TenVadResult {
    float probability;  // [0, 1]
    bool  is_speech;    // probability >= threshold
};

class TenVadWrapper {
public:
    TenVadWrapper();
    ~TenVadWrapper();

    TenVadWrapper(const TenVadWrapper&) = delete;
    TenVadWrapper& operator=(const TenVadWrapper&) = delete;

    bool init(const TenVadConfig& cfg);

    // Process one hop of int16 PCM (cfg.hop_size samples).
    TenVadResult process(const int16_t* pcm, int n_samples);

    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }
    bool initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    float threshold_   = 0.5f;
    void* handle_      = nullptr;  // ten_vad_handle_t
    int hop_size_      = 160;
};

} // namespace deusridet
