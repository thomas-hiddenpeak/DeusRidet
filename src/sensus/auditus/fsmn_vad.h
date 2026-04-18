// fsmn_vad.h — FunASR FSMN VAD native C++ inference (safetensors weights).
//
// Pipeline: PCM → GPU Fbank(80-bin, Hamming+preemph, 25ms/10ms) → LFR(5x) → CMVN → FP32 forward → softmax
// Model: ~430K params, 2×Linear + 4×FSMN blocks + 2×Linear output.
// Originally used DynamicQuantizeLinear ONNX; now running FP32 from PyTorch source weights.
//
// Pure CPU inference — model is too small for GPU kernel launch overhead.
// On Tegra unified memory, CPU and GPU share the same DRAM.
//
// Reference: https://github.com/modelscope/FunASR (MIT License)

#pragma once

#include "fsmn_fbank_gpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {

struct FsmnVadConfig {
    std::string model_path;      // path to fsmn_vad.safetensors
    std::string cmvn_path;       // path to am.mvn
    int sample_rate   = 16000;
    float threshold   = 0.5f;
    int n_mels        = 80;
    int frame_length_ms = 25;    // 25ms → 400 samples
    int frame_shift_ms  = 10;    // 10ms → 160 samples hop
    int lfr_m         = 5;       // concatenate 5 frames
    int lfr_n         = 1;
};

struct FsmnVadResult {
    float probability;
    bool  is_speech;
};

class FsmnVad {
public:
    FsmnVad();
    ~FsmnVad();

    FsmnVad(const FsmnVad&) = delete;
    FsmnVad& operator=(const FsmnVad&) = delete;

    bool init(const FsmnVadConfig& cfg);
    FsmnVadResult process(const int16_t* pcm, int n_samples);

    // Direct feature input for testing (bypasses GPU Fbank)
    float forward(const float* feats, int n_frames, int feat_dim);

    void reset_state();

    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }
    bool initialized() const { return initialized_; }

private:
    int apply_lfr_cmvn(std::vector<float>& out_feats);

    FsmnVadConfig cfg_;
    bool initialized_ = false;
    float threshold_   = 0.5f;

    FsmnFbankGpu fbank_gpu_;

    // Architecture constants
    static constexpr int kInputDim      = 400;  // 80 mel * 5 LFR
    static constexpr int kAffineIn      = 140;
    static constexpr int kLinearDim     = 250;
    static constexpr int kProjDim       = 128;
    static constexpr int kFsmnLayers    = 4;
    static constexpr int kLorder        = 20;   // left context order
    static constexpr int kCacheLen      = 19;   // lorder - 1
    static constexpr int kAffineOut     = 140;
    static constexpr int kOutputDim     = 248;

    // Weights (CPU, owned) — all stored as row-major (in, out) for x @ W
    // Input projection
    std::vector<float> in_linear1_w_;   // (400, 140)
    std::vector<float> in_linear1_b_;   // (140,)
    std::vector<float> in_linear2_w_;   // (140, 250)
    std::vector<float> in_linear2_b_;   // (250,)

    // FSMN blocks: linear(250→128) + depthwise_conv(128,k=20) + affine(128→250)
    struct FsmnBlock {
        std::vector<float> linear_w;    // (250, 128) — no bias
        std::vector<float> conv_w;      // (128, 20)
        std::vector<float> affine_w;    // (128, 250)
        std::vector<float> affine_b;    // (250,)
    };
    FsmnBlock fsmn_[kFsmnLayers];

    // Output projection
    std::vector<float> out_linear1_w_;  // (250, 140)
    std::vector<float> out_linear1_b_;  // (140,)
    std::vector<float> out_linear2_w_;  // (140, 248)
    std::vector<float> out_linear2_b_;  // (248,)

    // Streaming state: depthwise conv caches
    std::vector<float> caches_[kFsmnLayers];  // each (128 * 19)

    // CMVN parameters
    std::vector<float> cmvn_mean_;
    std::vector<float> cmvn_istd_;

    // Fbank frame accumulator
    std::vector<std::vector<float>> fbank_buf_;
    int lfr_consumed_ = 0;
};

} // namespace deusridet
