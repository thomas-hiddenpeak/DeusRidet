// fsmn_vad.h — FunASR FSMN VAD wrapper.
//
// Pipeline: PCM → GPU Fbank(80-bin, Hamming+preemph, 25ms/10ms) → LFR(5x) → CMVN → ORT → softmax
// Model: model_quant.onnx uses DynamicQuantizeLinear — TRT incompatible.
// Using ONNX Runtime CPU EP. On Tegra unified memory this shares the same DRAM.
//
// Reference: https://github.com/modelscope/FunASR (MIT License)

#pragma once

#include "fsmn_fbank_gpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {

struct FsmnVadConfig {
    std::string model_path;      // path to model_quant.onnx
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
    void reset_state();

    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }
    bool initialized() const { return initialized_; }

private:
    int apply_lfr_cmvn(std::vector<float>& out_feats);
    float run_onnx(const float* feats, int n_frames, int feat_dim);

    FsmnVadConfig cfg_;
    bool initialized_ = false;
    float threshold_   = 0.5f;

    FsmnFbankGpu fbank_gpu_;

    void* env_     = nullptr;  // Ort::Env*
    void* session_ = nullptr;  // Ort::Session*

    static constexpr int NUM_CACHES = 4;
    static constexpr int CACHE_DIM  = 128;
    static constexpr int CACHE_LEN  = 19;
    std::vector<float> caches_[NUM_CACHES];

    std::vector<float> cmvn_mean_;
    std::vector<float> cmvn_istd_;

    std::vector<std::vector<float>> fbank_buf_;
    int lfr_consumed_ = 0;
};

} // namespace deusridet
