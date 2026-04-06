// ten_vad_wrapper.cpp — TEN VAD C++ wrapper implementation.
//
// Adapted from TEN Framework (https://github.com/TEN-framework/ten-vad)
// Original: Apache 2.0 with additional conditions.

#include "ten_vad_wrapper.h"
#include "../../communis/log.h"

#include <ten_vad.h>

namespace deusridet {

TenVadWrapper::TenVadWrapper() = default;

TenVadWrapper::~TenVadWrapper() {
    if (handle_) {
        ten_vad_destroy(&handle_);
        handle_ = nullptr;
    }
}

bool TenVadWrapper::init(const TenVadConfig& cfg) {
    threshold_ = cfg.threshold;
    hop_size_  = cfg.hop_size;

    int ret = ten_vad_create2(&handle_, cfg.hop_size, cfg.threshold,
                              cfg.model_path.c_str());
    if (ret != 0 || !handle_) {
        LOG_ERROR("TenVAD", "Failed to create TEN VAD (ret=%d)", ret);
        return false;
    }

    initialized_ = true;
    LOG_INFO("TenVAD", "Loaded model: %s (hop=%d, threshold=%.2f)",
             cfg.model_path.c_str(), cfg.hop_size, cfg.threshold);
    return true;
}

TenVadResult TenVadWrapper::process(const int16_t* pcm, int n_samples) {
    TenVadResult result{};
    if (!initialized_ || !pcm || n_samples != hop_size_) return result;

    float prob = 0.0f;
    int flag = 0;
    int ret = ten_vad_process(handle_, pcm, n_samples, &prob, &flag);
    if (ret != 0) return result;

    result.probability = prob;
    result.is_speech = prob >= threshold_;
    return result;
}

} // namespace deusridet
