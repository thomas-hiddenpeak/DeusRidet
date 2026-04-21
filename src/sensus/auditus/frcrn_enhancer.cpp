/**
 * @file frcrn_enhancer.cpp
 * @philosophical_role FRCRN speech-enhancement host-side driver. The pre-ASR noise-reduction step — the entity cleans its own hearing before it attends.
 * @serves Auditus pipeline, optional pre-ASR stage.
 */
// frcrn_enhancer.cpp — FRCRN speech enhancement via custom CUDA inference.
//
// Delegates to FrcrnGpu for actual GPU inference.
// Maintains streaming accumulation buffer and int16/float conversion.
//
// Adapted from ModelScope iic/speech_frcrn_ans_cirm_16k (Apache-2.0).

#include "frcrn_enhancer.h"
#include "frcrn_gpu.h"
#include "../../communis/log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>

namespace deusridet {

FrcrnEnhancer::FrcrnEnhancer() = default;

FrcrnEnhancer::~FrcrnEnhancer() = default;

bool FrcrnEnhancer::init(const FrcrnConfig& cfg) {
    cfg_ = cfg;

    if (cfg_.weights_dir.empty()) {
        LOG_WARN("FRCRN", "No weights directory configured — disabled");
        return false;
    }

    gpu_ = std::make_unique<FrcrnGpu>();

    // Max samples: chunk_samples + some padding for hop alignment
    int max_samples = cfg_.chunk_samples + cfg_.hop_samples * 2;

    if (!gpu_->init(cfg_.weights_dir, max_samples)) {
        LOG_ERROR("FRCRN", "GPU initialization failed");
        gpu_.reset();
        return false;
    }

    initialized_ = true;
    LOG_INFO("FRCRN", "CUDA FRCRN initialized (chunk=%d samples = %d ms)",
             cfg_.chunk_samples, cfg_.chunk_samples * 1000 / cfg_.sample_rate);
    return true;
}

std::vector<float> FrcrnEnhancer::enhance(const float* pcm, int n_samples) {
    std::vector<float> output(n_samples, 0.0f);
    if (!initialized_ || !pcm || n_samples <= 0) return output;

    auto t0 = std::chrono::steady_clock::now();

    int out_len = gpu_->enhance_host(pcm, output.data(), n_samples);
    if (out_len <= 0) {
        // GPU enhance failed — return original audio
        std::memcpy(output.data(), pcm, n_samples * sizeof(float));
    }

    auto t1 = std::chrono::steady_clock::now();
    last_latency_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return output;
}

void FrcrnEnhancer::push_pcm(const int16_t* pcm, int n_samples) {
    if (!initialized_ || !pcm || n_samples <= 0) return;

    // Convert int16 → float and accumulate.
    size_t prev_size = accum_buf_.size();
    accum_buf_.resize(prev_size + n_samples);
    for (int i = 0; i < n_samples; i++) {
        accum_buf_[prev_size + i] = pcm[i] / 32768.0f;
    }

    // Process full chunks and queue output.
    int chunk = cfg_.chunk_samples;
    while ((int)accum_buf_.size() >= chunk) {
        std::vector<float> enhanced = enhance(accum_buf_.data(), chunk);

        // Convert float → int16 and append to output queue.
        size_t base = output_buf_.size();
        output_buf_.resize(base + chunk);
        for (int i = 0; i < chunk; i++) {
            float s = enhanced[i] * 32768.0f;
            s = std::max(-32768.0f, std::min(32767.0f, s));
            output_buf_[base + i] = (int16_t)s;
        }

        // Remove processed samples from input.
        accum_buf_.erase(accum_buf_.begin(), accum_buf_.begin() + chunk);
    }
}

int FrcrnEnhancer::pull_pcm(int16_t* out, int max_samples) {
    int avail = std::min(max_samples, (int)output_buf_.size());
    if (avail <= 0) return 0;
    std::memcpy(out, output_buf_.data(), avail * sizeof(int16_t));
    output_buf_.erase(output_buf_.begin(), output_buf_.begin() + avail);
    return avail;
}

void FrcrnEnhancer::enhance_inplace(int16_t* pcm, int n_samples) {
    if (!initialized_ || !pcm || n_samples <= 0) return;

    // Convert int16 → float.
    std::vector<float> float_buf(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float_buf[i] = pcm[i] / 32768.0f;
    }

    // Enhance.
    std::vector<float> enhanced = enhance(float_buf.data(), n_samples);

    // Convert back to int16.
    for (int i = 0; i < n_samples; i++) {
        float s = enhanced[i] * 32768.0f;
        s = std::max(-32768.0f, std::min(32767.0f, s));
        pcm[i] = (int16_t)s;
    }
}

void FrcrnEnhancer::reset() {
    accum_buf_.clear();
}

} // namespace deusridet
