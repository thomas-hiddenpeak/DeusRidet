/**
 * @file src/orator/speaker_change_detector.cu
 * @philosophical_role
 *   Speaker-change detector — cosine distance on consecutive WavLM windows. The lightweight test that says *a different voice just started*, driving segmentation upstream of full identification.
 * @serves
 *   Auditus pipeline segmentation (audio_pipeline.cpp) for multi-speaker turn-taking.
 */
// speaker_change_detector.cu — Lightweight speaker change detection
//
// Uses the WavLM CNN frontend to extract 512-dim features from short
// audio windows, then compares consecutive windows via cosine similarity.

#include "speaker_change_detector.h"
#include "wavlm_ecapa_encoder.h"
#include "../communis/log.h"

#include <cuda_runtime.h>
#include <cmath>
#include <numeric>

namespace deusridet {

// Average pool [C, T] over T dimension → [C]
// One thread per channel, sequential sum over T.
__global__ void avg_pool_time_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++)
        sum += in[c * T + t];
    out[c] = sum / (float)T;
}

SpeakerChangeDetector::SpeakerChangeDetector() = default;

SpeakerChangeDetector::~SpeakerChangeDetector() {
    if (d_window_) cudaFree(d_window_);
    if (d_pooled_) cudaFree(d_pooled_);
}

bool SpeakerChangeDetector::init(WavLMEcapaEncoder* encoder) {
    if (!encoder || !encoder->initialized()) {
        LOG_ERROR("SpkChange", "encoder not initialized");
        return false;
    }
    encoder_ = encoder;

    // Pre-allocate GPU buffers
    window_cap_ = kDefaultWindowSamples;
    cudaMalloc(&d_window_, window_cap_ * sizeof(float));
    cudaMalloc(&d_pooled_, kCnnDim * sizeof(float));

    prev_features_.resize(kCnnDim, 0.0f);
    initialized_ = true;
    LOG_INFO("SpkChange", "initialized (threshold=%.2f)", threshold_);
    return true;
}

bool SpeakerChangeDetector::feed(const float* pcm, int n_samples) {
    if (!initialized_ || n_samples < kMinSamples) return false;

    // Grow GPU buffer if needed
    if ((size_t)n_samples > window_cap_) {
        if (d_window_) cudaFree(d_window_);
        window_cap_ = (size_t)n_samples;
        cudaMalloc(&d_window_, window_cap_ * sizeof(float));
    }

    // Upload PCM to GPU
    cudaMemcpy(d_window_, pcm, n_samples * sizeof(float), cudaMemcpyHostToDevice);

    // Run WavLM CNN frontend → [512, T'] in encoder's scratch_a_
    int T_out = 0;
    float* d_cnn = encoder_->test_cnn(d_window_, n_samples, T_out);
    if (!d_cnn || T_out <= 0) return false;

    // Average pool [512, T'] → [512]
    int threads = ((kCnnDim + 31) / 32) * 32;
    avg_pool_time_kernel<<<1, threads>>>(d_cnn, d_pooled_, kCnnDim, T_out);

    // Copy pooled features to host
    std::vector<float> cur(kCnnDim);
    cudaMemcpy(cur.data(), d_pooled_, kCnnDim * sizeof(float), cudaMemcpyDeviceToHost);

    // Cosine similarity with previous window
    bool change = false;
    if (has_prev_) {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (int i = 0; i < kCnnDim; i++) {
            dot += prev_features_[i] * cur[i];
            na  += prev_features_[i] * prev_features_[i];
            nb  += cur[i] * cur[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        last_sim_ = (denom > 1e-8f) ? (dot / denom) : 0.0f;
        sim_history_.push_back(last_sim_);

        if (last_sim_ < threshold_ && samples_since_change_ >= min_interval_) {
            change = true;
            samples_since_change_ = 0;
        }
    }

    prev_features_ = cur;
    has_prev_ = true;
    samples_since_change_ += n_samples;
    return change;
}

void SpeakerChangeDetector::reset() {
    has_prev_ = false;
    last_sim_ = 1.0f;
    samples_since_change_ = 0;
    sim_history_.clear();
}

} // namespace deusridet
