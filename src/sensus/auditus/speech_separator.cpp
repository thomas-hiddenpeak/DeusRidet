// speech_separator.cpp — MossFormer2 native CUDA inference for speech separation.
//
// Segmented processing with overlap-add for audio > 2s.
// Lazy loading support to conserve memory when separation is not needed.
// GPU inference via MossFormer2 (cuBLAS + custom CUDA kernels).

#include "speech_separator.h"
#include "../../communis/log.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

namespace deusridet {

SpeechSeparator::SpeechSeparator() = default;

SpeechSeparator::~SpeechSeparator() {
    unload();
}

bool SpeechSeparator::init(const SpeechSeparatorConfig& cfg) {
    cfg_ = cfg;
    initialized_ = true;

    if (!cfg_.lazy_load) {
        return ensure_loaded();
    }

    LOG_INFO("Separator", "Initialized (lazy_load=%s, model=%s)",
             cfg_.lazy_load ? "true" : "false", cfg_.model_path.c_str());
    return true;
}

bool SpeechSeparator::ensure_loaded() {
    if (loaded_) return true;
    if (!initialized_) return false;

    // Create CUDA stream for I/O transfers.
    cudaStreamCreate(&cuda_stream_);

    // Initialize native MossFormer2 model.
    // MossFormer2::init creates its own internal stream for compute.
    if (!mf2_.init(cfg_.model_path, cfg_.max_chunk)) {
        LOG_ERROR("Separator", "MossFormer2 init failed: %s", cfg_.model_path.c_str());
        return false;
    }

    // Allocate GPU I/O buffers for max chunk size.
    size_t max_bytes = cfg_.max_chunk * sizeof(float);
    cudaMalloc(&d_input_,   max_bytes);
    cudaMalloc(&d_source1_, max_bytes);
    cudaMalloc(&d_source2_, max_bytes);

    loaded_ = true;
    LOG_INFO("Separator", "MossFormer2 native CUDA loaded: %s (max_chunk=%d)",
             cfg_.model_path.c_str(), cfg_.max_chunk);
    return true;
}

void SpeechSeparator::unload() {
    if (d_input_)   { cudaFree(d_input_);   d_input_   = nullptr; }
    if (d_source1_) { cudaFree(d_source1_); d_source1_ = nullptr; }
    if (d_source2_) { cudaFree(d_source2_); d_source2_ = nullptr; }
    if (cuda_stream_) { cudaStreamDestroy(cuda_stream_); cuda_stream_ = nullptr; }
    loaded_ = false;
}

SeparationResult SpeechSeparator::separate(const float* pcm, int n_samples) {
    SeparationResult result{};
    result.valid = false;

    if (!initialized_ || !pcm || n_samples <= 0) return result;

    if (!loaded_) {
        if (!ensure_loaded()) return result;
    }

    if (n_samples <= cfg_.max_chunk) {
        // Single chunk — direct processing.
        if (!separate_chunk(pcm, n_samples, result.source1, result.source2))
            return result;
    } else {
        // Segmented processing with overlap-add.
        result.source1.resize(n_samples, 0.0f);
        result.source2.resize(n_samples, 0.0f);

        int step = cfg_.max_chunk - cfg_.overlap_samples;
        if (step <= 0) step = cfg_.max_chunk / 2;

        // Overlap-add weights (Hann window on overlap regions).
        for (int offset = 0; offset < n_samples; offset += step) {
            int chunk_len = std::min(cfg_.max_chunk, n_samples - offset);
            if (chunk_len < 1600) break;  // skip tiny tail (<100ms)

            std::vector<float> s1, s2;
            if (!separate_chunk(pcm + offset, chunk_len, s1, s2))
                return result;

            // Overlap-add: simple crossfade in overlap region.
            for (int i = 0; i < chunk_len; i++) {
                int pos = offset + i;
                if (pos >= n_samples) break;

                float weight = 1.0f;
                // Fade-in at chunk start (if not first chunk).
                if (offset > 0 && i < cfg_.overlap_samples) {
                    weight = (float)i / (float)cfg_.overlap_samples;
                }
                // Fade-out at chunk end (if not last chunk).
                int remaining = chunk_len - i;
                if (offset + step < n_samples && remaining <= cfg_.overlap_samples) {
                    float fade_out = (float)remaining / (float)cfg_.overlap_samples;
                    weight = std::min(weight, fade_out);
                }

                if (offset > 0 && i < cfg_.overlap_samples) {
                    // In overlap region, blend with previous chunk.
                    result.source1[pos] = result.source1[pos] * (1.0f - weight) +
                                          s1[i] * weight;
                    result.source2[pos] = result.source2[pos] * (1.0f - weight) +
                                          s2[i] * weight;
                } else {
                    result.source1[pos] = s1[i];
                    result.source2[pos] = s2[i];
                }
            }
        }
    }

    // RMS normalization: the model outputs at an arbitrary scale.
    // ClearVoice normalizes each output source to match the input RMS.
    float rms_input = 0.0f;
    {
        double sum_sq = 0.0;
        for (int i = 0; i < n_samples; i++) sum_sq += (double)pcm[i] * pcm[i];
        rms_input = std::sqrt((float)(sum_sq / n_samples));
    }
    result.energy1 = compute_rms(result.source1);
    result.energy2 = compute_rms(result.source2);

    if (rms_input > 1e-8f) {
        if (result.energy1 > 1e-8f) {
            float scale1 = rms_input / result.energy1;
            for (float& v : result.source1) v *= scale1;
            result.energy1 = rms_input;
        }
        if (result.energy2 > 1e-8f) {
            float scale2 = rms_input / result.energy2;
            for (float& v : result.source2) v *= scale2;
            result.energy2 = rms_input;
        }
    }

    result.valid = true;
    return result;
}

bool SpeechSeparator::separate_chunk(const float* pcm, int n_samples,
                                      std::vector<float>& out1,
                                      std::vector<float>& out2) {
    if (!loaded_) return false;

    // Copy input H→D.
    cudaMemcpyAsync(d_input_, pcm, n_samples * sizeof(float),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaStreamSynchronize(cuda_stream_);

    // Run native MossFormer2 forward pass.
    if (!mf2_.forward(d_input_, d_source1_, d_source2_, n_samples)) {
        LOG_ERROR("Separator", "MossFormer2 forward failed (n=%d)", n_samples);
        return false;
    }

    // Copy outputs D→H.
    out1.resize(n_samples);
    out2.resize(n_samples);
    cudaMemcpy(out1.data(), d_source1_, n_samples * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out2.data(), d_source2_, n_samples * sizeof(float),
               cudaMemcpyDeviceToHost);

    return true;
}

float SpeechSeparator::compute_rms(const std::vector<float>& pcm) {
    if (pcm.empty()) return 0.0f;
    double sum_sq = 0.0;
    for (float v : pcm) sum_sq += (double)v * v;
    return std::sqrt((float)(sum_sq / pcm.size()));
}

} // namespace deusridet
