/**
 * @file speech_separator.cpp
 * @philosophical_role Speech separation driver on top of MossFormer2. Invoked only when the overlap detector fires — a deliberate cost-gated capability.
 * @serves Auditus pipeline, Orator per-speaker assignment.
 */
// speech_separator.cpp — MossFormer2 native CUDA inference for speech separation.
//
// Segmented processing follows ClearVoice-style fixed-window decoding for
// audio > 2s: pad, run full windows, discard edge regions, then stitch.
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

namespace {

int padded_length_for_decode(int n_samples, int window, int stride, int give_up) {
    if (n_samples <= window) return window;
    int min_last_start = std::max(0, n_samples - window + give_up);
    int steps = (min_last_start + stride - 1) / stride;
    return steps * stride + window;
}

} // namespace

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

    int window = cfg_.max_chunk;
    int stride = cfg_.max_chunk - cfg_.overlap_samples;
    if (stride <= 0) stride = cfg_.max_chunk / 2;
    int give_up = std::max(0, (window - stride) / 2);

    if (n_samples <= window) {
        // Single chunk — direct processing.
        if (!separate_chunk(pcm, n_samples, result.source1, result.source2))
            return result;
    } else {
        // Segmented processing with full-window padding. MossFormer2 is much
        // less stable when fed arbitrary short tails; ClearVoice always runs
        // fixed windows and drops edge regions before stitching.
        result.source1.resize(n_samples, 0.0f);
        result.source2.resize(n_samples, 0.0f);

        int padded_len = padded_length_for_decode(n_samples, window, stride, give_up);
        std::vector<float> padded(padded_len, 0.0f);
        std::memcpy(padded.data(), pcm, n_samples * sizeof(float));

        for (int offset = 0; offset + window <= padded_len; offset += stride) {
            std::vector<float> s1, s2;
            if (!separate_chunk(padded.data() + offset, window, s1, s2))
                return result;

            int dst_start = offset == 0 ? 0 : offset + give_up;
            int src_start = offset == 0 ? 0 : give_up;
            int dst_end = offset + window - give_up;
            dst_start = std::min(dst_start, n_samples);
            dst_end = std::min(dst_end, n_samples);
            if (dst_end <= dst_start) continue;

            int len = dst_end - dst_start;
            std::memcpy(result.source1.data() + dst_start, s1.data() + src_start,
                        len * sizeof(float));
            std::memcpy(result.source2.data() + dst_start, s2.data() + src_start,
                        len * sizeof(float));
            if (dst_end >= n_samples) break;
        }
    }

    // Keep raw model scale. Per-source RMS normalization can make a weak
    // residual sound like a real speaker; callers that need ClearVoice-style
    // listening output should normalize explicitly at the presentation layer.
    result.energy1 = compute_rms(result.source1);
    result.energy2 = compute_rms(result.source2);

    result.valid = true;
    return result;
}

bool SpeechSeparator::separate_chunk(const float* pcm, int n_samples,
                                      std::vector<float>& out1,
                                      std::vector<float>& out2) {
    if (!loaded_) return false;

    int forward_samples = n_samples;
    std::vector<float> padded;
    if (n_samples < cfg_.max_chunk) {
        padded.resize(cfg_.max_chunk, 0.0f);
        std::memcpy(padded.data(), pcm, n_samples * sizeof(float));
        pcm = padded.data();
        forward_samples = cfg_.max_chunk;
    }

    // Copy input H→D.
    cudaMemcpyAsync(d_input_, pcm, forward_samples * sizeof(float),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaStreamSynchronize(cuda_stream_);

    // Run native MossFormer2 forward pass.
    if (!mf2_.forward(d_input_, d_source1_, d_source2_, forward_samples)) {
        LOG_ERROR("Separator", "MossFormer2 forward failed (n=%d)", forward_samples);
        return false;
    }

    // Copy outputs D→H.
    out1.resize(forward_samples);
    out2.resize(forward_samples);
    cudaMemcpy(out1.data(), d_source1_, forward_samples * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out2.data(), d_source2_, forward_samples * sizeof(float),
               cudaMemcpyDeviceToHost);
    out1.resize(n_samples);
    out2.resize(n_samples);

    return true;
}

float SpeechSeparator::compute_rms(const std::vector<float>& pcm) {
    if (pcm.empty()) return 0.0f;
    double sum_sq = 0.0;
    for (float v : pcm) sum_sq += (double)v * v;
    return std::sqrt((float)(sum_sq / pcm.size()));
}

} // namespace deusridet
