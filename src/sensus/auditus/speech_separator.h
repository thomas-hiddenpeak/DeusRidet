// speech_separator.h — MossFormer2 Speech Separation via native CUDA.
//
// Model: MossFormer2_SS_16K from ClearerVoice-Studio (Apache-2.0)
// Architecture: Conv1d encoder → 24× MossFormer2 blocks → ConvTranspose1d decoder
// Input:  (1, variable_length) — mixed PCM float32, 16kHz mono
// Output: (1, same_length) × 2 — two separated speaker streams
// Weight size: ~213 MB safetensors (55.7M params, FP32)
//
// Supports lazy loading: model loaded on first overlap detection to save memory.
// For audio > 2s, internally segments with overlap-add stitching.
//
// Runs natively on GPU via cuBLAS + custom CUDA kernels (zero TRT dependency).

#pragma once

#include "mossformer2.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {

struct SpeechSeparatorConfig {
    std::string model_path;          // path to mossformer2_ss_16k.safetensors
    int sample_rate     = 16000;
    int max_chunk       = 32000;     // 2s native processing chunk
    int overlap_samples = 3200;      // 200ms overlap for stitching
    bool lazy_load      = true;      // don't load until first overlap
};

struct SeparationResult {
    std::vector<float> source1;     // separated speaker 1 PCM
    std::vector<float> source2;     // separated speaker 2 PCM
    float energy1;                  // RMS energy of source 1
    float energy2;                  // RMS energy of source 2
    bool valid;                     // separation succeeded
};

class SpeechSeparator {
public:
    SpeechSeparator();
    ~SpeechSeparator();

    SpeechSeparator(const SpeechSeparator&) = delete;
    SpeechSeparator& operator=(const SpeechSeparator&) = delete;

    // init() only stores config. If lazy_load=false, also loads the model.
    bool init(const SpeechSeparatorConfig& cfg);

    // Separate a mixed audio chunk into 2 speaker streams.
    // Input PCM should be float32 [-1, 1], 16kHz mono.
    // For audio > max_chunk, internally segments and stitches.
    SeparationResult separate(const float* pcm, int n_samples);

    bool initialized() const { return initialized_; }
    bool loaded() const { return loaded_; }

    // Lazy loading: call when first overlap detected.
    bool ensure_loaded();
    // Manual unload to free memory.
    void unload();

private:
    SpeechSeparatorConfig cfg_;
    bool initialized_ = false;
    bool loaded_ = false;

    MossFormer2 mf2_;
    cudaStream_t cuda_stream_ = nullptr;

    // GPU buffers (allocated for max_chunk size).
    float* d_input_   = nullptr;  // (max_chunk)
    float* d_source1_ = nullptr;  // (max_chunk)
    float* d_source2_ = nullptr;  // (max_chunk)

    // Process a single chunk (≤ max_chunk samples).
    bool separate_chunk(const float* pcm, int n_samples,
                        std::vector<float>& out1, std::vector<float>& out2);

    static float compute_rms(const std::vector<float>& pcm);
};

} // namespace deusridet
