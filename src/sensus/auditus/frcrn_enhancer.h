// frcrn_enhancer.h — FRCRN speech enhancement (denoising) via custom CUDA.
//
// Model: FRCRN_SE_16K (~55 MB safetensors, DCCRN dual-UNet architecture)
// Input:  raw PCM float32 [1, T], 16kHz mono
// Output: enhanced PCM float32 [1, T], same length
//
// Uses cuFFT (STFT/iSTFT), cuDNN (Conv2d/ConvTranspose2d), cuBLAS (FSMN linear),
// and custom CUDA kernels (BN, activations, FSMN conv, SE, complex ops).
//
// Adapted from ModelScope iic/speech_frcrn_ans_cirm_16k (Apache-2.0).
// Architecture: ConvSTFT → dual-UNet (complex mask via cIRM + tanh) → ConviSTFT
//
// For streaming use, this class accumulates PCM and processes in chunks
// with overlap-add stitching. Default chunk: 16000 samples (1s).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deusridet {

class FrcrnGpu;  // forward declaration

struct FrcrnConfig {
    std::string weights_dir;             // path to safetensors weights directory
    int sample_rate       = 16000;
    int chunk_samples     = 16000;       // 1s processing chunk
    int hop_samples       = 320;         // STFT hop — input length multiple
    bool enabled          = true;
};

class FrcrnEnhancer {
public:
    FrcrnEnhancer();
    ~FrcrnEnhancer();

    FrcrnEnhancer(const FrcrnEnhancer&) = delete;
    FrcrnEnhancer& operator=(const FrcrnEnhancer&) = delete;

    bool init(const FrcrnConfig& cfg);

    // Enhance a buffer of PCM. Input/output are float32 [-1, 1].
    // Returns enhanced PCM of same length as input.
    // Not thread-safe — call from one thread only.
    std::vector<float> enhance(const float* pcm, int n_samples);

    // Streaming interface: push int16 PCM chunks (from pipeline).
    // Internally accumulates to chunk_samples, processes, and queues output.
    // Returns enhanced int16 PCM from the output queue, up to n_samples.
    // May return empty vector (when still accumulating input).
    // The output is delayed by ~chunk_samples compared to input.
    // When flush=true, processes remaining buffer with zero-padding.
    //
    // Alternative: enhance_chunk() processes a single small chunk directly
    // with no accumulation — lower quality but zero latency.
    void push_pcm(const int16_t* pcm, int n_samples);

    // Pull enhanced PCM from the output queue.
    // Returns up to max_samples samples. May return fewer (or empty).
    int pull_pcm(int16_t* out, int max_samples);

    // How many enhanced samples are ready to pull.
    int available() const { return (int)output_buf_.size(); }

    // Direct per-chunk enhancement: no accumulation, process immediately.
    // Lower quality for very small chunks but zero latency.
    // In-place: overwrites pcm with enhanced audio.
    void enhance_inplace(int16_t* pcm, int n_samples);

    // Reset internal accumulation buffer.
    void reset();

    bool initialized() const { return initialized_; }
    float last_latency_ms() const { return last_latency_ms_; }

private:
    FrcrnConfig cfg_;
    bool initialized_ = false;
    float last_latency_ms_ = 0.0f;

    std::unique_ptr<FrcrnGpu> gpu_;

    // Streaming accumulation buffer (input) and output queue.
    std::vector<float> accum_buf_;
    std::vector<int16_t> output_buf_;
};

} // namespace deusridet
