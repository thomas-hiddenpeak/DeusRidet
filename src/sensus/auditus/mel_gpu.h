// mel_gpu.h — GPU Mel-spectrogram for real-time audio (Qwen3-ASR compatible).
//
// Parameters (matching Qwen3-ASR / Whisper):
//   sample_rate = 16000
//   n_fft       = 400  (25ms window)
//   hop_length  = 160  (10ms hop)
//   n_mels      = 128
//   window      = Hann
//
// Streaming design: push PCM samples, extract Mel frames as they become
// available. Internal GPU buffers hold the Hann window, Mel filterbank,
// and a rolling PCM buffer for overlap.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace deusridet {

struct MelConfig {
    int sample_rate = 16000;
    int n_fft       = 400;
    int hop_length  = 160;
    int n_mels      = 128;
    float ref_level = 1.0f;    // reference for log scaling
    float min_level = 1e-10f;  // floor for log
};

class MelSpectrogram {
public:
    MelSpectrogram();
    ~MelSpectrogram();

    MelSpectrogram(const MelSpectrogram&) = delete;
    MelSpectrogram& operator=(const MelSpectrogram&) = delete;

    // Initialize: allocate GPU buffers, precompute window + filterbank.
    bool init(const MelConfig& cfg);

    // Push int16 PCM samples (host pointer). Copies to GPU rolling buffer.
    // Returns number of new Mel frames produced.
    int push_pcm(const int16_t* pcm_host, int n_samples);

    // Get pointer to the Mel output buffer on GPU (float, [max_frames x n_mels]).
    // Valid frames are [0, frames_ready()).
    const float* mel_buffer() const { return d_mel_out_; }

    // Number of Mel frames ready since last reset.
    int frames_ready() const { return frames_produced_; }

    // Reset internal state (new utterance).
    void reset();

    // Get config.
    const MelConfig& config() const { return cfg_; }

private:
    MelConfig cfg_;
    bool initialized_ = false;

    // GPU buffers.
    float* d_hann_window_  = nullptr;  // [n_fft]
    float* d_mel_filters_  = nullptr;  // [n_mels x (n_fft/2+1)]
    float* d_pcm_rolling_  = nullptr;  // [max_pcm_samples] float PCM on GPU
    float* d_fft_scratch_  = nullptr;  // [n_fft] per-frame scratch
    float* d_power_spec_   = nullptr;  // [n_fft/2+1] power spectrum
    float* d_mel_out_      = nullptr;  // [max_frames x n_mels] output

    int pcm_buf_capacity_  = 0;   // max samples in rolling buffer
    int pcm_buf_len_       = 0;   // current samples in rolling buffer
    int frames_produced_   = 0;
    int max_frames_        = 0;   // max output frames allocated

    // Precomputation on host, uploaded to GPU.
    void precompute_hann_window();
    void precompute_mel_filterbank();
};

// Launch the Mel kernel for a batch of frames.
// d_pcm: float PCM on device, contiguous.
// d_window: Hann window [n_fft].
// d_mel_fb: Mel filterbank [n_mels x (n_fft/2+1)].
// d_out: output [n_frames x n_mels].
// n_frames: how many frames to process.
// pcm_offset: starting sample index for first frame.
void launch_mel_spectrogram(
    const float* d_pcm,
    const float* d_window,
    const float* d_mel_fb,
    float* d_out,
    int n_frames,
    int pcm_offset,
    int n_fft,
    int hop_length,
    int n_mels,
    float min_level,
    cudaStream_t stream = nullptr);

} // namespace deusridet
