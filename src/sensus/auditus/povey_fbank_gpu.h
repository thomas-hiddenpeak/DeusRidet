/**
 * @file povey_fbank_gpu.h
 * @philosophical_role Declaration of the fused GPU Fbank kernel (Povey/Hamming window) used by the speaker embedding frontend.
 * @serves Auditus speaker path (CAM++), FRCRN enhancer frontend.
 */
// povey_fbank_gpu.h — GPU Fbank feature extractor.
//
// Computes: window → DFT → power spectrum → Mel(80) → log on GPU.
// Supports Hamming window and Povey window (CAM++ speaker frontend).
// Parameters: n_fft=512, n_mels=80, frame_len=400 samples, hop=160.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace deusridet {

enum class FbankWindowType {
    HAMMING,  // 0.54 - 0.46*cos(2π*n/(N-1))
    POVEY,    // (0.5 - 0.5*cos(2π*n/(N-1)))^0.85 — Kaldi default, used by CAM++
};

// Launch the fused Fbank kernel.
void launch_povey_fbank(
    const float* d_pcm,
    const float* d_window,   // window [frame_len]
    const float* d_mel_fb,   // [n_mels * (n_fft/2+1)]
    float* d_out,            // [n_frames * n_mels]
    int n_frames,
    int pcm_offset,
    int frame_len,
    int n_fft,
    int hop_length,
    int n_mels,
    float min_level,
    cudaStream_t stream = nullptr);

// Streaming GPU Fbank extractor.
// Manages GPU buffers, accepts int16 PCM, produces fbank frames on host.
// Window type and PCM normalization are configurable:
//   FSMN VAD: Hamming window, int16 scale (normalize_pcm=false)
//   CAM++ speaker: Povey window, [-1,1] scale (normalize_pcm=true)
class PoveyFbankGpu {
public:
    PoveyFbankGpu();
    ~PoveyFbankGpu();

    PoveyFbankGpu(const PoveyFbankGpu&) = delete;
    PoveyFbankGpu& operator=(const PoveyFbankGpu&) = delete;

    bool init(int n_mels = 80, int frame_len = 400, int hop = 160,
              int n_fft = 512, int sample_rate = 16000,
              FbankWindowType window_type = FbankWindowType::HAMMING,
              bool normalize_pcm = false);

    // Push int16 PCM. Returns number of new fbank frames available.
    int push_pcm(const int16_t* pcm_host, int n_samples);

    // Read new fbank frames to host buffer. Returns frames read.
    // host_out must have space for at least max_frames * n_mels floats.
    int read_fbank(float* host_out, int max_frames);

    int frames_ready() const { return frames_produced_ - frames_read_; }
    int n_mels() const { return n_mels_; }
    bool initialized() const { return initialized_; }

    void reset();

private:
    bool initialized_ = false;
    int n_mels_     = 80;
    int frame_len_  = 400;
    int hop_        = 160;
    int n_fft_      = 512;
    int freq_bins_  = 257;  // n_fft/2 + 1

    bool normalize_pcm_ = false;
    FbankWindowType window_type_ = FbankWindowType::HAMMING;

    // GPU buffers.
    float* d_window_    = nullptr;  // [frame_len]
    float* d_mel_fb_    = nullptr;  // [n_mels * freq_bins]
    float* d_pcm_       = nullptr;  // rolling PCM
    float* d_fbank_out_ = nullptr;  // [max_frames * n_mels]

    int pcm_capacity_ = 0;
    int pcm_len_      = 0;
    int max_frames_   = 0;
    int frames_produced_ = 0;
    int frames_read_     = 0;
};

} // namespace deusridet
