// fsmn_fbank_gpu.h — GPU Fbank for FSMN VAD.
//
// Computes: Hamming window → DFT → power spectrum → Mel(80) → log
// on GPU. Same architecture as mel_gpu.h but with FSMN-specific params
// (n_fft=512, n_mels=80, frame_len=400, hop=160, Hamming window).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace deusridet {

// Launch the fused Fbank kernel for FSMN VAD.
void launch_fsmn_fbank(
    const float* d_pcm,
    const float* d_window,   // Hamming [frame_len]
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

// Streaming GPU Fbank extractor for FSMN VAD.
// Manages GPU buffers, accepts int16 PCM, produces fbank frames on host.
class FsmnFbankGpu {
public:
    FsmnFbankGpu();
    ~FsmnFbankGpu();

    FsmnFbankGpu(const FsmnFbankGpu&) = delete;
    FsmnFbankGpu& operator=(const FsmnFbankGpu&) = delete;

    bool init(int n_mels = 80, int frame_len = 400, int hop = 160,
              int n_fft = 512, int sample_rate = 16000);

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

    // GPU buffers.
    float* d_hamming_   = nullptr;  // [frame_len]
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
