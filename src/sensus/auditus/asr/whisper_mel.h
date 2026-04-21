/**
 * @file whisper_mel.h
 * @philosophical_role Declaration of the Whisper-family mel spectrogram GPU kernel (reflect-pad, framing, cuFFT, mel, log10 + Whisper normalise).
 * @serves ASR engine preprocessing.
 */
// whisper_mel.h — GPU Whisper Mel Spectrogram (128-channel, cuFFT)
//
// Computes STFT → power spectrum → mel filterbank → log10 + Whisper normalization.
// All on GPU using cuFFT batched R2C + cuBLAS SGEMM.
//
// Output layout: [N_MELS=128, T] row-major FP32 on GPU.
// The caller converts to BF16 and feeds to ASR encoder conv frontend.
//
// Adapted from qwen35-orin (src/plugins/asr/mel_gpu.h): GPU Whisper mel
// spectrogram with cuFFT + cuBLAS pipeline.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include <vector>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace deusridet {
namespace asr {

class WhisperMel {
public:
    WhisperMel();
    ~WhisperMel();

    // Initialize: builds Hann window + Slaney mel filterbank on GPU
    bool init();
    bool is_initialized() const { return initialized_; }

    // CPU PCM (float, 16kHz) → GPU mel [N_MELS, T] FP32
    // Internal buffer — overwritten on next compute() call
    struct Result { float* d_mel; int num_frames; };
    Result compute(const float* pcm, int num_samples);

    void sync() { if (stream_) cudaStreamSynchronize(stream_); }
    cudaStream_t cuda_stream() const { return stream_; }

    static constexpr int N_FFT   = 400;
    static constexpr int HOP     = 160;
    static constexpr int N_MELS  = 128;
    static constexpr int N_FREQS = N_FFT / 2 + 1;  // 201

private:
    bool initialized_ = false;
    cufftHandle fft_plan_ = 0;
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Constants (allocated once)
    float* d_window_  = nullptr;   // [N_FFT] Hann window (periodic)
    float* d_mel_fb_  = nullptr;   // [N_MELS, N_FREQS] Slaney mel filterbank

    // Work buffers (auto-grow)
    float* d_pcm_     = nullptr;   // padded PCM
    float* d_frames_  = nullptr;   // [T, N_FFT] windowed frames
    cufftComplex* d_fft_ = nullptr; // [T, N_FREQS]
    float* d_power_   = nullptr;   // [T, N_FREQS]
    float* d_mel_out_ = nullptr;   // [N_MELS, T] output (+1 float for max scratch)
    int buf_max_samples_ = 0;
    int buf_max_frames_  = 0;
    int cur_plan_frames_ = 0;

    bool ensure_buffers(int padded_samples, int num_frames);

    // Build Slaney mel filterbank [N_MELS, N_FREQS] on CPU → upload to GPU
    static std::vector<float> build_mel_filterbank();
};

} // namespace asr
} // namespace deusridet
