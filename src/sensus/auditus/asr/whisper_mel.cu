/**
 * @file src/sensus/auditus/asr/whisper_mel.cu
 * @philosophical_role
 *   Whisper-compatible mel — the mel configuration the Qwen3-ASR encoder was trained against (pad-reflect + log10 + Whisper normalise). Distinct from the Auditus mel used by VAD.
 * @serves
 *   Auditus ASR pipeline only (ASR encoder input).
 */
// whisper_mel.cu — GPU Whisper Mel Spectrogram implementation
//
// Pipeline: reflect-pad → frame+window → cuFFT R2C → power spectrum
//         → mel filterbank (SGEMM) → log10 + Whisper normalize
//
// Adapted from qwen35-orin (src/plugins/asr/mel_gpu.cu): GpuWhisperMel class
// and associated CUDA kernels for 128-channel Whisper mel spectrogram.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "whisper_mel.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace deusridet {
namespace asr {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Frame extraction + Hann windowing (1D grid, coalesced)
__global__ void whisper_frame_kernel(const float* __restrict__ pcm,
                                     const float* __restrict__ window,
                                     float* __restrict__ frames,
                                     int num_frames, int n_fft, int hop) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * n_fft;
    if (idx >= total) return;
    int t = idx / n_fft;
    int s = idx % n_fft;
    frames[idx] = pcm[t * hop + s] * window[s];
}

// Complex power spectrum: power[k] = re² + im²
__global__ void power_spectrum_kernel(const cufftComplex* __restrict__ fft_out,
                                      float* __restrict__ power,
                                      int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float re = fft_out[idx].x;
    float im = fft_out[idx].y;
    power[idx] = re * re + im * im;
}

// Transpose [rows, cols] → [cols, rows]
__global__ void transpose_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[idx];
}

// Phase 1: log10 + find max (single block)
__global__ void whisper_log10_kernel(float* __restrict__ mel, int total,
                                     float* __restrict__ d_max) {
    extern __shared__ float smem[];
    float local_max = -1e20f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        float v = log10f(fmaxf(mel[i], 1e-10f));
        mel[i] = v;
        local_max = fmaxf(local_max, v);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) d_max[0] = smem[0];
}

// Phase 2: normalize: clamp to (max - 8), then (v + 4) / 4
__global__ void whisper_normalize_kernel(float* __restrict__ mel, int total,
                                         const float* __restrict__ d_max) {
    float max_val = d_max[0];
    float floor_val = max_val - 8.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        float v = fmaxf(mel[i], floor_val);
        mel[i] = (v + 4.0f) / 4.0f;
    }
}

// ============================================================================
// Slaney Mel Filterbank (CPU, called once at init)
// ============================================================================

static float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

std::vector<float> WhisperMel::build_mel_filterbank() {
    constexpr int n_freqs = N_FREQS;      // 201
    constexpr int n_mels = N_MELS;        // 128
    constexpr int sample_rate = 16000;
    constexpr int n_fft = N_FFT;          // 400

    std::vector<float> fb(n_mels * n_freqs, 0.0f);

    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel((float)sample_rate / 2.0f);

    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (n_mels + 1));

    std::vector<float> fft_bins(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        fft_bins[i] = mel_points[i] * n_fft / sample_rate;

    for (int m = 0; m < n_mels; m++) {
        float left   = fft_bins[m];
        float center = fft_bins[m + 1];
        float right  = fft_bins[m + 2];

        for (int k = 0; k < n_freqs; k++) {
            float fk = (float)k;
            if (fk >= left && fk <= center)
                fb[m * n_freqs + k] = (fk - left) / (center - left);
            else if (fk > center && fk <= right)
                fb[m * n_freqs + k] = (right - fk) / (right - center);
        }

        // Slaney normalization
        float enorm = 2.0f / (mel_points[m + 2] - mel_points[m]);
        for (int k = 0; k < n_freqs; k++)
            fb[m * n_freqs + k] *= enorm;
    }

    return fb;
}

// ============================================================================
// WhisperMel class implementation
// ============================================================================

WhisperMel::WhisperMel() = default;

WhisperMel::~WhisperMel() {
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(d_window_);
    cudaFree(d_mel_fb_);
    cudaFree(d_pcm_);
    cudaFree(d_frames_);
    cudaFree(d_fft_);
    cudaFree(d_power_);
    cudaFree(d_mel_out_);
}

bool WhisperMel::init() {
    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_);
    cublasSetStream(cublas_, stream_);

    // Hann window (periodic: divide by N, not N-1)
    std::vector<float> win(N_FFT);
    for (int i = 0; i < N_FFT; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / N_FFT));
    cudaMalloc(&d_window_, N_FFT * sizeof(float));
    cudaMemcpy(d_window_, win.data(), N_FFT * sizeof(float), cudaMemcpyHostToDevice);

    // Slaney mel filterbank [128, 201]
    auto fb = build_mel_filterbank();
    cudaMalloc(&d_mel_fb_, N_MELS * N_FREQS * sizeof(float));
    cudaMemcpy(d_mel_fb_, fb.data(), N_MELS * N_FREQS * sizeof(float), cudaMemcpyHostToDevice);

    initialized_ = true;
    fprintf(stderr, "[WhisperMel] Initialized: %d mels, FFT %d, hop %d\n", N_MELS, N_FFT, HOP);
    return true;
}

bool WhisperMel::ensure_buffers(int padded_samples, int num_frames) {
    if (padded_samples > buf_max_samples_) {
        cudaFree(d_pcm_);
        buf_max_samples_ = padded_samples + padded_samples / 4;
        cudaMalloc(&d_pcm_, buf_max_samples_ * sizeof(float));
    }
    if (num_frames > buf_max_frames_) {
        cudaFree(d_frames_);
        cudaFree(d_fft_);
        cudaFree(d_power_);
        cudaFree(d_mel_out_);
        buf_max_frames_ = num_frames + num_frames / 4;
        cudaMalloc(&d_frames_, (size_t)buf_max_frames_ * N_FFT * sizeof(float));
        cudaMalloc(&d_fft_, (size_t)buf_max_frames_ * N_FREQS * sizeof(cufftComplex));
        cudaMalloc(&d_power_, (size_t)buf_max_frames_ * N_FREQS * sizeof(float));
        // +1 float for d_max scratch used by log10 kernel
        cudaMalloc(&d_mel_out_, ((size_t)N_MELS * buf_max_frames_ + 1) * sizeof(float));
    }
    if (num_frames != cur_plan_frames_) {
        if (fft_plan_) cufftDestroy(fft_plan_);
        cufftPlan1d(&fft_plan_, N_FFT, CUFFT_R2C, num_frames);
        cufftSetStream(fft_plan_, stream_);
        cur_plan_frames_ = num_frames;
    }
    return true;
}

WhisperMel::Result WhisperMel::compute(const float* pcm, int num_samples) {
    Result result{nullptr, 0};
    if (!initialized_ || num_samples < N_FFT) return result;

    // 1. Reflect-pad: add N_FFT/2 = 200 on each side
    int pad = N_FFT / 2;
    int padded_len = num_samples + 2 * pad;
    std::vector<float> padded(padded_len);
    for (int i = 0; i < padded_len; i++) {
        int src = i - pad;
        if (src < 0) src = -src;
        else if (src >= num_samples) src = 2 * num_samples - src - 2;
        padded[i] = (src >= 0 && src < num_samples) ? pcm[src] : 0.0f;
    }

    // Number of frames (Whisper drops last frame)
    int num_frames = (padded_len - N_FFT) / HOP + 1;
    if (num_frames > 1) num_frames--;
    if (num_frames <= 0) return result;

    ensure_buffers(padded_len, num_frames);

    const int BLOCK = 256;

    // 2. Upload padded PCM
    cudaMemcpyAsync(d_pcm_, padded.data(), padded_len * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    // 3. Frame extraction + Hann windowing
    int total_frame_elems = num_frames * N_FFT;
    int grid = (total_frame_elems + BLOCK - 1) / BLOCK;
    whisper_frame_kernel<<<grid, BLOCK, 0, stream_>>>(
        d_pcm_, d_window_, d_frames_, num_frames, N_FFT, HOP);

    // 4. Batched cuFFT R2C
    cufftExecR2C(fft_plan_, d_frames_, d_fft_);

    // 5. Power spectrum
    int total_freq = num_frames * N_FREQS;
    power_spectrum_kernel<<<(total_freq + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_fft_, d_power_, total_freq);

    // 6. Mel filterbank via cuBLAS SGEMM
    //    mel_fb is [N_MELS, N_FREQS] row-major = [N_FREQS, N_MELS] col-major
    //    power is [T, N_FREQS] row-major = [N_FREQS, T] col-major
    //    Result: C[N_MELS, T] col-major = A^T × B = [T, N_MELS] row-major
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    N_MELS, num_frames, N_FREQS,
                    &alpha,
                    d_mel_fb_, N_FREQS,
                    d_power_, N_FREQS,
                    &beta,
                    d_mel_out_, N_MELS);  // [N_MELS, T] col-major = [T, N_MELS] row-major
    }

    // 7. Transpose [T, N_MELS] row-major → [N_MELS, T] row-major
    //    Use d_power_ as temp (T*201 ≥ 128*T always)
    int total_mel = N_MELS * num_frames;
    transpose_kernel<<<(total_mel + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_mel_out_, d_power_, num_frames, N_MELS);
    cudaMemcpyAsync(d_mel_out_, d_power_, (size_t)total_mel * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // 8. Log10 + Whisper normalization (2-phase)
    float* d_max_scratch = d_mel_out_ + (size_t)N_MELS * buf_max_frames_;
    whisper_log10_kernel<<<1, 1024, 1024 * sizeof(float), stream_>>>(
        d_mel_out_, total_mel, d_max_scratch);
    whisper_normalize_kernel<<<(total_mel + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_mel_out_, total_mel, d_max_scratch);

    result.d_mel = d_mel_out_;
    result.num_frames = num_frames;
    return result;
}

} // namespace asr
} // namespace deusridet
