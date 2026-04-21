/**
 * @file src/sensus/auditus/mel_gpu.cu
 * @philosophical_role
 *   Mel-spectrogram kernel — the bridge from raw PCM to what an ASR encoder can actually consume. A perceptual primitive shared by every speech model in the project.
 * @serves
 *   Auditus pipeline mel stage feeding VAD and ASR.
 */
// mel_gpu.cu — GPU Mel-spectrogram kernel (Qwen3-ASR / Whisper compatible).
//
// Single fused kernel per frame: window → real-DFT → power spectrum →
// Mel filterbank → log scaling.
//
// n_fft = 400 is small enough to compute the DFT directly (no FFT butterfly
// needed — O(N²) with N=201 frequency bins is only ~80K FMAs per frame,
// well within a single SM's throughput at 10ms frame rate).
// This avoids cuFFT dependency and plan management overhead.
//
// Performance target: process 100 frames/sec (real-time at hop=160, sr=16k)
// with negligible GPU utilization — the bottleneck is audio capture, not compute.

#include "mel_gpu.h"
#include "../../communis/log.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// ============================================================================
// Mel spectrogram kernel: one block per frame, threads split across freq bins.
// ============================================================================

// Each block computes one Mel frame.
// threadIdx.x iterates over frequency bins (n_fft/2+1 = 201).
// Shared memory holds: windowed signal [n_fft] + power spectrum [n_fft/2+1].
__global__ void mel_spectrogram_kernel(
    const float* __restrict__ pcm,
    const float* __restrict__ window,     // [n_fft]
    const float* __restrict__ mel_fb,     // [n_mels * freq_bins]
    float*       __restrict__ out,        // [n_frames * n_mels]
    int pcm_offset,
    int n_fft,
    int hop_length,
    int n_mels,
    int freq_bins,                        // n_fft/2 + 1
    float min_level)
{
    extern __shared__ float smem[];
    float* s_windowed = smem;                    // [n_fft]
    float* s_power    = smem + n_fft;            // [freq_bins]

    int frame = blockIdx.x;
    int tid   = threadIdx.x;
    int bsz   = blockDim.x;

    // Step 1: Apply Hann window to PCM frame → shared memory.
    int frame_start = pcm_offset + frame * hop_length;
    for (int i = tid; i < n_fft; i += bsz) {
        s_windowed[i] = pcm[frame_start + i] * window[i];
    }
    __syncthreads();

    // Step 2: DFT → power spectrum.
    // Compute |X[k]|² for each frequency bin k.
    // X[k] = Σ_{n=0}^{N-1} x[n] * e^{-j 2π k n / N}
    //       = Σ x[n] cos(2πkn/N) - j Σ x[n] sin(2πkn/N)
    for (int k = tid; k < freq_bins; k += bsz) {
        float re = 0.0f, im = 0.0f;
        float phase_step = -2.0f * 3.14159265358979323846f * k / (float)n_fft;
        for (int n = 0; n < n_fft; n++) {
            float phase = phase_step * n;
            float c, s;
            __sincosf(phase, &s, &c);
            re += s_windowed[n] * c;
            im += s_windowed[n] * s;
        }
        s_power[k] = re * re + im * im;
    }
    __syncthreads();

    // Step 3: Mel filterbank + log scaling.
    // Each thread computes one mel bin.
    float* frame_out = out + frame * n_mels;
    for (int m = tid; m < n_mels; m += bsz) {
        float sum = 0.0f;
        const float* fb_row = mel_fb + m * freq_bins;
        for (int k = 0; k < freq_bins; k++) {
            sum += fb_row[k] * s_power[k];
        }
        // Log-mel: log(max(sum, min_level))
        frame_out[m] = __logf(fmaxf(sum, min_level));
    }
}

// ============================================================================
// Kernel launch wrapper
// ============================================================================

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
    cudaStream_t stream)
{
    if (n_frames <= 0) return;

    int freq_bins = n_fft / 2 + 1;
    // Block size: 256 threads handles both freq_bins (201) and n_mels (128).
    int block_size = 256;
    int smem_size = (n_fft + freq_bins) * sizeof(float);

    mel_spectrogram_kernel<<<n_frames, block_size, smem_size, stream>>>(
        d_pcm, d_window, d_mel_fb, d_out,
        pcm_offset, n_fft, hop_length, n_mels, freq_bins, min_level);
}

// ============================================================================
// MelSpectrogram class implementation
// ============================================================================

MelSpectrogram::MelSpectrogram() = default;

MelSpectrogram::~MelSpectrogram() {
    if (d_hann_window_)  cudaFree(d_hann_window_);
    if (d_mel_filters_)  cudaFree(d_mel_filters_);
    if (d_pcm_rolling_)  cudaFree(d_pcm_rolling_);
    if (d_mel_out_)      cudaFree(d_mel_out_);
    // d_fft_scratch_ and d_power_spec_ not separately allocated (in-kernel smem).
}

bool MelSpectrogram::init(const MelConfig& cfg) {
    cfg_ = cfg;

    int freq_bins = cfg_.n_fft / 2 + 1;

    // Allocate Hann window on GPU.
    cudaMalloc(&d_hann_window_, cfg_.n_fft * sizeof(float));

    // Allocate Mel filterbank on GPU.
    cudaMalloc(&d_mel_filters_, cfg_.n_mels * freq_bins * sizeof(float));

    // Rolling PCM buffer: hold up to 60 seconds of audio.
    pcm_buf_capacity_ = cfg_.sample_rate * 60;
    cudaMalloc(&d_pcm_rolling_, pcm_buf_capacity_ * sizeof(float));
    pcm_buf_len_ = 0;

    // Mel output buffer: max frames for 60 seconds.
    max_frames_ = (pcm_buf_capacity_ - cfg_.n_fft) / cfg_.hop_length + 1;
    cudaMalloc(&d_mel_out_, max_frames_ * cfg_.n_mels * sizeof(float));
    frames_produced_ = 0;

    // Precompute and upload constant data.
    precompute_hann_window();
    precompute_mel_filterbank();

    initialized_ = true;
    LOG_INFO("Mel", "Initialized: n_fft=%d hop=%d n_mels=%d (max %d frames)",
             cfg_.n_fft, cfg_.hop_length, cfg_.n_mels, max_frames_);
    return true;
}

void MelSpectrogram::precompute_hann_window() {
    std::vector<float> win(cfg_.n_fft);
    for (int i = 0; i < cfg_.n_fft; i++) {
        win[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / cfg_.n_fft));
    }
    cudaMemcpy(d_hann_window_, win.data(), cfg_.n_fft * sizeof(float),
               cudaMemcpyHostToDevice);
}

void MelSpectrogram::precompute_mel_filterbank() {
    int freq_bins = cfg_.n_fft / 2 + 1;
    int n_mels = cfg_.n_mels;
    float sr = (float)cfg_.sample_rate;

    // Mel scale conversion (HTK formula).
    auto hz_to_mel = [](float hz) -> float {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    };
    auto mel_to_hz = [](float mel) -> float {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    };

    float mel_low  = hz_to_mel(0.0f);
    float mel_high = hz_to_mel(sr / 2.0f);

    // n_mels + 2 equally spaced points in mel space.
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }

    // Convert back to Hz, then to FFT bin indices.
    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = hz_points[i] * cfg_.n_fft / sr;
    }

    // Build triangular filterbank [n_mels x freq_bins].
    std::vector<float> fb(n_mels * freq_bins, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        float left   = bin_points[m];
        float center = bin_points[m + 1];
        float right  = bin_points[m + 2];

        for (int k = 0; k < freq_bins; k++) {
            float fk = (float)k;
            if (fk >= left && fk <= center && center > left) {
                fb[m * freq_bins + k] = (fk - left) / (center - left);
            } else if (fk > center && fk <= right && right > center) {
                fb[m * freq_bins + k] = (right - fk) / (right - center);
            }
        }

        // Slaney normalization: 2 / (right - left).
        float norm = 2.0f / (hz_points[m + 2] - hz_points[m]);
        for (int k = 0; k < freq_bins; k++) {
            fb[m * freq_bins + k] *= norm;
        }
    }

    cudaMemcpy(d_mel_filters_, fb.data(), n_mels * freq_bins * sizeof(float),
               cudaMemcpyHostToDevice);
}

int MelSpectrogram::push_pcm(const int16_t* pcm_host, int n_samples) {
    if (!initialized_ || n_samples <= 0) return 0;

    // Check capacity.
    if (pcm_buf_len_ + n_samples > pcm_buf_capacity_) {
        LOG_WARN("Mel", "PCM buffer overflow, resetting (had %d samples)",
                 pcm_buf_len_);
        reset();
    }

    // Convert int16 → float32 on host and upload.
    // For small chunks (512 samples = 2KB), this is faster than a GPU kernel.
    std::vector<float> pcm_f(n_samples);
    for (int i = 0; i < n_samples; i++) {
        pcm_f[i] = pcm_host[i] / 32768.0f;
    }
    cudaMemcpy(d_pcm_rolling_ + pcm_buf_len_, pcm_f.data(),
               n_samples * sizeof(float), cudaMemcpyHostToDevice);
    pcm_buf_len_ += n_samples;

    // How many new frames can we compute?
    int total_possible = 0;
    if (pcm_buf_len_ >= cfg_.n_fft) {
        total_possible = (pcm_buf_len_ - cfg_.n_fft) / cfg_.hop_length + 1;
    }
    int new_frames = total_possible - frames_produced_;
    if (new_frames <= 0) return 0;

    // Clamp to output buffer capacity.
    if (frames_produced_ + new_frames > max_frames_) {
        new_frames = max_frames_ - frames_produced_;
        if (new_frames <= 0) return 0;
    }

    // Compute Mel for new frames.
    int pcm_offset = frames_produced_ * cfg_.hop_length;
    launch_mel_spectrogram(
        d_pcm_rolling_, d_hann_window_, d_mel_filters_,
        d_mel_out_ + frames_produced_ * cfg_.n_mels,
        new_frames, pcm_offset,
        cfg_.n_fft, cfg_.hop_length, cfg_.n_mels,
        cfg_.min_level);

    frames_produced_ += new_frames;
    return new_frames;
}

void MelSpectrogram::reset() {
    pcm_buf_len_ = 0;
    frames_produced_ = 0;
}

} // namespace deusridet
