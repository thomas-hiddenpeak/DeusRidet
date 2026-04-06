// fsmn_fbank_gpu.cu — GPU Fbank kernel for FSMN VAD.
//
// Fused kernel: Hamming window → DFT → power spectrum → Mel filterbank → log
// Parameters: n_fft=512, n_mels=80, frame_len=400 samples, hop=160
//
// Same architecture as mel_gpu.cu but with FSMN-specific parameters.
// One block per frame, threads split across frequency bins.

#include "fsmn_fbank_gpu.h"
#include "../../communis/log.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace deusridet {

// ============================================================================
// Fused Fbank kernel: one block per frame.
// threadIdx.x iterates over frequency bins (n_fft/2+1 = 257).
// ============================================================================

__global__ void fsmn_fbank_kernel(
    const float* __restrict__ pcm,        // float PCM on device
    const float* __restrict__ window,     // Hamming [frame_len]
    const float* __restrict__ mel_fb,     // [n_mels * freq_bins]
    float*       __restrict__ out,        // [n_frames * n_mels]
    int pcm_offset,                       // starting sample for first frame
    int frame_len,                        // 400 (actual window length)
    int n_fft,                            // 512 (zero-padded FFT size)
    int hop_length,                       // 160
    int n_mels,                           // 80
    int freq_bins,                        // n_fft/2 + 1 = 257
    float min_level)
{
    extern __shared__ float smem[];
    float* s_windowed = smem;             // [n_fft]
    float* s_power    = smem + n_fft;     // [freq_bins]

    int frame = blockIdx.x;
    int tid   = threadIdx.x;
    int bsz   = blockDim.x;

    // Step 1a: Load raw PCM into shared memory.
    int frame_start = pcm_offset + frame * hop_length;
    for (int i = tid; i < frame_len; i += bsz) {
        s_windowed[i] = pcm[frame_start + i];
    }
    // Zero-pad beyond frame_len.
    for (int i = frame_len + tid; i < n_fft; i += bsz) {
        s_windowed[i] = 0.0f;
    }
    __syncthreads();

    // Step 1b: Preemphasis (kaldi convention: coeff=0.97).
    // y[i] = x[i] - 0.97*x[i-1], y[0] = x[0] - 0.97*x[0].
    // Applied in reverse to avoid race (each thread reads neighbor below).
    for (int i = frame_len - 1 - tid; i >= 0; i -= bsz) {
        float prev = (i > 0) ? s_windowed[i - 1] : s_windowed[0];
        s_windowed[i] = s_windowed[i] - 0.97f * prev;
    }
    __syncthreads();

    // Step 1c: Apply Hamming window.
    for (int i = tid; i < frame_len; i += bsz) {
        s_windowed[i] *= window[i];
    }
    __syncthreads();

    // Step 2: DFT → power spectrum.
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

    // Step 3: Mel filterbank + log.
    float* frame_out = out + frame * n_mels;
    for (int m = tid; m < n_mels; m += bsz) {
        float sum = 0.0f;
        const float* fb_row = mel_fb + m * freq_bins;
        for (int k = 0; k < freq_bins; k++) {
            sum += fb_row[k] * s_power[k];
        }
        frame_out[m] = __logf(fmaxf(sum, min_level));
    }
}

// ============================================================================
// Kernel launch wrapper
// ============================================================================

void launch_fsmn_fbank(
    const float* d_pcm,
    const float* d_window,
    const float* d_mel_fb,
    float* d_out,
    int n_frames,
    int pcm_offset,
    int frame_len,
    int n_fft,
    int hop_length,
    int n_mels,
    float min_level,
    cudaStream_t stream)
{
    if (n_frames <= 0) return;

    int freq_bins = n_fft / 2 + 1;
    int block_size = 256;
    int smem_size = (n_fft + freq_bins) * sizeof(float);

    fsmn_fbank_kernel<<<n_frames, block_size, smem_size, stream>>>(
        d_pcm, d_window, d_mel_fb, d_out,
        pcm_offset, frame_len, n_fft, hop_length, n_mels, freq_bins, min_level);
}

// ============================================================================
// FsmnFbankGpu class implementation
// ============================================================================

FsmnFbankGpu::FsmnFbankGpu() = default;

FsmnFbankGpu::~FsmnFbankGpu() {
    if (d_hamming_)    cudaFree(d_hamming_);
    if (d_mel_fb_)     cudaFree(d_mel_fb_);
    if (d_pcm_)        cudaFree(d_pcm_);
    if (d_fbank_out_)  cudaFree(d_fbank_out_);
}

bool FsmnFbankGpu::init(int n_mels, int frame_len, int hop, int n_fft,
                         int sample_rate) {
    n_mels_ = n_mels;
    frame_len_ = frame_len;
    hop_ = hop;
    n_fft_ = n_fft;
    freq_bins_ = n_fft / 2 + 1;

    // Allocate Hamming window on GPU.
    cudaMalloc(&d_hamming_, frame_len_ * sizeof(float));

    // Allocate Mel filterbank on GPU.
    cudaMalloc(&d_mel_fb_, n_mels_ * freq_bins_ * sizeof(float));

    // Rolling PCM buffer: hold up to 10 seconds (plenty for VAD chunks).
    pcm_capacity_ = sample_rate * 10;
    cudaMalloc(&d_pcm_, pcm_capacity_ * sizeof(float));
    pcm_len_ = 0;

    // Output buffer: max frames for the PCM capacity.
    max_frames_ = (pcm_capacity_ - n_fft_) / hop_ + 1;
    cudaMalloc(&d_fbank_out_, max_frames_ * n_mels_ * sizeof(float));
    frames_produced_ = 0;

    // Precompute and upload Hamming window.
    {
        std::vector<float> win(frame_len_);
        for (int i = 0; i < frame_len_; i++) {
            win[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i /
                                            (frame_len_ - 1));
        }
        cudaMemcpy(d_hamming_, win.data(), frame_len_ * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    // Precompute and upload Mel filterbank (80 bins, triangular, HTK mel).
    {
        auto hz_to_mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) { return 700.0f * (expf(mel / 1127.0f) - 1.0f); };

        float mel_low  = hz_to_mel(0.0f);
        float mel_high = hz_to_mel((float)sample_rate / 2.0f);

        std::vector<float> mel_pts(n_mels_ + 2);
        for (int i = 0; i < n_mels_ + 2; i++)
            mel_pts[i] = mel_low + (mel_high - mel_low) * i / (n_mels_ + 1);

        std::vector<int> bin_pts(n_mels_ + 2);
        for (int i = 0; i < n_mels_ + 2; i++)
            bin_pts[i] = (int)floorf((n_fft_ + 1) * mel_to_hz(mel_pts[i]) /
                                      sample_rate);

        std::vector<float> fb(n_mels_ * freq_bins_, 0.0f);
        for (int m = 0; m < n_mels_; m++) {
            int f_left   = bin_pts[m];
            int f_center = bin_pts[m + 1];
            int f_right  = bin_pts[m + 2];
            for (int k = f_left; k <= f_center && k < freq_bins_; k++) {
                if (f_center > f_left)
                    fb[m * freq_bins_ + k] = (float)(k - f_left) /
                                              (f_center - f_left);
            }
            for (int k = f_center; k <= f_right && k < freq_bins_; k++) {
                if (f_right > f_center)
                    fb[m * freq_bins_ + k] = (float)(f_right - k) /
                                              (f_right - f_center);
            }
        }
        cudaMemcpy(d_mel_fb_, fb.data(),
                   n_mels_ * freq_bins_ * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    initialized_ = true;
    LOG_INFO("FsmnFbankGPU", "Initialized: frame=%d n_fft=%d hop=%d mels=%d",
             frame_len_, n_fft_, hop_, n_mels_);
    return true;
}

int FsmnFbankGpu::push_pcm(const int16_t* pcm_host, int n_samples) {
    if (!initialized_ || n_samples <= 0) return 0;

    if (pcm_len_ + n_samples > pcm_capacity_) {
        reset();
    }

    // Convert int16 → float on host and upload.
    std::vector<float> pcm_f(n_samples);
    for (int i = 0; i < n_samples; i++) {
        pcm_f[i] = pcm_host[i] / 32768.0f;
    }
    cudaMemcpy(d_pcm_ + pcm_len_, pcm_f.data(),
               n_samples * sizeof(float), cudaMemcpyHostToDevice);
    pcm_len_ += n_samples;

    // How many new frames?
    // Need frame_len_ samples for the first frame (zero-padded to n_fft_),
    // and n_fft_ worth for DFT. But the actual PCM needed per frame is
    // frame_start + frame_len_ (we only window frame_len_ samples).
    int total_possible = 0;
    if (pcm_len_ >= frame_len_) {
        total_possible = (pcm_len_ - frame_len_) / hop_ + 1;
    }
    int new_frames = total_possible - frames_produced_;
    if (new_frames <= 0) return 0;
    if (frames_produced_ + new_frames > max_frames_) {
        new_frames = max_frames_ - frames_produced_;
        if (new_frames <= 0) return 0;
    }

    int pcm_offset = frames_produced_ * hop_;
    launch_fsmn_fbank(
        d_pcm_, d_hamming_, d_mel_fb_,
        d_fbank_out_ + frames_produced_ * n_mels_,
        new_frames, pcm_offset,
        frame_len_, n_fft_, hop_, n_mels_, 1e-10f);

    frames_produced_ += new_frames;
    return new_frames;
}

int FsmnFbankGpu::read_fbank(float* host_out, int max_frames) {
    int avail = frames_produced_ - frames_read_;
    if (avail <= 0) return 0;
    int to_read = (avail < max_frames) ? avail : max_frames;
    cudaMemcpy(host_out,
               d_fbank_out_ + frames_read_ * n_mels_,
               to_read * n_mels_ * sizeof(float),
               cudaMemcpyDeviceToHost);
    frames_read_ += to_read;
    return to_read;
}

void FsmnFbankGpu::reset() {
    pcm_len_ = 0;
    frames_produced_ = 0;
    frames_read_ = 0;
}

} // namespace deusridet
