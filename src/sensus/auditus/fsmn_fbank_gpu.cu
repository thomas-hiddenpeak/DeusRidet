// fsmn_fbank_gpu.cu — GPU Fbank kernel.
//
// Fused kernel: window → DFT → power spectrum → Mel filterbank → log
// Parameters: n_fft=512, n_mels=80, frame_len=400 samples, hop=160
//
// Supports two window types:
//   HAMMING — for FSMN VAD (WavFrontend convention)
//   POVEY   — for CAM++ speaker encoder (Kaldi default, hann^0.85)
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
    // Layout: [raw PCM: frame_len] [windowed: n_fft] [power: freq_bins]
    float* s_raw      = smem;                          // [frame_len]
    float* s_windowed = smem + frame_len;              // [n_fft]
    float* s_power    = smem + frame_len + n_fft;      // [freq_bins]

    int frame = blockIdx.x;
    int tid   = threadIdx.x;
    int bsz   = blockDim.x;

    // Step 1a: Load raw PCM into shared memory.
    int frame_start = pcm_offset + frame * hop_length;
    for (int i = tid; i < frame_len; i += bsz) {
        s_raw[i] = pcm[frame_start + i];
    }
    __syncthreads();

    // Step 1b: Compute DC offset (mean of frame) via parallel reduction.
    // Kaldi default: remove_dc_offset=true.
    float local_sum = 0.0f;
    for (int i = tid; i < frame_len; i += bsz) {
        local_sum += s_raw[i];
    }
    // Use s_power as scratch for reduction (bsz <= freq_bins = 257).
    s_power[tid] = local_sum;
    __syncthreads();
    for (int stride = bsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_power[tid] += s_power[tid + stride];
        __syncthreads();
    }
    float dc_offset = s_power[0] / (float)frame_len;
    __syncthreads();

    // Step 1c: DC removal + preemphasis + windowing.
    // Read from s_raw (unmodified), write to s_windowed.
    // Preemphasis (Kaldi convention: coeff=0.97): y[i] = x[i] - 0.97*x[i-1].
    for (int i = tid; i < frame_len; i += bsz) {
        float x_cur  = s_raw[i] - dc_offset;
        float x_prev = (i > 0) ? (s_raw[i - 1] - dc_offset) : x_cur;
        s_windowed[i] = (x_cur - 0.97f * x_prev) * window[i];
    }
    // Zero-pad beyond frame_len.
    for (int i = frame_len + tid; i < n_fft; i += bsz) {
        s_windowed[i] = 0.0f;
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
    // Shared memory: [raw PCM: frame_len] + [windowed: n_fft] + [power: freq_bins]
    int smem_size = (frame_len + n_fft + freq_bins) * sizeof(float);

    fsmn_fbank_kernel<<<n_frames, block_size, smem_size, stream>>>(
        d_pcm, d_window, d_mel_fb, d_out,
        pcm_offset, frame_len, n_fft, hop_length, n_mels, freq_bins, min_level);
}

// ============================================================================
// FsmnFbankGpu class implementation
// ============================================================================

FsmnFbankGpu::FsmnFbankGpu() = default;

FsmnFbankGpu::~FsmnFbankGpu() {
    if (d_window_)     cudaFree(d_window_);
    if (d_mel_fb_)     cudaFree(d_mel_fb_);
    if (d_pcm_)        cudaFree(d_pcm_);
    if (d_fbank_out_)  cudaFree(d_fbank_out_);
}

bool FsmnFbankGpu::init(int n_mels, int frame_len, int hop, int n_fft,
                         int sample_rate, FbankWindowType window_type,
                         bool normalize_pcm) {
    n_mels_ = n_mels;
    frame_len_ = frame_len;
    hop_ = hop;
    n_fft_ = n_fft;
    freq_bins_ = n_fft / 2 + 1;
    window_type_ = window_type;
    normalize_pcm_ = normalize_pcm;

    // Allocate window on GPU.
    cudaMalloc(&d_window_, frame_len_ * sizeof(float));

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

    // Precompute and upload window function.
    {
        std::vector<float> win(frame_len_);
        for (int i = 0; i < frame_len_; i++) {
            float hann = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i /
                                              (frame_len_ - 1));
            if (window_type_ == FbankWindowType::POVEY) {
                // Povey window: hann^0.85 (Kaldi default)
                win[i] = powf(hann, 0.85f);
            } else {
                // Hamming window: 0.54 - 0.46*cos(...)
                win[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i /
                                                (frame_len_ - 1));
            }
        }
        cudaMemcpy(d_window_, win.data(), frame_len_ * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    // Precompute and upload Mel filterbank (80 bins, triangular, HTK mel).
    // Kaldi-style: triangular filters with interpolation in MEL domain.
    // CAM++ (FunASR) was trained with torchaudio.compliance.kaldi.fbank(low_freq=20).
    // Adapted from torchaudio.compliance.kaldi.get_mel_banks
    {
        auto hz_to_mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };

        float mel_low  = hz_to_mel(20.0f);   // Kaldi default low_freq=20 Hz
        float mel_high = hz_to_mel((float)sample_rate / 2.0f);
        float mel_delta = (mel_high - mel_low) / (n_mels_ + 1);
        float fft_bin_width = (float)sample_rate / n_fft_;

        std::vector<float> fb(n_mels_ * freq_bins_, 0.0f);
        for (int m = 0; m < n_mels_; m++) {
            float left_mel   = mel_low + m * mel_delta;
            float center_mel = mel_low + (m + 1) * mel_delta;
            float right_mel  = mel_low + (m + 2) * mel_delta;
            for (int k = 0; k < freq_bins_; k++) {
                float mel_k = hz_to_mel(k * fft_bin_width);
                float up   = (center_mel > left_mel)
                           ? (mel_k - left_mel) / (center_mel - left_mel) : 0.0f;
                float down = (right_mel > center_mel)
                           ? (right_mel - mel_k) / (right_mel - center_mel) : 0.0f;
                float w = up < down ? up : down;
                fb[m * freq_bins_ + k] = w > 0.0f ? w : 0.0f;
            }
        }
        cudaMemcpy(d_mel_fb_, fb.data(),
                   n_mels_ * freq_bins_ * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    initialized_ = true;
    const char* win_name = (window_type_ == FbankWindowType::POVEY) ? "Povey" : "Hamming";
    LOG_INFO("FsmnFbankGPU", "Initialized: frame=%d n_fft=%d hop=%d mels=%d window=%s normalize=%d",
             frame_len_, n_fft_, hop_, n_mels_, win_name, (int)normalize_pcm_);
    return true;
}

int FsmnFbankGpu::push_pcm(const int16_t* pcm_host, int n_samples) {
    if (!initialized_ || n_samples <= 0) return 0;

    if (pcm_len_ + n_samples > pcm_capacity_) {
        reset();
    }

    // Convert int16 → float on host and upload.
    // normalize_pcm_=true: divide by 32768 for [-1,1] range (CAM++ speaker)
    // normalize_pcm_=false: keep int16 scale (FSMN VAD with CMVN from am.mvn)
    std::vector<float> pcm_f(n_samples);
    if (normalize_pcm_) {
        for (int i = 0; i < n_samples; i++) {
            pcm_f[i] = pcm_host[i] / 32768.0f;
        }
    } else {
        for (int i = 0; i < n_samples; i++) {
            pcm_f[i] = (float)pcm_host[i];
        }
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
        d_pcm_, d_window_, d_mel_fb_,
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
