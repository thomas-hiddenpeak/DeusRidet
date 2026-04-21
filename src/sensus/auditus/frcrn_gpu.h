/**
 * @file frcrn_gpu.h
 * @philosophical_role Declaration of the FRCRN GPU forward pass (STFT + dual UNet1/UNet2 + iSTFT).
 * @serves frcrn_enhancer, Auditus.
 */
// frcrn_gpu.h — FRCRN speech enhancement: custom CUDA forward pass.
//
// Implements the full FRCRN (Frequency Recurrent CRN) inference pipeline
// on GPU using cuFFT for STFT/iSTFT, cuDNN for Conv2d/ConvTranspose2d,
// cuBLAS for linear layers, and custom CUDA kernels for everything else.
//
// Model: FRCRN_SE_16K (DCCRN architecture, dual UNet with FSMN blocks)
// Input:  float32 PCM [T] on GPU, 16kHz mono
// Output: float32 PCM [T] on GPU, denoised
//
// Architecture:
//   ConvSTFT (cuFFT R2C) → UNet1 (7-layer encoder/decoder) → tanh mask →
//   UNet2 → tanh mask → complex mask apply → ConviSTFT (cuFFT C2R + OLA)
//
// Memory: ~55 MB weights + ~30 MB activation buffers = ~85 MB GPU total.
//
// Adapted from ModelScope iic/speech_frcrn_ans_cirm_16k (Apache-2.0).
// Original: https://github.com/modelscope/modelscope

#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace deusridet {

// ============================================================================
// FRCRN constants (model_depth=14, model_complexity=45, complex=true)
// ============================================================================
namespace frcrn {
    constexpr int kWinLen     = 640;
    constexpr int kHop        = 320;
    constexpr int kFftLen     = 640;
    constexpr int kFreqBins   = kFftLen / 2 + 1;  // 321
    constexpr int kChannels   = 128;               // all encoder/decoder channels
    constexpr int kNumStages  = 7;                 // encoder/decoder depth
    constexpr int kFsmnOrder  = 20;                // FSMN convolution order
    constexpr int kSeReduce   = 8;                 // SE reduction ratio
    constexpr int kSeBotNeck  = kChannels / kSeReduce;  // 16

    // Encoder kernel sizes (kH, kW) for each stage
    constexpr int kEncKernH[kNumStages] = {5, 5, 5, 5, 5, 5, 2};
    constexpr int kEncKernW[kNumStages] = {2, 2, 2, 2, 2, 2, 2};
    // Decoder kernel sizes
    constexpr int kDecKernH[kNumStages] = {2, 5, 5, 5, 6, 5, 5};
    constexpr int kDecKernW[kNumStages] = {2, 2, 2, 2, 2, 2, 2};
    // All strides are (2,1), padding (0,1) — shared by all stages
    constexpr int kStrideH = 2, kStrideW = 1;
    constexpr int kPadH = 0, kPadW = 1;

    // Encoder input channels per stage (only stage 0 differs: 1 → 128)
    constexpr int kEncInCh[kNumStages]  = {1, 128, 128, 128, 128, 128, 128};
    constexpr int kEncOutCh[kNumStages] = {128, 128, 128, 128, 128, 128, 128};
    // Decoder input/output channels (dec0: 128→128, rest: 256→128, last: 256→1)
    // Note: input is doubled for skip connections (except dec0 which takes bottleneck)
    constexpr int kDecInCh[kNumStages]  = {128, 256, 256, 256, 256, 256, 256};
    constexpr int kDecOutCh[kNumStages] = {128, 128, 128, 128, 128, 128, 1};
}

// ============================================================================
// GPU weight reference (points into GPU memory)
// ============================================================================
struct GpuTensor {
    float* ptr = nullptr;
    int    numel = 0;
};

// ============================================================================
// FrcrnGpu — Main GPU inference class
// ============================================================================
class FrcrnGpu {
public:
    FrcrnGpu();
    ~FrcrnGpu();

    FrcrnGpu(const FrcrnGpu&) = delete;
    FrcrnGpu& operator=(const FrcrnGpu&) = delete;

    // Initialize: load weights from safetensors directory, allocate GPU buffers.
    // weights_dir should contain model.safetensors.
    // max_samples: maximum input length in samples (determines buffer sizes).
    bool init(const std::string& weights_dir, int max_samples = 48000,
              cudaStream_t stream = nullptr);

    // Enhance audio. PCM in/out are GPU-resident float32 [-1, 1].
    // n_samples must be <= max_samples and a multiple of kHop (320).
    // Returns actual output length (may differ slightly due to STFT framing).
    int enhance(const float* d_pcm_in, float* d_pcm_out, int n_samples);

    // Enhance audio from host memory. Handles H2D/D2H transfers internally.
    int enhance_host(const float* pcm_in, float* pcm_out, int n_samples);

    bool initialized() const { return initialized_; }
    float last_latency_ms() const { return last_lat_ms_; }

private:
    // Weight loading
    bool load_weights(const std::string& weights_dir);
    GpuTensor w(const std::string& name) const;

    // STFT / iSTFT
    void forward_stft(const float* d_pcm, int n_samples, int& n_frames);
    void forward_istft(int n_frames, float* d_pcm_out, int n_samples);

    // cuDNN Conv2d forward: [B, C_in, H, W] → [B, C_out, H_out, W_out]
    void forward_conv2d(const float* d_in, float* d_out,
                        int C_in, int C_out, int H, int W,
                        int kH, int kW, int sH, int sW, int pH, int pW,
                        const float* d_weight, const float* d_bias,
                        int& H_out, int& W_out);

    // cuDNN ConvTranspose2d forward (uses cudnnConvolutionBackwardData)
    void forward_conv_transpose2d(const float* d_in, float* d_out,
                                  int C_in, int C_out, int H, int W,
                                  int kH, int kW, int sH, int sW, int pH, int pW,
                                  const float* d_weight, const float* d_bias,
                                  int& H_out, int& W_out);

    // Complex Conv2d: 4 real Conv2d ops + complex combination
    void forward_complex_conv2d(const float* re_in, const float* im_in,
                                float* re_out, float* im_out,
                                int C_in, int C_out, int H, int W,
                                int kH, int kW, int sH, int sW, int pH, int pW,
                                const std::string& prefix,
                                int& H_out, int& W_out);

    // Complex ConvTranspose2d
    void forward_complex_tconv2d(const float* re_in, const float* im_in,
                                 float* re_out, float* im_out,
                                 int C_in, int C_out, int H, int W,
                                 int kH, int kW, int sH, int sW, int pH, int pW,
                                 const std::string& prefix,
                                 int& H_out, int& W_out);

    // Complex BatchNorm + LeakyReLU (fused)
    void forward_complex_bn_relu(float* re, float* im,
                                 int C, int H, int W,
                                 const std::string& prefix);

    // UniDeepFsmn forward (single real-valued FSMN block)
    void forward_fsmn(const float* d_in, float* d_out,
                      int H, int T,  // H = feature dim, T = sequence
                      const std::string& prefix);

    // ComplexUniDeepFsmn_L1 (1-layer complex FSMN)
    void forward_complex_fsmn_l1(float* re, float* im,
                                 int C, int H, int T,
                                 const std::string& prefix);

    // ComplexUniDeepFsmn (2-layer complex FSMN, bottleneck)
    void forward_complex_fsmn(float* re, float* im,
                              int C, int H, int T,
                              const std::string& prefix);

    // SE Layer (complex squeeze-excite)
    void forward_se_layer(float* re, float* im,
                          int C, int H, int W,
                          const std::string& prefix);

    // Full UNet forward pass
    void forward_unet(const float* re_in, const float* im_in,
                      float* re_out, float* im_out,
                      int H, int T,
                      const std::string& prefix);

    // ---- State ----
    bool initialized_ = false;
    float last_lat_ms_ = 0.0f;
    int max_samples_ = 0;
    int max_frames_ = 0;

    cudaStream_t stream_ = nullptr;
    cudnnHandle_t cudnn_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cufftHandle fft_plan_ = 0;
    cufftHandle ifft_plan_ = 0;

    // cuDNN workspace
    void*  d_cudnn_ws_ = nullptr;
    size_t cudnn_ws_size_ = 0;

    // ---- GPU weight storage ----
    float* d_weights_ = nullptr;        // single contiguous GPU allocation
    size_t weights_bytes_ = 0;
    std::unordered_map<std::string, GpuTensor> weight_map_;

    // ---- STFT buffers ----
    float* d_windowed_ = nullptr;       // [max_frames, kFftLen] windowed frames
    float* d_stft_window_ = nullptr;    // [kWinLen] sqrt(hann) window
    cufftComplex* d_stft_out_ = nullptr; // [max_frames, kFreqBins] complex STFT
    float* d_spec_re_ = nullptr;        // [kFreqBins, max_frames] real part
    float* d_spec_im_ = nullptr;        // [kFreqBins, max_frames] imaginary part

    // ---- iSTFT buffers ----
    cufftComplex* d_istft_in_ = nullptr; // [max_frames, kFreqBins]
    float* d_istft_out_ = nullptr;       // [max_frames, kFftLen]
    float* d_ola_buf_ = nullptr;         // overlap-add output [max_samples + kWinLen]
    float* d_ola_norm_ = nullptr;        // window normalization [max_samples + kWinLen]

    // ---- Activation scratch buffers (shared across UNet1 & UNet2) ----
    // Encoder skip connections: need to store SE outputs at each level
    // Max size per level: 2 * kChannels * max_H * max_T * sizeof(float)
    // We store real and imaginary separately.
    struct SkipBuf {
        float* re = nullptr;
        float* im = nullptr;
        int H = 0, W = 0;  // actual dimensions after SE (set during forward)
    };
    SkipBuf enc_skip_[frcrn::kNumStages + 1];  // +1 for input

    // General-purpose scratch buffers for intermediate results
    float* d_scratch_a_ = nullptr;  // large scratch
    float* d_scratch_b_ = nullptr;  // large scratch
    float* d_scratch_c_ = nullptr;  // medium scratch (for FSMN / SE)
    float* d_scratch_d_ = nullptr;  // medium scratch
    size_t scratch_a_size_ = 0;
    size_t scratch_b_size_ = 0;
    size_t scratch_c_size_ = 0;
    size_t scratch_d_size_ = 0;

    // Host staging buffer for enhance_host()
    float* d_pcm_staging_ = nullptr;

    // ---- UNet output buffers (for mask application) ----
    float* d_mask1_re_ = nullptr;   // UNet1 output (mask 1)
    float* d_mask1_im_ = nullptr;
    float* d_mask2_re_ = nullptr;   // UNet2 output (mask 2)
    float* d_mask2_im_ = nullptr;
};

// ============================================================================
// CUDA kernel launch declarations (implemented in frcrn_kernels.cu)
// ============================================================================
namespace frcrn_kernels {

// Apply analysis window and frame PCM for STFT
// PCM [n_samples] → windowed frames [n_frames, fft_len]
void launch_stft_frame(const float* d_pcm, const float* d_window,
                       float* d_frames, int n_samples, int n_frames,
                       int win_len, int hop, int fft_len,
                       cudaStream_t stream);

// Deinterleave cuFFT C2C/R2C output to split real/imag
// complex [n_frames, freq_bins] → real [freq_bins, n_frames], imag [freq_bins, n_frames]
void launch_stft_deinterleave(const cufftComplex* d_complex,
                              float* d_real, float* d_imag,
                              int n_frames, int freq_bins,
                              cudaStream_t stream);

// Interleave real/imag to cuFFT complex format + overlap-add after iFFT
void launch_istft_interleave(const float* d_real, const float* d_imag,
                             cufftComplex* d_complex,
                             int n_frames, int freq_bins,
                             cudaStream_t stream);

// iSTFT overlap-add: accumulate windowed iFFT frames
void launch_istft_ola(const float* d_frames, const float* d_window,
                      float* d_ola_buf, float* d_ola_norm,
                      int n_frames, int win_len, int hop, int fft_len,
                      int out_len, cudaStream_t stream);

// Normalize OLA output and copy to output
void launch_istft_normalize(const float* d_ola_buf, const float* d_ola_norm,
                            float* d_out, int n_samples,
                            cudaStream_t stream);

// Fused BatchNorm (eval mode) + LeakyReLU
// Input/output: [C, H, W], BN params: [C] each
void launch_bn_leakyrelu(float* d_inout, int C, int H, int W,
                         const float* d_gamma, const float* d_beta,
                         const float* d_mean, const float* d_var,
                         float eps, float neg_slope,
                         cudaStream_t stream);

// BatchNorm only (no activation) — for decoder where LeakyReLU is separate
void launch_bn(float* d_inout, int C, int H, int W,
               const float* d_gamma, const float* d_beta,
               const float* d_mean, const float* d_var,
               float eps, cudaStream_t stream);

// LeakyReLU in-place
void launch_leaky_relu(float* d_data, int n, float neg_slope,
                       cudaStream_t stream);

// Complex arithmetic: out_re = a_re - b_re, out_im = a_im + b_im
// (for complex Conv2d combination)
void launch_complex_combine(const float* a_re, const float* a_im,
                            const float* b_re, const float* b_im,
                            float* out_re, float* out_im,
                            int n, cudaStream_t stream);

// Elementwise add: out = a + b
void launch_add(const float* a, const float* b, float* out, int n,
                cudaStream_t stream);

// Elementwise add in-place: a += b
void launch_add_inplace(float* a, const float* b, int n,
                        cudaStream_t stream);

// Bias add: data[c * HW + i] += bias[c], for C channels of size HW
void launch_bias_add(float* d_data, const float* d_bias,
                     int C, int HW, cudaStream_t stream);

// im2col: unroll input [C_in, H, W] patches into [C_in*kH*kW, H_out*W_out]
void launch_im2col(const float* d_im, float* d_col,
                   int C_in, int H, int W,
                   int kH, int kW, int sH, int sW, int pH, int pW,
                   int H_out, int W_out, cudaStream_t stream);

// tanh activation in-place
void launch_tanh(float* d_data, int n, cudaStream_t stream);

// Complex mask apply:
//   out_re = spec_re * mask_re - spec_im * mask_im
//   out_im = spec_re * mask_im + spec_im * mask_re
void launch_complex_mask(const float* spec_re, const float* spec_im,
                         const float* mask_re, const float* mask_im,
                         float* out_re, float* out_im,
                         int n, cudaStream_t stream);

// FSMN depthwise conv: y[c,t] = sum_{k=0}^{order-1} w[c,k] * x[c, t-order+1+k]
// with left padding (causal). x is [C, T], w is [C, order].
void launch_fsmn_conv(const float* d_in, float* d_out,
                      const float* d_weight,  // [C, 1, order, 1] → use as [C, order]
                      int C, int T, int order,
                      cudaStream_t stream);

// SE global average pool: [C, H, W] → [C]
void launch_se_avg_pool(const float* d_in, float* d_out,
                        int C, int HW, cudaStream_t stream);

// SE apply scale: data[c, h, w] *= scale[c]
void launch_se_scale(float* d_data, const float* d_scale,
                     int C, int HW, cudaStream_t stream);

// Elementwise ReLU
void launch_relu(float* d_data, int n, cudaStream_t stream);

// Elementwise sigmoid
void launch_sigmoid(float* d_data, int n, cudaStream_t stream);

// Concatenate along channel dimension: [C1, H, W] + [C2, H, W] → [C1+C2, H, W]
void launch_concat_channels(const float* d_a, const float* d_b,
                            float* d_out, int C1, int C2, int HW,
                            cudaStream_t stream);

// Zero buffer
void launch_zero(float* d_buf, int n, cudaStream_t stream);

}  // namespace frcrn_kernels

}  // namespace deusridet
