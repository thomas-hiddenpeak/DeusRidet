// frcrn_kernels.cu — Custom CUDA kernels for FRCRN speech enhancement.
//
// Elementwise operations, fused BatchNorm+activation, STFT framing,
// iSTFT overlap-add, FSMN depthwise conv, SE pooling, complex arithmetic.
//
// All kernels use float32 (model is FP32). Batch size is always 1.

#include "frcrn_gpu.h"
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace deusridet {
namespace frcrn_kernels {

static constexpr int BLOCK = 256;

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// STFT framing: apply window and extract frames for cuFFT
// ============================================================================
__global__ void stft_frame_kernel(
    const float* __restrict__ pcm,
    const float* __restrict__ window,
    float* __restrict__ frames,
    int n_samples, int n_frames, int win_len, int hop, int fft_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * fft_len;
    if (idx >= total) return;

    int frame = idx / fft_len;
    int k = idx % fft_len;

    int sample_idx = frame * hop + k;
    float val = 0.0f;
    if (k < win_len && sample_idx < n_samples) {
        val = pcm[sample_idx] * window[k];
    }
    frames[idx] = val;
}

void launch_stft_frame(const float* d_pcm, const float* d_window,
                       float* d_frames, int n_samples, int n_frames,
                       int win_len, int hop, int fft_len,
                       cudaStream_t stream) {
    int total = n_frames * fft_len;
    stft_frame_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_pcm, d_window, d_frames, n_samples, n_frames, win_len, hop, fft_len);
}

// ============================================================================
// STFT deinterleave: cufftComplex [n_frames, freq_bins] →
//   real [freq_bins, n_frames], imag [freq_bins, n_frames]
// Transposes from frame-major to frequency-major layout.
// ============================================================================
__global__ void stft_deinterleave_kernel(
    const cufftComplex* __restrict__ complex_in,
    float* __restrict__ real_out,
    float* __restrict__ imag_out,
    int n_frames, int freq_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * freq_bins;
    if (idx >= total) return;

    int frame = idx / freq_bins;
    int bin = idx % freq_bins;

    cufftComplex c = complex_in[idx];  // [frame, bin]

    // Write transposed: [bin, frame]
    int out_idx = bin * n_frames + frame;
    real_out[out_idx] = c.x;
    imag_out[out_idx] = c.y;
}

void launch_stft_deinterleave(const cufftComplex* d_complex,
                              float* d_real, float* d_imag,
                              int n_frames, int freq_bins,
                              cudaStream_t stream) {
    int total = n_frames * freq_bins;
    stft_deinterleave_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_complex, d_real, d_imag, n_frames, freq_bins);
}

// ============================================================================
// iSTFT interleave: real [freq_bins, n_frames], imag → complex [n_frames, freq_bins]
// ============================================================================
__global__ void istft_interleave_kernel(
    const float* __restrict__ real_in,
    const float* __restrict__ imag_in,
    cufftComplex* __restrict__ complex_out,
    int n_frames, int freq_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * freq_bins;
    if (idx >= total) return;

    int frame = idx / freq_bins;
    int bin = idx % freq_bins;

    // Read transposed: [bin, frame]
    int in_idx = bin * n_frames + frame;
    cufftComplex c;
    c.x = real_in[in_idx];
    c.y = imag_in[in_idx];
    complex_out[idx] = c;  // [frame, bin]
}

void launch_istft_interleave(const float* d_real, const float* d_imag,
                             cufftComplex* d_complex,
                             int n_frames, int freq_bins,
                             cudaStream_t stream) {
    int total = n_frames * freq_bins;
    istft_interleave_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_real, d_imag, d_complex, n_frames, freq_bins);
}

// ============================================================================
// iSTFT overlap-add
// ============================================================================
__global__ void istft_ola_kernel(
    const float* __restrict__ frames,  // [n_frames, fft_len]
    const float* __restrict__ window,  // [win_len]
    float* __restrict__ ola_buf,
    float* __restrict__ ola_norm,
    int n_frames, int win_len, int hop, int fft_len, int out_len)
{
    // One thread per (frame, k) pair where k < win_len
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * win_len;
    if (idx >= total) return;

    int frame = idx / win_len;
    int k = idx % win_len;
    int out_pos = frame * hop + k;
    if (out_pos >= out_len) return;

    float w = window[k];
    // cuFFT C2R doesn't normalize by N — fold 1/N into OLA
    float inv_n = 1.0f / (float)fft_len;
    float val = frames[frame * fft_len + k] * w * inv_n;
    float w2 = w * w;

    atomicAdd(&ola_buf[out_pos], val);
    atomicAdd(&ola_norm[out_pos], w2);
}

void launch_istft_ola(const float* d_frames, const float* d_window,
                      float* d_ola_buf, float* d_ola_norm,
                      int n_frames, int win_len, int hop, int fft_len,
                      int out_len, cudaStream_t stream) {
    int total = n_frames * win_len;
    istft_ola_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_frames, d_window, d_ola_buf, d_ola_norm,
        n_frames, win_len, hop, fft_len, out_len);
}

__global__ void istft_normalize_kernel(
    const float* __restrict__ ola_buf,
    const float* __restrict__ ola_norm,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float norm = ola_norm[idx];
    out[idx] = (norm > 1e-8f) ? (ola_buf[idx] / norm) : 0.0f;
}

void launch_istft_normalize(const float* d_ola_buf, const float* d_ola_norm,
                            float* d_out, int n_samples,
                            cudaStream_t stream) {
    istft_normalize_kernel<<<div_ceil(n_samples, BLOCK), BLOCK, 0, stream>>>(
        d_ola_buf, d_ola_norm, d_out, n_samples);
}

// ============================================================================
// Fused BatchNorm (eval mode) + LeakyReLU
// data[c, h, w] = LeakyReLU( (data[c,h,w] - mean[c]) / sqrt(var[c]+eps) * gamma[c] + beta[c] )
// ============================================================================
__global__ void bn_leakyrelu_kernel(
    float* __restrict__ data,
    int C, int HW,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps, float neg_slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * HW;
    if (idx >= total) return;

    int c = idx / HW;
    float scale = gamma[c] * rsqrtf(var[c] + eps);
    float shift = beta[c] - mean[c] * scale;

    float val = data[idx] * scale + shift;
    // LeakyReLU
    val = (val >= 0.0f) ? val : val * neg_slope;
    data[idx] = val;
}

void launch_bn_leakyrelu(float* d_inout, int C, int H, int W,
                         const float* d_gamma, const float* d_beta,
                         const float* d_mean, const float* d_var,
                         float eps, float neg_slope,
                         cudaStream_t stream) {
    int total = C * H * W;
    bn_leakyrelu_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_inout, C, H * W, d_gamma, d_beta, d_mean, d_var, eps, neg_slope);
}

void launch_bn(float* d_inout, int C, int H, int W,
               const float* d_gamma, const float* d_beta,
               const float* d_mean, const float* d_var,
               float eps, cudaStream_t stream) {
    launch_bn_leakyrelu(d_inout, C, H, W, d_gamma, d_beta,
                        d_mean, d_var, eps, 1.0f, stream);
}

// ============================================================================
// Elementwise operations
// ============================================================================
__global__ void leaky_relu_kernel(float* data, int n, float neg_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = data[idx];
    data[idx] = (v >= 0.0f) ? v : v * neg_slope;
}

void launch_leaky_relu(float* d_data, int n, float neg_slope,
                       cudaStream_t stream) {
    leaky_relu_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(
        d_data, n, neg_slope);
}

// Complex combination: out_re = a - b, out_im = c + d
// For ComplexConv2d: re_out = conv_re(re_in) - conv_im(im_in)
//                    im_out = conv_re(im_in) + conv_im(re_in)
__global__ void complex_combine_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, const float* __restrict__ d,
    float* __restrict__ out_re, float* __restrict__ out_im, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out_re[idx] = a[idx] - b[idx];
    out_im[idx] = c[idx] + d[idx];
}

void launch_complex_combine(const float* a_re, const float* a_im,
                            const float* b_re, const float* b_im,
                            float* out_re, float* out_im,
                            int n, cudaStream_t stream) {
    complex_combine_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(
        a_re, a_im, b_re, b_im, out_re, out_im, n);
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

void launch_add(const float* a, const float* b, float* out, int n,
                cudaStream_t stream) {
    add_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(a, b, out, n);
}

__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] += b[idx];
}

void launch_add_inplace(float* a, const float* b, int n,
                        cudaStream_t stream) {
    add_inplace_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(a, b, n);
}

__global__ void bias_add_kernel(float* data, const float* bias, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    int c = idx / HW;
    data[idx] += bias[c];
}

void launch_bias_add(float* d_data, const float* d_bias,
                     int C, int HW, cudaStream_t stream) {
    int total = C * HW;
    bias_add_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_data, d_bias, C, HW);
}

// im2col: unroll input [C_in, H, W] into column matrix [C_in*kH*kW, H_out*W_out]
// for use with cuBLAS GEMM to compute Conv2d without cuDNN
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int C_in, int H, int W,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int H_out, int W_out)
{
    int total = C_in * kH * kW * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w_col = idx % W_out;
    int h_col = (idx / W_out) % H_out;
    int c_col = idx / (W_out * H_out);  // which element in the unrolled filter

    int c_im = c_col / (kH * kW);
    int kh = (c_col / kW) % kH;
    int kw = c_col % kW;

    int h_im = h_col * sH - pH + kh;
    int w_im = w_col * sW - pW + kw;

    if (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) {
        data_col[idx] = data_im[(c_im * H + h_im) * W + w_im];
    } else {
        data_col[idx] = 0.0f;
    }
}

void launch_im2col(const float* d_im, float* d_col,
                   int C_in, int H, int W,
                   int kH, int kW, int sH, int sW, int pH, int pW,
                   int H_out, int W_out, cudaStream_t stream) {
    int total = C_in * kH * kW * H_out * W_out;
    im2col_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_im, d_col, C_in, H, W, kH, kW, sH, sW, pH, pW, H_out, W_out);
}

__global__ void tanh_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = tanhf(data[idx]);
}

void launch_tanh(float* d_data, int n, cudaStream_t stream) {
    tanh_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(d_data, n);
}

// Complex mask: (spec_re + i*spec_im) * (mask_re + i*mask_im)
__global__ void complex_mask_kernel(
    const float* __restrict__ sr, const float* __restrict__ si,
    const float* __restrict__ mr, const float* __restrict__ mi,
    float* __restrict__ out_re, float* __restrict__ out_im, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = sr[idx], b = si[idx], c = mr[idx], d = mi[idx];
    out_re[idx] = a * c - b * d;
    out_im[idx] = a * d + b * c;
}

void launch_complex_mask(const float* spec_re, const float* spec_im,
                         const float* mask_re, const float* mask_im,
                         float* out_re, float* out_im,
                         int n, cudaStream_t stream) {
    complex_mask_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(
        spec_re, spec_im, mask_re, mask_im, out_re, out_im, n);
}

// ============================================================================
// FSMN depthwise causal convolution
// x: [C, T], weight: [C, order] (stored as [C, 1, order, 1] in PyTorch)
// y[c, t] = x[c, t] + sum_{k=0}^{order-1} w[c, k] * x[c, t - order + 1 + k]
// Left-padded (causal): for t < order-1, missing samples are zero.
// ============================================================================
__global__ void fsmn_conv_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    int C, int T, int order)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;

    int c = idx / T;
    int t = idx % T;

    float sum = 0.0f;
    const float* w = weight + c * order;
    for (int k = 0; k < order; k++) {
        int src_t = t - order + 1 + k;
        if (src_t >= 0) {
            sum += w[k] * x[c * T + src_t];
        }
    }
    // Residual connection: output = input + conv(projected)
    // Note: the residual is added in the FSMN forward, not here.
    y[idx] = sum;
}

void launch_fsmn_conv(const float* d_in, float* d_out,
                      const float* d_weight,
                      int C, int T, int order,
                      cudaStream_t stream) {
    int total = C * T;
    fsmn_conv_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_in, d_out, d_weight, C, T, order);
}

// ============================================================================
// SE Layer kernels
// ============================================================================

// Global average pool: [C, HW] → [C]
__global__ void se_avg_pool_kernel(
    const float* __restrict__ data, float* __restrict__ out,
    int C, int HW)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    const float* row = data + c * HW;
    for (int i = 0; i < HW; i++) {
        sum += row[i];
    }
    out[c] = sum / (float)HW;
}

void launch_se_avg_pool(const float* d_in, float* d_out,
                        int C, int HW, cudaStream_t stream) {
    se_avg_pool_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream>>>(
        d_in, d_out, C, HW);
}

// Scale: data[c, i] *= scale[c]
__global__ void se_scale_kernel(float* data, const float* scale, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    int c = idx / HW;
    data[idx] *= scale[c];
}

void launch_se_scale(float* d_data, const float* d_scale,
                     int C, int HW, cudaStream_t stream) {
    int total = C * HW;
    se_scale_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_data, d_scale, C, HW);
}

__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = fmaxf(data[idx], 0.0f);
}

void launch_relu(float* d_data, int n, cudaStream_t stream) {
    relu_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(d_data, n);
}

__global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = 1.0f / (1.0f + __expf(-data[idx]));
}

void launch_sigmoid(float* d_data, int n, cudaStream_t stream) {
    sigmoid_kernel<<<div_ceil(n, BLOCK), BLOCK, 0, stream>>>(d_data, n);
}

// ============================================================================
// Channel concatenation: [C1, HW] + [C2, HW] → [C1+C2, HW]
// ============================================================================
__global__ void concat_channels_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int C1, int C2, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (C1 + C2) * HW;
    if (idx >= total) return;

    int c = idx / HW;
    int hw = idx % HW;
    if (c < C1) {
        out[idx] = a[c * HW + hw];
    } else {
        out[idx] = b[(c - C1) * HW + hw];
    }
}

void launch_concat_channels(const float* d_a, const float* d_b,
                            float* d_out, int C1, int C2, int HW,
                            cudaStream_t stream) {
    int total = (C1 + C2) * HW;
    concat_channels_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream>>>(
        d_a, d_b, d_out, C1, C2, HW);
}

// ============================================================================
// Zero buffer
// ============================================================================
void launch_zero(float* d_buf, int n, cudaStream_t stream) {
    cudaMemsetAsync(d_buf, 0, n * sizeof(float), stream);
}

}  // namespace frcrn_kernels
}  // namespace deusridet
