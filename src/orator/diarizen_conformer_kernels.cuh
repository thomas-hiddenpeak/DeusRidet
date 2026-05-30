/**
 * @file diarizen_conformer_kernels.cuh
 * @philosophical_role Shared device kernels for the DiariZen Conformer head
 *     (P1b). Header-only, anonymous namespace: each including TU gets its own
 *     internal copy, so forward.cu can launch them without cross-TU linkage of
 *     internal-linkage symbols. Compute belongs on the GPU — every conformer
 *     elementwise / conv / softmax reduction lives here as a CUDA kernel.
 * @serves DiarizenConformerHead forward path.
 */
#pragma once

#include <cmath>

#include <cuda_runtime.h>

namespace deusridet {
namespace orator {
namespace {

// Swish(x) = x * sigmoid(x), in place.
__global__ void swish_kernel(float* __restrict__ x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

// GLU over the channel dim of a [T, 2C] map (frame-major): out[t, c] =
// a[t, c] * sigmoid(a[t, c + C]). One thread per (t, c).
__global__ void glu_tc_kernel(const float* __restrict__ in,  // [T, 2C]
                              float* __restrict__ out,        // [T, C]
                              int T, int C) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)T * C) return;
    int t = (int)(idx / C);
    int c = (int)(idx % C);
    float a = in[(long)t * 2 * C + c];
    float g = in[(long)t * 2 * C + C + c];
    out[idx] = a * (1.0f / (1.0f + expf(-g)));
}

// Depthwise 1-D conv with SAME padding (pad = K/2), per channel, on a [C, T]
// map. weight is [C, 1, K]. out[c, t] = bias[c] + sum_k w[c,k]*in[c, t+k-pad].
__global__ void depthwise_conv1d_kernel(const float* __restrict__ in,   // [C,T]
                                        const float* __restrict__ w,    // [C,K]
                                        const float* __restrict__ b,    // [C]
                                        float* __restrict__ out,        // [C,T]
                                        int C, int T, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)C * T) return;
    int c = (int)(idx / T);
    int t = (int)(idx % T);
    int pad = K / 2;
    const float* wc = w + (long)c * K;
    const float* ic = in + (long)c * T;
    float acc = b ? b[c] : 0.0f;
    for (int k = 0; k < K; ++k) {
        int ti = t + k - pad;
        if (ti >= 0 && ti < T) acc += wc[k] * ic[ti];
    }
    out[idx] = acc;
}

// BatchNorm1d in eval mode over a [C, T] map: out[c,t] = (in[c,t]-mean[c]) /
// sqrt(var[c]+eps) * gamma[c] + beta[c]. One thread per (c, t).
__global__ void batchnorm_ct_kernel(float* __restrict__ x,             // [C,T]
                                    const float* __restrict__ gamma,   // [C]
                                    const float* __restrict__ beta,    // [C]
                                    const float* __restrict__ mean,    // [C]
                                    const float* __restrict__ var,     // [C]
                                    int C, int T, float eps) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)C * T) return;
    int c = (int)(idx / T);
    float inv = rsqrtf(var[c] + eps);
    x[idx] = (x[idx] - mean[c]) * inv * gamma[c] + beta[c];
}

// Transpose [T, C] -> [C, T].
__global__ void transpose_tc_to_ct_kernel(const float* __restrict__ in,  // [T,C]
                                          float* __restrict__ out,        // [C,T]
                                          int T, int C) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)T * C) return;
    int t = (int)(idx / C);
    int c = (int)(idx % C);
    out[(long)c * T + t] = in[idx];
}

// Transpose [C, T] -> [T, C].
__global__ void transpose_ct_to_tc_kernel(const float* __restrict__ in,  // [C,T]
                                          float* __restrict__ out,        // [T,C]
                                          int C, int T) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)C * T) return;
    int c = (int)(idx / T);
    int t = (int)(idx % T);
    out[(long)t * C + c] = in[idx];
}

// Scaled-residual add: res[i] += scale * x[i].
__global__ void scaled_add_kernel(float* __restrict__ res,
                                  const float* __restrict__ x, float scale,
                                  long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] += scale * x[i];
}

// Multi-head self-attention scores and context are done by
// cublasSgemmStridedBatched (tensor-core path) in diarizen_conformer_forward.cu;
// no hand-rolled matmul kernel is needed here.

// Row softmax over the last dim (length T) of an [nh, T, T] tensor; one block
// per (j, q) row.
__global__ void softmax_rows_kernel_c(float* __restrict__ S, int T) {
    int row = blockIdx.x;          // 0 .. nh*T-1
    float* s = S + (long)row * T;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float m = -1e30f;
    for (int k = tid; k < T; k += blockDim.x) m = fmaxf(m, s[k]);
    red[tid] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
        __syncthreads();
    }
    float rmax = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int k = tid; k < T; k += blockDim.x) {
        float e = expf(s[k] - rmax);
        s[k] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        __syncthreads();
    }
    float rsum = red[0] + 1e-20f;
    for (int k = tid; k < T; k += blockDim.x) s[k] /= rsum;
}

// Attention context is done by cublasSgemmStridedBatched in
// diarizen_conformer_forward.cu; no hand-rolled matmul kernel here.

// LogSoftmax over the last dim (length C) of a [T, C] tensor; one block per row.
__global__ void logsoftmax_rows_kernel(float* __restrict__ X, int T, int C) {
    int row = blockIdx.x;
    if (row >= T) return;
    float* x = X + (long)row * C;
    float m = -1e30f;
    for (int c = 0; c < C; ++c) m = fmaxf(m, x[c]);
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) sum += expf(x[c] - m);
    float lse = m + logf(sum);
    for (int c = 0; c < C; ++c) x[c] = x[c] - lse;
}

}  // namespace
}  // namespace orator
}  // namespace deusridet
