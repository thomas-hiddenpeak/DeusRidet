/**
 * @file diarizen_wavlm_pruned_kernels.cuh
 * @philosophical_role Shared device kernels + host helpers for the DiariZen
 *     WavLM-pruned native forward pass. Split out so the CNN/front-end TU
 *     (diarizen_wavlm_pruned_forward.cu) and the transformer-layer TU
 *     (diarizen_wavlm_pruned_layers.cu) reuse one definition each instead of
 *     copying kernels (anti-entropy). Everything lives in an anonymous
 *     namespace, giving each including TU its own internal copy with no
 *     link-time symbol clash.
 * @serves DiarizenWavlmPruned forward path. Compute belongs on the GPU.
 */
#pragma once

#include "../communis/log.h"

#include <cmath>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {
namespace orator {
namespace {

constexpr const char* kLog = "DiariZenWavlm";
constexpr int kBlock = 256;

inline int div_ceil_(int a, int b) { return (a + b - 1) / b; }

inline bool cuda_ck_(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        LOG_ERROR(kLog, "CUDA %s failed: %s", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// fp16 -> fp32 element-wise (weights live in the fp16 arena; cuBLAS/cuDNN
// run in fp32 to match the reference activation precision).
__global__ void half_to_float_kernel(const __half* __restrict__ src,
                                      float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). Matches torch
// nn.functional.gelu default (not the tanh approximation).
__global__ void gelu_exact_kernel(float* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

// Add a per-feature bias to a frame-major [T, N] buffer in place.
__global__ void bias_add_rows_kernel(float* __restrict__ data,  // [T, N]
                                     const float* __restrict__ bias,  // [N]
                                     int T, int N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * N;
    if (idx >= total) return;
    data[idx] += bias[idx % N];
}

// Per-frame LayerNorm over the feature dimension of a frame-major [T, C]
// buffer (one block per frame t). Affine (gamma/beta), eps 1e-5. When `out`
// differs from `data` the normalised result is written to `out`, leaving the
// input intact (needed for pre-norm residuals).
__global__ void row_layer_norm_to_kernel(const float* __restrict__ data,  // [T,C]
                                         float* __restrict__ out,          // [T,C]
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         int T, int C) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* row = data + (size_t)t * C;
    float* orow = out + (size_t)t * C;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) sum += row[c];
    for (int o = warpSize / 2; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, o);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp] = sum;
    __syncthreads();
    __shared__ float s_mean, s_inv;
    int nw = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nw; ++i) tot += s_buf[i];
        s_mean = tot / C;
    }
    __syncthreads();
    float mean = s_mean;

    float vs = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = row[c] - mean;
        vs += d * d;
    }
    for (int o = warpSize / 2; o > 0; o >>= 1)
        vs += __shfl_down_sync(0xffffffff, vs, o);
    if (lane == 0) s_buf[warp] = vs;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nw; ++i) tot += s_buf[i];
        s_inv = rsqrtf(tot / C + 1e-5f);
    }
    __syncthreads();
    float inv = s_inv;

    for (int c = threadIdx.x; c < C; c += blockDim.x)
        orow[c] = (row[c] - mean) * inv * gamma[c] + beta[c];
}

}  // namespace
}  // namespace orator
}  // namespace deusridet
