/**
 * @file diarizen_resnet34_kernels.cuh
 * @philosophical_role Header-only CUDA kernels for the WeSpeaker ResNet34-LM
 *   embedder (P2a). Anonymous namespace so each including TU owns an internal
 *   copy and can launch them directly (mirrors the conformer-head pattern).
 * @serves DiarizenResnet34Embedder.
 */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {
namespace orator {
namespace {

// fp16 -> fp32.
__global__ void r34_half_to_float(const __half* __restrict__ in,
                                  float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// Apply folded BatchNorm (per-channel scale/bias) on an NCHW activation with
// N=1, optionally followed by ReLU. x[c*HW + i] = x*scale[c] + bias[c].
__global__ void r34_bn_relu(float* __restrict__ x, const float* __restrict__ scale,
                            const float* __restrict__ bias, int C, int HW,
                            int do_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * HW;
    if (idx >= total) return;
    int c = idx / HW;
    float v = x[idx] * scale[c] + bias[c];
    if (do_relu) v = v > 0.0f ? v : 0.0f;
    x[idx] = v;
}

// out[i] = relu(out[i] + res[i]).
__global__ void r34_add_relu(float* __restrict__ out, const float* __restrict__ res,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = out[i] + res[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

// Cepstral mean normalization: subtract per-mel time mean. fbank row-major
// [T, M]; one thread per mel column m.
__global__ void r34_cmn(float* __restrict__ fbank, int T, int M) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float s = 0.0f;
    for (int t = 0; t < T; ++t) s += fbank[t * M + m];
    float mean = s / (float)T;
    for (int t = 0; t < T; ++t) fbank[t * M + m] -= mean;
}

// Transpose [T, M] (row-major) -> [M, T] (row-major). Used to lay the fbank
// out as the conv input [1,1,M=freq,T=time].
__global__ void r34_transpose_TM_to_MT(const float* __restrict__ in,
                                       float* __restrict__ out, int T, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * M) return;
    int t = idx / M;
    int m = idx % M;
    out[m * T + t] = in[t * M + m];
}

// Nearest-neighbour resample of a 1-D weight vector, matching PyTorch
// F.interpolate(mode="nearest"): src = min(floor(dst * in/out), in-1).
__global__ void r34_interp_nearest(const float* __restrict__ win, int n_in,
                                   float* __restrict__ wout, int n_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    float scale = (float)n_in / (float)n_out;
    int src = (int)((float)i * scale);
    if (src >= n_in) src = n_in - 1;
    wout[i] = win[src];
}

// Weighted statistics pooling (StatsPool._pool). seq is [rows, T] row-major,
// weights w is [T]. One thread per row r:
//   v1 = sum(w) + 1e-8;  mean = sum(seq*w)/v1
//   v2 = sum(w^2);  var = sum((seq-mean)^2 * w)/(v1 - v2/v1 + 1e-8); std=sqrt(var)
// Writes mean to out[r] and std to out[rows + r] (concatenated pool vector).
__global__ void r34_stats_pool(const float* __restrict__ seq,
                               const float* __restrict__ w,
                               float* __restrict__ out, int rows, int T) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = seq + (size_t)r * T;
    float v1 = 0.0f, v2 = 0.0f, sw = 0.0f;
    for (int t = 0; t < T; ++t) {
        float wt = w[t];
        v1 += wt;
        v2 += wt * wt;
        sw += row[t] * wt;
    }
    v1 += 1e-8f;
    float mean = sw / v1;
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) {
        float d = row[t] - mean;
        acc += d * d * w[t];
    }
    float denom = v1 - v2 / v1 + 1e-8f;
    float var = acc / denom;
    out[r] = mean;
    out[rows + r] = sqrtf(var);
}

}  // namespace
}  // namespace orator
}  // namespace deusridet
