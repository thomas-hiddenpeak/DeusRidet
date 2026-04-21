/**
 * @file src/orator/wavlm_ecapa_kernels.cuh
 * @philosophical_role
 *   TU-local CUDA kernels and tiny utilities shared across the
 *   wavlm_ecapa_encoder.cu peer split. Textual include — every kernel is
 *   `static __global__` so each TU gets its own private instantiation.
 * @serves
 *   wavlm_ecapa_encoder.cu and its peers (frontend / transformer / ecapa /
 *   extract).
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

namespace deusridet {

static constexpr int BLOCK = 256;
static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

static __global__ void layer_norm_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = input + row * D;
    float* y = output + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        sum += x[i];
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    // Block reduction via shared memory
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_mean = total / D;
    }
    __syncthreads();
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[i] - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_inv_std = rsqrtf(total / D + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = (x[i] - mean) * inv_std;
        if (gamma) val = val * gamma[i];
        if (beta) val = val + beta[i];
        y[i] = val;
    }
}

static __global__ void wav_layer_norm_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int T) {
    // Phase 1: compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x)
        sum += input[i];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_mean = total / T;
    }
    __syncthreads();
    float mean = s_mean;

    // Phase 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        float v = input[i] - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_inv_std = rsqrtf(total / T + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Phase 3: normalize
    for (int i = threadIdx.x; i < T; i += blockDim.x)
        output[i] = (input[i] - mean) * inv_std;
}

static __global__ void f32_to_f16_wlecapa(const float* __restrict__ in,
                                    __half* __restrict__ out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float a = in[idx], b = in[idx + 1];
        *reinterpret_cast<__half2*>(out + idx) = __floats2half2_rn(a, b);
    } else if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

static __global__ void truncate_channels_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int C, int T_src, int T_dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T_dst) return;
    int c = idx / T_dst;
    int t = idx % T_dst;
    out[c * T_dst + t] = in[c * T_src + t];
}

static __global__ void conv1d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int Cin, int T_in, int Cout, int T_out,
                              int K, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * T_out) return;
    int co = idx / T_out;
    int to = idx % T_out;
    float sum = bias ? bias[co] : 0.0f;
    int t_start = to * stride;
    for (int ci = 0; ci < Cin; ++ci) {
        const float* w = weight + (co * Cin + ci) * K;
        const float* x = input + ci * T_in + t_start;
        for (int ki = 0; ki < K; ++ki)
            sum += w[ki] * x[ki];
    }
    output[idx] = sum;
}

static __global__ void layer_norm_channels_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            int C, int T) {
    int t = blockIdx.x;
    if (t >= T) return;

    // Compute mean over C channels at time t
    // input layout: [C, T], element [c, t] = input[c * T + t]
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        sum += input[c * T + t];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_mean = total / C;
    }
    __syncthreads();
    float mean = s_mean;

    // Variance
    float var_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = input[c * T + t] - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_inv_std = rsqrtf(total / C + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Normalize: same [C, T] layout
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = (input[c * T + t] - mean) * inv_std;
        if (gamma) val = val * gamma[c];
        if (beta) val = val + beta[c];
        output[c * T + t] = val;
    }
}

static __global__ void gelu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float x = data[idx];
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    data[idx] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

static __global__ void transpose_2d_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    output[n * M + m] = input[m * N + n];
}

static __global__ void bias_add_kernel(float* __restrict__ data,
                                const float* __restrict__ bias,
                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = idx % cols;
    data[idx] += bias[c];
}

static __global__ void bias_add_channel_kernel(float* __restrict__ data,
                                         const float* __restrict__ bias,
                                         int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int ch = idx / T;
    data[idx] += bias[ch];
}

static __global__ void grouped_conv1d_padded_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int C, int T_in, int K, int pad, int groups) {
    // Output T = T_in + 2*pad - K + 1; SamePad removes 1 → T_out = T_in
    int T_padded = T_in + 2 * pad;
    int T_out = T_padded - K + 1 - 1;  // -1 for SamePad (even kernel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T_out) return;

    int co = idx / T_out;
    int to = idx % T_out;
    int group_size = C / groups;
    int g = co / group_size;
    int co_in_group = co % group_size;

    float sum = bias ? bias[co] : 0.0f;
    int ci_start = g * group_size;
    for (int ci_g = 0; ci_g < group_size; ci_g++) {
        int ci = ci_start + ci_g;
        const float* w_ptr = weight + (co * group_size + ci_g) * K;
        for (int ki = 0; ki < K; ki++) {
            int t_in_padded = to + ki;
            int t_in = t_in_padded - pad;
            float x_val = (t_in >= 0 && t_in < T_in) ? input[ci * T_in + t_in] : 0.0f;
            sum += w_ptr[ki] * x_val;
        }
    }
    output[idx] = sum;
}

static __global__ void weight_norm_kernel(const float* __restrict__ g,
                                    const float* __restrict__ v,
                                    float* __restrict__ weight,
                                    int total_per_k, int K) {
    // Each block handles one k
    int k = blockIdx.x;
    if (k >= K) return;

    // Compute L2 norm of v[:,:,k]
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < total_per_k; i += blockDim.x) {
        float val = v[i * K + k];
        sum_sq += val * val;
    }
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum_sq;
    __syncthreads();
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_scale = g[k] / (sqrtf(total) + 1e-12f);
    }
    __syncthreads();
    float scale = s_scale;

    // Apply scale
    for (int i = threadIdx.x; i < total_per_k; i += blockDim.x) {
        weight[i * K + k] = v[i * K + k] * scale;
    }
}

static __global__ void skip_add_gelu_kernel(const float* __restrict__ input,
                                      const float* __restrict__ skip,
                                      float* __restrict__ output,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = input[idx] + skip[idx];
    output[idx] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

static __global__ void compute_rel_pos_bias_kernel(
        const float* __restrict__ rel_attn_bias,  // [num_buckets, num_heads]
        float* __restrict__ output,                // [num_heads, T, T]
        int T, int num_heads, int num_buckets, int max_distance) {
    // Each thread handles one (head, qi, ki) element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * T * T;
    if (idx >= total) return;

    int h = idx / (T * T);
    int remainder = idx % (T * T);
    int qi = remainder / T;
    int ki = remainder % T;

    // Compute relative position bucket
    int rel_pos = ki - qi;  // positive = forward
    int num_buckets_half = num_buckets / 2;
    int bucket;

    // Bidirectional: first half buckets for negative, second half for positive
    if (rel_pos > 0) {
        bucket = num_buckets_half;
        int abs_pos = rel_pos;
        int max_exact = num_buckets_half / 2;
        if (abs_pos < max_exact) {
            bucket += abs_pos;
        } else {
            // Log-space bucketing
            float log_ratio = logf((float)abs_pos / (float)max_exact)
                            / logf((float)max_distance / (float)max_exact);
            bucket += max_exact + (int)(log_ratio * (num_buckets_half - max_exact));
            if (bucket >= num_buckets) bucket = num_buckets - 1;
        }
    } else {
        bucket = 0;
        int abs_pos = -rel_pos;
        int max_exact = num_buckets_half / 2;
        if (abs_pos < max_exact) {
            bucket += abs_pos;
        } else {
            float log_ratio = logf((float)abs_pos / (float)max_exact)
                            / logf((float)max_distance / (float)max_exact);
            bucket += max_exact + (int)(log_ratio * (num_buckets_half - max_exact));
            if (bucket >= num_buckets_half) bucket = num_buckets_half - 1;
        }
    }

    // Lookup: rel_attn_bias is [num_buckets, num_heads], row-major
    output[idx] = rel_attn_bias[bucket * num_heads + h];
}

static __global__ void gru_rel_pos_kernel(float* __restrict__ attn_weights,
                                    const float* __restrict__ q,
                                    const float* __restrict__ grep_linear_w,
                                    const float* __restrict__ grep_linear_b,
                                    const float* __restrict__ grep_a,
                                    int T, int num_heads, int head_dim) {
    // This kernel is a simplified version — computes GRU gating per head
    // For each head h, compute:
    //   q_mean = mean(q[:, h, :, :]) over T → [head_dim]
    //   gate_in = Linear(q_mean) → [2]  (for groups of heads)
    //   gate = sigmoid(gate_in[0]) * 2 (rel_pos_gate), sigmoid(gate_in[1]) * 2 (linear_pos_gate)
    // Then scale the relative position bias per head
    //
    // This is complex enough to warrant its own implementation.
    // For now, we implement it correctly in the host-side orchestration.
}

static __global__ void softmax_kernel(float* __restrict__ data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // Find max (numerical stability)
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_data[i];
        if (v > max_val) max_val = v;
    }
    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        if (other > max_val) max_val = other;
    }
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = max_val;
    __syncthreads();
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int i = 0; i < num_warps; i++)
            if (s_buf[i] > m) m = s_buf[i];
        s_max = m;
    }
    __syncthreads();
    max_val = s_max;

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __expf(row_data[i] - max_val);
        row_data[i] = v;
        sum += v;
    }
    // Warp + block reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < num_warps; i++) total += s_buf[i];
        s_sum = total;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_data[i] *= inv_sum;
}

static __global__ void reshape_to_multihead_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int T, int H, int Dh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * Dh;
    if (idx >= total) return;
    int h = idx / (T * Dh);
    int td = idx % (T * Dh);
    int t = td / Dh;
    int d = td % Dh;
    output[idx] = input[t * H * Dh + h * Dh + d];
}

static __global__ void reshape_from_multihead_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int T, int H, int Dh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * Dh;
    if (idx >= total) return;
    int h = idx / (T * Dh);
    int td = idx % (T * Dh);
    int t = td / Dh;
    int d = td % Dh;
    output[t * H * Dh + h * Dh + d] = input[idx];
}

static __global__ void split_reshape_qkv_kernel(
        const float* __restrict__ qkv,
        float* __restrict__ Q,
        float* __restrict__ K,
        float* __restrict__ V,
        int T, int H, int Dh) {
    int D = H * Dh;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * Dh;
    if (idx >= total) return;
    int h = idx / (T * Dh);
    int t = (idx / Dh) % T;
    int d = idx % Dh;
    int src = t * 3 * D + h * Dh + d;
    Q[idx] = qkv[src];
    K[idx] = qkv[src + D];
    V[idx] = qkv[src + 2 * D];
}

static __global__ void gru_gate_bias_kernel(
        const float* __restrict__ x_flat,  // [T, D] — LN output (NOT Q-projected)
        const float* __restrict__ grep_w,  // [8, Dh]
        const float* __restrict__ grep_b,  // [8]
        const float* __restrict__ grep_a,  // [H] (flattened from [1, H, 1, 1])
        const float* __restrict__ pos_bias, // [H, T, T]
        float* __restrict__ attn,          // [H, T, T] — add gated bias in-place
        int T, int H, int Dh) {
    // Each block handles one (h, t) pair
    int ht = blockIdx.x;
    if (ht >= H * T) return;
    int h = ht / T;
    int t = ht % T;

    // Step 1: Compute grep_linear(x[t, h*Dh : (h+1)*Dh]) → [8]
    // x_flat[T, D] layout: element [t][h*Dh + d] = x_flat[t*D + h*Dh + d]
    int D = H * Dh;
    const float* x_ptr = x_flat + t * D + h * Dh;  // [Dh]
    float linear_out[8];
    for (int o = 0; o < 8; o++) {
        float sum = grep_b[o];
        for (int d = 0; d < Dh; d++) {
            sum += grep_w[o * Dh + d] * x_ptr[d];
        }
        linear_out[o] = sum;
    }

    // Step 2: Reshape [8] → [2, 4], sum over last dim → [2], sigmoid
    float s0 = linear_out[0] + linear_out[1] + linear_out[2] + linear_out[3];
    float s1 = linear_out[4] + linear_out[5] + linear_out[6] + linear_out[7];
    float ga = 1.0f / (1.0f + __expf(-s0));  // sigmoid(s0)
    float gb = 1.0f / (1.0f + __expf(-s1));  // sigmoid(s1)

    // Step 3: Gate computation
    float a_h = grep_a[h];
    float gate = ga * (gb * a_h - 1.0f) + 2.0f;

    // Step 4: Scale position bias and add to attention scores
    // attn[h, t, :] += gate * pos_bias[h, t, :]
    float* attn_row = attn + h * T * T + t * T;
    const float* bias_row = pos_bias + h * T * T + t * T;
    for (int t2 = threadIdx.x; t2 < T; t2 += blockDim.x) {
        attn_row[t2] += gate * bias_row[t2];
    }
}

static __global__ void vector_add_kernel(float* __restrict__ y,
                                   const float* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] += x[idx];
}

static __global__ void softmax_1d_kernel(const float* __restrict__ raw_weights,
                                   float* __restrict__ norm_weights, int n) {
    float max_val = -1e30f;
    for (int i = 0; i < n; i++)
        if (raw_weights[i] > max_val) max_val = raw_weights[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = expf(raw_weights[i] - max_val);
        norm_weights[i] = v;
        sum += v;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++)
        norm_weights[i] *= inv;
}

static __global__ void weighted_sum_kernel(const float* __restrict__ hidden_states,
                                     const float* __restrict__ norm_weights,
                                     float* __restrict__ output,
                                     int num_layers, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D) return;
    float sum = 0.0f;
    for (int l = 0; l < num_layers; l++)
        sum += norm_weights[l] * hidden_states[l * T * D + idx];
    output[idx] = sum;
}

static __global__ void utterance_mvn_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int T, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float mean = 0.0f;
    for (int t = 0; t < T; t++)
        mean += input[t * D + d];
    mean /= T;
    for (int t = 0; t < T; t++)
        output[t * D + d] = input[t * D + d] - mean;
}

static __global__ void batch_norm_1d_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      const float* __restrict__ running_mean,
                                      const float* __restrict__ running_var,
                                      int C, int T, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    float x = input[idx];
    float inv_std = rsqrtf(running_var[c] + eps);
    output[idx] = (x - running_mean[c]) * inv_std * weight[c] + bias[c];
}

static __global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (data[idx] < 0.0f) data[idx] = 0.0f;
}

static __global__ void im2col_1d_kernel(const float* __restrict__ input,
                                  float* __restrict__ cols,
                                  int C_in, int T_in, int K, int pad, int dilation,
                                  int T_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_in * K * T_out) return;
    int t_out = idx % T_out;
    int ck = idx / T_out;
    int c = ck / K;
    int k = ck % K;
    int t_in = t_out + k * dilation - pad;
    cols[idx] = (t_in >= 0 && t_in < T_in) ? input[c * T_in + t_in] : 0.0f;
}

static __global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = 1.0f / (1.0f + __expf(-data[idx]));
}

static __global__ void broadcast_mul_kernel(float* __restrict__ y,
                                      const float* __restrict__ x,
                                      int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    y[idx] *= x[c];
}

static __global__ void adaptive_avg_pool_1d_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++)
        sum += input[c * T + t];
    output[c] = sum / T;
}

static __global__ void pool_global_stats_kernel(const float* __restrict__ x,
                                          float* __restrict__ output,
                                          int C, int T) {
    int c = blockIdx.x;
    if (c >= C) return;
    // Compute mean and variance for this channel
    float sum = 0.0f, sum2 = 0.0f;
    for (int t = 0; t < T; t++) {
        float v = x[c * T + t];
        sum += v;
        sum2 += v * v;
    }
    float mean = sum / T;
    float var = sum2 / T - mean * mean;
    float std = sqrtf(fmaxf(var, 1.192092896e-07f));  // eps = FLT_EPSILON

    // Write to output: [x; mean_expanded; std_expanded]
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        output[c * T + t] = x[c * T + t];            // x itself
        output[(C + c) * T + t] = mean;               // mean repeated
        output[(2 * C + c) * T + t] = std;            // std repeated
    }
}

static __global__ void weighted_stats_kernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       float* __restrict__ mu,
                                       float* __restrict__ sg,
                                       int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m = 0.0f, s2 = 0.0f;
    for (int t = 0; t < T; t++) {
        float xi = x[c * T + t];
        float wi = w[c * T + t];
        m += xi * wi;
        s2 += xi * xi * wi;
    }
    mu[c] = m;
    sg[c] = sqrtf(fmaxf(s2 - m * m, 1e-4f));
}

static __global__ void l2_normalize_kernel(float* data, int n) {
    float sum2 = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum2 += data[i] * data[i];
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    // Cross-warp reduction via shared memory
    __shared__ float s_warp_sums[8];  // up to 256 threads = 8 warps
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) s_warp_sums[warp_id] = sum2;
    __syncthreads();
    // Thread 0 reduces all warp partial sums
    __shared__ float s_norm;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) total += s_warp_sums[w];
        s_norm = rsqrtf(total + 1e-12f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        data[i] *= s_norm;
}

} // namespace deusridet
