/**
 * @file src/machina/forward_kernels.cuh
 * @philosophical_role
 *   TU-local CUDA kernels shared across the forward.cu peer split. Every
 *   kernel is `static __global__` so each TU gets a private instantiation
 *   (CUDA RDC is off).
 * @serves
 *   forward.cu, forward_prefill.cu, forward_profile.cu.
 */
#pragma once

#include "forward.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace deusridet {

static __global__ void conv1d_batch_silu_kernel(
    __half* __restrict__ qkv,           // [M, stride] in/out (in-place)
    __half* __restrict__ conv_state,    // [conv_dim, kernel-1]
    const __half* __restrict__ conv_weight, // [conv_dim, kernel]
    int M, int conv_dim, int kernel_size, int stride)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int km1 = kernel_size - 1;

    // Load conv state and weights into registers
    float st[3], wt[4];
    #pragma unroll
    for (int j = 0; j < 3; j++)
        st[j] = __half2float(conv_state[ch * km1 + j]);
    #pragma unroll
    for (int j = 0; j < 4; j++)
        wt[j] = __half2float(conv_weight[ch * kernel_size + j]);

    // Process all M tokens sequentially for this channel
    for (int t = 0; t < M; t++) {
        float x_val = __half2float(qkv[t * stride + ch]);
        float acc = st[0]*wt[0] + st[1]*wt[1] + st[2]*wt[2] + x_val*wt[3];
        float silu_out = acc / (1.0f + __expf(-acc));
        qkv[t * stride + ch] = __float2half(silu_out);
        st[0] = st[1]; st[1] = st[2]; st[2] = x_val;
    }

    // Save final conv state
    conv_state[ch * km1 + 0] = __float2half(st[0]);
    conv_state[ch * km1 + 1] = __float2half(st[1]);
    conv_state[ch * km1 + 2] = __float2half(st[2]);
}

static __global__ void __launch_bounds__(128, 2)
deltanet_fused_head_kernel(
    const __half* __restrict__ qkv_batch,   // [M, qkv_stride] post-conv1d
    const float* __restrict__ A_log,        // [48]
    const __half* __restrict__ dt_bias,     // [48]
    float* __restrict__ dn_state,           // [48, 128, 128] recurrent state
    __half* __restrict__ output,            // [M, value_dim=6144]
    int M, int num_k_heads, int num_v_heads,
    int k_dim, int v_dim, int key_dim, int qkv_stride,
    int a_offset, int b_offset, int ab_stride, float eps)
{
    const int head = blockIdx.x;           // 0..47 (value head)
    const int tid = threadIdx.x;           // 0..127
    const int src_head = head / (num_v_heads / num_k_heads);  // Q/K source head
    const float q_scale = rsqrtf((float)k_dim);  // 1/sqrt(128)

    // Pre-load A_log and dt_bias for this head
    const float a_log_h = A_log[head];
    const float dt_bias_h = __half2float(dt_bias[head]);

    // State pointer for this head: S[128][128], thread tid owns column tid
    float* S = dn_state + (size_t)head * 128 * 128;

    // Shared memory: [128] for q + [128] for k + [4] for warp reduction
    extern __shared__ float smem[];
    float* sq = smem;              // [128]
    float* sk = sq + 128;          // [128]
    float* warp_sums = sk + 128;   // [4]

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // === Load entire state column into registers (one-time global read) ===
    float S_col[128];
    #pragma unroll
    for (int i = 0; i < 128; i++)
        S_col[i] = S[i * 128 + tid];

    for (int t = 0; t < M; t++) {
        // Pointers to this token's projected data
        const __half* qkv_t = qkv_batch + t * qkv_stride;
        const __half* q_src = qkv_t + src_head * k_dim;
        const __half* k_src = qkv_t + key_dim + src_head * k_dim;
        const __half* v_src = qkv_t + 2 * key_dim + head * v_dim;

        // === L2 normalize Q (with scale) ===
        float q_val = __half2float(q_src[tid]);
        float q_sq = q_val * q_val;
        for (int offset = 16; offset > 0; offset >>= 1)
            q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
        if (lane_id == 0) warp_sums[warp_id] = q_sq;
        __syncthreads();
        if (warp_id == 0) {
            q_sq = (lane_id < 4) ? warp_sums[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
        }
        if (tid == 0) warp_sums[0] = rsqrtf(q_sq + eps) * q_scale;
        __syncthreads();
        sq[tid] = q_val * warp_sums[0];

        // === L2 normalize K ===
        float k_val = __half2float(k_src[tid]);
        float k_sq = k_val * k_val;
        for (int offset = 16; offset > 0; offset >>= 1)
            k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
        if (lane_id == 0) warp_sums[warp_id] = k_sq;
        __syncthreads();
        if (warp_id == 0) {
            k_sq = (lane_id < 4) ? warp_sums[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
        }
        if (tid == 0) warp_sums[0] = rsqrtf(k_sq + eps);
        __syncthreads();
        sk[tid] = k_val * warp_sums[0];

        // === Compute g and beta for this head + token ===
        // a/b accessed via offset within qkv buffer (merged) or separate (ab_stride)
        float a_val = __half2float(qkv_batch[t * ab_stride + a_offset + head]);
        float b_val = __half2float(qkv_batch[t * ab_stride + b_offset + head]);
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float g_scalar = __expf(-__expf(a_log_h) * logf(1.0f + __expf(a_val + dt_bias_h)));
        float beta_scalar = 1.0f / (1.0f + __expf(-b_val));

        // === Load V into register ===
        float v_j = __half2float(v_src[tid]);

        // === Recurrent update (register-cached — zero intermediate global access) ===
        __syncthreads();  // ensure sk/sq are ready

        // Pass 1: decay + kv_mem dot product (all in registers + SMEM)
        float mem = 0.0f;
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            S_col[i] *= g_scalar;
            mem += S_col[i] * sk[i];
        }

        float delta = (v_j - mem) * beta_scalar;

        // Pass 2: rank-1 update + output dot product (all in registers + SMEM)
        float out_j = 0.0f;
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            S_col[i] += sk[i] * delta;
            out_j += S_col[i] * sq[i];
        }

        output[t * num_v_heads * v_dim + head * v_dim + tid] = __float2half(out_j);

        __syncthreads();  // sync before next token overwrites sq/sk
    }

    // === Store state column back (one-time global write) ===
    #pragma unroll
    for (int i = 0; i < 128; i++)
        S[i * 128 + tid] = S_col[i];
}

static __global__ void split_q_gate_batch_kernel(const __half* __restrict__ qg,
                                          __half* __restrict__ q,
                                          __half* __restrict__ gate,
                                          int M, int num_heads, int head_dim) {
    int token = blockIdx.y;
    int h = blockIdx.x;
    if (token >= M || h >= num_heads) return;
    int interleaved_stride = num_heads * head_dim * 2;
    int out_stride = num_heads * head_dim;
    const __half* src = qg + token * interleaved_stride + h * head_dim * 2;
    __half* q_dst = q + token * out_stride + h * head_dim;
    __half* g_dst = gate + token * out_stride + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_dst[d] = src[d];
        g_dst[d] = src[head_dim + d];
    }
}

static __global__ void kv_cache_write_batch_kernel(
    const __half* __restrict__ src_k,  // [M, num_kv_heads, head_dim]
    const __half* __restrict__ src_v,  // [M, num_kv_heads, head_dim]
    __half* __restrict__ k_cache,      // [num_kv_heads, max_kv_len, head_dim]
    __half* __restrict__ v_cache,      // [num_kv_heads, max_kv_len, head_dim]
    int pos_start, int M,
    int max_kv_len, int head_dim, int num_kv_heads)
{
    int h = blockIdx.x;   // kv head
    int t = blockIdx.y;   // token index within batch
    if (h >= num_kv_heads || t >= M) return;
    int pos = pos_start + t;
    size_t src_offset = (size_t)t * num_kv_heads * head_dim + h * head_dim;
    size_t cache_offset = (size_t)h * max_kv_len * head_dim + (size_t)pos * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        k_cache[cache_offset + d] = src_k[src_offset + d];
        v_cache[cache_offset + d] = src_v[src_offset + d];
    }
}

static __global__ void rope_batch_kernel(__half* __restrict__ q,
                                  __half* __restrict__ k,
                                  int num_q_heads, int num_kv_heads,
                                  int head_dim, int rotary_dim,
                                  int pos_start, int M, float theta) {
    int token = blockIdx.y;
    int head = blockIdx.x;
    int pair = threadIdx.x;
    int num_pairs = rotary_dim / 2;
    if (pair >= num_pairs || token >= M) return;

    int pos = pos_start + token;
    float freq = 1.0f / powf(theta, (2.0f * pair) / (float)rotary_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int q_stride = num_q_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    if (head < num_q_heads) {
        int idx0 = token * q_stride + head * head_dim + pair;
        int idx1 = token * q_stride + head * head_dim + pair + num_pairs;
        float x0 = __half2float(q[idx0]);
        float x1 = __half2float(q[idx1]);
        q[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
        q[idx1] = __float2half(x1 * cos_a + x0 * sin_a);
    }

    if (head < num_kv_heads) {
        int idx0 = token * kv_stride + head * head_dim + pair;
        int idx1 = token * kv_stride + head * head_dim + pair + num_pairs;
        float x0 = __half2float(k[idx0]);
        float x1 = __half2float(k[idx1]);
        k[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
        k[idx1] = __float2half(x1 * cos_a + x0 * sin_a);
    }
}

static __global__ void prefill_attention_kernel(
    const __half* __restrict__ Q,        // [M, num_attn_heads, head_dim]
    const __half* __restrict__ k_cache,  // [num_kv_heads, max_kv_len, head_dim]
    const __half* __restrict__ v_cache,  // [num_kv_heads, max_kv_len, head_dim]
    __half* __restrict__ out,            // [M, num_attn_heads, head_dim]
    int pos_start, int M,
    int max_kv_len, int head_dim, int num_kv_groups,
    float scale)
{
    const int h = blockIdx.x;       // query head index
    const int token = blockIdx.y;   // token within batch
    const int kv_h = h / num_kv_groups;
    const int d = threadIdx.x;
    const int seq_len = pos_start + token + 1;   // causal: attend to 0..pos_start+token

    int q_stride = gridDim.x * head_dim;  // num_attn_heads * head_dim
    const __half* q = Q + token * q_stride + h * head_dim;
    const __half* k = k_cache + (size_t)kv_h * max_kv_len * head_dim;
    const __half* v = v_cache + (size_t)kv_h * max_kv_len * head_dim;

    const float q_val = (d < head_dim) ? __half2float(q[d]) : 0.0f;
    const int warp_id = d / 32;
    const int lane = d % 32;
    const int num_warps = (head_dim + 31) / 32;

    __shared__ float warp_sums[8];
    __shared__ float s_broadcast;

    // Online softmax
    float m_running = -FLT_MAX;
    float l_running = 0.0f;
    float o_val = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        float k_val = (d < head_dim) ? __half2float(k[t * head_dim + d]) : 0.0f;
        float qk = q_val * k_val;

        #pragma unroll
        for (int off = 16; off > 0; off /= 2)
            qk += __shfl_xor_sync(0xFFFFFFFF, qk, off);

        if (lane == 0) warp_sums[warp_id] = qk;
        __syncthreads();

        if (d == 0) {
            float sum = 0;
            for (int w = 0; w < num_warps; w++) sum += warp_sums[w];
            s_broadcast = sum * scale;
        }
        __syncthreads();

        float s_t = s_broadcast;
        float m_new = fmaxf(m_running, s_t);
        float alpha = __expf(m_running - m_new);
        float p_t = __expf(s_t - m_new);

        float v_val = (d < head_dim) ? __half2float(v[t * head_dim + d]) : 0.0f;
        o_val = o_val * alpha + v_val * p_t;
        l_running = l_running * alpha + p_t;
        m_running = m_new;
    }

    if (d < head_dim && l_running > 0.0f) {
        int out_idx = token * q_stride + h * head_dim + d;
        out[out_idx] = __float2half(o_val / l_running);
    }
}

} // namespace deusridet
