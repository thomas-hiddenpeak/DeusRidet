/**
 * @file src/sensus/auditus/asr/asr_ops.cu
 * @philosophical_role
 *   ASR-specific operator kernels — ops the general Machina toolbox does not provide because they are ASR-shaped (relative position bias, conformer bits). Kept here to avoid polluting Machina.
 * @serves
 *   Auditus asr_encoder and asr_decoder only.
 */
// asr_ops.cu — ASR-specific CUDA operator implementations
//
// Adapted from qwen35-orin (src/plugins/asr/audio_ops.cu): complete set of
// CUDA kernels for Qwen3-ASR encoder/decoder inference on SM87.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "asr_ops.h"
#include <cmath>

namespace deusridet {
namespace asr_ops {

// ============================================================================
// Helper
// ============================================================================

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// RMSNorm (plain weight)
// ============================================================================

__global__ void rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float eps, int hidden_size)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (size_t)token * hidden_size;
    __nv_bfloat16* o_row = out + (size_t)token * hidden_size;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) sum_sq = shared[threadIdx.x];
    else sum_sq = 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) s_rsqrt = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();

    float scale = s_rsqrt;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * scale;
        o_row[i] = float_to_bf16(bf16_to_float(weight[i]) * v);
    }
}

void invoke_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    float eps, int num_tokens, int hidden_size, cudaStream_t stream) {
    int block = std::min(hidden_size, 1024);
    rmsnorm_kernel<<<num_tokens, block, 0, stream>>>(out, x, weight, eps, hidden_size);
}

// ============================================================================
// LayerNorm (with bias)
// ============================================================================

__global__ void layernorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float eps, int hidden_size)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (size_t)token * hidden_size;
    __nv_bfloat16* o_row = out + (size_t)token * hidden_size;

    // Pass 1: mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        sum += bf16_to_float(x_row[i]);
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) sum = shared[threadIdx.x];
    else sum = 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / hidden_size;
    __syncthreads();

    // Pass 2: variance
    float mean = s_mean;
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) var_sum = shared[threadIdx.x];
    else var_sum = 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) s_inv_std = rsqrtf(var_sum / hidden_size + eps);
    __syncthreads();

    float inv_std = s_inv_std;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (bf16_to_float(x_row[i]) - mean) * inv_std;
        o_row[i] = float_to_bf16(v * bf16_to_float(weight[i]) + bf16_to_float(bias[i]));
    }
}

void invoke_layernorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                      float eps, int num_tokens, int hidden_size, cudaStream_t stream) {
    int block = std::min(hidden_size, 1024);
    layernorm_kernel<<<num_tokens, block, 0, stream>>>(out, x, weight, bias, eps, hidden_size);
}

// ============================================================================
// Per-head RMSNorm (plain weight)
// ============================================================================

__global__ void per_head_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float eps, int num_tokens, int num_heads, int head_dim)
{
    int token_head = blockIdx.x;
    int token = token_head / num_heads;
    int head = token_head % num_heads;
    const __nv_bfloat16* x_ptr = x + ((size_t)token * num_heads + head) * head_dim;
    __nv_bfloat16* o_ptr = out + ((size_t)token * num_heads + head) * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = bf16_to_float(x_ptr[i]);
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) sum_sq = shared[threadIdx.x];
    else sum_sq = 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) s_rsqrt = rsqrtf(sum_sq / head_dim + eps);
    __syncthreads();

    float scale = s_rsqrt;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        o_ptr[i] = float_to_bf16(bf16_to_float(weight[i]) * bf16_to_float(x_ptr[i]) * scale);
    }
}

void invoke_per_head_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                              const __nv_bfloat16* weight,
                              float eps, int num_tokens, int num_heads, int head_dim,
                              cudaStream_t stream) {
    int total_heads = num_tokens * num_heads;
    int block = std::min(head_dim, 256);
    per_head_rmsnorm_kernel<<<total_heads, block, 0, stream>>>(
        out, x, weight, eps, num_tokens, num_heads, head_dim);
}

// ============================================================================
// SwiGLU
// ============================================================================

__global__ void swiglu_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = bf16_to_float(gate[idx]);
    float u = bf16_to_float(up[idx]);
    out[idx] = float_to_bf16(g / (1.0f + __expf(-g)) * u);
}

void invoke_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   int num_tokens, int intermediate_size, cudaStream_t stream) {
    int total = num_tokens * intermediate_size;
    swiglu_kernel<<<(total + 255) / 256, 256, 0, stream>>>(out, gate, up, total);
}

// ============================================================================
// GELU
// ============================================================================

__global__ void gelu_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float v = bf16_to_float(x[idx]);
    out[idx] = float_to_bf16(v * 0.5f * (1.0f + erff(v * 0.7071067811865476f)));
}

void invoke_gelu(__nv_bfloat16* out, const __nv_bfloat16* x,
                 int num_elements, cudaStream_t stream) {
    gelu_kernel<<<(num_elements + 255) / 256, 256, 0, stream>>>(out, x, num_elements);
}

// ============================================================================
// Sinusoidal Positional Embedding
// ============================================================================

__global__ void sinusoidal_pe_kernel(
    __nv_bfloat16* __restrict__ pe_out,
    int max_positions, int d_model, float log_timescale_base)
{
    int pos = blockIdx.x;
    int half = d_model / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float log_ts = -log_timescale_base * i / (half - 1);
        float angle = pos * __expf(log_ts);
        pe_out[(size_t)pos * d_model + i] = float_to_bf16(__sinf(angle));
        pe_out[(size_t)pos * d_model + half + i] = float_to_bf16(__cosf(angle));
    }
}

void compute_sinusoidal_pe(__nv_bfloat16* pe_out, int max_positions, int d_model,
                           float max_timescale, cudaStream_t stream) {
    float log_ts = logf(max_timescale);
    int block = std::min(d_model / 2, 256);
    sinusoidal_pe_kernel<<<max_positions, block, 0, stream>>>(
        pe_out, max_positions, d_model, log_ts);
}

// ============================================================================
// Add PE
// ============================================================================

__global__ void add_pe_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pe_table,
    int seq_len, int hidden_size, int pos_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * hidden_size) return;
    int t = idx / hidden_size, d = idx % hidden_size;
    hidden[idx] = float_to_bf16(
        bf16_to_float(hidden[idx]) + bf16_to_float(pe_table[(t + pos_offset) * hidden_size + d]));
}

__global__ void add_pe_chunked_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pe_table,
    int total_tokens, int hidden_size, int chunk_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * hidden_size) return;
    int t = idx / hidden_size, d = idx % hidden_size;
    int pos_in_chunk = t % chunk_len;
    hidden[idx] = float_to_bf16(
        bf16_to_float(hidden[idx]) + bf16_to_float(pe_table[pos_in_chunk * hidden_size + d]));
}

void invoke_add_pe(__nv_bfloat16* hidden_states, const __nv_bfloat16* pe_table,
                   int seq_len, int hidden_size, int pos_offset, cudaStream_t stream) {
    int total = seq_len * hidden_size;
    add_pe_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        hidden_states, pe_table, seq_len, hidden_size, pos_offset);
}

void invoke_add_pe_chunked(__nv_bfloat16* hidden_states, const __nv_bfloat16* pe_table,
                           int total_tokens, int hidden_size, int chunk_len, cudaStream_t stream) {
    int total = total_tokens * hidden_size;
    add_pe_chunked_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        hidden_states, pe_table, total_tokens, hidden_size, chunk_len);
}

// ============================================================================
// MRoPE (half-rotation, interleaved sections)
// ============================================================================

__global__ void mrope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int* __restrict__ pos_ids,
    int num_tokens, int num_q_heads, int num_kv_heads,
    int head_dim, int s0, int s1, int s2, float theta)
{
    int token = blockIdx.x;
    int d = blockIdx.y;
    if (d >= head_dim / 2) return;

    int dim_idx = 0;
    if ((d % 3 == 1) && (d < s1 * 3)) dim_idx = 1;
    if ((d % 3 == 2) && (d < s2 * 3)) dim_idx = 2;

    int pos = pos_ids[dim_idx * num_tokens + token];
    float freq = 1.0f / powf(theta, (float)(d * 2) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = __cosf(angle), sin_a = __sinf(angle);
    int d_hi = d + head_dim / 2;

    for (int h = threadIdx.x; h < num_q_heads; h += blockDim.x) {
        size_t base = ((size_t)token * num_q_heads + h) * head_dim;
        float x_lo = bf16_to_float(q[base + d]);
        float x_hi = bf16_to_float(q[base + d_hi]);
        q[base + d]    = float_to_bf16(x_lo * cos_a - x_hi * sin_a);
        q[base + d_hi] = float_to_bf16(x_hi * cos_a + x_lo * sin_a);
    }
    for (int h = threadIdx.x; h < num_kv_heads; h += blockDim.x) {
        size_t base = ((size_t)token * num_kv_heads + h) * head_dim;
        float x_lo = bf16_to_float(k[base + d]);
        float x_hi = bf16_to_float(k[base + d_hi]);
        k[base + d]    = float_to_bf16(x_lo * cos_a - x_hi * sin_a);
        k[base + d_hi] = float_to_bf16(x_hi * cos_a + x_lo * sin_a);
    }
}

void invoke_mrope(__nv_bfloat16* q, __nv_bfloat16* k, const int* pos_ids,
                  int num_tokens, int num_q_heads, int num_kv_heads,
                  int head_dim, int s0, int s1, int s2, float theta, cudaStream_t stream) {
    dim3 grid(num_tokens, head_dim / 2);
    int block = std::min(std::max(num_q_heads, num_kv_heads), 256);
    mrope_kernel<<<grid, block, 0, stream>>>(
        q, k, pos_ids, num_tokens, num_q_heads, num_kv_heads, head_dim, s0, s1, s2, theta);
}

// ============================================================================
// Fused QK RMSNorm + MRoPE
// ============================================================================

__global__ void fused_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const int* __restrict__ pos_ids,
    float eps, int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    int s0, int s1, int s2, float theta)
{
    int idx = blockIdx.x;
    int token = blockIdx.y;
    int tid = threadIdx.x;
    bool is_q = (idx < num_q_heads);
    int head = is_q ? idx : (idx - num_q_heads);

    __nv_bfloat16* data = is_q
        ? q + ((size_t)token * num_q_heads + head) * head_dim
        : k + ((size_t)token * num_kv_heads + head) * head_dim;
    const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;

    extern __shared__ float smem[];

    // RMSNorm
    float val = (tid < head_dim) ? bf16_to_float(data[tid]) : 0.0f;
    float sum_sq = val * val;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    __shared__ float s_shared[8];
    int wid = tid / 32, lid = tid % 32;
    if (lid == 0) s_shared[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        sum_sq = (lid < (blockDim.x + 31) / 32) ? s_shared[lid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_rsqrt;
    if (tid == 0) s_rsqrt = rsqrtf(sum_sq / head_dim + eps);
    __syncthreads();

    float normed = val * s_rsqrt * bf16_to_float(norm_w[tid]);
    if (tid < head_dim) smem[tid] = normed;
    __syncthreads();

    // RoPE (half rotation)
    int half_dim = head_dim / 2;
    if (tid < half_dim) {
        int d = tid, d_hi = d + half_dim;
        int dim_idx = 0;
        if ((d % 3 == 1) && (d < s1 * 3)) dim_idx = 1;
        if ((d % 3 == 2) && (d < s2 * 3)) dim_idx = 2;
        int pos = pos_ids[dim_idx * num_tokens + token];
        float freq = 1.0f / powf(theta, (float)(d * 2) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a, sin_a;
        sincosf(angle, &sin_a, &cos_a);
        float x_lo = smem[d], x_hi = smem[d_hi];
        data[d]    = float_to_bf16(x_lo * cos_a - x_hi * sin_a);
        data[d_hi] = float_to_bf16(x_hi * cos_a + x_lo * sin_a);
    }
}

void invoke_fused_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const int* pos_ids, float eps, int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    int s0, int s1, int s2, float theta, cudaStream_t stream) {
    dim3 grid(num_q_heads + num_kv_heads, num_tokens);
    int block = head_dim;
    int smem_bytes = head_dim * sizeof(float);
    fused_qk_norm_rope_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, q_norm_w, k_norm_w, pos_ids, eps,
        num_tokens, num_q_heads, num_kv_heads, head_dim, s0, s1, s2, theta);
}

// ============================================================================
// Bidirectional MHA (ASR Encoder)
// ============================================================================

__global__ void bidirectional_mha_kernel(
    __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_seqlens,
    int num_segments, int num_heads, int head_dim, float scale)
{
    int global_token = blockIdx.x;
    int head = blockIdx.z;
    int tid = threadIdx.x, bdim = blockDim.x;
    int warp_id = tid / 32, lane_id = tid % 32, num_warps = bdim / 32;

    int seg_start = 0, seg_end = 0;
    for (int s = 0; s < num_segments; s++) {
        if (global_token < cu_seqlens[s + 1]) {
            seg_start = cu_seqlens[s]; seg_end = cu_seqlens[s + 1]; break;
        }
    }
    int seg_len = seg_end - seg_start;

    const __nv_bfloat16* q_ptr = q + ((size_t)global_token * num_heads + head) * head_dim;
    __nv_bfloat16* o_ptr = attn_out + ((size_t)global_token * num_heads + head) * head_dim;

    extern __shared__ float smem[];
    float* scores = smem;
    __shared__ float reduce_buf[8];

    // QK^T scores
    float local_max = -1e20f;
    for (int j = tid; j < seg_len; j += bdim) {
        const __nv_bfloat16* k_ptr = k + ((size_t)(seg_start + j) * num_heads + head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += bf16_to_float(q_ptr[d]) * bf16_to_float(k_ptr[d]);
        dot *= scale;
        scores[j] = dot;
        local_max = fmaxf(local_max, dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    if (lane_id == 0) reduce_buf[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) { float m = reduce_buf[0]; for (int w = 1; w < num_warps; w++) m = fmaxf(m, reduce_buf[w]); reduce_buf[0] = m; }
    __syncthreads();
    float max_score = reduce_buf[0];

    // Softmax
    float local_sum = 0.0f;
    for (int j = tid; j < seg_len; j += bdim) {
        float e = exp2f((scores[j] - max_score) * 1.4426950408889634f);
        scores[j] = e; local_sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) reduce_buf[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) { float s = reduce_buf[0]; for (int w = 1; w < num_warps; w++) s += reduce_buf[w]; reduce_buf[0] = s; }
    __syncthreads();
    float inv_sum = 1.0f / reduce_buf[0];

    // Weighted V
    for (int d = tid; d < head_dim; d += bdim) {
        float acc = 0.0f;
        for (int j = 0; j < seg_len; j++) {
            const __nv_bfloat16* v_ptr = v + ((size_t)(seg_start + j) * num_heads + head) * head_dim;
            acc += scores[j] * bf16_to_float(v_ptr[d]);
        }
        o_ptr[d] = float_to_bf16(acc * inv_sum);
    }
}

void invoke_bidirectional_mha(
    __nv_bfloat16* attn_out, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, int total_tokens, int num_heads, int head_dim,
    const int* cu_seqlens, int num_segments, cudaStream_t stream) {
    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid(total_tokens, 1, num_heads);
    int block = head_dim;
    int max_seg_len = (total_tokens < 1536) ? total_tokens : 1536;
    bidirectional_mha_kernel<<<grid, block, max_seg_len * sizeof(float), stream>>>(
        attn_out, q, k, v, cu_seqlens, num_segments, num_heads, head_dim, scale);
}

} // namespace asr_ops
} // namespace deusridet
