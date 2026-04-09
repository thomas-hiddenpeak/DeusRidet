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

// ============================================================================
// Split-K Contiguous Decode Attention (T=1)
// ============================================================================

static constexpr int ATTN_PARTITION_SIZE = 128;

__global__ void contiguous_split_k_decode_kernel(
    float* __restrict__ partial_out, float* __restrict__ partial_m, float* __restrict__ partial_l,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float sm_scale, int num_partitions)
{
    const int q_head = blockIdx.x, part_id = blockIdx.y, tid = threadIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int part_start = part_id * ATTN_PARTITION_SIZE;
    int part_end = min(part_start + ATTN_PARTITION_SIZE, seq_len);
    const int out_idx = q_head * num_partitions + part_id;

    if (part_start >= seq_len) {
        partial_out[out_idx * head_dim + tid] = 0.0f;
        if (tid == 0) { partial_m[out_idx] = -1e20f; partial_l[out_idx] = 0.0f; }
        return;
    }

    float q_val = bf16_to_float(q[q_head * head_dim + tid]) * sm_scale;
    const int lane_id = tid & 31, warp_id = tid >> 5;
    const int part_len = part_end - part_start;

    extern __shared__ float smem[];
    float* scores = smem;
    float* warp_buf = smem + ATTN_PARTITION_SIZE;

    // QK^T
    for (int i = 0; i < part_len; i++) {
        const __nv_bfloat16* k_ptr = k_cache + ((size_t)(part_start + i) * num_kv_heads + kv_head) * head_dim;
        float partial = q_val * bf16_to_float(k_ptr[tid]);
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane_id == 0) warp_buf[warp_id] = partial;
        __syncthreads();
        if (tid == 0) { float dot = warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3]; scores[i] = dot; }
        __syncthreads();
    }

    // Softmax + V accumulation
    float local_m = -1e20f, local_l = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < part_len; i++) local_m = fmaxf(local_m, scores[i]);
        for (int i = 0; i < part_len; i++) {
            float e = exp2f((scores[i] - local_m) * 1.4426950408889634f);
            scores[i] = e; local_l += e;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < part_len; i++) {
        const __nv_bfloat16* v_ptr = v_cache + ((size_t)(part_start + i) * num_kv_heads + kv_head) * head_dim;
        acc += scores[i] * bf16_to_float(v_ptr[tid]);
    }
    partial_out[out_idx * head_dim + tid] = acc;
    if (tid == 0) { partial_m[out_idx] = local_m; partial_l[out_idx] = local_l; }
}

__global__ void contiguous_split_k_merge_kernel(
    __nv_bfloat16* __restrict__ attn_out,
    const float* __restrict__ partial_out, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, int head_dim, int num_partitions)
{
    const int q_head = blockIdx.x, tid = threadIdx.x;
    float m = -1e20f, l = 0.0f, acc = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        int idx = q_head * num_partitions + p;
        float pm = partial_m[idx], pl = partial_l[idx];
        float pv = partial_out[idx * head_dim + tid];
        if (pm > m) {
            float scale = exp2f((m - pm) * 1.4426950408889634f);
            acc = acc * scale + pv; l = l * scale + pl; m = pm;
        } else {
            float scale = exp2f((pm - m) * 1.4426950408889634f);
            acc += pv * scale; l += pl * scale;
        }
    }
    attn_out[q_head * head_dim + tid] = float_to_bf16((l > 0.0f) ? acc / l : 0.0f);
}

void invoke_causal_gqa_decode(
    __nv_bfloat16* attn_out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int batch_size, int num_q_heads, int num_kv_heads, int head_dim,
    int current_seq_len, cudaStream_t stream,
    float* attn_workspace, int attn_max_partitions)
{
    float sm_scale = 1.0f / sqrtf((float)head_dim);
    int num_parts = (current_seq_len + ATTN_PARTITION_SIZE - 1) / ATTN_PARTITION_SIZE;
    if (num_parts < 1) num_parts = 1;

    if (attn_workspace && attn_max_partitions > 0) {
        float* p_out = attn_workspace;
        float* p_m = p_out + num_q_heads * attn_max_partitions * head_dim;
        float* p_l = p_m + num_q_heads * attn_max_partitions;
        size_t smem_bytes = (ATTN_PARTITION_SIZE + 4) * sizeof(float);
        contiguous_split_k_decode_kernel<<<dim3(num_q_heads, num_parts), head_dim, smem_bytes, stream>>>(
            p_out, p_m, p_l, q, k_cache, v_cache,
            num_q_heads, num_kv_heads, head_dim, current_seq_len, sm_scale, num_parts);
        contiguous_split_k_merge_kernel<<<num_q_heads, head_dim, 0, stream>>>(
            attn_out, p_out, p_m, p_l, head_dim, num_parts);
    }
}

// ============================================================================
// Causal GQA Prefill via cuBLAS
// ============================================================================

__global__ void causal_softmax_bf16_kernel(
    __nv_bfloat16* __restrict__ scores, int seq_len)
{
    int row = blockIdx.x, tid = threadIdx.x;
    int attend_len = row + 1;
    __nv_bfloat16* row_ptr = scores + (size_t)row * seq_len;

    float local_max = -1e20f;
    for (int j = tid; j < attend_len; j += blockDim.x)
        local_max = fmaxf(local_max, bf16_to_float(row_ptr[j]));
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    __shared__ float reduce_buf[8];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) reduce_buf[warp] = local_max;
    __syncthreads();
    if (tid == 0) { float m = reduce_buf[0]; for (int w = 1; w < (blockDim.x + 31) / 32; w++) m = fmaxf(m, reduce_buf[w]); reduce_buf[0] = m; }
    __syncthreads();
    float max_val = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = tid; j < attend_len; j += blockDim.x) {
        float e = exp2f((bf16_to_float(row_ptr[j]) - max_val) * 1.4426950408889634f);
        row_ptr[j] = float_to_bf16(e); local_sum += e;
    }
    for (int j = attend_len + tid; j < seq_len; j += blockDim.x)
        row_ptr[j] = float_to_bf16(0.0f);
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (lane == 0) reduce_buf[warp] = local_sum;
    __syncthreads();
    if (tid == 0) { float s = reduce_buf[0]; for (int w = 1; w < (blockDim.x + 31) / 32; w++) s += reduce_buf[w]; reduce_buf[0] = (s > 0.0f) ? (1.0f / s) : 0.0f; }
    __syncthreads();
    float inv_sum = reduce_buf[0];

    for (int j = tid; j < attend_len; j += blockDim.x)
        row_ptr[j] = float_to_bf16(bf16_to_float(row_ptr[j]) * inv_sum);
}

void invoke_causal_gqa_prefill_cublas(
    __nv_bfloat16* attn_out, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* attn_score_buf, int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    cublasHandle_t handle, cudaStream_t stream)
{
    float sm_scale = 1.0f / sqrtf((float)head_dim);
    float zero = 0.0f, one = 1.0f;
    int gqa_ratio = num_q_heads / num_kv_heads;
    int q_stride = num_q_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;
    cublasSetStream(handle, stream);

    for (int qh = 0; qh < num_q_heads; qh++) {
        int kvh = qh / gqa_ratio;
        const __nv_bfloat16* Q_h = q + qh * head_dim;
        const __nv_bfloat16* K_kv = k + kvh * head_dim;
        const __nv_bfloat16* V_kv = v + kvh * head_dim;
        __nv_bfloat16* O_h = attn_out + qh * head_dim;

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_dim, &sm_scale,
            K_kv, CUDA_R_16BF, kv_stride, Q_h, CUDA_R_16BF, q_stride,
            &zero, attn_score_buf, CUDA_R_16BF, seq_len,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        int softmax_threads = std::min(256, ((seq_len + 31) / 32) * 32);
        causal_softmax_bf16_kernel<<<seq_len, softmax_threads, 0, stream>>>(attn_score_buf, seq_len);

        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len, &one,
            V_kv, CUDA_R_16BF, kv_stride, attn_score_buf, CUDA_R_16BF, seq_len,
            &zero, O_h, CUDA_R_16BF, q_stride,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
}

// ============================================================================
// Embedding Lookup
// ============================================================================

__global__ void embedding_lookup_kernel(
    __nv_bfloat16* __restrict__ out, const int* __restrict__ ids,
    const __nv_bfloat16* __restrict__ table, int num_tokens, int hidden_size)
{
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    const __nv_bfloat16* row = table + (size_t)ids[token] * hidden_size;
    __nv_bfloat16* out_row = out + (size_t)token * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) out_row[i] = row[i];
}

void invoke_embedding_lookup(__nv_bfloat16* out, const int* ids, const __nv_bfloat16* table,
                              int num_tokens, int hidden_size, cudaStream_t stream) {
    embedding_lookup_kernel<<<num_tokens, std::min(hidden_size, 256), 0, stream>>>(
        out, ids, table, num_tokens, hidden_size);
}

// ============================================================================
// Residual Add
// ============================================================================

__global__ void add_residual_kernel(__nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    a[idx] = float_to_bf16(bf16_to_float(a[idx]) + bf16_to_float(b[idx]));
}

void invoke_add_residual(__nv_bfloat16* a, const __nv_bfloat16* b,
                         int num_elements, cudaStream_t stream) {
    add_residual_kernel<<<(num_elements + 255) / 256, 256, 0, stream>>>(a, b, num_elements);
}

// ============================================================================
// BF16 Clamp
// ============================================================================

__global__ void bf16_clamp_kernel(__nv_bfloat16* __restrict__ x,
    int num_elements, float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float v = bf16_to_float(x[idx]);
    x[idx] = float_to_bf16(fminf(fmaxf(v, min_val), max_val));
}

void invoke_bf16_clamp(__nv_bfloat16* x, int num_elements,
                       float min_val, float max_val, cudaStream_t stream) {
    bf16_clamp_kernel<<<(num_elements + 255) / 256, 256, 0, stream>>>(x, num_elements, min_val, max_val);
}

// ============================================================================
// Write KV Cache
// ============================================================================

__global__ void write_kv_cache_kernel(
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    int start_pos, int num_tokens, int num_kv_heads, int head_dim)
{
    int kv_size = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * kv_size) return;
    int t = idx / kv_size, kv_idx = idx % kv_size;
    size_t cache_offset = ((size_t)(start_pos + t)) * kv_size + kv_idx;
    k_cache[cache_offset] = k[(size_t)t * kv_size + kv_idx];
    v_cache[cache_offset] = v[(size_t)t * kv_size + kv_idx];
}

void invoke_write_kv_cache(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                            const __nv_bfloat16* k, const __nv_bfloat16* v,
                            int start_pos, int num_tokens, int num_kv_heads,
                            int head_dim, cudaStream_t stream) {
    int total = num_tokens * num_kv_heads * head_dim;
    write_kv_cache_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        k_cache, v_cache, k, v, start_pos, num_tokens, num_kv_heads, head_dim);
}

// ============================================================================
// GPU Argmax
// ============================================================================

__global__ void argmax_kernel(const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ result_idx, int n)
{
    int tid = threadIdx.x;
    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }
    __shared__ float s_vals[32]; __shared__ int s_idxs[32];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) { s_vals[warp_id] = best_val; s_idxs[warp_id] = best_idx; }
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        best_val = (lane < num_warps) ? s_vals[lane] : -1e30f;
        best_idx = (lane < num_warps) ? s_idxs[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
        }
        if (lane == 0) *result_idx = best_idx;
    }
}

void invoke_argmax(const __nv_bfloat16* logits, int* result_idx, int n, cudaStream_t stream) {
    argmax_kernel<<<1, 256, 0, stream>>>(logits, result_idx, n);
}

// ============================================================================
// EOS Suppression
// ============================================================================

__global__ void suppress_eos_kernel(__nv_bfloat16* __restrict__ logits, int eos_id1, int eos_id2) {
    logits[eos_id1] = __float2bfloat16(-1e30f);
    logits[eos_id2] = __float2bfloat16(-1e30f);
}

void invoke_suppress_eos(__nv_bfloat16* logits, int eos_id1, int eos_id2, cudaStream_t stream) {
    suppress_eos_kernel<<<1, 1, 0, stream>>>(logits, eos_id1, eos_id2);
}

// ============================================================================
// F32 → BF16
// ============================================================================

__global__ void f32_to_bf16_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

void invoke_f32_to_bf16(__nv_bfloat16* out, const float* in, int n, cudaStream_t stream) {
    f32_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, in, n);
}

// ============================================================================
// Repetition Penalty
// ============================================================================

__global__ void repetition_penalty_kernel(
    __nv_bfloat16* __restrict__ logits, const int* __restrict__ token_ids,
    int num_tokens, float penalty)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens) {
        int token_id = token_ids[idx];
        float logit = __bfloat162float(logits[token_id]);
        logits[token_id] = __float2bfloat16(logit > 0.0f ? logit / penalty : logit * penalty);
    }
}

void invoke_repetition_penalty(__nv_bfloat16* logits, const int* token_ids,
                               int num_tokens, float penalty, cudaStream_t stream) {
    if (num_tokens <= 0 || penalty == 1.0f) return;
    repetition_penalty_kernel<<<(num_tokens + 255) / 256, 256, 0, stream>>>(
        logits, token_ids, num_tokens, penalty);
}

} // namespace asr_ops
} // namespace deusridet
