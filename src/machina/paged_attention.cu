// paged_attention.cu — Paged KV Cache attention kernels
//
// Paged variants of GQA decode/prefill attention and KV cache write.
// KV blocks are addressed through a block table for non-contiguous storage.
//
// Pool layout (per FA layer L, physical block B):
//   K base = pool + L * layer_stride + B * block_stride
//   V base = K base + kv_plane
//   where kv_plane = NUM_KV_HEADS * block_size * HEAD_DIM (in elements)
//         block_stride = 2 * kv_plane
//         layer_stride = max_blocks * block_stride
//
// Target: SM87 (Jetson AGX Orin)

#include "paged_attention.h"
#include "layer.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace deusridet {

// ============================================================================
// Pool addressing helpers (device-side)
// ============================================================================

// Compute element offset to K data for (fa_layer, phys_block, kv_head, pos_in_block, dim)
__device__ __forceinline__
size_t paged_k_offset(int fa_layer, int phys_block, int kv_head,
                      int pos_in_block, int dim,
                      int max_blocks, int block_size, int head_dim, int num_kv_heads) {
    size_t kv_plane     = (size_t)num_kv_heads * block_size * head_dim;
    size_t block_stride = 2 * kv_plane;
    size_t layer_stride = (size_t)max_blocks * block_stride;
    return fa_layer * layer_stride
         + phys_block * block_stride
         + (size_t)kv_head * block_size * head_dim
         + (size_t)pos_in_block * head_dim
         + dim;
}

// Same for V (offset by kv_plane from K)
__device__ __forceinline__
size_t paged_v_offset(int fa_layer, int phys_block, int kv_head,
                      int pos_in_block, int dim,
                      int max_blocks, int block_size, int head_dim, int num_kv_heads) {
    size_t kv_plane     = (size_t)num_kv_heads * block_size * head_dim;
    size_t block_stride = 2 * kv_plane;
    size_t layer_stride = (size_t)max_blocks * block_stride;
    return fa_layer * layer_stride
         + phys_block * block_stride
         + kv_plane  // V starts after K
         + (size_t)kv_head * block_size * head_dim
         + (size_t)pos_in_block * head_dim
         + dim;
}

// ============================================================================
// Paged KV Cache write — single token (decode)
//
// Grid = NUM_KV_HEADS, Block = 256 threads
// Each block handles one KV head, threads iterate over HEAD_DIM.
// ============================================================================

__global__ void paged_kv_cache_write_kernel(
    const __half* __restrict__ src_k,     // [NUM_KV_HEADS, HEAD_DIM]
    const __half* __restrict__ src_v,     // [NUM_KV_HEADS, HEAD_DIM]
    __half* __restrict__ kv_pool,
    const int* __restrict__ d_block_table,
    int fa_layer_idx,
    const int* __restrict__ d_pos,
    int max_blocks, int block_size, int head_dim, int num_kv_heads)
{
    int kv_h = blockIdx.x;
    if (kv_h >= num_kv_heads) return;

    int pos = *d_pos;
    int logical_block = pos / block_size;
    int pos_in_block  = pos % block_size;
    int phys_block    = d_block_table[logical_block];

    size_t k_off = paged_k_offset(fa_layer_idx, phys_block, kv_h,
                                  pos_in_block, 0,
                                  max_blocks, block_size, head_dim, num_kv_heads);
    size_t v_off = paged_v_offset(fa_layer_idx, phys_block, kv_h,
                                  pos_in_block, 0,
                                  max_blocks, block_size, head_dim, num_kv_heads);

    size_t src_off = (size_t)kv_h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        kv_pool[k_off + d] = src_k[src_off + d];
        kv_pool[v_off + d] = src_v[src_off + d];
    }
}

void paged_kv_cache_write(
    const __half* src_k, const __half* src_v,
    __half* kv_pool, const int* d_block_table,
    int fa_layer_idx, const int* d_pos,
    int max_phys_blocks, int block_size, cudaStream_t stream)
{
    using MC = ModelConfig;
    paged_kv_cache_write_kernel<<<MC::NUM_KV_HEADS, 256, 0, stream>>>(
        src_k, src_v, kv_pool, d_block_table,
        fa_layer_idx, d_pos,
        max_phys_blocks, block_size, MC::HEAD_DIM, MC::NUM_KV_HEADS);
}

// ============================================================================
// Paged KV Cache write — batched (prefill)
//
// Grid = (NUM_KV_HEADS, M), Block = 256
// ============================================================================

__global__ void paged_kv_cache_write_batch_kernel(
    const __half* __restrict__ src_k,     // [M, NUM_KV_HEADS, HEAD_DIM]
    const __half* __restrict__ src_v,     // [M, NUM_KV_HEADS, HEAD_DIM]
    __half* __restrict__ kv_pool,
    const int* __restrict__ d_block_table,
    int fa_layer_idx,
    int pos_start, int M,
    int max_blocks, int block_size, int head_dim, int num_kv_heads)
{
    int kv_h  = blockIdx.x;
    int token = blockIdx.y;
    if (kv_h >= num_kv_heads || token >= M) return;

    int pos = pos_start + token;
    int logical_block = pos / block_size;
    int pos_in_block  = pos % block_size;
    int phys_block    = d_block_table[logical_block];

    size_t k_off = paged_k_offset(fa_layer_idx, phys_block, kv_h,
                                  pos_in_block, 0,
                                  max_blocks, block_size, head_dim, num_kv_heads);
    size_t v_off = paged_v_offset(fa_layer_idx, phys_block, kv_h,
                                  pos_in_block, 0,
                                  max_blocks, block_size, head_dim, num_kv_heads);

    size_t src_off = ((size_t)token * num_kv_heads + kv_h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        kv_pool[k_off + d] = src_k[src_off + d];
        kv_pool[v_off + d] = src_v[src_off + d];
    }
}

void paged_kv_cache_write_batch(
    const __half* src_k, const __half* src_v,
    __half* kv_pool, const int* d_block_table,
    int fa_layer_idx, int pos_start, int M,
    int max_phys_blocks, int block_size, cudaStream_t stream)
{
    using MC = ModelConfig;
    dim3 grid(MC::NUM_KV_HEADS, M);
    paged_kv_cache_write_batch_kernel<<<grid, 256, 0, stream>>>(
        src_k, src_v, kv_pool, d_block_table,
        fa_layer_idx, pos_start, M,
        max_phys_blocks, block_size, MC::HEAD_DIM, MC::NUM_KV_HEADS);
}

// ============================================================================
// Paged GQA Decode Attention — Flash-Decoding with online softmax
//
// Same algorithm as gqa_decode_attention_kernel in forward.cu, but iterates
// through block table instead of contiguous KV cache.
//
// Grid = NUM_ATTN_HEADS (24), Block = HEAD_DIM (256)
// Shared memory: 36 bytes (8 warp sums + 1 broadcast float)
// ============================================================================

__global__ void paged_gqa_decode_attention_kernel(
    const __half* __restrict__ Q,           // [NUM_ATTN_HEADS, HEAD_DIM]
    const __half* __restrict__ kv_pool,
    const int* __restrict__ d_block_table,  // [num_logical_blocks]
    __half* __restrict__ out,               // [NUM_ATTN_HEADS, HEAD_DIM]
    int fa_layer_idx,
    int seq_len,             // total KV sequence length
    int max_blocks,          // physical block capacity
    int block_size,
    int head_dim,
    int num_kv_heads,
    int num_kv_groups,       // ATTN_HEADS / KV_HEADS
    float scale)
{
    const int h    = blockIdx.x;              // query head index
    const int kv_h = h / num_kv_groups;       // KV head index (GQA)
    const int d    = threadIdx.x;             // dimension index

    // Shared memory for cross-warp dot product reduction
    __shared__ float warp_sums[8];
    __shared__ float s_broadcast;

    const float q_val = (d < head_dim) ? __half2float(Q[h * head_dim + d]) : 0.0f;
    const int warp_id   = d / 32;
    const int lane      = d % 32;
    const int num_warps = (head_dim + 31) / 32;

    // Online softmax accumulators
    float m_running = -FLT_MAX;
    float l_running = 0.0f;
    float o_val     = 0.0f;

    // Precompute pool strides
    size_t kv_plane     = (size_t)num_kv_heads * block_size * head_dim;
    size_t block_stride = 2 * kv_plane;
    size_t layer_stride = (size_t)max_blocks * block_stride;
    size_t layer_base   = fa_layer_idx * layer_stride;

    int num_logical_blocks = (seq_len + block_size - 1) / block_size;

    for (int b = 0; b < num_logical_blocks; b++) {
        int phys_block = d_block_table[b];
        int block_len  = (b == num_logical_blocks - 1)
                       ? (seq_len - b * block_size)
                       : block_size;

        // Base addresses for this physical block's K and V
        size_t blk_base = layer_base + phys_block * block_stride;
        const __half* k_base = kv_pool + blk_base
                             + (size_t)kv_h * block_size * head_dim;
        const __half* v_base = kv_pool + blk_base + kv_plane
                             + (size_t)kv_h * block_size * head_dim;

        for (int t = 0; t < block_len; t++) {
            // QK^T: dot product via warp + cross-warp reduction
            float k_v = (d < head_dim) ? __half2float(k_base[t * head_dim + d]) : 0.0f;
            float qk = q_val * k_v;

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

            // Online softmax update
            float m_new = fmaxf(m_running, s_t);
            float alpha = __expf(m_running - m_new);
            float p_t   = __expf(s_t - m_new);

            float v_v = (d < head_dim) ? __half2float(v_base[t * head_dim + d]) : 0.0f;
            o_val     = o_val * alpha + v_v * p_t;
            l_running = l_running * alpha + p_t;
            m_running = m_new;
        }
    }

    // Final normalization
    if (d < head_dim && l_running > 0.0f) {
        out[h * head_dim + d] = __float2half(o_val / l_running);
    }
}

void paged_gqa_decode_attention(
    const __half* Q, __half* kv_pool, const int* d_block_table,
    __half* out, int fa_layer_idx, int seq_len,
    int max_phys_blocks, int block_size, float scale, cudaStream_t stream)
{
    using MC = ModelConfig;
    paged_gqa_decode_attention_kernel<<<MC::NUM_ATTN_HEADS, MC::HEAD_DIM, 0, stream>>>(
        Q, kv_pool, d_block_table, out,
        fa_layer_idx, seq_len,
        max_phys_blocks, block_size,
        MC::HEAD_DIM, MC::NUM_KV_HEADS, MC::NUM_KV_GROUPS, scale);
}

// ============================================================================
// Helper: split interleaved Q+Gate — local copy of forward.cu's kernel
// ============================================================================

static __global__ void paged_split_q_gate(const __half* __restrict__ qg,
                                          __half* __restrict__ q,
                                          __half* __restrict__ gate,
                                          int num_heads, int head_dim) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q[h * head_dim + d]    = qg[h * head_dim * 2 + d];
        gate[h * head_dim + d] = qg[h * head_dim * 2 + head_dim + d];
    }
}

// ============================================================================
// Paged Full Attention forward — single-token decode
//
// Same flow as full_attention_forward in forward.cu:
//   1. Q proj → split Q + Gate
//   2. K proj, V proj
//   3. Per-head Q/K RMSNorm
//   4. Partial RoPE
//   5. Paged KV write
//   6. Paged GQA decode attention
//   7. Attention gate
//   8. O proj
// ============================================================================

void full_attention_forward_paged(
    const __half* x,
    const FullAttentionWeights& attn,
    __half* kv_pool,
    const int* d_block_table,
    int fa_layer_idx,
    int pos, int seq_len,
    int max_phys_blocks, int block_size,
    InferenceState& state,
    cudaStream_t stream)
{
    using MC = ModelConfig;

    // 1. Q projection [5120→12288]
    fp16_gemv(x, attn.fp16_q.weight, state.q_buf,
              attn.fp16_q.in_features, attn.fp16_q.out_features, stream);

    // Deinterleave Q and Gate
    // Reuse dn_qkv for Q and mlp_gate for Gate (not used during FA)
    __half* q_ptr    = state.dn_qkv;
    __half* gate_ptr = state.mlp_gate;

    paged_split_q_gate<<<MC::NUM_ATTN_HEADS, 256, 0, stream>>>(
        state.q_buf, q_ptr, gate_ptr, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);

    // 2. K and V projections
    fp16_gemv(x, attn.fp16_k.weight, state.dn_z,
              attn.fp16_k.in_features, attn.fp16_k.out_features, stream);
    fp16_gemv(x, attn.fp16_v.weight, state.dn_z + MC::KV_PROJ_DIM,
              attn.fp16_v.in_features, attn.fp16_v.out_features, stream);
    __half* k_buf = state.dn_z;
    __half* v_buf = state.dn_z + MC::NUM_KV_HEADS * MC::HEAD_DIM;

    // 3. Per-head Q/K RMSNorm
    head_norm(q_ptr, attn.q_norm, q_ptr,
              MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(k_buf, attn.k_norm, k_buf,
              MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);

    // 4. Partial RoPE
    apply_rope(q_ptr, k_buf,
               MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS, MC::HEAD_DIM,
               MC::ROTARY_DIM, state.d_pos, MC::ROPE_THETA, stream);

    // 5. Paged KV write
    paged_kv_cache_write(k_buf, v_buf, kv_pool, d_block_table,
                         fa_layer_idx, state.d_pos,
                         max_phys_blocks, block_size, stream);

    // 6. Paged GQA decode attention
    float scale = 1.0f / sqrtf((float)MC::HEAD_DIM);
    paged_gqa_decode_attention(q_ptr, kv_pool, d_block_table,
                               state.attn_out, fa_layer_idx, seq_len,
                               max_phys_blocks, block_size, scale, stream);

    // 7. Attention gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, gate_ptr, state.attn_out,
                 MC::ATTN_OUT_DIM, stream);

    // 8. O projection: attn_out[6144] → norm_out[5120]
    fp16_gemv(state.attn_out, attn.fp16_o.weight, state.norm_out,
              attn.fp16_o.in_features, attn.fp16_o.out_features, stream);
}

// ============================================================================
// Batched split Q+Gate for paged prefill
//
// Q projection output is interleaved: [M, num_heads, head_dim*2] where
// each head has [Q_dim, Gate_dim]. Deinterleave into separate Q and Gate.
// Grid: (num_heads, M), Block: 256
// ============================================================================

static __global__ void paged_split_q_gate_batch_kernel(
    const __half* __restrict__ qg,    // [M, num_heads * head_dim * 2]
    __half* __restrict__ q,           // [M, num_heads * head_dim]
    __half* __restrict__ gate,        // [M, num_heads * head_dim]
    int M, int num_heads, int head_dim)
{
    int h = blockIdx.x;
    int token = blockIdx.y;
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

// ============================================================================
// Batched RoPE for paged prefill — M tokens at consecutive positions
//
// Grid: (max(num_q_heads, num_kv_heads), M), Block: rotary_dim/2
// ============================================================================

static __global__ void paged_rope_batch_kernel(
    __half* __restrict__ q,    // [M, num_q_heads, head_dim]
    __half* __restrict__ k,    // [M, num_kv_heads, head_dim]
    int num_q_heads, int num_kv_heads,
    int head_dim, int rotary_dim,
    int pos_start, int M, float theta)
{
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

// ============================================================================
// Paged prefill attention kernel — causal attention with block table
//
// Combines the causal per-query-token attention of prefill_attention_kernel
// with the block-table iteration of paged_gqa_decode_attention_kernel.
//
// Each block handles one (query_head, query_token) pair. The causal mask
// ensures token t only attends to positions 0..pos_start+t.
//
// Grid: (num_attn_heads, M), Block: head_dim (256)
// ============================================================================

static __global__ void paged_prefill_attention_kernel(
    const __half* __restrict__ Q,           // [M, num_attn_heads, head_dim]
    const __half* __restrict__ kv_pool,
    const int* __restrict__ d_block_table,
    __half* __restrict__ out,               // [M, num_attn_heads, head_dim]
    int fa_layer_idx,
    int pos_start, int M,
    int max_blocks, int block_size,
    int head_dim, int num_kv_heads, int num_kv_groups,
    float scale)
{
    const int h     = blockIdx.x;
    const int token = blockIdx.y;
    const int kv_h  = h / num_kv_groups;
    const int d     = threadIdx.x;
    const int causal_len = pos_start + token + 1;

    int q_stride = gridDim.x * head_dim;
    const float q_val = (d < head_dim)
        ? __half2float(Q[token * q_stride + h * head_dim + d]) : 0.0f;

    const int warp_id   = d / 32;
    const int lane      = d % 32;
    const int num_warps = (head_dim + 31) / 32;

    __shared__ float warp_sums[8];
    __shared__ float s_broadcast;

    // Pool addressing precomputation
    size_t kv_plane     = (size_t)num_kv_heads * block_size * head_dim;
    size_t blk_stride   = 2 * kv_plane;
    size_t layer_stride = (size_t)max_blocks * blk_stride;
    size_t layer_base   = fa_layer_idx * layer_stride;

    // Online softmax accumulators
    float m_running = -FLT_MAX;
    float l_running = 0.0f;
    float o_val     = 0.0f;

    int num_logical_blocks = (causal_len + block_size - 1) / block_size;

    for (int b = 0; b < num_logical_blocks; b++) {
        int phys_block = d_block_table[b];
        int block_start = b * block_size;
        int block_end   = block_start + block_size;
        if (block_end > causal_len) block_end = causal_len;
        int block_len = block_end - block_start;

        size_t blk_base = layer_base + phys_block * blk_stride;
        const __half* k_base = kv_pool + blk_base
                             + (size_t)kv_h * block_size * head_dim;
        const __half* v_base = kv_pool + blk_base + kv_plane
                             + (size_t)kv_h * block_size * head_dim;

        for (int t = 0; t < block_len; t++) {
            float k_v = (d < head_dim)
                ? __half2float(k_base[t * head_dim + d]) : 0.0f;
            float qk = q_val * k_v;

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
            float p_t   = __expf(s_t - m_new);

            float v_v = (d < head_dim)
                ? __half2float(v_base[t * head_dim + d]) : 0.0f;
            o_val     = o_val * alpha + v_v * p_t;
            l_running = l_running * alpha + p_t;
            m_running = m_new;
        }
    }

    if (d < head_dim && l_running > 0.0f) {
        out[token * q_stride + h * head_dim + d] = __float2half(o_val / l_running);
    }
}

// ============================================================================
// Paged Full Attention forward — batched prefill (complete implementation)
//
// Full pipeline: Q/K/V projections → split Q+Gate → head norms → RoPE →
// paged KV write → paged causal attention → sigmoid gate → O projection
// ============================================================================

void full_attention_forward_paged_prefill(
    const __half* x,
    const FullAttentionWeights& attn,
    __half* kv_pool,
    const int* d_block_table,
    int fa_layer_idx,
    int pos_start, int M,
    int seq_len,
    int max_phys_blocks, int block_size,
    InferenceState& state,
    cudaStream_t stream)
{
    using MC = ModelConfig;

    // 1. Q projection [M, 5120] → q_buf[M, 12288] (interleaved Q+Gate)
    linear_forward(x, attn.fp16_q, state.q_buf, M, stream);

    // 2. K projection [M, 5120] → kv_buf[M, 1024]
    linear_forward(x, attn.fp16_k, state.kv_buf, M, stream);

    // 3. V projection [M, 5120] → dn_z[M, 1024]
    linear_forward(x, attn.fp16_v, state.dn_z, M, stream);

    __half* k_batch = state.kv_buf;
    __half* v_batch = state.dn_z;

    // 4. Split Q and Gate from interleaved Q projection
    __half* q_sep = state.dn_qkv;    // [M, 6144] within [max_seq, 10496]
    __half* gate  = state.mlp_gate;   // [M, 6144] within [max_seq, 17408]
    {
        dim3 grid(MC::NUM_ATTN_HEADS, M);
        paged_split_q_gate_batch_kernel<<<grid, 256, 0, stream>>>(
            state.q_buf, q_sep, gate, M, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);
    }

    // 5. Per-head Q/K RMSNorm
    head_norm(q_sep, attn.q_norm, q_sep,
              M * MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(k_batch, attn.k_norm, k_batch,
              M * MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);

    // 6. Batched RoPE for M tokens at consecutive positions
    {
        int max_heads = (MC::NUM_ATTN_HEADS > MC::NUM_KV_HEADS)
                        ? MC::NUM_ATTN_HEADS : MC::NUM_KV_HEADS;
        dim3 grid(max_heads, M);
        paged_rope_batch_kernel<<<grid, MC::ROTARY_DIM / 2, 0, stream>>>(
            q_sep, k_batch,
            MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS,
            MC::HEAD_DIM, MC::ROTARY_DIM,
            pos_start, M, MC::ROPE_THETA);
    }

    // 7. Paged KV write for all M tokens
    paged_kv_cache_write_batch(k_batch, v_batch, kv_pool, d_block_table,
                               fa_layer_idx, pos_start, M,
                               max_phys_blocks, block_size, stream);

    // 8. Paged prefill attention (causal, iterating through block table)
    {
        float scale = 1.0f / sqrtf((float)MC::HEAD_DIM);
        dim3 grid(MC::NUM_ATTN_HEADS, M);
        paged_prefill_attention_kernel<<<grid, MC::HEAD_DIM, 0, stream>>>(
            q_sep, kv_pool, d_block_table, state.attn_out,
            fa_layer_idx, pos_start, M,
            max_phys_blocks, block_size,
            MC::HEAD_DIM, MC::NUM_KV_HEADS, MC::NUM_KV_GROUPS, scale);
    }

    // 9. Attention gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, gate, state.attn_out,
                 (size_t)M * MC::ATTN_OUT_DIM, stream);

    // 10. O projection [M, 6144] → norm_out[M, 5120]
    linear_forward(state.attn_out, attn.fp16_o, state.norm_out, M, stream);
}

} // namespace deusridet
