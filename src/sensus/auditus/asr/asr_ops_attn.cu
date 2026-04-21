/**
 * @file src/sensus/auditus/asr/asr_ops_attn.cu
 * @philosophical_role Attention, cache, and sampling ops for Qwen3-ASR — the back half of asr_ops.cu, split out under R1. Where asr_ops.cu holds norms and elementwise activations, this file holds the ops that *attend*: split-K contiguous decode, cuBLAS prefill softmax, KV cache writes, and the logit-side ops (argmax, EOS suppression, repetition penalty).
 * @serves Auditus asr_decoder (causal GQA), asr_encoder prefill path, and ASR sampling.
 */
// asr_ops_attn.cu — peer TU of asr_ops.cu under R1 800-line hard cap for .cu.

#include "asr_ops.h"
#include <cmath>

namespace deusridet {
namespace asr_ops {

// TU-local BF16 converters (duplicated from asr_ops.cu — __device__ __forceinline__,
// cheaper than exposing via header under R1 hard cap).
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
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
