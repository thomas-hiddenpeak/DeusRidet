/**
 * @file src/machina/forward.cu
 * @philosophical_role
 *   The forward pass itself — the sequence of layer calls that turns a token ID into a next-token distribution. If Conscientia is the loop, forward.cu is the body of one iteration of the loop.
 * @serves
 *   Conscientia per-tick decode; Actus diagnostic verbs (test_forward, bench_prefill, profile_forward).
 */
// forward.cu — Qwen3.5 forward pass implementation
//
// Single-token decode path. Each function assumes M=1.
// Target: SM87 (Jetson AGX Orin)

#include "forward.h"
#include "forward_kernels.cuh"
#include "layer.h"
#include "gptq.h"
#include "gptq_gemm_v2.h"
#include "marlin.h"
#include "fp16_gemm.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

namespace deusridet {

// ============================================================================
// SwiGLU MLP forward (M=1)
//
// gate_out = gate_proj(x)   — GPTQ Int4 or FP16
// up_out   = up_proj(x)     — GPTQ Int4 or FP16
// gate_out = silu(gate_out) * up_out
// mlp_out  = down_proj(gate_out) — GPTQ Int4 or FP16
// ============================================================================

void mlp_forward(const __half* x, const MLPWeights& mlp,
                 __half* residual,
                 InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    if (MC::MLP_IS_GPTQ) {
        // GPTQ-Int4 path
        gptq_gemm_v2(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                     1, mlp.gate_proj.K, mlp.gate_proj.N, stream);
        gptq_gemm_v2(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                     1, mlp.up_proj.K, mlp.up_proj.N, stream);

        silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                 MC::INTERMEDIATE_SIZE, stream);

        if (residual) {
            gptq_gemm_v2_add(state.mlp_gate, mlp.down_proj.qweight, residual,
                             mlp.down_proj.scales, 1, mlp.down_proj.K, mlp.down_proj.N, stream);
        } else {
            gptq_gemm_v2(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down,
                         mlp.down_proj.scales, 1, mlp.down_proj.K, mlp.down_proj.N, stream);
        }
    } else {
        // FP16 path (unquantized models)
        fp16_gemv(x, mlp.fp16_gate_proj.weight, state.mlp_gate,
                  mlp.fp16_gate_proj.in_features, mlp.fp16_gate_proj.out_features, stream);
        fp16_gemv(x, mlp.fp16_up_proj.weight, state.mlp_up,
                  mlp.fp16_up_proj.in_features, mlp.fp16_up_proj.out_features, stream);

        silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                 MC::INTERMEDIATE_SIZE, stream);

        if (residual) {
            fp16_gemv(state.mlp_gate, mlp.fp16_down_proj.weight, state.mlp_down,
                      mlp.fp16_down_proj.in_features, mlp.fp16_down_proj.out_features, stream);
            elementwise_add(residual, state.mlp_down, residual,
                            MC::HIDDEN_SIZE, stream);
        } else {
            fp16_gemv(state.mlp_gate, mlp.fp16_down_proj.weight, state.mlp_down,
                      mlp.fp16_down_proj.in_features, mlp.fp16_down_proj.out_features, stream);
        }
    }
}

// ============================================================================
// GQA Decode Attention — fused QK^T + softmax + V@scores
//
// Replaces per-KV-head cuBLAS calls (21ms overhead for tiny matrices) with a
// single kernel launch. Grid = num_attn_heads (24), Block = head_dim (256).
//
// Each block handles one query head:
//   1. QK^T: score[t] = Q[h]·K[kv_h][t] / sqrt(d) — dimension-parallel dot product
//   2. Softmax over scores[0..seq_len-1]
//   3. V@scores: out[d] = sum_t score[t] * V[kv_h][t][d]
//
// Shared memory: 8 floats (warp reduction) + seq_len_kv floats (scores)
// ============================================================================

// ============================================================================
// GQA Decode Attention — Flash-Decoding style with online softmax
//
// Inspired by Flash Attention v2 and FlashInfer's decode attention:
// - Online softmax: no score materialization, O(1) shared memory
// - Per-position processing: QK^T → online softmax update → V accumulation
// - Single pass: K and V loaded once per position (cache-friendly)
//
// Versus the previous implementation:
// - No O(seq_len) shared memory allocation for scores
// - No single-threaded softmax bottleneck
// - No separate V@scores pass
//
// Grid = num_attn_heads (24), Block = head_dim (256)
// Shared memory: 36 bytes (8 warp sums + 1 broadcast) — constant
// ============================================================================

__global__ void gqa_decode_attention_kernel(
    const __half* __restrict__ Q,           // [num_attn_heads, head_dim]
    const __half* __restrict__ k_cache,     // [num_kv_heads, max_kv_len, head_dim]
    const __half* __restrict__ v_cache,     // [num_kv_heads, max_kv_len, head_dim]
    __half* __restrict__ out,               // [num_attn_heads, head_dim]
    const int* __restrict__ d_pos,          // device pointer to current position
    int max_kv_len, int head_dim, int num_kv_groups,
    float scale)
{
    const int h = blockIdx.x;                     // query head index
    const int kv_h = h / num_kv_groups;           // KV head index (GQA)
    const int d = threadIdx.x;                    // dimension index

    const int seq_len_kv = *d_pos + 1;           // read from device memory

    // Static shared memory for cross-warp dot product reduction
    __shared__ float warp_sums[8];    // 8 warps for 256 threads
    __shared__ float s_broadcast;     // broadcast score to all threads

    const __half* q = Q + h * head_dim;
    const __half* k = k_cache + (size_t)kv_h * max_kv_len * head_dim;
    const __half* v = v_cache + (size_t)kv_h * max_kv_len * head_dim;

    const float q_val = (d < head_dim) ? __half2float(q[d]) : 0.0f;
    const int warp_id = d / 32;
    const int lane    = d % 32;
    const int num_warps = (head_dim + 31) / 32;

    // Online softmax accumulators (Flash Attention v2 style)
    // m: running max of scores, l: running sum of exp(score - m)
    // o: unnormalized output accumulator (normalized at end)
    float m_running = -FLT_MAX;
    float l_running = 0.0f;
    float o_val = 0.0f;

    for (int t = 0; t < seq_len_kv; t++) {
        // QK^T: dot product via warp + cross-warp reduction
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

        // Online softmax update
        float m_new = fmaxf(m_running, s_t);
        float alpha = expf(m_running - m_new);   // rescale old accumulators
        float p_t   = expf(s_t - m_new);          // unnormalized weight for this position

        // Accumulate (normalize once at the end)
        float v_val = (d < head_dim) ? __half2float(v[t * head_dim + d]) : 0.0f;
        o_val     = o_val * alpha + v_val * p_t;
        l_running = l_running * alpha + p_t;
        m_running = m_new;
    }

    // Final normalization
    if (d < head_dim && l_running > 0.0f) {
        out[h * head_dim + d] = __float2half(o_val / l_running);
    }
}

// ============================================================================
// Full Attention helper kernels
// ============================================================================

// Convert FP32 scores to FP16 in-place (overwrite into a separate FP16 buffer)
__global__ void f32_to_f16_kernel(const float* __restrict__ src,
                                  __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Deinterleave q_proj output: [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]
//   → separate Q[num_heads, head_dim] and Gate[num_heads, head_dim]
// blockIdx.x = head index, threadIdx.x iterates over head_dim
__global__ void split_q_gate_kernel(const __half* __restrict__ qg,
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

// Copy K and V into cache for all heads in one kernel
// src_k[num_kv_heads, head_dim], src_v[num_kv_heads, head_dim]
// k_cache[num_kv_heads, max_kv_len, head_dim]
// v_cache[num_kv_heads, max_kv_len, head_dim]
// d_pos: device pointer to current position
__global__ void kv_cache_write_kernel(const __half* __restrict__ src_k,
                                      const __half* __restrict__ src_v,
                                      __half* __restrict__ k_cache,
                                      __half* __restrict__ v_cache,
                                      const int* __restrict__ d_pos,
                                      int max_kv_len, int head_dim,
                                      int num_kv_heads) {
    // blockIdx.x = head, threadIdx.x iterates over head_dim
    int h = blockIdx.x;
    if (h >= num_kv_heads) return;
    int pos = *d_pos;
    size_t cache_offset = (size_t)h * max_kv_len * head_dim + (size_t)pos * head_dim;
    size_t src_offset = (size_t)h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        k_cache[cache_offset + d] = src_k[src_offset + d];
        v_cache[cache_offset + d] = src_v[src_offset + d];
    }
}

// ============================================================================
// Full Attention GQA forward (single-token decode, M=1)
//
// Reference: Qwen3_5Attention in modeling_qwen3_5.py
//
// 1. q_proj(x) → [12288] = [24*256 + 24*256], split into q[24,256] + gate[24,256]
// 2. k_proj(x) → [1024] = [4,256], v_proj(x) → [1024] = [4,256]
// 3. Q RMSNorm per-head, K RMSNorm per-head
// 4. Partial RoPE (first 64 dims)
// 5. Append K,V to kv_cache at position pos
// 6. QK^T / sqrt(head_dim) → softmax → V → out[24,256]
// 7. out = out * sigmoid(gate)  (attention gate)
// 8. o_proj(out) → [5120]
//
// KV cache layout: [2, num_kv_heads, max_kv_len, head_dim]
//   offset 0:                        K cache
//   offset num_kv_heads*max_kv*dim:  V cache
// ============================================================================

void full_attention_forward(const __half* x, const FullAttentionWeights& attn,
                            __half* kv_cache, int layer_idx,
                            int pos, int max_kv_len,
                            InferenceState& state, cudaStream_t stream,
                            bool trace) {
    using MC = ModelConfig;

    cudaEvent_t t0, t1;
    if (trace) { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    auto mark = [&](const char* label) {
        if (!trace) return;
        cudaEventRecord(t1, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("    [FA] %-28s %7.2f ms\n", label, ms);
        cudaEventRecord(t0, stream);
    };
    if (trace) cudaEventRecord(t0, stream);

    // 1. Q projection via FP16 GEMV: q_proj [5120→12288], interleaved Q+Gate
    fp16_gemv(x, attn.fp16_q.weight, state.q_buf,
              attn.fp16_q.in_features, attn.fp16_q.out_features, stream);
    mark("q_proj");

    // Deinterleave Q and Gate into separate contiguous buffers
    // Q → dn_qkv[6144] (reuse DeltaNet buffer, not used during full attention)
    // Gate → mlp_gate[6144] (reuse MLP buffer, not used until after attention)
    __half* q_ptr    = state.dn_qkv;   // [24, 256] contiguous Q
    __half* gate_ptr = state.mlp_gate;  // [24, 256] contiguous Gate
    split_q_gate_kernel<<<MC::NUM_ATTN_HEADS, 256, 0, stream>>>(
        state.q_buf, q_ptr, gate_ptr, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);

    // 2. K and V projections via FP16 GEMV: x[5120] → dn_z[2048] = [K(1024), V(1024)]
    fp16_gemv(x, attn.fp16_k.weight, state.dn_z,
              attn.fp16_k.in_features, attn.fp16_k.out_features, stream);
    fp16_gemv(x, attn.fp16_v.weight, state.dn_z + MC::KV_PROJ_DIM,
              attn.fp16_v.in_features, attn.fp16_v.out_features, stream);
    __half* k_buf = state.dn_z;
    __half* v_buf = state.dn_z + MC::NUM_KV_HEADS * MC::HEAD_DIM;
    mark("k_proj+v_proj+deinterleave");

    // 3. Per-head Q/K RMSNorm
    head_norm(q_ptr, attn.q_norm, q_ptr,
              MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(k_buf, attn.k_norm, k_buf,
              MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    mark("head_norm");

    // 4. Partial RoPE on Q and K (reads pos from state.d_pos)
    apply_rope(q_ptr, k_buf,
               MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS, MC::HEAD_DIM,
               MC::ROTARY_DIM, state.d_pos, MC::ROPE_THETA, stream);
    mark("rope");

    // 5. Append K,V to cache via single kernel (reads pos from state.d_pos)
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    __half* k_cache = kv_cache + (size_t)layer_idx * 2 * kv_plane;
    __half* v_cache_ptr = k_cache + kv_plane;

    kv_cache_write_kernel<<<MC::NUM_KV_HEADS, 256, 0, stream>>>(
        k_buf, v_buf, k_cache, v_cache_ptr,
        state.d_pos, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_HEADS);
    mark("kv_cache_write");

    // 6. Flash-decoding style GQA attention: online softmax + V accumulation
    float scale = 1.0f / sqrtf((float)MC::HEAD_DIM);

    gqa_decode_attention_kernel<<<MC::NUM_ATTN_HEADS, MC::HEAD_DIM, 0, stream>>>(
        q_ptr, k_cache, v_cache_ptr, state.attn_out,
        state.d_pos, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_GROUPS, scale);
    mark("fused QK+softmax+V@s");

    // 7. Attention gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, gate_ptr, state.attn_out,
                 MC::ATTN_OUT_DIM, stream);
    mark("sigmoid_gate");

    // 8. o_proj via FP16 GEMV: attn_out[6144] → norm_out[5120]
    fp16_gemv(state.attn_out, attn.fp16_o.weight, state.norm_out,
              attn.fp16_o.in_features, attn.fp16_o.out_features, stream);
    mark("o_proj");

    if (trace) { cudaEventDestroy(t0); cudaEventDestroy(t1); }
}

// ============================================================================
// L2 normalization kernel (per-head, for DeltaNet Q/K)
// ============================================================================

__global__ void l2norm_kernel(__half* __restrict__ x, int head_dim, float eps) {
    int head = blockIdx.x;
    __half* h = x + head * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(h[i]);
        sum_sq += v * v;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[4];  // max 128 threads = 4 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_inv;
    if (threadIdx.x == 0)
        s_inv = rsqrtf(sum_sq + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        h[i] = __float2half(__half2float(h[i]) * s_inv);
}

// L2 normalize + scale: x[i] = x[i] / ||x|| * scale
// Fused version that eliminates a separate scale_fp16 kernel launch.
__global__ void l2norm_scaled_kernel(__half* __restrict__ x, int head_dim,
                                     float eps, float scale) {
    int head = blockIdx.x;
    __half* h = x + head * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(h[i]);
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[4];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_inv_scaled;
    if (threadIdx.x == 0)
        s_inv_scaled = rsqrtf(sum_sq + eps) * scale;
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        h[i] = __float2half(__half2float(h[i]) * s_inv_scaled);
}

// ============================================================================
// DeltaNet recurrent step kernel (single token, one block per head)
//
// Per head h (48 total):
//   g_scalar = exp(g[h])   (decay factor)
//   beta_scalar = beta[h]
//   state[h] *= g_scalar
//   kv_mem = state[h]^T @ k_h    (matrix-vector: [v_dim] = [k_dim,v_dim]^T @ [k_dim])
//   delta = (v_h - kv_mem) * beta_scalar
//   state[h] += k_h ⊗ delta     (rank-1 outer product update)
//   out[h] = state[h]^T @ q_h
//
// state[h]: [k_dim=128, v_dim=128] in F32
// q_h, k_h: [k_dim=128] in FP16
// v_h: [v_dim=128] in FP16
// g[h], beta[h]: scalars from FP16
// ============================================================================

__global__ void deltanet_recurrent_kernel(
    const __half* __restrict__ q,    // [48, 128] l2normed + scaled
    const __half* __restrict__ k,    // [48, 128] l2normed
    const __half* __restrict__ v,    // [48, 128]
    const float* __restrict__ g,     // [48] (already: -exp(A_log) * softplus(a + dt_bias))
    const float* __restrict__ beta,  // [48] (sigmoid of b)
    float* __restrict__ state,       // [48, 128, 128]
    __half* __restrict__ out,        // [48, 128]
    int k_dim, int v_dim)
{
    int head = blockIdx.x;
    int tid = threadIdx.x;

    float g_scalar = expf(g[head]);
    float beta_scalar = beta[head];

    float* S = state + (size_t)head * k_dim * v_dim;  // [k_dim, v_dim]
    const __half* q_h = q + head * k_dim;
    const __half* k_h = k + head * k_dim;
    const __half* v_h = v + head * v_dim;
    __half* out_h = out + head * v_dim;

    // Load k_h and v_h into shared memory
    extern __shared__ float smem[];     // [k_dim + v_dim]
    float* sk = smem;                   // [k_dim]
    float* sv = sk + k_dim;             // [v_dim]

    for (int i = tid; i < k_dim; i += blockDim.x)
        sk[i] = __half2float(k_h[i]);
    for (int i = tid; i < v_dim; i += blockDim.x)
        sv[i] = __half2float(v_h[i]);
    __syncthreads();

    // For each row i of the state matrix (thread parallelism over v_dim):
    // Each thread handles one or more columns of v_dim.
    // We iterate over k_dim rows, accumulating kv_mem and out.

    // Strategy: each thread handles a column j of v_dim.
    // It computes kv_mem[j] = sum_i(S[i,j] * k[i]) across all k_dim rows.
    // Then delta[j] = (v[j] - kv_mem[j]) * beta
    // Then S[i,j] = S[i,j] * g + k[i] * delta[j] for all i
    // Then out[j] = sum_i(S[i,j] * q[i])

    // Optimized: eliminate s_kv_mem shared memory and unnecessary __syncthreads.
    // kv_mem[j] is computed per-thread (no cross-thread dependency on it).
    // The update and output are fused into a single pass over the state,
    // eliminating one full re-read of S from global/L2.

    for (int j = tid; j < v_dim; j += blockDim.x) {
        float v_j = sv[j];

        // Pass 1+2 fused: decay, kv_mem, delta, update, output in one pass
        float mem = 0.0f;
        for (int i = 0; i < k_dim; i++) {
            float s_ij = S[i * v_dim + j] * g_scalar;  // decay
            mem += s_ij * sk[i];
            S[i * v_dim + j] = s_ij;  // write decayed state
        }

        float delta = (v_j - mem) * beta_scalar;

        float out_j = 0.0f;
        for (int i = 0; i < k_dim; i++) {
            S[i * v_dim + j] += sk[i] * delta;
            out_j += S[i * v_dim + j] * __half2float(q_h[i]);
        }
        out_h[j] = __float2half(out_j);
    }
}

// ============================================================================
// Batched conv1d: one kernel processes all M tokens per channel
// Each thread handles one channel across all tokens (sequential state update).
// Eliminates M-1 kernel launches vs per-token conv1d.
// ============================================================================

// ============================================================================
// Fused DeltaNet head kernel: processes all M tokens for one head
// Fuses: repeat_interleave + compute_g_beta + l2norm_q + l2norm_k + recurrent
// Grid: num_v_heads (48), Block: 128 threads
// Register-cached state: S[128] floats per thread → eliminates ~132MB global
// memory traffic across all heads × tokens. Only load once / store once.
// ============================================================================

// ============================================================================
// DeltaNet forward (single-token decode)
// ============================================================================
// DeltaNet helper kernels
// ============================================================================

// Compute g and beta from a, b, A_log, dt_bias (all per-head scalars)
// g[h] = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
// beta[h] = sigmoid(b[h])
__global__ void compute_g_beta_kernel(const __half* __restrict__ a,
                                      const __half* __restrict__ b,
                                      const float* __restrict__ A_log,
                                      const __half* __restrict__ dt_bias,
                                      float* __restrict__ g,
                                      float* __restrict__ beta,
                                      int n) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    float a_val = __half2float(a[h]);
    float b_val = __half2float(b[h]);
    float dt_b  = __half2float(dt_bias[h]);
    float A_val = expf(A_log[h]);
    float x_ab = a_val + dt_b;
    float sp = (x_ab > 20.0f) ? x_ab : logf(1.0f + expf(x_ab));
    g[h] = -A_val * sp;
    beta[h] = 1.0f / (1.0f + expf(-b_val));
}

// Scale FP16 array by a scalar: x[i] *= scale
__global__ void scale_fp16_kernel(__half* __restrict__ x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = __float2half(__half2float(x[i]) * scale);
    }
}

// repeat_interleave: expand [num_k_heads, head_dim] → [num_v_heads, head_dim]
// Each of num_k_heads gets repeated (num_v_heads/num_k_heads) times.
__global__ void repeat_interleave_kernel(const __half* __restrict__ src,
                                         __half* __restrict__ dst,
                                         int num_k_heads, int head_dim,
                                         int ratio) {
    // blockIdx.x = destination head, threadIdx.x = element within head_dim
    int dst_h = blockIdx.x;
    int src_h = dst_h / ratio;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst[dst_h * head_dim + d] = src[src_h * head_dim + d];
    }
}

// ============================================================================
// DeltaNet forward (single-token decode)
//
// Reference: Qwen3_5GatedDeltaNet in modeling_qwen3_5.py
//
// 1. in_proj_qkv(x) → [10240], then conv1d_step + silu
// 2. Split → query[2048], key[2048], value[6144]
// 3. Reshape to heads: q[16,128], k[16,128], v[48,128]
// 4. repeat_interleave q,k: [16,128] → [48,128]
// 5. in_proj_z(x) → z[6144], in_proj_a(x) → a[48], in_proj_b(x) → b[48]
// 6. beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
// 7. l2norm q, k per head
// 8. scale q by 1/sqrt(128)
// 9. Recurrent step (update state, compute output)
// 10. Gated RMSNorm(out, z)
// 11. out_proj → [5120]
// ============================================================================

void deltanet_forward(const __half* x, const DeltaNetWeights& dn,
                      int dn_layer_idx,
                      InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // 1. QKV+A+B projections via FP16 GEMV → dn_qkv[10496]
    // Output layout: [0:10240]=qkv, [10240:10288]=a, [10288:10336]=b, [10336:10496]=pad
    fp16_gemv(x, dn.fp16_qkv.weight, state.dn_qkv,
              dn.fp16_qkv.in_features, dn.fp16_qkv.out_features, stream);
    fp16_gemv(x, dn.fp16_a.weight, state.dn_qkv + MC::LIN_CONV_DIM,
              dn.fp16_a.in_features, dn.fp16_a.out_features, stream);
    fp16_gemv(x, dn.fp16_b.weight, state.dn_qkv + MC::LIN_CONV_DIM + MC::LIN_NUM_V_HEADS,
              dn.fp16_b.in_features, dn.fp16_b.out_features, stream);

    // Fused Conv1d step + SiLU on qkv only (first 10240 elements)
    causal_conv1d_step_silu(state.dn_qkv, state.conv_states[dn_layer_idx],
                            dn.conv1d_weight, state.dn_qkv,
                            MC::LIN_CONV_DIM, MC::CONV_KERNEL, stream);

    // z projection via FP16 GEMV
    fp16_gemv(x, dn.fp16_z.weight, state.dn_z,
              dn.fp16_z.in_features, dn.fp16_z.out_features, stream);

    // Fused: repeat_interleave + compute_g_beta + l2norm_q + l2norm_k + recurrent
    // a/b already in dn_qkv at offsets LIN_CONV_DIM and LIN_CONV_DIM+48 from merged proj.
    {
        int fused_smem = (MC::LIN_K_HEAD_DIM + MC::LIN_K_HEAD_DIM + 4) * sizeof(float);
        deltanet_fused_head_kernel<<<MC::LIN_NUM_V_HEADS, 128, fused_smem, stream>>>(
            state.dn_qkv,
            dn.A_log, dn.dt_bias,
            state.dn_states[dn_layer_idx],
            state.attn_out,
            1, MC::LIN_NUM_K_HEADS, MC::LIN_NUM_V_HEADS,
            MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM,
            MC::LIN_KEY_DIM, MC::LIN_CONV_DIM,
            MC::LIN_CONV_DIM, MC::LIN_CONV_DIM + MC::LIN_NUM_V_HEADS,
            MC::LIN_QKV_AB_DIM, MC::RMS_EPS);
    }

    // 10. Gated RMSNorm: out[48, 128] with gate z[48, 128]
    rms_norm_gated(state.attn_out, state.dn_z,
                   dn.norm_weight, state.attn_out,
                   MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, stream);

    // 11. out_proj via FP16 GEMV: [6144] → [5120]
    fp16_gemv(state.attn_out, dn.fp16_out.weight, state.norm_out,
              dn.fp16_out.in_features, dn.fp16_out.out_features, stream);
}

} // namespace deusridet
