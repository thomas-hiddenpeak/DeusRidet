// forward.cu — Qwen3.5-27B forward pass implementation
//
// Single-token decode path. Each function assumes M=1.
// Target: SM87 (Jetson AGX Orin)

#include "forward.h"
#include "layer.h"
#include "gptq.h"
#include "marlin.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

namespace deusridet {

// ============================================================================
// SwiGLU MLP forward (M=1)
//
// gate_out = gate_proj(x)   — GPTQ Int4
// up_out   = up_proj(x)     — GPTQ Int4
// gate_out = silu(gate_out) * up_out
// mlp_out  = down_proj(gate_out) — GPTQ Int4
// ============================================================================

void mlp_forward(const __half* x, const MLPWeights& mlp,
                 __half* residual,
                 InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // Gate + Up projections via Marlin GEMM (weights are in Marlin format)
    marlin_gemm(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                state.marlin_workspace, 1, mlp.gate_proj.K, mlp.gate_proj.N, 128, stream);
    marlin_gemm(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                state.marlin_workspace, 1, mlp.up_proj.K, mlp.up_proj.N, 128, stream);

    // SiLU activation: mlp_gate = silu(mlp_gate) * mlp_up
    silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
             MC::INTERMEDIATE_SIZE, stream);

    // Down projection + residual add
    marlin_gemm(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down, mlp.down_proj.scales,
                state.marlin_workspace, 1, mlp.down_proj.K, mlp.down_proj.N, 128, stream);
    if (residual) {
        elementwise_add(residual, state.mlp_down, residual,
                        MC::HIDDEN_SIZE, stream);
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

    // 1. Q projection — q_proj outputs [12288] = interleaved [Q_h0(256), Gate_h0(256), ...]
    int8_linear_forward(x, attn.q_proj, state.q_buf, 1, stream);
    mark("q_proj");

    // Deinterleave Q and Gate into separate contiguous buffers
    // Q → dn_qkv[6144] (reuse DeltaNet buffer, not used during full attention)
    // Gate → mlp_gate[6144] (reuse MLP buffer, not used until after attention)
    __half* q_ptr    = state.dn_qkv;   // [24, 256] contiguous Q
    __half* gate_ptr = state.mlp_gate;  // [24, 256] contiguous Gate
    split_q_gate_kernel<<<MC::NUM_ATTN_HEADS, 256, 0, stream>>>(
        state.q_buf, q_ptr, gate_ptr, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);

    // 2. K, V projections (fused dual INT8 GEMV — shared SMEM x)
    int8_dual_linear_forward(x, attn.k_proj, state.kv_buf,
                                attn.v_proj, state.dn_z, 1, stream);
    __half* v_buf = state.dn_z;  // [kv_proj_dim=1024], reuse (not used in full attn)
    mark("k_proj+v_proj+deinterleave");

    // 3. Per-head Q/K RMSNorm
    head_norm(q_ptr, attn.q_norm, q_ptr,
              MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(state.kv_buf, attn.k_norm, state.kv_buf,
              MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    mark("head_norm");

    // 4. Partial RoPE on Q and K (reads pos from state.d_pos)
    apply_rope(q_ptr, state.kv_buf,
               MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS, MC::HEAD_DIM,
               MC::ROTARY_DIM, state.d_pos, MC::ROPE_THETA, stream);
    mark("rope");

    // 5. Append K,V to cache via single kernel (reads pos from state.d_pos)
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    __half* k_cache = kv_cache + (size_t)layer_idx * 2 * kv_plane;
    __half* v_cache_ptr = k_cache + kv_plane;

    kv_cache_write_kernel<<<MC::NUM_KV_HEADS, 256, 0, stream>>>(
        state.kv_buf, v_buf, k_cache, v_cache_ptr,
        state.d_pos, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_HEADS);
    mark("kv_cache_write");

    // 6. Flash-decoding style GQA attention: online softmax + V accumulation
    // No dynamic shared memory needed (uses static __shared__ for constant-size buffers)
    float scale = 1.0f / sqrtf((float)MC::HEAD_DIM);

    gqa_decode_attention_kernel<<<MC::NUM_ATTN_HEADS, MC::HEAD_DIM, 0, stream>>>(
        q_ptr, k_cache, v_cache_ptr, state.attn_out,
        state.d_pos, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_GROUPS, scale);
    mark("fused QK+softmax+V@s");

    // 7. Attention gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, gate_ptr, state.attn_out,
                 MC::ATTN_OUT_DIM, stream);
    mark("sigmoid_gate");

    // 8. o_proj: attn_out[6144] → norm_out[5120] (INT8 quantized)
    int8_linear_forward(state.attn_out, attn.o_proj, state.norm_out, 1, stream);
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
__global__ void conv1d_batch_silu_kernel(
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

// ============================================================================
// Fused DeltaNet head kernel: processes all M tokens for one head
// Fuses: repeat_interleave + compute_g_beta + l2norm_q + l2norm_k + recurrent
// Grid: num_v_heads (48), Block: 128 threads
// Register-cached state: S[128] floats per thread → eliminates ~132MB global
// memory traffic across all heads × tokens. Only load once / store once.
// ============================================================================
__global__ void __launch_bounds__(128, 2)
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

    // 1. QKV projection (INT8 quantized)
    int8_linear_forward(x, dn.in_proj_qkv, state.dn_qkv, 1, stream);

    // Fused Conv1d step + SiLU (eliminates one kernel launch + intermediate R/W)
    causal_conv1d_step_silu(state.dn_qkv, state.conv_states[dn_layer_idx],
                            dn.conv1d_weight, state.dn_qkv,
                            MC::LIN_CONV_DIM, MC::CONV_KERNEL, stream);

    // 2-3. Split qkv: [KEY_DIM, KEY_DIM, VALUE_DIM] = [2048, 2048, 6144]
    __half* dn_query = state.dn_qkv;                         // [16, 128]
    __half* dn_key   = state.dn_qkv + MC::LIN_KEY_DIM;       // [16, 128]
    __half* dn_value = state.dn_qkv + 2 * MC::LIN_KEY_DIM;   // [48, 128]

    // 4. repeat_interleave: q,k from 16 heads → 48 heads (ratio 3) via GPU kernel
    __half* q_expanded = state.attn_out;  // [48, 128] = [6144]
    __half* k_expanded = state.q_buf;     // [48, 128] (reuse q_buf, large enough)
    int ratio = MC::LIN_NUM_V_HEADS / MC::LIN_NUM_K_HEADS;  // 3

    repeat_interleave_kernel<<<MC::LIN_NUM_V_HEADS, 128, 0, stream>>>(
        dn_query, q_expanded, MC::LIN_NUM_K_HEADS, MC::LIN_K_HEAD_DIM, ratio);
    repeat_interleave_kernel<<<MC::LIN_NUM_V_HEADS, 128, 0, stream>>>(
        dn_key, k_expanded, MC::LIN_NUM_K_HEADS, MC::LIN_K_HEAD_DIM, ratio);

    // 5. z projection + fused a+b projection (single kernel, shared SMEM x)
    int8_linear_forward(x, dn.in_proj_z, state.dn_z, 1, stream);
    int8_dual_linear_forward(x, dn.in_proj_a, state.dn_a,
                                dn.in_proj_b, state.dn_b, 1, stream);

    // 6. Compute g and beta entirely on GPU (pre-allocated buffers)
    compute_g_beta_kernel<<<1, MC::LIN_NUM_V_HEADS, 0, stream>>>(
        state.dn_a, state.dn_b, dn.A_log, dn.dt_bias,
        state.dn_g, state.dn_beta, MC::LIN_NUM_V_HEADS);

    // 7-8. L2 normalize q (with fused 1/sqrt(k_dim) scale) and k (per head)
    float q_scale = 1.0f / sqrtf((float)MC::LIN_K_HEAD_DIM);
    l2norm_scaled_kernel<<<MC::LIN_NUM_V_HEADS, 128, 0, stream>>>(
        q_expanded, MC::LIN_K_HEAD_DIM, 1e-6f, q_scale);
    l2norm_kernel<<<MC::LIN_NUM_V_HEADS, 128, 0, stream>>>(
        k_expanded, MC::LIN_K_HEAD_DIM, 1e-6f);

    // 9. Recurrent step
    int smem_size = (MC::LIN_K_HEAD_DIM + MC::LIN_V_HEAD_DIM) * sizeof(float);
    deltanet_recurrent_kernel<<<MC::LIN_NUM_V_HEADS, 128, smem_size, stream>>>(
        q_expanded, k_expanded, dn_value,
        state.dn_g, state.dn_beta,
        state.dn_states[dn_layer_idx],
        state.attn_out,  // output [48, 128] = [6144]
        MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM);

    // 10. Gated RMSNorm: out[48, 128] with gate z[48, 128]
    rms_norm_gated(state.attn_out, state.dn_z,
                   dn.norm_weight, state.attn_out,
                   MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, stream);

    // 11. out_proj: [6144] → [5120], write to norm_out (INT8 quantized)
    int8_linear_forward(state.attn_out, dn.out_proj, state.norm_out, 1, stream);
}

// ============================================================================
// Complete single-token forward pass
//
// embed → for each layer: norm → attn/deltanet → residual → norm → MLP → residual
// → final_norm → lm_head → greedy sample
//
// Convention: after attn/deltanet, output is in state.norm_out[0..hidden-1]
//             after MLP, output is in state.mlp_down[0..hidden-1]
// ============================================================================

// ============================================================================
// Graph-capturable forward body
//
// Contains ALL GPU operations for a single decode token, with NO host sync.
// Reads token_id and pos from pinned staging buffers (set by caller before
// graph launch). This function is called once during graph capture, then
// the captured graph is replayed for all subsequent tokens.
//
// Prerequisite: h_token_pinned and h_pos_pinned are set by the caller.
// ============================================================================

static void forward_body(const ModelWeights& model,
                         InferenceState& state,
                         __half* kv_cache,
                         int max_kv_len,
                         cudaStream_t stream) {
    using MC = ModelConfig;

    // H2D from pinned staging (graph replays read current values from pinned memory)
    cudaMemcpyAsync(state.token_ids, state.h_token_pinned, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(state.d_pos, state.h_pos_pinned, sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Embedding lookup → hidden[0, 0..5119]
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, 1, MC::HIDDEN_SIZE, stream);

    // Copy to residual for first layer
    cudaMemcpyAsync(state.residual, state.hidden,
                    MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);

    int dn_layer_idx = 0;  // Counter for DeltaNet layers (0..47)

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        // Pre-attention RMSNorm
        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

        // Attention / DeltaNet
        if (lw.is_full_attention) {
            full_attention_forward(state.norm_out, lw.full_attn,
                                   kv_cache, layer, 0 /*pos read from d_pos*/, max_kv_len,
                                   state, stream);
        } else {
            deltanet_forward(state.norm_out, lw.delta_net,
                             dn_layer_idx, state, stream);
            dn_layer_idx++;
        }

        // Fused: residual += attn_output; norm_out = RMSNorm(residual)
        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

        // MLP with fused residual add: residual += down_proj(silu(gate)*up)
        mlp_forward(state.norm_out, lw.mlp, state.residual, state, stream);
    }

    // Final RMSNorm
    rms_norm(state.residual, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

    // LM head: hidden[5120] → logits[248320] (INT8 quantized — halves weight read)
    int8_linear_forward(state.hidden, model.lm_head_int8, state.logits, 1, stream);

    // GPU argmax (no sync — result extracted by caller after graph launch)
    argmax_async(state.logits, MC::VOCAB_SIZE, state.sample_out, stream);
}

// ============================================================================
// Single-token forward pass with CUDA Graph acceleration
//
// First call: captures the entire forward body into a CUDA Graph.
// Subsequent calls: replay the graph (eliminates ~1400 kernel launch overhead).
//
// Per-token-changing parameters (token_id, pos) are passed via pinned host
// staging → H2D memcpy (captured in graph; reads current pinned values at replay).
// ============================================================================

int forward_one_token(const ModelWeights& model,
                      InferenceState& state,
                      __half* kv_cache,
                      int token_id, int pos, int max_kv_len,
                      cudaStream_t stream) {
    // Use non-default compute stream for graph capture/replay
    // (cudaStreamBeginCapture requires non-default stream)
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Write per-token values to pinned staging buffers
    *state.h_token_pinned = token_id;
    *state.h_pos_pinned = pos;

    if (!state.graph_captured) {
        // First invocation: capture CUDA Graph
        cudaError_t err;

        err = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            LOG_ERROR("Machina", "Graph BeginCapture failed: %s", cudaGetErrorString(err));
        }

        forward_body(model, state, kv_cache, max_kv_len, s);

        err = cudaStreamEndCapture(s, &state.cuda_graph);
        if (err != cudaSuccess) {
            LOG_ERROR("Machina", "Graph EndCapture failed: %s", cudaGetErrorString(err));
        }

        // Check graph node count
        size_t num_nodes = 0;
        cudaGraphGetNodes(state.cuda_graph, nullptr, &num_nodes);
        LOG_INFO("Machina", "CUDA Graph captured: %zu nodes", num_nodes);

        err = cudaGraphInstantiate(&state.cuda_graph_exec, state.cuda_graph, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Machina", "Graph Instantiate failed: %s", cudaGetErrorString(err));
        }

        state.graph_captured = true;
    }

    // Replay captured graph (pinned staging contains current token_id & pos)
    cudaError_t err = cudaGraphLaunch(state.cuda_graph_exec, s);
    if (err != cudaSuccess) {
        LOG_ERROR("Machina", "Graph Launch failed: %s", cudaGetErrorString(err));
    }

    // Extract result (outside graph — single D2H + sync)
    int result;
    cudaMemcpyAsync(&result, state.sample_out, sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return result;
}

// ============================================================================
// Single-token forward pass with configurable sampling (no CUDA Graph)
//
// Runs the full forward body then applies top-k/top-p sampling.
// Intended for generation with non-greedy sampling strategies.
// ============================================================================

int forward_one_token_sampled(const ModelWeights& model,
                              InferenceState& state,
                              __half* kv_cache,
                              int token_id, int pos, int max_kv_len,
                              const SamplingParams& params,
                              cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Copy token_id and pos to device
    cudaMemcpyAsync(state.token_ids, &token_id, sizeof(int),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(state.d_pos, &pos, sizeof(int),
                    cudaMemcpyHostToDevice, s);

    // Embedding lookup → hidden
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, 1, MC::HIDDEN_SIZE, s);

    // Copy to residual
    cudaMemcpyAsync(state.residual, state.hidden,
                    MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, s);

    int dn_layer_idx = 0;
    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        if (lw.is_full_attention) {
            full_attention_forward(state.norm_out, lw.full_attn,
                                   kv_cache, layer, pos, max_kv_len,
                                   state, s);
        } else {
            deltanet_forward(state.norm_out, lw.delta_net,
                             dn_layer_idx, state, s);
            dn_layer_idx++;
        }

        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        mlp_forward(state.norm_out, lw.mlp, state.residual, state, s);
    }

    rms_norm(state.residual, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

    int8_linear_forward(state.hidden, model.lm_head_int8, state.logits, 1, s);

    // Top-k/top-p sampling
    unsigned long long seed = params.seed ? params.seed : ++state.rng_counter;
    sample_top_k_top_p(state.logits, state.probs, MC::VOCAB_SIZE,
                        params, seed, state.sample_out, s);

    int result;
    cudaMemcpyAsync(&result, state.sample_out, sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return result;
}

// ============================================================================
// Batched prefill: SwiGLU MLP (M>1)
//
// Uses Marlin GEMM for all GPTQ INT4 projections.
// Residual add done separately via elementwise_add.
// ============================================================================

static void mlp_forward_prefill(const __half* x, const MLPWeights& mlp,
                                __half* residual, int M,
                                InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;
    int N_inter = MC::INTERMEDIATE_SIZE;

    // gate_proj + up_proj via Marlin GEMM (weights in Marlin tile format)
    marlin_gemm(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                state.marlin_workspace, M, mlp.gate_proj.K, mlp.gate_proj.N, 128, stream);
    marlin_gemm(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                state.marlin_workspace, M, mlp.up_proj.K, mlp.up_proj.N, 128, stream);

    // SiLU: mlp_gate = silu(mlp_gate) * mlp_up
    silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
             M * N_inter, stream);

    // Down projection + residual add
    marlin_gemm(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down, mlp.down_proj.scales,
                state.marlin_workspace, M, mlp.down_proj.K, mlp.down_proj.N, 128, stream);
    if (residual) {
        elementwise_add(residual, state.mlp_down, residual,
                        M * MC::HIDDEN_SIZE, stream);
    }
}

// ============================================================================
// Batched prefill: split_q_gate for M tokens
// q_interleaved[M, 24, 512] → Q[M, 24, 256] + Gate[M, 24, 256]
// ============================================================================

__global__ void split_q_gate_batch_kernel(const __half* __restrict__ qg,
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

// ============================================================================
// Batched prefill: kv_cache_write for M tokens at consecutive positions
// ============================================================================

__global__ void kv_cache_write_batch_kernel(
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

// ============================================================================
// Batched prefill: RoPE for M tokens at consecutive positions
// ============================================================================

__global__ void rope_batch_kernel(__half* __restrict__ q,
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

// ============================================================================
// Prefill causal attention kernel (M queries attending to their own KV)
//
// Simple implementation: one block per (query_head, token). Each block
// computes attention for one query head at one token position.
// For small M (≤ 128), this gives sufficient parallelism.
//
// Grid: (num_attn_heads, M), Block: head_dim (256)
// ============================================================================

__global__ void prefill_attention_kernel(
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

// ============================================================================
// Batched prefill: Full Attention GQA (M tokens)
//
// Strategy: batch projections (GEMM), batch element-wise ops,
// batch causal self-attention, batch output projection.
// ============================================================================

static void full_attention_prefill(const __half* x, const FullAttentionWeights& attn,
                                   __half* kv_cache, int layer_idx,
                                   int pos_start, int M, int max_kv_len,
                                   InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // 1. Batched Q, K, V projections (INT8 GEMM)
    // Q: x[M, 5120] → q_buf[M, 12288]
    int8_linear_forward(x, attn.q_proj, state.q_buf, M, stream);

    // K+V: merged projection → attn_out[M, 2048], then split
    int8_linear_forward(x, attn.kv_proj, state.attn_out, M, stream);
    // Split: attn_out[M, 2048] → kv_buf[M, 1024] (K) + dn_z[M, 1024] (V)
    cudaMemcpy2DAsync(state.kv_buf, MC::KV_PROJ_DIM * sizeof(__half),
                      state.attn_out, MC::FA_KV_DIM * sizeof(__half),
                      MC::KV_PROJ_DIM * sizeof(__half), M,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(state.dn_z, MC::KV_PROJ_DIM * sizeof(__half),
                      state.attn_out + MC::KV_PROJ_DIM, MC::FA_KV_DIM * sizeof(__half),
                      MC::KV_PROJ_DIM * sizeof(__half), M,
                      cudaMemcpyDeviceToDevice, stream);

    // 2. Batched split_q_gate: q_buf[M, 12288] → Q_sep[M, 6144] + Gate[M, 6144]
    // Q_sep → dn_qkv[M, 10240] (only first 6144 per row used)
    // Gate → mlp_gate[M, 17408] (only first 6144 per row used)
    __half* Qsep = state.dn_qkv;
    __half* Gate = state.mlp_gate;
    {
        dim3 grid(MC::NUM_ATTN_HEADS, M);
        split_q_gate_batch_kernel<<<grid, 256, 0, stream>>>(
            state.q_buf, Qsep, Gate,
            M, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);
    }

    // 3. Batched head_norm for Q (M * 24 heads) and K (M * 4 heads)
    head_norm(Qsep, attn.q_norm, Qsep,
              M * MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(state.kv_buf, attn.k_norm, state.kv_buf,
              M * MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);

    // 4. Batched RoPE for M tokens at consecutive positions
    {
        int max_heads = (MC::NUM_ATTN_HEADS > MC::NUM_KV_HEADS)
                        ? MC::NUM_ATTN_HEADS : MC::NUM_KV_HEADS;
        dim3 grid(max_heads, M);
        rope_batch_kernel<<<grid, MC::ROTARY_DIM / 2, 0, stream>>>(
            Qsep, state.kv_buf,
            MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS,
            MC::HEAD_DIM, MC::ROTARY_DIM,
            pos_start, M, MC::ROPE_THETA);
    }

    // 5. Batched KV cache write
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    __half* k_cache = kv_cache + (size_t)layer_idx * 2 * kv_plane;
    __half* v_cache_ptr = k_cache + kv_plane;
    {
        dim3 grid(MC::NUM_KV_HEADS, M);
        kv_cache_write_batch_kernel<<<grid, 256, 0, stream>>>(
            state.kv_buf, state.dn_z, k_cache, v_cache_ptr,
            pos_start, M, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_HEADS);
    }

    // 6. Batched causal attention
    // Grid: (num_attn_heads, M), each block does one head for one token
    {
        float scale_val = 1.0f / sqrtf((float)MC::HEAD_DIM);
        dim3 grid(MC::NUM_ATTN_HEADS, M);
        // Output to attn_out[M, 6144]
        prefill_attention_kernel<<<grid, MC::HEAD_DIM, 0, stream>>>(
            Qsep, k_cache, v_cache_ptr, state.attn_out,
            pos_start, M, max_kv_len, MC::HEAD_DIM,
            MC::NUM_KV_GROUPS, scale_val);
    }

    // 7. Batched sigmoid_gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, Gate, state.attn_out,
                 M * MC::ATTN_OUT_DIM, stream);

    // 8. Batched o_proj: attn_out[M, 6144] → norm_out[M, 5120]
    int8_linear_forward(state.attn_out, attn.o_proj, state.norm_out, M, stream);
}

// ============================================================================
// Batched prefill: DeltaNet (M tokens)
//
// Strategy: batch projections (INT8 GEMM), process conv1d + recurrent
// sequentially per token, batch output (rms_norm_gated + out_proj).
// ============================================================================

static void deltanet_prefill(const __half* x, const DeltaNetWeights& dn,
                             int dn_layer_idx, int M,
                             InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // 1. Merged qkv+a+b projection: x[M, 5120] → dn_qkv[M, 10368]
    //    Eliminates separate ab_proj launch. a/b at columns 10240..10335.
    int8_linear_forward(x, dn.in_proj_qkv_ab, state.dn_qkv, M, stream);

    // Z: x[M, 5120] → dn_z[M, 6144]
    int8_linear_forward(x, dn.in_proj_z, state.dn_z, M, stream);

    // 2. Batch conv1d: all M tokens in one launch (stride=10368 for merged buffer)
    {
        int conv_blocks = (MC::LIN_CONV_DIM + 255) / 256;
        conv1d_batch_silu_kernel<<<conv_blocks, 256, 0, stream>>>(
            state.dn_qkv, state.conv_states[dn_layer_idx],
            dn.conv1d_weight, M, MC::LIN_CONV_DIM, MC::CONV_KERNEL,
            MC::LIN_QKV_AB_DIM);
    }

    // 3. Fused head kernel: repeat_interleave + g/beta + L2norm + recurrent
    //    a/b read from within dn_qkv buffer at offsets 10240 and 10288
    {
        int fused_smem = (MC::LIN_K_HEAD_DIM + MC::LIN_K_HEAD_DIM + 4) * sizeof(float);
        deltanet_fused_head_kernel<<<MC::LIN_NUM_V_HEADS, 128, fused_smem, stream>>>(
            state.dn_qkv,
            dn.A_log, dn.dt_bias,
            state.dn_states[dn_layer_idx],
            state.attn_out,
            M, MC::LIN_NUM_K_HEADS, MC::LIN_NUM_V_HEADS,
            MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM,
            MC::LIN_KEY_DIM, MC::LIN_QKV_AB_DIM,
            MC::LIN_CONV_DIM, MC::LIN_CONV_DIM + MC::LIN_NUM_V_HEADS,
            MC::LIN_QKV_AB_DIM, MC::RMS_EPS);
    }

    // 4. Batched gated RMSNorm: attn_out[M, 48, 128] with gate dn_z[M, 48, 128]
    rms_norm_gated(state.attn_out, state.dn_z,
                   dn.norm_weight, state.attn_out,
                   M * MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, stream);

    // 5. Batched out_proj: attn_out[M, 6144] → norm_out[M, 5120]
    int8_linear_forward(state.attn_out, dn.out_proj, state.norm_out, M, stream);
}

// ============================================================================
// Complete batched prefill forward pass
//
// embed[M] → for each layer: norm → attn/deltanet → residual → norm → MLP
// → final_norm → lm_head (last token only) → greedy sample
// ============================================================================

int forward_prefill(const ModelWeights& model,
                    InferenceState& state,
                    __half* kv_cache,
                    const int* h_token_ids, int M,
                    int pos_start, int max_kv_len,
                    cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Copy token IDs to device
    cudaMemcpyAsync(state.token_ids, h_token_ids, M * sizeof(int),
                    cudaMemcpyHostToDevice, s);

    // Embedding lookup → hidden[M, H]
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, M, MC::HIDDEN_SIZE, s);

    // Copy to residual
    cudaMemcpyAsync(state.residual, state.hidden,
                    (size_t)M * MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, s);

    int dn_layer_idx = 0;

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        // Pre-attention RMSNorm: residual[M, H] → norm_out[M, H]
        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // Attention / DeltaNet (writes output to norm_out[M, H])
        if (lw.is_full_attention) {
            full_attention_prefill(state.norm_out, lw.full_attn,
                                   kv_cache, layer, pos_start, M, max_kv_len,
                                   state, s);
        } else {
            deltanet_prefill(state.norm_out, lw.delta_net,
                             dn_layer_idx, M, state, s);
            dn_layer_idx++;
        }

        // Fused: residual += attn_output; norm_out = RMSNorm(residual)
        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // MLP with residual add
        mlp_forward_prefill(state.norm_out, lw.mlp, state.residual, M, state, s);
    }

    // Final RMSNorm — only the last token matters for logits
    __half* last_hidden = state.residual + (size_t)(M - 1) * MC::HIDDEN_SIZE;
    rms_norm(last_hidden, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

    // LM head: hidden[5120] → logits[248320] (last token only)
    int8_linear_forward(state.hidden, model.lm_head_int8, state.logits, 1, s);

    // GPU argmax
    argmax_async(state.logits, MC::VOCAB_SIZE, state.sample_out, s);

    // Extract result
    int result;
    cudaMemcpyAsync(&result, state.sample_out, sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    return result;
}

// ============================================================================
// Sub-layer profiler: measures per-operation timing within DN, FA, MLP.
// Runs one representative layer of each type with fine-grained events.
// Call AFTER a warmup pass so buffers are populated.
// ============================================================================

void profile_sublayer_prefill(const ModelWeights& model,
                              InferenceState& state,
                              __half* kv_cache,
                              int M, int pos_start, int max_kv_len,
                              cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Find first DN and first FA layer
    int first_dn = -1, first_fa = -1;
    for (int i = 0; i < MC::NUM_LAYERS; i++) {
        if (!model.layers[i].is_full_attention && first_dn < 0) first_dn = i;
        if (model.layers[i].is_full_attention && first_fa < 0) first_fa = i;
    }

    constexpr int NEV = 16;
    cudaEvent_t ev[NEV];
    for (int i = 0; i < NEV; i++)
        cudaEventCreateWithFlags(&ev[i], cudaEventDefault);
    auto ms = [&](int a, int b) {
        float t; cudaEventElapsedTime(&t, ev[a], ev[b]); return t;
    };

    // State buffers already contain data from warmup — shapes are correct.
    // Timing depends on weight memory locations and tensor shapes, not data values.

    // === DN Sub-layer (one layer) ===
    if (first_dn >= 0) {
        const auto& dn = model.layers[first_dn].delta_net;
        const __half* x = state.norm_out;
        int e = 0;
        cudaEventRecord(ev[e++], s);  // 0
        int8_linear_forward(x, dn.in_proj_qkv_ab, state.dn_qkv, M, s);
        cudaEventRecord(ev[e++], s);  // 1
        int8_linear_forward(x, dn.in_proj_z, state.dn_z, M, s);
        cudaEventRecord(ev[e++], s);  // 2
        // (ab_proj merged into qkv_ab — no separate event)
        {
            int conv_blocks = (MC::LIN_CONV_DIM + 255) / 256;
            conv1d_batch_silu_kernel<<<conv_blocks, 256, 0, s>>>(
                state.dn_qkv, state.conv_states[0],
                dn.conv1d_weight, M, MC::LIN_CONV_DIM, MC::CONV_KERNEL,
                MC::LIN_QKV_AB_DIM);
        }
        cudaEventRecord(ev[e++], s);  // 3
        {
            int fused_smem = (MC::LIN_K_HEAD_DIM + MC::LIN_K_HEAD_DIM + 4) * sizeof(float);
            deltanet_fused_head_kernel<<<MC::LIN_NUM_V_HEADS, 128, fused_smem, s>>>(
                state.dn_qkv,
                dn.A_log, dn.dt_bias,
                state.dn_states[0],
                state.attn_out,
                M, MC::LIN_NUM_K_HEADS, MC::LIN_NUM_V_HEADS,
                MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM,
                MC::LIN_KEY_DIM, MC::LIN_QKV_AB_DIM,
                MC::LIN_CONV_DIM, MC::LIN_CONV_DIM + MC::LIN_NUM_V_HEADS,
                MC::LIN_QKV_AB_DIM, MC::RMS_EPS);
        }
        cudaEventRecord(ev[e++], s);  // 4
        rms_norm_gated(state.attn_out, state.dn_z,
                       dn.norm_weight, state.attn_out,
                       M * MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, s);
        cudaEventRecord(ev[e++], s);  // 5
        int8_linear_forward(state.attn_out, dn.out_proj, state.norm_out, M, s);
        cudaEventRecord(ev[e++], s);  // 6
        cudaStreamSynchronize(s);

        float dn_total = ms(0, 6);
        printf("\n=== DN Sub-layer (layer %d, M=%d) ===\n", first_dn, M);
        printf("  qkv_ab_proj (INT8 5120→10368): %6.3f ms  (%4.1f%%)\n", ms(0,1), 100*ms(0,1)/dn_total);
        printf("  z_proj    (INT8 5120→6144):    %6.3f ms  (%4.1f%%)\n", ms(1,2), 100*ms(1,2)/dn_total);
        printf("  conv1d_silu:                   %6.3f ms  (%4.1f%%)\n", ms(2,3), 100*ms(2,3)/dn_total);
        printf("  fused_head (recurrent):        %6.3f ms  (%4.1f%%)\n", ms(3,4), 100*ms(3,4)/dn_total);
        printf("  rms_norm_gated:                %6.3f ms  (%4.1f%%)\n", ms(4,5), 100*ms(4,5)/dn_total);
        printf("  out_proj  (INT8 6144→5120):    %6.3f ms  (%4.1f%%)\n", ms(5,6), 100*ms(5,6)/dn_total);
        printf("  TOTAL (1 DN layer):            %6.3f ms  (×48 = %.1f ms)\n", dn_total, dn_total*48);
    }

    // === MLP Sub-layer (one layer) ===
    {
        int mlp_layer = (first_dn >= 0) ? first_dn : 0;
        const auto& mlp = model.layers[mlp_layer].mlp;
        const __half* x = state.norm_out;
        int e = 0;
        cudaEventRecord(ev[e++], s);  // 0
        marlin_gemm(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                    state.marlin_workspace, M, mlp.gate_proj.K, mlp.gate_proj.N, 128, s);
        cudaEventRecord(ev[e++], s);  // 1
        marlin_gemm(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                    state.marlin_workspace, M, mlp.up_proj.K, mlp.up_proj.N, 128, s);
        cudaEventRecord(ev[e++], s);  // 2
        silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                 M * MC::INTERMEDIATE_SIZE, s);
        cudaEventRecord(ev[e++], s);  // 3
        marlin_gemm(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down, mlp.down_proj.scales,
                    state.marlin_workspace, M, mlp.down_proj.K, mlp.down_proj.N, 128, s);
        cudaEventRecord(ev[e++], s);  // 4
        elementwise_add(state.residual, state.mlp_down, state.residual,
                        M * MC::HIDDEN_SIZE, s);
        cudaEventRecord(ev[e++], s);  // 5
        cudaStreamSynchronize(s);

        float mlp_total = ms(0, 5);
        printf("\n=== MLP Sub-layer (layer %d, M=%d) ===\n", mlp_layer, M);
        printf("  gate_proj (Marlin 5120→17408):%6.3f ms  (%4.1f%%)\n", ms(0,1), 100*ms(0,1)/mlp_total);
        printf("  up_proj   (Marlin 5120→17408):%6.3f ms  (%4.1f%%)\n", ms(1,2), 100*ms(1,2)/mlp_total);
        printf("  silu_mul:                    %6.3f ms  (%4.1f%%)\n", ms(2,3), 100*ms(2,3)/mlp_total);
        printf("  down_proj (Marlin 17408→5120):%6.3f ms  (%4.1f%%)\n", ms(3,4), 100*ms(3,4)/mlp_total);
        printf("  elementwise_add:             %6.3f ms  (%4.1f%%)\n", ms(4,5), 100*ms(4,5)/mlp_total);
        printf("  TOTAL (1 MLP layer):         %6.3f ms  (×64 = %.1f ms)\n", mlp_total, mlp_total*64);
    }

    // === FA Sub-layer (one layer) ===
    if (first_fa >= 0) {
        const auto& attn = model.layers[first_fa].full_attn;
        const __half* x = state.norm_out;
        __half* Qsep = state.dn_qkv;
        __half* Gate = state.mlp_gate;
        int e = 0;
        cudaEventRecord(ev[e++], s);  // 0
        int8_linear_forward(x, attn.q_proj, state.q_buf, M, s);
        cudaEventRecord(ev[e++], s);  // 1
        int8_linear_forward(x, attn.kv_proj, state.attn_out, M, s);
        cudaMemcpy2DAsync(state.kv_buf, MC::KV_PROJ_DIM * sizeof(__half),
                          state.attn_out, MC::FA_KV_DIM * sizeof(__half),
                          MC::KV_PROJ_DIM * sizeof(__half), M,
                          cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(state.dn_z, MC::KV_PROJ_DIM * sizeof(__half),
                          state.attn_out + MC::KV_PROJ_DIM, MC::FA_KV_DIM * sizeof(__half),
                          MC::KV_PROJ_DIM * sizeof(__half), M,
                          cudaMemcpyDeviceToDevice, s);
        cudaEventRecord(ev[e++], s);  // 2
        // split + norm + RoPE + KV write
        {
            dim3 grid_sq(MC::NUM_ATTN_HEADS, M);
            split_q_gate_batch_kernel<<<grid_sq, 256, 0, s>>>(
                state.q_buf, Qsep, Gate, M, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);
            head_norm(Qsep, attn.q_norm, Qsep,
                      M * MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, s);
            head_norm(state.kv_buf, attn.k_norm, state.kv_buf,
                      M * MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, s);
            int mh = (MC::NUM_ATTN_HEADS > MC::NUM_KV_HEADS) ? MC::NUM_ATTN_HEADS : MC::NUM_KV_HEADS;
            dim3 grid_r(mh, M);
            rope_batch_kernel<<<grid_r, MC::ROTARY_DIM / 2, 0, s>>>(
                Qsep, state.kv_buf, MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS,
                MC::HEAD_DIM, MC::ROTARY_DIM, pos_start, M, MC::ROPE_THETA);
            size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
            __half* k_c = kv_cache + (size_t)first_fa * 2 * kv_plane;
            __half* v_c = k_c + kv_plane;
            dim3 grid_kv(MC::NUM_KV_HEADS, M);
            kv_cache_write_batch_kernel<<<grid_kv, 256, 0, s>>>(
                state.kv_buf, state.dn_z, k_c, v_c,
                pos_start, M, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_HEADS);
        }
        cudaEventRecord(ev[e++], s);  // 3
        {
            float scale_val = 1.0f / sqrtf((float)MC::HEAD_DIM);
            dim3 grid_a(MC::NUM_ATTN_HEADS, M);
            size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
            __half* k_c = kv_cache + (size_t)first_fa * 2 * kv_plane;
            __half* v_c = k_c + kv_plane;
            prefill_attention_kernel<<<grid_a, MC::HEAD_DIM, 0, s>>>(
                Qsep, k_c, v_c, state.attn_out,
                pos_start, M, max_kv_len, MC::HEAD_DIM,
                MC::NUM_KV_GROUPS, scale_val);
        }
        cudaEventRecord(ev[e++], s);  // 4
        sigmoid_gate(state.attn_out, Gate, state.attn_out,
                     M * MC::ATTN_OUT_DIM, s);
        cudaEventRecord(ev[e++], s);  // 5
        int8_linear_forward(state.attn_out, attn.o_proj, state.norm_out, M, s);
        cudaEventRecord(ev[e++], s);  // 6
        cudaStreamSynchronize(s);

        float fa_total = ms(0, 6);
        printf("\n=== FA Sub-layer (layer %d, M=%d) ===\n", first_fa, M);
        printf("  q_proj    (INT8 5120→12288): %6.3f ms  (%4.1f%%)\n", ms(0,1), 100*ms(0,1)/fa_total);
        printf("  kv_proj   (INT8 5120→1024×2):%6.3f ms  (%4.1f%%)\n", ms(1,2), 100*ms(1,2)/fa_total);
        printf("  split+norm+RoPE+kvwrite:     %6.3f ms  (%4.1f%%)\n", ms(2,3), 100*ms(2,3)/fa_total);
        printf("  attention (causal):          %6.3f ms  (%4.1f%%)\n", ms(3,4), 100*ms(3,4)/fa_total);
        printf("  sigmoid_gate:                %6.3f ms  (%4.1f%%)\n", ms(4,5), 100*ms(4,5)/fa_total);
        printf("  o_proj    (INT8 6144→5120):  %6.3f ms  (%4.1f%%)\n", ms(5,6), 100*ms(5,6)/fa_total);
        printf("  TOTAL (1 FA layer):          %6.3f ms  (×16 = %.1f ms)\n", fa_total, fa_total*16);
    }

    for (int i = 0; i < NEV; i++) cudaEventDestroy(ev[i]);
}

// ============================================================================
// Profile prefill: records events at component boundaries, sync once at end.
// No pipeline drain — events are async timestamps.
// ============================================================================

void profile_forward_prefill(const ModelWeights& model,
                             InferenceState& state,
                             __half* kv_cache,
                             const int* h_token_ids, int M,
                             int pos_start, int max_kv_len,
                             cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // 4 events per layer + 1 final = 257
    constexpr int EPL = 4;
    constexpr int N_EV = MC::NUM_LAYERS * EPL + 1;
    cudaEvent_t ev[N_EV];
    for (int i = 0; i < N_EV; i++)
        cudaEventCreateWithFlags(&ev[i], cudaEventDefault);

    // Setup (same as forward_prefill)
    cudaMemcpyAsync(state.token_ids, h_token_ids, M * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, M, MC::HIDDEN_SIZE, s);
    cudaMemcpyAsync(state.residual, state.hidden,
                    (size_t)M * MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, s);

    int dn_layer_idx = 0;

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];
        int base = layer * EPL;

        // ev[0]: before pre_norm
        cudaEventRecord(ev[base + 0], s);

        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // ev[1]: after pre_norm, before attn/DN
        cudaEventRecord(ev[base + 1], s);

        if (lw.is_full_attention) {
            full_attention_prefill(state.norm_out, lw.full_attn,
                                   kv_cache, layer, pos_start, M, max_kv_len,
                                   state, s);
        } else {
            deltanet_prefill(state.norm_out, lw.delta_net,
                             dn_layer_idx, M, state, s);
            dn_layer_idx++;
        }

        // ev[2]: after attn/DN, before residual+post_norm
        cudaEventRecord(ev[base + 2], s);

        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // ev[3]: after post_norm, before MLP
        cudaEventRecord(ev[base + 3], s);

        mlp_forward_prefill(state.norm_out, lw.mlp, state.residual, M, state, s);
    }

    // Final event after last MLP
    cudaEventRecord(ev[N_EV - 1], s);

    // Single sync point — no pipeline drain during the loop
    cudaStreamSynchronize(s);

    // Aggregate
    float t_norm = 0, t_dn = 0, t_fa = 0, t_mlp = 0;
    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];
        int base = layer * EPL;
        float ms;

        cudaEventElapsedTime(&ms, ev[base + 0], ev[base + 1]);
        t_norm += ms;

        cudaEventElapsedTime(&ms, ev[base + 1], ev[base + 2]);
        if (lw.is_full_attention) t_fa += ms;
        else t_dn += ms;

        cudaEventElapsedTime(&ms, ev[base + 2], ev[base + 3]);
        t_norm += ms;

        // MLP end = next layer's ev[0], or final event
        cudaEventElapsedTime(&ms, ev[base + 3], ev[base + EPL]);
        t_mlp += ms;
    }

    float total;
    cudaEventElapsedTime(&total, ev[0], ev[N_EV - 1]);

    printf("\n=== Prefill profile (M=%d, %d layers) ===\n", M, MC::NUM_LAYERS);
    printf("  DeltaNet SSM (48 layers):     %7.2f ms  (%4.1f%%)\n", t_dn, 100*t_dn/total);
    printf("  Full Attention (16 layers):   %7.2f ms  (%4.1f%%)\n", t_fa, 100*t_fa/total);
    printf("  MLP Marlin (64 layers):       %7.2f ms  (%4.1f%%)\n", t_mlp, 100*t_mlp/total);
    printf("  Norms (pre+post, 64 layers):  %7.2f ms  (%4.1f%%)\n", t_norm, 100*t_norm/total);
    printf("  Total (layers only):          %7.2f ms\n", total);
    printf("  Per token (M=%d):             %7.2f ms/tok\n", M, total / M);

    for (int i = 0; i < N_EV; i++) cudaEventDestroy(ev[i]);
}

// ============================================================================
// Profile forward pass — time each component type across all layers
// ============================================================================

void profile_forward(const ModelWeights& model,
                     InferenceState& state,
                     __half* kv_cache,
                     int token_id, int pos, int max_kv_len,
                     cudaStream_t stream) {
    using MC = ModelConfig;

    // Run a warmup token first to populate caches etc.
    forward_one_token(model, state, kv_cache, token_id, pos, max_kv_len, stream);

    // Now profile at pos+1
    pos++;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    auto timed = [&](const char* label, auto fn) {
        cudaEventRecord(e0, stream);
        fn();
        cudaEventRecord(e1, stream);
        cudaStreamSynchronize(stream);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        printf("  %-32s  %7.2f ms\n", label, ms);
        return ms;
    };

    printf("\n=== Forward pass profile (pos=%d) ===\n", pos);

    // Setup
    cudaMemcpyAsync(state.token_ids, &token_id, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(state.d_pos, &pos, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, 1, MC::HIDDEN_SIZE, stream);
    cudaMemcpyAsync(state.residual, state.hidden,
                    MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    float total_dn_attn = 0, total_fa_attn = 0, total_mlp = 0, total_norm = 0;
    int dn_layer_idx = 0;

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        // Norm + Attn + Norm + MLP timed as a unit per layer type
        if (layer == 0 || layer == 3) {
            // Time individual components for first DeltaNet (layer0) and first FullAttn (layer3)
            const char* lt = lw.is_full_attention ? "FullAttn" : "DeltaNet";
            char buf[64];

            snprintf(buf, sizeof(buf), "L%d %s pre_norm", layer, lt);
            timed(buf, [&]{ rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                                      1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream); });

            snprintf(buf, sizeof(buf), "L%d %s attn", layer, lt);
            float attn_ms;
            if (lw.is_full_attention) {
                attn_ms = timed(buf, [&]{ full_attention_forward(state.norm_out, lw.full_attn,
                                    kv_cache, layer, pos, max_kv_len, state, stream, true); });
                total_fa_attn += attn_ms;
            } else {
                attn_ms = timed(buf, [&]{ deltanet_forward(state.norm_out, lw.delta_net,
                                    dn_layer_idx, state, stream); });
                dn_layer_idx++;
                total_dn_attn += attn_ms;
            }

            elementwise_add(state.residual, state.norm_out, state.residual,
                            MC::HIDDEN_SIZE, stream);

            snprintf(buf, sizeof(buf), "L%d %s post_norm", layer, lt);
            timed(buf, [&]{ rms_norm(state.residual, lw.post_attn_layernorm, state.norm_out,
                                      1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream); });

            snprintf(buf, sizeof(buf), "L%d %s mlp", layer, lt);
            float mlp_ms = timed(buf, [&]{ mlp_forward(state.norm_out, lw.mlp, nullptr, state, stream); });
            total_mlp += mlp_ms;

            elementwise_add(state.residual, state.mlp_down, state.residual,
                            MC::HIDDEN_SIZE, stream);
        } else {
            // Bulk timing for remaining layers
            cudaEventRecord(e0, stream);
            rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                     1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);
            if (lw.is_full_attention) {
                full_attention_forward(state.norm_out, lw.full_attn,
                                       kv_cache, layer, pos, max_kv_len, state, stream);
            } else {
                deltanet_forward(state.norm_out, lw.delta_net,
                                 dn_layer_idx, state, stream);
                dn_layer_idx++;
            }
            elementwise_add(state.residual, state.norm_out, state.residual,
                            MC::HIDDEN_SIZE, stream);
            rms_norm(state.residual, lw.post_attn_layernorm, state.norm_out,
                     1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);
            mlp_forward(state.norm_out, lw.mlp, state.residual, state, stream);
            cudaEventRecord(e1, stream);
            cudaStreamSynchronize(stream);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            // Estimate split: ~70% attn, ~30% MLP based on weight sizes
            if (lw.is_full_attention) total_fa_attn += ms * 0.55f;
            else total_dn_attn += ms * 0.55f;
            total_mlp += ms * 0.35f;
            total_norm += ms * 0.10f;
        }
    }

    // LM head
    Linear lm_head_linear;
    lm_head_linear.weight = model.lm_head;
    lm_head_linear.in_features = MC::HIDDEN_SIZE;
    lm_head_linear.out_features = MC::VOCAB_SIZE;
    float lm_ms = timed("lm_head", [&]{ linear_forward(state.hidden, lm_head_linear, state.logits, 1, stream); });

    float sample_ms = timed("greedy_sample", [&]{ greedy_sample(state.logits, MC::VOCAB_SIZE, state.sample_out, stream); });

    printf("\n  --- Summary ---\n");
    printf("  DeltaNet attn (48 layers):    %7.1f ms\n", total_dn_attn);
    printf("  FullAttn (16 layers):         %7.1f ms\n", total_fa_attn);
    printf("  MLP (64 layers):              %7.1f ms\n", total_mlp);
    printf("  Norms:                        %7.1f ms\n", total_norm);
    printf("  LM Head:                      %7.1f ms\n", lm_ms);
    printf("  Greedy sample:                %7.1f ms\n", sample_ms);
    printf("  Estimated total:              %7.1f ms\n",
           total_dn_attn + total_fa_attn + total_mlp + total_norm + lm_ms + sample_ms);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

} // namespace deusridet
