// forward.cu — Qwen3.5-27B forward pass implementation
//
// Single-token decode path. Each function assumes M=1.
// Target: SM87 (Jetson AGX Orin)

#include "forward.h"
#include "layer.h"
#include "gptq.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

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
                 InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // gate_proj: x[5120] → mlp_gate[17408]
    gptq_linear(x, mlp.gate_proj, state.mlp_gate, 1, stream);

    // up_proj: x[5120] → mlp_up[17408]
    gptq_linear(x, mlp.up_proj, state.mlp_up, 1, stream);

    // SiLU(gate) * up → mlp_gate (reuse buffer)
    silu_inplace(state.mlp_gate, MC::INTERMEDIATE_SIZE, stream);
    elementwise_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                    MC::INTERMEDIATE_SIZE, stream);

    // down_proj: mlp_gate[17408] → mlp_down[5120]
    gptq_linear(state.mlp_gate, mlp.down_proj, state.mlp_down, 1, stream);
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
                            InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // 1. Q projection — q_proj outputs [12288] = Q[6144] + gate[6144]
    linear_forward(x, attn.q_proj, state.q_buf, 1, stream);

    // Split: first half is Q, second half is gate
    // q_buf[0..6143] = Q (24 heads * 256 dim)
    // q_buf[6144..12287] = gate (24 heads * 256 dim)
    __half* q_ptr    = state.q_buf;                     // [24, 256]
    __half* gate_ptr = state.q_buf + MC::ATTN_OUT_DIM;  // [24, 256]

    // 2. K, V projections
    linear_forward(x, attn.k_proj, state.kv_buf, 1, stream);
    // Need V in a separate buffer — reuse dn_z (not used in full attention layers)
    __half* v_buf = state.dn_z;  // [kv_proj_dim=1024]
    linear_forward(x, attn.v_proj, v_buf, 1, stream);

    // 3. Per-head Q/K RMSNorm
    head_norm(q_ptr, attn.q_norm, q_ptr,
              MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(state.kv_buf, attn.k_norm, state.kv_buf,
              MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);

    // 4. Partial RoPE on Q and K
    apply_rope(q_ptr, state.kv_buf,
               MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS, MC::HEAD_DIM,
               MC::ROTARY_DIM, pos, MC::ROPE_THETA, stream);

    // 5. Append K,V to cache
    // K cache: [num_kv_heads, max_kv_len, head_dim]
    // V cache: offset by num_kv_heads * max_kv_len * head_dim
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    __half* k_cache = kv_cache + (size_t)layer_idx * 2 * kv_plane;
    __half* v_cache = k_cache + kv_plane;

    // Copy current K into cache at position pos
    // K: [4, 256] → K_cache[h, pos, :] for each head h
    for (int h = 0; h < MC::NUM_KV_HEADS; h++) {
        cudaMemcpyAsync(
            k_cache + (size_t)h * max_kv_len * MC::HEAD_DIM + (size_t)pos * MC::HEAD_DIM,
            state.kv_buf + h * MC::HEAD_DIM,
            MC::HEAD_DIM * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(
            v_cache + (size_t)h * max_kv_len * MC::HEAD_DIM + (size_t)pos * MC::HEAD_DIM,
            v_buf + h * MC::HEAD_DIM,
            MC::HEAD_DIM * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
    }

    // 6. GQA attention: Q[24,256] @ K[4, pos+1, 256]^T → scores → softmax → V
    // Use cuBLAS for the matmuls. Each Q head attends to its KV group.
    // For single-token decode, this is essentially 24 dot products per cached token.
    // We compute this head-by-head on CPU to keep it simple for the skeleton.

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    int seq_len_kv = pos + 1;  // total cached tokens including current

    // Allocate temporary score buffer on device: [num_attn_heads, seq_len_kv] in FP32
    float* scores = nullptr;
    cudaMalloc(&scores, (size_t)MC::NUM_ATTN_HEADS * seq_len_kv * sizeof(float));

    // For each Q head group, compute Q @ K^T
    // GQA: heads 0-5 use KV head 0, heads 6-11 use KV head 1, etc.
    float alpha = 1.0f / sqrtf((float)MC::HEAD_DIM);
    float beta_zero = 0.0f;

    for (int kv_h = 0; kv_h < MC::NUM_KV_HEADS; kv_h++) {
        int q_start = kv_h * MC::NUM_KV_GROUPS;
        int num_q = MC::NUM_KV_GROUPS;  // 6 Q heads per KV head

        // Q_group: [num_q, head_dim] — these are contiguous in q_ptr
        // K_head:  [seq_len_kv, head_dim] — K_cache[kv_h, :pos+1, :]
        // scores_group: [num_q, seq_len_kv]

        const __half* Q_group = q_ptr + q_start * MC::HEAD_DIM;
        const __half* K_head  = k_cache + (size_t)kv_h * max_kv_len * MC::HEAD_DIM;
        float* scores_group   = scores + q_start * seq_len_kv;

        // S = Q @ K^T → [num_q, seq_len_kv]
        // In cuBLAS col-major: S^T = K @ Q^T
        //   K: [seq_len_kv, head_dim] row-major → col-major [head_dim, seq_len_kv]
        //   Q: [num_q, head_dim] row-major → col-major [head_dim, num_q]
        // cublasSgemm: C = alpha * A * B + beta * C
        //   where A = K^T (trans), B = Q (no trans)... this gets confusing.
        // Let's use GemmEx with FP16 input and FP32 output.

        // Actually, cublasGemmEx: C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C
        // We want: scores_group[seq_len_kv, num_q] (col-major of [num_q, seq_len_kv] row-major)
        // = alpha * K[seq_len_kv, head_dim] * Q^T[head_dim, num_q]
        // So: M=seq_len_kv, N=num_q, K=head_dim
        // A=K (no trans), ld=head_dim... but wait, K is stored as [max_kv_len, head_dim]
        // and we only want first seq_len_kv rows. In col-major, K is [head_dim, max_kv_len].
        // We want CUBLAS_OP_T on K to get [seq_len_kv, head_dim] from [head_dim, seq_len_kv].
        // Actually no. Let me think more carefully.

        // K_head row-major: [max_kv_len, head_dim], padded rows, using first seq_len_kv rows
        // In col-major interpretation: it's [head_dim, max_kv_len] with ld=head_dim
        // Actually col-major means we read column by column. A row-major [M,N] matrix
        // stored linearly is the same as col-major [N,M] with ld=N.

        // K_head row-major [seq_len_kv, head_dim]:
        //   col-major interpretation: [head_dim, seq_len_kv], ld=head_dim
        //   but actual storage stride between columns is head_dim (correct if contiguous)
        //   Wait, it's actually stored with stride max_kv_len*head_dim between heads,
        //   but within each head, rows are stride head_dim.
        //   So K_head[i,j] = K_head + i*head_dim + j → col-major [head_dim, seq_len_kv], ld=head_dim? No.
        //   K_head[row, col] at address K_head + row * head_dim + col.
        //   In column-major, this is matrix [head_dim, seq_len_kv] with ld=head_dim.
        //   Wait: for col-major, element (i,j) = base + j * ld + i.
        //   So if we say this is col-major [head_dim, max_kv_len], then element (i,j) = base + j*head_dim + i.
        //   That means physical(row=r, col=c) = base + r*head_dim + c matches (i=c, j=r).
        //   So row-major [rows, cols] = col-major [cols, rows] with ld=cols.

        // K: row-major [seq_kv, hd] = col-major [hd, seq_kv], ld=hd
        //    But the actual stride may be max_kv_len * hd per head... no, within one head,
        //    K is contiguous [max_kv_len, hd]. We only use first seq_kv rows.
        //    Col-major: [hd, max_kv_len], ld=hd, but we only read N=seq_kv columns.
        //    Stride between column j and j+1 = hd. Wait but max_kv_len rows...
        //    Row-major stride between rows = hd. Col-major ld = hd for [hd, max_kv_len].
        //    Hmm, ld = hd is the stride between columns. Each column has hd elements.
        //    This works because row-major [max_kv_len, hd] is contiguous.

        // Q_group: row-major [num_q, hd] = col-major [hd, num_q], ld=hd

        // We want: scores = Q @ K^T → [num_q, seq_kv]
        // col-major scores: [seq_kv, num_q], ld=seq_kv

        // scores^T = K @ Q^T
        // In cuBLAS: C = alpha * A * B + beta * C
        // C = scores: col-major [seq_kv, num_q], M=seq_kv, N=num_q
        // A = K: we have col-major [hd, seq_kv]. We need A to be [seq_kv, hd] → op=CUBLAS_OP_T on [hd, seq_kv] → gives [seq_kv, hd]. K=hd. ld_A = hd.
        //   Wait, op(A) must be [M,K] = [seq_kv, hd]. A is stored col-major [hd, seq_kv], ld=hd.
        //   CUBLAS_OP_T gives A^T = [seq_kv, hd]. Yes!
        // B = Q: we need op(B) = [hd, num_q]. B is col-major [hd, num_q], ld=hd.
        //   CUBLAS_OP_N gives [hd, num_q]. Yes!

        // cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_kv, num_q, hd,
        //              &alpha, K_head, CUDA_R_16F, hd, Q_group, CUDA_R_16F, hd,
        //              &beta, scores_group, CUDA_R_32F, seq_kv, CUDA_R_32F, ...)

        cublasGemmEx(handle,
                     CUBLAS_OP_T,     // A^T: K^T
                     CUBLAS_OP_N,     // B: Q
                     seq_len_kv, num_q, MC::HEAD_DIM,
                     &alpha,
                     K_head,      CUDA_R_16F, MC::HEAD_DIM,  // A = K col-major [hd, max_kv], ld=hd
                     Q_group,     CUDA_R_16F, MC::HEAD_DIM,  // B = Q col-major [hd, num_q], ld=hd
                     &beta_zero,
                     scores_group, CUDA_R_32F, seq_len_kv,   // C = [seq_kv, num_q], ld=seq_kv
                     CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // Softmax on scores: [num_attn_heads, seq_len_kv]
    softmax(scores, scores, MC::NUM_ATTN_HEADS, seq_len_kv, stream);

    // Attention output: attn_out[h, :] = scores[h, :] @ V_cache[kv_h(h), :seq_kv, :]
    // attn_out: [num_attn_heads, head_dim] = [24, 256]

    for (int kv_h = 0; kv_h < MC::NUM_KV_HEADS; kv_h++) {
        int q_start = kv_h * MC::NUM_KV_GROUPS;
        int num_q = MC::NUM_KV_GROUPS;

        const float* scores_group = scores + q_start * seq_len_kv;
        const __half* V_head = v_cache + (size_t)kv_h * max_kv_len * MC::HEAD_DIM;
        __half* out_group = state.attn_out + q_start * MC::HEAD_DIM;

        __half* scores_h16 = nullptr;
        cudaMalloc(&scores_h16, (size_t)num_q * seq_len_kv * sizeof(__half));        {
            size_t n = (size_t)num_q * seq_len_kv;
            std::vector<float> h_f32(n);
            std::vector<uint16_t> h_f16(n);
            cudaMemcpy(h_f32.data(), scores_group, n * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < n; i++) {
                __half h = __float2half(h_f32[i]);
                memcpy(&h_f16[i], &h, sizeof(uint16_t));
            }
            cudaMemcpy(scores_h16, h_f16.data(), n * sizeof(__half),
                       cudaMemcpyHostToDevice);
        }

        __half alpha_h = __float2half(1.0f);
        __half beta_h  = __float2half(0.0f);

        // C[hd, num_q] = V[hd, seq_kv] * scores[seq_kv, num_q]
        cublasGemmEx(handle,
                     CUBLAS_OP_N,     // V: col-major [hd, max_kv], no trans
                     CUBLAS_OP_N,     // scores: col-major [seq_kv, num_q]
                     MC::HEAD_DIM, num_q, seq_len_kv,
                     &alpha_h,
                     V_head,      CUDA_R_16F, MC::HEAD_DIM,
                     scores_h16,  CUDA_R_16F, seq_len_kv,
                     &beta_h,
                     out_group,   CUDA_R_16F, MC::HEAD_DIM,
                     CUDA_R_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        cudaFree(scores_h16);
    }

    cudaFree(scores);

    // 7. Attention gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, gate_ptr, state.attn_out,
                 MC::ATTN_OUT_DIM, stream);

    // 8. o_proj: attn_out[6144] → hidden[5120]
    // Store in attn_out temporarily, actual output goes to mlp_down for reuse
    // Actually, write to attn_out buffer. The caller handles residual add.
    // But attn_out is [6144] and o_proj output is [5120]... use a separate buf.
    // Let's write o_proj output to norm_out (reusable scratch).
    linear_forward(state.attn_out, attn.o_proj, state.norm_out, 1, stream);

    // Copy o_proj result to attn_out head area (actually, the orchestrator will
    // read from norm_out for the residual add). We'll use a convention:
    // After full_attention_forward, the attention output is in state.norm_out[0..5119].
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
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float s_inv;
    if (threadIdx.x == 0)
        s_inv = rsqrtf(sum_sq + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        h[i] = __float2half(__half2float(h[i]) * s_inv);
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
    extern __shared__ float smem[];     // [k_dim + v_dim + v_dim]
    float* sk = smem;                   // [k_dim]
    float* sv = sk + k_dim;             // [v_dim]
    float* s_kv_mem = sv + v_dim;       // [v_dim]

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

    for (int j = tid; j < v_dim; j += blockDim.x) {
        // Compute kv_mem[j]
        float mem = 0.0f;
        for (int i = 0; i < k_dim; i++) {
            float s_ij = S[i * v_dim + j];
            s_ij *= g_scalar;  // decay in-place
            S[i * v_dim + j] = s_ij;
            mem += s_ij * sk[i];
        }
        s_kv_mem[j] = mem;
    }
    __syncthreads();

    // Compute delta and update state, then output
    for (int j = tid; j < v_dim; j += blockDim.x) {
        float delta = (sv[j] - s_kv_mem[j]) * beta_scalar;

        float out_j = 0.0f;
        for (int i = 0; i < k_dim; i++) {
            float update = sk[i] * delta;
            S[i * v_dim + j] += update;
            out_j += S[i * v_dim + j] * __half2float(q_h[i]);
        }
        out_h[j] = __float2half(out_j);
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

    // 1. QKV projection
    linear_forward(x, dn.in_proj_qkv, state.dn_qkv, 1, stream);

    // Conv1d step (also applies SiLU)
    causal_conv1d_step(state.dn_qkv, state.conv_states[dn_layer_idx],
                       dn.conv1d_weight, state.dn_qkv,
                       MC::LIN_CONV_DIM, MC::CONV_KERNEL, stream);

    // SiLU on conv output
    silu_inplace(state.dn_qkv, MC::LIN_CONV_DIM, stream);

    // 2-3. Split qkv: the layout is [KEY_DIM, KEY_DIM, VALUE_DIM] = [2048, 2048, 6144]
    // query = dn_qkv[0..2047]           → [16, 128]
    // key   = dn_qkv[2048..4095]        → [16, 128]
    // value = dn_qkv[4096..10239]       → [48, 128]
    __half* dn_query = state.dn_qkv;                         // [16, 128]
    __half* dn_key   = state.dn_qkv + MC::LIN_KEY_DIM;       // [16, 128]
    __half* dn_value = state.dn_qkv + 2 * MC::LIN_KEY_DIM;   // [48, 128]

    // 4. repeat_interleave: q,k from 16 heads → 48 heads (ratio 3)
    // We need expanded q,k of size [48, 128]. Use attn_out as temporary.
    // attn_out is [6144], can hold [48, 128].
    __half* q_expanded = state.attn_out;           // [48, 128]
    __half* k_expanded = state.q_buf;              // [48, 128] (reuse q_buf, large enough)

    // Repeat interleave on GPU: each of the 16 heads gets repeated 3 times
    // Simple approach: cudaMemcpy in a loop
    int ratio = MC::LIN_NUM_V_HEADS / MC::LIN_NUM_K_HEADS;  // 3
    for (int h = 0; h < MC::LIN_NUM_K_HEADS; h++) {
        for (int r = 0; r < ratio; r++) {
            int dst_h = h * ratio + r;
            cudaMemcpyAsync(q_expanded + dst_h * MC::LIN_K_HEAD_DIM,
                            dn_query + h * MC::LIN_K_HEAD_DIM,
                            MC::LIN_K_HEAD_DIM * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(k_expanded + dst_h * MC::LIN_K_HEAD_DIM,
                            dn_key + h * MC::LIN_K_HEAD_DIM,
                            MC::LIN_K_HEAD_DIM * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // 5. Projections for z, a, b
    linear_forward(x, dn.in_proj_z, state.dn_z, 1, stream);           // → [6144]
    linear_forward(x, dn.in_proj_a, state.dn_a, 1, stream);           // → [48] FP16
    linear_forward(x, dn.in_proj_b, state.dn_b, 1, stream);           // → [48] FP16

    // 6. Compute g and beta on CPU (only 48 values each)
    // g = -exp(A_log) * softplus(a + dt_bias)
    // beta = sigmoid(b)
    float g_host[MC::LIN_NUM_V_HEADS];
    float beta_host[MC::LIN_NUM_V_HEADS];
    float A_log_host[MC::LIN_NUM_V_HEADS];
    __half dt_bias_host_h[MC::LIN_NUM_V_HEADS];
    __half a_host_h[MC::LIN_NUM_V_HEADS];
    __half b_host_h[MC::LIN_NUM_V_HEADS];

    cudaStreamSynchronize(stream);  // Need values on host

    cudaMemcpy(a_host_h, state.dn_a, MC::LIN_NUM_V_HEADS * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host_h, state.dn_b, MC::LIN_NUM_V_HEADS * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(A_log_host, dn.A_log, MC::LIN_NUM_V_HEADS * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dt_bias_host_h, dn.dt_bias, MC::LIN_NUM_V_HEADS * sizeof(__half),
               cudaMemcpyDeviceToHost);

    for (int h = 0; h < MC::LIN_NUM_V_HEADS; h++) {
        float a_val = __half2float(a_host_h[h]);
        float b_val = __half2float(b_host_h[h]);
        float dt_b  = __half2float(dt_bias_host_h[h]);
        float A_val = expf(A_log_host[h]);

        float x_ab = a_val + dt_b;
        float softplus = (x_ab > 20.0f) ? x_ab : logf(1.0f + expf(x_ab));
        g_host[h] = -A_val * softplus;
        beta_host[h] = 1.0f / (1.0f + expf(-b_val));
    }

    // Copy g and beta to device
    float* g_dev = nullptr;
    float* beta_dev = nullptr;
    cudaMalloc(&g_dev, MC::LIN_NUM_V_HEADS * sizeof(float));
    cudaMalloc(&beta_dev, MC::LIN_NUM_V_HEADS * sizeof(float));
    cudaMemcpyAsync(g_dev, g_host, MC::LIN_NUM_V_HEADS * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(beta_dev, beta_host, MC::LIN_NUM_V_HEADS * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 7. L2 normalize q and k (per head)
    l2norm_kernel<<<MC::LIN_NUM_V_HEADS, 128, 0, stream>>>(
        q_expanded, MC::LIN_K_HEAD_DIM, 1e-6f);
    l2norm_kernel<<<MC::LIN_NUM_V_HEADS, 128, 0, stream>>>(
        k_expanded, MC::LIN_K_HEAD_DIM, 1e-6f);

    // 8. Scale q by 1/sqrt(k_dim) — host roundtrip for 6144 values
    float scale = 1.0f / sqrtf((float)MC::LIN_K_HEAD_DIM);
    {
        size_t n = MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM;
        std::vector<uint16_t> q_host(n);
        cudaMemcpy(q_host.data(), q_expanded, n * sizeof(__half), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; i++) {
            __half h;
            memcpy(&h, &q_host[i], sizeof(__half));
            h = __float2half(__half2float(h) * scale);
            memcpy(&q_host[i], &h, sizeof(__half));
        }
        cudaMemcpyAsync(q_expanded, q_host.data(), n * sizeof(__half),
                        cudaMemcpyHostToDevice, stream);
    }

    // 9. Recurrent step
    int smem_size = (MC::LIN_K_HEAD_DIM + MC::LIN_V_HEAD_DIM + MC::LIN_V_HEAD_DIM) * sizeof(float);
    deltanet_recurrent_kernel<<<MC::LIN_NUM_V_HEADS, 128, smem_size, stream>>>(
        q_expanded, k_expanded, dn_value,
        g_dev, beta_dev,
        state.dn_states[dn_layer_idx],
        state.attn_out,  // output [48, 128] = [6144]
        MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM);

    cudaFree(g_dev);
    cudaFree(beta_dev);

    // 10. Gated RMSNorm: out[48, 128] with gate z[48, 128]
    // attn_out holds the output [6144], dn_z holds z [6144]
    // Reshape conceptually as [48 rows, 128 dim]
    rms_norm_gated(state.attn_out, state.dn_z,
                   dn.norm_weight, state.attn_out,
                   MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, stream);

    // 11. out_proj: [6144] → [5120], write to norm_out (same convention as full_attn)
    linear_forward(state.attn_out, dn.out_proj, state.norm_out, 1, stream);
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

int forward_one_token(const ModelWeights& model,
                      InferenceState& state,
                      __half* kv_cache,
                      int token_id, int pos, int max_kv_len,
                      cudaStream_t stream) {
    using MC = ModelConfig;

    // Copy token ID to device
    cudaMemcpyAsync(state.token_ids, &token_id, sizeof(int),
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
                                   kv_cache, layer, pos, max_kv_len,
                                   state, stream);
            // Output in state.norm_out
        } else {
            deltanet_forward(state.norm_out, lw.delta_net,
                             dn_layer_idx, state, stream);
            dn_layer_idx++;
            // Output in state.norm_out
        }

        // Residual connection: residual = residual + attn_output
        elementwise_add(state.residual, state.norm_out, state.residual,
                        MC::HIDDEN_SIZE, stream);

        // Post-attention RMSNorm
        rms_norm(state.residual, lw.post_attn_layernorm, state.norm_out,
                 1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

        // MLP
        mlp_forward(state.norm_out, lw.mlp, state, stream);
        // Output in state.mlp_down

        // Residual connection: residual = residual + mlp_output
        elementwise_add(state.residual, state.mlp_down, state.residual,
                        MC::HIDDEN_SIZE, stream);
    }

    // Final RMSNorm
    rms_norm(state.residual, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

    // LM head: hidden[5120] → logits[248320]
    // lm_head is [vocab_size, hidden_size] — same as Linear
    Linear lm_head_linear;
    lm_head_linear.weight = model.lm_head;
    lm_head_linear.in_features = MC::HIDDEN_SIZE;
    lm_head_linear.out_features = MC::VOCAB_SIZE;
    linear_forward(state.hidden, lm_head_linear, state.logits, 1, stream);

    // Synchronize before sampling
    cudaStreamSynchronize(stream);

    // Greedy sample
    return greedy_sample(state.logits, MC::VOCAB_SIZE);
}

} // namespace deusridet
