// layer.h — CUDA kernel declarations for Qwen3.5 layer operations
//
// All kernels operate on FP16 data (BF16 converted at load time).
// cuBLAS handles non-quantized matmuls; GPTQ kernels handle quantized MLP.

#pragma once

#include "model.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace deusridet {

// ============================================================================
// cuBLAS handle (singleton, created on first use)
// ============================================================================

cublasHandle_t get_cublas_handle();

// ============================================================================
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
// Weights are precomputed as (1+w) at load time for regular norms.
// ============================================================================

// In-place RMSNorm: x[rows, dim] *= weight[dim] / rms(x)
void rms_norm(const __half* x, const __half* weight, __half* out,
              int rows, int dim, float eps,
              cudaStream_t stream = 0);

// Fused Residual + RMSNorm: residual += x; out = norm(residual) * weight
// Saves a kernel launch + memory round-trip over separate add + norm.
void residual_rms_norm(__half* residual, const __half* x,
                       const __half* weight, __half* out,
                       int rows, int dim, float eps,
                       cudaStream_t stream = 0);

// Gated RMSNorm: out = weight * norm(x) * silu(gate)
// x[n, dim], gate[n, dim], weight[dim], out[n, dim]
void rms_norm_gated(const __half* x, const __half* gate,
                    const float* weight, __half* out,
                    int n, int dim, float eps,
                    cudaStream_t stream = 0);

// ============================================================================
// Embedding lookup
// ============================================================================

// out[seq_len, hidden] = embed_table[token_ids[i], :]
void embedding_lookup(const __half* embed_table, const int* token_ids,
                      __half* out, int seq_len, int hidden_size,
                      cudaStream_t stream = 0);

// ============================================================================
// Element-wise operations
// ============================================================================

// SiLU activation: x = x * sigmoid(x), in-place
void silu_inplace(__half* x, int n, cudaStream_t stream = 0);

// Element-wise multiply: out = a * b
void elementwise_mul(const __half* a, const __half* b, __half* out,
                     int n, cudaStream_t stream = 0);

// Fused SiLU + multiply: out = silu(gate) * up
void silu_mul(const __half* gate, const __half* up, __half* out,
              int n, cudaStream_t stream = 0);

// Element-wise add: out = a + b
void elementwise_add(const __half* a, const __half* b, __half* out,
                     int n, cudaStream_t stream = 0);

// Sigmoid gate: out = x * sigmoid(gate)
void sigmoid_gate(const __half* x, const __half* gate, __half* out,
                  int n, cudaStream_t stream = 0);

// ============================================================================
// Linear (cuBLAS GEMM for non-quantized layers)
// ============================================================================

// Y[N] = x[K] @ W^T[K,N]  where W is [N,K] row-major. Single-vector GEMV.
void fp16_gemv(const __half* x, const __half* W, __half* y,
               int K, int N, cudaStream_t stream = 0);

// Y[M,N] = X[M,K] @ W^T[K,N]  where W is [N,K] row-major
// Uses cuBLAS FP16 GEMM.
void linear_forward(const __half* X, const Linear& weight, __half* Y,
                    int M, cudaStream_t stream = 0);

// INT8 quantized linear forward: Y[M,N] = X[M,K] @ W_int8^T[K,N] * scales[N]
// For M=1: custom INT8 GEMV. For M>1: dequant to FP16 temp + cuBLAS.
void int8_linear_forward(const __half* X, const Int8Linear& weight, __half* Y,
                         int M, cudaStream_t stream = 0);

// Dual INT8 linear forward: compute two matmuls sharing the same x in SMEM.
// Y1 = w1 @ X, Y2 = w2 @ X. One kernel launch, one x load.
void int8_dual_linear_forward(const __half* X,
                               const Int8Linear& w1, __half* Y1,
                               const Int8Linear& w2, __half* Y2,
                               int M, cudaStream_t stream = 0);

// Quantize FP16 weights to INT8 (per-channel symmetric) on GPU.
// Allocates int8 weight + scales via cudaMalloc. src_fp16 is [N, K] row-major.
void quantize_fp16_to_int8(const __half* src_fp16, Int8Linear& dst,
                           int out_features, int in_features,
                           cudaStream_t stream = 0);

// Quantize INT8 per-channel weights to GPTQ INT4 per-group format on GPU.
// Allocates qweight + scales via cudaMalloc. INT8 weight is [N, K] row-major.
// Output: qweight [K/8, N] packed uint32, scales [K/group_size, N] FP16.
// The INT8 per-channel scale is folded into the per-group INT4 scale.
void quantize_int8_to_gptq_int4(const Int8Linear& src, GptqWeight& dst,
                                int group_size, cudaStream_t stream = 0);

// ============================================================================
// RoPE (partial rotary position embedding)
// ============================================================================

// Apply partial RoPE to q and k in-place.
// q[num_heads, head_dim], k[num_kv_heads, head_dim].
// Only first rotary_dim elements are rotated. Interleaved pairing.
// pos: absolute position index.
void apply_rope(__half* q, __half* k,
                int num_heads, int num_kv_heads, int head_dim,
                int rotary_dim, const int* d_pos, float theta,
                cudaStream_t stream = 0);

// ============================================================================
// Per-head RMSNorm for Q/K normalization
// ============================================================================

// Normalize each head independently: head[i] /= rms(head[i]) * weight[i%dim]
// x[num_heads, head_dim], weight[head_dim] (precomputed 1+w)
void head_norm(const __half* x, const __half* weight, __half* out,
               int num_heads, int head_dim, float eps,
               cudaStream_t stream = 0);

// ============================================================================
// Softmax (for attention scores)
// ============================================================================

// Row-wise softmax: out[rows, cols] = softmax(x[rows, cols])
void softmax(const float* x, float* out, int rows, int cols,
             cudaStream_t stream = 0);

// ============================================================================
// Causal conv1d step (single-token, for DeltaNet decode)
// ============================================================================

// Process one new input column through a causal 1D convolution.
// Updates conv_state in-place (shift + insert).
// x_in[conv_dim]: new input
// conv_state[conv_dim, kernel-1]: rolling state
// conv_weight[conv_dim, kernel]: weights
// x_out[conv_dim]: result (dot product along kernel dimension per channel)
void causal_conv1d_step(const __half* x_in, __half* conv_state,
                        const __half* conv_weight, __half* x_out,
                        int conv_dim, int kernel_size,
                        cudaStream_t stream = 0);

// Fused conv1d step + SiLU: conv → silu in registers (no intermediate write)
void causal_conv1d_step_silu(const __half* x_in, __half* conv_state,
                              const __half* conv_weight, __half* x_out,
                              int conv_dim, int kernel_size,
                              cudaStream_t stream = 0);

// ============================================================================
// Sampling
// ============================================================================

// Launch argmax kernel without sync (graph-capturable)
void argmax_async(const __half* logits, int vocab_size, int* d_token_out,
                  cudaStream_t stream = 0);

// Greedy: return argmax of logits[vocab_size] (includes D2H + sync)
int greedy_sample(const __half* logits, int vocab_size, int* d_token_out,
                  cudaStream_t stream = 0);

// Top-k / Top-p sampling (GPU, single block)
// probs_workspace: pre-allocated float[vocab_size] for intermediate probabilities.
// rng_seed: per-call seed for PRNG (caller should increment each call).
// Result written to d_token_out[0].
void sample_top_k_top_p(const __half* logits, float* probs_workspace,
                         int vocab_size, const SamplingParams& params,
                         unsigned long long rng_seed,
                         int* d_token_out, cudaStream_t stream = 0);

} // namespace deusridet
