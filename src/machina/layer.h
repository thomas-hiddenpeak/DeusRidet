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

// Element-wise add: out = a + b
void elementwise_add(const __half* a, const __half* b, __half* out,
                     int n, cudaStream_t stream = 0);

// Sigmoid gate: out = x * sigmoid(gate)
void sigmoid_gate(const __half* x, const __half* gate, __half* out,
                  int n, cudaStream_t stream = 0);

// ============================================================================
// Linear (cuBLAS GEMM for non-quantized layers)
// ============================================================================

// Y[M,N] = X[M,K] @ W^T[K,N]  where W is [N,K] row-major
// Uses cuBLAS FP16 GEMM.
void linear_forward(const __half* X, const Linear& weight, __half* Y,
                    int M, cudaStream_t stream = 0);

// ============================================================================
// RoPE (partial rotary position embedding)
// ============================================================================

// Apply partial RoPE to q and k in-place.
// q[num_heads, head_dim], k[num_kv_heads, head_dim].
// Only first rotary_dim elements are rotated. Interleaved pairing.
// pos: absolute position index.
void apply_rope(__half* q, __half* k,
                int num_heads, int num_kv_heads, int head_dim,
                int rotary_dim, int pos, float theta,
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

// ============================================================================
// Sampling
// ============================================================================

// Greedy: return argmax of logits[vocab_size]
int greedy_sample(const __half* logits, int vocab_size);

} // namespace deusridet
