// asr_ops.h — ASR-specific CUDA operator library
//
// Standalone operator set for Qwen3-ASR inference. Not shared with LLM kernels
// to allow independent optimization.
//
// Adapted from qwen35-orin (src/plugins/asr/audio_ops.h): audio-specific CUDA
// operators for ASR encoder/decoder inference.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

namespace deusridet {
namespace asr_ops {

// ============================================================================
// Normalization
// ============================================================================

// RMSNorm (plain weight): y = w * x * rsqrt(mean(x²) + eps)
void invoke_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    float eps, int num_tokens, int hidden_size, cudaStream_t stream = 0);

// LayerNorm (with bias): y = (x - mean) / sqrt(var + eps) * w + b
void invoke_layernorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                      float eps, int num_tokens, int hidden_size, cudaStream_t stream = 0);

// Per-head RMSNorm (plain weight)
void invoke_per_head_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                              const __nv_bfloat16* weight,
                              float eps, int num_tokens, int num_heads, int head_dim,
                              cudaStream_t stream = 0);

// Fused per-head QK RMSNorm + MRoPE (3 kernels → 1)
void invoke_fused_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_norm_w,
    const int* pos_ids,
    float eps,
    int num_tokens,
    int num_q_heads, int num_kv_heads,
    int head_dim,
    int s0, int s1, int s2,
    float theta,
    cudaStream_t stream = 0);

// ============================================================================
// Activations
// ============================================================================

// SwiGLU: out = silu(gate) * up
void invoke_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   int num_tokens, int intermediate_size, cudaStream_t stream = 0);

// GELU: out = x * 0.5 * (1 + erf(x / sqrt(2)))
void invoke_gelu(__nv_bfloat16* out, const __nv_bfloat16* x,
                 int num_elements, cudaStream_t stream = 0);

// ============================================================================
// Positional Encoding
// ============================================================================

// Sinusoidal PE (Whisper-style)
void compute_sinusoidal_pe(__nv_bfloat16* pe_out,
                           int max_positions, int d_model,
                           float max_timescale = 10000.0f,
                           cudaStream_t stream = 0);

// Add PE: hidden += pe[pos_offset : pos_offset + seq_len]
void invoke_add_pe(__nv_bfloat16* hidden_states,
                   const __nv_bfloat16* pe_table,
                   int seq_len, int hidden_size,
                   int pos_offset = 0,
                   cudaStream_t stream = 0);

// Per-chunk PE: each chunk independently uses PE[0..chunk_len-1]
void invoke_add_pe_chunked(__nv_bfloat16* hidden_states,
                           const __nv_bfloat16* pe_table,
                           int total_tokens, int hidden_size,
                           int chunk_len,
                           cudaStream_t stream = 0);

// MRoPE (Multimodal Rotary Position Embedding)
void invoke_mrope(__nv_bfloat16* q, __nv_bfloat16* k,
                  const int* pos_ids,
                  int num_tokens,
                  int num_q_heads, int num_kv_heads,
                  int head_dim,
                  int s0, int s1, int s2,
                  float theta = 1000000.0f,
                  cudaStream_t stream = 0);

// ============================================================================
// Attention
// ============================================================================

// Bidirectional MHA (ASR Encoder)
void invoke_bidirectional_mha(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    int total_tokens,
    int num_heads, int head_dim,
    const int* cu_seqlens,
    int num_segments,
    cudaStream_t stream = 0);

// Causal GQA Decode Attention (T=1, split-K)
void invoke_causal_gqa_decode(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int batch_size,
    int num_q_heads, int num_kv_heads, int head_dim,
    int current_seq_len,
    cudaStream_t stream = 0,
    float* attn_workspace = nullptr,
    int attn_max_partitions = 0);

// Causal GQA Prefill via cuBLAS GEMM
void invoke_causal_gqa_prefill_cublas(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* attn_score_buf,
    int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    cublasHandle_t handle, cudaStream_t stream = 0);

// ============================================================================
// Misc
// ============================================================================

// Embedding lookup: out[i] = table[ids[i]]
void invoke_embedding_lookup(__nv_bfloat16* out, const int* ids,
                              const __nv_bfloat16* table,
                              int num_tokens, int hidden_size,
                              cudaStream_t stream = 0);

// Residual add: a += b
void invoke_add_residual(__nv_bfloat16* a, const __nv_bfloat16* b,
                         int num_elements, cudaStream_t stream = 0);

// BF16 clamp (overflow protection)
void invoke_bf16_clamp(__nv_bfloat16* x, int num_elements,
                       float min_val, float max_val, cudaStream_t stream = 0);

// Write contiguous KV cache
void invoke_write_kv_cache(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                            const __nv_bfloat16* k, const __nv_bfloat16* v,
                            int start_pos, int num_tokens,
                            int num_kv_heads, int head_dim,
                            cudaStream_t stream = 0);

// GPU Argmax
void invoke_argmax(const __nv_bfloat16* logits, int* result_idx, int n,
                   cudaStream_t stream = 0);

// EOS suppression
void invoke_suppress_eos(__nv_bfloat16* logits, int eos_id1, int eos_id2,
                         cudaStream_t stream = 0);

// F32 → BF16 conversion
void invoke_f32_to_bf16(__nv_bfloat16* out, const float* in, int n,
                        cudaStream_t stream = 0);

// Repetition penalty
void invoke_repetition_penalty(__nv_bfloat16* logits, const int* token_ids,
                               int num_tokens, float penalty,
                               cudaStream_t stream = 0);

} // namespace asr_ops
} // namespace deusridet
