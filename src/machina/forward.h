// forward.h — Qwen3.5-27B forward pass skeleton
//
// Single-token decode path for initial testing.
// This file declares the forward pass orchestrator and per-layer functions.

#pragma once

#include "model.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {

// ============================================================================
// Per-layer forward functions (single-token decode, M=1)
// ============================================================================

// SwiGLU MLP: out = down_proj( silu(gate_proj(x)) * up_proj(x) )
// x[hidden_size], uses state scratch buffers.
// If residual != nullptr, fuses residual += down_proj output (eliminates separate add).
// Otherwise writes to mlp_down[hidden_size].
void mlp_forward(const __half* x, const MLPWeights& mlp,
                 __half* residual,
                 InferenceState& state, cudaStream_t stream = 0);

// Full Attention (GQA) single-token decode step.
// x[hidden_size], uses q_buf/kv_buf/attn_out scratch, writes to attn_out.
// Appends K/V to kv_cache at position pos.
// kv_cache: [num_layers, 2, num_kv_heads, max_kv_len, head_dim]
// Returns output in state.attn_out[attn_out_dim=6144]
void full_attention_forward(const __half* x, const FullAttentionWeights& attn,
                            __half* kv_cache, int layer_idx,
                            int pos, int max_kv_len,
                            InferenceState& state, cudaStream_t stream = 0,
                            bool trace = false);

// DeltaNet recurrent step (single-token decode).
// x[hidden_size], updates dn_states and conv_states in-place.
// dn_layer_idx: index into dn_states/conv_states arrays (0..47)
// Returns output in state.attn_out[6144] (reusing same buffer as full attn)
void deltanet_forward(const __half* x, const DeltaNetWeights& dn,
                      int dn_layer_idx,
                      InferenceState& state, cudaStream_t stream = 0);

// ============================================================================
// Complete forward pass
// ============================================================================

// Single-token forward pass through entire model (greedy sampling).
// Uses CUDA Graph acceleration: first call captures, subsequent calls replay.
// Returns the sampled token ID.
int forward_one_token(const ModelWeights& model,
                      InferenceState& state,
                      __half* kv_cache,
                      int token_id, int pos, int max_kv_len,
                      cudaStream_t stream = 0);

// Single-token forward pass with configurable sampling.
// Does NOT use CUDA Graph (sampling kernel changes with params).
// Returns the sampled token ID.
int forward_one_token_sampled(const ModelWeights& model,
                              InferenceState& state,
                              __half* kv_cache,
                              int token_id, int pos, int max_kv_len,
                              const SamplingParams& params,
                              cudaStream_t stream = 0);

// Profile per-component timing of a single forward pass.
// Prints detailed breakdown to stdout.
void profile_forward(const ModelWeights& model,
                     InferenceState& state,
                     __half* kv_cache,
                     int token_id, int pos, int max_kv_len,
                     cudaStream_t stream = 0);

// ============================================================================
// Batched prefill forward pass (M>1 tokens)
// ============================================================================

// Process M prompt tokens through the entire model in a single batched pass.
// Uses GEMM (M>1) for linear layers instead of per-token GEMV.
// Returns the sampled token ID from the last token's logits.
//
// token_ids: host array of M token IDs
// M: number of prompt tokens
// pos_start: starting position index (0 for initial prompt)
// max_kv_len: maximum KV cache length
int forward_prefill(const ModelWeights& model,
                    InferenceState& state,
                    __half* kv_cache,
                    const int* token_ids, int M,
                    int pos_start, int max_kv_len,
                    cudaStream_t stream = 0);

// Profile prefill: same as forward_prefill but records per-component event
// timestamps and prints breakdown. Events are non-blocking (no pipeline drain).
void profile_forward_prefill(const ModelWeights& model,
                             InferenceState& state,
                             __half* kv_cache,
                             const int* token_ids, int M,
                             int pos_start, int max_kv_len,
                             cudaStream_t stream = 0);

// Sub-layer profiler: per-operation timing within DN, FA, MLP.
// Call AFTER a warmup pass with buffers populated.
void profile_sublayer_prefill(const ModelWeights& model,
                              InferenceState& state,
                              __half* kv_cache,
                              int M, int pos_start, int max_kv_len,
                              cudaStream_t stream = 0);

} // namespace deusridet
