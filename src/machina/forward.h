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
// x[hidden_size], uses state scratch buffers, writes to mlp_down[hidden_size]
void mlp_forward(const __half* x, const MLPWeights& mlp,
                 InferenceState& state, cudaStream_t stream = 0);

// Full Attention (GQA) single-token decode step.
// x[hidden_size], uses q_buf/kv_buf/attn_out scratch, writes to attn_out.
// Appends K/V to kv_cache at position pos.
// kv_cache: [num_layers, 2, num_kv_heads, max_kv_len, head_dim]
// Returns output in state.attn_out[attn_out_dim=6144]
void full_attention_forward(const __half* x, const FullAttentionWeights& attn,
                            __half* kv_cache, int layer_idx,
                            int pos, int max_kv_len,
                            InferenceState& state, cudaStream_t stream = 0);

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

// Single-token forward pass through entire model.
// Writes logits to state.logits[vocab_size].
// Returns the sampled token ID (greedy).
int forward_one_token(const ModelWeights& model,
                      InferenceState& state,
                      __half* kv_cache,
                      int token_id, int pos, int max_kv_len,
                      cudaStream_t stream = 0);

} // namespace deusridet
