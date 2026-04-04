// model.h — Qwen3.5-27B model weight structures and forward pass
//
// Hybrid DeltaNet SSM + GQA architecture:
//   64 layers: 48 DeltaNet (linear_attention) + 16 Full Attention (every 4th)
//   GPTQ-Int4 quantized MLP, BF16 attention weights converted to FP16 at load
//   SwiGLU MLP, partial RoPE (25%), attention output gate, (1+w) RMSNorm
//
// Weight loading converts BF16→FP16 at load time for uniform FP16 compute.
// Regular RMSNorm weights are precomputed as (1+w) at load time.

#pragma once

#include "gptq.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace deusridet {

// ============================================================================
// Model configuration (from Qwen3.5-27B config.json)
// ============================================================================

struct ModelConfig {
    static constexpr int NUM_LAYERS         = 64;
    static constexpr int HIDDEN_SIZE        = 5120;
    static constexpr int INTERMEDIATE_SIZE  = 17408;
    static constexpr int VOCAB_SIZE         = 248320;
    static constexpr float RMS_EPS          = 1e-6f;

    // Full Attention (GQA)
    static constexpr int NUM_ATTN_HEADS     = 24;
    static constexpr int NUM_KV_HEADS       = 4;
    static constexpr int HEAD_DIM           = 256;
    static constexpr int NUM_KV_GROUPS      = 6;   // ATTN_HEADS / KV_HEADS
    static constexpr int Q_PROJ_DIM         = 12288; // ATTN_HEADS * HEAD_DIM * 2 (gate)
    static constexpr int KV_PROJ_DIM        = 1024;  // KV_HEADS * HEAD_DIM
    static constexpr int ATTN_OUT_DIM       = 6144;  // ATTN_HEADS * HEAD_DIM
    static constexpr int FULL_ATTN_INTERVAL = 4;

    // RoPE (partial rotary, interleaved M-RoPE)
    static constexpr float ROPE_THETA       = 10000000.0f;
    static constexpr float PARTIAL_ROTARY   = 0.25f;
    static constexpr int ROTARY_DIM         = 64;  // HEAD_DIM * 0.25

    // DeltaNet SSM
    static constexpr int LIN_NUM_K_HEADS    = 16;
    static constexpr int LIN_NUM_V_HEADS    = 48;
    static constexpr int LIN_K_HEAD_DIM     = 128;
    static constexpr int LIN_V_HEAD_DIM     = 128;
    static constexpr int LIN_KEY_DIM        = 2048;  // 16 * 128
    static constexpr int LIN_VALUE_DIM      = 6144;  // 48 * 128
    static constexpr int LIN_CONV_DIM       = 10240; // KEY_DIM*2 + VALUE_DIM
    static constexpr int CONV_KERNEL        = 4;

    static constexpr bool is_full_attention(int layer_idx) {
        return (layer_idx % FULL_ATTN_INTERVAL) == (FULL_ATTN_INTERVAL - 1);
    }
};

// ============================================================================
// Non-quantized linear layer (FP16 weights on device)
// ============================================================================

struct Linear {
    __half* weight = nullptr;  // [out_features, in_features] row-major
    int in_features  = 0;
    int out_features = 0;
};

// ============================================================================
// INT8 quantized linear layer (per-channel symmetric quantization)
// ============================================================================
// Quantized at model load time from FP16/BF16 weights.
// Dequant: w_fp32 = scale[n] * (float)int8_weight[n][k]
// Memory: N*K bytes (weight) + N*4 bytes (scales) → ~50% of FP16

struct Int8Linear {
    int8_t* weight = nullptr;  // [out_features, in_features] row-major INT8
    float*  scales = nullptr;  // [out_features] per-output-channel FP32
    int in_features  = 0;
    int out_features = 0;
};

// ============================================================================
// Per-layer weight structures
// ============================================================================

struct DeltaNetWeights {
    Int8Linear in_proj_qkv;   // 5120 → 10240 (key*2 + value)
    Int8Linear in_proj_z;     // 5120 → 6144  (value_dim, for gate)
    Int8Linear in_proj_a;     // 5120 → 48    (num_v_heads, for decay)
    Int8Linear in_proj_b;     // 5120 → 48    (num_v_heads, for beta)
    Int8Linear out_proj;      // 6144 → 5120

    __half* conv1d_weight = nullptr;  // [conv_dim, kernel] = [10240, 4]
    float*  A_log         = nullptr;  // [48]
    __half* dt_bias       = nullptr;  // [48]
    float*  norm_weight   = nullptr;  // [128] gated RMSNorm (w, NOT 1+w)
};

struct FullAttentionWeights {
    Int8Linear q_proj;   // 5120 → 12288 (includes output gate, split in kernel)
    Int8Linear k_proj;   // 5120 → 1024
    Int8Linear v_proj;   // 5120 → 1024
    Int8Linear o_proj;   // 6144 → 5120

    __half* q_norm = nullptr;  // [256] precomputed (1+w)
    __half* k_norm = nullptr;  // [256] precomputed (1+w)
};

struct MLPWeights {
    GptqWeight gate_proj;  // 5120 → 17408
    GptqWeight up_proj;    // 5120 → 17408
    GptqWeight down_proj;  // 17408 → 5120
};

struct LayerWeights {
    __half* input_layernorm  = nullptr;  // [5120] precomputed (1+w)
    __half* post_attn_layernorm = nullptr;  // [5120] precomputed (1+w)

    bool is_full_attention = false;

    DeltaNetWeights   delta_net;
    FullAttentionWeights full_attn;
    MLPWeights mlp;
};

// ============================================================================
// Complete model
// ============================================================================

struct ModelWeights {
    __half* embed_tokens = nullptr;  // [vocab_size, hidden_size]
    __half* final_norm   = nullptr;  // [hidden_size] precomputed (1+w)
    __half* lm_head      = nullptr;  // [vocab_size, hidden_size] FP16 (for prefill)
    Int8Linear lm_head_int8;         // INT8 quantized lm_head for decode GEMV

    LayerWeights layers[ModelConfig::NUM_LAYERS];

    size_t total_bytes = 0;  // total device memory

    // Pool-allocated blocks (one per shard). Individual tensor pointers
    // above point into these pools. Only pool blocks are cudaFree'd.
    std::vector<void*> pool_blocks;
};

// ============================================================================
// API
// ============================================================================

// Load all model weights from safetensors shards into device memory.
// BF16 weights are converted to FP16. RMSNorm weights are precomputed as (1+w).
bool load_model_weights(const std::string& model_dir, ModelWeights& weights);

// Release all device memory held by the model.
void free_model_weights(ModelWeights& weights);

// ============================================================================
// Sampling parameters
// ============================================================================

struct SamplingParams {
    float temperature = 1.0f;   // Logit scaling (>1 = more random, <1 = more deterministic)
    int   top_k       = 50;     // Keep top-k highest-probability tokens (0 = disabled)
    float top_p       = 0.9f;   // Nucleus: keep smallest set with cumulative prob >= top_p
    float rep_penalty = 1.0f;   // Repetition penalty (1.0 = disabled)
    unsigned long long seed = 0; // RNG seed (0 = use counter)
};

// ============================================================================
// Inference state (scratch buffers for forward pass)
// ============================================================================

struct InferenceState {
    // Hidden state buffers (double-buffered for residual connections)
    __half* hidden      = nullptr;  // [max_seq, hidden_size]
    __half* residual    = nullptr;  // [max_seq, hidden_size]
    __half* norm_out    = nullptr;  // [max_seq, hidden_size]

    // Attention scratch
    __half* attn_out    = nullptr;  // [max_seq, attn_out_dim] (6144)
    __half* q_buf       = nullptr;  // [max_seq, q_proj_dim] (12288)
    __half* kv_buf      = nullptr;  // [max_seq, kv_proj_dim] (1024)

    // Full Attention pre-allocated scratch
    float*  attn_scores = nullptr;  // [num_attn_heads, max_seq] for Q@K^T
    __half* scores_h16  = nullptr;  // [num_kv_groups, max_seq] for FP16 score conversion

    // MLP scratch
    __half* mlp_gate    = nullptr;  // [max_seq, intermediate_size] (17408)
    __half* mlp_up      = nullptr;  // [max_seq, intermediate_size]
    __half* mlp_down    = nullptr;  // [max_seq, hidden_size]

    // DeltaNet scratch
    __half* dn_qkv      = nullptr;  // [max_seq, lin_conv_dim] (10240)
    __half* dn_z        = nullptr;  // [max_seq, lin_value_dim] (6144)
    __half* dn_a        = nullptr;  // [max_seq, num_v_heads] (48)
    __half* dn_b        = nullptr;  // [max_seq, num_v_heads] (48)
    float*  dn_g        = nullptr;  // [num_v_heads] (48) decay factors
    float*  dn_beta     = nullptr;  // [num_v_heads] (48) beta factors

    // DeltaNet recurrent states (one per linear attention layer)
    // state[layer][head]: [key_head_dim, value_head_dim] = [128, 128]
    float** dn_states   = nullptr;  // array of device pointers
    int     num_dn_layers = 0;

    // DeltaNet conv states: [layer][conv_dim, kernel-1] = [10240, 3]
    __half** conv_states = nullptr;

    // Token input
    int*    token_ids   = nullptr;  // [max_seq] device
    int*    sample_out  = nullptr;  // [1] device — greedy sample output

    // Logits output
    __half* logits      = nullptr;  // [vocab_size]

    // Sampling workspace (FP32 probabilities for top-k/top-p)
    float*  probs       = nullptr;  // [vocab_size]
    unsigned long long rng_counter = 0;  // running counter for PRNG seeding

    // --- CUDA Graph support ---
    // Device-side pos for graph-capturable kernels (RoPE, KV cache write, attention)
    int*    d_pos       = nullptr;  // [1] device — current sequence position
    // Pinned host staging (CUDA graph reads from these at replay time)
    int*    h_pos_pinned    = nullptr;  // pinned host — pos staging
    int*    h_token_pinned  = nullptr;  // pinned host — token_id staging
    // Compute stream (non-default, required for graph capture)
    cudaStream_t compute_stream = nullptr;
    // Auxiliary stream for concurrent MLP gate+up projections
    cudaStream_t aux_stream     = nullptr;
    cudaEvent_t  aux_fork_event = nullptr;  // main→aux dependency (X is ready)
    cudaEvent_t  aux_join_event = nullptr;  // aux→main dependency (up_proj done)
    // Graph state
    cudaGraph_t     cuda_graph      = nullptr;
    cudaGraphExec_t cuda_graph_exec = nullptr;
    bool            graph_captured  = false;

    int max_seq_len = 0;

    // Allocate all buffers for given max sequence length
    bool allocate(int max_seq);
    void free();
};

} // namespace deusridet
