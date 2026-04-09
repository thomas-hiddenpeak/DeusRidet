// model.h — Qwen3.5 model weight structures and forward pass
//
// Supports multiple Qwen3.5 variants (9B BF16, 27B GPTQ-Int4, etc.)
// Hybrid DeltaNet SSM + GQA architecture:
//   GPTQ-Int4 or FP16 MLP, BF16 attention weights converted to FP16 at load
//   SwiGLU MLP, partial RoPE (25%), attention output gate, (1+w) RMSNorm
//
// ModelConfig is runtime-initialized from config.json via ModelConfig::init().
// Weight loading converts BF16→FP16 at load time for uniform FP16 compute.
// Regular RMSNorm weights are precomputed as (1+w) at load time.

#pragma once

#include "gptq.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace deusridet {

// ============================================================================
// Model configuration — runtime-initialized from config.json
// ============================================================================

struct ModelConfig {
    // Core dimensions (set by init())
    static int NUM_LAYERS;
    static int HIDDEN_SIZE;
    static int INTERMEDIATE_SIZE;
    static int VOCAB_SIZE;
    static float RMS_EPS;

    // Full Attention (GQA)
    static int NUM_ATTN_HEADS;
    static int NUM_KV_HEADS;
    static int HEAD_DIM;
    static int NUM_KV_GROUPS;         // ATTN_HEADS / KV_HEADS
    static int Q_PROJ_DIM;            // ATTN_HEADS * HEAD_DIM * 2 (gate)
    static int KV_PROJ_DIM;           // KV_HEADS * HEAD_DIM
    static int ATTN_OUT_DIM;          // ATTN_HEADS * HEAD_DIM
    static int FULL_ATTN_INTERVAL;

    // RoPE (partial rotary, interleaved M-RoPE)
    static float ROPE_THETA;
    static float PARTIAL_ROTARY;
    static int ROTARY_DIM;            // HEAD_DIM * PARTIAL_ROTARY

    // DeltaNet SSM
    static int LIN_NUM_K_HEADS;
    static int LIN_NUM_V_HEADS;
    static int LIN_K_HEAD_DIM;
    static int LIN_V_HEAD_DIM;
    static int LIN_KEY_DIM;           // K_HEADS * K_HEAD_DIM
    static int LIN_VALUE_DIM;         // V_HEADS * V_HEAD_DIM
    static int LIN_CONV_DIM;          // KEY_DIM*2 + VALUE_DIM
    static int CONV_KERNEL;

    // Merged projection dimensions (padded to tile boundary, 256)
    static int LIN_QKV_AB_DIM;        // LIN_CONV_DIM+V_HEADS+V_HEADS, padded to 256
    static int FA_KV_DIM;             // KV_PROJ_DIM + KV_PROJ_DIM

    // Quantization mode
    static bool MLP_IS_GPTQ;          // true if MLP uses GPTQ-Int4, false if FP16

    // Number of Full Attention layers (derived)
    static int NUM_FA_LAYERS;

    static bool is_full_attention(int layer_idx) {
        return (layer_idx % FULL_ATTN_INTERVAL) == (FULL_ATTN_INTERVAL - 1);
    }

    // Initialize from model directory's config.json.
    // Must be called before loading weights or allocating inference state.
    static bool init(const std::string& model_dir);

    // Check if already initialized
    static bool initialized();
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
    // FP16 weights (point into pool_blocks, no extra alloc)
    Linear fp16_qkv;   // 5120 → 10240
    Linear fp16_z;     // 5120 → 6144
    Linear fp16_a;     // 5120 → 48   (num_v_heads, for decay)
    Linear fp16_b;     // 5120 → 48   (num_v_heads, for beta)
    Linear fp16_out;   // 6144 → 5120

    // Repacked FP16 weights for prefill fp16_gemm (created post-load)
    __half* repacked_qkv_ab = nullptr;  // merged qkv+a+b [10496, 5120] tile-repacked
    __half* repacked_z      = nullptr;  // [6144, 5120] tile-repacked
    __half* repacked_out    = nullptr;  // [5120, 6144] tile-repacked

    __half* conv1d_weight = nullptr;  // [conv_dim, kernel] = [10240, 4]
    float*  A_log         = nullptr;  // [48]
    __half* dt_bias       = nullptr;  // [48]
    float*  norm_weight   = nullptr;  // [128] gated RMSNorm (w, NOT 1+w)
};

struct FullAttentionWeights {
    // FP16 weights (point into pool_blocks, no extra alloc)
    Linear fp16_q;   // 5120 → 12288
    Linear fp16_k;   // 5120 → 1024
    Linear fp16_v;   // 5120 → 1024
    Linear fp16_o;   // 6144 → 5120

    // Repacked FP16 weights for prefill fp16_gemm (created post-load)
    __half* repacked_q  = nullptr;  // [12288, 5120] tile-repacked
    __half* repacked_kv = nullptr;  // merged k+v [2048, 5120] tile-repacked
    __half* repacked_o  = nullptr;  // [5120, 6144] tile-repacked

    __half* q_norm = nullptr;  // [256] precomputed (1+w)
    __half* k_norm = nullptr;  // [256] precomputed (1+w)
};

struct MLPWeights {
    // GPTQ-Int4 weights (for quantized models, e.g. 27B-GPTQ)
    GptqWeight gate_proj;  // 5120 → 17408 (27B)
    GptqWeight up_proj;    // 5120 → 17408
    GptqWeight down_proj;  // 17408 → 5120

    // FP16 Linear weights (for unquantized models, e.g. 9B-BF16)
    Linear fp16_gate_proj;
    Linear fp16_up_proj;
    Linear fp16_down_proj;

    // Repacked FP16 weights for prefill fp16_gemm (unquantized models only)
    __half* repacked_gate = nullptr;
    __half* repacked_up   = nullptr;
    __half* repacked_down = nullptr;
};

struct LayerWeights {
    __half* input_layernorm  = nullptr;  // [hidden_size] precomputed (1+w)
    __half* post_attn_layernorm = nullptr;  // [hidden_size] precomputed (1+w)

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

    std::vector<LayerWeights> layers;  // [NUM_LAYERS] — sized by load_model_weights

    size_t total_bytes = 0;  // total device memory

    // Pool-allocated blocks (one per shard). Individual tensor pointers
    // above point into these pools. Only pool blocks are cudaFree'd.
    std::vector<void*> pool_blocks;
};

// ============================================================================
// API
// ============================================================================

// Load all model weights from safetensors shards into device memory.
// Calls ModelConfig::init() first to read config.json and set parameters.
// BF16 weights are converted to FP16. RMSNorm weights are precomputed as (1+w).
// If repack_marlin=false, GPTQ weights are left in original format (for custom GPTQ kernel).
bool load_model_weights(const std::string& model_dir, ModelWeights& weights,
                        bool repack_marlin = true);

// Create merged+repacked FP16 projection weights for prefill optimization.
// Must be called after load_model_weights. Creates:
//   - DN repacked_qkv_ab: qkv+a+b merged+repacked [10496, 5120] × 48 layers
//   - DN repacked_z, repacked_out × 48 layers
//   - FA repacked_q, repacked_kv (k+v merged), repacked_o × 16 layers
bool merge_projection_weights(ModelWeights& weights);

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
    __half* dn_qkv      = nullptr;  // [max_seq, qkv_ab_dim=10496] (qkv+a+b, used by fused kernel both prefill & decode)
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

    // Marlin GEMM workspace (global barrier lock buffer)
    int*    marlin_workspace = nullptr;

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
