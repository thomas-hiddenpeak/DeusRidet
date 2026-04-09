// asr_decoder.h — Qwen3-ASR Text Decoder
//
// 28-layer Qwen3 decoder:
//   - RMSNorm (plain weight, eps=1e-6)
//   - GQA: 16 Q heads, 8 KV heads, head_dim=128
//   - Per-head Q/K RMSNorm + MRoPE (interleaved sections)
//   - SwiGLU MLP (no bias)
//   - Contiguous KV cache (non-paged, single-sequence ASR)
//
// Weight prefix: thinker.model.layers.{i}.*
//                thinker.model.embed_tokens.weight
//                thinker.model.norm.weight
//                thinker.lm_head.weight (= embed_tokens, tied)
//
// Adapted from qwen35-orin (src/plugins/asr/asr_decoder.h): decoder class
// with GQA, MRoPE, contiguous KV cache, and autoregressive decode.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include "asr_config.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>

namespace deusridet {
namespace asr {

// Per-layer decoder weight pointers (all projections no bias)
struct DecoderLayerWeights {
    // Pre-attention norm
    __nv_bfloat16* input_layernorm_w = nullptr;       // [hidden_size=2048]

    // Self-attention (GQA)
    __nv_bfloat16* q_proj_w = nullptr;                // [q_dim=2048, hidden_size=2048]
    __nv_bfloat16* k_proj_w = nullptr;                // [kv_dim=1024, hidden_size=2048]
    __nv_bfloat16* v_proj_w = nullptr;                // [kv_dim=1024, hidden_size=2048]
    __nv_bfloat16* qkv_proj_w = nullptr;              // [q_dim+2*kv_dim=4096, h=2048] merged
    __nv_bfloat16* o_proj_w = nullptr;                // [hidden_size=2048, q_dim=2048]
    __nv_bfloat16* q_norm_w = nullptr;                // [head_dim=128] per-head RMSNorm
    __nv_bfloat16* k_norm_w = nullptr;                // [head_dim=128]

    // Post-attention norm
    __nv_bfloat16* post_attention_layernorm_w = nullptr; // [hidden_size=2048]

    // MLP (SwiGLU, no bias)
    __nv_bfloat16* gate_proj_w = nullptr;             // [intermediate=6144, hidden=2048]
    __nv_bfloat16* up_proj_w = nullptr;               // [intermediate=6144, hidden=2048]
    __nv_bfloat16* down_proj_w = nullptr;             // [hidden=2048, intermediate=6144]
};

class TextDecoder {
public:
    TextDecoder(const ASRConfig& config, int max_seq_len = 512);
    ~TextDecoder();

    // Bind shared weights
    void set_embed_weights(__nv_bfloat16* embed_tokens_w,  // [vocab_size, hidden_size]
                           __nv_bfloat16* lm_head_w,       // [vocab_size, hidden_size] (tied)
                           __nv_bfloat16* final_norm_w);   // [hidden_size]

    // Bind layer weights
    void set_layer_weights(int layer_idx, const DecoderLayerWeights& weights);

    // Initialize KV cache + workspace
    void initialize(cudaStream_t stream = 0);

    // Reset KV cache (call before new transcription)
    void reset_cache();

    // Prepare merged QKV weights (call once after all weights are set)
    void prepare_optimized_weights(cudaStream_t stream = 0);

    // Prefill: input embeddings → logits for last token
    // input_embeds: [seq_len, hidden_size] GPU BF16
    // position_ids: [3, seq_len] GPU int (MRoPE 3D positions)
    // logits_out: [vocab_size] GPU BF16 (last token only)
    void forward_prefill(const __nv_bfloat16* input_embeds,
                         const int* position_ids,
                         int seq_len,
                         __nv_bfloat16* logits_out,
                         cudaStream_t stream = 0);

    // Decode: single token step
    // token_id: current token
    // position_ids: [3] GPU int (3D position)
    // logits_out: [vocab_size] GPU BF16
    void forward_decode(int token_id,
                        const int* position_ids,
                        __nv_bfloat16* logits_out,
                        cudaStream_t stream = 0);

    int current_seq_len() const { return cache_seq_len_; }

private:
    ASRConfig config_;
    int max_seq_len_;
    cublasHandle_t cublas_ = nullptr;

    // Shared weights (externally owned)
    __nv_bfloat16* embed_tokens_w_ = nullptr;
    __nv_bfloat16* lm_head_w_ = nullptr;
    __nv_bfloat16* final_norm_w_ = nullptr;

    // Layer weights
    std::vector<DecoderLayerWeights> layer_weights_;

    // Merged weight allocations (freed in destructor)
    std::vector<void*> merged_allocations_;

    // KV Cache: per-layer, [max_seq_len, num_kv_heads, head_dim]
    std::vector<__nv_bfloat16*> k_cache_;
    std::vector<__nv_bfloat16*> v_cache_;
    int cache_seq_len_ = 0;

    // Workspace
    __nv_bfloat16* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    // Device token ID for decode step
    int* token_id_gpu_ = nullptr;

    // Split-K attention workspace
    float* attn_split_k_ws_ = nullptr;
    int attn_max_partitions_ = 0;

    // Prefill attention score buffer: [max_seq, max_seq] BF16
    __nv_bfloat16* prefill_attn_buf_ = nullptr;

    bool initialized_ = false;

    void decoder_layer_forward_prefill(
        int layer_idx,
        __nv_bfloat16* hidden_states,
        const int* position_ids,
        int seq_len,
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);

    void decoder_layer_forward_decode(
        int layer_idx,
        __nv_bfloat16* hidden_states,
        const int* position_ids,
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);
};

} // namespace asr
} // namespace deusridet
