/**
 * @file src/sensus/auditus/asr/asr_decoder.cu
 * @philosophical_role
 *   Qwen3-ASR text decoder — 28-layer GQA decoder turning encoder output into token IDs. The last perceptual step before Conscientia sees words.
 * @serves
 *   Auditus ASR pipeline tail; publishes tokens to the facade which forwards them to Conscientia via Nexus.
 */
// asr_decoder.cu — Qwen3-ASR Text Decoder implementation
//
// 28-layer GQA decoder with MRoPE, per-head Q/K RMSNorm, SwiGLU MLP.
// Supports prefill (T>1) and decode (T=1) paths.
// Uses cuBLAS GEMM for both paths (no fused GEMV dependency).
//
// Adapted from qwen35-orin (src/plugins/asr/asr_decoder.cu): decoder
// layer implementation with GQA prefill/decode, QKV merge, KV cache.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "asr_decoder.h"
#include "asr_ops.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace deusridet {
namespace asr {

// BF16 linear: out = input @ weight^T (no bias)
static void linear_nobias(
    cublasHandle_t handle,
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int M, int K, int N,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 weight, CUDA_R_16BF, K,
                 input, CUDA_R_16BF, K,
                 &beta,
                 out, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ============================================================================
// TextDecoder implementation
// ============================================================================

TextDecoder::TextDecoder(const ASRConfig& config, int max_seq_len)
    : config_(config)
    , max_seq_len_(max_seq_len)
    , layer_weights_(config.decoder_layers)
    , k_cache_(config.decoder_layers, nullptr)
    , v_cache_(config.decoder_layers, nullptr) {}

TextDecoder::~TextDecoder() {
    for (int i = 0; i < config_.decoder_layers; i++) {
        if (k_cache_[i]) cudaFree(k_cache_[i]);
        if (v_cache_[i]) cudaFree(v_cache_[i]);
    }
    for (auto p : merged_allocations_) cudaFree(p);
    if (workspace_) cudaFree(workspace_);
    if (token_id_gpu_) cudaFree(token_id_gpu_);
    if (attn_split_k_ws_) cudaFree(attn_split_k_ws_);
    if (prefill_attn_buf_) cudaFree(prefill_attn_buf_);
    if (cublas_) cublasDestroy(cublas_);
}

void TextDecoder::set_embed_weights(
    __nv_bfloat16* embed_tokens_w,
    __nv_bfloat16* lm_head_w,
    __nv_bfloat16* final_norm_w)
{
    embed_tokens_w_ = embed_tokens_w;
    lm_head_w_ = lm_head_w;
    final_norm_w_ = final_norm_w;
}

void TextDecoder::set_layer_weights(int layer_idx, const DecoderLayerWeights& weights) {
    layer_weights_[layer_idx] = weights;
}

void TextDecoder::initialize(cudaStream_t stream) {
    if (initialized_) return;

    cublasCreate(&cublas_);

    // Pre-allocate cuBLAS workspace
    size_t cublas_ws_size = 4 * 1024 * 1024;  // 4 MB
    void* cublas_ws = nullptr;
    cudaMalloc(&cublas_ws, cublas_ws_size);
    cublasSetWorkspace(cublas_, cublas_ws, cublas_ws_size);
    merged_allocations_.push_back(cublas_ws);  // tracked for cleanup

    int h = config_.decoder_hidden_size;
    int kv_dim = config_.decoder_kv_dim();
    int q_dim = config_.decoder_q_dim();
    int ffn = config_.decoder_intermediate_size;
    int num_layers = config_.decoder_layers;

    // KV cache: [max_seq_len, num_kv_heads, head_dim] per layer
    size_t kv_per_layer = (size_t)max_seq_len_ * kv_dim;
    for (int i = 0; i < num_layers; i++) {
        cudaMalloc(&k_cache_[i], kv_per_layer * sizeof(__nv_bfloat16));
        cudaMalloc(&v_cache_[i], kv_per_layer * sizeof(__nv_bfloat16));
    }

    // Workspace for prefill (max T = max_seq_len_)
    workspace_size_ = (size_t)max_seq_len_ * h          // hidden_states (copy)
                    + (size_t)max_seq_len_ * h           // norm_buf
                    + (size_t)max_seq_len_ * q_dim       // q_buf
                    + (size_t)max_seq_len_ * kv_dim      // k_buf
                    + (size_t)max_seq_len_ * kv_dim      // v_buf
                    + (size_t)max_seq_len_ * h           // attn_out
                    + (size_t)max_seq_len_ * ffn          // gate_buf
                    + (size_t)max_seq_len_ * ffn          // up_buf
                    + (size_t)config_.vocab_size           // logits
                    + 1024;

    cudaMalloc(&workspace_, workspace_size_ * sizeof(__nv_bfloat16));
    cudaMemset(workspace_, 0, workspace_size_ * sizeof(__nv_bfloat16));

    cudaMalloc(&token_id_gpu_, sizeof(int));

    // Split-K attention workspace
    int num_q_heads = config_.decoder_num_attention_heads;
    int head_dim = config_.decoder_head_dim;
    attn_max_partitions_ = (max_seq_len_ + 127) / 128;
    size_t attn_ws_size = (size_t)num_q_heads * attn_max_partitions_ * head_dim * sizeof(float)
                        + (size_t)num_q_heads * attn_max_partitions_ * sizeof(float) * 2;
    cudaMalloc(&attn_split_k_ws_, attn_ws_size);
    cudaMemset(attn_split_k_ws_, 0, attn_ws_size);

    // Prefill attention score buffer
    cudaMalloc(&prefill_attn_buf_,
               (size_t)max_seq_len_ * max_seq_len_ * sizeof(__nv_bfloat16));

    cache_seq_len_ = 0;
    initialized_ = true;

    float kv_mb = (float)num_layers * kv_per_layer * 2 * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float ws_mb = workspace_size_ * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    fprintf(stderr, "[ASR Decoder] initialized: %d layers, max_seq=%d, KV %.1f MB, workspace %.1f MB\n",
            num_layers, max_seq_len_, kv_mb, ws_mb);
}

void TextDecoder::reset_cache() {
    cache_seq_len_ = 0;
}

// ============================================================================
// prepare_optimized_weights: merge QKV into single buffer
// ============================================================================

void TextDecoder::prepare_optimized_weights(cudaStream_t stream) {
    int h = config_.decoder_hidden_size;
    int q_dim = config_.decoder_q_dim();
    int kv_dim = config_.decoder_kv_dim();
    int qkv_dim = q_dim + 2 * kv_dim;

    for (int layer = 0; layer < config_.decoder_layers; layer++) {
        auto& lw = layer_weights_[layer];

        // Merge QKV: [q_dim, h] + [kv_dim, h] + [kv_dim, h] → [qkv_dim, h]
        __nv_bfloat16* merged;
        cudaMalloc(&merged, (size_t)qkv_dim * h * sizeof(__nv_bfloat16));
        merged_allocations_.push_back(merged);

        cudaMemcpyAsync(merged,
                        lw.q_proj_w, (size_t)q_dim * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(merged + (size_t)q_dim * h,
                        lw.k_proj_w, (size_t)kv_dim * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(merged + (size_t)(q_dim + kv_dim) * h,
                        lw.v_proj_w, (size_t)kv_dim * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);

        lw.qkv_proj_w = merged;
        // Point individual projections into merged buffer
        lw.q_proj_w = merged;
        lw.k_proj_w = merged + (size_t)q_dim * h;
        lw.v_proj_w = merged + (size_t)(q_dim + kv_dim) * h;
    }

    cudaStreamSynchronize(stream);
    fprintf(stderr, "[ASR Decoder] Optimized: %d layers QKV merged [%d, %d]\n",
            config_.decoder_layers, qkv_dim, h);
}

// ============================================================================
// Prefill layer
// ============================================================================

void TextDecoder::decoder_layer_forward_prefill(
    int layer_idx,
    __nv_bfloat16* hidden_states,
    const int* position_ids,
    int seq_len,
    __nv_bfloat16* workspace_base,
    cudaStream_t stream)
{
    const auto& lw = layer_weights_[layer_idx];
    int h = config_.decoder_hidden_size;
    int q_dim = config_.decoder_q_dim();
    int kv_dim = config_.decoder_kv_dim();
    int num_q_heads = config_.decoder_num_attention_heads;
    int num_kv_heads = config_.decoder_num_kv_heads;
    int head_dim = config_.decoder_head_dim;
    float eps = config_.rms_norm_eps;
    int ffn = config_.decoder_intermediate_size;

    __nv_bfloat16* norm_buf  = workspace_base;
    __nv_bfloat16* q_buf     = norm_buf  + (size_t)seq_len * h;
    __nv_bfloat16* k_buf     = q_buf     + (size_t)seq_len * q_dim;
    __nv_bfloat16* v_buf     = k_buf     + (size_t)seq_len * kv_dim;
    __nv_bfloat16* attn_out  = v_buf     + (size_t)seq_len * kv_dim;
    __nv_bfloat16* gate_buf  = attn_out  + (size_t)seq_len * h;
    __nv_bfloat16* up_buf    = gate_buf  + (size_t)seq_len * ffn;

    // Self-Attention
    asr_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.input_layernorm_w,
                            eps, seq_len, h, stream);

    linear_nobias(cublas_, q_buf, norm_buf, lw.q_proj_w, seq_len, h, q_dim, stream);
    linear_nobias(cublas_, k_buf, norm_buf, lw.k_proj_w, seq_len, h, kv_dim, stream);
    linear_nobias(cublas_, v_buf, norm_buf, lw.v_proj_w, seq_len, h, kv_dim, stream);

    asr_ops::invoke_fused_qk_norm_rope(
        q_buf, k_buf, lw.q_norm_w, lw.k_norm_w,
        position_ids, eps, seq_len, num_q_heads, num_kv_heads, head_dim,
        config_.mrope_section[0], config_.mrope_section[1],
        config_.mrope_section[2], config_.rope_theta, stream);

    asr_ops::invoke_write_kv_cache(k_cache_[layer_idx], v_cache_[layer_idx],
                                   k_buf, v_buf,
                                   0, seq_len,
                                   num_kv_heads, head_dim, stream);

    asr_ops::invoke_causal_gqa_prefill_cublas(
        attn_out, q_buf,
        k_cache_[layer_idx], v_cache_[layer_idx],
        prefill_attn_buf_,
        seq_len, num_q_heads, num_kv_heads, head_dim,
        cublas_, stream);

    linear_nobias(cublas_, norm_buf, attn_out, lw.o_proj_w, seq_len, q_dim, h, stream);
    asr_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream);

    // MLP (SwiGLU)
    asr_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.post_attention_layernorm_w,
                            eps, seq_len, h, stream);

    linear_nobias(cublas_, gate_buf, norm_buf, lw.gate_proj_w, seq_len, h, ffn, stream);
    linear_nobias(cublas_, up_buf, norm_buf, lw.up_proj_w, seq_len, h, ffn, stream);
    asr_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, seq_len, ffn, stream);

    linear_nobias(cublas_, norm_buf, gate_buf, lw.down_proj_w, seq_len, ffn, h, stream);
    asr_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream);
}

// ============================================================================
// Decode layer (T=1)
// ============================================================================

void TextDecoder::decoder_layer_forward_decode(
    int layer_idx,
    __nv_bfloat16* hidden_states,
    const int* position_ids,
    __nv_bfloat16* workspace_base,
    cudaStream_t stream)
{
    const auto& lw = layer_weights_[layer_idx];
    int h = config_.decoder_hidden_size;
    int q_dim = config_.decoder_q_dim();
    int kv_dim = config_.decoder_kv_dim();
    int num_q_heads = config_.decoder_num_attention_heads;
    int num_kv_heads = config_.decoder_num_kv_heads;
    int head_dim = config_.decoder_head_dim;
    float eps = config_.rms_norm_eps;
    int ffn = config_.decoder_intermediate_size;

    // Workspace (T=1)
    __nv_bfloat16* norm_buf  = workspace_base;
    __nv_bfloat16* q_buf     = norm_buf  + h;
    __nv_bfloat16* k_buf     = q_buf     + q_dim;
    __nv_bfloat16* v_buf     = k_buf     + kv_dim;
    __nv_bfloat16* attn_out  = v_buf     + kv_dim;
    __nv_bfloat16* proj_out  = attn_out  + h;
    __nv_bfloat16* gate_buf  = proj_out  + h;
    __nv_bfloat16* up_buf    = gate_buf  + ffn;

    // Self-Attention
    asr_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.input_layernorm_w,
                            eps, 1, h, stream);

    // QKV via cuBLAS GEMM (M=1 → effectively GEMV)
    linear_nobias(cublas_, q_buf, norm_buf, lw.q_proj_w, 1, h, q_dim, stream);
    linear_nobias(cublas_, k_buf, norm_buf, lw.k_proj_w, 1, h, kv_dim, stream);
    linear_nobias(cublas_, v_buf, norm_buf, lw.v_proj_w, 1, h, kv_dim, stream);

    asr_ops::invoke_fused_qk_norm_rope(
        q_buf, k_buf, lw.q_norm_w, lw.k_norm_w,
        position_ids, eps, 1, num_q_heads, num_kv_heads, head_dim,
        config_.mrope_section[0], config_.mrope_section[1],
        config_.mrope_section[2], config_.rope_theta, stream);

    asr_ops::invoke_write_kv_cache(k_cache_[layer_idx], v_cache_[layer_idx],
                                   k_buf, v_buf,
                                   cache_seq_len_, 1,
                                   num_kv_heads, head_dim, stream);

    asr_ops::invoke_causal_gqa_decode(
        attn_out, q_buf,
        k_cache_[layer_idx], v_cache_[layer_idx],
        1, num_q_heads, num_kv_heads, head_dim,
        cache_seq_len_ + 1,
        stream,
        attn_split_k_ws_, attn_max_partitions_);

    linear_nobias(cublas_, proj_out, attn_out, lw.o_proj_w, 1, q_dim, h, stream);
    asr_ops::invoke_add_residual(hidden_states, proj_out, h, stream);

    // MLP
    asr_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.post_attention_layernorm_w,
                            eps, 1, h, stream);

    linear_nobias(cublas_, gate_buf, norm_buf, lw.gate_proj_w, 1, h, ffn, stream);
    linear_nobias(cublas_, up_buf, norm_buf, lw.up_proj_w, 1, h, ffn, stream);
    asr_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, 1, ffn, stream);

    linear_nobias(cublas_, proj_out, gate_buf, lw.down_proj_w, 1, ffn, h, stream);
    asr_ops::invoke_add_residual(hidden_states, proj_out, h, stream);
}

// ============================================================================
// forward_prefill
// ============================================================================

void TextDecoder::forward_prefill(
    const __nv_bfloat16* input_embeds,
    const int* position_ids,
    int seq_len,
    __nv_bfloat16* logits_out,
    cudaStream_t stream)
{
    if (!initialized_) {
        fprintf(stderr, "[ASR Decoder] ERROR: not initialized\n");
        return;
    }
    if (seq_len > max_seq_len_) {
        fprintf(stderr, "[ASR Decoder] ERROR: seq_len=%d > max_seq_len=%d\n",
                seq_len, max_seq_len_);
        return;
    }

    int h = config_.decoder_hidden_size;

    // Copy input to workspace (layers modify in-place)
    __nv_bfloat16* hidden_states = workspace_;
    cudaMemcpyAsync(hidden_states, input_embeds,
                    (size_t)seq_len * h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);

    __nv_bfloat16* layer_ws = hidden_states + (size_t)max_seq_len_ * h;

    for (int layer = 0; layer < config_.decoder_layers; layer++) {
        decoder_layer_forward_prefill(layer, hidden_states, position_ids,
                                      seq_len, layer_ws, stream);
    }

    // Final RMSNorm on last token
    __nv_bfloat16* last_hidden = hidden_states + (size_t)(seq_len - 1) * h;
    __nv_bfloat16* norm_out = layer_ws;
    asr_ops::invoke_rmsnorm(norm_out, last_hidden, final_norm_w_,
                            config_.rms_norm_eps, 1, h, stream);

    // LM head: [1, h] → [1, vocab_size]
    linear_nobias(cublas_, logits_out, norm_out, lm_head_w_,
                  1, h, config_.vocab_size, stream);

    cache_seq_len_ = seq_len;
}

// ============================================================================
// forward_decode
// ============================================================================

void TextDecoder::forward_decode(
    int token_id,
    const int* position_ids,
    __nv_bfloat16* logits_out,
    cudaStream_t stream)
{
    if (!initialized_) {
        fprintf(stderr, "[ASR Decoder] ERROR: not initialized\n");
        return;
    }
    if (cache_seq_len_ >= max_seq_len_) {
        fprintf(stderr, "[ASR Decoder] ERROR: KV cache full (%d/%d)\n",
                cache_seq_len_, max_seq_len_);
        return;
    }

    int h = config_.decoder_hidden_size;

    // Embed token
    __nv_bfloat16* hidden_states = workspace_;
    cudaMemcpyAsync(token_id_gpu_, &token_id, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    asr_ops::invoke_embedding_lookup(hidden_states, token_id_gpu_,
                                     embed_tokens_w_, 1, h, stream);

    __nv_bfloat16* layer_ws = hidden_states + h;

    for (int layer = 0; layer < config_.decoder_layers; layer++) {
        decoder_layer_forward_decode(layer, hidden_states, position_ids,
                                     layer_ws, stream);
    }

    // Final RMSNorm + LM head
    __nv_bfloat16* norm_out = layer_ws;
    asr_ops::invoke_rmsnorm(norm_out, hidden_states, final_norm_w_,
                            config_.rms_norm_eps, 1, h, stream);

    linear_nobias(cublas_, logits_out, norm_out, lm_head_w_,
                  1, h, config_.vocab_size, stream);

    cache_seq_len_++;
}

} // namespace asr
} // namespace deusridet
