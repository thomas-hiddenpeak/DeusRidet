/**
 * @file model_runtime.cpp
 * @philosophical_role Runtime body — the repacked FP16 projections, the pool-freeing ritual, and the scratch space (InferenceState) where a forward pass actually breathes. Where model.cpp declares what exists, model_runtime.cpp prepares and disposes the workbench.
 * @serves merge_projection_weights, free_model_weights, InferenceState::{allocate,free}.
 */
// model_runtime.cpp — post-load FP16 repack for prefill GEMM, pool release,
// and InferenceState scratch allocation. Peer TU of model.cpp under R1 cap.

#include "model.h"
#include "layer.h"
#include "fp16_gemm.h"
#include "marlin.h"
#include "allocator.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <vector>

namespace deusridet {

// ============================================================================
// merge_projection_weights — create repacked FP16 weights for prefill GEMM
//
// For prefill (M>1), the custom fp16_gemm kernel requires tile-repacked weights.
// This function creates repacked FP16 weights for all attention projections:
//   DN: qkv+a+b merged → repacked_qkv_ab [10496,5120]
//       z → repacked_z [6144,5120], out → repacked_out [5120,6144]
//   FA: q → repacked_q [12288,5120]
//       k+v merged → repacked_kv [2048,5120], o → repacked_o [5120,6144]
// Decode (M=1) uses original FP16 row-major weights with fp16_gemv.
// ============================================================================

// Helper: allocate + repack a single FP16 weight for fp16_gemm
static bool repack_fp16_weight(const __half* src, __half*& dst, int N, int K) {
    size_t bytes = (size_t)N * K * sizeof(__half);
    if (cudaMalloc(&dst, bytes) != cudaSuccess) return false;
    fp16_repack_b(src, dst, N, K);
    return true;
}

// Helper: create merged FP16 from multiple row-major sources, then repack
static bool merge_and_repack_fp16(const Linear* srcs, int nsrc,
                                  __half*& repacked, int merged_N, int K) {
    size_t bytes = (size_t)merged_N * K * sizeof(__half);

    // Temporary merged buffer
    __half* temp = nullptr;
    if (cudaMalloc(&temp, bytes) != cudaSuccess) return false;
    cudaMemset(temp, 0, bytes);  // zero-fill (handles padding rows)

    // Copy rows from each source
    size_t row_offset = 0;
    for (int i = 0; i < nsrc; i++) {
        int N_src = srcs[i].out_features;
        cudaMemcpy(temp + row_offset * K, srcs[i].weight,
                   (size_t)N_src * K * sizeof(__half), cudaMemcpyDeviceToDevice);
        row_offset += N_src;
    }

    // Allocate repacked buffer and repack
    if (cudaMalloc(&repacked, bytes) != cudaSuccess) {
        cudaFree(temp);
        return false;
    }
    fp16_repack_b(temp, repacked, merged_N, K);
    cudaFree(temp);
    return true;
}

bool merge_projection_weights(ModelWeights& weights) {
    using MC = ModelConfig;
    size_t total_bytes = 0;
    int num_layers = (int)weights.layers.size();

    for (int i = 0; i < num_layers; i++) {
        auto& lw = weights.layers[i];

        if (!MC::is_full_attention(i)) {
            auto& dn = lw.delta_net;

            // Merged qkv+a+b → repacked [10496, 5120]
            Linear srcs[3] = { dn.fp16_qkv, dn.fp16_a, dn.fp16_b };
            if (!merge_and_repack_fp16(srcs, 3, dn.repacked_qkv_ab,
                                       MC::LIN_QKV_AB_DIM, MC::HIDDEN_SIZE)) {
                LOG_ERROR("Model", "Failed to repack DN qkv_ab for layer %d", i);
                return false;
            }
            total_bytes += (size_t)MC::LIN_QKV_AB_DIM * MC::HIDDEN_SIZE * sizeof(__half);

            // z → repacked [6144, 5120]
            if (!repack_fp16_weight(dn.fp16_z.weight, dn.repacked_z,
                                    dn.fp16_z.out_features, dn.fp16_z.in_features)) {
                LOG_ERROR("Model", "Failed to repack DN z for layer %d", i);
                return false;
            }
            total_bytes += (size_t)dn.fp16_z.out_features * dn.fp16_z.in_features * sizeof(__half);

            // out → repacked [5120, 6144]
            if (!repack_fp16_weight(dn.fp16_out.weight, dn.repacked_out,
                                    dn.fp16_out.out_features, dn.fp16_out.in_features)) {
                LOG_ERROR("Model", "Failed to repack DN out for layer %d", i);
                return false;
            }
            total_bytes += (size_t)dn.fp16_out.out_features * dn.fp16_out.in_features * sizeof(__half);
        } else {
            auto& fa = lw.full_attn;

            // q → repacked [12288, 5120]
            if (!repack_fp16_weight(fa.fp16_q.weight, fa.repacked_q,
                                    fa.fp16_q.out_features, fa.fp16_q.in_features)) {
                LOG_ERROR("Model", "Failed to repack FA q for layer %d", i);
                return false;
            }
            total_bytes += (size_t)fa.fp16_q.out_features * fa.fp16_q.in_features * sizeof(__half);

            // Merged k+v → repacked [2048, 5120]
            Linear kv_srcs[2] = { fa.fp16_k, fa.fp16_v };
            if (!merge_and_repack_fp16(kv_srcs, 2, fa.repacked_kv,
                                       MC::FA_KV_DIM, MC::HIDDEN_SIZE)) {
                LOG_ERROR("Model", "Failed to repack FA kv for layer %d", i);
                return false;
            }
            total_bytes += (size_t)MC::FA_KV_DIM * MC::HIDDEN_SIZE * sizeof(__half);

            // o → repacked [5120, 6144]
            if (!repack_fp16_weight(fa.fp16_o.weight, fa.repacked_o,
                                    fa.fp16_o.out_features, fa.fp16_o.in_features)) {
                LOG_ERROR("Model", "Failed to repack FA o for layer %d", i);
                return false;
            }
            total_bytes += (size_t)fa.fp16_o.out_features * fa.fp16_o.in_features * sizeof(__half);
        }

        // Repack FP16 MLP weights (unquantized models only)
        if (!MC::MLP_IS_GPTQ) {
            auto& mlp = lw.mlp;
            if (mlp.fp16_gate_proj.weight) {
                if (!repack_fp16_weight(mlp.fp16_gate_proj.weight, mlp.repacked_gate,
                                        mlp.fp16_gate_proj.out_features, mlp.fp16_gate_proj.in_features)) {
                    LOG_ERROR("Model", "Failed to repack MLP gate for layer %d", i);
                    return false;
                }
                total_bytes += (size_t)mlp.fp16_gate_proj.out_features * mlp.fp16_gate_proj.in_features * sizeof(__half);
            }
            if (mlp.fp16_up_proj.weight) {
                if (!repack_fp16_weight(mlp.fp16_up_proj.weight, mlp.repacked_up,
                                        mlp.fp16_up_proj.out_features, mlp.fp16_up_proj.in_features)) {
                    LOG_ERROR("Model", "Failed to repack MLP up for layer %d", i);
                    return false;
                }
                total_bytes += (size_t)mlp.fp16_up_proj.out_features * mlp.fp16_up_proj.in_features * sizeof(__half);
            }
            if (mlp.fp16_down_proj.weight) {
                if (!repack_fp16_weight(mlp.fp16_down_proj.weight, mlp.repacked_down,
                                        mlp.fp16_down_proj.out_features, mlp.fp16_down_proj.in_features)) {
                    LOG_ERROR("Model", "Failed to repack MLP down for layer %d", i);
                    return false;
                }
                total_bytes += (size_t)mlp.fp16_down_proj.out_features * mlp.fp16_down_proj.in_features * sizeof(__half);
            }
        }
    }

    cudaDeviceSynchronize();
    LOG_INFO("Model", "Repacked FP16 projection weights: %.1f MB (%d DN, %d FA layers%s)",
             total_bytes / 1048576.0, num_layers - MC::NUM_FA_LAYERS, MC::NUM_FA_LAYERS,
             MC::MLP_IS_GPTQ ? "" : " + MLP");
    return true;
}

// ============================================================================
// free_model_weights — release pool blocks
// ============================================================================

void free_model_weights(ModelWeights& w) {
    // All tensor pointers are sub-allocated from pool blocks.
    // Only free the pool blocks themselves.
    for (void* block : w.pool_blocks) {
        cudaFree(block);
    }
    w.pool_blocks.clear();

    // Free separately-allocated INT8 lm_head (not in pool)
    if (w.lm_head_int8.weight) { cudaFree(w.lm_head_int8.weight); }
    if (w.lm_head_int8.scales) { cudaFree(w.lm_head_int8.scales); }
    w.lm_head_int8 = Int8Linear{};

    // Free repacked FP16 weights (allocated by merge_projection_weights)
    auto free_half = [](__half*& p) { if (p) { cudaFree(p); p = nullptr; } };
    for (size_t i = 0; i < w.layers.size(); i++) {
        auto& lw = w.layers[i];
        free_half(lw.delta_net.repacked_qkv_ab);
        free_half(lw.delta_net.repacked_z);
        free_half(lw.delta_net.repacked_out);
        free_half(lw.full_attn.repacked_q);
        free_half(lw.full_attn.repacked_kv);
        free_half(lw.full_attn.repacked_o);
        free_half(lw.mlp.repacked_gate);
        free_half(lw.mlp.repacked_up);
        free_half(lw.mlp.repacked_down);
    }

    // Clear all layer data
    w.layers.clear();

    w.embed_tokens = nullptr;
    w.final_norm   = nullptr;
    w.lm_head      = nullptr;
    w.total_bytes = 0;
}

// ============================================================================
// InferenceState allocation
// ============================================================================

bool InferenceState::allocate(int max_seq) {
    using MC = ModelConfig;
    max_seq_len = max_seq;
    size_t total = 0;

    auto alloc_fp16 = [&](size_t numel) -> __half* {
        size_t bytes = numel * sizeof(__half);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, bytes);
        total += bytes;
        return static_cast<__half*>(p);
    };

    auto alloc_f32 = [&](size_t numel) -> float* {
        size_t bytes = numel * sizeof(float);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, bytes);
        total += bytes;
        return static_cast<float*>(p);
    };

    size_t S = (size_t)max_seq;
    size_t H = MC::HIDDEN_SIZE;

    hidden   = alloc_fp16(S * H);
    residual = alloc_fp16(S * H);
    norm_out = alloc_fp16(S * H);
    attn_out = alloc_fp16(S * MC::ATTN_OUT_DIM);
    q_buf    = alloc_fp16(S * MC::Q_PROJ_DIM);
    kv_buf   = alloc_fp16(S * MC::KV_PROJ_DIM);
    mlp_gate = alloc_fp16(S * MC::INTERMEDIATE_SIZE);
    mlp_up   = alloc_fp16(S * MC::INTERMEDIATE_SIZE);
    mlp_down = alloc_fp16(S * H);
    dn_qkv   = alloc_fp16(S * MC::LIN_QKV_AB_DIM);  // wider for merged qkv+ab prefill
    dn_z     = alloc_fp16(S * MC::LIN_VALUE_DIM);
    dn_a     = alloc_fp16(S * MC::LIN_NUM_V_HEADS);
    dn_b     = alloc_fp16(S * MC::LIN_NUM_V_HEADS);
    dn_g     = alloc_f32(MC::LIN_NUM_V_HEADS);
    dn_beta  = alloc_f32(MC::LIN_NUM_V_HEADS);
    logits   = alloc_fp16(MC::VOCAB_SIZE);
    probs    = alloc_f32(MC::VOCAB_SIZE);  // Sampling workspace (FP32)

    // Marlin GEMM workspace: global barrier lock buffer
    {
        size_t ws = marlin_workspace_size(MC::INTERMEDIATE_SIZE);
        void* p = nullptr;
        if (cudaMalloc(&p, ws) != cudaSuccess) return false;
        cudaMemset(p, 0, ws);
        marlin_workspace = static_cast<int*>(p);
        total += ws;
    }

    // Full Attention scratch: scores and FP16 conversion buffer
    attn_scores = alloc_f32((size_t)MC::NUM_ATTN_HEADS * S);
    scores_h16  = alloc_fp16((size_t)MC::NUM_KV_GROUPS * S);

    // Token IDs on device
    {
        size_t bytes = S * sizeof(int);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return false;
        cudaMemset(p, 0, bytes);
        token_ids = static_cast<int*>(p);
        total += bytes;
    }

    // Sample output (single int for GPU argmax)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return false;
        sample_out = static_cast<int*>(p);
        total += sizeof(int);
    }

    // Device pos for graph-capturable kernels
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return false;
        d_pos = static_cast<int*>(p);
        total += sizeof(int);
    }

    // Pinned host staging for CUDA Graph
    cudaHostAlloc(&h_pos_pinned, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_token_pinned, sizeof(int), cudaHostAllocDefault);

    // Compute stream for graph capture (cannot capture on default stream 0)
    cudaStreamCreate(&compute_stream);

    // Auxiliary stream for concurrent MLP gate+up projections (prefill only)
    cudaStreamCreate(&aux_stream);
    cudaEventCreateWithFlags(&aux_fork_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&aux_join_event, cudaEventDisableTiming);

    // DeltaNet recurrent states: one [128, 128] per head per layer (F32)
    // Count linear attention layers
    num_dn_layers = 0;
    for (int i = 0; i < MC::NUM_LAYERS; i++) {
        if (!MC::is_full_attention(i)) num_dn_layers++;
    }

    size_t state_size = (size_t)MC::LIN_K_HEAD_DIM * MC::LIN_V_HEAD_DIM;  // 128*128
    size_t heads = MC::LIN_NUM_V_HEADS;  // 48

    dn_states = new float*[num_dn_layers];
    for (int i = 0; i < num_dn_layers; i++) {
        dn_states[i] = alloc_f32(heads * state_size);
        if (!dn_states[i]) return false;
    }

    // Conv states: [10240, 3] per linear attention layer (FP16)
    size_t conv_state_size = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1);
    conv_states = new __half*[num_dn_layers];
    for (int i = 0; i < num_dn_layers; i++) {
        conv_states[i] = alloc_fp16(conv_state_size);
        if (!conv_states[i]) return false;
    }

    if (!hidden || !residual || !norm_out || !logits) {
        LOG_ERROR("Model", "InferenceState allocation failed");
        return false;
    }

    LOG_INFO("Model", "InferenceState: %.1f MB for max_seq=%d (%d DN layers)",
             total / 1048576.0, max_seq, num_dn_layers);
    return true;
}

void InferenceState::free() {
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(hidden);   safe_free(residual);  safe_free(norm_out);
    safe_free(attn_out); safe_free(q_buf);     safe_free(kv_buf);
    safe_free(mlp_gate); safe_free(mlp_up);    safe_free(mlp_down);
    safe_free(dn_qkv);  safe_free(dn_z);      safe_free(dn_a);
    safe_free(dn_b);    safe_free(dn_g);       safe_free(dn_beta);
    safe_free(token_ids);  safe_free(sample_out);  safe_free(logits);  safe_free(probs);
    safe_free(marlin_workspace);
    safe_free(attn_scores); safe_free(scores_h16);
    safe_free(d_pos);
    if (h_pos_pinned) { cudaFreeHost(h_pos_pinned); h_pos_pinned = nullptr; }
    if (h_token_pinned) { cudaFreeHost(h_token_pinned); h_token_pinned = nullptr; }
    if (cuda_graph_exec) { cudaGraphExecDestroy(cuda_graph_exec); cuda_graph_exec = nullptr; }
    if (cuda_graph) { cudaGraphDestroy(cuda_graph); cuda_graph = nullptr; }
    if (compute_stream) { cudaStreamDestroy(compute_stream); compute_stream = nullptr; }
    if (aux_stream) { cudaStreamDestroy(aux_stream); aux_stream = nullptr; }
    if (aux_fork_event) { cudaEventDestroy(aux_fork_event); aux_fork_event = nullptr; }
    if (aux_join_event) { cudaEventDestroy(aux_join_event); aux_join_event = nullptr; }
    graph_captured = false;

    if (dn_states) {
        for (int i = 0; i < num_dn_layers; i++) safe_free(dn_states[i]);
        delete[] dn_states;
        dn_states = nullptr;
    }
    if (conv_states) {
        for (int i = 0; i < num_dn_layers; i++) safe_free(conv_states[i]);
        delete[] conv_states;
        conv_states = nullptr;
    }
    *this = InferenceState{};
}


} // namespace deusridet
