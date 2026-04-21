/**
 * @file src/machina/forward_prefill.cu
 * @philosophical_role
 *   Peer TU of forward.cu under R1 800-line hard cap — batched prefill path (mlp/attention/deltanet prefill + forward_prefill).
 * @serves
 *   Conscientia + Actus diagnostic verbs.
 */
#include "forward.h"
#include "forward_kernels.cuh"
#include "layer.h"
#include "gptq.h"
#include "gptq_gemm_v2.h"
#include "marlin.h"
#include "fp16_gemm.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

namespace deusridet {

// ============================================================================
// Batched prefill: SwiGLU MLP (M>1)
//
// Uses Marlin GEMM for all GPTQ INT4 projections.
// Residual add done separately via elementwise_add.
// ============================================================================

void mlp_forward_prefill(const __half* x, const MLPWeights& mlp,
                        __half* residual, int M,
                        InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;
    int N_inter = MC::INTERMEDIATE_SIZE;

    if (MC::MLP_IS_GPTQ) {
        // GPTQ-Int4 path
        gptq_gemm_v2(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                     M, mlp.gate_proj.K, mlp.gate_proj.N, stream);
        gptq_gemm_v2(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                     M, mlp.up_proj.K, mlp.up_proj.N, stream);

        silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                 M * N_inter, stream);

        if (residual) {
            gptq_gemm_v2_add(state.mlp_gate, mlp.down_proj.qweight, residual,
                             mlp.down_proj.scales, M, mlp.down_proj.K, mlp.down_proj.N, stream);
        } else {
            gptq_gemm_v2(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down,
                         mlp.down_proj.scales, M, mlp.down_proj.K, mlp.down_proj.N, stream);
        }
    } else {
        // FP16 path (repacked weights for prefill GEMM)
        if (mlp.repacked_gate) {
            fp16_gemm(x, mlp.repacked_gate, state.mlp_gate,
                      M, MC::HIDDEN_SIZE, N_inter, stream);
            fp16_gemm(x, mlp.repacked_up, state.mlp_up,
                      M, MC::HIDDEN_SIZE, N_inter, stream);
        } else {
            linear_forward(x, mlp.fp16_gate_proj, state.mlp_gate, M, stream);
            linear_forward(x, mlp.fp16_up_proj, state.mlp_up, M, stream);
        }

        silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                 M * N_inter, stream);

        if (mlp.repacked_down) {
            fp16_gemm(state.mlp_gate, mlp.repacked_down, state.mlp_down,
                      M, N_inter, MC::HIDDEN_SIZE, stream);
        } else {
            linear_forward(state.mlp_gate, mlp.fp16_down_proj, state.mlp_down, M, stream);
        }

        if (residual) {
            elementwise_add(residual, state.mlp_down, residual,
                            M * MC::HIDDEN_SIZE, stream);
        }
    }
}

// ============================================================================
// Batched prefill: split_q_gate for M tokens
// q_interleaved[M, 24, 512] → Q[M, 24, 256] + Gate[M, 24, 256]
// ============================================================================


// ============================================================================
// Batched prefill: kv_cache_write for M tokens at consecutive positions
// ============================================================================


// ============================================================================
// Batched prefill: RoPE for M tokens at consecutive positions
// ============================================================================


// ============================================================================
// Prefill causal attention kernel (M queries attending to their own KV)
//
// Simple implementation: one block per (query_head, token). Each block
// computes attention for one query head at one token position.
// For small M (≤ 128), this gives sufficient parallelism.
//
// Grid: (num_attn_heads, M), Block: head_dim (256)
// ============================================================================


// ============================================================================
// Batched prefill: Full Attention GQA (M tokens)
//
// Strategy: batch projections (GEMM), batch element-wise ops,
// batch causal self-attention, batch output projection.
// ============================================================================

void full_attention_prefill(const __half* x, const FullAttentionWeights& attn,
                           __half* kv_cache, int layer_idx,
                           int pos_start, int M, int max_kv_len,
                           InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // 1. Batched Q, K, V projections (FP16 GEMM with repacked weights)
    // Q: x[M, 5120] → q_buf[M, 12288]
    fp16_gemm(x, attn.repacked_q, state.q_buf,
              M, MC::HIDDEN_SIZE, MC::Q_PROJ_DIM, stream);

    // K+V: merged projection → attn_out[M, 2048], then split
    fp16_gemm(x, attn.repacked_kv, state.attn_out,
              M, MC::HIDDEN_SIZE, MC::FA_KV_DIM, stream);
    // Split: attn_out[M, 2048] → kv_buf[M, 1024] (K) + dn_z[M, 1024] (V)
    cudaMemcpy2DAsync(state.kv_buf, MC::KV_PROJ_DIM * sizeof(__half),
                      state.attn_out, MC::FA_KV_DIM * sizeof(__half),
                      MC::KV_PROJ_DIM * sizeof(__half), M,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(state.dn_z, MC::KV_PROJ_DIM * sizeof(__half),
                      state.attn_out + MC::KV_PROJ_DIM, MC::FA_KV_DIM * sizeof(__half),
                      MC::KV_PROJ_DIM * sizeof(__half), M,
                      cudaMemcpyDeviceToDevice, stream);

    // 2. Batched split_q_gate: q_buf[M, 12288] → Q_sep[M, 6144] + Gate[M, 6144]
    // Q_sep → dn_qkv[M, 10240] (only first 6144 per row used)
    // Gate → mlp_gate[M, 17408] (only first 6144 per row used)
    __half* Qsep = state.dn_qkv;
    __half* Gate = state.mlp_gate;
    {
        dim3 grid(MC::NUM_ATTN_HEADS, M);
        split_q_gate_batch_kernel<<<grid, 256, 0, stream>>>(
            state.q_buf, Qsep, Gate,
            M, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);
    }

    // 3. Batched head_norm for Q (M * 24 heads) and K (M * 4 heads)
    head_norm(Qsep, attn.q_norm, Qsep,
              M * MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);
    head_norm(state.kv_buf, attn.k_norm, state.kv_buf,
              M * MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, stream);

    // 4. Batched RoPE for M tokens at consecutive positions
    {
        int max_heads = (MC::NUM_ATTN_HEADS > MC::NUM_KV_HEADS)
                        ? MC::NUM_ATTN_HEADS : MC::NUM_KV_HEADS;
        dim3 grid(max_heads, M);
        rope_batch_kernel<<<grid, MC::ROTARY_DIM / 2, 0, stream>>>(
            Qsep, state.kv_buf,
            MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS,
            MC::HEAD_DIM, MC::ROTARY_DIM,
            pos_start, M, MC::ROPE_THETA);
    }

    // 5. Batched KV cache write
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    __half* k_cache = kv_cache + (size_t)layer_idx * 2 * kv_plane;
    __half* v_cache_ptr = k_cache + kv_plane;
    {
        dim3 grid(MC::NUM_KV_HEADS, M);
        kv_cache_write_batch_kernel<<<grid, 256, 0, stream>>>(
            state.kv_buf, state.dn_z, k_cache, v_cache_ptr,
            pos_start, M, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_HEADS);
    }

    // 6. Batched causal attention
    // Grid: (num_attn_heads, M), each block does one head for one token
    {
        float scale_val = 1.0f / sqrtf((float)MC::HEAD_DIM);
        dim3 grid(MC::NUM_ATTN_HEADS, M);
        // Output to attn_out[M, 6144]
        prefill_attention_kernel<<<grid, MC::HEAD_DIM, 0, stream>>>(
            Qsep, k_cache, v_cache_ptr, state.attn_out,
            pos_start, M, max_kv_len, MC::HEAD_DIM,
            MC::NUM_KV_GROUPS, scale_val);
    }

    // 7. Batched sigmoid_gate: attn_out = attn_out * sigmoid(gate)
    sigmoid_gate(state.attn_out, Gate, state.attn_out,
                 M * MC::ATTN_OUT_DIM, stream);

    // 8. Batched o_proj: attn_out[M, 6144] → norm_out[M, 5120] (FP16 GEMM)
    fp16_gemm(state.attn_out, attn.repacked_o, state.norm_out,
              M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, stream);
}

// ============================================================================
// Batched prefill: DeltaNet (M tokens)
//
// Strategy: batch projections (INT8 GEMM), process conv1d + recurrent
// sequentially per token, batch output (rms_norm_gated + out_proj).
// ============================================================================

void deltanet_prefill(const __half* x, const DeltaNetWeights& dn,
                      int dn_layer_idx, int M,
                      InferenceState& state, cudaStream_t stream) {
    using MC = ModelConfig;

    // 1. Merged qkv+a+b projection: x[M, 5120] → dn_qkv[M, 10496] (FP16 GEMM)
    fp16_gemm(x, dn.repacked_qkv_ab, state.dn_qkv,
              M, MC::HIDDEN_SIZE, MC::LIN_QKV_AB_DIM, stream);

    // Z: x[M, 5120] → dn_z[M, 6144] (FP16 GEMM)
    fp16_gemm(x, dn.repacked_z, state.dn_z,
              M, MC::HIDDEN_SIZE, MC::LIN_VALUE_DIM, stream);

    // 2. Batch conv1d: all M tokens in one launch (stride=10368 for merged buffer)
    {
        int conv_blocks = (MC::LIN_CONV_DIM + 255) / 256;
        conv1d_batch_silu_kernel<<<conv_blocks, 256, 0, stream>>>(
            state.dn_qkv, state.conv_states[dn_layer_idx],
            dn.conv1d_weight, M, MC::LIN_CONV_DIM, MC::CONV_KERNEL,
            MC::LIN_QKV_AB_DIM);
    }

    // 3. Fused head kernel: repeat_interleave + g/beta + L2norm + recurrent
    //    a/b read from within dn_qkv buffer at offsets 10240 and 10288
    {
        int fused_smem = (MC::LIN_K_HEAD_DIM + MC::LIN_K_HEAD_DIM + 4) * sizeof(float);
        deltanet_fused_head_kernel<<<MC::LIN_NUM_V_HEADS, 128, fused_smem, stream>>>(
            state.dn_qkv,
            dn.A_log, dn.dt_bias,
            state.dn_states[dn_layer_idx],
            state.attn_out,
            M, MC::LIN_NUM_K_HEADS, MC::LIN_NUM_V_HEADS,
            MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM,
            MC::LIN_KEY_DIM, MC::LIN_QKV_AB_DIM,
            MC::LIN_CONV_DIM, MC::LIN_CONV_DIM + MC::LIN_NUM_V_HEADS,
            MC::LIN_QKV_AB_DIM, MC::RMS_EPS);
    }

    // 4. Batched gated RMSNorm: attn_out[M, 48, 128] with gate dn_z[M, 48, 128]
    rms_norm_gated(state.attn_out, state.dn_z,
                   dn.norm_weight, state.attn_out,
                   M * MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, stream);

    // 5. Batched out_proj: attn_out[M, 6144] → norm_out[M, 5120] (FP16 GEMM)
    fp16_gemm(state.attn_out, dn.repacked_out, state.norm_out,
              M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, stream);
}

// ============================================================================
// Complete batched prefill forward pass
//
// embed[M] → for each layer: norm → attn/deltanet → residual → norm → MLP
// → final_norm → lm_head (last token only) → greedy sample
// ============================================================================

int forward_prefill(const ModelWeights& model,
                    InferenceState& state,
                    __half* kv_cache,
                    const int* h_token_ids, int M,
                    int pos_start, int max_kv_len,
                    cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Copy token IDs to device
    cudaMemcpyAsync(state.token_ids, h_token_ids, M * sizeof(int),
                    cudaMemcpyHostToDevice, s);

    // Embedding lookup → hidden[M, H]
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, M, MC::HIDDEN_SIZE, s);

    // Copy to residual
    cudaMemcpyAsync(state.residual, state.hidden,
                    (size_t)M * MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, s);

    int dn_layer_idx = 0;

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        // Pre-attention RMSNorm: residual[M, H] → norm_out[M, H]
        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // Attention / DeltaNet (writes output to norm_out[M, H])
        if (lw.is_full_attention) {
            full_attention_prefill(state.norm_out, lw.full_attn,
                                   kv_cache, layer, pos_start, M, max_kv_len,
                                   state, s);
        } else {
            deltanet_prefill(state.norm_out, lw.delta_net,
                             dn_layer_idx, M, state, s);
            dn_layer_idx++;
        }

        // Fused: residual += attn_output; norm_out = RMSNorm(residual)
        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // MLP with residual add
        mlp_forward_prefill(state.norm_out, lw.mlp, state.residual, M, state, s);
    }

    // Final RMSNorm — only the last token matters for logits
    __half* last_hidden = state.residual + (size_t)(M - 1) * MC::HIDDEN_SIZE;
    rms_norm(last_hidden, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

    // LM head: hidden[5120] → logits[248320] (last token only)
    int8_linear_forward(state.hidden, model.lm_head_int8, state.logits, 1, s);

    // GPU argmax
    argmax_async(state.logits, MC::VOCAB_SIZE, state.sample_out, s);

    // Extract result
    int result;
    cudaMemcpyAsync(&result, state.sample_out, sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    return result;
}


} // namespace deusridet
