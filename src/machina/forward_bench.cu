/**
 * @file src/machina/forward_bench.cu
 * @philosophical_role
 *   Peer TU of forward.cu under R1 800-line hard cap — profile_forward (decode) + bench_prefill_projections.
 * @serves
 *   Conscientia + Actus diagnostic verbs.
 */
#include "forward.h"
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
// Profile forward pass — time each component type across all layers
// ============================================================================

void profile_forward(const ModelWeights& model,
                     InferenceState& state,
                     __half* kv_cache,
                     int token_id, int pos, int max_kv_len,
                     cudaStream_t stream) {
    using MC = ModelConfig;

    // Run a warmup token first to populate caches etc.
    forward_one_token(model, state, kv_cache, token_id, pos, max_kv_len, stream);

    // Now profile at pos+1
    pos++;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    auto timed = [&](const char* label, auto fn) {
        cudaEventRecord(e0, stream);
        fn();
        cudaEventRecord(e1, stream);
        cudaStreamSynchronize(stream);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        printf("  %-32s  %7.2f ms\n", label, ms);
        return ms;
    };

    printf("\n=== Forward pass profile (pos=%d) ===\n", pos);

    // Setup
    cudaMemcpyAsync(state.token_ids, &token_id, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(state.d_pos, &pos, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, 1, MC::HIDDEN_SIZE, stream);
    cudaMemcpyAsync(state.residual, state.hidden,
                    MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    float total_dn_attn = 0, total_fa_attn = 0, total_mlp = 0, total_norm = 0;
    int dn_layer_idx = 0;

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        // Norm + Attn + Norm + MLP timed as a unit per layer type
        if (layer == 0 || layer == 3) {
            // Time individual components for first DeltaNet (layer0) and first FullAttn (layer3)
            const char* lt = lw.is_full_attention ? "FullAttn" : "DeltaNet";
            char buf[64];

            snprintf(buf, sizeof(buf), "L%d %s pre_norm", layer, lt);
            timed(buf, [&]{ rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                                      1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream); });

            snprintf(buf, sizeof(buf), "L%d %s attn", layer, lt);
            float attn_ms;
            if (lw.is_full_attention) {
                attn_ms = timed(buf, [&]{ full_attention_forward(state.norm_out, lw.full_attn,
                                    kv_cache, layer, pos, max_kv_len, state, stream, true); });
                total_fa_attn += attn_ms;
            } else {
                attn_ms = timed(buf, [&]{ deltanet_forward(state.norm_out, lw.delta_net,
                                    dn_layer_idx, state, stream); });
                dn_layer_idx++;
                total_dn_attn += attn_ms;
            }

            elementwise_add(state.residual, state.norm_out, state.residual,
                            MC::HIDDEN_SIZE, stream);

            snprintf(buf, sizeof(buf), "L%d %s post_norm", layer, lt);
            timed(buf, [&]{ rms_norm(state.residual, lw.post_attn_layernorm, state.norm_out,
                                      1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream); });

            snprintf(buf, sizeof(buf), "L%d %s mlp", layer, lt);
            float mlp_ms = timed(buf, [&]{ mlp_forward(state.norm_out, lw.mlp, state.residual, state, stream); });
            total_mlp += mlp_ms;

        } else {
            // Bulk timing for remaining layers
            cudaEventRecord(e0, stream);
            rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                     1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);
            if (lw.is_full_attention) {
                full_attention_forward(state.norm_out, lw.full_attn,
                                       kv_cache, layer, pos, max_kv_len, state, stream);
            } else {
                deltanet_forward(state.norm_out, lw.delta_net,
                                 dn_layer_idx, state, stream);
                dn_layer_idx++;
            }
            elementwise_add(state.residual, state.norm_out, state.residual,
                            MC::HIDDEN_SIZE, stream);
            rms_norm(state.residual, lw.post_attn_layernorm, state.norm_out,
                     1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);
            mlp_forward(state.norm_out, lw.mlp, state.residual, state, stream);
            cudaEventRecord(e1, stream);
            cudaStreamSynchronize(stream);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            // Estimate split: ~70% attn, ~30% MLP based on weight sizes
            if (lw.is_full_attention) total_fa_attn += ms * 0.55f;
            else total_dn_attn += ms * 0.55f;
            total_mlp += ms * 0.35f;
            total_norm += ms * 0.10f;
        }
    }

    // LM head
    Linear lm_head_linear;
    lm_head_linear.weight = model.lm_head;
    lm_head_linear.in_features = MC::HIDDEN_SIZE;
    lm_head_linear.out_features = MC::VOCAB_SIZE;
    float lm_ms = timed("lm_head", [&]{ linear_forward(state.hidden, lm_head_linear, state.logits, 1, stream); });

    float sample_ms = timed("greedy_sample", [&]{ greedy_sample(state.logits, MC::VOCAB_SIZE, state.sample_out, stream); });

    printf("\n  --- Summary ---\n");
    printf("  DeltaNet attn (48 layers):    %7.1f ms\n", total_dn_attn);
    printf("  FullAttn (16 layers):         %7.1f ms\n", total_fa_attn);
    printf("  MLP (64 layers):              %7.1f ms\n", total_mlp);
    printf("  Norms:                        %7.1f ms\n", total_norm);
    printf("  LM Head:                      %7.1f ms\n", lm_ms);
    printf("  Greedy sample:                %7.1f ms\n", sample_ms);
    printf("  Estimated total:              %7.1f ms\n",
           total_dn_attn + total_fa_attn + total_mlp + total_norm + lm_ms + sample_ms);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

// ============================================================================
// Benchmark: Marlin INT4 vs cuBLAS FP16 attention projections at various M
//
// For the continuous consciousness use case, prefill frames range from
// small (1-16 tokens during conversation) to large (100-1000+ during
// context merging, dream consolidation). This benchmark measures actual
// throughput at each M to determine the optimal crossover point.
// ============================================================================

void bench_prefill_projections(const ModelWeights& model,
                               InferenceState& state,
                               cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Find representative layers
    int dn_idx = -1, fa_idx = -1;
    for (int i = 0; i < MC::NUM_LAYERS; i++) {
        if (!model.layers[i].is_full_attention && dn_idx < 0) dn_idx = i;
        if (model.layers[i].is_full_attention && fa_idx < 0) fa_idx = i;
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // M values to test — small to large, covering all regimes
    const int M_vals[] = {1, 4, 11, 32, 64, 128, 256, 512, 1024, 2048};
    const int N_M = sizeof(M_vals) / sizeof(M_vals[0]);
    const int WARMUP = 3;
    const int ITERS  = 10;

    // Check max M supported by state
    int max_m = state.max_seq_len;

    printf("\n=== Prefill Projection Benchmark: Marlin INT4 vs cuBLAS FP16 ===\n");
    printf("Warmup: %d, Iterations: %d per measurement\n", WARMUP, ITERS);
    printf("Max M supported by state: %d\n\n", max_m);

    // Helper: time a lambda over ITERS iterations (after WARMUP)
    auto bench = [&](auto fn) -> float {
        for (int i = 0; i < WARMUP; i++) fn();
        cudaStreamSynchronize(s);
        cudaEventRecord(e0, s);
        for (int i = 0; i < ITERS; i++) fn();
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / ITERS;
    };

    // ---- DeltaNet projections (48 layers) ----
    if (dn_idx >= 0) {
        const auto& dn = model.layers[dn_idx].delta_net;
        const __half* x = state.norm_out;  // dummy input, shape doesn't matter for timing

        printf("--- DeltaNet Projections (layer %d, ×48) ---\n", dn_idx);
        printf("%-6s | %-40s | %-40s | %s\n", "M", "Custom FP16 (ms)", "cuBLAS FP16 (ms)", "Speedup");
        printf("%-6s | %-12s %-12s %-12s | %-12s %-12s %-12s | %s\n",
               "", "qkv_ab", "z", "out", "qkv", "z", "out", "custom/cublas");
        printf("%s\n", std::string(120, '-').c_str());

        for (int mi = 0; mi < N_M; mi++) {
            int M = M_vals[mi];
            if (M > max_m) break;

            // Custom FP16 GEMM (repacked weights)
            float m_qkv = bench([&]{ fp16_gemm(x, dn.repacked_qkv_ab, state.dn_qkv,
                M, MC::HIDDEN_SIZE, MC::LIN_QKV_AB_DIM, s); });
            float m_z = bench([&]{ fp16_gemm(x, dn.repacked_z, state.dn_z,
                M, MC::HIDDEN_SIZE, MC::LIN_VALUE_DIM, s); });
            float m_out = bench([&]{ fp16_gemm(state.attn_out, dn.repacked_out, state.norm_out,
                M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s); });

            // cuBLAS FP16 — use original separate projections
            float f_qkv = bench([&]{ linear_forward(x, dn.fp16_qkv, state.dn_qkv, M, s); });
            float f_z   = bench([&]{ linear_forward(x, dn.fp16_z, state.dn_z, M, s); });
            float f_out = bench([&]{ linear_forward(state.attn_out, dn.fp16_out, state.norm_out, M, s); });

            float m_total = (m_qkv + m_z + m_out) * 48;
            float f_total = (f_qkv + f_z + f_out) * 48;

            printf("%-6d | %-12.3f %-12.3f %-12.3f | %-12.3f %-12.3f %-12.3f | %.2fx\n",
                   M, m_qkv, m_z, m_out, f_qkv, f_z, f_out, f_total / m_total);
            printf("       | total×48: %-28.1f | total×48: %-28.1f |\n", m_total, f_total);
        }
        printf("\n");
    }

    // ---- Full Attention projections (16 layers) ----
    if (fa_idx >= 0) {
        const auto& fa = model.layers[fa_idx].full_attn;
        const __half* x = state.norm_out;

        printf("--- Full Attention Projections (layer %d, ×16) ---\n", fa_idx);
        printf("%-6s | %-40s | %-40s | %s\n", "M", "Custom FP16 (ms)", "cuBLAS FP16 (ms)", "Speedup");
        printf("%-6s | %-12s %-12s %-12s | %-12s %-12s %-12s | %s\n",
               "", "q", "kv", "o", "q", "k+v", "o", "custom/cublas");
        printf("%s\n", std::string(120, '-').c_str());

        for (int mi = 0; mi < N_M; mi++) {
            int M = M_vals[mi];
            if (M > max_m) break;

            // Custom FP16 GEMM (repacked weights)
            float m_q = bench([&]{ fp16_gemm(x, fa.repacked_q, state.q_buf,
                M, MC::HIDDEN_SIZE, MC::Q_PROJ_DIM, s); });
            float m_kv = bench([&]{ fp16_gemm(x, fa.repacked_kv, state.attn_out,
                M, MC::HIDDEN_SIZE, MC::FA_KV_DIM, s); });
            float m_o = bench([&]{ fp16_gemm(state.attn_out, fa.repacked_o, state.norm_out,
                M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s); });

            // cuBLAS FP16 — separate q, k, v projections (not merged)
            float f_q = bench([&]{ linear_forward(x, fa.fp16_q, state.q_buf, M, s); });
            float f_kv = bench([&]{
                linear_forward(x, fa.fp16_k, state.kv_buf, M, s);
                linear_forward(x, fa.fp16_v, state.dn_z, M, s);
            });
            float f_o = bench([&]{ linear_forward(state.attn_out, fa.fp16_o, state.norm_out, M, s); });

            float m_total = (m_q + m_kv + m_o) * 16;
            float f_total = (f_q + f_kv + f_o) * 16;

            printf("%-6d | %-12.3f %-12.3f %-12.3f | %-12.3f %-12.3f %-12.3f | %.2fx\n",
                   M, m_q, m_kv, m_o, f_q, f_kv, f_o, f_total / m_total);
            printf("       | total×16: %-28.1f | total×16: %-28.1f |\n", m_total, f_total);
        }
        printf("\n");
    }

    // ---- MLP (reference, Marlin only, 64 layers) ----
    {
        const auto& mlp = model.layers[0].mlp;
        const __half* x = state.norm_out;

        printf("--- MLP Projections (layer 0, ×64, Marlin INT4 only) ---\n");
        printf("%-6s | %-12s %-12s %-12s | %-12s\n", "M", "gate", "up", "down", "total×64");
        printf("%s\n", std::string(80, '-').c_str());

        for (int mi = 0; mi < N_M; mi++) {
            int M = M_vals[mi];
            if (M > max_m) break;

            float m_gate = bench([&]{ marlin_gemm(x, mlp.gate_proj.qweight, state.mlp_gate,
                mlp.gate_proj.scales, state.marlin_workspace, M,
                mlp.gate_proj.K, mlp.gate_proj.N, 128, s); });
            float m_up = bench([&]{ marlin_gemm(x, mlp.up_proj.qweight, state.mlp_up,
                mlp.up_proj.scales, state.marlin_workspace, M,
                mlp.up_proj.K, mlp.up_proj.N, 128, s); });
            float m_down = bench([&]{ marlin_gemm(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down,
                mlp.down_proj.scales, state.marlin_workspace, M,
                mlp.down_proj.K, mlp.down_proj.N, 128, s); });

            float total = (m_gate + m_up + m_down) * 64;
            printf("%-6d | %-12.3f %-12.3f %-12.3f | %-12.1f\n",
                   M, m_gate, m_up, m_down, total);
        }
        printf("\n");
    }

    // ---- Custom FP16 GEMM kernel benchmark (repacked vs cuBLAS) ----
    {
        printf("--- Custom FP16 GEMM (repacked) vs cuBLAS FP16 ---\n");
        printf("(DeltaNet layer %d: qkv_ab 5120→10496, z 5120→6144, out 6144→5120)\n", dn_idx);
        printf("%-6s | %-36s | %-36s\n", "M",
               "Custom FP16 (ms)", "cuBLAS FP16 (ms)");
        printf("%-6s | %-12s %-12s %-12s | %-12s %-12s %-12s\n",
               "", "qkv_ab", "z", "out", "qkv", "z", "out");
        printf("%s\n", std::string(100, '-').c_str());

        const auto& dn = model.layers[dn_idx >= 0 ? dn_idx : 0].delta_net;
        const __half* x = state.norm_out;

        for (int mi = 0; mi < N_M; mi++) {
            int M = M_vals[mi];
            if (M > max_m || M < 2) continue;  // fp16_gemm requires M>=2

            // Custom FP16 kernel (repacked weights)
            float c_qkv = bench([&]{ fp16_gemm(x, dn.repacked_qkv_ab, state.dn_qkv,
                M, MC::HIDDEN_SIZE, MC::LIN_QKV_AB_DIM, s); });
            float c_z   = bench([&]{ fp16_gemm(x, dn.repacked_z, state.dn_z,
                M, MC::HIDDEN_SIZE, MC::LIN_VALUE_DIM, s); });
            float c_out = bench([&]{ fp16_gemm(state.attn_out, dn.repacked_out, state.norm_out,
                M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s); });

            // cuBLAS FP16
            float f_qkv = bench([&]{ linear_forward(x, dn.fp16_qkv, state.dn_qkv, M, s); });
            float f_z   = bench([&]{ linear_forward(x, dn.fp16_z, state.dn_z, M, s); });
            float f_out = bench([&]{ linear_forward(state.attn_out, dn.fp16_out, state.norm_out, M, s); });

            float c_total = (c_qkv + c_z + c_out) * 48;
            float f_total = (f_qkv + f_z + f_out) * 48;

            printf("%-6d | %-12.3f %-12.3f %-12.3f | %-12.3f %-12.3f %-12.3f\n",
                   M, c_qkv, c_z, c_out, f_qkv, f_z, f_out);
            printf("       | DN×48: %-10.1f (%.0f%%)      | DN×48: %-10.1f (100%%)\n",
                   c_total, c_total/f_total*100, f_total);
        }
        printf("\n");
    }

    // ---- Summary: estimated full prefill (Custom FP16 attn + Marlin MLP) ----
    printf("--- Estimated Full Prefill (64 layers) ---\n");
    printf("%-6s | %-15s %-15s | %-7s\n", "M", "FP16 Custom", "FP16 cuBLAS", "Ratio");
    printf("%s\n", std::string(60, '-').c_str());

    for (int mi = 0; mi < N_M; mi++) {
        int M = M_vals[mi];
        if (M > max_m) break;
        if (M < 2) continue;  // fp16_gemm requires M>=2

        const auto& dn = model.layers[dn_idx >= 0 ? dn_idx : 0].delta_net;
        const auto& fa = model.layers[fa_idx >= 0 ? fa_idx : 0].full_attn;
        const auto& mlp = model.layers[0].mlp;
        const __half* x = state.norm_out;

        // Custom FP16 attention + Marlin INT4 MLP
        float custom_dn = bench([&]{
            fp16_gemm(x, dn.repacked_qkv_ab, state.dn_qkv,
                M, MC::HIDDEN_SIZE, MC::LIN_QKV_AB_DIM, s);
            fp16_gemm(x, dn.repacked_z, state.dn_z,
                M, MC::HIDDEN_SIZE, MC::LIN_VALUE_DIM, s);
            fp16_gemm(state.attn_out, dn.repacked_out, state.norm_out,
                M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s);
        });
        float custom_fa = bench([&]{
            fp16_gemm(x, fa.repacked_q, state.q_buf,
                M, MC::HIDDEN_SIZE, MC::Q_PROJ_DIM, s);
            fp16_gemm(x, fa.repacked_kv, state.attn_out,
                M, MC::HIDDEN_SIZE, MC::FA_KV_DIM, s);
            fp16_gemm(state.attn_out, fa.repacked_o, state.norm_out,
                M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s);
        });
        float int4_mlp = bench([&]{
            marlin_gemm(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                state.marlin_workspace, M, mlp.gate_proj.K, mlp.gate_proj.N, 128, s);
            marlin_gemm(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                state.marlin_workspace, M, mlp.up_proj.K, mlp.up_proj.N, 128, s);
            marlin_gemm(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down, mlp.down_proj.scales,
                state.marlin_workspace, M, mlp.down_proj.K, mlp.down_proj.N, 128, s);
        });
        float est_custom = custom_dn * 48 + custom_fa * 16 + int4_mlp * 64;

        // cuBLAS FP16 attention + Marlin INT4 MLP
        float fp16_dn = bench([&]{
            linear_forward(x, dn.fp16_qkv, state.dn_qkv, M, s);
            linear_forward(x, dn.fp16_z, state.dn_z, M, s);
            linear_forward(state.attn_out, dn.fp16_out, state.norm_out, M, s);
        });
        float fp16_fa = bench([&]{
            linear_forward(x, fa.fp16_q, state.q_buf, M, s);
            linear_forward(x, fa.fp16_k, state.kv_buf, M, s);
            linear_forward(x, fa.fp16_v, state.dn_z, M, s);
            linear_forward(state.attn_out, fa.fp16_o, state.norm_out, M, s);
        });
        float est_fp16 = fp16_dn * 48 + fp16_fa * 16 + int4_mlp * 64;

        printf("%-6d | %10.1f ms   %10.1f ms   | %5.2fx\n",
               M, est_custom, est_fp16, est_fp16 / est_custom);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}


} // namespace deusridet
