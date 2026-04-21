/**
 * @file src/machina/forward_profile.cu
 * @philosophical_role
 *   Peer TU of forward.cu under R1 800-line hard cap — sub-layer profiler + profile_forward_prefill.
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
// Sub-layer profiler: measures per-operation timing within DN, FA, MLP.
// Runs one representative layer of each type with fine-grained events.
// Call AFTER a warmup pass so buffers are populated.
// ============================================================================

void profile_sublayer_prefill(const ModelWeights& model,
                              InferenceState& state,
                              __half* kv_cache,
                              int M, int pos_start, int max_kv_len,
                              cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Find first DN and first FA layer
    int first_dn = -1, first_fa = -1;
    for (int i = 0; i < MC::NUM_LAYERS; i++) {
        if (!model.layers[i].is_full_attention && first_dn < 0) first_dn = i;
        if (model.layers[i].is_full_attention && first_fa < 0) first_fa = i;
    }

    constexpr int NEV = 16;
    cudaEvent_t ev[NEV];
    for (int i = 0; i < NEV; i++)
        cudaEventCreateWithFlags(&ev[i], cudaEventDefault);
    auto ms = [&](int a, int b) {
        float t; cudaEventElapsedTime(&t, ev[a], ev[b]); return t;
    };

    // State buffers already contain data from warmup — shapes are correct.
    // Timing depends on weight memory locations and tensor shapes, not data values.

    // === DN Sub-layer (one layer) ===
    if (first_dn >= 0) {
        const auto& dn = model.layers[first_dn].delta_net;
        const __half* x = state.norm_out;
        int e = 0;
        cudaEventRecord(ev[e++], s);  // 0
        fp16_gemm(x, dn.repacked_qkv_ab, state.dn_qkv,
                  M, MC::HIDDEN_SIZE, MC::LIN_QKV_AB_DIM, s);
        cudaEventRecord(ev[e++], s);  // 1
        fp16_gemm(x, dn.repacked_z, state.dn_z,
                  M, MC::HIDDEN_SIZE, MC::LIN_VALUE_DIM, s);
        cudaEventRecord(ev[e++], s);  // 2
        // (ab_proj merged into qkv_ab — no separate event)
        {
            int conv_blocks = (MC::LIN_CONV_DIM + 255) / 256;
            conv1d_batch_silu_kernel<<<conv_blocks, 256, 0, s>>>(
                state.dn_qkv, state.conv_states[0],
                dn.conv1d_weight, M, MC::LIN_CONV_DIM, MC::CONV_KERNEL,
                MC::LIN_QKV_AB_DIM);
        }
        cudaEventRecord(ev[e++], s);  // 3
        {
            int fused_smem = (MC::LIN_K_HEAD_DIM + MC::LIN_K_HEAD_DIM + 4) * sizeof(float);
            deltanet_fused_head_kernel<<<MC::LIN_NUM_V_HEADS, 128, fused_smem, s>>>(
                state.dn_qkv,
                dn.A_log, dn.dt_bias,
                state.dn_states[0],
                state.attn_out,
                M, MC::LIN_NUM_K_HEADS, MC::LIN_NUM_V_HEADS,
                MC::LIN_K_HEAD_DIM, MC::LIN_V_HEAD_DIM,
                MC::LIN_KEY_DIM, MC::LIN_QKV_AB_DIM,
                MC::LIN_CONV_DIM, MC::LIN_CONV_DIM + MC::LIN_NUM_V_HEADS,
                MC::LIN_QKV_AB_DIM, MC::RMS_EPS);
        }
        cudaEventRecord(ev[e++], s);  // 4
        rms_norm_gated(state.attn_out, state.dn_z,
                       dn.norm_weight, state.attn_out,
                       M * MC::LIN_NUM_V_HEADS, MC::LIN_V_HEAD_DIM, MC::RMS_EPS, s);
        cudaEventRecord(ev[e++], s);  // 5
        fp16_gemm(state.attn_out, dn.repacked_out, state.norm_out,
                  M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s);
        cudaEventRecord(ev[e++], s);  // 6
        cudaStreamSynchronize(s);

        float dn_total = ms(0, 6);
        printf("\n=== DN Sub-layer (layer %d, M=%d) ===\n", first_dn, M);
        printf("  qkv_ab_proj (FP16 5120→10496):  %6.3f ms  (%4.1f%%)\n", ms(0,1), 100*ms(0,1)/dn_total);
        printf("  z_proj    (FP16 5120→6144):      %6.3f ms  (%4.1f%%)\n", ms(1,2), 100*ms(1,2)/dn_total);
        printf("  conv1d_silu:                   %6.3f ms  (%4.1f%%)\n", ms(2,3), 100*ms(2,3)/dn_total);
        printf("  fused_head (recurrent):        %6.3f ms  (%4.1f%%)\n", ms(3,4), 100*ms(3,4)/dn_total);
        printf("  rms_norm_gated:                %6.3f ms  (%4.1f%%)\n", ms(4,5), 100*ms(4,5)/dn_total);
        printf("  out_proj  (FP16 6144→5120):    %6.3f ms  (%4.1f%%)\n", ms(5,6), 100*ms(5,6)/dn_total);
        printf("  TOTAL (1 DN layer):            %6.3f ms  (×48 = %.1f ms)\n", dn_total, dn_total*48);
    }

    // === MLP Sub-layer (one layer) ===
    {
        int mlp_layer = (first_dn >= 0) ? first_dn : 0;
        const auto& mlp = model.layers[mlp_layer].mlp;
        const __half* x = state.norm_out;
        int e = 0;
        cudaEventRecord(ev[e++], s);  // 0
        marlin_gemm(x, mlp.gate_proj.qweight, state.mlp_gate, mlp.gate_proj.scales,
                    state.marlin_workspace, M, mlp.gate_proj.K, mlp.gate_proj.N, 128, s);
        cudaEventRecord(ev[e++], s);  // 1
        marlin_gemm(x, mlp.up_proj.qweight, state.mlp_up, mlp.up_proj.scales,
                    state.marlin_workspace, M, mlp.up_proj.K, mlp.up_proj.N, 128, s);
        cudaEventRecord(ev[e++], s);  // 2
        silu_mul(state.mlp_gate, state.mlp_up, state.mlp_gate,
                 M * MC::INTERMEDIATE_SIZE, s);
        cudaEventRecord(ev[e++], s);  // 3
        // Down projection + residual add
        marlin_gemm(state.mlp_gate, mlp.down_proj.qweight, state.mlp_down, mlp.down_proj.scales,
                    state.marlin_workspace, M, mlp.down_proj.K, mlp.down_proj.N, 128, s);
        elementwise_add(state.residual, state.mlp_down, state.residual,
                        M * MC::HIDDEN_SIZE, s);
        cudaEventRecord(ev[e++], s);  // 4
        cudaStreamSynchronize(s);

        float mlp_total = ms(0, 4);
        printf("\n=== MLP Sub-layer (layer %d, M=%d) ===\n", mlp_layer, M);
        printf("  gate_proj (Marlin 5120→17408):%6.3f ms  (%4.1f%%)\n", ms(0,1), 100*ms(0,1)/mlp_total);
        printf("  up_proj   (Marlin 5120→17408):%6.3f ms  (%4.1f%%)\n", ms(1,2), 100*ms(1,2)/mlp_total);
        printf("  silu_mul:                    %6.3f ms  (%4.1f%%)\n", ms(2,3), 100*ms(2,3)/mlp_total);
        printf("  down_proj+add (Marlin+add):   %6.3f ms  (%4.1f%%)\n", ms(3,4), 100*ms(3,4)/mlp_total);
        printf("  TOTAL (1 MLP layer):         %6.3f ms  (×64 = %.1f ms)\n", mlp_total, mlp_total*64);
    }

    // === FA Sub-layer (one layer) ===
    if (first_fa >= 0) {
        const auto& attn = model.layers[first_fa].full_attn;
        const __half* x = state.norm_out;
        __half* Qsep = state.dn_qkv;
        __half* Gate = state.mlp_gate;
        int e = 0;
        cudaEventRecord(ev[e++], s);  // 0
        fp16_gemm(x, attn.repacked_q, state.q_buf,
                  M, MC::HIDDEN_SIZE, MC::Q_PROJ_DIM, s);
        cudaEventRecord(ev[e++], s);  // 1
        fp16_gemm(x, attn.repacked_kv, state.attn_out,
                  M, MC::HIDDEN_SIZE, MC::FA_KV_DIM, s);
        cudaMemcpy2DAsync(state.kv_buf, MC::KV_PROJ_DIM * sizeof(__half),
                          state.attn_out, MC::FA_KV_DIM * sizeof(__half),
                          MC::KV_PROJ_DIM * sizeof(__half), M,
                          cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(state.dn_z, MC::KV_PROJ_DIM * sizeof(__half),
                          state.attn_out + MC::KV_PROJ_DIM, MC::FA_KV_DIM * sizeof(__half),
                          MC::KV_PROJ_DIM * sizeof(__half), M,
                          cudaMemcpyDeviceToDevice, s);
        cudaEventRecord(ev[e++], s);  // 2
        // split + norm + RoPE + KV write
        {
            dim3 grid_sq(MC::NUM_ATTN_HEADS, M);
            split_q_gate_batch_kernel<<<grid_sq, 256, 0, s>>>(
                state.q_buf, Qsep, Gate, M, MC::NUM_ATTN_HEADS, MC::HEAD_DIM);
            head_norm(Qsep, attn.q_norm, Qsep,
                      M * MC::NUM_ATTN_HEADS, MC::HEAD_DIM, MC::RMS_EPS, s);
            head_norm(state.kv_buf, attn.k_norm, state.kv_buf,
                      M * MC::NUM_KV_HEADS, MC::HEAD_DIM, MC::RMS_EPS, s);
            int mh = (MC::NUM_ATTN_HEADS > MC::NUM_KV_HEADS) ? MC::NUM_ATTN_HEADS : MC::NUM_KV_HEADS;
            dim3 grid_r(mh, M);
            rope_batch_kernel<<<grid_r, MC::ROTARY_DIM / 2, 0, s>>>(
                Qsep, state.kv_buf, MC::NUM_ATTN_HEADS, MC::NUM_KV_HEADS,
                MC::HEAD_DIM, MC::ROTARY_DIM, pos_start, M, MC::ROPE_THETA);
            size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
            __half* k_c = kv_cache + (size_t)first_fa * 2 * kv_plane;
            __half* v_c = k_c + kv_plane;
            dim3 grid_kv(MC::NUM_KV_HEADS, M);
            kv_cache_write_batch_kernel<<<grid_kv, 256, 0, s>>>(
                state.kv_buf, state.dn_z, k_c, v_c,
                pos_start, M, max_kv_len, MC::HEAD_DIM, MC::NUM_KV_HEADS);
        }
        cudaEventRecord(ev[e++], s);  // 3
        {
            float scale_val = 1.0f / sqrtf((float)MC::HEAD_DIM);
            dim3 grid_a(MC::NUM_ATTN_HEADS, M);
            size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
            __half* k_c = kv_cache + (size_t)first_fa * 2 * kv_plane;
            __half* v_c = k_c + kv_plane;
            prefill_attention_kernel<<<grid_a, MC::HEAD_DIM, 0, s>>>(
                Qsep, k_c, v_c, state.attn_out,
                pos_start, M, max_kv_len, MC::HEAD_DIM,
                MC::NUM_KV_GROUPS, scale_val);
        }
        cudaEventRecord(ev[e++], s);  // 4
        sigmoid_gate(state.attn_out, Gate, state.attn_out,
                     M * MC::ATTN_OUT_DIM, s);
        cudaEventRecord(ev[e++], s);  // 5
        fp16_gemm(state.attn_out, attn.repacked_o, state.norm_out,
                  M, MC::ATTN_OUT_DIM, MC::HIDDEN_SIZE, s);
        cudaEventRecord(ev[e++], s);  // 6
        cudaStreamSynchronize(s);

        float fa_total = ms(0, 6);
        printf("\n=== FA Sub-layer (layer %d, M=%d) ===\n", first_fa, M);
        printf("  q_proj    (FP16 5120→12288):   %6.3f ms  (%4.1f%%)\n", ms(0,1), 100*ms(0,1)/fa_total);
        printf("  kv_proj   (FP16 5120→1024×2):  %6.3f ms  (%4.1f%%)\n", ms(1,2), 100*ms(1,2)/fa_total);
        printf("  split+norm+RoPE+kvwrite:     %6.3f ms  (%4.1f%%)\n", ms(2,3), 100*ms(2,3)/fa_total);
        printf("  attention (causal):          %6.3f ms  (%4.1f%%)\n", ms(3,4), 100*ms(3,4)/fa_total);
        printf("  sigmoid_gate:                %6.3f ms  (%4.1f%%)\n", ms(4,5), 100*ms(4,5)/fa_total);
        printf("  o_proj    (FP16 6144→5120):   %6.3f ms  (%4.1f%%)\n", ms(5,6), 100*ms(5,6)/fa_total);
        printf("  TOTAL (1 FA layer):          %6.3f ms  (×16 = %.1f ms)\n", fa_total, fa_total*16);
    }

    for (int i = 0; i < NEV; i++) cudaEventDestroy(ev[i]);
}

// ============================================================================
// Profile prefill: records events at component boundaries, sync once at end.
// No pipeline drain — events are async timestamps.
// ============================================================================

void profile_forward_prefill(const ModelWeights& model,
                             InferenceState& state,
                             __half* kv_cache,
                             const int* h_token_ids, int M,
                             int pos_start, int max_kv_len,
                             cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // 4 events per layer + 1 final
    constexpr int EPL = 4;
    const int N_EV = MC::NUM_LAYERS * EPL + 1;
    std::vector<cudaEvent_t> ev(N_EV);
    for (int i = 0; i < N_EV; i++)
        cudaEventCreateWithFlags(&ev[i], cudaEventDefault);

    // Setup (same as forward_prefill)
    cudaMemcpyAsync(state.token_ids, h_token_ids, M * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, M, MC::HIDDEN_SIZE, s);
    cudaMemcpyAsync(state.residual, state.hidden,
                    (size_t)M * MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, s);

    int dn_layer_idx = 0;

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];
        int base = layer * EPL;

        // ev[0]: before pre_norm
        cudaEventRecord(ev[base + 0], s);

        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // ev[1]: after pre_norm, before attn/DN
        cudaEventRecord(ev[base + 1], s);

        if (lw.is_full_attention) {
            full_attention_prefill(state.norm_out, lw.full_attn,
                                   kv_cache, layer, pos_start, M, max_kv_len,
                                   state, s);
        } else {
            deltanet_prefill(state.norm_out, lw.delta_net,
                             dn_layer_idx, M, state, s);
            dn_layer_idx++;
        }

        // ev[2]: after attn/DN, before residual+post_norm
        cudaEventRecord(ev[base + 2], s);

        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          M, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        // ev[3]: after post_norm, before MLP
        cudaEventRecord(ev[base + 3], s);

        mlp_forward_prefill(state.norm_out, lw.mlp, state.residual, M, state, s);
    }

    // Final event after last MLP
    cudaEventRecord(ev[N_EV - 1], s);

    // Single sync point — no pipeline drain during the loop
    cudaStreamSynchronize(s);

    // Aggregate
    float t_norm = 0, t_dn = 0, t_fa = 0, t_mlp = 0;
    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];
        int base = layer * EPL;
        float ms;

        cudaEventElapsedTime(&ms, ev[base + 0], ev[base + 1]);
        t_norm += ms;

        cudaEventElapsedTime(&ms, ev[base + 1], ev[base + 2]);
        if (lw.is_full_attention) t_fa += ms;
        else t_dn += ms;

        cudaEventElapsedTime(&ms, ev[base + 2], ev[base + 3]);
        t_norm += ms;

        // MLP end = next layer's ev[0], or final event
        cudaEventElapsedTime(&ms, ev[base + 3], ev[base + EPL]);
        t_mlp += ms;
    }

    float total;
    cudaEventElapsedTime(&total, ev[0], ev[N_EV - 1]);

    printf("\n=== Prefill profile (M=%d, %d layers) ===\n", M, MC::NUM_LAYERS);
    printf("  DeltaNet SSM (48 layers):     %7.2f ms  (%4.1f%%)\n", t_dn, 100*t_dn/total);
    printf("  Full Attention (16 layers):   %7.2f ms  (%4.1f%%)\n", t_fa, 100*t_fa/total);
    printf("  MLP GPTQ-v2 (64 layers):      %7.2f ms  (%4.1f%%)\n", t_mlp, 100*t_mlp/total);
    printf("  Norms (pre+post, 64 layers):  %7.2f ms  (%4.1f%%)\n", t_norm, 100*t_norm/total);
    printf("  Total (layers only):          %7.2f ms\n", total);
    printf("  Per token (M=%d):             %7.2f ms/tok\n", M, total / M);

    for (int i = 0; i < N_EV; i++) cudaEventDestroy(ev[i]);
}


} // namespace deusridet
