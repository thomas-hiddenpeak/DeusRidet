/**
 * @file src/sensus/auditus/mossformer2_flash.cu
 * @philosophical_role
 *   FFConvM + flash-attention + FSMN block forward methods. Peer TU of
 *   mossformer2.cu under R1 800-line hard cap.
 * @serves
 *   Auditus pipeline; called by forward_masknet for each layer.
 */
// mossformer2_flash.cu — peer TU of mossformer2.cu.

#include "mossformer2.h"
#include "mossformer2_internal.h"
#include "mossformer2_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace deusridet {

using namespace mf2;

// ============================================================================

void MossFormer2::forward_ffconvm_scale(const float* d_in, float* d_out,
                                        float* d_tmp,
                                        int L, int dim_in, int dim_out,
                                        float* norm_g,
                                        float* linear_w, float* linear_b,
                                        float* dw_w) {
    int nout = L * dim_out;

    // ScaleNorm → d_tmp (needs L*dim_in, may exceed d_out capacity)
    k_scalenorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_in, d_tmp, norm_g, L, dim_in);

    // Linear(dim_in→dim_out): d_tmp[L,dim_in] @ linear_w^T → d_out[L,dim_out]
    gemm_nt(cublas_, d_tmp, linear_w, d_out, L, dim_out, dim_in);
    if (linear_b)
        k_bias_row<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
            d_out, linear_b, L, dim_out);

    // SiLU
    k_silu<<<cdiv(nout, BLK), BLK, 0, stream_>>>(d_out, nout);

    // ConvModule: transpose → DWConv → transpose → add residual
    k_transpose<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
        d_out, d_tmp, L, dim_out);             // d_tmp = [dim_out, L]
    k_dwconv<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
        d_tmp, d_work_b_, dw_w, dim_out, L, kDWKernel, kDWPad);  // d_work_b_ temp
    k_transpose<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
        d_work_b_, d_tmp, dim_out, L);         // d_tmp = [L, dim_out]
    k_add<<<cdiv(nout, BLK), BLK, 0, stream_>>>(d_out, d_tmp, nout);
}

// FFConvM with LayerNorm (FSMN: to_u, to_v)
void MossFormer2::forward_ffconvm_layer(const float* d_in, float* d_out,
                                        float* d_tmp,
                                        int L, int dim_in, int dim_out,
                                        float* norm_w, float* norm_b,
                                        float* linear_w, float* linear_b,
                                        float* dw_w) {
    int nout = L * dim_out;

    k_layernorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_in, d_out, norm_w, norm_b, L, dim_in);

    gemm_nt(cublas_, d_out, linear_w, d_tmp, L, dim_out, dim_in);
    if (linear_b)
        k_bias_row<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
            d_tmp, linear_b, L, dim_out);

    k_silu<<<cdiv(nout, BLK), BLK, 0, stream_>>>(d_tmp, nout);

    k_transpose<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
        d_tmp, d_out, L, dim_out);
    k_dwconv<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
        d_out, d_fsmn_c_, dw_w, dim_out, L, kDWKernel, kDWPad);
    k_transpose<<<cdiv(nout, BLK), BLK, 0, stream_>>>(
        d_fsmn_c_, d_out, dim_out, L);
    k_add<<<cdiv(nout, BLK), BLK, 0, stream_>>>(d_out, d_tmp, nout);
}

// ============================================================================
// FLASH Layer — FLASH_ShareA_FFConvM
// ============================================================================

void MossFormer2::forward_flash_layer(int idx, int L) {
    auto& fw = flash_w_[idx];
    int n512 = L * kEncDim;
    int n_vu = L * kVUDim;
    int Lpad = cdiv(L, kGroupSize) * kGroupSize;
    int G = Lpad / kGroupSize;

    // ---- Save residual ----
    cudaMemcpyAsync(d_work_c_, d_x_, n512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // ---- Token shift ----
    // Read from d_work_c_ (copy), write to d_x_ (shifted)
    k_token_shift<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_c_, d_x_, L, kEncDim);

    // ---- to_hidden: FFConvM(ScaleNorm, 512→2048) → d_hidden_ [L, 2048] ----
    forward_ffconvm_scale(d_x_, d_hidden_, d_work_a_, L, kEncDim, kHiddenDim,
                          fw.hidden_norm_g, fw.hidden_linear_w,
                          fw.hidden_linear_b, fw.hidden_dw_w);

    // ---- Deinterleave d_hidden_[L,2048] → v[L,1024], u[L,1024] ----
    // v → d_work_a_[0], u → d_work_a_[Lpad*kVUDim]
    float* v_ptr = d_work_a_;
    float* u_ptr = d_work_a_ + Lpad * kVUDim;
    k_deinterleave<<<cdiv(n_vu, BLK), BLK, 0, stream_>>>(
        d_hidden_, v_ptr, u_ptr, L, kVUDim);
    // Zero-pad v and u from L to Lpad
    if (Lpad > L) {
        int pad_n = (Lpad - L) * kVUDim;
        k_zero<<<cdiv(pad_n, BLK), BLK, 0, stream_>>>(v_ptr + L * kVUDim, pad_n);
        k_zero<<<cdiv(pad_n, BLK), BLK, 0, stream_>>>(u_ptr + L * kVUDim, pad_n);
    }

    // ---- to_qk: FFConvM(ScaleNorm, 512→128) → d_qk_ [L, 128] ----
    forward_ffconvm_scale(d_x_, d_qk_, d_hidden_, L, kEncDim, kQKDim,
                          fw.qk_norm_g, fw.qk_linear_w,
                          fw.qk_linear_b, fw.qk_dw_w);

    // ---- OffsetScale → 4 outputs in d_hidden_ ----
    int nqk = L * kQKDim;
    float* quad_q = d_hidden_;
    float* lin_q  = d_hidden_ + Lpad * kQKDim;
    float* quad_k = d_hidden_ + 2 * Lpad * kQKDim;
    float* lin_k  = d_hidden_ + 3 * Lpad * kQKDim;

    k_offset_scale<<<cdiv(nqk, BLK), BLK, 0, stream_>>>(
        d_qk_, quad_q, lin_q, quad_k, lin_k,
        fw.offset_gamma, fw.offset_beta, L, kQKDim);

    // Zero-pad QK outputs from L to Lpad
    if (Lpad > L) {
        int pad_qk = (Lpad - L) * kQKDim;
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(quad_q + L * kQKDim, pad_qk);
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(lin_q  + L * kQKDim, pad_qk);
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(quad_k + L * kQKDim, pad_qk);
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(lin_k  + L * kQKDim, pad_qk);
    }

    // ---- RoPE on all 4 QK vectors ----
    int rope_n = Lpad * kRopeFreqs;
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(quad_q, fw.rope_freqs, Lpad, kQKDim);
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(lin_q,  fw.rope_freqs, Lpad, kQKDim);
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(quad_k, fw.rope_freqs, Lpad, kQKDim);
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(lin_k,  fw.rope_freqs, Lpad, kQKDim);

    // ---- Quadratic attention (batched GEMM over G groups) ----
    // sim[G,256,256] stored in d_qk_ (reused, needs Lpad*256 ≤ Lpad*128? No!)
    // Actually sim needs G*256*256 = Lpad*256 floats. d_qk_ is Lpad*128. Not enough.
    // Use d_x_ for sim (n512 = L*512, sim needs Lpad*256 ≈ 4096*256 = 1M,
    // d_x_ holds max_L*512 ≈ 4000*512 = 2M). Fits.
    float* sim = d_x_;   // safe: d_x_ is not needed until we write residual+to_out back

    // sim = quad_q @ quad_k^T / group_size
    float alpha_sim = 1.f / kGroupSize;
    float beta_zero = 0.f;
    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        kGroupSize, kGroupSize, kQKDim,
        &alpha_sim,
        quad_k, kQKDim, kGroupSize * kQKDim,   // A (transposed)
        quad_q, kQKDim, kGroupSize * kQKDim,   // B
        &beta_zero,
        sim,    kGroupSize, kGroupSize * kGroupSize,  // C
        G);

    // attn = relu(sim)^2
    int n_sim = Lpad * kGroupSize;
    k_relu_sq<<<cdiv(n_sim, BLK), BLK, 0, stream_>>>(sim, n_sim);

    // att_v[Lpad,1024] = attn[G,256,256] @ v[G,256,1024] → store in d_work_b_
    float* att_v = d_work_b_;
    float* att_u = d_work_b_ + Lpad * kVUDim;
    float alpha_one = 1.f;

    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        kVUDim, kGroupSize, kGroupSize,
        &alpha_one,
        v_ptr,  kVUDim, kGroupSize * kVUDim,    // first arg = V
        sim,    kGroupSize, kGroupSize * kGroupSize,  // second arg = attn
        &beta_zero,
        att_v,  kVUDim, kGroupSize * kVUDim,
        G);

    // att_u[Lpad,1024] = attn[G,256,256] @ u[G,256,1024] → store in d_work_b_+offset
    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        kVUDim, kGroupSize, kGroupSize,
        &alpha_one,
        u_ptr,  kVUDim, kGroupSize * kVUDim,
        sim,    kGroupSize, kGroupSize * kGroupSize,
        &beta_zero,
        att_u,  kVUDim, kGroupSize * kVUDim,
        G);

    // ---- Linear attention (global, non-causal) ----
    // Python: lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
    // where n = original sequence length (NOT group_size)
    // lin_kv[128,1024] = lin_k^T[128,Lpad] @ v[Lpad,1024] / L
    float* lin_kv = d_qk_;
    float alpha_lin = 1.f / L;

    // lin_kv = lin_k^T @ v / L
    gemm_tn(cublas_, lin_k, v_ptr, lin_kv, kQKDim, kVUDim, Lpad, alpha_lin, 0.f);

    // att_v += lin_q @ lin_kv (accumulate with beta=1)
    gemm_nn(cublas_, lin_q, lin_kv, att_v, Lpad, kVUDim, kQKDim, 1.f, 1.f);

    // lin_ku = lin_k^T @ u / L → reuse lin_kv location
    float* lin_ku = d_qk_;
    gemm_tn(cublas_, lin_k, u_ptr, lin_ku, kQKDim, kVUDim, Lpad, alpha_lin, 0.f);

    // att_u += lin_q @ lin_ku
    gemm_nn(cublas_, lin_q, lin_ku, att_u, Lpad, kVUDim, kQKDim, 1.f, 1.f);

    // ---- Gate: out[L,1024] = (att_u * v) * sigmoid(att_v * u) ----
    // Store gate output in d_hidden_[0..L*1024-1]
    float* gate_out = d_hidden_;
    k_gate<<<cdiv(n_vu, BLK), BLK, 0, stream_>>>(
        att_u, v_ptr, att_v, u_ptr, gate_out, n_vu);

    // ---- to_out: FFConvM(ScaleNorm, 1024→512) → d_x_ [L, 512] ----
    // d_x_ was used as sim scratch but we're done with it
    forward_ffconvm_scale(gate_out, d_x_, d_work_a_, L, kVUDim, kEncDim,
                          fw.out_norm_g, fw.out_linear_w,
                          fw.out_linear_b, fw.out_dw_w);

    // ---- Add residual: d_x_ += saved residual in d_work_c_ ----
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_x_, d_work_c_, n512);
}

// ============================================================================
// FLASH Layer debug substep — runs forward_flash_layer up to a given substep
// Steps: 1=token_shift(d_x_), 2=to_hidden(d_hidden_), 3=to_qk(d_qk_),
//        4=offset_scale+rope(d_hidden_ has 4 QK vecs), 5=attention(d_work_b_),
//        6=gate(d_hidden_), 7=to_out+residual(d_x_)
// ============================================================================

void MossFormer2::dbg_forward_flash_substep(int idx, int L, int substep) {
    auto& fw = flash_w_[idx];
    int n512 = L * kEncDim;
    int n_vu = L * kVUDim;
    int Lpad = cdiv(L, kGroupSize) * kGroupSize;
    int G = Lpad / kGroupSize;

    // ---- Save residual ----
    cudaMemcpyAsync(d_work_c_, d_x_, n512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // ---- Token shift ----
    k_token_shift<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_c_, d_x_, L, kEncDim);
    if (substep <= 1) return;

    // ---- to_hidden → d_hidden_ [L, 2048] ----
    forward_ffconvm_scale(d_x_, d_hidden_, d_work_a_, L, kEncDim, kHiddenDim,
                          fw.hidden_norm_g, fw.hidden_linear_w,
                          fw.hidden_linear_b, fw.hidden_dw_w);
    if (substep <= 2) return;

    // ---- Deinterleave → v, u ----
    float* v_ptr = d_work_a_;
    float* u_ptr = d_work_a_ + Lpad * kVUDim;
    k_deinterleave<<<cdiv(n_vu, BLK), BLK, 0, stream_>>>(
        d_hidden_, v_ptr, u_ptr, L, kVUDim);
    if (Lpad > L) {
        int pad_n = (Lpad - L) * kVUDim;
        k_zero<<<cdiv(pad_n, BLK), BLK, 0, stream_>>>(v_ptr + L * kVUDim, pad_n);
        k_zero<<<cdiv(pad_n, BLK), BLK, 0, stream_>>>(u_ptr + L * kVUDim, pad_n);
    }

    // ---- to_qk → d_qk_ [L, 128] ----
    forward_ffconvm_scale(d_x_, d_qk_, d_hidden_, L, kEncDim, kQKDim,
                          fw.qk_norm_g, fw.qk_linear_w,
                          fw.qk_linear_b, fw.qk_dw_w);
    if (substep <= 3) return;

    // ---- OffsetScale + RoPE ----
    int nqk = L * kQKDim;
    float* quad_q = d_hidden_;
    float* lin_q  = d_hidden_ + Lpad * kQKDim;
    float* quad_k = d_hidden_ + 2 * Lpad * kQKDim;
    float* lin_k  = d_hidden_ + 3 * Lpad * kQKDim;
    k_offset_scale<<<cdiv(nqk, BLK), BLK, 0, stream_>>>(
        d_qk_, quad_q, lin_q, quad_k, lin_k,
        fw.offset_gamma, fw.offset_beta, L, kQKDim);
    if (Lpad > L) {
        int pad_qk = (Lpad - L) * kQKDim;
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(quad_q + L * kQKDim, pad_qk);
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(lin_q  + L * kQKDim, pad_qk);
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(quad_k + L * kQKDim, pad_qk);
        k_zero<<<cdiv(pad_qk, BLK), BLK, 0, stream_>>>(lin_k  + L * kQKDim, pad_qk);
    }
    int rope_n = Lpad * kRopeFreqs;
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(quad_q, fw.rope_freqs, Lpad, kQKDim);
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(lin_q,  fw.rope_freqs, Lpad, kQKDim);
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(quad_k, fw.rope_freqs, Lpad, kQKDim);
    k_rope<<<cdiv(rope_n, BLK), BLK, 0, stream_>>>(lin_k,  fw.rope_freqs, Lpad, kQKDim);
    if (substep <= 4) return;

    // ---- Attention (quadratic + linear) ----
    float* sim = d_x_;
    float alpha_sim = 1.f / kGroupSize;
    float beta_zero = 0.f;
    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        kGroupSize, kGroupSize, kQKDim, &alpha_sim,
        quad_k, kQKDim, kGroupSize * kQKDim,
        quad_q, kQKDim, kGroupSize * kQKDim,
        &beta_zero, sim, kGroupSize, kGroupSize * kGroupSize, G);
    int n_sim = Lpad * kGroupSize;
    k_relu_sq<<<cdiv(n_sim, BLK), BLK, 0, stream_>>>(sim, n_sim);
    float* att_v = d_work_b_;
    float* att_u = d_work_b_ + Lpad * kVUDim;
    float alpha_one = 1.f;
    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        kVUDim, kGroupSize, kGroupSize, &alpha_one,
        v_ptr, kVUDim, kGroupSize * kVUDim,
        sim, kGroupSize, kGroupSize * kGroupSize,
        &beta_zero, att_v, kVUDim, kGroupSize * kVUDim, G);
    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        kVUDim, kGroupSize, kGroupSize, &alpha_one,
        u_ptr, kVUDim, kGroupSize * kVUDim,
        sim, kGroupSize, kGroupSize * kGroupSize,
        &beta_zero, att_u, kVUDim, kGroupSize * kVUDim, G);
    float* lin_kv = d_qk_;
    float alpha_lin = 1.f / L;
    gemm_tn(cublas_, lin_k, v_ptr, lin_kv, kQKDim, kVUDim, Lpad, alpha_lin, 0.f);
    gemm_nn(cublas_, lin_q, lin_kv, att_v, Lpad, kVUDim, kQKDim, 1.f, 1.f);
    float* lin_ku = d_qk_;
    gemm_tn(cublas_, lin_k, u_ptr, lin_ku, kQKDim, kVUDim, Lpad, alpha_lin, 0.f);
    gemm_nn(cublas_, lin_q, lin_ku, att_u, Lpad, kVUDim, kQKDim, 1.f, 1.f);
    if (substep <= 5) return;

    // ---- Gate ----
    float* gate_out = d_hidden_;
    k_gate<<<cdiv(n_vu, BLK), BLK, 0, stream_>>>(
        att_u, v_ptr, att_v, u_ptr, gate_out, n_vu);
    if (substep <= 6) return;

    // ---- to_out + residual ----
    forward_ffconvm_scale(gate_out, d_x_, d_work_a_, L, kVUDim, kEncDim,
                          fw.out_norm_g, fw.out_linear_w,
                          fw.out_linear_b, fw.out_dw_w);
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_x_, d_work_c_, n512);
}

// ============================================================================
// FSMN substep debug — runs forward_fsmn_block up to a given substep
// Steps: 1=conv1+prelu, 2=norm1, 3=to_u, 4=to_v, 5=uni_fsmn, 6=gate,
//        7=norm2, 8=conv2+residual
// ============================================================================

void MossFormer2::dbg_forward_fsmn_substep(int idx, int L, int substep) {
    auto& fw = fsmn_w_[idx];
    int n512 = L * kEncDim;
    int n256 = kFsmnInner * L;

    cudaMemcpyAsync(d_work_c_, d_x_, n512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // conv1 + PReLU → d_fsmn_a_ [256,L]
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_x_, d_work_a_, L, kEncDim);
    gemm_CL(cublas_, fw.conv1_w, d_work_a_, d_fsmn_a_, kFsmnInner, kEncDim, L);
    k_bias_ch<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_fsmn_a_, fw.conv1_b, kFsmnInner, L);
    k_prelu1<<<cdiv(n256, BLK), BLK, 0, stream_>>>(d_fsmn_a_, fw.conv1_prelu, n256);
    if (substep <= 1) return;

    // norm1 → d_fsmn_b_ [256,L]
    k_clnorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_fsmn_a_, d_fsmn_b_, fw.norm1_w, fw.norm1_b, kFsmnInner, L);
    if (substep <= 2) return;

    // Transpose → d_fsmn_a_ [L,256]
    k_transpose<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_fsmn_b_, d_fsmn_a_, kFsmnInner, L);

    // to_u → d_fsmn_b_ [L,256]
    forward_ffconvm_layer(d_fsmn_a_, d_fsmn_b_, d_hidden_, L,
                          kFsmnInner, kFsmnInner,
                          fw.to_u_norm_w, fw.to_u_norm_b,
                          fw.to_u_linear_w, fw.to_u_linear_b,
                          fw.to_u_dw_w);
    if (substep <= 3) return;

    // to_v → d_hidden_ [L,256]
    float* x_v = d_hidden_;
    forward_ffconvm_layer(d_fsmn_a_, x_v, d_hidden_ + L * kFsmnInner, L,
                          kFsmnInner, kFsmnInner,
                          fw.to_v_norm_w, fw.to_v_norm_b,
                          fw.to_v_linear_w, fw.to_v_linear_b,
                          fw.to_v_dw_w);
    if (substep <= 4) return;

    // UniDeepFsmn → d_fsmn_b_ [L,256]
    forward_uni_fsmn(d_fsmn_b_, d_fsmn_b_, L, fw);
    if (substep <= 5) return;

    // Gate: d_fsmn_b_ = x_v * d_fsmn_b_ + d_fsmn_a_
    k_mul<<<cdiv(L * kFsmnInner, BLK), BLK, 0, stream_>>>(
        x_v, d_fsmn_b_, d_fsmn_b_, L * kFsmnInner);
    k_add<<<cdiv(L * kFsmnInner, BLK), BLK, 0, stream_>>>(
        d_fsmn_b_, d_fsmn_a_, L * kFsmnInner);
    if (substep <= 6) return;

    // norm2 → d_fsmn_b_ [256,L]
    k_transpose<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_fsmn_b_, d_fsmn_a_, L, kFsmnInner);
    k_clnorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_fsmn_a_, d_fsmn_b_, fw.norm2_w, fw.norm2_b, kFsmnInner, L);
    if (substep <= 7) return;

    // conv2 + residual → d_x_ [L,512]
    gemm_CL(cublas_, fw.conv2_w, d_fsmn_b_, d_work_a_, kEncDim, kFsmnInner, L);
    k_bias_ch<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_a_, fw.conv2_b, kEncDim, L);
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_a_, d_x_, kEncDim, L);
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_x_, d_work_c_, n512);
}

// ============================================================================
// FSMN Block — Gated_FSMN_Block_Dilated
// ============================================================================

void MossFormer2::forward_fsmn_block(int idx, int L) {
    auto& fw = fsmn_w_[idx];
    int n512 = L * kEncDim;
    int n256 = kFsmnInner * L;

    // Save input for final residual: d_x_ [L, 512]
    // We'll need it at the very end. Use d_work_c_ (safe during FSMN).
    cudaMemcpyAsync(d_work_c_, d_x_, n512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // ---- conv1: Conv1d(512→256, k=1) + PReLU(1) ----
    // Transpose d_x_[L,512] → d_work_a_[512,L]
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_x_, d_work_a_, L, kEncDim);
    // Conv1d(k=1): W[256,512]@X[512,L] → Y[256,L]
    gemm_CL(cublas_, fw.conv1_w, d_work_a_, d_fsmn_a_, kFsmnInner, kEncDim, L);
    k_bias_ch<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_fsmn_a_, fw.conv1_b, kFsmnInner, L);
    // PReLU(1) — scalar alpha
    k_prelu1<<<cdiv(n256, BLK), BLK, 0, stream_>>>(d_fsmn_a_, fw.conv1_prelu, n256);

    // ---- norm1: CLayerNorm(256) [256,L] → d_fsmn_b_ [256,L] ----
    k_clnorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_fsmn_a_, d_fsmn_b_, fw.norm1_w, fw.norm1_b, kFsmnInner, L);

    // ---- Gated FSMN: input is norm1.T = [L, 256] ----
    // Transpose d_fsmn_b_[256,L] → d_fsmn_a_[L,256]
    k_transpose<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_fsmn_b_, d_fsmn_a_, kFsmnInner, L);

    // to_u: FFConvM(LayerNorm, 256→256) → d_fsmn_b_ [L, 256]
    forward_ffconvm_layer(d_fsmn_a_, d_fsmn_b_, d_hidden_, L,
                          kFsmnInner, kFsmnInner,
                          fw.to_u_norm_w, fw.to_u_norm_b,
                          fw.to_u_linear_w, fw.to_u_linear_b,
                          fw.to_u_dw_w);

    // to_v: FFConvM(LayerNorm, 256→256) → d_qk_ [L, 256]
    // d_qk_ is [max_L*128] but we need [L*256]. Check: max_L*128 vs L*256.
    // For max_samples=32000: max_L=3999, so max_L*128 = 511872.
    // L*256 can be up to 3999*256 = 1023744. This doesn't fit in d_qk_!
    // Use d_hidden_[0..L*256-1] instead (d_hidden_ is max_L*2048, plenty).
    float* x_v = d_hidden_;
    forward_ffconvm_layer(d_fsmn_a_, x_v, d_hidden_ + L * kFsmnInner, L,
                          kFsmnInner, kFsmnInner,
                          fw.to_v_norm_w, fw.to_v_norm_b,
                          fw.to_v_linear_w, fw.to_v_linear_b,
                          fw.to_v_dw_w);

    // UniDeepFsmn_dilated: d_fsmn_b_ → d_fsmn_b_ (in-place via internal scratch)
    forward_uni_fsmn(d_fsmn_b_, d_fsmn_b_, L, fw);

    // output = x_v * x_u + input (where input = d_fsmn_a_)
    k_mul<<<cdiv(n256 / L * L, BLK), BLK, 0, stream_>>>(
        x_v, d_fsmn_b_, d_fsmn_b_, L * kFsmnInner);
    k_add<<<cdiv(L * kFsmnInner, BLK), BLK, 0, stream_>>>(
        d_fsmn_b_, d_fsmn_a_, L * kFsmnInner);

    // ---- norm2: transpose → CLayerNorm(256) ----
    // d_fsmn_b_ [L,256] → transpose → d_fsmn_a_ [256,L]
    k_transpose<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_fsmn_b_, d_fsmn_a_, L, kFsmnInner);
    k_clnorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_fsmn_a_, d_fsmn_b_, fw.norm2_w, fw.norm2_b, kFsmnInner, L);

    // ---- conv2: Conv1d(256→512, k=1) + bias ----
    gemm_CL(cublas_, fw.conv2_w, d_fsmn_b_, d_work_a_, kEncDim, kFsmnInner, L);
    k_bias_ch<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_a_, fw.conv2_b, kEncDim, L);

    // ---- Transpose [512,L] → d_x_ [L,512] ----
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_a_, d_x_, kEncDim, L);

    // ---- Add residual ----
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_x_, d_work_c_, n512);
}

// ============================================================================

} // namespace deusridet
