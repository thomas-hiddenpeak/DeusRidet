/**
 * @file src/sensus/auditus/mossformer2_lifecycle.cu
 * @philosophical_role
 *   Lifecycle, weight loading/resolution, scratch alloc, top-level forward
 *   wrapper, encoder/decoder, FSMN tail. Peer TU of mossformer2.cu under
 *   R1 800-line hard cap.
 * @serves
 *   Auditus pipeline. Methods called from mossformer2.cu kernels indirectly
 *   via the public MossFormer2 surface.
 */
// mossformer2_lifecycle.cu — peer TU of mossformer2.cu.

#include "mossformer2.h"
#include "mossformer2_internal.h"
#include "mossformer2_kernels.cuh"
#include "../../machina/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace deusridet {

using namespace mf2;


// ============================================================================
// Weight loading
// ============================================================================

bool MossFormer2::load_weights(const std::string& path) {
    SafetensorsFile sf(path);
    auto names = sf.tensor_names();
    if (names.empty()) {
        LOG_ERROR("MF2", "No tensors in %s", path.c_str());
        return false;
    }

    size_t total = 0;
    for (auto& n : names) {
        auto t = sf.get_tensor(n);
        if (t) total += t->nbytes();
    }

    CK(cudaMalloc(&d_weights_, total));
    weights_bytes_ = total;

    size_t off = 0;
    for (auto& n : names) {
        auto t = sf.get_tensor(n);
        if (!t) continue;
        float* dst = (float*)((char*)d_weights_ + off);
        size_t bytes = t->nbytes();
        CK(cudaMemcpy(dst, t->data(), bytes, cudaMemcpyHostToDevice));
        wmap_[n] = {dst, (int)(bytes / sizeof(float))};
        off += bytes;
    }

    LOG_INFO("MF2", "Loaded %zu tensors (%.1f MB) from %s",
             wmap_.size(), total / (1024.0 * 1024.0), path.c_str());
    return true;
}

float* MossFormer2::wp(const std::string& name) const {
    auto it = wmap_.find(name);
    if (it == wmap_.end()) {
        LOG_ERROR("MF2", "Weight not found: %s", name.c_str());
        return nullptr;
    }
    return it->second.ptr;
}

// ============================================================================
// Resolve per-layer weight pointers
// ============================================================================

bool MossFormer2::resolve_layer_weights() {
    enc_w_         = wp("enc.conv1d.weight");
    dec_w_         = wp("dec.weight");
    mn_norm_w_     = wp("mask_net.norm.weight");
    mn_norm_b_     = wp("mask_net.norm.bias");
    mn_conv_enc_w_ = wp("mask_net.conv1d_encoder.weight");
    pos_inv_freq_  = wp("mask_net.pos_enc.inv_freq");
    pos_scale_     = wp("mask_net.pos_enc.scale");
    blk_norm_w_    = wp("mask_net.mdl.intra_mdl.norm.weight");
    blk_norm_b_    = wp("mask_net.mdl.intra_mdl.norm.bias");
    intra_norm_w_  = wp("mask_net.mdl.intra_norm.weight");
    intra_norm_b_  = wp("mask_net.mdl.intra_norm.bias");
    prelu_w_       = wp("mask_net.prelu.weight");
    conv_out_w_    = wp("mask_net.conv1d_out.weight");
    conv_out_b_    = wp("mask_net.conv1d_out.bias");
    output_w_      = wp("mask_net.output.0.weight");
    output_b_      = wp("mask_net.output.0.bias");
    outgate_w_     = wp("mask_net.output_gate.0.weight");
    outgate_b_     = wp("mask_net.output_gate.0.bias");
    conv1_dec_w_   = wp("mask_net.conv1_decoder.weight");

    if (!enc_w_ || !dec_w_ || !mn_norm_w_ || !mn_conv_enc_w_) {
        LOG_ERROR("MF2", "Missing top-level weights");
        return false;
    }

    char buf[256];
    for (int i = 0; i < kNumLayers; i++) {
        auto& f = flash_w_[i];
        auto fl = [&](const char* s) -> float* {
            snprintf(buf, sizeof(buf),
                     "mask_net.mdl.intra_mdl.mossformerM.layers.%d.%s", i, s);
            return wp(buf);
        };
        f.hidden_norm_g   = fl("to_hidden.mdl.0.g");
        f.hidden_linear_w = fl("to_hidden.mdl.1.weight");
        f.hidden_linear_b = fl("to_hidden.mdl.1.bias");
        f.hidden_dw_w     = fl("to_hidden.mdl.3.sequential.1.conv.weight");
        f.qk_norm_g       = fl("to_qk.mdl.0.g");
        f.qk_linear_w     = fl("to_qk.mdl.1.weight");
        f.qk_linear_b     = fl("to_qk.mdl.1.bias");
        f.qk_dw_w         = fl("to_qk.mdl.3.sequential.1.conv.weight");
        f.offset_gamma     = fl("qk_offset_scale.gamma");
        f.offset_beta      = fl("qk_offset_scale.beta");
        f.rope_freqs       = fl("rotary_pos_emb.freqs");
        f.out_norm_g       = fl("to_out.mdl.0.g");
        f.out_linear_w     = fl("to_out.mdl.1.weight");
        f.out_linear_b     = fl("to_out.mdl.1.bias");
        f.out_dw_w         = fl("to_out.mdl.3.sequential.1.conv.weight");

        if (!f.hidden_norm_g || !f.hidden_linear_w || !f.out_norm_g) {
            LOG_ERROR("MF2", "Missing FLASH layer %d weights", i);
            return false;
        }
    }

    for (int i = 0; i < kNumLayers; i++) {
        auto& f = fsmn_w_[i];
        auto fn = [&](const char* s) -> float* {
            snprintf(buf, sizeof(buf),
                     "mask_net.mdl.intra_mdl.mossformerM.fsmn.%d.%s", i, s);
            return wp(buf);
        };
        f.conv1_w       = fn("conv1.0.weight");
        f.conv1_b       = fn("conv1.0.bias");
        f.conv1_prelu   = fn("conv1.1.weight");
        f.norm1_w       = fn("norm1.weight");
        f.norm1_b       = fn("norm1.bias");
        f.to_u_norm_w   = fn("gated_fsmn.to_u.mdl.0.weight");
        f.to_u_norm_b   = fn("gated_fsmn.to_u.mdl.0.bias");
        f.to_u_linear_w = fn("gated_fsmn.to_u.mdl.1.weight");
        f.to_u_linear_b = fn("gated_fsmn.to_u.mdl.1.bias");
        f.to_u_dw_w     = fn("gated_fsmn.to_u.mdl.3.sequential.1.conv.weight");
        f.to_v_norm_w   = fn("gated_fsmn.to_v.mdl.0.weight");
        f.to_v_norm_b   = fn("gated_fsmn.to_v.mdl.0.bias");
        f.to_v_linear_w = fn("gated_fsmn.to_v.mdl.1.weight");
        f.to_v_linear_b = fn("gated_fsmn.to_v.mdl.1.bias");
        f.to_v_dw_w     = fn("gated_fsmn.to_v.mdl.3.sequential.1.conv.weight");
        f.fsmn_linear_w = fn("gated_fsmn.fsmn.linear.weight");
        f.fsmn_linear_b = fn("gated_fsmn.fsmn.linear.bias");
        f.fsmn_project_w = fn("gated_fsmn.fsmn.project.weight");
        f.ddc_conv1_w   = fn("gated_fsmn.fsmn.conv.conv1.weight");
        f.ddc_norm1_w   = fn("gated_fsmn.fsmn.conv.norm1.weight");
        f.ddc_norm1_b   = fn("gated_fsmn.fsmn.conv.norm1.bias");
        f.ddc_prelu1_w  = fn("gated_fsmn.fsmn.conv.prelu1.weight");
        f.ddc_conv2_w   = fn("gated_fsmn.fsmn.conv.conv2.weight");
        f.ddc_norm2_w   = fn("gated_fsmn.fsmn.conv.norm2.weight");
        f.ddc_norm2_b   = fn("gated_fsmn.fsmn.conv.norm2.bias");
        f.ddc_prelu2_w  = fn("gated_fsmn.fsmn.conv.prelu2.weight");
        f.norm2_w       = fn("norm2.weight");
        f.norm2_b       = fn("norm2.bias");
        f.conv2_w       = fn("conv2.weight");
        f.conv2_b       = fn("conv2.bias");

        if (!f.conv1_w || !f.fsmn_linear_w || !f.ddc_conv1_w) {
            LOG_ERROR("MF2", "Missing FSMN layer %d weights", i);
            return false;
        }
    }

    LOG_INFO("MF2", "All layer weights resolved");
    return true;
}

// ============================================================================
// Scratch allocation
// ============================================================================

bool MossFormer2::alloc_scratch(int max_L) {
    max_L_ = max_L;
    int L = max_L;
    size_t tot = 0;

    auto al = [&](float*& p, size_t n, const char* nm) -> bool {
        size_t b = n * sizeof(float);
        if (cudaMalloc(&p, b) != cudaSuccess) {
            LOG_ERROR("MF2", "Scratch fail: %s (%zu B)", nm, b);
            return false;
        }
        tot += b;
        return true;
    };

    // Lpad = L rounded up to kGroupSize boundary (needed for batched attention)
    int Lpad = ((L + kGroupSize - 1) / kGroupSize) * kGroupSize;

    if (!al(d_enc_out_, kEncDim * L,           "enc_out"))  return false;
    if (!al(d_x_,       L * kEncDim,            "x"))       return false;
    if (!al(d_skip_,    L * kEncDim,            "skip"))    return false;
    if (!al(d_hidden_,  L * kHiddenDim,         "hidden"))  return false;
    if (!al(d_qk_,      L * kQKDim,             "qk"))      return false;
    if (!al(d_work_a_,  Lpad * kHiddenDim,      "work_a"))  return false;
    if (!al(d_work_b_,  Lpad * kHiddenDim,      "work_b"))  return false;
    if (!al(d_work_c_,  L * kEncDim,            "work_c"))  return false;
    if (!al(d_fsmn_a_,  L * kFsmnInner,         "fsmn_a"))  return false;
    if (!al(d_fsmn_b_,  L * kFsmnInner,         "fsmn_b"))  return false;
    if (!al(d_fsmn_c_,  L * kFsmnInner,         "fsmn_c"))  return false;
    if (!al(d_ddc_cat_, L * kFsmnInner * 3,     "ddc_cat")) return false;
    if (!al(d_masks_,   kNumSpk * kEncDim * L,  "masks"))   return false;
    if (!al(d_dec_tmp_, max_samples_,           "dec_tmp")) return false;
    if (!al(d_gn_stats_, 2,                      "gn_stats")) return false;

    LOG_INFO("MF2", "Scratch: %.1f MB (max_L=%d)", tot / (1024.0 * 1024.0), max_L);
    return true;
}

void MossFormer2::free_scratch() {
    auto f = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    f(d_enc_out_); f(d_x_); f(d_skip_); f(d_hidden_); f(d_qk_);
    f(d_work_a_); f(d_work_b_); f(d_work_c_);
    f(d_fsmn_a_); f(d_fsmn_b_); f(d_fsmn_c_); f(d_ddc_cat_);
    f(d_masks_); f(d_dec_tmp_); f(d_gn_stats_);
}

// ============================================================================
// Init
// ============================================================================

bool MossFormer2::init(const std::string& model_path, int max_samples,
                       cudaStream_t stream) {
    stream_ = stream;
    max_samples_ = max_samples;
    max_L_ = (max_samples - kEncKernel) / kEncStride + 1;

    LOG_INFO("MF2", "Init: max_samples=%d max_L=%d", max_samples, max_L_);

    if (!load_weights(model_path)) return false;
    if (!resolve_layer_weights()) return false;

    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("MF2", "cuBLAS create failed");
        return false;
    }
    if (stream_) cublasSetStream(cublas_, stream_);
    if (!alloc_scratch(max_L_)) return false;

    initialized_ = true;
    LOG_INFO("MF2", "Ready: %zu tensors %.1f MB",
             wmap_.size(), weights_bytes_ / (1024.0 * 1024.0));
    return true;
}

// ============================================================================
// Forward — Encoder
// ============================================================================

void MossFormer2::forward_encoder(const float* d_pcm, int n_samples, int L) {
    int n = kEncDim * L;
    k_enc_conv<<<cdiv(n, BLK), BLK, 0, stream_>>>(
        d_pcm, d_enc_out_, enc_w_, n_samples, L);
}

// ============================================================================
// Forward — MaskNet (staged debug version)
// ============================================================================

void MossFormer2::dbg_forward_masknet_stage(int L, int stage) {
    int n512 = kEncDim * L;

    // 1. GroupNorm(1, 512) on encoder output [512,L]
    cudaMemsetAsync(d_gn_stats_, 0, 2 * sizeof(float), stream_);
    k_gn1_stats<<<cdiv(n512, BLK), BLK, 2 * BLK * sizeof(float), stream_>>>(
        d_enc_out_, d_gn_stats_, n512);
    k_gn1_norm<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_enc_out_, d_work_c_, mn_norm_w_, mn_norm_b_, d_gn_stats_, kEncDim, L);
    if (stage <= 1) return;

    // 2. Conv1d(512→512, k=1)
    gemm_CL(cublas_, mn_conv_enc_w_, d_work_c_, d_work_a_, kEncDim, kEncDim, L);
    if (stage <= 2) return;

    // 3. Transpose + SinuEmb
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_a_, d_x_, kEncDim, L);
    k_sinuemb<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_c_, pos_inv_freq_, pos_scale_, L, kEncDim);
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_x_, d_work_c_, n512);
    if (stage <= 3) return;

    // 4. Save skip
    cudaMemcpyAsync(d_skip_, d_x_, n512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    if (stage <= 4) return;

    // 5. N layers (stage 5 = 0 layers done, 5+N = N layers done)
    int n_layers = std::min(stage - 5, (int)kNumLayers);
    for (int i = 0; i < n_layers; i++) {
        forward_flash_layer(i, L);
        forward_fsmn_block(i, L);
    }
    // if stage < 5 + kNumLayers + 1, return here
    if (stage < 5 + kNumLayers + 1) return;

    // 6+. Continue with post-layers
    // 29 = LayerNorm (MossFormerM final)
    k_layernorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_x_, d_work_c_, blk_norm_w_, blk_norm_b_, L, kEncDim);
    if (stage <= 29) return;

    // 30 = Transpose [L,512]→[512,L]
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_c_, d_x_, L, kEncDim);
    if (stage <= 30) return;

    // 31 = GroupNorm(1,512) — intra_norm
    cudaMemsetAsync(d_gn_stats_, 0, 2 * sizeof(float), stream_);
    k_gn1_stats<<<cdiv(n512, BLK), BLK, 2 * BLK * sizeof(float), stream_>>>(
        d_x_, d_gn_stats_, n512);
    k_gn1_norm<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_x_, d_work_a_, intra_norm_w_, intra_norm_b_, d_gn_stats_, kEncDim, L);
    if (stage <= 31) return;

    // 32 = Add skip (transpose skip [L,512]→[512,L] then add)
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_skip_, d_work_c_, L, kEncDim);
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_work_a_, d_work_c_, n512);
    if (stage <= 32) return;

    // 33 = PReLU
    k_prelu1<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_work_a_, prelu_w_, n512);
    // Remainder handled by full forward_masknet
}

// ============================================================================
// Forward — MaskNet
// ============================================================================

void MossFormer2::forward_masknet(int L) {
    int n512 = kEncDim * L;

    // 1. GroupNorm(1, 512) — normalize over ALL C*L elements [512,L]
    cudaMemsetAsync(d_gn_stats_, 0, 2 * sizeof(float), stream_);
    k_gn1_stats<<<cdiv(n512, BLK), BLK, 2 * BLK * sizeof(float), stream_>>>(
        d_enc_out_, d_gn_stats_, n512);
    k_gn1_norm<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_enc_out_, d_work_c_, mn_norm_w_, mn_norm_b_, d_gn_stats_, kEncDim, L);

    // 2. Conv1d(512→512, k=1): W[512,512]@X[512,L]→Y[512,L]
    gemm_CL(cublas_, mn_conv_enc_w_, d_work_c_, d_work_a_, kEncDim, kEncDim, L);

    // 3. Transpose [512,L]→[L,512], add sinusoidal embedding
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_a_, d_x_, kEncDim, L);
    k_sinuemb<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_c_, pos_inv_freq_, pos_scale_, L, kEncDim);
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_x_, d_work_c_, n512);

    // 4. Save skip
    cudaMemcpyAsync(d_skip_, d_x_, n512 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // 5. 24× [FLASH + FSMN]
    for (int i = 0; i < kNumLayers; i++) {
        forward_flash_layer(i, L);
        forward_fsmn_block(i, L);
    }

    // 6. LayerNorm (MossFormerM final norm)
    k_layernorm<<<L, BLK, BLK * sizeof(float), stream_>>>(
        d_x_, d_work_c_, blk_norm_w_, blk_norm_b_, L, kEncDim);

    // 7. Transpose [L,512]→[512,L]
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_work_c_, d_x_, L, kEncDim);

    // 8. GroupNorm(1, 512) — intra_norm, over all C*L elements
    cudaMemsetAsync(d_gn_stats_, 0, 2 * sizeof(float), stream_);
    k_gn1_stats<<<cdiv(n512, BLK), BLK, 2 * BLK * sizeof(float), stream_>>>(
        d_x_, d_gn_stats_, n512);
    k_gn1_norm<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_x_, d_work_a_, intra_norm_w_, intra_norm_b_, d_gn_stats_, kEncDim, L);

    // 9. Add skip (skip is [L,512], need [512,L] → transpose then add)
    k_transpose<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
        d_skip_, d_work_c_, L, kEncDim);
    k_add<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_work_a_, d_work_c_, n512);

    // 10. PReLU
    k_prelu1<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_work_a_, prelu_w_, n512);

    // 11. Conv1d(512→1024, k=1) + bias
    int n1024 = kEncDim * kNumSpk * L;
    gemm_CL(cublas_, conv_out_w_, d_work_a_, d_hidden_,
             kEncDim * kNumSpk, kEncDim, L);
    k_bias_ch<<<cdiv(n1024, BLK), BLK, 0, stream_>>>(
        d_hidden_, conv_out_b_, kEncDim * kNumSpk, L);

    // 12. Per-speaker gating + mask generation
    for (int spk = 0; spk < kNumSpk; spk++) {
        float* src = d_hidden_ + spk * kEncDim * L;
        float* msk = d_masks_  + spk * kEncDim * L;

        // output: Conv1d(512,512,1) + bias + Tanh
        gemm_CL(cublas_, output_w_, src, d_work_a_, kEncDim, kEncDim, L);
        k_bias_ch<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
            d_work_a_, output_b_, kEncDim, L);
        k_tanh<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_work_a_, n512);

        // output_gate: Conv1d(512,512,1) + bias + Sigmoid
        gemm_CL(cublas_, outgate_w_, src, d_work_c_, kEncDim, kEncDim, L);
        k_bias_ch<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
            d_work_c_, outgate_b_, kEncDim, L);
        k_sigmoid<<<cdiv(n512, BLK), BLK, 0, stream_>>>(d_work_c_, n512);

        // output * output_gate
        k_mul<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
            d_work_a_, d_work_c_, d_work_a_, n512);

        // conv1_decoder: Conv1d(512,512,1) + ReLU
        gemm_CL(cublas_, conv1_dec_w_, d_work_a_, msk, kEncDim, kEncDim, L);
        k_relu<<<cdiv(n512, BLK), BLK, 0, stream_>>>(msk, n512);
    }

    // 13. Apply mask: mask *= enc_out
    for (int spk = 0; spk < kNumSpk; spk++) {
        float* msk = d_masks_ + spk * kEncDim * L;
        k_mul<<<cdiv(n512, BLK), BLK, 0, stream_>>>(
            d_enc_out_, msk, msk, n512);
    }
}

// ============================================================================
// Forward — Decoder
// ============================================================================

void MossFormer2::forward_decoder(int L, int n_samples,
                                  float* d_out1, float* d_out2) {
    int Tout = (L - 1) * kEncStride + kEncKernel;

    for (int spk = 0; spk < kNumSpk; spk++) {
        float* src = d_masks_ + spk * kEncDim * L;
        float* dst = (spk == 0) ? d_out1 : d_out2;

        k_dec_conv<<<cdiv(Tout, BLK), BLK, 0, stream_>>>(
            src, d_dec_tmp_, dec_w_, L, Tout);

        int cp = std::min(Tout, n_samples);
        cudaMemcpyAsync(dst, d_dec_tmp_, cp * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        if (cp < n_samples) {
            int rem = n_samples - cp;
            k_zero<<<cdiv(rem, BLK), BLK, 0, stream_>>>(dst + cp, rem);
        }
    }
}

// ============================================================================
// FFConvM with ScaleNorm (FLASH: to_hidden, to_qk, to_out)
// d_in [L,dim_in] → d_out [L,dim_out], d_tmp scratch [L,max(dim_in,dim_out)]
// Uses d_work_b_ as DWConv intermediate (caller must not overlap)
// DilatedDenseNet — 2-layer dilated dense convolution
// Input/output: [256, L] channel-first. Uses d_ddc_cat_ for concat.
// ============================================================================

void MossFormer2::forward_ddc(const float* d_in, float* d_out, int L,
                              const FsmnWeights& fw) {
    int C = kFsmnInner;  // 256
    int n = C * L;

    // d_in is [256, L]. d_out will be [256, L].
    // DDC: skip = input
    // Layer 1: Conv2d(C,1,K,dil=1) → InstanceNorm → PReLU → concat with skip
    // Layer 2: Conv2d(C,2,K,dil=2) → InstanceNorm → PReLU → take last C channels

    // ---- Layer 1: dilated conv (dil=1) ----
    // Input: d_in [256, L], weight [256, 1, 39, 1] → group conv with cpg=1
    int pad1 = (kDDCKernel - 1) / 2;  // 19
    k_ddc_conv<<<cdiv(n, BLK), BLK, 0, stream_>>>(
        d_in, d_out, fw.ddc_conv1_w, C, 1, L, kDDCKernel, 1, pad1);
    // InstanceNorm
    k_instnorm<<<C, BLK, BLK * sizeof(float), stream_>>>(
        d_out, d_out, fw.ddc_norm1_w, fw.ddc_norm1_b, C, L);
    // PReLU (per-channel)
    k_prelu_ch<<<cdiv(n, BLK), BLK, 0, stream_>>>(
        d_out, fw.ddc_prelu1_w, C, L);

    // Concat: [conv1_out(256,L), skip(256,L)] → d_ddc_cat_ [512, L]
    // Python: cat([out, skip], dim=1) — conv output first, then original input
    k_cat_ch<<<cdiv(2 * n, BLK), BLK, 0, stream_>>>(
        d_out, d_in, d_ddc_cat_, C, C, L);

    // ---- Layer 2: dilated conv (dil=2) ----
    // Input: d_ddc_cat_ [512, L] as groups, weight [256, 2, 39, 1], cpg=2
    int pad2 = (kDDCKernel - 1) / 2;  // 19 (symmetric: pad_length/dil = 38/2 = 19)
    k_ddc_conv<<<cdiv(n, BLK), BLK, 0, stream_>>>(
        d_ddc_cat_, d_out, fw.ddc_conv2_w, C, 2, L, kDDCKernel, 2, pad2);
    // InstanceNorm
    k_instnorm<<<C, BLK, BLK * sizeof(float), stream_>>>(
        d_out, d_out, fw.ddc_norm2_w, fw.ddc_norm2_b, C, L);
    // PReLU (per-channel)
    k_prelu_ch<<<cdiv(n, BLK), BLK, 0, stream_>>>(
        d_out, fw.ddc_prelu2_w, C, L);
}

// ============================================================================
// UniDeepFsmn_dilated
// Input d_in [L, 256], output d_out [L, 256]
// ============================================================================

void MossFormer2::forward_uni_fsmn(const float* d_in, float* d_out, int L,
                                   const FsmnWeights& fw) {
    int n256 = L * kFsmnInner;

    // Linear(256→256) + ReLU
    // d_in [L, 256], weight [256, 256]
    gemm_nt(cublas_, d_in, fw.fsmn_linear_w, d_work_a_, L, kFsmnInner, kFsmnInner);
    k_bias_row<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_work_a_, fw.fsmn_linear_b, L, kFsmnInner);
    k_relu<<<cdiv(n256, BLK), BLK, 0, stream_>>>(d_work_a_, n256);

    // Project(256→256) — no bias
    gemm_nt(cublas_, d_work_a_, fw.fsmn_project_w, d_work_a_ + n256,
            L, kFsmnInner, kFsmnInner);

    // Transpose [L, 256] → [256, L] for channel-first DDC
    float* ddc_in = d_work_a_ + 2 * n256;
    k_transpose<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        d_work_a_ + n256, ddc_in, L, kFsmnInner);

    // DilatedDenseNet: [256, L] → [256, L]
    float* ddc_out = d_work_a_ + 3 * n256;
    forward_ddc(ddc_in, ddc_out, L, fw);

    // Transpose [256, L] → [L, 256] into scratch (d_work_a_ reuse ok, linear/relu done)
    // Cannot transpose directly to d_out: d_in may alias d_out, and we need
    // d_in intact for the residual add.
    float* ddc_T = d_work_a_;
    k_transpose<<<cdiv(n256, BLK), BLK, 0, stream_>>>(
        ddc_out, ddc_T, kFsmnInner, L);

    // Add DDC result to input: d_out += ddc_T  (safe even if d_in == d_out)
    if (d_out != d_in) {
        cudaMemcpyAsync(d_out, d_in, n256 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
    k_add<<<cdiv(n256, BLK), BLK, 0, stream_>>>(d_out, ddc_T, n256);
}

// ============================================================================
// Attention (integrated into forward_flash_layer)
// ============================================================================

// ============================================================================
// Top-level forward
// ============================================================================

bool MossFormer2::forward(const float* d_pcm_in, float* d_source1,
                          float* d_source2, int n_samples) {
    if (!initialized_) {
        LOG_ERROR("MF2", "Not initialized");
        return false;
    }
    if (n_samples > max_samples_ || n_samples < kEncKernel) {
        LOG_ERROR("MF2", "Bad n_samples=%d (max=%d min=%d)",
                  n_samples, max_samples_, kEncKernel);
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int L = (n_samples - kEncKernel) / kEncStride + 1;

    forward_encoder(d_pcm_in, n_samples, L);
    forward_masknet(L);
    forward_decoder(L, n_samples, d_source1, d_source2);

    cudaStreamSynchronize(stream_);
    auto t1 = std::chrono::high_resolution_clock::now();
    last_lat_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
    LOG_DEBUG("MF2", "forward: n=%d L=%d %.2fms", n_samples, L, last_lat_ms_);
    return true;
}

} // namespace deusridet
