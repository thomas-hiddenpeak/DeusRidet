/**
 * @file src/orator/wavlm_ecapa_transformer.cu
 * @philosophical_role
 *   Peer TU of wavlm_ecapa_encoder.cu under R1 800-line hard cap — positional conv + transformer encoder + diagnostic test_* probes.
 * @serves
 *   Orator speaker embedding extraction.
 */
#include "wavlm_ecapa_encoder.h"
#include "wavlm_ecapa_kernels.cuh"
#include "../communis/log.h"
#include "../machina/safetensors.h"

#include <cmath>
#include <cstdio>
#include <cassert>

namespace deusridet {

// Duplicated from main TU (originally lived in featurizer block, used by
// test_projection here under R1 split).
static const char* FE_LN_W = "frontend.upstream.upstream.model.layer_norm.weight";
static const char* FE_LN_B = "frontend.upstream.upstream.model.layer_norm.bias";
static const char* FE_PROJ_W = "frontend.upstream.upstream.model.post_extract_proj.weight";
static const char* FE_PROJ_B = "frontend.upstream.upstream.model.post_extract_proj.bias";


// ============================================================================
// Positional convolution: weight_norm Conv1d + SamePad + GELU
// ============================================================================

// Grouped Conv1d with padding: groups=16, kern=128, pad=64, stride=1
// Input: [C, T_in] (channels first), Output: [C, T_in] (after SamePad removes 1)
// Weight: [C, C/groups, K] precomputed from weight_norm(g, v)

// Compute effective pos_conv weight from weight_norm decomposition
// weight_g: [1, 1, K], weight_v: [C, C/groups, K]
// For each k: norm_k = ||v[:,:,k]||_2, weight[:,:,k] = g[k] * v[:,:,k] / norm_k

static const char* POS_CONV_G = "frontend.upstream.upstream.model.encoder.pos_conv.0.weight_g";
static const char* POS_CONV_V = "frontend.upstream.upstream.model.encoder.pos_conv.0.weight_v";
static const char* POS_CONV_B = "frontend.upstream.upstream.model.encoder.pos_conv.0.bias";

// skip_add + GELU: output = GELU(input + skip) (elementwise)

// ============================================================================
// Encoder key helpers
// ============================================================================
static const char* ENC_FINAL_LN_W = "frontend.upstream.upstream.model.encoder.layer_norm.weight";
static const char* ENC_FINAL_LN_B = "frontend.upstream.upstream.model.encoder.layer_norm.bias";

static std::string enc_layer_key(int layer, const char* suffix) {
    return "frontend.upstream.upstream.model.encoder.layers."
           + std::to_string(layer) + "." + suffix;
}

// ============================================================================
// Transformer self-attention forward (single layer)
// ============================================================================

// Compute relative position bias for all heads: [num_heads, T, T]
// Uses bucketed relative position indices → lookup from learned embeddings

// GRU-based relative position gating
// gate = sigmoid(Linear(mean_pool(attn_weights_per_head)))
// attn_weights *= gate.unsqueeze(-1)
// grep_linear: [8, 64] (2 * (num_heads/2) × head_dim)
// grep_a: [1, num_heads, 1, 1] → used as exp base

// Softmax along last dimension: input [rows, cols], in-place

// Reshape [T, H*Dh] (head-interleaved) → [H, T, Dh] (head-contiguous)
// Input[t][h*Dh + d] → Output[h][t][d]

// Reshape [H, T, Dh] (head-contiguous) → [T, H*Dh] (head-interleaved)
// Input[h][t][d] → Output[t][h*Dh + d]

// Split merged QKV [T, 3*D] → three [H, T, Dh] tensors in one kernel
// qkv[t, proj*D + h*Dh + d] → Q/K/V[h, t, d]  where proj ∈ {0,1,2}

// GRU relative position gating + position bias scaling
// For each (h, t): compute gate from Q vector, then scale position bias row
// Q: [H, T, Dh], grep_linear_w: [8, Dh], grep_linear_b: [8], grep_a: [H]
// pos_bias: [H, T, T], attn: [H, T, T] (output: attn += gate * pos_bias)
// Note: WavLM applies grep_linear to the un-projected LN output (reshaped as heads),
// NOT to the Q-projected output. The input `x_flat` is [T, D] where D = H*Dh.

// Simple vector add: y[i] += x[i]

// ============================================================================
// Transformer layer forward
// ============================================================================

void WavLMEcapaEncoder::forward_transformer_layer(
        float* d_x, int T, int layer_idx, float* d_pos_bias) {
    int D = WavLMConfig::embed_dim;
    int H = WavLMConfig::num_heads;
    int Dh = WavLMConfig::head_dim;
    int Dff = WavLMConfig::ffn_dim;

    // Buffer layout:
    // scratch_b_: [T,D] for LN,  then [H,T,Dh] for attn_out, then [T, Dff] for FFN
    // scratch_c_: [H,T,Dh]*3 for Q,K,V + [H,T,T] for attn_scores

    float* d_ln = scratch_b_;
    float* d_q_mh = scratch_c_;                      // [H, T, Dh] = [16, T, 64]
    float* d_k_mh = scratch_c_ + H * T * Dh;         // [H, T, Dh]
    float* d_v_mh = scratch_c_ + 2 * H * T * Dh;     // [H, T, Dh]
    float* d_attn = scratch_c_ + 3 * H * T * Dh;     // [H, T, T]

    // ── Self-Attention ──

    // 1. Pre-LN
    auto& sa_ln_w = w(enc_layer_key(layer_idx, "self_attn_layer_norm.weight"));
    auto& sa_ln_b = w(enc_layer_key(layer_idx, "self_attn_layer_norm.bias"));
    forward_layer_norm(d_x, d_ln, T, D, sa_ln_w.ptr, sa_ln_b.ptr);

    // 2. Merged QKV projection → [T, 3*D], then split+reshape to [H, T, Dh] × 3
    float* d_proj_qkv = scratch_b_ + T * D;  // temp [T, 3*D]

    auto& qkv_w = w(enc_layer_key(layer_idx, "self_attn.qkv_merged.weight"));
    auto& qkv_b = w(enc_layer_key(layer_idx, "self_attn.qkv_merged.bias"));
    forward_linear(d_ln, d_proj_qkv, T, D, 3 * D,
        qkv_w.ptr, qkv_b.ptr, qkv_w.fp16);

    int mh_total = H * T * Dh;
    split_reshape_qkv_kernel<<<div_ceil(mh_total, BLOCK), BLOCK, 0, stream_>>>(
        d_proj_qkv, d_q_mh, d_k_mh, d_v_mh, T, H, Dh);

    // 3. QK^T / sqrt(Dh) → [H, T, T]
    // We want attn_data[q*T+k] = Q[q]·K[k]/sqrt(d), so softmax can be applied row-wise.
    // In cuBLAS col-major: C_cm[k,q] = K[k]·Q[q] gives C_mem[k+q*T] = C_mem[q*T+k] for row-major.
    // So: C = K^T @ Q (A=K with OP_T, B=Q with OP_N)
    float alpha_scale = 1.0f / sqrtf((float)Dh);
    float beta_zero = 0.0f;
    {
        // Convert Q, K to FP16 for Tensor Core batched GEMM
        int qk_count = H * T * Dh;
        int fp16_blocks = (qk_count / 2 + 255) / 256;
        f32_to_f16_wlecapa<<<fp16_blocks, 256, 0, stream_>>>(d_k_mh, d_gemm_b_, qk_count);
        f32_to_f16_wlecapa<<<fp16_blocks, 256, 0, stream_>>>(d_q_mh, d_gemm_a_, qk_count);

        cublasGemmStridedBatchedEx(cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, Dh,
            &alpha_scale,
            d_gemm_b_, CUDA_R_16F, Dh, (long long)T * Dh,   // A = K_fp16
            d_gemm_a_, CUDA_R_16F, Dh, (long long)T * Dh,   // B = Q_fp16
            &beta_zero,
            d_attn, CUDA_R_32F, T, (long long)T * T,         // C = FP32 attn scores
            H,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // 4. GRU gating + position bias scaling
    // WavLM applies grep_linear to the un-projected LN output (d_ln), not to Q
    // attn[h,t,:] += gate(x_ln[t,h]) * pos_bias[h,t,:]
    gru_gate_bias_kernel<<<H * T, std::min(T, 256), 0, stream_>>>(
        d_ln,  // [T, D] — LN output BEFORE Q projection
        w(enc_layer_key(layer_idx, "self_attn.grep_linear.weight")).ptr,
        w(enc_layer_key(layer_idx, "self_attn.grep_linear.bias")).ptr,
        w(enc_layer_key(layer_idx, "self_attn.grep_a")).ptr,
        d_pos_bias, d_attn,
        T, H, Dh);

    // 5. Softmax along last dimension: [H*T rows, each of T columns]
    int sm_threads = ((std::min(T, 256) + 31) / 32) * 32;  // round up to warp boundary
    softmax_kernel<<<H * T, sm_threads, 0, stream_>>>(
        d_attn, H * T, T);

    // 6. attn_scores @ V → attn_output [H, T, Dh]
    // attn_h = [T, T], V_h = [T, Dh] → [T, Dh]
    // In col-major: attn_cm = [T, T], V_cm = [Dh, T]
    // output_cm = V_cm @ attn_cm^T = [Dh, T] @ [T, T] = [Dh, T]
    // But we want: output = attn @ V = [T, T] @ [T, Dh] = [T, Dh]
    // col-major: out_cm = [Dh, T] = V_cm @ attn_cm^T
    // Hmm, attn_cm is row-major [T,T] interpreted as col-major [T,T].
    // [T,T] row-major = [T,T] col-major (same for square, but elements are transposed!)
    // Actually for row-major [T,T], element [i,j] = data[i*T+j].
    // Col-major interpretation: element [i,j] = data[i+j*T], so col-major [i,j] = row-major [j,i].
    // So col-major view of row-major [T,T] is the transpose.
    // attn_cm[i,j] = attn_rm[j,i] = softmax(QK^T)[j, i]
    // We want: out = attn_rm @ V = out[t, d] = sum_t2 attn_rm[t, t2] * V[t2, d]
    // In col-major: out_cm[d, t] = sum_t2 V_cm[d, t2] * attn_rm[t, t2]
    //             = sum_t2 V_cm[d, t2] * attn_cm[t2, t]  (since attn_cm = attn_rm^T)
    // This is: out_cm = V_cm @ attn_cm
    // cublasSgemm(N, N, Dh, T, T, alpha, V_cm, Dh, attn_cm, T, beta, out_cm, Dh)

    // Store output in scratch_b_ [H, T, Dh] — where d_ln was (safe to reuse)
    float* d_attn_out_mh = scratch_b_;
    float alpha_one = 1.0f;
    {
        // Convert V and attn to FP16 for Tensor Core batched GEMM
        int v_count = H * T * Dh;
        int attn_count = H * T * T;
        f32_to_f16_wlecapa<<<(v_count / 2 + 255) / 256, 256, 0, stream_>>>(
            d_v_mh, d_gemm_a_, v_count);
        f32_to_f16_wlecapa<<<(attn_count / 2 + 255) / 256, 256, 0, stream_>>>(
            d_attn, d_gemm_b_, attn_count);

        cublasGemmStridedBatchedEx(cublas_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Dh, T, T,
            &alpha_one,
            d_gemm_a_, CUDA_R_16F, Dh, (long long)T * Dh,    // A = V_fp16
            d_gemm_b_, CUDA_R_16F, T, (long long)T * T,       // B = attn_fp16
            &beta_zero,
            d_attn_out_mh, CUDA_R_32F, Dh, (long long)T * Dh, // C = FP32
            H,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // 7. Reshape [H, T, Dh] → [T, H*Dh] = [T, D] and output projection
    float* d_attn_flat = scratch_c_;  // reuse scratch_c_ start (Q no longer needed)
    reshape_from_multihead_kernel<<<div_ceil(mh_total, BLOCK), BLOCK, 0, stream_>>>(
        d_attn_out_mh, d_attn_flat, T, H, Dh);

    // out_proj: Linear(D, D) → scratch_b_
    float* d_sa_out = scratch_b_;
    forward_linear(d_attn_flat, d_sa_out, T, D, D,
        w(enc_layer_key(layer_idx, "self_attn.out_proj.weight")).ptr,
        w(enc_layer_key(layer_idx, "self_attn.out_proj.bias")).ptr,
        w(enc_layer_key(layer_idx, "self_attn.out_proj.weight")).fp16);

    // 8. Residual add: d_x += d_sa_out
    vector_add_kernel<<<div_ceil(T * D, BLOCK), BLOCK, 0, stream_>>>(
        d_x, d_sa_out, T * D);

    // ── Feed-Forward ──

    // 9. Pre-LN for FFN
    auto& ff_ln_w = w(enc_layer_key(layer_idx, "final_layer_norm.weight"));
    auto& ff_ln_b = w(enc_layer_key(layer_idx, "final_layer_norm.bias"));
    forward_layer_norm(d_x, d_ln, T, D, ff_ln_w.ptr, ff_ln_b.ptr);

    // 10. FC1: [T, D] → [T, Dff] (Dff = 4096)
    float* d_fc1 = scratch_c_;  // [T, 4096] — fits in scratch_c_
    forward_linear(d_ln, d_fc1, T, D, Dff,
        w(enc_layer_key(layer_idx, "fc1.weight")).ptr,
        w(enc_layer_key(layer_idx, "fc1.bias")).ptr,
        w(enc_layer_key(layer_idx, "fc1.weight")).fp16);

    // 11. GELU
    gelu_kernel<<<div_ceil(T * Dff, BLOCK), BLOCK, 0, stream_>>>(
        d_fc1, T * Dff);

    // 12. FC2: [T, Dff] → [T, D]
    float* d_fc2 = scratch_b_;
    forward_linear(d_fc1, d_fc2, T, Dff, D,
        w(enc_layer_key(layer_idx, "fc2.weight")).ptr,
        w(enc_layer_key(layer_idx, "fc2.bias")).ptr,
        w(enc_layer_key(layer_idx, "fc2.weight")).fp16);

    // 13. Residual add: d_x += d_fc2
    vector_add_kernel<<<div_ceil(T * D, BLOCK), BLOCK, 0, stream_>>>(
        d_x, d_fc2, T * D);
}

// ============================================================================
// Test interfaces for layer-by-layer debugging
// ============================================================================

float* WavLMEcapaEncoder::test_cnn(const float* d_wav, int n_samples, int& T_out) {
    if (!initialized_) return nullptr;
    ensure_scratch(n_samples);

    // Step 1: Waveform normalization (if normalize_input)
    float* d_normed = scratch_c_;  // use scratch_c as temp
    if (WavLMConfig::normalize_input) {
        wav_layer_norm_kernel<<<1, BLOCK, 0, stream_>>>(d_wav, d_normed, n_samples);
    } else {
        cudaMemcpyAsync(d_normed, d_wav, n_samples * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // Step 2: CNN feature extraction
    // Result goes to scratch_a or scratch_b (alternating)
    forward_cnn(d_normed, n_samples, scratch_a_, T_out);
    cudaStreamSynchronize(stream_);

    return scratch_a_;
}

float* WavLMEcapaEncoder::test_projection(const float* d_cnn, int T, int& T_out) {
    if (!initialized_) return nullptr;

    // d_cnn: [512, T] on GPU (CNN output, channels first)
    // Step 1: Transpose [512, T] → [T, 512]
    int total = WavLMConfig::cnn_dim * T;
    transpose_2d_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
        d_cnn, scratch_b_, WavLMConfig::cnn_dim, T);

    // Step 2: LayerNorm(512) → [T, 512]
    forward_layer_norm(scratch_b_, scratch_c_, T, WavLMConfig::cnn_dim,
                       w(FE_LN_W).ptr, w(FE_LN_B).ptr);

    // Step 3: Linear(512 → 1024) → [T, 1024]
    forward_linear(scratch_c_, scratch_a_, T,
                   WavLMConfig::cnn_dim, WavLMConfig::embed_dim,
                   w(FE_PROJ_W).ptr, w(FE_PROJ_B).ptr,
                   w(FE_PROJ_W).fp16);

    T_out = T;
    cudaStreamSynchronize(stream_);
    return scratch_a_;  // [T, 1024] row-major
}

float* WavLMEcapaEncoder::test_pos_conv(const float* d_proj, int T, int& T_out) {
    if (!initialized_) return nullptr;

    // d_proj: [T, 1024] row-major
    // Positional conv operates on [C, T] (channels first)
    int C = WavLMConfig::embed_dim;
    int K = WavLMConfig::pos_conv_kernel;
    int total = T * C;

    // Step 1: Transpose [T, 1024] → [1024, T] into scratch_b_
    transpose_2d_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
        d_proj, scratch_b_, T, C);

    // Step 2: Compute weight_norm weight if not cached
    if (!pos_conv_weight_computed_) {
        int group_size = C / WavLMConfig::pos_conv_groups;
        int total_per_k = C * group_size;  // 1024 * 64 = 65536
        cudaMalloc(&d_pos_conv_weight_, C * group_size * K * sizeof(float));
        weight_norm_kernel<<<K, BLOCK, 0, stream_>>>(
            w(POS_CONV_G).ptr, w(POS_CONV_V).ptr, d_pos_conv_weight_,
            total_per_k, K);
        pos_conv_weight_computed_ = true;
    }

    // Step 3: Grouped Conv1d via cuDNN (pad=64, K=128 → T+1 outputs), then SamePad truncate
    int pad = K / 2;  // 64
    int T_full = T + 2 * pad - K + 1;  // = T + 1 for symmetric pad
    // cuDNN output → scratch_a_ (free after step 1), then truncate to scratch_c_
    forward_conv1d_cudnn(scratch_b_, scratch_a_, C, C, T, K, 1, pad,
                         WavLMConfig::pos_conv_groups, 1,
                         d_pos_conv_weight_, w(POS_CONV_B).ptr);
    // SamePad: drop last column  [1024, T+1] → [1024, T]
    truncate_channels_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, scratch_c_, C, T_full, T);

    // Step 4: GELU on conv output (in-place on scratch_c_)
    gelu_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
        scratch_c_, total);

    // Step 5: Residual add: scratch_b_ += scratch_c_ (input + GELU(conv))
    {
        float alpha = 1.0f;
        cublasSaxpy(cublas_, total, &alpha, scratch_c_, 1, scratch_b_, 1);
    }

    // Step 6: Transpose [1024, T] → [T, 1024] into scratch_a_
    transpose_2d_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
        scratch_b_, scratch_a_, C, T);

    T_out = T;
    cudaStreamSynchronize(stream_);
    return scratch_a_;  // [T, 1024] row-major
}

float* WavLMEcapaEncoder::test_encoder(const float* d_pos, int T, int& T_out) {
    if (!initialized_) return nullptr;

    int D = WavLMConfig::embed_dim;
    int H = WavLMConfig::num_heads;

    // d_pos: [T, 1024] from test_pos_conv, held in scratch_a_
    // We'll work in scratch_a_ as the running activation buffer.

    // Step 1: Compute relative position bias using layer 0's weights (reused for all layers)
    if (pos_bias_T_ != T) {
        if (d_pos_bias_) cudaFree(d_pos_bias_);
        cudaMalloc(&d_pos_bias_, H * T * T * sizeof(float));
        pos_bias_T_ = T;

        int total = H * T * T;
        auto& rel_attn_w = w(enc_layer_key(0, "self_attn.relative_attention_bias.weight"));
        compute_rel_pos_bias_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
            rel_attn_w.ptr, d_pos_bias_, T, H,
            WavLMConfig::num_buckets, WavLMConfig::max_distance);
    }

    // Step 2: Store initial hidden state (layer 0 = input to encoder)
    // d_hidden_states_[0] = d_pos (copy since scratch_a_ will be modified)
    cudaMemcpyAsync(d_hidden_states_, d_pos, T * D * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // Step 3: Run 24 transformer layers
    // d_x lives in scratch_a_ — forward_transformer_layer modifies it in-place
    for (int i = 0; i < WavLMConfig::num_layers; i++) {
        forward_transformer_layer(scratch_a_, T, i, d_pos_bias_);

        // Store hidden state after each layer
        float* hs_slot = d_hidden_states_ + (i + 1) * T * D;
        cudaMemcpyAsync(hs_slot, scratch_a_, T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // Step 4: Final layer norm (encoder.layer_norm)
    // Applied to the last layer output in scratch_a_
    forward_layer_norm(scratch_a_, scratch_b_, T, D,
                       w(ENC_FINAL_LN_W).ptr, w(ENC_FINAL_LN_B).ptr);
    // Copy back to scratch_a_ AND overwrite hidden_states[24] with LN output
    cudaMemcpyAsync(scratch_a_, scratch_b_, T * D * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    float* hs_last = d_hidden_states_ + WavLMConfig::num_layers * T * D;
    cudaMemcpyAsync(hs_last, scratch_b_, T * D * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    T_out = T;
    cudaStreamSynchronize(stream_);
    return scratch_a_;  // [T, 1024] — final layer norm output
}

const float* WavLMEcapaEncoder::get_hidden_state(int layer) const {
    if (!d_hidden_states_ || layer < 0 || layer > WavLMConfig::num_layers)
        return nullptr;
    return d_hidden_states_ + layer * scratch_max_T_ * WavLMConfig::embed_dim;
}

} // namespace deusridet
