/**
 * @file src/orator/wavlm_ecapa_encoder.cu
 * @philosophical_role
 *   WavLM-Large + ECAPA-TDNN encoder — turns a short speech window into a 192-dim speaker embedding. The perceptual act of abstracting identity from voice.
 * @serves
 *   Orator speaker_vector_store (for DB search); Auditus speaker path (for per-segment identification).
 */
// wavlm_ecapa_encoder.cu — WavLM-Large + ECAPA-TDNN implementation (Part 1)
//
// Implements: weight loading, scratch management, CNN feature extractor.
// See wavlm_ecapa_encoder.h for architecture overview.

#include "wavlm_ecapa_encoder.h"
#include "wavlm_ecapa_kernels.cuh"
#include "../communis/log.h"
#include "../machina/safetensors.h"

#include <cmath>
#include <cstdio>
#include <cassert>

namespace deusridet {


// ============================================================================
// CUDA Kernels — CNN Feature Extractor
// ============================================================================

// Layer normalization: x = gamma * (x - mean) / sqrt(var + eps) + beta
// Operates on last dimension: input [N, D], normalizes over D.
// One block per row (N), block-wide reduction for mean/var.

// Whole-sequence normalization for WavLM input (normalize=True):
// F.layer_norm(wav, wav.shape) — normalizes over all T samples.
// input [T], output [T]. Single block handles reduction.

// FP32 → FP16 conversion kernel for GEMM inputs.
// Uses half2 for 2× throughput.
// Named with _wlecapa suffix to avoid ODR collision with machina/forward.cu.

// Copy [C, T_src] → [C, T_dst] (drop trailing columns, for SamePad truncation)

// Conv1d: input [Cin, T_in], weight [Cout, Cin, K], bias optional
// output [Cout, T_out] where T_out = (T_in - K) / stride + 1
// No padding (WavLM CNN feature extractor has no padding).
// Kept as fallback; primary path uses cuDNN.

// LayerNorm on [C, T] along C dimension (for each time step):
// WavLM extractor_mode="layer_norm" applies LN per time frame on 512 channels.
// The input is [C, T] (channels first from Conv1d), LN is over C per t.
// We treat each column t as a "row" of length C.

// GELU activation (exact, matches PyTorch GELU)

// ============================================================================
// Weight loading
// ============================================================================

WavLMEcapaEncoder::WavLMEcapaEncoder() = default;

WavLMEcapaEncoder::~WavLMEcapaEncoder() {
    for (auto& [name, w] : weights_) {
        if (w.ptr) cudaFree(w.ptr);
        if (w.fp16) cudaFree(w.fp16);
    }
    if (scratch_a_) cudaFree(scratch_a_);
    if (scratch_b_) cudaFree(scratch_b_);
    if (scratch_c_) cudaFree(scratch_c_);
    if (d_hidden_states_) cudaFree(d_hidden_states_);
    if (d_im2col_) cudaFree(d_im2col_);
    if (d_pcm_buf_) cudaFree(d_pcm_buf_);
    if (d_gemm_a_) cudaFree(d_gemm_a_);
    if (d_gemm_b_) cudaFree(d_gemm_b_);
    if (d_pos_conv_weight_) cudaFree(d_pos_conv_weight_);
    if (d_pos_bias_) cudaFree(d_pos_bias_);
    if (d_cudnn_ws_) cudaFree(d_cudnn_ws_);
    if (cublas_) cublasDestroy(cublas_);
    if (cudnn_) cudnnDestroy(cudnn_);
    if (stream_) cudaStreamDestroy(stream_);
}

GpuWeight WavLMEcapaEncoder::upload_weight(const std::string& key,
                                            const float* data, int numel) {
    GpuWeight gw;
    gw.numel = numel;
    // FP32 copy (for custom kernels: conv1d, layer_norm, BN, etc.)
    cudaMalloc(&gw.ptr, numel * sizeof(float));
    cudaMemcpy(gw.ptr, data, numel * sizeof(float), cudaMemcpyHostToDevice);
    // FP16 copy (for Tensor Core GEMM via cublasGemmEx)
    cudaMalloc(&gw.fp16, numel * sizeof(__half));
    // Convert on GPU: launch kernel on default stream (weight loading, not perf-critical)
    int blocks = (numel / 2 + 255) / 256;
    f32_to_f16_wlecapa<<<blocks, 256>>>(gw.ptr, gw.fp16, numel);
    cudaDeviceSynchronize();
    return gw;
}

const GpuWeight& WavLMEcapaEncoder::w(const std::string& key) const {
    auto it = weights_.find(key);
    if (it == weights_.end()) {
        LOG_ERROR("WavLMEcapa", "weight not found: %s", key.c_str());
        static GpuWeight empty;
        return empty;
    }
    return it->second;
}

bool WavLMEcapaEncoder::load_weights(const std::string& model_path) {
    LOG_INFO("WavLMEcapa", "loading weights from %s", model_path.c_str());

    SafetensorsFile sf(model_path);
    auto names = sf.tensor_names();
    LOG_INFO("WavLMEcapa", "%zu tensors in safetensors", names.size());

    int loaded = 0;
    size_t total_bytes = 0;

    for (const auto& name : names) {
        auto tensor = sf.get_tensor(name);
        if (!tensor) {
            LOG_WARN("WavLMEcapa", "failed to load tensor %s", name.c_str());
            continue;
        }

        int numel = (int)tensor->numel();

        // Tensor data is mmap'd float32 (safetensors uses F32)
        const float* data = reinterpret_cast<const float*>(tensor->data());
        weights_[name] = upload_weight(name, data, numel);
        total_bytes += numel * sizeof(float);
        loaded++;
    }

    LOG_INFO("WavLMEcapa", "loaded %d tensors (%.1f MB GPU)",
             loaded, (float)(total_bytes / (1024.0 * 1024.0)));
    return loaded > 0;
}

// Definition duplicated here from transformer peer under R1 split (static
// linkage keeps each TU's instance private; merge_qkv_weights below uses it).
static std::string enc_layer_key(int layer, const char* suffix) {
    return "frontend.upstream.upstream.model.encoder.layers."
           + std::to_string(layer) + "." + suffix;
}

void WavLMEcapaEncoder::merge_qkv_weights() {
    int D = WavLMConfig::embed_dim;  // 1024
    for (int i = 0; i < WavLMConfig::num_layers; i++) {
        std::string q_w = enc_layer_key(i, "self_attn.q_proj.weight");
        std::string k_w = enc_layer_key(i, "self_attn.k_proj.weight");
        std::string v_w = enc_layer_key(i, "self_attn.v_proj.weight");
        std::string q_b = enc_layer_key(i, "self_attn.q_proj.bias");
        std::string k_b = enc_layer_key(i, "self_attn.k_proj.bias");
        std::string v_b = enc_layer_key(i, "self_attn.v_proj.bias");

        // Merged weight: [3*D, D] = Q[D,D] | K[D,D] | V[D,D]
        GpuWeight mw;
        mw.numel = 3 * D * D;
        cudaMalloc(&mw.ptr, mw.numel * sizeof(float));
        cudaMemcpy(mw.ptr,             weights_[q_w].ptr, D * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(mw.ptr + D * D,     weights_[k_w].ptr, D * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(mw.ptr + 2 * D * D, weights_[v_w].ptr, D * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMalloc(&mw.fp16, mw.numel * sizeof(__half));
        f32_to_f16_wlecapa<<<(mw.numel / 2 + 255) / 256, 256>>>(mw.ptr, mw.fp16, mw.numel);
        weights_[enc_layer_key(i, "self_attn.qkv_merged.weight")] = mw;

        // Merged bias: [3*D]
        GpuWeight mb;
        mb.numel = 3 * D;
        cudaMalloc(&mb.ptr, mb.numel * sizeof(float));
        cudaMemcpy(mb.ptr,         weights_[q_b].ptr, D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(mb.ptr + D,     weights_[k_b].ptr, D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(mb.ptr + 2 * D, weights_[v_b].ptr, D * sizeof(float), cudaMemcpyDeviceToDevice);
        mb.fp16 = nullptr;  // bias not used in FP16 path
        weights_[enc_layer_key(i, "self_attn.qkv_merged.bias")] = mb;

        // Free individual Q/K/V weight+bias buffers
        for (const auto& key : {q_w, k_w, v_w, q_b, k_b, v_b}) {
            auto& gw = weights_[key];
            if (gw.ptr)  { cudaFree(gw.ptr);  gw.ptr  = nullptr; }
            if (gw.fp16) { cudaFree(gw.fp16); gw.fp16 = nullptr; }
            weights_.erase(key);
        }
    }
    cudaDeviceSynchronize();
    LOG_INFO("WavLMEcapa", "merged QKV weights for %d transformer layers",
             WavLMConfig::num_layers);
}

// ============================================================================
// Scratch management
// ============================================================================

bool WavLMEcapaEncoder::ensure_scratch(int n_samples) {
    // Compute max T' through CNN:
    // Layer 0: (T - 10) / 5 + 1
    // Layer 1-4: (T' - 3) / 2 + 1
    // Layer 5-6: (T' - 2) / 2 + 1
    int T = n_samples;
    T = (T - 10) / 5 + 1;
    for (int i = 1; i <= 4; i++) T = (T - 3) / 2 + 1;
    for (int i = 5; i <= 6; i++) T = (T - 2) / 2 + 1;
    int T_prime = T;

    if (T_prime <= scratch_max_T_) return true;

    // Free old buffers
    if (scratch_a_) cudaFree(scratch_a_);
    if (scratch_b_) cudaFree(scratch_b_);
    if (scratch_c_) cudaFree(scratch_c_);
    if (d_hidden_states_) cudaFree(d_hidden_states_);
    if (d_im2col_) cudaFree(d_im2col_);
    if (d_gemm_a_) cudaFree(d_gemm_a_);
    if (d_gemm_b_) cudaFree(d_gemm_b_);

    // Size each scratch to hold the largest intermediate
    // CNN: max is layer0 output = n_samples * 512 / 5 (approx)
    int max_cnn_T = (n_samples - 10) / 5 + 1;
    size_t cnn_size = (size_t)WavLMConfig::cnn_dim * max_cnn_T;

    // Transformer: T' * embed_dim
    size_t xfmr_size = (size_t)T_prime * WavLMConfig::embed_dim;

    // Attention: num_heads * T' * T' for attention weights
    size_t attn_size = (size_t)WavLMConfig::num_heads * T_prime * T_prime;

    // ECAPA: max is 4608 * T' (from pool_global_stats: cat(x4, mean, std))
    size_t ecapa_size = (size_t)4608 * T_prime;

    // Take max of all
    size_t max_per_buf = std::max({cnn_size, xfmr_size, attn_size, ecapa_size,
                                   (size_t)n_samples});

    // Allocate 3 scratch buffers
    scratch_size_ = max_per_buf;
    cudaMalloc(&scratch_a_, max_per_buf * sizeof(float));
    cudaMalloc(&scratch_b_, max_per_buf * sizeof(float));
    cudaMalloc(&scratch_c_, max_per_buf * sizeof(float));

    // Hidden states: 25 * T' * 1024
    size_t hs_size = (size_t)WavLMConfig::num_hidden_states * T_prime *
                     WavLMConfig::embed_dim;
    hidden_states_size_ = hs_size;
    cudaMalloc(&d_hidden_states_, hs_size * sizeof(float));

    // Im2col buffer: max K=5, C_in=1024, T_out=T'
    size_t im2col_max = (size_t)WavLMConfig::embed_dim * 5 * T_prime;
    im2col_size_ = im2col_max;
    cudaMalloc(&d_im2col_, im2col_max * sizeof(float));

    // FP16 GEMM scratch: 2 buffers, each max(T'*4096, 16*T'*T', C_in*K*T')
    size_t fp16_max = std::max({(size_t)T_prime * 4096,
                                 (size_t)16 * T_prime * T_prime,
                                 im2col_max});
    gemm_fp16_size_ = fp16_max;
    cudaMalloc(&d_gemm_a_, fp16_max * sizeof(__half));
    cudaMalloc(&d_gemm_b_, fp16_max * sizeof(__half));

    scratch_max_T_ = T_prime;

    LOG_INFO("WavLMEcapa", "scratch allocated T'=%d (%.1f MB/buf, %.1f MB hs, %.1f MB im2col)",
             T_prime, (float)(max_per_buf * 4.0 / (1024 * 1024)),
             (float)(hs_size * 4.0 / (1024 * 1024)),
             (float)(im2col_max * 4.0 / (1024 * 1024)));

    return scratch_a_ && scratch_b_ && scratch_c_ && d_hidden_states_ && d_im2col_;
}

// ============================================================================
// Init
// ============================================================================

bool WavLMEcapaEncoder::init(const std::string& model_path) {
    if (initialized_) return true;
    model_path_ = model_path;

    // Create CUDA resources
    cublasCreate(&cublas_);
    cudaStreamCreate(&stream_);
    cublasSetStream(cublas_, stream_);
    // Enable TF32 Tensor Core math for all cublasSgemm calls (~2-3× GEMM speedup on SM87).
    cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);

    // cuDNN for CNN feature extractor + positional conv
    cudnnCreate(&cudnn_);
    cudnnSetStream(cudnn_, stream_);

    // Load weights
    if (!load_weights(model_path)) {
        LOG_ERROR("WavLMEcapa", "failed to load weights");
        return false;
    }

    // Merge Q/K/V projections into single [3D, D] weight per layer (reduces 72→24 cuBLAS calls)
    merge_qkv_weights();

    initialized_ = true;
    LOG_INFO("WavLMEcapa", "initialized successfully");
    return true;
}

// ============================================================================
// CNN Feature Extractor forward
// ============================================================================

// Weight key helpers
static std::string cnn_conv_w(int layer) {
    // s3prl key: frontend.upstream.upstream.model.feature_extractor.conv_layers.{i}.0.weight
    return "frontend.upstream.upstream.model.feature_extractor.conv_layers."
           + std::to_string(layer) + ".0.weight";
}

static std::string cnn_ln_w(int layer) {
    // extractor_mode=layer_norm: LN at position [2] in Sequential
    // Sequential(Conv1d, Dropout, Sequential(TransposeLast, LN, TransposeLast), GELU)
    // LN is at .2.1 — the Fp32LayerNorm inside the inner Sequential
    return "frontend.upstream.upstream.model.feature_extractor.conv_layers."
           + std::to_string(layer) + ".2.1.weight";
}

static std::string cnn_ln_b(int layer) {
    return "frontend.upstream.upstream.model.feature_extractor.conv_layers."
           + std::to_string(layer) + ".2.1.bias";
}

void WavLMEcapaEncoder::forward_cnn(const float* d_wav, int n_samples,
                                     float* d_out, int& T_out) {
    // Input: d_wav [n_samples] (already layer-norm'd if needed)
    // We treat it as [1, n_samples] = [Cin=1, T_in]
    const float* d_input = d_wav;
    float* buf_a = scratch_a_;
    float* buf_b = scratch_b_;

    int Cin = 1;
    int T = n_samples;

    for (int i = 0; i < WavLMConfig::cnn_layers; i++) {
        int K = WavLMConfig::cnn_kernels[i];
        int S = WavLMConfig::cnn_strides[i];
        int Cout = WavLMConfig::cnn_dim;  // all layers output 512
        int T_next = (T - K) / S + 1;

        // Conv1d → buf_a always, LN → buf_b always.
        // Sequential execution on same stream guarantees:
        //   conv reads d_input (buf_b or d_wav), writes buf_a
        //   LN reads buf_a, writes buf_b
        // No aliasing conflict.
        const auto& wt = w(cnn_conv_w(i));

        // cuDNN Conv1d (as Conv2d H=1): no padding, stride S, groups=1
        forward_conv1d_cudnn(d_input, buf_a, Cin, Cout, T, K, S, 0, 1, 1,
                             wt.ptr, nullptr);

        // LayerNorm over C dimension per time step (extractor_mode=layer_norm)
        // LN input is [Cout, T_next], normalize over Cout for each t
        const auto& ln_w = w(cnn_ln_w(i));
        const auto& ln_b = w(cnn_ln_b(i));

        layer_norm_channels_kernel<<<T_next, std::min(Cout, 512), 0, stream_>>>(
            buf_a, buf_b, ln_w.ptr, ln_b.ptr, Cout, T_next);

        // GELU activation (in-place on buf_b)
        gelu_kernel<<<div_ceil(Cout * T_next, BLOCK), BLOCK, 0, stream_>>>(
            buf_b, Cout * T_next);

        d_input = buf_b;
        Cin = Cout;
        T = T_next;
    }

    // Copy result to d_out if not already there
    if (d_input != d_out) {
        cudaMemcpyAsync(d_out, d_input, Cin * T * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
    T_out = T;
}

// ============================================================================
// Transpose kernel: [M, N] row-major → [N, M] row-major
// Equivalently: input[m * N + n] → output[n * M + m]
// ============================================================================

// Bias add: output[row * cols + c] += bias[c]

// Channel-first bias: data[ch * T + t] += bias[ch]
// For col-major GEMM output [T, C_out] or row-major [C_out, T]

// ============================================================================
// Feature projection: LayerNorm(512) + Linear(512 → 1024)
// ============================================================================

static const char* FE_LN_W = "frontend.upstream.upstream.model.layer_norm.weight";
static const char* FE_LN_B = "frontend.upstream.upstream.model.layer_norm.bias";
static const char* FE_PROJ_W = "frontend.upstream.upstream.model.post_extract_proj.weight";
static const char* FE_PROJ_B = "frontend.upstream.upstream.model.post_extract_proj.bias";

void WavLMEcapaEncoder::forward_layer_norm(const float* d_in, float* d_out,
                                            int T, int dim,
                                            const float* d_gamma,
                                            const float* d_beta) {
    // Input/output: [T, dim] row-major. LN over last dimension (dim).
    // One block per row.
    int block = std::min(dim, 1024);
    layer_norm_kernel<<<T, block, 0, stream_>>>(
        d_in, d_out, d_gamma, d_beta, T, dim);
}

void WavLMEcapaEncoder::forward_linear(const float* d_in, float* d_out,
                                        int rows, int in_dim, int out_dim,
                                        const float* d_weight,
                                        const float* d_bias,
                                        const __half* d_weight_fp16) {
    float alpha = 1.0f, beta = 0.0f;

    if (d_weight_fp16) {
        // Tensor Core FP16 path: convert activation to FP16, use cublasGemmEx.
        int in_count = rows * in_dim;
        int conv_blocks = (in_count / 2 + 255) / 256;
        f32_to_f16_wlecapa<<<conv_blocks, 256, 0, stream_>>>(d_in, d_gemm_a_, in_count);

        // y = x @ W^T + bias  →  col-major: Y_cm = W_cm^T @ X_cm
        cublasGemmEx(cublas_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     out_dim, rows, in_dim,
                     &alpha,
                     d_weight_fp16, CUDA_R_16F, in_dim,
                     d_gemm_a_, CUDA_R_16F, in_dim,
                     &beta,
                     d_out, CUDA_R_32F, out_dim,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        // FP32 TF32 fallback
        cublasSgemm(cublas_,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    out_dim, rows, in_dim,
                    &alpha,
                    d_weight, in_dim,
                    d_in, in_dim,
                    &beta,
                    d_out, out_dim);
    }

    // Add bias
    if (d_bias) {
        int total = rows * out_dim;
        bias_add_kernel<<<div_ceil(total, BLOCK), BLOCK, 0, stream_>>>(
            d_out, d_bias, rows, out_dim);
    }
}
} // namespace deusridet
