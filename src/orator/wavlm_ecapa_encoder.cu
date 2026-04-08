// wavlm_ecapa_encoder.cu — WavLM-Large + ECAPA-TDNN implementation (Part 1)
//
// Implements: weight loading, scratch management, CNN feature extractor.
// See wavlm_ecapa_encoder.h for architecture overview.

#include "wavlm_ecapa_encoder.h"
#include "../communis/log.h"
#include "../machina/safetensors.h"

#include <cmath>
#include <cstdio>
#include <cassert>

namespace deusridet {

static constexpr int BLOCK = 256;
static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// CUDA Kernels — CNN Feature Extractor
// ============================================================================

// Layer normalization: x = gamma * (x - mean) / sqrt(var + eps) + beta
// Operates on last dimension: input [N, D], normalizes over D.
// One block per row (N), block-wide reduction for mean/var.
__global__ void layer_norm_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = input + row * D;
    float* y = output + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        sum += x[i];
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    // Block reduction via shared memory
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_mean = total / D;
    }
    __syncthreads();
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[i] - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_inv_std = rsqrtf(total / D + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = (x[i] - mean) * inv_std;
        if (gamma) val = val * gamma[i];
        if (beta) val = val + beta[i];
        y[i] = val;
    }
}

// Whole-sequence normalization for WavLM input (normalize=True):
// F.layer_norm(wav, wav.shape) — normalizes over all T samples.
// input [T], output [T]. Single block handles reduction.
__global__ void wav_layer_norm_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int T) {
    // Phase 1: compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x)
        sum += input[i];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_mean = total / T;
    }
    __syncthreads();
    float mean = s_mean;

    // Phase 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        float v = input[i] - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_inv_std = rsqrtf(total / T + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Phase 3: normalize
    for (int i = threadIdx.x; i < T; i += blockDim.x)
        output[i] = (input[i] - mean) * inv_std;
}

// FP32 → FP16 conversion kernel for GEMM inputs.
// Uses half2 for 2× throughput.
// Named with _wlecapa suffix to avoid ODR collision with machina/forward.cu.
__global__ void f32_to_f16_wlecapa(const float* __restrict__ in,
                                    __half* __restrict__ out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float a = in[idx], b = in[idx + 1];
        *reinterpret_cast<__half2*>(out + idx) = __floats2half2_rn(a, b);
    } else if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// Copy [C, T_src] → [C, T_dst] (drop trailing columns, for SamePad truncation)
__global__ void truncate_channels_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int C, int T_src, int T_dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T_dst) return;
    int c = idx / T_dst;
    int t = idx % T_dst;
    out[c * T_dst + t] = in[c * T_src + t];
}

// Conv1d: input [Cin, T_in], weight [Cout, Cin, K], bias optional
// output [Cout, T_out] where T_out = (T_in - K) / stride + 1
// No padding (WavLM CNN feature extractor has no padding).
// Kept as fallback; primary path uses cuDNN.
__global__ void conv1d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int Cin, int T_in, int Cout, int T_out,
                              int K, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * T_out) return;
    int co = idx / T_out;
    int to = idx % T_out;
    float sum = bias ? bias[co] : 0.0f;
    int t_start = to * stride;
    for (int ci = 0; ci < Cin; ++ci) {
        const float* w = weight + (co * Cin + ci) * K;
        const float* x = input + ci * T_in + t_start;
        for (int ki = 0; ki < K; ++ki)
            sum += w[ki] * x[ki];
    }
    output[idx] = sum;
}

// LayerNorm on [C, T] along C dimension (for each time step):
// WavLM extractor_mode="layer_norm" applies LN per time frame on 512 channels.
// The input is [C, T] (channels first from Conv1d), LN is over C per t.
// We treat each column t as a "row" of length C.
__global__ void layer_norm_channels_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            int C, int T) {
    int t = blockIdx.x;
    if (t >= T) return;

    // Compute mean over C channels at time t
    // input layout: [C, T], element [c, t] = input[c * T + t]
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        sum += input[c * T + t];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_mean = total / C;
    }
    __syncthreads();
    float mean = s_mean;

    // Variance
    float var_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = input[c * T + t] - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_inv_std = rsqrtf(total / C + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Normalize: same [C, T] layout
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = (input[c * T + t] - mean) * inv_std;
        if (gamma) val = val * gamma[c];
        if (beta) val = val + beta[c];
        output[c * T + t] = val;
    }
}

// GELU activation (exact, matches PyTorch GELU)
__global__ void gelu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float x = data[idx];
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    data[idx] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

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

// Forward declaration — defined later near transformer layer helpers.
static std::string enc_layer_key(int layer, const char* suffix);

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
__global__ void transpose_2d_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    output[n * M + m] = input[m * N + n];
}

// Bias add: output[row * cols + c] += bias[c]
__global__ void bias_add_kernel(float* __restrict__ data,
                                const float* __restrict__ bias,
                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = idx % cols;
    data[idx] += bias[c];
}

// Channel-first bias: data[ch * T + t] += bias[ch]
// For col-major GEMM output [T, C_out] or row-major [C_out, T]
__global__ void bias_add_channel_kernel(float* __restrict__ data,
                                         const float* __restrict__ bias,
                                         int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int ch = idx / T;
    data[idx] += bias[ch];
}

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

// ============================================================================
// Positional convolution: weight_norm Conv1d + SamePad + GELU
// ============================================================================

// Grouped Conv1d with padding: groups=16, kern=128, pad=64, stride=1
// Input: [C, T_in] (channels first), Output: [C, T_in] (after SamePad removes 1)
// Weight: [C, C/groups, K] precomputed from weight_norm(g, v)
__global__ void grouped_conv1d_padded_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int C, int T_in, int K, int pad, int groups) {
    // Output T = T_in + 2*pad - K + 1; SamePad removes 1 → T_out = T_in
    int T_padded = T_in + 2 * pad;
    int T_out = T_padded - K + 1 - 1;  // -1 for SamePad (even kernel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T_out) return;

    int co = idx / T_out;
    int to = idx % T_out;
    int group_size = C / groups;
    int g = co / group_size;
    int co_in_group = co % group_size;

    float sum = bias ? bias[co] : 0.0f;
    int ci_start = g * group_size;
    for (int ci_g = 0; ci_g < group_size; ci_g++) {
        int ci = ci_start + ci_g;
        const float* w_ptr = weight + (co * group_size + ci_g) * K;
        for (int ki = 0; ki < K; ki++) {
            int t_in_padded = to + ki;
            int t_in = t_in_padded - pad;
            float x_val = (t_in >= 0 && t_in < T_in) ? input[ci * T_in + t_in] : 0.0f;
            sum += w_ptr[ki] * x_val;
        }
    }
    output[idx] = sum;
}

// Compute effective pos_conv weight from weight_norm decomposition
// weight_g: [1, 1, K], weight_v: [C, C/groups, K]
// For each k: norm_k = ||v[:,:,k]||_2, weight[:,:,k] = g[k] * v[:,:,k] / norm_k
__global__ void weight_norm_kernel(const float* __restrict__ g,
                                    const float* __restrict__ v,
                                    float* __restrict__ weight,
                                    int total_per_k, int K) {
    // Each block handles one k
    int k = blockIdx.x;
    if (k >= K) return;

    // Compute L2 norm of v[:,:,k]
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < total_per_k; i += blockDim.x) {
        float val = v[i * K + k];
        sum_sq += val * val;
    }
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = sum_sq;
    __syncthreads();
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < (int)(blockDim.x / warpSize); i++) total += s_buf[i];
        s_scale = g[k] / (sqrtf(total) + 1e-12f);
    }
    __syncthreads();
    float scale = s_scale;

    // Apply scale
    for (int i = threadIdx.x; i < total_per_k; i += blockDim.x) {
        weight[i * K + k] = v[i * K + k] * scale;
    }
}

static const char* POS_CONV_G = "frontend.upstream.upstream.model.encoder.pos_conv.0.weight_g";
static const char* POS_CONV_V = "frontend.upstream.upstream.model.encoder.pos_conv.0.weight_v";
static const char* POS_CONV_B = "frontend.upstream.upstream.model.encoder.pos_conv.0.bias";

// skip_add + GELU: output = GELU(input + skip) (elementwise)
__global__ void skip_add_gelu_kernel(const float* __restrict__ input,
                                      const float* __restrict__ skip,
                                      float* __restrict__ output,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = input[idx] + skip[idx];
    output[idx] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

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
__global__ void compute_rel_pos_bias_kernel(
        const float* __restrict__ rel_attn_bias,  // [num_buckets, num_heads]
        float* __restrict__ output,                // [num_heads, T, T]
        int T, int num_heads, int num_buckets, int max_distance) {
    // Each thread handles one (head, qi, ki) element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * T * T;
    if (idx >= total) return;

    int h = idx / (T * T);
    int remainder = idx % (T * T);
    int qi = remainder / T;
    int ki = remainder % T;

    // Compute relative position bucket
    int rel_pos = ki - qi;  // positive = forward
    int num_buckets_half = num_buckets / 2;
    int bucket;

    // Bidirectional: first half buckets for negative, second half for positive
    if (rel_pos > 0) {
        bucket = num_buckets_half;
        int abs_pos = rel_pos;
        int max_exact = num_buckets_half / 2;
        if (abs_pos < max_exact) {
            bucket += abs_pos;
        } else {
            // Log-space bucketing
            float log_ratio = logf((float)abs_pos / (float)max_exact)
                            / logf((float)max_distance / (float)max_exact);
            bucket += max_exact + (int)(log_ratio * (num_buckets_half - max_exact));
            if (bucket >= num_buckets) bucket = num_buckets - 1;
        }
    } else {
        bucket = 0;
        int abs_pos = -rel_pos;
        int max_exact = num_buckets_half / 2;
        if (abs_pos < max_exact) {
            bucket += abs_pos;
        } else {
            float log_ratio = logf((float)abs_pos / (float)max_exact)
                            / logf((float)max_distance / (float)max_exact);
            bucket += max_exact + (int)(log_ratio * (num_buckets_half - max_exact));
            if (bucket >= num_buckets_half) bucket = num_buckets_half - 1;
        }
    }

    // Lookup: rel_attn_bias is [num_buckets, num_heads], row-major
    output[idx] = rel_attn_bias[bucket * num_heads + h];
}

// GRU-based relative position gating
// gate = sigmoid(Linear(mean_pool(attn_weights_per_head)))
// attn_weights *= gate.unsqueeze(-1)
// grep_linear: [8, 64] (2 * (num_heads/2) × head_dim)
// grep_a: [1, num_heads, 1, 1] → used as exp base
__global__ void gru_rel_pos_kernel(float* __restrict__ attn_weights,
                                    const float* __restrict__ q,
                                    const float* __restrict__ grep_linear_w,
                                    const float* __restrict__ grep_linear_b,
                                    const float* __restrict__ grep_a,
                                    int T, int num_heads, int head_dim) {
    // This kernel is a simplified version — computes GRU gating per head
    // For each head h, compute:
    //   q_mean = mean(q[:, h, :, :]) over T → [head_dim]
    //   gate_in = Linear(q_mean) → [2]  (for groups of heads)
    //   gate = sigmoid(gate_in[0]) * 2 (rel_pos_gate), sigmoid(gate_in[1]) * 2 (linear_pos_gate)
    // Then scale the relative position bias per head
    //
    // This is complex enough to warrant its own implementation.
    // For now, we implement it correctly in the host-side orchestration.
}

// Softmax along last dimension: input [rows, cols], in-place
__global__ void softmax_kernel(float* __restrict__ data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // Find max (numerical stability)
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_data[i];
        if (v > max_val) max_val = v;
    }
    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        if (other > max_val) max_val = other;
    }
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp_id] = max_val;
    __syncthreads();
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int i = 0; i < num_warps; i++)
            if (s_buf[i] > m) m = s_buf[i];
        s_max = m;
    }
    __syncthreads();
    max_val = s_max;

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __expf(row_data[i] - max_val);
        row_data[i] = v;
        sum += v;
    }
    // Warp + block reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < num_warps; i++) total += s_buf[i];
        s_sum = total;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_data[i] *= inv_sum;
}

// Reshape [T, H*Dh] (head-interleaved) → [H, T, Dh] (head-contiguous)
// Input[t][h*Dh + d] → Output[h][t][d]
__global__ void reshape_to_multihead_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int T, int H, int Dh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * Dh;
    if (idx >= total) return;
    int h = idx / (T * Dh);
    int td = idx % (T * Dh);
    int t = td / Dh;
    int d = td % Dh;
    output[idx] = input[t * H * Dh + h * Dh + d];
}

// Reshape [H, T, Dh] (head-contiguous) → [T, H*Dh] (head-interleaved)
// Input[h][t][d] → Output[t][h*Dh + d]
__global__ void reshape_from_multihead_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int T, int H, int Dh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * Dh;
    if (idx >= total) return;
    int h = idx / (T * Dh);
    int td = idx % (T * Dh);
    int t = td / Dh;
    int d = td % Dh;
    output[t * H * Dh + h * Dh + d] = input[idx];
}

// Split merged QKV [T, 3*D] → three [H, T, Dh] tensors in one kernel
// qkv[t, proj*D + h*Dh + d] → Q/K/V[h, t, d]  where proj ∈ {0,1,2}
__global__ void split_reshape_qkv_kernel(
        const float* __restrict__ qkv,
        float* __restrict__ Q,
        float* __restrict__ K,
        float* __restrict__ V,
        int T, int H, int Dh) {
    int D = H * Dh;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * Dh;
    if (idx >= total) return;
    int h = idx / (T * Dh);
    int t = (idx / Dh) % T;
    int d = idx % Dh;
    int src = t * 3 * D + h * Dh + d;
    Q[idx] = qkv[src];
    K[idx] = qkv[src + D];
    V[idx] = qkv[src + 2 * D];
}

// GRU relative position gating + position bias scaling
// For each (h, t): compute gate from Q vector, then scale position bias row
// Q: [H, T, Dh], grep_linear_w: [8, Dh], grep_linear_b: [8], grep_a: [H]
// pos_bias: [H, T, T], attn: [H, T, T] (output: attn += gate * pos_bias)
// Note: WavLM applies grep_linear to the un-projected LN output (reshaped as heads),
// NOT to the Q-projected output. The input `x_flat` is [T, D] where D = H*Dh.
__global__ void gru_gate_bias_kernel(
        const float* __restrict__ x_flat,  // [T, D] — LN output (NOT Q-projected)
        const float* __restrict__ grep_w,  // [8, Dh]
        const float* __restrict__ grep_b,  // [8]
        const float* __restrict__ grep_a,  // [H] (flattened from [1, H, 1, 1])
        const float* __restrict__ pos_bias, // [H, T, T]
        float* __restrict__ attn,          // [H, T, T] — add gated bias in-place
        int T, int H, int Dh) {
    // Each block handles one (h, t) pair
    int ht = blockIdx.x;
    if (ht >= H * T) return;
    int h = ht / T;
    int t = ht % T;

    // Step 1: Compute grep_linear(x[t, h*Dh : (h+1)*Dh]) → [8]
    // x_flat[T, D] layout: element [t][h*Dh + d] = x_flat[t*D + h*Dh + d]
    int D = H * Dh;
    const float* x_ptr = x_flat + t * D + h * Dh;  // [Dh]
    float linear_out[8];
    for (int o = 0; o < 8; o++) {
        float sum = grep_b[o];
        for (int d = 0; d < Dh; d++) {
            sum += grep_w[o * Dh + d] * x_ptr[d];
        }
        linear_out[o] = sum;
    }

    // Step 2: Reshape [8] → [2, 4], sum over last dim → [2], sigmoid
    float s0 = linear_out[0] + linear_out[1] + linear_out[2] + linear_out[3];
    float s1 = linear_out[4] + linear_out[5] + linear_out[6] + linear_out[7];
    float ga = 1.0f / (1.0f + __expf(-s0));  // sigmoid(s0)
    float gb = 1.0f / (1.0f + __expf(-s1));  // sigmoid(s1)

    // Step 3: Gate computation
    float a_h = grep_a[h];
    float gate = ga * (gb * a_h - 1.0f) + 2.0f;

    // Step 4: Scale position bias and add to attention scores
    // attn[h, t, :] += gate * pos_bias[h, t, :]
    float* attn_row = attn + h * T * T + t * T;
    const float* bias_row = pos_bias + h * T * T + t * T;
    for (int t2 = threadIdx.x; t2 < T; t2 += blockDim.x) {
        attn_row[t2] += gate * bias_row[t2];
    }
}

// Simple vector add: y[i] += x[i]
__global__ void vector_add_kernel(float* __restrict__ y,
                                   const float* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] += x[idx];
}

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

// ============================================================================
// Featurizer: softmax weighted sum of hidden states
// ============================================================================

// Softmax over a small 1D vector (25 elements) — single thread block
__global__ void softmax_1d_kernel(const float* __restrict__ raw_weights,
                                   float* __restrict__ norm_weights, int n) {
    float max_val = -1e30f;
    for (int i = 0; i < n; i++)
        if (raw_weights[i] > max_val) max_val = raw_weights[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = expf(raw_weights[i] - max_val);
        norm_weights[i] = v;
        sum += v;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++)
        norm_weights[i] *= inv;
}

// Weighted sum: output[t*D+d] = sum_l norm_weights[l] * hidden_states[l*T*D + t*D + d]
__global__ void weighted_sum_kernel(const float* __restrict__ hidden_states,
                                     const float* __restrict__ norm_weights,
                                     float* __restrict__ output,
                                     int num_layers, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D) return;
    float sum = 0.0f;
    for (int l = 0; l < num_layers; l++)
        sum += norm_weights[l] * hidden_states[l * T * D + idx];
    output[idx] = sum;
}

// ============================================================================
// UtteranceMVN: mean subtraction over time
// ============================================================================

// Compute mean per feature dim across T frames, then subtract
// Input/output: [T, D] row-major
__global__ void utterance_mvn_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int T, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float mean = 0.0f;
    for (int t = 0; t < T; t++)
        mean += input[t * D + d];
    mean /= T;
    for (int t = 0; t < T; t++)
        output[t * D + d] = input[t * D + d] - mean;
}

// ============================================================================
// ECAPA-TDNN kernels
// ============================================================================

// BatchNorm1d eval mode: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
// Input: [C, T] (channel-first, 1D) — one element per thread
__global__ void batch_norm_1d_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      const float* __restrict__ running_mean,
                                      const float* __restrict__ running_var,
                                      int C, int T, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    float x = input[idx];
    float inv_std = rsqrtf(running_var[c] + eps);
    output[idx] = (x - running_mean[c]) * inv_std * weight[c] + bias[c];
}

// ReLU in-place
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (data[idx] < 0.0f) data[idx] = 0.0f;
}

// Conv1d: [C_in, T] → [C_out, T_out], with kernel K, stride 1, padding P, dilation D
// Using im2col + cuBLAS GEMM
// For K=1: direct GEMM (y = W @ x), weight shape [C_out, C_in, 1] → [C_out, C_in]
// For K>1: im2col then GEMM

// im2col for 1D conv: extract [C_in * K, T_out] column matrix from [C_in, T] input
__global__ void im2col_1d_kernel(const float* __restrict__ input,
                                  float* __restrict__ cols,
                                  int C_in, int T_in, int K, int pad, int dilation,
                                  int T_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_in * K * T_out) return;
    int t_out = idx % T_out;
    int ck = idx / T_out;
    int c = ck / K;
    int k = ck % K;
    int t_in = t_out + k * dilation - pad;
    cols[idx] = (t_in >= 0 && t_in < T_in) ? input[c * T_in + t_in] : 0.0f;
}

// Sigmoid kernel
__global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = 1.0f / (1.0f + __expf(-data[idx]));
}

// Element-wise multiply: y[i] *= x[i] (for SE scaling with broadcast)
// x is [C, 1], y is [C, T] — broadcast multiply
__global__ void broadcast_mul_kernel(float* __restrict__ y,
                                      const float* __restrict__ x,
                                      int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    y[idx] *= x[c];
}

// Adaptive avg pool over last dimension: [C, T] → [C, 1]
__global__ void adaptive_avg_pool_1d_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++)
        sum += input[c * T + t];
    output[c] = sum / T;
}

// ============================================================================
// Pooling kernels
// ============================================================================

// Compute global_x = cat(x, mean(x).expand, std(x).expand) along channel dim
// x: [C, T], output: [3*C, T]
__global__ void pool_global_stats_kernel(const float* __restrict__ x,
                                          float* __restrict__ output,
                                          int C, int T) {
    int c = blockIdx.x;
    if (c >= C) return;
    // Compute mean and variance for this channel
    float sum = 0.0f, sum2 = 0.0f;
    for (int t = 0; t < T; t++) {
        float v = x[c * T + t];
        sum += v;
        sum2 += v * v;
    }
    float mean = sum / T;
    float var = sum2 / T - mean * mean;
    float std = sqrtf(fmaxf(var, 1.192092896e-07f));  // eps = FLT_EPSILON

    // Write to output: [x; mean_expanded; std_expanded]
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        output[c * T + t] = x[c * T + t];            // x itself
        output[(C + c) * T + t] = mean;               // mean repeated
        output[(2 * C + c) * T + t] = std;            // std repeated
    }
}

// Weighted stats: mu[c] = sum_t(x[c,t] * w[c,t]), sg[c] = sqrt(sum_t(x^2*w) - mu^2)
__global__ void weighted_stats_kernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       float* __restrict__ mu,
                                       float* __restrict__ sg,
                                       int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m = 0.0f, s2 = 0.0f;
    for (int t = 0; t < T; t++) {
        float xi = x[c * T + t];
        float wi = w[c * T + t];
        m += xi * wi;
        s2 += xi * xi * wi;
    }
    mu[c] = m;
    sg[c] = sqrtf(fmaxf(s2 - m * m, 1e-4f));
}

// L2 normalize: x[i] /= sqrt(sum(x[i]^2))
__global__ void l2_normalize_kernel(float* data, int n) {
    float sum2 = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum2 += data[i] * data[i];
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    // Cross-warp reduction via shared memory
    __shared__ float s_warp_sums[8];  // up to 256 threads = 8 warps
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) s_warp_sums[warp_id] = sum2;
    __syncthreads();
    // Thread 0 reduces all warp partial sums
    __shared__ float s_norm;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) total += s_warp_sums[w];
        s_norm = rsqrtf(total + 1e-12f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        data[i] *= s_norm;
}

// ============================================================================
// ECAPA-TDNN forward helpers
// ============================================================================

// Conv1d forward: input [C_in, T] → output [C_out, T_out]
// Weight: [C_out, C_in, K], Bias: [C_out]
void WavLMEcapaEncoder::forward_conv1d(const float* d_in, float* d_out,
                                         int C_in, int C_out, int T, int K,
                                         int pad, int dilation,
                                         const float* d_weight, const float* d_bias,
                                         const __half* d_weight_fp16) {
    int T_out = T;
    float alpha = 1.0f, beta = 0.0f;

    if (K == 1) {
        if (d_weight_fp16) {
            int in_count = C_in * T;
            int conv_blocks = (in_count / 2 + 255) / 256;
            f32_to_f16_wlecapa<<<conv_blocks, 256, 0, stream_>>>(d_in, d_gemm_a_, in_count);

            cublasGemmEx(cublas_,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         T, C_out, C_in,
                         &alpha,
                         d_gemm_a_, CUDA_R_16F, T,
                         d_weight_fp16, CUDA_R_16F, C_in,
                         &beta,
                         d_out, CUDA_R_32F, T,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                         T, C_out, C_in,
                         &alpha, d_in, T, d_weight, C_in,
                         &beta, d_out, T);
        }
    } else {
        int col_size = C_in * K * T_out;
        float* d_cols = d_im2col_;

        im2col_1d_kernel<<<div_ceil(col_size, BLOCK), BLOCK, 0, stream_>>>(
            d_in, d_cols, C_in, T, K, pad, dilation, T_out);

        if (d_weight_fp16) {
            int conv_blocks = (col_size / 2 + 255) / 256;
            f32_to_f16_wlecapa<<<conv_blocks, 256, 0, stream_>>>(d_cols, d_gemm_a_, col_size);

            cublasGemmEx(cublas_,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         T_out, C_out, C_in * K,
                         &alpha,
                         d_gemm_a_, CUDA_R_16F, T_out,
                         d_weight_fp16, CUDA_R_16F, C_in * K,
                         &beta,
                         d_out, CUDA_R_32F, T_out,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                         T_out, C_out, C_in * K,
                         &alpha, d_cols, T_out, d_weight, C_in * K,
                         &beta, d_out, T_out);
        }
    }

    if (d_bias) {
        bias_add_channel_kernel<<<div_ceil(C_out * T_out, BLOCK), BLOCK, 0, stream_>>>(
            d_out, d_bias, C_out, T_out);
    }
}

// cuDNN-accelerated Conv1d (as Conv2d with H=1).
// Handles groups, stride, padding, dilation. Used for CNN extractor + pos conv.
void WavLMEcapaEncoder::forward_conv1d_cudnn(
        const float* d_in, float* d_out,
        int C_in, int C_out, int T, int K,
        int stride, int pad, int groups, int dilation,
        const float* d_weight, const float* d_bias) {

    int T_out = (T + 2 * pad - dilation * (K - 1) - 1) / stride + 1;

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, C_in, 1, T);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, C_out, 1, T_out);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               C_out, C_in / groups, 1, K);
    cudnnSetConvolution2dDescriptor(conv_desc,
        0, pad,       // padH, padW
        1, stride,    // strideH, strideW
        1, dilation,  // dilationH, dilationW
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(conv_desc, groups);
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

    // Find best algorithm
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_, in_desc, filt_desc,
        conv_desc, out_desc, 1, &returned, &algo_perf);
    auto algo = algo_perf.algo;

    // Ensure workspace
    size_t ws_need = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_, in_desc, filt_desc,
        conv_desc, out_desc, algo, &ws_need);
    if (ws_need > cudnn_ws_size_) {
        if (d_cudnn_ws_) cudaFree(d_cudnn_ws_);
        cudaMalloc(&d_cudnn_ws_, ws_need);
        cudnn_ws_size_ = ws_need;
    }

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_, &alpha,
        in_desc, d_in, filt_desc, d_weight,
        conv_desc, algo, d_cudnn_ws_, ws_need,
        &beta, out_desc, d_out);

    if (d_bias) {
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, C_out, 1, 1);
        float ab = 1.0f, bb = 1.0f;
        cudnnAddTensor(cudnn_, &ab, bias_desc, d_bias, &bb, out_desc, d_out);
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// BatchNorm1d eval-mode forward
void WavLMEcapaEncoder::forward_batch_norm_1d(const float* d_in, float* d_out,
                                                int C, int T, const std::string& prefix) {
    auto& bn_w = w(prefix + ".weight");
    auto& bn_b = w(prefix + ".bias");
    auto& bn_rm = w(prefix + ".running_mean");
    auto& bn_rv = w(prefix + ".running_var");
    batch_norm_1d_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(
        d_in, d_out, bn_w.ptr, bn_b.ptr, bn_rm.ptr, bn_rv.ptr, C, T, 1e-5f);
}

// ECAPA SE block: AdaptiveAvgPool → Conv1d(C→bottleneck, K=1) → ReLU → BN → Conv1d(bottleneck→C, K=1) → Sigmoid
// Modifies d_x in-place: d_x *= sigmoid(fc2(bn(relu(fc1(avg_pool(d_x))))))
void WavLMEcapaEncoder::forward_se_block(float* d_x, int C, int T,
                                           const std::string& prefix) {
    int bottleneck = 128;
    // Reuse scratch_b_ tail for SE intermediates
    // Need [C] for pool, [bottleneck] for fc1, [bottleneck] for bn, [C] for fc2
    float* d_pool = scratch_b_ + scratch_max_T_ * WavLMConfig::embed_dim;  // [C]
    float* d_se1 = d_pool + C;          // [bottleneck]
    float* d_se2 = d_se1 + bottleneck;  // [C]

    // AdaptiveAvgPool1d: [C, T] → [C, 1]
    adaptive_avg_pool_1d_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream_>>>(
        d_x, d_pool, C, T);

    // Conv1d(C→128, K=1) = linear: d_se1 = W @ d_pool + bias
    // W: [128, C, 1], pool is [C, 1], output [128, 1]
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                 1, bottleneck, C,
                 &alpha,
                 d_pool, 1,
                 w(prefix + ".1.weight").ptr, C,
                 &beta,
                 d_se1, 1);
    // Add bias: [bottleneck, 1] channel-first
    bias_add_channel_kernel<<<div_ceil(bottleneck, BLOCK), BLOCK, 0, stream_>>>(
        d_se1, w(prefix + ".1.bias").ptr, bottleneck, 1);

    // ReLU
    relu_kernel<<<div_ceil(bottleneck, BLOCK), BLOCK, 0, stream_>>>(d_se1, bottleneck);

    // BN(128)
    // For [128, 1] — just apply element-wise
    batch_norm_1d_kernel<<<div_ceil(bottleneck, BLOCK), BLOCK, 0, stream_>>>(
        d_se1, d_se1, w(prefix + ".3.weight").ptr, w(prefix + ".3.bias").ptr,
        w(prefix + ".3.running_mean").ptr, w(prefix + ".3.running_var").ptr,
        bottleneck, 1, 1e-5f);

    // Conv1d(128→C, K=1) = linear
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                 1, C, bottleneck,
                 &alpha,
                 d_se1, 1,
                 w(prefix + ".4.weight").ptr, bottleneck,
                 &beta,
                 d_se2, 1);
    bias_add_channel_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream_>>>(
        d_se2, w(prefix + ".4.bias").ptr, C, 1);

    // Sigmoid
    sigmoid_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream_>>>(d_se2, C);

    // Multiply: d_x[c, t] *= d_se2[c]
    broadcast_mul_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_x, d_se2, C, T);
}

// ECAPA Res2Net block
void WavLMEcapaEncoder::forward_ecapa_block(float* d_x, int C, int T,
                                              int dilation, const std::string& prefix) {
    int width = 128;  // C / model_scale = 1024 / 8
    int nums = 7;     // model_scale - 1

    // d_x is [C, T] channel-first
    // Need temp buffers for conv1 output, res2net splits, conv3 output
    // Use scratch_b_ for intermediate (scratch_a_ has the residual input)
    float* d_conv1 = scratch_b_;  // [C, T]

    // conv1: Conv1d(C→C, K=1) + ReLU + BN
    forward_conv1d(d_x, d_conv1, C, C, T, 1, 0, 1,
                   w(prefix + ".conv1.weight").ptr, w(prefix + ".conv1.bias").ptr,
                   w(prefix + ".conv1.weight").fp16);
    relu_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_conv1, C * T);
    forward_batch_norm_1d(d_conv1, d_conv1, C, T, prefix + ".bn1");

    // Res2Net: split d_conv1 [C=1024, T] into 8 groups of [128, T]
    // Process groups: sp starts as group[0], then for i=1..6: sp = conv(sp + group[i])
    // Accumulate outputs: out = cat(processed_groups)
    float* d_out = scratch_c_;  // [C, T] for the concatenated output
    // d_sp buffer for the cascading intermediate
    float* d_sp = d_out + C * T;  // [width, T]

    for (int i = 0; i < nums; i++) {
        float* group_i = d_conv1 + i * width * T;  // pointer into d_conv1
        if (i == 0) {
            // sp = spx[0] — just copy
            cudaMemcpyAsync(d_sp, group_i, width * T * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream_);
        } else {
            // sp = sp + spx[i]
            vector_add_kernel<<<div_ceil(width * T, BLOCK), BLOCK, 0, stream_>>>(
                d_sp, group_i, width * T);
        }

        // Dilated conv: Conv1d(128→128, K=3, pad=dilation, dil=dilation) + ReLU + BN
        float* d_conv_out = d_out + i * width * T;  // write directly to output slot
        forward_conv1d(d_sp, d_conv_out, width, width, T, 3, dilation, dilation,
                       w(prefix + ".convs." + std::to_string(i) + ".weight").ptr,
                       w(prefix + ".convs." + std::to_string(i) + ".bias").ptr,
                       w(prefix + ".convs." + std::to_string(i) + ".weight").fp16);
        relu_kernel<<<div_ceil(width * T, BLOCK), BLOCK, 0, stream_>>>(d_conv_out, width * T);
        forward_batch_norm_1d(d_conv_out, d_conv_out, width, T,
                              prefix + ".bns." + std::to_string(i));

        // Update sp for next iteration (conv output is the new sp)
        cudaMemcpyAsync(d_sp, d_conv_out, width * T * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // Append last group (spx[nums] = group[7]) — passes through unchanged
    float* group_last = d_conv1 + nums * width * T;
    cudaMemcpyAsync(d_out + nums * width * T, group_last, width * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // conv3: Conv1d(C→C, K=1) + ReLU + BN
    float* d_conv3 = scratch_b_;  // reuse
    forward_conv1d(d_out, d_conv3, C, C, T, 1, 0, 1,
                   w(prefix + ".conv3.weight").ptr, w(prefix + ".conv3.bias").ptr,
                   w(prefix + ".conv3.weight").fp16);
    relu_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_conv3, C * T);
    forward_batch_norm_1d(d_conv3, d_conv3, C, T, prefix + ".bn3");

    // SE block
    forward_se_block(d_conv3, C, T, prefix + ".se.se");

    // Residual add: d_conv3 += d_x (original input)
    vector_add_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_conv3, d_x, C * T);

    // Copy result back to d_x
    cudaMemcpyAsync(d_x, d_conv3, C * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
}

// ============================================================================
// Full extract pipeline
// ============================================================================

std::vector<float> WavLMEcapaEncoder::extract(const float* pcm, int n_samples) {
    if (!initialized_) return {};
    ensure_scratch(n_samples);

    // Grow pre-allocated PCM buffer if needed.
    size_t need = (size_t)n_samples;
    if (need > pcm_buf_size_) {
        if (d_pcm_buf_) cudaFree(d_pcm_buf_);
        pcm_buf_size_ = std::max(need, (size_t)(16000 * 10));  // min 10s @ 16kHz
        cudaMalloc(&d_pcm_buf_, pcm_buf_size_ * sizeof(float));
    }
    cudaMemcpyAsync(d_pcm_buf_, pcm, n_samples * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    return extract_gpu(d_pcm_buf_, n_samples);
}

std::vector<float> WavLMEcapaEncoder::extract_gpu(const float* d_pcm, int n_samples) {
    if (!initialized_) return {};
    ensure_scratch(n_samples);

    // Timing events for latency breakdown.
    cudaEvent_t ev_start, ev_cnn, ev_enc, ev_done;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_cnn);
    cudaEventCreate(&ev_enc);
    cudaEventCreate(&ev_done);
    cudaEventRecord(ev_start, stream_);

    int D = WavLMConfig::embed_dim;
    int num_hs = WavLMConfig::num_hidden_states;  // 25

    // ── 1. WavLM CNN + Projection + PosConv + Encoder ──
    int T;
    test_cnn(d_pcm, n_samples, T);
    cudaEventRecord(ev_cnn, stream_);
    test_projection(scratch_a_, T, T);
    test_pos_conv(scratch_a_, T, T);
    test_encoder(scratch_a_, T, T);  // populates d_hidden_states_
    cudaEventRecord(ev_enc, stream_);

    // ── 2. Featurizer: softmax weighted sum of 25 hidden states ──
    float* d_nw = scratch_b_;  // [25] — normalized weights
    softmax_1d_kernel<<<1, 1, 0, stream_>>>(
        w("frontend.featurizer.weights").ptr, d_nw, num_hs);

    float* d_feat = scratch_a_;  // [T, D] — featurizer output
    weighted_sum_kernel<<<div_ceil(T * D, BLOCK), BLOCK, 0, stream_>>>(
        d_hidden_states_, d_nw, d_feat, num_hs, T, D);

    // ── 3. UtteranceMVN: subtract mean per feature dim ──
    float* d_mvn = scratch_b_;  // [T, D]
    utterance_mvn_kernel<<<div_ceil(D, BLOCK), BLOCK, 0, stream_>>>(
        d_feat, d_mvn, T, D);

    // ── 4. ECAPA-TDNN ──
    // Need channel-first layout: [1024, T] from [T, 1024]
    // d_mvn is [T, D] row-major. Transpose to [D, T] = [1024, T]
    float* d_ecapa = scratch_a_;  // [1024, T] channel-first
    transpose_2d_kernel<<<div_ceil(T * D, BLOCK), BLOCK, 0, stream_>>>(
        d_mvn, d_ecapa, T, D);

    // Initial conv + ReLU + BN: [1024, T] → [1024, T]
    float* d_xe = scratch_c_;  // [1024, T]
    forward_conv1d(d_ecapa, d_xe, 1024, 1024, T, 5, 2, 1,
                   w("encoder.conv.weight").ptr, w("encoder.conv.bias").ptr,
                   w("encoder.conv.weight").fp16);
    relu_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(d_xe, 1024 * T);
    forward_batch_norm_1d(d_xe, d_xe, 1024, T, "encoder.bn");

    // Three ECAPA blocks with different dilations
    // We need to keep x_e, x1, x2, x3 for the skip connections
    // x_e is in scratch_c_ [1024, T]
    // For each block, the input must be in a buffer that won't be overwritten
    // Block input: scratch_a_ = working buffer, scratch_b_/scratch_c_ = intermediates

    // Save x_e to a safe location
    float* d_xe_saved = d_hidden_states_;  // Reuse hidden states buffer temporarily (25 * T * D is big enough)
    cudaMemcpyAsync(d_xe_saved, d_xe, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // layer1: input = x_e, output = x1
    cudaMemcpyAsync(scratch_a_, d_xe, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    forward_ecapa_block(scratch_a_, 1024, T, 2, "encoder.layer1");
    // scratch_a_ now has x1; save it
    float* d_x1 = d_xe_saved + 1024 * T;
    cudaMemcpyAsync(d_x1, scratch_a_, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // layer2: input = x_e + x1
    // scratch_a_ = x_e + x1
    cudaMemcpyAsync(scratch_a_, d_xe_saved, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    vector_add_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, d_x1, 1024 * T);
    forward_ecapa_block(scratch_a_, 1024, T, 3, "encoder.layer2");
    float* d_x2 = d_x1 + 1024 * T;
    cudaMemcpyAsync(d_x2, scratch_a_, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // layer3: input = x_e + x1 + x2
    cudaMemcpyAsync(scratch_a_, d_xe_saved, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    vector_add_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, d_x1, 1024 * T);
    vector_add_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, d_x2, 1024 * T);
    forward_ecapa_block(scratch_a_, 1024, T, 4, "encoder.layer3");
    float* d_x3 = d_x2 + 1024 * T;
    cudaMemcpyAsync(d_x3, scratch_a_, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // Concatenate x1, x2, x3 → [3072, T]
    // They're already contiguous in memory: d_x1, d_x2, d_x3
    float* d_cat = d_x1;  // [3072, T] starting at d_x1

    // layer4: Conv1d(3072→1536, K=1) + ReLU
    float* d_x4 = scratch_a_;  // [1536, T]
    forward_conv1d(d_cat, d_x4, 3072, 1536, T, 1, 0, 1,
                   w("encoder.layer4.weight").ptr, w("encoder.layer4.bias").ptr,
                   w("encoder.layer4.weight").fp16);
    relu_kernel<<<div_ceil(1536 * T, BLOCK), BLOCK, 0, stream_>>>(d_x4, 1536 * T);

    // ── 5. Channel Attention Stat Pooling ──
    // global_x = cat(x4, mean(x4).expand, std(x4).expand) → [4608, T]
    int C_pool = 1536;
    float* d_global = scratch_b_;  // [4608, T]
    pool_global_stats_kernel<<<C_pool, std::min(T, 256), 0, stream_>>>(
        d_x4, d_global, C_pool, T);

    // Attention: Conv1d(4608→128, K=1) → ReLU → BN → Conv1d(128→1536, K=1) → Softmax
    float* d_attn_w = scratch_c_;  // [128, T]
    forward_conv1d(d_global, d_attn_w, 4608, 128, T, 1, 0, 1,
                   w("pooling.attention.0.weight").ptr, w("pooling.attention.0.bias").ptr,
                   w("pooling.attention.0.weight").fp16);
    relu_kernel<<<div_ceil(128 * T, BLOCK), BLOCK, 0, stream_>>>(d_attn_w, 128 * T);
    forward_batch_norm_1d(d_attn_w, d_attn_w, 128, T, "pooling.attention.2");

    float* d_attn_out = d_attn_w + 128 * T;  // [1536, T]
    forward_conv1d(d_attn_w, d_attn_out, 128, 1536, T, 1, 0, 1,
                   w("pooling.attention.3.weight").ptr, w("pooling.attention.3.bias").ptr,
                   w("pooling.attention.3.weight").fp16);

    // Softmax along T dimension (dim=2 in [1, 1536, T])
    // Each channel's T values get softmaxed independently
    int sm_threads_pool = ((std::min(T, 256) + 31) / 32) * 32;
    softmax_kernel<<<C_pool, sm_threads_pool, 0, stream_>>>(
        d_attn_out, C_pool, T);

    // Weighted stats: mu, sg
    float* d_mu = scratch_c_;         // [1536]
    float* d_sg = d_mu + C_pool;      // [1536]
    weighted_stats_kernel<<<div_ceil(C_pool, BLOCK), BLOCK, 0, stream_>>>(
        d_x4, d_attn_out, d_mu, d_sg, C_pool, T);

    // cat(mu, sg) → [3072]
    float* d_pool_out = d_mu;  // mu and sg are contiguous → [3072]

    // ── 6. Projector: BN(3072) → Linear(3072→192) ──
    // BN on [3072, 1] (treating as [C=3072, T=1])
    float* d_bn = scratch_b_;  // [3072]
    batch_norm_1d_kernel<<<div_ceil(3072, BLOCK), BLOCK, 0, stream_>>>(
        d_pool_out, d_bn,
        w("projector.bn.weight").ptr, w("projector.bn.bias").ptr,
        w("projector.bn.running_mean").ptr, w("projector.bn.running_var").ptr,
        3072, 1, 1e-5f);

    // Linear(3072→192): d_bn [1, 3072] → d_fc [1, 192]
    float* d_fc = d_bn + 3072;  // [192]
    forward_linear(d_bn, d_fc, 1, 3072, 192,
                   w("projector.fc.weight").ptr, w("projector.fc.bias").ptr,
                   w("projector.fc.weight").fp16);

    // L2 normalize
    l2_normalize_kernel<<<1, 192, 0, stream_>>>(d_fc, 192);

    cudaEventRecord(ev_done, stream_);
    cudaStreamSynchronize(stream_);

    // Compute latency breakdown.
    cudaEventElapsedTime(&last_lat_cnn_ms_,     ev_start, ev_cnn);
    cudaEventElapsedTime(&last_lat_encoder_ms_, ev_cnn,   ev_enc);
    cudaEventElapsedTime(&last_lat_ecapa_ms_,   ev_enc,   ev_done);
    cudaEventElapsedTime(&last_lat_total_ms_,   ev_start, ev_done);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_cnn);
    cudaEventDestroy(ev_enc);
    cudaEventDestroy(ev_done);

    // Copy to host
    std::vector<float> result(192);
    cudaMemcpy(result.data(), d_fc, 192 * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

} // namespace deusridet
