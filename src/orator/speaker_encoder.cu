// speaker_encoder.cu — CAM++ GPU Speaker Encoder implementation.
//
// Adapted from qwen35-orin speaker_encoder_gpu.cu (Thomas Zhu)
// Original: CAM++ architecture from FunASR (MIT License)
//
// Architecture: FCM(ResNet) → TDNN → 3×CAMDenseTDNNBlock → StatsPool → Dense → L2
// All computation on GPU: cuBLAS SGEMM + custom CUDA kernels.
// Key: pre-allocated ScratchPool with ping-pong concat buffers.

#include "speaker_encoder.h"
#include "../communis/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace deusridet {

// ============================================================================
// CUDA Kernels
// ============================================================================

static constexpr int BLOCK = 256;
static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// BN + ReLU fused
__global__ void spk_bn_relu_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ var,
                                    int C, int spatial, bool do_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    float inv_std = rsqrtf(var[c] + 1e-5f);
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;
    float val = g * (input[idx] - mean[c]) * inv_std + b;
    output[idx] = do_relu ? fmaxf(val, 0.0f) : val;
}

__global__ void spk_relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = fmaxf(data[idx], 0.0f);
}

// Conv2d: [Cin, H, W] → [Cout, H', W']
__global__ void spk_conv2d_kernel(const float* __restrict__ input,
                                   const float* __restrict__ weight,
                                   float* __restrict__ output,
                                   int Cin, int H, int W,
                                   int Cout, int H_out, int W_out,
                                   int k, int stride_h, int stride_w,
                                   int pad_h, int pad_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * H_out * W_out) return;
    int co = idx / (H_out * W_out);
    int rem = idx % (H_out * W_out);
    int ho = rem / W_out;
    int wo = rem % W_out;
    float sum = 0;
    for (int ci = 0; ci < Cin; ++ci)
        for (int kh = 0; kh < k; ++kh)
            for (int kw = 0; kw < k; ++kw) {
                int hi = ho * stride_h - pad_h + kh;
                int wi = wo * stride_w - pad_w + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W)
                    sum += weight[co * Cin * k * k + ci * k * k + kh * k + kw]
                         * input[ci * H * W + hi * W + wi];
            }
    output[idx] = sum;
}

// Conv1d with dilation
__global__ void spk_conv1d_kernel(const float* __restrict__ input,
                                   const float* __restrict__ weight,
                                   const float* __restrict__ bias,
                                   float* __restrict__ output,
                                   int Cin, int T, int Cout, int T_out,
                                   int k, int stride, int pad, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * T_out) return;
    int co = idx / T_out;
    int to = idx % T_out;
    float sum = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < Cin; ++ci)
        for (int ki = 0; ki < k; ++ki) {
            int ti = to * stride - pad + ki * dilation;
            if (ti >= 0 && ti < T)
                sum += weight[co * Cin * k + ci * k + ki] * input[ci * T + ti];
        }
    output[idx] = sum;
}

__global__ void spk_add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

// Segment pooling: each timestep gets average of its segment
__global__ void spk_seg_pool_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int C, int T, int seg_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    int t = idx % T;
    int seg_start = (t / seg_len) * seg_len;
    int seg_end = min(seg_start + seg_len, T);
    float sum = 0;
    for (int i = seg_start; i < seg_end; ++i) sum += input[c * T + i];
    output[idx] = sum / (seg_end - seg_start);
}

// Context = global_mean + seg_pool
__global__ void spk_context_kernel(const float* __restrict__ input,
                                    const float* __restrict__ seg_pool,
                                    float* __restrict__ context,
                                    int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    float global_mean = 0;
    for (int t = 0; t < T; ++t) global_mean += input[c * T + t];
    global_mean /= T;
    context[idx] = global_mean + seg_pool[idx];
}

// sigmoid(gate) * output
__global__ void spk_sigmoid_mul_kernel(float* __restrict__ output,
                                        const float* __restrict__ gate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] *= 1.0f / (1.0f + __expf(-gate[idx]));
}

// StatsPool: [C, T] → [2*C] (mean + std)
__global__ void spk_stats_pool_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0;
    for (int t = 0; t < T; ++t) sum += input[c * T + t];
    float mean = sum / T;
    float var_sum = 0;
    for (int t = 0; t < T; ++t) {
        float diff = input[c * T + t] - mean;
        var_sum += diff * diff;
    }
    output[c] = mean;
    output[C + c] = sqrtf(var_sum / max(1, T - 1) + 1e-2f);
}

// BN without affine (for final embedding)
__global__ void spk_bn_no_affine_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         const float* __restrict__ mean,
                                         const float* __restrict__ var, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    output[c] = (input[c] - mean[c]) * rsqrtf(var[c] + 1e-5f);
}

__global__ void spk_l2_normalize_kernel(float* data, int C) {
    float norm = 0;
    for (int i = 0; i < C; ++i) norm += data[i] * data[i];
    norm = rsqrtf(norm + 1e-12f);
    for (int i = 0; i < C; ++i) data[i] *= norm;
}

__global__ void spk_add_bias_relu_kernel(float* data,
                                          const float* __restrict__ bias,
                                          int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    data[idx] = fmaxf(data[idx] + (bias ? bias[c] : 0.0f), 0.0f);
}

__global__ void spk_add_bias_kernel(float* data,
                                     const float* __restrict__ bias,
                                     int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    data[idx] += bias ? bias[c] : 0.0f;
}

__global__ void spk_copy_rows_kernel(float* dst, const float* src,
                                      int C, int T, int dst_offset_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    dst[dst_offset_channels * T + idx] = src[idx];
}

// CMN + Transpose: [T, F] → [F, T] with per-feature mean subtraction.
// CAM++ extract_feature() applies CMN: feature - feature.mean(dim=0).
// Each block handles one frequency bin f: compute mean over T, subtract, transpose.
__global__ void spk_cmn_transpose_kernel(const float* __restrict__ mel,
                                          float* __restrict__ out,
                                          int T, int F) {
    int f = blockIdx.x;
    if (f >= F) return;

    // Phase 1: compute mean of mel[:, f] via shared memory reduction.
    extern __shared__ float smem[];
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x)
        local_sum += mel[t * F + f];
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = smem[0] / (float)T;

    // Phase 2: subtract mean and transpose.
    for (int t = threadIdx.x; t < T; t += blockDim.x)
        out[f * T + t] = mel[t * F + f] - mean;
}

// ============================================================================
// ScratchPool
// ============================================================================

bool SpeakerScratchPool::alloc(int max_T, int max_spatial) {
    size_t a_sz = max_spatial;
    size_t b_sz = max_spatial;
    size_t fcm_scratch = (size_t)32 * 40 * max_T;
    size_t c_sz = std::max((size_t)128 * max_T, fcm_scratch);
    size_t d_sz = std::max((size_t)128 * max_T, fcm_scratch);
    size_t e_sz = 128 * max_T;
    size_t f_sz = 64  * max_T;
    size_t cat_sz = 1024 * max_T;
    total_bytes = (a_sz + b_sz + c_sz + d_sz + e_sz + f_sz + 2 * cat_sz) * sizeof(float);

    float* base = nullptr;
    if (cudaMalloc(&base, total_bytes) != cudaSuccess) return false;
    cudaMemset(base, 0, total_bytes);
    size_t off = 0;
    a = base + off; off += a_sz;
    b = base + off; off += b_sz;
    c = base + off; off += c_sz;
    d = base + off; off += d_sz;
    e = base + off; off += e_sz;
    f = base + off; off += f_sz;
    concat[0] = base + off; off += cat_sz;
    concat[1] = base + off;
    which_concat = 0;
    return true;
}

void SpeakerScratchPool::free() {
    if (a) { cudaFree(a); a = nullptr; }
    b = c = d = e = f = nullptr;
    concat[0] = concat[1] = nullptr;
    total_bytes = 0;
}

// ============================================================================
// SpeakerEncoder lifecycle
// ============================================================================

SpeakerEncoder::SpeakerEncoder() = default;

SpeakerEncoder::~SpeakerEncoder() {
    scratch_.free();
    if (stream_) cudaStreamDestroy(stream_);
    if (cublas_) cublasDestroy(cublas_);
    for (auto& kv : gpu_tensors_) {
        if (kv.second) cudaFree(kv.second);
    }
}

bool SpeakerEncoder::init(const SpeakerEncoderConfig& cfg) {
    cfg_ = cfg;
    auto cpu_tensors = load_safetensors(cfg_.model_path);
    if (cpu_tensors.empty()) {
        LOG_ERROR("Speaker", "Failed to load safetensors: %s", cfg_.model_path.c_str());
        return false;
    }

    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("Speaker", "cuBLAS create failed");
        return false;
    }

    size_t total_bytes = 0;
    for (auto& kv : cpu_tensors) {
        float* d_ptr = nullptr;
        size_t bytes = kv.second.size() * sizeof(float);
        if (cudaMalloc(&d_ptr, bytes) != cudaSuccess) {
            LOG_ERROR("Speaker", "cudaMalloc failed for %s (%zu B)", kv.first.c_str(), bytes);
            return false;
        }
        cudaMemcpy(d_ptr, kv.second.data(), bytes, cudaMemcpyHostToDevice);
        gpu_tensors_[kv.first] = d_ptr;
        tensor_sizes_[kv.first] = (int)kv.second.size();
        total_bytes += bytes;
    }

    cudaStreamCreate(&stream_);
    ensure_scratch(1000);  // pre-allocate for typical segments

    initialized_ = true;
    LOG_INFO("Speaker", "CAM++ loaded: %zu tensors, %.1f MB weights, %.1f MB scratch",
             gpu_tensors_.size(), total_bytes / (1024.0f * 1024.0f),
             scratch_.total_bytes / (1024.0f * 1024.0f));
    return true;
}

bool SpeakerEncoder::ensure_scratch(int T) {
    if (T <= scratch_max_T_) return true;
    scratch_.free();
    int T2 = (T + 2 * 2 - 1 * (5 - 1) - 1) / 2 + 1;
    int max_fcm = 32 * 80 * T;
    int max_block = 1024 * T2;
    int max_spatial = std::max(max_fcm, max_block);
    if (!scratch_.alloc(std::max(T, T2), max_spatial)) {
        LOG_ERROR("Speaker", "scratch alloc failed for T=%d", T);
        scratch_max_T_ = 0;
        return false;
    }
    scratch_max_T_ = T;
    return true;
}

const float* SpeakerEncoder::get_gpu(const std::string& name) const {
    auto it = gpu_tensors_.find(name);
    return (it != gpu_tensors_.end()) ? it->second : nullptr;
}

float SpeakerEncoder::cosine_similarity(const std::vector<float>& a,
                                         const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

// ============================================================================
// FCM ResBlock
// ============================================================================

void SpeakerEncoder::gpu_res_block(const float* d_input, float* d_output,
                                    int C, int H, int W,
                                    const std::string& prefix, int stride,
                                    float* scratch_a, float* scratch_b,
                                    cudaStream_t stream) {
    int pad = 1;
    int H2 = (H + 2 * pad - 3) / stride + 1;
    int conv_size = C * H2 * W;

    spk_conv2d_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        d_input, get_gpu(prefix + ".conv1.weight"), scratch_a,
        C, H, W, C, H2, W, 3, stride, 1, 1, 1);
    spk_bn_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        scratch_a, scratch_a,
        get_gpu(prefix + ".bn1.weight"), get_gpu(prefix + ".bn1.bias"),
        get_gpu(prefix + ".bn1.running_mean"), get_gpu(prefix + ".bn1.running_var"),
        C, H2 * W, true);

    spk_conv2d_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        scratch_a, get_gpu(prefix + ".conv2.weight"), scratch_b,
        C, H2, W, C, H2, W, 3, 1, 1, 1, 1);
    spk_bn_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        scratch_b, scratch_b,
        get_gpu(prefix + ".bn2.weight"), get_gpu(prefix + ".bn2.bias"),
        get_gpu(prefix + ".bn2.running_mean"), get_gpu(prefix + ".bn2.running_var"),
        C, H2 * W, false);

    if (stride != 1) {
        spk_conv2d_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
            d_input, get_gpu(prefix + ".shortcut.0.weight"), d_output,
            C, H, W, C, H2, W, 1, stride, 1, 0, 0);
        spk_bn_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
            d_output, d_output,
            get_gpu(prefix + ".shortcut.1.weight"), get_gpu(prefix + ".shortcut.1.bias"),
            get_gpu(prefix + ".shortcut.1.running_mean"), get_gpu(prefix + ".shortcut.1.running_var"),
            C, H2 * W, false);
        spk_add_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(d_output, scratch_b, conv_size);
    } else {
        cudaMemcpyAsync(d_output, d_input, conv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        spk_add_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(d_output, scratch_b, conv_size);
    }
    spk_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(d_output, conv_size);
}

// ============================================================================
// CAM Dense TDNN Block
// ============================================================================

void SpeakerEncoder::gpu_cam_dense_block(SpeakerScratchPool& sp, int in_dim,
                                          int T, const std::string& prefix,
                                          int num_layers, int dilation,
                                          cublasHandle_t cublas,
                                          cudaStream_t stream) {
    const int growth = 32, bn_ch = 128, k = 3;
    int pad = (k - 1) / 2 * dilation;
    int cur_dim = in_dim;

    for (int l = 0; l < num_layers; ++l) {
        std::string lp = prefix + ".tdnnd" + std::to_string(l + 1);

        // BN + ReLU on concat → sp.a
        spk_bn_relu_kernel<<<div_ceil(cur_dim * T, BLOCK), BLOCK, 0, stream>>>(
            sp.cur_concat(), sp.a,
            get_gpu(lp + ".nonlinear1.batchnorm.weight"),
            get_gpu(lp + ".nonlinear1.batchnorm.bias"),
            get_gpu(lp + ".nonlinear1.batchnorm.running_mean"),
            get_gpu(lp + ".nonlinear1.batchnorm.running_var"),
            cur_dim, T, true);

        // linear1: Conv1d(cur_dim→128, k=1) = GEMM → sp.b
        {
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        T, bn_ch, cur_dim,
                        &alpha, sp.a, T,
                        get_gpu(lp + ".linear1.weight"), cur_dim,
                        &beta, sp.b, T);
        }

        // BN + ReLU on sp.b
        spk_bn_relu_kernel<<<div_ceil(bn_ch * T, BLOCK), BLOCK, 0, stream>>>(
            sp.b, sp.b,
            get_gpu(lp + ".nonlinear2.batchnorm.weight"),
            get_gpu(lp + ".nonlinear2.batchnorm.bias"),
            get_gpu(lp + ".nonlinear2.batchnorm.running_mean"),
            get_gpu(lp + ".nonlinear2.batchnorm.running_var"),
            bn_ch, T, true);

        // CAM layer → sp.c [growth, T]
        gpu_cam_layer(sp, bn_ch, growth, T, lp + ".cam_layer", k, dilation, pad, cublas, stream);

        // Append to concat: copy old concat + new growth
        cudaMemcpyAsync(sp.next_concat(), sp.cur_concat(),
                        cur_dim * T * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        spk_copy_rows_kernel<<<div_ceil(growth * T, BLOCK), BLOCK, 0, stream>>>(
            sp.next_concat(), sp.c, growth, T, cur_dim);
        sp.swap_concat();
        cur_dim += growth;
    }
}

// ============================================================================
// CAM Layer
// ============================================================================

void SpeakerEncoder::gpu_cam_layer(SpeakerScratchPool& sp, int bn_ch, int out_ch,
                                    int T, const std::string& prefix,
                                    int k, int dilation, int padding,
                                    cublasHandle_t cublas,
                                    cudaStream_t stream) {
    // local: Conv1d(128→32, k=3, dilation) → sp.c
    int local_size = out_ch * T;
    spk_conv1d_kernel<<<div_ceil(local_size, BLOCK), BLOCK, 0, stream>>>(
        sp.b, get_gpu(prefix + ".linear_local.weight"),
        get_gpu(prefix + ".linear_local.bias"),
        sp.c, bn_ch, T, out_ch, T, k, 1, padding, dilation);

    // Segment pooling → sp.d
    int ctx_size = bn_ch * T;
    spk_seg_pool_kernel<<<div_ceil(ctx_size, BLOCK), BLOCK, 0, stream>>>(
        sp.b, sp.d, bn_ch, T, 100);

    // Context = global_mean + seg_pool → sp.e
    spk_context_kernel<<<div_ceil(ctx_size, BLOCK), BLOCK, 0, stream>>>(
        sp.b, sp.d, sp.e, bn_ch, T);

    // linear1: GEMM(128→64) → sp.d
    int mid = bn_ch / 2;
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    T, mid, bn_ch,
                    &alpha, sp.e, T,
                    get_gpu(prefix + ".linear1.weight"), bn_ch,
                    &beta, sp.d, T);
    }
    spk_add_bias_relu_kernel<<<div_ceil(mid * T, BLOCK), BLOCK, 0, stream>>>(
        sp.d, get_gpu(prefix + ".linear1.bias"), mid, T);

    // linear2: GEMM(64→32) → sp.e
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    T, out_ch, mid,
                    &alpha, sp.d, T,
                    get_gpu(prefix + ".linear2.weight"), mid,
                    &beta, sp.e, T);
    }
    spk_add_bias_kernel<<<div_ceil(out_ch * T, BLOCK), BLOCK, 0, stream>>>(
        sp.e, get_gpu(prefix + ".linear2.bias"), out_ch, T);

    // gate × local
    spk_sigmoid_mul_kernel<<<div_ceil(local_size, BLOCK), BLOCK, 0, stream>>>(
        sp.c, sp.e, local_size);
}

// ============================================================================
// Transit Layer
// ============================================================================

void SpeakerEncoder::gpu_transit(SpeakerScratchPool& sp, int in_dim, int T,
                                  const std::string& prefix, int out_dim,
                                  cublasHandle_t cublas,
                                  cudaStream_t stream) {
    spk_bn_relu_kernel<<<div_ceil(in_dim * T, BLOCK), BLOCK, 0, stream>>>(
        sp.cur_concat(), sp.a,
        get_gpu(prefix + ".nonlinear.batchnorm.weight"),
        get_gpu(prefix + ".nonlinear.batchnorm.bias"),
        get_gpu(prefix + ".nonlinear.batchnorm.running_mean"),
        get_gpu(prefix + ".nonlinear.batchnorm.running_var"),
        in_dim, T, true);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                T, out_dim, in_dim,
                &alpha, sp.a, T,
                get_gpu(prefix + ".linear.weight"), in_dim,
                &beta, sp.next_concat(), T);
    sp.swap_concat();
}

// ============================================================================
// Core forward pass
// ============================================================================

void SpeakerEncoder::forward_one(const float* d_mel, int T,
                                  SpeakerScratchPool& sp,
                                  cudaStream_t stream, cublasHandle_t cublas,
                                  float* d_emb_out) {
    int T2 = (T + 2 * 2 - 1 * (5 - 1) - 1) / 2 + 1;
    sp.which_concat = 0;

    // CMN + Transpose: [T, 80] → [80, T] (utterance-level mean subtraction)
    float* d_x = sp.a;
    int cmn_smem = 256 * sizeof(float);
    spk_cmn_transpose_kernel<<<80, 256, cmn_smem, stream>>>(d_mel, d_x, T, 80);

    // ======================== FCM ========================
    int H = 80;
    int conv1_size = 32 * H * T;
    float* d_fcm = sp.b;
    spk_conv2d_kernel<<<div_ceil(conv1_size, BLOCK), BLOCK, 0, stream>>>(
        d_x, get_gpu("head.conv1.weight"), d_fcm,
        1, H, T, 32, H, T, 3, 1, 1, 1, 1);
    spk_bn_relu_kernel<<<div_ceil(conv1_size, BLOCK), BLOCK, 0, stream>>>(
        d_fcm, d_fcm,
        get_gpu("head.bn1.weight"), get_gpu("head.bn1.bias"),
        get_gpu("head.bn1.running_mean"), get_gpu("head.bn1.running_var"),
        32, H * T, true);

    // ResBlocks
    gpu_res_block(d_fcm, d_x, 32, H, T, "head.layer1.0", 2, sp.c, sp.d, stream);
    H = (H + 2 - 3) / 2 + 1;  // 40
    gpu_res_block(d_x, d_fcm, 32, H, T, "head.layer1.1", 1, sp.c, sp.d, stream);
    gpu_res_block(d_fcm, d_x, 32, H, T, "head.layer2.0", 2, sp.c, sp.d, stream);
    H = (H + 2 - 3) / 2 + 1;  // 20
    gpu_res_block(d_x, d_fcm, 32, H, T, "head.layer2.1", 1, sp.c, sp.d, stream);

    int H2 = (H + 2 - 3) / 2 + 1;  // 10
    int conv2_size = 32 * H2 * T;
    spk_conv2d_kernel<<<div_ceil(conv2_size, BLOCK), BLOCK, 0, stream>>>(
        d_fcm, get_gpu("head.conv2.weight"), d_x,
        32, H, T, 32, H2, T, 3, 2, 1, 1, 1);
    spk_bn_relu_kernel<<<div_ceil(conv2_size, BLOCK), BLOCK, 0, stream>>>(
        d_x, d_x,
        get_gpu("head.bn2.weight"), get_gpu("head.bn2.bias"),
        get_gpu("head.bn2.running_mean"), get_gpu("head.bn2.running_var"),
        32, H2 * T, true);

    // ======================== TDNN ========================
    int tdnn_size = 128 * T2;
    float* d_tdnn = sp.b;
    spk_conv1d_kernel<<<div_ceil(tdnn_size, BLOCK), BLOCK, 0, stream>>>(
        d_x, get_gpu("xvector.tdnn.linear.weight"),
        get_gpu("xvector.tdnn.linear.bias"),
        d_tdnn, 32 * H2, T, 128, T2, 5, 2, 2, 1);
    spk_bn_relu_kernel<<<div_ceil(tdnn_size, BLOCK), BLOCK, 0, stream>>>(
        d_tdnn, d_tdnn,
        get_gpu("xvector.tdnn.nonlinear.batchnorm.weight"),
        get_gpu("xvector.tdnn.nonlinear.batchnorm.bias"),
        get_gpu("xvector.tdnn.nonlinear.batchnorm.running_mean"),
        get_gpu("xvector.tdnn.nonlinear.batchnorm.running_var"),
        128, T2, true);

    cudaMemcpyAsync(sp.cur_concat(), d_tdnn, 128 * T2 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // ======================== DenseTDNN Blocks ========================
    int cur_dim = 128;
    gpu_cam_dense_block(sp, cur_dim, T2, "xvector.block1", 12, 1, cublas, stream);
    cur_dim += 12 * 32;  // 512
    gpu_transit(sp, cur_dim, T2, "xvector.transit1", cur_dim / 2, cublas, stream);
    cur_dim /= 2;  // 256

    gpu_cam_dense_block(sp, cur_dim, T2, "xvector.block2", 24, 2, cublas, stream);
    cur_dim += 24 * 32;  // 1024
    gpu_transit(sp, cur_dim, T2, "xvector.transit2", cur_dim / 2, cublas, stream);
    cur_dim /= 2;  // 512

    gpu_cam_dense_block(sp, cur_dim, T2, "xvector.block3", 16, 2, cublas, stream);
    cur_dim += 16 * 32;  // 1024
    gpu_transit(sp, cur_dim, T2, "xvector.transit3", cur_dim / 2, cublas, stream);
    cur_dim /= 2;  // 512

    // ======================== Out BN + ReLU ========================
    int embed_channels = cur_dim;
    float* d_final = sp.cur_concat();
    spk_bn_relu_kernel<<<div_ceil(embed_channels * T2, BLOCK), BLOCK, 0, stream>>>(
        d_final, d_final,
        get_gpu("xvector.out_nonlinear.batchnorm.weight"),
        get_gpu("xvector.out_nonlinear.batchnorm.bias"),
        get_gpu("xvector.out_nonlinear.batchnorm.running_mean"),
        get_gpu("xvector.out_nonlinear.batchnorm.running_var"),
        embed_channels, T2, true);

    // ======================== StatsPool ========================
    float* d_pooled = sp.a;
    spk_stats_pool_kernel<<<div_ceil(embed_channels, BLOCK), BLOCK, 0, stream>>>(
        d_final, d_pooled, embed_channels, T2);

    // ======================== Dense → BN → L2 ========================
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cublas, CUBLAS_OP_T, embed_channels * 2, 192,
                    &alpha, get_gpu("xvector.dense.linear.weight"),
                    embed_channels * 2, d_pooled, 1, &beta, d_emb_out, 1);
    }
    spk_bn_no_affine_kernel<<<1, BLOCK, 0, stream>>>(
        d_emb_out, d_emb_out,
        get_gpu("xvector.dense.nonlinear.batchnorm.running_mean"),
        get_gpu("xvector.dense.nonlinear.batchnorm.running_var"), 192);
    spk_l2_normalize_kernel<<<1, 1, 0, stream>>>(d_emb_out, 192);
}

// ============================================================================
// Public extract methods
// ============================================================================

std::vector<float> SpeakerEncoder::extract_gpu(const float* d_mel, int T) {
    if (!initialized_ || T < 10) return {};
    if (!ensure_scratch(T)) return {};
    cublasSetStream(cublas_, stream_);

    float* d_emb = scratch_.b;
    forward_one(d_mel, T, scratch_, stream_, cublas_, d_emb);

    cudaStreamSynchronize(stream_);
    std::vector<float> result(192);
    cudaMemcpy(result.data(), d_emb, 192 * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

std::vector<float> SpeakerEncoder::extract(const float* mel, int T) {
    if (!initialized_ || T < 10) return {};
    if (!ensure_scratch(T)) return {};
    cublasSetStream(cublas_, stream_);

    // Upload mel to GPU
    float* d_mel = scratch_.a;
    cudaMemcpyAsync(d_mel, mel, T * 80 * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // CMN + Transpose done inside forward_one
    float* d_emb = scratch_.b;
    forward_one(d_mel, T, scratch_, stream_, cublas_, d_emb);

    cudaStreamSynchronize(stream_);
    std::vector<float> result(192);
    cudaMemcpy(result.data(), d_emb, 192 * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// ============================================================================
// Safetensors loader
// ============================================================================

SpeakerEncoder::TensorMap SpeakerEncoder::load_safetensors(const std::string& path) {
    TensorMap result;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        LOG_ERROR("Speaker", "Cannot open: %s", path.c_str());
        return result;
    }

    uint64_t header_size = 0;
    ifs.read(reinterpret_cast<char*>(&header_size), 8);
    if (header_size > 10000000) {
        LOG_ERROR("Speaker", "Header too large: %lu", header_size);
        return result;
    }

    std::string header(header_size, '\0');
    ifs.read(&header[0], header_size);

    size_t data_base = 8 + header_size;
    size_t pos = 0;

    while (pos < header.size()) {
        size_t ks = header.find('"', pos);
        if (ks == std::string::npos) break;
        size_t ke = header.find('"', ks + 1);
        if (ke == std::string::npos) break;
        std::string key = header.substr(ks + 1, ke - ks - 1);
        pos = ke + 1;

        if (key == "__metadata__") {
            size_t brace = header.find('{', pos);
            if (brace != std::string::npos) {
                int depth = 1; pos = brace + 1;
                while (pos < header.size() && depth > 0) {
                    if (header[pos] == '{') depth++;
                    else if (header[pos] == '}') depth--;
                    pos++;
                }
            }
            continue;
        }

        size_t op = header.find("data_offsets", pos);
        if (op == std::string::npos) break;
        size_t br = header.find('[', op);
        size_t cm = header.find(',', br);
        size_t eb = header.find(']', cm);

        uint64_t start = std::stoull(header.substr(br + 1, cm - br - 1));
        uint64_t end   = std::stoull(header.substr(cm + 1, eb - cm - 1));
        size_t num_floats = (end - start) / sizeof(float);

        std::vector<float> data(num_floats);
        ifs.seekg(data_base + start);
        ifs.read(reinterpret_cast<char*>(data.data()), end - start);
        result[key] = std::move(data);
        pos = eb + 1;
    }

    LOG_INFO("Speaker", "Loaded %zu tensors from %s", result.size(), path.c_str());
    return result;
}

} // namespace deusridet
