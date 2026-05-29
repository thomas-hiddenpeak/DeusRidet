/**
 * @file diarizen_wavlm_pruned_forward.cu
 * @philosophical_role P1a-step2 of the DiariZen native CUDA port: the
 *     forward pass of the WavLM-pruned encoder. Compute belongs on the
 *     GPU (philosophy.instructions.md). This TU is a peer of
 *     diarizen_wavlm_pruned.cu (loader) and reaches into the same class;
 *     it is split out to respect the .cu hard size limit and to keep the
 *     loader path free of cuDNN/cuBLAS state.
 * @serves DiarizenWavlmPruned. Step 2a implements the CNN feature
 *     extractor only (bit-checked against the `cnn_out` reference tap);
 *     feature_projection / pos_conv / 24 transformer layers / weight_sum
 *     / proj / lnorm land in 2b..2e.
 *
 * Stage 2a — CNN feature extractor (matches diarizen
 * wav2vec2/components.py FeatureExtractor + ConvLayerBlock, extractor_mode
 * = "layer_norm"):
 *   for each of 7 pruned conv layers:
 *     x = conv1d(x, weight, stride, no-bias, no-pad)   # [C_out, T']
 *     x = layer_norm_over_channels(x)                  # per-frame, eps 1e-5
 *     x = gelu(x)                                      # exact (erf) form
 *   x = x^T                                            # [T, C]
 *   x = x * dummy_weight[C]                            # per-channel scale
 */
#include "diarizen_wavlm_pruned.h"

#include "../communis/log.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace deusridet {
namespace orator {

namespace {

constexpr const char* kLog = "DiariZenWavlm";
constexpr int kBlock = 256;

inline int div_ceil_(int a, int b) { return (a + b - 1) / b; }

inline bool cuda_ck_(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        LOG_ERROR(kLog, "CUDA %s failed: %s", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// fp16 -> fp32 element-wise (weights live in the fp16 arena; cuDNN/cuBLAS
// run the CNN in fp32 to match the reference activation precision).
__global__ void half_to_float_kernel(const __half* __restrict__ src,
                                      float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// Single-block reduction of sum and sum-of-squares over the full waveform.
// out[0] = sum, out[1] = sum of squares. Launch with one block.
__global__ void waveform_stats_kernel(const float* __restrict__ x, int n,
                                      float* __restrict__ out) {
    __shared__ float s_sum[kBlock];
    __shared__ float s_sq[kBlock];
    float ls = 0.0f, lq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        ls += v;
        lq += v * v;
    }
    s_sum[threadIdx.x] = ls;
    s_sq[threadIdx.x] = lq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = s_sum[0];
        out[1] = s_sq[0];
    }
}

// Normalize the waveform in place: (x - mean) / sqrt(var_biased + 1e-5),
// matching torch F.layer_norm over the time dimension (WavLM input front
// end). stats[0] = sum, stats[1] = sum of squares.
__global__ void normalize_waveform_kernel(float* __restrict__ x, int n,
                                          const float* __restrict__ stats) {
    float mean = stats[0] / n;
    float var = stats[1] / n - mean * mean;
    float inv = rsqrtf(var + 1e-5f);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (x[i] - mean) * inv;
}

// Per-frame LayerNorm over the channel dimension, operating in place on a
// channel-major [C, T] buffer. One block per frame t; threads stride over
// channels reading data[c * T + t]. eps = 1e-5 (torch default).
__global__ void layer_norm_channels_kernel(float* __restrict__ data,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           int C, int T) {
    int t = blockIdx.x;
    if (t >= T) return;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        sum += data[c * T + t];
    for (int o = warpSize / 2; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, o);

    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp] = sum;
    __syncthreads();
    __shared__ float s_mean, s_inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nw; ++i) tot += s_buf[i];
        s_mean = tot / C;
    }
    __syncthreads();
    float mean = s_mean;

    float vs = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = data[c * T + t] - mean;
        vs += d * d;
    }
    for (int o = warpSize / 2; o > 0; o >>= 1)
        vs += __shfl_down_sync(0xffffffff, vs, o);
    if (lane == 0) s_buf[warp] = vs;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nw; ++i) tot += s_buf[i];
        s_inv = rsqrtf(tot / C + 1e-5f);
    }
    __syncthreads();
    float inv = s_inv;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = (data[c * T + t] - mean) * inv;
        data[c * T + t] = v * gamma[c] + beta[c];
    }
}

// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). Matches torch
// nn.functional.gelu default (not the tanh approximation).
__global__ void gelu_exact_kernel(float* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

// Transpose channel-major [C, T] -> frame-major [T, C] and apply the
// per-channel dummy_weight scale in the same pass.
__global__ void transpose_scale_kernel(const float* __restrict__ src,  // [C,T]
                                       float* __restrict__ dst,         // [T,C]
                                       const float* __restrict__ scale, // [C]
                                       int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx >= total) return;
    int c = idx / T;
    int t = idx % T;
    dst[t * C + c] = src[c * T + t] * scale[c];
}

}  // namespace

// --------------------------------------------------------------------------
// Handle management
// --------------------------------------------------------------------------
bool DiarizenWavlmPruned::ensure_handles_() {
    if (cudnn_) return true;
    cudnnHandle_t h = nullptr;
    if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) {
        LOG_ERROR(kLog, "cudnnCreate failed");
        return false;
    }
    cudnn_ = h;
    return true;
}

// --------------------------------------------------------------------------
// CNN feature extractor (P1a-step2a)
// --------------------------------------------------------------------------
std::vector<float>
DiarizenWavlmPruned::debug_cnn_features(const float* pcm, int n_samples,
                                        int& T_out) {
    T_out = 0;
    if (!loaded_) {
        LOG_ERROR(kLog, "debug_cnn_features called before load()");
        return {};
    }
    if (!ensure_handles_()) return {};
    auto cudnn = static_cast<cudnnHandle_t>(cudnn_);

    constexpr int kCnn = DiarizenWavlmPrunedArch::kCnnLayers;
    const int kernels[kCnn] = {10, 3, 3, 3, 3, 2, 2};
    const int strides[kCnn] = {5, 2, 2, 2, 2, 2, 2};

    // Resolve per-layer channel widths + weight/LN views from the arena.
    int cin[kCnn], cout[kCnn];
    const DiarizenWavlmPrunedTensorView* w_conv[kCnn];
    const DiarizenWavlmPrunedTensorView* w_lng[kCnn];
    const DiarizenWavlmPrunedTensorView* w_lnb[kCnn];
    for (int i = 0; i < kCnn; ++i) {
        char key[128];
        std::snprintf(key, sizeof(key),
                      "wavlm_model.feature_extractor.conv_layers.%d.conv.weight",
                      i);
        w_conv[i] = find(key);
        std::snprintf(key, sizeof(key),
                      "wavlm_model.feature_extractor.conv_layers.%d.layer_norm.weight",
                      i);
        w_lng[i] = find(key);
        std::snprintf(key, sizeof(key),
                      "wavlm_model.feature_extractor.conv_layers.%d.layer_norm.bias",
                      i);
        w_lnb[i] = find(key);
        if (!w_conv[i] || !w_lng[i] || !w_lnb[i]) {
            LOG_ERROR(kLog, "missing CNN tensor for conv layer %d", i);
            return {};
        }
        // conv.weight shape [C_out, C_in, K]
        cout[i] = w_conv[i]->shape[0];
        cin[i]  = w_conv[i]->shape[1];
    }
    const auto* dummy = find("wavlm_model.feature_extractor.dummy_weight");
    if (!dummy) {
        LOG_ERROR(kLog, "missing feature_extractor.dummy_weight");
        return {};
    }

    // ---- Allocate GPU scratch ------------------------------------------
    // Two ping-pong activation buffers sized for the largest conv output
    // (layer 0: 512 x ~51199). Plus an fp32 weight staging buffer sized
    // for the largest conv weight.
    auto fail = [&](const char* what, void* a, void* b, void* c, void* d,
                    void* e) -> std::vector<float> {
        LOG_ERROR(kLog, "debug_cnn_features: %s", what);
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (c) cudaFree(c);
        if (d) cudaFree(d);
        if (e) cudaFree(e);
        return {};
    };

    // Compute conv output lengths.
    int T = n_samples;
    int t_len[kCnn];
    long max_act = 0;
    size_t max_w = 0;
    int tt = T;
    for (int i = 0; i < kCnn; ++i) {
        tt = (tt - (kernels[i] - 1) - 1) / strides[i] + 1;
        t_len[i] = tt;
        long act = (long)cout[i] * tt;
        if (act > max_act) max_act = act;
        size_t wn = (size_t)cout[i] * cin[i] * kernels[i];
        if (wn > max_w) max_w = wn;
    }
    const int T_final = t_len[kCnn - 1];

    float *d_pcm = nullptr, *d_a = nullptr, *d_b = nullptr, *d_w = nullptr,
          *d_out = nullptr;
    if (!cuda_ck_(cudaMalloc(&d_pcm, (size_t)n_samples * sizeof(float)),
                  "malloc pcm"))
        return {};
    if (!cuda_ck_(cudaMalloc(&d_a, (size_t)max_act * sizeof(float)), "malloc a"))
        return fail("malloc a", d_pcm, nullptr, nullptr, nullptr, nullptr);
    if (!cuda_ck_(cudaMalloc(&d_b, (size_t)max_act * sizeof(float)), "malloc b"))
        return fail("malloc b", d_pcm, d_a, nullptr, nullptr, nullptr);
    if (!cuda_ck_(cudaMalloc(&d_w, max_w * sizeof(float)), "malloc w"))
        return fail("malloc w", d_pcm, d_a, d_b, nullptr, nullptr);
    if (!cuda_ck_(cudaMalloc(&d_out, (size_t)T_final * cout[kCnn - 1] *
                                         sizeof(float)),
                  "malloc out"))
        return fail("malloc out", d_pcm, d_a, d_b, d_w, nullptr);

    if (!cuda_ck_(cudaMemcpy(d_pcm, pcm, (size_t)n_samples * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "memcpy pcm"))
        return fail("memcpy pcm", d_pcm, d_a, d_b, d_w, d_out);

    // WavLM input front end: per-window layer-norm of the raw waveform
    // (zero mean, unit variance, eps 1e-5). The CNN reference is computed
    // on this normalised signal, not the raw PCM.
    {
        float* d_stats = nullptr;
        if (!cuda_ck_(cudaMalloc(&d_stats, 2 * sizeof(float)), "malloc stats"))
            return fail("malloc stats", d_pcm, d_a, d_b, d_w, d_out);
        waveform_stats_kernel<<<1, kBlock>>>(d_pcm, n_samples, d_stats);
        normalize_waveform_kernel<<<div_ceil_(n_samples, kBlock), kBlock>>>(
            d_pcm, n_samples, d_stats);
        cudaFree(d_stats);
    }

    // cuDNN scratch descriptors reused across layers.
    cudnnTensorDescriptor_t in_d, out_d;
    cudnnFilterDescriptor_t filt_d;
    cudnnConvolutionDescriptor_t conv_d;
    cudnnCreateTensorDescriptor(&in_d);
    cudnnCreateTensorDescriptor(&out_d);
    cudnnCreateFilterDescriptor(&filt_d);
    cudnnCreateConvolutionDescriptor(&conv_d);

    // Input starts as [C_in=1, T] in d_pcm; treat d_pcm as the first "act".
    const float* cur = d_pcm;
    int cur_T = n_samples;
    float* dst_act = d_a;     // first conv writes here

    for (int i = 0; i < kCnn; ++i) {
        const int Ci = cin[i], Co = cout[i], K = kernels[i], S = strides[i];
        const int To = t_len[i];

        // Stage conv weight to fp32.
        const int wn = Co * Ci * K;
        half_to_float_kernel<<<div_ceil_(wn, kBlock), kBlock>>>(
            w_conv[i]->data, d_w, wn);

        cudnnSetTensor4dDescriptor(in_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, Ci, 1, cur_T);
        cudnnSetTensor4dDescriptor(out_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, Co, 1, To);
        cudnnSetFilter4dDescriptor(filt_d, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                   Co, Ci, 1, K);
        cudnnSetConvolution2dDescriptor(conv_d, 0, 0, 1, S, 1, 1,
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT);

        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t perf;
        cudnnGetConvolutionForwardAlgorithm_v7(cudnn, in_d, filt_d, conv_d,
                                               out_d, 1, &returned, &perf);
        cudnnConvolutionFwdAlgo_t algo = perf.algo;

        size_t ws = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_d, filt_d, conv_d,
                                                out_d, algo, &ws);
        if (ws > cudnn_ws_bytes_) {
            if (cudnn_ws_) cudaFree(cudnn_ws_);
            if (!cuda_ck_(cudaMalloc(&cudnn_ws_, ws), "malloc conv ws")) {
                cudnnDestroyTensorDescriptor(in_d);
                cudnnDestroyTensorDescriptor(out_d);
                cudnnDestroyFilterDescriptor(filt_d);
                cudnnDestroyConvolutionDescriptor(conv_d);
                return fail("malloc conv ws", d_pcm, d_a, d_b, d_w, d_out);
            }
            cudnn_ws_bytes_ = ws;
        }

        float alpha = 1.0f, beta = 0.0f;
        cudnnStatus_t cs = cudnnConvolutionForward(
            cudnn, &alpha, in_d, cur, filt_d, d_w, conv_d, algo, cudnn_ws_,
            cudnn_ws_bytes_, &beta, out_d, dst_act);
        if (cs != CUDNN_STATUS_SUCCESS) {
            LOG_ERROR(kLog, "conv layer %d failed: %s", i,
                      cudnnGetErrorString(cs));
            cudnnDestroyTensorDescriptor(in_d);
            cudnnDestroyTensorDescriptor(out_d);
            cudnnDestroyFilterDescriptor(filt_d);
            cudnnDestroyConvolutionDescriptor(conv_d);
            return fail("conv forward", d_pcm, d_a, d_b, d_w, d_out);
        }

        // LayerNorm over channels (in place) using fp32 gamma/beta staged
        // into the unused part of spare (small: Co each).
        // Stage LN params to fp32 in d_w tail region is risky; allocate a
        // tiny local fp32 buffer instead.
        float* ln_gb = nullptr;
        cudaMalloc(&ln_gb, (size_t)2 * Co * sizeof(float));
        half_to_float_kernel<<<div_ceil_(Co, kBlock), kBlock>>>(
            w_lng[i]->data, ln_gb, Co);
        half_to_float_kernel<<<div_ceil_(Co, kBlock), kBlock>>>(
            w_lnb[i]->data, ln_gb + Co, Co);
        layer_norm_channels_kernel<<<To, kBlock>>>(dst_act, ln_gb, ln_gb + Co,
                                                   Co, To);
        gelu_exact_kernel<<<div_ceil_(Co * To, kBlock), kBlock>>>(dst_act,
                                                                  Co * To);
        cudaFree(ln_gb);

        // Advance: output becomes next input. Ping-pong a<->b; the very
        // first input (d_pcm) is never overwritten.
        cur = dst_act;
        cur_T = To;
        float* next = (dst_act == d_a) ? d_b : d_a;
        dst_act = next;
    }

    cudnnDestroyTensorDescriptor(in_d);
    cudnnDestroyTensorDescriptor(out_d);
    cudnnDestroyFilterDescriptor(filt_d);
    cudnnDestroyConvolutionDescriptor(conv_d);

    // `cur` now points at [C=211, T_final] channel-major. Transpose to
    // [T_final, 211] and apply dummy_weight in one pass.
    const int Cf = cout[kCnn - 1];
    float* d_dummy = nullptr;
    cudaMalloc(&d_dummy, (size_t)Cf * sizeof(float));
    half_to_float_kernel<<<div_ceil_(Cf, kBlock), kBlock>>>(dummy->data,
                                                            d_dummy, Cf);
    transpose_scale_kernel<<<div_ceil_(Cf * T_final, kBlock), kBlock>>>(
        cur, d_out, d_dummy, Cf, T_final);
    cudaFree(d_dummy);

    if (!cuda_ck_(cudaDeviceSynchronize(), "cnn sync"))
        return fail("sync", d_pcm, d_a, d_b, d_w, d_out);

    std::vector<float> host((size_t)T_final * Cf);
    cudaMemcpy(host.data(), d_out, host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_pcm);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_w);
    cudaFree(d_out);

    T_out = T_final;
    return host;
}

}  // namespace orator
}  // namespace deusridet
