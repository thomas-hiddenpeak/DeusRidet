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
#include <initializer_list>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "diarizen_wavlm_pruned_kernels.cuh"

namespace deusridet {
namespace orator {

namespace {

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
// channel-major [C, T] buffer. One thread per frame t; the thread reduces
// serially over channels reading data[c * T + t]. Because neighbouring threads
// own neighbouring frames, every load data[c*T + (t0..t0+31)] is a contiguous
// 32-float run -> fully coalesced. (The earlier one-block-per-frame version
// strided each warp by T and was the pipeline's top kernel at 20%.) eps = 1e-5.
__global__ void layer_norm_channels_kernel(float* __restrict__ data,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           int C, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) sum += data[(long)c * T + t];
    float mean = sum / C;

    float vs = 0.0f;
    for (int c = 0; c < C; ++c) {
        float d = data[(long)c * T + t] - mean;
        vs += d * d;
    }
    float inv = rsqrtf(vs / C + 1e-5f);

    for (int c = 0; c < C; ++c) {
        long idx = (long)c * T + t;
        data[idx] = (data[idx] - mean) * inv * gamma[c] + beta[c];
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

// Transpose frame-major [T, C] -> channel-major [C, T].
__global__ void transpose_TC_to_CT_kernel(const float* __restrict__ src, // [T,C]
                                          float* __restrict__ dst,        // [C,T]
                                          int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * C;
    if (idx >= total) return;
    int t = idx / C;
    int c = idx % C;
    dst[c * T + t] = src[t * C + c];
}

// weight_norm reconstruction (dim=2): per kernel-tap k, compute the L2 norm
// of v[:, :, k] over the (out, in) axes. v is [O, I, K] contiguous.
__global__ void posconv_norm_kernel(const float* __restrict__ v,
                                    int O, int I, int K,
                                    float* __restrict__ norm) {  // [K]
    int k = blockIdx.x;
    if (k >= K) return;
    float acc = 0.0f;
    int OI = O * I;
    for (int oi = threadIdx.x; oi < OI; oi += blockDim.x) {
        float val = v[(size_t)oi * K + k];
        acc += val * val;
    }
    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    __shared__ float s_buf[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) s_buf[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nw; ++i) tot += s_buf[i];
        norm[k] = sqrtf(tot);
    }
}

// W[o, i, k] = g[k] * v[o, i, k] / norm[k]. g is the weight_norm scale
// vector of shape [1, 1, K]; v is [O, I, K]; output W matches v's layout.
__global__ void posconv_weight_kernel(const float* __restrict__ v,
                                      const float* __restrict__ g,    // [K]
                                      const float* __restrict__ norm, // [K]
                                      int O, int I, int K,
                                      float* __restrict__ W) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)O * I * K;
    if (idx >= total) return;
    int k = idx % K;
    W[idx] = g[k] * v[idx] / norm[k];
}

// Add a per-channel bias to a channel-major [C, T] buffer in place.
__global__ void bias_add_channels_kernel(float* __restrict__ data, // [C,T]
                                         const float* __restrict__ bias, // [C]
                                         int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx >= total) return;
    data[idx] += bias[idx / T];
}

// Transpose channel-major [C, src_stride] -> frame-major [T, C] and add into
// an existing frame-major [T, C] residual buffer (out += src^T), reading only
// the first T of src_stride columns (used to trim pos_conv's SamePad frame).
__global__ void transpose_CT_to_TC_add_kernel(const float* __restrict__ src, // [C,src_stride]
                                              float* __restrict__ out,        // [T,C]
                                              int C, int T, int src_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx >= total) return;
    int c = idx / T;
    int t = idx % T;
    out[(size_t)t * C + c] += src[(size_t)c * src_stride + t];
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
    cudnnSetStream(h, stream_);  // route conv forward onto the bound stream
    return true;
}

void DiarizenWavlmPruned::set_stream(cudaStream_t s) {
    stream_ = s;
    if (cudnn_) cudnnSetStream(static_cast<cudnnHandle_t>(cudnn_), s);
}

// --------------------------------------------------------------------------
// CNN feature extractor (P1a-step2a) — internal, returns a GPU buffer
// --------------------------------------------------------------------------
float* DiarizenWavlmPruned::run_cnn_(const float* pcm, int n_samples,
                                     int& T_out) {
    T_out = 0;
    if (!loaded_) {
        LOG_ERROR(kLog, "run_cnn_ called before load()");
        return nullptr;
    }
    if (!ensure_handles_()) return nullptr;
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
            return nullptr;
        }
        // conv.weight shape [C_out, C_in, K]
        cout[i] = w_conv[i]->shape[0];
        cin[i]  = w_conv[i]->shape[1];
    }
    const auto* dummy = find("wavlm_model.feature_extractor.dummy_weight");
    if (!dummy) {
        LOG_ERROR(kLog, "missing feature_extractor.dummy_weight");
        return nullptr;
    }

    // ---- Allocate GPU scratch ------------------------------------------
    // Two ping-pong activation buffers sized for the largest conv output
    // (layer 0: 512 x ~51199). Plus an fp32 weight staging buffer sized
    // for the largest conv weight.
    auto fail = [&](const char* what, void* a, void* b, void* c, void* d,
                    void* e) -> float* {
        LOG_ERROR(kLog, "run_cnn_: %s", what);
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (c) cudaFree(c);
        if (d) cudaFree(d);
        if (e) cudaFree(e);
        return nullptr;
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

    if (!cuda_ck_(cudaMemcpyAsync(d_pcm, pcm, (size_t)n_samples * sizeof(float),
                             cudaMemcpyHostToDevice, stream_),
                  "memcpy pcm"))
        return fail("memcpy pcm", d_pcm, d_a, d_b, d_w, d_out);

    // WavLM input front end: per-window layer-norm of the raw waveform
    // (zero mean, unit variance, eps 1e-5). The CNN reference is computed
    // on this normalised signal, not the raw PCM.
    {
        float* d_stats = nullptr;
        if (!cuda_ck_(cudaMalloc(&d_stats, 2 * sizeof(float)), "malloc stats"))
            return fail("malloc stats", d_pcm, d_a, d_b, d_w, d_out);
        waveform_stats_kernel<<<1, kBlock, 0, stream_>>>(d_pcm, n_samples, d_stats);
        normalize_waveform_kernel<<<div_ceil_(n_samples, kBlock), kBlock, 0,
                                   stream_>>>(d_pcm, n_samples, d_stats);
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
        half_to_float_kernel<<<div_ceil_(wn, kBlock), kBlock, 0, stream_>>>(
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
        half_to_float_kernel<<<div_ceil_(Co, kBlock), kBlock, 0, stream_>>>(
            w_lng[i]->data, ln_gb, Co);
        half_to_float_kernel<<<div_ceil_(Co, kBlock), kBlock, 0, stream_>>>(
            w_lnb[i]->data, ln_gb + Co, Co);
        layer_norm_channels_kernel<<<div_ceil_(To, kBlock), kBlock, 0, stream_>>>(
            dst_act, ln_gb, ln_gb + Co, Co, To);
        gelu_exact_kernel<<<div_ceil_(Co * To, kBlock), kBlock, 0, stream_>>>(dst_act,
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
    half_to_float_kernel<<<div_ceil_(Cf, kBlock), kBlock, 0, stream_>>>(dummy->data,
                                                            d_dummy, Cf);
    transpose_scale_kernel<<<div_ceil_(Cf * T_final, kBlock), kBlock, 0, stream_>>>(
        cur, d_out, d_dummy, Cf, T_final);
    cudaFree(d_dummy);

    if (!cuda_ck_(cudaStreamSynchronize(stream_), "cnn sync"))
        return fail("sync", d_pcm, d_a, d_b, d_w, d_out);

    // Free all scratch except d_out, which is handed back to the caller.
    cudaFree(d_pcm);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_w);

    T_out = T_final;
    return d_out;
}

// --------------------------------------------------------------------------
// CNN feature extractor (P1a-step2a) — public host wrapper
// --------------------------------------------------------------------------
std::vector<float>
DiarizenWavlmPruned::debug_cnn_features(const float* pcm, int n_samples,
                                        int& T_out) {
    int T = 0;
    float* d_out = run_cnn_(pcm, n_samples, T);
    if (!d_out) return {};
    const int Cf = DiarizenWavlmPrunedArch::kFeatProjInDim;  // 211
    std::vector<float> host((size_t)T * Cf);
    cudaMemcpyAsync(host.data(), d_out, host.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    cudaFree(d_out);
    T_out = T;
    return host;
}

// --------------------------------------------------------------------------
// Encoder front end (= tap 0, P1a-step2b): feature_projection (LayerNorm ->
// Linear) -> positional convolution -> residual add. No transformer.layer_norm
// (it is not part of any of the 25 weight_sum taps; see header). Returns a GPU
// [T, 1024] frame-major buffer (caller frees); shared by debug_tap0 and
// debug_layers.
// --------------------------------------------------------------------------
float*
DiarizenWavlmPruned::run_frontend_(const float* pcm, int n_samples, int& T_out) {
    T_out = 0;
    int T = 0;
    float* d_cnn = run_cnn_(pcm, n_samples, T);  // [T, 211] frame-major
    if (!d_cnn) return nullptr;

    constexpr int Cin = DiarizenWavlmPrunedArch::kFeatProjInDim;  // 211
    constexpr int Cout = DiarizenWavlmPrunedArch::kHiddenDim;     // 1024
    constexpr int kPosK = 128;        // pos_conv kernel size
    constexpr int kPosGroups = 16;    // pos_conv groups
    constexpr int kPosCinG = Cout / kPosGroups;  // in channels per group = 64
    constexpr int kPosPad = 64;       // pos_conv padding

    auto bail = [&](const char* what,
                    std::initializer_list<void*> ptrs) -> float* {
        LOG_ERROR(kLog, "run_frontend_: %s", what);
        for (void* p : ptrs)
            if (p) cudaFree(p);
        return nullptr;
    };

    // Resolve weight views.
    const auto* fp_lng = find("wavlm_model.encoder.feature_projection.layer_norm.weight");
    const auto* fp_lnb = find("wavlm_model.encoder.feature_projection.layer_norm.bias");
    const auto* fp_w   = find("wavlm_model.encoder.feature_projection.projection.weight");
    const auto* fp_b   = find("wavlm_model.encoder.feature_projection.projection.bias");
    const auto* pc_b   = find("wavlm_model.encoder.transformer.pos_conv_embed.conv.bias");
    const auto* pc_g   = find("wavlm_model.encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0");
    const auto* pc_v   = find("wavlm_model.encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1");
    if (!fp_lng || !fp_lnb || !fp_w || !fp_b || !pc_b || !pc_g || !pc_v)
        return bail("missing tap0 tensor", {d_cnn});

    // ---- feature_projection: LayerNorm(211) over features ----------------
    float* d_lngb = nullptr;
    cudaMalloc(&d_lngb, (size_t)2 * Cin * sizeof(float));
    half_to_float_kernel<<<div_ceil_(Cin, kBlock), kBlock, 0, stream_>>>(fp_lng->data, d_lngb, Cin);
    half_to_float_kernel<<<div_ceil_(Cin, kBlock), kBlock, 0, stream_>>>(fp_lnb->data, d_lngb + Cin, Cin);
    row_layer_norm_to_kernel<<<T, kBlock, 0, stream_>>>(d_cnn, d_cnn, d_lngb, d_lngb + Cin, T, Cin);
    cudaFree(d_lngb);

    // ---- feature_projection: Linear 211 -> 1024 (+bias) ------------------
    float* d_projW = nullptr;  // [1024,211] fp32
    cudaMalloc(&d_projW, (size_t)Cout * Cin * sizeof(float));
    half_to_float_kernel<<<div_ceil_(Cout * Cin, kBlock), kBlock, 0, stream_>>>(
        fp_w->data, d_projW, Cout * Cin);
    float* d_projB = nullptr;
    cudaMalloc(&d_projB, (size_t)Cout * sizeof(float));
    half_to_float_kernel<<<div_ceil_(Cout, kBlock), kBlock, 0, stream_>>>(fp_b->data, d_projB, Cout);

    float* d_hidden = nullptr;  // [T,1024] frame-major
    cudaMalloc(&d_hidden, (size_t)T * Cout * sizeof(float));

    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS)
        return bail("cublasCreate", {d_cnn, d_projW, d_projB, d_hidden});
    diarizen_set_gemm_math_(blas);  // tensor-core GEMM (env-gated)
    cublasSetStream(blas, stream_);

    // Y[T,1024] = X[T,211] * W^T, W is [1024,211]. Column-major mapping:
    // C(1024 x T) = op(W)^T (1024 x 211) * X_colmajor(211 x T).
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, Cout, T, Cin, &alpha,
                d_projW, Cin, d_cnn, Cin, &beta, d_hidden, Cout);
    bias_add_rows_kernel<<<div_ceil_(T * Cout, kBlock), kBlock, 0, stream_>>>(d_hidden, d_projB, T, Cout);
    cublasDestroy(blas);
    cudaFree(d_cnn);
    cudaFree(d_projW);
    cudaFree(d_projB);

    // ---- positional convolution -----------------------------------------
    // Transpose hidden [T,1024] -> channel-major [1024,T] for conv input.
    float* d_ct = nullptr;
    cudaMalloc(&d_ct, (size_t)Cout * T * sizeof(float));
    transpose_TC_to_CT_kernel<<<div_ceil_(T * Cout, kBlock), kBlock, 0, stream_>>>(
        d_hidden, d_ct, T, Cout);

    // Reconstruct weight_norm weight W = g * v / ||v||_(out,in) per k.
    const long vnum = (long)Cout * kPosCinG * kPosK;
    float* d_v = nullptr;
    cudaMalloc(&d_v, vnum * sizeof(float));
    half_to_float_kernel<<<div_ceil_((int)vnum, kBlock), kBlock, 0, stream_>>>(pc_v->data, d_v, (int)vnum);
    float* d_g = nullptr;
    cudaMalloc(&d_g, (size_t)kPosK * sizeof(float));
    half_to_float_kernel<<<div_ceil_(kPosK, kBlock), kBlock, 0, stream_>>>(pc_g->data, d_g, kPosK);
    float* d_norm = nullptr;
    cudaMalloc(&d_norm, (size_t)kPosK * sizeof(float));
    posconv_norm_kernel<<<kPosK, kBlock, 0, stream_>>>(d_v, Cout, kPosCinG, kPosK, d_norm);
    float* d_pcW = nullptr;
    cudaMalloc(&d_pcW, vnum * sizeof(float));
    posconv_weight_kernel<<<div_ceil_((int)vnum, kBlock), kBlock, 0, stream_>>>(
        d_v, d_g, d_norm, Cout, kPosCinG, kPosK, d_pcW);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_norm);

    // Grouped conv1d via cuDNN: in [1,1024,1,T] pad_w=64 -> out [1,1024,1,T+1].
    const int T_conv = T + 2 * kPosPad - kPosK + 1;  // = T + 1
    float* d_conv = nullptr;
    cudaMalloc(&d_conv, (size_t)Cout * T_conv * sizeof(float));
    auto cudnn = static_cast<cudnnHandle_t>(cudnn_);
    cudnnTensorDescriptor_t in_d, out_d;
    cudnnFilterDescriptor_t filt_d;
    cudnnConvolutionDescriptor_t conv_d;
    cudnnCreateTensorDescriptor(&in_d);
    cudnnCreateTensorDescriptor(&out_d);
    cudnnCreateFilterDescriptor(&filt_d);
    cudnnCreateConvolutionDescriptor(&conv_d);
    cudnnSetTensor4dDescriptor(in_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Cout, 1, T);
    cudnnSetTensor4dDescriptor(out_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Cout, 1, T_conv);
    cudnnSetFilter4dDescriptor(filt_d, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               Cout, kPosCinG, 1, kPosK);
    cudnnSetConvolution2dDescriptor(conv_d, 0, kPosPad, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(conv_d, kPosGroups);

    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, in_d, filt_d, conv_d, out_d,
                                           1, &returned, &perf);
    size_t ws = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_d, filt_d, conv_d, out_d,
                                            perf.algo, &ws);
    if (ws > cudnn_ws_bytes_) {
        if (cudnn_ws_) cudaFree(cudnn_ws_);
        cudaMalloc(&cudnn_ws_, ws);
        cudnn_ws_bytes_ = ws;
    }
    cudnnStatus_t cs = cudnnConvolutionForward(
        cudnn, &alpha, in_d, d_ct, filt_d, d_pcW, conv_d, perf.algo,
        cudnn_ws_, cudnn_ws_bytes_, &beta, out_d, d_conv);
    cudnnDestroyTensorDescriptor(in_d);
    cudnnDestroyTensorDescriptor(out_d);
    cudnnDestroyFilterDescriptor(filt_d);
    cudnnDestroyConvolutionDescriptor(conv_d);
    cudaFree(d_ct);
    cudaFree(d_pcW);
    if (cs != CUDNN_STATUS_SUCCESS) {
        LOG_ERROR(kLog, "pos_conv failed: %s", cudnnGetErrorString(cs));
        return bail("pos_conv forward", {d_hidden, d_conv});
    }

    // Bias + exact GELU on the full [1024, T+1] conv output (pre-trim).
    float* d_convB = nullptr;
    cudaMalloc(&d_convB, (size_t)Cout * sizeof(float));
    half_to_float_kernel<<<div_ceil_(Cout, kBlock), kBlock, 0, stream_>>>(pc_b->data, d_convB, Cout);
    bias_add_channels_kernel<<<div_ceil_(Cout * T_conv, kBlock), kBlock, 0, stream_>>>(
        d_conv, d_convB, Cout, T_conv);
    gelu_exact_kernel<<<div_ceil_(Cout * T_conv, kBlock), kBlock, 0, stream_>>>(d_conv, Cout * T_conv);
    cudaFree(d_convB);

    // Residual: hidden += pos_conv^T (trimmed to first T of T+1 frames).
    transpose_CT_to_TC_add_kernel<<<div_ceil_(Cout * T, kBlock), kBlock, 0, stream_>>>(
        d_conv, d_hidden, Cout, T, T_conv);
    cudaFree(d_conv);

    if (!cuda_ck_(cudaStreamSynchronize(stream_), "tap0 sync"))
        return bail("sync", {d_hidden});

    T_out = T;
    return d_hidden;
}

// Public host wrapper around run_frontend_; bit-checked vs layer_hiddens[0].
std::vector<float>
DiarizenWavlmPruned::debug_tap0(const float* pcm, int n_samples, int& T_out) {
    int T = 0;
    float* d = run_frontend_(pcm, n_samples, T);
    if (!d) return {};
    const int Cout = DiarizenWavlmPrunedArch::kHiddenDim;
    std::vector<float> host((size_t)T * Cout);
    cudaMemcpyAsync(host.data(), d, host.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    cudaFree(d);
    T_out = T;
    return host;
}

}  // namespace orator
}  // namespace deusridet
