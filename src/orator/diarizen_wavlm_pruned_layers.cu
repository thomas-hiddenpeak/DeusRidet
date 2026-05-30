/**
 * @file diarizen_wavlm_pruned_layers.cu
 * @philosophical_role The 24 PRE-/POST-norm transformer EncoderLayers of the
 *     DiariZen WavLM-pruned encoder (P1a-step2c). Peer TU to
 *     diarizen_wavlm_pruned_forward.cu (the CNN + front end); split out to
 *     respect the 800-line .cu hard limit and to keep the gated relative-
 *     position attention math in one auditable place. Compute belongs on the
 *     GPU: every GEMM runs through cuBLAS and every elementwise / softmax /
 *     attention reduction is a CUDA kernel; the CPU only builds the integer
 *     relative-position bucket table once (N<=799^2 integer bookkeeping with
 *     no GPU entry point that is cheaper than the H2D copy it feeds).
 * @serves Orator subsystem — produces the per-layer hidden taps consumed by
 *     the weight-sum + proj + lnorm tail (P1a-step2d) and ultimately replaces
 *     tools/diarizen_worker.py's S-stage.
 *
 * Architecture (verified against the torchaudio-style reference
 * diarizen/models/module/wav2vec2/components.py, and confirmed on the loaded
 * model object): the Transformer wrapper has layer_norm_first = False (so the
 * tap-0 front end carries NO transformer.layer_norm), but each EncoderLayer
 * has layer_norm_first = True -> the layers are PRE-NORM. Each layer:
 *   residual = x; x = layer_norm(x); x = attn(x); x = residual + x
 *   x = x + feed_forward(final_layer_norm(x))
 * Fully head-pruned layers (9, 12, 16, 17) skip the attention sub-block AND
 * its layer_norm entirely, running only the pre-norm FFN residual. Attention
 * uses gated relative-position bias with per-layer surviving-head selection.
 */
#include "diarizen_wavlm_pruned.h"

#include "../communis/log.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "diarizen_wavlm_pruned_kernels.cuh"

namespace deusridet {
namespace orator {

// Device fp32 weight cache: convert each fp16 weight once, reuse forever.
// Removes the per-chunk half->float + cudaMalloc/cudaFree churn that
// dominated the sliding-window forward on Tegra (see header doc).
float* DiarizenWavlmPruned::weight_f32(const std::string& name,
                                       std::size_t* out_numel) const {
    auto it = f32_cache_.find(name);
    const auto* v = find(name);
    if (!v || !v->data) {
        LOG_ERROR(kLog, "tensor missing: %s", name.c_str());
        return nullptr;
    }
    if (out_numel) *out_numel = v->numel;
    if (it != f32_cache_.end()) return it->second;
    float* d = nullptr;
    if (cudaMalloc(&d, v->numel * sizeof(float)) != cudaSuccess) return nullptr;
    half_to_float_kernel<<<div_ceil_((int)v->numel, kBlock), kBlock>>>(
        v->data, d, (int)v->numel);
    f32_cache_.emplace(name, d);
    return d;
}

// Persistent scratch pool: recycle transient forward buffers by exact byte
// size to avoid per-chunk cudaMalloc/cudaFree on Tegra. Buffers are
// uninitialised on acquire (every consumer overwrites before read), so
// recycling is bit-equivalent to fresh allocation.
void* DiarizenWavlmPruned::scratch_acquire(std::size_t bytes) const {
    if (bytes == 0) return nullptr;
    auto it = scratch_pool_.find(bytes);
    if (it != scratch_pool_.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        return p;
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

void DiarizenWavlmPruned::scratch_release(void* ptr, std::size_t bytes) const {
    if (!ptr || bytes == 0) return;
    scratch_pool_[bytes].push_back(ptr);
}

namespace {

// Gather the [16, T, T] relative-position bias from rel_attn_embed.weight
// ([320, 16] row-major) using a precomputed per-(q,k) bucket index. One
// thread per (head, q, k). out[h, q, k] = emb[bucket[q, k] * 16 + h].
__global__ void gather_pos_bias_kernel(const int* __restrict__ bucket,  // [T*T]
                                       const float* __restrict__ emb,   // [320,16]
                                       float* __restrict__ out,         // [16,T,T]
                                       int T) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)16 * T * T;
    if (idx >= total) return;
    long tt = (long)T * T;
    int h = (int)(idx / tt);
    long r = idx % tt;
    int q = (int)(r / T);
    int k = (int)(r % T);
    int b = bucket[(long)q * T + k];
    out[idx] = emb[(long)b * 16 + h];
}

// Gated relative-position scalar per (head h in 0..15, query t). Reads the
// attention input x [T, 1024] reshaped to (16 heads, 64), applies the shared
// gru_rel_pos_linear (64->8), folds 8->2 by summing 4-wide groups, sigmoids,
// then gate_a_1 = ga*(gb*const[h] - 1) + 2. One thread per (h, t).
__global__ void gru_gate_kernel(const float* __restrict__ x,       // [T,1024]
                                const float* __restrict__ W,       // [8,64]
                                const float* __restrict__ b,       // [8]
                                const float* __restrict__ cst,     // [16]
                                float* __restrict__ gate,          // [16,T]
                                int T) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)16 * T) return;
    int h = (int)(idx / T);
    int t = (int)(idx % T);
    const float* xh = x + (long)t * 1024 + h * 64;
    float lin[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
        float a = b[o];
        const float* wo = W + o * 64;
        for (int d = 0; d < 64; ++d) a += wo[d] * xh[d];
        lin[o] = a;
    }
    float s0 = lin[0] + lin[1] + lin[2] + lin[3];
    float s1 = lin[4] + lin[5] + lin[6] + lin[7];
    float ga = 1.0f / (1.0f + expf(-s0));
    float gb = 1.0f / (1.0f + expf(-s1));
    gate[(long)h * T + t] = ga * (gb * cst[h] - 1.0f) + 2.0f;
}

// Attention scores for the surviving heads. One thread per (j, q, k) where j
// indexes the pruned head set (0..nh-1) and o = rh[j] is its original id in
// 0..15. score = scaling * (Q_j[q] . K_j[k]) + gate[o, q] * pos_bias[o, q, k].
__global__ void attn_scores_kernel(const float* __restrict__ Q,   // [T,nh*64]
                                   const float* __restrict__ K,   // [T,nh*64]
                                   const int* __restrict__ rh,    // [nh]
                                   const float* __restrict__ gate,// [16,T]
                                   const float* __restrict__ posb,// [16,T,T]
                                   float* __restrict__ S,         // [nh,T,T]
                                   int T, int nh, float scaling) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nh * T * T;
    if (idx >= total) return;
    long tt = (long)T * T;
    int j = (int)(idx / tt);
    long r = idx % tt;
    int q = (int)(r / T);
    int k = (int)(r % T);
    int o = rh[j];
    const float* qp = Q + (long)q * nh * 64 + j * 64;
    const float* kp = K + (long)k * nh * 64 + j * 64;
    float dot = 0.0f;
    for (int d = 0; d < 64; ++d) dot += qp[d] * kp[d];
    float bias = gate[(long)o * T + q] * posb[(long)o * tt + (long)q * T + k];
    S[idx] = scaling * dot + bias;
}

// Row softmax over the key axis. One block per (j, q) row of length T; block
// reduction for max then sum. In place on S [nh, T, T].
__global__ void softmax_rows_kernel(float* __restrict__ S, int T) {
    long row = blockIdx.x;  // 0 .. nh*T-1
    float* s = S + row * (long)T;

    __shared__ float red[256];
    float lm = -1e30f;
    for (int k = threadIdx.x; k < T; k += blockDim.x) lm = fmaxf(lm, s[k]);
    red[threadIdx.x] = lm;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
        __syncthreads();
    }
    float mx = red[0];
    __syncthreads();

    float ls = 0.0f;
    for (int k = threadIdx.x; k < T; k += blockDim.x) {
        float e = expf(s[k] - mx);
        s[k] = e;
        ls += e;
    }
    red[threadIdx.x] = ls;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int k = threadIdx.x; k < T; k += blockDim.x) s[k] *= inv;
}

// Context: ctx[q, j, d] = sum_k S[j, q, k] * V[k, j, d]. One thread per
// (q, j, d) over the pruned head set.
__global__ void attn_context_kernel(const float* __restrict__ S,   // [nh,T,T]
                                    const float* __restrict__ V,   // [T,nh*64]
                                    float* __restrict__ ctx,       // [T,nh*64]
                                    int T, int nh) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * nh * 64;
    if (idx >= total) return;
    int q = (int)(idx / (nh * 64));
    int rem = (int)(idx % (nh * 64));
    int j = rem / 64;
    int d = rem % 64;
    const float* srow = S + ((long)j * T + q) * T;
    float acc = 0.0f;
    for (int k = 0; k < T; ++k)
        acc += srow[k] * V[(long)k * nh * 64 + j * 64 + d];
    ctx[(long)q * nh * 64 + j * 64 + d] = acc;
}

// a += b, elementwise.
__global__ void add_inplace_kernel(float* __restrict__ a,
                                   const float* __restrict__ b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

}  // namespace

// --------------------------------------------------------------------------
// Member implementations.
// --------------------------------------------------------------------------

float* DiarizenWavlmPruned::compute_position_bias_(int T) {
    const auto* emb =
        find("wavlm_model.encoder.transformer.layers.0.attention.rel_attn_embed.weight");
    if (!emb || !emb->data) {
        LOG_ERROR(kLog, "rel_attn_embed.weight missing");
        return nullptr;
    }

    // Relative-position bucket table (bidirectional, num_buckets=320,
    // max_distance=800). Integer bookkeeping, computed once on the host.
    const int num_buckets = 320, max_distance = 800;
    const int nb = num_buckets / 2;     // 160
    const int max_exact = nb / 2;       // 80
    const double lden = std::log((double)max_distance / max_exact);
    std::vector<int> bucket((size_t)T * T);
    for (int q = 0; q < T; ++q) {
        for (int k = 0; k < T; ++k) {
            int rel = k - q;
            int rb = (rel > 0) ? nb : 0;
            int arel = rel < 0 ? -rel : rel;
            int val;
            if (arel < max_exact) {
                val = arel;
            } else {
                int large = max_exact +
                    (int)(std::log((double)arel / max_exact) / lden *
                          (nb - max_exact));
                val = large > nb - 1 ? nb - 1 : large;
            }
            bucket[(size_t)q * T + k] = rb + val;
        }
    }

    int en = (int)emb->numel;  // 320 * 16
    const std::size_t bytes_emb = (std::size_t)en * sizeof(float);
    float* d_emb = static_cast<float*>(scratch_acquire(bytes_emb));
    if (!d_emb) return nullptr;
    half_to_float_kernel<<<div_ceil_(en, kBlock), kBlock>>>(emb->data, d_emb, en);

    const std::size_t bytes_bucket = bucket.size() * sizeof(int);
    int* d_bucket = static_cast<int*>(scratch_acquire(bytes_bucket));
    cudaMemcpy(d_bucket, bucket.data(), bucket.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    const std::size_t bytes_bias = (std::size_t)16 * T * T * sizeof(float);
    float* d_bias = static_cast<float*>(scratch_acquire(bytes_bias));
    if (!d_bias) {
        scratch_release(d_emb, bytes_emb);
        scratch_release(d_bucket, bytes_bucket);
        return nullptr;
    }
    long tot = (long)16 * T * T;
    gather_pos_bias_kernel<<<(int)((tot + kBlock - 1) / kBlock), kBlock>>>(
        d_bucket, d_emb, d_bias, T);
    scratch_release(d_emb, bytes_emb);
    scratch_release(d_bucket, bytes_bucket);
    return d_bias;
}

namespace {

// Fetch a per-layer tensor by suffix and return a cached fp32 device buffer
// (owned by the object; caller must NOT free). Returns nullptr on lookup
// failure.
float* layer_f32(const DiarizenWavlmPruned& self, int layer,
                 const char* suffix, std::size_t* out_numel) {
    char name[256];
    std::snprintf(name, sizeof(name),
                  "wavlm_model.encoder.transformer.layers.%d.%s", layer, suffix);
    return self.weight_f32(name, out_numel);
}

// Y[T, M] = X[T, K] @ W^T, W is [M, K] row-major. Optionally adds bias[M].
void linear_(cublasHandle_t blas, const float* d_X, const float* d_W,
             const float* d_bias, float* d_Y, int T, int M, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &alpha, d_W, K, d_X, K,
                &beta, d_Y, M);
    if (d_bias)
        bias_add_rows_kernel<<<div_ceil_(T * M, kBlock), kBlock>>>(d_Y, d_bias,
                                                                   T, M);
}

// Fetch a top-level tensor by exact name into a cached fp32 device buffer
// (owned by the object; caller must NOT free).
float* top_f32(const DiarizenWavlmPruned& self, const char* name,
               std::size_t* out_numel) {
    return self.weight_f32(name, out_numel);
}

// Learned weighted sum over the 25 hidden taps. taps is [25, T, H] frame-
// major per slot; ws is [25]. out[t, c] = sum_k ws[k] * taps[k, t, c].
// One thread per (t, c).
__global__ void weighted_sum_kernel(const float* __restrict__ taps,  // [25,T,H]
                                    const float* __restrict__ ws,    // [25]
                                    float* __restrict__ out,         // [T,H]
                                    long TH) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TH) return;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 25; ++k) acc += ws[k] * taps[(long)k * TH + idx];
    out[idx] = acc;
}

}  // namespace

bool DiarizenWavlmPruned::run_encoder_layer_(int layer, float* d_hidden, int T,
                                             const float* d_pos_bias,
                                             void* cublas) {
    auto blas = static_cast<cublasHandle_t>(cublas);
    const int H = DiarizenWavlmPrunedArch::kHiddenDim;  // 1024
    const auto& dims = layer_dims_[layer];
    const int nh = dims.num_heads;
    const int ffn = dims.ffn_inner;

    // ---- attention sub-block (PRE-NORM; skipped for fully head-pruned
    //      layers, in which case layer_norm is NOT applied either) ---------
    if (nh > 0) {
        const int inner = nh * 64;
        std::size_t n_ignored = 0;
        float* Wq = layer_f32(*this, layer, "attention.q_proj.weight", &n_ignored);
        float* bq = layer_f32(*this, layer, "attention.q_proj.bias", &n_ignored);
        float* Wk = layer_f32(*this, layer, "attention.k_proj.weight", &n_ignored);
        float* bk = layer_f32(*this, layer, "attention.k_proj.bias", &n_ignored);
        float* Wv = layer_f32(*this, layer, "attention.v_proj.weight", &n_ignored);
        float* bv = layer_f32(*this, layer, "attention.v_proj.bias", &n_ignored);
        float* Wo = layer_f32(*this, layer, "attention.out_proj.weight", &n_ignored);
        float* bo = layer_f32(*this, layer, "attention.out_proj.bias", &n_ignored);
        float* Wg = layer_f32(*this, layer, "attention.gru_rel_pos_linear.weight", &n_ignored);
        float* bg = layer_f32(*this, layer, "attention.gru_rel_pos_linear.bias", &n_ignored);
        float* cst = layer_f32(*this, layer, "attention.gru_rel_pos_const", &n_ignored);
        float* ln_w = layer_f32(*this, layer, "layer_norm.weight", &n_ignored);
        float* ln_b = layer_f32(*this, layer, "layer_norm.bias", &n_ignored);
        if (!Wq || !bq || !Wk || !bk || !Wv || !bv || !Wo || !bo || !Wg ||
            !bg || !cst || !ln_w || !ln_b) {
            LOG_ERROR(kLog, "layer %d attention weights missing", layer);
            return false;
        }

        // Pre-norm: xn = layer_norm(d_hidden); attention reads xn, residual
        // stream (d_hidden) is preserved for the post-attention add.
        const std::size_t bytes_TH = (std::size_t)T * H * sizeof(float);
        const std::size_t bytes_Tinner = (std::size_t)T * inner * sizeof(float);
        float* xn = static_cast<float*>(scratch_acquire(bytes_TH));
        row_layer_norm_to_kernel<<<T, kBlock>>>(d_hidden, xn, ln_w, ln_b, T, H);

        float* Q = static_cast<float*>(scratch_acquire(bytes_Tinner));
        float* K = static_cast<float*>(scratch_acquire(bytes_Tinner));
        float* V = static_cast<float*>(scratch_acquire(bytes_Tinner));
        linear_(blas, xn, Wq, bq, Q, T, inner, H);
        linear_(blas, xn, Wk, bk, K, T, inner, H);
        linear_(blas, xn, Wv, bv, V, T, inner, H);

        // Gated relative-position scalar per (head, query) over all 16 heads.
        const std::size_t bytes_gate = (std::size_t)16 * T * sizeof(float);
        float* gate = static_cast<float*>(scratch_acquire(bytes_gate));
        gru_gate_kernel<<<div_ceil_(16 * T, kBlock), kBlock>>>(xn, Wg, bg,
                                                               cst, gate, T);

        // remaining_heads -> device.
        const std::size_t bytes_rh = (std::size_t)nh * sizeof(int);
        int* d_rh = static_cast<int*>(scratch_acquire(bytes_rh));
        cudaMemcpy(d_rh, remaining_heads_[layer].data(), nh * sizeof(int),
                   cudaMemcpyHostToDevice);

        const std::size_t bytes_S = (std::size_t)nh * T * T * sizeof(float);
        float* S = static_cast<float*>(scratch_acquire(bytes_S));
        long stot = (long)nh * T * T;
        const float scaling = 0.125f;  // 64^-0.5
        attn_scores_kernel<<<(int)((stot + kBlock - 1) / kBlock), kBlock>>>(
            Q, K, d_rh, gate, d_pos_bias, S, T, nh, scaling);
        softmax_rows_kernel<<<nh * T, 256>>>(S, T);

        float* ctx = static_cast<float*>(scratch_acquire(bytes_Tinner));
        long ctot = (long)T * inner;
        attn_context_kernel<<<(int)((ctot + kBlock - 1) / kBlock), kBlock>>>(
            S, V, ctx, T, nh);

        // out_proj then residual add: d_hidden = d_hidden + out_proj(ctx).
        float* attn_out = static_cast<float*>(scratch_acquire(bytes_TH));
        linear_(blas, ctx, Wo, bo, attn_out, T, H, inner);
        add_inplace_kernel<<<div_ceil_(T * H, kBlock), kBlock>>>(
            d_hidden, attn_out, (long)T * H);

        scratch_release(xn, bytes_TH);
        scratch_release(Q, bytes_Tinner);
        scratch_release(K, bytes_Tinner);
        scratch_release(V, bytes_Tinner);
        scratch_release(gate, bytes_gate);
        scratch_release(d_rh, bytes_rh);
        scratch_release(S, bytes_S);
        scratch_release(ctx, bytes_Tinner);
        scratch_release(attn_out, bytes_TH);
        // Wq..ln_b are device-resident in the f32 weight cache; not freed here.
    }

    // ---- feed forward (PRE-NORM): x = x + output_dense(gelu(
    //      intermediate_dense(final_layer_norm(x)))) ----------------------
    float* fln_w = layer_f32(*this, layer, "final_layer_norm.weight", nullptr);
    float* fln_b = layer_f32(*this, layer, "final_layer_norm.bias", nullptr);
    if (!fln_w || !fln_b) return false;
    float* fn = static_cast<float*>(scratch_acquire((std::size_t)T * H * sizeof(float)));
    row_layer_norm_to_kernel<<<T, kBlock>>>(d_hidden, fn, fln_w, fln_b, T, H);
    // fln_w/fln_b are cache-owned; not freed here.

    float* Wi = layer_f32(*this, layer, "feed_forward.intermediate_dense.weight", nullptr);
    float* bi = layer_f32(*this, layer, "feed_forward.intermediate_dense.bias", nullptr);
    float* Wo2 = layer_f32(*this, layer, "feed_forward.output_dense.weight", nullptr);
    float* bo2 = layer_f32(*this, layer, "feed_forward.output_dense.bias", nullptr);
    if (!Wi || !bi || !Wo2 || !bo2) return false;

    const std::size_t bytes_TH_ff = (std::size_t)T * H * sizeof(float);
    const std::size_t bytes_Tffn = (std::size_t)T * ffn * sizeof(float);
    float* inter = static_cast<float*>(scratch_acquire(bytes_Tffn));
    linear_(blas, fn, Wi, bi, inter, T, ffn, H);
    gelu_exact_kernel<<<div_ceil_(T * ffn, kBlock), kBlock>>>(inter, T * ffn);
    float* ff = static_cast<float*>(scratch_acquire(bytes_TH_ff));
    linear_(blas, inter, Wo2, bo2, ff, T, H, ffn);
    add_inplace_kernel<<<div_ceil_(T * H, kBlock), kBlock>>>(d_hidden, ff,
                                                             (long)T * H);
    scratch_release(fn, bytes_TH_ff);
    scratch_release(inter, bytes_Tffn);
    scratch_release(ff, bytes_TH_ff);
    // Wi/bi/Wo2/bo2 are cache-owned; not freed here.
    return true;
}

std::vector<float> DiarizenWavlmPruned::debug_layers(const float* pcm,
                                                     int n_samples,
                                                     int up_to_layer,
                                                     int& T_out) {
    int T = 0;
    float* d_hidden = run_frontend_(pcm, n_samples, T);
    if (!d_hidden) return {};
    const int H = DiarizenWavlmPrunedArch::kHiddenDim;

    if (up_to_layer > 0) {
        const std::size_t bytes_pos = (std::size_t)16 * T * T * sizeof(float);
        float* d_pos_bias = compute_position_bias_(T);
        if (!d_pos_bias) {
            cudaFree(d_hidden);
            return {};
        }
        cublasHandle_t blas = nullptr;
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_hidden);
            scratch_release(d_pos_bias, bytes_pos);
            return {};
        }
        for (int l = 0; l < up_to_layer && l < DiarizenWavlmPrunedArch::kTransformerLayers; ++l) {
            if (!run_encoder_layer_(l, d_hidden, T, d_pos_bias, blas)) {
                cublasDestroy(blas);
                cudaFree(d_hidden);
                scratch_release(d_pos_bias, bytes_pos);
                return {};
            }
        }
        cublasDestroy(blas);
        scratch_release(d_pos_bias, bytes_pos);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_hidden);
        return {};
    }
    std::vector<float> host((size_t)T * H);
    cudaMemcpy(host.data(), d_hidden, host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_hidden);
    T_out = T;
    return host;
}

std::vector<float> DiarizenWavlmPruned::debug_lnorm_tail(const float* pcm,
                                                         int n_samples,
                                                         int& T_out) {
    int T = 0;
    float* d_hidden = run_frontend_(pcm, n_samples, T);
    if (!d_hidden) return {};
    const int H = DiarizenWavlmPrunedArch::kHiddenDim;       // 1024
    const int kNumTaps = DiarizenWavlmPrunedArch::kTransformerLayers + 1;  // 25
    const long TH = (long)T * H;

    // Collect all 25 taps (front end + 24 layers) into [25, T, H].
    const std::size_t bytes_taps = (std::size_t)kNumTaps * TH * sizeof(float);
    float* d_taps = static_cast<float*>(scratch_acquire(bytes_taps));
    if (!d_taps) {
        cudaFree(d_hidden);
        return {};
    }
    cudaMemcpy(d_taps, d_hidden, TH * sizeof(float), cudaMemcpyDeviceToDevice);

    const std::size_t bytes_pos = (std::size_t)16 * T * T * sizeof(float);
    float* d_pos_bias = compute_position_bias_(T);
    cublasHandle_t blas = nullptr;
    if (!d_pos_bias || cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_hidden); scratch_release(d_taps, bytes_taps);
        if (d_pos_bias) scratch_release(d_pos_bias, bytes_pos);
        return {};
    }
    for (int l = 0; l < DiarizenWavlmPrunedArch::kTransformerLayers; ++l) {
        if (!run_encoder_layer_(l, d_hidden, T, d_pos_bias, blas)) {
            cublasDestroy(blas); cudaFree(d_hidden);
            scratch_release(d_taps, bytes_taps);
            scratch_release(d_pos_bias, bytes_pos);
            return {};
        }
        cudaMemcpy(d_taps + (long)(l + 1) * TH, d_hidden, TH * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
    scratch_release(d_pos_bias, bytes_pos);
    cudaFree(d_hidden);

    // weight_sum.weight [1, 25] -> ws[25]; summed[t, c] = sum_k ws[k]*tap_k.
    float* d_ws = top_f32(*this, "weight_sum.weight", nullptr);
    float* d_sum = static_cast<float*>(scratch_acquire((std::size_t)TH * sizeof(float)));
    if (!d_ws || !d_sum) {
        cublasDestroy(blas); scratch_release(d_taps, bytes_taps);
        if (d_sum) scratch_release(d_sum, (std::size_t)TH * sizeof(float));
        return {};
    }
    weighted_sum_kernel<<<div_ceil_((int)TH, kBlock), kBlock>>>(d_taps, d_ws,
                                                                d_sum, TH);
    scratch_release(d_taps, bytes_taps);
    // d_ws is cache-owned; not freed here.

    // proj: [T, 256] = summed[T, 1024] @ proj.weight^T + proj.bias.
    const int D = 256;
    float* d_pw = top_f32(*this, "proj.weight", nullptr);
    float* d_pb = top_f32(*this, "proj.bias", nullptr);
    const std::size_t bytes_sum = (std::size_t)TH * sizeof(float);
    const std::size_t bytes_proj = (std::size_t)T * D * sizeof(float);
    float* d_proj = static_cast<float*>(scratch_acquire(bytes_proj));
    if (!d_pw || !d_pb || !d_proj) {
        cublasDestroy(blas); scratch_release(d_sum, bytes_sum);
        if (d_proj) scratch_release(d_proj, bytes_proj);
        return {};
    }
    linear_(blas, d_sum, d_pw, d_pb, d_proj, T, D, H);
    cublasDestroy(blas);
    scratch_release(d_sum, bytes_sum);
    // d_pw/d_pb are cache-owned; not freed here.

    // lnorm: LayerNorm(256) in place over the feature dim.
    float* d_lw = top_f32(*this, "lnorm.weight", nullptr);
    float* d_lb = top_f32(*this, "lnorm.bias", nullptr);
    if (!d_lw || !d_lb) {
        scratch_release(d_proj, bytes_proj);
        return {};
    }
    row_layer_norm_to_kernel<<<T, kBlock>>>(d_proj, d_proj, d_lw, d_lb, T, D);
    // d_lw/d_lb are cache-owned; not freed here.

    if (cudaDeviceSynchronize() != cudaSuccess) {
        scratch_release(d_proj, bytes_proj);
        return {};
    }
    std::vector<float> host((size_t)T * D);
    cudaMemcpy(host.data(), d_proj, host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    scratch_release(d_proj, bytes_proj);
    T_out = T;
    return host;
}

}  // namespace orator
}  // namespace deusridet
