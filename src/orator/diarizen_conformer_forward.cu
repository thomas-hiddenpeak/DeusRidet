/**
 * @file diarizen_conformer_forward.cu
 * @philosophical_role Forward orchestration peer TU for the DiariZen Conformer
 *     head (P1b). Holds run_block_ (one Conformer block: macaron-FFN, MHSA,
 *     depthwise-conv module, macaron-FFN, final LN) and the public debug_*
 *     taps that drive bit-equality. Split from diarizen_conformer_head.cu to
 *     respect the 800-line .cu hard limit. Compute belongs on the GPU: cuBLAS
 *     GEMMs + CUDA kernels only; the CPU sequences the four blocks.
 * @serves DiarizenConformerHead — produces conformer_out / classifier_logits /
 *     classifier_probs taps.
 */
#include "diarizen_conformer_head.h"

#include "../communis/log.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "diarizen_wavlm_pruned_kernels.cuh"
#include "diarizen_conformer_kernels.cuh"

namespace deusridet {
namespace orator {

namespace {
constexpr const char* kFLog = "DiariZenConformer";
}  // namespace

// Persistent fp32 weight cache (declared in the header). Converts the fp16
// arena tensor to a device-resident fp32 buffer on first use and returns the
// cached pointer thereafter, so the four Conformer blocks no longer pay a
// cudaMalloc + convert + cudaFree per weight per chunk (each malloc/free walks
// the Tegra VMM map). Bit-equality preserving: same values, fetched once.
float* DiarizenConformerHead::weight_f32(const std::string& name) const {
    auto it = f32_cache_.find(name);
    if (it != f32_cache_.end()) return it->second;
    const auto* v = find(name);
    if (!v || !v->data) {
        LOG_ERROR(kFLog, "tensor missing: %s", name.c_str());
        return nullptr;
    }
    float* d = nullptr;
    if (cudaMalloc(&d, v->numel * sizeof(float)) != cudaSuccess) return nullptr;
    half_to_float_kernel<<<div_ceil_((int)v->numel, kBlock), kBlock>>>(
        v->data, d, (int)v->numel);
    f32_cache_.emplace(name, d);
    return d;
}

// Persistent scratch pool (declared in the header): recycle transient forward
// buffers by exact byte size so the four Conformer blocks no longer pay a
// cudaMalloc/cudaFree per buffer per chunk (each Tegra alloc/free walks the
// global VMM map). Buffers are uninitialised on acquire; every consumer
// overwrites before read, so recycling is bit-equivalent to fresh allocation.
void* DiarizenConformerHead::scratch_acquire(std::size_t bytes) const {
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

void DiarizenConformerHead::scratch_release(void* ptr, std::size_t bytes) const {
    if (!ptr || bytes == 0) return;
    scratch_pool_[bytes].push_back(ptr);
}

namespace {

// Fetch a tensor by exact name as a device fp32 buffer. Returns the head's
// persistent cached pointer (NOT a fresh buffer) — callers MUST NOT free it.
float* fetch_f32(const DiarizenConformerHead& self, const char* name,
                 std::size_t* out_numel) {
    float* d = self.weight_f32(name);
    if (d && out_numel) {
        const auto* v = self.find(name);
        if (v) *out_numel = v->numel;
    }
    return d;
}

// Convenience: fetch "conformer.conformer_layer.<L>.<suffix>" (cached).
float* layer_f32c(const DiarizenConformerHead& self, int layer,
                  const char* suffix) {
    char name[256];
    std::snprintf(name, sizeof(name), "conformer.conformer_layer.%d.%s", layer,
                  suffix);
    return self.weight_f32(name);
}

// Y[T, M] = X[T, K] @ W^T + bias[M], W is [M, K] row-major.
void clinear_(cublasHandle_t blas, const float* d_X, const float* d_W,
              const float* d_bias, float* d_Y, int T, int M, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &alpha, d_W, K, d_X, K,
                &beta, d_Y, M);
    if (d_bias)
        bias_add_rows_kernel<<<div_ceil_(T * M, kBlock), kBlock>>>(d_Y, d_bias,
                                                                   T, M);
}

// Macaron FFN: x += 0.5 * w_2(swish(w_1(ln_norm(x)))).
bool ffn_(const DiarizenConformerHead& self, cublasHandle_t blas, int layer,
          const char* prefix, float* d_x, int T) {
    const int C = DiarizenConformerArch::kFeatDim;
    const int FF = DiarizenConformerArch::kFfnHidden;
    char s[128];
    std::snprintf(s, sizeof(s), "%s.ln_norm.weight", prefix);
    float* lnw = layer_f32c(self, layer, s);
    std::snprintf(s, sizeof(s), "%s.ln_norm.bias", prefix);
    float* lnb = layer_f32c(self, layer, s);
    std::snprintf(s, sizeof(s), "%s.w_1.weight", prefix);
    float* w1 = layer_f32c(self, layer, s);
    std::snprintf(s, sizeof(s), "%s.w_1.bias", prefix);
    float* b1 = layer_f32c(self, layer, s);
    std::snprintf(s, sizeof(s), "%s.w_2.weight", prefix);
    float* w2 = layer_f32c(self, layer, s);
    std::snprintf(s, sizeof(s), "%s.w_2.bias", prefix);
    float* b2 = layer_f32c(self, layer, s);
    if (!lnw || !lnb || !w1 || !b1 || !w2 || !b2) return false;

    float *xn = nullptr, *h1 = nullptr, *h2 = nullptr;
    const std::size_t bTC = (std::size_t)T * C * sizeof(float);
    const std::size_t bTFF = (std::size_t)T * FF * sizeof(float);
    xn = static_cast<float*>(self.scratch_acquire(bTC));
    h1 = static_cast<float*>(self.scratch_acquire(bTFF));
    h2 = static_cast<float*>(self.scratch_acquire(bTC));
    row_layer_norm_to_kernel<<<T, kBlock>>>(d_x, xn, lnw, lnb, T, C);
    clinear_(blas, xn, w1, b1, h1, T, FF, C);
    swish_kernel<<<div_ceil_(T * FF, kBlock), kBlock>>>(h1, (long)T * FF);
    clinear_(blas, h1, w2, b2, h2, T, C, FF);
    scaled_add_kernel<<<div_ceil_(T * C, kBlock), kBlock>>>(d_x, h2, 0.5f,
                                                            (long)T * C);
    self.scratch_release(xn, bTC);
    self.scratch_release(h1, bTFF);
    self.scratch_release(h2, bTC);  // scratch only; weights cached
    return true;
}

// MHSA module: x += linearO(MHSA(ln_norm(x))), scale 1/sqrt(d_k).
bool mha_(const DiarizenConformerHead& self, cublasHandle_t blas, int layer,
          float* d_x, int T) {
    const int C = DiarizenConformerArch::kFeatDim;
    const int nh = DiarizenConformerArch::kNumHead;
    const int dk = DiarizenConformerArch::kHeadDim;
    float* lnw = layer_f32c(self, layer, "mha.ln_norm.weight");
    float* lnb = layer_f32c(self, layer, "mha.ln_norm.bias");
    float* wq = layer_f32c(self, layer, "mha.mha.linearQ.weight");
    float* bq = layer_f32c(self, layer, "mha.mha.linearQ.bias");
    float* wk = layer_f32c(self, layer, "mha.mha.linearK.weight");
    float* bk = layer_f32c(self, layer, "mha.mha.linearK.bias");
    float* wv = layer_f32c(self, layer, "mha.mha.linearV.weight");
    float* bv = layer_f32c(self, layer, "mha.mha.linearV.bias");
    float* wo = layer_f32c(self, layer, "mha.mha.linearO.weight");
    float* bo = layer_f32c(self, layer, "mha.mha.linearO.bias");
    if (!lnw || !lnb || !wq || !bq || !wk || !bk || !wv || !bv || !wo || !bo)
        return false;

    float *xn, *Q, *K, *V, *S, *ctx, *o;
    const std::size_t bTC = (std::size_t)T * C * sizeof(float);
    const std::size_t bS = (std::size_t)nh * T * T * sizeof(float);
    xn = static_cast<float*>(self.scratch_acquire(bTC));
    Q = static_cast<float*>(self.scratch_acquire(bTC));
    K = static_cast<float*>(self.scratch_acquire(bTC));
    V = static_cast<float*>(self.scratch_acquire(bTC));
    S = static_cast<float*>(self.scratch_acquire(bS));
    ctx = static_cast<float*>(self.scratch_acquire(bTC));
    o = static_cast<float*>(self.scratch_acquire(bTC));
    row_layer_norm_to_kernel<<<T, kBlock>>>(d_x, xn, lnw, lnb, T, C);
    clinear_(blas, xn, wq, bq, Q, T, C, C);
    clinear_(blas, xn, wk, bk, K, T, C, C);
    clinear_(blas, xn, wv, bv, V, T, C, C);
    const float scale = 1.0f / std::sqrt((float)dk);
    const float zero = 0.0f, one = 1.0f;
    // Scores via batched GEMM (replaces hand-rolled mhsa_scores_kernel, which
    // was 13.1% of pipeline GPU time): per head j, S_j[q,k] = scale * Q_j·K_j.
    // Row-major S_j[q,k] is the column-major (k,q) matrix K_jᵀ·Q_j.
    cublasSgemmStridedBatched(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, T, T, dk, &scale,
        K, nh * dk, dk,                    // A = per-head K, lda, strideA
        Q, nh * dk, dk,                    // B = per-head Q, ldb, strideB
        &zero, S, T, (long long)T * T, nh);
    softmax_rows_kernel_c<<<nh * T, 256>>>(S, T);
    // Context via batched GEMM: ctx_j[q,d] = sum_k S_j[q,k] · V_j[k,d].
    cublasSgemmStridedBatched(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, dk, T, T, &one,
        V, nh * dk, dk,                    // A = per-head V, lda, strideA
        S, T, (long long)T * T,            // B = per-head S, ldb, strideB
        &zero, ctx, nh * dk, dk, nh);
    clinear_(blas, ctx, wo, bo, o, T, C, C);
    scaled_add_kernel<<<div_ceil_(T * C, kBlock), kBlock>>>(d_x, o, 1.0f,
                                                            (long)T * C);
    self.scratch_release(xn, bTC);
    self.scratch_release(Q, bTC);
    self.scratch_release(K, bTC);
    self.scratch_release(V, bTC);
    self.scratch_release(S, bS);
    self.scratch_release(ctx, bTC);
    self.scratch_release(o, bTC);  // scratch only; weights cached
    return true;
}

// Convolution module: x += pw2(swish(bn(dw(glu(pw1(ln_norm(x))))))).
bool conv_(const DiarizenConformerHead& self, cublasHandle_t blas, int layer,
           float* d_x, int T) {
    const int C = DiarizenConformerArch::kFeatDim;
    const int K = DiarizenConformerArch::kKernelSize;
    float* lnw = layer_f32c(self, layer, "conv.ln_norm.weight");
    float* lnb = layer_f32c(self, layer, "conv.ln_norm.bias");
    float* pw1 = layer_f32c(self, layer, "conv.pointwise_conv1.weight");
    float* pb1 = layer_f32c(self, layer, "conv.pointwise_conv1.bias");
    float* dww = layer_f32c(self, layer, "conv.depthwise_conv.weight");
    float* dwb = layer_f32c(self, layer, "conv.depthwise_conv.bias");
    float* bnw = layer_f32c(self, layer, "conv.bn_norm.weight");
    float* bnb = layer_f32c(self, layer, "conv.bn_norm.bias");
    float* bnm = layer_f32c(self, layer, "conv.bn_norm.running_mean");
    float* bnv = layer_f32c(self, layer, "conv.bn_norm.running_var");
    float* pw2 = layer_f32c(self, layer, "conv.pointwise_conv2.weight");
    float* pb2 = layer_f32c(self, layer, "conv.pointwise_conv2.bias");
    if (!lnw || !lnb || !pw1 || !pb1 || !dww || !dwb || !bnw || !bnb || !bnm ||
        !bnv || !pw2 || !pb2)
        return false;

    float *xn, *pc1, *glu, *gct, *dw, *swt, *pc2;
    const std::size_t bTC = (std::size_t)T * C * sizeof(float);
    const std::size_t bT2C = (std::size_t)T * 2 * C * sizeof(float);
    const std::size_t bCT = (std::size_t)C * T * sizeof(float);
    xn = static_cast<float*>(self.scratch_acquire(bTC));
    pc1 = static_cast<float*>(self.scratch_acquire(bT2C));
    glu = static_cast<float*>(self.scratch_acquire(bTC));
    gct = static_cast<float*>(self.scratch_acquire(bCT));
    dw = static_cast<float*>(self.scratch_acquire(bCT));
    swt = static_cast<float*>(self.scratch_acquire(bTC));
    pc2 = static_cast<float*>(self.scratch_acquire(bTC));
    row_layer_norm_to_kernel<<<T, kBlock>>>(d_x, xn, lnw, lnb, T, C);
    // pointwise_conv1 (1x1) == per-frame linear C -> 2C.
    clinear_(blas, xn, pw1, pb1, pc1, T, 2 * C, C);
    glu_tc_kernel<<<div_ceil_(T * C, kBlock), kBlock>>>(pc1, glu, T, C);
    transpose_tc_to_ct_kernel<<<div_ceil_(T * C, kBlock), kBlock>>>(glu, gct, T,
                                                                    C);
    depthwise_conv1d_kernel<<<div_ceil_(C * T, kBlock), kBlock>>>(gct, dww, dwb,
                                                                  dw, C, T, K);
    batchnorm_ct_kernel<<<div_ceil_(C * T, kBlock), kBlock>>>(dw, bnw, bnb, bnm,
                                                              bnv, C, T, 1e-5f);
    swish_kernel<<<div_ceil_(C * T, kBlock), kBlock>>>(dw, (long)C * T);
    transpose_ct_to_tc_kernel<<<div_ceil_(C * T, kBlock), kBlock>>>(dw, swt, C,
                                                                    T);
    // pointwise_conv2 (1x1) == per-frame linear C -> C.
    clinear_(blas, swt, pw2, pb2, pc2, T, C, C);
    scaled_add_kernel<<<div_ceil_(T * C, kBlock), kBlock>>>(d_x, pc2, 1.0f,
                                                            (long)T * C);
    self.scratch_release(xn, bTC);
    self.scratch_release(pc1, bT2C);
    self.scratch_release(glu, bTC);
    self.scratch_release(gct, bCT);
    self.scratch_release(dw, bCT);
    self.scratch_release(swt, bTC);
    self.scratch_release(pc2, bTC);  // scratch only; weights cached
    return true;
}

}  // namespace

bool DiarizenConformerHead::run_block_(int layer, float* d_x, int T,
                                       void* cublas) {
    auto blas = static_cast<cublasHandle_t>(cublas);
    const int C = DiarizenConformerArch::kFeatDim;
    if (!ffn_(*this, blas, layer, "ffn1", d_x, T)) return false;
    if (!mha_(*this, blas, layer, d_x, T)) return false;
    if (!conv_(*this, blas, layer, d_x, T)) return false;
    if (!ffn_(*this, blas, layer, "ffn2", d_x, T)) return false;
    // Final per-block LayerNorm, in place.
    {
        char nw[160], nb[160];
        std::snprintf(nw, sizeof(nw),
                      "conformer.conformer_layer.%d.ln_norm.weight", layer);
        std::snprintf(nb, sizeof(nb),
                      "conformer.conformer_layer.%d.ln_norm.bias", layer);
        float* w = weight_f32(nw);
        float* b = weight_f32(nb);
        if (!w || !b) return false;
        row_layer_norm_to_kernel<<<T, kBlock>>>(d_x, d_x, w, b, T, C);
    }
    return true;
}

std::vector<float> DiarizenConformerHead::run_(const float* feat, int T,
                                               int stage) {
    if (!loaded_ || T <= 0 || !feat) return {};
    const int C = DiarizenConformerArch::kFeatDim;
    const int NC = DiarizenConformerArch::kNumClasses;

    const std::size_t bytes_x = (std::size_t)T * C * sizeof(float);
    float* d_x = static_cast<float*>(scratch_acquire(bytes_x));
    if (!d_x) return {};
    cudaMemcpy(d_x, feat, bytes_x, cudaMemcpyHostToDevice);

    // Cached cublas handle: created once and reused for every chunk, so the
    // seg loop no longer pays a cublasCreate/Destroy (and its implicit device
    // sync) per chunk. TF32 math mode is reasserted each call (cheap).
    if (!blas_) {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            scratch_release(d_x, bytes_x);
            return {};
        }
        blas_ = h;
    }
    cublasHandle_t blas = static_cast<cublasHandle_t>(blas_);
    // Route dense GEMMs through TF32 tensor cores (Ampere sm87). ResNet34 in
    // this same pipeline already runs TF32 convs, so the precision regime is
    // established; ~3-4x over fp32 ampere_sgemm on the linear layers.
    diarizen_set_gemm_math_(blas);
    for (int l = 0; l < DiarizenConformerArch::kNumLayer; ++l) {
        if (!run_block_(l, d_x, T, blas)) {
            scratch_release(d_x, bytes_x);
            return {};
        }
    }

    int out_dim = C;
    float* d_out = d_x;
    float* d_logits = nullptr;
    const std::size_t bytes_logits = (std::size_t)T * NC * sizeof(float);
    if (stage >= 1) {
        float* cw = fetch_f32(*this, "classifier.weight", nullptr);
        float* cb = fetch_f32(*this, "classifier.bias", nullptr);
        if (!cw || !cb) {
            scratch_release(d_x, bytes_x);
            return {};
        }
        d_logits = static_cast<float*>(scratch_acquire(bytes_logits));
        clinear_(blas, d_x, cw, cb, d_logits, T, NC, C);  // cw/cb cached
        out_dim = NC;
        d_out = d_logits;
        if (stage >= 2)
            logsoftmax_rows_kernel<<<T, 1>>>(d_logits, T, NC);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        scratch_release(d_x, bytes_x);
        if (d_logits) scratch_release(d_logits, bytes_logits);
        return {};
    }
    std::vector<float> host((size_t)T * out_dim);
    cudaMemcpy(host.data(), d_out, host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    scratch_release(d_x, bytes_x);
    if (d_logits) scratch_release(d_logits, bytes_logits);
    return host;
}

std::vector<float> DiarizenConformerHead::debug_conformer(const float* feat,
                                                          int T) {
    return run_(feat, T, 0);
}
std::vector<float> DiarizenConformerHead::debug_logits(const float* feat, int T) {
    return run_(feat, T, 1);
}
std::vector<float> DiarizenConformerHead::debug_probs(const float* feat, int T) {
    return run_(feat, T, 2);
}

}  // namespace orator
}  // namespace deusridet
