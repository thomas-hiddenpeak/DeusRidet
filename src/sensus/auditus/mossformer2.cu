// mossformer2.cu — MossFormer2 native CUDA forward pass implementation.
//
// Adapted from ClearerVoice-Studio MossFormer2 (Apache-2.0).
// Original: https://github.com/modelscope/ClearerVoice-Studio

#include "mossformer2.h"
#include "../../communis/log.h"
#include "../../machina/safetensors.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace deusridet {

using namespace mf2;

static constexpr int BLK = 256;
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

#define CK(call) do {                                                       \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess)                                                  \
        LOG_ERROR("MF2", "CUDA %s:%d: %s", __FILE__, __LINE__,             \
                  cudaGetErrorString(_e));                                   \
} while(0)

// ============================================================================
// GEMM helpers
// ============================================================================

// Channel-first Conv1d(k=1): Y[Co,L] = W[Co,Ci] @ X[Ci,L]
// All stored row-major. cuBLAS col-major: Y^T = X^T @ W^T
static void gemm_CL(cublasHandle_t h,
                     const float* W, const float* X, float* Y,
                     int Co, int Ci, int L) {
    float a = 1.f, b = 0.f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                L, Co, Ci, &a, X, L, W, Ci, &b, Y, L);
}

// Row-major Linear: Y[M,N] = A[M,K] @ B^T[K,N]
// B stored [N,K] (PyTorch weight convention). Transposed in GEMM.
static void gemm_nt(cublasHandle_t h,
                     const float* A, const float* BT, float* C,
                     int M, int N, int K) {
    float a = 1.f, b = 0.f;
    // C^T[N,M] = BT^T_col[N,K] * A^T_col[K,M]
    // BT is [N,K] row-major = [K,N] col-major → need CUBLAS_OP_T
    // Actually: A[M,K] row = [K,M] col; BT[N,K] row = [K,N] col
    // Want C[M,N] row = [N,M] col
    // C_col = BT_col^T * A_col = [N,K] * [K,M] = [N,M] ✓
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &a, BT, K, A, K, &b, C, N);
}

// Row-major: C[M,N] = A[M,K] @ B[K,N] (no transpose)
static void gemm_nn(cublasHandle_t h,
                     const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha = 1.f, float beta = 0.f) {
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// Row-major: C[M,N] = A^T[M,K] @ B[K,N], A stored as [K,M]
static void gemm_tn(cublasHandle_t h,
                     const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha = 1.f, float beta = 0.f) {
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K, &alpha, B, N, A, M, &beta, C, N);
}

// ============================================================================
// Elementwise kernels
// ============================================================================

__global__ void k_relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.f);
}

__global__ void k_silu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v / (1.f + __expf(-v)); }
}

__global__ void k_prelu1(float* x, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v > 0.f ? v : v * a[0]; }
}

__global__ void k_prelu_ch(float* x, const float* a, int C, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C * S) { int c = i / S; float v = x[i]; x[i] = v > 0.f ? v : v * a[c]; }
}

__global__ void k_sigmoid(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.f / (1.f + __expf(-x[i]));
}

__global__ void k_tanh(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

__global__ void k_mul(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}

__global__ void k_add(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Bias for row-major [rows, C]: data[r*C+c] += bias[c]
__global__ void k_bias_row(float* d, const float* bias, int rows, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * C) d[i] += bias[i % C];
}

// Bias for channel-first [C, L]: data[c*L+l] += bias[c]
__global__ void k_bias_ch(float* d, const float* bias, int C, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C * L) d[i] += bias[i / L];
}

__global__ void k_zero(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.f;
}

// ============================================================================
// Norms
// ============================================================================

// LayerNorm: x[L,C] → y[L,C], per-row over C
__global__ void k_layernorm(const float* x, float* y,
                            const float* g, const float* b,
                            int L, int C) {
    int r = blockIdx.x;
    if (r >= L) return;
    const float* xr = x + r * C; float* yr = y + r * C;
    extern __shared__ float sm[];

    float s = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) s += xr[i];
    sm[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float mean = sm[0] / C;

    s = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float d = xr[i] - mean; s += d * d;
    }
    sm[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float is = rsqrtf(sm[0] / C + 1e-8f);

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        yr[i] = (xr[i] - mean) * is * g[i] + b[i];
}

// CumulativeLayerNorm: [C,L] channel-first, per-column over C
__global__ void k_clnorm(const float* x, float* y,
                         const float* g, const float* b,
                         int C, int L) {
    int t = blockIdx.x;
    if (t >= L) return;
    extern __shared__ float sm[];

    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += x[c * L + t];
    sm[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float mean = sm[0] / C;

    s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = x[c * L + t] - mean; s += d * d;
    }
    sm[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float is = rsqrtf(sm[0] / C + 1e-8f);

    for (int c = threadIdx.x; c < C; c += blockDim.x)
        y[c * L + t] = (x[c * L + t] - mean) * is * g[c] + b[c];
}

// GroupNorm(num_groups=1): normalize over ALL C*L elements globally.
// Input/output layout: [C, L] (channel-first).
// Two-kernel approach: k_gn1_stats → k_gn1_norm.

// Pass 1: compute sum and sum-of-squares via atomicAdd
__global__ void k_gn1_stats(const float* x, float* stats, int N) {
    extern __shared__ float sm[];  // [2 * blockDim.x]
    int tid = threadIdx.x;
    float s = 0.f, sq = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N;
         i += gridDim.x * blockDim.x) {
        float v = x[i]; s += v; sq += v * v;
    }
    sm[tid] = s; sm[tid + blockDim.x] = sq; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (tid < d) {
            sm[tid] += sm[tid + d];
            sm[tid + blockDim.x] += sm[tid + blockDim.x + d];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&stats[0], sm[0]);
        atomicAdd(&stats[1], sm[blockDim.x]);
    }
}

// Pass 2: normalize x[C,L] → y[C,L] with per-channel affine g[C], b[C]
__global__ void k_gn1_norm(const float* x, float* y,
                           const float* g, const float* b,
                           const float* stats, int C, int L) {
    float N_f = (float)(C * L);
    float mean = stats[0] / N_f;
    float var  = stats[1] / N_f - mean * mean;
    float is   = rsqrtf(var + 1e-8f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * L) return;
    int c = idx / L;
    y[idx] = (x[idx] - mean) * is * g[c] + b[c];
}

// ScaleNorm: y = x / (||x||₂ * D^{-0.5}) * g, per-row
__global__ void k_scalenorm(const float* x, float* y,
                            const float* gp, int L, int D) {
    int r = blockIdx.x;
    if (r >= L) return;
    const float* xr = x + r * D; float* yr = y + r * D;
    float sc = rsqrtf((float)D);
    extern __shared__ float sm[];

    float sq = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) sq += xr[i] * xr[i];
    sm[threadIdx.x] = sq; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float nv = sqrtf(sm[0]) * sc;
    float inv = (nv > 1e-5f) ? (1.f / nv) : 0.f;
    float g = gp[0];

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = xr[i] * inv * g;
}

// InstanceNorm: [C,T] channel-first, per-channel over T
__global__ void k_instnorm(const float* x, float* y,
                           const float* g, const float* b,
                           int C, int T) {
    int c = blockIdx.x;
    if (c >= C) return;
    const float* xc = x + c * T; float* yc = y + c * T;
    extern __shared__ float sm[];

    float s = 0.f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) s += xc[t];
    sm[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float mean = sm[0] / T;

    s = 0.f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float d = xc[t] - mean; s += d * d;
    }
    sm[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
        __syncthreads();
    }
    float is = rsqrtf(sm[0] / T + 1e-5f);
    float gv = g ? g[c] : 1.f;
    float bv = b ? b[c] : 0.f;

    for (int t = threadIdx.x; t < T; t += blockDim.x)
        yc[t] = (xc[t] - mean) * is * gv + bv;
}

// ============================================================================
// Conv kernels
// ============================================================================

// Encoder: [1,T] → [512,L], fused ReLU
__global__ void k_enc_conv(const float* pcm, float* out, const float* w,
                           int T, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kEncDim * L) return;
    int c = idx / L, l = idx % L;
    const float* wc = w + c * kEncKernel;
    float s = 0.f;
    int t0 = l * kEncStride;
    for (int k = 0; k < kEncKernel; k++) s += pcm[t0 + k] * wc[k];
    out[c * L + l] = fmaxf(s, 0.f);
}

// Decoder: ConvTranspose1d [512,L] → [1,T]
__global__ void k_dec_conv(const float* enc, float* pcm, const float* w,
                           int L, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float s = 0.f;
    int lmin = max(0, (t - kEncKernel + kEncStride) / kEncStride);
    int lmax = min(L - 1, t / kEncStride);
    for (int l = lmin; l <= lmax; l++) {
        int k = t - l * kEncStride;
        if (k >= 0 && k < kEncKernel)
            for (int c = 0; c < kEncDim; c++)
                s += enc[c * L + l] * w[c * kEncKernel + k];
    }
    pcm[t] = s;
}

// Depthwise Conv1d: [C,L] → [C,L], weight [C,1,K]
__global__ void k_dwconv(const float* x, float* y, const float* w,
                         int C, int L, int K, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * L) return;
    int c = idx / L, l = idx % L;
    const float* wc = w + c * K;
    float s = 0.f;
    for (int k = 0; k < K; k++) {
        int t = l - pad + k;
        if (t >= 0 && t < L) s += x[c * L + t] * wc[k];
    }
    y[c * L + l] = s;
}

// DDC Conv2d: grouped dilated
__global__ void k_ddc_conv(const float* x, float* y, const float* w,
                           int Co, int cpg, int T, int K, int dil, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Co * T) return;
    int c = idx / T, t = idx % T;
    const float* wc = w + c * cpg * K;
    float s = 0.f;
    for (int g = 0; g < cpg; g++) {
        const float* xi = x + (c * cpg + g) * T;
        const float* wg = wc + g * K;
        for (int k = 0; k < K; k++) {
            int ti = t + (k - pad) * dil;
            if (ti >= 0 && ti < T) s += xi[ti] * wg[k];
        }
    }
    y[c * T + t] = s;
}

// Transpose [R,C] → [C,R]
__global__ void k_transpose(const float* in, float* out, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    int r = idx / C, c = idx % C;
    out[c * R + r] = in[r * C + c];
}

// Concat [C1,T]+[C2,T] → [C1+C2,T]
__global__ void k_cat_ch(const float* a, const float* b, float* o,
                         int C1, int C2, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (C1 + C2) * T) return;
    int c = idx / T, t = idx % T;
    o[idx] = (c < C1) ? a[c * T + t] : b[(c - C1) * T + t];
}

// ============================================================================
// FLASH-specific kernels
// ============================================================================

// Sinusoidal pos embedding: [L,D]
__global__ void k_sinuemb(float* out, const float* inv_freq,
                          const float* scale, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int l = idx / D, d = idx % D;
    float ang = (float)l * inv_freq[d / 2];
    float s = scale[0];
    out[idx] = (d & 1) ? __cosf(ang) * s : __sinf(ang) * s;
}

// OffsetScale: qk[L,D] → 4 outputs, gamma/beta [4,D]
__global__ void k_offset_scale(const float* qk,
                               float* o0, float* o1,
                               float* o2, float* o3,
                               const float* gm, const float* bt,
                               int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int d = idx % D;
    float v = qk[idx];
    o0[idx] = v * gm[d]         + bt[d];
    o1[idx] = v * gm[D + d]     + bt[D + d];
    o2[idx] = v * gm[2*D + d]   + bt[2*D + d];
    o3[idx] = v * gm[3*D + d]   + bt[3*D + d];
}

// Token shift: split [L,D] into two halves, shift first half by 1 time step
// out[t, c] = (c < D/2) ? (t > 0 ? src[t-1, c] : 0) : src[t, c]
__global__ void k_token_shift(const float* src, float* out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int t = idx / D, c = idx % D;
    int half = D / 2;
    if (c < half)
        out[idx] = (t > 0) ? src[(t - 1) * D + c] : 0.f;
    else
        out[idx] = src[idx];
}

// Deinterleave: src[L, 2*D] → a[L, D], b[L, D]
__global__ void k_deinterleave(const float* src, float* a, float* b, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int l = idx / D, d = idx % D;
    a[idx] = src[l * 2 * D + d];
    b[idx] = src[l * 2 * D + D + d];
}

// ReLU squared: x = relu(x)^2
__global__ void k_relu_sq(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = fmaxf(x[i], 0.f); x[i] = v * v; }
}

// Gate: out = (att_u * v) * sigmoid(att_v * u)
__global__ void k_gate(const float* att_u, const float* v,
                       const float* att_v, const float* u,
                       float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (att_u[i] * v[i]) / (1.f + __expf(-(att_v[i] * u[i])));
}

// RoPE: rotate first kRopeDim dims of x[L,D]
__global__ void k_rope(float* x, const float* freqs, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * kRopeFreqs) return;
    int l = idx / kRopeFreqs, p = idx % kRopeFreqs;
    float ang = (float)l * freqs[p];
    float cs, sn;
    __sincosf(ang, &sn, &cs);
    int base = l * D + p * 2;
    float x0 = x[base], x1 = x[base + 1];
    x[base]     = x0 * cs - x1 * sn;
    x[base + 1] = x0 * sn + x1 * cs;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

MossFormer2::MossFormer2() = default;

MossFormer2::~MossFormer2() {
    free_scratch();
    if (cublas_) cublasDestroy(cublas_);
    if (d_weights_) { cudaFree(d_weights_); d_weights_ = nullptr; }
}

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
