/**
 * @file src/sensus/auditus/mossformer2_kernels.cuh
 * @philosophical_role
 *   All MossFormer2 CUDA kernels, made TU-local via static __global__ so
 *   each peer TU (mossformer2.cu / _lifecycle.cu / _flash.cu) gets its own
 *   instantiation. Avoids RDC; satisfies R1 800-line cap by relocation.
 * @serves
 *   mossformer2.cu, mossformer2_lifecycle.cu, mossformer2_flash.cu.
 */
// mossformer2_kernels.cuh — TU-local kernel definitions.

#ifndef DEUSRIDET_SENSUS_AUDITUS_MOSSFORMER2_KERNELS_CUH_
#define DEUSRIDET_SENSUS_AUDITUS_MOSSFORMER2_KERNELS_CUH_

#include <cuda_runtime.h>

#include "mossformer2.h"  // for mf2::kEncDim, kEncKernel, kEncStride, kRopeFreqs

namespace deusridet {

using namespace mf2;

// ============================================================================
// Elementwise kernels
// ============================================================================

static __global__ void k_relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.f);
}

static __global__ void k_silu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v / (1.f + __expf(-v)); }
}

static __global__ void k_prelu1(float* x, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v > 0.f ? v : v * a[0]; }
}

static __global__ void k_prelu_ch(float* x, const float* a, int C, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C * S) { int c = i / S; float v = x[i]; x[i] = v > 0.f ? v : v * a[c]; }
}

static __global__ void k_sigmoid(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.f / (1.f + __expf(-x[i]));
}

static __global__ void k_tanh(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

static __global__ void k_mul(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}

static __global__ void k_add(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Bias for row-major [rows, C]: data[r*C+c] += bias[c]
static __global__ void k_bias_row(float* d, const float* bias, int rows, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * C) d[i] += bias[i % C];
}

// Bias for channel-first [C, L]: data[c*L+l] += bias[c]
static __global__ void k_bias_ch(float* d, const float* bias, int C, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C * L) d[i] += bias[i / L];
}

static __global__ void k_zero(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.f;
}

// ============================================================================
// Norms
// ============================================================================

// LayerNorm: x[L,C] → y[L,C], per-row over C
static __global__ void k_layernorm(const float* x, float* y,
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
static __global__ void k_clnorm(const float* x, float* y,
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
static __global__ void k_gn1_stats(const float* x, float* stats, int N) {
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
static __global__ void k_gn1_norm(const float* x, float* y,
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
static __global__ void k_scalenorm(const float* x, float* y,
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
static __global__ void k_instnorm(const float* x, float* y,
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
static __global__ void k_enc_conv(const float* pcm, float* out, const float* w,
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
static __global__ void k_dec_conv(const float* enc, float* pcm, const float* w,
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
static __global__ void k_dwconv(const float* x, float* y, const float* w,
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
static __global__ void k_ddc_conv(const float* x, float* y, const float* w,
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
static __global__ void k_transpose(const float* in, float* out, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    int r = idx / C, c = idx % C;
    out[c * R + r] = in[r * C + c];
}

// Concat [C1,T]+[C2,T] → [C1+C2,T]
static __global__ void k_cat_ch(const float* a, const float* b, float* o,
                         int C1, int C2, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (C1 + C2) * T) return;
    int c = idx / T, t = idx % T;
    o[idx] = (c < C1) ? a[c * T + t] : b[(c - C1) * T + t];
}

// ============================================================================
// FLASH-specific kernels
// ============================================================================

// Sinusoidal pos embedding: [L,D] — concatenated layout matching PyTorch:
//   out[l, 0..D/2-1] = sin(l * inv_freq[d]) * scale
//   out[l, D/2..D-1]  = cos(l * inv_freq[d]) * scale
static __global__ void k_sinuemb(float* out, const float* inv_freq,
                          const float* scale, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int l = idx / D, d = idx % D;
    int half = D / 2;
    float ang = (float)l * inv_freq[d < half ? d : d - half];
    float s = scale[0];
    out[idx] = (d < half) ? __sinf(ang) * s : __cosf(ang) * s;
}

// OffsetScale: qk[L,D] → 4 outputs, gamma/beta [4,D]
static __global__ void k_offset_scale(const float* qk,
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
static __global__ void k_token_shift(const float* src, float* out, int L, int D) {
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
static __global__ void k_deinterleave(const float* src, float* a, float* b, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int l = idx / D, d = idx % D;
    a[idx] = src[l * 2 * D + d];
    b[idx] = src[l * 2 * D + D + d];
}

// ReLU squared: x = relu(x)^2
static __global__ void k_relu_sq(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = fmaxf(x[i], 0.f); x[i] = v * v; }
}

// Gate: out = (att_u * v) * sigmoid(att_v * u)
static __global__ void k_gate(const float* att_u, const float* v,
                       const float* att_v, const float* u,
                       float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (att_u[i] * v[i]) / (1.f + __expf(-(att_v[i] * u[i])));
}

// RoPE: rotate first kRopeDim dims of x[L,D]
static __global__ void k_rope(float* x, const float* freqs, int L, int D) {
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

}  // namespace deusridet

#endif  // DEUSRIDET_SENSUS_AUDITUS_MOSSFORMER2_KERNELS_CUH_
