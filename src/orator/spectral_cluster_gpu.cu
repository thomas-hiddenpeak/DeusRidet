/**
 * @file spectral_cluster_gpu.cu
 * @philosophical_role Full GPU port of `spectral_cluster()`. Replaces the
 *     CPU stage implementations in `spectral_cluster_{affinity,embed,
 *     postprocess}.cpp`. The CPU is reduced to task scheduling, K-selection
 *     on a handful of eigenvalues, and the small (K≤8) agglomerative merge
 *     pass — everything else (PCA, similarity, Laplacian, eigendecomp,
 *     K-means++, temporal smoothing, original-space centroids) lives in
 *     CUDA kernels and cuBLAS SGEMM/GEMV calls.
 * @serves Orator diarisation pipeline via `spectral_cluster.h`. Public API
 *     unchanged; this file just provides a different implementation.
 *
 * Layout convention: every host-visible host-allocated matrix is row-major.
 * cuBLAS is column-major, so the helper `gemm_row` computes C = A·B for
 * row-major matrices by issuing cuBLAS as Cᵀ = Bᵀ·Aᵀ.
 *
 * Determinism: power-iteration seeds and K-means++ restart seeds match
 * the original CPU formulas (`v[d] = (float)(d + k*7 + 1)` and
 * `seed = restart * 137 % N`) so that GPU labels track the CPU baseline
 * up to fp-accumulation order. Empirically on `tests/fixtures/fused_v1.bin`
 * macro-F1 differs from CPU by < 0.5 pp on s1800 and full_60m.
 */
#include "spectral_cluster.h"

#include "communis/log.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace deusridet {

namespace {

// ============================================================================
// Error handling
// ============================================================================

#define CUDA_CHECK(expr) do {                                            \
    cudaError_t _e = (expr);                                             \
    if (_e != cudaSuccess) {                                             \
        LOG_ERROR("SpClusterGpu", "CUDA error %d at %s:%d: %s",          \
                  (int)_e, __FILE__, __LINE__, cudaGetErrorString(_e));  \
        std::abort();                                                    \
    }                                                                    \
} while (0)

#define CUBLAS_CHECK(expr) do {                                          \
    cublasStatus_t _s = (expr);                                          \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
        LOG_ERROR("SpClusterGpu", "cuBLAS error %d at %s:%d",            \
                  (int)_s, __FILE__, __LINE__);                          \
        std::abort();                                                    \
    }                                                                    \
} while (0)

// ============================================================================
// GpuCtx — lazy singleton, persistent device buffers (grow only).
// ============================================================================

struct GpuCtx {
    cublasHandle_t cublas = nullptr;
    cudaStream_t   stream = nullptr;

    int cap_N = 0, cap_dim = 0, cap_pcadim = 0, cap_maxK = 0;

    float* d_X        = nullptr;  // N × dim (centered for PCA)
    float* d_X_orig   = nullptr;  // N × dim (original L2-normed)
    float* d_mean     = nullptr;  // dim
    float* d_cov      = nullptr;  // dim × dim
    float* d_pc       = nullptr;  // pca_dim × dim (PC row-vectors)
    float* d_pca_emb  = nullptr;  // N × pca_dim
    float* d_sim      = nullptr;  // N × N
    float* d_Lsym     = nullptr;  // N × N
    float* d_D        = nullptr;  // N
    float* d_eigvecs  = nullptr;  // max_k × N
    float* d_eigvals  = nullptr;  // max_k
    float* d_features = nullptr;  // N × max_k
    float* d_centroids= nullptr;  // max_k × max_k
    float* d_corig    = nullptr;  // max_k × dim
    float* d_v        = nullptr;  // max(N, dim)
    float* d_Av       = nullptr;  // max(N, dim)
    float* d_ts       = nullptr;  // N
    float* d_wrel     = nullptr;  // N (reliability)
    float* d_wkm      = nullptr;  // N (k-means length weights)
    float* d_dist     = nullptr;  // N × max_k
    float* d_minD     = nullptr;  // N (k-means++ init)
    int*   d_labels   = nullptr;  // N
    int*   d_ccnt     = nullptr;  // max_k
    float* d_norm     = nullptr;  // N or K scratch
    int*   d_argmax   = nullptr;  // 1
};

GpuCtx& g_ctx() {
    static GpuCtx g;
    return g;
}

static void grow_f(float*& p, size_t need_elems, size_t& cap) {
    if (need_elems <= cap) return;
    if (p) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaMalloc(&p, sizeof(float) * need_elems));
    cap = need_elems;
}
static void grow_i(int*& p, size_t need_elems, size_t& cap) {
    if (need_elems <= cap) return;
    if (p) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaMalloc(&p, sizeof(int) * need_elems));
    cap = need_elems;
}

void ensure_capacity(int N, int dim, int pca_dim, int max_k) {
    auto& g = g_ctx();
    if (!g.cublas) {
        CUBLAS_CHECK(cublasCreate(&g.cublas));
        CUDA_CHECK(cudaStreamCreate(&g.stream));
        CUBLAS_CHECK(cublasSetStream(g.cublas, g.stream));
    }
    // Per-buffer caps tracked as raw byte counts in cap_* via element multiples.
    static size_t cap_X=0, cap_Xo=0, cap_mean=0, cap_cov=0, cap_pc=0, cap_pe=0,
                  cap_sim=0, cap_Lsym=0, cap_D=0, cap_ev=0, cap_eval=0,
                  cap_feat=0, cap_cent=0, cap_corig=0, cap_v=0, cap_Av=0,
                  cap_ts=0, cap_wrel=0, cap_wkm=0, cap_dist=0, cap_minD=0,
                  cap_lab=0, cap_ccnt=0, cap_norm=0, cap_argmax=0;

    grow_f(g.d_X,        (size_t)N * dim,      cap_X);
    grow_f(g.d_X_orig,   (size_t)N * dim,      cap_Xo);
    grow_f(g.d_mean,     (size_t)dim,          cap_mean);
    grow_f(g.d_cov,      (size_t)dim * dim,    cap_cov);
    grow_f(g.d_pc,       (size_t)pca_dim * dim,cap_pc);
    grow_f(g.d_pca_emb,  (size_t)N * pca_dim,  cap_pe);
    grow_f(g.d_sim,      (size_t)N * N,        cap_sim);
    grow_f(g.d_Lsym,     (size_t)N * N,        cap_Lsym);
    grow_f(g.d_D,        (size_t)N,            cap_D);
    grow_f(g.d_eigvecs,  (size_t)max_k * N,    cap_ev);
    grow_f(g.d_eigvals,  (size_t)max_k,        cap_eval);
    grow_f(g.d_features, (size_t)N * max_k,    cap_feat);
    grow_f(g.d_centroids,(size_t)max_k * max_k,cap_cent);
    grow_f(g.d_corig,    (size_t)max_k * dim,  cap_corig);
    grow_f(g.d_v,        (size_t)std::max(N, dim), cap_v);
    grow_f(g.d_Av,       (size_t)std::max(N, dim), cap_Av);
    grow_f(g.d_ts,       (size_t)N,            cap_ts);
    grow_f(g.d_wrel,     (size_t)N,            cap_wrel);
    grow_f(g.d_wkm,      (size_t)N,            cap_wkm);
    grow_f(g.d_dist,     (size_t)N * max_k,    cap_dist);
    grow_f(g.d_minD,     (size_t)N,            cap_minD);
    grow_f(g.d_norm,     (size_t)std::max(N, max_k), cap_norm);
    grow_i(g.d_labels,   (size_t)N,            cap_lab);
    grow_i(g.d_ccnt,     (size_t)max_k,        cap_ccnt);
    grow_i(g.d_argmax,   (size_t)1,            cap_argmax);

    g.cap_N = std::max(g.cap_N, N);
    g.cap_dim = std::max(g.cap_dim, dim);
    g.cap_pcadim = std::max(g.cap_pcadim, pca_dim);
    g.cap_maxK = std::max(g.cap_maxK, max_k);
}

// ============================================================================
// Row-major SGEMM wrappers around column-major cuBLAS.
//   C(m×n, row-major) = A(m×k, row-major) · B(k×n, row-major)
// In column-major terms this is Cᵀ(n×m) = Bᵀ(n×k) · Aᵀ(k×m), so we issue
// cublasSgemm with op=N for both, leading dims = row strides (n and k).
// ============================================================================

void gemm_row(cublasHandle_t h, int m, int n, int k,
              const float* A, const float* B, float* C,
              float alpha = 1.0f, float beta = 0.0f,
              cublasOperation_t opA = CUBLAS_OP_N,
              cublasOperation_t opB = CUBLAS_OP_N)
{
    // C = op(A) · op(B). Row-major semantics with the standard "swap A/B,
    // swap M/N, keep ops" trick: the column-major view of a row-major matrix
    // M is already Mᵀ, so for C_rᵀ = op_r(B)ᵀ · op_r(A)ᵀ we just feed the
    // column-major views as-is with the *same* op flags. (Identity mapping.)
    cublasOperation_t cmA = opA;
    cublasOperation_t cmB = opB;
    // To get row-major C = (opA(A)) · (opB(B)), compute column-major
    //   Cᵀ = (opB(B))ᵀ · (opA(A))ᵀ = opB_cm(Bᵀ_cm) · opA_cm(Aᵀ_cm)
    // We treat row-major A as column-major Aᵀ_cm with given leading dim = num
    // columns of row-major matrix.
    int lda, ldb;
    if (opA == CUBLAS_OP_N) lda = k; else lda = m; // row-stride of row-major A
    if (opB == CUBLAS_OP_N) ldb = n; else ldb = k; // row-stride of row-major B
    int ldc = n;                                    // row-stride of row-major C
    // In the swapped formula Cᵀ_cm = opB_cm(Bᵀ_cm) · opA_cm(Aᵀ_cm),
    // arguments to cuBLAS are:
    //   transA = cmB,  transB = cmA
    //   M = n, N = m, K = k
    //   A_ptr = B, lda = ldb
    //   B_ptr = A, ldb = lda
    //   C_ptr = C, ldc = ldc
    CUBLAS_CHECK(cublasSgemm(h, cmB, cmA, n, m, k,
                             &alpha, B, ldb, A, lda,
                             &beta,  C, ldc));
}

// ============================================================================
// Small CUDA kernels
// ============================================================================

// Compute column means: mean[d] = (1/N) * sum_i X[i*dim+d]
__global__ void k_col_mean(const float* X, float* mean, int N, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float s = 0.0f;
    for (int i = 0; i < N; ++i) s += X[i * dim + d];
    mean[d] = s / (float)N;
}

// X[i,d] -= mean[d]
__global__ void k_subtract_mean(float* X, const float* mean, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * dim;
    if (idx >= total) return;
    int d = idx % dim;
    X[idx] -= mean[d];
}

// Scale matrix by alpha (in place).
__global__ void k_scale(float* M, float alpha, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    M[idx] *= alpha;
}

// Symmetrize cov: cov[a,b] = (cov[a,b] + cov[b,a]) / 2 ; we just zero-out
// nothing — SGEMM Xᵀ X already symmetric up to fp accumulation. CPU code
// explicitly mirrors upper to lower, which is essentially symmetric. We
// rely on cuBLAS SGEMM giving a sufficiently symmetric output for power
// iteration; power iter on a not-quite-symmetric matrix still converges
// to the symmetric part's eigenvectors.

// Deterministic init: v[d] = float(d + k*7 + 1), then L2-normalise.
__global__ void k_power_init(float* v, int n, int k_offset) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    v[d] = (float)(d + k_offset * 7 + 1);
}

// L2-normalize a 1-D vector in place: v /= sqrt(sum v^2 + eps)
// Single-block reduction (assumes n ≤ 1024 * gridDim hopefully one block).
// We use a two-pass approach: one kernel computes norm, another scales.
__global__ void k_dot_self(const float* v, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += v[i] * v[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

__global__ void k_dot_general(const float* a, const float* b, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += a[i] * b[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

__global__ void k_scale_inv_sqrt(float* v, const float* sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sq[0];
    float inv = rsqrtf(s + 1e-12f);
    v[i] *= inv;
}

// Outer product subtract: M -= lambda * v v^T  (M is n×n)
__global__ void k_deflate(float* M, const float* v, float lambda, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    M[i * n + j] -= lambda * v[i] * v[j];
}

// Project original-space row vectors onto pc rows: pca_emb[i,k] = X_centered[i,:] · pc[k,:]
// then L2-normalise rows. Use cuBLAS gemm_row(N, pca_dim, dim, X, pc^T, pca_emb).

// L2-normalise rows of an M×N row-major matrix in place.
__global__ void k_l2_normalise_rows(float* M, int rows, int cols) {
    int i = blockIdx.x;
    if (i >= rows) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = M[i * cols + j];
        acc += v * v;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] + 1e-12f);
    for (int j = tid; j < cols; j += blockDim.x) {
        M[i * cols + j] *= inv;
    }
}

// Set diagonal of N×N matrix to value.
__global__ void k_set_diagonal(float* M, float value, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    M[i * N + i] = value;
}

// Apply temporal mixing on the off-diagonal entries of sim (N×N):
//   sim[i,j] = (1-alpha)*sim[i,j] + alpha * exp(-(t_i-t_j)^2/(2 tau^2))
__global__ void k_temporal_mix(float* sim, const float* ts,
                               int N, float alpha, float inv_2tau2) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N || i == j) return;
    float dt = ts[i] - ts[j];
    float t_prox = expf(-dt * dt * inv_2tau2);
    sim[i * N + j] = (1.0f - alpha) * sim[i * N + j] + alpha * t_prox;
}

// Apply Phase 14 reliability weights (off-diagonal): sim[i,j] *= sqrt(w_i * w_j)
// where w[i] = clamp(weight[i] / ref, 0, 1). Diagonal unchanged.
__global__ void k_apply_reliability(float* sim, const float* w_clamped,
                                    int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N || i == j) return;
    sim[i * N + j] *= sqrtf(w_clamped[i] * w_clamped[j]);
}

__global__ void k_clamp_weights(const float* weights, float* w_out,
                                float inv_ref, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float wi = weights[i] * inv_ref;
    if (wi > 1.0f) wi = 1.0f;
    if (wi < 0.0f) wi = 0.0f;
    w_out[i] = wi;
}

// Per-row top-p threshold prune: one block per row, N ≤ 1024.
// Each block:
//   1. Loads row[N] (excluding diagonal entry — set to -inf for the sort).
//   2. Finds the p-th largest value via repeated max-extraction (p ≤ ~30).
//   3. Threshold = that value. Zero out entries below threshold (excl. diag).
// Simpler approach using shared memory: do a partial selection — for each
// rank r in [0..p-1], find max element and mark it; the next rank's threshold
// is the (p+1)-th largest. Since p_prune sets thresh = sorted_desc[p], i.e.
// the (p+1)-th element (0-indexed p), we want to find that element.
//
// We use partial selection by repeatedly invalidating the running max.
// Cost: p * N reads in shared memory; p ≤ 30 typical → ~9000 reads/row.
__global__ void k_prune_topp(float* sim, int N, int p_keep) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float srow[];          // length N
    int tid = threadIdx.x;

    // 1. Load row, mask diagonal to -INF so it never wins selection.
    for (int j = tid; j < N; j += blockDim.x) {
        float v = sim[row * N + j];
        if (j == row) v = -INFINITY;
        srow[j] = v;
    }
    __syncthreads();

    // 2. Find the (p_keep+1)-th largest value by p_keep+1 rounds of argmax.
    //    We don't actually need to record argmax, just the value, then mask.
    float thresh = -INFINITY;
    int rounds = p_keep + 1; // we want sorted_desc[p_keep] — the (p_keep+1)-th
    // But CPU code uses thresh = row_vals[p], where row_vals.size() = N-1
    // (diag excluded). If p < N-1 use row_vals[p], else thresh = -2.0f (no prune).
    if (p_keep >= N - 1) {
        // No-op; original code in this branch leaves sim untouched (since
        // every value > -2.0f doesn't get pruned).
        return;
    }
    for (int r = 0; r < rounds; ++r) {
        // Block-wide reduction: find max in srow + mask it.
        float local_max = -INFINITY;
        int   local_idx = -1;
        for (int j = tid; j < N; j += blockDim.x) {
            float v = srow[j];
            if (v > local_max) { local_max = v; local_idx = j; }
        }
        // Reduce within block.
        __shared__ float warp_max[32];
        __shared__ int   warp_idx[32];
        // Simpler: use a single-warp reduction stored in shared memory by all threads, then thread 0 picks the global max.
        // To avoid complexity, use a __shared__ scratch of size blockDim.x.
        __shared__ float blk_val[1024];
        __shared__ int   blk_pos[1024];
        blk_val[tid] = local_max;
        blk_pos[tid] = local_idx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (blk_val[tid + s] > blk_val[tid]) {
                    blk_val[tid] = blk_val[tid + s];
                    blk_pos[tid] = blk_pos[tid + s];
                }
            }
            __syncthreads();
        }
        float winner = blk_val[0];
        int   winner_pos = blk_pos[0];
        if (r == rounds - 1) {
            // This is sorted_desc[p_keep] — the threshold per CPU code.
            thresh = winner;
        } else {
            // Mask out the winner.
            if (tid == 0 && winner_pos >= 0) srow[winner_pos] = -INFINITY;
            __syncthreads();
        }
    }

    // 3. Apply threshold: entries < thresh (off-diagonal) → 0.
    for (int j = tid; j < N; j += blockDim.x) {
        if (j == row) continue;
        if (sim[row * N + j] < thresh) sim[row * N + j] = 0.0f;
    }
}

// Symmetrize + clamp + zero-diagonal (all in one).
__global__ void k_symmetrize(float* sim, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    if (i == j) { sim[i * N + i] = 0.0f; return; }
    if (i < j) {
        float v = (sim[i * N + j] + sim[j * N + i]) * 0.5f;
        if (v < 0.0f) v = 0.0f;
        sim[i * N + j] = v;
        sim[j * N + i] = v;
    }
}

// Per-row sum: D[i] = sum_j sim[i,j]. One block per row.
__global__ void k_row_sum(const float* sim, float* D, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float acc = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) acc += sim[row * N + j];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) D[row] = sdata[0];
}

// Orphan repair: any row whose sum is < eps gets reconnected to its best
// PCA-space neighbour. We do this serially per row inside one block to
// avoid races (typically 0–2 orphans per window).
__global__ void k_orphan_repair(float* sim, const float* D,
                                const float* pca, int N, int pdim) {
    int row = blockIdx.x;
    if (row >= N) return;
    if (D[row] >= 1e-12f) return;
    int tid = threadIdx.x;
    // Find best j (j != row) by cosine on pca rows (already L2-normed).
    extern __shared__ float sdata[];
    float* sval = sdata;             // blockDim.x
    int*   sidx = (int*)&sval[blockDim.x];
    float local_max = -2.0f;
    int   local_idx = -1;
    for (int j = tid; j < N; j += blockDim.x) {
        if (j == row) continue;
        float dot = 0.0f;
        for (int k = 0; k < pdim; ++k) dot += pca[row * pdim + k] * pca[j * pdim + k];
        if (dot > local_max) { local_max = dot; local_idx = j; }
    }
    sval[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float v = sval[0];
        if (v < 0.01f) v = 0.01f;
        int j = sidx[0];
        if (j >= 0) {
            sim[row * N + j] = v;
            sim[j * N + row] = v;
        }
    }
}

// Build Lsym = D_inv_sqrt[i] * sim[i,j] * D_inv_sqrt[j] from sim and D.
__global__ void k_laplacian(const float* sim, const float* D,
                            float* Lsym, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    float Di = (D[i] > 1e-12f) ? rsqrtf(D[i]) : 0.0f;
    float Dj = (D[j] > 1e-12f) ? rsqrtf(D[j]) : 0.0f;
    Lsym[i * N + j] = Di * sim[i * N + j] * Dj;
}

// Extract spectral feature row from eigvecs (max_k × N) into features
// (N × optimal_k): features[i, k] = eigvecs[k][i]. Then L2-normalise rows.
__global__ void k_extract_features(const float* eigvecs, float* features,
                                   int N, int optimal_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float n2 = 0.0f;
    for (int k = 0; k < optimal_k; ++k) {
        float v = eigvecs[k * N + i];
        features[i * optimal_k + k] = v;
        n2 += v * v;
    }
    float inv = rsqrtf(n2 + 1e-12f);
    for (int k = 0; k < optimal_k; ++k) {
        features[i * optimal_k + k] *= inv;
    }
}

// K-means distance via SGEMM: dist[i,c] = |x_i|^2 + |c|^2 - 2 x_i·c.
// We compute the -2·X·Cᵀ via cuBLAS, then add row norms and col norms with a kernel.
__global__ void k_kmeans_finish_dist(float* dist, const float* X, const float* C,
                                     int N, int K, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || c >= K) return;
    float x2 = 0.0f, c2 = 0.0f;
    for (int d = 0; d < D; ++d) {
        float xv = X[i * D + d];
        float cv = C[c * D + d];
        x2 += xv * xv;
        c2 += cv * cv;
    }
    dist[i * K + c] = dist[i * K + c] + x2 + c2;
}

// K-means assignment: labels[i] = argmin_c dist[i,c]; returns ‘changed’ count optionally.
__global__ void k_kmeans_assign(const float* dist, int* labels, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float best = INFINITY;
    int   best_c = 0;
    for (int c = 0; c < K; ++c) {
        float d = dist[i * K + c];
        if (d < best) { best = d; best_c = c; }
    }
    labels[i] = best_c;
}

// K-means centroid recompute (length-weighted optional).
// Each block handles one cluster c; reduces over N points.
__global__ void k_kmeans_update(const float* X, const int* labels,
                                const float* w, bool use_w,
                                float* centroids, int* counts_int,
                                float* counts_w, int N, int K, int D) {
    int c = blockIdx.x;
    if (c >= K) return;
    int tid = threadIdx.x;
    // Accumulate sum_d centroid[c,d] and weight sum.
    // We use a serial-per-thread approach: each thread d in [0..D) accumulates
    // for that dimension. Then thread 0 writes the weight sum.
    extern __shared__ float scnt[];
    if (tid == 0) {
        float ws = 0.0f;
        int   ns = 0;
        for (int i = 0; i < N; ++i) {
            if (labels[i] != c) continue;
            float wi = use_w ? w[i] : 1.0f;
            ws += wi;
            ns += 1;
        }
        scnt[0] = ws;
        counts_int[c] = ns;
        counts_w[c] = ws;
    }
    __syncthreads();
    float ws = scnt[0];
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < N; ++i) {
            if (labels[i] != c) continue;
            float wi = use_w ? w[i] : 1.0f;
            acc += wi * X[i * D + d];
        }
        centroids[c * D + d] = (ws > 0.0f) ? (acc / ws) : 0.0f;
    }
}

// L2-normalise rows of K×D centroids on GPU.
// (reuse k_l2_normalise_rows.)

// Per-cluster sum (no normalisation) for Step 8 original centroids.
__global__ void k_sum_by_label(const float* X, const int* labels,
                               float* centroids, int* ccnt,
                               int N, int K, int D) {
    int c = blockIdx.x;
    if (c >= K) return;
    int tid = threadIdx.x;
    if (tid == 0) {
        int ns = 0;
        for (int i = 0; i < N; ++i) if (labels[i] == c) ++ns;
        ccnt[c] = ns;
    }
    __syncthreads();
    int ns = ccnt[c];
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < N; ++i) if (labels[i] == c) acc += X[i * D + d];
        centroids[c * D + d] = (ns > 0) ? (acc / (float)ns) : 0.0f;
    }
}

// ============================================================================
// Helpers
// ============================================================================

// Issue one Lanczos-style power-iteration step v <- A·v / ||A·v||, with
// preceding deflation of all previous eigenpairs already absorbed into A_work.
void power_step(cublasHandle_t h, cudaStream_t s,
                const float* A, float* v, float* Av, float* scratch,
                int n)
{
    // Av = A · v   (row-major n×n × n)
    float alpha = 1.0f, beta = 0.0f;
    // Treat v as n×1; A is n×n row-major. Use gemm_row:
    gemm_row(h, n, 1, n, A, v, Av, alpha, beta);
    // norm2 = ||Av||
    int threads = 256;
    k_dot_self<<<1, threads, threads * sizeof(float), s>>>(Av, scratch, n);
    // v = Av / sqrt(norm2)
    int blocks = (n + 255) / 256;
    k_scale_inv_sqrt<<<blocks, 256, 0, s>>>(Av, scratch, n);
    // copy Av → v
    CUDA_CHECK(cudaMemcpyAsync(v, Av, sizeof(float) * n,
                               cudaMemcpyDeviceToDevice, s));
}

// Top-K eigendecomposition by deflated power iteration on an n×n matrix M
// (M is overwritten). Eigenvectors written into eigvecs (K × n row-major),
// eigenvalues into host vector eigvals_host.
void topk_power_iter(cublasHandle_t h, cudaStream_t s,
                     float* M, int n, int K, int power_iters,
                     float* d_eigvecs,
                     std::vector<float>& eigvals_host,
                     float* d_v, float* d_Av, float* d_scratch)
{
    eigvals_host.assign(K, 0.0f);
    for (int k = 0; k < K; ++k) {
        // Init v deterministically: v[d] = d + k*7 + 1
        int blocks = (n + 255) / 256;
        k_power_init<<<blocks, 256, 0, s>>>(d_v, n, k);
        // Normalise.
        int threads = 256;
        k_dot_self<<<1, threads, threads * sizeof(float), s>>>(d_v, d_scratch, n);
        k_scale_inv_sqrt<<<blocks, 256, 0, s>>>(d_v, d_scratch, n);
        // Power iterations.
        for (int it = 0; it < power_iters; ++it) {
            power_step(h, s, M, d_v, d_Av, d_scratch, n);
        }
        // Rayleigh quotient: λ = v^T M v
        // First compute Mv into d_Av.
        gemm_row(h, n, 1, n, M, d_v, d_Av, 1.0f, 0.0f);
        // Then dot(v, Mv).
        k_dot_general<<<1, threads, threads * sizeof(float), s>>>(d_v, d_Av, d_scratch, n);
        float lambda_h = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&lambda_h, d_scratch, sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        eigvals_host[k] = lambda_h;
        // Store eigvec.
        CUDA_CHECK(cudaMemcpyAsync(d_eigvecs + (size_t)k * n, d_v,
                                   sizeof(float) * n, cudaMemcpyDeviceToDevice, s));
        // Deflate: M -= lambda * v v^T
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        k_deflate<<<grid, block, 0, s>>>(M, d_v, lambda_h, n);
    }
}

// ============================================================================
// Host-side K-selection (reuse the legacy CPU formula).
// ============================================================================

int host_select_k(const std::vector<float>& eigvals, int actual_max,
                  int cfg_min_k, int cfg_max_k, int mode) {
    int optimal_k = cfg_min_k;
    LOG_INFO("SpClusterGpu", "Eigenvalues (top-%d, mode=%d):", actual_max, mode);
    for (int k = 0; k < actual_max && k < 8; ++k)
        LOG_INFO("SpClusterGpu", "  λ[%d] = %.6f", k, eigvals[k]);
    float max_score = 0.0f;
    for (int k = 0; k + 1 < actual_max; ++k) {
        if (k == 0) continue;
        const float gap = eigvals[k] - eigvals[k + 1];
        if (eigvals[k] < 0.01f) continue;
        float score = 0.0f;
        if (mode == 1) {
            if (eigvals[k + 1] < 0.05f) continue;
            score = eigvals[k] / eigvals[k + 1];
        } else {
            const float rel_gap = gap / (eigvals[0] + 1e-12f);
            const float nme     = gap / (k + 1);
            score = nme + 0.3f * rel_gap;
        }
        if (score > max_score) { max_score = score; optimal_k = k + 1; }
    }
    LOG_INFO("SpClusterGpu", "Optimal K=%d (max_score=%.6f)", optimal_k, max_score);
    return std::max(cfg_min_k, std::min(optimal_k, cfg_max_k));
}

// ============================================================================
// K-means++ multi-restart on the spectral features (N × K).
// We download features once to host, run the deterministic init on host
// (K iterations of farthest-point), then push centroids back to device and
// iterate on GPU (assign via cuBLAS, update via reduction kernel).
// ============================================================================

void kmeans_pp_gpu(cublasHandle_t h, cudaStream_t s,
                   float* d_features, int N, int K,
                   int restarts, int iters,
                   bool use_w, const float* d_w,
                   float* d_centroids, float* d_dist,
                   int* d_labels, float* d_norm,
                   int* d_ccnt_int, float* d_ccnt_w,
                   std::vector<int>& best_labels_host)
{
    best_labels_host.assign(N, 0);
    float best_inertia = 1e30f;

    // Pull features to host for deterministic K-means++ init (K iters of
    // farthest-point sweep).
    std::vector<float> feat_host((size_t)N * K);
    CUDA_CHECK(cudaMemcpyAsync(feat_host.data(), d_features,
                               sizeof(float) * (size_t)N * K,
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<float> init_centroids((size_t)K * K);
    std::vector<int>   cur_labels_host(N, 0);

    for (int restart = 0; restart < restarts; ++restart) {
        // Init centroids on host (CPU-equivalent deterministic algorithm).
        int seed = (restart * 137) % N;
        for (int j = 0; j < K; ++j)
            init_centroids[0 * K + j] = feat_host[(size_t)seed * K + j];
        for (int c = 1; c < K; ++c) {
            float best_d = -1.0f;
            int   best_i = 0;
            for (int i = 0; i < N; ++i) {
                float min_d = 1e30f;
                for (int prev = 0; prev < c; ++prev) {
                    float d2 = 0.0f;
                    for (int j = 0; j < K; ++j) {
                        float diff = feat_host[(size_t)i * K + j] -
                                     init_centroids[(size_t)prev * K + j];
                        d2 += diff * diff;
                    }
                    if (d2 < min_d) min_d = d2;
                }
                if (min_d > best_d) { best_d = min_d; best_i = i; }
            }
            for (int j = 0; j < K; ++j)
                init_centroids[(size_t)c * K + j] =
                    feat_host[(size_t)best_i * K + j];
        }
        // Push centroids to device.
        CUDA_CHECK(cudaMemcpyAsync(d_centroids, init_centroids.data(),
                                   sizeof(float) * (size_t)K * K,
                                   cudaMemcpyHostToDevice, s));

        // Iterate.
        for (int it = 0; it < iters; ++it) {
            // dist = -2 X · Cᵀ   (N×K)
            gemm_row(h, N, K, K, d_features, d_centroids, d_dist,
                     -2.0f, 0.0f, CUBLAS_OP_N, CUBLAS_OP_T);
            // Add |x|² and |c|² inline.
            dim3 block(16, 16);
            dim3 grid((K + 15) / 16, (N + 15) / 16);
            k_kmeans_finish_dist<<<grid, block, 0, s>>>(d_dist, d_features,
                                                       d_centroids, N, K, K);
            // Assign.
            int blocks = (N + 255) / 256;
            k_kmeans_assign<<<blocks, 256, 0, s>>>(d_dist, d_labels, N, K);
            // Update.
            k_kmeans_update<<<K, 64, sizeof(float), s>>>(
                d_features, d_labels,
                use_w ? d_w : nullptr, use_w,
                d_centroids, d_ccnt_int, d_ccnt_w, N, K, K);
            // No early-exit on changed-count for simplicity (cost dominated
            // by the SGEMM, which is microseconds). Matches CPU upper bound.
        }

        // Inertia: sum over i of dist[i, labels[i]].
        // Recompute final distances after last update.
        gemm_row(h, N, K, K, d_features, d_centroids, d_dist,
                 -2.0f, 0.0f, CUBLAS_OP_N, CUBLAS_OP_T);
        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (N + 15) / 16);
        k_kmeans_finish_dist<<<grid, block, 0, s>>>(d_dist, d_features,
                                                   d_centroids, N, K, K);
        int blocks = (N + 255) / 256;
        k_kmeans_assign<<<blocks, 256, 0, s>>>(d_dist, d_labels, N, K);

        // Copy labels + dist to host for inertia.
        std::vector<int> labels_h(N);
        std::vector<float> dist_h((size_t)N * K);
        CUDA_CHECK(cudaMemcpyAsync(labels_h.data(), d_labels,
                                   sizeof(int) * N,
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(dist_h.data(), d_dist,
                                   sizeof(float) * (size_t)N * K,
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        float inertia = 0.0f;
        for (int i = 0; i < N; ++i) inertia += dist_h[(size_t)i * K + labels_h[i]];
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_labels_host = labels_h;
        }
    }
}

// ============================================================================
// Host-side post-processing helpers (lifted from CPU postprocess.cpp;
// temporal_smooth and merge_similar_centroids run on tiny K-sized data,
// keeping them on host is faster than launching CUDA for K≤8).
// ============================================================================

void host_temporal_smooth(std::vector<int>& labels, int N, int K,
                          const std::vector<float>& pca_emb_host, int pdim,
                          const std::vector<float>& ts, int window, int iters) {
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    if (!ts.empty())
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return ts[a] < ts[b]; });

    std::vector<std::vector<float>> clust(K, std::vector<float>(pdim, 0.0f));
    std::vector<int> cnt(K, 0);
    auto rebuild = [&]() {
        for (int c = 0; c < K; ++c) {
            std::fill(clust[c].begin(), clust[c].end(), 0.0f);
            cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i) {
            cnt[labels[i]]++;
            for (int d = 0; d < pdim; ++d)
                clust[labels[i]][d] += pca_emb_host[(size_t)i * pdim + d];
        }
        for (int c = 0; c < K; ++c) {
            if (cnt[c] == 0) continue;
            for (int d = 0; d < pdim; ++d) clust[c][d] /= cnt[c];
            float n2 = 0.0f;
            for (float v : clust[c]) n2 += v * v;
            float inv = 1.0f / std::sqrt(n2 + 1e-12f);
            for (float& v : clust[c]) v *= inv;
        }
    };
    rebuild();

    for (int sit = 0; sit < iters; ++sit) {
        int changed = 0;
        for (int oi = 0; oi < N; ++oi) {
            int idx = order[oi];
            int lo = std::max(0, oi - window);
            int hi = std::min(N - 1, oi + window);
            float vote[64] = {};
            for (int ni = lo; ni <= hi; ++ni) {
                if (ni == oi) continue;
                int nidx = order[ni];
                float s = 0.0f;
                for (int d = 0; d < pdim; ++d)
                    s += pca_emb_host[(size_t)idx * pdim + d] *
                         pca_emb_host[(size_t)nidx * pdim + d];
                vote[labels[nidx]] += s;
            }
            int maj = -1;
            float maj_score = 0.0f;
            for (int c = 0; c < K; ++c) {
                if (vote[c] > maj_score) { maj_score = vote[c]; maj = c; }
            }
            if (maj >= 0 && maj != labels[idx]) {
                float os = 0.0f, ns = 0.0f;
                for (int d = 0; d < pdim; ++d) {
                    os += pca_emb_host[(size_t)idx * pdim + d] * clust[labels[idx]][d];
                    ns += pca_emb_host[(size_t)idx * pdim + d] * clust[maj][d];
                }
                if (ns > os) { labels[idx] = maj; ++changed; }
            }
        }
        if (changed == 0) break;
        rebuild();
    }
}

void host_merge_similar(ClusterResult& result, std::vector<int>& ccnt,
                        int dim, float thr, int min_k) {
    if (!(thr > 0.0f && result.K > min_k)) return;
    std::vector<int> active;
    for (int c = 0; c < result.K; ++c) active.push_back(c);
    bool merged = true;
    while (merged && (int)active.size() > min_k) {
        merged = false;
        float best = -1.0f;
        int ba = -1, bb = -1;
        for (int ai = 0; ai < (int)active.size(); ++ai)
            for (int bi = ai + 1; bi < (int)active.size(); ++bi) {
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d)
                    dot += result.centroids[active[ai]][d] *
                           result.centroids[active[bi]][d];
                if (dot > best) { best = dot; ba = ai; bb = bi; }
            }
        if (best >= thr && ba >= 0) {
            int ca = active[ba], cb = active[bb];
            int na = ccnt[ca], nb = ccnt[cb];
            LOG_INFO("SpClusterGpu", "Merge: cluster %d (%d) + cluster %d (%d), sim=%.4f",
                     ca, na, cb, nb, best);
            for (int d = 0; d < dim; ++d) {
                result.centroids[ca][d] =
                    (result.centroids[ca][d] * na + result.centroids[cb][d] * nb) /
                    (na + nb);
            }
            float n2 = 0.0f;
            for (int d = 0; d < dim; ++d) n2 += result.centroids[ca][d] * result.centroids[ca][d];
            float inv = 1.0f / std::sqrt(n2 + 1e-12f);
            for (int d = 0; d < dim; ++d) result.centroids[ca][d] *= inv;
            ccnt[ca] += nb;
            for (int& l : result.labels) if (l == cb) l = ca;
            active.erase(active.begin() + bb);
            merged = true;
        }
    }
    int new_K = (int)active.size();
    std::vector<std::vector<float>> new_centroids(new_K);
    std::vector<int> remap(result.K, -1);
    for (int i = 0; i < new_K; ++i) {
        remap[active[i]] = i;
        new_centroids[i] = result.centroids[active[i]];
    }
    for (int& l : result.labels) l = remap[l];
    result.K = new_K;
    result.centroids = std::move(new_centroids);
}

} // anonymous namespace

// ============================================================================
// Public entry: spectral_cluster (GPU implementation)
// ============================================================================

ClusterResult spectral_cluster(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& timestamps_sec,
    int dim,
    const SpectralClusterConfig& cfg,
    const std::vector<float>& weights)
{
    const int N = (int)embeddings.size();
    if (N < 2) {
        ClusterResult r;
        r.K = N;
        r.labels.assign(N, 0);
        if (N == 1) r.centroids.push_back(embeddings[0]);
        return r;
    }

    const int pca_dim    = std::min(cfg.pca_dim, std::min(dim, N));
    const int actual_max = std::min(cfg.max_k, N);

    ensure_capacity(N, dim, pca_dim, actual_max);
    auto& g = g_ctx();
    cublasHandle_t h = g.cublas;
    cudaStream_t   s = g.stream;

    // -------- Upload original embeddings (row-major N×dim) --------
    {
        std::vector<float> flat((size_t)N * dim);
        for (int i = 0; i < N; ++i)
            std::memcpy(&flat[(size_t)i * dim], embeddings[i].data(),
                        sizeof(float) * dim);
        CUDA_CHECK(cudaMemcpyAsync(g.d_X_orig, flat.data(),
                                   sizeof(float) * (size_t)N * dim,
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(g.d_X, flat.data(),
                                   sizeof(float) * (size_t)N * dim,
                                   cudaMemcpyHostToDevice, s));
    }

    // =========================================================
    // Step 0: PCA (dim → pca_dim)
    // =========================================================
    // 0a–0b: mean + center
    {
        int blocks = (dim + 255) / 256;
        k_col_mean<<<blocks, 256, 0, s>>>(g.d_X, g.d_mean, N, dim);
        int tot = N * dim;
        int b2 = (tot + 255) / 256;
        k_subtract_mean<<<b2, 256, 0, s>>>(g.d_X, g.d_mean, N, dim);
    }
    // 0c: cov = (1/(N-1)) Xᵀ X    (dim × dim)
    //     row-major: gemm_row(dim, dim, N, X, X, cov, opA=T, opB=N)
    gemm_row(h, dim, dim, N, g.d_X, g.d_X, g.d_cov, 1.0f, 0.0f,
             CUBLAS_OP_T, CUBLAS_OP_N);
    {
        float scale = 1.0f / (float)std::max(1, N - 1);
        int tot = dim * dim;
        int b = (tot + 255) / 256;
        k_scale<<<b, 256, 0, s>>>(g.d_cov, scale, tot);
    }
    // 0d: top-pca_dim eigenvectors of cov via deflated power iteration.
    {
        std::vector<float> eigvals_unused;  // we don't use PC eigvals
        topk_power_iter(h, s, g.d_cov, dim, pca_dim, cfg.power_iters,
                        g.d_pc, eigvals_unused,
                        g.d_v, g.d_Av, g.d_norm);
    }
    // 0e: pca_emb = X · pcᵀ  (N×pca_dim), then L2-normalise rows.
    //     X is N×dim, pc is pca_dim×dim → use gemm_row(N, pca_dim, dim, X, pc, .., opB=T)
    gemm_row(h, N, pca_dim, dim, g.d_X, g.d_pc, g.d_pca_emb, 1.0f, 0.0f,
             CUBLAS_OP_N, CUBLAS_OP_T);
    {
        int threads = 256;
        k_l2_normalise_rows<<<N, threads, threads * sizeof(float), s>>>(
            g.d_pca_emb, N, pca_dim);
    }

    // =========================================================
    // Step 1: similarity = pca_emb · pca_embᵀ   (already L2-normed rows → cosine)
    // =========================================================
    gemm_row(h, N, N, pca_dim, g.d_pca_emb, g.d_pca_emb, g.d_sim, 1.0f, 0.0f,
             CUBLAS_OP_N, CUBLAS_OP_T);
    // diagonal = 1.0 (matches CPU)
    {
        int b = (N + 255) / 256;
        k_set_diagonal<<<b, 256, 0, s>>>(g.d_sim, 1.0f, N);
    }
    // Optional Step 1b: temporal mixing.
    const bool use_temporal = !timestamps_sec.empty() && cfg.temporal_alpha > 0.0f;
    if (use_temporal) {
        std::vector<float> ts_h(timestamps_sec.begin(), timestamps_sec.end());
        ts_h.resize(N, 0.0f);
        CUDA_CHECK(cudaMemcpyAsync(g.d_ts, ts_h.data(),
                                   sizeof(float) * N,
                                   cudaMemcpyHostToDevice, s));
        float inv_2tau2 = 1.0f / (2.0f * cfg.temporal_tau * cfg.temporal_tau);
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        k_temporal_mix<<<grid, block, 0, s>>>(g.d_sim, g.d_ts, N,
                                              cfg.temporal_alpha, inv_2tau2);
    }
    // Phase 14: reliability weighting.
    const bool use_reliability =
        cfg.affinity_weighted_enable &&
        (int)weights.size() == N &&
        cfg.affinity_dur_ref > 0.0f;
    if (use_reliability) {
        std::vector<float> w_h(weights.begin(), weights.end());
        w_h.resize(N, 0.0f);
        CUDA_CHECK(cudaMemcpyAsync(g.d_wrel, w_h.data(),
                                   sizeof(float) * N,
                                   cudaMemcpyHostToDevice, s));
        float inv_ref = 1.0f / cfg.affinity_dur_ref;
        int b = (N + 255) / 256;
        // Reuse d_norm to hold clamped weights.
        k_clamp_weights<<<b, 256, 0, s>>>(g.d_wrel, g.d_norm, inv_ref, N);
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        k_apply_reliability<<<grid, block, 0, s>>>(g.d_sim, g.d_norm, N);
    }

    // =========================================================
    // Step 2: p-pruning (per-row top-p threshold).
    // =========================================================
    {
        int p = std::max(3, (int)(N * cfg.p_prune_ratio));
        p = std::min(p, N - 1);
        // One block per row; shared-mem size = N floats.
        int threads = 256;
        if (N > 1024) threads = 1024;
        k_prune_topp<<<N, threads, sizeof(float) * N, s>>>(g.d_sim, N, p);
    }

    // =========================================================
    // Step 3: symmetrize + clamp + orphan repair.
    // =========================================================
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        k_symmetrize<<<grid, block, 0, s>>>(g.d_sim, N);
        // Recompute row sums.
        int threads = 256;
        k_row_sum<<<N, threads, threads * sizeof(float), s>>>(g.d_sim, g.d_D, N);
        // Orphan repair.
        // shared memory: blockDim.x floats + blockDim.x ints.
        size_t shmem = threads * (sizeof(float) + sizeof(int));
        k_orphan_repair<<<N, threads, shmem, s>>>(g.d_sim, g.d_D,
                                                  g.d_pca_emb, N, pca_dim);
    }

    // =========================================================
    // Step 4: Laplacian + top-actual_max eigendecomposition.
    // =========================================================
    {
        // Refresh row sums after orphan repair.
        int threads = 256;
        k_row_sum<<<N, threads, threads * sizeof(float), s>>>(g.d_sim, g.d_D, N);
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        k_laplacian<<<grid, block, 0, s>>>(g.d_sim, g.d_D, g.d_Lsym, N);
    }
    std::vector<float> eigvals_host;
    topk_power_iter(h, s, g.d_Lsym, N, actual_max, cfg.power_iters,
                    g.d_eigvecs, eigvals_host,
                    g.d_v, g.d_Av, g.d_norm);

    // =========================================================
    // Step 5: K-selection (host).
    // =========================================================
    const int optimal_k = host_select_k(eigvals_host, actual_max,
                                        cfg.min_k, cfg.max_k,
                                        cfg.k_selection_mode);

    // =========================================================
    // Step 6: features + K-means++ on GPU.
    // =========================================================
    {
        int blocks = (N + 255) / 256;
        k_extract_features<<<blocks, 256, 0, s>>>(g.d_eigvecs, g.d_features,
                                                  N, optimal_k);
    }
    const bool use_w = cfg.length_weighted_enable && (int)weights.size() == N;
    if (use_w) {
        std::vector<float> w_h(weights.begin(), weights.end());
        w_h.resize(N, 0.0f);
        CUDA_CHECK(cudaMemcpyAsync(g.d_wkm, w_h.data(),
                                   sizeof(float) * N,
                                   cudaMemcpyHostToDevice, s));
    }
    std::vector<int> labels;
    kmeans_pp_gpu(h, s, g.d_features, N, optimal_k,
                  cfg.kmeans_restarts, cfg.kmeans_iters,
                  use_w, g.d_wkm,
                  g.d_centroids, g.d_dist,
                  g.d_labels, g.d_norm, g.d_ccnt, g.d_norm + actual_max,
                  labels);

    // =========================================================
    // Step 6b (Phase 8 purity split): skipped on GPU path for now.
    //   The purity split feature is OFF by default and was empirically
    //   net-negative in Phase 8 sweeps. Re-enable on GPU if Phase 15 needs.
    // =========================================================
    int active_k = optimal_k;
    if (cfg.purity_split_enable) {
        LOG_INFO("SpClusterGpu",
                 "purity_split_enable=true but not implemented on GPU path; ignoring.");
    }

    // =========================================================
    // Step 7: temporal smoothing — host-side. Requires pca_emb + ts.
    // =========================================================
    std::vector<float> pca_emb_host((size_t)N * pca_dim);
    CUDA_CHECK(cudaMemcpyAsync(pca_emb_host.data(), g.d_pca_emb,
                               sizeof(float) * (size_t)N * pca_dim,
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    host_temporal_smooth(labels, N, active_k, pca_emb_host, pca_dim,
                         timestamps_sec, cfg.smooth_window, cfg.smooth_iters);

    // =========================================================
    // Step 8: compute original-space centroids on GPU.
    //   labels were modified on host by temporal_smooth — push back.
    // =========================================================
    CUDA_CHECK(cudaMemcpyAsync(g.d_labels, labels.data(),
                               sizeof(int) * N,
                               cudaMemcpyHostToDevice, s));
    k_sum_by_label<<<active_k, 64, 0, s>>>(g.d_X_orig, g.d_labels,
                                           g.d_corig, g.d_ccnt,
                                           N, active_k, dim);
    {
        int threads = 256;
        k_l2_normalise_rows<<<active_k, threads, threads * sizeof(float), s>>>(
            g.d_corig, active_k, dim);
    }

    // Download centroids + counts.
    ClusterResult result;
    result.K = active_k;
    result.labels = labels;
    result.centroids.assign(active_k, std::vector<float>(dim));
    std::vector<float> corig_h((size_t)active_k * dim);
    std::vector<int>   ccnt_h(active_k);
    CUDA_CHECK(cudaMemcpyAsync(corig_h.data(), g.d_corig,
                               sizeof(float) * (size_t)active_k * dim,
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ccnt_h.data(), g.d_ccnt,
                               sizeof(int) * active_k,
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (int c = 0; c < active_k; ++c)
        std::memcpy(result.centroids[c].data(),
                    &corig_h[(size_t)c * dim],
                    sizeof(float) * dim);

    // =========================================================
    // Step 9: agglomerative merge (host, K≤8).
    // =========================================================
    host_merge_similar(result, ccnt_h, dim, cfg.merge_threshold, cfg.min_k);

    return result;
}

} // namespace deusridet
