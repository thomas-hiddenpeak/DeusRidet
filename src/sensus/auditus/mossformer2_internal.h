/**
 * @file src/sensus/auditus/mossformer2_internal.h
 * @philosophical_role
 *   TU-local helpers shared across mossformer2*.cu peers. Not part of the
 *   public Auditus surface — exists only to satisfy the R1 file-size
 *   discipline without duplicating GEMM wrappers and macros.
 * @serves
 *   mossformer2.cu, mossformer2_lifecycle.cu, mossformer2_flash.cu.
 */
// mossformer2_internal.h — internal helpers (not for outside consumers).

#ifndef DEUSRIDET_SENSUS_AUDITUS_MOSSFORMER2_INTERNAL_H_
#define DEUSRIDET_SENSUS_AUDITUS_MOSSFORMER2_INTERNAL_H_

#include "../../communis/log.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace deusridet {

static constexpr int BLK = 256;
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

#define CK(call) do {                                                       \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess)                                                  \
        LOG_ERROR("MF2", "CUDA %s:%d: %s", __FILE__, __LINE__,             \
                  cudaGetErrorString(_e));                                   \
} while(0)

// Channel-first Conv1d(k=1): Y[Co,L] = W[Co,Ci] @ X[Ci,L]
static inline void gemm_CL(cublasHandle_t h,
                            const float* W, const float* X, float* Y,
                            int Co, int Ci, int L) {
    float a = 1.f, b = 0.f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                L, Co, Ci, &a, X, L, W, Ci, &b, Y, L);
}

// Row-major Linear: Y[M,N] = A[M,K] @ B^T[K,N], B stored [N,K].
static inline void gemm_nt(cublasHandle_t h,
                            const float* A, const float* BT, float* C,
                            int M, int N, int K) {
    float a = 1.f, b = 0.f;
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &a, BT, K, A, K, &b, C, N);
}

// Row-major: C[M,N] = A[M,K] @ B[K,N] (no transpose).
static inline void gemm_nn(cublasHandle_t h,
                            const float* A, const float* B, float* C,
                            int M, int N, int K,
                            float alpha = 1.f, float beta = 0.f) {
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// Row-major: C[M,N] = A^T[M,K] @ B[K,N], A stored as [K,M].
static inline void gemm_tn(cublasHandle_t h,
                            const float* A, const float* B, float* C,
                            int M, int N, int K,
                            float alpha = 1.f, float beta = 0.f) {
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K, &alpha, B, N, A, M, &beta, C, N);
}

}  // namespace deusridet

#endif  // DEUSRIDET_SENSUS_AUDITUS_MOSSFORMER2_INTERNAL_H_
