// gptq.h — GPTQ-Int4 dequantization and linear layer kernels
//
// Supports GPTQ: bits=4, group_size=128, sym=true, desc_act=false
// Weight packing: 8 INT4 nibbles per uint32, LSB-first along K dimension
//
// Dequantization: W[k,n] = scales[k/128, n] * (qw_4bit - 8)
//   where qw_4bit = (qweight[k/8, n] >> (4*(k%8))) & 0xF
//
// Two kernel variants:
//   - GEMV (decode, batch=1): y[1,N] = x[1,K] @ W_q[K,N]  (memory-bound)
//   - GEMM (prefill, batch>1): Y[M,N] = X[M,K] @ W_q[K,N] (compute-bound)
//
// Target: SM87 (Ampere / Jetson AGX Orin)

#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>

namespace deusridet {

// GPTQ configuration (compile-time constants for this model)
constexpr int GPTQ_BITS       = 4;
constexpr int GPTQ_GROUP_SIZE = 128;
constexpr int GPTQ_PACK_FACTOR = 8;    // 32 / 4 = 8 INT4 per uint32
constexpr int GPTQ_ZERO_POINT = 8;     // symmetric quantization center

// GPTQ weight descriptor (non-owning device pointers to GPU-resident weight data)
struct GptqWeight {
    const uint32_t* qweight;   // [K/8, N] packed INT4 (device memory)
    const __half*   scales;    // [K/128, N] FP16 (device memory)
    // qzeros not needed: sym=true, all zeros = 8 (constant)
    int K;                     // input dimension (unpacked)
    int N;                     // output dimension
};

// ============================================================================
// GEMV — Decode path (batch=1, memory-bound)
// ============================================================================
// Computes y[N] = x[K] @ W_q[K,N] with on-the-fly INT4 dequantization
//
// x:      input vector [K] in FP16
// weight: GPTQ weight descriptor
// y:      output vector [N] in FP16
// stream: CUDA stream
void gptq_gemv(const __half* x,
               const GptqWeight& weight,
               __half* y,
               cudaStream_t stream = 0);

// Fused GEMV + residual add: y[n] = (x @ W_q)[n] + residual[n]
void gptq_gemv_add(const __half* x,
                   const GptqWeight& weight,
                   __half* y,
                   const __half* residual,
                   cudaStream_t stream = 0);

// Dual GEMV: gate_proj + up_proj sharing x in SMEM (one load, one launch)
void gptq_dual_gemv(const __half* x,
                    const GptqWeight& w_a, const GptqWeight& w_b,
                    __half* y_a, __half* y_b,
                    cudaStream_t stream = 0);

// ============================================================================
// GEMM — Prefill path (batch>1, compute-bound for larger M)
// ============================================================================
// Computes Y[M,N] = X[M,K] @ W_q[K,N] with on-the-fly INT4 dequantization
//
// X:      input matrix [M, K] in FP16, row-major
// weight: GPTQ weight descriptor
// Y:      output matrix [M, N] in FP16, row-major
// M:      batch dimension (number of tokens)
// stream: CUDA stream
void gptq_gemm(const __half* X,
               const GptqWeight& weight,
               __half* Y,
               int M,
               cudaStream_t stream = 0);

// Batch GEMV for small M: loads weights once, computes M dot products from L2-cached X.
// Faster than tiled GEMM for M ≤ ~32 (avoids BM padding waste, better weight reuse).
void gptq_batch_gemv(const __half* X,
                     const GptqWeight& weight,
                     __half* Y,
                     int M,
                     cudaStream_t stream = 0);

// ============================================================================
// WMMA GEMM — Tensor core path (M>1, primary prefill kernel)
// ============================================================================
// Uses WMMA m16n16k16 FP16 tensor cores with in-SMEM INT4 dequantization.
// Requires: K % 64 == 0, N % 64 == 0 (holds for all MLP projections).
// M is padded to 16 internally; output buffers must be >= ceil16(M) rows.
void gptq_wmma_gemm(const __half* X,
                    const GptqWeight& weight,
                    __half* Y,
                    int M,
                    cudaStream_t stream = 0);

// Fused WMMA GEMM with residual add: residual += W @ X
// Eliminates standalone elementwise_add kernel.
void gptq_wmma_gemm_add(const __half* X,
                          const GptqWeight& weight,
                          __half* residual, int res_N,
                          int M, cudaStream_t stream = 0);

// ============================================================================
// Auto-dispatch: GEMV for M=1, batch GEMV for small M (small N), GEMM otherwise
// ============================================================================
inline void gptq_linear(const __half* X,
                         const GptqWeight& weight,
                         __half* Y,
                         int M,
                         cudaStream_t stream = 0)
{
    if (M == 1) {
        gptq_gemv(X, weight, Y, stream);
    } else if (weight.K % 64 == 0 && weight.N % 64 == 0) {
        // Tensor core WMMA path: dequant INT4 in SMEM + m16n16k16 FP16 mma.
        // Reads INT4 weights once from DRAM → approaching bandwidth limit.
        gptq_wmma_gemm(X, weight, Y, M, stream);
    } else {
        gptq_gemm(X, weight, Y, M, stream);
    }
}

// ============================================================================
// Benchmark utility
// ============================================================================
struct GptqBenchResult {
    float gemv_us;     // microseconds for single GEMV
    float gemm_us;     // microseconds for GEMM at given M
    float gemv_gbps;   // effective bandwidth (GB/s)
    float gemm_tflops; // effective TFLOPS
    int   M;           // batch size used for GEMM
    int   K;
    int   N;
    bool  correct;     // correctness check passed
};

// Run benchmark: correctness check + timing for a given (K, N, M)
GptqBenchResult gptq_benchmark(int K, int N, int M,
                                int warmup_iters = 10,
                                int bench_iters = 50);

} // namespace deusridet
