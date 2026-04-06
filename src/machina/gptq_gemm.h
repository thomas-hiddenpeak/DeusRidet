// gptq_gemm.h — GPTQ INT4 GEMM/GEMV kernels optimized for SM87 (Orin)
//
// Custom GPTQ inference kernels for DeusRidet, replacing Marlin for MLP
// projections. Designed from scratch for SM87's 16-SM topology:
//   - Non-persistent grid (simple 2D launch, no lock-based reduction)
//   - Original GPTQ weight format (no proprietary permutation)
//   - m16n8k16 Tensor Core MMA with register-level INT4→FP16 dequant
//   - Tile sizes chosen for SM87 occupancy and L1 balance
//
// Weight format: original GPTQ (NOT Marlin-permuted)
//   qweight: uint32 [K/8, N] — 8 INT4 packed per uint32 along K
//   scales:  FP16 [K/group_size, N] — one scale per 128 K-rows per N-col
//   dequant: fp16_val = (INT4_val - 8) * scale (symmetric, zero_point=-8)
//
// Requires: N divisible by 64, K divisible by 64, group_size divisible by 64

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace deusridet {

// ============================================================================
// GPTQ INT4 GEMM: C[M,N] = A[M,K] @ dequant(qweight, scales)
// ============================================================================
//
// A:        [M, K] FP16 row-major input activations
// qweight:  [K/8, N] uint32 packed INT4 (original GPTQ format)
// C:        [M, N] FP16 row-major output
// scales:   [K/group_size, N] FP16 quantization scales
// M:        batch size (rows of A). M=0 is a no-op.
// K, N:     weight dimensions. K and N must be divisible by 64.
// group_size: quantization group size (128 for our GPTQ)
void gptq_gemm(
    const __half* A,
    const uint32_t* qweight,
    __half* C,
    const __half* scales,
    int M, int K, int N,
    int group_size = 128,
    cudaStream_t stream = 0);

// ============================================================================
// GPTQ INT4 GEMV: y[N] = x[K] @ dequant(qweight, scales)
// ============================================================================
//
// Specialized M=1 kernel (bandwidth-bound). Each thread block handles a
// slice of N outputs, loading the full K dimension.
void gptq_gemv(
    const __half* x,
    const uint32_t* qweight,
    __half* y,
    const __half* scales,
    int K, int N,
    int group_size = 128,
    cudaStream_t stream = 0);

// ============================================================================
// Standalone benchmark: synthetic data, correctness + timing
// ============================================================================
void bench_new_gptq_kernels();

} // namespace deusridet
