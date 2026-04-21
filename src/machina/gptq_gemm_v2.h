/**
 * @file gptq_gemm_v2.h
 * @philosophical_role Declaration of GPTQ INT4 GEMM v2 (Marlin-format weights, SM87-tuned). The second Machina GEMM dialect: quantized weights, higher arithmetic intensity per byte loaded.
 * @serves Machina forward pass for GPTQ-Int4 models.
 */
// gptq_gemm_v2.h — GPTQ INT4 GEMM v2: Marlin-format weights, SM87-tuned
//
// Absorbs Marlin's key advantages (pre-permuted weight layout, 4-stage
// cp.async pipeline, SMEM-B, ldmatrix) into a clean non-persistent kernel.
// Reads weights in Marlin tile format (after repack_all_marlin).
//
// Differences from Marlin:
//   - Non-persistent 2D CTA grid: each CTA owns one output tile, no global reduction
//   - No slice/stripe scheduling, no lock-based barriers
//   - Simpler control flow → fewer registers for bookkeeping, easier to fuse ops
//   - SM87-tuned: tile/SMEM sizes validated for 128KB unified L1/SMEM
//
// Tile: BM=16-64, BN=128, BK=128, 256 threads, 4-stage pipeline
// ~97KB SMEM (tmb=4), 31KB L1
//
// Weight format: Marlin-repacked (same as marlin.cu repack_gptq_to_marlin)
//   qweight: [K/16, 2*N] uint32, pre-permuted for m16n8k16 MMA fragments
//   scales:  [K/128, N] FP16, column-permuted in 64-element blocks
//
// Requires: K % 128 == 0, N % 128 == 0, group_size = 128

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace deusridet {

// GPTQ INT4 GEMM (Marlin-format weights): C[M,N] = A[M,K] @ dequant(B,scales)
//
// A:        [M, K] FP16 row-major input activations
// B:        [K/16, 2*N] uint32 Marlin-permuted packed INT4
// C:        [M, N] FP16 row-major output
// scales:   [K/128, N] FP16 Marlin-permuted scales
// M:        batch size (rows of A). M=0 is a no-op.
// K, N:     weight dimensions. K must be divisible by 64, N by 128.
void gptq_gemm_v2(
    const __half* A,
    const uint32_t* B,
    __half* C,
    const __half* scales,
    int M, int K, int N,
    cudaStream_t stream = 0);

// Standalone benchmark: v2 correctness + head-to-head performance vs Marlin
void bench_gptq_v2_kernels();

// Fused GEMM + residual add: C[i] += A @ dequant(B,scales)
// C must be pre-initialized with the residual values.
void gptq_gemm_v2_add(
    const __half* A,
    const uint32_t* B,
    __half* C,
    const __half* scales,
    int M, int K, int N,
    cudaStream_t stream = 0);

} // namespace deusridet
