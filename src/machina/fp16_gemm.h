// fp16_gemm.h — FP16 GEMM kernel for SM87 (Orin)
//
// Custom high-efficiency GEMM for FP16 attention weight projections.
// Target: exceed cuBLAS on SM87 where cuBLAS reaches only ~164 GB/s
// (vs hardware achievable ~198 GB/s).
//
// Design:
//   - Weight repacking: B from [N,K] row-major → tile-level [K,N]
//     (one-time cost at model load, enables ldmatrix.x2.trans)
//   - Smart grid dispatch: non-persistent when tiles fill SMs evenly,
//     persistent (capped at 32 TBs) when wave imbalance would waste >25%
//   - 4-stage cp.async software pipeline for A (input) and B (weight)
//   - m16n8k16 tensor core MMA with FP32 accumulation
//   - XOR-based SMEM swizzle for bank-conflict-free A and B access
//   - A loaded via ldmatrix.x4, B via ldmatrix.x2.trans
//   - cp.async.ca for A (L1 caching enables cross-TB A tile reuse)
//
// Target: SM87, 16 SMs, 163 KB SMEM per SM, 2 TBs/SM (64KB each)
//
// Tile config:
//   BLOCK_M: 16..64 (template, multiples of 16)
//   TILE_N: 64
//   TILE_K: 64
//   SMEM: ~64 KB (4 stages × (8KB A + 8KB B))

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {

// FP16 GEMM: C[M,N] = A[M,K] @ B_repacked^T
//
// A:    [M, K] FP16 row-major input activations
// B:    Repacked weight matrix (see fp16_repack_b). NOT original [N,K].
// C:    [M, N] FP16 row-major output
// M:    number of rows (sequence length)
// K:    input dimension (must be divisible by 64)
// N:    output dimension (must be divisible by 64)
//
// Automatically selects thread_m_blocks based on M.
void fp16_gemm(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int K, int N,
    cudaStream_t stream = 0);

// Repack B weights from [N, K] row-major to tile-level [K, N] layout
// for ldmatrix.x2.trans access. One-time cost at model load.
//
// B_src:  [N, K] FP16 row-major (original weights)
// B_dst:  [N/64, K/64, 64, 64] FP16 (preallocated, same total size N*K)
// N:      output dimension (must be divisible by 64)
// K:      input dimension (must be divisible by 64)
void fp16_repack_b(
    const __half* B_src,
    __half* B_dst,
    int N, int K);

} // namespace deusridet
