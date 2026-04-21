/**
 * @file marlin.h
 * @philosophical_role Declaration of the Marlin GPTQ INT4 kernel adapted for Orin. Upstream attribution preserved in the .cu; this header is the adapter seam.
 * @serves Machina INT4 fast path.
 */
// marlin.h — Marlin-style GPTQ INT4 GEMM kernel for SM87 (Orin)
//
// Adapted from IST-DASLab/marlin (Apache 2.0 License) for DeusRidet.
// Original: https://github.com/IST-DASLab/marlin
// Paper: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs"
//        Frantar et al., 2024 (arXiv:2408.11743)
//
// Key techniques from Marlin:
//   - cp.async hardware async DRAM→SMEM copy (bypasses registers + L1)
//   - 4-stage software pipeline for full load/compute overlap
//   - lop3.b32 dequantization (INT4→FP16 via bit manipulation, no int→float)
//   - XOR-based SMEM layout for bank-conflict-free access
//   - Persistent kernel with striped SM partitioning
//   - m16n8k16 tensor core MMA instructions
//
// Requires weight repacking from standard GPTQ format to Marlin tile format.
// Repacking is done once at model load time via GPU kernel (in-place).
//
// Target: SM87 (Ampere / Jetson AGX Orin), GPTQ group_size=128, sym=true

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {

// ============================================================================
// Weight repacking: GPTQ → Marlin format (one-time, at model load)
// ============================================================================

// Repack all MLP GPTQ weights in the model (64 layers × 3 projections).
// Allocates a single 42.5 MB temp buffer, repacks all tensors, frees temp.
// Call after load_model_weights(). Returns extra bytes allocated for workspace.
struct ModelWeights;  // forward declaration
size_t repack_all_marlin(ModelWeights& weights, cudaStream_t stream = 0);

// Repack a single GPTQ weight tensor from GPTQ format to Marlin tile format.
// Operates in-place on qweight and scales. temp_buf must be at least
// (K/8)*N*sizeof(uint32_t) bytes. d_perm[1024] and d_scale_perm[64] are
// device-resident permutation tables (see compute_marlin_perm/scale_perm).
void repack_gptq_to_marlin(
    uint32_t* qweight, __half* scales,
    int K, int N,
    int* d_perm, int* d_scale_perm,
    void* temp_buf, size_t temp_buf_bytes,
    cudaStream_t stream = 0);

// Upload Marlin permutation tables to device. Caller owns the returned pointers.
void upload_marlin_perm_tables(int** d_perm, int** d_scale_perm);

// ============================================================================
// Marlin GEMM dispatch
// ============================================================================

// Marlin GEMM: C[M,N] = A[M,K] @ B_marlin[K,N]
//
// A:         [M, K] FP16 row-major input
// B:         Marlin-packed INT4 weight (output of repack_gptq_to_marlin)
// C:         [M, N] FP16 row-major output
// s:         Marlin-permuted FP16 scales
// workspace: zeroed int32 buffer, at least marlin_workspace_size(N) bytes
// M, K, N:   problem dimensions (K and N must match repacked weight)
// groupsize: quantization group size (128 for our GPTQ)
void marlin_gemm(
    const __half* A,
    const uint32_t* B,
    __half* C,
    const __half* s,
    int* workspace,
    int M, int K, int N,
    int groupsize = 128,
    cudaStream_t stream = 0);

// Marlin GEMM with fused residual accumulation: C[i] += (A @ B)[i]
// In-place — reads old C, adds GEMM result, writes back. Only safe for
// M ≤ 64 (single-slice execution, no global reduce). For M > 64, use
// marlin_gemm() + separate elementwise_add().
void marlin_gemm_add(
    const __half* A,
    const uint32_t* B,
    __half* C,
    const __half* s,
    int* workspace,
    int M, int K, int N,
    int groupsize = 128,
    cudaStream_t stream = 0);

// Required workspace size in bytes for given N dimension.
// Workspace must be zeroed before each marlin_gemm call.
inline int marlin_workspace_size(int N) {
    // locks array: max(N/128, N/256) * max_par=16 ints
    // Conservative upper bound
    return (N / 64 + 1) * 16 * sizeof(int);
}

} // namespace deusridet
