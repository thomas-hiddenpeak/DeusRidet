/**
 * @file src/machina/gptq_gemm.cu
 * @philosophical_role
 *   Peer TU of gptq.cu under R1 800-line hard cap — CUDA-core GEMM (M>1 fallback).
 * @serves
 *   Machina forward.cu GPTQ paths.
 */
#include "gptq_internal.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace deusridet {

// ============================================================================
// GEMM kernel — Prefill path (M>1)
// ============================================================================
//
// Tile-based approach: each block computes a [BM x BN] tile of Y.
// K is iterated in steps of BK (aligned to group_size for scale reuse).
//
// Per iteration:
//   1. Load X tile [BM, BK] from global to shared memory (FP16)
//   2. Load qweight tile [BK/8, BN] from global, dequant to shared memory (FP16)
//   3. Compute partial Y += X_tile @ W_tile using thread-level FMA
//
// Memory optimization:
//   - qweight loads are coalesced along N
//   - X loads are coalesced along K
//   - Dequant happens in shared memory to amortize global bandwidth
//   - BK = 128 aligns with group_size → one scale per BK tile per column

constexpr int GEMM_BM = 32;   // tile rows (tokens)
constexpr int GEMM_BN = 64;   // tile columns (output features)
constexpr int GEMM_BK = 128;  // tile K (= group_size, one scale per tile)
constexpr int GEMM_BLOCK_DIM = 256;

// Thread mapping within block: 256 = 8 (M-dim) x 32 (N-dim)
constexpr int GEMM_TM = 4;    // elements per thread in M dimension
constexpr int GEMM_TN = 2;    // elements per thread in N dimension
constexpr int GEMM_WARP_N = 32;

__global__ void gptq_gemm_kernel(
    const __half*   __restrict__ X,       // [M, K] row-major
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    __half*         __restrict__ Y,       // [M, N] row-major
    int M, int K, int N)
{
    const int bm = blockIdx.y * GEMM_BM;
    const int bn = blockIdx.x * GEMM_BN;

    // Thread indices within block
    const int tid = threadIdx.x;
    const int warp_m = tid / GEMM_WARP_N;   // 0..7 (M-dim thread index)
    const int warp_n = tid % GEMM_WARP_N;   // 0..31 (N-dim thread index)

    // Each thread accumulates TM x TN output values in FP32
    float acc[GEMM_TM][GEMM_TN];
    #pragma unroll
    for (int i = 0; i < GEMM_TM; i++) {
        #pragma unroll
        for (int j = 0; j < GEMM_TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Shared memory for dequantized weights and X tile
    // W_tile: [BK, BN] FP16 = 128 * 64 * 2 = 16 KB
    // X_tile: [BM, BK] FP16 = 32 * 128 * 2 = 8 KB
    // Total: 24 KB — fits in default 48KB shared memory
    __shared__ __half W_tile[GEMM_BK][GEMM_BN];
    __shared__ __half X_tile[GEMM_BM][GEMM_BK];

    const int num_k_tiles = (K + GEMM_BK - 1) / GEMM_BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_base = kt * GEMM_BK;

        // --- Load X tile [BM, BK] into shared memory ---
        // 256 threads load BM*BK = 32*128 = 4096 elements → 16 per thread
        {
            int total_elems = GEMM_BM * GEMM_BK;
            for (int idx = tid; idx < total_elems; idx += GEMM_BLOCK_DIM) {
                int row = idx / GEMM_BK;
                int col = idx % GEMM_BK;
                int global_m = bm + row;
                int global_k = k_base + col;
                if (global_m < M && global_k < K) {
                    X_tile[row][col] = X[global_m * K + global_k];
                } else {
                    X_tile[row][col] = __float2half(0.0f);
                }
            }
        }

        // --- Load and dequant qweight tile [BK, BN] into shared memory ---
        // qweight is [K/8, N], we need BK/8 = 16 packed rows, BN = 64 columns
        // 16 * 64 = 1024 loads, each yielding 8 values → 8192 writes
        // 256 threads, 4 packed rows per thread
        {
            int packed_rows = GEMM_BK / GPTQ_PACK_FACTOR;  // 128/8 = 16
            int total_loads = packed_rows * GEMM_BN;  // 16 * 64 = 1024
            int group = (k_base / GPTQ_GROUP_SIZE);  // BK = group_size → one group per tile

            for (int idx = tid; idx < total_loads; idx += GEMM_BLOCK_DIM) {
                int pk_local = idx / GEMM_BN;   // packed row within tile (0..15)
                int n_local  = idx % GEMM_BN;   // column within tile

                int pk_global = (k_base / GPTQ_PACK_FACTOR) + pk_local;
                int n_global  = bn + n_local;

                uint32_t packed = 0;
                __half s = __float2half(0.0f);
                if (n_global < N && pk_global < (K / GPTQ_PACK_FACTOR)) {
                    packed = qweight[pk_global * N + n_global];
                    s = scales[group * N + n_global];
                }

                #pragma unroll
                for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                    int k_local = pk_local * GPTQ_PACK_FACTOR + i;
                    int q_val = extract_int4(packed, i);
                    W_tile[k_local][n_local] = dequant_int4(s, q_val);
                }
            }
        }

        __syncthreads();

        // --- Compute: accumulate X_tile @ W_tile ---
        // Each thread computes TM x TN elements
        // Thread (warp_m, warp_n) handles rows [warp_m*TM .. warp_m*TM+TM-1]
        //                               cols  [warp_n*TN .. warp_n*TN+TN-1]
        #pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            #pragma unroll
            for (int tm = 0; tm < GEMM_TM; tm++) {
                float x_val = __half2float(X_tile[warp_m * GEMM_TM + tm][k]);
                #pragma unroll
                for (int tn = 0; tn < GEMM_TN; tn++) {
                    float w_val = __half2float(W_tile[k][warp_n * GEMM_TN + tn]);
                    acc[tm][tn] += x_val * w_val;
                }
            }
        }

        __syncthreads();
    }

    // --- Write output ---
    #pragma unroll
    for (int tm = 0; tm < GEMM_TM; tm++) {
        int global_m = bm + warp_m * GEMM_TM + tm;
        #pragma unroll
        for (int tn = 0; tn < GEMM_TN; tn++) {
            int global_n = bn + warp_n * GEMM_TN + tn;
            if (global_m < M && global_n < N) {
                Y[global_m * N + global_n] = __float2half(acc[tm][tn]);
            }
        }
    }
}

void gptq_gemm(const __half* X,
               const GptqWeight& weight,
               __half* Y,
               int M,
               cudaStream_t stream)
{
    dim3 grid((weight.N + GEMM_BN - 1) / GEMM_BN,
              (M + GEMM_BM - 1) / GEMM_BM);

    gptq_gemm_kernel<<<grid, GEMM_BLOCK_DIM, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        M, weight.K, weight.N);
}

} // namespace deusridet
