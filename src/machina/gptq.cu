// gptq.cu — GPTQ-Int4 dequant + GEMV/GEMM CUDA kernels for SM87
//
// Kernel design for Jetson AGX Orin (SM87, Ampere):
//   - 16 SMs, 128 FP32 cores each, L2 = 4 MB
//   - Memory BW ~192 GB/s (unified DRAM)
//   - Shared memory: up to 164 KB/SM (48 KB default)
//   - FP16 tensor core available (WMMA m16n16k16)
//
// GPTQ layout: qweight[K/8, N] (uint32), scales[K/128, N] (FP16)
//   bits=4, group_size=128, sym=true, zero_point=8
//
// GEMV strategy (M=1, memory-bound):
//   Each warp handles a slice of output columns. Each thread processes
//   multiple K elements, accumulating in FP32 for numerical stability.
//   Final warp reduction → FP16 output.
//
// GEMM strategy (M>1, CUDA core fallback):
//   Tile-based: each thread block computes a tile of Y[BM, BN].
//   Load qweight tile into shared memory, dequant on the fly.
//
// WMMA GEMM strategy (M>1, tensor core):
//   Tile-based with WMMA m16n16k16 fragments.
//   BK=128 aligned to group_size → one scale per tile per column.
//   INT4 dequant in registers during SMEM cooperative load.

#include "gptq.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace deusridet {

// ============================================================================
// Shared device utilities
// ============================================================================

__device__ __forceinline__ int extract_int4(uint32_t packed, int index) {
    // Extract 4-bit value at position index (0–7), LSB-first
    return (packed >> (index * 4)) & 0xF;
}

__device__ __forceinline__ __half dequant_int4(__half scale, int q_val) {
    // W = scale * (q - 8), computed in FP32 for precision then converted to FP16
    float w = __half2float(scale) * (float)(q_val - GPTQ_ZERO_POINT);
    return __float2half(w);
}

// ============================================================================
// GEMV kernel — Decode path (M=1)
// ============================================================================
//
// Optimized INT4 GEMV with:
//   1. x vector cached in shared memory (eliminates redundant DRAM reads)
//   2. Scale hoisted: one load per group of 16 packed rows (15x fewer reads)
//   3. 4-way loop unrolling for memory pipelining (4 qweight loads in-flight)
//   4. 4 independent FP32 accumulators for ILP across FMA pipeline stages
//   5. Block=512 (TILE_N=64, K_THREADS=8) for high occupancy on SM87
//
// Memory layout: qweight[K/8, N] row-major, scales[K/128, N] row-major.
// Coalesced access: adjacent threads read adjacent N columns.
//
// Shared mem modes:
//   USE_SMEM_X=true:  x[K] (__half) + reduce[TILE_N * K_THREADS] (float)
//     Gate/up (K=5120): 12.3KB → 3 blocks/SM, 48 warps (100% occupancy)
//   USE_SMEM_X=false: reduce[TILE_N * K_THREADS] (float) only, x from L2
//     Down (K=17408): 2KB → 3 blocks/SM, 48 warps (100% occupancy)
//
// Scale deferral: scale multiply hoisted to group boundary (once per 128
// elements) instead of per-element. Reduces FP32 ops by ~15%, pushing
// the kernel from borderline compute-bound to solidly memory-bound.

constexpr int GEMV_TILE_N = 64;       // output columns per block
constexpr int GEMV_K_THREADS = 8;     // threads splitting K dimension
constexpr int GEMV_BLOCK_DIM = GEMV_TILE_N * GEMV_K_THREADS;  // 512 threads

// Templatized for optional fused residual add (ADD_RES):
//   false: y[n] = gemv_result
//   true:  y[n] = gemv_result + residual[n]  (saves a separate add kernel)
// Templatized for SMEM x vs L2 x (USE_SMEM_X):
//   true:  x loaded to shared memory (best for small K ≤ 10240)
//   false: x read from L2 cache directly (best for large K > 10240)
template<bool ADD_RES, bool USE_SMEM_X>
__global__ void gptq_gemv_kernel(
    const __half*   __restrict__ x,         // [K]
    const uint32_t* __restrict__ qweight,   // [K/8, N]
    const __half*   __restrict__ scales,    // [K/128, N]
    __half*         __restrict__ y,         // [N]
    const __half*   __restrict__ residual,  // [N] (only read if ADD_RES)
    int K, int N)
{
    extern __shared__ char smem_raw[];
    __half* smem_x;
    float*  smem_reduce;

    if constexpr (USE_SMEM_X) {
        smem_x = reinterpret_cast<__half*>(smem_raw);
        smem_reduce = reinterpret_cast<float*>(smem_raw + K * sizeof(__half));
    } else {
        smem_reduce = reinterpret_cast<float*>(smem_raw);
    }

    const int local_n = threadIdx.x % GEMV_TILE_N;
    const int k_tid   = threadIdx.x / GEMV_TILE_N;
    const int n = blockIdx.x * GEMV_TILE_N + local_n;

    // Cooperative vectorized load of x into shared memory (float4 = 8 halves)
    if constexpr (USE_SMEM_X) {
        const int vec_elems = K / 8;
        const float4* x_f4 = reinterpret_cast<const float4*>(x);
        float4* smem_f4 = reinterpret_cast<float4*>(smem_x);
        for (int i = threadIdx.x; i < vec_elems; i += GEMV_BLOCK_DIM)
            smem_f4[i] = x_f4[i];
        __syncthreads();
    }

    if (n >= N) return;

    const int packed_K = K / GPTQ_PACK_FACTOR;
    const int rows_per_thread = (packed_K + GEMV_K_THREADS - 1) / GEMV_K_THREADS;
    const int pk_start = k_tid * rows_per_thread;
    const int pk_end   = min(pk_start + rows_per_thread, packed_K);

    float acc = 0.0f;

    // Process packed rows with scale deferral: scale multiply hoisted to group
    // boundary. Inside the inner loop, accumulate raw (unscaled) dot products.
    // This removes one FP32 multiply per element from the hot path.
    constexpr int ROWS_PER_GROUP = GPTQ_GROUP_SIZE / GPTQ_PACK_FACTOR; // 16

    int pk = pk_start;
    while (pk < pk_end) {
        const int group = (pk * GPTQ_PACK_FACTOR) / GPTQ_GROUP_SIZE;
        const float s = __half2float(scales[group * N + n]);
        const int group_end = min((group + 1) * ROWS_PER_GROUP, pk_end);

        float raw0 = 0.0f, raw1 = 0.0f, raw2 = 0.0f, raw3 = 0.0f;

        // 4-way unrolled inner loop
        for (; pk + 3 < group_end; pk += 4) {
            uint32_t p0 = qweight[(pk    ) * N + n];
            uint32_t p1 = qweight[(pk + 1) * N + n];
            uint32_t p2 = qweight[(pk + 2) * N + n];
            uint32_t p3 = qweight[(pk + 3) * N + n];

            int kb = pk * GPTQ_PACK_FACTOR;

            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv;
                if constexpr (USE_SMEM_X) xv = __half2float(smem_x[kb + i]);
                else xv = __half2float(x[kb + i]);
                raw0 += (float)((int)((p0 >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv;
                if constexpr (USE_SMEM_X) xv = __half2float(smem_x[kb + 8 + i]);
                else xv = __half2float(x[kb + 8 + i]);
                raw1 += (float)((int)((p1 >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv;
                if constexpr (USE_SMEM_X) xv = __half2float(smem_x[kb + 16 + i]);
                else xv = __half2float(x[kb + 16 + i]);
                raw2 += (float)((int)((p2 >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv;
                if constexpr (USE_SMEM_X) xv = __half2float(smem_x[kb + 24 + i]);
                else xv = __half2float(x[kb + 24 + i]);
                raw3 += (float)((int)((p3 >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
        }

        // Remainder within group (0-3 packed rows)
        for (; pk < group_end; pk++) {
            uint32_t packed = qweight[pk * N + n];
            int kb = pk * GPTQ_PACK_FACTOR;
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv;
                if constexpr (USE_SMEM_X) xv = __half2float(smem_x[kb + i]);
                else xv = __half2float(x[kb + i]);
                raw0 += (float)((int)((packed >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
        }

        // Scale multiply deferred to group boundary (one multiply per group)
        acc += s * (raw0 + raw1 + raw2 + raw3);
    }

    // K-thread reduction via shared memory
    smem_reduce[k_tid * GEMV_TILE_N + local_n] = acc;
    __syncthreads();

    if (k_tid == 0) {
        float sum = smem_reduce[local_n];
        #pragma unroll
        for (int t = 1; t < GEMV_K_THREADS; t++)
            sum += smem_reduce[t * GEMV_TILE_N + local_n];
        if constexpr (ADD_RES)
            sum += __half2float(residual[n]);
        y[n] = __float2half(sum);
    }
}

void gptq_gemv(const __half* x,
               const GptqWeight& weight,
               __half* y,
               cudaStream_t stream)
{
    int N = weight.N;
    int K = weight.K;
    int grid = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    size_t smem = K * sizeof(__half) + GEMV_TILE_N * GEMV_K_THREADS * sizeof(float);

    gptq_gemv_kernel<false, true><<<grid, GEMV_BLOCK_DIM, smem, stream>>>(
        x, weight.qweight, weight.scales, y, nullptr, K, N);
}

// Fused GEMV + residual add: y[n] = (x @ W_q)[n] + residual[n]
// Eliminates a separate elementwise_add kernel + one full read/write pass.
void gptq_gemv_add(const __half* x,
                   const GptqWeight& weight,
                   __half* y,
                   const __half* residual,
                   cudaStream_t stream)
{
    int N = weight.N;
    int K = weight.K;
    int grid = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    size_t smem = K * sizeof(__half) + GEMV_TILE_N * GEMV_K_THREADS * sizeof(float);

    gptq_gemv_kernel<true, true><<<grid, GEMV_BLOCK_DIM, smem, stream>>>(
        x, weight.qweight, weight.scales, y, residual, K, N);
}

// ============================================================================
// Dual GEMV — gate_proj + up_proj computed together
// ============================================================================
// Both weight matrices must have the same K and N dimensions.
// x cached in SMEM (broadcast to all threads).
// Scale deferral + 2-way unrolled inner loop: 4 qweight loads per step
// (2 gate + 2 up). 4-way was tested but causes register pressure
// regression even with scale deferral.

__global__ void gptq_dual_gemv_kernel(
    const __half*   __restrict__ x,
    const uint32_t* __restrict__ qw_a,    // gate_proj qweight
    const __half*   __restrict__ sc_a,    // gate_proj scales
    const uint32_t* __restrict__ qw_b,    // up_proj qweight
    const __half*   __restrict__ sc_b,    // up_proj scales
    __half*         __restrict__ y_a,     // gate output
    __half*         __restrict__ y_b,     // up output
    int K, int N)
{
    extern __shared__ char smem_raw[];
    __half* smem_x = reinterpret_cast<__half*>(smem_raw);
    float* smem_red_a = reinterpret_cast<float*>(smem_raw + K * sizeof(__half));
    float* smem_red_b = smem_red_a + GEMV_TILE_N * GEMV_K_THREADS;

    const int local_n = threadIdx.x % GEMV_TILE_N;
    const int k_tid   = threadIdx.x / GEMV_TILE_N;
    const int n = blockIdx.x * GEMV_TILE_N + local_n;

    // Load x to SMEM (vectorized)
    {
        const int vec_elems = K / 8;
        const float4* x_f4 = reinterpret_cast<const float4*>(x);
        float4* smem_f4 = reinterpret_cast<float4*>(smem_x);
        for (int i = threadIdx.x; i < vec_elems; i += GEMV_BLOCK_DIM)
            smem_f4[i] = x_f4[i];
    }
    __syncthreads();

    if (n >= N) return;

    const int packed_K = K / GPTQ_PACK_FACTOR;
    const int rows_per_thread = (packed_K + GEMV_K_THREADS - 1) / GEMV_K_THREADS;
    const int pk_start = k_tid * rows_per_thread;
    const int pk_end   = min(pk_start + rows_per_thread, packed_K);

    float a_acc = 0.0f, b_acc = 0.0f;

    constexpr int ROWS_PER_GROUP = GPTQ_GROUP_SIZE / GPTQ_PACK_FACTOR;
    int pk = pk_start;
    while (pk < pk_end) {
        const int group = (pk * GPTQ_PACK_FACTOR) / GPTQ_GROUP_SIZE;
        const float sa = __half2float(sc_a[group * N + n]);
        const float sb = __half2float(sc_b[group * N + n]);
        const int group_end = min((group + 1) * ROWS_PER_GROUP, pk_end);

        float ra0 = 0, ra1 = 0;
        float rb0 = 0, rb1 = 0;

        // 2-way unrolled: 4 qweight loads per step (2 gate + 2 up)
        for (; pk + 1 < group_end; pk += 2) {
            uint32_t pa0 = qw_a[(pk    ) * N + n];
            uint32_t pa1 = qw_a[(pk + 1) * N + n];
            uint32_t pb0 = qw_b[(pk    ) * N + n];
            uint32_t pb1 = qw_b[(pk + 1) * N + n];
            int kb = pk * GPTQ_PACK_FACTOR;

            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv = __half2float(smem_x[kb + i]);
                ra0 += (float)((int)((pa0 >> (i*4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
                rb0 += (float)((int)((pb0 >> (i*4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv = __half2float(smem_x[kb + 8 + i]);
                ra1 += (float)((int)((pa1 >> (i*4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
                rb1 += (float)((int)((pb1 >> (i*4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
        }
        for (; pk < group_end; pk++) {
            uint32_t pa = qw_a[pk * N + n];
            uint32_t pb = qw_b[pk * N + n];
            int kb = pk * GPTQ_PACK_FACTOR;
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                float xv = __half2float(smem_x[kb + i]);
                ra0 += (float)((int)((pa >> (i*4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
                rb0 += (float)((int)((pb >> (i*4)) & 0xF) - GPTQ_ZERO_POINT) * xv;
            }
        }

        // Scale multiply deferred to group boundary
        a_acc += sa * (ra0 + ra1);
        b_acc += sb * (rb0 + rb1);
    }

    smem_red_a[k_tid * GEMV_TILE_N + local_n] = a_acc;
    smem_red_b[k_tid * GEMV_TILE_N + local_n] = b_acc;
    __syncthreads();

    if (k_tid == 0) {
        float sum_a = smem_red_a[local_n], sum_b = smem_red_b[local_n];
        #pragma unroll
        for (int t = 1; t < GEMV_K_THREADS; t++) {
            sum_a += smem_red_a[t * GEMV_TILE_N + local_n];
            sum_b += smem_red_b[t * GEMV_TILE_N + local_n];
        }
        y_a[n] = __float2half(sum_a);
        y_b[n] = __float2half(sum_b);
    }
}

void gptq_dual_gemv(const __half* x,
                    const GptqWeight& w_a, const GptqWeight& w_b,
                    __half* y_a, __half* y_b,
                    cudaStream_t stream)
{
    int N = w_a.N;
    int K = w_a.K;
    int grid = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    size_t smem = K * sizeof(__half)
               + 2 * GEMV_TILE_N * GEMV_K_THREADS * sizeof(float);

    gptq_dual_gemv_kernel<<<grid, GEMV_BLOCK_DIM, smem, stream>>>(
        x, w_a.qweight, w_a.scales, w_b.qweight, w_b.scales,
        y_a, y_b, K, N);
}

// ============================================================================
// Batch GEMV — Prefill path (M>1, small M)
//
// Y[M,N] = X[M,K] @ W_q[K,N] with on-the-fly INT4 dequantization
// Loads each weight row ONCE and computes M dot products simultaneously.
// X rows read from L2 cache (M×K×2 ≤ ~256KB fits in 4MB L2).
// Weight bandwidth identical to M=1 GEMV → near-M× speedup.
//
// Thread mapping: 4 warps/block, one output column per warp.
// Each lane processes K/32 packed rows, accumulating M dot products.
// Scale deferral: multiply by scale once per GPTQ group (every 128 elements).
// ============================================================================

__global__ void gptq_batch_gemv_kernel(
    const __half*   __restrict__ X,        // [M, K] row-major FP16
    const uint32_t* __restrict__ qweight,  // [K/8, N]
    const __half*   __restrict__ scales,   // [K/128, N]
    __half*         __restrict__ Y,        // [M, N] row-major FP16
    int M, int K, int N)
{
    constexpr int WARPS = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n = blockIdx.x * WARPS + warp_id;
    if (n >= N) return;

    const int packed_K = K / GPTQ_PACK_FACTOR;

    // M accumulators
    float acc[128];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    constexpr int ROWS_PER_GROUP = GPTQ_GROUP_SIZE / GPTQ_PACK_FACTOR; // 16

    // Each lane strides by 32 through packed_K
    for (int pk = lane_id; pk < packed_K; pk += 32) {
        int group = (pk * GPTQ_PACK_FACTOR) / GPTQ_GROUP_SIZE;
        float s = __half2float(scales[group * N + n]);
        uint32_t packed = qweight[pk * N + n];
        int kb = pk * GPTQ_PACK_FACTOR;

        // Dequantize 8 weight values (with deferred scale)
        float w_raw[8];
        #pragma unroll
        for (int i = 0; i < GPTQ_PACK_FACTOR; i++)
            w_raw[i] = (float)(extract_int4(packed, i) - GPTQ_ZERO_POINT);

        // For each M row: dot product with these 8 weights
        for (int m = 0; m < M; m++) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < GPTQ_PACK_FACTOR; i++)
                sum += w_raw[i] * __half2float(X[(size_t)m * K + kb + i]);
            acc[m] += s * sum;
        }
    }

    // Warp reduction + write for each M row
    for (int m = 0; m < M; m++) {
        float val = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off /= 2)
            val += __shfl_xor_sync(0xFFFFFFFF, val, off);
        if (lane_id == 0)
            Y[(size_t)m * N + n] = __float2half(val);
    }
}

void gptq_batch_gemv(const __half* X,
                     const GptqWeight& weight,
                     __half* Y,
                     int M,
                     cudaStream_t stream)
{
    int N = weight.N;
    int K = weight.K;
    int grid = (N + 3) / 4;  // 4 warps per block
    gptq_batch_gemv_kernel<<<grid, 128, 0, stream>>>(
        X, weight.qweight, weight.scales, Y, M, K, N);
}

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

// ============================================================================
// WMMA GEMM kernel — Tensor core path (M>1)
// ============================================================================
//
// Uses WMMA m16n16k16 FP16 tensor core instructions for GPTQ-Int4 GEMM.
// Each block computes a [TC_BM, TC_BN] output tile.
// INT4 weights are dequantized in shared memory during cooperative load.
//
// Tile: BM=16, BN=64, BK=64 (half a GPTQ group → one scale per tile/col)
// Block: 128 threads = 4 warps, each warp → 16×16 output via WMMA
// SMEM padded: X[16, 68] + W^T[64, 68] = 10.6 KB (pad +4 avoids SMEM bank conflicts)
// Occupancy: 4 blocks/SM → 16 warps/SM
//
// Requirements: K % 64 == 0, N % 64 == 0 (both hold for all MLP projections)
// M is padded to multiple of 16 internally (output buffers sized [max_seq, dim])

constexpr int TC_BM = 16;
constexpr int TC_BN = 64;
constexpr int TC_BK = 64;
constexpr int TC_BK_PAD = 72;  // +8 halfs: float4-aligned, bank shift 4/row
constexpr int TC_WARPS = 4;
constexpr int TC_BLOCK = TC_WARPS * 32;  // 128 threads

__global__ void __launch_bounds__(TC_BLOCK, 2)
gptq_wmma_gemm_kernel(
    const __half*   __restrict__ X,       // [M_pad, K] row-major FP16
    const uint32_t* __restrict__ qweight, // [K/8, N] packed INT4
    const __half*   __restrict__ scales,  // [K/128, N] FP16
    __half*         __restrict__ Y,       // [M_pad, N] row-major FP16
    int K, int N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * TC_BN;   // output column offset
    const int bm = blockIdx.y * TC_BM;   // output row offset
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Each warp handles a 16×16 output sub-tile at column bn + warp_id*16
    const int warp_col = bn + warp_id * 16;

    // Shared memory — K-inner layout for W: [BN, BK_PAD] enables float4 stores
    __shared__ __half smem_x[TC_BM * TC_BK_PAD];       // 16×72 = 2.25 KB
    __shared__ __half smem_w[TC_BN * TC_BK_PAD];        // 64×72 = 9 KB (K-inner)
    // Base value table: base_values[i] = (i - 8) as FP16, 32 bytes
    __shared__ __half base_values[16];
    if (tid < 16) base_values[tid] = __int2half_rn((int)tid - GPTQ_ZERO_POINT);
    __syncthreads();

    // FP32 accumulator fragment
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Thread's constant column in the N-tile (tid % 64)
    const int my_nc = tid & (TC_BN - 1);
    const int pk_start = tid / TC_BN;  // 0 or 1

    const int num_k_tiles = K / TC_BK;

    // === Register-level double-buffer: prefetch next tile during WMMA ===
    // Two register sets: cur (being written to SMEM) and nxt (being loaded from DRAM)
    // With __launch_bounds__(128, 2), we have 64 regs/thread — enough for both sets.
    float4 cur_x, nxt_x;
    uint32_t cur_pk[4], nxt_pk[4];
    half2 cur_s2, nxt_s2;

    // Pre-load tile 0 into 'cur' registers
    {
        const int k_base = 0;
        const int row = tid / 8, col = (tid & 7) * 8;
        cur_x = *(const float4*)(X + (size_t)(bm + row) * K + k_base + col);

        const int group = k_base / GPTQ_GROUP_SIZE;
        const uint32_t* qw_ptr = qweight + (size_t)(k_base / GPTQ_PACK_FACTOR) * N + bn;
        cur_s2 = __half2half2(scales[(size_t)group * N + bn + my_nc]);

        cur_pk[0] = qw_ptr[(pk_start + 0) * N + my_nc];
        cur_pk[1] = qw_ptr[(pk_start + 2) * N + my_nc];
        cur_pk[2] = qw_ptr[(pk_start + 4) * N + my_nc];
        cur_pk[3] = qw_ptr[(pk_start + 6) * N + my_nc];
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // === Phase 1: write current tile from regs to SMEM ===
        {
            const int row = tid / 8, col = (tid & 7) * 8;
            *(float4*)(smem_x + row * TC_BK_PAD + col) = cur_x;
        }

        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint32_t packed = cur_pk[r];
            int pk_row = pk_start + r * 2;
            __half vals[8];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int nib0 = (packed >> (p * 8)) & 0xF;
                int nib1 = (packed >> (p * 8 + 4)) & 0xF;
                half2 r2 = __hmul2(cur_s2, __halves2half2(base_values[nib0], base_values[nib1]));
                vals[p * 2]     = __low2half(r2);
                vals[p * 2 + 1] = __high2half(r2);
            }
            *(float4*)(smem_w + my_nc * TC_BK_PAD + pk_row * 8) = *(float4*)vals;
        }

        __syncthreads();

        // === Phase 2: WMMA compute + prefetch next tile ===
        // Issue DRAM loads for tile kt+1 BEFORE WMMA so loads overlap with compute
        if (kt + 1 < num_k_tiles) {
            const int k_next = (kt + 1) * TC_BK;
            const int row = tid / 8, col = (tid & 7) * 8;
            nxt_x = *(const float4*)(X + (size_t)(bm + row) * K + k_next + col);

            const int group = k_next / GPTQ_GROUP_SIZE;
            const uint32_t* qw_ptr = qweight + (size_t)(k_next / GPTQ_PACK_FACTOR) * N + bn;
            nxt_s2 = __half2half2(scales[(size_t)group * N + bn + my_nc]);

            nxt_pk[0] = qw_ptr[(pk_start + 0) * N + my_nc];
            nxt_pk[1] = qw_ptr[(pk_start + 2) * N + my_nc];
            nxt_pk[2] = qw_ptr[(pk_start + 4) * N + my_nc];
            nxt_pk[3] = qw_ptr[(pk_start + 6) * N + my_nc];
        }

        // WMMA compute (reads SMEM, doesn't touch DRAM — loads overlap)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            #pragma unroll
            for (int wk = 0; wk < TC_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x + wk * 16, TC_BK_PAD);
                wmma::load_matrix_sync(b_frag, smem_w + warp_id * 16 * TC_BK_PAD + wk * 16, TC_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }

        __syncthreads();

        // Swap: next becomes current for the next iteration
        cur_x = nxt_x;
        cur_s2 = nxt_s2;
        cur_pk[0] = nxt_pk[0]; cur_pk[1] = nxt_pk[1];
        cur_pk[2] = nxt_pk[2]; cur_pk[3] = nxt_pk[3];
    }

    // === Store: FP32 accum → FP16 output ===
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
    for (int i = 0; i < acc.num_elements; i++) {
        acc_h.x[i] = __float2half(acc.x[i]);
    }
    wmma::store_matrix_sync(Y + (size_t)bm * N + warp_col, acc_h, N, wmma::mem_row_major);
}

void gptq_wmma_gemm(const __half* X,
                     const GptqWeight& weight,
                     __half* Y,
                     int M,
                     cudaStream_t stream)
{
    // Pad M to multiple of 16 (output buffers are sized [max_seq, dim])
    int M_pad = (M + TC_BM - 1) / TC_BM * TC_BM;

    dim3 grid(weight.N / TC_BN, M_pad / TC_BM);

    gptq_wmma_gemm_kernel<<<grid, TC_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        weight.K, weight.N);
}

// ============================================================================
// GPU-based dequant kernel for reference: fully dequantize W_q → W_fp32
// Used by benchmark correctness check (replaces slow CPU matmul)
// ============================================================================

__global__ void gptq_dequant_kernel(
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    float*          __restrict__ W_fp32,  // [K, N] output
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;

    int k = idx / N;
    int n = idx % N;
    int pk = k / GPTQ_PACK_FACTOR;
    int ki = k % GPTQ_PACK_FACTOR;
    int group = k / GPTQ_GROUP_SIZE;

    uint32_t packed = qweight[pk * N + n];
    int q_val = (packed >> (ki * 4)) & 0xF;
    float s = __half2float(scales[group * N + n]);
    W_fp32[k * N + n] = s * (float)(q_val - GPTQ_ZERO_POINT);
}

GptqBenchResult gptq_benchmark(int K, int N, int M,
                                int warmup_iters,
                                int bench_iters)
{
    GptqBenchResult result = {};
    result.K = K;
    result.N = N;
    result.M = M;

    // Allocate host data for initialization
    int packed_K = K / GPTQ_PACK_FACTOR;
    int num_groups = K / GPTQ_GROUP_SIZE;

    size_t x_size   = (size_t)M * K * sizeof(__half);
    size_t qw_size  = (size_t)packed_K * N * sizeof(uint32_t);
    size_t sc_size  = (size_t)num_groups * N * sizeof(__half);
    size_t y_size   = (size_t)M * N * sizeof(__half);

    __half*   h_X = (__half*)malloc(x_size);
    uint32_t* h_qw = (uint32_t*)malloc(qw_size);
    __half*   h_sc = (__half*)malloc(sc_size);
    __half*   h_Y = (__half*)malloc(y_size);

    // Initialize with deterministic pseudo-random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_X[i] = __float2half(((float)(rand() % 1000) - 500.0f) / 500.0f);
    }
    for (int i = 0; i < packed_K * N; i++) {
        h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
    }
    for (int i = 0; i < num_groups * N; i++) {
        h_sc[i] = __float2half(((float)(rand() % 200) - 100.0f) / 100.0f * 0.1f);
    }

    // Device allocation
    __half*   d_X;
    uint32_t* d_qw;
    __half*   d_sc;
    __half*   d_Y;
    float*    d_W_ref;   // dequantized weights for cuBLAS reference
    float*    d_X_fp32;  // FP32 input for cuBLAS
    float*    d_Y_ref;   // FP32 reference output

    cudaMalloc(&d_X, x_size);
    cudaMalloc(&d_qw, qw_size);
    cudaMalloc(&d_sc, sc_size);
    cudaMalloc(&d_Y, y_size);
    cudaMalloc(&d_W_ref, (size_t)K * N * sizeof(float));
    cudaMalloc(&d_X_fp32, (size_t)M * K * sizeof(float));
    cudaMalloc(&d_Y_ref, (size_t)M * N * sizeof(float));

    cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qw, h_qw, qw_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, h_sc, sc_size, cudaMemcpyHostToDevice);

    GptqWeight weight;
    weight.qweight = d_qw;
    weight.scales  = d_sc;
    weight.K       = K;
    weight.N       = N;

    // GPU reference: dequant to FP32 + cuBLAS SGEMM
    // 1. Dequantize qweight → W_fp32 on GPU
    {
        int total_elems = K * N;
        int block = 256;
        int grid = (total_elems + block - 1) / block;
        gptq_dequant_kernel<<<grid, block>>>(d_qw, d_sc, d_W_ref, K, N);
    }

    // 2. Convert X from FP16 to FP32 for cuBLAS SGEMM reference
    {
        float* h_X_fp32 = (float*)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) {
            h_X_fp32[i] = __half2float(h_X[i]);
        }
        cudaMemcpy(d_X_fp32, h_X_fp32, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice);
        free(h_X_fp32);
    }

    // 3. cuBLAS SGEMM: Y_ref = X_fp32 @ W_fp32^T ... but our layout is row-major
    //    Y[M,N] = X[M,K] * W[K,N] (row-major)
    //    cuBLAS expects column-major, so: Y^T[N,M] = W^T[N,K] * X^T[K,M]
    //    i.e. cublasSgemm(N, M, K, W_ref, N, X_fp32, K, Y_ref, N)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_W_ref, N,     // W[K,N] stored row-major = W^T[N,K] col-major
                    d_X_fp32, K,    // X[M,K] stored row-major = X^T[K,M] col-major
                    &beta,
                    d_Y_ref, N);    // Y[M,N] stored row-major = Y^T[N,M] col-major
        cudaDeviceSynchronize();
        cublasDestroy(handle);
    }

    // 4. Run GPTQ kernel under test
    gptq_linear(d_X, weight, d_Y, M);
    cudaDeviceSynchronize();

    // 5. Compare on host
    cudaMemcpy(h_Y, d_Y, y_size, cudaMemcpyDeviceToHost);
    float* h_Y_ref = (float*)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(h_Y_ref, d_Y_ref, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    double sum_err = 0.0, sum_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(h_Y[i]);
        float ref_val = h_Y_ref[i];
        float err = fabsf(gpu_val - ref_val);
        sum_err += (double)(err * err);
        sum_ref += (double)(ref_val * ref_val);
    }
    float rmse = (sum_ref > 0) ? (float)sqrt(sum_err / sum_ref) : 0.0f;
    // FP16 dequant precision allows ~2-3% relative error for large K reductions
    result.correct = (rmse < 0.05f);

    free(h_Y_ref);
    cudaFree(d_W_ref);
    cudaFree(d_X_fp32);
    cudaFree(d_Y_ref);

    // Benchmark: only run the relevant kernel (GEMV for M=1, GEMM for M>1)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (M == 1) {
        // Benchmark GEMV
        for (int i = 0; i < warmup_iters; i++) {
            gptq_gemv(d_X, weight, d_Y);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < bench_iters; i++) {
            gptq_gemv(d_X, weight, d_Y);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        result.gemv_us = ms * 1000.0f / bench_iters;

        size_t bytes_read = qw_size + sc_size + (size_t)K * sizeof(__half);
        size_t bytes_write = (size_t)N * sizeof(__half);
        result.gemv_gbps = (bytes_read + bytes_write) / (result.gemv_us * 1e3f);
    } else {
        // Benchmark GEMM
        for (int i = 0; i < warmup_iters; i++) {
            gptq_gemm(d_X, weight, d_Y, M);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < bench_iters; i++) {
            gptq_gemm(d_X, weight, d_Y, M);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        result.gemm_us = ms * 1000.0f / bench_iters;

        double flops = 2.0 * M * N * K;
        result.gemm_tflops = (float)(flops / ((double)result.gemm_us * 1e6));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_qw);
    cudaFree(d_sc);
    cudaFree(d_Y);
    free(h_X);
    free(h_qw);
    free(h_sc);
    free(h_Y);

    return result;
}

} // namespace deusridet
