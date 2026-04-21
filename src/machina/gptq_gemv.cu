/**
 * @file src/machina/gptq_gemv.cu
 * @philosophical_role
 *   Peer TU of gptq.cu under R1 800-line hard cap — GEMV (single + dual + batch).
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

} // namespace deusridet
