/**
 * @file src/machina/layer_int8.cu
 * @philosophical_role
 *   INT8 GEMV / GEMM kernels and dispatchers — peer TU of layer.cu under
 *   R1 800-line hard cap.
 * @serves
 *   Machina forward.cu int8 paths.
 */
// layer_int8.cu — peer TU of layer.cu (int8 gemv/batch/gemm/tc/linear).

#include "layer.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cfloat>
#include <vector>

namespace deusridet {

// ============================================================================
// INT8 GEMV kernel — Decode path (M=1)
//
// Per-channel symmetric quantization: w_fp32 = scale[n] * (float)int8_weight[n][k]
// Same warp-per-row design as FP16 GEMV, but reads 1 byte per weight instead of 2.
// Vectorized: int4 (16 bytes = 16 int8 values per thread per step).
// Each step also reads 16 FP16 x values (2 float4 loads).
// Warp stride: 32 lanes × 16 = 512 elements per iteration.
//
// Expected ~50% less weight data read → ~50% faster for weight-dominated GEMVs.
// ============================================================================

constexpr int INT8_GEMV_WARPS = 4;
constexpr int INT8_GEMV_BLOCK = INT8_GEMV_WARPS * 32;  // 128 threads

__global__ void int8_gemv_kernel(
    const __half*  __restrict__ x,       // [K]
    const int8_t*  __restrict__ W,       // [N, K] row-major INT8
    const float*   __restrict__ scales,  // [N] per-output-channel
    __half*        __restrict__ y,       // [N]
    int K, int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n = blockIdx.x * INT8_GEMV_WARPS + warp_id;
    if (n >= N) return;

    const int8_t* w_row = W + (size_t)n * K;
    const float scale = scales[n];
    float acc = 0.0f;

    // Vectorized: 16 int8 values (float4 = 16 bytes) + 16 FP16 x values (2 float4)
    // Warp stride: 32 × 16 = 512 elements per iteration
    const int vec_end = (K / 512) * 512;
    for (int k = lane_id * 16; k < vec_end; k += 512) {
        // Load 16 INT8 weights as float4 (16 bytes)
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        // Load 16 FP16 x values as 2 float4 (32 bytes)
        float4 xv0 = *reinterpret_cast<const float4*>(x + k);
        float4 xv1 = *reinterpret_cast<const float4*>(x + k + 8);

        const int8_t* wp = reinterpret_cast<const int8_t*>(&wv);
        const __half* xp = reinterpret_cast<const __half*>(&xv0);
        const __half* xp1 = reinterpret_cast<const __half*>(&xv1);

        #pragma unroll
        for (int i = 0; i < 8; i++)
            acc += (float)wp[i] * __half2float(xp[i]);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            acc += (float)wp[8 + i] * __half2float(xp1[i]);
    }

    // Scalar remainder (handles K not divisible by 512)
    for (int k = vec_end + lane_id; k < K; k += 32)
        acc += (float)w_row[k] * __half2float(x[k]);

    // Apply per-channel scale
    acc *= scale;

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane_id == 0)
        y[n] = __float2half(acc);
}

static void int8_gemv(const __half* x, const int8_t* W, const float* scales,
                      __half* y, int K, int N, cudaStream_t stream) {
    int grid = (N + INT8_GEMV_WARPS - 1) / INT8_GEMV_WARPS;
    int8_gemv_kernel<<<grid, INT8_GEMV_BLOCK, 0, stream>>>(x, W, scales, y, K, N);
}

// ============================================================================
// INT8 batch GEMV — Prefill path (M>1, small M)
//
// Y[M,N] = X[M,K] @ W^T[K,N] with per-channel scales
// Loads each weight row ONCE and computes M dot products simultaneously.
// X rows are read from L2 cache (total X = M×K×2 bytes, fits in 4MB L2).
// Weight bandwidth = same as M=1 GEMV → near-11× speedup for M=11.
//
// Thread mapping: same as int8_gemv_kernel (4 warps/block, warp-per-row-n)
// Register usage: M FP32 accumulators per thread (~11 extra for M=11)
// ============================================================================

__global__ void int8_batch_gemv_kernel(
    const __half* __restrict__ X,       // [M, K] row-major FP16
    const int8_t* __restrict__ W,       // [N, K] row-major INT8
    const float*  __restrict__ scales,  // [N] per-output-channel
    __half*       __restrict__ Y,       // [M, N] row-major FP16
    int M, int K, int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n = blockIdx.x * INT8_GEMV_WARPS + warp_id;
    if (n >= N) return;

    const int8_t* w_row = W + (size_t)n * K;
    const float scale = scales[n];

    // M accumulators — weight loaded once, used for all M rows
    float acc[128];  // max M=128 (matches max_seq allocation)
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    // Vectorized: 16 INT8 weights + M × 16 FP16 x values per step
    const int vec_end = (K / 512) * 512;
    for (int k = lane_id * 16; k < vec_end; k += 512) {
        // Load weight once (16 INT8 values)
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        const int8_t* wp = reinterpret_cast<const int8_t*>(&wv);

        // Apply to all M input rows
        for (int m = 0; m < M; m++) {
            float4 xv0 = *reinterpret_cast<const float4*>(X + (size_t)m * K + k);
            float4 xv1 = *reinterpret_cast<const float4*>(X + (size_t)m * K + k + 8);
            const __half* xp = reinterpret_cast<const __half*>(&xv0);
            const __half* xp1 = reinterpret_cast<const __half*>(&xv1);

            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                sum += (float)wp[i] * __half2float(xp[i]);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                sum += (float)wp[8 + i] * __half2float(xp1[i]);
            acc[m] += sum;
        }
    }

    // Scalar remainder
    for (int k = vec_end + lane_id; k < K; k += 32) {
        int8_t wval = w_row[k];
        for (int m = 0; m < M; m++)
            acc[m] += (float)wval * __half2float(X[(size_t)m * K + k]);
    }

    // Scale + warp reduction + write for each M row
    for (int m = 0; m < M; m++) {
        float val = acc[m] * scale;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

        if (lane_id == 0)
            Y[(size_t)m * N + n] = __float2half(val);
    }
}

static void int8_batch_gemv(const __half* X, const int8_t* W, const float* scales,
                             __half* Y, int M, int K, int N, cudaStream_t stream) {
    int grid = (N + INT8_GEMV_WARPS - 1) / INT8_GEMV_WARPS;
    int8_batch_gemv_kernel<<<grid, INT8_GEMV_BLOCK, 0, stream>>>(X, W, scales, Y, M, K, N);
}

// ============================================================================
// INT8 GEMM kernel — Prefill path (M>1)
//
// Y[M,N] = X[M,K] @ W^T[K,N] with per-channel scales
// W is [N,K] row-major INT8. scales[N] is FP32 per-output-channel.
//
// Tile: BM=32, BN=64, BK=128, Block=256 (mirrors GPTQ GEMM structure)
// Each thread computes TM×TN = 4×2 output elements.
// Scale deferred to output write (one multiply per output element).
//
// SMEM layout: X_tile[BM][BK] FP16 (8KB) + W_tile[BK][BN] FP16 (16KB) = 24KB
// ============================================================================

constexpr int I8_BM = 32;
constexpr int I8_BN = 64;
constexpr int I8_BK = 128;
constexpr int I8_BLOCK = 256;
constexpr int I8_TM = 4;
constexpr int I8_TN = 2;

__global__ void int8_gemm_kernel(
    const __half*  __restrict__ X,       // [M, K] row-major FP16
    const int8_t*  __restrict__ W,       // [N, K] row-major INT8
    const float*   __restrict__ scales,  // [N] per-channel FP32
    __half*        __restrict__ Y,       // [M, N] row-major FP16
    int M, int K, int N)
{
    const int bm = blockIdx.y * I8_BM;
    const int bn = blockIdx.x * I8_BN;
    const int tid = threadIdx.x;
    const int warp_m = tid / 32;   // 0..7
    const int warp_n = tid % 32;   // 0..31

    float acc[I8_TM][I8_TN];
    #pragma unroll
    for (int i = 0; i < I8_TM; i++)
        for (int j = 0; j < I8_TN; j++)
            acc[i][j] = 0.0f;

    __shared__ __half X_tile[I8_BM][I8_BK];
    __shared__ __half W_tile[I8_BK][I8_BN];

    const int num_k_tiles = (K + I8_BK - 1) / I8_BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_base = kt * I8_BK;

        // Load X tile [BM, BK] into SMEM (coalesced along K)
        {
            int total = I8_BM * I8_BK;
            for (int idx = tid; idx < total; idx += I8_BLOCK) {
                int row = idx / I8_BK;
                int col = idx % I8_BK;
                int gm = bm + row;
                int gk = k_base + col;
                X_tile[row][col] = (gm < M && gk < K)
                    ? X[gm * K + gk] : __float2half(0.0f);
            }
        }

        // Load W tile: read INT8 from W[N,K], convert to FP16, store as [BK,BN]
        // Thread mapping: k varies fastest for coalesced INT8 reads along K
        {
            int total = I8_BK * I8_BN;
            for (int idx = tid; idx < total; idx += I8_BLOCK) {
                int n_local = idx / I8_BK;
                int k_local = idx % I8_BK;
                int gn = bn + n_local;
                int gk = k_base + k_local;
                int8_t wval = (gn < N && gk < K) ? W[(size_t)gn * K + gk] : 0;
                W_tile[k_local][n_local] = __float2half((float)wval);
            }
        }

        __syncthreads();

        // Compute: accumulate X_tile @ W_tile
        #pragma unroll
        for (int k = 0; k < I8_BK; k++) {
            #pragma unroll
            for (int tm = 0; tm < I8_TM; tm++) {
                float x_val = __half2float(X_tile[warp_m * I8_TM + tm][k]);
                #pragma unroll
                for (int tn = 0; tn < I8_TN; tn++) {
                    acc[tm][tn] += x_val * __half2float(W_tile[k][warp_n * I8_TN + tn]);
                }
            }
        }

        __syncthreads();
    }

    // Write output with per-channel scale
    #pragma unroll
    for (int tm = 0; tm < I8_TM; tm++) {
        int gm = bm + warp_m * I8_TM + tm;
        #pragma unroll
        for (int tn = 0; tn < I8_TN; tn++) {
            int gn = bn + warp_n * I8_TN + tn;
            if (gm < M && gn < N) {
                Y[gm * N + gn] = __float2half(acc[tm][tn] * scales[gn]);
            }
        }
    }
}

static void int8_gemm(const __half* X, const int8_t* W, const float* scales,
                       __half* Y, int M, int K, int N, cudaStream_t stream) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    int8_gemm_kernel<<<grid, I8_BLOCK, 0, stream>>>(X, W, scales, Y, M, K, N);
}

// ============================================================================
// INT8 WMMA GEMM — Tensor core path (M>1)
// ============================================================================
//
// Uses WMMA m16n16k16 FP16 tensor cores with in-SMEM INT8→FP16 dequantization.
// W is [N, K] row-major INT8, scales [N] FP32 per-channel.
//
// Key design: W tile stored in SMEM as [BN, BK_PAD] (W's natural row-major layout,
// NOT transposed) with K-dimension padding for bank-conflict avoidance.
// WMMA uses col_major matrix_b to implicitly transpose W → W^T.
//
// Load pattern: each warp reads one W row (one N value) along K per iteration.
// 32 lanes × 4 bytes = 128 bytes = one full BK segment. Perfectly coalesced.
// 4 warps × 16 iterations = 64 rows = BN.
//
// Tile: BM=16, BN=64, BK=128.  Block: 128 threads = 4 warps.
// SMEM: X[16,136] (4.25 KB) + W[64,136] (17 KB) = 21.25 KB.
// launch_bounds(128, 4): 4 blocks/SM = 16 warps = 33% occupancy.
// The compiler spills ~11 regs to L1 (fast) to fit the 128-reg budget.
// With double-buffer prefetch, loads overlap with WMMA; inter-block
// interleaving provides additional DRAM latency hiding.
//
// Requirements: K % 128 == 0, N % 64 == 0.

constexpr int I8TC_BM = 16;
constexpr int I8TC_BN = 64;
constexpr int I8TC_BK = 128;
constexpr int I8TC_BK_PAD = I8TC_BK + 8;  // 136: pad K stride for SMEM bank avoidance
constexpr int I8TC_BLOCK = 128;  // 4 warps

__global__ void __launch_bounds__(I8TC_BLOCK, 4)
int8_wmma_gemm_kernel(
    const __half*  __restrict__ X,       // [M_pad, K] row-major FP16
    const int8_t*  __restrict__ W,       // [N, K] row-major INT8
    const float*   __restrict__ scales,  // [N] per-channel FP32
    __half*        __restrict__ Y,       // [M_pad, N] row-major FP16
    int K, int N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * I8TC_BN;
    const int bm = blockIdx.y * I8TC_BM;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // SMEM: X as [BM, BK_PAD] row-major, W as [BN, BK_PAD] (K-inner)
    __shared__ __half smem_x[I8TC_BM * I8TC_BK_PAD];        // 16×136 = 4.25 KB
    __shared__ __half smem_w[I8TC_BN * I8TC_BK_PAD];         // 64×136 = 17 KB

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int num_k_tiles = K / I8TC_BK;
    const int x_row = tid / 8, x_col = (tid & 7) * 16;

    // Pre-compute per-channel scales: constant across all K-tiles (hoist from loop)
    // Each warp processes 16 rows of W (4 groups × 4 rows)
    half2 cached_s2[16];
    #pragma unroll
    for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int n_local = warp_id + (grp * 4 + i) * 4;
            const int n_global = bn + n_local;
            cached_s2[grp * 4 + i] = __half2half2(__float2half(scales[n_global]));
        }
    }

    // === Register-level double-buffer: prefetch next tile during WMMA ===
    float4 cur_x0, cur_x1, nxt_x0, nxt_x1;
    uint32_t cur_pk[16], nxt_pk[16];

    // Pre-load tile 0 into 'cur' registers
    {
        const __half* src = X + (size_t)(bm + x_row) * K + x_col;
        cur_x0 = *(const float4*)(src);
        cur_x1 = *(const float4*)(src + 8);
    }
    #pragma unroll
    for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int n_local = warp_id + (grp * 4 + i) * 4;
            const int n_global = bn + n_local;
            cur_pk[grp * 4 + i] = *reinterpret_cast<const uint32_t*>(
                W + (size_t)n_global * K + lane_id * 4);
        }
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // === Phase 1: write current tile to SMEM (dequant from registers) ===

        // Write X to SMEM
        {
            __half* dst = smem_x + x_row * I8TC_BK_PAD + x_col;
            *(float4*)(dst)     = cur_x0;
            *(float4*)(dst + 8) = cur_x1;
        }

        // Dequant W: registers → FP16 → SMEM (using cached scales)
        #pragma unroll
        for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int idx = grp * 4 + i;
                const int n_local = warp_id + (grp * 4 + i) * 4;
                const half2 s2 = cached_s2[idx];
                uint32_t pk = cur_pk[idx];
                half2 v01 = __hmul2(s2, __halves2half2(
                    __int2half_rn((int)(int8_t)(pk & 0xFF)),
                    __int2half_rn((int)(int8_t)((pk >> 8) & 0xFF))));
                half2 v23 = __hmul2(s2, __halves2half2(
                    __int2half_rn((int)(int8_t)((pk >> 16) & 0xFF)),
                    __int2half_rn((int)(int8_t)((pk >> 24) & 0xFF))));
                __half* dst = smem_w + n_local * I8TC_BK_PAD + lane_id * 4;
                *(half2*)(dst)     = v01;
                *(half2*)(dst + 2) = v23;
            }
        }

        __syncthreads();

        // === Phase 2: WMMA compute + prefetch next tile ===
        // Issue DRAM loads for tile kt+1 BEFORE WMMA so loads overlap with tensor core
        if (kt + 1 < num_k_tiles) {
            const int k_next = (kt + 1) * I8TC_BK;
            {
                const __half* src = X + (size_t)(bm + x_row) * K + k_next + x_col;
                nxt_x0 = *(const float4*)(src);
                nxt_x1 = *(const float4*)(src + 8);
            }
            #pragma unroll
            for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int n_local = warp_id + (grp * 4 + i) * 4;
                    const int n_global = bn + n_local;
                    nxt_pk[grp * 4 + i] = *reinterpret_cast<const uint32_t*>(
                        W + (size_t)n_global * K + k_next + lane_id * 4);
                }
            }
        }

        // WMMA compute: 8 K-chunks (reads SMEM, concurrent with DRAM prefetch)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            #pragma unroll
            for (int wk = 0; wk < I8TC_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x + wk * 16, I8TC_BK_PAD);
                wmma::load_matrix_sync(b_frag,
                    smem_w + warp_id * 16 * I8TC_BK_PAD + wk * 16, I8TC_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }

        __syncthreads();

        // Swap: nxt → cur
        cur_x0 = nxt_x0; cur_x1 = nxt_x1;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            cur_pk[i] = nxt_pk[i];
    }

    // === Store: FP32 → FP16 ===
    const int warp_col = bn + warp_id * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
    for (int i = 0; i < acc.num_elements; i++) {
        acc_h.x[i] = __float2half(acc.x[i]);
    }
    wmma::store_matrix_sync(Y + (size_t)bm * N + warp_col, acc_h, N, wmma::mem_row_major);
}

static void int8_wmma_gemm(const __half* X, const int8_t* W, const float* scales,
                            __half* Y, int M, int K, int N, cudaStream_t stream) {
    int M_pad = (M + I8TC_BM - 1) / I8TC_BM * I8TC_BM;
    dim3 grid(N / I8TC_BN, M_pad / I8TC_BM);
    int8_wmma_gemm_kernel<<<grid, I8TC_BLOCK, 0, stream>>>(X, W, scales, Y, K, N);
}

void int8_linear_forward(const __half* X, const Int8Linear& weight, __half* Y,
                         int M, cudaStream_t stream) {
    if (M == 1) {
        int8_gemv(X, weight.weight, weight.scales, Y,
                  weight.in_features, weight.out_features, stream);
        return;
    }
    int K = weight.in_features;
    int N = weight.out_features;
    if (K % I8TC_BK == 0 && N % I8TC_BN == 0) {
        // Tensor core WMMA path: dequant INT8 in SMEM + m16n16k16 FP16 mma.
        int8_wmma_gemm(X, weight.weight, weight.scales, Y,
                       M, K, N, stream);
    } else {
        // Fallback for non-aligned dimensions (e.g. in_proj_a/b with N=48)
        int8_batch_gemv(X, weight.weight, weight.scales, Y,
                        M, K, N, stream);
    }
}


} // namespace deusridet
