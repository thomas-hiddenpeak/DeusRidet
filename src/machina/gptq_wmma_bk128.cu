/**
 * @file src/machina/gptq_wmma_bk128.cu
 * @philosophical_role
 *   Peer TU of gptq.cu under R1 800-line hard cap — WMMA bk=128 (BK2_*) — gemm + add variants.
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
// BK=128 WMMA GEMM — V3 (full-group tile, halved K-loop)
// ============================================================================
//
// BK=128 matches GPTQ group_size → exactly one scale per tile per column.
// BN=32 (not 64) keeps SMEM small enough for adequate L1:
//   SMEM = X[16,136] + W[32,136] = 13056 bytes = 12.75 KB
//   2 blocks/SM → 25.5 KB SMEM, L1 = 102.5 KB (vs V1's 105 KB)
//
// Benefits vs V1 (BK=64, BN=64):
//   - 50% fewer K-loop iterations (40 vs 80) → 50% fewer __syncthreads
//   - One scale load per tile (BK=128 = group) vs one per 2 tiles
//   - 128 threads all do load+dequant, only warps 0,1 do WMMA (BN=32 = 2×16)
//     Warps 2,3 are "load assistants" — keeps store simple (no accumulator merge)
//
// Trade-off: BN=32 → 2× more grid blocks, but same 2 blocks/SM occupancy.
//   Grid: (N/32, M_pad/16) vs V1 (N/64, M_pad/16)
//
// Requirements: K % 128 == 0, N % 32 == 0

constexpr int BK2_BM = 16;
constexpr int BK2_BN = 32;
constexpr int BK2_BK = 128;
constexpr int BK2_BK_PAD = 136;  // +8: float4-aligned, 4-bank shift per row
constexpr int BK2_BLOCK = 128;   // 4 warps

__global__ void __launch_bounds__(BK2_BLOCK, 2)
gptq_wmma_gemm_bk128_kernel(
    const __half*   __restrict__ X,       // [M_pad, K]
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    __half*         __restrict__ Y,       // [M_pad, N]
    int K, int N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * BK2_BN;
    const int bm = blockIdx.y * BK2_BM;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Shared memory: K-inner layout with padding
    __shared__ __half smem_x[BK2_BM * BK2_BK_PAD];     // 16 × 136 = 4352 bytes
    __shared__ __half smem_w[BK2_BN * BK2_BK_PAD];     // 32 × 136 = 8704 bytes

    // FP32 accumulator — only warps 0,1 use this
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Thread column in N-tile: tid % 32
    const int my_nc = tid & (BK2_BN - 1);
    // qweight group: tid / 32 → 0..3, each loads 4 packed rows (stride 4)
    const int pk_group = tid / BK2_BN;  // 0..3

    const int num_k_tiles = K / BK2_BK;

    // Register double-buffer
    float4 cur_x0, cur_x1, nxt_x0, nxt_x1;  // 2 float4 per thread = 16 halves
    uint32_t cur_pk[4], nxt_pk[4];
    half2 cur_s2, nxt_s2;

    // X load mapping: 128 threads load 16×128 halves
    // row = tid / 16 (0..7), col_base = (tid & 15) * 8
    // Each thread loads 2 rows: row and row+8
    const int x_row0 = tid / 16;
    const int x_row1 = x_row0 + 8;
    const int x_col = (tid & 15) * 8;

    // Pre-load tile 0
    {
        cur_x0 = *(const float4*)(X + (size_t)(bm + x_row0) * K + x_col);
        cur_x1 = *(const float4*)(X + (size_t)(bm + x_row1) * K + x_col);

        const uint32_t* qw_ptr = qweight + bn;  // pk_base = 0, k_base = 0
        cur_s2 = __half2half2(scales[bn + my_nc]);  // group 0

        cur_pk[0] = qw_ptr[(pk_group +  0) * N + my_nc];
        cur_pk[1] = qw_ptr[(pk_group +  4) * N + my_nc];
        cur_pk[2] = qw_ptr[(pk_group +  8) * N + my_nc];
        cur_pk[3] = qw_ptr[(pk_group + 12) * N + my_nc];
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // === Phase 1: write current tile from regs to SMEM (all 128 threads) ===
        *(float4*)(smem_x + x_row0 * BK2_BK_PAD + x_col) = cur_x0;
        *(float4*)(smem_x + x_row1 * BK2_BK_PAD + x_col) = cur_x1;

        // W: dequant 4 packed uint32 → 32 FP16 values each
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint32_t packed = cur_pk[r];
            int pk_row = pk_group + r * 4;  // packed row in tile: 0..15
            __half vals[8];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int nib0 = (packed >> (p * 8)) & 0xF;
                int nib1 = (packed >> (p * 8 + 4)) & 0xF;
                half2 r2 = __hmul2(cur_s2, __halves2half2(
                    __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                    __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                vals[p * 2]     = __low2half(r2);
                vals[p * 2 + 1] = __high2half(r2);
            }
            *(float4*)(smem_w + my_nc * BK2_BK_PAD + pk_row * 8) = *(float4*)vals;
        }

        __syncthreads();

        // === Phase 2: WMMA compute (warps 0,1) + prefetch next tile (all) ===
        if (kt + 1 < num_k_tiles) {
            const int k_next = (kt + 1) * BK2_BK;
            nxt_x0 = *(const float4*)(X + (size_t)(bm + x_row0) * K + k_next + x_col);
            nxt_x1 = *(const float4*)(X + (size_t)(bm + x_row1) * K + k_next + x_col);

            const int group = k_next / GPTQ_GROUP_SIZE;
            const uint32_t* qw_ptr = qweight + (size_t)(k_next / GPTQ_PACK_FACTOR) * N + bn;
            nxt_s2 = __half2half2(scales[(size_t)group * N + bn + my_nc]);

            nxt_pk[0] = qw_ptr[(pk_group +  0) * N + my_nc];
            nxt_pk[1] = qw_ptr[(pk_group +  4) * N + my_nc];
            nxt_pk[2] = qw_ptr[(pk_group +  8) * N + my_nc];
            nxt_pk[3] = qw_ptr[(pk_group + 12) * N + my_nc];
        }

        // WMMA: 8 iterations (BK=128 / 16), only warps 0,1 (BN=32 = 2×16)
        if (warp_id < 2) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            #pragma unroll
            for (int wk = 0; wk < BK2_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x + wk * 16, BK2_BK_PAD);
                wmma::load_matrix_sync(b_frag, smem_w + warp_id * 16 * BK2_BK_PAD + wk * 16, BK2_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }

        __syncthreads();

        // Swap registers
        cur_x0 = nxt_x0; cur_x1 = nxt_x1;
        cur_s2 = nxt_s2;
        cur_pk[0] = nxt_pk[0]; cur_pk[1] = nxt_pk[1];
        cur_pk[2] = nxt_pk[2]; cur_pk[3] = nxt_pk[3];
    }

    // === Store: warps 0,1 write FP32 accum → FP16 output ===
    if (warp_id < 2) {
        const int warp_col = bn + warp_id * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
        for (int i = 0; i < acc.num_elements; i++) {
            acc_h.x[i] = __float2half(acc.x[i]);
        }
        wmma::store_matrix_sync(Y + (size_t)bm * N + warp_col, acc_h, N,
                                wmma::mem_row_major);
    }
}

void gptq_wmma_gemm_bk128(const __half* X,
                           const GptqWeight& weight,
                           __half* Y,
                           int M,
                           cudaStream_t stream)
{
    int M_pad = (M + BK2_BM - 1) / BK2_BM * BK2_BM;
    dim3 grid(weight.N / BK2_BN, M_pad / BK2_BM);
    gptq_wmma_gemm_bk128_kernel<<<grid, BK2_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        weight.K, weight.N);
}

// ============================================================================
// BK=128 WMMA GEMM + residual add — V3
// Same as gptq_wmma_gemm_bk128_kernel but output is added to residual.
// ============================================================================

__global__ void __launch_bounds__(BK2_BLOCK, 2)
gptq_wmma_gemm_add_bk128_kernel(
    const __half*   __restrict__ X,        // [M_pad, K]
    const uint32_t* __restrict__ qweight,  // [K/8, N]
    const __half*   __restrict__ scales,   // [K/128, N]
    __half*         __restrict__ residual, // [M_pad, res_N] in/out
    int K, int N, int res_N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * BK2_BN;
    const int bm = blockIdx.y * BK2_BM;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    __shared__ __half smem_x[BK2_BM * BK2_BK_PAD];
    __shared__ __half smem_w[BK2_BN * BK2_BK_PAD];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int my_nc = tid & (BK2_BN - 1);
    const int pk_group = tid / BK2_BN;
    const int num_k_tiles = K / BK2_BK;

    float4 cur_x0, cur_x1, nxt_x0, nxt_x1;
    uint32_t cur_pk[4], nxt_pk[4];
    half2 cur_s2, nxt_s2;

    const int x_row0 = tid / 16;
    const int x_row1 = x_row0 + 8;
    const int x_col = (tid & 15) * 8;

    // Pre-load tile 0
    {
        cur_x0 = *(const float4*)(X + (size_t)(bm + x_row0) * K + x_col);
        cur_x1 = *(const float4*)(X + (size_t)(bm + x_row1) * K + x_col);

        const uint32_t* qw_ptr = qweight + bn;
        cur_s2 = __half2half2(scales[bn + my_nc]);

        cur_pk[0] = qw_ptr[(pk_group +  0) * N + my_nc];
        cur_pk[1] = qw_ptr[(pk_group +  4) * N + my_nc];
        cur_pk[2] = qw_ptr[(pk_group +  8) * N + my_nc];
        cur_pk[3] = qw_ptr[(pk_group + 12) * N + my_nc];
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        *(float4*)(smem_x + x_row0 * BK2_BK_PAD + x_col) = cur_x0;
        *(float4*)(smem_x + x_row1 * BK2_BK_PAD + x_col) = cur_x1;

        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint32_t packed = cur_pk[r];
            int pk_row = pk_group + r * 4;
            __half vals[8];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                int nib0 = (packed >> (p * 8)) & 0xF;
                int nib1 = (packed >> (p * 8 + 4)) & 0xF;
                half2 r2 = __hmul2(cur_s2, __halves2half2(
                    __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                    __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                vals[p * 2]     = __low2half(r2);
                vals[p * 2 + 1] = __high2half(r2);
            }
            *(float4*)(smem_w + my_nc * BK2_BK_PAD + pk_row * 8) = *(float4*)vals;
        }

        __syncthreads();

        if (kt + 1 < num_k_tiles) {
            const int k_next = (kt + 1) * BK2_BK;
            nxt_x0 = *(const float4*)(X + (size_t)(bm + x_row0) * K + k_next + x_col);
            nxt_x1 = *(const float4*)(X + (size_t)(bm + x_row1) * K + k_next + x_col);

            const int group = k_next / GPTQ_GROUP_SIZE;
            const uint32_t* qw_ptr = qweight + (size_t)(k_next / GPTQ_PACK_FACTOR) * N + bn;
            nxt_s2 = __half2half2(scales[(size_t)group * N + bn + my_nc]);

            nxt_pk[0] = qw_ptr[(pk_group +  0) * N + my_nc];
            nxt_pk[1] = qw_ptr[(pk_group +  4) * N + my_nc];
            nxt_pk[2] = qw_ptr[(pk_group +  8) * N + my_nc];
            nxt_pk[3] = qw_ptr[(pk_group + 12) * N + my_nc];
        }

        if (warp_id < 2) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            #pragma unroll
            for (int wk = 0; wk < BK2_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x + wk * 16, BK2_BK_PAD);
                wmma::load_matrix_sync(b_frag, smem_w + warp_id * 16 * BK2_BK_PAD + wk * 16, BK2_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }

        __syncthreads();

        cur_x0 = nxt_x0; cur_x1 = nxt_x1;
        cur_s2 = nxt_s2;
        cur_pk[0] = nxt_pk[0]; cur_pk[1] = nxt_pk[1];
        cur_pk[2] = nxt_pk[2]; cur_pk[3] = nxt_pk[3];
    }

    // Store: warps 0,1 write WMMA output to SMEM, then warps 0,1 add to residual
    // Reuse smem_x[0..511] as staging (16×32 = 512 halves, fits in 16×136)
    {
        if (warp_id < 2) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
            for (int i = 0; i < acc.num_elements; i++)
                acc_h.x[i] = __float2half(acc.x[i]);
            // Store to smem_x with stride BK2_BN (=32) at column offset warp_id*16
            wmma::store_matrix_sync(smem_x + warp_id * 16, acc_h, BK2_BN,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();

    // 64 threads (warps 0,1) read 8 halves each and add to residual
    // 64 × 8 = 512 = 16 × 32 ✓
    if (tid < 64) {
        const int elem_base = tid * 8;
        const int r = elem_base / BK2_BN;
        const int c = elem_base % BK2_BN;
        const size_t gidx = (size_t)(bm + r) * res_N + bn + c;
        float4 y4 = *(const float4*)(smem_x + r * BK2_BN + c);
        float4 res4 = *(const float4*)(residual + gidx);
        half2* y2  = reinterpret_cast<half2*>(&y4);
        half2* r2p = reinterpret_cast<half2*>(&res4);
        r2p[0] = __hadd2(y2[0], r2p[0]);
        r2p[1] = __hadd2(y2[1], r2p[1]);
        r2p[2] = __hadd2(y2[2], r2p[2]);
        r2p[3] = __hadd2(y2[3], r2p[3]);
        *(float4*)(residual + gidx) = res4;
    }
}

void gptq_wmma_gemm_add_bk128(const __half* X,
                               const GptqWeight& weight,
                               __half* residual, int res_N,
                               int M, cudaStream_t stream)
{
    int M_pad = (M + BK2_BM - 1) / BK2_BM * BK2_BM;
    dim3 grid(weight.N / BK2_BN, M_pad / BK2_BM);
    gptq_wmma_gemm_add_bk128_kernel<<<grid, BK2_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, residual,
        weight.K, weight.N, res_N);
}


} // namespace deusridet
