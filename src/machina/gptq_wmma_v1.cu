/**
 * @file src/machina/gptq_wmma_v1.cu
 * @philosophical_role
 *   Peer TU of gptq.cu under R1 800-line hard cap — WMMA v1 (TC_BM/BN/BK) — gemm + add variants.
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
// WMMA GEMM kernel — Tensor core path (M>1)
// ============================================================================
//
// Uses WMMA m16n16k16 FP16 tensor core instructions for GPTQ-Int4 GEMM.
// Each block computes a [TC_BM, TC_BN] output tile.
// INT4 weights are dequantized in shared memory during cooperative load.
//
// Tile: BM=16, BN=64, BK=64 (half a GPTQ group → one scale per tile/col)
// Block: 128 threads = 4 warps, each warp → 16×16 output via WMMA
// SMEM padded: X[16, 72] + W^T[64, 72] = 11.5 KB → 2 blocks/SM = 23 KB, L1 = 105 KB
// Note: BK=64 (not 128) is optimal — larger BK increases SMEM, reducing L1 cache
// for X tensor reuse. SM87's 128 KB L1+SMEM budget makes L1 the limiting factor.
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
    __shared__ __half smem_x[TC_BM * TC_BK_PAD];        // 16×72 = 2.25 KB
    __shared__ __half smem_w[TC_BN * TC_BK_PAD];        // 64×72 = 9 KB (K-inner)

    // FP32 accumulator fragment
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Thread's constant column in the N-tile (tid % 64)
    const int my_nc = tid & (TC_BN - 1);
    const int pk_start = tid / TC_BN;  // 0 or 1

    const int num_k_tiles = K / TC_BK;

    // === Register-level double-buffer: prefetch next tile during WMMA ===
    // Two register sets: cur (being written to SMEM) and nxt (being loaded from DRAM)
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
                // Inline dequant: register ALU (IADD + I2FP.F16) replaces SMEM lookup
                // Eliminates data-dependent SMEM bank conflicts (was 4-way on base_values[16])
                half2 r2 = __hmul2(cur_s2, __halves2half2(
                    __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                    __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                vals[p * 2]     = __low2half(r2);
                vals[p * 2 + 1] = __high2half(r2);
            }
            *(float4*)(smem_w + my_nc * TC_BK_PAD + pk_row * 8) = *(float4*)vals;
        }

        __syncthreads();

        // === Phase 2: WMMA compute + prefetch next tile ===
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

        // Swap: next becomes current
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

void gptq_wmma_gemm_v1(const __half* X,
                        const GptqWeight& weight,
                        __half* Y,
                        int M,
                        cudaStream_t stream)
{
    int M_pad = (M + TC_BM - 1) / TC_BM * TC_BM;
    dim3 grid(weight.N / TC_BN, M_pad / TC_BM);
    gptq_wmma_gemm_kernel<<<grid, TC_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        weight.K, weight.N);
}

// ============================================================================
// Fused WMMA GEMM with SiLU activation on input and residual add on output.
//
// Computes: residual += W @ silu(gate) * up
//
// - Input X is computed on-the-fly as silu(gate[row,k]) * up[row,k]
//   during the SMEM load phase (eliminates standalone silu_mul kernel)
// - Output is added to residual in-place (eliminates elementwise_add kernel)
// - Saves 2 kernel launches per layer + ~49 MB DRAM traffic per pass
// ============================================================================

// (silu_add variant shelved — silu in prefetch stalls DRAM pipeline, see devlog)

// ============================================================================
// Fused WMMA GEMM with residual add on output.
//
// Computes: residual += W @ X
//
// Same as gptq_wmma_gemm_kernel but output is added to residual instead of
// written to a separate buffer. Eliminates standalone elementwise_add kernel
// (1 launch + 14 MB DRAM traffic per pass).
// ============================================================================

__global__ void __launch_bounds__(TC_BLOCK, 2)
gptq_wmma_gemm_add_kernel(
    const __half*   __restrict__ X,        // [M_pad, K]
    const uint32_t* __restrict__ qweight,  // [K/8, N] packed INT4
    const __half*   __restrict__ scales,   // [K/128, N] FP16
    __half*         __restrict__ residual, // [M_pad, res_N] in/out
    int K, int N, int res_N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * TC_BN;
    const int bm = blockIdx.y * TC_BM;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    __shared__ __half smem_x[TC_BM * TC_BK_PAD];
    __shared__ __half smem_w[TC_BN * TC_BK_PAD];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int my_nc = tid & (TC_BN - 1);
    const int pk_start = tid / TC_BN;
    const int num_k_tiles = K / TC_BK;

    float4 cur_x, nxt_x;
    uint32_t cur_pk[4], nxt_pk[4];
    half2 cur_s2, nxt_s2;

    // Pre-load tile 0
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
                half2 r2 = __hmul2(cur_s2, __halves2half2(
                    __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                    __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                vals[p * 2]     = __low2half(r2);
                vals[p * 2 + 1] = __high2half(r2);
            }
            *(float4*)(smem_w + my_nc * TC_BK_PAD + pk_row * 8) = *(float4*)vals;
        }

        __syncthreads();

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

        cur_x = nxt_x;
        cur_s2 = nxt_s2;
        cur_pk[0] = nxt_pk[0]; cur_pk[1] = nxt_pk[1];
        cur_pk[2] = nxt_pk[2]; cur_pk[3] = nxt_pk[3];
    }

    // Store: WMMA output to SMEM, then add to residual
    {
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
        for (int i = 0; i < acc.num_elements; i++)
            acc_h.x[i] = __float2half(acc.x[i]);
        // Reuse smem_x (16×72 ≥ 16×64 needed) with stride TC_BN
        wmma::store_matrix_sync(smem_x + warp_id * 16, acc_h, TC_BN, wmma::mem_row_major);
    }
    __syncthreads();

    // Each thread: read 8 halves from SMEM, add residual, write back
    {
        const int elem_base = tid * 8;
        const int r = elem_base / TC_BN;
        const int c = elem_base % TC_BN;
        const size_t gidx = (size_t)(bm + r) * res_N + bn + c;
        float4 y4 = *(const float4*)(smem_x + r * TC_BN + c);
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

void gptq_wmma_gemm_add_v1(const __half* X,
                            const GptqWeight& weight,
                            __half* residual, int res_N,
                            int M, cudaStream_t stream)
{
    int M_pad = (M + TC_BM - 1) / TC_BM * TC_BM;
    dim3 grid(weight.N / TC_BN, M_pad / TC_BM);
    gptq_wmma_gemm_add_kernel<<<grid, TC_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, residual,
        weight.K, weight.N, res_N);
}


// ============================================================================
// Public dispatchers — current default routes M-padded V1 kernel.
// (Originally lived in the warp-spec section; moved here under R1 split so
//  TC_* constants and the V1 kernels resolve in the same TU.)
// ============================================================================

void gptq_wmma_gemm(const __half* X,
                     const GptqWeight& weight,
                     __half* Y,
                     int M,
                     cudaStream_t stream)
{
    int M_pad = (M + TC_BM - 1) / TC_BM * TC_BM;
    dim3 grid(weight.N / TC_BN, M_pad / TC_BM);
    gptq_wmma_gemm_kernel<<<grid, TC_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        weight.K, weight.N);
}

void gptq_wmma_gemm_add(const __half* X,
                          const GptqWeight& weight,
                          __half* residual, int res_N,
                          int M, cudaStream_t stream)
{
    int M_pad = (M + TC_BM - 1) / TC_BM * TC_BM;
    dim3 grid(weight.N / TC_BN, M_pad / TC_BM);
    gptq_wmma_gemm_add_kernel<<<grid, TC_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, residual,
        weight.K, weight.N, res_N);
}

} // namespace deusridet
