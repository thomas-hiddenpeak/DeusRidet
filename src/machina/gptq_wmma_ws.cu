/**
 * @file src/machina/gptq_wmma_ws.cu
 * @philosophical_role
 *   Peer TU of gptq.cu under R1 800-line hard cap — WMMA warp-spec (WS_*) — gemm + add variants.
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
// Warp-specialized WMMA GEMM — V2 (producer/consumer overlap)
// ============================================================================
//
// Addresses the fundamental GPTQ bottleneck: 15:1 dequant-ALU-to-HMMA ratio.
// In V1, all warps do dequant→sync→WMMA→sync serially. DRAM bandwidth is
// underutilized because no loads are in-flight during the 121-ALU dequant phase.
//
// V2 strategy: warp specialization with SMEM double-buffer.
//   - Warps 0,1 (producers): load qweight/X from DRAM + dequant + STS
//   - Warps 2,3 (consumers): WMMA from SMEM + accumulate
//   - Each syncthreads boundary, producers fill buf[ping] while consumers
//     compute from buf[pong]. Inter-block TLP (3 blocks/SM = 12 warps)
//     keeps warp schedulers busy when consumers are faster than producers.
//
// Tile: BM=16, BN=32, BK=64
// SMEM: double-buf [2 × (32×72 + 16×72)] = 2 × 6.75 KB = 13.5 KB
// Occupancy: 3 blocks/SM → 40.5 KB SMEM, 87.5 KB L1
//
// Compared to V1 (BN=64, 2 blocks/SM, 105 KB L1):
//   - Less L1 (87.5 vs 105 KB) but true producer/consumer overlap
//   - 50% more warps per SM (12 vs 8) → better latency hiding
//   - 2× grid blocks but 3 blocks/SM → same SM occupancy
// ============================================================================

constexpr int WS_BM = 16;
constexpr int WS_BN = 32;
constexpr int WS_BK = 64;
constexpr int WS_BK_PAD = 72;   // +8 halfs: float4-aligned
constexpr int WS_BLOCK = 128;   // 4 warps

__global__ void __launch_bounds__(WS_BLOCK, 3)
gptq_wmma_gemm_ws_kernel(
    const __half*   __restrict__ X,
    const uint32_t* __restrict__ qweight,
    const __half*   __restrict__ scales,
    __half*         __restrict__ Y,
    int K, int N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * WS_BN;
    const int bm = blockIdx.y * WS_BM;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Warp roles: 0,1 = producer (load+dequant), 2,3 = consumer (WMMA)
    const bool is_producer = (warp_id < 2);
    const int cons_warp = warp_id - 2;        // 0 or 1 for consumers

    // Double-buffered SMEM
    __shared__ __half smem_x[2][WS_BM * WS_BK_PAD];    // 2 × 16×72 = 4.5 KB
    __shared__ __half smem_w[2][WS_BN * WS_BK_PAD];    // 2 × 32×72 = 9 KB

    // Consumer accumulators
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Producer qweight mapping (64 threads handle BN×BK = 32×64 dequant)
    const int prod_nc = tid & (WS_BN - 1);   // 0..31
    const int pk_start = (tid / WS_BN) & 1;  // 0 or 1

    // Producer X load mapping: 64 threads load 16×64 = 128 float4s (2 each)
    // First float4: row = tid/4, col = (tid&3)*8  (covers rows 0..15, cols 0..31)
    // Second float4: same row, col + 32            (covers rows 0..15, cols 32..63)
    const int xrow0 = tid / 4;       // only warps 0,1 have tid 0..63 → row 0..15
    const int xcol0 = (tid & 3) * 8;

    const int num_k_tiles = K / WS_BK;

    // Named barrier protocol for producer-consumer overlap:
    // bar 0: "buf[0] produced" — producers arrive (non-blocking), consumers sync (wait)
    // bar 1: "buf[1] produced"
    // bar 2: "buf[0] consumed" — consumers arrive (non-blocking), producers sync (wait)
    // bar 3: "buf[1] consumed"
    // For kt < 2, producers skip the consume-wait (buffers start empty).

    // === Main pipeline ===
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int buf = kt & 1;
        const int consume_bar = 2 + buf;   // barrier for "buf consumed"
        const int produce_bar = buf;        // barrier for "buf produced"

        if (is_producer) {
            // Wait for consumers to finish reading this buffer from prev iteration
            if (kt >= 2) {
                asm volatile("bar.sync %0, 128;" :: "r"(consume_bar));
            }

            const int k_base = kt * WS_BK;

            // Load X to SMEM (2 float4 per producer thread)
            if (xrow0 < WS_BM) {
                *(float4*)(smem_x[buf] + xrow0 * WS_BK_PAD + xcol0) =
                    *(const float4*)(X + (size_t)(bm + xrow0) * K + k_base + xcol0);
                *(float4*)(smem_x[buf] + xrow0 * WS_BK_PAD + xcol0 + 32) =
                    *(const float4*)(X + (size_t)(bm + xrow0) * K + k_base + xcol0 + 32);
            }

            // Load + dequant qweight to SMEM
            const int group = k_base / GPTQ_GROUP_SIZE;
            const uint32_t* qw_ptr = qweight + (size_t)(k_base / GPTQ_PACK_FACTOR) * N + bn;
            half2 s2 = __half2half2(scales[(size_t)group * N + bn + prod_nc]);

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                uint32_t packed = qw_ptr[(pk_start + r * 2) * N + prod_nc];
                int pk_row = pk_start + r * 2;
                __half vals[8];
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    int nib0 = (packed >> (p * 8)) & 0xF;
                    int nib1 = (packed >> (p * 8 + 4)) & 0xF;
                    half2 r2 = __hmul2(s2, __halves2half2(
                        __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                        __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                    vals[p * 2]     = __low2half(r2);
                    vals[p * 2 + 1] = __high2half(r2);
                }
                *(float4*)(smem_w[buf] + prod_nc * WS_BK_PAD + pk_row * 8) = *(float4*)vals;
            }

            // Signal: buf produced (non-blocking arrive)
            asm volatile("bar.arrive %0, 128;" :: "r"(produce_bar));

        } else {
            // Consumer: wait for producers to finish buf
            asm volatile("bar.sync %0, 128;" :: "r"(produce_bar));

            // WMMA from buf
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            #pragma unroll
            for (int wk = 0; wk < WS_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x[buf] + wk * 16, WS_BK_PAD);
                wmma::load_matrix_sync(b_frag,
                    smem_w[buf] + cons_warp * 16 * WS_BK_PAD + wk * 16,
                    WS_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            // Signal: buf consumed (non-blocking arrive)
            asm volatile("bar.arrive %0, 128;" :: "r"(consume_bar));
        }
    }

    // Wait for all outstanding barriers to settle
    __syncthreads();

    // === Store: consumer warps write FP32 accum → FP16 output ===
    if (!is_producer) {
        const int warp_col = bn + cons_warp * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
        for (int i = 0; i < acc.num_elements; i++) {
            acc_h.x[i] = __float2half(acc.x[i]);
        }
        wmma::store_matrix_sync(Y + (size_t)bm * N + warp_col, acc_h, N,
                                wmma::mem_row_major);
    }
}

// ============================================================================
// Warp-specialized WMMA GEMM + residual add — V2
// ============================================================================

__global__ void __launch_bounds__(WS_BLOCK, 3)
gptq_wmma_gemm_add_ws_kernel(
    const __half*   __restrict__ X,
    const uint32_t* __restrict__ qweight,
    const __half*   __restrict__ scales,
    __half*         __restrict__ residual,
    int K, int N, int res_N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * WS_BN;
    const int bm = blockIdx.y * WS_BM;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    const bool is_producer = (warp_id < 2);
    const int cons_warp = warp_id - 2;

    __shared__ __half smem_x[2][WS_BM * WS_BK_PAD];
    __shared__ __half smem_w[2][WS_BN * WS_BK_PAD];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int prod_nc = tid & (WS_BN - 1);
    const int pk_start = (tid / WS_BN) & 1;

    // X load: all 128 threads cooperate
    const int xrow = tid / 8;
    const int xcol = (tid & 7) * 8;

    const int num_k_tiles = K / WS_BK;

    // Prologue: all threads load X, producers dequant W
    {
        const int k_base = 0;
        *(float4*)(smem_x[0] + xrow * WS_BK_PAD + xcol) =
            *(const float4*)(X + (size_t)(bm + xrow) * K + k_base + xcol);

        if (is_producer) {
            const int group = k_base / GPTQ_GROUP_SIZE;
            const uint32_t* qw_ptr = qweight + (size_t)(k_base / GPTQ_PACK_FACTOR) * N + bn;
            half2 s2 = __half2half2(scales[(size_t)group * N + bn + prod_nc]);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                uint32_t packed = qw_ptr[(pk_start + r * 2) * N + prod_nc];
                int pk_row = pk_start + r * 2;
                __half vals[8];
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    int nib0 = (packed >> (p * 8)) & 0xF;
                    int nib1 = (packed >> (p * 8 + 4)) & 0xF;
                    half2 r2 = __hmul2(s2, __halves2half2(
                        __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                        __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                    vals[p * 2]     = __low2half(r2);
                    vals[p * 2 + 1] = __high2half(r2);
                }
                *(float4*)(smem_w[0] + prod_nc * WS_BK_PAD + pk_row * 8) = *(float4*)vals;
            }
        }
    }
    __syncthreads();

    // Main loop
    for (int kt = 1; kt < num_k_tiles; kt++) {
        const int ping = kt & 1;
        const int pong = ping ^ 1;

        // All threads load X
        {
            const int k_base = kt * WS_BK;
            *(float4*)(smem_x[ping] + xrow * WS_BK_PAD + xcol) =
                *(const float4*)(X + (size_t)(bm + xrow) * K + k_base + xcol);
        }

        if (is_producer) {
            const int k_base = kt * WS_BK;
            const int group = k_base / GPTQ_GROUP_SIZE;
            const uint32_t* qw_ptr = qweight + (size_t)(k_base / GPTQ_PACK_FACTOR) * N + bn;
            half2 s2 = __half2half2(scales[(size_t)group * N + bn + prod_nc]);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                uint32_t packed = qw_ptr[(pk_start + r * 2) * N + prod_nc];
                int pk_row = pk_start + r * 2;
                __half vals[8];
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    int nib0 = (packed >> (p * 8)) & 0xF;
                    int nib1 = (packed >> (p * 8 + 4)) & 0xF;
                    half2 r2 = __hmul2(s2, __halves2half2(
                        __int2half_rn(nib0 - GPTQ_ZERO_POINT),
                        __int2half_rn(nib1 - GPTQ_ZERO_POINT)));
                    vals[p * 2]     = __low2half(r2);
                    vals[p * 2 + 1] = __high2half(r2);
                }
                *(float4*)(smem_w[ping] + prod_nc * WS_BK_PAD + pk_row * 8) = *(float4*)vals;
            }
        } else {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            #pragma unroll
            for (int wk = 0; wk < WS_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x[pong] + wk * 16, WS_BK_PAD);
                wmma::load_matrix_sync(b_frag,
                    smem_w[pong] + cons_warp * 16 * WS_BK_PAD + wk * 16,
                    WS_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }
        __syncthreads();
    }

    // Epilogue: consumers compute last tile
    if (!is_producer) {
        const int last_buf = (num_k_tiles - 1) & 1;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        #pragma unroll
        for (int wk = 0; wk < WS_BK / 16; wk++) {
            wmma::load_matrix_sync(a_frag, smem_x[last_buf] + wk * 16, WS_BK_PAD);
            wmma::load_matrix_sync(b_frag,
                smem_w[last_buf] + cons_warp * 16 * WS_BK_PAD + wk * 16,
                WS_BK_PAD);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    // Store: consumer warps write to SMEM, then ALL threads do residual add
    // Reuse smem_x[0] (16×72 ≥ 16×32 needed) with stride WS_BN for output staging
    if (!is_producer) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
        for (int i = 0; i < acc.num_elements; i++)
            acc_h.x[i] = __float2half(acc.x[i]);
        wmma::store_matrix_sync(smem_x[0] + cons_warp * 16, acc_h, WS_BN,
                                wmma::mem_row_major);
    }
    __syncthreads();

    // 128 threads, 16×32 = 512 elements → 4 per thread
    {
        const int elem_base = tid * 4;
        if (elem_base < WS_BM * WS_BN) {
            const int r = elem_base / WS_BN;
            const int c = elem_base % WS_BN;
            const size_t gidx = (size_t)(bm + r) * res_N + bn + c;
            // Load 4 halves (half2×2)
            half2 y0 = *(const half2*)(smem_x[0] + r * WS_BN + c);
            half2 y1 = *(const half2*)(smem_x[0] + r * WS_BN + c + 2);
            half2 r0 = *(const half2*)(residual + gidx);
            half2 r1 = *(const half2*)(residual + gidx + 2);
            *(half2*)(residual + gidx)     = __hadd2(y0, r0);
            *(half2*)(residual + gidx + 2) = __hadd2(y1, r1);
        }
    }
}


} // namespace deusridet
