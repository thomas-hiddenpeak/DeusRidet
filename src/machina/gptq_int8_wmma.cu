/**
 * @file src/machina/gptq_int8_wmma.cu
 * @philosophical_role
 *   Peer TU of gptq.cu under R1 800-line hard cap — INT8 WMMA (I8_*) — quant-then-mma path.
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
// INT8 WMMA GPTQ kernel — V4 (integer dequant, tensor core INT8)
// ============================================================================
//
// Eliminates FP16 dequant overhead by keeping weights in INT8 domain:
//   1. Dequant INT4 → INT8 (subtract zero_point=8, no float conversion)
//   2. Quantize X from FP16 to INT8 per-tile (per-row scale)
//   3. INT8 WMMA m16n16k16: int8 × int8 → int32 accumulator
//   4. Post-scale via SMEM: each thread reads its output elements,
//      applies x_scale[row] * w_scale[col], accumulates FP32 in registers
//
// Dequant per nibble: shift(1) + mask(1) + sub(1) = 3 ALU (vs 5 for FP16 path)
// = 40% reduction in dequant ALU per nibble.
//
// Tile: BM=16, BN=64, BK=64 (same as V1)
// SMEM: X_i8[16,72] + W_i8[64,72] + x_scale[16] + w_scale[64] + scratch[4×256×4]
//       = 1152 + 4608 + 64 + 256 + 4096 = 10176 bytes ≈ 10 KB
//   3 blocks/SM → 30 KB SMEM, L1 ≈ 98 KB
//
// Requirements: K % 64 == 0, N % 64 == 0

constexpr int I8_BM = 16;
constexpr int I8_BN = 64;
constexpr int I8_BK = 64;
constexpr int I8_BK_PAD = 80;  // 16-byte aligned for INT8 WMMA load_matrix_sync
constexpr int I8_BLOCK = 128;

__global__ void __launch_bounds__(I8_BLOCK, 2)
gptq_int8_wmma_gemm_kernel(
    const __half*   __restrict__ X,       // [M_pad, K]
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    __half*         __restrict__ Y,       // [M_pad, N]
    int K, int N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * I8_BN;
    const int bm = blockIdx.y * I8_BM;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // SMEM layout
    __shared__ signed char smem_x_i8[I8_BM * I8_BK_PAD];       // 1152 B
    __shared__ signed char smem_w_i8[I8_BN * I8_BK_PAD];       // 4608 B
    __shared__ float smem_x_scale[I8_BM];                       // 64 B
    __shared__ float smem_w_scale[I8_BN];                       // 256 B
    __shared__ int smem_scratch[4 * 16 * 16];                   // 4096 B (4 warp tiles)

    // Per-thread FP32 running total: 8 output elements per thread
    // Thread tid handles output elements [tid*8 .. tid*8+7] in row-major 16×64
    float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Thread's column in N-tile
    const int my_nc = tid & (I8_BN - 1);     // 0..63
    const int pk_start = tid / I8_BN;         // 0 or 1

    // Output element mapping for this thread
    const int out_row = (tid * 8) / I8_BN;    // row in output tile
    const int out_col = (tid * 8) % I8_BN;    // starting col in output tile

    const int num_k_tiles = K / I8_BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_base = kt * I8_BK;

        // === Load X, compute per-row max_abs, quantize to INT8 ===
        const int x_row = tid / 8;
        const int x_col_base = (tid & 7) * 8;
        __half x_vals[8];

        *(float4*)x_vals = *(const float4*)(X + (size_t)(bm + x_row) * K + k_base + x_col_base);

        float local_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            local_max = fmaxf(local_max, fabsf(__half2float(x_vals[i])));

        // Reduce over 8 threads with same row (row-mates at tid & 7 = 0..7)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 4));

        float x_scale = local_max / 127.0f;
        if (x_scale < 1e-8f) x_scale = 1e-8f;
        float rcp_x_scale = 1.0f / x_scale;

        if ((tid & 7) == 0)
            smem_x_scale[x_row] = x_scale;

        // Quantize and store INT8
        signed char x_i8[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v = __float2int_rn(__half2float(x_vals[i]) * rcp_x_scale);
            x_i8[i] = (signed char)max(-128, min(127, v));
        }
        *(int2*)(smem_x_i8 + x_row * I8_BK_PAD + x_col_base) = *(int2*)x_i8;

        // === Load W: dequant INT4 → INT8 (sub zero_point only) ===
        const int group = k_base / GPTQ_GROUP_SIZE;
        const uint32_t* qw_ptr = qweight + (size_t)(k_base / GPTQ_PACK_FACTOR) * N + bn;
        float w_scale_f = __half2float(scales[(size_t)group * N + bn + my_nc]);

        if (pk_start == 0)
            smem_w_scale[my_nc] = w_scale_f;

        uint32_t pk[4];
        pk[0] = qw_ptr[(pk_start + 0) * N + my_nc];
        pk[1] = qw_ptr[(pk_start + 2) * N + my_nc];
        pk[2] = qw_ptr[(pk_start + 4) * N + my_nc];
        pk[3] = qw_ptr[(pk_start + 6) * N + my_nc];

        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint32_t packed = pk[r];
            int pk_row = pk_start + r * 2;
            signed char vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                vals[i] = (signed char)(((packed >> (i * 4)) & 0xF) - GPTQ_ZERO_POINT);
            *(int2*)(smem_w_i8 + my_nc * I8_BK_PAD + pk_row * 8) = *(int2*)vals;
        }

        __syncthreads();

        // === INT8 WMMA compute ===
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> i32_acc;
            wmma::fill_fragment(i32_acc, 0);

            const int warp_col = warp_id * 16;
            #pragma unroll
            for (int wk = 0; wk < I8_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x_i8 + wk * 16, I8_BK_PAD);
                wmma::load_matrix_sync(b_frag, smem_w_i8 + warp_col * I8_BK_PAD + wk * 16, I8_BK_PAD);
                wmma::mma_sync(i32_acc, a_frag, b_frag, i32_acc);
            }

            // Store INT32 accumulator to SMEM scratch (row-major 16×16 per warp)
            wmma::store_matrix_sync(smem_scratch + warp_id * 256, i32_acc, 16,
                                    wmma::mem_row_major);
        }

        __syncthreads();

        // === Post-scale: read scaled values, accumulate in per-thread FP32 ===
        {
            float xs = smem_x_scale[out_row];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int c = out_col + i;
                int warp_of_c = c / 16;
                int local_c = c & 15;
                float ws = smem_w_scale[c];
                float ival = (float)smem_scratch[warp_of_c * 256 + out_row * 16 + local_c];
                total[i] += ival * xs * ws;
            }
        }

        __syncthreads();
    }

    // === Store: convert FP32 totals to FP16 and write to global ===
    {
        __half out[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            out[i] = __float2half(total[i]);
        *(float4*)(Y + (size_t)(bm + out_row) * N + bn + out_col) = *(float4*)out;
    }
}

void gptq_int8_wmma_gemm(const __half* X,
                          const GptqWeight& weight,
                          __half* Y,
                          int M,
                          cudaStream_t stream)
{
    int M_pad = (M + I8_BM - 1) / I8_BM * I8_BM;
    dim3 grid(weight.N / I8_BN, M_pad / I8_BM);
    gptq_int8_wmma_gemm_kernel<<<grid, I8_BLOCK, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        weight.K, weight.N);
}


} // namespace deusridet
