/**
 * @file src/machina/gptq_gemm_v2.cu
 * @philosophical_role
 *   GPTQ GEMM v2 — the second-generation path, Marlin-tile loop with non-persistent CTA scheduling. Kept alongside gptq.cu so the older GEMV path remains available for shapes where v2 is not a win.
 * @serves
 *   Machina linear layers where M>1 prefill benefits from tile-level blocking; selected at runtime by shape.
 */
// gptq_gemm_v2.cu — GPTQ INT4 GEMM v2: Marlin-format weights, SM87-tuned
//
// Non-persistent CTA-level tiling with Marlin's proven inner loop.
// Each CTA computes one [BM, BN=128] output tile, iterating over full K.
// No global reduction, no lock-based barriers, no slice/stripe scheduling.
//
// Adapted from IST-DASLab/marlin (Apache 2.0 License).
// Original: https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
// Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
//
// Changes: stripped persistent grid, global reduction, parallel batching.
// Kept: weight format, dequant, MMA, SMEM layout, pipeline structure.
// Tile config: tnb=8, tkb=8 (BN=128, BK=128) — best for SM87 Orin.

#include "gptq_gemm_v2.h"
#include "marlin.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace deusridet {

// ============================================================================
// Section 1: Constants
// ============================================================================

static constexpr int THREADS = 256;
static constexpr int STAGES  = 4;

// Tile sizes for SM87: BN=128, BK=128 (narrow config)
// THREAD_M_BLOCKS: 1-4, selected at runtime via template dispatch
// BM = thread_m_blocks * 16 (16, 32, 48, or 64)
// THREAD_N_BLOCKS=8 and THREAD_K_BLOCKS=8 are kernel template parameters.
static constexpr int GROUP_BLOCKS    = 8;   // group_size=128 → 128/16 = 8

static constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================================
// Section 2: Register-level vector types (identical to Marlin)
// ============================================================================

template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

using I4 = Vec<int, 4>;
using FragA = Vec<half2, 4>;   // A operand for m16n8k16
using FragB = Vec<half2, 2>;   // B operand for m16n8k16
using FragC = Vec<float, 4>;   // Accumulator
using FragS = Vec<half2, 1>;   // Scale

// ============================================================================
// Section 3: PTX inline assembly (identical to Marlin)
// ============================================================================

// Predicated async global→shared copy (16 bytes), cg cache policy for A.
__device__ __forceinline__ void cp_async4_pred(
    void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

// Streaming async copy for weights B (evict-first L2 hint: accessed once).
__device__ __forceinline__ void cp_async4_stream(
    void* smem_ptr, const void* glob_ptr)
{
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// m16n8k16 tensor core MMA: C += A * B
__device__ __forceinline__ void mma(
    const FragA& a_frag, const FragB& frag_b, FragC& frag_c)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );
}

// Load 16x16 A fragment from shared memory via ldmatrix.
__device__ __forceinline__ void ldsm4(FragA& frag_a, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

// 3-input LUT-based logical operation.
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
    );
    return res;
}

// ============================================================================
// Section 4: INT4→FP16 dequantization (identical to Marlin)
// ============================================================================

// Converts packed int32 of 4 INT4 values → FragB of 4 FP16 values.
// Symmetric quantization: fp16 = (int4_val - 8) * scale
__device__ __forceinline__ FragB dequant(int q) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    const int SUB = 0x64086408;
    const int MUL = 0x2c002c00;
    const int ADD = 0xd480d480;
    FragB frag_b;
    frag_b[0] = __hsub2(
        *reinterpret_cast<half2*>(&lo),
        *reinterpret_cast<const half2*>(&SUB)
    );
    frag_b[1] = __hfma2(
        *reinterpret_cast<half2*>(&hi),
        *reinterpret_cast<const half2*>(&MUL),
        *reinterpret_cast<const half2*>(&ADD)
    );
    return frag_b;
}

// Apply per-group scale to dequantized B fragment.
__device__ __forceinline__ void scale(FragB& frag_b, FragS& frag_s, int i) {
    half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
    frag_b[0] = __hmul2(frag_b[0], s);
    frag_b[1] = __hmul2(frag_b[1], s);
}

// ============================================================================
// Section 5: Main kernel
// ============================================================================

template <int thread_m_blocks, int thread_n_blocks, int thread_k_blocks>
__global__ void __launch_bounds__(THREADS)
gptq_gemm_v2_kernel(
    const int4* __restrict__ A,   // FP16 input [M, K]
    const int4* __restrict__ B,   // Marlin-format INT4 [K/16, 2*N]
          int4* __restrict__ C,   // FP16 output [M, N]
    const int4* __restrict__ s,   // FP16 scales [K/128, N] Marlin-permuted
    int prob_m, int prob_n, int prob_k,
    const int4* residual          // fused add: C[i] = residual[i] + result (nullptr=disabled)
) {
    // ---- CTA tile assignment (non-persistent: one tile per CTA) ----
    const int tile_m = blockIdx.x;  // M tile index
    const int tile_n = blockIdx.y;  // N tile index

    // Strides in int4 (16-byte) units
    int a_gl_stride = prob_k / 8;                      // A row stride
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8;  // A SMEM row stride
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
    int a_gl_rd_delta_i = a_gl_stride * (THREADS / a_gl_rd_delta_o);
    constexpr int a_sh_wr_delta = a_sh_stride * (THREADS / a_gl_rd_delta_o);
    constexpr int a_sh_rd_delta_o = 2 * ((THREADS / 32) / (thread_n_blocks / 4));
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

    int b_gl_stride = 16 * prob_n / 32;
    constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (THREADS / b_sh_stride);
    constexpr int b_sh_wr_delta = THREADS;
    constexpr int b_sh_rd_delta = THREADS;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

    int s_gl_stride = prob_n / 8;
    constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
    constexpr int s_sh_stage = s_sh_stride;
    int s_gl_rd_delta = s_gl_stride;

    // ---- Thread-level read/write indices ----
    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
    // Offset to this CTA's M tile (row 0 of this tile in global A)
    a_gl_rd += a_gl_stride * (16 * thread_m_blocks * tile_m);

    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) +
                  (threadIdx.x % 32) / 16;
    a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
    // Offset to this CTA's N tile
    b_gl_rd += b_sh_stride * tile_n;

    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s_gl_rd = s_sh_stride * tile_n + threadIdx.x;
    int s_sh_wr = threadIdx.x;
    int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                  (threadIdx.x % 32) / 4;

    // M-boundary predication for A loads
    bool a_sh_wr_pred[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr <
                           a_sh_stride * prob_m -
                           a_sh_stride * (16 * thread_m_blocks * tile_m);

    bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

    // XOR-based SMEM layout for bank-conflict-free A access
    auto transform_a = [&](int i) {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };

    int a_sh_wr_trans[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) {
        #pragma unroll
        for (int j = 0; j < thread_m_blocks; j++)
            a_sh_rd_trans[i][j] = transform_a(
                a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
    }

    // Pre-split B global pointers for pipelining
    const int4* B_ptr[b_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    // ---- Dynamic shared memory: A | B | S ----
    extern __shared__ int4 sh[];
    int4* sh_a = sh;
    int4* sh_b = sh_a + (STAGES * a_sh_stage);
    int4* sh_s = sh_b + (STAGES * b_sh_stage);

    // ---- Register double-buffer for SMEM→register loads ----
    FragA frag_a[2][thread_m_blocks];
    I4 frag_b_quant[2];
    FragC frag_c[thread_m_blocks][4][2];
    FragS frag_s[2][4];

    // ---- Zero accumulators ----
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
        reinterpret_cast<float*>(frag_c)[i] = 0;

    // ---- Number of K-tiles this CTA must process ----
    const int k_tiles = prob_k / 16 / thread_k_blocks;

    // ---- Lambda: fetch A + B + S for one pipeline stage ----
    int a_gl_rd_cur = a_gl_rd;  // track A global read position

    auto fetch_to_shared = [&](int pipe, int k_iter, bool pred = true) {
        if (pred) {
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(
                    &sh_a_stage[a_sh_wr_trans[i]],
                    &A[a_gl_rd_delta_i * i + a_gl_rd_cur],
                    a_sh_wr_pred[i]);
            }
            int4* sh_b_stage = sh_b + b_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++) {
                cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr],
                                 B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            // Scales: one load per GROUP_BLOCKS k-tiles
            if (k_iter % (GROUP_BLOCKS / thread_k_blocks) == 0) {
                int4* sh_s_stage = sh_s + s_sh_stage * pipe;
                if (s_sh_wr_pred)
                    cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
                s_gl_rd += s_gl_rd_delta;
            }
        }
        cp_async_fence();
        a_gl_rd_cur += a_gl_rd_delta_o;
    };

    // ---- Lambda: SMEM → register load ----
    auto fetch_to_registers = [&](int k, int pipe) {
        int4* sh_s_stage = sh_s + s_sh_stage *
            ((GROUP_BLOCKS / thread_k_blocks) *
             (pipe / (GROUP_BLOCKS / thread_k_blocks)));
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];

        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
            ldsm4(frag_a[k % 2][i],
                   &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4* sh_b_stage = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4*>(
            &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    // ---- Lambda: MMA compute for one sub-tile ----
    auto matmul = [&](int k) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int b_quant = frag_b_quant[k % 2][j];
            int b_quant_shift = b_quant >> 8;
            FragB frag_b0 = dequant(b_quant);
            scale(frag_b0, frag_s[k % 2][j], 0);
            FragB frag_b1 = dequant(b_quant_shift);
            scale(frag_b1, frag_s[k % 2][j], 1);
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    // ---- Lambda: thread-block reduction (2-way for this config) ----
    auto thread_block_reduce = [&]() {
        constexpr int red_off = THREADS / b_sh_stride / 2;
        if constexpr (red_off >= 1) {
            int red_idx = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta = b_sh_stride;
            int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) +
                            (threadIdx.x % b_sh_stride);

            #pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
                #pragma unroll
                for (int i = red_off; i > 0; i /= 2) {
                    if (i <= red_idx && red_idx < 2 * i) {
                        #pragma unroll
                        for (int j = 0; j < 4 * 2; j++) {
                            int red_sh_wr = red_sh_delta * j +
                                            (red_sh_rd - red_sh_stride * i);
                            if (i < red_off) {
                                float* c_rd = reinterpret_cast<float*>(
                                    &sh[red_sh_delta * j + red_sh_rd]);
                                float* c_wr = reinterpret_cast<float*>(
                                    &sh[red_sh_wr]);
                                #pragma unroll
                                for (int k = 0; k < 4; k++)
                                    reinterpret_cast<FragC*>(frag_c)
                                        [4 * 2 * m_block + j][k] +=
                                        c_rd[k] + c_wr[k];
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4*>(
                                &frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0) {
                    #pragma unroll
                    for (int i = 0; i < 4 * 2; i++) {
                        float* c_rd = reinterpret_cast<float*>(
                            &sh[red_sh_delta * i + red_sh_rd]);
                        #pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<FragC*>(frag_c)
                                [4 * 2 * m_block + i][j] += c_rd[j];
                    }
                }
                __syncthreads();
            }
        }
    };

    // ---- Lambda: write result to global memory ----
    auto write_result = [&]() {
        int c_gl_stride = prob_n / 8;
        constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
        int c_gl_wr_delta = c_gl_stride * (THREADS / (2 * thread_n_blocks));
        constexpr int c_sh_rd_delta =
            c_sh_stride * (THREADS / (2 * thread_n_blocks));

        int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                      (threadIdx.x % (2 * thread_n_blocks));
        // Offset to this CTA's tile in output
        c_gl_wr += (2 * thread_n_blocks) * tile_n;
        c_gl_wr += c_gl_stride * (16 * thread_m_blocks * tile_m);
        int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) +
                      (threadIdx.x % 32) % 4;
        c_sh_wr += 32 * (threadIdx.x / 32);
        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                      (threadIdx.x % (2 * thread_n_blocks));

        int c_gl_wr_end = c_gl_stride * prob_m;

        // Load per-column scales for final application (group_blocks=-1 only)
        // For grouped quant (our case), scales were already applied in matmul

        auto write = [&](int idx, float c0, float c1, FragS& /*unused*/) {
            half2 res = __halves2half2(__float2half(c0), __float2half(c1));
            ((half2*)sh)[idx] = res;
        };

        if (threadIdx.x / 32 < thread_n_blocks / 4) {
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wr = c_sh_wr + 8 * j;
                    write(wr + (4 * c_sh_stride) * 0 + 0,
                          frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[0][0]);
                    write(wr + (4 * c_sh_stride) * 8 + 0,
                          frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[0][0]);
                    write(wr + (4 * c_sh_stride) * 0 + 4,
                          frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[0][0]);
                    write(wr + (4 * c_sh_stride) * 8 + 4,
                          frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[0][0]);
                }
                c_sh_wr += 16 * (4 * c_sh_stride);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0;
             i < ceildiv(16 * thread_m_blocks,
                         THREADS / (2 * thread_n_blocks));
             i++)
        {
            if (c_gl_wr < c_gl_wr_end) {
                if (residual) {
                    int4 r = sh[c_sh_rd];
                    int4 o = residual[c_gl_wr];
                    half2* rh = reinterpret_cast<half2*>(&r);
                    const half2* oh = reinterpret_cast<const half2*>(&o);
                    rh[0] = __hadd2(rh[0], oh[0]);
                    rh[1] = __hadd2(rh[1], oh[1]);
                    rh[2] = __hadd2(rh[2], oh[2]);
                    rh[3] = __hadd2(rh[3], oh[3]);
                    C[c_gl_wr] = r;
                } else {
                    C[c_gl_wr] = sh[c_sh_rd];
                }
                c_gl_wr += c_gl_wr_delta;
                c_sh_rd += c_sh_rd_delta;
            }
        }
    };

    // ====================================================================
    // Main execution: fill pipeline, iterate K, reduce, write
    // ====================================================================

    // Prologue: fill first (STAGES-1) pipeline stages
    #pragma unroll
    for (int i = 0; i < STAGES - 1; i++)
        fetch_to_shared(i, i, i < k_tiles);

    // Wait for stage 0, load first sub-tile to registers
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    fetch_to_registers(0, 0);

    // Main K-tile loop
    int k_iter = 0;
    while (k_iter < k_tiles) {
        #pragma unroll
        for (int pipe = 0; pipe < STAGES;) {
            #pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % STAGES);
                if (k == b_sh_wr_iters - 2) {
                    // Issue async load for next pipeline stage
                    int next_k = k_iter + STAGES - 1;
                    fetch_to_shared((pipe + STAGES - 1) % STAGES,
                                    next_k,
                                    next_k < k_tiles);
                    pipe++;
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();
                }
                matmul(k);
            }
            k_iter++;
            if (k_iter >= k_tiles) break;
        }
    }

    // Drain pipeline
    cp_async_wait<0>();
    __syncthreads();  // Ensure all async copies visible to all threads

    // Thread-block reduction
    thread_block_reduce();

    // Write output
    write_result();
}

// ============================================================================
// Section 6: Host dispatch — dual-config tile + multi-config M selection
//
// Two tile configurations:
//   Wide:   tnb=16, tkb=4  (BN=256, BK=64)  — Marlin-matched, high compute intensity
//   Narrow: tnb=8,  tkb=8  (BN=128, BK=128) — more CTAs for small-N shapes
//
// ============================================================================
// Section 6: Host dispatch — narrow config only (BN=128, BK=128)
//
// NOTE: Wide config (BN=256, BK=64) was attempted for Marlin-matched compute
// intensity but produces non-deterministic results on SM87 Tegra (Orin).
// The root cause appears to be a code generation or hardware issue specific
// to the <tmb, 16, 4> template instantiation. Extensive debugging confirmed:
//   - Not an optimization bug (-O0 reproduces)
//   - Not a fast-math issue (reproduces without --use_fast_math)
//   - Not register pressure (reproduces with maxrregcount=128)
//   - Not missing sync (adding __syncthreads after cp_async_wait<0> doesn't help)
//   - Partially mitigated by SMEM zeroing (suggests stale SMEM, but doesn't fully fix)
// The narrow config (tnb=8, tkb=8) is fully deterministic and correct.
// ============================================================================

// SMEM bytes as a function of all tile parameters
template <int tmb, int tnb, int tkb>
static constexpr int smem_bytes_for() {
    return (STAGES * (
        (16 * tkb / 8 * 16 * tmb) +  // A: a_sh_stage
        (32 * tnb / 4 * tkb) +        // B: b_sh_stage
        (16 * tnb / 8)                 // S: s_sh_stage
    )) * 16;  // int4 = 16 bytes
}

// One-time max dynamic SMEM configuration per template instantiation
template <int tmb, int tnb, int tkb>
static void configure_smem_once() {
    static bool done = false;
    if (!done) {
        constexpr int bytes = smem_bytes_for<tmb, tnb, tkb>();
        cudaFuncSetAttribute(gptq_gemm_v2_kernel<tmb, tnb, tkb>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        done = true;
    }
}

static void gptq_gemm_v2_impl(
    const __half* A, const uint32_t* B, __half* C, const __half* scales,
    int M, int K, int N, cudaStream_t stream,
    const int4* residual_ptr)
{
    if (M == 0 || K == 0 || N == 0) return;

    if (N % 128 != 0 || K % 128 != 0) {
        fprintf(stderr, "gptq_gemm_v2: requires N%%128==0 && K%%128==0 "
                "(got N=%d K=%d)\n", N, K);
        return;
    }

    // Select thread_m_blocks based on M
    int tmb;
    if      (M <= 16) tmb = 1;
    else if (M <= 32) tmb = 2;
    else if (M <= 48) tmb = 3;
    else              tmb = 4;

    const int4* A4 = reinterpret_cast<const int4*>(A);
    const int4* B4 = reinterpret_cast<const int4*>(B);
    int4* C4 = reinterpret_cast<int4*>(C);
    const int4* s4 = reinterpret_cast<const int4*>(scales);

    // Narrow config: tnb=8, tkb=8 → BN=128, BK=128
    #define LAUNCH_V2(TMB)                                                     \
    {                                                                          \
        constexpr int tnb = 8, tkb = 8;                                        \
        constexpr int smem = smem_bytes_for<TMB, tnb, tkb>();                  \
        int gm = ceildiv(M, 16 * TMB);                                        \
        int gn = N / (16 * tnb);                                              \
        dim3 grid(gm, gn);                                                     \
        configure_smem_once<TMB, tnb, tkb>();                                  \
        gptq_gemm_v2_kernel<TMB, tnb, tkb><<<grid, THREADS, smem, stream>>>(  \
            A4, B4, C4, s4, M, N, K, residual_ptr);                           \
    }

    switch (tmb) {
        case 1: LAUNCH_V2(1); break;
        case 2: LAUNCH_V2(2); break;
        case 3: LAUNCH_V2(3); break;
        default: LAUNCH_V2(4); break;
    }

    #undef LAUNCH_V2
}

void gptq_gemm_v2(
    const __half* A, const uint32_t* B, __half* C, const __half* scales,
    int M, int K, int N, cudaStream_t stream)
{
    gptq_gemm_v2_impl(A, B, C, scales, M, K, N, stream, nullptr);
}

void gptq_gemm_v2_add(
    const __half* A, const uint32_t* B, __half* C, const __half* scales,
    int M, int K, int N, cudaStream_t stream)
{
    gptq_gemm_v2_impl(A, B, C, scales, M, K, N, stream,
                       reinterpret_cast<const int4*>(C));
}

// ============================================================================
// Section 7: Standalone benchmark — v2 correctness + head-to-head vs Marlin
// ============================================================================

// CPU reference for correctness validation (original GPTQ format, before repack)
static void cpu_gptq_ref(
    const __half* A, const uint32_t* qw, float* C, const __half* sc,
    int M, int K, int N, int gs)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                float a = __half2float(A[m * K + k]);
                uint32_t packed = qw[(k / 8) * N + n];
                int q = (packed >> ((k % 8) * 4)) & 0xF;
                float s = __half2float(sc[(k / gs) * N + n]);
                sum += (double)a * (double)((float)(q - 8) * s);
            }
            C[m * N + n] = (float)sum;
        }
    }
}

// Minimal naive GPTQ GEMM kernel for correctness reference in benchmarks.
// Dequantizes INT4 (group_size=gs, symmetric) and accumulates in FP32.
static __global__ void gptq_gemm_naive_ref(
    const __half* __restrict__ A,
    const uint32_t* __restrict__ qweight,
    __half* __restrict__ C,
    const __half* __restrict__ scales,
    int M, int K, int N, int gs)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        int grp = k / gs;
        float s = __half2float(scales[grp * N + col]);
        uint32_t packed = qweight[(k / 8) * N + col];
        int shift = (k % 8) * 4;
        int q = (int)((packed >> shift) & 0xF) - 8;
        float w = (float)q * s;
        acc += __half2float(A[row * K + k]) * w;
    }
    C[row * N + col] = __float2half(acc);
}

void bench_gptq_v2_kernels() {
    printf("\n=== GPTQ INT4 v2 Benchmark (Marlin-format, SM87) ===\n");
    printf("v2: BN=128/BK=128, non-persistent, multi-config M dispatch\n");
    printf("Comparison: v2 vs Marlin (persistent grid)\n\n");

    int *d_perm = nullptr, *d_scale_perm = nullptr;
    upload_marlin_perm_tables(&d_perm, &d_scale_perm);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ================================================================
    // Diagnostic: small matrix, CPU reference correctness
    // ================================================================
    {
        printf("--- DIAGNOSTIC: K=256, N=256, M=32 ---\n");
        constexpr int K = 256, N = 256, M = 32, gs = 128;

        size_t a_bytes  = (size_t)M * K * sizeof(__half);
        size_t qw_bytes = (size_t)(K / 8) * N * sizeof(uint32_t);
        size_t sc_bytes = (size_t)(K / gs) * N * sizeof(__half);
        size_t c_bytes  = (size_t)M * N * sizeof(__half);

        __half*   h_A   = (__half*)malloc(a_bytes);
        uint32_t* h_qw  = (uint32_t*)malloc(qw_bytes);
        __half*   h_sc  = (__half*)malloc(sc_bytes);
        float*    h_ref = (float*)malloc(M * N * sizeof(float));
        __half*   h_v2  = (__half*)malloc(c_bytes);
        __half*   h_ml  = (__half*)malloc(c_bytes);

        srand(42);
        for (int i = 0; i < M * K; i++)
            h_A[i] = __float2half(((rand() % 1000) - 500) / 500.0f);
        for (int i = 0; i < (K / 8) * N; i++)
            h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
        for (int i = 0; i < (K / gs) * N; i++)
            h_sc[i] = __float2half(((rand() % 200) - 100) / 1000.0f);

        // CPU reference from original GPTQ format
        cpu_gptq_ref(h_A, h_qw, h_ref, h_sc, M, K, N, gs);

        __half *d_A, *d_sc, *d_v2, *d_ml;
        uint32_t *d_qw;
        cudaMalloc(&d_A,  a_bytes);
        cudaMalloc(&d_qw, qw_bytes);
        cudaMalloc(&d_sc, sc_bytes);
        cudaMalloc(&d_v2, c_bytes);
        cudaMalloc(&d_ml, c_bytes);
        cudaMemcpy(d_A,  h_A,  a_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_qw, h_qw, qw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc, h_sc, sc_bytes,  cudaMemcpyHostToDevice);

        // Repack to Marlin format (in-place)
        void* tmp;
        cudaMalloc(&tmp, qw_bytes);
        repack_gptq_to_marlin(d_qw, d_sc, K, N, d_perm, d_scale_perm,
                              tmp, qw_bytes);
        cudaFree(tmp);
        cudaDeviceSynchronize();

        int ws_bytes = marlin_workspace_size(N);
        int* d_ws;
        cudaMalloc(&d_ws, ws_bytes);

        // Run v2
        cudaMemset(d_v2, 0, c_bytes);
        gptq_gemm_v2(d_A, d_qw, d_v2, d_sc, M, K, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  v2 launch error: %s\n", cudaGetErrorString(err));
        }

        // Run Marlin
        cudaMemset(d_ws, 0, ws_bytes);
        cudaMemset(d_ml, 0, c_bytes);
        marlin_gemm(d_A, d_qw, d_ml, d_sc, d_ws, M, K, N, gs);
        cudaDeviceSynchronize();

        cudaMemcpy(h_v2, d_v2, c_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ml, d_ml, c_bytes, cudaMemcpyDeviceToHost);

        // v2 vs CPU
        {
            double se = 0, sr = 0;
            for (int i = 0; i < M * N; i++) {
                float g = __half2float(h_v2[i]), r = h_ref[i];
                se += (double)(g - r) * (g - r);
                sr += (double)r * r;
            }
            float rmse = (sr > 0) ? (float)sqrt(se / sr) : 0;
            printf("  v2 vs CPU:     [0,0] CPU=%.4f v2=%.4f  RMSE=%.6f (%s)\n",
                   h_ref[0], __half2float(h_v2[0]), rmse,
                   rmse < 0.05f ? "PASS" : "FAIL");
        }
        // Marlin vs CPU
        {
            double se = 0, sr = 0;
            for (int i = 0; i < M * N; i++) {
                float g = __half2float(h_ml[i]), r = h_ref[i];
                se += (double)(g - r) * (g - r);
                sr += (double)r * r;
            }
            float rmse = (sr > 0) ? (float)sqrt(se / sr) : 0;
            printf("  Marlin vs CPU: [0,0] CPU=%.4f ML=%.4f  RMSE=%.6f (%s)\n",
                   h_ref[0], __half2float(h_ml[0]), rmse,
                   rmse < 0.05f ? "PASS" : "FAIL");
        }
        // v2 vs Marlin
        {
            double se = 0, sr = 0;
            for (int i = 0; i < M * N; i++) {
                float g = __half2float(h_v2[i]), r = __half2float(h_ml[i]);
                se += (double)(g - r) * (g - r);
                sr += (double)r * r;
            }
            float rmse = (sr > 0) ? (float)sqrt(se / sr) : 0;
            printf("  v2 vs Marlin:  RMSE=%.6f (%s)\n\n",
                   rmse, rmse < 0.01f ? "MATCH" : "MISMATCH");
        }

        cudaFree(d_A); cudaFree(d_qw); cudaFree(d_sc);
        cudaFree(d_v2); cudaFree(d_ml); cudaFree(d_ws);
        free(h_A); free(h_qw); free(h_sc);
        free(h_ref); free(h_v2); free(h_ml);
    }

    // ================================================================
    // Diagnostic 2: M=128 with CPU reference (debug large-M RMSE)
    // ================================================================
    {
        printf("--- DIAGNOSTIC 2: K=256, N=256, M=128 ---\n");
        constexpr int K = 256, N = 256, M = 128, gs = 128;

        size_t a_bytes  = (size_t)M * K * sizeof(__half);
        size_t qw_bytes = (size_t)(K / 8) * N * sizeof(uint32_t);
        size_t sc_bytes = (size_t)(K / gs) * N * sizeof(__half);
        size_t c_bytes  = (size_t)M * N * sizeof(__half);

        __half*   h_A   = (__half*)malloc(a_bytes);
        uint32_t* h_qw  = (uint32_t*)malloc(qw_bytes);
        __half*   h_sc  = (__half*)malloc(sc_bytes);
        float*    h_ref = (float*)malloc(M * N * sizeof(float));
        __half*   h_v2  = (__half*)malloc(c_bytes);
        __half*   h_ml  = (__half*)malloc(c_bytes);

        srand(42);
        for (int i = 0; i < M * K; i++)
            h_A[i] = __float2half(((rand() % 1000) - 500) / 500.0f);
        for (int i = 0; i < (K / 8) * N; i++)
            h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
        for (int i = 0; i < (K / gs) * N; i++)
            h_sc[i] = __float2half(((rand() % 200) - 100) / 1000.0f);

        cpu_gptq_ref(h_A, h_qw, h_ref, h_sc, M, K, N, gs);

        __half *d_A, *d_sc, *d_v2, *d_ml; uint32_t *d_qw;
        cudaMalloc(&d_A, a_bytes); cudaMalloc(&d_qw, qw_bytes);
        cudaMalloc(&d_sc, sc_bytes); cudaMalloc(&d_v2, c_bytes);
        cudaMalloc(&d_ml, c_bytes);
        cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_qw, h_qw, qw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc, h_sc, sc_bytes, cudaMemcpyHostToDevice);

        void* tmp; cudaMalloc(&tmp, qw_bytes);
        repack_gptq_to_marlin(d_qw, d_sc, K, N, d_perm, d_scale_perm, tmp, qw_bytes);
        cudaFree(tmp); cudaDeviceSynchronize();

        int ws_bytes = marlin_workspace_size(N);
        int* d_ws; cudaMalloc(&d_ws, ws_bytes);

        cudaMemset(d_v2, 0, c_bytes);
        gptq_gemm_v2(d_A, d_qw, d_v2, d_sc, M, K, N);
        cudaMemset(d_ws, 0, ws_bytes);
        cudaMemset(d_ml, 0, c_bytes);
        marlin_gemm(d_A, d_qw, d_ml, d_sc, d_ws, M, K, N, gs);
        cudaDeviceSynchronize();

        cudaMemcpy(h_v2, d_v2, c_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ml, d_ml, c_bytes, cudaMemcpyDeviceToHost);

        // Sample values from row 0, row 64, row 127
        int rows[] = {0, 64, 127};
        for (int r : rows) {
            printf("  Row %-3d col 0: CPU=%.4f v2=%.4f ML=%.4f\n",
                   r, h_ref[r*N], __half2float(h_v2[r*N]), __half2float(h_ml[r*N]));
        }

        // RMSE checks
        auto rmse_check = [&](const char* label, __half* gpuBuf) {
            double se = 0, sr = 0;
            for (int i = 0; i < M * N; i++) {
                float g = __half2float(gpuBuf[i]), r = h_ref[i];
                se += (double)(g - r) * (g - r);
                sr += (double)r * r;
            }
            float rmse = (sr > 0) ? (float)sqrt(se / sr) : 0;
            printf("  %s vs CPU: RMSE=%.6f (%s)\n", label, rmse,
                   rmse < 0.05f ? "PASS" : "FAIL");
        };
        rmse_check("v2    ", h_v2);
        rmse_check("Marlin", h_ml);

        // v2 vs Marlin
        {
            double se = 0, sr = 0;
            for (int i = 0; i < M * N; i++) {
                float g = __half2float(h_v2[i]), r = __half2float(h_ml[i]);
                se += (double)(g - r) * (g - r);
                sr += (double)r * r;
            }
            float rmse = (sr > 0) ? (float)sqrt(se / sr) : 0;
            printf("  v2 vs Marlin:  RMSE=%.6f\n\n", rmse);
        }

        cudaFree(d_A); cudaFree(d_qw); cudaFree(d_sc);
        cudaFree(d_v2); cudaFree(d_ml); cudaFree(d_ws);
        free(h_A); free(h_qw); free(h_sc);
        free(h_ref); free(h_v2); free(h_ml);
    }

    // ================================================================
    // Main benchmark: real model shapes
    // Verify v2 against naive GPU kernel (original GPTQ format),
    // then benchmark v2 vs Marlin for timing comparison.
    // ================================================================

    struct Case { const char* name; int K, N; };
    Case cases[] = {
        {"gate_proj (5120->17408)", 5120, 17408},
        {"down_proj (17408->5120)", 17408, 5120},
    };
    int M_vals[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    constexpr int MAX_M = 512;
    constexpr int WARMUP = 5, ITERS = 20;
    constexpr float PEAK_TFLOPS = 69.0f;

    for (auto& c : cases) {
        printf("--- %s ---\n", c.name);
        printf("%-6s %10s %10s %8s %9s %6s %10s\n",
               "M", "v2(us)", "Marlin(us)", "v2/ML", "v2 TFLOP", "TC%", "v2 RMSE");
        printf("--------------------------------------------------------------\n");

        int K = c.K, N = c.N;
        constexpr int gs = 128;

        size_t a_bytes  = (size_t)MAX_M * K * sizeof(__half);
        size_t qw_bytes = (size_t)(K / 8) * N * sizeof(uint32_t);
        size_t sc_bytes = (size_t)(K / gs) * N * sizeof(__half);
        size_t c_bytes  = (size_t)MAX_M * N * sizeof(__half);

        __half*   h_A  = (__half*)malloc(a_bytes);
        uint32_t* h_qw = (uint32_t*)malloc(qw_bytes);
        __half*   h_sc = (__half*)malloc(sc_bytes);

        srand(42);
        for (int i = 0; i < MAX_M * K; i++)
            h_A[i] = __float2half(((rand() % 1000) - 500) / 500.0f);
        for (int i = 0; i < (K / 8) * N; i++)
            h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
        for (int i = 0; i < (K / gs) * N; i++)
            h_sc[i] = __float2half(((rand() % 200) - 100) / 1000.0f);

        // Device buffers: original GPTQ format (for naive ref) + Marlin format
        __half *d_A, *d_v2, *d_ml, *d_ref;
        uint32_t *d_qw_orig, *d_qw_marlin;
        __half *d_sc_orig, *d_sc_marlin;
        cudaMalloc(&d_A,  a_bytes);
        cudaMalloc(&d_qw_orig, qw_bytes);
        cudaMalloc(&d_sc_orig, sc_bytes);
        cudaMalloc(&d_qw_marlin, qw_bytes);
        cudaMalloc(&d_sc_marlin, sc_bytes);
        cudaMalloc(&d_v2, c_bytes);
        cudaMalloc(&d_ml, c_bytes);
        cudaMalloc(&d_ref, c_bytes);

        cudaMemcpy(d_A,  h_A,  a_bytes,  cudaMemcpyHostToDevice);
        cudaMemcpy(d_qw_orig, h_qw, qw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc_orig, h_sc, sc_bytes, cudaMemcpyHostToDevice);
        // Copy for Marlin repack (in-place)
        cudaMemcpy(d_qw_marlin, h_qw, qw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc_marlin, h_sc, sc_bytes, cudaMemcpyHostToDevice);

        // Repack the copy to Marlin format
        void* tmp;
        cudaMalloc(&tmp, qw_bytes);
        repack_gptq_to_marlin(d_qw_marlin, d_sc_marlin, K, N,
                              d_perm, d_scale_perm, tmp, qw_bytes);
        cudaFree(tmp);
        cudaDeviceSynchronize();

        int ws_bytes = marlin_workspace_size(N);
        int* d_ws;
        cudaMalloc(&d_ws, ws_bytes);

        __half* h_v2  = (__half*)malloc(c_bytes);
        __half* h_ref = (__half*)malloc(c_bytes);

        for (int M : M_vals) {
            size_t out_bytes = (size_t)M * N * sizeof(__half);

            // --- Naive GPU reference (original GPTQ weights) ---
            cudaMemset(d_ref, 0, out_bytes);
            dim3 ref_grid((N + 15) / 16, (M + 15) / 16);
            dim3 ref_block(16, 16);
            gptq_gemm_naive_ref<<<ref_grid, ref_block>>>(
                d_A, d_qw_orig, d_ref, d_sc_orig, M, K, N, gs);

            // --- v2 kernel (Marlin-format weights) ---
            cudaMemset(d_v2, 0, out_bytes);
            gptq_gemm_v2(d_A, d_qw_marlin, d_v2, d_sc_marlin, M, K, N);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("%-6d  v2 error: %s\n", M, cudaGetErrorString(err));
                continue;
            }
            cudaDeviceSynchronize();

            // --- Correctness: v2 vs naive GPU reference RMSE ---
            cudaMemcpy(h_v2,  d_v2,  out_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ref, d_ref, out_bytes, cudaMemcpyDeviceToHost);

            double se = 0, sr = 0;
            for (int i = 0; i < M * N; i++) {
                float g = __half2float(h_v2[i]);
                float r = __half2float(h_ref[i]);
                se += (double)(g - r) * (g - r);
                sr += (double)r * r;
            }
            float rmse = (sr > 0) ? (float)sqrt(se / sr) : 0;

            // --- Benchmark v2 ---
            for (int i = 0; i < WARMUP; i++)
                gptq_gemm_v2(d_A, d_qw_marlin, d_v2, d_sc_marlin, M, K, N);
            cudaDeviceSynchronize();

            cudaEventRecord(e0);
            for (int i = 0; i < ITERS; i++)
                gptq_gemm_v2(d_A, d_qw_marlin, d_v2, d_sc_marlin, M, K, N);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms_v2;
            cudaEventElapsedTime(&ms_v2, e0, e1);
            float us_v2 = ms_v2 * 1000.0f / ITERS;

            // --- Benchmark Marlin ---
            for (int i = 0; i < WARMUP; i++) {
                cudaMemset(d_ws, 0, ws_bytes);
                marlin_gemm(d_A, d_qw_marlin, d_ml, d_sc_marlin,
                            d_ws, M, K, N, gs);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(e0);
            for (int i = 0; i < ITERS; i++) {
                cudaMemset(d_ws, 0, ws_bytes);
                marlin_gemm(d_A, d_qw_marlin, d_ml, d_sc_marlin,
                            d_ws, M, K, N, gs);
            }
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms_ml;
            cudaEventElapsedTime(&ms_ml, e0, e1);
            float us_ml = ms_ml * 1000.0f / ITERS;

            float ratio = us_v2 / us_ml;
            float tflops = (float)(2.0 * M * K * N / (us_v2 * 1e6));
            float tc_pct = tflops / PEAK_TFLOPS * 100.0f;

            printf("%-6d %10.1f %10.1f %7.2fx %9.3f %5.1f%% %10.6f\n",
                   M, us_v2, us_ml, ratio, tflops, tc_pct, rmse);
        }

        cudaFree(d_A); cudaFree(d_qw_orig); cudaFree(d_sc_orig);
        cudaFree(d_qw_marlin); cudaFree(d_sc_marlin);
        cudaFree(d_v2); cudaFree(d_ml); cudaFree(d_ref); cudaFree(d_ws);
        free(h_A); free(h_qw); free(h_sc);
        free(h_v2); free(h_ref);
        printf("\n");
    }

    cudaFree(d_perm);
    cudaFree(d_scale_perm);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

} // namespace deusridet
