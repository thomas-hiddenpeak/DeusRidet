// marlin.cu — Marlin GPTQ INT4 GEMM kernel adapted for SM87 (Orin)
//
// Adapted from IST-DASLab/marlin (Apache 2.0 License).
// Original: https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
// Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
//
// Modifications for DeusRidet:
//   - Wrapped in deusridet namespace
//   - Added C++ weight repacking (replaces Python pack())
//   - Integrated with GptqWeight / ModelWeights structures
//   - SM87 target (96KB SMEM verified within 128KB unified budget)
//   - Stripped per-column quantization configs (we always use group_size=128)

#include "marlin.h"
#include "model.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace deusridet {

// ============================================================================
// Section 1: Constants and utility types
// ============================================================================

static constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

// Register-level vector: groups of registers for tensor core operands.
// All index accesses must be compile-time constants (use #pragma unroll).
template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

// MMA fragment types for m16n8k16 tensor core:
// See: PTX ISA § Matrix Fragments for mma.m16n8k16
using FragA = Vec<half2, 4>;   // A operand: 4 half2 = 8 halves
using FragB = Vec<half2, 2>;   // B operand: 2 half2 = 4 halves
using FragC = Vec<float, 4>;   // Accumulator: 4 floats
using FragS = Vec<half2, 1>;   // Scale: 1 half2 = 2 halves

// ============================================================================
// Section 2: PTX inline assembly wrappers
// ============================================================================

// Predicated async global→shared copy (16 bytes). Used for input A.
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

// Async global→shared copy with L2 evict-first hint (16 bytes).
// Used for quantized weights B: accessed once, should not pollute L2.
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

// Async copy fence (commit group).
__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy groups are still pending.
template <int n>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// m16n8k16 tensor core MMA: C += A * B (FP16 inputs, FP32 accumulate).
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

// Load 16x16 A-operand fragment from shared memory via ldmatrix.
__device__ __forceinline__ void ldsm4(FragA& frag_a, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

// 3-input logical operation (LUT-based).
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
    );
    return res;
}

// Efficient INT4→FP16 dequantization via bit manipulation (lop3).
// Converts a packed int32 of 4 INT4 values to a FragB of 4 FP16 values.
// Strategy: pack INT4 nibbles into FP16 exponent field, subtract bias.
// Adapted from NVIDIA/FasterTransformer interleaved_numeric_conversion.h
__device__ __forceinline__ FragB dequant(int q) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;
    // (a & b) | c  →  lop3 with LUT = (0xf0 & 0xcc) | 0xaa
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    // Fuse symmetric zero point (-8) directly into SUB/ADD constants
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

// Multiply dequantized fragment by quantization scale (grouped quantization).
__device__ __forceinline__ void scale(FragB& frag_b, FragS& frag_s, int i) {
    half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
    frag_b[0] = __hmul2(frag_b[0], s);
    frag_b[1] = __hmul2(frag_b[1], s);
}

// Global barrier: wait until lock reaches expected count.
__device__ __forceinline__ void barrier_acquire(int* lock, int count) {
    if (threadIdx.x == 0) {
        int state = -1;
        do
            asm volatile(
                "ld.global.acquire.gpu.b32 %0, [%1];\n"
                : "=r"(state) : "l"(lock));
        while (state != count);
    }
    __syncthreads();
}

// Release barrier and increment visit count.
__device__ __forceinline__ void barrier_release(int* lock, bool reset = false) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) { lock[0] = 0; return; }
        int val = 1;
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile(
            "red.relaxed.gpu.global.add.s32 [%0], %1;\n"
            : : "l"(lock), "r"(val));
    }
}

// ============================================================================
// Section 3: Marlin kernel template
// ============================================================================

template <
    const int threads,          // threads per block (256)
    const int thread_m_blocks,  // 16-row blocks in M dimension per threadblock
    const int thread_n_blocks,  // 16-col blocks in N dimension per threadblock
    const int thread_k_blocks,  // 16-element blocks in K dimension per threadblock
    const int stages,           // async pipeline stages (4)
    const int group_blocks = -1 // consecutive 16x16 blocks per scale group (-1 = per-column)
>
__global__ void Marlin(
    const int4* __restrict__ A,   // FP16 input [M, K]
    const int4* __restrict__ B,   // INT4 quantized weights [K/16, N*16/8] in Marlin format
          int4* __restrict__ C,   // FP16 output [M, N]
    const int4* __restrict__ s,   // FP16 scales [K/groupsize, N] Marlin-permuted
    int prob_m, int prob_n, int prob_k,
    int* locks
) {
    // Striped partitioning: each threadblock processes one "stripe" of B.
    // Stripes ensure good utilization across all SMs with minimal global reductions.
    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks) {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m = 16 * thread_m_blocks;
    }

    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

    // Ensure stripe boundaries align with group boundaries
    if (group_blocks != -1)
        iters = (group_blocks / thread_k_blocks) *
                ceildiv(iters, (group_blocks / thread_k_blocks));

    int slice_row = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col = slice_col_par;
    int slice_iters;
    int slice_count = 0;
    int slice_idx;

    // Handle parallel batch problem instances
    if (slice_col_par >= n_tiles) {
        A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
        C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
        locks += (slice_col_par / n_tiles) * n_tiles;
        slice_col = slice_col_par % n_tiles;
    }

    // Compute slice metadata for synchronization
    auto init_slice = [&]() {
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
            slice_iters = 0;
        if (slice_iters == 0) return;
        if (slice_row + slice_iters > k_tiles)
            slice_iters = k_tiles - slice_row;
        slice_count = 1;
        slice_idx = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1)) {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0) slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0))
                slice_idx = slice_count - 1;
            else {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0) slice_idx--;
            }
        }
        if (slice_col == n_tiles) {
            A += 16 * thread_m_blocks * prob_k / 8;
            C += 16 * thread_m_blocks * prob_n / 8;
            locks += n_tiles;
            slice_col = 0;
        }
    };
    init_slice();

    // Stride calculations (all in int4 = 16-byte units)
    int a_gl_stride = prob_k / 8;
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

    int b_gl_stride = 16 * prob_n / 32;
    constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
    constexpr int b_sh_wr_delta = threads;
    constexpr int b_sh_rd_delta = threads;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

    int s_gl_stride = prob_n / 8;
    constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
    constexpr int s_sh_stage = s_sh_stride;
    int s_gl_rd_delta = s_gl_stride;

    // Thread-level read/write indices
    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) +
                  (threadIdx.x % 32) / 16;
    a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                  s_sh_stride * slice_col + threadIdx.x;
    int s_sh_wr = threadIdx.x;
    int s_sh_rd;
    if (group_blocks != -1)
        s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                  (threadIdx.x % 32) / 4;
    else
        s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                  (threadIdx.x % 32) % 4;

    // Predication for boundary handling
    bool a_sh_wr_pred[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
    bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

    // XOR-based SMEM layout for fully bank-conflict-free A tile access
    auto transform_a = [&](int i) {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };

    // Precompute transformed SMEM indices (all accesses are static after unrolling)
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

    // Pre-split B pointers to break dependency chains between iterations
    const int4* B_ptr[b_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    // Dynamic shared memory: A tiles | B tiles | S tiles
    extern __shared__ int4 sh[];
    int4* sh_a = sh;
    int4* sh_b = sh_a + (stages * a_sh_stage);
    int4* sh_s = sh_b + (stages * b_sh_stage);

    // Register double-buffer for SMEM→register loads
    FragA frag_a[2][thread_m_blocks];
    I4 frag_b_quant[2];
    FragC frag_c[thread_m_blocks][4][2];
    FragS frag_s[2][4];

    // Zero accumulators
    auto zero_accums = [&]() {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
            reinterpret_cast<float*>(frag_c)[i] = 0;
    };

    // Async fetch next A, B, s tiles from global to shared memory pipeline stage
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
        if (pred) {
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(
                    &sh_a_stage[a_sh_wr_trans[i]],
                    &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                    a_sh_wr_pred[i]);
            }
            int4* sh_b_stage = sh_b + b_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++) {
                cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            if (group_blocks != -1 &&
                pipe % (group_blocks / thread_k_blocks) == 0)
            {
                int4* sh_s_stage = sh_s + s_sh_stage * pipe;
                if (s_sh_wr_pred)
                    cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
                s_gl_rd += s_gl_rd_delta;
            }
        }
        cp_async_fence();
    };

    // Wait for next pipeline stage to be ready in shared memory
    auto wait_for_stage = [&]() {
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    // Load sub-tile from shared memory into register double-buffer
    auto fetch_to_registers = [&](int k, int pipe) {
        if (group_blocks != -1) {
            int4* sh_s_stage = sh_s + s_sh_stage *
                ((group_blocks / thread_k_blocks) *
                 (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        }
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
            ldsm4(frag_a[k % 2][i],
                   &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4* sh_b_stage = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4*>(
            &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    // Execute tensor core matmul for one sub-tile
    auto matmul = [&](int k) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int b_quant = frag_b_quant[k % 2][j];
            int b_quant_shift = b_quant >> 8;
            FragB frag_b0 = dequant(b_quant);
            if (group_blocks != -1)
                scale(frag_b0, frag_s[k % 2][j], 0);
            FragB frag_b1 = dequant(b_quant_shift);
            if (group_blocks != -1)
                scale(frag_b1, frag_s[k % 2][j], 1);
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    // Thread-block level reduction (multiple warps → single result per output position)
    auto thread_block_reduce = [&]() {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1) {
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

    // Global reduction across threadblocks in the same column slice (via L2 cache)
    auto global_reduce = [&](bool first = false, bool last = false) {
        constexpr int active_threads = 32 * thread_n_blocks / 4;
        if (threadIdx.x < active_threads) {
            int c_gl_stride = prob_n / 8;
            int c_gl_wr_delta_o = 8 * c_gl_stride;
            int c_gl_wr_delta_i = 4 * (active_threads / 32);
            int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                          4 * (threadIdx.x / 32) + threadIdx.x % 4;
            c_gl_wr += (2 * thread_n_blocks) * slice_col;
            constexpr int c_sh_wr_delta = active_threads;
            int c_sh_wr = threadIdx.x;
            int row = (threadIdx.x % 32) / 4;

            if (!first) {
                #pragma unroll
                for (int i = 0; i < thread_m_blocks * 4; i++) {
                    cp_async4_pred(
                        &sh[c_sh_wr + c_sh_wr_delta * i],
                        &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                           c_gl_wr_delta_i * (i % 2)],
                        i < (thread_m_blocks - 1) * 4 ||
                            8 * (i / 2) + row < prob_m);
                }
                cp_async_fence();
                cp_async_wait<0>();
            }

            #pragma unroll
            for (int i = 0; i < thread_m_blocks * 4; i++) {
                if (i < (thread_m_blocks - 1) * 4 ||
                    8 * (i / 2) + row < prob_m)
                {
                    if (!first) {
                        int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
                        #pragma unroll
                        for (int j = 0; j < 2 * 4; j++) {
                            reinterpret_cast<float*>(frag_c)
                                [4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                                __half2float(
                                    reinterpret_cast<__half*>(&c_red)[j]);
                        }
                    }
                    if (!last) {
                        int4 c;
                        #pragma unroll
                        for (int j = 0; j < 2 * 4; j++) {
                            reinterpret_cast<__half*>(&c)[j] = __float2half(
                                reinterpret_cast<float*>(frag_c)
                                    [4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
                        }
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                          c_gl_wr_delta_i * (i % 2)] = c;
                    }
                }
            }
        }
    };

    // Write final result, reshuffling from fragment layout to row-major
    auto write_result = [&]() {
        int c_gl_stride = prob_n / 8;
        constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
        int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
        constexpr int c_sh_rd_delta =
            c_sh_stride * (threads / (2 * thread_n_blocks));

        int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                      (threadIdx.x % (2 * thread_n_blocks));
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
        int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) +
                      (threadIdx.x % 32) % 4;
        c_sh_wr += 32 * (threadIdx.x / 32);
        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                      (threadIdx.x % (2 * thread_n_blocks));

        int c_gl_wr_end = c_gl_stride * prob_m;

        auto write = [&](int idx, float c0, float c1, FragS& s) {
            half2 res = __halves2half2(__float2half(c0), __float2half(c1));
            if (group_blocks == -1)  // per-column scale applied here
                res = __hmul2(res, s[0]);
            ((half2*)sh)[idx] = res;
        };

        if (threadIdx.x / 32 < thread_n_blocks / 4) {
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wr = c_sh_wr + 8 * j;
                    write(wr + (4 * c_sh_stride) * 0 + 0,
                          frag_c[i][j][0][0], frag_c[i][j][0][1],
                          frag_s[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * c_sh_stride) * 8 + 0,
                          frag_c[i][j][0][2], frag_c[i][j][0][3],
                          frag_s[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * c_sh_stride) * 0 + 4,
                          frag_c[i][j][1][0], frag_c[i][j][1][1],
                          frag_s[j / 2][2 * (j % 2) + 1]);
                    write(wr + (4 * c_sh_stride) * 8 + 4,
                          frag_c[i][j][1][2], frag_c[i][j][1][3],
                          frag_s[j / 2][2 * (j % 2) + 1]);
                }
                c_sh_wr += 16 * (4 * c_sh_stride);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0;
             i < ceildiv(16 * thread_m_blocks,
                         threads / (2 * thread_n_blocks));
             i++)
        {
            if (c_gl_wr < c_gl_wr_end) {
                C[c_gl_wr] = sh[c_sh_rd];
                c_gl_wr += c_gl_wr_delta;
                c_sh_rd += c_sh_rd_delta;
            }
        }
    };

    // ---- Main execution pipeline ----

    auto start_pipes = [&]() {
        #pragma unroll
        for (int i = 0; i < stages - 1; i++)
            fetch_to_shared(i, i, i < slice_iters);
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    // Main loop: interleaved fetch + compute
    while (slice_iters) {
        #pragma unroll
        for (int pipe = 0; pipe < stages;) {
            #pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2) {
                    fetch_to_shared((pipe + stages - 1) % stages,
                                    pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0) break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;

        // End-of-slice processing
        if (slice_iters == 0) {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;

            if (group_blocks == -1 && last) {
                if (s_sh_wr_pred)
                    cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);
                cp_async_fence();
            }
            thread_block_reduce();
            if (group_blocks == -1 && last) {
                cp_async_wait<0>();
                __syncthreads();
                if (threadIdx.x / 32 < thread_n_blocks / 4) {
                    reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
                    reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
                }
            }

            if (slice_count > 1) {
                barrier_acquire(&locks[slice_col], slice_idx);
                global_reduce(slice_idx == 0, last);
                barrier_release(&locks[slice_col], last);
            }
            if (last)
                write_result();

            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters) {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                          (threadIdx.x % a_gl_rd_delta_o);
                #pragma unroll
                for (int i = 0; i < b_sh_wr_iters; i++)
                    B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
                if (slice_col == 0) {
                    #pragma unroll
                    for (int i = 0; i < b_sh_wr_iters; i++)
                        B_ptr[i] -= b_gl_stride;
                }
                s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}

// ============================================================================
// Section 4: Kernel dispatch
// ============================================================================

static const int THREADS = 256;
static const int STAGES = 4;
static const int SHARED_MEM = 96 * 1024;  // 96 KB dynamic SMEM

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
    else if (                                                                     \
        thread_m_blocks == THREAD_M_BLOCKS &&                                     \
        thread_n_blocks == THREAD_N_BLOCKS &&                                     \
        thread_k_blocks == THREAD_K_BLOCKS &&                                     \
        group_blocks == GROUP_BLOCKS                                              \
    ) {                                                                           \
        cudaFuncSetAttribute(                                                     \
            Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,                     \
                   THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>,                        \
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);             \
        Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,                         \
               THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>                             \
            <<<blocks, THREADS, SHARED_MEM, stream>>>(                            \
                A_ptr, B_ptr, C_ptr, s_ptr, prob_m, prob_n, prob_k, locks);       \
    }

// Cached SM count — queried once, reused for all marlin_gemm calls
static int get_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return sms;
}

void marlin_gemm(
    const __half* A, const uint32_t* B, __half* C, const __half* s,
    int* workspace, int M, int K, int N,
    int groupsize, cudaStream_t stream)
{
    int sms = get_sm_count();

    int tot_m = M;
    int tot_m_blocks = ceildiv(tot_m, 16);
    int pad = 16 * tot_m_blocks - tot_m;

    // Auto-select tile dimensions based on batch size
    int thread_k, thread_n;
    if (M <= 16) {
        thread_k = 128;
        thread_n = 128;
    } else {
        thread_k = 64;
        thread_n = 256;
    }

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
    int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
    int blocks = sms;

    if (N % thread_n != 0 || K % thread_k != 0) {
        fprintf(stderr, "marlin_gemm: shape error N=%d K=%d thread_n=%d thread_k=%d\n",
                N, K, thread_n, thread_k);
        return;
    }
    if (group_blocks != -1 && K % (group_blocks * 16) != 0) {
        fprintf(stderr, "marlin_gemm: group alignment error\n");
        return;
    }
    if (M == 0 || N == 0 || K == 0) return;

    const int4* A_ptr = reinterpret_cast<const int4*>(A);
    const int4* B_ptr = reinterpret_cast<const int4*>(B);
    int4* C_ptr = reinterpret_cast<int4*>(C);
    const int4* s_ptr = reinterpret_cast<const int4*>(s);

    int* locks = workspace;

    // Zero workspace locks only when multiple threadblocks may write to the
    // same output column slice (parallel > 1, i.e. M > 64 for thread_m=4).
    // For small M (≤64), each column slice is handled by a single TB — no
    // global reduction, no locks needed. This eliminates 192 cudaMemsetAsync
    // calls per forward pass for typical decode/small-prefill.
    int max_par = 16;
    if (tot_m_blocks > 4) {
        int n_tiles_max = N / thread_n;
        cudaMemsetAsync(locks, 0, n_tiles_max * max_par * sizeof(int), stream);
    }

    int prob_n = N;
    int prob_k = K;

    for (int i = 0; i < tot_m_blocks; i += 4) {
        int thread_m_blocks = tot_m_blocks - i;
        int prob_m = tot_m - 16 * i;
        int par = 1;
        if (thread_m_blocks > 4) {
            par = (16 * thread_m_blocks - pad) / 64;
            if (par > max_par) par = max_par;
            prob_m = 64 * par;
            i += 4 * (par - 1);
            thread_m_blocks = 4;
        }

        // Kernel config selection for group_size=128 (group_blocks=8)
        if (false) {}
        CALL_IF(1,  8,  8,  8)
        CALL_IF(1, 16,  4,  8)
        CALL_IF(2, 16,  4,  8)
        CALL_IF(3, 16,  4,  8)
        CALL_IF(4, 16,  4,  8)
        // Also support per-column quantization for potential future use
        CALL_IF(1,  8,  8, -1)
        CALL_IF(1, 16,  4, -1)
        CALL_IF(2, 16,  4, -1)
        CALL_IF(3, 16,  4, -1)
        CALL_IF(4, 16,  4, -1)
        else {
            fprintf(stderr, "marlin_gemm: no kernel for m_blocks=%d n_blocks=%d k_blocks=%d g=%d\n",
                    thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks);
        }

        A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
        C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
    }
}

#undef CALL_IF

// ============================================================================
// Section 5: Weight repacking (GPTQ → Marlin format)
// ============================================================================

// Precompute the Marlin weight permutation table (1024 entries).
// Matches the _get_perms() function from Python marlin/__init__.py
static std::vector<int> compute_marlin_perm() {
    std::vector<int> perm;
    perm.reserve(1024);
    for (int i = 0; i < 32; i++) {
        std::vector<int> perm1;
        int col = i / 4;
        for (int block = 0; block < 2; block++) {
            int rows[] = {2 * (i % 4), 2 * (i % 4) + 1,
                          2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1};
            for (int r = 0; r < 4; r++)
                perm1.push_back(16 * rows[r] + col + 8 * block);
        }
        for (int j = 0; j < 4; j++)
            for (auto p : perm1)
                perm.push_back(p + 256 * j);
    }
    // Apply interleave: [0,2,4,6,1,3,5,7]
    int interleave[] = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<int> result(1024);
    for (int i = 0; i < 1024 / 8; i++) {
        for (int j = 0; j < 8; j++)
            result[i * 8 + j] = perm[i * 8 + interleave[j]];
    }
    return result;
}

// Precompute scale permutation (64 entries for grouped quantization).
static std::vector<int> compute_scale_perm() {
    std::vector<int> perm;
    perm.reserve(64);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            perm.push_back(i + 8 * j);
    return perm;
}

// GPU kernel: repack GPTQ qweight to Marlin B format.
// Each thread produces one output uint32 (8 packed INT4 values).
__global__ void repack_qweight_kernel(
    const uint32_t* __restrict__ in,   // GPTQ qweight [K/8, N] (copy in temp)
    uint32_t* __restrict__ out,        // Marlin B [K/16, 2*N]
    const int* __restrict__ perm,      // [1024] Marlin permutation
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (K / 16) * (2 * N);
    if (idx >= total) return;

    int tr = idx / (2 * N);     // tile row [0, K/16)
    int pc = idx % (2 * N);     // packed column [0, 2*N)

    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col_perm_idx = 8 * pc + i;
        int block = col_perm_idx / 1024;
        int within = col_perm_idx % 1024;
        int col_tiled = block * 1024 + perm[within];

        int n_tile = col_tiled / 256;
        int k_within = (col_tiled % 256) / 16;
        int n_within = col_tiled % 16;

        int k = tr * 16 + k_within;
        int n = n_tile * 16 + n_within;

        // Read INT4 from GPTQ packed format
        int k_packed = k / 8;
        int k_bit = k % 8;
        uint32_t packed = in[k_packed * N + n];
        int val = (packed >> (k_bit * 4)) & 0xF;

        result |= ((uint32_t)val << (i * 4));
    }
    out[idx] = result;
}

// GPU kernel: permute scale columns within 64-element blocks.
__global__ void repack_scales_kernel(
    const __half* __restrict__ in,      // original scales [K/128, N] (copy in temp)
    __half* __restrict__ out,           // permuted scales [K/128, N]
    const int* __restrict__ scale_perm, // [64] scale permutation
    int num_groups, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_groups * N;
    if (idx >= total) return;

    int g = idx / N;
    int n = idx % N;
    int block = n / 64;
    int within = n % 64;
    int src_n = block * 64 + scale_perm[within];

    out[g * N + n] = in[g * N + src_n];
}

void repack_gptq_to_marlin(
    uint32_t* qweight, __half* scales,
    int K, int N,
    int* d_perm, int* d_scale_perm,
    void* temp_buf, size_t temp_buf_bytes,
    cudaStream_t stream)
{
    // --- Repack qweight ---
    int qw_size = (K / 8) * N;    // original size in uint32
    int B_size  = (K / 16) * 2 * N; // Marlin size in uint32 (same total bytes)
    size_t qw_bytes = (size_t)qw_size * sizeof(uint32_t);

    // Use pre-allocated temp buffer (caller ensures it's big enough)
    uint32_t* temp_qw = static_cast<uint32_t*>(temp_buf);
    cudaMemcpyAsync(temp_qw, qweight, qw_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    int threads_per_block = 256;
    int blocks = ceildiv(B_size, threads_per_block);
    repack_qweight_kernel<<<blocks, threads_per_block, 0, stream>>>(
        temp_qw, qweight, d_perm, K, N);

    // --- Repack scales ---
    int num_groups = K / 128;
    int s_size = num_groups * N;
    size_t s_bytes = (size_t)s_size * sizeof(__half);

    // Reuse same temp buffer for scales (smaller than qweight)
    __half* temp_s = reinterpret_cast<__half*>(temp_buf);
    cudaMemcpyAsync(temp_s, scales, s_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    int s_blocks = ceildiv(s_size, threads_per_block);
    repack_scales_kernel<<<s_blocks, threads_per_block, 0, stream>>>(
        temp_s, scales, d_scale_perm, num_groups, N);
}

// Repack all MLP weights in the model. Returns workspace bytes allocated.
size_t repack_all_marlin(ModelWeights& weights, cudaStream_t stream) {
    using MC = ModelConfig;
    printf("[marlin] Repacking GPTQ weights to Marlin format...\n");

    auto t0 = std::chrono::steady_clock::now();

    // Upload permutation tables to device (once)
    static std::vector<int> h_perm = compute_marlin_perm();
    static std::vector<int> h_scale_perm = compute_scale_perm();
    int* d_perm = nullptr;
    int* d_scale_perm = nullptr;
    cudaMalloc(&d_perm, 1024 * sizeof(int));
    cudaMemcpy(d_perm, h_perm.data(), 1024 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_scale_perm, 64 * sizeof(int));
    cudaMemcpy(d_scale_perm, h_scale_perm.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate single temp buffer for largest tensor: max(gate/up, down) qweight
    // gate/up: (5120/8)*17408 = 11141120 uint32 = 42.5 MB
    // down:    (17408/8)*5120 = 11141120 uint32 = 42.5 MB (same!)
    size_t max_qw_bytes = (size_t)(MC::HIDDEN_SIZE / 8) * MC::INTERMEDIATE_SIZE * sizeof(uint32_t);
    void* temp_buf = nullptr;
    cudaMalloc(&temp_buf, max_qw_bytes);

    for (int li = 0; li < MC::NUM_LAYERS; li++) {
        auto& mlp = weights.layers[li].mlp;

        repack_gptq_to_marlin(
            const_cast<uint32_t*>(mlp.gate_proj.qweight),
            const_cast<__half*>(mlp.gate_proj.scales),
            mlp.gate_proj.K, mlp.gate_proj.N,
            d_perm, d_scale_perm, temp_buf, max_qw_bytes, stream);

        repack_gptq_to_marlin(
            const_cast<uint32_t*>(mlp.up_proj.qweight),
            const_cast<__half*>(mlp.up_proj.scales),
            mlp.up_proj.K, mlp.up_proj.N,
            d_perm, d_scale_perm, temp_buf, max_qw_bytes, stream);

        repack_gptq_to_marlin(
            const_cast<uint32_t*>(mlp.down_proj.qweight),
            const_cast<__half*>(mlp.down_proj.scales),
            mlp.down_proj.K, mlp.down_proj.N,
            d_perm, d_scale_perm, temp_buf, max_qw_bytes, stream);
    }

    cudaStreamSynchronize(stream);

    // Free temp resources (only used during repacking)
    cudaFree(temp_buf);
    cudaFree(d_perm);
    cudaFree(d_scale_perm);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[marlin] Repacked %d GPTQ weights in %.1f ms (temp buf %.1f MB, freed)\n",
           MC::NUM_LAYERS * 3, ms, max_qw_bytes / 1048576.0);

    int max_N = MC::INTERMEDIATE_SIZE;
    size_t ws_bytes = marlin_workspace_size(max_N);
    return ws_bytes;
}

} // namespace deusridet
