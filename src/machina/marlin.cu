/**
 * @file src/machina/marlin.cu
 * @philosophical_role
 *   Marlin INT4 GEMM — the kernel by which a quantized weight becomes a fast product. GPTQ is the compression scheme; Marlin is how Machina *earns* that compression back as throughput on SM87.
 * @serves
 *   Machina GPTQ path (gptq.cu, gptq_gemm_v2.cu) for quantized linear layers across the Qwen3 stack.
 */
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
#include "marlin_kernel.cuh"

// ============================================================================

// ============================================================================
// Section 4: Kernel dispatch
// ============================================================================

static const int THREADS = 256;
static const int STAGES = 4;

// Compute exact SMEM per config instead of flat 96 KB.
// SM87 has 128 KB unified L1/SMEM — right-sizing frees L1 for register spills
// and global store traffic. E.g. config (1,8,8,8) uses 49 KB → 79 KB L1
// vs old 96 KB SMEM → only 32 KB L1.
#define MARLIN_SMEM_BYTES(M, N, K) \
    (STAGES * ((16*(K)/8 * 16*(M)) + (32*(N)/4 * (K)) + (16*(N)/8)) * 16)

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
    else if (                                                                     \
        thread_m_blocks == THREAD_M_BLOCKS &&                                     \
        thread_n_blocks == THREAD_N_BLOCKS &&                                     \
        thread_k_blocks == THREAD_K_BLOCKS &&                                     \
        group_blocks == GROUP_BLOCKS                                              \
    ) {                                                                           \
        constexpr int _smem = MARLIN_SMEM_BYTES(                                  \
            THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS);                   \
        cudaFuncSetAttribute(                                                     \
            Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,                     \
                   THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>,                        \
            cudaFuncAttributeMaxDynamicSharedMemorySize, _smem);                  \
        Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,                         \
               THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>                             \
            <<<blocks, THREADS, _smem, stream>>>(                                 \
                A_ptr, B_ptr, C_ptr, s_ptr, prob_m, prob_n, prob_k, locks,        \
                residual_ptr);                                                    \
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

static void marlin_gemm_impl(
    const __half* A, const uint32_t* B, __half* C, const __half* s,
    int* workspace, int M, int K, int N,
    int groupsize, cudaStream_t stream,
    const int4* residual_ptr)
{
    int sms = get_sm_count();

    int tot_m = M;
    int tot_m_blocks = ceildiv(tot_m, 16);
    int pad = 16 * tot_m_blocks - tot_m;

    // All batch sizes use thread_k=64, thread_n=256. For M≤16 this selects
    // config (1,16,4,8): 42KB SMEM → 86KB L1 on SM87, higher compute intensity
    // per pipeline iteration (6KB data/iter vs 12KB for old k=128,n=128).
    // N must be divisible by 256 — pad if needed (see LIN_QKV_AB_DIM).
    int thread_k = 64;
    int thread_n = 256;

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
        CALL_IF(2,  8,  8,  8)
        CALL_IF(2, 16,  4,  8)
        CALL_IF(3,  8,  8,  8)
        CALL_IF(3, 16,  4,  8)
        CALL_IF(4,  8,  8,  8)
        CALL_IF(4, 16,  4,  8)
        // Also support per-column quantization for potential future use
        CALL_IF(1,  8,  8, -1)
        CALL_IF(1, 16,  4, -1)
        CALL_IF(2,  8,  8, -1)
        CALL_IF(2, 16,  4, -1)
        CALL_IF(3,  8,  8, -1)
        CALL_IF(3, 16,  4, -1)
        CALL_IF(4,  8,  8, -1)
        CALL_IF(4, 16,  4, -1)
        else {
            fprintf(stderr, "marlin_gemm: no kernel for m_blocks=%d n_blocks=%d k_blocks=%d g=%d\n",
                    thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks);
        }

        A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
        C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
        if (residual_ptr)
            residual_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
    }
}

#undef CALL_IF
#undef MARLIN_SMEM_BYTES

void marlin_gemm(
    const __half* A, const uint32_t* B, __half* C, const __half* s,
    int* workspace, int M, int K, int N,
    int groupsize, cudaStream_t stream)
{
    marlin_gemm_impl(A, B, C, s, workspace, M, K, N, groupsize, stream, nullptr);
}

// WARNING: marlin_gemm_add is BROKEN for in-place mode (C == residual) when
// slice_count > 1.  The global_reduce path uses C as scratch for partial sums,
// which corrupts the residual before write_result reads it.  This happens for
// almost all practical shapes on 16 SMs.  To use safely, pass a SEPARATE output
// buffer C != residual, then copy C → residual afterwards.  For now, prefer
// marlin_gemm + elementwise_add instead.
void marlin_gemm_add(
    const __half* A, const uint32_t* B, __half* C, const __half* s,
    int* workspace, int M, int K, int N,
    int groupsize, cudaStream_t stream)
{
    marlin_gemm_impl(A, B, C, s, workspace, M, K, N, groupsize, stream,
                     reinterpret_cast<const int4*>(C));
}


} // namespace deusridet
