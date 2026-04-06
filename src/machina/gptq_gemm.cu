// gptq_gemm.cu — GPTQ INT4 GEMM kernel for SM87 (Orin)
//
// Custom GEMM: C[M,N] = A[M,K] @ dequant(B_int4[K,N])
// Uses original GPTQ weight format, m16n8k16 Tensor Core MMA.
// Non-persistent grid, each CTA computes one [BM, BN] output tile.

#include "gptq_gemm.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cmath>
#include <cstdlib>
#include <chrono>

namespace deusridet {

// ============================================================================
// Constants
// ============================================================================

static constexpr int BM = 64;       // M-tile per CTA
static constexpr int BN = 64;       // N-tile per CTA
static constexpr int BK = 64;       // K-tile per outer loop iteration
static constexpr int THREADS = 128; // 4 warps per CTA
static constexpr int WARPS = 4;
static constexpr int WARP_M = 32;   // M-tile per warp (2×2 warp arrangement)
static constexpr int WARP_N = 32;   // N-tile per warp

// SMEM padding to avoid bank conflicts
static constexpr int A_PAD = 8;     // pad A rows (128+8=136 bytes/row, stride not power of 2)
static constexpr int B_PAD = 0;     // B_fp16 rows: stride must be even for uint32 reads

// Pipeline stages for async global→SMEM prefetch
static constexpr int STAGES = 2;

// ============================================================================
// PTX helpers (shared with fp16_gemm / marlin)
// ============================================================================

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = pred ? 16 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(s), "l"(gmem), "r"(src_size));
}

__device__ __forceinline__ void mma_m16n8k16(
    const uint32_t* a, const uint32_t* b, float* c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]));
}

// ============================================================================
// GPTQ dequant: uint32 (8 packed INT4) → 8 FP16 values
// ============================================================================
// Symmetric quantization: fp16 = (int4_val - 8) * scale
// Vectorized: lop3 embeds nibbles as FP16(1024+val), hsub2/hmul2 for math,
// prmt reorders interleaved pairs into K-contiguous order.

__device__ __forceinline__ void dequant_8_fast(
    uint32_t packed, __half scale_h, __half* out)
{
    // Mask low nibble of each 16-bit half, OR with FP16(1024.0) exponent
    // lop3(a,b,c, 0xEA) = (a & b) | c
    static constexpr uint32_t MASK  = 0x000F000F;
    static constexpr uint32_t EMBED = 0x64006400;  // FP16(1024.0) in both halves

    uint32_t r0, r1, r2, r3;
    uint32_t s4 = packed >> 4, s8 = packed >> 8, s12 = packed >> 12;
    // r0 = {fp16(1024+n4), fp16(1024+n0)}
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r0) : "r"(packed), "r"(MASK), "r"(EMBED));
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r1) : "r"(s4),     "r"(MASK), "r"(EMBED));
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r2) : "r"(s8),     "r"(MASK), "r"(EMBED));
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r3) : "r"(s12),    "r"(MASK), "r"(EMBED));

    // Subtract 1032.0 (=1024+8) and multiply by scale, all in FP16×2
    const __half sub_val = __ushort_as_half(static_cast<unsigned short>(0x6408));
    __half2 sub2 = __half2half2(sub_val);
    __half2 sc2  = __half2half2(scale_h);

    __half2 h0 = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&r0), sub2), sc2);
    __half2 h1 = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&r1), sub2), sc2);
    __half2 h2 = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&r2), sub2), sc2);
    __half2 h3 = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&r3), sub2), sc2);

    // Reorder: (n0,n4),(n1,n5),(n2,n6),(n3,n7) → n0,n1,n2,n3,n4,n5,n6,n7
    // prmt(a,b,sel): bytes 0-3 from a, bytes 4-7 from b
    uint32_t u0 = *reinterpret_cast<uint32_t*>(&h0);
    uint32_t u1 = *reinterpret_cast<uint32_t*>(&h1);
    uint32_t u2 = *reinterpret_cast<uint32_t*>(&h2);
    uint32_t u3 = *reinterpret_cast<uint32_t*>(&h3);

    uint32_t o01, o23, o45, o67;
    // o01 = {n1, n0}: low halves of u0,u1
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(o01) : "r"(u0), "r"(u1), "r"(0x5410));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(o23) : "r"(u2), "r"(u3), "r"(0x5410));
    // o45 = {n5, n4}: high halves of u0,u1
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(o45) : "r"(u0), "r"(u1), "r"(0x7632));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(o67) : "r"(u2), "r"(u3), "r"(0x7632));

    *reinterpret_cast<uint32_t*>(out + 0) = o01;
    *reinterpret_cast<uint32_t*>(out + 2) = o23;
    *reinterpret_cast<uint32_t*>(out + 4) = o45;
    *reinterpret_cast<uint32_t*>(out + 6) = o67;
}

// ============================================================================
// Naive verification kernel — no SMEM, no MMA, just scalar GEMM with dequant
// ============================================================================

__global__ void gptq_gemm_naive_kernel(
    const __half* __restrict__ A,
    const uint32_t* __restrict__ qweight,
    __half* __restrict__ C,
    const __half* __restrict__ scales,
    int M, int K, int N, int group_size)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a_val = __half2float(A[m * K + k]);
        uint32_t packed = qweight[(k / 8) * N + n];
        int int4_val = (packed >> ((k % 8) * 4)) & 0xF;
        float scale = __half2float(scales[(k / group_size) * N + n]);
        float w_val = (float(int4_val) - 8.0f) * scale;
        sum += a_val * w_val;
    }
    C[m * N + n] = __float2half(sum);
}

// ============================================================================
// Main GEMM kernel — Register-B: B loaded from global to registers per warp,
// dequant in registers, direct PTX mma.m16n8k16. Only A uses SMEM.
// This eliminates B-related SMEM and __syncthreads, spreading B loads over
// the compute phase for better memory utilization.
// ============================================================================

template <int MIN_BLOCKS>
__global__ __launch_bounds__(128, MIN_BLOCKS) void gptq_gemm_kernel(
    const __half* __restrict__ A,
    const uint32_t* __restrict__ qweight,
    __half* __restrict__ C,
    const __half* __restrict__ scales,
    int M, int K, int N, int group_size)
{
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int warp_m = (warp_id / 2) * WARP_M;  // 0 or 32
    const int warp_n = (warp_id % 2) * WARP_N;  // 0 or 32

    const int m_start = bm * BM;
    const int n_start = bn * BN;

    // ---- SMEM: double-buffered A (no B buffers needed) ----
    constexpr int A_ELEMS = BM * (BK + A_PAD);  // 4608 halfs = 9216B per buffer
    extern __shared__ char smem_raw[];

    // ---- Accumulators: 2 mt × 4 nt mma.m16n8k16 per k16 step ----
    // Total: 2 mt × 4 nt × 4 floats = 32 floats
    float acc[2][4][4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        for (int nt = 0; nt < 4; nt++)
            for (int i = 0; i < 4; i++)
                acc[mt][nt][i] = 0.0f;

    // Lane-local fragment indices for PTX mma.m16n8k16
    const int groupID = lane / 4;   // 0..7
    const int pos     = lane % 4;   // 0..3

    // ==================================================================
    // Main K loop — double-buffered A with cp.async overlap
    // ==================================================================
    const int num_k_tiles = K / BK;
    const int nib_shift = pos * 8;

    __half* const A_buf0 = reinterpret_cast<__half*>(smem_raw);
    __half* const A_buf1 = A_buf0 + A_ELEMS;

    // ---- Prologue: async load A[kt=0] to buf 0 ----
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int elem_idx = tid * 32 + i * 8;
        int row = elem_idx / BK;
        int col = elem_idx % BK;
        int g_row = m_start + row;
        __half* smem_dst = A_buf0 + row * (BK + A_PAD) + col;
        cp_async16_zfill(smem_dst, A + g_row * K + col, g_row < M);
    }
    cp_async_commit();

    for (int kt = 0; kt < num_k_tiles; kt++) {
        __half* A_cur = (kt & 1) ? A_buf1 : A_buf0;
        __half* A_nxt = (kt & 1) ? A_buf0 : A_buf1;

        // Wait for current tile's async load, sync across CTA
        cp_async_wait<0>();
        __syncthreads();

        // ---- Issue async load for next A tile (overlaps with compute below) ----
        if (kt + 1 < num_k_tiles) {
            const int next_k = (kt + 1) * BK;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int elem_idx = tid * 32 + i * 8;
                int row = elem_idx / BK;
                int col = elem_idx % BK;
                int g_row = m_start + row;
                __half* smem_dst = A_nxt + row * (BK + A_PAD) + col;
                cp_async16_zfill(smem_dst, A + g_row * K + next_k + col, g_row < M);
            }
            cp_async_commit();
        }

        // Scale for this BK tile
        const int scale_row = (kt * BK) / group_size;
        __half2 scale2[4];
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            int n_col = n_start + warp_n + nt * 8 + groupID;
            __half s = scales[scale_row * N + n_col];
            scale2[nt] = {s, s};
        }

        // ---- k16 loop: 4 steps per BK=64 tile ----
        #pragma unroll
        for (int k16 = 0; k16 < 4; k16++) {
            const int k_row0 = kt * (BK / 8) + k16 * 2;

            // ---- Load A fragments from SMEM (PTX layout) ----
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                int base_row = warp_m + mt * 16;
                const __half* a_base = A_cur + k16 * 16;
                a_frag[mt][0] = *reinterpret_cast<const uint32_t*>(
                    a_base + (base_row + groupID)     * (BK + A_PAD) + 2 * pos);
                a_frag[mt][1] = *reinterpret_cast<const uint32_t*>(
                    a_base + (base_row + groupID + 8) * (BK + A_PAD) + 2 * pos);
                a_frag[mt][2] = *reinterpret_cast<const uint32_t*>(
                    a_base + (base_row + groupID)     * (BK + A_PAD) + 8 + 2 * pos);
                a_frag[mt][3] = *reinterpret_cast<const uint32_t*>(
                    a_base + (base_row + groupID + 8) * (BK + A_PAD) + 8 + 2 * pos);
            }

            // ---- Phase 1: Pre-issue all B loads for this k16 step ----
            uint32_t bp0[4], bp1[4];
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int n_col = n_start + warp_n + nt * 8 + groupID;
                bp0[nt] = qweight[k_row0 * N + n_col];
                bp1[nt] = qweight[(k_row0 + 1) * N + n_col];
            }

            // ---- Phase 2: Dequant + MMA (B data should be in registers now) ----
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                uint32_t byte0 = (bp0[nt] >> nib_shift) & 0xFF;
                uint32_t byte1 = (bp1[nt] >> nib_shift) & 0xFF;
                uint32_t emb0_raw = (0x6400u | (byte0 & 0xF))
                                  | ((0x6400u | (byte0 >> 4)) << 16);
                uint32_t emb1_raw = (0x6400u | (byte1 & 0xF))
                                  | ((0x6400u | (byte1 >> 4)) << 16);

                __half2 emb0 = *reinterpret_cast<__half2*>(&emb0_raw);
                __half2 emb1 = *reinterpret_cast<__half2*>(&emb1_raw);
                const __half2 c1032 = {__ushort_as_half(0x6408u), __ushort_as_half(0x6408u)};
                __half2 val0 = __hmul2(__hsub2(emb0, c1032), scale2[nt]);
                __half2 val1 = __hmul2(__hsub2(emb1, c1032), scale2[nt]);

                uint32_t b[2];
                b[0] = *reinterpret_cast<uint32_t*>(&val0);
                b[1] = *reinterpret_cast<uint32_t*>(&val1);

                #pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    mma_m16n8k16(a_frag[mt], b, acc[mt][nt]);
                }
            }
        }
    }

    // ==================================================================
    // Write output C: acc floats → FP16 global (half2 vectorized stores)
    // Each MMA tile is 16×8. Thread holds c[0..3] for two adjacent columns
    // in two rows (groupID and groupID+8). Pack c[0],c[1] and c[2],c[3]
    // as half2 for 32-bit coalesced stores (16 STG.32 instead of 32 STG.16).
    // ==================================================================
    for (int mt = 0; mt < 2; mt++) {
        for (int nt = 0; nt < 4; nt++) {
            int tile_m = m_start + warp_m + mt * 16;
            int tile_n = n_start + warp_n + nt * 8;

            int r0 = tile_m + groupID;
            int c0 = tile_n + 2 * pos;  // always even → half2-aligned

            if (r0 < M && c0 + 1 < N) {
                __half2 val01 = __halves2half2(__float2half(acc[mt][nt][0]),
                                              __float2half(acc[mt][nt][1]));
                *reinterpret_cast<__half2*>(&C[r0 * N + c0]) = val01;
            } else if (r0 < M && c0 < N) {
                C[r0 * N + c0] = __float2half(acc[mt][nt][0]);
            }

            int r1 = tile_m + groupID + 8;
            if (r1 < M && c0 + 1 < N) {
                __half2 val23 = __halves2half2(__float2half(acc[mt][nt][2]),
                                              __float2half(acc[mt][nt][3]));
                *reinterpret_cast<__half2*>(&C[r1 * N + c0]) = val23;
            } else if (r1 < M && c0 < N) {
                C[r1 * N + c0] = __float2half(acc[mt][nt][2]);
            }
        }
    }
}

// ============================================================================
// Host dispatch
// ============================================================================

void gptq_gemm(
    const __half* A, const uint32_t* qweight, __half* C,
    const __half* scales,
    int M, int K, int N, int group_size, cudaStream_t stream)
{
    if (M == 0 || K == 0 || N == 0) return;

    // SMEM: double-buffered A (register-B kernel has no B in SMEM)
    constexpr int A_BYTES       = BM * (BK + A_PAD) * sizeof(__half);
    constexpr int SMEM_BYTES    = 2 * A_BYTES;  // double-buffered A

    // Dual-path: 5 blocks/SM for small M (more occupancy), 4 blocks for large M (more regs)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(gptq_gemm_kernel<5>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        cudaFuncSetAttribute(gptq_gemm_kernel<4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        smem_configured = true;
    }

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    if (M <= 64)
        gptq_gemm_kernel<5><<<grid, THREADS, SMEM_BYTES, stream>>>(
            A, qweight, C, scales, M, K, N, group_size);
    else
        gptq_gemm_kernel<4><<<grid, THREADS, SMEM_BYTES, stream>>>(
            A, qweight, C, scales, M, K, N, group_size);
}

// ============================================================================
// GPTQ INT4 GEMV (M=1) — bandwidth-bound, simple kernel
// ============================================================================

static constexpr int GEMV_THREADS = 256;
static constexpr int GEMV_N_PER_BLOCK = 64;  // N outputs per block

__global__ void gptq_gemv_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ qweight,
    __half* __restrict__ y,
    const __half* __restrict__ scales,
    int K, int N, int group_size)
{
    const int n_start = blockIdx.x * GEMV_N_PER_BLOCK;
    const int tid = threadIdx.x;

    // Each thread accumulates one or more N outputs
    // With 256 threads and 64 N per block: some threads are idle
    // Better: each thread handles 1 N column, all 256 threads reduce along K

    // Strategy: 256 threads split K, each summing a portion, then warp-reduce
    const int n_col = n_start + (tid % GEMV_N_PER_BLOCK);
    if (n_col >= N) return;

    const int k_per_thread = K / GEMV_THREADS;
    const int k_start_t = (tid / GEMV_N_PER_BLOCK) * (K / (GEMV_THREADS / GEMV_N_PER_BLOCK));

    // Simple approach: each of 4 thread groups (64 threads each) handles one K slice
    // tid / 64 = k_group (0-3), tid % 64 = n_offset
    const int k_groups = GEMV_THREADS / GEMV_N_PER_BLOCK;  // 4
    const int k_group = tid / GEMV_N_PER_BLOCK;
    const int n_offset = tid % GEMV_N_PER_BLOCK;
    const int n_global = n_start + n_offset;
    if (n_global >= N) return;

    const int k_chunk = K / k_groups;
    const int k_begin = k_group * k_chunk;
    const int k_end = (k_group == k_groups - 1) ? K : k_begin + k_chunk;

    float sum = 0.0f;
    for (int k = k_begin; k < k_end; k += 8) {
        uint32_t packed = qweight[(k / 8) * N + n_global];
        float s = __half2float(scales[(k / group_size) * N + n_global]);

        float x_vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            x_vals[i] = __half2float(x[k + i]);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int val = (packed >> (i * 4)) & 0xF;
            sum += x_vals[i] * (float(val) - 8.0f) * s;
        }
    }

    // Reduce across k_groups via shared memory
    __shared__ float reduce_smem[GEMV_THREADS];
    reduce_smem[tid] = sum;
    __syncthreads();

    // k_group 0 accumulates from all groups for each N column
    if (k_group == 0) {
        float total = reduce_smem[n_offset];
        for (int g = 1; g < k_groups; g++)
            total += reduce_smem[g * GEMV_N_PER_BLOCK + n_offset];
        y[n_global] = __float2half(total);
    }
}

void gptq_gemv(
    const __half* x, const uint32_t* qweight, __half* y,
    const __half* scales,
    int K, int N, int group_size, cudaStream_t stream)
{
    if (K == 0 || N == 0) return;

    int blocks = (N + GEMV_N_PER_BLOCK - 1) / GEMV_N_PER_BLOCK;
    gptq_gemv_kernel<<<blocks, GEMV_THREADS, 0, stream>>>(
        x, qweight, y, scales, K, N, group_size);
}

// ============================================================================
// CPU reference dequant + GEMM for correctness checking
// ============================================================================

// ============================================================================
// Standalone benchmark
// ============================================================================

void bench_new_gptq_kernels() {
    printf("\n=== New GPTQ INT4 Kernel Benchmark (SM87 Orin) ===\n");
    printf("Format: original GPTQ (qweight[K/8,N] uint32, scales[K/gs,N] FP16)\n");
    printf("Dequant: symmetric, zero_point=-8, group_size=128\n\n");

    // ---- Quick diagnostic with small matrix ----
    {
        printf("--- DIAGNOSTIC: K=128, N=64, M=32 ---\n");
        int K = 128, N = 64, M = 32;
        const int group_size = 128;
        size_t a_bytes = (size_t)M * K * sizeof(__half);
        size_t qw_bytes = (size_t)(K/8) * N * sizeof(uint32_t);
        size_t sc_bytes = (size_t)(K/group_size) * N * sizeof(__half);
        size_t c_bytes = (size_t)M * N * sizeof(__half);

        __half* h_A = (__half*)malloc(a_bytes);
        uint32_t* h_qw = (uint32_t*)malloc(qw_bytes);
        __half* h_sc = (__half*)malloc(sc_bytes);
        __half* h_C = (__half*)malloc(c_bytes);
        __half* h_ref_C = (__half*)malloc(c_bytes);

        srand(42);
        for (int i = 0; i < M * K; i++)
            h_A[i] = __float2half(((rand() % 1000) - 500) / 500.0f);
        for (int i = 0; i < (K/8) * N; i++)
            h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
        for (int i = 0; i < (K/group_size) * N; i++)
            h_sc[i] = __float2half(((rand() % 200) - 100) / 1000.0f);

        __half *d_A, *d_C, *d_ref_C; uint32_t* d_qw; __half* d_sc;
        cudaMalloc(&d_A, a_bytes);
        cudaMalloc(&d_qw, qw_bytes);
        cudaMalloc(&d_sc, sc_bytes);
        cudaMalloc(&d_C, c_bytes);
        cudaMalloc(&d_ref_C, c_bytes);
        cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_qw, h_qw, qw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc, h_sc, sc_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, c_bytes);

        // GPU naive reference
        dim3 ref_grid((N + 15) / 16, (M + 15) / 16);
        dim3 ref_block(16, 16);
        gptq_gemm_naive_kernel<<<ref_grid, ref_block>>>(d_A, d_qw, d_ref_C, d_sc, M, K, N, group_size);

        // Our kernel
        gptq_gemm(d_A, d_qw, d_C, d_sc, M, K, N, group_size);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, c_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref_C, d_ref_C, c_bytes, cudaMemcpyDeviceToHost);

        // Compare
        printf("  [0,0] REF=%.4f GPU=%.4f\n", __half2float(h_ref_C[0]), __half2float(h_C[0]));
        printf("  [0,1] REF=%.4f GPU=%.4f\n", __half2float(h_ref_C[1]), __half2float(h_C[1]));
        printf("  [1,0] REF=%.4f GPU=%.4f\n", __half2float(h_ref_C[N]), __half2float(h_C[N]));

        double se = 0, sr = 0;
        for (int i = 0; i < M * N; i++) {
            float g = __half2float(h_C[i]), r = __half2float(h_ref_C[i]);
            se += (double)(g-r)*(g-r); sr += (double)r*r;
        }
        float rmse = (sr > 0) ? (float)sqrt(se/sr) : 0;
        printf("  RMSE = %.6f (%s)\n\n", rmse, rmse < 0.05f ? "PASS" : "FAIL");

        cudaFree(d_A); cudaFree(d_qw); cudaFree(d_sc); cudaFree(d_C); cudaFree(d_ref_C);
        free(h_A); free(h_qw); free(h_sc); free(h_C); free(h_ref_C);
    }

    struct Case { const char* name; int K, N; };
    Case cases[] = {
        {"gate_proj (5120->17408)", 5120, 17408},
        {"down_proj (17408->5120)", 17408, 5120},
    };
    int M_vals[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
    const int MAX_M = 1024;
    const int WARMUP = 5, ITERS = 20;

    // Orin specs for utilization calculation
    constexpr float PEAK_TFLOPS = 69.0f;   // FP16 TC peak
    constexpr float PEAK_BW_GBS = 192.0f;  // DRAM bandwidth

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (auto& c : cases) {
        printf("--- %s ---\n", c.name);
        printf("%-6s %10s %8s %6s %8s %6s %10s %8s\n",
            "M", "Time(us)", "TFLOPS", "TC%", "GB/s", "BW%", "ArithInt", "Correct");
        printf("--------------------------------------------------------------------------\n");

        int K = c.K, N = c.N;
        const int group_size = 128;

        // Allocate for max M
        size_t a_bytes = (size_t)MAX_M * K * sizeof(__half);
        size_t qw_bytes = (size_t)(K/8) * N * sizeof(uint32_t);
        size_t sc_bytes = (size_t)(K/group_size) * N * sizeof(__half);
        size_t c_bytes = (size_t)MAX_M * N * sizeof(__half);

        __half* h_A = (__half*)malloc(a_bytes);
        uint32_t* h_qw = (uint32_t*)malloc(qw_bytes);
        __half* h_sc = (__half*)malloc(sc_bytes);
        __half* h_C = (__half*)malloc(c_bytes);

        srand(42);
        for (int i = 0; i < MAX_M * K; i++)
            h_A[i] = __float2half(((rand() % 1000) - 500) / 500.0f);
        for (int i = 0; i < (K/8) * N; i++)
            h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
        for (int i = 0; i < (K/group_size) * N; i++)
            h_sc[i] = __float2half(((rand() % 200) - 100) / 1000.0f);

        __half *d_A, *d_C; uint32_t* d_qw; __half* d_sc;
        cudaMalloc(&d_A, a_bytes);
        cudaMalloc(&d_qw, qw_bytes);
        cudaMalloc(&d_sc, sc_bytes);
        cudaMalloc(&d_C, c_bytes);
        cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_qw, h_qw, qw_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc, h_sc, sc_bytes, cudaMemcpyHostToDevice);

        for (int M : M_vals) {
            // GPU naive kernel reference for correctness (much faster than CPU)
            bool check = (M <= 64);
            __half* d_ref_C = nullptr;
            if (check) {
                cudaMalloc(&d_ref_C, (size_t)M * N * sizeof(__half));
                dim3 ref_grid((N + 15) / 16, (M + 15) / 16);
                dim3 ref_block(16, 16);
                gptq_gemm_naive_kernel<<<ref_grid, ref_block>>>(d_A, d_qw, d_ref_C, d_sc, M, K, N, group_size);
                cudaDeviceSynchronize();
            }

            // Run kernel
            cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half));
            if (M == 1) gptq_gemv(d_A, d_qw, d_C, d_sc, K, N, group_size);
            else        gptq_gemm(d_A, d_qw, d_C, d_sc, M, K, N, group_size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("  CUDA err: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();

            // Check correctness via GPU naive kernel
            bool correct = true;
            if (check) {
                __half* h_ref_C = (__half*)malloc((size_t)M * N * sizeof(__half));
                cudaMemcpy(h_C, d_C, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_ref_C, d_ref_C, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost);
                double se = 0, sr = 0;
                for (int i = 0; i < M * N; i++) {
                    float g = __half2float(h_C[i]), r = __half2float(h_ref_C[i]);
                    se += (double)(g-r)*(g-r); sr += (double)r*r;
                }
                float rmse = (sr > 0) ? (float)sqrt(se/sr) : 0;
                correct = (rmse < 0.05f);
                if (!correct) printf("  RMSE = %.4f\n", rmse);
                free(h_ref_C);
                cudaFree(d_ref_C);
            }

            // Benchmark
            for (int i = 0; i < WARMUP; i++) {
                if (M == 1) gptq_gemv(d_A, d_qw, d_C, d_sc, K, N, group_size);
                else        gptq_gemm(d_A, d_qw, d_C, d_sc, M, K, N, group_size);
            }
            cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int i = 0; i < ITERS; i++) {
                if (M == 1) gptq_gemv(d_A, d_qw, d_C, d_sc, K, N, group_size);
                else        gptq_gemm(d_A, d_qw, d_C, d_sc, M, K, N, group_size);
            }
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            float us = ms * 1000.0f / ITERS;
            float tflops = (float)(2.0 * M * K * N / (us * 1e6));
            float tc_pct = tflops / PEAK_TFLOPS * 100.0f;

            // Bytes: A[M,K]*2 + qweight[K/8,N]*4 + scales[K/gs,N]*2 + C[M,N]*2
            double bytes_a = (double)M * K * 2;
            double bytes_w = (double)(K / 8) * N * 4;          // INT4 packed
            double bytes_s = (double)(K / group_size) * N * 2;  // scales
            double bytes_c = (double)M * N * 2;                 // output
            double total_bytes = bytes_a + bytes_w + bytes_s + bytes_c;
            float gbs = (float)(total_bytes / (us * 1e3));  // GB/s
            float bw_pct = gbs / PEAK_BW_GBS * 100.0f;

            // Arithmetic intensity = FLOPs / Bytes
            double flops = 2.0 * M * K * N;
            float arith_int = (float)(flops / total_bytes);

            printf("%-6d %10.1f %8.3f %5.1f%% %7.1f %5.1f%% %10.1f %8s\n",
                M, us, tflops, tc_pct, gbs, bw_pct, arith_int,
                correct ? "OK" : "FAIL");
        }

        cudaFree(d_A); cudaFree(d_qw); cudaFree(d_sc); cudaFree(d_C);
        free(h_A); free(h_qw); free(h_sc); free(h_C);
        printf("\n");
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

} // namespace deusridet
