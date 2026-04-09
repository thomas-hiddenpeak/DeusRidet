// fp16_gemm.cu — FP16 GEMM kernel for SM87 (Orin)
//
// Computes C[M,N] = A[M,K] @ B_repacked^T where B is pre-repacked
// from [N,K] row-major to tile-level [K,N] via fp16_repack_b().
//
// Architecture inspired by IST-DASLab/marlin persistent kernel design,
// adapted for native FP16 weights (no dequantization).
// Original Marlin: https://github.com/IST-DASLab/marlin (Apache 2.0)
//
// Key differences from Marlin:
//   - No INT4 dequant → B loaded as FP16 directly
//   - B pre-repacked to [K,N] tile layout for ldmatrix.x2.trans
//   - XOR swizzle for both A and B (no padding, zero bank conflicts)
//   - Simpler dispatch (no quantization group handling)

#include "fp16_gemm.h"
#include <cstdio>
#include <cstdint>

namespace deusridet {

// ============================================================================
// Section 1: Constants
// ============================================================================

static constexpr int FP16_THREADS = 256;  // 8 warps per TB
static constexpr int FP16_STAGES  = 4;    // async pipeline depth (64KB SMEM → 2 TBs/SM = 16 warps)

// Tile dimensions (in elements, not blocks)
static constexpr int TILE_N = 64;   // N cols per threadblock
static constexpr int TILE_K = 64;   // K elements per pipeline stage

// B weight is pre-repacked from [N,K] to tile-level [K,N] via fp16_repack_b().
// In SMEM, B tiles are [TILE_K rows, TILE_N cols] with XOR swizzle on N-int4:
//   Row r at col_int4 c → physical int4 at (c ^ (r%8)). For 8 K-rows at same
//   N-column: offsets c^0..c^7 — all distinct banks.
// This layout allows ldmatrix.x2.trans to produce correct m16n8k16 B fragments.

static constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// Section 2: PTX inline assembly helpers
// ============================================================================

// Predicated async global→shared copy (16 bytes = 8 FP16 values).
// Uses .ca cache policy: caches in L1+L2 for data reuse across TBs on same SM.
__device__ __forceinline__ void cp_async4_pred(
    void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

// Async copy with L2 evict-first hint (for weights: read once, don't pollute L2).
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

// m16n8k16 tensor core MMA: C += A * B (FP16→FP32 accumulate).
__device__ __forceinline__ void mma_m16n8k16(
    const uint32_t* a, const uint32_t* b, float* c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );
}

// Load 16×16 A-operand fragment from shared memory via ldmatrix.
__device__ __forceinline__ void ldsm4(uint32_t* frag, const void* smem_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem)
    );
}

// Load 8×16 B-operand fragment via ldmatrix with hardware transpose.
// Source: B in [K, N] layout in SMEM (K-major, N-cols).
// Each thread provides the address of its K-row (lane_id % 16).
// Lanes 0-7 → first 8×8 (K=0..7), lanes 8-15 → second 8×8 (K=8..15).
// .trans shuffles data across threads to match m16n8k16 B fragment layout.
__device__ __forceinline__ void ldsm2_trans(uint32_t* frag, const void* smem_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(smem)
    );
}

// ============================================================================
// Section 3: Kernel template
// ============================================================================
//
// Template parameter: thread_m_blocks = number of 16-row blocks in M.
//   1 → M≤16, 2 → M≤32, 3 → M≤48, 4 → M≤64
//
// Warp layout (8 warps, 256 threads):
//   warp_m = warp_id / WARPS_N (0..WARPS_M-1)
//   warp_n = warp_id % WARPS_N (0..WARPS_N-1)
//
// Each warp processes one m16×n_per_warp sub-tile of the output.
// All warps cooperatively load A and B tiles via cp.async.

template <int thread_m_blocks>
__global__ void fp16_gemm_kernel(
    const __half* __restrict__ A,     // [M, K] row-major
    const __half* __restrict__ B,     // [N, K] row-major
    __half* __restrict__ C,           // [M, N] row-major
    int prob_m, int prob_n, int prob_k)
{
    // Warp/thread decomposition
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Warp layout: WARPS_M × WARPS_N covering BLOCK_M × TILE_N output tile
    // TILE_N = 64 = 8 × n8-blocks. With 8 warps:
    //   thread_m_blocks=4: 4 warps_m × 2 warps_n, each warp = 16m × 32n
    //   thread_m_blocks=2: 2 warps_m × 4 warps_n, each warp = 16m × 16n
    //   thread_m_blocks=1: 1 warp_m  × 8 warps_n, each warp = 16m × 8n
    //   thread_m_blocks=3: 3 warps_m × ... doesn't divide evenly
    //     → use 4 warps_m × 2 warps_n, warp 6-7 idle for m-overflow
    static constexpr int WARPS_M = (thread_m_blocks >= 3) ? 4 :
                                   (thread_m_blocks >= 2) ? 2 : 1;
    static constexpr int WARPS_N = 8 / WARPS_M;
    static constexpr int N_PER_WARP = TILE_N / WARPS_N;  // n-cols per warp
    static constexpr int N_MMA_PER_WARP = N_PER_WARP / 8; // mma ops along N per warp

    const int warp_m = warp_id / WARPS_N;  // which M-block this warp handles
    const int warp_n = warp_id % WARPS_N;  // which N-block this warp handles

    // Skip excess warps when thread_m_blocks doesn't divide evenly
    const bool warp_active = (warp_m < thread_m_blocks);

    // ======================================================================
    // Persistent kernel: striped N-partitioning
    // ======================================================================

    const int n_tiles = prob_n / TILE_N;
    const int k_tiles = prob_k / TILE_K;  // number of K-iterations per N-tile

    // Each TB loops over N-tiles in striped fashion
    int tile_n = blockIdx.x;  // starting N-tile for this TB

    // ======================================================================
    // Shared memory layout
    // ======================================================================
    //
    // A tiles: [BLOCK_M, TILE_K] with XOR swizzle
    //   Per stage: BLOCK_M * TILE_K * 2 bytes = m_blocks * 2048 bytes
    //
    // B tiles: [TILE_N, TILE_K] with XOR swizzle (no padding)
    //   Per stage: TILE_N * TILE_K * 2 = 64 * 64 * 2 = 8192 bytes
    //
    // Total per stage: m_blocks*2048 + 8192
    // For m_blocks=4: 8192 + 8192 = 16384, × 4 stages = 65536 bytes = 64 KB → 2 TBs/SM

    static constexpr int BLOCK_M = 16 * thread_m_blocks;
    static constexpr int A_SMEM_STRIDE = TILE_K;  // in halves (no padding for A, use XOR)
    static constexpr int A_STAGE_SIZE = BLOCK_M * A_SMEM_STRIDE;  // in halves
    static constexpr int B_STAGE_SIZE = TILE_K * TILE_N;  // in halves, [K, N] repacked layout

    extern __shared__ __half sh_mem[];
    __half* sh_a = sh_mem;  // [STAGES][BLOCK_M][TILE_K] A tiles
    __half* sh_b = sh_a + FP16_STAGES * A_STAGE_SIZE;  // [STAGES][TILE_K][TILE_N] B tiles (K-major)

    // ======================================================================
    // A-tile cp.async indices
    // ======================================================================
    // A[M, K] row-major. We load A[0:BLOCK_M, k_tile*64 : k_tile*64+64].
    // Each cp.async4 copies 16 bytes = 8 halves.
    // Elems per row = TILE_K = 64 halves → 8 cp.async4 per row.
    // Total per stage = BLOCK_M * 8 = BLOCK_M * 8 loads.
    // With 256 threads: each thread does ceil(BLOCK_M*8 / 256) loads.

    static constexpr int A_CP_PER_ROW = TILE_K / 8;           // 8
    static constexpr int A_CP_TOTAL   = BLOCK_M * A_CP_PER_ROW; // m_blocks*128
    static constexpr int A_CP_PER_THREAD = ceildiv(A_CP_TOTAL, FP16_THREADS);

    // ======================================================================
    // B-tile cp.async indices
    // ======================================================================
    // B is pre-repacked to [N/64, K/64, 64_K, 64_N] — tile-level K-major.
    // Each tile [TILE_K, TILE_N]: rows are K-dim (64), cols are N-dim (64).
    // Data per K-row = TILE_N = 64 halves = 128 bytes → 8 cp.async4 per row.
    // XOR swizzle applied to SMEM destination on N-int4 columns.
    // Total per stage = 64 * 8 = 512 loads. 512/256 = 2 per thread.

    static constexpr int B_CP_PER_ROW = TILE_N / 8;             // 8 (N-int4 per K-row)
    static constexpr int B_CP_TOTAL   = TILE_K * B_CP_PER_ROW;  // 512
    static constexpr int B_CP_PER_THREAD = ceildiv(B_CP_TOTAL, FP16_THREADS);

    // ======================================================================
    // XOR swizzle for A SMEM (bank-conflict-free ldmatrix access)
    // ======================================================================
    // A is stored as [BLOCK_M, TILE_K] in SMEM. ldmatrix reads 16×16 sub-tiles.
    // Without swizzle, rows 0 and 8 map to same banks (stride=128 bytes = 32 banks).
    // XOR: smem_col_int4 ^= (row % 8). Undone during ldmatrix read.
    auto a_swizzle = [](int row, int col_int4) -> int {
        return row * (A_SMEM_STRIDE / 8) + (col_int4 ^ (row % 8));
    };

    // XOR swizzle for B SMEM [TILE_K, TILE_N]
    // Rows are K-dim (0..63), cols are N-dim int4s (0..7).
    // ldmatrix.x2.trans reads 16 consecutive K-rows at same N-int4 column.
    // XOR ensures 8 K-rows at same logical N-col hit different banks.
    auto b_swizzle = [](int k_row, int n_int4) -> int {
        return k_row * (TILE_N / 8) + (n_int4 ^ (k_row % 8));
    };

    // ======================================================================
    // Register accumulators
    // ======================================================================
    // Each warp accumulates a [16, N_PER_WARP] output sub-tile in FP32.
    // N_PER_WARP / 8 mma ops, each producing 4 floats per thread.
    float acc[N_MMA_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_MMA_PER_WARP; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    // ======================================================================
    // Main persistent loop over N-tiles
    // ======================================================================

    for (; tile_n < n_tiles; tile_n += gridDim.x) {
        const int n_base = tile_n * TILE_N;

        // Zero accumulators
        #pragma unroll
        for (int i = 0; i < N_MMA_PER_WARP; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = 0.0f;

        // ==================================================================
        // Async pipeline: prefetch first (STAGES-1) tiles
        // ==================================================================

        auto fetch_tile = [&](int stage, int k_tile) {
            if (k_tile >= k_tiles) {
                cp_async_fence();
                return;
            }
            __half* a_stage = sh_a + stage * A_STAGE_SIZE;
            __half* b_stage = sh_b + stage * B_STAGE_SIZE;

            int a_k_offset = k_tile * TILE_K;

            // Load A tile: A[0:BLOCK_M, k_offset : k_offset+64]
            #pragma unroll
            for (int i = 0; i < A_CP_PER_THREAD; i++) {
                int cp_idx = threadIdx.x + i * FP16_THREADS;
                if (cp_idx < A_CP_TOTAL) {
                    int row = cp_idx / A_CP_PER_ROW;
                    int col_int4 = cp_idx % A_CP_PER_ROW;
                    int smem_idx = a_swizzle(row, col_int4);
                    bool valid = (row < prob_m);
                    const void* glob = &A[row * prob_k + a_k_offset + col_int4 * 8];
                    cp_async4_pred(
                        reinterpret_cast<int4*>(a_stage) + smem_idx,
                        glob, valid);
                }
            }

            // Load B tile: repacked B[n_tile, k_tile] → SMEM[TILE_K, TILE_N]
            // Repacked layout: [N/64, K/64, 64_K, 64_N] contiguous tiles.
            // Each K-row has TILE_N halves = 8 int4s.
            #pragma unroll
            for (int i = 0; i < B_CP_PER_THREAD; i++) {
                int cp_idx = threadIdx.x + i * FP16_THREADS;
                if (cp_idx < B_CP_TOTAL) {
                    int k_row = cp_idx / B_CP_PER_ROW;    // K-row within tile (0..63)
                    int n_int4 = cp_idx % B_CP_PER_ROW;   // N-int4 within tile (0..7)
                    int smem_idx = b_swizzle(k_row, n_int4);
                    int b_tile_base = (tile_n * k_tiles + k_tile) * (TILE_K * TILE_N);
                    const void* glob = &B[b_tile_base + k_row * TILE_N + n_int4 * 8];
                    cp_async4_stream(
                        reinterpret_cast<int4*>(b_stage) + smem_idx, glob);
                }
            }

            cp_async_fence();
        };

        // Prefetch first stages
        #pragma unroll
        for (int s = 0; s < FP16_STAGES - 1; s++)
            fetch_tile(s, s);

        // ==================================================================
        // K-loop: process all K-tiles with sliding window pipeline
        // ==================================================================

        for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
            // Wait for current stage data
            cp_async_wait<FP16_STAGES - 2>();
            __syncthreads();

            int stage = k_tile % FP16_STAGES;
            __half* a_stage = sh_a + stage * A_STAGE_SIZE;
            __half* b_stage = sh_b + stage * B_STAGE_SIZE;

            // Prefetch next stage
            int next_k = k_tile + FP16_STAGES - 1;
            int next_stage = next_k % FP16_STAGES;
            fetch_tile(next_stage, next_k);

            if (!warp_active) {
                __syncthreads();
                continue;
            }

            // ==============================================================
            // Compute: iterate over 4 K-sub-blocks of 16 within TILE_K=64
            // ==============================================================

            #pragma unroll
            for (int k_sub = 0; k_sub < TILE_K / 16; k_sub++) {
                // Load A fragment [16, 16] for this warp's M-block
                uint32_t frag_a[4];
                {
                    int a_row = warp_m * 16 + (lane_id % 16);
                    int a_col_int4 = k_sub * 2 + (lane_id / 16);
                    int smem_idx = a_swizzle(a_row, a_col_int4);
                    ldsm4(frag_a, reinterpret_cast<int4*>(a_stage) + smem_idx);
                }

                // For each N-sub-block of 8 within this warp's N-range
                #pragma unroll
                for (int n_sub = 0; n_sub < N_MMA_PER_WARP; n_sub++) {
                    // B fragment via ldmatrix.x2.trans from [K, N] SMEM layout.
                    // Each thread addresses K-row = k_sub*16 + (lane_id % 16),
                    // N-column = warp's N-offset + n_sub (in int4 units).
                    // ldmatrix.x2.trans reads 2 × 8×8 sub-matrices (K=0..7, K=8..15)
                    // and transposes to match m16n8k16 B fragment register layout.
                    uint32_t frag_b[2];
                    {
                        int k_row = k_sub * 16 + (lane_id % 16);
                        int n_int4 = warp_n * (N_PER_WARP / 8) + n_sub;
                        int smem_idx = b_swizzle(k_row, n_int4);
                        ldsm2_trans(frag_b,
                            reinterpret_cast<int4*>(b_stage) + smem_idx);
                    }

                    // MMA: acc[n_sub] += frag_a × frag_b
                    mma_m16n8k16(frag_a, frag_b, acc[n_sub]);
                }
            }

            __syncthreads();
        }  // k_tile loop

        // ==================================================================
        // Write result: convert FP32 accumulators → FP16 and store to C
        // ==================================================================

        if (!warp_active) continue;

        // mma.m16n8k16 C fragment layout per thread:
        //   c[0], c[1]: rows (lane_id/4)*2 + {0,1}, col (lane_id%4)*2
        //   c[2], c[3]: rows (lane_id/4)*2 + {8,9}, col (lane_id%4)*2
        //   Wait — that's not right. Let me use the standard m16n8k16 mapping:
        //
        //   For mma.m16n8k16 output C[16,8], each thread holds 4 floats:
        //     Thread (lane_id) stores:
        //       c[0] → C[ lane_id/4       , (lane_id%4)*2     ]  (row 0-7, even cols)
        //       c[1] → C[ lane_id/4       , (lane_id%4)*2 + 1 ]  (row 0-7, odd cols)
        //       c[2] → C[ lane_id/4 + 8   , (lane_id%4)*2     ]  (row 8-15, even cols)
        //       c[3] → C[ lane_id/4 + 8   , (lane_id%4)*2 + 1 ]  (row 8-15, odd cols)
        //
        //   With warp_m and warp_n offsets, the global C position is:
        //     row = warp_m*16 + {lane_id/4, lane_id/4+8}
        //     col = n_base + warp_n*N_PER_WARP + n_sub*8 + {(lane_id%4)*2, +1}

        for (int n_sub = 0; n_sub < N_MMA_PER_WARP; n_sub++) {
            int col_base = n_base + warp_n * N_PER_WARP + n_sub * 8 + (lane_id % 4) * 2;

            // Top half: rows lane_id/4 + 0
            {
                int row = warp_m * 16 + lane_id / 4;
                if (row < prob_m) {
                    __half* out = &C[row * prob_n + col_base];
                    out[0] = __float2half(acc[n_sub][0]);
                    out[1] = __float2half(acc[n_sub][1]);
                }
            }
            // Bottom half: rows lane_id/4 + 8
            {
                int row = warp_m * 16 + lane_id / 4 + 8;
                if (row < prob_m) {
                    __half* out = &C[row * prob_n + col_base];
                    out[0] = __float2half(acc[n_sub][2]);
                    out[1] = __float2half(acc[n_sub][3]);
                }
            }
        }

    }  // tile_n loop
}

// ============================================================================
// Section 4: Dispatch
// ============================================================================

#define FP16_SMEM_BYTES(M_BLOCKS) \
    (FP16_STAGES * ((16 * (M_BLOCKS) * TILE_K + TILE_K * TILE_N) * 2))

#define FP16_LAUNCH(M_BLOCKS) \
    do { \
        constexpr int smem = FP16_SMEM_BYTES(M_BLOCKS); \
        cudaFuncSetAttribute( \
            fp16_gemm_kernel<M_BLOCKS>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        fp16_gemm_kernel<M_BLOCKS><<<blocks, FP16_THREADS, smem, stream>>>( \
            A, B, C, M, N, K); \
    } while (0)

void fp16_gemm(
    const __half* A, const __half* B, __half* C,
    int M, int K, int N, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return;
    if (N % TILE_N != 0 || K % TILE_K != 0) {
        fprintf(stderr, "fp16_gemm: N=%d must be divisible by %d, K=%d by %d\n",
                N, TILE_N, K, TILE_K);
        return;
    }

    int n_tiles = N / TILE_N;
    // Use non-persistent (1 TB per tile) when tiles fill SM evenly.
    // Use capped grid (persistent) when n_tiles would leave SMs idle in last wave.
    // SM87: 16 SMs, 2 TBs/SM (64KB SMEM with 4 stages) → 32 concurrent TBs.
    constexpr int MAX_CONCURRENT = 32;  // 16 SMs × 2 TBs
    int blocks;
    if (n_tiles % MAX_CONCURRENT == 0 || n_tiles <= MAX_CONCURRENT) {
        blocks = n_tiles;  // perfectly balanced or single wave
    } else {
        blocks = MAX_CONCURRENT;  // persistent: distribute tiles across TBs
    }
    int m_blocks = ceildiv(M, 16);

    // NOTE: template<3> has a broken WARPS_M=2 (should be 4) causing OOB
    // reads on rows [32..BLOCK_M-1]. Skip it and use template<4> which
    // correctly sets WARPS_M=4 with warp_active guard for excess warps.
    if (m_blocks <= 1) FP16_LAUNCH(1);
    else if (m_blocks <= 2) FP16_LAUNCH(2);
    else                    FP16_LAUNCH(4);
}

#undef FP16_LAUNCH
#undef FP16_SMEM_BYTES

// ============================================================================
// Section 5: Weight repacking for ldmatrix.x2.trans
// ============================================================================
//
// Repack B from [N, K] row-major to tile-level [K, N] layout:
//   [N/TILE_N, K/TILE_K, TILE_K, TILE_N]
//
// Within each tile, data is stored K-major: B_repacked[k][n] = B_orig[n][k].
// This allows ldmatrix.x2.trans to produce correct m16n8k16 B fragments
// directly from SMEM, eliminating 32 manual half2 loads per MMA batch.
//
// One-time cost at model load. N must be divisible by TILE_N (64),
// K must be divisible by TILE_K (64).

// GPU kernel: repack B from [N,K] row-major to tile-level [K,N] layout.
// Each thread handles one element: B_dst[tile_k, tile_n] = B_src[n_global, k_global]
__global__ void fp16_repack_b_kernel(
    const __half* __restrict__ B_src,
    __half* __restrict__ B_dst,
    int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    // Source: row-major [N, K] → element at (n, k)
    int n = idx / K;
    int k = idx % K;

    // Destination tile coords
    int nt = n / TILE_N;
    int n_local = n % TILE_N;
    int kt = k / TILE_K;
    int k_local = k % TILE_K;
    int k_tiles = K / TILE_K;

    // dst layout: [nt, kt, k_local, n_local]
    int dst_idx = ((nt * k_tiles + kt) * TILE_K + k_local) * TILE_N + n_local;
    B_dst[dst_idx] = B_src[idx];
}

void fp16_repack_b(
    const __half* __restrict__ B_src,
    __half* __restrict__ B_dst,
    int N, int K)
{
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp16_repack_b_kernel<<<blocks, threads>>>(B_src, B_dst, N, K);
    cudaDeviceSynchronize();
}

} // namespace deusridet
