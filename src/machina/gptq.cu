// gptq.cu — GPTQ-Int4 dequant + GEMV/GEMM CUDA kernels for SM87
//
// Kernel design for Jetson AGX Orin (SM87, Ampere):
//   - 16 SMs, 128 FP32 cores each, L2 = 4 MB
//   - Memory BW ~192 GB/s (unified DRAM)
//   - Shared memory: up to 164 KB/SM (48 KB default)
//   - No FP4/FP8 tensor core, FP16 tensor core available
//
// GPTQ layout: qweight[K/8, N] (uint32), scales[K/128, N] (FP16)
//   bits=4, group_size=128, sym=true, zero_point=8
//
// GEMV strategy (M=1, memory-bound):
//   Each warp handles a slice of output columns. Each thread processes
//   multiple K elements, accumulating in FP32 for numerical stability.
//   Final warp reduction → FP16 output.
//
// GEMM strategy (M>1):
//   Tile-based: each thread block computes a tile of Y[BM, BN].
//   Load qweight tile into shared memory, dequant on the fly during
//   the K-dimension accumulation. Uses FP16 accumulation with FP32
//   master accumulators for critical paths.

#include "gptq.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace deusridet {

// ============================================================================
// Shared device utilities
// ============================================================================

__device__ __forceinline__ int extract_int4(uint32_t packed, int index) {
    // Extract 4-bit value at position index (0–7), LSB-first
    return (packed >> (index * 4)) & 0xF;
}

__device__ __forceinline__ __half dequant_int4(__half scale, int q_val) {
    // W = scale * (q - 8), computed in FP32 for precision then converted to FP16
    float w = __half2float(scale) * (float)(q_val - GPTQ_ZERO_POINT);
    return __float2half(w);
}

// ============================================================================
// GEMV kernel — Decode path (M=1)
// ============================================================================
//
// Grid:  (N / TILE_N) blocks
// Block: (TILE_N, K_THREADS) where K_THREADS process the K dimension
//
// Each thread block computes TILE_N output elements.
// Within a block, K_THREADS threads split the K loop.
// After the K loop, column-wise reduction across K_THREADS using shared mem.
//
// Memory access: qweight[K/8, N] is read with coalesced access along N.
// Each thread reads one uint32 (8 INT4 values sharing the same N column).
// This gives 8 K-elements per load, amortizing memory access.

constexpr int GEMV_TILE_N = 64;       // output columns per block
constexpr int GEMV_K_THREADS = 8;     // threads splitting K dimension
constexpr int GEMV_BLOCK_DIM = GEMV_TILE_N * GEMV_K_THREADS;  // 512 threads

__global__ void gptq_gemv_kernel(
    const __half*   __restrict__ x,       // [K]
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    __half*         __restrict__ y,       // [N]
    int K, int N)
{
    const int n_base = blockIdx.x * GEMV_TILE_N;
    const int local_n = threadIdx.x % GEMV_TILE_N;  // which output column
    const int k_tid   = threadIdx.x / GEMV_TILE_N;  // which K-slice

    const int n = n_base + local_n;
    if (n >= N) return;

    // Number of packed K rows total
    const int packed_K = K / GPTQ_PACK_FACTOR;

    // Each K-thread processes packed_K / GEMV_K_THREADS rows
    // Each packed row = 8 INT4 values = 8 K elements
    const int rows_per_thread = (packed_K + GEMV_K_THREADS - 1) / GEMV_K_THREADS;
    const int pk_start = k_tid * rows_per_thread;
    const int pk_end   = min(pk_start + rows_per_thread, packed_K);

    float acc = 0.0f;

    for (int pk = pk_start; pk < pk_end; pk++) {
        uint32_t packed = qweight[pk * N + n];
        int k_base = pk * GPTQ_PACK_FACTOR;

        // Process 8 INT4 values from one packed uint32
        #pragma unroll
        for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
            int k = k_base + i;
            int q_val = extract_int4(packed, i);
            int group = k / GPTQ_GROUP_SIZE;
            __half s = scales[group * N + n];
            __half w = dequant_int4(s, q_val);
            acc += __half2float(w) * __half2float(x[k]);
        }
    }

    // K-thread reduction using shared memory
    __shared__ float smem[GEMV_TILE_N * GEMV_K_THREADS];
    smem[k_tid * GEMV_TILE_N + local_n] = acc;
    __syncthreads();

    // Thread 0 of each column reduces
    if (k_tid == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int t = 0; t < GEMV_K_THREADS; t++) {
            sum += smem[t * GEMV_TILE_N + local_n];
        }
        y[n] = __float2half(sum);
    }
}

void gptq_gemv(const __half* x,
               const GptqWeight& weight,
               __half* y,
               cudaStream_t stream)
{
    int N = weight.N;
    int grid = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;

    gptq_gemv_kernel<<<grid, GEMV_BLOCK_DIM, 0, stream>>>(
        x, weight.qweight, weight.scales, y,
        weight.K, weight.N);
}

// ============================================================================
// GEMM kernel — Prefill path (M>1)
// ============================================================================
//
// Tile-based approach: each block computes a [BM x BN] tile of Y.
// K is iterated in steps of BK (aligned to group_size for scale reuse).
//
// Per iteration:
//   1. Load X tile [BM, BK] from global to shared memory (FP16)
//   2. Load qweight tile [BK/8, BN] from global, dequant to shared memory (FP16)
//   3. Compute partial Y += X_tile @ W_tile using thread-level FMA
//
// Memory optimization:
//   - qweight loads are coalesced along N
//   - X loads are coalesced along K
//   - Dequant happens in shared memory to amortize global bandwidth
//   - BK = 128 aligns with group_size → one scale per BK tile per column

constexpr int GEMM_BM = 32;   // tile rows (tokens)
constexpr int GEMM_BN = 64;   // tile columns (output features)
constexpr int GEMM_BK = 128;  // tile K (= group_size, one scale per tile)
constexpr int GEMM_BLOCK_DIM = 256;

// Thread mapping within block: 256 = 8 (M-dim) x 32 (N-dim)
constexpr int GEMM_TM = 4;    // elements per thread in M dimension
constexpr int GEMM_TN = 2;    // elements per thread in N dimension
constexpr int GEMM_WARP_N = 32;

__global__ void gptq_gemm_kernel(
    const __half*   __restrict__ X,       // [M, K] row-major
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    __half*         __restrict__ Y,       // [M, N] row-major
    int M, int K, int N)
{
    const int bm = blockIdx.y * GEMM_BM;
    const int bn = blockIdx.x * GEMM_BN;

    // Thread indices within block
    const int tid = threadIdx.x;
    const int warp_m = tid / GEMM_WARP_N;   // 0..7 (M-dim thread index)
    const int warp_n = tid % GEMM_WARP_N;   // 0..31 (N-dim thread index)

    // Each thread accumulates TM x TN output values in FP32
    float acc[GEMM_TM][GEMM_TN];
    #pragma unroll
    for (int i = 0; i < GEMM_TM; i++) {
        #pragma unroll
        for (int j = 0; j < GEMM_TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Shared memory for dequantized weights and X tile
    // W_tile: [BK, BN] FP16 = 128 * 64 * 2 = 16 KB
    // X_tile: [BM, BK] FP16 = 32 * 128 * 2 = 8 KB
    // Total: 24 KB — fits in default 48KB shared memory
    __shared__ __half W_tile[GEMM_BK][GEMM_BN];
    __shared__ __half X_tile[GEMM_BM][GEMM_BK];

    const int num_k_tiles = (K + GEMM_BK - 1) / GEMM_BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_base = kt * GEMM_BK;

        // --- Load X tile [BM, BK] into shared memory ---
        // 256 threads load BM*BK = 32*128 = 4096 elements → 16 per thread
        {
            int total_elems = GEMM_BM * GEMM_BK;
            for (int idx = tid; idx < total_elems; idx += GEMM_BLOCK_DIM) {
                int row = idx / GEMM_BK;
                int col = idx % GEMM_BK;
                int global_m = bm + row;
                int global_k = k_base + col;
                if (global_m < M && global_k < K) {
                    X_tile[row][col] = X[global_m * K + global_k];
                } else {
                    X_tile[row][col] = __float2half(0.0f);
                }
            }
        }

        // --- Load and dequant qweight tile [BK, BN] into shared memory ---
        // qweight is [K/8, N], we need BK/8 = 16 packed rows, BN = 64 columns
        // 16 * 64 = 1024 loads, each yielding 8 values → 8192 writes
        // 256 threads, 4 packed rows per thread
        {
            int packed_rows = GEMM_BK / GPTQ_PACK_FACTOR;  // 128/8 = 16
            int total_loads = packed_rows * GEMM_BN;  // 16 * 64 = 1024
            int group = (k_base / GPTQ_GROUP_SIZE);  // BK = group_size → one group per tile

            for (int idx = tid; idx < total_loads; idx += GEMM_BLOCK_DIM) {
                int pk_local = idx / GEMM_BN;   // packed row within tile (0..15)
                int n_local  = idx % GEMM_BN;   // column within tile

                int pk_global = (k_base / GPTQ_PACK_FACTOR) + pk_local;
                int n_global  = bn + n_local;

                uint32_t packed = 0;
                __half s = __float2half(0.0f);
                if (n_global < N && pk_global < (K / GPTQ_PACK_FACTOR)) {
                    packed = qweight[pk_global * N + n_global];
                    s = scales[group * N + n_global];
                }

                #pragma unroll
                for (int i = 0; i < GPTQ_PACK_FACTOR; i++) {
                    int k_local = pk_local * GPTQ_PACK_FACTOR + i;
                    int q_val = extract_int4(packed, i);
                    W_tile[k_local][n_local] = dequant_int4(s, q_val);
                }
            }
        }

        __syncthreads();

        // --- Compute: accumulate X_tile @ W_tile ---
        // Each thread computes TM x TN elements
        // Thread (warp_m, warp_n) handles rows [warp_m*TM .. warp_m*TM+TM-1]
        //                               cols  [warp_n*TN .. warp_n*TN+TN-1]
        #pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            #pragma unroll
            for (int tm = 0; tm < GEMM_TM; tm++) {
                float x_val = __half2float(X_tile[warp_m * GEMM_TM + tm][k]);
                #pragma unroll
                for (int tn = 0; tn < GEMM_TN; tn++) {
                    float w_val = __half2float(W_tile[k][warp_n * GEMM_TN + tn]);
                    acc[tm][tn] += x_val * w_val;
                }
            }
        }

        __syncthreads();
    }

    // --- Write output ---
    #pragma unroll
    for (int tm = 0; tm < GEMM_TM; tm++) {
        int global_m = bm + warp_m * GEMM_TM + tm;
        #pragma unroll
        for (int tn = 0; tn < GEMM_TN; tn++) {
            int global_n = bn + warp_n * GEMM_TN + tn;
            if (global_m < M && global_n < N) {
                Y[global_m * N + global_n] = __float2half(acc[tm][tn]);
            }
        }
    }
}

void gptq_gemm(const __half* X,
               const GptqWeight& weight,
               __half* Y,
               int M,
               cudaStream_t stream)
{
    dim3 grid((weight.N + GEMM_BN - 1) / GEMM_BN,
              (M + GEMM_BM - 1) / GEMM_BM);

    gptq_gemm_kernel<<<grid, GEMM_BLOCK_DIM, 0, stream>>>(
        X, weight.qweight, weight.scales, Y,
        M, weight.K, weight.N);
}

// ============================================================================
// Benchmark + correctness check
// ============================================================================

static void cpu_gptq_matmul(const __half* X, int M,
                             const uint32_t* qweight,
                             const __half* scales,
                             int K, int N,
                             float* Y_ref)
{
    // Reference: Y_ref[m, n] = sum_k X[m,k] * dequant(qweight, scales, k, n)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                int pk = k / GPTQ_PACK_FACTOR;
                int ki = k % GPTQ_PACK_FACTOR;
                uint32_t packed = qweight[pk * N + n];
                int q_val = (packed >> (ki * 4)) & 0xF;
                int group = k / GPTQ_GROUP_SIZE;
                float s = __half2float(scales[group * N + n]);
                float w = s * (float)(q_val - GPTQ_ZERO_POINT);
                sum += (double)__half2float(X[m * K + k]) * (double)w;
            }
            Y_ref[m * N + n] = (float)sum;
        }
    }
}

GptqBenchResult gptq_benchmark(int K, int N, int M,
                                int warmup_iters,
                                int bench_iters)
{
    GptqBenchResult result = {};
    result.K = K;
    result.N = N;
    result.M = M;

    // Allocate host data
    int packed_K = K / GPTQ_PACK_FACTOR;
    int num_groups = K / GPTQ_GROUP_SIZE;

    size_t x_size   = (size_t)M * K * sizeof(__half);
    size_t qw_size  = (size_t)packed_K * N * sizeof(uint32_t);
    size_t sc_size  = (size_t)num_groups * N * sizeof(__half);
    size_t y_size   = (size_t)M * N * sizeof(__half);

    __half*   h_X = (__half*)malloc(x_size);
    uint32_t* h_qw = (uint32_t*)malloc(qw_size);
    __half*   h_sc = (__half*)malloc(sc_size);
    __half*   h_Y = (__half*)malloc(y_size);
    float*    h_Y_ref = (float*)malloc((size_t)M * N * sizeof(float));

    // Initialize with deterministic pseudo-random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_X[i] = __float2half(((float)(rand() % 1000) - 500.0f) / 500.0f);
    }
    for (int i = 0; i < packed_K * N; i++) {
        // Random packed INT4 values (each nibble 0–15)
        h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
    }
    for (int i = 0; i < num_groups * N; i++) {
        h_sc[i] = __float2half(((float)(rand() % 200) - 100.0f) / 100.0f * 0.1f);
    }

    // CPU reference
    cpu_gptq_matmul(h_X, M, h_qw, h_sc, K, N, h_Y_ref);

    // Device allocation
    __half*   d_X;
    uint32_t* d_qw;
    __half*   d_sc;
    __half*   d_Y;

    cudaMalloc(&d_X, x_size);
    cudaMalloc(&d_qw, qw_size);
    cudaMalloc(&d_sc, sc_size);
    cudaMalloc(&d_Y, y_size);

    cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qw, h_qw, qw_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, h_sc, sc_size, cudaMemcpyHostToDevice);

    GptqWeight weight;
    weight.qweight = d_qw;
    weight.scales  = d_sc;
    weight.K       = K;
    weight.N       = N;

    // Correctness check
    gptq_linear(d_X, weight, d_Y, M);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Y, d_Y, y_size, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    double sum_err = 0.0, sum_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(h_Y[i]);
        float ref_val = h_Y_ref[i];
        float err = fabsf(gpu_val - ref_val);
        max_err = fmaxf(max_err, err);
        sum_err += (double)(err * err);
        sum_ref += (double)(ref_val * ref_val);
    }
    float rmse = (sum_ref > 0) ? (float)sqrt(sum_err / sum_ref) : 0.0f;
    // FP16 dequant precision allows ~2-3% relative error for large K reductions
    result.correct = (rmse < 0.05f);

    // Benchmark GEMV (M=1)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < warmup_iters; i++) {
            gptq_gemv(d_X, weight, d_Y);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < bench_iters; i++) {
            gptq_gemv(d_X, weight, d_Y);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        result.gemv_us = ms * 1000.0f / bench_iters;

        // Effective bandwidth: read qweight + scales + x, write y
        size_t bytes_read = qw_size + sc_size + (size_t)K * sizeof(__half);
        size_t bytes_write = (size_t)N * sizeof(__half);
        result.gemv_gbps = (bytes_read + bytes_write) / (result.gemv_us * 1e3f);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Benchmark GEMM (M=M)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < warmup_iters; i++) {
            gptq_gemm(d_X, weight, d_Y, M);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < bench_iters; i++) {
            gptq_gemm(d_X, weight, d_Y, M);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        result.gemm_us = ms * 1000.0f / bench_iters;

        // Effective TFLOPS: 2*M*N*K FLOPs (multiply-add)
        double flops = 2.0 * M * N * K;
        result.gemm_tflops = (float)(flops / ((double)result.gemm_us * 1e6));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_qw);
    cudaFree(d_sc);
    cudaFree(d_Y);
    free(h_X);
    free(h_qw);
    free(h_sc);
    free(h_Y);
    free(h_Y_ref);

    return result;
}

} // namespace deusridet
