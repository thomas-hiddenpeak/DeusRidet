/**
 * @file src/machina/gptq.cu
 * @philosophical_role
 *   GPTQ-Int4 dequant + GEMV/GEMM kernels — the scheme that lets the entity's mind fit in Orin DRAM. Every quantized linear layer asks this file how to become a product again.
 * @serves
 *   Machina forward pass for all GPTQ-quantized layers; Actus diagnostic (bench_gptq, test_gptq) via the engine library.
 */
// gptq.cu — GPTQ-Int4 dequant + GEMV/GEMM CUDA kernels for SM87
//
// Kernel design for Jetson AGX Orin (SM87, Ampere):
//   - 16 SMs, 128 FP32 cores each, L2 = 4 MB
//   - Memory BW ~192 GB/s (unified DRAM)
//   - Shared memory: up to 164 KB/SM (48 KB default)
//   - FP16 tensor core available (WMMA m16n16k16)
//
// GPTQ layout: qweight[K/8, N] (uint32), scales[K/128, N] (FP16)
//   bits=4, group_size=128, sym=true, zero_point=8
//
// GEMV strategy (M=1, memory-bound):
//   Each warp handles a slice of output columns. Each thread processes
//   multiple K elements, accumulating in FP32 for numerical stability.
//   Final warp reduction → FP16 output.
//
// GEMM strategy (M>1, CUDA core fallback):
//   Tile-based: each thread block computes a tile of Y[BM, BN].
//   Load qweight tile into shared memory, dequant on the fly.
//
// WMMA GEMM strategy (M>1, tensor core):
//   Tile-based with WMMA m16n16k16 fragments.
//   BK=128 aligned to group_size → one scale per tile per column.
//   INT4 dequant in registers during SMEM cooperative load.

#include "gptq.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
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
// GPU-based dequant kernel for reference: fully dequantize W_q → W_fp32
// Used by benchmark correctness check (replaces slow CPU matmul)
// ============================================================================

__global__ void gptq_dequant_kernel(
    const uint32_t* __restrict__ qweight, // [K/8, N]
    const __half*   __restrict__ scales,  // [K/128, N]
    float*          __restrict__ W_fp32,  // [K, N] output
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;

    int k = idx / N;
    int n = idx % N;
    int pk = k / GPTQ_PACK_FACTOR;
    int ki = k % GPTQ_PACK_FACTOR;
    int group = k / GPTQ_GROUP_SIZE;

    uint32_t packed = qweight[pk * N + n];
    int q_val = (packed >> (ki * 4)) & 0xF;
    float s = __half2float(scales[group * N + n]);
    W_fp32[k * N + n] = s * (float)(q_val - GPTQ_ZERO_POINT);
}

GptqBenchResult gptq_benchmark(int K, int N, int M,
                                int warmup_iters,
                                int bench_iters)
{
    GptqBenchResult result = {};
    result.K = K;
    result.N = N;
    result.M = M;

    // Allocate host data for initialization
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

    // Initialize with deterministic pseudo-random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_X[i] = __float2half(((float)(rand() % 1000) - 500.0f) / 500.0f);
    }
    for (int i = 0; i < packed_K * N; i++) {
        h_qw[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
    }
    for (int i = 0; i < num_groups * N; i++) {
        h_sc[i] = __float2half(((float)(rand() % 200) - 100.0f) / 100.0f * 0.1f);
    }

    // Device allocation
    __half*   d_X;
    uint32_t* d_qw;
    __half*   d_sc;
    __half*   d_Y;
    float*    d_W_ref;   // dequantized weights for cuBLAS reference
    float*    d_X_fp32;  // FP32 input for cuBLAS
    float*    d_Y_ref;   // FP32 reference output

    cudaMalloc(&d_X, x_size);
    cudaMalloc(&d_qw, qw_size);
    cudaMalloc(&d_sc, sc_size);
    cudaMalloc(&d_Y, y_size);
    cudaMalloc(&d_W_ref, (size_t)K * N * sizeof(float));
    cudaMalloc(&d_X_fp32, (size_t)M * K * sizeof(float));
    cudaMalloc(&d_Y_ref, (size_t)M * N * sizeof(float));

    cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qw, h_qw, qw_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, h_sc, sc_size, cudaMemcpyHostToDevice);

    GptqWeight weight;
    weight.qweight = d_qw;
    weight.scales  = d_sc;
    weight.K       = K;
    weight.N       = N;

    // GPU reference: dequant to FP32 + cuBLAS SGEMM
    // 1. Dequantize qweight → W_fp32 on GPU
    {
        int total_elems = K * N;
        int block = 256;
        int grid = (total_elems + block - 1) / block;
        gptq_dequant_kernel<<<grid, block>>>(d_qw, d_sc, d_W_ref, K, N);
    }

    // 2. Convert X from FP16 to FP32 for cuBLAS SGEMM reference
    {
        float* h_X_fp32 = (float*)malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < M * K; i++) {
            h_X_fp32[i] = __half2float(h_X[i]);
        }
        cudaMemcpy(d_X_fp32, h_X_fp32, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice);
        free(h_X_fp32);
    }

    // 3. cuBLAS SGEMM: Y_ref = X_fp32 @ W_fp32^T ... but our layout is row-major
    //    Y[M,N] = X[M,K] * W[K,N] (row-major)
    //    cuBLAS expects column-major, so: Y^T[N,M] = W^T[N,K] * X^T[K,M]
    //    i.e. cublasSgemm(N, M, K, W_ref, N, X_fp32, K, Y_ref, N)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_W_ref, N,     // W[K,N] stored row-major = W^T[N,K] col-major
                    d_X_fp32, K,    // X[M,K] stored row-major = X^T[K,M] col-major
                    &beta,
                    d_Y_ref, N);    // Y[M,N] stored row-major = Y^T[N,M] col-major
        cudaDeviceSynchronize();
        cublasDestroy(handle);
    }

    // 4. Run GPTQ kernel under test
    gptq_linear(d_X, weight, d_Y, M);
    cudaDeviceSynchronize();

    // 5. Compare on host
    cudaMemcpy(h_Y, d_Y, y_size, cudaMemcpyDeviceToHost);
    float* h_Y_ref = (float*)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(h_Y_ref, d_Y_ref, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    double sum_err = 0.0, sum_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(h_Y[i]);
        float ref_val = h_Y_ref[i];
        float err = fabsf(gpu_val - ref_val);
        sum_err += (double)(err * err);
        sum_ref += (double)(ref_val * ref_val);
    }
    float rmse = (sum_ref > 0) ? (float)sqrt(sum_err / sum_ref) : 0.0f;
    // FP16 dequant precision allows ~2-3% relative error for large K reductions
    result.correct = (rmse < 0.05f);

    free(h_Y_ref);
    cudaFree(d_W_ref);
    cudaFree(d_X_fp32);
    cudaFree(d_Y_ref);

    // Benchmark: only run the relevant kernel (GEMV for M=1, GEMM for M>1)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (M == 1) {
        // Benchmark GEMV
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

        size_t bytes_read = qw_size + sc_size + (size_t)K * sizeof(__half);
        size_t bytes_write = (size_t)N * sizeof(__half);
        result.gemv_gbps = (bytes_read + bytes_write) / (result.gemv_us * 1e3f);
    } else {
        // Benchmark GEMM
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

        double flops = 2.0 * M * N * K;
        result.gemm_tflops = (float)(flops / ((double)result.gemm_us * 1e6));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_qw);
    cudaFree(d_sc);
    cudaFree(d_Y);
    free(h_X);
    free(h_qw);
    free(h_sc);
    free(h_Y);

    return result;
}

} // namespace deusridet
