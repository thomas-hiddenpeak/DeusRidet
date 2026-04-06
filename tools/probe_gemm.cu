// probe_gemm.cu — Minimal GEMM probe for ncu profiling
//
// Compares cuBLAS vs custom FP16 GEMM on representative shapes.
// Build: nvcc -O3 --use_fast_math -gencode arch=compute_87,code=sm_87
//        -o probe_gemm probe_gemm.cu -lcublas -lcudart

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

// Import our custom kernel
#include "../src/machina/fp16_gemm.h"

#include <curand.h>
#include <cmath>

// Minimal cuBLAS GEMM wrapper (matches linear_forward)
void cublas_fp16_gemm(cublasHandle_t handle,
                      const __half* A, const __half* B, __half* C,
                      int M, int K, int N, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);
    // C[M,N] = A[M,K] @ B^T[K,N], B is [N,K] row-major
    // cuBLAS col-major: C^T[N,M] = B[N,K] @ A^T[K,M]
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, CUDA_R_16F, K,
                 A, CUDA_R_16F, K,
                 &beta,
                 C, CUDA_R_16F, N,
                 CUDA_R_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Fill buffer with small random values via cuRAND
__global__ void scale_to_half(const float* in, __half* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i] * 0.2f - 0.1f);
}

void fill_random(__half* d_ptr, size_t n) {
    float* d_tmp;
    cudaMalloc(&d_tmp, n * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, d_tmp, n);
    curandDestroyGenerator(gen);
    scale_to_half<<<(n + 255) / 256, 256>>>(d_tmp, d_ptr, n);
    cudaDeviceSynchronize();
    cudaFree(d_tmp);
}

// Repack B on device: d_B [N,K] → d_B_repacked [N/64, K/64, 64, 64] K-major tiles
void repack_b_device(__half* d_B, __half* d_B_repacked, int N, int K) {
    size_t sz = (size_t)N * K * 2;
    __half* h_B = new __half[N * K];
    __half* h_B_r = new __half[N * K];
    cudaMemcpy(h_B, d_B, sz, cudaMemcpyDeviceToHost);
    deusridet::fp16_repack_b(h_B, h_B_r, N, K);
    cudaMemcpy(d_B_repacked, h_B_r, sz, cudaMemcpyHostToDevice);
    delete[] h_B;
    delete[] h_B_r;
}

int main(int argc, char** argv) {
    // Default: DN qkv projection [M=64, K=5120, N=10240]
    int M = 64, K = 5120, N = 10240;
    int mode = 0;  // 0=cuBLAS, 1=custom, 2=correctness check

    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) M = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) N = atoi(argv[4]);

    const char* mode_str[] = {"cuBLAS", "custom", "correctness", "timing"};
    printf("GEMM probe: M=%d K=%d N=%d mode=%s\n", M, K, N,
           mode <= 3 ? mode_str[mode] : "unknown");

    // Allocate
    __half *d_A, *d_B, *d_B_repacked, *d_C_cublas, *d_C_custom;
    cudaMalloc(&d_A, (size_t)M * K * 2);
    cudaMalloc(&d_B, (size_t)N * K * 2);
    cudaMalloc(&d_B_repacked, (size_t)N * K * 2);
    cudaMalloc(&d_C_cublas, (size_t)M * N * 2);
    cudaMalloc(&d_C_custom, (size_t)M * N * 2);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    if (mode == 2) {
        // === Correctness check: compare custom vs cuBLAS ===
        fill_random(d_A, (size_t)M * K);
        fill_random(d_B, (size_t)N * K);
        repack_b_device(d_B, d_B_repacked, N, K);

        // cuBLAS reference (uses original B)
        cublas_fp16_gemm(handle, d_A, d_B, d_C_cublas, M, K, N, 0);
        cudaDeviceSynchronize();

        // Custom kernel (uses repacked B)
        deusridet::fp16_gemm(d_A, d_B_repacked, d_C_custom, M, K, N, 0);
        cudaDeviceSynchronize();

        // Copy to host and compare
        size_t out_size = (size_t)M * N;
        __half* h_ref = new __half[out_size];
        __half* h_cus = new __half[out_size];
        cudaMemcpy(h_ref, d_C_cublas, out_size * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cus, d_C_custom, out_size * 2, cudaMemcpyDeviceToHost);

        float max_abs_err = 0, max_rel_err = 0;
        double sum_sq_err = 0;
        int nan_count = 0, zero_ref = 0, large_err_count = 0;
        for (size_t i = 0; i < out_size; i++) {
            float ref = __half2float(h_ref[i]);
            float cus = __half2float(h_cus[i]);
            if (std::isnan(cus)) { nan_count++; continue; }
            float err = fabsf(ref - cus);
            sum_sq_err += (double)err * err;
            if (err > max_abs_err) max_abs_err = err;
            float denom = fmaxf(fabsf(ref), 1e-6f);
            float rel = err / denom;
            if (rel > max_rel_err) max_rel_err = rel;
            if (fabsf(ref) < 1e-8f) zero_ref++;
            if (rel > 0.05f) large_err_count++;  // >5% relative error
        }
        float rmse = sqrtf((float)(sum_sq_err / out_size));

        printf("=== Correctness Report ===\n");
        printf("  Output size:    %zu elements (%d × %d)\n", out_size, M, N);
        printf("  Max abs error:  %.6f\n", max_abs_err);
        printf("  Max rel error:  %.4f%%\n", max_rel_err * 100);
        printf("  RMSE:           %.6f\n", rmse);
        printf("  NaN count:      %d\n", nan_count);
        printf("  Large err (>5%%): %d / %zu (%.2f%%)\n",
               large_err_count, out_size,
               100.0f * large_err_count / out_size);

        // Print first few elements for visual inspection
        printf("  First 8 values [row 0]:\n");
        printf("    cuBLAS: ");
        for (int i = 0; i < 8 && i < N; i++)
            printf("%.4f ", __half2float(h_ref[i]));
        printf("\n    Custom: ");
        for (int i = 0; i < 8 && i < N; i++)
            printf("%.4f ", __half2float(h_cus[i]));
        printf("\n");

        bool pass = (nan_count == 0) && (max_abs_err < 0.2f) && (rmse < 0.01f);
        printf("  Result: %s\n", pass ? "PASS ✓" : "FAIL ✗");

        delete[] h_ref;
        delete[] h_cus;
    } else if (mode == 3) {
        // === Timing comparison: cuBLAS vs custom ===
        fill_random(d_A, (size_t)M * K);
        fill_random(d_B, (size_t)N * K);
        repack_b_device(d_B, d_B_repacked, N, K);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int warmup = 10, iters = 50;

        // Warmup cuBLAS
        for (int i = 0; i < warmup; i++)
            cublas_fp16_gemm(handle, d_A, d_B, d_C_cublas, M, K, N, 0);
        cudaDeviceSynchronize();

        // Time cuBLAS
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++)
            cublas_fp16_gemm(handle, d_A, d_B, d_C_cublas, M, K, N, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cublas_ms;
        cudaEventElapsedTime(&cublas_ms, start, stop);
        float cublas_us = cublas_ms * 1000.0f / iters;

        // Warmup custom (uses repacked B)
        for (int i = 0; i < warmup; i++)
            deusridet::fp16_gemm(d_A, d_B_repacked, d_C_custom, M, K, N, 0);
        cudaDeviceSynchronize();

        // Time custom
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++)
            deusridet::fp16_gemm(d_A, d_B_repacked, d_C_custom, M, K, N, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float custom_ms;
        cudaEventElapsedTime(&custom_ms, start, stop);
        float custom_us = custom_ms * 1000.0f / iters;

        float data_mb = (float)(M * K + N * K + M * N) * 2.0f / (1024*1024);
        float cublas_bw = data_mb / (cublas_us / 1e6) / 1024;  // GB/s
        float custom_bw = data_mb / (custom_us / 1e6) / 1024;

        printf("=== Timing (%d iters) ===\n", iters);
        printf("  cuBLAS:  %.1f us  (%.1f GB/s)\n", cublas_us, cublas_bw);
        printf("  Custom:  %.1f us  (%.1f GB/s)\n", custom_us, custom_bw);
        printf("  Speedup: %.2fx\n", cublas_us / custom_us);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        // === Profile mode: run one kernel for ncu ===
        cudaMemset(d_A, 0, (size_t)M * K * 2);
        cudaMemset(d_B, 0, (size_t)N * K * 2);

        // For custom kernel, need repacked B
        if (mode == 1) {
            repack_b_device(d_B, d_B_repacked, N, K);
        }

        __half* d_C = (mode == 0) ? d_C_cublas : d_C_custom;

        // Warmup
        for (int i = 0; i < 3; i++) {
            if (mode == 0)
                cublas_fp16_gemm(handle, d_A, d_B, d_C, M, K, N, 0);
            else
                deusridet::fp16_gemm(d_A, d_B_repacked, d_C, M, K, N, 0);
        }
        cudaDeviceSynchronize();

        // Measured run (ncu captures this one)
        if (mode == 0)
            cublas_fp16_gemm(handle, d_A, d_B, d_C, M, K, N, 0);
        else
            deusridet::fp16_gemm(d_A, d_B_repacked, d_C, M, K, N, 0);
        cudaDeviceSynchronize();

        printf("Done.\n");
    }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_repacked);
    cudaFree(d_C_cublas);
    cudaFree(d_C_custom);
    return 0;
}
