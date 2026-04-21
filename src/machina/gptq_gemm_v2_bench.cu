/**
 * @file src/machina/gptq_gemm_v2_bench.cu
 * @philosophical_role
 *   Standalone correctness + head-to-head benchmark for gptq_gemm_v2 vs Marlin.
 *   Split from gptq_gemm_v2.cu under R1 800-line hard cap.
 * @serves
 *   Dev-only tooling; linked into the benchmark executable via the machina
 *   library. The production path never calls bench_gptq_v2_kernels().
 */
// gptq_gemm_v2_bench.cu — peer TU of gptq_gemm_v2.cu
//
// Contains only `bench_gptq_v2_kernels()` plus its static CPU/naive helpers.
// The fast-path kernels, dispatcher, and gptq_gemm_v2{,_add} launchers remain
// in gptq_gemm_v2.cu.

#include "gptq_gemm_v2.h"
#include "marlin.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace deusridet {

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
