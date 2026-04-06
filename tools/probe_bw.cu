// probe_bw.cu — Pure DRAM bandwidth ceiling measurement for SM87 Orin.
// Tests streaming read (copy kernel) vs theoretical 207 GB/s.
// Usage: ./probe_bw [size_mb]  (default: 100 MB)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Vectorized copy kernel: float4 = 16 bytes per thread per access
__global__ void copy_f4(const float4* __restrict__ src,
                        float4* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        dst[i] = src[i];
}

// Read-only kernel (sum to prevent optimization): measures pure read BW
__global__ void read_f4(const float4* __restrict__ src, float* out, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // Prevent dead code elimination
    if (idx == 0) *out = sum;
}

// cp.async based copy (load to SMEM, then store to global)
__global__ void copy_cpasync(const float4* __restrict__ src,
                             float4* __restrict__ dst, int n) {
    extern __shared__ float4 smem[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_stride = block_size * gridDim.x;

    for (int base = bid * block_size + tid; base < n; base += grid_stride) {
        // Load via cp.async to SMEM
        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem[tid]));
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_addr), "l"(&src[base]));
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        dst[base] = smem[tid];
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    int size_mb = (argc > 1) ? atoi(argv[1]) : 100;
    size_t bytes = (size_t)size_mb * 1024 * 1024;
    int n_f4 = bytes / sizeof(float4);

    printf("DRAM Bandwidth Probe: %d MB (%d float4 elements)\n", size_mb, n_f4);
    printf("Hardware: SM87 Orin, theoretical 207 GB/s\n\n");

    float4 *d_src, *d_dst;
    float *d_out;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_src, 0x42, bytes);
    cudaMemset(d_dst, 0, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 5;
    const int iters = 20;
    float ms;

    // Grid: 16 SMs, plenty of blocks for full occupancy
    int threads = 256;
    int blocks = 16 * 8;  // 8 blocks per SM

    // === Test 1: Copy (read + write) ===
    for (int i = 0; i < warmup; i++)
        copy_f4<<<blocks, threads>>>(d_src, d_dst, n_f4);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        copy_f4<<<blocks, threads>>>(d_src, d_dst, n_f4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    float copy_us = ms * 1000.0f / iters;
    float copy_gbps = (2.0f * bytes) / (copy_us * 1e3f);  // read + write
    printf("Copy (R+W):     %8.1f us  %6.1f GB/s (read: %.1f GB/s)\n",
           copy_us, copy_gbps, copy_gbps / 2.0f);

    // === Test 2: Read-only ===
    for (int i = 0; i < warmup; i++)
        read_f4<<<blocks, threads>>>(d_src, d_out, n_f4);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        read_f4<<<blocks, threads>>>(d_src, d_out, n_f4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    float read_us = ms * 1000.0f / iters;
    float read_gbps = (float)bytes / (read_us * 1e3f);
    printf("Read-only:      %8.1f us  %6.1f GB/s\n", read_us, read_gbps);

    // === Test 3: cp.async copy ===
    int smem_bytes = threads * sizeof(float4);
    cudaFuncSetAttribute(copy_cpasync,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    for (int i = 0; i < warmup; i++)
        copy_cpasync<<<blocks, threads, smem_bytes>>>(d_src, d_dst, n_f4);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        copy_cpasync<<<blocks, threads, smem_bytes>>>(d_src, d_dst, n_f4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    float cpasync_us = ms * 1000.0f / iters;
    float cpasync_gbps = (2.0f * bytes) / (cpasync_us * 1e3f);
    printf("cp.async (R+W): %8.1f us  %6.1f GB/s (read: %.1f GB/s)\n",
           cpasync_us, cpasync_gbps, cpasync_gbps / 2.0f);

    // === Test 4: cudaMemcpy D2D ===
    for (int i = 0; i < warmup; i++)
        cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    float memcpy_us = ms * 1000.0f / iters;
    float memcpy_gbps = (2.0f * bytes) / (memcpy_us * 1e3f);
    printf("cudaMemcpy D2D: %8.1f us  %6.1f GB/s (read: %.1f GB/s)\n",
           memcpy_us, memcpy_gbps, memcpy_gbps / 2.0f);

    printf("\n=== GEMM-comparable read BW (write is ~1%% of total for M=64): ===\n");
    printf("Copy read-half: %.1f GB/s\n", copy_gbps / 2.0f);
    printf("Pure read:      %.1f GB/s\n", read_gbps);
    printf("Target: 198 GB/s, Hardware peak: 207 GB/s\n");

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
