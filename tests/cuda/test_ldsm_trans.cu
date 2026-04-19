// test_ldsm_trans.cu — Verify ldmatrix.x2.trans fragment layout on SM87
// Build: nvcc -O3 -gencode arch=compute_87,code=sm_87 -o build/test_ldsm_trans tests/cuda/test_ldsm_trans.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

__device__ void ldsm2_trans(uint32_t* frag, const void* smem_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(smem)
    );
}

// Test kernel: load a known 8×16 pattern via ldmatrix.x2.trans
// Source: B_smem[n][k] = 100*n + k  (8 N-rows, 16 K-cols)
// Expected MMA B fragment for thread T:
//   b[0] = {B[T/4, T%4*2], B[T/4, T%4*2+1]}
//   b[1] = {B[T/4, T%4*2+8], B[T/4, T%4*2+9]}
__global__ void test_ldmatrix_trans(uint32_t* out_b0, uint32_t* out_b1,
                                    uint32_t* exp_b0, uint32_t* exp_b1) {
    __shared__ __half smem[8 * 16];  // [8, 16] = 8 N-rows, 16 K-cols

    int tid = threadIdx.x;
    int lane = tid % 32;

    // Fill SMEM with known pattern: B[n][k] = 100*n + k
    if (tid < 128) {
        int n = tid / 16;
        int k = tid % 16;
        smem[n * 16 + k] = __float2half((float)(100 * n + k));
    }
    __syncthreads();

    if (tid < 32) {
        // ldmatrix.x2.trans addresses:
        // Lane 0-7: row lane%8, K=0..7 (first 8x8)
        // Lane 8-15: row lane%8, K=8..15 (second 8x8)
        int lane16 = lane % 16;
        int n_idx = lane16 % 8;
        int k_group = lane16 / 8;
        const __half* ptr = &smem[n_idx * 16 + k_group * 8];

        uint32_t frag[2];
        ldsm2_trans(frag, ptr);

        out_b0[lane] = frag[0];
        out_b1[lane] = frag[1];

        // Expected values (manual construction matching MMA B fragment)
        int n = lane / 4;  // 0..7
        int k_pair = lane % 4;  // 0..3
        __half e0_a = __float2half((float)(100 * n + k_pair * 2));
        __half e0_b = __float2half((float)(100 * n + k_pair * 2 + 1));
        __half e1_a = __float2half((float)(100 * n + k_pair * 2 + 8));
        __half e1_b = __float2half((float)(100 * n + k_pair * 2 + 9));

        uint32_t e0, e1;
        *(reinterpret_cast<__half*>(&e0) + 0) = e0_a;
        *(reinterpret_cast<__half*>(&e0) + 1) = e0_b;
        *(reinterpret_cast<__half*>(&e1) + 0) = e1_a;
        *(reinterpret_cast<__half*>(&e1) + 1) = e1_b;

        exp_b0[lane] = e0;
        exp_b1[lane] = e1;
    }
}

int main() {
    uint32_t *d_out0, *d_out1, *d_exp0, *d_exp1;
    cudaMalloc(&d_out0, 32 * 4);
    cudaMalloc(&d_out1, 32 * 4);
    cudaMalloc(&d_exp0, 32 * 4);
    cudaMalloc(&d_exp1, 32 * 4);

    test_ldmatrix_trans<<<1, 128>>>(d_out0, d_out1, d_exp0, d_exp1);
    cudaDeviceSynchronize();

    uint32_t h_out0[32], h_out1[32], h_exp0[32], h_exp1[32];
    cudaMemcpy(h_out0, d_out0, 32*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out1, d_out1, 32*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exp0, d_exp0, 32*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exp1, d_exp1, 32*4, cudaMemcpyDeviceToHost);

    printf("Lane | out_b0       exp_b0     | out_b1       exp_b1     | match?\n");
    printf("-----|--------------------------|--------------------------|\n");
    int mismatches = 0;
    for (int i = 0; i < 32; i++) {
        __half* oh0 = reinterpret_cast<__half*>(&h_out0[i]);
        __half* oh1 = reinterpret_cast<__half*>(&h_out1[i]);
        __half* eh0 = reinterpret_cast<__half*>(&h_exp0[i]);
        __half* eh1 = reinterpret_cast<__half*>(&h_exp1[i]);

        bool match = (h_out0[i] == h_exp0[i]) && (h_out1[i] == h_exp1[i]);
        if (!match) mismatches++;

        printf("%4d | (%6.0f,%6.0f) (%6.0f,%6.0f) | (%6.0f,%6.0f) (%6.0f,%6.0f) | %s\n",
               i,
               __half2float(oh0[0]), __half2float(oh0[1]),
               __half2float(eh0[0]), __half2float(eh0[1]),
               __half2float(oh1[0]), __half2float(oh1[1]),
               __half2float(eh1[0]), __half2float(eh1[1]),
               match ? "OK" : "MISMATCH");
    }
    printf("\n%d mismatches out of 32 lanes\n", mismatches);

    cudaFree(d_out0); cudaFree(d_out1);
    cudaFree(d_exp0); cudaFree(d_exp1);
    return 0;
}
