// test_mma_ptx.cu - Full corrected test: ldmatrix + MMA
// Test 1: ldmatrix.x4 for A (verify register order matches MMA)
// Test 2: ldmatrix.x2.trans for B
// Test 3: Full GEMM: ldmatrix A + ldmatrix.trans B + MMA

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

// ==================== Test 1: ldmatrix.x4 for A ====================
// Load A[16][16] row-major from SMEM using ldmatrix.x4
// Verify fragment register order matches corrected MMA layout:
//   a0={A[g,2l],A[g,2l+1]}, a1={A[g+8,2l],A[g+8,2l+1]}, 
//   a2={A[g,2l+8],A[g,2l+9]}, a3={A[g+8,2l+8],A[g+8,2l+9]}
__global__ void test_ldmatrix_a(const half *A_row, uint32_t *out) {
    __shared__ half sA[16*16];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) sA[i] = A_row[i];
    __syncthreads();

    // ldmatrix.x4 needs each thread to point to a 128-bit row in SMEM
    // For the 16x16 row-major matrix, the WMMA fragment dump showed:
    // T00: a0={0,1}, a1={128,129}, a2={8,9}, a3={136,137}
    // Which means r0 loads row 0 cols 0-7, r1 loads row 8 cols 0-7,
    //             r2 loads row 0 cols 8-15, r3 loads row 8 cols 8-15
    // But ldmatrix.x4 loads r0,r1,r2,r3 from addresses provided by
    // the thread. Each thread provides ONE address; collectively 32 threads
    // provide 32 addresses, and ldmatrix picks 4 (from threads 0,8,16,24
    // for the respective registers... or from threads t%8 groups?)

    // Actually, ldmatrix.x4 loads 4 matrices of 8x8 each.
    // Each matrix uses 8 consecutive threads' addresses.
    // matrix 0: threads 0-7 -> r0
    // matrix 1: threads 8-15 -> r1
    // matrix 2: threads 16-23 -> r2
    // matrix 3: threads 24-31 -> r3

    // For the WMMA ordering, we need:
    // r0 = A rows 0-7, cols 0-7 (top-left 8x8)
    // r1 = A rows 8-15, cols 0-7 (bottom-left 8x8)
    // r2 = A rows 0-7, cols 8-15 (top-right 8x8)
    // r3 = A rows 8-15, cols 8-15 (bottom-right 8x8)

    // Addressing for each 8-thread group:
    // Threads 0-7: point to rows 0-7, col 0 -> sA[tid*16]
    // Threads 8-15: point to rows 8-15, col 0 -> sA[(tid-8+8)*16] = sA[tid*16]
    // Threads 16-23: point to rows 0-7, col 8 -> sA[(tid-16)*16 + 8]
    // Threads 24-31: point to rows 8-15, col 8 -> sA[(tid-24+8)*16 + 8]

    int group = tid / 8;   // 0,1,2,3
    int lane = tid % 8;    // 0..7
    uint32_t addr;
    switch(group) {
        case 0: addr = __cvta_generic_to_shared(&sA[lane * 16]); break;        // rows 0-7, col 0
        case 1: addr = __cvta_generic_to_shared(&sA[(lane+8) * 16]); break;    // rows 8-15, col 0
        case 2: addr = __cvta_generic_to_shared(&sA[lane * 16 + 8]); break;    // rows 0-7, col 8
        case 3: addr = __cvta_generic_to_shared(&sA[(lane+8) * 16 + 8]); break;// rows 8-15, col 8
    }

    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));

    // Store all registers
    out[tid*4+0] = r0;
    out[tid*4+1] = r1;
    out[tid*4+2] = r2;
    out[tid*4+3] = r3;
}

// ==================== Test 2: ldmatrix.x2.trans for B ====================
// B[16][8] row-major in SMEM, load as col-major fragment
// Fragment layout: b0={B[2l,g], B[2l+1,g]}, b1={B[2l+8,g], B[2l+9,g]}
__global__ void test_ldmatrix_b(const half *B_row, uint32_t *out) {
    __shared__ half sB[16*8];
    int tid = threadIdx.x;
    for (int i = tid; i < 128; i += 32) sB[i] = B_row[i];
    __syncthreads();

    // ldmatrix.x2.trans loads 2 transposed 8x8 matrices
    // Matrix 0: threads 0-7 -> r0 (k=0..7)
    // Matrix 1: threads 8-15 -> r1 (k=8..15)
    // But we have 32 threads... threads 16-31 use same addresses as 0-15

    // For B[16][8] stored row-major with stride 8:
    // Matrix 0 (k=0..7): threads 0-7 point to rows 0-7 -> sB[lane*8]
    // Matrix 1 (k=8..15): threads 8-15 point to rows 8-15 -> sB[(lane+8)*8]
    // Threads 16-23: same as 0-7, threads 24-31: same as 8-15

    int group = (tid / 8) % 2;
    int lane = tid % 8;
    uint32_t addr;
    if (group == 0)
        addr = __cvta_generic_to_shared(&sB[lane * 8]);
    else
        addr = __cvta_generic_to_shared(&sB[(lane + 8) * 8]);

    uint32_t r0, r1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));

    out[tid*2+0] = r0;
    out[tid*2+1] = r1;
}

// ==================== Test 3: Full ldmatrix + MMA ====================
__global__ void test_ldmatrix_mma(const half *A_row, const half *B_row, half *C) {
    __shared__ half sA[16*16], sB[16*8];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) sA[i] = A_row[i];
    for (int i = tid; i < 128; i += 32) sB[i] = B_row[i];
    __syncthreads();

    int g = tid / 4, l = tid % 4;

    // Load A via ldmatrix.x4
    int agroup = tid / 8, alane = tid % 8;
    uint32_t aaddr;
    switch(agroup) {
        case 0: aaddr = __cvta_generic_to_shared(&sA[alane * 16]); break;
        case 1: aaddr = __cvta_generic_to_shared(&sA[(alane+8) * 16]); break;
        case 2: aaddr = __cvta_generic_to_shared(&sA[alane * 16 + 8]); break;
        case 3: aaddr = __cvta_generic_to_shared(&sA[(alane+8) * 16 + 8]); break;
    }
    uint32_t a0, a1, a2, a3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(aaddr));

    // Load B via ldmatrix.x2.trans
    int bgroup = (tid / 8) % 2, blane = tid % 8;
    uint32_t baddr;
    if (bgroup == 0)
        baddr = __cvta_generic_to_shared(&sB[blane * 8]);
    else
        baddr = __cvta_generic_to_shared(&sB[(blane + 8) * 8]);
    uint32_t b0, b1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(baddr));

    // MMA
    uint32_t c0 = 0, c1 = 0, d0, d1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                   "r"(b0), "r"(b1), "r"(c0), "r"(c1));

    half2 h0 = *reinterpret_cast<half2*>(&d0);
    half2 h1 = *reinterpret_cast<half2*>(&d1);
    C[g*8 + l*2]       = __low2half(h0);
    C[g*8 + l*2 + 1]   = __high2half(h0);
    C[(g+8)*8 + l*2]   = __low2half(h1);
    C[(g+8)*8 + l*2+1] = __high2half(h1);
}

int main() {
    // A[m][k] = m*16+k, B[k][n] = k*8+n+1 (unique values)
    half hA[256], hB[128];
    for (int i = 0; i < 256; i++) hA[i] = __float2half((float)i);
    for (int i = 0; i < 128; i++) hB[i] = __float2half((float)(i+1));

    half *dA, *dB;
    cudaMalloc(&dA, 512); cudaMalloc(&dB, 256);
    cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);

    // ===== Test 1: ldmatrix.x4 A =====
    printf("=== Test 1: ldmatrix.x4 A fragment ===\n");
    {
        uint32_t *dout, hout[32*4];
        cudaMalloc(&dout, 32*4*4);
        test_ldmatrix_a<<<1,32>>>(dA, dout);
        cudaDeviceSynchronize();
        cudaMemcpy(hout, dout, 32*4*4, cudaMemcpyDeviceToHost);

        // Check against expected MMA A fragment:
        // a0={A[g,2l],A[g,2l+1]}, a1={A[g+8,2l],A[g+8,2l+1]},
        // a2={A[g,2l+8],A[g,2l+9]}, a3={A[g+8,2l+8],A[g+8,2l+9]}
        int errs = 0;
        for (int t = 0; t < 32; t++) {
            int g = t/4, l = t%4;
            half2 r0 = *reinterpret_cast<half2*>(&hout[t*4+0]);
            half2 r1 = *reinterpret_cast<half2*>(&hout[t*4+1]);
            half2 r2 = *reinterpret_cast<half2*>(&hout[t*4+2]);
            half2 r3 = *reinterpret_cast<half2*>(&hout[t*4+3]);
            float e0a = g*16+2*l, e0b = g*16+2*l+1;
            float e1a = (g+8)*16+2*l, e1b = (g+8)*16+2*l+1;
            float e2a = g*16+2*l+8, e2b = g*16+2*l+9;
            float e3a = (g+8)*16+2*l+8, e3b = (g+8)*16+2*l+9;
            bool ok = (fabsf(__half2float(__low2half(r0))-e0a)<0.5f &&
                       fabsf(__half2float(__high2half(r0))-e0b)<0.5f &&
                       fabsf(__half2float(__low2half(r1))-e1a)<0.5f &&
                       fabsf(__half2float(__high2half(r1))-e1b)<0.5f &&
                       fabsf(__half2float(__low2half(r2))-e2a)<0.5f &&
                       fabsf(__half2float(__high2half(r2))-e2b)<0.5f &&
                       fabsf(__half2float(__low2half(r3))-e3a)<0.5f &&
                       fabsf(__half2float(__high2half(r3))-e3b)<0.5f);
            if (!ok) errs++;
            if (t < 4 || !ok)
                printf("  T%02d g=%d l=%d: r0={%.0f,%.0f} r1={%.0f,%.0f} r2={%.0f,%.0f} r3={%.0f,%.0f} exp a0={%.0f,%.0f} a1={%.0f,%.0f} a2={%.0f,%.0f} a3={%.0f,%.0f} %s\n",
                       t,g,l,
                       __half2float(__low2half(r0)), __half2float(__high2half(r0)),
                       __half2float(__low2half(r1)), __half2float(__high2half(r1)),
                       __half2float(__low2half(r2)), __half2float(__high2half(r2)),
                       __half2float(__low2half(r3)), __half2float(__high2half(r3)),
                       e0a,e0b,e1a,e1b,e2a,e2b,e3a,e3b,
                       ok?"OK":"MISMATCH");
        }
        printf("  %s (%d mismatches)\n", errs==0?"PASSED":"FAILED", errs);
        cudaFree(dout);
    }

    // ===== Test 2: ldmatrix.x2.trans B =====
    printf("\n=== Test 2: ldmatrix.x2.trans B fragment ===\n");
    {
        uint32_t *dout, hout[32*2];
        cudaMalloc(&dout, 32*2*4);
        test_ldmatrix_b<<<1,32>>>(dB, dout);
        cudaDeviceSynchronize();
        cudaMemcpy(hout, dout, 32*2*4, cudaMemcpyDeviceToHost);

        // Expected B fragment: b0={B[2l,g], B[2l+1,g]}, b1={B[2l+8,g], B[2l+9,g]}
        // B[k][n] row-major = k*8+n+1. So B[2l,g] = (2l)*8+g+1
        int errs = 0;
        for (int t = 0; t < 32; t++) {
            int g = t/4, l = t%4;
            half2 r0 = *reinterpret_cast<half2*>(&hout[t*2+0]);
            half2 r1 = *reinterpret_cast<half2*>(&hout[t*2+1]);
            float e0a = (2*l)*8+g+1, e0b = (2*l+1)*8+g+1;
            float e1a = (2*l+8)*8+g+1, e1b = (2*l+9)*8+g+1;
            bool ok = (fabsf(__half2float(__low2half(r0))-e0a)<0.5f &&
                       fabsf(__half2float(__high2half(r0))-e0b)<0.5f &&
                       fabsf(__half2float(__low2half(r1))-e1a)<0.5f &&
                       fabsf(__half2float(__high2half(r1))-e1b)<0.5f);
            if (!ok) errs++;
            if (t < 4 || !ok)
                printf("  T%02d g=%d l=%d: b0={%.0f,%.0f} b1={%.0f,%.0f} exp b0={%.0f,%.0f} b1={%.0f,%.0f} %s\n",
                       t,g,l,
                       __half2float(__low2half(r0)), __half2float(__high2half(r0)),
                       __half2float(__low2half(r1)), __half2float(__high2half(r1)),
                       e0a,e0b,e1a,e1b,
                       ok?"OK":"MISMATCH");
        }
        printf("  %s (%d mismatches)\n", errs==0?"PASSED":"FAILED", errs);
        cudaFree(dout);
    }

    // ===== Test 3: Full ldmatrix MMA with random data =====
    printf("\n=== Test 3: Full ldmatrix + MMA (random) ===\n");
    {
        half hAr[256], hBr[128];
        srand(42);
        for (int i = 0; i < 256; i++) hAr[i] = __float2half((rand()%100-50)/50.f);
        for (int i = 0; i < 128; i++) hBr[i] = __float2half((rand()%100-50)/50.f);

        float fA[256], fB[128], fCref[128];
        for (int i = 0; i < 256; i++) fA[i] = __half2float(hAr[i]);
        for (int i = 0; i < 128; i++) fB[i] = __half2float(hBr[i]);
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 8; n++) {
                float s = 0;
                for (int k = 0; k < 16; k++) s += fA[m*16+k] * fB[k*8+n];
                fCref[m*8+n] = s;
            }

        half *dAr, *dBr, *dCr;
        cudaMalloc(&dAr, 512); cudaMalloc(&dBr, 256); cudaMalloc(&dCr, 256);
        cudaMemcpy(dAr, hAr, 512, cudaMemcpyHostToDevice);
        cudaMemcpy(dBr, hBr, 256, cudaMemcpyHostToDevice);
        cudaMemset(dCr, 0, 256);
        test_ldmatrix_mma<<<1,32>>>(dAr, dBr, dCr);
        cudaDeviceSynchronize();
        half hCr[128];
        cudaMemcpy(hCr, dCr, 256, cudaMemcpyDeviceToHost);

        float maxe = 0; int errs = 0;
        for (int i = 0; i < 128; i++) {
            float g = __half2float(hCr[i]), d = fabsf(g - fCref[i]);
            maxe = fmaxf(maxe, d);
            if (d > 0.5f) errs++;
        }
        printf("  max_err=%.4f errs(>0.5)=%d\n", maxe, errs);
        printf("  C[0][0]=%.4f ref=%.4f\n", __half2float(hCr[0]), fCref[0]);
        printf("  %s\n", errs==0?"PASSED":"FAILED");
        cudaFree(dAr); cudaFree(dBr); cudaFree(dCr);
    }

    cudaFree(dA); cudaFree(dB);
    return 0;
}
