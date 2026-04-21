/**
 * @file src/machina/layer_int8_dual.cu
 * @philosophical_role
 *   INT8 dual-linear (Q/K combined, gate/up combined) + quantization helpers.
 *   Peer TU of layer.cu under R1 800-line hard cap.
 * @serves
 *   Machina forward.cu attention/MLP fused linears; weight quantization.
 */
// layer_int8_dual.cu — peer TU of layer.cu (dual gemv + quantize).

#include "layer.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cfloat>
#include <vector>

namespace deusridet {

// Duplicated TU-local from layer_int8.cu (R1 split — these constants are
// not exposed via header).
constexpr int INT8_GEMV_WARPS = 4;
constexpr int INT8_GEMV_BLOCK = INT8_GEMV_WARPS * 32;
constexpr int I8TC_BM = 16;
constexpr int I8TC_BN = 64;
constexpr int I8TC_BK = 128;
constexpr int I8TC_BK_PAD = I8TC_BK + 8;
constexpr int I8TC_BLOCK = 128;


// ============================================================================
// Dual INT8 batch GEMV — two weight matrices sharing one X, M>1 tokens.
//
// Y1[M,N1] = X[M,K] @ W1^T   and   Y2[M,N2] = X[M,K] @ W2^T
// Single kernel launch. Warps are assigned to (N1+N2) outputs; each selects
// which matrix to read from and which output buffer to write.
// Used for: DeltaNet in_proj_a+b (2×48), FullAttn k_proj+v_proj (2×1024).
// ============================================================================

__global__ void int8_dual_batch_gemv_kernel(
    const __half*  __restrict__ X,       // [M, K]
    const int8_t*  __restrict__ W1, const float* __restrict__ scales1,
    __half*        __restrict__ Y1,      // [M, N1]
    const int8_t*  __restrict__ W2, const float* __restrict__ scales2,
    __half*        __restrict__ Y2,      // [M, N2]
    int M, int K, int N1, int N2)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int global_n = blockIdx.x * INT8_GEMV_WARPS + warp_id;
    const int total_N = N1 + N2;
    if (global_n >= total_N) return;

    // Select matrix (warp-uniform branch)
    const int8_t* w_row;
    float scale;
    __half* y;
    int n, N_out;
    if (global_n < N1) {
        n = global_n;
        w_row = W1 + (size_t)n * K;
        scale = scales1[n];
        y = Y1;
        N_out = N1;
    } else {
        n = global_n - N1;
        w_row = W2 + (size_t)n * K;
        scale = scales2[n];
        y = Y2;
        N_out = N2;
    }

    float acc[128];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    const int vec_end = (K / 512) * 512;
    for (int k = lane_id * 16; k < vec_end; k += 512) {
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        const int8_t* wp = reinterpret_cast<const int8_t*>(&wv);

        for (int m = 0; m < M; m++) {
            float4 xv0 = *reinterpret_cast<const float4*>(X + (size_t)m * K + k);
            float4 xv1 = *reinterpret_cast<const float4*>(X + (size_t)m * K + k + 8);
            const __half* xp = reinterpret_cast<const __half*>(&xv0);
            const __half* xp1 = reinterpret_cast<const __half*>(&xv1);

            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                sum += (float)wp[i] * __half2float(xp[i]);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                sum += (float)wp[8 + i] * __half2float(xp1[i]);
            acc[m] += sum;
        }
    }

    for (int k = vec_end + lane_id; k < K; k += 32) {
        int8_t wval = w_row[k];
        for (int m = 0; m < M; m++)
            acc[m] += (float)wval * __half2float(X[(size_t)m * K + k]);
    }

    for (int m = 0; m < M; m++) {
        float val = acc[m] * scale;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (lane_id == 0)
            y[(size_t)m * N_out + n] = __float2half(val);
    }
}

// ============================================================================
// Dual INT8 GEMV — two weight matrices sharing one x in SMEM
//
// Computes y1[N1] = W1[N1,K] @ x   and   y2[N2] = W2[N2,K] @ x
// in a single kernel launch. Saves one x load and one kernel dispatch.
// Used for: DeltaNet in_proj_a+b (2×48), FullAttn k_proj+v_proj (2×1024).
// ============================================================================

__global__ void int8_dual_gemv_kernel(
    const __half*  __restrict__ x,
    const int8_t*  __restrict__ W1, const float* __restrict__ scales1,
    __half*        __restrict__ y1,
    const int8_t*  __restrict__ W2, const float* __restrict__ scales2,
    __half*        __restrict__ y2,
    int K, int N1, int N2)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int global_n = blockIdx.x * INT8_GEMV_WARPS + warp_id;
    const int total_N = N1 + N2;

    // Cooperative load of x into SMEM (shared by all warps for both matrices)
    extern __shared__ __half smem_x[];
    const int vec_k = K / 8;
    for (int i = threadIdx.x; i < vec_k; i += blockDim.x)
        reinterpret_cast<float4*>(smem_x)[i] = reinterpret_cast<const float4*>(x)[i];
    __syncthreads();

    if (global_n >= total_N) return;

    // Select which matrix this warp operates on (warp-uniform branch)
    const int8_t* w_row;
    float scale;
    __half* y;
    int n;
    if (global_n < N1) {
        n = global_n;
        w_row = W1 + (size_t)n * K;
        scale = scales1[n];
        y = y1;
    } else {
        n = global_n - N1;
        w_row = W2 + (size_t)n * K;
        scale = scales2[n];
        y = y2;
    }

    float acc = 0.0f;
    const int vec_end = (K / 512) * 512;
    for (int k = lane_id * 16; k < vec_end; k += 512) {
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        float4 xv0 = *reinterpret_cast<const float4*>(smem_x + k);
        float4 xv1 = *reinterpret_cast<const float4*>(smem_x + k + 8);

        const int8_t* wp = reinterpret_cast<const int8_t*>(&wv);
        const __half* xp = reinterpret_cast<const __half*>(&xv0);
        const __half* xp1 = reinterpret_cast<const __half*>(&xv1);

        #pragma unroll
        for (int i = 0; i < 8; i++)
            acc += (float)wp[i] * __half2float(xp[i]);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            acc += (float)wp[8 + i] * __half2float(xp1[i]);
    }

    for (int k = vec_end + lane_id; k < K; k += 32)
        acc += (float)w_row[k] * __half2float(smem_x[k]);

    acc *= scale;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane_id == 0) y[n] = __float2half(acc);
}

void int8_dual_linear_forward(const __half* X,
                               const Int8Linear& w1, __half* Y1,
                               const Int8Linear& w2, __half* Y2,
                               int M, cudaStream_t stream) {
    if (M == 1) {
        int K = w1.in_features;  // both must have same K
        int N1 = w1.out_features;
        int N2 = w2.out_features;
        int grid = (N1 + N2 + INT8_GEMV_WARPS - 1) / INT8_GEMV_WARPS;
        int smem = K * sizeof(__half);
        int8_dual_gemv_kernel<<<grid, INT8_GEMV_BLOCK, smem, stream>>>(
            X, w1.weight, w1.scales, Y1,
               w2.weight, w2.scales, Y2,
            K, N1, N2);
        return;
    }
    // M>1: use dual batch GEMV only when individual calls would NOT use WMMA.
    // WMMA (tensor core) path is faster for WMMA-compatible dimensions.
    {
        int K = w1.in_features;
        int N1 = w1.out_features;
        int N2 = w2.out_features;
        bool w1_wmma = (K % I8TC_BK == 0 && N1 % I8TC_BN == 0);
        bool w2_wmma = (K % I8TC_BK == 0 && N2 % I8TC_BN == 0);
        if (w1_wmma || w2_wmma) {
            // At least one uses WMMA — separate calls are faster
            int8_linear_forward(X, w1, Y1, M, stream);
            int8_linear_forward(X, w2, Y2, M, stream);
        } else {
            // Both would use batch_gemv — fuse into single launch
            int grid = (N1 + N2 + INT8_GEMV_WARPS - 1) / INT8_GEMV_WARPS;
            int8_dual_batch_gemv_kernel<<<grid, INT8_GEMV_BLOCK, 0, stream>>>(
                X, w1.weight, w1.scales, Y1,
                   w2.weight, w2.scales, Y2,
                M, K, N1, N2);
        }
    }
}

// ============================================================================
// GPU quantization kernel: FP16 → INT8 per-channel symmetric
//
// Phase 1: find max|w| per row (per-output-channel) → scale = max_abs / 127
// Phase 2: quantize w_int8[n][k] = round(w_fp16[n][k] / scale[n])
// ============================================================================

__global__ void find_absmax_kernel(
    const __half* __restrict__ W,  // [N, K] row-major FP16
    float* __restrict__ scales,    // [N] output: scale per row
    int K)
{
    const int n = blockIdx.x;
    const __half* row = W + (size_t)n * K;
    float max_val = 0.0f;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = fabsf(__half2float(row[k]));
        max_val = fmaxf(max_val, v);
    }

    // Block-level max reduction via warp shuffles
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));

    // Inter-warp reduction via shared memory
    __shared__ float warp_max[8];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_max[warp] = max_val;
    __syncthreads();

    if (warp == 0) {
        max_val = (lane < (blockDim.x + 31) / 32) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        if (lane == 0)
            scales[n] = max_val / 127.0f;
    }
}

__global__ void quantize_to_int8_kernel(
    const __half* __restrict__ W,      // [N, K] FP16
    const float*  __restrict__ scales, // [N]
    int8_t*       __restrict__ W_int8, // [N, K] INT8 output
    int K)
{
    const int n = blockIdx.x;
    const __half* row_in = W + (size_t)n * K;
    int8_t* row_out = W_int8 + (size_t)n * K;
    float s = scales[n];
    float rcp = (s > 0.0f) ? (1.0f / s) : 0.0f;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __half2float(row_in[k]) * rcp;
        int q = __float2int_rn(v);  // round to nearest
        q = max(-127, min(127, q));
        row_out[k] = (int8_t)q;
    }
}

void quantize_fp16_to_int8(const __half* src_fp16, Int8Linear& dst,
                           int out_features, int in_features,
                           cudaStream_t stream) {
    dst.in_features = in_features;
    dst.out_features = out_features;

    size_t w_bytes = (size_t)out_features * in_features * sizeof(int8_t);
    size_t s_bytes = out_features * sizeof(float);
    cudaMalloc(&dst.weight, w_bytes);
    cudaMalloc(&dst.scales, s_bytes);

    int block = 256;
    find_absmax_kernel<<<out_features, block, 0, stream>>>(
        src_fp16, dst.scales, in_features);
    quantize_to_int8_kernel<<<out_features, block, 0, stream>>>(
        src_fp16, dst.scales, dst.weight, in_features);
    cudaStreamSynchronize(stream);
}

// ============================================================================
// INT8 → GPTQ INT4 re-quantization (per-group symmetric)
//
// Converts INT8 per-channel weights to GPTQ INT4 per-group format.
// Input:  int8_weight[N, K] row-major, float scales[N] per-channel
// Output: qweight[K/8, N] packed uint32, scales[K/gs, N] FP16
//
// Algorithm per group of gs=128 elements along K for each output channel n:
//   max_abs = max(|int8_weight[n, g*128 .. g*128+127]|)
//   group_scale = int8_scale[n] * max_abs / 7.0
//   nibble[k] = clamp(round(int8_val * 7.0 / max(1, max_abs)) + 8, 0, 15)
// ============================================================================

// One block per (group, n) — 128 threads handle 128 K-elements.
// Fused: find max_abs, quantize, pack 16 uint32s, write scale.
__global__ void int8_to_gptq_int4_kernel(
    const int8_t* __restrict__ in_weight,  // [N, K] row-major
    const float*  __restrict__ in_scales,  // [N]
    uint32_t*     __restrict__ out_qw,     // [K/8, N]
    __half*       __restrict__ out_scales,  // [num_groups, N]
    int N, int K, int group_size)
{
    // blockIdx.x = group index, blockIdx.y = n (output channel)
    int g = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;  // 0..127

    if (n >= N) return;

    int k_base = g * group_size;
    int k = k_base + tid;

    // Load one INT8 value per thread
    int val = (k < K) ? (int)in_weight[(size_t)n * K + k] : 0;
    int abs_val = (val >= 0) ? val : -val;

    // Warp reduction for max_abs (4 warps × 32 threads = 128)
    __shared__ int smem_max[4];
    unsigned mask = 0xFFFFFFFF;
    int warp_max = abs_val;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        warp_max = max(warp_max, __shfl_xor_sync(mask, warp_max, offset));

    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) smem_max[warp_id] = warp_max;
    __syncthreads();

    int max_abs;
    if (tid < 4) {
        max_abs = smem_max[tid];
    } else {
        max_abs = 0;
    }
    if (tid < 4) {
        #pragma unroll
        for (int offset = 2; offset >= 1; offset >>= 1)
            max_abs = max(max_abs, __shfl_xor_sync(0xF, max_abs, offset));
        smem_max[0] = max_abs;
    }
    __syncthreads();
    max_abs = smem_max[0];

    // Compute and write group scale (one thread)
    if (tid == 0) {
        float scale = in_scales[n] * (float)max(1, max_abs) / 7.0f;
        int num_groups = K / group_size;
        out_scales[g * N + n] = __float2half(scale);
    }

    // Quantize: nibble = clamp(round(val * 7 / max(1, max_abs)) + 8, 0, 15)
    float rcp = (max_abs > 0) ? (7.0f / (float)max_abs) : 0.0f;
    int nibble = __float2int_rn((float)val * rcp) + 8;
    nibble = max(0, min(15, nibble));

    // Pack: 8 threads with consecutive k values contribute to one uint32
    int pack_group = tid / 8;   // 0..15 (16 uint32s per group)
    int pack_lane  = tid % 8;   // 0..7

    // Shared memory gather for packing (works across warps)
    __shared__ int smem_nibbles[128];
    smem_nibbles[tid] = nibble;
    __syncthreads();

    if (pack_lane == 0) {
        uint32_t packed = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            packed |= ((uint32_t)smem_nibbles[pack_group * 8 + b] << (b * 4));
        }
        // Write packed uint32: qweight[(k_base/8 + pack_group) * N + n]
        int qw_row = k_base / 8 + pack_group;
        out_qw[(size_t)qw_row * N + n] = packed;
    }
}

void quantize_int8_to_gptq_int4(const Int8Linear& src, GptqWeight& dst,
                                int group_size, cudaStream_t stream) {
    int N = src.out_features;
    int K = src.in_features;
    int num_groups = K / group_size;

    dst.K = K;
    dst.N = N;

    size_t qw_bytes = (size_t)(K / 8) * N * sizeof(uint32_t);
    size_t sc_bytes = (size_t)num_groups * N * sizeof(__half);
    uint32_t* qw_ptr = nullptr;
    __half*   sc_ptr = nullptr;
    cudaMalloc(&qw_ptr, qw_bytes);
    cudaMalloc(&sc_ptr, sc_bytes);
    cudaMemsetAsync(qw_ptr, 0, qw_bytes, stream);

    dst.qweight = qw_ptr;
    dst.scales  = sc_ptr;

    dim3 grid(num_groups, N);
    int8_to_gptq_int4_kernel<<<grid, group_size, 0, stream>>>(
        src.weight, src.scales, qw_ptr, sc_ptr, N, K, group_size);
}

} // namespace deusridet
