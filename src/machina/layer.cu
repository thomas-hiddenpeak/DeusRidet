/**
 * @file src/machina/layer.cu
 * @philosophical_role
 *   Per-layer primitives — RMSNorm, embedding lookup, element-wise fusion, sampling. The small verbs Machina stitches together to form a forward pass. Part 1 of the layer operator set.
 * @serves
 *   Machina forward.cu stages; Conscientia per-tick decode; Actus probes (test_forward) via public headers.
 */
// layer.cu — CUDA kernels for Qwen3.5 layer operations (Part 1)
//
// Implements: RMSNorm, Gated RMSNorm, embedding lookup, element-wise ops,
//             cuBLAS linear forward, greedy sampling.
// Target: SM87 (Jetson AGX Orin)

#include "layer.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <cfloat>
#include <vector>

namespace deusridet {

// ============================================================================
// cuBLAS singleton
// ============================================================================

static cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t get_cublas_handle() {
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
        cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    }
    return g_cublas_handle;
}

// ============================================================================
// RMSNorm kernel — register-cached two-pass
//
// Each block handles one row. Two-pass:
//   Pass 1: load x → float registers + accumulate sum_sq (one global read)
//   Pass 2: normalize from registers + weight (no re-read of x)
// For dim=5120, 256 threads: 20 elements/thread, 20 float regs = 80 bytes.
// ============================================================================

__global__ void rms_norm_kernel(const __half* __restrict__ x,
                                const __half* __restrict__ weight,
                                __half* __restrict__ out,
                                int dim, float eps) {
    const int row = blockIdx.x;
    const __half* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    const int elems_per_thread = (dim + blockDim.x - 1) / blockDim.x;

    // Register cache (max 40 elements per thread for dim up to 10240 at 256 threads)
    float cache[40];

    // Pass 1: load x into registers, accumulate sum of squares
    float sum_sq = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int i = threadIdx.x + e * blockDim.x;
        if (i < dim) {
            float v = __half2float(x_row[i]);
            cache[e] = v;
            sum_sq += v * v;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[8];  // max 256 threads = 8 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    }
    __syncthreads();

    // Pass 2: normalize from register cache (no global re-read of x)
    for (int e = 0; e < elems_per_thread; e++) {
        int i = threadIdx.x + e * blockDim.x;
        if (i < dim) {
            out_row[i] = __float2half(cache[e] * inv_rms * __half2float(weight[i]));
        }
    }
}

void rms_norm(const __half* x, const __half* weight, __half* out,
              int rows, int dim, float eps, cudaStream_t stream) {
    int threads = (dim < 256) ? 128 : 256;
    rms_norm_kernel<<<rows, threads, 0, stream>>>(x, weight, out, dim, eps);
}

// ============================================================================
// Fused Residual + RMSNorm — register-cached
//
// residual += x; out = norm(residual) * weight
// Pass 1: read residual + x, add, store in float registers, accumulate sum_sq
// Pass 2: write residual + normalized output from registers (no global re-read)
// Saves 128 global reads of residual per forward pass vs non-cached version.
// ============================================================================

__global__ void residual_rms_norm_kernel(
    __half* __restrict__ residual,          // [rows, dim] — updated in-place
    const __half* __restrict__ x,           // [rows, dim] — added to residual
    const __half* __restrict__ weight,      // [dim] — norm weight (precomputed 1+w)
    __half* __restrict__ out,               // [rows, dim] — normalized output
    int dim, float eps)
{
    const int row = blockIdx.x;
    __half* res_row = residual + row * dim;
    const __half* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    const int elems_per_thread = (dim + blockDim.x - 1) / blockDim.x;
    float cache[40];  // register cache (max dim=10240 @ 256 threads)

    // Pass 1: residual += x, cache in registers, accumulate sum_sq
    float sum_sq = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int i = threadIdx.x + e * blockDim.x;
        if (i < dim) {
            float r = __half2float(res_row[i]) + __half2float(x_row[i]);
            cache[e] = r;
            sum_sq += r * r;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    __syncthreads();

    // Pass 2: write residual + normalized output from register cache
    for (int e = 0; e < elems_per_thread; e++) {
        int i = threadIdx.x + e * blockDim.x;
        if (i < dim) {
            float r = cache[e];
            res_row[i] = __float2half(r);
            out_row[i] = __float2half(r * inv_rms * __half2float(weight[i]));
        }
    }
}

void residual_rms_norm(__half* residual, const __half* x,
                       const __half* weight, __half* out,
                       int rows, int dim, float eps, cudaStream_t stream) {
    int threads = (dim < 256) ? 128 : 256;
    residual_rms_norm_kernel<<<rows, threads, 0, stream>>>(
        residual, x, weight, out, dim, eps);
}

// ============================================================================
// Gated RMSNorm: out = weight * norm(x) * silu(gate)
// For DeltaNet: applied per-head (dim=128, many rows)
// ============================================================================

__global__ void rms_norm_gated_kernel(const __half* __restrict__ x,
                                      const __half* __restrict__ gate,
                                      const float* __restrict__ weight,
                                      __half* __restrict__ out,
                                      int dim, float eps) {
    const int row = blockIdx.x;
    const __half* x_row = x + row * dim;
    const __half* g_row = gate + row * dim;
    __half* out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    // Warp reduction (dim=128, one warp is enough if threads<=128)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float warp_sums[4];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normed = __half2float(x_row[i]) * inv_rms;
        float w = weight[i];
        float g = __half2float(g_row[i]);
        float silu_g = g / (1.0f + __expf(-g));
        out_row[i] = __float2half(w * normed * silu_g);
    }
}

void rms_norm_gated(const __half* x, const __half* gate,
                    const float* weight, __half* out,
                    int n, int dim, float eps, cudaStream_t stream) {
    int threads = (dim <= 128) ? 128 : 256;
    rms_norm_gated_kernel<<<n, threads, 0, stream>>>(x, gate, weight, out, dim, eps);
}

// ============================================================================
// Embedding lookup
// ============================================================================

__global__ void embedding_kernel(const __half* __restrict__ table,
                                 const int* __restrict__ ids,
                                 __half* __restrict__ out,
                                 int hidden_size) {
    const int token = blockIdx.x;
    const int id = ids[token];
    const __half* row = table + (long long)id * hidden_size;
    __half* dst = out + (long long)token * hidden_size;

    // Vectorized copy: float4 = 8 halves per thread per step
    const int vec_dim = hidden_size / 8;
    const float4* src4 = reinterpret_cast<const float4*>(row);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int i = threadIdx.x; i < vec_dim; i += blockDim.x)
        dst4[i] = src4[i];
    // Scalar remainder
    for (int i = vec_dim * 8 + threadIdx.x; i < hidden_size; i += blockDim.x)
        dst[i] = row[i];
}

void embedding_lookup(const __half* table, const int* ids, __half* out,
                      int seq_len, int hidden_size, cudaStream_t stream) {
    embedding_kernel<<<seq_len, 256, 0, stream>>>(table, ids, out, hidden_size);
}

// ============================================================================
// Element-wise operations
// ============================================================================

__global__ void silu_kernel(__half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        x[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

void silu_inplace(__half* x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

// ============================================================================
// Fused SiLU + elementwise multiply: out = silu(gate) * up
//
// Saves one kernel launch + one read/write round-trip on intermediate_size.
// Used once per MLP (64 calls per forward → 64 launches saved).
// ============================================================================

__global__ void silu_mul_kernel(const __half* __restrict__ gate,
                                const __half* __restrict__ up,
                                __half* __restrict__ out, int n) {
    // Vectorized: process 8 halves per thread via float4
    const int vec_n = n / 8;
    const float4* g4 = reinterpret_cast<const float4*>(gate);
    const float4* u4 = reinterpret_cast<const float4*>(up);
    float4* o4 = reinterpret_cast<float4*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        float4 gv = g4[idx];
        float4 uv = u4[idx];
        const __half2* gp = reinterpret_cast<const __half2*>(&gv);
        const __half2* up2 = reinterpret_cast<const __half2*>(&uv);
        __half2 result[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float g0 = __half2float(gp[i].x), g1 = __half2float(gp[i].y);
            float u0 = __half2float(up2[i].x), u1 = __half2float(up2[i].y);
            float s0 = (g0 / (1.0f + __expf(-g0))) * u0;
            float s1 = (g1 / (1.0f + __expf(-g1))) * u1;
            result[i] = __halves2half2(__float2half(s0), __float2half(s1));
        }
        o4[idx] = *reinterpret_cast<float4*>(result);
    }
    // Scalar tail for n not divisible by 8
    if (idx == 0) {
        for (int i = vec_n * 8; i < n; i++) {
            float g = __half2float(gate[i]);
            float u = __half2float(up[i]);
            out[i] = __float2half((g / (1.0f + __expf(-g))) * u);
        }
    }
}

void silu_mul(const __half* gate, const __half* up, __half* out,
              int n, cudaStream_t stream) {
    int vec_n = n / 8;
    int threads = 256;
    int blocks = (vec_n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

__global__ void mul_kernel(const __half* __restrict__ a,
                           const __half* __restrict__ b,
                           __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
    }
}

void elementwise_mul(const __half* a, const __half* b, __half* out,
                     int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void add_kernel(const __half* __restrict__ a,
                           const __half* __restrict__ b,
                           __half* __restrict__ out, int n) {
    // Vectorized float4 path: process 8 halves per thread (4 half2 in a float4)
    const int vec_n = n / 8;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* o4 = reinterpret_cast<float4*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        float4 av = a4[idx];
        float4 bv = b4[idx];
        const __half2* ap = reinterpret_cast<const __half2*>(&av);
        const __half2* bp = reinterpret_cast<const __half2*>(&bv);
        __half2 result[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            result[i] = __hadd2(ap[i], bp[i]);
        }
        o4[idx] = *reinterpret_cast<float4*>(result);
    }

    // Scalar tail for remaining elements
    const int tail_start = vec_n * 8;
    const int tail_idx = tail_start + (blockIdx.x * blockDim.x + threadIdx.x) - vec_n * blockDim.x;
    // Only last block handles tail
    if (idx >= vec_n && tail_idx < n) {
        out[tail_idx] = __hadd(a[tail_idx], b[tail_idx]);
    }
}

void elementwise_add(const __half* a, const __half* b, __half* out,
                     int n, cudaStream_t stream) {
    int threads = 256;
    int vec_n = n / 8;
    int blocks = (vec_n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void sigmoid_gate_kernel(const __half* __restrict__ x,
                                    const __half* __restrict__ gate,
                                    __half* __restrict__ out, int n) {
    const int vec_n = n / 8;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* g4 = reinterpret_cast<const float4*>(gate);
    float4* o4 = reinterpret_cast<float4*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        float4 xv = x4[idx];
        float4 gv = g4[idx];
        const __half2* xp = reinterpret_cast<const __half2*>(&xv);
        const __half2* gp = reinterpret_cast<const __half2*>(&gv);
        __half2 result[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float v0 = __half2float(xp[i].x), v1 = __half2float(xp[i].y);
            float g0 = __half2float(gp[i].x), g1 = __half2float(gp[i].y);
            result[i] = __halves2half2(
                __float2half(v0 / (1.0f + __expf(-g0))),
                __float2half(v1 / (1.0f + __expf(-g1))));
        }
        o4[idx] = *reinterpret_cast<float4*>(result);
    }
}

void sigmoid_gate(const __half* x, const __half* gate, __half* out,
                  int n, cudaStream_t stream) {
    int vec_n = n / 8;
    int threads = 256;
    int blocks = (vec_n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    sigmoid_gate_kernel<<<blocks, threads, 0, stream>>>(x, gate, out, n);
}

// ============================================================================
// FP16 GEMV kernel — y[N] = W[N,K] @ x[K]
//
// For M=1 decode: weight matrix W is [N,K] row-major, each output row
// is one dot product. This kernel reads W row-by-row with coalesced
// access along K, much faster than cuBLAS OP_T transpose for M=1.
//
// Strategy: same as GPTQ GEMV — tile N in blocks, split K across threads.
// ============================================================================

// FP16 GEMV: y[N] = W[N,K] @ x[K], W is [N,K] row-major
//
// Design: 1 warp per output row. 32 lanes iterate along K cooperatively,
// reading consecutive FP16 values within each row (coalesced). Float4
// vectorized loads (8 halves per thread per step) for maximum bandwidth.
// Final reduction via warp shuffle.
//
// Memory access: w_row[lane_id*8..lane_id*8+7] — adjacent lanes read
// adjacent 16-byte chunks from the same row. Perfectly coalesced.
//
constexpr int FP16_GEMV_WARPS = 4;   // output rows per block
constexpr int FP16_GEMV_BLOCK = FP16_GEMV_WARPS * 32;  // 128 threads

__global__ void fp16_gemv_kernel(
    const __half* __restrict__ x,   // [K]
    const __half* __restrict__ W,   // [N, K] row-major
    __half* __restrict__ y,         // [N]
    int K, int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n = blockIdx.x * FP16_GEMV_WARPS + warp_id;
    if (n >= N) return;

    const __half* w_row = W + (size_t)n * K;
    float acc = 0.0f;

    // Vectorized: 8 halves (float4 = 16 bytes) per lane per step
    // Warp stride: 32 lanes × 8 halves = 256 halves per iteration
    const int vec_end = (K / 256) * 256;
    for (int k = lane_id * 8; k < vec_end; k += 256) {
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        float4 xv = *reinterpret_cast<const float4*>(x + k);
        const __half2* wp = reinterpret_cast<const __half2*>(&wv);
        const __half2* xp = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc += __half2float(wp[i].x) * __half2float(xp[i].x)
                 + __half2float(wp[i].y) * __half2float(xp[i].y);
        }
    }

    // Scalar remainder (handles K not divisible by 256)
    for (int k = vec_end + lane_id; k < K; k += 32) {
        acc += __half2float(w_row[k]) * __half2float(x[k]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[n] = __float2half(acc);
    }
}

void fp16_gemv(const __half* x, const __half* W, __half* y,
                      int K, int N, cudaStream_t stream) {
    int grid = (N + FP16_GEMV_WARPS - 1) / FP16_GEMV_WARPS;
    fp16_gemv_kernel<<<grid, FP16_GEMV_BLOCK, 0, stream>>>(x, W, y, K, N);
}

// ============================================================================
// Linear forward: GEMV for M=1, cuBLAS GEMM for M>1
//
// Y[M,N] = X[M,K] @ W^T[K,N] where W is [N,K] row-major.
// For M=1: use custom FP16 GEMV (coalesced row reads — much faster).
// For M>1: cuBLAS uses column-major: Y^T = W @ X^T
// ============================================================================

void linear_forward(const __half* X, const Linear& weight, __half* Y,
                    int M, cudaStream_t stream) {
    int K = weight.in_features;
    int N = weight.out_features;

    if (M == 1) {
        fp16_gemv(X, weight.weight, Y, K, N, stream);
        return;
    }

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha_h,
                 weight.weight, CUDA_R_16F, K,
                 X,              CUDA_R_16F, K,
                 &beta_h,
                 Y,              CUDA_R_16F, N,
                 CUDA_R_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ============================================================================
// INT8 GEMV kernel — Decode path (M=1)
//
// Per-channel symmetric quantization: w_fp32 = scale[n] * (float)int8_weight[n][k]
// Same warp-per-row design as FP16 GEMV, but reads 1 byte per weight instead of 2.
// Vectorized: int4 (16 bytes = 16 int8 values per thread per step).
// Each step also reads 16 FP16 x values (2 float4 loads).
// Warp stride: 32 lanes × 16 = 512 elements per iteration.
//
// Expected ~50% less weight data read → ~50% faster for weight-dominated GEMVs.
// ============================================================================

constexpr int INT8_GEMV_WARPS = 4;
constexpr int INT8_GEMV_BLOCK = INT8_GEMV_WARPS * 32;  // 128 threads

__global__ void int8_gemv_kernel(
    const __half*  __restrict__ x,       // [K]
    const int8_t*  __restrict__ W,       // [N, K] row-major INT8
    const float*   __restrict__ scales,  // [N] per-output-channel
    __half*        __restrict__ y,       // [N]
    int K, int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n = blockIdx.x * INT8_GEMV_WARPS + warp_id;
    if (n >= N) return;

    const int8_t* w_row = W + (size_t)n * K;
    const float scale = scales[n];
    float acc = 0.0f;

    // Vectorized: 16 int8 values (float4 = 16 bytes) + 16 FP16 x values (2 float4)
    // Warp stride: 32 × 16 = 512 elements per iteration
    const int vec_end = (K / 512) * 512;
    for (int k = lane_id * 16; k < vec_end; k += 512) {
        // Load 16 INT8 weights as float4 (16 bytes)
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        // Load 16 FP16 x values as 2 float4 (32 bytes)
        float4 xv0 = *reinterpret_cast<const float4*>(x + k);
        float4 xv1 = *reinterpret_cast<const float4*>(x + k + 8);

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

    // Scalar remainder (handles K not divisible by 512)
    for (int k = vec_end + lane_id; k < K; k += 32)
        acc += (float)w_row[k] * __half2float(x[k]);

    // Apply per-channel scale
    acc *= scale;

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane_id == 0)
        y[n] = __float2half(acc);
}

static void int8_gemv(const __half* x, const int8_t* W, const float* scales,
                      __half* y, int K, int N, cudaStream_t stream) {
    int grid = (N + INT8_GEMV_WARPS - 1) / INT8_GEMV_WARPS;
    int8_gemv_kernel<<<grid, INT8_GEMV_BLOCK, 0, stream>>>(x, W, scales, y, K, N);
}

// ============================================================================
// INT8 batch GEMV — Prefill path (M>1, small M)
//
// Y[M,N] = X[M,K] @ W^T[K,N] with per-channel scales
// Loads each weight row ONCE and computes M dot products simultaneously.
// X rows are read from L2 cache (total X = M×K×2 bytes, fits in 4MB L2).
// Weight bandwidth = same as M=1 GEMV → near-11× speedup for M=11.
//
// Thread mapping: same as int8_gemv_kernel (4 warps/block, warp-per-row-n)
// Register usage: M FP32 accumulators per thread (~11 extra for M=11)
// ============================================================================

__global__ void int8_batch_gemv_kernel(
    const __half* __restrict__ X,       // [M, K] row-major FP16
    const int8_t* __restrict__ W,       // [N, K] row-major INT8
    const float*  __restrict__ scales,  // [N] per-output-channel
    __half*       __restrict__ Y,       // [M, N] row-major FP16
    int M, int K, int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int n = blockIdx.x * INT8_GEMV_WARPS + warp_id;
    if (n >= N) return;

    const int8_t* w_row = W + (size_t)n * K;
    const float scale = scales[n];

    // M accumulators — weight loaded once, used for all M rows
    float acc[128];  // max M=128 (matches max_seq allocation)
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    // Vectorized: 16 INT8 weights + M × 16 FP16 x values per step
    const int vec_end = (K / 512) * 512;
    for (int k = lane_id * 16; k < vec_end; k += 512) {
        // Load weight once (16 INT8 values)
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        const int8_t* wp = reinterpret_cast<const int8_t*>(&wv);

        // Apply to all M input rows
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

    // Scalar remainder
    for (int k = vec_end + lane_id; k < K; k += 32) {
        int8_t wval = w_row[k];
        for (int m = 0; m < M; m++)
            acc[m] += (float)wval * __half2float(X[(size_t)m * K + k]);
    }

    // Scale + warp reduction + write for each M row
    for (int m = 0; m < M; m++) {
        float val = acc[m] * scale;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

        if (lane_id == 0)
            Y[(size_t)m * N + n] = __float2half(val);
    }
}

static void int8_batch_gemv(const __half* X, const int8_t* W, const float* scales,
                             __half* Y, int M, int K, int N, cudaStream_t stream) {
    int grid = (N + INT8_GEMV_WARPS - 1) / INT8_GEMV_WARPS;
    int8_batch_gemv_kernel<<<grid, INT8_GEMV_BLOCK, 0, stream>>>(X, W, scales, Y, M, K, N);
}

// ============================================================================
// INT8 GEMM kernel — Prefill path (M>1)
//
// Y[M,N] = X[M,K] @ W^T[K,N] with per-channel scales
// W is [N,K] row-major INT8. scales[N] is FP32 per-output-channel.
//
// Tile: BM=32, BN=64, BK=128, Block=256 (mirrors GPTQ GEMM structure)
// Each thread computes TM×TN = 4×2 output elements.
// Scale deferred to output write (one multiply per output element).
//
// SMEM layout: X_tile[BM][BK] FP16 (8KB) + W_tile[BK][BN] FP16 (16KB) = 24KB
// ============================================================================

constexpr int I8_BM = 32;
constexpr int I8_BN = 64;
constexpr int I8_BK = 128;
constexpr int I8_BLOCK = 256;
constexpr int I8_TM = 4;
constexpr int I8_TN = 2;

__global__ void int8_gemm_kernel(
    const __half*  __restrict__ X,       // [M, K] row-major FP16
    const int8_t*  __restrict__ W,       // [N, K] row-major INT8
    const float*   __restrict__ scales,  // [N] per-channel FP32
    __half*        __restrict__ Y,       // [M, N] row-major FP16
    int M, int K, int N)
{
    const int bm = blockIdx.y * I8_BM;
    const int bn = blockIdx.x * I8_BN;
    const int tid = threadIdx.x;
    const int warp_m = tid / 32;   // 0..7
    const int warp_n = tid % 32;   // 0..31

    float acc[I8_TM][I8_TN];
    #pragma unroll
    for (int i = 0; i < I8_TM; i++)
        for (int j = 0; j < I8_TN; j++)
            acc[i][j] = 0.0f;

    __shared__ __half X_tile[I8_BM][I8_BK];
    __shared__ __half W_tile[I8_BK][I8_BN];

    const int num_k_tiles = (K + I8_BK - 1) / I8_BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_base = kt * I8_BK;

        // Load X tile [BM, BK] into SMEM (coalesced along K)
        {
            int total = I8_BM * I8_BK;
            for (int idx = tid; idx < total; idx += I8_BLOCK) {
                int row = idx / I8_BK;
                int col = idx % I8_BK;
                int gm = bm + row;
                int gk = k_base + col;
                X_tile[row][col] = (gm < M && gk < K)
                    ? X[gm * K + gk] : __float2half(0.0f);
            }
        }

        // Load W tile: read INT8 from W[N,K], convert to FP16, store as [BK,BN]
        // Thread mapping: k varies fastest for coalesced INT8 reads along K
        {
            int total = I8_BK * I8_BN;
            for (int idx = tid; idx < total; idx += I8_BLOCK) {
                int n_local = idx / I8_BK;
                int k_local = idx % I8_BK;
                int gn = bn + n_local;
                int gk = k_base + k_local;
                int8_t wval = (gn < N && gk < K) ? W[(size_t)gn * K + gk] : 0;
                W_tile[k_local][n_local] = __float2half((float)wval);
            }
        }

        __syncthreads();

        // Compute: accumulate X_tile @ W_tile
        #pragma unroll
        for (int k = 0; k < I8_BK; k++) {
            #pragma unroll
            for (int tm = 0; tm < I8_TM; tm++) {
                float x_val = __half2float(X_tile[warp_m * I8_TM + tm][k]);
                #pragma unroll
                for (int tn = 0; tn < I8_TN; tn++) {
                    acc[tm][tn] += x_val * __half2float(W_tile[k][warp_n * I8_TN + tn]);
                }
            }
        }

        __syncthreads();
    }

    // Write output with per-channel scale
    #pragma unroll
    for (int tm = 0; tm < I8_TM; tm++) {
        int gm = bm + warp_m * I8_TM + tm;
        #pragma unroll
        for (int tn = 0; tn < I8_TN; tn++) {
            int gn = bn + warp_n * I8_TN + tn;
            if (gm < M && gn < N) {
                Y[gm * N + gn] = __float2half(acc[tm][tn] * scales[gn]);
            }
        }
    }
}

static void int8_gemm(const __half* X, const int8_t* W, const float* scales,
                       __half* Y, int M, int K, int N, cudaStream_t stream) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    int8_gemm_kernel<<<grid, I8_BLOCK, 0, stream>>>(X, W, scales, Y, M, K, N);
}

// ============================================================================
// INT8 WMMA GEMM — Tensor core path (M>1)
// ============================================================================
//
// Uses WMMA m16n16k16 FP16 tensor cores with in-SMEM INT8→FP16 dequantization.
// W is [N, K] row-major INT8, scales [N] FP32 per-channel.
//
// Key design: W tile stored in SMEM as [BN, BK_PAD] (W's natural row-major layout,
// NOT transposed) with K-dimension padding for bank-conflict avoidance.
// WMMA uses col_major matrix_b to implicitly transpose W → W^T.
//
// Load pattern: each warp reads one W row (one N value) along K per iteration.
// 32 lanes × 4 bytes = 128 bytes = one full BK segment. Perfectly coalesced.
// 4 warps × 16 iterations = 64 rows = BN.
//
// Tile: BM=16, BN=64, BK=128.  Block: 128 threads = 4 warps.
// SMEM: X[16,136] (4.25 KB) + W[64,136] (17 KB) = 21.25 KB.
// launch_bounds(128, 4): 4 blocks/SM = 16 warps = 33% occupancy.
// The compiler spills ~11 regs to L1 (fast) to fit the 128-reg budget.
// With double-buffer prefetch, loads overlap with WMMA; inter-block
// interleaving provides additional DRAM latency hiding.
//
// Requirements: K % 128 == 0, N % 64 == 0.

constexpr int I8TC_BM = 16;
constexpr int I8TC_BN = 64;
constexpr int I8TC_BK = 128;
constexpr int I8TC_BK_PAD = I8TC_BK + 8;  // 136: pad K stride for SMEM bank avoidance
constexpr int I8TC_BLOCK = 128;  // 4 warps

__global__ void __launch_bounds__(I8TC_BLOCK, 4)
int8_wmma_gemm_kernel(
    const __half*  __restrict__ X,       // [M_pad, K] row-major FP16
    const int8_t*  __restrict__ W,       // [N, K] row-major INT8
    const float*   __restrict__ scales,  // [N] per-channel FP32
    __half*        __restrict__ Y,       // [M_pad, N] row-major FP16
    int K, int N)
{
    using namespace nvcuda;

    const int bn = blockIdx.x * I8TC_BN;
    const int bm = blockIdx.y * I8TC_BM;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // SMEM: X as [BM, BK_PAD] row-major, W as [BN, BK_PAD] (K-inner)
    __shared__ __half smem_x[I8TC_BM * I8TC_BK_PAD];        // 16×136 = 4.25 KB
    __shared__ __half smem_w[I8TC_BN * I8TC_BK_PAD];         // 64×136 = 17 KB

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int num_k_tiles = K / I8TC_BK;
    const int x_row = tid / 8, x_col = (tid & 7) * 16;

    // Pre-compute per-channel scales: constant across all K-tiles (hoist from loop)
    // Each warp processes 16 rows of W (4 groups × 4 rows)
    half2 cached_s2[16];
    #pragma unroll
    for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int n_local = warp_id + (grp * 4 + i) * 4;
            const int n_global = bn + n_local;
            cached_s2[grp * 4 + i] = __half2half2(__float2half(scales[n_global]));
        }
    }

    // === Register-level double-buffer: prefetch next tile during WMMA ===
    float4 cur_x0, cur_x1, nxt_x0, nxt_x1;
    uint32_t cur_pk[16], nxt_pk[16];

    // Pre-load tile 0 into 'cur' registers
    {
        const __half* src = X + (size_t)(bm + x_row) * K + x_col;
        cur_x0 = *(const float4*)(src);
        cur_x1 = *(const float4*)(src + 8);
    }
    #pragma unroll
    for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int n_local = warp_id + (grp * 4 + i) * 4;
            const int n_global = bn + n_local;
            cur_pk[grp * 4 + i] = *reinterpret_cast<const uint32_t*>(
                W + (size_t)n_global * K + lane_id * 4);
        }
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // === Phase 1: write current tile to SMEM (dequant from registers) ===

        // Write X to SMEM
        {
            __half* dst = smem_x + x_row * I8TC_BK_PAD + x_col;
            *(float4*)(dst)     = cur_x0;
            *(float4*)(dst + 8) = cur_x1;
        }

        // Dequant W: registers → FP16 → SMEM (using cached scales)
        #pragma unroll
        for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int idx = grp * 4 + i;
                const int n_local = warp_id + (grp * 4 + i) * 4;
                const half2 s2 = cached_s2[idx];
                uint32_t pk = cur_pk[idx];
                half2 v01 = __hmul2(s2, __halves2half2(
                    __int2half_rn((int)(int8_t)(pk & 0xFF)),
                    __int2half_rn((int)(int8_t)((pk >> 8) & 0xFF))));
                half2 v23 = __hmul2(s2, __halves2half2(
                    __int2half_rn((int)(int8_t)((pk >> 16) & 0xFF)),
                    __int2half_rn((int)(int8_t)((pk >> 24) & 0xFF))));
                __half* dst = smem_w + n_local * I8TC_BK_PAD + lane_id * 4;
                *(half2*)(dst)     = v01;
                *(half2*)(dst + 2) = v23;
            }
        }

        __syncthreads();

        // === Phase 2: WMMA compute + prefetch next tile ===
        // Issue DRAM loads for tile kt+1 BEFORE WMMA so loads overlap with tensor core
        if (kt + 1 < num_k_tiles) {
            const int k_next = (kt + 1) * I8TC_BK;
            {
                const __half* src = X + (size_t)(bm + x_row) * K + k_next + x_col;
                nxt_x0 = *(const float4*)(src);
                nxt_x1 = *(const float4*)(src + 8);
            }
            #pragma unroll
            for (int grp = 0; grp < I8TC_BN / 4 / 4; grp++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int n_local = warp_id + (grp * 4 + i) * 4;
                    const int n_global = bn + n_local;
                    nxt_pk[grp * 4 + i] = *reinterpret_cast<const uint32_t*>(
                        W + (size_t)n_global * K + k_next + lane_id * 4);
                }
            }
        }

        // WMMA compute: 8 K-chunks (reads SMEM, concurrent with DRAM prefetch)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            #pragma unroll
            for (int wk = 0; wk < I8TC_BK / 16; wk++) {
                wmma::load_matrix_sync(a_frag, smem_x + wk * 16, I8TC_BK_PAD);
                wmma::load_matrix_sync(b_frag,
                    smem_w + warp_id * 16 * I8TC_BK_PAD + wk * 16, I8TC_BK_PAD);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }

        __syncthreads();

        // Swap: nxt → cur
        cur_x0 = nxt_x0; cur_x1 = nxt_x1;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            cur_pk[i] = nxt_pk[i];
    }

    // === Store: FP32 → FP16 ===
    const int warp_col = bn + warp_id * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_h;
    for (int i = 0; i < acc.num_elements; i++) {
        acc_h.x[i] = __float2half(acc.x[i]);
    }
    wmma::store_matrix_sync(Y + (size_t)bm * N + warp_col, acc_h, N, wmma::mem_row_major);
}

static void int8_wmma_gemm(const __half* X, const int8_t* W, const float* scales,
                            __half* Y, int M, int K, int N, cudaStream_t stream) {
    int M_pad = (M + I8TC_BM - 1) / I8TC_BM * I8TC_BM;
    dim3 grid(N / I8TC_BN, M_pad / I8TC_BM);
    int8_wmma_gemm_kernel<<<grid, I8TC_BLOCK, 0, stream>>>(X, W, scales, Y, K, N);
}

void int8_linear_forward(const __half* X, const Int8Linear& weight, __half* Y,
                         int M, cudaStream_t stream) {
    if (M == 1) {
        int8_gemv(X, weight.weight, weight.scales, Y,
                  weight.in_features, weight.out_features, stream);
        return;
    }
    int K = weight.in_features;
    int N = weight.out_features;
    if (K % I8TC_BK == 0 && N % I8TC_BN == 0) {
        // Tensor core WMMA path: dequant INT8 in SMEM + m16n16k16 FP16 mma.
        int8_wmma_gemm(X, weight.weight, weight.scales, Y,
                       M, K, N, stream);
    } else {
        // Fallback for non-aligned dimensions (e.g. in_proj_a/b with N=48)
        int8_batch_gemv(X, weight.weight, weight.scales, Y,
                        M, K, N, stream);
    }
}

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

// ============================================================================
// Partial RoPE (half-half pairing, single-token)
//
// Qwen3.5 uses partial_rotary_factor=0.25, so only first 64 of 256 dims
// are rotated. Uses rotate_half (NOT interleaved) pairing:
//   pairs are (i, i + rotary_dim/2) for i = 0..rotary_dim/2-1
//   i.e. (0,32), (1,33), ..., (31,63)
// freq_i = 1.0 / (theta^(2i/rotary_dim))
// ============================================================================

__global__ void rope_kernel(__half* __restrict__ q,
                            __half* __restrict__ k,
                            int num_q_heads, int num_kv_heads,
                            int head_dim, int rotary_dim,
                            const int* __restrict__ d_pos, float theta) {
    // Each thread handles one pair in one head
    int head = blockIdx.x;
    int pair = threadIdx.x;  // pair index within rotary_dim/2
    int num_pairs = rotary_dim / 2;
    if (pair >= num_pairs) return;

    int pos = *d_pos;
    float freq = 1.0f / powf(theta, (2.0f * pair) / (float)rotary_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Half-half pairing: pair (pair, pair + num_pairs) within each head
    // Apply to Q head
    if (head < num_q_heads) {
        int idx0 = head * head_dim + pair;
        int idx1 = head * head_dim + pair + num_pairs;
        float x0 = __half2float(q[idx0]);
        float x1 = __half2float(q[idx1]);
        q[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
        q[idx1] = __float2half(x1 * cos_a + x0 * sin_a);
    }

    // Apply to K head (only first num_kv_heads)
    if (head < num_kv_heads) {
        int idx0 = head * head_dim + pair;
        int idx1 = head * head_dim + pair + num_pairs;
        float x0 = __half2float(k[idx0]);
        float x1 = __half2float(k[idx1]);
        k[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
        k[idx1] = __float2half(x1 * cos_a + x0 * sin_a);
    }
}

void apply_rope(__half* q, __half* k,
                int num_heads, int num_kv_heads, int head_dim,
                int rotary_dim, const int* d_pos, float theta,
                cudaStream_t stream) {
    int num_pairs = rotary_dim / 2;
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    rope_kernel<<<max_heads, num_pairs, 0, stream>>>(
        q, k, num_heads, num_kv_heads, head_dim, rotary_dim, d_pos, theta);
}

// ============================================================================
// Per-head RMSNorm (for Q/K norms in full attention)
// Each block handles one head of head_dim elements.
// ============================================================================

__global__ void head_norm_kernel(const __half* __restrict__ x,
                                 const __half* __restrict__ weight,
                                 __half* __restrict__ out,
                                 int head_dim, float eps) {
    int head = blockIdx.x;
    const __half* x_head = x + head * head_dim;
    __half* out_head = out + head * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(x_head[i]);
        sum_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(x_head[i]) * inv_rms;
        out_head[i] = __float2half(v * __half2float(weight[i]));
    }
}

void head_norm(const __half* x, const __half* weight, __half* out,
               int num_heads, int head_dim, float eps,
               cudaStream_t stream) {
    int threads = (head_dim <= 128) ? 128 : 256;
    head_norm_kernel<<<num_heads, threads, 0, stream>>>(
        x, weight, out, head_dim, eps);
}

// ============================================================================
// Row-wise softmax (FP32)
// One block per row. First compute max, then exp(x-max), then normalize.
// ============================================================================

__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               int cols) {
    int row = blockIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    // Find max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x_row[i];
        if (v > max_val) max_val = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        if (other > max_val) max_val = other;
    }
    __shared__ float warp_max[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x / 32)) ? warp_max[lane_id] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            if (other > max_val) max_val = other;
        }
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x_row[i] - s_max);
        out_row[i] = e;
        sum_exp += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    __shared__ float warp_sum[8];
    if (lane_id == 0) warp_sum[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane_id < (blockDim.x / 32)) ? warp_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        }
    }
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = 1.0f / sum_exp;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] *= s_inv;
    }
}

void softmax(const float* x, float* out, int rows, int cols,
             cudaStream_t stream) {
    int threads = (cols <= 128) ? 128 : 256;
    softmax_kernel<<<rows, threads, 0, stream>>>(x, out, cols);
}

// ============================================================================
// Causal conv1d step — register-optimized (kernel_size=4)
//
// conv_state[conv_dim, kernel-1] stores the last (kernel-1) inputs per channel.
// On each step: load state and weight to registers, compute weighted sum,
// shift state, insert new input. All memory ops coalesced.
//
// Optimized: preload conv_state[3] and conv_weight[4] to registers.
// Conv computation and state update entirely in registers, then one write-back.
// ============================================================================

__global__ void conv1d_step_kernel(const __half* __restrict__ x_in,
                                   __half* __restrict__ conv_state,
                                   const __half* __restrict__ conv_weight,
                                   __half* __restrict__ x_out,
                                   int conv_dim, int kernel_size) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int km1 = kernel_size - 1;  // 3

    // Load state and weight to registers (eliminates repeated global reads)
    float st[3], wt[4];
    #pragma unroll
    for (int j = 0; j < 3; j++)
        st[j] = __half2float(conv_state[ch * km1 + j]);
    #pragma unroll
    for (int j = 0; j < 4; j++)
        wt[j] = __half2float(conv_weight[ch * kernel_size + j]);

    float x_val = __half2float(x_in[ch]);

    // Compute: dot product of [state[0], state[1], state[2], x_in] with weights
    float acc = st[0] * wt[0] + st[1] * wt[1] + st[2] * wt[2] + x_val * wt[3];
    x_out[ch] = __float2half(acc);

    // Update state: shift left, insert new input (write once)
    conv_state[ch * km1 + 0] = __float2half(st[1]);
    conv_state[ch * km1 + 1] = __float2half(st[2]);
    conv_state[ch * km1 + 2] = x_in[ch];
}

void causal_conv1d_step(const __half* x_in, __half* conv_state,
                        const __half* conv_weight, __half* x_out,
                        int conv_dim, int kernel_size,
                        cudaStream_t stream) {
    int threads = 256;
    int blocks = (conv_dim + threads - 1) / threads;
    conv1d_step_kernel<<<blocks, threads, 0, stream>>>(
        x_in, conv_state, conv_weight, x_out, conv_dim, kernel_size);
}

// ============================================================================
// Fused Conv1d step + SiLU: conv → silu in registers (no intermediate write)
//
// Replaces: causal_conv1d_step + silu_inplace (2 kernels, 1 round-trip)
// Saves: conv_dim × 2 bytes write + conv_dim × 2 bytes read = 40KB for 10240.
// ============================================================================

__global__ void conv1d_step_silu_kernel(const __half* __restrict__ x_in,
                                        __half* __restrict__ conv_state,
                                        const __half* __restrict__ conv_weight,
                                        __half* __restrict__ x_out,
                                        int conv_dim, int kernel_size) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int km1 = kernel_size - 1;

    float st[3], wt[4];
    #pragma unroll
    for (int j = 0; j < 3; j++)
        st[j] = __half2float(conv_state[ch * km1 + j]);
    #pragma unroll
    for (int j = 0; j < 4; j++)
        wt[j] = __half2float(conv_weight[ch * kernel_size + j]);

    float x_val = __half2float(x_in[ch]);
    float acc = st[0] * wt[0] + st[1] * wt[1] + st[2] * wt[2] + x_val * wt[3];

    // Fused SiLU: silu(x) = x / (1 + exp(-x))
    float silu_out = acc / (1.0f + __expf(-acc));
    x_out[ch] = __float2half(silu_out);

    conv_state[ch * km1 + 0] = __float2half(st[1]);
    conv_state[ch * km1 + 1] = __float2half(st[2]);
    conv_state[ch * km1 + 2] = x_in[ch];
}

void causal_conv1d_step_silu(const __half* x_in, __half* conv_state,
                              const __half* conv_weight, __half* x_out,
                              int conv_dim, int kernel_size,
                              cudaStream_t stream) {
    int threads = 256;
    int blocks = (conv_dim + threads - 1) / threads;
    conv1d_step_silu_kernel<<<blocks, threads, 0, stream>>>(
        x_in, conv_state, conv_weight, x_out, conv_dim, kernel_size);
}

// ============================================================================
// Greedy sampling: argmax on CPU
// ============================================================================

// GPU argmax kernel for greedy sampling
// Single block, 1024 threads — each thread scans vocab_size/1024 elements,
// then shared memory reduction to find global max.
// Result written to dev_result[0].
__global__ void argmax_kernel(const __half* __restrict__ logits, int* result,
                              int vocab_size) {
    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    int tid = threadIdx.x;
    float best_val = -1e30f;
    int   best_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) result[0] = s_idx[0];
}

void argmax_async(const __half* logits, int vocab_size, int* d_token_out,
                  cudaStream_t stream) {
    argmax_kernel<<<1, 1024, 0, stream>>>(logits, d_token_out, vocab_size);
}

int greedy_sample(const __half* logits, int vocab_size, int* d_token_out,
                  cudaStream_t stream) {
    argmax_async(logits, vocab_size, d_token_out, stream);
    int result;
    cudaMemcpyAsync(&result, d_token_out, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return result;
}

// ============================================================================
// Top-k / Top-p sampling (GPU, single block)
//
// Design inspired by FlashInfer's rejection sampling approach:
// - Binary search threshold for combined top-k + top-p filtering
// - Multinomial sample from filtered distribution
// - Runs in a single kernel launch with 3 barrier-separated phases
//
// Phase 1: Online softmax with temperature (find max, compute exp/sum)
// Phase 2: Binary search for probability threshold T such that
//          count(prob >= T) <= top_k AND sum(prob >= T) >= top_p
// Phase 3: Gather candidates above T, multinomial sample
// ============================================================================

__global__ void top_k_top_p_sample_kernel(
    const __half* __restrict__ logits,
    float* __restrict__ probs_workspace,    // [vocab_size] FP32
    int vocab_size,
    float temperature, int top_k, float top_p,
    unsigned long long rng_seed,
    int* __restrict__ token_out)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // 1024

    // Shared memory for warp reductions
    __shared__ float s_warp[32];    // max 32 warps
    __shared__ float s_global;
    __shared__ float s_global2;

    // --- Phase 1: Temperature + online softmax ---
    // Pass 1a: find global max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float v = __half2float(logits[i]) / temperature;
        probs_workspace[i] = v;  // store scaled logits
        local_max = fmaxf(local_max, v);
    }
    // Warp reduction for max
    for (int o = 16; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, o));
    if (tid % 32 == 0) s_warp[tid / 32] = local_max;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < nthreads / 32) ? s_warp[tid] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, o));
        if (tid == 0) s_global = v;
    }
    __syncthreads();
    float global_max = s_global;

    // Pass 1b: compute exp(logit - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float e = expf(probs_workspace[i] - global_max);
        probs_workspace[i] = e;
        local_sum += e;
    }
    // Warp + block reduction for sum
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, o);
    if (tid % 32 == 0) s_warp[tid / 32] = local_sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < nthreads / 32) ? s_warp[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, o);
        if (tid == 0) s_global = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_global;

    // Normalize to probabilities
    for (int i = tid; i < vocab_size; i += nthreads)
        probs_workspace[i] *= inv_sum;
    __syncthreads();

    // --- Phase 2: Binary search for threshold ---
    // Find threshold T such that:
    //   count(prob >= T) <= top_k AND sum(prob >= T) >= top_p
    float lo = 0.0f, hi = 1.0f;
    for (int iter = 0; iter < 32; iter++) {
        float mid = (lo + hi) * 0.5f;

        // Count elements >= mid and sum their probabilities
        int local_count = 0;
        float local_psum = 0.0f;
        for (int i = tid; i < vocab_size; i += nthreads) {
            float p = probs_workspace[i];
            if (p >= mid) {
                local_count++;
                local_psum += p;
            }
        }

        // Reduce count
        for (int o = 16; o > 0; o >>= 1)
            local_count += __shfl_xor_sync(0xFFFFFFFF, local_count, o);
        if (tid % 32 == 0) s_warp[tid / 32] = __int_as_float(local_count);
        __syncthreads();
        int total_count = 0;
        if (tid < 32) {
            int v = (tid < nthreads / 32) ? __float_as_int(s_warp[tid]) : 0;
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            if (tid == 0) s_global = __int_as_float(v);
        }
        __syncthreads();
        total_count = __float_as_int(s_global);

        // Reduce probability sum
        for (int o = 16; o > 0; o >>= 1)
            local_psum += __shfl_xor_sync(0xFFFFFFFF, local_psum, o);
        if (tid % 32 == 0) s_warp[tid / 32] = local_psum;
        __syncthreads();
        float total_psum = 0.0f;
        if (tid < 32) {
            float v = (tid < nthreads / 32) ? s_warp[tid] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            if (tid == 0) s_global2 = v;
        }
        __syncthreads();
        total_psum = s_global2;

        // Binary search logic:
        // If too many elements above mid, raise threshold
        // If prob mass below top_p, lower threshold
        if (total_count > top_k || total_psum > top_p + 0.01f)
            lo = mid;
        else
            hi = mid;
    }
    float threshold = lo;
    __syncthreads();

    // --- Phase 3: Gather filtered probabilities + multinomial sample ---
    // Compute cumulative probability only for elements above threshold
    // Use parallel scan: each thread accumulates its local portion
    float local_filtered_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float p = probs_workspace[i];
        if (p >= threshold) local_filtered_sum += p;
    }
    // Block reduce for total filtered mass
    for (int o = 16; o > 0; o >>= 1)
        local_filtered_sum += __shfl_xor_sync(0xFFFFFFFF, local_filtered_sum, o);
    if (tid % 32 == 0) s_warp[tid / 32] = local_filtered_sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < nthreads / 32) ? s_warp[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, o);
        if (tid == 0) s_global = v;
    }
    __syncthreads();
    float filtered_mass = s_global;

    // Generate random number (LCG PRNG, fast and sufficient for sampling)
    // PCG-style: rng_seed provides per-call uniqueness
    unsigned long long rng_state = rng_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    float r = (float)((rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    r *= filtered_mass;  // Scale to filtered probability mass

    // Sequential scan to find the sampled token (thread 0 only)
    if (tid == 0) {
        float cum = 0.0f;
        int selected = 0;
        for (int i = 0; i < vocab_size; i++) {
            float p = probs_workspace[i];
            if (p >= threshold) {
                cum += p;
                if (cum >= r) {
                    selected = i;
                    break;
                }
            }
        }
        token_out[0] = selected;
    }
}

void sample_top_k_top_p(const __half* logits, float* probs_workspace,
                         int vocab_size, const SamplingParams& params,
                         unsigned long long rng_seed,
                         int* d_token_out, cudaStream_t stream) {
    top_k_top_p_sample_kernel<<<1, 1024, 0, stream>>>(
        logits, probs_workspace, vocab_size,
        params.temperature, params.top_k, params.top_p,
        rng_seed, d_token_out);
}

} // namespace deusridet
