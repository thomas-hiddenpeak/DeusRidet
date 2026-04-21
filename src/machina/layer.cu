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

} // namespace deusridet
