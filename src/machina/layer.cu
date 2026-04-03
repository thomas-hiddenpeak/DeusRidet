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
// RMSNorm kernel
//
// Each block handles one row. Warp-level reduction for mean(x^2).
// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// ============================================================================

__global__ void rms_norm_kernel(const __half* __restrict__ x,
                                const __half* __restrict__ weight,
                                __half* __restrict__ out,
                                int dim, float eps) {
    const int row = blockIdx.x;
    const __half* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    // Compute sum of squares with thread-level accumulation
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
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

    // Apply normalization and weight
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]) * inv_rms;
        out_row[i] = __float2half(v * __half2float(weight[i]));
    }
}

void rms_norm(const __half* x, const __half* weight, __half* out,
              int rows, int dim, float eps, cudaStream_t stream) {
    int threads = (dim < 256) ? 128 : 256;
    rms_norm_kernel<<<rows, threads, 0, stream>>>(x, weight, out, dim, eps);
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
        float silu_g = g / (1.0f + expf(-g));
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

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = row[i];
    }
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
    }
}

void elementwise_add(const __half* a, const __half* b, __half* out,
                     int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void sigmoid_gate_kernel(const __half* __restrict__ x,
                                    const __half* __restrict__ gate,
                                    __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        float g = __half2float(gate[i]);
        out[i] = __float2half(v / (1.0f + expf(-g)));
    }
}

void sigmoid_gate(const __half* x, const __half* gate, __half* out,
                  int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_gate_kernel<<<blocks, threads, 0, stream>>>(x, gate, out, n);
}

// ============================================================================
// Linear forward via cuBLAS
//
// Y[M,N] = X[M,K] @ W^T[K,N] where W is [N,K] row-major.
// cuBLAS uses column-major, so we compute: Y^T = W @ X^T
//   i.e. cublasGemmEx(N, M, K, W, N, X, K, Y, N)
// ============================================================================

void linear_forward(const __half* X, const Linear& weight, __half* Y,
                    int M, cudaStream_t stream) {
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    int K = weight.in_features;
    int N = weight.out_features;

    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    cublasGemmEx(handle,
                 CUBLAS_OP_T,     // W^T: W is [N,K] row-major → column-major [K,N]
                 CUBLAS_OP_N,     // X: [M,K] row-major → column-major [K,M]
                 N, M, K,
                 &alpha_h,
                 weight.weight, CUDA_R_16F, K,  // A = W^T: ld = K
                 X,              CUDA_R_16F, K,  // B = X:   ld = K
                 &beta_h,
                 Y,              CUDA_R_16F, N,  // C = Y:   ld = N
                 CUDA_R_16F,                     // compute type
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ============================================================================
// Partial RoPE (interleaved pairing, single-token)
//
// Qwen3.5 uses partial_rotary_factor=0.25, so only first 64 of 256 dims
// are rotated. Interleaved means pairs are (0,1), (2,3), ... not (0,D/2).
// freq_i = 1.0 / (theta^(2i/rotary_dim))
// ============================================================================

__global__ void rope_kernel(__half* __restrict__ q,
                            __half* __restrict__ k,
                            int num_q_heads, int num_kv_heads,
                            int head_dim, int rotary_dim,
                            int pos, float theta) {
    // Each thread handles one pair in one head
    int head = blockIdx.x;
    int pair = threadIdx.x;  // pair index within rotary_dim/2
    int num_pairs = rotary_dim / 2;
    if (pair >= num_pairs) return;

    float freq = 1.0f / powf(theta, (2.0f * pair) / (float)rotary_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Apply to Q head
    if (head < num_q_heads) {
        int idx = head * head_dim + pair * 2;
        float x0 = __half2float(q[idx]);
        float x1 = __half2float(q[idx + 1]);
        q[idx]     = __float2half(x0 * cos_a - x1 * sin_a);
        q[idx + 1] = __float2half(x0 * sin_a + x1 * cos_a);
    }

    // Apply to K head (only first num_kv_heads)
    if (head < num_kv_heads) {
        int idx = head * head_dim + pair * 2;
        float x0 = __half2float(k[idx]);
        float x1 = __half2float(k[idx + 1]);
        k[idx]     = __float2half(x0 * cos_a - x1 * sin_a);
        k[idx + 1] = __float2half(x0 * sin_a + x1 * cos_a);
    }
}

void apply_rope(__half* q, __half* k,
                int num_heads, int num_kv_heads, int head_dim,
                int rotary_dim, int pos, float theta,
                cudaStream_t stream) {
    int num_pairs = rotary_dim / 2;
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    rope_kernel<<<max_heads, num_pairs, 0, stream>>>(
        q, k, num_heads, num_kv_heads, head_dim, rotary_dim, pos, theta);
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
// Causal conv1d step (single token, for DeltaNet decode)
//
// conv_state[conv_dim, kernel-1] stores the last (kernel-1) inputs per channel.
// On each step: shift left, insert new input, compute weighted sum.
// ============================================================================

__global__ void conv1d_step_kernel(const __half* __restrict__ x_in,
                                   __half* __restrict__ conv_state,
                                   const __half* __restrict__ conv_weight,
                                   __half* __restrict__ x_out,
                                   int conv_dim, int kernel_size) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    int km1 = kernel_size - 1;  // 3 for kernel_size=4

    // Shift state left: state[ch, j] = state[ch, j+1] for j=0..km1-2
    for (int j = 0; j < km1 - 1; j++) {
        conv_state[ch * km1 + j] = conv_state[ch * km1 + j + 1];
    }
    // Insert new input at the end
    conv_state[ch * km1 + (km1 - 1)] = x_in[ch];

    // Compute weighted sum: sum over kernel dimension
    // weight layout: [conv_dim, kernel_size]
    float acc = 0.0f;
    for (int j = 0; j < km1; j++) {
        acc += __half2float(conv_state[ch * km1 + j]) *
               __half2float(conv_weight[ch * kernel_size + j]);
    }
    // Last kernel element uses the new input directly
    acc += __half2float(x_in[ch]) *
           __half2float(conv_weight[ch * kernel_size + km1]);

    x_out[ch] = __float2half(acc);
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
// Greedy sampling: argmax on CPU
// ============================================================================

int greedy_sample(const __half* logits, int vocab_size) {
    std::vector<__half> host(vocab_size);
    cudaMemcpy(host.data(), logits, vocab_size * sizeof(__half),
               cudaMemcpyDeviceToHost);

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
        float v = __half2float(host[i]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace deusridet
