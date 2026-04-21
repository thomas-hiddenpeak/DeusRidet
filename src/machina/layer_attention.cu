/**
 * @file src/machina/layer_attention.cu
 * @philosophical_role
 *   Attention primitives (RoPE, head-norm, softmax) + Mamba conv1d step +
 *   sampling kernels. Peer TU of layer.cu under R1 800-line hard cap.
 * @serves
 *   Machina forward.cu attention path; conv1d_step paths; sampler.
 */
// layer_attention.cu — peer TU of layer.cu (attention/conv/sampling).

#include "layer.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cfloat>
#include <vector>

namespace deusridet {


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
