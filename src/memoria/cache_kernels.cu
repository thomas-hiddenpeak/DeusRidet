/**
 * @file src/memoria/cache_kernels.cu
 * @philosophical_role
 *   Paged KV cache extract/inject kernels — the physical act of moving a block between GPU resident memory and SSD overflow. Forgetting is not deletion; it is descent into a deeper tier.
 * @serves
 *   Memoria cache_manager tier transitions; Machina paged_attention pointer lookups.
 */
// cache_kernels.cu — CUDA kernels for paged KV cache extract/inject
//
// One thread per __half element. Bandwidth-limited on DRAM (~192 GB/s Orin).
// ~0.28 ms for 1024 tokens per qwen35-thor measurements.
//
// Adapted from qwen35-thor (cache_kernels): scatter/gather pattern.
// Target: SM87 (Jetson AGX Orin)

#include "cache_kernels.h"

namespace deusridet {

// ============================================================================
// Extract kernel: paged → flat
//
// Grid: (total_elements + 255) / 256
// Block: 256 threads
//
// Flat layout: [layer][kv=2][token][head][dim]
// Paged layout: pool[layer * layer_stride + block * block_stride + {0|kv_plane} + head * block_size * dim + pos_in_block * dim + d]
// ============================================================================

__global__ void extract_kv_from_pages_kernel(
    __half* __restrict__ dst,
    const __half* __restrict__ kv_pool,
    const int* __restrict__ block_table,
    int start_token,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_phys_blocks)
{
    // Total elements = num_layers * 2 * num_tokens * num_kv_heads * head_dim
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_per_kv = num_tokens * num_kv_heads * head_dim;
    int total_per_layer = 2 * total_per_kv;
    int total = num_layers * total_per_layer;
    if (idx >= total) return;

    // Decompose flat index → (layer, kv, token, head, dim)
    int rem = idx;
    int layer = rem / total_per_layer;
    rem -= layer * total_per_layer;
    int kv = rem / total_per_kv;     // 0=K, 1=V
    rem -= kv * total_per_kv;
    int token_local = rem / (num_kv_heads * head_dim);
    rem -= token_local * (num_kv_heads * head_dim);
    int head = rem / head_dim;
    int d = rem - head * head_dim;

    int global_token = start_token + token_local;
    int logical_block = global_token / block_size;
    int pos_in_block = global_token - logical_block * block_size;

    int phys_block = block_table[logical_block];
    if (phys_block < 0) {
        dst[idx] = __float2half(0.0f);  // SSD or invalid — zero fill
        return;
    }

    // Pool addressing:
    // layer_stride = max_phys_blocks * block_stride
    // block_stride = 2 * kv_plane (K then V contiguous per block)
    // kv_plane = num_kv_heads * block_size * head_dim
    int kv_plane = num_kv_heads * block_size * head_dim;
    int block_stride = 2 * kv_plane;
    int layer_stride = max_phys_blocks * block_stride;

    int pool_offset = layer * layer_stride
                    + phys_block * block_stride
                    + kv * kv_plane
                    + head * block_size * head_dim
                    + pos_in_block * head_dim
                    + d;

    dst[idx] = kv_pool[pool_offset];
}

// ============================================================================
// Inject kernel: flat → paged
// ============================================================================

__global__ void inject_kv_to_pages_kernel(
    __half* __restrict__ kv_pool,
    const __half* __restrict__ src,
    const int* __restrict__ block_table,
    int start_token,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_phys_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_per_kv = num_tokens * num_kv_heads * head_dim;
    int total_per_layer = 2 * total_per_kv;
    int total = num_layers * total_per_layer;
    if (idx >= total) return;

    int rem = idx;
    int layer = rem / total_per_layer;
    rem -= layer * total_per_layer;
    int kv = rem / total_per_kv;
    rem -= kv * total_per_kv;
    int token_local = rem / (num_kv_heads * head_dim);
    rem -= token_local * (num_kv_heads * head_dim);
    int head = rem / head_dim;
    int d = rem - head * head_dim;

    int global_token = start_token + token_local;
    int logical_block = global_token / block_size;
    int pos_in_block = global_token - logical_block * block_size;

    int phys_block = block_table[logical_block];
    if (phys_block < 0) return;  // SSD or invalid — skip

    int kv_plane = num_kv_heads * block_size * head_dim;
    int block_stride = 2 * kv_plane;
    int layer_stride = max_phys_blocks * block_stride;

    int pool_offset = layer * layer_stride
                    + phys_block * block_stride
                    + kv * kv_plane
                    + head * block_size * head_dim
                    + pos_in_block * head_dim
                    + d;

    kv_pool[pool_offset] = src[idx];
}

// ============================================================================
// Host wrappers
// ============================================================================

void invoke_extract_kv_from_pages(
    __half* dst,
    const __half* kv_pool,
    const int* block_table,
    int start_token,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_phys_blocks,
    cudaStream_t stream)
{
    int total = num_layers * 2 * num_tokens * num_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    extract_kv_from_pages_kernel<<<blocks, threads, 0, stream>>>(
        dst, kv_pool, block_table, start_token, num_tokens,
        num_layers, num_kv_heads, head_dim, block_size, max_phys_blocks);
}

void invoke_inject_kv_to_pages(
    __half* kv_pool,
    const __half* src,
    const int* block_table,
    int start_token,
    int num_tokens,
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_phys_blocks,
    cudaStream_t stream)
{
    int total = num_layers * 2 * num_tokens * num_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    inject_kv_to_pages_kernel<<<blocks, threads, 0, stream>>>(
        kv_pool, src, block_table, start_token, num_tokens,
        num_layers, num_kv_heads, head_dim, block_size, max_phys_blocks);
}

} // namespace deusridet
