/**
 * @file cache_kernels.h
 * @philosophical_role Declaration of GPU extract/inject kernels for paged KV blocks. The movement of memory between tiers is itself a cognitive act.
 * @serves KVSwapper, CacheManager fast paths.
 */
// cache_kernels.h — CUDA kernels for paged KV cache extract/inject
//
// Scatter/gather operations between paged KV layout and flat contiguous
// buffers. Used for:
//   - SSD swap staging (paged GPU → flat CPU staging → SSD)
//   - Prefix cache store/restore
//   - External tools that need contiguous KV representations
//
// Adapted from qwen35-thor (cache_kernels): one-thread-per-element
// scatter/gather at ~230 GB/s bandwidth utilization.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {

// Extract KV data from paged layout into a flat contiguous buffer.
//
// Gathers K and V for `num_tokens` starting at `start_token` across
// `num_layers` FA layers from the paged KV pool into `dst`.
//
// dst layout: [num_layers][2][num_tokens][num_kv_heads][head_dim]
//   (2 = K then V, contiguous per layer)
//
// Args:
//   dst: flat output buffer (device memory)
//   kv_pool: base of paged KV pool
//   block_table: [max_logical_blocks] logical→physical mapping
//   start_token: first token position to extract
//   num_tokens: number of tokens to extract
//   num_layers: number of FA layers
//   num_kv_heads: KV heads per layer
//   head_dim: dimension per head
//   block_size: tokens per block
//   max_phys_blocks: total physical blocks (for stride computation)
//   stream: CUDA stream
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
    cudaStream_t stream);

// Inject KV data from a flat contiguous buffer into paged layout.
//
// Inverse of extract: scatters flat KV data back into the paged pool.
// Same layout conventions as extract.
//
// Args: same as extract, but src is input and kv_pool is output.
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
    cudaStream_t stream);

} // namespace deusridet
