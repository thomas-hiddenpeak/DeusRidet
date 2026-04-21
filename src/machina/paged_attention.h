/**
 * @file paged_attention.h
 * @philosophical_role Declaration of paged KV-cache attention kernels. Paged because the entity's working memory is NOT a contiguous tape — blocks are allocated, evicted, and resurrected.
 * @serves Machina forward pass, Memoria cache manager.
 */
// paged_attention.h — Paged KV Cache attention kernel declarations
//
// Paged variants of the GQA decode/prefill attention and KV cache write.
// KV data is stored in fixed-size blocks addressed via a block table.
// Supports non-contiguous KV sequences for eviction and SSD offload.

#pragma once

#include "model.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace deusridet {

// ============================================================================
// Paged KV Cache write (single token, decode)
// ============================================================================

// Write K and V for one token into the correct physical block.
// pos: absolute sequence position (determines logical block + offset).
// d_block_table: device array [max_logical_blocks], maps logical→physical block.
void paged_kv_cache_write(
    const __half* src_k,         // [NUM_KV_HEADS, HEAD_DIM]
    const __half* src_v,         // [NUM_KV_HEADS, HEAD_DIM]
    __half* kv_pool,             // entire KV pool base
    const int* d_block_table,    // [max_logical_blocks] device
    int fa_layer_idx,            // 0..15
    const int* d_pos,            // device pointer to current position
    int max_phys_blocks,         // for addressing
    int block_size,              // tokens per block
    cudaStream_t stream = 0);

// ============================================================================
// Paged KV Cache write (batched, prefill)
// ============================================================================

// Write K and V for M consecutive tokens starting at pos_start.
void paged_kv_cache_write_batch(
    const __half* src_k,         // [M, NUM_KV_HEADS, HEAD_DIM]
    const __half* src_v,         // [M, NUM_KV_HEADS, HEAD_DIM]
    __half* kv_pool,
    const int* d_block_table,
    int fa_layer_idx,
    int pos_start, int M,
    int max_phys_blocks,
    int block_size,
    cudaStream_t stream = 0);

// ============================================================================
// Paged GQA Decode Attention — Flash-Decoding with online softmax
// ============================================================================

// Single-token decode attention over paged KV blocks.
// Iterates through block table, performing online softmax across blocks.
// Grid = NUM_ATTN_HEADS (24), Block = HEAD_DIM (256).
void paged_gqa_decode_attention(
    const __half* Q,             // [NUM_ATTN_HEADS, HEAD_DIM]
    __half* kv_pool,
    const int* d_block_table,    // [max_logical_blocks] device
    __half* out,                 // [NUM_ATTN_HEADS, HEAD_DIM]
    int fa_layer_idx,
    int seq_len,                 // total KV sequence length
    int max_phys_blocks,
    int block_size,
    float scale,                 // 1/sqrt(head_dim)
    cudaStream_t stream = 0);

// ============================================================================
// Paged Full Attention forward (single-token decode)
// ============================================================================

// Complete FA layer forward with paged KV: Q/K/V proj, norms, RoPE,
// paged KV write, paged attention, gate, o_proj.
void full_attention_forward_paged(
    const __half* x,
    const FullAttentionWeights& attn,
    __half* kv_pool,
    const int* d_block_table,
    int fa_layer_idx,
    int pos, int seq_len,
    int max_phys_blocks,
    int block_size,
    InferenceState& state,
    cudaStream_t stream = 0);

// ============================================================================
// Paged Full Attention forward (batched prefill)
// ============================================================================

void full_attention_forward_paged_prefill(
    const __half* x,             // [M, HIDDEN_SIZE]
    const FullAttentionWeights& attn,
    __half* kv_pool,
    const int* d_block_table,
    int fa_layer_idx,
    int pos_start, int M,
    int seq_len,                 // total KV length after this prefill
    int max_phys_blocks,
    int block_size,
    InferenceState& state,
    cudaStream_t stream = 0);

} // namespace deusridet
