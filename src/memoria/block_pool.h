// block_pool.h — GPU KV Cache block pool allocator
//
// Pre-allocates a contiguous GPU buffer and manages fixed-size KV blocks
// via a free stack. Each physical block stores K+V for one Full Attention
// layer at BLOCK_SIZE token positions.
//
// Pool layout (per FA layer, per physical block):
//   K: [NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM] in FP16
//   V: [NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM] in FP16
//
// Total per block per layer = 2 * 4 * 256 * 256 * 2 = 1 MB
// Total per block (16 FA layers) = 16 MB

#pragma once

#include "../machina/model.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <mutex>

namespace deusridet {

// ============================================================================
// Configuration
// ============================================================================

struct PagedKVConfig {
    static constexpr int BLOCK_SIZE    = 256;   // tokens per block
    // NUM_KV_HEADS and HEAD_DIM are the same across Qwen3.5 variants (4, 256)
    static constexpr int NUM_KV_HEADS  = 4;
    static constexpr int HEAD_DIM      = 256;

    // NUM_FA_LAYERS is model-dependent — use ModelConfig::NUM_FA_LAYERS at runtime
    // For compile-time layout calculations, use the constexpr values above.

    // Elements per K (or V) block for one layer: heads * block_size * dim
    static constexpr size_t KV_BLOCK_ELEMS = NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM;
    // Bytes per K+V block for one layer
    static constexpr size_t BLOCK_LAYER_BYTES = 2 * KV_BLOCK_ELEMS * sizeof(__half);

    // Bytes per block across all FA layers (runtime, depends on model)
    static size_t block_total_bytes() {
        return ModelConfig::NUM_FA_LAYERS * BLOCK_LAYER_BYTES;
    }
};

// ============================================================================
// BlockPool — GPU memory pool for KV cache blocks
// ============================================================================

class BlockPool {
public:
    BlockPool() = default;
    ~BlockPool();

    // Non-copyable, non-movable
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    // Allocate the GPU pool with the given memory budget (bytes).
    // Returns false on failure. Computes max_blocks from budget.
    bool init(size_t budget_bytes, int block_size = PagedKVConfig::BLOCK_SIZE);

    // Allocate a physical block. Returns block ID (>= 0) or -1 if exhausted.
    int alloc_block();

    // Free a physical block back to the pool.
    void free_block(int block_id);

    // Number of blocks currently available.
    int free_count() const;

    // Total number of physical blocks.
    int max_blocks() const { return max_blocks_; }

    // Tokens per block.
    int block_size() const { return block_size_; }

    // Base pointer to the KV pool on device.
    __half* pool_ptr() const { return pool_; }

    // Get K cache base pointer for a given FA layer and physical block.
    // K: [NUM_KV_HEADS, block_size, HEAD_DIM]
    __half* k_ptr(int fa_layer_idx, int block_id) const;

    // Get V cache base pointer for a given FA layer and physical block.
    // V: [NUM_KV_HEADS, block_size, HEAD_DIM]
    __half* v_ptr(int fa_layer_idx, int block_id) const;

    // Byte offset from pool base to K data for (fa_layer, block_id).
    size_t k_offset(int fa_layer_idx, int block_id) const;

    // Byte offset from pool base to V data for (fa_layer, block_id).
    size_t v_offset(int fa_layer_idx, int block_id) const;

    // Total allocated GPU memory in bytes.
    size_t total_bytes() const { return total_bytes_; }

    // Release all GPU memory.
    void destroy();

private:
    __half* pool_       = nullptr;
    int max_blocks_     = 0;
    int block_size_     = PagedKVConfig::BLOCK_SIZE;
    size_t total_bytes_ = 0;

    // Layout strides (in __half elements, not bytes)
    size_t layer_stride_ = 0;  // distance between consecutive FA layers
    size_t block_stride_ = 0;  // distance between consecutive blocks within one layer
    size_t kv_plane_     = 0;  // K or V plane size per block (in elements)

    // Free stack (LIFO for cache locality)
    std::vector<int> free_stack_;
    mutable std::mutex mu_;
};

} // namespace deusridet
