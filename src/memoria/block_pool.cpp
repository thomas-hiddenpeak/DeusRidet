/**
 * @file block_pool.cpp
 * @philosophical_role Pool of fixed-size KV blocks on GPU. The working memory granule — what is remembered, forgotten, and swapped is always a whole block.
 * @serves CacheManager, KVSwapper, Memoria importance scorer.
 */
// block_pool.cpp — GPU KV Cache block pool implementation

#include "block_pool.h"
#include "../communis/log.h"
#include <algorithm>

namespace deusridet {

BlockPool::~BlockPool() {
    destroy();
}

bool BlockPool::init(size_t budget_bytes, int block_size) {
    using C = PagedKVConfig;
    block_size_ = block_size;

    // Compute per-block sizes with configurable block_size
    kv_plane_     = (size_t)C::NUM_KV_HEADS * block_size_ * C::HEAD_DIM;   // elements per K (or V)
    block_stride_ = 2 * kv_plane_;                                          // K + V elements per block per layer
    layer_stride_ = 0;  // set after computing max_blocks

    // Bytes per block across all FA layers
    int num_fa = ModelConfig::NUM_FA_LAYERS;
    size_t block_total = (size_t)num_fa * block_stride_ * sizeof(__half);

    max_blocks_ = static_cast<int>(budget_bytes / block_total);
    if (max_blocks_ < 4) {
        LOG_ERROR("Memoria", "BlockPool: budget %.1f MB too small (need >= 4 blocks × %.1f MB)",
                  budget_bytes / 1048576.0, block_total / 1048576.0);
        return false;
    }

    layer_stride_ = (size_t)max_blocks_ * block_stride_;  // all blocks for one layer
    total_bytes_  = (size_t)num_fa * layer_stride_ * sizeof(__half);

    // Allocate GPU pool
    cudaError_t err = cudaMalloc(&pool_, total_bytes_);
    if (err != cudaSuccess) {
        LOG_ERROR("Memoria", "BlockPool: cudaMalloc(%.1f MB) failed: %s",
                  total_bytes_ / 1048576.0, cudaGetErrorString(err));
        pool_ = nullptr;
        return false;
    }
    cudaMemset(pool_, 0, total_bytes_);

    // Initialize free stack (all blocks available)
    free_stack_.resize(max_blocks_);
    for (int i = 0; i < max_blocks_; i++) {
        free_stack_[i] = max_blocks_ - 1 - i;  // push in reverse for LIFO order [0..N-1]
    }

    LOG_INFO("Memoria", "BlockPool: %d blocks × %d tokens = %dK max, %.1f MB GPU",
             max_blocks_, block_size_, max_blocks_ * block_size_ / 1024,
             total_bytes_ / 1048576.0);
    return true;
}

int BlockPool::alloc_block() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_stack_.empty()) return -1;
    int id = free_stack_.back();
    free_stack_.pop_back();
    return id;
}

void BlockPool::free_block(int block_id) {
    std::lock_guard<std::mutex> lock(mu_);
    free_stack_.push_back(block_id);
}

int BlockPool::free_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(free_stack_.size());
}

__half* BlockPool::k_ptr(int fa_layer_idx, int block_id) const {
    return pool_ + fa_layer_idx * layer_stride_ + block_id * block_stride_;
}

__half* BlockPool::v_ptr(int fa_layer_idx, int block_id) const {
    return pool_ + fa_layer_idx * layer_stride_ + block_id * block_stride_ + kv_plane_;
}

size_t BlockPool::k_offset(int fa_layer_idx, int block_id) const {
    return (fa_layer_idx * layer_stride_ + block_id * block_stride_) * sizeof(__half);
}

size_t BlockPool::v_offset(int fa_layer_idx, int block_id) const {
    return (fa_layer_idx * layer_stride_ + block_id * block_stride_ + kv_plane_) * sizeof(__half);
}

void BlockPool::destroy() {
    if (pool_) {
        cudaFree(pool_);
        pool_ = nullptr;
    }
    free_stack_.clear();
    max_blocks_ = 0;
    total_bytes_ = 0;
}

} // namespace deusridet
