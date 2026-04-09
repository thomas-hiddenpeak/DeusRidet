// cache_manager.cpp — Unified KV Cache management implementation
//
// Adapted from qwen35-thor (cache_manager): SSD swap via KVSwapper,
// importance-guided eviction, block location tracking.

#include "cache_manager.h"
#include "kv_swapper.h"
#include "../communis/log.h"
#include <sys/stat.h>
#include <cstring>
#include <algorithm>

namespace deusridet {

CacheManager::~CacheManager() {
    destroy();
}

bool CacheManager::init(size_t kv_budget_bytes, const std::string& cache_dir,
                        int block_size, KVSwapper* swapper,
                        ImportanceScorer* scorer) {
    block_size_ = block_size;
    cache_dir_  = cache_dir;
    swapper_    = swapper;
    scorer_     = scorer;

    // Initialize block pool
    if (!pool_.init(kv_budget_bytes, block_size)) {
        return false;
    }

    // Compute max logical blocks based on pool capacity
    // Allow 2x physical blocks for logical space (SSD overflow)
    max_logical_blocks_ = pool_.max_blocks() * 2;

    h_block_table_.resize(max_logical_blocks_, -1);
    block_location_.resize(max_logical_blocks_, BlockLocation::INVALID);
    ssd_paths_.resize(max_logical_blocks_);

    // Allocate device block table
    size_t table_bytes = max_logical_blocks_ * sizeof(int);
    cudaError_t err = cudaMalloc(&d_block_table_, table_bytes);
    if (err != cudaSuccess) {
        LOG_ERROR("Memoria", "CacheManager: d_block_table alloc failed: %s",
                  cudaGetErrorString(err));
        return false;
    }
    cudaMemset(d_block_table_, 0xFF, table_bytes);  // -1 = invalid

    // Setup SSD cache directory
    if (!cache_dir_.empty()) {
        struct stat st;
        if (stat(cache_dir_.c_str(), &st) != 0) {
            if (mkdir(cache_dir_.c_str(), 0755) != 0) {
                LOG_WARN("Memoria", "Cannot create SSD cache dir: %s", cache_dir_.c_str());
                cache_dir_.clear();
            }
        }
        ssd_enabled_ = !cache_dir_.empty();
    }

    LOG_INFO("Memoria", "CacheManager: %d phys blocks, %d max logical, SSD=%s",
             pool_.max_blocks(), max_logical_blocks_,
             ssd_enabled_ ? cache_dir_.c_str() : "disabled");
    return true;
}

bool CacheManager::ensure_blocks_for(int pos) {
    std::lock_guard<std::mutex> lock(mu_);
    int logical_idx = pos / block_size_;

    if (logical_idx >= max_logical_blocks_) {
        LOG_ERROR("Memoria", "Position %d exceeds max logical blocks (%d)",
                  pos, max_logical_blocks_);
        return false;
    }

    // Already allocated on GPU?
    if (block_location_[logical_idx] == BlockLocation::GPU) {
        return true;
    }

    // On SSD — swap back to GPU
    if (block_location_[logical_idx] == BlockLocation::SSD) {
        return swap_block_from_ssd(logical_idx);
    }

    // Not allocated — grab from pool
    return alloc_logical_block(logical_idx) >= 0;
}

bool CacheManager::ensure_blocks_for_range(int pos_start, int M) {
    int first_block = pos_start / block_size_;
    int last_block  = (pos_start + M - 1) / block_size_;

    for (int b = first_block; b <= last_block; b++) {
        int pos = b * block_size_;
        if (!ensure_blocks_for(pos)) return false;
    }
    return true;
}

void CacheManager::sync_block_table(cudaStream_t stream) {
    size_t bytes = max_logical_blocks_ * sizeof(int);
    cudaMemcpyAsync(d_block_table_, h_block_table_.data(), bytes,
                    cudaMemcpyHostToDevice, stream);
}

int CacheManager::num_logical_blocks() const {
    return (seq_len_ + block_size_ - 1) / block_size_;
}

int CacheManager::evict_blocks(int n) {
    int evicted = 0;
    while (evicted < n && !lru_order_.empty()) {
        int logical_idx = pick_eviction_candidate();
        if (logical_idx < 0) break;

        // Remove from LRU
        lru_order_.erase(
            std::remove(lru_order_.begin(), lru_order_.end(), logical_idx),
            lru_order_.end());

        int phys_id = h_block_table_[logical_idx];
        if (phys_id < 0) continue;

        // SSD write via KVSwapper (if available), else discard
        if (ssd_enabled_ && swapper_) {
            if (swap_block_to_ssd(logical_idx)) {
                // Block is on SSD — physical block freed, logical still valid
                evicted++;
                continue;
            }
            LOG_WARN("Memoria", "SSD swap-out failed for block %d, discarding",
                     logical_idx);
        }

        // Discard: free physical block, mark invalid
        pool_.free_block(phys_id);
        h_block_table_[logical_idx] = -1;
        block_location_[logical_idx] = BlockLocation::INVALID;
        evicted++;
    }

    if (evicted > 0) {
        LOG_INFO("Memoria", "Evicted %d blocks (SSD=%s), %d free now",
                 evicted, (ssd_enabled_ && swapper_) ? "yes" : "no",
                 pool_.free_count());
    }
    return evicted;
}

bool CacheManager::ensure_free_blocks(int n_free) {
    while (pool_.free_count() < n_free) {
        if (lru_order_.empty()) return false;
        if (evict_blocks(1) == 0) return false;
    }
    return true;
}

void CacheManager::prefetch_block(int logical_idx) {
    if (logical_idx < 0 || logical_idx >= max_logical_blocks_) return;
    if (block_location_[logical_idx] != BlockLocation::SSD) return;
    if (!swapper_) return;

    // Submit async prefetch — KVSwapper will read from SSD in background
    SwapRequest req;
    req.direction = SwapDirection::SSD_TO_GPU;
    req.logical_block_idx = logical_idx;
    req.physical_block_id = -1;  // not yet assigned; swapper caches to staging
    req.ssd_path = ssd_path_for(logical_idx);
    req.callback = nullptr;  // prefetch is fire-and-forget
    swapper_->submit(std::move(req));
}

// ============================================================================
// Eviction candidate selection
// ============================================================================

int CacheManager::pick_eviction_candidate() {
    if (lru_order_.empty()) return -1;

    // If ImportanceScorer is attached, pick least important GPU block
    if (scorer_) {
        // Build list of GPU-resident logical indices
        std::vector<int> gpu_blocks;
        gpu_blocks.reserve(lru_order_.size());
        for (int idx : lru_order_) {
            if (block_location_[idx] == BlockLocation::GPU)
                gpu_blocks.push_back(idx);
        }
        if (!gpu_blocks.empty()) {
            return scorer_->least_important(gpu_blocks.data(),
                                            static_cast<int>(gpu_blocks.size()));
        }
    }

    // Fallback: LRU (oldest first)
    return lru_order_.front();
}

// ============================================================================
// SSD swap helpers
// ============================================================================

std::string CacheManager::ssd_path_for(int logical_idx) const {
    return cache_dir_ + "/block_" + std::to_string(logical_idx) + ".kv";
}

bool CacheManager::swap_block_to_ssd(int logical_idx) {
    int phys_id = h_block_table_[logical_idx];
    if (phys_id < 0) return false;

    std::string path = ssd_path_for(logical_idx);

    // Synchronous swap-out (blocking for correctness on eviction path)
    SwapRequest req;
    req.direction = SwapDirection::GPU_TO_SSD;
    req.logical_block_idx = logical_idx;
    req.physical_block_id = phys_id;
    req.ssd_path = path;

    bool success = false;
    req.callback = [&success](bool ok) { success = ok; };
    swapper_->submit(std::move(req));
    swapper_->drain();  // wait for completion

    if (!success) return false;

    // Free physical block, mark logical as SSD
    pool_.free_block(phys_id);
    h_block_table_[logical_idx] = -1;
    block_location_[logical_idx] = BlockLocation::SSD;
    ssd_paths_[logical_idx] = path;
    return true;
}

bool CacheManager::swap_block_from_ssd(int logical_idx) {
    if (block_location_[logical_idx] != BlockLocation::SSD) return false;

    // Allocate a fresh physical block (may trigger further eviction)
    int phys_id = pool_.alloc_block();
    if (phys_id < 0) {
        // Need to evict one block to make space
        // Temporarily unlock to allow eviction (caller holds mu_)
        // Use direct eviction of the LRU front to avoid recursion
        int victim = pick_eviction_candidate();
        if (victim < 0 || victim == logical_idx) return false;

        int victim_phys = h_block_table_[victim];
        if (victim_phys >= 0) {
            // Discard victim (we can't swap to SSD while swapping in)
            pool_.free_block(victim_phys);
            h_block_table_[victim] = -1;
            block_location_[victim] = BlockLocation::INVALID;
            lru_order_.erase(
                std::remove(lru_order_.begin(), lru_order_.end(), victim),
                lru_order_.end());
        }

        phys_id = pool_.alloc_block();
        if (phys_id < 0) return false;
    }

    // Read from SSD to GPU
    SwapRequest req;
    req.direction = SwapDirection::SSD_TO_GPU;
    req.logical_block_idx = logical_idx;
    req.physical_block_id = phys_id;
    req.ssd_path = ssd_paths_[logical_idx];

    bool success = false;
    req.callback = [&success](bool ok) { success = ok; };
    swapper_->submit(std::move(req));
    swapper_->drain();

    if (!success) {
        pool_.free_block(phys_id);
        return false;
    }

    // Update tracking
    h_block_table_[logical_idx] = phys_id;
    block_location_[logical_idx] = BlockLocation::GPU;
    lru_order_.push_back(logical_idx);
    return true;
}

int CacheManager::gpu_blocks_used() const {
    return pool_.max_blocks() - pool_.free_count();
}

int CacheManager::ssd_blocks_used() const {
    int count = 0;
    for (auto loc : block_location_) {
        if (loc == BlockLocation::SSD) count++;
    }
    return count;
}

int CacheManager::alloc_logical_block(int logical_idx) {
    // Try direct allocation
    int phys_id = pool_.alloc_block();

    // Pool exhausted — evict oldest block
    if (phys_id < 0) {
        if (!ensure_free_blocks(1)) {
            LOG_ERROR("Memoria", "Cannot allocate block %d: pool exhausted, eviction failed",
                      logical_idx);
            return -1;
        }
        phys_id = pool_.alloc_block();
        if (phys_id < 0) return -1;
    }

    h_block_table_[logical_idx] = phys_id;
    block_location_[logical_idx] = BlockLocation::GPU;
    lru_order_.push_back(logical_idx);

    return phys_id;
}

void CacheManager::destroy() {
    if (d_block_table_) {
        cudaFree(d_block_table_);
        d_block_table_ = nullptr;
    }
    pool_.destroy();
    h_block_table_.clear();
    block_location_.clear();
    lru_order_.clear();
    seq_len_ = 0;
}

} // namespace deusridet
