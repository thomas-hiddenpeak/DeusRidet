// cache_manager.h — Unified KV Cache management for consciousness stream
//
// Manages the complete lifecycle of paged KV blocks:
//   - Block table: maps logical block index → physical block ID
//   - Allocation: grab blocks from BlockPool as sequence grows
//   - Eviction: importance-guided (via ImportanceScorer), with SSD offload
//   - SSD offload: delegates to KVSwapper for GPU↔SSD transfers
//   - Swap-in: restore SSD-resident blocks to GPU when accessed
//   - SSM state snapshots: periodic checkpointing of DeltaNet states
//
// The consciousness stream is a single infinite-lifetime sequence.
// Unlike request-response servers that evict entire requests,
// CacheManager evicts individual blocks WITHIN the stream.
//
// Adapted from qwen35-thor (cache_manager, block_tracker, kv_swapper):
// Block location tracking, staging buffer consolidation, FADV_DONTNEED,
// importance-guided eviction (replaces pure FIFO/LRU).

#pragma once

#include "block_pool.h"
#include "importance_scorer.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <deque>
#include <string>
#include <mutex>
#include <cstdint>

namespace deusridet {

// Forward declarations
class KVSwapper;

// Block location for the block tracker
enum class BlockLocation : int8_t {
    GPU     =  0,  // resident on GPU
    SSD     = -2,  // swapped to SSD
    INVALID = -1   // not allocated
};

// ============================================================================
// CacheManager — unified KV block management
// ============================================================================

class CacheManager {
public:
    CacheManager() = default;
    ~CacheManager();

    // Non-copyable
    CacheManager(const CacheManager&) = delete;
    CacheManager& operator=(const CacheManager&) = delete;

    // Initialize with memory budget and optional SSD cache directory.
    // kv_budget_bytes: GPU memory for KV blocks
    // cache_dir: directory for SSD overflow (empty = no SSD)
    // block_size: tokens per block
    // swapper: optional KVSwapper for SSD offload (caller owns lifetime)
    // scorer: optional ImportanceScorer for guided eviction (caller owns lifetime)
    bool init(size_t kv_budget_bytes, const std::string& cache_dir = "",
              int block_size = PagedKVConfig::BLOCK_SIZE,
              KVSwapper* swapper = nullptr,
              ImportanceScorer* scorer = nullptr);

    // ── Token-level API ─────────────────────────────────────────────

    // Ensure blocks are allocated for writing at position pos.
    // Allocates new blocks as needed, evicts if pool exhausted.
    // Returns false if unable to allocate (pool + SSD both exhausted).
    bool ensure_blocks_for(int pos);

    // Ensure blocks for a range [pos_start, pos_start + M).
    bool ensure_blocks_for_range(int pos_start, int M);

    // ── Block table access ──────────────────────────────────────────

    // Device block table pointer (for passing to paged attention kernels).
    const int* d_block_table() const { return d_block_table_; }

    // Update device block table from host mirror (call after any change).
    void sync_block_table(cudaStream_t stream = 0);

    // Current sequence length (number of tokens written).
    int seq_len() const { return seq_len_; }
    void set_seq_len(int len) { seq_len_ = len; }

    // Number of logical blocks in use.
    int num_logical_blocks() const;

    // ── Eviction ────────────────────────────────────────────────────

    // Evict N GPU-resident blocks to SSD (importance-guided if scorer
    // attached, otherwise LRU). Returns number evicted.
    int evict_blocks(int n);

    // Evict blocks until at least n_free blocks are available in the pool.
    // Returns true if target met.
    bool ensure_free_blocks(int n_free);

    // ── Swap-in ─────────────────────────────────────────────────────

    // Prefetch an SSD-resident block into GPU (async, non-blocking).
    // The block will be GPU-ready by the time it is accessed.
    void prefetch_block(int logical_idx);

    // ── Block pool access ───────────────────────────────────────────

    BlockPool& block_pool() { return pool_; }
    const BlockPool& block_pool() const { return pool_; }

    // ── Statistics ──────────────────────────────────────────────────

    int gpu_blocks_used() const;
    int ssd_blocks_used() const;
    int free_blocks() const { return pool_.free_count(); }

    // Release all resources.
    void destroy();

private:
    BlockPool pool_;
    KVSwapper* swapper_ = nullptr;           // optional SSD offload (not owned)
    ImportanceScorer* scorer_ = nullptr;     // optional importance scoring (not owned)

    // Host block table: logical_block_idx → physical_block_id
    // -1 = not allocated, physical_id >= 0 = GPU-resident
    // SSD-resident blocks: h_block_table_[i] = -1, location = SSD
    std::vector<int> h_block_table_;
    int* d_block_table_ = nullptr;  // device mirror
    int max_logical_blocks_ = 0;

    // Block location tracker
    std::vector<BlockLocation> block_location_;  // indexed by logical block

    // LRU order: front = oldest (eviction candidate), back = newest
    std::deque<int> lru_order_;  // logical block indices of GPU-resident blocks

    // SSD file paths for swapped blocks (indexed by logical block)
    std::vector<std::string> ssd_paths_;

    int seq_len_ = 0;        // current sequence length
    int block_size_ = 256;
    std::string cache_dir_;
    bool ssd_enabled_ = false;

    mutable std::mutex mu_;

    // Allocate a new logical block, handling eviction if needed.
    int alloc_logical_block(int logical_idx);

    // Build SSD path for a logical block.
    std::string ssd_path_for(int logical_idx) const;

    // Pick the best eviction candidate (importance-guided or LRU).
    int pick_eviction_candidate();

    // Swap a single block from GPU to SSD. Returns true on success.
    bool swap_block_to_ssd(int logical_idx);

    // Swap a single block from SSD back to GPU. Returns true on success.
    bool swap_block_from_ssd(int logical_idx);
};

} // namespace deusridet
