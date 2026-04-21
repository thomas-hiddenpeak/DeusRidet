/**
 * @file kv_swapper.h
 * @philosophical_role Declaration of the async swap interface. Memory migration is visible but non-blocking — the entity keeps thinking while it rearranges what it remembers.
 * @serves Memoria CacheManager.
 */
// kv_swapper.h — GPU↔SSD KV block swap for cache overflow
//
// Handles asynchronous transfer of KV blocks between GPU and SSD:
//   - Swap-out: GPU → staging buffer → fwrite → SSD file
//   - Swap-in:  SSD file → fread → staging buffer → GPU
//   - Prefetch: background swap-in so blocks are ready when accessed
//
// Uses FADV_DONTNEED after SSD write — critical on Tegra unified memory
// to release physical pages back to the GPU allocator.
//
// Adapted from qwen35-thor (kv_swapper): staging buffer consolidation,
// prefetch thread, FADV_DONTNEED pattern.

#pragma once

#include "block_pool.h"
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <functional>

namespace deusridet {

// ============================================================================
// Swap request
// ============================================================================

enum class SwapDirection { GPU_TO_SSD, SSD_TO_GPU };

struct SwapRequest {
    SwapDirection direction;
    int logical_block_idx;
    int physical_block_id;    // GPU block (source for out, target for in)
    std::string ssd_path;     // file path on SSD
    std::function<void(bool)> callback;  // completion callback (success)
};

// ============================================================================
// KVSwapper — async GPU↔SSD block transfer with prefetch
// ============================================================================

class KVSwapper {
public:
    KVSwapper() = default;
    ~KVSwapper();

    KVSwapper(const KVSwapper&) = delete;
    KVSwapper& operator=(const KVSwapper&) = delete;

    // Initialize with block pool reference and SSD directory.
    // staging_blocks: number of CPU-pinned staging blocks for async DMA.
    bool init(BlockPool& pool, const std::string& ssd_dir, int staging_blocks = 2);

    // Submit a swap request (non-blocking, queued for background thread).
    void submit(SwapRequest req);

    // Wait for all pending swaps to complete.
    void drain();

    // Shutdown the background thread.
    void shutdown();

    // Number of pending requests.
    int pending() const;

    // ── Prefetch API ────────────────────────────────────────────────

    // Submit an async prefetch: reads SSD file into a staging cache.
    // When swap_in is later called for the same block, it skips disk I/O.
    void prefetch(int logical_block_idx, const std::string& ssd_path);

    // Check if a block has been prefetched and is ready.
    bool is_prefetched(int logical_block_idx) const;

    // Drop a prefetched block (if the block was freed/invalidated).
    void drop_prefetch(int logical_block_idx);

private:
    BlockPool* pool_ = nullptr;
    std::string ssd_dir_;

    // CPU-pinned staging buffers for async GPU↔host DMA
    std::vector<void*> staging_bufs_;
    size_t staging_buf_size_ = 0;

    // Prefetch cache: logical_block_idx → staging buffer with SSD data
    // Protected by mu_. Data copied from SSD into these CPU buffers ahead of time.
    std::unordered_map<int, std::vector<char>> prefetch_cache_;
    std::unordered_set<int> prefetch_pending_;  // blocks being prefetched

    // Background I/O thread
    std::thread io_thread_;
    std::queue<SwapRequest> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool running_ = false;

    // CUDA stream for async H2D / D2H copies
    cudaStream_t copy_stream_ = nullptr;

    void io_loop();
    bool swap_out(const SwapRequest& req);
    bool swap_in(const SwapRequest& req);
};

} // namespace deusridet
