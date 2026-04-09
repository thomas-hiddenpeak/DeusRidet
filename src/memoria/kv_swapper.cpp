// kv_swapper.cpp — GPU↔SSD KV block swap implementation
//
// Background I/O thread handles swap requests asynchronously.
// Uses CPU-pinned staging buffers for GPU DMA + file I/O.
// Applies FADV_DONTNEED after writes to release page cache on Tegra.
//
// Prefetch support: blocks can be pre-read from SSD into a CPU cache.
// When swap_in is called for a prefetched block, disk I/O is skipped
// (1 ms vs 33 ms per qwen35-thor measurements).
//
// Adapted from qwen35-thor (kv_swapper): staging consolidation,
// prefetch thread, FADV_DONTNEED pattern.

#include "kv_swapper.h"
#include "../communis/log.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>

namespace deusridet {

KVSwapper::~KVSwapper() {
    shutdown();
}

bool KVSwapper::init(BlockPool& pool, const std::string& ssd_dir, int staging_blocks) {
    pool_ = &pool;
    ssd_dir_ = ssd_dir;

    // Compute staging buffer size: one full block across all FA layers
    using C = PagedKVConfig;
    staging_buf_size_ = C::block_total_bytes();

    // Allocate CPU-pinned staging buffers
    staging_bufs_.resize(staging_blocks);
    for (int i = 0; i < staging_blocks; i++) {
        cudaError_t err = cudaHostAlloc(&staging_bufs_[i], staging_buf_size_,
                                        cudaHostAllocDefault);
        if (err != cudaSuccess) {
            LOG_ERROR("Memoria", "KVSwapper: staging alloc failed: %s",
                      cudaGetErrorString(err));
            return false;
        }
    }

    // Create CUDA stream for async copies
    cudaStreamCreate(&copy_stream_);

    // Ensure SSD directory exists
    struct stat st;
    if (stat(ssd_dir_.c_str(), &st) != 0) {
        mkdir(ssd_dir_.c_str(), 0755);
    }

    // Start background I/O thread
    running_ = true;
    io_thread_ = std::thread(&KVSwapper::io_loop, this);

    LOG_INFO("Memoria", "KVSwapper: %d staging bufs × %.1f MB, dir=%s",
             staging_blocks, staging_buf_size_ / 1048576.0,
             ssd_dir_.c_str());
    return true;
}

void KVSwapper::submit(SwapRequest req) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.push(std::move(req));
    }
    cv_.notify_one();
}

void KVSwapper::drain() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this]{ return queue_.empty(); });
}

void KVSwapper::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        running_ = false;
    }
    cv_.notify_all();
    if (io_thread_.joinable()) io_thread_.join();

    // Free staging buffers
    for (auto* buf : staging_bufs_) {
        if (buf) cudaFreeHost(buf);
    }
    staging_bufs_.clear();

    // Free prefetch cache
    prefetch_cache_.clear();
    prefetch_pending_.clear();

    if (copy_stream_) {
        cudaStreamDestroy(copy_stream_);
        copy_stream_ = nullptr;
    }
}

int KVSwapper::pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(queue_.size());
}

// ============================================================================
// Prefetch API
// ============================================================================

void KVSwapper::prefetch(int logical_block_idx, const std::string& ssd_path) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        // Already prefetched or pending?
        if (prefetch_cache_.count(logical_block_idx) ||
            prefetch_pending_.count(logical_block_idx)) {
            return;
        }
        prefetch_pending_.insert(logical_block_idx);
    }

    // Submit as a regular swap-in, but with physical_block_id = -1
    // (signals prefetch-only: read to CPU cache, not GPU)
    SwapRequest req;
    req.direction = SwapDirection::SSD_TO_GPU;
    req.logical_block_idx = logical_block_idx;
    req.physical_block_id = -1;  // sentinel for prefetch
    req.ssd_path = ssd_path;
    req.callback = nullptr;
    submit(std::move(req));
}

bool KVSwapper::is_prefetched(int logical_block_idx) const {
    std::lock_guard<std::mutex> lock(mu_);
    return prefetch_cache_.count(logical_block_idx) > 0;
}

void KVSwapper::drop_prefetch(int logical_block_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    prefetch_cache_.erase(logical_block_idx);
    prefetch_pending_.erase(logical_block_idx);
}

// ============================================================================
// Background I/O thread
// ============================================================================

void KVSwapper::io_loop() {
    while (true) {
        SwapRequest req;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this]{ return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            req = std::move(queue_.front());
            queue_.pop();
        }

        bool ok = false;
        if (req.direction == SwapDirection::GPU_TO_SSD) {
            ok = swap_out(req);
        } else {
            ok = swap_in(req);
        }

        if (req.callback) req.callback(ok);

        // Notify drain() waiters
        cv_.notify_all();
    }
}

bool KVSwapper::swap_out(const SwapRequest& req) {
    using C = PagedKVConfig;
    void* staging = staging_bufs_[0];  // use first staging buffer

    // Consolidated copy: gather all FA layers' K+V for this block into staging
    size_t offset = 0;
    int num_fa = ModelConfig::NUM_FA_LAYERS;
    for (int layer = 0; layer < num_fa; layer++) {
        __half* k_src = pool_->k_ptr(layer, req.physical_block_id);
        __half* v_src = pool_->v_ptr(layer, req.physical_block_id);
        size_t kv_bytes = C::KV_BLOCK_ELEMS * sizeof(__half);

        cudaMemcpyAsync(static_cast<char*>(staging) + offset,
                        k_src, kv_bytes, cudaMemcpyDeviceToHost, copy_stream_);
        offset += kv_bytes;

        cudaMemcpyAsync(static_cast<char*>(staging) + offset,
                        v_src, kv_bytes, cudaMemcpyDeviceToHost, copy_stream_);
        offset += kv_bytes;
    }
    cudaStreamSynchronize(copy_stream_);

    // Write consolidated block to SSD
    int fd = open(req.ssd_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOG_ERROR("Memoria", "KVSwapper: open(%s) for write failed", req.ssd_path.c_str());
        return false;
    }

    ssize_t written = write(fd, staging, offset);
    if (written != (ssize_t)offset) {
        LOG_ERROR("Memoria", "KVSwapper: write failed (%zd/%zu)", written, offset);
        close(fd);
        return false;
    }

    // FADV_DONTNEED: release page cache — critical on Tegra unified memory
    // Page cache pages = physical RAM that GPU could allocate
    posix_fadvise(fd, 0, offset, POSIX_FADV_DONTNEED);
    close(fd);

    return true;
}

bool KVSwapper::swap_in(const SwapRequest& req) {
    using C = PagedKVConfig;
    size_t total = C::block_total_bytes();

    // Check if this is a prefetch-only request (physical_block_id == -1)
    bool is_prefetch = (req.physical_block_id < 0);

    // Check prefetch cache first (skip disk I/O if available)
    const char* src_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = prefetch_cache_.find(req.logical_block_idx);
        if (it != prefetch_cache_.end() && !is_prefetch) {
            src_data = it->second.data();
        }
    }

    void* staging = staging_bufs_[0];

    if (src_data && !is_prefetch) {
        // Prefetch hit: copy from CPU cache (skip disk I/O entirely)
        memcpy(staging, src_data, total);
        {
            std::lock_guard<std::mutex> lock(mu_);
            prefetch_cache_.erase(req.logical_block_idx);
            prefetch_pending_.erase(req.logical_block_idx);
        }
    } else {
        // Read from SSD into staging
        int fd = open(req.ssd_path.c_str(), O_RDONLY);
        if (fd < 0) {
            LOG_ERROR("Memoria", "KVSwapper: open(%s) for read failed",
                      req.ssd_path.c_str());
            return false;
        }

        ssize_t nread = read(fd, is_prefetch ? staging : staging, total);
        posix_fadvise(fd, 0, total, POSIX_FADV_DONTNEED);
        close(fd);

        if (nread != (ssize_t)total) {
            LOG_ERROR("Memoria", "KVSwapper: read failed (%zd/%zu)", nread, total);
            return false;
        }

        // If prefetch-only: store in prefetch cache and return
        if (is_prefetch) {
            std::vector<char> buf(total);
            memcpy(buf.data(), staging, total);
            {
                std::lock_guard<std::mutex> lock(mu_);
                prefetch_cache_[req.logical_block_idx] = std::move(buf);
                prefetch_pending_.erase(req.logical_block_idx);
            }
            return true;
        }
    }

    // Copy from staging to GPU (scatter into paged layout)
    size_t offset = 0;
    int num_fa = ModelConfig::NUM_FA_LAYERS;
    for (int layer = 0; layer < num_fa; layer++) {
        __half* k_dst = pool_->k_ptr(layer, req.physical_block_id);
        __half* v_dst = pool_->v_ptr(layer, req.physical_block_id);
        size_t kv_bytes = C::KV_BLOCK_ELEMS * sizeof(__half);

        cudaMemcpyAsync(k_dst, static_cast<char*>(staging) + offset,
                        kv_bytes, cudaMemcpyHostToDevice, copy_stream_);
        offset += kv_bytes;

        cudaMemcpyAsync(v_dst, static_cast<char*>(staging) + offset,
                        kv_bytes, cudaMemcpyHostToDevice, copy_stream_);
        offset += kv_bytes;
    }
    cudaStreamSynchronize(copy_stream_);

    return true;
}

} // namespace deusridet
