// allocator.cpp — Memory allocator implementations for Tegra iGPU
//
// Adapted from qwen35-orin (src/engine/allocator.cpp): allocator architecture
// and mmap strategy adapted to DeusRidet's consciousness-centric design.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "allocator.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace deusridet {

// ============================================================================
// DeviceAllocator
// ============================================================================

void* DeviceAllocator::allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc failed (") + std::to_string(size) +
            " bytes): " + cudaGetErrorString(err));
    }
    return ptr;
}

void DeviceAllocator::deallocate(void* ptr) {
    if (ptr) cudaFree(ptr);
}

// ============================================================================
// UnifiedAllocator
// ============================================================================

void* UnifiedAllocator::allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMallocManaged failed (") + std::to_string(size) +
            " bytes): " + cudaGetErrorString(err));
    }
    return ptr;
}

void UnifiedAllocator::deallocate(void* ptr) {
    if (ptr) cudaFree(ptr);
}

// ============================================================================
// MmapAllocator
// ============================================================================
//
// Adaptive mmap strategy based on file size vs available memory:
//   - Small files (< 25% of MemAvailable): MAP_POPULATE for bulk preread
//   - Large files: on-demand faulting with MADV_WILLNEED async preread
//
// Adapted from qwen35-orin (src/engine/allocator.cpp): mmap strategy with
// adaptive populate/willneed based on available memory.

MmapAllocator::MmapAllocator(const std::string& file_path)
    : fd_(-1), base_ptr_(MAP_FAILED), size_(0)
{
    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file for mmap: " + file_path);
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Failed to stat file: " + file_path);
    }
    size_ = sb.st_size;

    // Query available memory to decide mmap strategy
    size_t avail_kb = 0;
    {
        FILE* f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "MemAvailable:", 13) == 0) {
                    sscanf(line + 13, " %zu", &avail_kb);
                    break;
                }
            }
            fclose(f);
        }
    }
    size_t avail_bytes = avail_kb * 1024;
    bool use_populate = (avail_bytes > 0 && size_ < avail_bytes / 4);

    int flags = MAP_PRIVATE;
    if (use_populate) flags |= MAP_POPULATE;

    base_ptr_ = mmap(nullptr, size_, PROT_READ, flags, fd_, 0);
    if (base_ptr_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for file: " + file_path);
    }

    if (use_populate) {
        madvise(base_ptr_, size_, MADV_HUGEPAGE);
    } else {
        madvise(base_ptr_, size_, MADV_SEQUENTIAL);
        madvise(base_ptr_, size_, MADV_WILLNEED);
    }

    // Note: cudaHostRegister does not work with PROT_READ-only mmap on Tegra.
    // Weight data must be explicitly copied to device memory for GPU access.
    // This is actually preferred for inference weights: device memory avoids
    // coherency overhead on Tegra iGPU.
}

MmapAllocator::~MmapAllocator() {
    if (base_ptr_ != MAP_FAILED) {
        munmap(base_ptr_, size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

} // namespace deusridet
