// allocator.h — Memory allocators for Tegra unified memory architecture
//
// Three allocator types optimized for Jetson AGX Orin (SM87):
//   - DeviceAllocator:  cudaMalloc for GPU-only buffers (KV Cache, activations)
//   - UnifiedAllocator: cudaMallocManaged for CPU/GPU shared data
//   - MmapAllocator:    mmap for zero-copy safetensors weight loading
//
// On Tegra iGPU, CPU and GPU share the same physical DRAM. cudaMalloc still
// allocates from system DRAM but creates GPU page table mappings directly —
// no coherency overhead compared to managed memory.
// See: CUDA for Tegra Application Note

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace deusridet {

// Data type enumeration
enum class DataType {
    FP32,
    FP16,
    BF16,
    INT32,    // GPTQ qweight/qzeros packed container
    INT8,
    U8,
    INT4,     // GPTQ packed (2 values per byte)
    UNKNOWN
};

inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:  return 4;
        case DataType::FP16:  return 2;
        case DataType::BF16:  return 2;
        case DataType::INT32: return 4;
        case DataType::INT8:  return 1;
        case DataType::U8:    return 1;
        case DataType::INT4:  return 0;  // special: 2 values per byte
        default: throw std::runtime_error("Unknown data type");
    }
}

inline const char* dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:  return "FP32";
        case DataType::FP16:  return "FP16";
        case DataType::BF16:  return "BF16";
        case DataType::INT32: return "INT32";
        case DataType::INT8:  return "INT8";
        case DataType::U8:    return "U8";
        case DataType::INT4:  return "INT4";
        default:              return "UNKNOWN";
    }
}

// Abstract allocator interface
class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void  deallocate(void* ptr) = 0;
};

// Device memory allocator (cudaMalloc)
// Preferred for GPU-only buffers: KV Cache blocks, intermediate activations,
// scratch space. Avoids coherency overhead on Tegra iGPU.
class DeviceAllocator : public Allocator {
public:
    void* allocate(size_t size) override;
    void  deallocate(void* ptr) override;
};

// Unified memory allocator (cudaMallocManaged)
// For data accessed by both CPU and GPU. On Tegra, this is physically the
// same DRAM but with coherency management at kernel launch/sync boundaries.
class UnifiedAllocator : public Allocator {
public:
    void* allocate(size_t size) override;
    void  deallocate(void* ptr) override;
};

// Memory-mapped file allocator for zero-copy weight loading
// Maps an entire file into the process address space via mmap.
// On Tegra iGPU, mmap'd regions are directly accessible by GPU without
// explicit copy — the CPU page tables are sufficient.
class MmapAllocator {
public:
    explicit MmapAllocator(const std::string& file_path);
    ~MmapAllocator();

    MmapAllocator(const MmapAllocator&) = delete;
    MmapAllocator& operator=(const MmapAllocator&) = delete;

    void*  base_ptr() const { return base_ptr_; }
    size_t size()     const { return size_; }

private:
    int    fd_;
    void*  base_ptr_;
    size_t size_;
};

} // namespace deusridet
