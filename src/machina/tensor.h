// tensor.h — Lightweight tensor descriptor
//
// Non-owning view or allocator-backed storage for multi-dimensional data.
// Designed for Tegra iGPU where mmap'd weights are directly GPU-accessible.

#pragma once

#include "allocator.h"
#include <vector>
#include <memory>
#include <numeric>
#include <functional>

namespace deusridet {

class Tensor {
public:
    // Construct with allocator (owns memory)
    Tensor(const std::vector<int64_t>& shape, DataType dtype,
           std::shared_ptr<Allocator> allocator)
        : shape_(shape), dtype_(dtype),
          numel_(compute_numel(shape)),
          nbytes_(numel_ * dtype_size(dtype)),
          allocator_(allocator)
    {
        data_ptr_ = allocator_->allocate(nbytes_);
    }

    // Construct from external pointer (non-owning view, e.g. mmap'd weights)
    Tensor(const std::vector<int64_t>& shape, DataType dtype, void* data_ptr)
        : shape_(shape), dtype_(dtype),
          numel_(compute_numel(shape)),
          nbytes_(numel_ * dtype_size(dtype)),
          data_ptr_(data_ptr)
    {}

    // Construct with explicit byte count (for packed formats like INT4)
    Tensor(const std::vector<int64_t>& shape, DataType dtype,
           void* data_ptr, size_t explicit_nbytes)
        : shape_(shape), dtype_(dtype),
          numel_(compute_numel(shape)),
          nbytes_(explicit_nbytes),
          data_ptr_(data_ptr)
    {}

    ~Tensor() {
        if (allocator_ && data_ptr_) {
            allocator_->deallocate(data_ptr_);
        }
    }

    // Move only
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept
        : shape_(std::move(o.shape_)), dtype_(o.dtype_),
          numel_(o.numel_), nbytes_(o.nbytes_),
          data_ptr_(o.data_ptr_), allocator_(std::move(o.allocator_))
    {
        o.data_ptr_ = nullptr;
    }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            if (allocator_ && data_ptr_) allocator_->deallocate(data_ptr_);
            shape_ = std::move(o.shape_);
            dtype_ = o.dtype_;
            numel_ = o.numel_;
            nbytes_ = o.nbytes_;
            data_ptr_ = o.data_ptr_;
            allocator_ = std::move(o.allocator_);
            o.data_ptr_ = nullptr;
        }
        return *this;
    }

    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype()  const { return dtype_; }
    void*    data()   const { return data_ptr_; }
    size_t   numel()  const { return numel_; }
    size_t   nbytes() const { return nbytes_; }

    template<typename T>
    T* data_as() const { return static_cast<T*>(data_ptr_); }

    static size_t compute_numel(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), (size_t)1,
                               std::multiplies<size_t>());
    }

private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    size_t   numel_;
    size_t   nbytes_;
    void*    data_ptr_ = nullptr;
    std::shared_ptr<Allocator> allocator_;
};

} // namespace deusridet
