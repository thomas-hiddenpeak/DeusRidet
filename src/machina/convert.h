// convert.h — Device-side data type conversion utilities

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace deusridet {

// Convert BF16 → FP16 in-place on device (both 2 bytes per element).
// Optionally adds 1.0f to each value (for RMSNorm weight precomputation).
// d_ptr contains raw BF16 bytes on entry, FP16 values on exit.
void bf16_to_fp16_gpu(void* d_ptr, size_t numel, bool add_one, cudaStream_t stream);

// Add 1.0f to FP16 values in-place on device (for RMSNorm weight precomputation).
void fp16_add_one_gpu(__half* d_ptr, size_t numel, cudaStream_t stream);

} // namespace deusridet
