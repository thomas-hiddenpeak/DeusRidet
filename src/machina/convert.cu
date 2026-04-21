/**
 * @file src/machina/convert.cu
 * @philosophical_role
 *   BF16↔FP16 conversion — the quiet translator between weight-storage dtype (BF16 from safetensors) and compute dtype (FP16 for Tensor Cores). A boundary kernel, not a headline kernel.
 * @serves
 *   Machina weight loading (machina/safetensors → device allocator) and any path crossing a precision boundary.
 */
// convert.cu — Device-side data type conversion kernels
//
// BF16 → FP16 in-place: both are 16-bit, so the conversion reads each
// element as uint16_t (BF16 raw), converts via float, and writes back
// as __half.  Each thread handles one element — no aliasing issue.

#include "convert.h"
#include <cstdint>

namespace deusridet {

static __global__ void bf16_to_fp16_kernel(uint16_t* data, size_t n, bool add_one) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = __uint_as_float((uint32_t)data[i] << 16);
        if (add_one) val += 1.0f;
        reinterpret_cast<__half*>(data)[i] = __float2half(val);
    }
}

static __global__ void fp16_add_one_kernel(__half* data, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = __float2half(__half2float(data[i]) + 1.0f);
    }
}

void bf16_to_fp16_gpu(void* d_ptr, size_t numel, bool add_one, cudaStream_t stream) {
    if (numel == 0) return;
    const int block = 256;
    int grid = (int)((numel + block - 1) / block);
    bf16_to_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<uint16_t*>(d_ptr), numel, add_one);
}

void fp16_add_one_gpu(__half* d_ptr, size_t numel, cudaStream_t stream) {
    if (numel == 0) return;
    const int block = 256;
    int grid = (int)((numel + block - 1) / block);
    fp16_add_one_kernel<<<grid, block, 0, stream>>>(d_ptr, numel);
}

} // namespace deusridet
