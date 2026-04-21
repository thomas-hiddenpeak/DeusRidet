/**
 * @file src/machina/gptq_internal.cuh
 * @philosophical_role
 *   TU-local INT4 dequant helpers shared across the gptq.cu peer split.
 *   Textual include only — no symbols escape the including TU.
 * @serves
 *   gptq.cu, gptq_gemv.cu, gptq_gemm.cu, gptq_wmma_*.cu, gptq_int8_wmma.cu.
 */
#pragma once

#include "gptq.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace deusridet {

static __device__ __forceinline__ int extract_int4(uint32_t packed, int index) {
    return (packed >> (index * 4)) & 0xF;
}

static __device__ __forceinline__ __half dequant_int4(__half scale, int q_val) {
    float w = __half2float(scale) * (float)(q_val - GPTQ_ZERO_POINT);
    return __float2half(w);
}

} // namespace deusridet
