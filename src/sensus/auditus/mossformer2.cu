/**
 * @file src/sensus/auditus/mossformer2.cu
 * @philosophical_role
 *   MossFormer2 native CUDA inference — entry TU. After R1 800-line split,
 *   this file holds only the constructor/destructor; kernels live in
 *   mossformer2_kernels.cuh, methods in mossformer2_lifecycle.cu and
 *   mossformer2_flash.cu.
 * @serves
 *   Auditus pipeline (audio_pipeline.cpp) multi-speaker path.
 */
// mossformer2.cu — entry TU after split.
//
// Adapted from ClearerVoice-Studio MossFormer2 (Apache-2.0).
// Original: https://github.com/modelscope/ClearerVoice-Studio

#include "mossformer2.h"
#include "mossformer2_kernels.cuh"
#include "mossformer2_internal.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace deusridet {

using namespace mf2;

MossFormer2::MossFormer2() = default;

MossFormer2::~MossFormer2() {
    free_scratch();
    if (cublas_) cublasDestroy(cublas_);
    if (d_weights_) { cudaFree(d_weights_); d_weights_ = nullptr; }
}

}  // namespace deusridet
