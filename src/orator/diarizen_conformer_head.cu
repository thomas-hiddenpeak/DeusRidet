/**
 * @file diarizen_conformer_head.cu
 * @philosophical_role P1b of the DiariZen native CUDA port: the Conformer
 *     segmentation head (4 blocks of macaron-FFN / MHSA / depthwise-conv /
 *     macaron-FFN) plus the powerset classifier. Consumes the [T, 256]
 *     WavLM tail feature and emits per-frame logits. Every GEMM is cuBLAS;
 *     every norm/conv/softmax/activation is a CUDA kernel. The CPU only
 *     sequences the four blocks and owns the fp16 weight arena.
 * @serves DiarizenConformerHead. Bit-checked against the reference taps
 *     conformer_out / classifier_logits / classifier_probs.
 *
 * Verified against diarizen/models/module/conformer.py with use_posi=False
 * (pos_k=None) and output_activate_function=False:
 *   block(x): x=ffn1(x); x=mha(x); x=conv(x); x=ffn2(x); x=ln_norm(x)
 *   ffn(x):   res + 0.5 * w_2(swish(w_1(ln_norm(x))))
 *   mha(x):   res + linearO(MHSA(ln_norm(x)))            (scale 1/sqrt(64))
 *   conv(x):  res + pw2(swish(bn(dw(glu(pw1(ln_norm(x)^T))))))^T
 * classifier: Linear(256,16); activation: LogSoftmax(dim=-1).
 */
#include "diarizen_conformer_head.h"

#include "../communis/log.h"
#include "../machina/safetensors.h"
#include "../machina/tensor.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "diarizen_wavlm_pruned_kernels.cuh"

namespace deusridet {
namespace orator {

namespace {

constexpr const char* kCLog = "DiariZenConformer";

}  // namespace

// --------------------------------------------------------------------------
// Lifecycle + loader
// --------------------------------------------------------------------------
DiarizenConformerHead::DiarizenConformerHead() = default;
DiarizenConformerHead::~DiarizenConformerHead() { release_(); }

void DiarizenConformerHead::release_() {
    if (arena_) {
        cudaFree(arena_);
        arena_ = nullptr;
    }
    for (auto& kv : f32_cache_) cudaFree(kv.second);
    f32_cache_.clear();
    for (auto& kv : scratch_pool_)
        for (void* p : kv.second) cudaFree(p);
    scratch_pool_.clear();
    if (blas_) {
        cublasDestroy(static_cast<cublasHandle_t>(blas_));
        blas_ = nullptr;
    }
    arena_bytes_ = 0;
    tensors_.clear();
    loaded_ = false;
}

const DiarizenConformerTensorView* DiarizenConformerHead::find(
    const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool DiarizenConformerHead::load(const std::string& path) {
    release_();
    LOG_INFO(kCLog, "loading conformer-head weights: %s", path.c_str());
    SafetensorsFile sf(path);
    auto names = sf.tensor_names();
    if (names.empty()) {
        LOG_ERROR(kCLog, "no tensors found in %s", path.c_str());
        return false;
    }

    constexpr std::size_t kTensorAlign = 16;
    std::size_t total = 0;
    for (const auto& n : names) {
        auto t = sf.get_tensor(n);
        if (!t) {
            LOG_ERROR(kCLog, "tensor missing: %s", n.c_str());
            return false;
        }
        if (t->dtype() != DataType::FP16) {
            LOG_ERROR(kCLog, "tensor %s is not fp16 (dtype=%d)", n.c_str(),
                      static_cast<int>(t->dtype()));
            return false;
        }
        total += (t->nbytes() + kTensorAlign - 1) & ~(kTensorAlign - 1);
    }
    constexpr std::size_t kAlign = 256;
    const std::size_t arena_total = (total + kAlign - 1) & ~(kAlign - 1);
    LOG_INFO(kCLog, "allocating arena %.2f MB (%zu tensors)",
             arena_total / (1024.0 * 1024.0), names.size());
    if (cudaMalloc(&arena_, arena_total) != cudaSuccess) {
        LOG_ERROR(kCLog, "cudaMalloc arena failed");
        return false;
    }
    arena_bytes_ = arena_total;

    std::size_t cursor = 0;
    tensors_.reserve(names.size());
    for (const auto& n : names) {
        auto t = sf.get_tensor(n);
        const std::size_t nb = t->nbytes();
        const std::size_t aligned = (nb + kTensorAlign - 1) & ~(kTensorAlign - 1);
        if (cursor + aligned > arena_total) {
            LOG_ERROR(kCLog, "arena overflow on %s", n.c_str());
            release_();
            return false;
        }
        void* dst = static_cast<char*>(arena_) + cursor;
        if (cudaMemcpy(dst, t->data(), nb, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            LOG_ERROR(kCLog, "cudaMemcpy tensor %s failed", n.c_str());
            release_();
            return false;
        }
        DiarizenConformerTensorView v;
        v.data = static_cast<const __half*>(dst);
        v.numel = t->numel();
        const auto& shape = t->shape();
        v.dim = static_cast<int>(shape.size());
        for (int d = 0; d < v.dim && d < 4; ++d)
            v.shape[d] = static_cast<int>(shape[d]);
        tensors_.emplace(n, v);
        cursor += aligned;
    }
    loaded_ = true;
    LOG_INFO(kCLog, "loaded %zu tensors into %.2f MB arena", tensors_.size(),
             arena_bytes_ / (1024.0 * 1024.0));
    return true;
}

}  // namespace orator
}  // namespace deusridet
