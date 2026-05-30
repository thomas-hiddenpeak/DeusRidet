/**
 * @file diarizen_conformer_head.h
 * @philosophical_role Native CUDA replacement for the Conformer segmentation
 *     head (stage after WavLM) of the DiariZen-v2 pipeline. P1b of the native
 *     port roadmap; see docs/{en,zh}/architecture/12-diarizen.md. Consumes the
 *     [T, 256] feature produced by DiarizenWavlmPruned's weight_sum+proj+lnorm
 *     tail (P1a-step2d, the `wavlm_lnorm_out` tap) and produces per-frame
 *     powerset logits over 16 classes. Compute belongs on the GPU: every
 *     GEMM runs through cuBLAS, every elementwise/softmax/conv reduction is a
 *     CUDA kernel; the CPU only sequences the four Conformer blocks.
 * @serves Orator subsystem — replaces the C-stage of tools/diarizen_worker.py.
 *
 * Weight provenance: BUT-FIT/diarizen-wavlm-large-s80-md-v2 non-WavLM
 * parameters, converted to ~/models/dev/diarizen_v2/conformer_head.safetensors
 * (150 tensors, float16). Architecture (verified on the loaded model):
 * ConformerEncoder(attention_in=256, ffn_hidden=1024, num_head=4,
 * num_layer=4, kernel_size=31, use_posi=False, output_activate_function=False)
 * followed by classifier Linear(256, 16) and LogSoftmax(dim=-1).
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_fp16.h>

namespace deusridet {
namespace orator {

// Compile-time invariants of the DiariZen-v2 Conformer head, validated at
// load time against the weight shapes.
struct DiarizenConformerArch {
    static constexpr int kFeatDim     = 256;   ///< attention_in
    static constexpr int kFfnHidden   = 1024;  ///< positionwise FFN width
    static constexpr int kNumHead     = 4;     ///< MHSA heads
    static constexpr int kHeadDim     = 64;    ///< kFeatDim / kNumHead
    static constexpr int kNumLayer    = 4;     ///< Conformer blocks
    static constexpr int kKernelSize  = 31;    ///< depthwise conv (SAME pad 15)
    static constexpr int kNumClasses  = 16;    ///< powerset classifier out
};

// A single GPU buffer view: pointer + element count. Everything is fp16.
struct DiarizenConformerTensorView {
    const __half* data    = nullptr;
    std::size_t   numel   = 0;
    int           dim     = 0;
    int           shape[4] = {0, 0, 0, 0};
};

/**
 * @brief CUDA-side DiariZen Conformer segmentation head.
 *
 * Owns one contiguous fp16 GPU arena that backs every tensor; the per-tensor
 * views are offsets into that arena (O(1) cudaMalloc on Tegra). forward()
 * takes the [T, 256] WavLM feature on host, runs the four Conformer blocks +
 * classifier on the GPU, and returns per-frame logits (or log-probs).
 */
class DiarizenConformerHead {
public:
    DiarizenConformerHead();
    ~DiarizenConformerHead();

    DiarizenConformerHead(const DiarizenConformerHead&)            = delete;
    DiarizenConformerHead& operator=(const DiarizenConformerHead&) = delete;

    /// Load weights from a single safetensors file (all fp16). Returns false
    /// and emits LOG_ERROR on failure.
    bool load(const std::string& safetensors_path);

    bool is_loaded() const { return loaded_; }
    std::size_t arena_bytes() const { return arena_bytes_; }
    std::size_t tensor_count() const { return tensors_.size(); }

    /// Look up a tensor by its safetensors name. Returns nullptr-bearing view
    /// if absent. Stable for the object lifetime.
    const DiarizenConformerTensorView* find(const std::string& name) const;

    /// Persistent fp32 weight cache: converts an fp16 arena tensor to a
    /// device-resident fp32 buffer on first use and returns the cached pointer
    /// on every subsequent call. Eliminates the per-chunk
    /// cudaMalloc/convert/cudaFree churn (each Tegra cudaMalloc/Free walks the
    /// global VMM map) that dominated the Conformer seg stage. Owned by the
    /// head; callers MUST NOT free the result. Mirrors the WavLM weight_f32.
    float* weight_f32(const std::string& name) const;

    /// P1b bit-equality tap: run the four Conformer blocks over the [T, 256]
    /// feature `feat` (host, frame-major) and return the conformer output
    /// [T, 256] flattened frame-major. Bit-checked vs `conformer_out`.
    std::vector<float> debug_conformer(const float* feat, int T);

    /// P1b bit-equality tap: conformer + classifier linear, returns logits
    /// [T, 16]. Bit-checked vs `classifier_logits`.
    std::vector<float> debug_logits(const float* feat, int T);

    /// P1b bit-equality tap: + LogSoftmax(dim=-1), returns log-probs [T, 16].
    /// Bit-checked vs `classifier_probs`.
    std::vector<float> debug_probs(const float* feat, int T);

private:
    // Internal: run conformer blocks (+ optional classifier/logsoftmax) over a
    // GPU [T, 256] buffer. stage: 0 = conformer only, 1 = +classifier,
    // 2 = +logsoftmax. Returns a fresh host vector; empty on error.
    std::vector<float> run_(const float* feat, int T, int stage);
    bool run_block_(int layer, float* d_x, int T, void* cublas);

    void release_();

    bool        loaded_      = false;
    void*       arena_       = nullptr;   ///< owned, GPU
    std::size_t arena_bytes_ = 0;
    std::unordered_map<std::string, DiarizenConformerTensorView> tensors_;
    mutable std::unordered_map<std::string, float*> f32_cache_;  ///< owned, GPU
};

}  // namespace orator
}  // namespace deusridet
