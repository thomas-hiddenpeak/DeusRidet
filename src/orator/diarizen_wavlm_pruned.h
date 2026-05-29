/**
 * @file diarizen_wavlm_pruned.h
 * @philosophical_role Native CUDA replacement for the WavLM-pruned encoder
 *     stage (S) of the DiariZen-v2 segmentation pipeline. P1a of the native
 *     port roadmap; see docs/{en,zh}/architecture/12-diarizen.md and the
 *     "Architectural anchor" section. Tools belongs on the GPU; CPU is
 *     orchestration only (philosophy.instructions.md "Compute Belongs on
 *     the GPU"). This loader-only step is the first verifiable chunk:
 *     map all 449 safetensors entries to GPU pointers, expose per-layer
 *     pruned dimensions, and validate against shapes.json. Forward pass
 *     ships in P1a-step2.
 * @serves Orator subsystem - replaces tools/diarizen_worker.py for the
 *     S-stage. The downstream Conformer head (P1b) consumes
 *     `wavlm_lnorm_out` produced by this class.
 *
 * Weight provenance: BUT-FIT/diarizen-wavlm-large-s80-md-v2, converted to
 * a single safetensors at ~/models/dev/diarizen_v2/wavlm_pruned.safetensors
 * (449 tensors, float16, ~127 MB). Pruning is heterogeneous: each
 * transformer layer has its own (head_dim_total, ffn_intermediate) shape,
 * and the CNN feature extractor's last layer width is 211 instead of 512.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_fp16.h>

namespace deusridet {
namespace orator {

// One transformer layer's pruned per-tensor shapes. Captured at load time
// so the forward pass can launch the right GEMM sizes without re-reading
// safetensors metadata. Hidden width (1024) is the model invariant and is
// stored once on the outer struct.
struct DiarizenWavlmPrunedLayerDims {
    int layer_index   = -1;
    int attn_inner    = 0;   ///< k/q/v projection out width = num_heads * 64
                             ///< (e.g. 320=5h, 192=3h, ...); 0 = entire
                             ///< attention sub-block pruned away (known:
                             ///< layers 9, 12, 16, 17 in s80-md-v2).
    int num_heads     = 0;   ///< attn_inner / kHeadDim; 0 when attention pruned.
                             ///< BUT-FIT structured pruning removes whole
                             ///< heads (head_dim stays 64); it does NOT
                             ///< narrow per-head width.
    int attn_head_dim = 0;   ///< always kHeadDim (64) when present, else 0.
    int ffn_inner    = 0;   ///< intermediate_dense out width (e.g. 1092); always > 0
    int gru_rel_pos_inner = 0;  ///< gru_rel_pos_linear out width (8 in known weights);
                                ///< 0 when attention is pruned away
};

// Compile-time-ish invariants of the wavlm-large-s80-md-v2 architecture
// (verified at load time against the weight shapes).
struct DiarizenWavlmPrunedArch {
    static constexpr int kSampleRate          = 16000;
    static constexpr int kCnnLayers           = 7;
    static constexpr int kTransformerLayers   = 24;
    static constexpr int kNumAttnHeads        = 16;   ///< total_num_heads (gru gate + rel-pos use all 16)
    static constexpr int kHeadDim             = 64;   ///< embed_dim / total_num_heads, fixed across pruning
    static constexpr int kHiddenDim           = 1024;  ///< post feature_projection.projection
    static constexpr int kFeatProjInDim       = 211;   ///< CNN out width (pruned)
    static constexpr int kFinalProjOutDim     = 256;   ///< proj/lnorm output (P1a tap)
    static constexpr int kLayerTaps           = 25;    ///< CNN + 24 transformer
    static constexpr int kCnnStrideTotal      = 320;   ///< 16 kHz / 320 -> 50 Hz frames
};

// A single GPU buffer view: pointer + element count (no element-type
// templating to keep the table flat; everything is fp16 except a few
// running-mean tensors which we coerce at load time).
struct DiarizenWavlmPrunedTensorView {
    const __half* data    = nullptr;
    std::size_t   numel   = 0;
    int           dim     = 0;
    int           shape[4] = {0, 0, 0, 0};  ///< up to 4-D is sufficient
};

/**
 * @brief CUDA-side WavLM-pruned encoder (loader-only in P1a-step1).
 *
 * Owns one contiguous fp16 GPU arena that backs every tensor; the per-
 * tensor views in `tensors_` are offsets into that arena. This keeps the
 * allocation count to O(1) cuda calls regardless of the 449 tensors
 * stored, which matters on Tegra where every cudaMalloc walks the global
 * VMM map.
 *
 * Forward(), per-layer hidden tap collection, and the weight-sum + proj +
 * lnorm tail (= "wavlm_lnorm_out" of the Python reference) are NOT yet
 * implemented; they are P1a-step2 / P1a-step3 follow-ups.
 */
class DiarizenWavlmPruned {
public:
    DiarizenWavlmPruned();
    ~DiarizenWavlmPruned();

    DiarizenWavlmPruned(const DiarizenWavlmPruned&)            = delete;
    DiarizenWavlmPruned& operator=(const DiarizenWavlmPruned&) = delete;

    /// Load weights from a single safetensors file (no sharding for the
    /// pruned model - it fits in one 127 MB file). Returns false and
    /// emits LOG_ERROR on failure; the object is left in an unloaded
    /// state and forward() will be unusable.
    bool load(const std::string& safetensors_path);

    bool is_loaded() const { return loaded_; }

    /// Bytes allocated on the GPU (single contiguous arena).
    std::size_t arena_bytes() const { return arena_bytes_; }

    /// Number of tensors registered in the view table (should equal the
    /// safetensors entry count once load() succeeds).
    std::size_t tensor_count() const { return tensors_.size(); }

    /// Per-layer pruned dimensions, indexed 0..23. Empty until load().
    const std::vector<DiarizenWavlmPrunedLayerDims>& layer_dims() const {
        return layer_dims_;
    }

    /// Per-layer surviving attention head ids (original 0..15 indices),
    /// loaded from the `<weights>_heads.json` sidecar. Size 24; an empty
    /// inner list means the layer's attention is fully pruned. Used by the
    /// forward pass to select rows of the [16, T, T] relative-position bias.
    const std::vector<int>& remaining_heads(int layer) const {
        return remaining_heads_[layer];
    }

    /// Look up a tensor by its safetensors name. Returns nullptr-bearing
    /// view if absent. Stable for the lifetime of the object.
    const DiarizenWavlmPrunedTensorView* find(const std::string& name) const;

    /// Diagnostic dump to stderr: per-layer attn_inner / ffn_inner and
    /// total arena bytes. Used by the smoke test target.
    void log_summary() const;

    /// P1a-step2a milestone: run ONLY the CNN feature extractor (7 pruned
    /// conv blocks -> per-frame channel LayerNorm -> exact-erf GELU ->
    /// dummy_weight scale). Input is host float32 PCM normalised to
    /// [-1, 1]; output is the [T, 211] feature map flattened row-major
    /// (frame-major). Returns empty on error. Lazily creates the cuDNN
    /// handle on first call. Used by the bit-equality test against the
    /// `cnn_out` tap of the Python reference dump.
    std::vector<float> debug_cnn_features(const float* pcm, int n_samples,
                                          int& T_out);

    /// P1a-step2b milestone: run the CNN feature extractor followed by the
    /// encoder front end (feature_projection LayerNorm+Linear -> positional
    /// convolution -> residual add). This equals tap 0 of the 25 WavLM
    /// taps consumed by weight_sum (the transformer.layer_norm is NOT part
    /// of any tap and is omitted, matching extract_features /
    /// get_intermediate_outputs). Output is the [T, 1024] hidden flattened
    /// frame-major. Returns empty on error. Bit-checked against the
    /// `layer_hiddens[0]` reference tap.
    std::vector<float> debug_tap0(const float* pcm, int n_samples, int& T_out);

    /// P1a-step2c milestone: run the encoder front end followed by the first
    /// `up_to_layer` transformer EncoderLayers (PRE-NORM WavLM-pruned, with
    /// gated relative-position attention and per-layer head pruning). With
    /// up_to_layer == 0 this equals tap 0; with up_to_layer == k it equals
    /// the reference `layer_hiddens[k]`. Output is the [T, 1024] hidden
    /// flattened frame-major. Returns empty on error.
    std::vector<float> debug_layers(const float* pcm, int n_samples,
                                    int up_to_layer, int& T_out);

    /// P1a-step2d milestone: full WavLM-pruned tail. Collects all 25 hidden
    /// taps (front end + 24 layers), forms the learned weighted sum
    /// (weight_sum.weight [1, 25], plain Linear with no bias), projects
    /// 1024 -> 256 (proj.weight/bias), then applies LayerNorm(256)
    /// (lnorm.weight/bias). Output is the [T, 256] feature flattened
    /// frame-major. Bit-checked against the `wavlm_lnorm_out` reference tap.
    std::vector<float> debug_lnorm_tail(const float* pcm, int n_samples,
                                        int& T_out);

private:
    bool loaded_ = false;
    void*       arena_  = nullptr;   ///< owned, GPU
    std::size_t arena_bytes_ = 0;

    std::unordered_map<std::string, DiarizenWavlmPrunedTensorView> tensors_;
    std::vector<DiarizenWavlmPrunedLayerDims> layer_dims_;
    std::vector<std::vector<int>> remaining_heads_;  ///< 24 lists, from sidecar

    // Lazily-created compute handles (forward path only; loader does not
    // need them). Declared void* to keep cudnn/cublas headers out of this
    // public header — the .cu casts them back.
    void* cudnn_  = nullptr;   ///< cudnnHandle_t
    void* cudnn_ws_ = nullptr; ///< conv workspace (GPU), grown on demand
    std::size_t cudnn_ws_bytes_ = 0;

    bool ensure_handles_();
    void release_();

    /// Internal: run the CNN feature extractor and return a freshly
    /// allocated GPU buffer of shape [T_out, 211] (frame-major). Caller
    /// owns the pointer and must cudaFree it. Returns nullptr on error.
    float* run_cnn_(const float* pcm, int n_samples, int& T_out);

    /// Internal: run CNN + feature_projection + pos_conv (= tap 0) and
    /// return a freshly allocated GPU buffer of shape [T_out, 1024]
    /// (frame-major). Caller owns the pointer and must cudaFree it. Used by
    /// both debug_tap0 and debug_layers. Returns nullptr on error.
    float* run_frontend_(const float* pcm, int n_samples, int& T_out);

    /// Internal (defined in diarizen_wavlm_pruned_layers.cu): run a single
    /// PRE-NORM transformer EncoderLayer in place on the device hidden
    /// buffer d_hidden [T, 1024]. `layer` selects the per-layer weights and
    /// pruned head set. `d_pos_bias` is the shared [16, T, T] relative
    /// position bias (computed once and reused); may be nullptr for layers
    /// whose attention is fully pruned. cublas/cudnn handles are passed in.
    bool run_encoder_layer_(int layer, float* d_hidden, int T,
                            const float* d_pos_bias, void* cublas);

    /// Internal: compute the shared [16, T, T] relative position bias from
    /// rel_attn_embed (layer 0 only). Returns a freshly allocated device
    /// buffer; caller frees. Returns nullptr on error.
    float* compute_position_bias_(int T);
};

}  // namespace orator
}  // namespace deusridet
