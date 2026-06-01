/**
 * @file diarizen_wavlm_pruned.cu
 * @philosophical_role P1a-step1 of the DiariZen native CUDA port. Loader
 *     only: maps every fp16 tensor of the BUT-FIT wavlm_pruned.safetensors
 *     into a single contiguous GPU arena, captures per-layer pruned
 *     dimensions, and validates the architecture against the known
 *     wavlm-large-s80-md-v2 invariants. Forward pass deliberately omitted
 *     and lives in P1a-step2.
 * @serves DiarizenWavlmPruned. The Conformer head (P1b) is its sole
 *     consumer once forward() lands; until then the smoke-test target
 *     `test_diarizen_wavlm_pruned_loader` exercises the loader path.
 */
#include "diarizen_wavlm_pruned.h"

#include "../communis/log.h"
#include "../machina/safetensors.h"
#include "../machina/tensor.h"

#include <cstring>
#include <cstdio>
#include <regex>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace deusridet {
namespace orator {

namespace {

constexpr const char* kLog = "DiariZenWavlm";

inline bool cuda_ok_(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        LOG_ERROR(kLog, "CUDA %s failed: %s", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// Parse a `wavlm_model.encoder.transformer.layers.<N>.<rest>` key. Returns
// (layer_index, "<rest>"). Returns (-1, name) when the key does not match,
// so we can skip non-layer tensors quickly.
struct LayerKey {
    int  layer = -1;
    std::string rest;
};

LayerKey parse_layer_key_(const std::string& name) {
    static const std::regex re(
        R"(^wavlm_model\.encoder\.transformer\.layers\.(\d+)\.(.+)$)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return {-1, name};
    return {std::stoi(m[1].str()), m[2].str()};
}

}  // namespace

// --------------------------------------------------------------------------
// Lifecycle
// --------------------------------------------------------------------------
DiarizenWavlmPruned::DiarizenWavlmPruned() = default;

DiarizenWavlmPruned::~DiarizenWavlmPruned() {
    release_();
}

void DiarizenWavlmPruned::release_() {
    if (cudnn_ws_) {
        cudaFree(cudnn_ws_);
        cudnn_ws_ = nullptr;
        cudnn_ws_bytes_ = 0;
    }
    if (cudnn_) {
        cudnnDestroy(static_cast<cudnnHandle_t>(cudnn_));
        cudnn_ = nullptr;
    }
    if (arena_) {
        cudaFree(arena_);
        arena_ = nullptr;
    }
    arena_bytes_ = 0;
    for (auto& kv : f32_cache_) {
        if (kv.second) cudaFree(kv.second);
    }
    f32_cache_.clear();
    for (auto& kv : scratch_pool_) {
        for (void* p : kv.second) cudaFree(p);
    }
    scratch_pool_.clear();
    tensors_.clear();
    layer_dims_.clear();
    remaining_heads_.clear();
    loaded_ = false;
}

void DiarizenWavlmPruned::release_scratch() {
    // Vires V3 glymphatic clearance: drop only the transient forward
    // by-products (the size-keyed scratch free-list and the on-demand cuDNN
    // conv workspace). Persistent weights (arena_, f32_cache_), the cuDNN
    // handle, and loaded_ are kept, so the encoder stays ready; the pool and
    // workspace lazily re-grow on the next forward and contents are always
    // overwritten before read, making this bit-equivalent to fresh allocation.
    for (auto& kv : scratch_pool_) {
        for (void* p : kv.second) cudaFree(p);
    }
    scratch_pool_.clear();
    if (cudnn_ws_) {
        cudaFree(cudnn_ws_);
        cudnn_ws_ = nullptr;
        cudnn_ws_bytes_ = 0;
    }
}

// --------------------------------------------------------------------------
// load
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
bool DiarizenWavlmPruned::load(const std::string& path) {
    release_();

    LOG_INFO(kLog, "loading wavlm-pruned weights: %s", path.c_str());
    SafetensorsFile sf(path);
    auto names = sf.tensor_names();
    if (names.empty()) {
        LOG_ERROR(kLog, "no tensors found in %s", path.c_str());
        return false;
    }

    // Pass 1: validate dtype + accumulate arena bytes. Match the per-
    // tensor alignment used in pass 2 (16 bytes) so the running total
    // here matches the cursor walk below; otherwise the last tensors
    // overflow the arena.
    constexpr std::size_t kTensorAlign = 16;
    std::size_t total = 0;
    for (const auto& n : names) {
        auto t = sf.get_tensor(n);
        if (!t) {
            LOG_ERROR(kLog, "tensor missing during enumeration: %s",
                      n.c_str());
            return false;
        }
        if (t->dtype() != DataType::FP16) {
            LOG_ERROR(kLog,
                      "tensor %s is not fp16 (got dtype=%d); the "
                      "wavlm-large-s80-md-v2 conversion should produce "
                      "only fp16 tensors",
                      n.c_str(), static_cast<int>(t->dtype()));
            return false;
        }
        total += (t->nbytes() + kTensorAlign - 1) & ~(kTensorAlign - 1);
    }

    // Pad to 256-byte alignment for safety against any future kernel that
    // wants cuBLAS-friendly bases.
    constexpr std::size_t kAlign = 256;
    const std::size_t arena_total = (total + kAlign - 1) & ~(kAlign - 1);
    LOG_INFO(kLog, "allocating arena %.2f MB (%zu tensors, raw %.2f MB)",
             arena_total / (1024.0 * 1024.0), names.size(),
             total / (1024.0 * 1024.0));

    if (!cuda_ok_(cudaMalloc(&arena_, arena_total), "cudaMalloc arena"))
        return false;
    arena_bytes_ = arena_total;

    // Pass 2: copy each tensor into the arena and record its view.
    std::size_t cursor = 0;
    tensors_.reserve(names.size());
    layer_dims_.assign(DiarizenWavlmPrunedArch::kTransformerLayers, {});
    for (int i = 0; i < DiarizenWavlmPrunedArch::kTransformerLayers; ++i) {
        layer_dims_[i].layer_index = i;
    }

    for (const auto& n : names) {
        auto t = sf.get_tensor(n);
        const std::size_t nb = t->nbytes();
        // Per-tensor alignment too (small enough vs arena).
        const std::size_t aligned = (nb + kTensorAlign - 1) & ~(kTensorAlign - 1);
        if (cursor + aligned > arena_total) {
            LOG_ERROR(kLog,
                      "arena overflow on tensor %s (cursor=%zu, need=%zu)",
                      n.c_str(), cursor, aligned);
            release_();
            return false;
        }
        void* dst = static_cast<char*>(arena_) + cursor;
        if (!cuda_ok_(cudaMemcpy(dst, t->data(), nb, cudaMemcpyHostToDevice),
                      "cudaMemcpy tensor")) {
            release_();
            return false;
        }

        DiarizenWavlmPrunedTensorView v;
        v.data  = static_cast<const __half*>(dst);
        v.numel = t->numel();
        const auto& shape = t->shape();
        v.dim = static_cast<int>(shape.size());
        for (int d = 0; d < v.dim && d < 4; ++d) {
            v.shape[d] = static_cast<int>(shape[d]);
        }
        tensors_.emplace(n, v);

        // Capture per-layer pruned dims.
        const auto lk = parse_layer_key_(n);
        if (lk.layer >= 0 && lk.layer < (int)layer_dims_.size()) {
            auto& dims = layer_dims_[lk.layer];
            if (lk.rest == "attention.k_proj.weight" && v.dim == 2) {
                dims.attn_inner = v.shape[0];
                dims.attn_head_dim = DiarizenWavlmPrunedArch::kHeadDim;
                dims.num_heads =
                    v.shape[0] / DiarizenWavlmPrunedArch::kHeadDim;
            } else if (lk.rest == "feed_forward.intermediate_dense.weight"
                       && v.dim == 2) {
                dims.ffn_inner = v.shape[0];
            } else if (lk.rest == "attention.gru_rel_pos_linear.weight"
                       && v.dim == 2) {
                dims.gru_rel_pos_inner = v.shape[0];
            }
        }

        cursor += aligned;
    }

    // Final validation: ffn must exist on every layer; attention can be
    // fully pruned away (BUT-FIT's structured pruning removes the entire
    // attention sub-block on some layers — known cases: layers 9, 12,
    // 16, 17 in wavlm-large-s80-md-v2). attn_inner == 0 is a legal
    // "no-op attention" marker; the forward pass (P1a-step2) treats
    // such a layer as residual + FFN only.
    for (const auto& d : layer_dims_) {
        if (d.ffn_inner <= 0) {
            LOG_ERROR(kLog,
                      "layer %d missing ffn_inner (got %d) - checkpoint "
                      "structurally broken",
                      d.layer_index, d.ffn_inner);
            release_();
            return false;
        }
        if (d.attn_inner > 0 &&
            d.attn_inner % DiarizenWavlmPrunedArch::kHeadDim != 0) {
            LOG_ERROR(kLog,
                      "layer %d attn_inner=%d not divisible by head_dim=%d",
                      d.layer_index, d.attn_inner,
                      DiarizenWavlmPrunedArch::kHeadDim);
            release_();
            return false;
        }
    }

    // Sanity-check the top-level tail tensors against the architectural
    // constants so a wrong checkpoint fails loudly here, not 30 minutes
    // into the first segmentation pass.
    auto must = [&](const std::string& name, int d0, int d1) -> bool {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            LOG_ERROR(kLog, "missing required tensor: %s", name.c_str());
            return false;
        }
        const auto& v = it->second;
        if (v.shape[0] != d0 || (d1 > 0 && v.shape[1] != d1)) {
            LOG_ERROR(kLog,
                      "tensor %s shape mismatch: got [%d,%d], expected [%d,%d]",
                      name.c_str(), v.shape[0], v.shape[1], d0, d1);
            return false;
        }
        return true;
    };
    const bool tail_ok =
        must("weight_sum.weight", 1,
             DiarizenWavlmPrunedArch::kLayerTaps) &&
        must("proj.weight",
             DiarizenWavlmPrunedArch::kFinalProjOutDim,
             DiarizenWavlmPrunedArch::kHiddenDim) &&
        must("lnorm.weight",
             DiarizenWavlmPrunedArch::kFinalProjOutDim, 0) &&
        must("wavlm_model.encoder.feature_projection.projection.weight",
             DiarizenWavlmPrunedArch::kHiddenDim,
             DiarizenWavlmPrunedArch::kFeatProjInDim);
    if (!tail_ok) {
        release_();
        return false;
    }

    // Load the remaining_heads sidecar (<weights w/o .safetensors>_heads.json).
    // The surviving attention-head ids are not stored in safetensors; they are
    // exported by tools/diarizen_dump_reference.py --dump-heads. Parsing is a
    // tiny one-shot integer op so it stays on the CPU (anti-entropy: do NOT
    // hardcode the 24-list table in the .cu).
    {
        std::string side = path;
        const std::string ext = ".safetensors";
        if (side.size() > ext.size() &&
            side.compare(side.size() - ext.size(), ext.size(), ext) == 0)
            side = side.substr(0, side.size() - ext.size());
        side += "_heads.json";

        remaining_heads_.assign(DiarizenWavlmPrunedArch::kTransformerLayers, {});
        FILE* hf = std::fopen(side.c_str(), "rb");
        if (!hf) {
            LOG_ERROR(kLog, "missing remaining_heads sidecar: %s", side.c_str());
            release_();
            return false;
        }
        std::fseek(hf, 0, SEEK_END);
        long hn = std::ftell(hf);
        std::fseek(hf, 0, SEEK_SET);
        std::string js((size_t)hn, '\0');
        size_t hgot = std::fread(&js[0], 1, (size_t)hn, hf);
        std::fclose(hf);
        js.resize(hgot);

        // Minimal parser: find "remaining_heads", then scan nested [ ... ]
        // integer lists. The outer array holds exactly kTransformerLayers
        // inner arrays.
        size_t pos = js.find("remaining_heads");
        if (pos == std::string::npos) pos = 0;
        pos = js.find('[', pos);          // outer array open
        int li = -1;
        bool in_inner = false;
        while (pos < js.size() &&
               li < DiarizenWavlmPrunedArch::kTransformerLayers) {
            char c = js[pos];
            if (!in_inner) {
                if (c == '[') { ++li; in_inner = true; }
                else if (c == ']') break;     // outer close
                ++pos;
            } else {
                if (c == ']') { in_inner = false; ++pos; continue; }
                if (c == '-' || (c >= '0' && c <= '9')) {
                    long v = std::strtol(js.c_str() + pos, nullptr, 10);
                    if (li >= 0 && li < (int)remaining_heads_.size())
                        remaining_heads_[li].push_back((int)v);
                    // advance past the number
                    if (c == '-') ++pos;
                    while (pos < js.size() && js[pos] >= '0' && js[pos] <= '9')
                        ++pos;
                } else {
                    ++pos;
                }
            }
        }
        // Sanity: surviving-head count must equal num_heads parsed from
        // the safetensors projection widths.
        for (int i = 0; i < DiarizenWavlmPrunedArch::kTransformerLayers; ++i) {
            if ((int)remaining_heads_[i].size() != layer_dims_[i].num_heads) {
                LOG_ERROR(kLog,
                          "layer %d head mismatch: sidecar %zu vs weights %d",
                          i, remaining_heads_[i].size(),
                          layer_dims_[i].num_heads);
                release_();
                return false;
            }
        }
    }

    loaded_ = true;
    LOG_INFO(kLog,
             "loaded %zu tensors into %.2f MB arena; %d transformer layers",
             tensors_.size(),
             arena_bytes_ / (1024.0 * 1024.0),
             DiarizenWavlmPrunedArch::kTransformerLayers);
    return true;
}

// --------------------------------------------------------------------------
// Accessors / diagnostics
// --------------------------------------------------------------------------
const DiarizenWavlmPrunedTensorView*
DiarizenWavlmPruned::find(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

void DiarizenWavlmPruned::log_summary() const {
    if (!loaded_) {
        LOG_INFO(kLog, "(not loaded)");
        return;
    }
    LOG_INFO(kLog,
             "arena=%.2f MB  tensors=%zu  layers=%d",
             arena_bytes_ / (1024.0 * 1024.0),
             tensors_.size(),
             static_cast<int>(layer_dims_.size()));
    for (const auto& d : layer_dims_) {
        std::fprintf(stderr,
                     "  layer %2d  attn_inner=%4d  num_heads=%2d  ffn_inner=%4d\n",
                     d.layer_index, d.attn_inner, d.num_heads,
                     d.ffn_inner);
    }
}

}  // namespace orator
}  // namespace deusridet
