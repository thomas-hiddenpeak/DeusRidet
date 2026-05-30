// @role: DiariZen native pipeline implementation (P3a). Chains the three
//   native stages and the pyannote post-processing. This increment lands
//   load() + get_embeddings windowing (P3a-3); reconstruct / aggregate /
//   speaker_count / Binarize and the full diarize() follow in P3a-4/P3a-5.
#include "diarizen_pipeline.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../communis/log.h"
#include "diarizen_clustering.h"
#include "diarizen_resnet34_embedder.h"
#include "diarizen_segmenter.h"

namespace deusridet {
namespace orator {

namespace {
constexpr int kWindowSamples = DiarizenSegmenterArch::kWindowSamples;  // 256000
constexpr int kStepSamples   = DiarizenSegmenterArch::kStepSamples;    // 25600
constexpr int kEmbedDim      = DiarizenResnet34Arch::kEmbedDim;        // 256
// min_num_frames = ceil(num_frames * min_num_samples / (duration*sr)).
// For the verified DiariZen-v2 config: min_num_samples=400, duration=16 s,
// sr=16 kHz, num_frames=799 -> ceil(799*400/256000) = 2.
constexpr int kMinNumSamples = 400;
}  // namespace

struct DiarizenPipeline::Impl {
    DiarizenPipelineConfig    cfg;
    DiarizenSegmenter         segmenter;
    DiarizenResnet34Embedder  embedder;
    DiarizenClustering        clustering;
    bool                      loaded = false;
    std::string               err;

    // Crop chunk c from `wave` into `out` (length kWindowSamples), zero-padded
    // (pyannote Audio.crop mode="pad"). start = c * kStepSamples.
    static void crop_chunk_(const float* wave, int n_samples, int c,
                            std::vector<float>& out) {
        out.assign(kWindowSamples, 0.0f);
        const long start = static_cast<long>(c) * kStepSamples;
        for (int i = 0; i < kWindowSamples; ++i) {
            const long src = start + i;
            if (src >= 0 && src < n_samples) out[i] = wave[src];
        }
    }

    // pyannote get_embeddings for one (already loaded) waveform + binarized
    // segmentation. emb_out is [C * S * kEmbedDim]; inactive speakers receive
    // the embedder's zero-mask constant (exactly as the reference does).
    bool get_embeddings(const float* wave, int n_samples, const float* seg,
                        int C, int F, int S, std::vector<float>& emb_out) {
        if (!embedder.is_loaded()) {
            err = "get_embeddings: embedder not loaded";
            return false;
        }
        const int min_num_frames = static_cast<int>(
            std::ceil(static_cast<double>(F) * kMinNumSamples /
                      (static_cast<double>(kWindowSamples))));
        emb_out.assign(static_cast<std::size_t>(C) * S * kEmbedDim, 0.0f);

        std::vector<float> chunk(kWindowSamples);
        std::vector<float> clean(F), full(F);
        for (int c = 0; c < C; ++c) {
            crop_chunk_(wave, n_samples, c, chunk);
            // clean_frames[f] = (sum_s seg < 2); clean_mask = seg * clean_frames.
            for (int s = 0; s < S; ++s) {
                double clean_sum = 0.0, full_sum = 0.0;
                for (int f = 0; f < F; ++f) {
                    const std::size_t base = (static_cast<std::size_t>(c) * F + f) * S;
                    float count = 0.0f;
                    for (int t = 0; t < S; ++t) count += seg[base + t];
                    const float a = seg[base + s];
                    // NaN-safe (pyannote nan_to_num(nan=0.0)).
                    const float av = std::isnan(a) ? 0.0f : a;
                    full[f] = av;
                    const float cf = (count < 2.0f) ? 1.0f : 0.0f;
                    clean[f] = av * cf;
                    clean_sum += clean[f];
                    full_sum += full[f];
                }
                const float* used =
                    (clean_sum > min_num_frames) ? clean.data() : full.data();
                (void)full_sum;
                float* dst =
                    &emb_out[(static_cast<std::size_t>(c) * S + s) * kEmbedDim];
                if (!embedder.embed(chunk.data(), kWindowSamples, used, F, dst)) {
                    err = "get_embeddings: embed() failed at chunk " +
                          std::to_string(c) + " speaker " + std::to_string(s);
                    return false;
                }
            }
        }
        return true;
    }
};

DiarizenPipeline::DiarizenPipeline() : impl_(std::make_unique<Impl>()) {}
DiarizenPipeline::~DiarizenPipeline() = default;
DiarizenPipeline::DiarizenPipeline(DiarizenPipeline&&) noexcept = default;
DiarizenPipeline& DiarizenPipeline::operator=(DiarizenPipeline&&) noexcept =
    default;

bool DiarizenPipeline::load(const DiarizenPipelineConfig& cfg) {
    impl_->cfg = cfg;
    impl_->err.clear();
    if (!impl_->segmenter.load(cfg.wavlm_safetensors,
                               cfg.conformer_safetensors)) {
        impl_->err = "load: segmenter.load failed";
        LOG_ERROR("DiarizenPipeline", "%s", impl_->err.c_str());
        return false;
    }
    if (!impl_->embedder.load(cfg.resnet34_safetensors)) {
        impl_->err = "load: embedder.load failed";
        LOG_ERROR("DiarizenPipeline", "%s", impl_->err.c_str());
        return false;
    }
    if (!impl_->clustering.load_priors(cfg.plda_dir)) {
        impl_->err = "load: clustering.load_priors failed";
        LOG_ERROR("DiarizenPipeline", "%s", impl_->err.c_str());
        return false;
    }
    impl_->loaded = true;
    LOG_INFO("DiarizenPipeline", "loaded all native stages");
    return true;
}

bool DiarizenPipeline::is_loaded() const { return impl_->loaded; }

const std::string& DiarizenPipeline::last_error() const noexcept {
    return impl_->err;
}

bool DiarizenPipeline::debug_get_embeddings(const float* wave, int n_samples,
                                            const float* seg, int num_chunks,
                                            int num_frames, int num_speakers,
                                            std::vector<float>& emb_out) {
    return impl_->get_embeddings(wave, n_samples, seg, num_chunks, num_frames,
                                 num_speakers, emb_out);
}

std::vector<DiarizenSegment> DiarizenPipeline::diarize(const float* /*wave*/,
                                                       int /*n_samples*/) {
    // Full chain lands in P3a-4/P3a-5.
    impl_->err = "diarize: not yet implemented (P3a-4/P3a-5)";
    return {};
}

}  // namespace orator
}  // namespace deusridet
