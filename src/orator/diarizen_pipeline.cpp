// @role: DiariZen native pipeline implementation (P3a). Chains the three
//   native stages and the pyannote post-processing. This increment lands
//   load() + get_embeddings windowing (P3a-3); reconstruct / aggregate /
//   speaker_count / Binarize and the full diarize() follow in P3a-4/P3a-5.
#include "diarizen_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
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
        // Env-gated sub-stage profiler: splits embed into backbone (1/chunk)
        // vs pool (S/chunk) so we can see the per-chunk launch hog. Off unless
        // DEUSRIDET_DIARIZEN_EMB_PROF=1.
        const bool emb_prof = [] {
            const char* e = std::getenv("DEUSRIDET_DIARIZEN_EMB_PROF");
            return e && e[0] == '1';
        }();
        using emb_clock = std::chrono::steady_clock;
        auto emb_ms = [](emb_clock::time_point a, emb_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        double ms_back = 0, ms_pool = 0;
        for (int c = 0; c < C; ++c) {
            crop_chunk_(wave, n_samples, c, chunk);
            // The ResNet34 backbone depends only on the chunk waveform, so
            // compute it once and reuse it for every speaker of this chunk
            // (the mask only enters the pooling head). Cuts the dominant conv
            // cost by a factor of S relative to a full embed() per speaker.
            const auto tb0 = emb_clock::now();
            if (!embedder.embed_backbone(chunk.data(), kWindowSamples)) {
                err = "get_embeddings: embed_backbone() failed at chunk " +
                      std::to_string(c);
                return false;
            }
            if (emb_prof) ms_back += emb_ms(tb0, emb_clock::now());
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
                const auto tp0 = emb_clock::now();
                if (!embedder.embed_pool(used, F, dst)) {
                    err = "get_embeddings: embed_pool() failed at chunk " +
                          std::to_string(c) + " speaker " + std::to_string(s);
                    return false;
                }
                if (emb_prof) ms_pool += emb_ms(tp0, emb_clock::now());
            }
        }
        if (emb_prof) {
            std::fprintf(stderr,
                "[emb-prof] C=%d S=%d | backbone=%.1f pool=%.1f ms\n",
                C, S, ms_back, ms_pool);
        }
        return true;
    }

    // pyannote reconstruct + speaker_count + to_diarization. seg is the
    // binarized segmentation [C*F*S]; hard the per-chunk cluster ids [C*S].
    // Produces count_out [nf] (rounded float) and binary_out [nf*ncl].
    // CPU-resident: pure frame-bookkeeping (overlap-add over ~C*F*ncl ≈ 1e5
    // adds, one-shot per diarize call) — orchestration glue between GPU
    // stages, not a tensor op scaling with the heavy data.
    bool post_process(const float* seg, const int* hard, int C, int F, int S,
                      std::vector<float>& count_out,
                      std::vector<float>& binary_out, int& nf, int& ncl) {
        // Frame geometry (matches Inference.aggregate / closest_frame).
        constexpr double kStep = 0.02;   // frames.step (receptive_field)
        constexpr double kDur  = 0.025;  // frames.duration
        constexpr double kChStep = 1.6;  // chunk step (seconds)
        constexpr double kChDur  = 16.0; // chunk duration (seconds)
        auto closest = [&](double t) {
            return static_cast<int>(std::nearbyint((t - 0.5 * kDur) / kStep));
        };
        nf = closest(kChDur + (C - 1) * kChStep + 0.5 * kDur) + 1;
        if (nf <= 0) {
            err = "post_process: non-positive frame count";
            return false;
        }
        // num_clusters = max(hard) + 1.
        int max_k = -1;
        for (int i = 0; i < C * S; ++i) max_k = std::max(max_k, hard[i]);
        ncl = max_k + 1;
        if (ncl <= 0) {
            err = "post_process: no active clusters";
            return false;
        }

        // --- speaker_count: overlap-add sum-over-speakers, /overlap, rint ----
        std::vector<float> csum(nf, 0.0f), ccount(nf, 0.0f), cmask(nf, 0.0f);
        std::vector<float> dout(static_cast<std::size_t>(nf) * ncl, 0.0f);
        std::vector<float> dmask(static_cast<std::size_t>(nf) * ncl, 0.0f);
        for (int c = 0; c < C; ++c) {
            const int sf = closest(c * kChStep + 0.5 * kDur);
            for (int f = 0; f < F; ++f) {
                const int g = sf + f;
                if (g < 0 || g >= nf) continue;
                const std::size_t base = (static_cast<std::size_t>(c) * F + f) * S;
                // count: sum over local speakers (mask=1, segmentation binary)
                float ssum = 0.0f;
                for (int s = 0; s < S; ++s) ssum += seg[base + s];
                csum[g] += ssum;
                ccount[g] += 1.0f;
                cmask[g] = 1.0f;
                // reconstruct+aggregate: per cluster k, max over local speakers
                // in cluster k; NaN (mask 0) where cluster absent in chunk.
                for (int k = 0; k < ncl; ++k) {
                    float mx = -std::numeric_limits<float>::infinity();
                    bool present = false;
                    for (int s = 0; s < S; ++s) {
                        if (hard[c * S + s] == k) {
                            present = true;
                            mx = std::max(mx, seg[base + s]);
                        }
                    }
                    if (present) {
                        const std::size_t di =
                            static_cast<std::size_t>(g) * ncl + k;
                        dout[di] += mx;  // skip_average: no divide
                        dmask[di] = 1.0f;
                    }
                }
            }
        }
        // count: average, missing->0, rint.
        count_out.assign(nf, 0.0f);
        for (int g = 0; g < nf; ++g) {
            float v = (cmask[g] == 0.0f)
                          ? 0.0f
                          : csum[g] / std::max(ccount[g], 1e-12f);
            count_out[g] = std::nearbyint(v);
        }
        // discrete activations: missing->0 (skip_average already applied).
        for (std::size_t i = 0; i < dout.size(); ++i)
            if (dmask[i] == 0.0f) dout[i] = 0.0f;

        // --- to_diarization binary: top-count[g] speakers per frame ----------
        binary_out.assign(static_cast<std::size_t>(nf) * ncl, 0.0f);
        std::vector<int> order(ncl);
        for (int g = 0; g < nf; ++g) {
            for (int k = 0; k < ncl; ++k) order[k] = k;
            const float* act = &dout[static_cast<std::size_t>(g) * ncl];
            // np.argsort(-act): ascending by -act == descending by act, ties
            // keep ascending index order (stable) — matches the fixture.
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return act[a] > act[b]; });
            const int cnt = static_cast<int>(count_out[g]);
            for (int i = 0; i < cnt && i < ncl; ++i)
                binary_out[static_cast<std::size_t>(g) * ncl + order[i]] = 1.0f;
        }
        return true;
    }

    // pyannote Binarize (onset=offset=0.5, no pad, no min-duration). binary is
    // [nf*ncl]; frame middle = i*0.02 + 0.0125 s. Emits one DiarizenSegment per
    // contiguous active run per cluster, label "speaker<k>".
    std::vector<DiarizenSegment> binarize(const std::vector<float>& binary,
                                          int nf, int ncl) {
        constexpr double kStep = 0.02, kDur = 0.025, kOn = 0.5;
        auto mid = [&](int i) { return i * kStep + 0.5 * kDur; };
        std::vector<DiarizenSegment> out;
        for (int k = 0; k < ncl; ++k) {
            double start = mid(0);
            bool is_active = binary[static_cast<std::size_t>(0) * ncl + k] > kOn;
            double t = start;
            for (int i = 1; i < nf; ++i) {
                t = mid(i);
                const float y = binary[static_cast<std::size_t>(i) * ncl + k];
                if (is_active) {
                    if (y < kOn) {
                        out.push_back({start, t, "speaker" + std::to_string(k)});
                        start = t;
                        is_active = false;
                    }
                } else if (y > kOn) {
                    start = t;
                    is_active = true;
                }
            }
            if (is_active)
                out.push_back({start, t, "speaker" + std::to_string(k)});
        }
        return out;
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

bool DiarizenPipeline::debug_post_process(const float* seg, const int* hard,
                                          int num_chunks, int num_frames,
                                          int num_speakers,
                                          std::vector<float>& count_out,
                                          std::vector<float>& binary_out,
                                          int& num_out_frames,
                                          int& num_clusters) {
    return impl_->post_process(seg, hard, num_chunks, num_frames, num_speakers,
                               count_out, binary_out, num_out_frames,
                               num_clusters);
}

std::vector<DiarizenSegment> DiarizenPipeline::diarize(const float* wave,
                                                       int n_samples) {
    impl_->err.clear();
    if (!impl_->loaded) {
        impl_->err = "diarize: pipeline not loaded";
        return {};
    }
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = clk::now();
    // 1. Sliding-window segmentation (WavLM + Conformer, median-filtered).
    DiarizenSegmentation seg = impl_->segmenter.segment(
        wave, n_samples, impl_->cfg.apply_median_filtering);
    if (seg.empty()) {
        impl_->err = "diarize: segmentation empty";
        return {};
    }
    const int C = seg.num_chunks, F = seg.num_frames, S = seg.num_speakers;
    auto t1 = clk::now();

    // 2. Per-(chunk, speaker) speaker embeddings (inactive -> constant).
    std::vector<float> emb;
    if (!impl_->get_embeddings(wave, n_samples, seg.data.data(), C, F, S, emb)) {
        return {};  // err set
    }
    auto t2 = clk::now();

    // 3. VBx clustering -> per-chunk hard cluster ids (-2 = inactive).
    std::vector<std::int8_t> hard8;
    if (!impl_->clustering.cluster(emb.data(), C, S, DiarizenResnet34Arch::kEmbedDim,
                                   seg.data.data(), F, hard8)) {
        impl_->err = "diarize: clustering failed";
        return {};
    }
    std::vector<int> hard(hard8.begin(), hard8.end());
    auto t3 = clk::now();

    // 4. reconstruct + speaker_count + to_diarization.
    std::vector<float> count, binary;
    int nf = 0, ncl = 0;
    if (!impl_->post_process(seg.data.data(), hard.data(), C, F, S, count,
                             binary, nf, ncl)) {
        return {};  // err set
    }
    auto t4 = clk::now();

    // 5. Binarize -> labelled intervals.
    auto out = impl_->binarize(binary, nf, ncl);
    auto t5 = clk::now();
    LOG_INFO("DiarizenPipeline",
             "diarize stages (C=%d F=%d S=%d): seg=%.0fms embed=%.0fms "
             "cluster=%.0fms post=%.0fms binarize=%.0fms total=%.0fms",
             C, F, S, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4),
             ms(t4, t5), ms(t0, t5));
    return out;
}

}  // namespace orator
}  // namespace deusridet
