/**
 * @file diarizen_segmenter.cpp
 * @philosophical_role P1c orchestration body. Sequences the GPU WavLM-pruned
 *     encoder and Conformer head over a sliding window and performs the host
 *     bookkeeping (powerset argmax/decode, median filter, last-chunk zero pad)
 *     that turns 16-way logits into the binarized segmentation. No CUDA kernels
 *     live here: the compute is already on the GPU inside the two sub-models.
 * @serves DiarizenSegmenter — the P3a facade's segmentation entry point.
 */
#include "diarizen_segmenter.h"

#include "../communis/log.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace deusridet {
namespace orator {

namespace {
constexpr const char* kSLog = "DiariZenSegmenter";
}  // namespace

DiarizenSegmenter::DiarizenSegmenter() : mapping_(build_mapping_()) {}
DiarizenSegmenter::~DiarizenSegmenter() = default;

bool DiarizenSegmenter::is_loaded() const {
    return wavlm_.is_loaded() && conformer_.is_loaded();
}

bool DiarizenSegmenter::load(const std::string& wavlm_safetensors,
                             const std::string& conformer_safetensors) {
    if (!wavlm_.load(wavlm_safetensors)) {
        LOG_ERROR(kSLog, "WavLM-pruned load failed: %s",
                  wavlm_safetensors.c_str());
        return false;
    }
    if (!conformer_.load(conformer_safetensors)) {
        LOG_ERROR(kSLog, "Conformer head load failed: %s",
                  conformer_safetensors.c_str());
        return false;
    }
    return true;
}

// Canonical pyannote Powerset(4, 4).build_mapping(): subsets of {0..3} ordered
// by ascending set size, each in itertools.combinations order. Row k is the
// multilabel indicator of powerset class k. Matches the reference matrix the
// dump tool prints.
std::vector<int> DiarizenSegmenter::build_mapping_() {
    const int n = DiarizenSegmenterArch::kNumSpeakers;     // 4
    std::vector<int> mapping;                              // [16 * 4]
    mapping.reserve(DiarizenSegmenterArch::kPowersetClasses * n);
    // set_size = 0
    for (int s = 0; s < n; ++s) mapping.push_back(0);
    // set_size = 1..n via lexicographic combinations (matches itertools).
    std::vector<int> idx;
    for (int set_size = 1; set_size <= n; ++set_size) {
        idx.assign(set_size, 0);
        for (int i = 0; i < set_size; ++i) idx[i] = i;
        while (true) {
            std::vector<int> row(n, 0);
            for (int v : idx) row[v] = 1;
            for (int s = 0; s < n; ++s) mapping.push_back(row[s]);
            // advance combination (n choose set_size), lexicographic.
            int i = set_size - 1;
            while (i >= 0 && idx[i] == n - set_size + i) --i;
            if (i < 0) break;
            ++idx[i];
            for (int j = i + 1; j < set_size; ++j) idx[j] = idx[j - 1] + 1;
        }
    }
    return mapping;  // 16 rows * 4 cols
}

// median_filter(data, size=(1, 11, 1), mode='reflect') along the frame axis.
// scipy 'reflect' is half-sample symmetric (edge duplicated). Window of 11 is
// odd, so the median is the 6th smallest of the 11 values; on binary inputs
// that is the majority vote.
void DiarizenSegmenter::median_filter_frames_(std::vector<float>& data,
                                              int num_chunks, int num_frames,
                                              int num_speakers) {
    const int W   = DiarizenSegmenterArch::kMedianWindow;  // 11
    const int rad = W / 2;                                 // 5
    std::vector<float> out(data.size());
    std::vector<float> win(W);
    for (int c = 0; c < num_chunks; ++c) {
        for (int s = 0; s < num_speakers; ++s) {
            const std::size_t base = (std::size_t)c * num_frames * num_speakers;
            for (int f = 0; f < num_frames; ++f) {
                for (int k = -rad; k <= rad; ++k) {
                    int fi = f + k;
                    // half-sample reflect (edge duplicated).
                    if (num_frames == 1) {
                        fi = 0;
                    } else {
                        int period = 2 * num_frames;
                        fi %= period;
                        if (fi < 0) fi += period;
                        if (fi >= num_frames) fi = period - 1 - fi;
                    }
                    win[k + rad] = data[base + (std::size_t)fi * num_speakers + s];
                }
                std::nth_element(win.begin(), win.begin() + rad, win.end());
                out[base + (std::size_t)f * num_speakers + s] = win[rad];
            }
        }
    }
    data.swap(out);
}

DiarizenSegmentation DiarizenSegmenter::segment(const float* wave,
                                                int n_samples,
                                                bool apply_median) {
    DiarizenSegmentation result;
    if (!is_loaded()) {
        LOG_ERROR(kSLog, "segment() called before load()");
        return result;
    }
    if (!wave || n_samples <= 0) {
        LOG_ERROR(kSLog, "invalid waveform (n_samples=%d)", n_samples);
        return result;
    }

    const int win   = DiarizenSegmenterArch::kWindowSamples;  // 256000
    const int step  = DiarizenSegmenterArch::kStepSamples;    // 25600
    const int nspk  = DiarizenSegmenterArch::kNumSpeakers;    // 4

    // Chunk layout mirrors pyannote Inference.slide: complete chunks via
    // unfold, plus one zero-padded last chunk when the tail is non-empty.
    int num_complete = 0;
    if (n_samples >= win) num_complete = (n_samples - win) / step + 1;
    const bool has_last =
        (n_samples < win) || ((n_samples - win) % step > 0);
    const int num_chunks = num_complete + (has_last ? 1 : 0);
    if (num_chunks <= 0) {
        LOG_ERROR(kSLog, "no chunks for n_samples=%d", n_samples);
        return result;
    }

    result.num_chunks   = num_chunks;
    result.num_frames   = DiarizenSegmenterArch::kFramesPerChunk;
    result.num_speakers = nspk;
    result.data.assign((std::size_t)num_chunks * result.num_frames * nspk, 0.0f);

    // WavLM front-end + encoder runs in batches of `batch` chunks so the dense
    // transformer linears fuse into tall [B*T, H] GEMMs that fill the GPU,
    // instead of one batch=1 launch per chunk. Conformer decode stays per chunk
    // (its head is cheap; batching it is a separate step). The batched tail is
    // numerically identical to B independent debug_lnorm_tail calls.
    int batch = 8;
    if (const char* e = std::getenv("DEUSRIDET_DIARIZEN_BATCH")) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 64) batch = v;
    }
    std::vector<float> chunks((std::size_t)batch * win);
    for (int c0 = 0; c0 < num_chunks; c0 += batch) {
        const int B = std::min(batch, num_chunks - c0);
        // Fill B chunk windows (zero-padded past the tail).
        for (int b = 0; b < B; ++b) {
            const long start = (long)(c0 + b) * step;
            float* dst = chunks.data() + (std::size_t)b * win;
            for (int i = 0; i < win; ++i) {
                const long idx = start + i;
                dst[i] = (idx < n_samples) ? wave[idx] : 0.0f;
            }
        }

        int T_lnorm = 0;
        std::vector<float> feat =
            wavlm_.debug_lnorm_tail_batch(chunks.data(), B, win, T_lnorm);
        if (feat.empty() || T_lnorm <= 0) {
            LOG_ERROR(kSLog, "WavLM batched tail failed at chunk %d", c0);
            return DiarizenSegmentation{};
        }
        const int D = 256;
        for (int b = 0; b < B; ++b) {
            const int c = c0 + b;
            std::vector<float> logits = conformer_.debug_logits(
                feat.data() + (std::size_t)b * T_lnorm * D, T_lnorm);
            if (logits.empty()) {
                LOG_ERROR(kSLog, "Conformer logits failed on chunk %d", c);
                return DiarizenSegmentation{};
            }
            const int T = std::min(T_lnorm, result.num_frames);

            // Powerset decode (soft=False): per frame argmax over 16 classes,
            // then map through the powerset->multilabel matrix.
            for (int f = 0; f < T; ++f) {
                const float* lr =
                    logits.data() +
                    (std::size_t)f * DiarizenSegmenterArch::kPowersetClasses;
                int best = 0;
                float bv = lr[0];
                for (int k = 1; k < DiarizenSegmenterArch::kPowersetClasses;
                     ++k) {
                    if (lr[k] > bv) { bv = lr[k]; best = k; }
                }
                const int* mrow = mapping_.data() + (std::size_t)best * nspk;
                float* d = result.data.data() +
                           ((std::size_t)c * result.num_frames + f) * nspk;
                for (int s = 0; s < nspk; ++s) d[s] = (float)mrow[s];
            }
        }
    }

    if (apply_median) {
        median_filter_frames_(result.data, result.num_chunks, result.num_frames,
                              result.num_speakers);
    }
    return result;
}

}  // namespace orator
}  // namespace deusridet
