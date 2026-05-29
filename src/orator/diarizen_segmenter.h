/**
 * @file diarizen_segmenter.h
 * @philosophical_role P1c segmentation orchestrator for the native DiariZen-v2
 *     port. Chains the GPU WavLM-pruned encoder (P1a) and Conformer head (P1b)
 *     over a sliding 16 s / 1.6 s window, decodes the 16-way powerset logits to
 *     a [799, 4] multilabel map per chunk (argmax + powerset->multilabel), and
 *     applies the median filter that the upstream pipeline runs before
 *     clustering. The heavy per-chunk forward stays on the GPU; only the
 *     argmax/mapping/median bookkeeping (O(num_chunks*799), tiny serial glue)
 *     runs on the host.
 * @serves DiariZen pipeline facade (P3a) — produces binarized_segmentations,
 *     the input to the ResNet34 embedder (P2a) and VBx clustering (P2b).
 */
#pragma once

#include <string>
#include <vector>

#include "diarizen_conformer_head.h"
#include "diarizen_wavlm_pruned.h"

namespace deusridet {
namespace orator {

/// Fixed geometry of the DiariZen-v2 segmentation stage (verified on the live
/// pipeline): 16 s window at 16 kHz, 1.6 s step (= 0.1 * 16 s), 799 frames per
/// chunk (RF step 0.02 s), 4 local speakers, 16 powerset classes.
struct DiarizenSegmenterArch {
    static constexpr int kSampleRate    = 16000;
    static constexpr int kWindowSamples = 256000;  // 16 s
    static constexpr int kStepSamples   = 25600;   // 1.6 s
    static constexpr int kFramesPerChunk = 799;
    static constexpr int kNumSpeakers    = 4;
    static constexpr int kPowersetClasses = 16;
    static constexpr int kMedianWindow   = 11;     // size=(1,11,1) reflect
};

/// Per-chunk multilabel segmentation, shape [num_chunks, 799, 4], frame-major
/// within each chunk. `data[(c*799 + f)*4 + s]` is 1.0 if speaker `s` is active
/// in frame `f` of chunk `c`, else 0.0.
struct DiarizenSegmentation {
    int num_chunks   = 0;
    int num_frames   = DiarizenSegmenterArch::kFramesPerChunk;
    int num_speakers = DiarizenSegmenterArch::kNumSpeakers;
    std::vector<float> data;  // [num_chunks * 799 * 4]

    bool empty() const { return data.empty(); }
};

/// Orchestrates the WavLM-pruned encoder + Conformer head over a sliding window
/// to produce the binarized segmentation consumed by the clustering stage.
class DiarizenSegmenter {
public:
    DiarizenSegmenter();
    ~DiarizenSegmenter();

    DiarizenSegmenter(const DiarizenSegmenter&)            = delete;
    DiarizenSegmenter& operator=(const DiarizenSegmenter&) = delete;

    /// Load both sub-models. Returns false (and LOG_ERROR) on any failure.
    bool load(const std::string& wavlm_safetensors,
              const std::string& conformer_safetensors);

    bool is_loaded() const;

    /// Run the full sliding-window segmentation over a 16 kHz mono waveform.
    /// `apply_median` toggles the post-decode median filter (default true,
    /// matching the live pipeline). Returns an empty result on error.
    DiarizenSegmentation segment(const float* wave, int n_samples,
                                 bool apply_median = true);

    /// P1c bit-equality tap: same as `segment` but without the median filter,
    /// exposing the raw per-chunk multilabel (vs the `seg_raw` reference).
    DiarizenSegmentation segment_raw(const float* wave, int n_samples) {
        return segment(wave, n_samples, /*apply_median=*/false);
    }

private:
    // Build the canonical pyannote powerset->multilabel mapping [16, 4].
    static std::vector<int> build_mapping_();
    // In-place median filter along the frame axis (size 11, reflect) per
    // (chunk, speaker). `data` is [num_chunks, 799, 4].
    static void median_filter_frames_(std::vector<float>& data, int num_chunks,
                                      int num_frames, int num_speakers);

    DiarizenWavlmPruned   wavlm_;
    DiarizenConformerHead conformer_;
    std::vector<int>      mapping_;  // [16 * 4], row-major
};

}  // namespace orator
}  // namespace deusridet
