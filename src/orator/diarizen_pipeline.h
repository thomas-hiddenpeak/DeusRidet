// @philosophical_role Orator — the native DiariZen-v2 pipeline. Chains the
//   three native stages (Segmenter -> ResNet34 Embedder -> VBx Clustering) and
//   the pyannote post-processing (get_embeddings windowing, reconstruct,
//   Inference.aggregate overlap-add, Binarize) into a single in-process call.
//   This is the artefact that retires tools/diarizen_worker.py: once it
//   reproduces the worker's labelled intervals on tests/test.mp3, the Python
//   subprocess and its IPC facade are deleted and clustering runs inside
//   awaken.
// @serves Orator subsystem (docs/{en,zh}/architecture/12-diarizen.md, P3a/P3b).
//   R3 boundary: no Python, torch, or pyannote types cross this header; the
//   only inputs are a 16 kHz mono waveform and model paths, the only output is
//   a list of labelled time intervals.
#ifndef DEUSRIDET_ORATOR_DIARIZEN_PIPELINE_H
#define DEUSRIDET_ORATOR_DIARIZEN_PIPELINE_H

#include <memory>
#include <string>
#include <vector>

#include "diarizen_facade.h"  // DiarizenSegment (shared output type)

namespace deusridet {
namespace orator {

// Paths to the four native model assets. Defaults point at the verified
// /home/rm01/models/dev/diarizen_v2 layout used by the 93.5% live verdict.
struct DiarizenPipelineConfig {
    // WavLM-pruned encoder + 4-layer Conformer EEND head (segmentation).
    std::string wavlm_safetensors;
    std::string conformer_safetensors;
    // WeSpeaker ResNet34-LM speaker embedder.
    std::string resnet34_safetensors;
    // Directory holding xvec_transform.npz + plda.npz (VBx priors).
    std::string plda_dir = "/home/rm01/models/dev/diarizen_v2";

    // Pipeline hyper-parameters (mirror pyannote __call__ defaults).
    bool   apply_median_filtering = true;
    bool   embedding_exclude_overlap = true;
    int    min_speakers = 1;
    int    max_speakers = 20;
    double binarize_onset  = 0.5;
    double binarize_offset = 0.5;
};

// Native, in-process replacement for DiarizenFacade. Owns the three native
// model stages; thread-compatible (one outstanding diarize() per instance).
class DiarizenPipeline {
   public:
    DiarizenPipeline();
    ~DiarizenPipeline();

    DiarizenPipeline(const DiarizenPipeline&) = delete;
    DiarizenPipeline& operator=(const DiarizenPipeline&) = delete;
    DiarizenPipeline(DiarizenPipeline&&) noexcept;
    DiarizenPipeline& operator=(DiarizenPipeline&&) noexcept;

    // @role Load all four model assets. Returns false (and LOG_ERROR) on any
    //       failure; last_error() describes why. Idempotent.
    bool load(const DiarizenPipelineConfig& cfg);
    bool is_loaded() const;

    // @role Run the full pipeline on a 16 kHz mono waveform and return the
    //       labelled intervals (labels "speaker0".."speakerN"). Empty vector
    //       on failure (check last_error()). Reproduces pyannote __call__
    //       segment-for-segment.
    std::vector<DiarizenSegment> diarize(const float* wave, int n_samples);

    const std::string& last_error() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace orator
}  // namespace deusridet

#endif  // DEUSRIDET_ORATOR_DIARIZEN_PIPELINE_H
