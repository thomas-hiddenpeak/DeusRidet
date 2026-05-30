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

namespace deusridet {
namespace orator {

/// One labelled time interval produced by DiariZen-v2 diarisation. Labels
/// are pipeline-local strings (e.g. "speaker0", "speaker1"); the caller is
/// responsible for mapping them onto persistent global identities. This is
/// the shared output type of the native pipeline (moved here when the
/// Python-IPC facade was retired in P3b-3).
struct DiarizenSegment {
    double      start_sec = 0.0;
    double      end_sec   = 0.0;
    std::string label;
};

// Paths to the four native model assets. Defaults point at the verified
// /home/rm01/models/dev/diarizen_v2 layout used by the 93.5% live verdict.
struct DiarizenPipelineConfig {
    // WavLM-pruned encoder + 4-layer Conformer EEND head (segmentation).
    std::string wavlm_safetensors =
        "/home/rm01/models/dev/diarizen_v2/wavlm_pruned.safetensors";
    std::string conformer_safetensors =
        "/home/rm01/models/dev/diarizen_v2/conformer_head.safetensors";
    // WeSpeaker ResNet34-LM speaker embedder.
    std::string resnet34_safetensors =
        "/home/rm01/models/dev/diarizen_v2/wespeaker_resnet34.safetensors";
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

// Native, in-process DiariZen-v2 pipeline (retired the Python-IPC bridge).
// Owns the three native model stages; thread-compatible (one outstanding
// diarize() per instance).
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

    // --- bit-equality debug taps (P3a harness) -----------------------------
    // @role get_embeddings windowing tap. Given a waveform and the binarized
    //   segmentation [num_chunks * num_frames * num_speakers], reproduce
    //   pyannote get_embeddings: crop each chunk (mode="pad"), build the
    //   clean/full activity mask, and run the ResNet34 embedder per
    //   (chunk, speaker). Writes embeddings [num_chunks * num_speakers * 256].
    //   Requires the embedder to be loaded. Returns false on error.
    bool debug_get_embeddings(const float* wave, int n_samples,
                              const float* seg, int num_chunks, int num_frames,
                              int num_speakers, std::vector<float>& emb_out);

    // @role reconstruct + speaker_count + to_diarization tap. Given the
    //   binarized segmentation [C*F*S] and the per-chunk hard cluster ids
    //   [C*S], reproduce pyannote reconstruct (max over local speakers per
    //   cluster, skip -2), speaker_count (overlap-add sum, rint->uint8) and
    //   to_diarization (overlap-add skip_average, argsort top-count binary).
    //   Writes count_out [num_out_frames] (rounded float) and binary_out
    //   [num_out_frames * num_clusters]. Returns false on error.
    bool debug_post_process(const float* seg, const int* hard, int num_chunks,
                            int num_frames, int num_speakers,
                            std::vector<float>& count_out,
                            std::vector<float>& binary_out, int& num_out_frames,
                            int& num_clusters);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace orator
}  // namespace deusridet

#endif  // DEUSRIDET_ORATOR_DIARIZEN_PIPELINE_H
