// overlap_detector.h — Learned Overlap Speech Detection via pyannote/segmentation-3.0.
//
// Model: PyanNet (SincNet + LSTM + Linear), ~5 MB, MIT license.
// Input:  (1, 1, 160000) — 10s mono PCM @ 16kHz float32
// Output: (1, num_frames, 7) — powerset encoding
//   Classes: [non-speech, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3]
//   Overlap = argmax in {4, 5, 6}
//
// Runs on GPU via native CUDA (safetensors weights).
//
// Streaming: accumulate PCM into 10s windows with 5s hop (50% overlap),
// report overlap for the center 5s of each window output.

#pragma once

#include "pyannote_seg3.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {

struct OverlapDetectorConfig {
    std::string model_path;          // path to pyannote_seg3.safetensors
    float overlap_threshold = 0.5f;  // softmax threshold for overlap classes
    int chunk_samples = 160000;      // 10s @ 16kHz (model native window)
    int hop_samples   = 80000;       // 5s hop for streaming (50% overlap)
    bool enabled      = true;
};

struct OverlapResult {
    bool is_overlap;           // any frame in analysis window has overlap
    float overlap_ratio;       // fraction of frames with overlap [0.0, 1.0]
    int overlap_start_frame;   // first overlapping frame index
    int overlap_end_frame;     // last overlapping frame index
    int num_frames;            // total frames in result
    // Per-frame detail
    std::vector<bool> frame_overlap;   // per-frame overlap flag
    std::vector<int>  frame_num_spk;   // per-frame speaker count (0, 1, 2, 3)
};

class OverlapDetector {
public:
    OverlapDetector();
    ~OverlapDetector();

    OverlapDetector(const OverlapDetector&) = delete;
    OverlapDetector& operator=(const OverlapDetector&) = delete;

    bool init(const OverlapDetectorConfig& cfg);

    // Process exactly chunk_samples (160000) of PCM, return overlap analysis.
    OverlapResult detect(const float* pcm, int n_samples);

    // Streaming mode: feed arbitrary PCM, returns true when a new 10s window
    // has been processed and result is available.
    bool feed(const float* pcm, int n_samples, OverlapResult& result);

    // Reset streaming state (e.g., on segment boundary).
    void reset();

    bool initialized() const { return initialized_; }

    // Model info
    int num_output_frames() const { return num_frames_; }
    int num_classes() const { return num_classes_; }

private:
    OverlapDetectorConfig cfg_;
    bool initialized_ = false;

    PyannoteSeg3 seg3_;
    cudaStream_t stream_ = nullptr;

    // GPU buffers
    float* d_input_  = nullptr;  // (1, 1, chunk_samples)
    float* d_output_ = nullptr;  // (1, num_frames, num_classes)

    // Model output dimensions
    int num_frames_  = PyannoteSeg3::kNumFrames;   // 589
    int num_classes_ = PyannoteSeg3::kNumClasses;   // 7

    // Streaming buffer
    std::vector<float> stream_buf_;

    // Host output buffer
    std::vector<float> h_output_;

    // Convert powerset logits to overlap result.
    void decode_powerset(const float* logits, int num_frames, int num_classes,
                         OverlapResult& result);
};

} // namespace deusridet
