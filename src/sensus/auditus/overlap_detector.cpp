// overlap_detector.cpp — pyannote/segmentation-3.0 native CUDA inference.
//
// Powerset decoding: 7 classes → per-frame overlap detection.
// Streaming: 10s window, 5s hop.
// GPU inference via PyannoteSeg3 (safetensors weights, native CUDA kernels).

#include "overlap_detector.h"
#include "../../communis/log.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

namespace deusridet {

OverlapDetector::OverlapDetector() = default;

OverlapDetector::~OverlapDetector() {
    if (d_input_)  { cudaFree(d_input_);  d_input_  = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (stream_)   { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

bool OverlapDetector::init(const OverlapDetectorConfig& cfg) {
    cfg_ = cfg;

    // Create CUDA stream.
    cudaStreamCreate(&stream_);

    // Init native CUDA model.
    if (!seg3_.init(cfg_.model_path, stream_)) {
        LOG_ERROR("OD", "PyannoteSeg3 init failed: %s", cfg_.model_path.c_str());
        return false;
    }

    num_frames_  = PyannoteSeg3::kNumFrames;
    num_classes_ = PyannoteSeg3::kNumClasses;

    // Allocate GPU buffers.
    size_t input_bytes  = cfg_.chunk_samples * sizeof(float);
    size_t output_bytes = num_frames_ * num_classes_ * sizeof(float);

    cudaMalloc(&d_input_,  input_bytes);
    cudaMalloc(&d_output_, output_bytes);

    // Zero input buffer.
    cudaMemset(d_input_, 0, input_bytes);

    // Allocate host output buffer.
    h_output_.resize(num_frames_ * num_classes_);

    stream_buf_.clear();
    initialized_ = true;

    LOG_INFO("OD", "Native CUDA loaded: %s (chunk=%d, frames=%d, classes=%d)",
             cfg_.model_path.c_str(), cfg_.chunk_samples,
             num_frames_, num_classes_);
    return true;
}

OverlapResult OverlapDetector::detect(const float* pcm, int n_samples) {
    OverlapResult result{};
    if (!initialized_ || !pcm || n_samples <= 0) return result;

    // Pad or truncate to chunk_samples.
    std::vector<float> input_buf(cfg_.chunk_samples, 0.0f);
    int copy_len = std::min(n_samples, cfg_.chunk_samples);
    std::memcpy(input_buf.data(), pcm, copy_len * sizeof(float));

    // Copy input H→D.
    size_t input_bytes = cfg_.chunk_samples * sizeof(float);
    cudaMemcpyAsync(d_input_, input_buf.data(), input_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // Run inference via native CUDA.
    int frames = seg3_.forward(d_input_, d_output_, cfg_.chunk_samples);
    if (frames <= 0) {
        LOG_ERROR("OD", "PyannoteSeg3 forward failed");
        return result;
    }

    // Copy output D→H.
    size_t output_bytes = num_frames_ * num_classes_ * sizeof(float);
    cudaMemcpyAsync(h_output_.data(), d_output_, output_bytes,
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    decode_powerset(h_output_.data(), num_frames_, num_classes_, result);
    return result;
}

bool OverlapDetector::feed(const float* pcm, int n_samples, OverlapResult& result) {
    if (!initialized_ || !pcm || n_samples <= 0) return false;

    // Append to streaming buffer.
    size_t old_size = stream_buf_.size();
    stream_buf_.resize(old_size + n_samples);
    std::memcpy(stream_buf_.data() + old_size, pcm, n_samples * sizeof(float));

    // Check if we have enough for a full chunk.
    if ((int)stream_buf_.size() < cfg_.chunk_samples) return false;

    // Process the first chunk_samples from the buffer.
    result = detect(stream_buf_.data(), cfg_.chunk_samples);

    // Advance by hop_samples (keep overlap for next window).
    int advance = cfg_.hop_samples;
    if (advance >= (int)stream_buf_.size()) {
        stream_buf_.clear();
    } else {
        stream_buf_.erase(stream_buf_.begin(),
                          stream_buf_.begin() + advance);
    }

    return true;
}

void OverlapDetector::reset() {
    stream_buf_.clear();
}

void OverlapDetector::decode_powerset(const float* logits, int num_frames,
                                       int num_classes, OverlapResult& result) {
    // Powerset classes for pyannote segmentation-3.0:
    //   0: non-speech
    //   1: spk1
    //   2: spk2
    //   3: spk3
    //   4: spk1+2 (overlap)
    //   5: spk1+3 (overlap)
    //   6: spk2+3 (overlap)
    //
    // Overlap = argmax in {4, 5, 6}
    // Speaker count:
    //   0 → non-speech (0 speakers)
    //   1,2,3 → single speaker (1 speaker)
    //   4,5,6 → overlap (2 speakers)

    result.num_frames = num_frames;
    result.frame_overlap.resize(num_frames);
    result.frame_num_spk.resize(num_frames);
    result.is_overlap = false;
    result.overlap_ratio = 0.0f;
    result.overlap_start_frame = -1;
    result.overlap_end_frame = -1;

    int overlap_count = 0;

    for (int f = 0; f < num_frames; f++) {
        const float* row = logits + f * num_classes;

        // Argmax over classes.
        int best_class = 0;
        float best_val = row[0];
        for (int c = 1; c < num_classes && c < 7; c++) {
            if (row[c] > best_val) {
                best_val = row[c];
                best_class = c;
            }
        }

        // Determine speaker count and overlap.
        int num_spk = 0;
        bool is_ovl = false;
        if (best_class == 0) {
            num_spk = 0;  // non-speech
        } else if (best_class >= 1 && best_class <= 3) {
            num_spk = 1;  // single speaker
        } else if (best_class >= 4 && best_class <= 6) {
            num_spk = 2;  // overlap
            is_ovl = true;
        }

        // Also check with softmax + threshold for more nuanced detection.
        // Compute softmax and check if any overlap class exceeds threshold.
        if (!is_ovl && cfg_.overlap_threshold < 1.0f) {
            float max_val = *std::max_element(row, row + std::min(num_classes, 7));
            float sum_exp = 0.0f;
            float ovl_sum = 0.0f;
            for (int c = 0; c < num_classes && c < 7; c++) {
                float e = std::exp(row[c] - max_val);
                sum_exp += e;
                if (c >= 4) ovl_sum += e;  // overlap classes
            }
            float ovl_prob = ovl_sum / sum_exp;
            if (ovl_prob >= cfg_.overlap_threshold) {
                is_ovl = true;
                num_spk = 2;
            }
        }

        result.frame_overlap[f] = is_ovl;
        result.frame_num_spk[f] = num_spk;

        if (is_ovl) {
            overlap_count++;
            if (result.overlap_start_frame < 0)
                result.overlap_start_frame = f;
            result.overlap_end_frame = f;
        }
    }

    result.is_overlap = (overlap_count > 0);
    result.overlap_ratio = (num_frames > 0) ?
        (float)overlap_count / (float)num_frames : 0.0f;
}

} // namespace deusridet
