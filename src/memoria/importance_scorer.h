// importance_scorer.h — Attention-score based KV block importance tracking
//
// Runs asynchronously on a separate CUDA stream after each Prefill frame.
// Accumulates per-block attention weights across all Full Attention layers.
// Blocks with consistently low importance become eviction candidates.
//
// Design:
//   - Each block has a running importance score (FP32)
//   - After attention, scores are decayed and updated with new attention weights
//   - Eviction policy queries the scorer for the least important block

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace deusridet {

class ImportanceScorer {
public:
    ImportanceScorer() = default;
    ~ImportanceScorer();

    ImportanceScorer(const ImportanceScorer&) = delete;
    ImportanceScorer& operator=(const ImportanceScorer&) = delete;

    // Initialize scorer for max_blocks logical blocks.
    // decay: per-update multiplicative decay for old scores (e.g. 0.95).
    bool init(int max_blocks, float decay = 0.95f);

    // Update block importance scores on GPU.
    // block_scores: [num_blocks] GPU buffer with new importance values from attention.
    // num_blocks: number of logical blocks with data.
    // Runs on scorer_stream (async).
    void update(const float* d_block_scores, int num_blocks);

    // Get the index of the least important GPU-resident block.
    // gpu_resident: host array of block IDs that are on GPU.
    // n: number of GPU-resident blocks.
    // Returns the logical block index with lowest score.
    int least_important(const int* gpu_resident, int n) const;

    // Get importance score for a specific logical block (host-side).
    float score(int logical_block) const;

    // Copy scores from GPU to host mirror (sync).
    void sync_to_host();

    // Release resources.
    void destroy();

    // CUDA stream used for async scoring
    cudaStream_t stream() const { return scorer_stream_; }

private:
    float* d_scores_  = nullptr;  // [max_blocks] device
    std::vector<float> h_scores_;  // host mirror
    int max_blocks_   = 0;
    float decay_      = 0.95f;
    cudaStream_t scorer_stream_ = nullptr;
};

} // namespace deusridet
