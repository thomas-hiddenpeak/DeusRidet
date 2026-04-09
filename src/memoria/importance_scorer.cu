// importance_scorer.cu — Attention-score KV block importance kernels
//
// Simple exponential decay + additive update model:
//   score[b] = score[b] * decay + new_score[b]
//
// Target: SM87 (Jetson AGX Orin)

#include "importance_scorer.h"
#include "../communis/log.h"
#include <cfloat>
#include <algorithm>

namespace deusridet {

// ============================================================================
// Decay + update kernel
// One thread per block (max ~1000 blocks, trivially parallel).
// ============================================================================

__global__ void importance_update_kernel(
    float* __restrict__ scores,
    const float* __restrict__ new_scores,
    int num_blocks,
    float decay)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;
    scores[b] = scores[b] * decay + new_scores[b];
}

// ============================================================================
// ImportanceScorer implementation
// ============================================================================

ImportanceScorer::~ImportanceScorer() {
    destroy();
}

bool ImportanceScorer::init(int max_blocks, float decay) {
    max_blocks_ = max_blocks;
    decay_ = decay;

    size_t bytes = max_blocks * sizeof(float);
    cudaError_t err = cudaMalloc(&d_scores_, bytes);
    if (err != cudaSuccess) {
        LOG_ERROR("Memoria", "ImportanceScorer: alloc failed: %s",
                  cudaGetErrorString(err));
        return false;
    }
    cudaMemset(d_scores_, 0, bytes);

    h_scores_.resize(max_blocks, 0.0f);

    cudaStreamCreate(&scorer_stream_);

    LOG_INFO("Memoria", "ImportanceScorer: %d blocks, decay=%.3f", max_blocks, decay);
    return true;
}

void ImportanceScorer::update(const float* d_block_scores, int num_blocks) {
    if (!d_scores_ || num_blocks <= 0) return;
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    importance_update_kernel<<<blocks, threads, 0, scorer_stream_>>>(
        d_scores_, d_block_scores, num_blocks, decay_);
}

int ImportanceScorer::least_important(const int* gpu_resident, int n) const {
    if (n <= 0) return -1;
    int min_idx = gpu_resident[0];
    float min_score = h_scores_[min_idx];
    for (int i = 1; i < n; i++) {
        int idx = gpu_resident[i];
        if (h_scores_[idx] < min_score) {
            min_score = h_scores_[idx];
            min_idx = idx;
        }
    }
    return min_idx;
}

float ImportanceScorer::score(int logical_block) const {
    if (logical_block < 0 || logical_block >= max_blocks_) return 0.0f;
    return h_scores_[logical_block];
}

void ImportanceScorer::sync_to_host() {
    if (!d_scores_) return;
    cudaStreamSynchronize(scorer_stream_);
    cudaMemcpy(h_scores_.data(), d_scores_,
               max_blocks_ * sizeof(float), cudaMemcpyDeviceToHost);
}

void ImportanceScorer::destroy() {
    if (d_scores_) {
        cudaFree(d_scores_);
        d_scores_ = nullptr;
    }
    if (scorer_stream_) {
        cudaStreamDestroy(scorer_stream_);
        scorer_stream_ = nullptr;
    }
    h_scores_.clear();
    max_blocks_ = 0;
}

} // namespace deusridet
