/**
 * @file spectral_cluster.h
 * @philosophical_role Declaration of spectral clustering for unsupervised speaker diarization. When the entity does not yet know the voices, it at least counts them.
 * @serves Orator diarisation pipeline.
 */
// spectral_cluster.h — Warm-up spectral clustering for online speaker diarization.
//
// Adapted from qwen35-orin transcription_pipeline.cpp (Phase 3b):
//   PCA 192→16, cosine similarity + temporal mixing, p-pruning,
//   normalized Laplacian eigendecomposition (power iteration),
//   eigengap K-selection, K-means++ with multi-restart.
//
// All CPU, no external dependencies (LAPACK-free).
// Designed for small batch sizes (20–80 embeddings) collected during
// the warm-up phase of online speaker identification.
//
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include <vector>

namespace deusridet {

struct SpectralClusterConfig {
    int   pca_dim          = 16;      // PCA target dimension
    float temporal_alpha   = 0.93f;   // temporal proximity weight
    float temporal_tau     = 3.125f;  // Gaussian kernel bandwidth (seconds)
    float p_prune_ratio    = 0.10f;   // keep top 10% neighbors
    int   max_k            = 8;       // maximum clusters
    int   min_k            = 2;       // minimum clusters
    float merge_threshold  = 0.55f;   // post-clustering centroid merge threshold (original space)
    int   kmeans_restarts  = 20;      // K-means++ random restarts
    int   kmeans_iters     = 100;     // max iterations per restart
    int   power_iters      = 300;     // power iteration steps
    int   smooth_window    = 1;       // temporal smoothing window
    int   smooth_iters     = 3;       // max smoothing passes
};

struct ClusterResult {
    int K = 0;                                   // number of clusters found
    std::vector<int> labels;                     // per-embedding cluster label [0, K)
    std::vector<std::vector<float>> centroids;   // K × dim centroids in ORIGINAL space (L2-normed)
};

// Run spectral clustering on a batch of speaker embeddings.
//
// embeddings:      N × dim row-major (each vector should be L2-normalized)
// timestamps_sec:  N timestamps (mid-point of each segment, in seconds).
//                  Pass empty to disable temporal mixing.
// dim:             embedding dimension
// cfg:             algorithm parameters
//
// Returns ClusterResult with K, labels[N], and centroids[K][dim].
//
// Body lives in spectral_cluster.cpp — the 560-line algorithm pushed the
// header past the R1 500-line hard cap and forced every includer to drag
// the implementation through its parser. See that file for the full
// PCA → similarity → Laplacian → K-means++ → merge pipeline.
ClusterResult spectral_cluster(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& timestamps_sec,
    int dim,
    const SpectralClusterConfig& cfg = {});

} // namespace deusridet
