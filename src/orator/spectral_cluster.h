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
    // K-selection criterion:
    //   0 = nme+rel_gap (legacy, biased toward small K)
    //   1 = eigenvalue-ratio λ[k]/λ[k+1] (parameter-free; argmax over k≥1)
    int   k_selection_mode = 0;

    // Phase 8 — per-cluster purity post-filter.
    // After K-means (Step 6), each cluster c with at least
    // `purity_min_cluster_size` members has its members' cosine to the
    // L2-normed cluster centroid (in ORIGINAL embedding space) measured.
    // A cluster is considered contaminated when
    //   mean(member·centroid) < purity_min_mean_cos
    // We then run a K=2 K-means++ on the cluster's original-space
    // embeddings; the split is accepted only when the resulting two
    // sub-centroids have cosine < `purity_accept_max_subsim` (i.e. they
    // are far enough apart to really be two speakers).
    // Default OFF so existing eval baseline (s1800=0.7935 /
    // full_60m=0.7025 with W=180) is byte-identical until opt-in.
    bool  purity_split_enable          = false;
    int   purity_min_cluster_size      = 8;
    float purity_min_mean_cos          = 0.60f;
    float purity_accept_max_subsim     = 0.85f;
    int   purity_split_kmeans_iters    = 25;
    int   purity_split_kmeans_restarts = 4;

    // Phase 9 — length-weighted K-means centroid update.
    // When ON and `weights` is supplied to spectral_cluster(), each
    // point contributes w_i to its assigned centroid (instead of 1).
    // No-op when OFF or when weights is empty.
    bool  length_weighted_enable       = false;

    // Phase 14 — reliability-weighted affinity matrix.
    // When ON and `weights` (per-segment durations, in seconds) is
    // supplied, each off-diagonal sim[i,j] is scaled by
    //     sqrt( min(1, w_i / affinity_dur_ref) *
    //           min(1, w_j / affinity_dur_ref) )
    // so that short (< affinity_dur_ref) segments contribute proportionally
    // less to the affinity graph that drives the eigendecomposition and
    // ultimately the K-means assignment. The Phase-13 audit showed that
    // dur < 1.0 s segments dominate the error mass (~2× enrichment) yet
    // their cosine to a clean centroid is the noisiest signal in the
    // pipeline. This knob attacks the problem inside the affinity matrix
    // rather than as a coarse post-pass.
    // Default OFF so existing eval baseline is byte-identical until opt-in.
    bool  affinity_weighted_enable     = false;
    float affinity_dur_ref             = 1.5f;
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
    const SpectralClusterConfig& cfg = {},
    const std::vector<float>& weights = {});

} // namespace deusridet
