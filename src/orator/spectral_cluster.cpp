/**
 * @file spectral_cluster.cpp
 * @philosophical_role Top-level orchestrator for `spectral_cluster()`.
 *     Previously a 560-line monolith; after Step 11 of the file-shape
 *     campaign the body dispatches to ten named stages — each an
 *     independently inspectable and trace-taggable helper — living in
 *     the peer TU `spectral_cluster_stages.cpp`. The orchestrator
 *     exists to make the shape of the algorithm legible at a glance.
 * @serves Orator diarisation pipeline via spectral_cluster.h.
 */
#include "spectral_cluster.h"
#include "spectral_cluster_stages.h"

#include <algorithm>
#include <vector>

namespace deusridet {

ClusterResult spectral_cluster(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& timestamps_sec,
    int dim,
    const SpectralClusterConfig& cfg)
{
    using namespace spectral_detail;

    const int N = (int)embeddings.size();

    // Trivial cases: 0 or 1 embedding — nothing to cluster.
    if (N < 2) {
        ClusterResult r;
        r.K = N;
        r.labels.assign(N, 0);
        if (N == 1) r.centroids.push_back(embeddings[0]);
        return r;
    }

    const int pca_dim = std::min(cfg.pca_dim, std::min(dim, N));
    const int actual_max = std::min(cfg.max_k, N);

    // Step 0 — PCA dim → pca_dim (L2-normed rows).
    auto pca_emb = pca_reduce(embeddings, dim, pca_dim, cfg.power_iters);

    // Step 1 (+ 1b) — cosine similarity + optional temporal mixing.
    auto sim = build_similarity(
        pca_emb, pca_dim, N,
        timestamps_sec, cfg.temporal_alpha, cfg.temporal_tau);

    // Step 2 — p-pruning (keep top-p neighbours per row).
    p_prune(sim, N, cfg.p_prune_ratio);

    // Step 3 — symmetrise, clamp, orphan repair.
    symmetrize_and_repair_orphans(sim, N, pca_emb, pca_dim);

    // Step 4 — normalised Laplacian eigendecomposition.
    std::vector<std::vector<float>> eigvecs;
    std::vector<float> eigvals;
    laplacian_eigendecomp(sim, N, actual_max, cfg.power_iters, eigvecs, eigvals);

    // Step 5 — eigengap K-selection.
    const int optimal_k = select_k_by_eigengap(eigvals, actual_max, cfg.min_k, cfg.max_k);

    // Step 6 — K-means++ on spectral features with multi-restart.
    auto labels = kmeans_pp_spectral(
        eigvecs, N, optimal_k, cfg.kmeans_restarts, cfg.kmeans_iters);

    // Step 7 — temporal smoothing (majority vote guarded by centroid similarity).
    temporal_smooth(
        labels, N, optimal_k, pca_emb, pca_dim,
        timestamps_sec, cfg.smooth_window, cfg.smooth_iters);

    // Step 8 — centroids in ORIGINAL (pre-PCA) space, L2-normed.
    ClusterResult result;
    result.K = optimal_k;
    result.labels = std::move(labels);
    std::vector<int> ccnt;
    compute_original_centroids(result, embeddings, dim, ccnt);

    // Step 9 — agglomeratively merge similar centroids (in original space).
    merge_similar_centroids(result, ccnt, dim, cfg.merge_threshold, cfg.min_k);

    return result;
}

} // namespace deusridet
