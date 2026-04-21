/**
 * @file spectral_cluster_stages.h
 * @philosophical_role Private decomposition of `spectral_cluster()` into the
 *     ten named stages of the algorithm. Step 11 of the file-shape campaign
 *     reaches into the function body: the top-level routine is a ~70-line
 *     dispatcher, and each algorithmic stage is now an independently
 *     inspectable, trace-taggable helper.
 * @serves `spectral_cluster.cpp` orchestration; implementations live in
 *     three peer TUs by phase:
 *       - `spectral_cluster_affinity.cpp`       — Stages 0–3
 *       - `spectral_cluster_embed.cpp`          — Stages 4–6
 *       - `spectral_cluster_postprocess.cpp`    — Stages 7–9
 *
 * Stage numbering matches the `// ===== Step N =====` headings of the
 * original monolithic body (pre-split at commit `0e65664`):
 *   0 — PCA dim → pca_dim
 *   1 — cosine similarity (with optional temporal mixing = old Step 1b)
 *   2 — p-pruning
 *   3 — symmetrize + clamp + orphan repair
 *   4 — normalized Laplacian eigendecomposition
 *   5 — eigengap K-selection
 *   6 — K-means++ on spectral features with multi-restart
 *   7 — temporal smoothing
 *   8 — centroid computation in original (pre-PCA) space
 *   9 — agglomerative merge of similar centroids
 *
 * No public consumer should include this header. It is a peer-TU detail.
 */
#pragma once

#include "spectral_cluster.h"

#include <vector>

namespace deusridet::spectral_detail {

// Step 0: PCA reduce (dim → pca_dim). Returns N row-vectors of length
// pca_dim, each L2-normalised.
std::vector<std::vector<float>> pca_reduce(
    const std::vector<std::vector<float>>& embeddings,
    int dim,
    int pca_dim,
    int power_iters);

// Step 1 (+ optional 1b temporal mixing): cosine similarity matrix.
// Returns a symmetric N×N matrix stored row-major.
std::vector<float> build_similarity(
    const std::vector<std::vector<float>>& pca_emb,
    int pca_dim,
    int N,
    const std::vector<float>& timestamps_sec,
    float temporal_alpha,
    float temporal_tau);

// Step 2: keep only the top-p neighbours per row (in place).
void p_prune(std::vector<float>& sim, int N, float p_prune_ratio);

// Step 3: symmetrise, clamp to non-negative, repair orphans by reconnecting
// each row's nearest PCA-space neighbour (in place).
void symmetrize_and_repair_orphans(
    std::vector<float>& sim,
    int N,
    const std::vector<std::vector<float>>& pca_emb,
    int pca_dim);

// Step 4: top-`max_k` eigenpairs of the normalised Laplacian via deflated
// power iteration.
void laplacian_eigendecomp(
    const std::vector<float>& sim,
    int N,
    int max_k,
    int power_iters,
    std::vector<std::vector<float>>& eigvecs_out,
    std::vector<float>& eigvals_out);

// Step 5: choose K by the largest gap-score (NME + relative gap).
int select_k_by_eigengap(
    const std::vector<float>& eigvals,
    int actual_max,
    int cfg_min_k,
    int cfg_max_k);

// Step 6: K-means++ on spectral features; multi-restart keeps the lowest-
// inertia labelling.
std::vector<int> kmeans_pp_spectral(
    const std::vector<std::vector<float>>& eigvecs,
    int N,
    int optimal_k,
    int kmeans_restarts,
    int kmeans_iters);

// Step 7: majority-vote smoothing by temporal neighbour, guarded by a
// centroid-similarity confirmation (in place).
void temporal_smooth(
    std::vector<int>& labels,
    int N,
    int optimal_k,
    const std::vector<std::vector<float>>& pca_emb,
    int pca_dim,
    const std::vector<float>& timestamps_sec,
    int smooth_window,
    int smooth_iters);

// Step 8: compute centroids in the ORIGINAL (pre-PCA) embedding space,
// L2-normalise each. Returns per-cluster member counts via `ccnt_out`.
void compute_original_centroids(
    ClusterResult& result,
    const std::vector<std::vector<float>>& embeddings,
    int dim,
    std::vector<int>& ccnt_out);

// Step 9: agglomeratively merge cluster pairs whose centroid cosine
// similarity exceeds `merge_threshold`, down to `min_k`. Relabels and
// compacts `result` in place.
void merge_similar_centroids(
    ClusterResult& result,
    std::vector<int>& ccnt,
    int dim,
    float merge_threshold,
    int min_k);

} // namespace deusridet::spectral_detail
