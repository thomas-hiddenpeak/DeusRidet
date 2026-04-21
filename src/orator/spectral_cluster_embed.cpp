/**
 * @file spectral_cluster_embed.cpp
 * @philosophical_role Stages 4–6 of `spectral_cluster()`: extract the
 *     spectral embedding from the affinity matrix, pick K by eigengap, and
 *     partition the points by K-means++ in the spectral subspace. This is
 *     the "how many speakers, and which point belongs to which" phase.
 * @serves `spectral_cluster.cpp` as a decomposition peer alongside
 *     `spectral_cluster_affinity.cpp` and `spectral_cluster_postprocess.cpp`.
 *     Bodies copied verbatim from the pre-split
 *     `spectral_cluster_stages.cpp` (commit `a34b4a9`); only the file
 *     boundary moved. Contract in `spectral_cluster_stages.h`.
 */
#include "spectral_cluster_stages.h"

#include "communis/log.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace deusridet::spectral_detail {

// ===== Step 4: Normalized Laplacian eigendecomposition =====
void laplacian_eigendecomp(
    const std::vector<float>& sim,
    int N,
    int max_k,
    int power_iters,
    std::vector<std::vector<float>>& eigvecs_out,
    std::vector<float>& eigvals_out)
{
    // 4a: degree vector
    std::vector<float> D(N, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            D[i] += sim[i * N + j];

    // 4b: D^{-1/2} S D^{-1/2}
    std::vector<float> D_inv_sqrt(N);
    for (int i = 0; i < N; ++i)
        D_inv_sqrt[i] = (D[i] > 1e-12f) ? 1.0f / sqrtf(D[i]) : 0.0f;

    std::vector<float> Lsym(N * N, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Lsym[i * N + j] = D_inv_sqrt[i] * sim[i * N + j] * D_inv_sqrt[j];

    // 4c: power iteration for top-max_k eigenvectors
    eigvecs_out.assign(max_k, std::vector<float>(N, 0));
    eigvals_out.assign(max_k, 0);
    std::vector<float> Lwork(Lsym);
    for (int k = 0; k < max_k; ++k) {
        std::vector<float> v(N);
        for (int i = 0; i < N; ++i)
            v[i] = (float)(i + k * 7 + 1);
        float vnorm = 0;
        for (float x : v) vnorm += x * x;
        vnorm = sqrtf(vnorm + 1e-12f);
        for (float& x : v) x /= vnorm;

        for (int iter = 0; iter < power_iters; ++iter) {
            std::vector<float> Av(N, 0.0f);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    Av[i] += Lwork[i * N + j] * v[j];
            float norm2 = 0;
            for (float x : Av) norm2 += x * x;
            float inorm = 1.0f / sqrtf(norm2 + 1e-12f);
            for (int i = 0; i < N; ++i)
                v[i] = Av[i] * inorm;
        }

        float lambda = 0;
        for (int i = 0; i < N; ++i) {
            float Av_i = 0;
            for (int j = 0; j < N; ++j)
                Av_i += Lwork[i * N + j] * v[j];
            lambda += v[i] * Av_i;
        }
        eigvals_out[k] = lambda;
        eigvecs_out[k] = v;

        // deflate
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                Lwork[i * N + j] -= lambda * v[i] * v[j];
    }
}

// ===== Step 5: Eigengap K-selection =====
int select_k_by_eigengap(
    const std::vector<float>& eigvals,
    int actual_max,
    int cfg_min_k,
    int cfg_max_k)
{
    int optimal_k = cfg_min_k;
    LOG_INFO("SpCluster", "Eigenvalues (top-%d):", actual_max);
    for (int k = 0; k < actual_max && k < 8; ++k)
        LOG_INFO("SpCluster", "  λ[%d] = %.6f", k, eigvals[k]);

    float max_gap_score = 0;
    for (int k = 0; k + 1 < actual_max; ++k) {
        float gap = eigvals[k] - eigvals[k + 1];
        if (eigvals[k] < 0.01f) continue;
        float rel_gap = gap / (eigvals[0] + 1e-12f);
        float nme = gap / (k + 1);
        float score = nme + 0.3f * rel_gap;
        LOG_INFO("SpCluster", "  gap[%d→%d]: gap=%.6f rel=%.4f nme=%.6f score=%.6f",
                 k, k + 1, gap, rel_gap, nme, score);
        if (score > max_gap_score) {
            max_gap_score = score;
            optimal_k = k + 1;
        }
    }
    LOG_INFO("SpCluster", "Optimal K=%d (max_gap_score=%.6f)", optimal_k, max_gap_score);
    return std::max(cfg_min_k, std::min(optimal_k, cfg_max_k));
}

// ===== Step 6: K-means++ on spectral features =====
std::vector<int> kmeans_pp_spectral(
    const std::vector<std::vector<float>>& eigvecs,
    int N,
    int optimal_k,
    int kmeans_restarts,
    int kmeans_iters)
{
    // 6a: extract spectral features (N × optimal_k), L2-normalize rows
    std::vector<float> features(N * optimal_k);
    for (int i = 0; i < N; ++i) {
        float rnorm = 0;
        for (int k = 0; k < optimal_k; ++k) {
            features[i * optimal_k + k] = eigvecs[k][i];
            rnorm += eigvecs[k][i] * eigvecs[k][i];
        }
        rnorm = 1.0f / sqrtf(rnorm + 1e-12f);
        for (int k = 0; k < optimal_k; ++k)
            features[i * optimal_k + k] *= rnorm;
    }

    // 6b: K-means++ with multi-restart
    std::vector<int> labels(N, 0);
    float best_inertia = 1e30f;

    for (int restart = 0; restart < kmeans_restarts; ++restart) {
        std::vector<std::vector<float>> centroids(optimal_k, std::vector<float>(optimal_k, 0));
        std::vector<int> cur_labels(N, 0);

        // K-means++ init: first centroid
        int seed = restart * 137 % N;
        for (int j = 0; j < optimal_k; ++j)
            centroids[0][j] = features[seed * optimal_k + j];

        // remaining centroids (farthest point heuristic)
        for (int c = 1; c < optimal_k; ++c) {
            float best_d = -1;
            int best_i = 0;
            for (int i = 0; i < N; ++i) {
                float min_d = 1e30f;
                for (int prev = 0; prev < c; ++prev) {
                    float d = 0;
                    for (int j = 0; j < optimal_k; ++j) {
                        float diff = features[i * optimal_k + j] - centroids[prev][j];
                        d += diff * diff;
                    }
                    min_d = std::min(min_d, d);
                }
                if (min_d > best_d) {
                    best_d = min_d;
                    best_i = i;
                }
            }
            for (int j = 0; j < optimal_k; ++j)
                centroids[c][j] = features[best_i * optimal_k + j];
        }

        // iterate
        for (int iter = 0; iter < kmeans_iters; ++iter) {
            int changed = 0;
            for (int i = 0; i < N; ++i) {
                float best_d = 1e30f;
                int best_c = 0;
                for (int c = 0; c < optimal_k; ++c) {
                    float d = 0;
                    for (int j = 0; j < optimal_k; ++j) {
                        float diff = features[i * optimal_k + j] - centroids[c][j];
                        d += diff * diff;
                    }
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                if (best_c != cur_labels[i]) ++changed;
                cur_labels[i] = best_c;
            }

            // update centroids
            for (int c = 0; c < optimal_k; ++c)
                std::fill(centroids[c].begin(), centroids[c].end(), 0.0f);
            std::vector<int> cnt(optimal_k, 0);
            for (int i = 0; i < N; ++i) {
                cnt[cur_labels[i]]++;
                for (int j = 0; j < optimal_k; ++j)
                    centroids[cur_labels[i]][j] += features[i * optimal_k + j];
            }
            for (int c = 0; c < optimal_k; ++c)
                if (cnt[c] > 0)
                    for (int j = 0; j < optimal_k; ++j)
                        centroids[c][j] /= cnt[c];

            if (changed == 0) break;
        }

        // inertia
        float inertia = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < optimal_k; ++j) {
                float diff = features[i * optimal_k + j] - centroids[cur_labels[i]][j];
                inertia += diff * diff;
            }

        if (inertia < best_inertia) {
            best_inertia = inertia;
            labels = cur_labels;
        }
    }
    return labels;
}

} // namespace deusridet::spectral_detail
