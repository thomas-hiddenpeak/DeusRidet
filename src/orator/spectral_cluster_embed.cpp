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
//
// mode = 0 (legacy NME + rel_gap): score[k] = gap/(k+1) + 0.3*gap/λ[0].
//     The 1/(k+1) NME term strongly biases the argmax toward small K and on
//     384-D fused embeddings collapses every window to K=2 (the dominant
//     conversational split) even when 4 speakers are clearly present.
//
// mode = 1 (eigenvalue ratio): score[k] = λ[k] / λ[k+1].
//     Parameter-free. K = 1 + argmax_{k≥1} score[k]. Robust on the s1800
//     fixture: picks K=4 (true value) instead of K=2.
int select_k_by_eigengap(
    const std::vector<float>& eigvals,
    int actual_max,
    int cfg_min_k,
    int cfg_max_k,
    int mode)
{
    int optimal_k = cfg_min_k;
    LOG_INFO("SpCluster", "Eigenvalues (top-%d, mode=%d):", actual_max, mode);
    for (int k = 0; k < actual_max && k < 8; ++k)
        LOG_INFO("SpCluster", "  λ[%d] = %.6f", k, eigvals[k]);

    float max_score = 0;
    for (int k = 0; k + 1 < actual_max; ++k) {
        // Skip trivial first eigengap: λ[0] is the connected-component eigenvalue
        // and dominates the score, forcing K=1 in degenerate cases.
        if (k == 0) continue;
        const float gap = eigvals[k] - eigvals[k + 1];
        if (eigvals[k] < 0.01f) continue;
        float score = 0.0f;
        if (mode == 1) {
            // Eigenvalue ratio λ[k]/λ[k+1]. Argmax marks the elbow.
            // Guard: when λ[k+1] is near zero (degenerate small-N windows or
            // disconnected affinity graph) the ratio explodes and dominates;
            // require λ[k+1] ≥ 0.05 to consider this gap a meaningful elbow.
            if (eigvals[k + 1] < 0.05f) {
                LOG_INFO("SpCluster", "  ratio[%d→%d]: λ_k+1=%.6f < 0.05 (skip)",
                         k, k + 1, eigvals[k + 1]);
                continue;
            }
            score = eigvals[k] / eigvals[k + 1];
            LOG_INFO("SpCluster", "  ratio[%d→%d]: λ_k=%.6f λ_k+1=%.6f score=%.6f",
                     k, k + 1, eigvals[k], eigvals[k + 1], score);
        } else {
            const float rel_gap = gap / (eigvals[0] + 1e-12f);
            const float nme     = gap / (k + 1);
            score = nme + 0.3f * rel_gap;
            LOG_INFO("SpCluster", "  gap[%d→%d]: gap=%.6f rel=%.4f nme=%.6f score=%.6f",
                     k, k + 1, gap, rel_gap, nme, score);
        }
        if (score > max_score) {
            max_score = score;
            optimal_k = k + 1;
        }
    }
    LOG_INFO("SpCluster", "Optimal K=%d (max_score=%.6f)", optimal_k, max_score);
    return std::max(cfg_min_k, std::min(optimal_k, cfg_max_k));
}

// ===== Step 6: K-means++ on spectral features =====
std::vector<int> kmeans_pp_spectral(
    const std::vector<std::vector<float>>& eigvecs,
    int N,
    int optimal_k,
    int kmeans_restarts,
    int kmeans_iters,
    const std::vector<float>& weights)
{
    const bool use_w = (int)weights.size() == N;
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

            // update centroids (length-weighted when use_w; uniform otherwise)
            for (int c = 0; c < optimal_k; ++c)
                std::fill(centroids[c].begin(), centroids[c].end(), 0.0f);
            std::vector<float> cnt(optimal_k, 0.0f);
            for (int i = 0; i < N; ++i) {
                const float w = use_w ? weights[i] : 1.0f;
                cnt[cur_labels[i]] += w;
                for (int j = 0; j < optimal_k; ++j)
                    centroids[cur_labels[i]][j] += w * features[i * optimal_k + j];
            }
            for (int c = 0; c < optimal_k; ++c)
                if (cnt[c] > 0.0f)
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

// ===== Step 6b (Phase 8): per-cluster purity post-filter =====
// Splits contaminated clusters (low mean intra-cluster cosine) by running
// a K=2 K-means++ on members in the ORIGINAL embedding space. The split
// is accepted only when the two sub-centroids are far enough apart.
//
// All distance work happens on unit-norm vectors, so squared Euclidean
// distance and cosine distance differ by a constant factor — we use
// squared Euclidean for the K-means loop and cosine for the acceptance
// gate (more interpretable).
int purity_split_clusters(
    std::vector<int>& labels,
    int current_k,
    const std::vector<std::vector<float>>& embeddings,
    int dim,
    int min_cluster_size,
    float min_mean_cos,
    float accept_max_subsim,
    int split_kmeans_iters,
    int split_kmeans_restarts)
{
    const int N = (int)labels.size();
    if (N <= 0 || current_k <= 0 || dim <= 0) return current_k;

    int K = current_k;
    int splits_done = 0;

    // Snapshot the original label set so we never recursively split a
    // cluster we just produced this pass.
    const int original_k = current_k;

    for (int c = 0; c < original_k; ++c) {
        // Gather members of cluster c.
        std::vector<int> members;
        members.reserve(N);
        for (int i = 0; i < N; ++i) if (labels[i] == c) members.push_back(i);
        const int M = (int)members.size();
        if (M < min_cluster_size) continue;

        // L2-normed centroid in original space.
        std::vector<float> centroid(dim, 0.0f);
        for (int idx : members) {
            const auto& e = embeddings[idx];
            for (int d = 0; d < dim; ++d) centroid[d] += e[d];
        }
        float cnorm = 0.0f;
        for (int d = 0; d < dim; ++d) cnorm += centroid[d] * centroid[d];
        cnorm = 1.0f / sqrtf(cnorm + 1e-12f);
        for (int d = 0; d < dim; ++d) centroid[d] *= cnorm;

        // Mean cosine of members to centroid (members already L2-normed).
        float sum_cos = 0.0f;
        for (int idx : members) {
            const auto& e = embeddings[idx];
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) s += e[d] * centroid[d];
            sum_cos += s;
        }
        float mean_cos = sum_cos / (float)M;
        if (mean_cos >= min_mean_cos) continue;  // cluster is tight; skip.

        // ── K=2 K-means++ on member original embeddings, multi-restart ──
        std::vector<int> best_sub(M, 0);
        float best_inertia = 1e30f;
        std::vector<float> best_c0(dim, 0.0f), best_c1(dim, 0.0f);

        for (int restart = 0; restart < split_kmeans_restarts; ++restart) {
            // K-means++ init: first centroid = deterministic per-restart pick.
            int i0 = (restart * 1009) % M;
            std::vector<float> c0(dim), c1(dim);
            for (int d = 0; d < dim; ++d) c0[d] = embeddings[members[i0]][d];

            // Second centroid = farthest member from c0 (deterministic).
            int i1 = i0;
            float far_d = -1.0f;
            for (int m = 0; m < M; ++m) {
                const auto& e = embeddings[members[m]];
                float d2 = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = e[d] - c0[d];
                    d2 += diff * diff;
                }
                if (d2 > far_d) { far_d = d2; i1 = m; }
            }
            for (int d = 0; d < dim; ++d) c1[d] = embeddings[members[i1]][d];

            std::vector<int> sub(M, 0);
            for (int iter = 0; iter < split_kmeans_iters; ++iter) {
                int changed = 0;
                // Assign.
                for (int m = 0; m < M; ++m) {
                    const auto& e = embeddings[members[m]];
                    float d0 = 0.0f, d1 = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        float a = e[d] - c0[d];
                        float b = e[d] - c1[d];
                        d0 += a * a;
                        d1 += b * b;
                    }
                    int chosen = (d0 <= d1) ? 0 : 1;
                    if (chosen != sub[m]) ++changed;
                    sub[m] = chosen;
                }

                // Update (L2-normed centroids — embeddings are unit vectors).
                std::fill(c0.begin(), c0.end(), 0.0f);
                std::fill(c1.begin(), c1.end(), 0.0f);
                int n0 = 0, n1 = 0;
                for (int m = 0; m < M; ++m) {
                    const auto& e = embeddings[members[m]];
                    if (sub[m] == 0) {
                        for (int d = 0; d < dim; ++d) c0[d] += e[d];
                        ++n0;
                    } else {
                        for (int d = 0; d < dim; ++d) c1[d] += e[d];
                        ++n1;
                    }
                }
                auto renorm = [&](std::vector<float>& v) {
                    float nrm = 0.0f;
                    for (int d = 0; d < dim; ++d) nrm += v[d] * v[d];
                    nrm = 1.0f / sqrtf(nrm + 1e-12f);
                    for (int d = 0; d < dim; ++d) v[d] *= nrm;
                };
                if (n0 > 0) renorm(c0);
                if (n1 > 0) renorm(c1);
                if (changed == 0) break;
            }

            // Inertia (squared-Euclidean to chosen centroid).
            float inertia = 0.0f;
            for (int m = 0; m < M; ++m) {
                const auto& e = embeddings[members[m]];
                const auto& cc = (sub[m] == 0) ? c0 : c1;
                for (int d = 0; d < dim; ++d) {
                    float diff = e[d] - cc[d];
                    inertia += diff * diff;
                }
            }
            if (inertia < best_inertia) {
                best_inertia = inertia;
                best_sub = sub;
                best_c0 = c0;
                best_c1 = c1;
            }
        }

        // Acceptance gate: sub-centroid cosine must be < accept_max_subsim.
        float sub_cos = 0.0f;
        for (int d = 0; d < dim; ++d) sub_cos += best_c0[d] * best_c1[d];
        if (sub_cos >= accept_max_subsim) continue;  // not actually 2 spks.

        // Guard against degenerate splits (all-on-one-side).
        int n0 = 0, n1 = 0;
        for (int s : best_sub) { if (s == 0) ++n0; else ++n1; }
        if (n0 == 0 || n1 == 0) continue;

        // Commit split: side 0 keeps label c, side 1 becomes new label K.
        const int new_label = K;
        for (int m = 0; m < M; ++m) {
            if (best_sub[m] == 1) labels[members[m]] = new_label;
        }
        ++K;
        ++splits_done;
    }

    if (splits_done > 0) {
        LOG_INFO("SpCluster",
                 "purity_split: %d cluster(s) split; K: %d -> %d",
                 splits_done, current_k, K);
    }
    return K;
}

} // namespace deusridet::spectral_detail
