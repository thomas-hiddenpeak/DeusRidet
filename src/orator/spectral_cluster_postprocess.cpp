/**
 * @file spectral_cluster_postprocess.cpp
 * @philosophical_role Stages 7–9 of `spectral_cluster()`: smooth the raw
 *     label sequence against temporal neighbours, compute centroids in the
 *     original (pre-PCA) embedding space, and agglomeratively merge
 *     over-split clusters. This is the "what do the clusters actually look
 *     like, and which pairs are really the same speaker" phase.
 * @serves `spectral_cluster.cpp` as a decomposition peer alongside
 *     `spectral_cluster_affinity.cpp` and `spectral_cluster_embed.cpp`.
 *     Bodies copied verbatim from the pre-split
 *     `spectral_cluster_stages.cpp` (commit `a34b4a9`); only the file
 *     boundary moved. Contract in `spectral_cluster_stages.h`.
 */
#include "spectral_cluster_stages.h"

#include "communis/log.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace deusridet::spectral_detail {

// ===== Step 7: Temporal smoothing =====
void temporal_smooth(
    std::vector<int>& labels,
    int N,
    int optimal_k,
    const std::vector<std::vector<float>>& pca_emb,
    int pca_dim,
    const std::vector<float>& timestamps_sec,
    int smooth_window,
    int smooth_iters)
{
    // sort by timestamp (or index if no timestamps)
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    if (!timestamps_sec.empty()) {
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return timestamps_sec[a] < timestamps_sec[b];
        });
    }

    // cluster centroids in PCA space (for centroid confirmation)
    std::vector<std::vector<float>> clust_pca(optimal_k, std::vector<float>(pca_dim, 0));
    std::vector<int> clust_cnt(optimal_k, 0);
    for (int i = 0; i < N; ++i) {
        clust_cnt[labels[i]]++;
        for (int d = 0; d < pca_dim; ++d)
            clust_pca[labels[i]][d] += pca_emb[i][d];
    }
    for (int c = 0; c < optimal_k; ++c) {
        if (clust_cnt[c] > 0) {
            for (int d = 0; d < pca_dim; ++d)
                clust_pca[c][d] /= clust_cnt[c];
            float n2 = 0;
            for (float v : clust_pca[c]) n2 += v * v;
            float inv = 1.0f / sqrtf(n2 + 1e-12f);
            for (float& v : clust_pca[c]) v *= inv;
        }
    }

    for (int sit = 0; sit < smooth_iters; ++sit) {
        int changed = 0;
        for (int oi = 0; oi < N; ++oi) {
            int idx = order[oi];
            int lo = std::max(0, oi - smooth_window);
            int hi = std::min(N - 1, oi + smooth_window);

            float vote[64] = {};
            for (int ni = lo; ni <= hi; ++ni) {
                if (ni == oi) continue;
                int nidx = order[ni];
                float s = 0;
                for (int d = 0; d < pca_dim; ++d)
                    s += pca_emb[idx][d] * pca_emb[nidx][d];
                vote[labels[nidx]] += s;
            }

            int maj = -1;
            float maj_score = 0;
            for (int c = 0; c < optimal_k; ++c) {
                if (vote[c] > maj_score) {
                    maj_score = vote[c];
                    maj = c;
                }
            }

            if (maj >= 0 && maj != labels[idx]) {
                float old_sim = 0, new_sim = 0;
                for (int d = 0; d < pca_dim; ++d) {
                    old_sim += pca_emb[idx][d] * clust_pca[labels[idx]][d];
                    new_sim += pca_emb[idx][d] * clust_pca[maj][d];
                }
                if (new_sim > old_sim) {
                    labels[idx] = maj;
                    ++changed;
                }
            }
        }
        if (changed == 0) break;

        // recompute centroids
        for (int c = 0; c < optimal_k; ++c) {
            std::fill(clust_pca[c].begin(), clust_pca[c].end(), 0.0f);
            clust_cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i) {
            clust_cnt[labels[i]]++;
            for (int d = 0; d < pca_dim; ++d)
                clust_pca[labels[i]][d] += pca_emb[i][d];
        }
        for (int c = 0; c < optimal_k; ++c) {
            if (clust_cnt[c] > 0) {
                for (int d = 0; d < pca_dim; ++d)
                    clust_pca[c][d] /= clust_cnt[c];
                float n2 = 0;
                for (float v : clust_pca[c]) n2 += v * v;
                float inv = 1.0f / sqrtf(n2 + 1e-12f);
                for (float& v : clust_pca[c]) v *= inv;
            }
        }
    }
}

// ===== Step 8: Compute centroids in ORIGINAL space =====
void compute_original_centroids(
    ClusterResult& result,
    const std::vector<std::vector<float>>& embeddings,
    int dim,
    std::vector<int>& ccnt_out)
{
    const int N = (int)embeddings.size();
    const int K = result.K;
    result.centroids.assign(K, std::vector<float>(dim, 0.0f));
    ccnt_out.assign(K, 0);
    for (int i = 0; i < N; ++i) {
        ccnt_out[result.labels[i]]++;
        for (int d = 0; d < dim; ++d)
            result.centroids[result.labels[i]][d] += embeddings[i][d];
    }
    for (int c = 0; c < K; ++c) {
        if (ccnt_out[c] > 0) {
            for (int d = 0; d < dim; ++d)
                result.centroids[c][d] /= ccnt_out[c];
            // L2-normalize
            float n2 = 0;
            for (int d = 0; d < dim; ++d)
                n2 += result.centroids[c][d] * result.centroids[c][d];
            float inv = 1.0f / sqrtf(n2 + 1e-12f);
            for (int d = 0; d < dim; ++d)
                result.centroids[c][d] *= inv;
        }
    }
}

// ===== Step 9: Post-clustering centroid merge =====
void merge_similar_centroids(
    ClusterResult& result,
    std::vector<int>& ccnt,
    int dim,
    float merge_threshold,
    int min_k)
{
    if (!(merge_threshold > 0.0f && result.K > min_k))
        return;

    // Map: active cluster IDs. Initially 0..K-1.
    std::vector<int> active;
    for (int c = 0; c < result.K; ++c)
        active.push_back(c);

    bool merged_any = true;
    while (merged_any && (int)active.size() > min_k) {
        merged_any = false;
        // Find the most similar pair.
        float best_sim = -1;
        int best_a = -1, best_b = -1;
        for (int ai = 0; ai < (int)active.size(); ++ai) {
            for (int bi = ai + 1; bi < (int)active.size(); ++bi) {
                float dot = 0;
                for (int d = 0; d < dim; ++d)
                    dot += result.centroids[active[ai]][d] *
                           result.centroids[active[bi]][d];
                if (dot > best_sim) {
                    best_sim = dot;
                    best_a = ai;
                    best_b = bi;
                }
            }
        }
        if (best_sim >= merge_threshold && best_a >= 0) {
            int ca = active[best_a], cb = active[best_b];
            int na = ccnt[ca], nb = ccnt[cb];
            LOG_INFO("SpCluster", "Merge: cluster %d (%d) + cluster %d (%d), sim=%.4f",
                     ca, na, cb, nb, best_sim);
            // Merge cb into ca: weighted centroid average.
            for (int d = 0; d < dim; ++d) {
                result.centroids[ca][d] = (result.centroids[ca][d] * na +
                                           result.centroids[cb][d] * nb) /
                                          (na + nb);
            }
            // L2-normalize merged centroid.
            float n2 = 0;
            for (int d = 0; d < dim; ++d)
                n2 += result.centroids[ca][d] * result.centroids[ca][d];
            float inv = 1.0f / sqrtf(n2 + 1e-12f);
            for (int d = 0; d < dim; ++d)
                result.centroids[ca][d] *= inv;
            ccnt[ca] += nb;
            // Relabel all cb → ca.
            for (int& lbl : result.labels)
                if (lbl == cb) lbl = ca;
            // Remove cb from active list.
            active.erase(active.begin() + best_b);
            merged_any = true;
        }
    }

    // Compact: renumber labels to 0..K'-1.
    int new_K = (int)active.size();
    std::vector<std::vector<float>> new_centroids(new_K);
    std::vector<int> remap(result.K, -1);
    for (int i = 0; i < new_K; ++i) {
        remap[active[i]] = i;
        new_centroids[i] = result.centroids[active[i]];
    }
    for (int& lbl : result.labels)
        lbl = remap[lbl];
    result.K = new_K;
    result.centroids = std::move(new_centroids);
}

} // namespace deusridet::spectral_detail
