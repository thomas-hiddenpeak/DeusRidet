/**
 * @file spectral_cluster_stages.cpp
 * @philosophical_role Stage implementations for `spectral_cluster()`. Each
 *     function here maps one-to-one onto one `// ===== Step N =====` block
 *     of the original 560-line monolithic body. Bodies are copied verbatim
 *     from pre-split `spectral_cluster.cpp` (commit `0e65664`); the only
 *     delta is parameter plumbing (what was a local variable in the
 *     monolith is now a parameter or a return value).
 * @serves `spectral_cluster.cpp` as its decomposition peer. Private
 *     contract declared in `spectral_cluster_stages.h`; no other TU should
 *     include that header.
 */
#include "spectral_cluster_stages.h"

#include "communis/log.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace deusridet::spectral_detail {

// ===== Step 0: PCA (dim → pca_dim) =====
std::vector<std::vector<float>> pca_reduce(
    const std::vector<std::vector<float>>& embeddings,
    int dim,
    int pca_dim,
    int power_iters)
{
    const int N = (int)embeddings.size();

    // 0a: compute mean
    std::vector<float> mean(dim, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < dim; ++d)
            mean[d] += embeddings[i][d];
    float inv_n = 1.0f / N;
    for (int d = 0; d < dim; ++d)
        mean[d] *= inv_n;

    // 0b: center data
    std::vector<float> X(N * dim);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < dim; ++d)
            X[i * dim + d] = embeddings[i][d] - mean[d];

    // 0c: covariance matrix (dim × dim)
    std::vector<float> cov(dim * dim, 0.0f);
    for (int i = 0; i < N; ++i) {
        const float* xi = &X[i * dim];
        for (int a = 0; a < dim; ++a) {
            float xa = xi[a];
            for (int b = a; b < dim; ++b)
                cov[a * dim + b] += xa * xi[b];
        }
    }
    float inv_nm1 = 1.0f / std::max(1, N - 1);
    for (int a = 0; a < dim; ++a) {
        for (int b = a; b < dim; ++b) {
            cov[a * dim + b] *= inv_nm1;
            cov[b * dim + a] = cov[a * dim + b];
        }
    }

    // 0d: power iteration for top-pca_dim eigenvectors of cov
    std::vector<std::vector<float>> pc(pca_dim, std::vector<float>(dim));
    {
        std::vector<float> Cwork(cov);
        for (int k = 0; k < pca_dim; ++k) {
            std::vector<float> v(dim);
            // deterministic init
            for (int d = 0; d < dim; ++d)
                v[d] = (float)(d + k * 7 + 1);
            float vnorm = 0;
            for (float x : v) vnorm += x * x;
            vnorm = sqrtf(vnorm + 1e-12f);
            for (float& x : v) x /= vnorm;

            for (int iter = 0; iter < power_iters; ++iter) {
                std::vector<float> Av(dim, 0.0f);
                for (int i = 0; i < dim; ++i)
                    for (int j = 0; j < dim; ++j)
                        Av[i] += Cwork[i * dim + j] * v[j];
                float norm2 = 0;
                for (float x : Av) norm2 += x * x;
                float inorm = 1.0f / sqrtf(norm2 + 1e-12f);
                for (int i = 0; i < dim; ++i)
                    v[i] = Av[i] * inorm;
            }

            // eigenvalue = v^T C v
            float lambda = 0;
            for (int i = 0; i < dim; ++i) {
                float Av_i = 0;
                for (int j = 0; j < dim; ++j)
                    Av_i += Cwork[i * dim + j] * v[j];
                lambda += v[i] * Av_i;
            }
            pc[k] = v;

            // deflate: C -= lambda * v v^T
            for (int i = 0; i < dim; ++i)
                for (int j = 0; j < dim; ++j)
                    Cwork[i * dim + j] -= lambda * v[i] * v[j];
        }
    }

    // 0e: project onto PCs and L2-normalize
    std::vector<std::vector<float>> pca_emb(N, std::vector<float>(pca_dim));
    for (int i = 0; i < N; ++i) {
        const float* xi = &X[i * dim];
        float norm2 = 0;
        for (int k = 0; k < pca_dim; ++k) {
            float proj = 0;
            for (int d = 0; d < dim; ++d)
                proj += xi[d] * pc[k][d];
            pca_emb[i][k] = proj;
            norm2 += proj * proj;
        }
        float inorm = 1.0f / sqrtf(norm2 + 1e-12f);
        for (int k = 0; k < pca_dim; ++k)
            pca_emb[i][k] *= inorm;
    }
    return pca_emb;
}

// ===== Step 1 (+ 1b): cosine similarity matrix + optional temporal mixing =====
std::vector<float> build_similarity(
    const std::vector<std::vector<float>>& pca_emb,
    int pca_dim,
    int N,
    const std::vector<float>& timestamps_sec,
    float temporal_alpha,
    float temporal_tau)
{
    std::vector<float> sim(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        sim[i * N + i] = 1.0f;
        for (int j = i + 1; j < N; ++j) {
            float dot = 0;
            for (int k = 0; k < pca_dim; ++k)
                dot += pca_emb[i][k] * pca_emb[j][k];
            sim[i * N + j] = dot;
            sim[j * N + i] = dot;
        }
    }

    bool use_temporal = !timestamps_sec.empty() && temporal_alpha > 0.0f;
    if (use_temporal) {
        float inv_2tau2 = 1.0f / (2.0f * temporal_tau * temporal_tau);
        float alpha = temporal_alpha;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                float dt = timestamps_sec[i] - timestamps_sec[j];
                float t_prox = expf(-dt * dt * inv_2tau2);
                float combined = (1.0f - alpha) * sim[i * N + j] + alpha * t_prox;
                sim[i * N + j] = combined;
                sim[j * N + i] = combined;
            }
        }
    }
    return sim;
}

// ===== Step 2: P-pruning =====
void p_prune(std::vector<float>& sim, int N, float p_prune_ratio)
{
    int p = std::max(3, (int)(N * p_prune_ratio));
    p = std::min(p, N - 1);
    for (int i = 0; i < N; ++i) {
        std::vector<float> row_vals;
        row_vals.reserve(N - 1);
        for (int j = 0; j < N; ++j)
            if (j != i) row_vals.push_back(sim[i * N + j]);
        std::sort(row_vals.rbegin(), row_vals.rend());
        float thresh = (p < (int)row_vals.size()) ? row_vals[p] : -2.0f;
        for (int j = 0; j < N; ++j)
            if (j != i && sim[i * N + j] < thresh)
                sim[i * N + j] = 0.0f;
    }
}

// ===== Step 3: Symmetrize + clamp + orphan repair =====
void symmetrize_and_repair_orphans(
    std::vector<float>& sim,
    int N,
    const std::vector<std::vector<float>>& pca_emb,
    int pca_dim)
{
    for (int i = 0; i < N; ++i) {
        sim[i * N + i] = 0.0f;
        for (int j = i + 1; j < N; ++j) {
            float val = (sim[i * N + j] + sim[j * N + i]) * 0.5f;
            val = std::max(0.0f, val);
            sim[i * N + j] = val;
            sim[j * N + i] = val;
        }
    }
    // orphan repair
    for (int i = 0; i < N; ++i) {
        float row_sum = 0;
        for (int j = 0; j < N; ++j)
            if (j != i) row_sum += sim[i * N + j];
        if (row_sum < 1e-12f) {
            float best = -2; int best_j = 0;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                float dot = 0;
                for (int k = 0; k < pca_dim; ++k)
                    dot += pca_emb[i][k] * pca_emb[j][k];
                if (dot > best) { best = dot; best_j = j; }
            }
            float v = std::max(0.01f, best);
            sim[i * N + best_j] = v;
            sim[best_j * N + i] = v;
        }
    }
}

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
