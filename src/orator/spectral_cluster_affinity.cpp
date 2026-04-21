/**
 * @file spectral_cluster_affinity.cpp
 * @philosophical_role Stages 0–3 of `spectral_cluster()`: embed the points
 *     into a compact PCA subspace and build the cleaned, symmetrised
 *     affinity matrix that downstream eigendecomposition will read. This is
 *     the "how close is every pair" phase — no eigenstructure yet, only
 *     geometry and pruning.
 * @serves `spectral_cluster.cpp` as a decomposition peer alongside
 *     `spectral_cluster_embed.cpp` and `spectral_cluster_postprocess.cpp`.
 *     Bodies copied verbatim from the pre-split
 *     `spectral_cluster_stages.cpp` (commit `a34b4a9`); only the file
 *     boundary moved. Contract in `spectral_cluster_stages.h`.
 */
#include "spectral_cluster_stages.h"

#include <algorithm>
#include <cmath>
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

} // namespace deusridet::spectral_detail
