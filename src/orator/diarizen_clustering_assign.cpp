// @role: DiariZenClustering whole-stage assignment (P2b-4). filter_embeddings
//        -> AHC -> VBx EM -> PLDA-weighted centroids -> cosine cdist ->
//        constrained (Hungarian) argmax -> renumber. Reproduces pyannote
//        VBxClustering.__call__ bit-for-bit. Small-N serial CPU glue.
#include "diarizen_clustering.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace deusridet {
namespace orator {

namespace {
// Temporary sub-stage profiler for the clustering CPU path. Gated by
// DEUSRIDET_DIARIZEN_CLUSTER_PROF=1 so it is silent by default. Used to
// decide which sub-stage (AHC / fea / VBx / cdist / assignment) to port to
// the GPU per the project's GPU-first rule.
inline bool cluster_prof_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("DEUSRIDET_DIARIZEN_CLUSTER_PROF");
        return e && e[0] == '1';
    }();
    return on;
}
using prof_clock = std::chrono::steady_clock;
inline double ms_since(const prof_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(prof_clock::now() - t0)
        .count();
}
}  // namespace

// Rectangular linear_sum_assignment (maximize) — a faithful port of scipy
// scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (Crouse 2016 shortest
// augmenting path). The exact tie-breaking matters: the diarization soft matrix
// has many ties (inactive/duplicate speaker rows collapse to the global min),
// so only scipy's deterministic rule reproduces hard_clusters bit-for-bit.
// Returns assign[s] = chosen column k (or -1 if speaker s is unassigned).
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Python round(): half-to-even.
inline long py_round(double x) {
    return static_cast<long>(std::nearbyint(x));  // to-nearest-even by default
}

intptr_t augmenting_path(intptr_t nc, const std::vector<double>& cost,
                         std::vector<double>& u, std::vector<double>& v,
                         std::vector<intptr_t>& path,
                         std::vector<intptr_t>& row4col,
                         std::vector<double>& spc, intptr_t i,
                         std::vector<char>& SR, std::vector<char>& SC,
                         std::vector<intptr_t>& remaining, double* p_minVal) {
    double minVal = 0;
    intptr_t num_remaining = nc;
    for (intptr_t it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
    std::fill(SR.begin(), SR.end(), 0);
    std::fill(SC.begin(), SC.end(), 0);
    std::fill(spc.begin(), spc.end(), kInf);

    intptr_t sink = -1;
    while (sink == -1) {
        intptr_t index = -1;
        double lowest = kInf;
        SR[i] = 1;
        for (intptr_t it = 0; it < num_remaining; ++it) {
            const intptr_t j = remaining[it];
            const double r = minVal + cost[i * nc + j] - u[i] - v[j];
            if (r < spc[j]) { path[j] = i; spc[j] = r; }
            if (spc[j] < lowest ||
                (spc[j] == lowest && row4col[j] == -1)) {
                lowest = spc[j];
                index = it;
            }
        }
        minVal = lowest;
        if (minVal == kInf) return -1;
        const intptr_t j = remaining[index];
        if (row4col[j] == -1) sink = j; else i = row4col[j];
        SC[j] = 1;
        remaining[index] = remaining[--num_remaining];
    }
    *p_minVal = minVal;
    return sink;
}

// soft: [S0*K0] row-major (maximize). assign[s] = k or -1.
void lsap_max(const std::vector<double>& soft, int S0, int K0,
              std::vector<int>& assign) {
    assign.assign(S0, -1);
    if (S0 == 0 || K0 == 0) return;
    const bool transpose = K0 < S0;  // ensure nr <= nc
    intptr_t nr = S0, nc = K0;
    std::vector<double> cost(static_cast<std::size_t>(S0) * K0);
    if (transpose) {
        for (int i = 0; i < S0; ++i)
            for (int j = 0; j < K0; ++j)
                cost[static_cast<std::size_t>(j) * S0 + i] =
                    -soft[static_cast<std::size_t>(i) * K0 + j];
        nr = K0;
        nc = S0;
    } else {
        for (std::size_t idx = 0; idx < cost.size(); ++idx)
            cost[idx] = -soft[idx];
    }

    std::vector<double> u(nr, 0.0), v(nc, 0.0), spc(nc);
    std::vector<intptr_t> path(nc, -1), col4row(nr, -1), row4col(nc, -1),
        remaining(nc);
    std::vector<char> SR(nr), SC(nc);

    for (intptr_t curRow = 0; curRow < nr; ++curRow) {
        double minVal;
        const intptr_t sink =
            augmenting_path(nc, cost, u, v, path, row4col, spc, curRow, SR, SC,
                            remaining, &minVal);
        if (sink < 0) { assign.assign(S0, -1); return; }  // infeasible
        u[curRow] += minVal;
        for (intptr_t i = 0; i < nr; ++i)
            if (SR[i] && i != curRow)
                u[i] += minVal - spc[col4row[i]];
        for (intptr_t j = 0; j < nc; ++j)
            if (SC[j]) v[j] -= minVal - spc[j];
        intptr_t j = sink;
        while (true) {
            const intptr_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) break;
        }
    }

    if (transpose) {
        // col4row indexed by centroid (row in transposed) -> speaker (col).
        for (intptr_t k = 0; k < nr; ++k)
            if (col4row[k] >= 0) assign[col4row[k]] = static_cast<int>(k);
    } else {
        for (intptr_t s = 0; s < nr; ++s)
            if (col4row[s] >= 0) assign[s] = static_cast<int>(col4row[s]);
    }
}

}  // namespace

bool DiarizenClustering::cluster(const float* embeddings, int C, int S, int dim,
                                 const float* seg, int F,
                                 std::vector<std::int8_t>& hard_out) {
    if (!priors_.loaded) return false;
    hard_out.assign(static_cast<std::size_t>(C) * S, -2);

    const bool prof = cluster_prof_enabled();
    const auto t_start = prof_clock::now();
    double ms_filter = 0, ms_ahc = 0, ms_fea = 0, ms_vbx = 0, ms_cent = 0,
           ms_soft = 0, ms_assign = 0;

    // --- filter_embeddings -------------------------------------------------
    auto collect = [&](long min_frames, std::vector<int>& cidx,
                       std::vector<int>& sidx) {
        cidx.clear();
        sidx.clear();
        for (int c = 0; c < C; ++c) {
            for (int s = 0; s < S; ++s) {
                // active: sum_f seg>0
                double act = 0.0;
                for (int f = 0; f < F; ++f)
                    act += seg[((std::size_t)c * F + f) * S + s];
                if (act <= 0.0) continue;
                // valid: no NaN in embedding row
                const float* e =
                    embeddings + ((std::size_t)c * S + s) * dim;
                bool valid = true;
                for (int d = 0; d < dim; ++d)
                    if (std::isnan(e[d])) { valid = false; break; }
                if (!valid) continue;
                // clean-frame count: frames where exactly one speaker active
                long clean = 0;
                for (int f = 0; f < F; ++f) {
                    double tot = 0.0;
                    for (int ss = 0; ss < S; ++ss)
                        tot += seg[((std::size_t)c * F + f) * S + ss];
                    if (tot == 1.0 &&
                        seg[((std::size_t)c * F + f) * S + s] > 0.0)
                        ++clean;
                }
                if (clean >= min_frames) {
                    cidx.push_back(c);
                    sidx.push_back(s);
                }
            }
        }
    };

    const long min_frames = py_round(cfg_.min_frames_ratio * F);
    std::vector<int> cidx, sidx;
    collect(min_frames, cidx, sidx);
    if (static_cast<int>(cidx.size()) < 2) collect(0, cidx, sidx);

    const int N = static_cast<int>(cidx.size());
    if (prof) { ms_filter = ms_since(t_start); }
    if (N < 2) {
        // trivial path: all zeros (BaseClustering / VBx <2 branch).
        std::fill(hard_out.begin(), hard_out.end(), 0);
        return true;
    }

    // train_embeddings [N, dim]
    std::vector<float> train(static_cast<std::size_t>(N) * dim);
    for (int i = 0; i < N; ++i) {
        const float* e =
            embeddings + ((std::size_t)cidx[i] * S + sidx[i]) * dim;
        std::copy(e, e + dim, train.data() + (std::size_t)i * dim);
    }

    // --- AHC ---------------------------------------------------------------
    const auto t_ahc0 = prof_clock::now();
    std::vector<double> normed(static_cast<std::size_t>(N) * dim);
    for (int r = 0; r < N; ++r) {
        const float* x = train.data() + (std::size_t)r * dim;
        double nrm = 0.0;
        for (int d = 0; d < dim; ++d) nrm += static_cast<double>(x[d]) * x[d];
        nrm = std::sqrt(nrm);
        double* o = normed.data() + (std::size_t)r * dim;
        for (int d = 0; d < dim; ++d) o[d] = static_cast<double>(x[d]) / nrm;
    }
    std::vector<int> ahc;
    agglomerative_(normed, N, dim, ahc);
    int K0 = 0;
    for (int v : ahc) K0 = std::max(K0, v + 1);
    if (prof) ms_ahc = ms_since(t_ahc0);

    // --- VBx EM ------------------------------------------------------------
    const auto t_fea0 = prof_clock::now();
    std::vector<double> fea;
    compute_fea_(train.data(), N, dim, fea);  // [N, pdim]
    if (prof) ms_fea = ms_since(t_fea0);
    const auto t_vbx0 = prof_clock::now();
    std::vector<double> q, pi;
    vbx_em_(fea, N, priors_.pdim, ahc, K0, q, pi);
    if (prof) ms_vbx = ms_since(t_vbx0);

    // --- centroids = (q[:, sp>1e-7].T @ train) -----------------------------
    const auto t_cent0 = prof_clock::now();
    std::vector<int> keep;
    for (int k = 0; k < K0; ++k)
        if (pi[k] > 1e-7) keep.push_back(k);
    const int Kc = static_cast<int>(keep.size());
    std::vector<double> centroids(static_cast<std::size_t>(Kc) * dim, 0.0);
    for (int ci = 0; ci < Kc; ++ci) {
        const int k = keep[ci];
        double* cen = centroids.data() + (std::size_t)ci * dim;
        for (int i = 0; i < N; ++i) {
            const double w = q[(std::size_t)i * K0 + k];
            const float* x = train.data() + (std::size_t)i * dim;
            for (int d = 0; d < dim; ++d) cen[d] += w * x[d];
        }
    }
    // precompute centroid norms
    std::vector<double> cnorm(Kc, 0.0);
    for (int ci = 0; ci < Kc; ++ci) {
        const double* cen = centroids.data() + (std::size_t)ci * dim;
        double s2 = 0.0;
        for (int d = 0; d < dim; ++d) s2 += cen[d] * cen[d];
        cnorm[ci] = std::sqrt(s2);
    }

    if (prof) ms_cent = ms_since(t_cent0);

    // --- soft = 2 - cdist(embeddings, centroids, cosine) -------------------
    // NaN rows (inactive speakers) yield NaN distances -> NaN soft.
    const auto t_soft0 = prof_clock::now();
    const std::size_t CS = static_cast<std::size_t>(C) * S;
    std::vector<double> soft(CS * Kc);
    std::vector<char> isnan_row(CS, 0);
    double soft_min = std::numeric_limits<double>::infinity();
    for (std::size_t cs = 0; cs < CS; ++cs) {
        const float* e = embeddings + cs * dim;
        bool nanrow = false;
        for (int d = 0; d < dim; ++d)
            if (std::isnan(e[d])) { nanrow = true; break; }
        double enorm = 0.0;
        if (!nanrow)
            for (int d = 0; d < dim; ++d) enorm += (double)e[d] * e[d];
        enorm = std::sqrt(enorm);
        isnan_row[cs] = nanrow ? 1 : 0;
        for (int ci = 0; ci < Kc; ++ci) {
            double val;
            if (nanrow) {
                val = std::numeric_limits<double>::quiet_NaN();
            } else {
                const double* cen = centroids.data() + (std::size_t)ci * dim;
                double dot = 0.0;
                for (int d = 0; d < dim; ++d) dot += (double)e[d] * cen[d];
                const double denom = enorm * cnorm[ci];
                const double cdistv = 1.0 - dot / denom;  // cosine distance
                val = 2.0 - cdistv;
                if (val < soft_min) soft_min = val;
            }
            soft[cs * Kc + ci] = val;
        }
    }
    // nan_to_num(soft, nan=nanmin(soft))
    for (std::size_t i = 0; i < soft.size(); ++i)
        if (std::isnan(soft[i])) soft[i] = soft_min;

    if (prof) ms_soft = ms_since(t_soft0);

    // --- constrained_argmax per chunk (Hungarian, maximize) ----------------
    const auto t_assign0 = prof_clock::now();
    std::vector<std::int32_t> hard_raw(CS, -2);
    std::vector<double> cost(static_cast<std::size_t>(S) * Kc);
    std::vector<int> assign;
    for (int c = 0; c < C; ++c) {
        for (int s = 0; s < S; ++s)
            for (int ci = 0; ci < Kc; ++ci)
                cost[(std::size_t)s * Kc + ci] =
                    soft[((std::size_t)c * S + s) * Kc + ci];
        lsap_max(cost, S, Kc, assign);
        for (int s = 0; s < S; ++s)
            if (assign[s] >= 0)
                hard_raw[(std::size_t)c * S + s] = assign[s];
    }

    // --- np.unique(return_inverse) renumber (includes -2) ------------------
    std::vector<std::int32_t> uniq(hard_raw.begin(), hard_raw.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    for (std::size_t i = 0; i < CS; ++i) {
        const int idx = static_cast<int>(
            std::lower_bound(uniq.begin(), uniq.end(), hard_raw[i]) -
            uniq.begin());
        hard_out[i] = static_cast<std::int8_t>(idx);
    }
    if (prof) {
        ms_assign = ms_since(t_assign0);
        std::fprintf(stderr,
            "[cluster-prof] N=%d K0=%d Kc=%d C=%d S=%d | filter=%.1f ahc=%.1f "
            "fea=%.1f vbx=%.1f cent=%.1f soft=%.1f assign=%.1f total=%.1f ms\n",
            N, K0, Kc, C, S, ms_filter, ms_ahc, ms_fea, ms_vbx, ms_cent,
            ms_soft, ms_assign, ms_since(t_start));
    }
    return true;
}

}  // namespace orator
}  // namespace deusridet
