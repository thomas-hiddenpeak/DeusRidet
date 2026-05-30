// @role: DiariZenClustering AHC (P2b-2) — generic centroid-linkage
//        agglomeration + fcluster(distance) cut. VBx EM + assignment
//        (P2b-3/4) follow below once verified. Small-N serial CPU glue.
#include "diarizen_clustering.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace deusridet {
namespace orator {

namespace {
// Centroid Lance-Williams update on current (euclidean) distances, matching
// scipy _hierarchy _centroid:
//   d(x∪y, i) = sqrt( (sx*dxi^2 + sy*dyi^2)/(sx+sy)
//                     - sx*sy*dxy^2/(sx+sy)^2 )
inline double centroid_lw(double dxi, double dyi, double dxy, int sx, int sy) {
    const double s = static_cast<double>(sx) + sy;
    const double t = (sx * dxi * dxi + sy * dyi * dyi) / s -
                     (static_cast<double>(sx) * sy) * dxy * dxy / (s * s);
    return std::sqrt(t < 0.0 ? 0.0 : t);
}
}  // namespace

// normed: [n, dim] L2-normalized rows. Produces 0-based flat labels (renumbered
// by sorted unique, matching np.unique(return_inverse)).
void DiarizenClustering::agglomerative_(const std::vector<double>& normed, int n,
                                        int dim, std::vector<int>& labels) const {
    labels.assign(n, 0);
    if (n <= 1) return;

    const int total = 2 * n - 1;  // node ids: leaves 0..n-1, merges n..2n-2
    const double INF = std::numeric_limits<double>::infinity();

    // Dense distance among active clusters (ids 0..total-1).
    std::vector<double> D(static_cast<std::size_t>(total) * total, INF);
    auto at = [&](int i, int j) -> double& {
        return D[static_cast<std::size_t>(i) * total + j];
    };
    // pdist euclidean on the normalized rows.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double s = 0.0;
            const double* a = normed.data() + static_cast<std::size_t>(i) * dim;
            const double* b = normed.data() + static_cast<std::size_t>(j) * dim;
            for (int d = 0; d < dim; ++d) {
                const double diff = a[d] - b[d];
                s += diff * diff;
            }
            const double dd = std::sqrt(s);
            at(i, j) = dd;
            at(j, i) = dd;
        }
    }

    std::vector<int> size(total, 0);
    std::vector<char> active(total, 0);
    for (int i = 0; i < n; ++i) { size[i] = 1; active[i] = 1; }

    std::vector<int> child0(total, -1), child1(total, -1);
    std::vector<double> height(total, 0.0);

    // Generic agglomeration: each step merge the global-minimum active pair.
    for (int k = 0; k < n - 1; ++k) {
        double best = INF;
        int ba = -1, bb = -1;
        for (int i = 0; i < total; ++i) {
            if (!active[i]) continue;
            for (int j = i + 1; j < total; ++j) {
                if (!active[j]) continue;
                const double dd = at(i, j);
                if (dd < best) { best = dd; ba = i; bb = j; }
            }
        }
        const int nid = n + k;  // new cluster id
        child0[nid] = ba;
        child1[nid] = bb;
        height[nid] = best;
        size[nid] = size[ba] + size[bb];
        active[ba] = 0;
        active[bb] = 0;
        // Update distances from the new cluster to every other active cluster.
        for (int z = 0; z < total; ++z) {
            if (!active[z] || z == nid) continue;
            const double dnew = centroid_lw(at(ba, z), at(bb, z), best,
                                            size[ba], size[bb]);
            at(nid, z) = dnew;
            at(z, nid) = dnew;
        }
        active[nid] = 1;
    }

    // MD[node] = max merge height in the subtree rooted at node (leaves -> 0).
    std::vector<double> MD(total, 0.0);
    for (int nid = n; nid < total; ++nid) {
        double m = height[nid];
        const int c0 = child0[nid], c1 = child1[nid];
        if (c0 >= n) m = std::max(m, MD[c0]);
        if (c1 >= n) m = std::max(m, MD[c1]);
        MD[nid] = m;
    }

    // fcluster(criterion='distance') == scipy cluster_monocrit(Z, MD, t): a
    // stack DFS from the root, descending the (smaller-id) left child first,
    // numbering clusters 1..K in leader-encounter order. The diarizen wrapper
    // then does `- 1` and np.unique(return_inverse); since the numbering is
    // already 1..K in order, that yields T-1 (0-based). We must reproduce the
    // exact numbering because it propagates to the final cluster ids.
    const double t = cfg_.ahc_threshold;
    std::vector<int> T(n, 0);
    std::vector<int> stack(total, 0);
    std::vector<char> visited(total, 0);
    int n_cluster = 0;
    int cluster_leader = -1;
    int sp = 0;
    stack[0] = total - 1;  // root = 2n-2
    while (sp >= 0) {
        const int root = stack[sp];
        const int i_lc = child0[root];
        const int i_rc = child1[root];
        if (cluster_leader == -1 && MD[root] <= t) {
            cluster_leader = root;
            ++n_cluster;
        }
        if (i_lc >= n && !visited[i_lc]) {
            visited[i_lc] = 1;
            stack[++sp] = i_lc;
            continue;
        }
        if (i_rc >= n && !visited[i_rc]) {
            visited[i_rc] = 1;
            stack[++sp] = i_rc;
            continue;
        }
        if (i_lc < n) {
            if (cluster_leader == -1) ++n_cluster;
            T[i_lc] = n_cluster;
        }
        if (i_rc < n) {
            if (cluster_leader == -1) ++n_cluster;
            T[i_rc] = n_cluster;
        }
        if (cluster_leader == root) cluster_leader = -1;
        --sp;
    }
    // ahc = fcluster - 1 (np.unique(return_inverse) is identity on 1..K).
    for (int i = 0; i < n; ++i) labels[i] = T[i] - 1;
}

bool DiarizenClustering::debug_ahc(const float* train_emb, int n, int xdim,
                                   std::vector<int>& ahc_out) {
    if (!priors_.loaded) return false;
    // L2-normalize rows (train_embeddings_normed).
    std::vector<double> normed(static_cast<std::size_t>(n) * xdim);
    for (int r = 0; r < n; ++r) {
        const float* x = train_emb + static_cast<std::size_t>(r) * xdim;
        double nrm = 0.0;
        for (int i = 0; i < xdim; ++i) nrm += static_cast<double>(x[i]) * x[i];
        nrm = std::sqrt(nrm);
        double* o = normed.data() + static_cast<std::size_t>(r) * xdim;
        for (int i = 0; i < xdim; ++i) o[i] = static_cast<double>(x[i]) / nrm;
    }
    agglomerative_(normed, n, xdim, ahc_out);
    return true;
}

bool DiarizenClustering::cluster(const float*, int, int, int, const float*, int,
                                 std::vector<std::int8_t>&) {
    return false;  // TODO(P2b-4)
}

}  // namespace orator
}  // namespace deusridet
