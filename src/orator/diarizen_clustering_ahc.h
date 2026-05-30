// @philosophical_role Orator — GPU centroid-linkage agglomeration. The AHC
//   merge is the one clustering sub-stage that is NOT tiny: profiling the
//   full 60-min session showed agglomerative_ at ~41 s (86 % of the cluster
//   stage, N≈3894 nodes, O(N^3) generic merge) running entirely on one CPU
//   core while the GPU sat idle. Per the project's GPU-first rule this work
//   belongs on the GPU; only the O(N) fcluster DFS numbering stays on the
//   host (genuinely serial, sub-millisecond).
// @serves DiarizenClustering::agglomerative_ — replaces the CPU pdist + merge
//   loop with three CUDA kernels (pdist, per-step lexicographic argmin,
//   Lance-Williams update) that reproduce the CPU result bit-for-bit.
#ifndef DEUSRIDET_ORATOR_DIARIZEN_CLUSTERING_AHC_H
#define DEUSRIDET_ORATOR_DIARIZEN_CLUSTERING_AHC_H

#include <vector>

namespace deusridet {
namespace orator {

// GPU centroid-linkage merge. normed: [n*dim] fp64, L2-normalized rows. On
// success fills, for merge nodes nid in n..2n-2: child0[nid], child1[nid]
// (child0 < child1, the lexicographically-first global-minimum active pair),
// height[nid] (the merge distance); and size[node] for all node ids 0..2n-2
// (leaves = 1). All four vectors must be pre-sized to total = 2n-1. Returns
// false on any CUDA error (caller falls back to the CPU merge); on false the
// output vectors are left untouched/partial and must not be used.
//
// Bit-identical to the CPU pdist (sequential fp64 sum) + generic centroid
// merge (Lance-Williams update, strict-less global-min with row-major
// tie-break == lexicographic (dist, i, j)).
bool agglomerative_merge_gpu(const std::vector<double>& normed, int n, int dim,
                             std::vector<int>& child0, std::vector<int>& child1,
                             std::vector<double>& height,
                             std::vector<int>& size);

}  // namespace orator
}  // namespace deusridet

#endif  // DEUSRIDET_ORATOR_DIARIZEN_CLUSTERING_AHC_H
