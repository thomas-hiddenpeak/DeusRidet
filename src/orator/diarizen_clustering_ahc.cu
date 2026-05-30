/**
 * @file diarizen_clustering_ahc.cu
 * @philosophical_role GPU centroid-linkage agglomeration for the DiariZen-v2
 *   clustering stage. The merge is inherently serial (n-1 dependent steps),
 *   but the two heavy per-step operations — the global-minimum scan over the
 *   active pair set (O(active^2)) and the Lance-Williams distance update
 *   (O(active)) — are data-parallel, and the one-shot pairwise distance is a
 *   clean per-pair kernel. All three run on the GPU; only the tree-numbering
 *   DFS (O(n), sub-millisecond) stays on the host. Reproduces the CPU result
 *   bit-for-bit: the pdist keeps the same sequential fp64 summation order, the
 *   argmin uses a lexicographic (dist, i, j) comparator that matches the CPU's
 *   strict-less row-major scan tie-break, and the Lance-Williams kernel mirrors
 *   centroid_lw's exact fp64 arithmetic.
 * @serves DiarizenClustering::agglomerative_.
 */
#include "diarizen_clustering_ahc.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../communis/log.h"

namespace deusridet {
namespace orator {

namespace {
constexpr const char* kFLog = "DiariZenAHC";

// A candidate active pair: distance + the two node ids (i < j). Compared
// lexicographically by (dist, i, j) so the reduction reproduces the CPU's
// "first strictly-smaller in row-major (i,j) scan" tie-break exactly.
struct Cand {
    double dist;
    int i;
    int j;
};

__device__ __forceinline__ bool cand_less(const Cand& a, const Cand& b) {
    if (a.dist < b.dist) return true;
    if (a.dist > b.dist) return false;
    if (a.i < b.i) return true;
    if (a.i > b.i) return false;
    return a.j < b.j;
}

__device__ __forceinline__ Cand cand_max() {
    Cand c;
    c.dist = DBL_MAX;
    c.i = 0x7fffffff;
    c.j = 0x7fffffff;
    return c;
}

// pdist euclidean on the L2-normalized leaf rows. One thread per (i,j), i<j.
// Sequential fp64 accumulation over d preserves the CPU summation order, so
// every D[i,j] is bit-identical to the host loop.
__global__ void pdist_kernel(const double* __restrict__ normed, int n, int dim,
                             int total, double* __restrict__ D) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n || j <= i) return;
    const double* a = normed + static_cast<std::size_t>(i) * dim;
    const double* b = normed + static_cast<std::size_t>(j) * dim;
    double s = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double diff = a[d] - b[d];
        s += diff * diff;
    }
    const double dd = sqrt(s);
    D[static_cast<std::size_t>(i) * total + j] = dd;
    D[static_cast<std::size_t>(j) * total + i] = dd;
}

// Global-minimum over all active pairs (p<q) in the compact active list.
// Each block reduces its 16x16 tile of candidate pairs to one, writing to
// blockBest[blockIdx.y*gridDim.x + blockIdx.x].
__global__ void argmin_kernel(const int* __restrict__ act, int na, int total,
                              const double* __restrict__ D,
                              Cand* __restrict__ blockBest) {
    __shared__ Cand sm[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    Cand c = cand_max();
    if (p < na && q < na && p < q) {
        const int idA = act[p];
        const int idB = act[q];
        const int i = idA < idB ? idA : idB;
        const int j = idA < idB ? idB : idA;
        c.dist = D[static_cast<std::size_t>(i) * total + j];
        c.i = i;
        c.j = j;
    }
    sm[tid] = c;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && cand_less(sm[tid + s], sm[tid])) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) blockBest[blockIdx.y * gridDim.x + blockIdx.x] = sm[0];
}

// Final reduction of the per-block winners to a single best (single block,
// grid-stride). m = number of blocks emitted by argmin_kernel.
__global__ void reduce_kernel(const Cand* __restrict__ in, int m,
                              Cand* __restrict__ out) {
    __shared__ Cand sm[256];
    const int tid = threadIdx.x;
    Cand c = cand_max();
    for (int k = tid; k < m; k += blockDim.x)
        if (cand_less(in[k], c)) c = in[k];
    sm[tid] = c;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && cand_less(sm[tid + s], sm[tid])) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sm[0];
}

// Lance-Williams centroid update of the new node's distance to every active z.
// Mirrors centroid_lw exactly (same fp64 op order) so D[nid,z] is bit-identical.
__global__ void lw_kernel(const int* __restrict__ actz, int naz, int total,
                          int nid, int ba, int bb, double best, int size_ba,
                          int size_bb, double* __restrict__ D) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= naz) return;
    const int z = actz[t];
    const double dxi = D[static_cast<std::size_t>(ba) * total + z];
    const double dyi = D[static_cast<std::size_t>(bb) * total + z];
    const double s = static_cast<double>(size_ba) + size_bb;
    const double tt =
        (size_ba * dxi * dxi + size_bb * dyi * dyi) / s -
        (static_cast<double>(size_ba) * size_bb) * best * best / (s * s);
    const double dnew = sqrt(tt < 0.0 ? 0.0 : tt);
    D[static_cast<std::size_t>(nid) * total + z] = dnew;
    D[static_cast<std::size_t>(z) * total + nid] = dnew;
}

inline bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    LOG_WARN(kFLog, "CUDA error at %s: %s — falling back to CPU merge", what,
             cudaGetErrorString(e));
    return false;
}
}  // namespace

bool agglomerative_merge_gpu(const std::vector<double>& normed, int n, int dim,
                             std::vector<int>& child0, std::vector<int>& child1,
                             std::vector<double>& height,
                             std::vector<int>& size) {
    if (n < 2) return false;
    const int total = 2 * n - 1;
    const std::size_t Dn = static_cast<std::size_t>(total) * total;

    double* dN = nullptr;
    double* dD = nullptr;
    int* dAct = nullptr;
    int* dActz = nullptr;
    Cand* dBlock = nullptr;
    Cand* dBest = nullptr;
    bool ok = false;

    do {
        if (!ck(cudaMalloc(&dN, static_cast<std::size_t>(n) * dim *
                                    sizeof(double)),
                "malloc normed"))
            break;
        if (!ck(cudaMalloc(&dD, Dn * sizeof(double)), "malloc D")) break;
        if (!ck(cudaMalloc(&dAct, static_cast<std::size_t>(n) * sizeof(int)),
                "malloc act"))
            break;
        if (!ck(cudaMalloc(&dActz, static_cast<std::size_t>(n) * sizeof(int)),
                "malloc actz"))
            break;
        const int gmax = (n + 15) / 16;
        if (!ck(cudaMalloc(&dBlock, static_cast<std::size_t>(gmax) * gmax *
                                        sizeof(Cand)),
                "malloc block"))
            break;
        if (!ck(cudaMalloc(&dBest, sizeof(Cand)), "malloc best")) break;
        if (!ck(cudaMemcpy(dN, normed.data(),
                           static_cast<std::size_t>(n) * dim * sizeof(double),
                           cudaMemcpyHostToDevice),
                "H2D normed"))
            break;

        {
            const dim3 blk(16, 16);
            const dim3 grd((n + 15) / 16, (n + 15) / 16);
            pdist_kernel<<<grd, blk>>>(dN, n, dim, total, dD);
            if (!ck(cudaGetLastError(), "pdist")) break;
        }

        for (int i = 0; i < n; ++i) size[i] = 1;

        std::vector<int> act(n);
        for (int i = 0; i < n; ++i) act[i] = i;
        int na = n;

        bool err = false;
        for (int k = 0; k < n - 1; ++k) {
            const int nid = n + k;
            if (!ck(cudaMemcpy(dAct, act.data(),
                               static_cast<std::size_t>(na) * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "H2D act")) {
                err = true;
                break;
            }
            const int g = (na + 15) / 16;
            const dim3 blk(16, 16);
            const dim3 grd(g, g);
            argmin_kernel<<<grd, blk>>>(dAct, na, total, dD, dBlock);
            if (!ck(cudaGetLastError(), "argmin")) {
                err = true;
                break;
            }
            reduce_kernel<<<1, 256>>>(dBlock, g * g, dBest);
            if (!ck(cudaGetLastError(), "reduce")) {
                err = true;
                break;
            }
            Cand best;
            if (!ck(cudaMemcpy(&best, dBest, sizeof(Cand),
                               cudaMemcpyDeviceToHost),
                    "D2H best")) {
                err = true;
                break;
            }
            const int ba = best.i;
            const int bb = best.j;
            child0[nid] = ba;
            child1[nid] = bb;
            height[nid] = best.dist;
            size[nid] = size[ba] + size[bb];

            // active list minus ba,bb (the z's for the Lance-Williams update).
            std::vector<int> actz;
            actz.reserve(na - 2);
            for (int t = 0; t < na; ++t) {
                const int v = act[t];
                if (v != ba && v != bb) actz.push_back(v);
            }
            const int naz = static_cast<int>(actz.size());
            if (naz > 0) {
                if (!ck(cudaMemcpy(dActz, actz.data(),
                                   static_cast<std::size_t>(naz) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "H2D actz")) {
                    err = true;
                    break;
                }
                const int lb = 256;
                const int lg = (naz + lb - 1) / lb;
                lw_kernel<<<lg, lb>>>(dActz, naz, total, nid, ba, bb, best.dist,
                                      size[ba], size[bb], dD);
                if (!ck(cudaGetLastError(), "lw")) {
                    err = true;
                    break;
                }
            }
            actz.push_back(nid);  // new node becomes active
            act.swap(actz);
            na = na - 1;
        }
        if (err) break;
        if (!ck(cudaDeviceSynchronize(), "sync")) break;
        ok = true;
    } while (false);

    if (dN) cudaFree(dN);
    if (dD) cudaFree(dD);
    if (dAct) cudaFree(dAct);
    if (dActz) cudaFree(dActz);
    if (dBlock) cudaFree(dBlock);
    if (dBest) cudaFree(dBest);
    return ok;
}

}  // namespace orator
}  // namespace deusridet
