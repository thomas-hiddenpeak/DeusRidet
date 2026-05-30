// @philosophical_role Orator — speaker re-attribution. The VBx clustering stage
//   of the native DiariZen-v2 pipeline. This is small-N serial glue: an
//   external-library algorithm (PLDA + agglomerative + Bayesian HMM EM +
//   Hungarian assignment) with no GPU entry point, so per the project's
//   GPU-first rule it runs on the CPU by exception (case c).
// @serves Reproduces pyannote VBxClustering.__call__ bit-for-bit so the
//   diarizen_worker.py subprocess can be retired and clustering can run
//   in-process inside awaken.
#ifndef DEUSRIDET_ORATOR_DIARIZEN_CLUSTERING_H
#define DEUSRIDET_ORATOR_DIARIZEN_CLUSTERING_H

#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {
namespace orator {

// Hyper-parameters mirror the DiariZen-v2 config (clustering.args):
//   method=VBxClustering, ahc_criterion=distance, ahc_threshold=0.6,
//   Fa=0.07, Fb=0.8, lda_dim=128, max_iters=20, metric=cosine,
//   constrained_assignment=true, init_smoothing=7.0, loopProb=0.0.
struct DiarizenClusteringConfig {
    double ahc_threshold = 0.6;
    double Fa = 0.07;
    double Fb = 0.8;
    int lda_dim = 128;
    int max_iters = 20;
    double init_smoothing = 7.0;
    double vbx_epsilon = 1e-4;
    double min_frames_ratio = 0.1;
};

// PLDA priors loaded from xvec_transform.npz + plda.npz and pre-transformed
// exactly as diarizen.clustering.VBx.vbx_setup does (centering, whitening, LDA,
// generalized eigendecomposition reordered to descending eigenvalues).
struct DiarizenPldaPriors {
    int xdim = 256;   // input embedding dim
    int pdim = 128;   // PLDA latent dim
    // xvec_transform
    std::vector<double> mean1;  // [xdim]
    std::vector<float> mean2;   // [pdim]
    std::vector<float> lda;     // [xdim, pdim] row-major
    // plda (post vbx_setup reorder)
    std::vector<double> plda_mu;   // [pdim]
    std::vector<double> plda_tr;   // [pdim, pdim] row-major (wccn.T[::-1])
    std::vector<double> plda_psi;  // [pdim] descending eigenvalues
    bool loaded = false;
};

class DiarizenClustering {
   public:
    DiarizenClustering() = default;

    // Load + transform the PLDA priors. plda_dir contains xvec_transform.npz and
    // plda.npz. Returns false on any parse/shape error.
    bool load_priors(const std::string& plda_dir);
    bool priors_loaded() const { return priors_.loaded; }

    // --- whole-stage entry (P2b-4) -----------------------------------------
    // embeddings: [num_chunks * num_local_speakers * dim] row-major, NaN rows
    //   for inactive speakers. seg: [num_chunks * num_frames * num_local] binary
    //   median-filtered segmentation. Writes hard_clusters [num_chunks *
    //   num_local] (int8, -2 for never-assigned). Returns false on error.
    bool cluster(const float* embeddings, int num_chunks, int num_local,
                 int dim, const float* seg, int num_frames,
                 std::vector<std::int8_t>& hard_out);

    // --- debug taps (bit-equality harness) ---------------------------------
    // PLDA-space features for a [N, xdim] block of train embeddings.
    bool debug_fea(const float* train_emb, int n, int xdim,
                   std::vector<float>& fea_out);
    // AHC labels (0-based, renumbered) for a [N, xdim] block.
    bool debug_ahc(const float* train_emb, int n, int xdim,
                   std::vector<int>& ahc_out);
    // VBx EM responsibilities gamma [N, K0] (row-major) + priors pi [K0] for a
    // [N, xdim] block. K0 = AHC cluster count.
    bool debug_vbx(const float* train_emb, int n, int xdim,
                   std::vector<double>& gamma_out, int& K0,
                   std::vector<double>& pi_out);

    const DiarizenClusteringConfig& config() const { return cfg_; }
    DiarizenClusteringConfig& mutable_config() { return cfg_; }
    const DiarizenPldaPriors& priors() const { return priors_; }

   private:
    DiarizenClusteringConfig cfg_;
    DiarizenPldaPriors priors_;

    // x_tf + plda_tf: [N, xdim] -> [N, pdim]. Host doubles internally.
    void compute_fea_(const float* train_emb, int n, int xdim,
                      std::vector<double>& fea) const;
    // centroid-linkage + fcluster(distance, threshold) -> 0-based labels.
    void agglomerative_(const std::vector<double>& normed, int n, int dim,
                        std::vector<int>& labels) const;
    // VBx Bayesian-HMM EM, GMM branch (loopProb=0). fea [N, pdim], ahc labels
    // -> gamma [N, K0] + pi [K0]. Phi = plda_psi[:pdim].
    void vbx_em_(const std::vector<double>& fea, int N, int pdim,
                 const std::vector<int>& ahc, int K0,
                 std::vector<double>& gamma, std::vector<double>& pi) const;
};

}  // namespace orator
}  // namespace deusridet

#endif  // DEUSRIDET_ORATOR_DIARIZEN_CLUSTERING_H
