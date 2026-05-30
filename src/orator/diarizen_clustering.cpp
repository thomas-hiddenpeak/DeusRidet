// @role: DiariZenClustering::load_priors / compute_fea — PLDA prior setup and
//        the x_tf+plda_tf feature transform (P2b-1). Whole-stage clustering
//        (AHC, VBx EM, assignment) lives in diarizen_clustering_vbx.cpp.
#include "diarizen_clustering.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../communis/log.h"

namespace deusridet {
namespace orator {

namespace {
constexpr const char* kLog = "DiariZenClustering";

// ---- minimal .npy / .npz (STORED) reader ---------------------------------
struct NpyArray {
    std::vector<std::int64_t> shape;
    std::string dtype;  // e.g. "<f8", "<f4"
    std::vector<std::uint8_t> data;
};

// Parse one .npy blob (numpy format v1/v2) from memory.
bool parse_npy(const std::uint8_t* p, std::size_t n, NpyArray& out) {
    if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) return false;
    const std::uint8_t major = p[6];
    std::size_t hlen;
    std::size_t hoff;
    if (major == 1) {
        hlen = p[8] | (p[9] << 8);
        hoff = 10;
    } else {
        hlen = static_cast<std::size_t>(p[8]) | (p[9] << 8) |
               (static_cast<std::size_t>(p[10]) << 16) |
               (static_cast<std::size_t>(p[11]) << 24);
        hoff = 12;
    }
    if (hoff + hlen > n) return false;
    std::string header(reinterpret_cast<const char*>(p + hoff), hlen);
    // descr
    auto dpos = header.find("'descr'");
    auto q1 = header.find('\'', dpos + 7);
    auto q2 = header.find('\'', q1 + 1);
    out.dtype = header.substr(q1 + 1, q2 - q1 - 1);
    // shape
    auto spos = header.find("'shape'");
    auto lp = header.find('(', spos);
    auto rp = header.find(')', lp);
    std::string shp = header.substr(lp + 1, rp - lp - 1);
    out.shape.clear();
    std::size_t i = 0;
    while (i < shp.size()) {
        while (i < shp.size() && (shp[i] == ' ' || shp[i] == ',')) ++i;
        if (i >= shp.size()) break;
        std::size_t j = i;
        while (j < shp.size() && shp[j] >= '0' && shp[j] <= '9') ++j;
        if (j > i) out.shape.push_back(std::stoll(shp.substr(i, j - i)));
        i = j;
    }
    std::size_t off = hoff + hlen;
    out.data.assign(p + off, p + n);
    return true;
}

// Read a STORED (uncompressed) .npz; fills name->NpyArray.
bool read_npz(const std::string& path, std::map<std::string, NpyArray>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
    std::size_t pos = 0;
    while (pos + 30 <= buf.size()) {
        if (std::memcmp(&buf[pos], "PK\x03\x04", 4) != 0) break;
        const std::uint16_t method = buf[pos + 8] | (buf[pos + 9] << 8);
        const std::uint32_t csize = buf[pos + 18] | (buf[pos + 19] << 8) |
                                    (buf[pos + 20] << 16) | (buf[pos + 21] << 24);
        const std::uint16_t nlen = buf[pos + 26] | (buf[pos + 27] << 8);
        const std::uint16_t elen = buf[pos + 28] | (buf[pos + 29] << 8);
        std::string name(reinterpret_cast<const char*>(&buf[pos + 30]), nlen);
        const std::size_t doff = pos + 30 + nlen + elen;
        if (method != 0) {  // priors are STORED; refuse deflate
            LOG_ERROR(kLog, "npz member %s is compressed (method=%d)",
                      name.c_str(), method);
            return false;
        }
        NpyArray arr;
        if (!parse_npy(&buf[doff], csize, arr)) return false;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy")
            name = name.substr(0, name.size() - 4);
        out[name] = std::move(arr);
        pos = doff + csize;
    }
    return !out.empty();
}

std::vector<double> as_f64(const NpyArray& a) {
    const std::size_t n = a.data.size() / (a.dtype == "<f8" ? 8 : 4);
    std::vector<double> v(n);
    if (a.dtype == "<f8") {
        const double* s = reinterpret_cast<const double*>(a.data.data());
        for (std::size_t i = 0; i < n; ++i) v[i] = s[i];
    } else {
        const float* s = reinterpret_cast<const float*>(a.data.data());
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<double>(s[i]);
    }
    return v;
}
std::vector<float> as_f32(const NpyArray& a) {
    const std::size_t n = a.data.size() / (a.dtype == "<f8" ? 8 : 4);
    std::vector<float> v(n);
    if (a.dtype == "<f8") {
        const double* s = reinterpret_cast<const double*>(a.data.data());
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(s[i]);
    } else {
        const float* s = reinterpret_cast<const float*>(a.data.data());
        for (std::size_t i = 0; i < n; ++i) v[i] = s[i];
    }
    return v;
}
}  // namespace

bool DiarizenClustering::load_priors(const std::string& plda_dir) {
    using Eigen::MatrixXd;
    std::map<std::string, NpyArray> xv, pl;
    if (!read_npz(plda_dir + "/xvec_transform.npz", xv)) {
        LOG_ERROR(kLog, "cannot read xvec_transform.npz in %s", plda_dir.c_str());
        return false;
    }
    if (!read_npz(plda_dir + "/plda.npz", pl)) {
        LOG_ERROR(kLog, "cannot read plda.npz in %s", plda_dir.c_str());
        return false;
    }
    if (!xv.count("mean1") || !xv.count("mean2") || !xv.count("lda") ||
        !pl.count("mu") || !pl.count("tr") || !pl.count("psi")) {
        LOG_ERROR(kLog, "missing prior arrays");
        return false;
    }

    DiarizenPldaPriors& P = priors_;
    P.mean1 = as_f64(xv["mean1"]);          // [256]
    P.mean2 = as_f32(xv["mean2"]);          // [128]
    P.lda = as_f32(xv["lda"]);              // [256,128]
    P.xdim = static_cast<int>(xv["lda"].shape[0]);
    P.pdim = static_cast<int>(xv["lda"].shape[1]);
    P.plda_mu = as_f64(pl["mu"]);           // [128]

    const std::vector<double> tr = as_f64(pl["tr"]);   // [pdim,pdim] row-major
    const std::vector<double> psi = as_f64(pl["psi"]);  // [pdim]
    const int d = P.pdim;

    // tr matrix (row-major) -> Eigen.
    MatrixXd TR(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) TR(i, j) = tr[i * d + j];

    // W = inv(tr^T @ tr)
    MatrixXd W = (TR.transpose() * TR).inverse();
    // B = inv((tr^T / psi) @ tr), where (tr.T / psi)[i,j] = tr.T[i,j]/psi[j].
    MatrixXd TRtScaled(d, d);  // tr.T with column j divided by psi[j]
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) TRtScaled(i, j) = TR(j, i) / psi[j];
    MatrixXd B = (TRtScaled * TR).inverse();

    // acvar, wccn = eigh(B, W): solve B v = lambda W v, ascending eigenvalues,
    // eigenvectors normalized so wccn^T W wccn = I (matches scipy.linalg.eigh).
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(
        B, W, Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
    if (es.info() != Eigen::Success) {
        LOG_ERROR(kLog, "generalized eigensolver failed");
        return false;
    }
    const Eigen::VectorXd acvar = es.eigenvalues();        // ascending
    const MatrixXd wccn = es.eigenvectors();               // columns

    // plda_psi = acvar[::-1]; plda_tr = wccn.T[::-1] (row k = eigvec for k-th
    // largest eigenvalue).
    P.plda_psi.resize(d);
    P.plda_tr.resize(static_cast<std::size_t>(d) * d);
    for (int k = 0; k < d; ++k) {
        const int src = d - 1 - k;
        P.plda_psi[k] = acvar[src];
        for (int j = 0; j < d; ++j) P.plda_tr[k * d + j] = wccn(j, src);
    }

    P.loaded = true;
    LOG_INFO(kLog, "loaded PLDA priors: xdim=%d pdim=%d psi[0]=%.6f psi[last]=%.6f",
             P.xdim, P.pdim, P.plda_psi.front(), P.plda_psi.back());
    return true;
}

void DiarizenClustering::compute_fea_(const float* train_emb, int n, int xdim,
                                      std::vector<double>& fea) const {
    const DiarizenPldaPriors& P = priors_;
    const int d = P.pdim;
    const double s0 = std::sqrt(static_cast<double>(P.xdim));  // sqrt(lda rows)
    const double s1 = std::sqrt(static_cast<double>(d));       // sqrt(lda cols)
    fea.assign(static_cast<std::size_t>(n) * d, 0.0);

    std::vector<double> v(xdim), y(d);
    for (int r = 0; r < n; ++r) {
        const float* x = train_emb + static_cast<std::size_t>(r) * xdim;
        // l2_norm(x - mean1) * sqrt(xdim)
        double nrm = 0.0;
        for (int i = 0; i < xdim; ++i) {
            v[i] = static_cast<double>(x[i]) - P.mean1[i];
            nrm += v[i] * v[i];
        }
        nrm = std::sqrt(nrm);
        for (int i = 0; i < xdim; ++i) v[i] = v[i] / nrm * s0;
        // lda^T @ v - mean2
        for (int j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int i = 0; i < xdim; ++i)
                acc += static_cast<double>(P.lda[i * d + j]) * v[i];
            y[j] = acc - static_cast<double>(P.mean2[j]);
        }
        // l2_norm(y) * sqrt(pdim) -> x_tf
        double n2 = 0.0;
        for (int j = 0; j < d; ++j) n2 += y[j] * y[j];
        n2 = std::sqrt(n2);
        for (int j = 0; j < d; ++j) y[j] = y[j] / n2 * s1;
        // plda_tf: fea[k] = dot(plda_tr_row_k, y - plda_mu)
        double* fr = fea.data() + static_cast<std::size_t>(r) * d;
        for (int k = 0; k < d; ++k) {
            double acc = 0.0;
            for (int j = 0; j < d; ++j)
                acc += P.plda_tr[k * d + j] * (y[j] - P.plda_mu[j]);
            fr[k] = acc;
        }
    }
}

bool DiarizenClustering::debug_fea(const float* train_emb, int n, int xdim,
                                   std::vector<float>& fea_out) {
    if (!priors_.loaded) return false;
    std::vector<double> fea;
    compute_fea_(train_emb, n, xdim, fea);
    fea_out.assign(fea.size(), 0.0f);
    for (std::size_t i = 0; i < fea.size(); ++i)
        fea_out[i] = static_cast<float>(fea[i]);
    return true;
}

}  // namespace orator
}  // namespace deusridet
