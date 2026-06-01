// @philosophical_role Implementation of the cross-window persistent-identity
//   registry (see diarizen_identity_registry.h). Voiceprint-first binding with
//   a time-overlap fallback for labels that carry no usable embedding.
// @serves DiarizenPeriodicWorker.
#include "diarizen_identity_registry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace deusridet {
namespace orator {

namespace {

inline double overlap_seconds(double a0, double a1, double b0, double b1) {
    const double lo = std::max(a0, b0);
    const double hi = std::min(a1, b1);
    return (hi > lo) ? (hi - lo) : 0.0;
}

// Cosine of two L2-normalised vectors == dot product. -1 if either is empty.
inline double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return -1.0;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += (double)a[i] * b[i];
    return dot;
}

// Parse "speaker<k>" -> k, or -1 if the label is not in that form.
inline int parse_cluster_id(const std::string& label) {
    const size_t n = 7;  // strlen("speaker")
    if (label.size() <= n || label.compare(0, n, "speaker") != 0) return -1;
    char* end = nullptr;
    long v = std::strtol(label.c_str() + n, &end, 10);
    if (end == label.c_str() + n) return -1;
    return static_cast<int>(v);
}

}  // namespace

size_t DiarizenIdentityRegistry::stitch(
    std::vector<DiarizenSegment>& segs, double origin_sec,
    const std::vector<std::vector<float>>& centroids) {
    if (segs.empty()) return 0;

    // Gather per-local-label absolute intervals, the window end (fallback
    // pruning), and each label's voiceprint (empty if no usable centroid).
    double window_end = 0.0;
    std::unordered_map<std::string, std::vector<Occ>> label_iv;
    std::unordered_map<std::string, std::vector<float>> label_emb;
    for (const auto& s : segs) {
        const double a0 = s.start_sec + origin_sec;
        const double a1 = s.end_sec + origin_sec;
        window_end = std::max(window_end, a1);
        label_iv[s.label].push_back(Occ{a0, a1});
        if (label_emb.find(s.label) == label_emb.end()) {
            std::vector<float> emb;
            const int k = parse_cluster_id(s.label);
            if (k >= 0 && k < (int)centroids.size() && !centroids[k].empty()) {
                double nrm = 0.0;
                for (float v : centroids[k]) nrm += (double)v * v;
                if (nrm > 1e-9) emb = centroids[k];  // reject all-zero
            }
            label_emb[s.label] = std::move(emb);
        }
    }

    struct Cand {
        std::string label;
        int gid;
        double score;
    };

    // --- Pass 1: voiceprint binding (greedy one-to-one cosine) --------------
    std::vector<Cand> cands;
    for (const auto& kv : label_emb) {
        if (kv.second.empty()) continue;  // no embedding -> fallback later
        for (const auto& pk : proto_) {
            const double c = cosine(kv.second, pk.second);
            if (c >= cfg_.match_min_cos) cands.push_back({kv.first, pk.first, c});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });

    std::unordered_map<std::string, int> label_to_gid;
    std::unordered_map<int, bool> gid_taken;
    for (const auto& c : cands) {
        if (label_to_gid.count(c.label)) continue;
        if (gid_taken.count(c.gid)) continue;
        label_to_gid[c.label] = c.gid;
        gid_taken[c.gid] = true;
    }

    // --- Pass 2: time-overlap fallback (only labels with no embedding) ------
    std::vector<Cand> ov_cands;
    for (const auto& kv : label_iv) {
        if (label_to_gid.count(kv.first)) continue;
        if (!label_emb[kv.first].empty()) continue;  // had embedding, no match
        for (const auto& hk : hist_) {
            if (gid_taken.count(hk.first)) continue;
            double ov = 0.0;
            for (const auto& iv : kv.second)
                for (const auto& h : hk.second)
                    ov += overlap_seconds(iv.start_sec, iv.end_sec,
                                          h.start_sec, h.end_sec);
            if (ov >= cfg_.match_min_overlap_sec)
                ov_cands.push_back({kv.first, hk.first, ov});
        }
    }
    std::sort(ov_cands.begin(), ov_cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    for (const auto& c : ov_cands) {
        if (label_to_gid.count(c.label)) continue;
        if (gid_taken.count(c.gid)) continue;
        label_to_gid[c.label] = c.gid;
        gid_taken[c.gid] = true;
    }

    // --- Mint fresh identities for everything still unbound -----------------
    for (const auto& kv : label_iv) {
        if (label_to_gid.count(kv.first)) continue;
        label_to_gid[kv.first] = next_gid_++;
    }

    // --- Update prototypes (EMA) and occupancy history ----------------------
    const double cutoff = window_end - cfg_.history_horizon_sec;
    for (const auto& kv : label_iv) {
        const int gid = label_to_gid[kv.first];
        auto& dq = hist_[gid];
        for (const auto& iv : kv.second) dq.push_back(iv);
        while (!dq.empty() && dq.front().end_sec < cutoff) dq.pop_front();
        const auto& emb = label_emb[kv.first];
        if (emb.empty()) continue;
        auto it = proto_.find(gid);
        if (it == proto_.end() || it->second.empty()) {
            proto_[gid] = emb;  // first acoustic evidence
        } else {
            auto& p = it->second;
            const double a = cfg_.ema_alpha;
            double nrm = 0.0;
            for (size_t i = 0; i < p.size(); ++i) {
                p[i] = static_cast<float>((1.0 - a) * p[i] + a * emb[i]);
                nrm += (double)p[i] * p[i];
            }
            const float inv =
                (nrm > 0.0) ? static_cast<float>(1.0 / std::sqrt(nrm)) : 0.0f;
            for (auto& v : p) v *= inv;  // renormalise to unit length
        }
    }

    // --- Rewrite labels in place to stable global identity strings ----------
    for (auto& s : segs) s.label = "S" + std::to_string(label_to_gid[s.label]);

    std::unordered_map<int, bool> seen;
    for (const auto& kv : label_to_gid) seen[kv.second] = true;
    return seen.size();
}

}  // namespace orator
}  // namespace deusridet
