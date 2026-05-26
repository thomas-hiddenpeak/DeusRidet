/**
 * @file orator_reclusterer_eval.cpp
 * @philosophical_role Detached evaluator for `OratorReclusterer` against
 *     a pre-computed GT-aligned embedding dump (tests/fixtures/fused_v1.bin).
 *     Bypasses VAD, dual_db, and the live identify/match gates.
 *
 * @internal_check_only This tool is **internal-check-only** per
 *     `.github/instructions/benchmarks.instructions.md`. Numbers produced
 *     here (macro F1, Hungarian-mapped accuracy, K_pred) describe a slice
 *     of the system, NOT the system. They MUST NOT be used to:
 *       - set or change any default value;
 *       - flip any always-on / always-off behaviour;
 *       - declare a phase positive or negative;
 *       - claim a quality regression or improvement.
 *     Permitted uses: GPU-vs-CPU bit-equality regression, kernel micro-
 *     benchmarks, one-shot algorithmic sanity. For behavioural verdicts,
 *     use `tools/replay_to_transcript.py` against the live `awaken`
 *     pipeline streaming `tests/test.mp3`, and read the report directly.
 *
 * @serves Internal sanity only.
 */
#include "src/orator/orator_reclusterer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace dr = deusridet::orator;

namespace {

struct FixtureRecord {
    double t_center;
    double t_start;
    double t_end;
    int    gt;
    std::vector<float> emb;
};

struct Fixture {
    int dim = 0;
    int n_speakers = 0;
    std::vector<FixtureRecord> recs;
};

bool load_fixture(const std::string& path, Fixture& out) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "cannot open fixture: %s\n", path.c_str());
        return false;
    }
    uint32_t magic = 0;
    int32_t  n = 0, dim = 0, K = 0;
    uint32_t strategy_index = 0;
    if (std::fread(&magic, 4, 1, fp) != 1 ||
        std::fread(&n,     4, 1, fp) != 1 ||
        std::fread(&dim,   4, 1, fp) != 1 ||
        std::fread(&K,     4, 1, fp) != 1 ||
        std::fread(&strategy_index, 4, 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }
    if (magic != 0x4F524554u) {
        std::fprintf(stderr, "bad magic 0x%x\n", magic);
        std::fclose(fp);
        return false;
    }
    out.dim = dim;
    out.n_speakers = K;
    out.recs.resize(n);
    for (int i = 0; i < n; ++i) {
        FixtureRecord& r = out.recs[i];
        if (std::fread(&r.t_center, 8, 1, fp) != 1 ||
            std::fread(&r.t_start,  8, 1, fp) != 1 ||
            std::fread(&r.t_end,    8, 1, fp) != 1 ||
            std::fread(&r.gt,       4, 1, fp) != 1) {
            std::fclose(fp);
            return false;
        }
        r.emb.resize(dim);
        if (std::fread(r.emb.data(), 4, dim, fp) != static_cast<size_t>(dim)) {
            std::fclose(fp);
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

// Hungarian-mapped macro F1: build a confusion matrix between predicted
// labels and GT, find best one-to-one mapping that maximises the sum of
// per-class F1 (we approximate with greedy assignment on per-pair F1
// scores — sufficient for K≤8).
struct MacroResult {
    double macro_f1;
    int    K_pred;
    int    K_used; // pred clusters mapped to a GT label
    std::map<int, int> mapping; // pred -> gt
};

double class_f1(int tp, int fp, int fn) {
    if (tp == 0) return 0.0;
    const double p = double(tp) / double(tp + fp);
    const double r = double(tp) / double(tp + fn);
    return 2.0 * p * r / (p + r);
}

MacroResult macro_f1(const std::vector<int>& pred, const std::vector<int>& gt,
                     int K_gt) {
    MacroResult R{};
    // Map predicted labels into a dense [0, K_pred) namespace.
    std::map<int, int> pred_to_dense;
    for (int p : pred) {
        if (p < 0) continue;
        if (!pred_to_dense.count(p)) {
            const int idx = static_cast<int>(pred_to_dense.size());
            pred_to_dense[p] = idx;
        }
    }
    const int Kp = static_cast<int>(pred_to_dense.size());
    R.K_pred = Kp;
    if (Kp == 0 || K_gt == 0) return R;

    // confusion[Kp][K_gt]
    std::vector<std::vector<int>> conf(Kp, std::vector<int>(K_gt, 0));
    std::vector<int> pred_count(Kp, 0), gt_count(K_gt, 0);
    for (size_t i = 0; i < pred.size(); ++i) {
        const int p = pred[i];
        const int g = gt[i];
        if (p < 0 || g < 0 || g >= K_gt) continue;
        const int pi = pred_to_dense[p];
        conf[pi][g] += 1;
        pred_count[pi] += 1;
        gt_count[g] += 1;
    }

    // Pre-compute pair F1 for greedy assignment.
    std::vector<std::tuple<double, int, int>> pairs;
    pairs.reserve(static_cast<size_t>(Kp) * K_gt);
    for (int p = 0; p < Kp; ++p) {
        for (int g = 0; g < K_gt; ++g) {
            const int tp = conf[p][g];
            const int fp = pred_count[p] - tp;
            const int fn = gt_count[g]   - tp;
            const double f1 = class_f1(tp, fp, fn);
            pairs.emplace_back(f1, p, g);
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    std::vector<char> p_used(Kp, 0), g_used(K_gt, 0);
    double sum_f1 = 0.0;
    int matched_g = 0;
    // Build inverse mapping pred-dense -> pred-real
    std::vector<int> dense_to_pred(Kp, -1);
    for (auto& kv : pred_to_dense) dense_to_pred[kv.second] = kv.first;

    for (auto& t : pairs) {
        double f1 = std::get<0>(t);
        int p = std::get<1>(t);
        int g = std::get<2>(t);
        if (p_used[p] || g_used[g]) continue;
        if (f1 <= 0.0) continue;
        p_used[p] = 1;
        g_used[g] = 1;
        sum_f1 += f1;
        matched_g += 1;
        R.mapping[dense_to_pred[p]] = g;
        if (matched_g == K_gt) break;
    }
    R.macro_f1 = sum_f1 / double(K_gt);
    R.K_used   = matched_g;
    return R;
}

} // namespace

int main(int argc, char** argv) {
    std::string fixture_path = "/home/rm01/DeusRidet/tests/fixtures/fused_v1.bin";
    double end_sec   = 1800.0;     // s1800 by default
    double window_sec = 180.0;
    double interval_sec = 30.0;
    int    min_k = 2, max_k = 6;
    float  link_threshold = 0.55f;
    float  centroid_ema   = 0.20f;
    float  merge_threshold = -1.0f;   // disabled by default; -1 means no merge
    int    min_segments = 12;
    int    max_segments = 300;
    bool   force_final = true;     // run a final force_run after stream ends
    bool   diag = false;
    int    k_mode = 0;             // 0=nme (legacy), 1=eigenvalue ratio
    bool   purity = false;         // Phase 8 — purity post-filter
    double purity_min_cos = 0.60;
    double purity_max_subsim = 0.85;
    bool   lenw = false;           // Phase 9 — length-weighted K-means
    int    max_globals = -1;       // Phase 10 — hard cap on global speakers (-1 disables)
    bool   aff_weighted = false;   // Phase 14 — reliability-weighted affinity
    double aff_dur_ref  = 1.5;     // Phase 14 — duration reference (s)
    std::string dump_pred_path;    // Phase 13 — write per-seg (gt,pred) JSONL
    double smooth_short_dur = 0.0; // Phase 13 — short-seg temporal smoothing dur threshold (s); 0=off
    int    smooth_neighbors = 8;   // Phase 13 — # of past+future neighbors to consult
    int    smooth_min_votes = 2;   // Phase 13 — minimum reliable-neighbor vote count to override

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
        auto nexti = [&](int& v)   { if (i + 1 < argc) v = std::atoi(argv[++i]); };
        if      (a == "--fixture" && i + 1 < argc) fixture_path = argv[++i];
        else if (a == "--end-sec")  next(end_sec);
        else if (a == "--W")        next(window_sec);
        else if (a == "--S")        next(interval_sec);
        else if (a == "--min-k")    nexti(min_k);
        else if (a == "--max-k")    nexti(max_k);
        else if (a == "--thr")      { double v = 0.55; next(v); link_threshold = float(v); }
        else if (a == "--ema")      { double v = 0.20; next(v); centroid_ema   = float(v); }
        else if (a == "--merge-thr"){ double v = 0.85; next(v); merge_threshold= float(v); }
        else if (a == "--min-segs") nexti(min_segments);
        else if (a == "--max-segs") nexti(max_segments);
        else if (a == "--no-final-force") force_final = false;
        else if (a == "--diag") diag = true;
        else if (a == "--k-mode") nexti(k_mode);
        else if (a == "--purity") purity = true;
        else if (a == "--purity-min-cos")    next(purity_min_cos);
        else if (a == "--purity-max-subsim") next(purity_max_subsim);
        else if (a == "--lenw") lenw = true;
        else if (a == "--max-globals") nexti(max_globals);
        else if (a == "--aff-weighted") aff_weighted = true;
        else if (a == "--aff-dur-ref") next(aff_dur_ref);
        else if (a == "--dump-pred" && i + 1 < argc) dump_pred_path = argv[++i];
        else if (a == "--smooth-short")     next(smooth_short_dur);
        else if (a == "--smooth-neighbors") nexti(smooth_neighbors);
        else if (a == "--smooth-min-votes") nexti(smooth_min_votes);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    Fixture fx;
    if (!load_fixture(fixture_path, fx)) return 1;
    std::fprintf(stderr,
                 "[eval] fixture: %d recs, dim=%d, K_gt=%d, t_range=[%.1f, %.1f] s\n",
                 (int)fx.recs.size(), fx.dim, fx.n_speakers,
                 fx.recs.empty() ? 0.0 : fx.recs.front().t_center,
                 fx.recs.empty() ? 0.0 : fx.recs.back().t_center);
    std::fprintf(stderr,
                 "[eval] cfg: end=%.1fs W=%.1fs S=%.1fs min_k=%d max_k=%d thr=%.2f min_segs=%d max_segs=%d\n",
                 end_sec, window_sec, interval_sec, min_k, max_k, link_threshold,
                 min_segments, max_segments);

    dr::OratorReclustererConfig cfg;
    cfg.embedding_dim   = fx.dim;
    cfg.window_sec      = window_sec;
    cfg.interval_sec    = interval_sec;
    cfg.min_segments    = min_segments;
    cfg.max_segments    = max_segments;
    cfg.min_k           = min_k;
    cfg.max_k           = max_k;
    cfg.link_threshold  = link_threshold;
    cfg.centroid_ema    = centroid_ema;
    cfg.global_id_base  = 1000;
    cfg.global_merge_threshold = merge_threshold;
    cfg.k_selection_mode = k_mode;
    cfg.purity_split_enable      = purity;
    cfg.purity_min_mean_cos      = float(purity_min_cos);
    cfg.purity_accept_max_subsim = float(purity_max_subsim);
    cfg.length_weighted_enable   = lenw;
    cfg.max_global_speakers      = max_globals;
    cfg.affinity_weighted_enable = aff_weighted;
    cfg.affinity_dur_ref         = float(aff_dur_ref);

    dr::OratorReclusterer rec(cfg);

    // Stream segments in chronological order up to end_sec. We keep a map
    // segment_id -> final_pred and update it from RelabelEvents as they
    // arrive. We also remember each segment's GT label keyed by segment_id.
    std::unordered_map<uint64_t, int> final_pred;
    std::unordered_map<uint64_t, int> gt_by_segid;

    std::vector<dr::RelabelEvent> ev_buf;
    int n_events_total = 0;
    int n_recs_used = 0;
    for (size_t i = 0; i < fx.recs.size(); ++i) {
        const FixtureRecord& r = fx.recs[i];
        if (r.t_center > end_sec) break;

        dr::ReclusterSegment seg;
        seg.segment_id = static_cast<uint64_t>(i + 1);
        seg.t_center_sec = r.t_center;
        seg.t_start_sec  = r.t_start;
        seg.t_end_sec    = r.t_end;
        seg.tentative_speaker_id = -1;
        seg.embedding   = r.emb;
        gt_by_segid[seg.segment_id] = r.gt;

        rec.push(seg);
        rec.tick(r.t_center);

        ev_buf.clear();
        rec.drain_relabels(ev_buf);
        for (const auto& ev : ev_buf) {
            final_pred[ev.segment_id] = ev.new_speaker_id;
        }
        n_events_total += static_cast<int>(ev_buf.size());
        n_recs_used += 1;
    }

    // Final force pass to commit any segments still in the trailing window.
    if (force_final && n_recs_used > 0) {
        const double now = fx.recs[n_recs_used - 1].t_center + 1e-3;
        const int n = rec.force_run(now);
        ev_buf.clear();
        rec.drain_relabels(ev_buf);
        for (const auto& ev : ev_buf) {
            final_pred[ev.segment_id] = ev.new_speaker_id;
        }
        n_events_total += n;
    }

    std::fprintf(stderr, "[eval] streamed %d segs, %d relabel events, %d globals\n",
                 n_recs_used, n_events_total, rec.global_speaker_count());

    // Phase 13 — temporal smoothing for short segments.
    // If a seg with dur < smooth_short_dur has a majority of reliable neighbors
    // (dur >= smooth_short_dur) labelled X, override its label to X.
    int n_smoothed = 0;
    if (smooth_short_dur > 0.0 && n_recs_used > 0) {
        const int N = static_cast<int>(fx.recs.size());
        std::vector<int> new_labels(N, -2); // -2 = no change; -1 = unlabeled
        for (int i = 0; i < N; ++i) {
            const FixtureRecord& r = fx.recs[i];
            if (!gt_by_segid.count(static_cast<uint64_t>(i + 1))) continue;
            const double dur = r.t_end - r.t_start;
            if (dur >= smooth_short_dur) continue;
            const uint64_t sid_i = static_cast<uint64_t>(i + 1);
            auto pit = final_pred.find(sid_i);
            if (pit == final_pred.end()) continue;
            const int cur = pit->second;
            // Count reliable neighbor labels.
            std::map<int, int> votes;
            const int lo = std::max(0, i - smooth_neighbors);
            const int hi = std::min(N - 1, i + smooth_neighbors);
            int reliable = 0;
            for (int j = lo; j <= hi; ++j) {
                if (j == i) continue;
                const FixtureRecord& rj = fx.recs[j];
                const double dj = rj.t_end - rj.t_start;
                if (dj < smooth_short_dur) continue;
                const uint64_t sid_j = static_cast<uint64_t>(j + 1);
                auto pj = final_pred.find(sid_j);
                if (pj == final_pred.end()) continue;
                votes[pj->second] += 1;
                reliable += 1;
            }
            if (reliable == 0) continue;
            // Find top vote.
            int top_label = -1, top_count = 0;
            for (const auto& kv : votes) {
                if (kv.second > top_count) { top_count = kv.second; top_label = kv.first; }
            }
            if (top_label == cur) continue;
            if (top_count < smooth_min_votes) continue;
            if (top_count * 2 <= reliable) continue; // need strict majority
            new_labels[i] = top_label;
        }
        // Apply changes after the scan.
        for (int i = 0; i < N; ++i) {
            if (new_labels[i] == -2) continue;
            const uint64_t sid = static_cast<uint64_t>(i + 1);
            final_pred[sid] = new_labels[i];
            ++n_smoothed;
        }
        std::fprintf(stderr, "[smooth-short] dur<%.2f, K=%d, min_votes=%d: %d labels overridden\n",
                     smooth_short_dur, smooth_neighbors, smooth_min_votes, n_smoothed);
    }


    // Build dense pred + gt vectors for the segments that received a label.
    std::vector<int> pred_v, gt_v;
    pred_v.reserve(n_recs_used);
    gt_v.reserve(n_recs_used);
    int n_uncommitted = 0;
    for (size_t i = 0; i < fx.recs.size(); ++i) {
        const uint64_t sid = static_cast<uint64_t>(i + 1);
        if (!gt_by_segid.count(sid)) continue; // beyond end_sec
        auto it = final_pred.find(sid);
        if (it == final_pred.end()) {
            n_uncommitted += 1;
            continue;
        }
        pred_v.push_back(it->second);
        gt_v.push_back(gt_by_segid[sid]);
    }
    if (n_uncommitted > 0) {
        std::fprintf(stderr, "[eval] %d segments never received a committed label\n", n_uncommitted);
    }

    MacroResult M = macro_f1(pred_v, gt_v, fx.n_speakers);
    std::fprintf(stderr,
                 "[eval] macro_f1=%.4f  K_pred=%d  K_mapped=%d/%d  n_scored=%zu\n",
                 M.macro_f1, M.K_pred, M.K_used, fx.n_speakers, pred_v.size());

    if (!dump_pred_path.empty()) {
        std::FILE* fp = std::fopen(dump_pred_path.c_str(), "w");
        if (!fp) {
            std::fprintf(stderr, "[dump-pred] cannot open: %s\n", dump_pred_path.c_str());
        } else {
            int n_written = 0;
            for (size_t i = 0; i < fx.recs.size(); ++i) {
                const uint64_t sid = static_cast<uint64_t>(i + 1);
                if (!gt_by_segid.count(sid)) continue;
                const FixtureRecord& r = fx.recs[i];
                auto it = final_pred.find(sid);
                const int pred = (it == final_pred.end()) ? -1 : it->second;
                int mapped_gt = -1;
                if (pred >= 0) {
                    auto mit = M.mapping.find(pred);
                    if (mit != M.mapping.end()) mapped_gt = mit->second;
                }
                const int gt = gt_by_segid[sid];
                const int correct = (mapped_gt == gt) ? 1 : 0;
                std::fprintf(fp,
                    "{\"idx\":%zu,\"sid\":%llu,\"t_start\":%.3f,\"t_end\":%.3f,"
                    "\"t_center\":%.3f,\"gt\":%d,\"pred\":%d,\"mapped_gt\":%d,\"correct\":%d}\n",
                    i, (unsigned long long)sid, r.t_start, r.t_end, r.t_center,
                    gt, pred, mapped_gt, correct);
                ++n_written;
            }
            std::fclose(fp);
            std::fprintf(stderr, "[dump-pred] wrote %d rows -> %s\n", n_written, dump_pred_path.c_str());
        }
    }

    if (diag) {
        // Per-predicted-global histogram over GT classes.
        std::map<int, std::vector<int>> hist; // pred_id -> [count per gt]
        for (size_t i = 0; i < pred_v.size(); ++i) {
            auto& h = hist[pred_v[i]];
            if (h.empty()) h.assign(fx.n_speakers, 0);
            if (gt_v[i] >= 0 && gt_v[i] < fx.n_speakers) h[gt_v[i]] += 1;
        }
        std::fprintf(stderr, "[diag] per-pred-global GT distribution (pred_id -> [gt0,gt1,...]):\n");
        for (const auto& kv : hist) {
            std::fprintf(stderr, "[diag]   pred=%d  total=%d  hist=[", kv.first,
                         std::accumulate(kv.second.begin(), kv.second.end(), 0));
            for (size_t j = 0; j < kv.second.size(); ++j) {
                std::fprintf(stderr, "%s%d", j ? "," : "", kv.second[j]);
            }
            int mapped = -1;
            auto it = M.mapping.find(kv.first);
            if (it != M.mapping.end()) mapped = it->second;
            std::fprintf(stderr, "]  mapped_to_gt=%d\n", mapped);
        }
        // Pairwise centroid cosine sims among current globals.
        const auto& gs = rec.globals();
        std::vector<std::pair<int, const dr::GlobalSpeaker*>> v;
        for (const auto& kv : gs) v.emplace_back(kv.first, &kv.second);
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        std::fprintf(stderr, "[diag] pairwise centroid cosine sims:\n");
        for (size_t i = 0; i < v.size(); ++i) {
            for (size_t j = i + 1; j < v.size(); ++j) {
                const auto& a = v[i].second->centroid;
                const auto& b = v[j].second->centroid;
                double s = 0.0;
                for (size_t k = 0; k < a.size() && k < b.size(); ++k) s += double(a[k]) * double(b[k]);
                std::fprintf(stderr, "[diag]   cos(%d, %d) = %.4f  (supports %d / %d)\n",
                             v[i].first, v[j].first, s,
                             v[i].second->support_count, v[j].second->support_count);
            }
        }
    }

    std::printf("{\"macro_f1\":%.4f,\"K_pred\":%d,\"K_mapped\":%d,\"K_gt\":%d,"
                "\"n_scored\":%zu,\"n_uncommitted\":%d,\"n_events\":%d,"
                "\"end_sec\":%.1f,\"window_sec\":%.1f,\"interval_sec\":%.1f,"
                "\"min_k\":%d,\"max_k\":%d,\"link_threshold\":%.3f,"
                "\"ema\":%.3f,\"merge_thr\":%.3f,\"k_mode\":%d,"
                "\"purity\":%d,\"lw\":%d,\"kc\":%d}\n",
                M.macro_f1, M.K_pred, M.K_used, fx.n_speakers,
                pred_v.size(), n_uncommitted, n_events_total,
                end_sec, window_sec, interval_sec, min_k, max_k, link_threshold,
                centroid_ema, merge_threshold, k_mode, purity ? 1 : 0,
                lenw ? 1 : 0, max_globals);
    return 0;
}
