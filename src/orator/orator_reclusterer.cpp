/**
 * @file orator_reclusterer.cpp
 * @philosophical_role The reflective hearing — converts the existing
 *     `spectral_cluster()` warm-up primitive into a continuously-running
 *     background corrector over a rolling window.
 * @serves Speaker diarisation pipeline. Replaces AHC-centroid-cosine
 *     `SpeakerVectorStore::identify()` for the persistent-ID role.
 */
#include "orator_reclusterer.h"

#include "hungarian.h"
#include "spectral_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace deusridet::orator {

namespace {

// Phase 3b.1 — env-gated audit log of every reclusterer pass. When
// `DEUSRIDET_RECLUSTERER_DEBUG=1` is set, run_pass() dumps the spectral
// output, the global-vs-cluster cosine matrix, the Hungarian assignment,
// and every merge action to stderr. Off by default (zero cost when off).
bool debug_enabled() {
    static const bool on = []() {
        const char* v = std::getenv("DEUSRIDET_RECLUSTERER_DEBUG");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

void l2_normalise(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    if (s <= 0.0) return;
    const float inv = static_cast<float>(1.0 / std::sqrt(s));
    for (float& x : v) x *= inv;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(s);
}

// EMA towards `target` (assumed L2-norm 1), then renormalise.
void update_centroid_ema(std::vector<float>& c,
                         const std::vector<float>& target,
                         float rate) {
    if (c.size() != target.size()) {
        c = target;
        return;
    }
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = (1.0f - rate) * c[i] + rate * target[i];
    }
    l2_normalise(c);
}

} // namespace

OratorReclusterer::OratorReclusterer(OratorReclustererConfig cfg)
    : cfg_(std::move(cfg)) {
    next_global_id_ = cfg_.global_id_base;
}

void OratorReclusterer::push(const ReclusterSegment& seg) {
    if (static_cast<int>(seg.embedding.size()) != cfg_.embedding_dim) {
        // Boundary check: drop misshapen embeddings rather than crash.
        std::fprintf(stderr,
                     "[orator-recluster] dropped seg %llu: embedding dim %zu != cfg %d\n",
                     static_cast<unsigned long long>(seg.segment_id),
                     seg.embedding.size(), cfg_.embedding_dim);
        return;
    }
    Slot slot;
    slot.seg = seg;
    l2_normalise(slot.seg.embedding);
    slot.committed_speaker_id = -1;
    slot.ever_committed       = false;
    buffer_.push_back(std::move(slot));

    // Soft cap — drop oldest if we exceed max_segments.
    while (static_cast<int>(buffer_.size()) > cfg_.max_segments) {
        buffer_.pop_front();
    }
}

void OratorReclusterer::prune_window(double now_sec) {
    const double cutoff = now_sec - cfg_.window_sec;
    while (!buffer_.empty() && buffer_.front().seg.t_center_sec < cutoff) {
        buffer_.pop_front();
    }
}

int OratorReclusterer::tick(double now_sec) {
    if (last_run_sec_ <= -1e17) {
        // First call — set the clock but don't run until we accumulate data.
        last_run_sec_ = now_sec;
        return 0;
    }
    if (now_sec - last_run_sec_ < cfg_.interval_sec) return 0;
    return force_run(now_sec);
}

int OratorReclusterer::force_run(double now_sec) {
    last_run_sec_ = now_sec;
    prune_window(now_sec);
    if (static_cast<int>(buffer_.size()) < cfg_.min_segments) return 0;
    return run_pass(now_sec);
}

int OratorReclusterer::run_pass(double now_sec) {
    const int N   = static_cast<int>(buffer_.size());
    const int dim = cfg_.embedding_dim;

    // Build input to spectral_cluster().
    std::vector<std::vector<float>> X;
    X.reserve(N);
    std::vector<float> ts;
    ts.reserve(N);
    std::vector<float> weights;     // Phase 9 — per-segment durations (s).
    weights.reserve(N);
    for (const Slot& s : buffer_) {
        X.push_back(s.seg.embedding);
        ts.push_back(static_cast<float>(s.seg.t_center_sec));
        const float dur = static_cast<float>(
            s.seg.t_end_sec - s.seg.t_start_sec);
        // Clamp to a small floor so a degenerate zero-length segment
        // does not vanish from the centroid sum.
        weights.push_back(dur > 0.05f ? dur : 0.05f);
    }

    SpectralClusterConfig sc;
    // For 384D fused embeddings the existing warm-up defaults (designed for
    // 192D CAM++) work well; we only raise the PCA target a notch. Temporal
    // proximity weighting is disabled — over a 600 s window the warm-up's
    // tau=3.125 s Gaussian collapses every distant pair to ~0 affinity and
    // destroys the global cluster structure that the offline pass exists to
    // recover (PoC ablation: macro 0.55 with temporal_alpha=0.93, 0.76 with
    // 0.0).
    sc.pca_dim          = 32;
    sc.temporal_alpha   = 0.0f;
    sc.temporal_tau     = 3.125f;
    sc.p_prune_ratio    = 0.10f;
    sc.max_k            = cfg_.max_k;
    sc.min_k            = cfg_.min_k;
    sc.merge_threshold  = 0.55f;
    sc.kmeans_restarts  = 20;
    sc.kmeans_iters     = 100;
    sc.power_iters      = 300;
    sc.smooth_window    = 1;
    sc.smooth_iters     = 3;
    sc.k_selection_mode = cfg_.k_selection_mode;

    // Phase 8 — purity post-filter pass-through.
    sc.purity_split_enable          = cfg_.purity_split_enable;
    sc.purity_min_cluster_size      = cfg_.purity_min_cluster_size;
    sc.purity_min_mean_cos          = cfg_.purity_min_mean_cos;
    sc.purity_accept_max_subsim     = cfg_.purity_accept_max_subsim;
    sc.purity_split_kmeans_iters    = cfg_.purity_split_kmeans_iters;
    sc.purity_split_kmeans_restarts = cfg_.purity_split_kmeans_restarts;

    // Phase 9 — length-weighted K-means pass-through.
    sc.length_weighted_enable = cfg_.length_weighted_enable;

    // Phase 14 — reliability-weighted affinity pass-through.
    sc.affinity_weighted_enable = cfg_.affinity_weighted_enable;
    sc.affinity_dur_ref         = cfg_.affinity_dur_ref;

    ClusterResult cr = spectral_cluster(X, ts, dim, sc, weights);
    if (cr.K <= 0 || static_cast<int>(cr.labels.size()) != N) return 0;

    // Build local centroids in original space — spectral_cluster already
    // returns them L2-normalised. local_centroids[k] corresponds to cluster k.
    const int K = cr.K;
    const auto& local_centroids = cr.centroids;

    if (debug_enabled()) {
        std::fprintf(stderr,
            "\n[recluster-audit] === pass @ now=%.3fs  N=%d  K=%d  G=%zu ===\n",
            now_sec, N, K, globals_.size());
        // Per-cluster composition: list raw_id, segment_id, duration.
        std::vector<int> sizes(K, 0);
        std::vector<double> dur_sum(K, 0.0);
        for (int i = 0; i < N; ++i) {
            const int k = cr.labels[i];
            if (k < 0 || k >= K) continue;
            sizes[k] += 1;
            dur_sum[k] += buffer_[i].seg.t_end_sec - buffer_[i].seg.t_start_sec;
        }
        for (int k = 0; k < K; ++k) {
            std::fprintf(stderr,
                "[recluster-audit]   cluster %d: size=%d dur=%.2fs members=[",
                k, sizes[k], dur_sum[k]);
            int printed = 0;
            for (int i = 0; i < N; ++i) {
                if (cr.labels[i] != k) continue;
                if (printed > 0) std::fprintf(stderr, " ");
                std::fprintf(stderr, "(seg=%llu raw=%d t=%.2f-%.2f)",
                    static_cast<unsigned long long>(buffer_[i].seg.segment_id),
                    buffer_[i].seg.tentative_speaker_id,
                    buffer_[i].seg.t_start_sec,
                    buffer_[i].seg.t_end_sec);
                printed += 1;
            }
            std::fprintf(stderr, "]\n");
        }
    }

    // Hungarian match local clusters → existing globals.
    // Collect existing globals into a stable vector.
    std::vector<GlobalSpeaker*> globs;
    globs.reserve(globals_.size());
    for (auto& kv : globals_) globs.push_back(&kv.second);
    const int G = static_cast<int>(globs.size());

    // local_to_global[k] = id of matched global speaker, or -1 if a new one is needed.
    std::vector<int> local_to_global(K, -1);
    std::vector<float> match_conf(K, 0.0f);

    if (G > 0) {
        // Cost = -cos similarity (Hungarian minimises). Range -> [0, 2].
        std::vector<double> cost(static_cast<size_t>(K) * G, 0.0);
        for (int k = 0; k < K; ++k) {
            for (int g = 0; g < G; ++g) {
                const float c = cosine(local_centroids[k], globs[g]->centroid);
                cost[k * G + g] = 1.0 - static_cast<double>(c); // [0, 2]
            }
        }
        auto assign = solve_assignment(cost, K, G);
        if (debug_enabled()) {
            std::fprintf(stderr,
                "[recluster-audit]   cosine matrix (cluster x global):\n");
            std::fprintf(stderr, "[recluster-audit]            ");
            for (int g = 0; g < G; ++g) {
                std::fprintf(stderr, " g%d(id=%d)  ", g, globs[g]->id);
            }
            std::fprintf(stderr, "\n");
            for (int k = 0; k < K; ++k) {
                std::fprintf(stderr, "[recluster-audit]   k=%d:    ", k);
                for (int g = 0; g < G; ++g) {
                    const float sim = 1.0f - static_cast<float>(cost[k * G + g]);
                    const bool mark = (assign[k] == g);
                    std::fprintf(stderr, "%s%.4f%s  ",
                        mark ? "[" : " ",
                        sim,
                        mark ? "]" : " ");
                }
                std::fprintf(stderr, "\n");
            }
            std::fprintf(stderr,
                "[recluster-audit]   link_threshold=%.3f  centroid_ema=%.3f\n",
                cfg_.link_threshold, cfg_.centroid_ema);
        }
        for (int k = 0; k < K; ++k) {
            const int g = assign[k];
            if (g < 0) continue;
            const float sim = 1.0f - static_cast<float>(cost[k * G + g]);
            if (sim >= cfg_.link_threshold) {
                local_to_global[k] = globs[g]->id;
                match_conf[k]      = sim;
                if (debug_enabled()) {
                    std::fprintf(stderr,
                        "[recluster-audit]   k=%d -> reuse global id=%d sim=%.4f (>=%.3f)\n",
                        k, globs[g]->id, sim, cfg_.link_threshold);
                }
            } else if (debug_enabled()) {
                std::fprintf(stderr,
                    "[recluster-audit]   k=%d -> REJECT global id=%d sim=%.4f (<%.3f)  will allocate new id\n",
                    k, globs[g]->id, sim, cfg_.link_threshold);
            }
        }
    } else if (debug_enabled()) {
        std::fprintf(stderr,
            "[recluster-audit]   no existing globals (first pass) — every cluster gets a fresh id\n");
    }

    // Allocate fresh global IDs for unmatched local clusters.
    for (int k = 0; k < K; ++k) {
        if (local_to_global[k] != -1) continue;
        const int new_id = next_global_id_++;
        GlobalSpeaker gs;
        gs.id            = new_id;
        gs.centroid      = local_centroids[k];
        gs.support_count = 0;
        gs.last_seen_sec = now_sec;
        globals_.emplace(new_id, std::move(gs));
        local_to_global[k] = new_id;
        match_conf[k]      = 1.0f; // self-match
        if (debug_enabled()) {
            std::fprintf(stderr,
                "[recluster-audit]   k=%d -> NEW global id=%d\n", k, new_id);
        }
    }

    // Update global centroids via EMA and bookkeeping.
    std::vector<int> support_delta(K, 0);
    for (int i = 0; i < N; ++i) {
        const int k = cr.labels[i];
        if (k < 0 || k >= K) continue;
        support_delta[k] += 1;
    }
    for (int k = 0; k < K; ++k) {
        auto it = globals_.find(local_to_global[k]);
        if (it == globals_.end()) continue;
        update_centroid_ema(it->second.centroid, local_centroids[k], cfg_.centroid_ema);
        it->second.support_count += support_delta[k];
        it->second.last_seen_sec  = now_sec;
    }

    // Merge near-duplicate global centroids. K_pred>K_gt is usually caused by
    // one true speaker being split across two persistent globals. We collapse
    // any pair with cos sim ≥ global_merge_threshold, keeping the one with
    // larger support, weighted-averaging the centroids, and remapping both
    // the just-built local_to_global[] and every previously-committed slot
    // in the buffer so subsequent ticks see a consistent identity space.
    if (cfg_.global_merge_threshold > 0.0f && globals_.size() >= 2) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<int> ids;
            ids.reserve(globals_.size());
            for (auto& kv : globals_) ids.push_back(kv.first);
            for (size_t a = 0; a < ids.size() && !changed; ++a) {
                for (size_t b = a + 1; b < ids.size(); ++b) {
                    auto ita = globals_.find(ids[a]);
                    auto itb = globals_.find(ids[b]);
                    if (ita == globals_.end() || itb == globals_.end()) continue;
                    const float sim = cosine(ita->second.centroid, itb->second.centroid);
                    if (sim < cfg_.global_merge_threshold) continue;
                    // Support-disparity gate. Only merge if one of the two
                    // is clearly smaller than the other; this prevents two
                    // equally-large but topic-mixed clusters from collapsing.
                    const int sa = std::max(1, ita->second.support_count);
                    const int sb = std::max(1, itb->second.support_count);
                    const int s_min = std::min(sa, sb);
                    const int s_max = std::max(sa, sb);
                    if (cfg_.global_merge_support_ratio > 0.0f &&
                        static_cast<float>(s_min) > cfg_.global_merge_support_ratio * static_cast<float>(s_max)) {
                        continue;
                    }
                    int keep_id = ids[a], drop_id = ids[b];
                    if (itb->second.support_count > ita->second.support_count) {
                        keep_id = ids[b]; drop_id = ids[a];
                    }
                    GlobalSpeaker& keep = globals_[keep_id];
                    GlobalSpeaker& drop = globals_[drop_id];
                    const float wa = static_cast<float>(std::max(1, keep.support_count));
                    const float wb = static_cast<float>(std::max(1, drop.support_count));
                    const float tot = wa + wb;
                    for (size_t i = 0; i < keep.centroid.size(); ++i) {
                        keep.centroid[i] = (wa * keep.centroid[i] + wb * drop.centroid[i]) / tot;
                    }
                    l2_normalise(keep.centroid);
                    keep.support_count += drop.support_count;
                    keep.last_seen_sec  = std::max(keep.last_seen_sec, drop.last_seen_sec);
                    globals_.erase(drop_id);
                    if (debug_enabled()) {
                        std::fprintf(stderr,
                            "[recluster-audit]   GLOBAL-MERGE drop id=%d -> keep id=%d sim=%.4f (>=%.3f)  support keep_after=%d\n",
                            drop_id, keep_id, sim, cfg_.global_merge_threshold,
                            keep.support_count);
                    }
                    // Remap current local->global.
                    for (int k = 0; k < K; ++k) {
                        if (local_to_global[k] == drop_id) local_to_global[k] = keep_id;
                    }
                    // Remap previously committed labels in the buffer so the
                    // baseline comparison below doesn't fire spurious events.
                    for (Slot& s : buffer_) {
                        if (s.ever_committed && s.committed_speaker_id == drop_id) {
                            s.committed_speaker_id = keep_id;
                        }
                    }
                    // Retroactively relabel every historical segment that
                    // was committed to the dropped global — including those
                    // already evicted from the window. Emits RelabelEvents
                    // so downstream consumers (eval, WebUI) can patch their
                    // local mappings.
                    for (auto& kv : committed_history_) {
                        if (kv.second != drop_id) continue;
                        RelabelEvent ev;
                        ev.segment_id     = kv.first;
                        ev.old_speaker_id = drop_id;
                        ev.new_speaker_id = keep_id;
                        ev.confidence     = sim;
                        pending_.push_back(ev);
                        kv.second = keep_id;
                    }
                    changed = true;
                    break;
                }
            }
        }
    }

    // Phase 10 — hard K-cap on global speakers. Independent of cosine
    // similarity: while we have more globals than the configured cap,
    // collapse the smallest-support global into its nearest surviving
    // neighbour. Same remapping machinery as the cosine-merge block.
    if (cfg_.max_global_speakers > 0 &&
        static_cast<int>(globals_.size()) > cfg_.max_global_speakers) {
        while (static_cast<int>(globals_.size()) > cfg_.max_global_speakers &&
               globals_.size() >= 2) {
            // Smallest-support global.
            int drop_id = -1;
            int min_support = 0;
            for (const auto& kv : globals_) {
                if (drop_id < 0 || kv.second.support_count < min_support) {
                    drop_id     = kv.first;
                    min_support = kv.second.support_count;
                }
            }
            if (drop_id < 0) break;
            auto drop_it = globals_.find(drop_id);
            if (drop_it == globals_.end()) break;

            // Nearest survivor by centroid cosine.
            int keep_id = -1;
            float best_sim = -2.0f;
            for (const auto& kv : globals_) {
                if (kv.first == drop_id) continue;
                const float sim = cosine(drop_it->second.centroid, kv.second.centroid);
                if (sim > best_sim) { best_sim = sim; keep_id = kv.first; }
            }
            if (keep_id < 0) break;

            GlobalSpeaker& keep = globals_[keep_id];
            GlobalSpeaker& drop = drop_it->second;
            const float wa = static_cast<float>(std::max(1, keep.support_count));
            const float wb = static_cast<float>(std::max(1, drop.support_count));
            const float tot = wa + wb;
            for (size_t i = 0; i < keep.centroid.size(); ++i) {
                keep.centroid[i] = (wa * keep.centroid[i] + wb * drop.centroid[i]) / tot;
            }
            l2_normalise(keep.centroid);
            keep.support_count += drop.support_count;
            keep.last_seen_sec  = std::max(keep.last_seen_sec, drop.last_seen_sec);
            globals_.erase(drop_id);

            // Remap current local->global so commit phase sees one identity.
            for (int k = 0; k < K; ++k) {
                if (local_to_global[k] == drop_id) local_to_global[k] = keep_id;
            }
            // Remap previously committed labels in the live buffer.
            for (Slot& s : buffer_) {
                if (s.ever_committed && s.committed_speaker_id == drop_id) {
                    s.committed_speaker_id = keep_id;
                }
            }
            // Retroactively relabel every committed segment in history.
            for (auto& kv : committed_history_) {
                if (kv.second != drop_id) continue;
                RelabelEvent ev;
                ev.segment_id     = kv.first;
                ev.old_speaker_id = drop_id;
                ev.new_speaker_id = keep_id;
                ev.confidence     = best_sim;
                pending_.push_back(ev);
                kv.second = keep_id;
            }
        }
    }

    // Commit new labels back to the buffer and emit RelabelEvents wherever
    // the global id differs from what was last committed.
    int n_events = 0;
    for (int i = 0; i < N; ++i) {
        Slot& slot     = buffer_[i];
        const int k    = cr.labels[i];
        if (k < 0 || k >= K) continue;
        const int new_id = local_to_global[k];

        // Compare against the most recent authoritative label. The first
        // time we author this segment, treat the tentative online id as the
        // baseline (so if the reclusterer agrees, no event fires).
        const int baseline = slot.ever_committed
                                 ? slot.committed_speaker_id
                                 : slot.seg.tentative_speaker_id;

        if (new_id != baseline) {
            RelabelEvent ev;
            ev.segment_id     = slot.seg.segment_id;
            ev.old_speaker_id = baseline;
            ev.new_speaker_id = new_id;
            ev.confidence     = match_conf[k];
            pending_.push_back(ev);
            n_events += 1;
        }
        slot.committed_speaker_id = new_id;
        slot.ever_committed       = true;
        committed_history_[slot.seg.segment_id] = new_id;
    }

    if (debug_enabled()) {
        std::fprintf(stderr,
            "[recluster-audit]   local_to_global=[");
        for (int k = 0; k < K; ++k) {
            std::fprintf(stderr, "%s%d->%d", k > 0 ? " " : "",
                k, local_to_global[k]);
        }
        std::fprintf(stderr, "]  globals_now=%zu  relabel_events=%d\n",
            globals_.size(), n_events);
    }

    return n_events;
}

void OratorReclusterer::drain_relabels(std::vector<RelabelEvent>& out) {
    out.insert(out.end(), pending_.begin(), pending_.end());
    pending_.clear();
}

} // namespace deusridet::orator
