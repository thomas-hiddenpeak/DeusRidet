/**
 * @file orator_online.cpp
 * @philosophical_role Implementation of the clean three-concern online speaker
 *     facade. See orator_online.h for the philosophy. The hot path is
 *     deliberately short and read-only on the store; the only writes are a
 *     confirmed ① registration and a gated ③ exemplar admission.
 * @serves Orator online identity.
 */
#include "orator/orator_online.h"

#include <algorithm>
#include <cmath>

namespace deusridet::orator {

namespace {

// CPU cosine over a handful of pending entries (N < ~5, dim 384). One-shot,
// tiny N, host-side orchestration — correctly on the CPU per the GPU-first
// rule (the per-segment GPU search already happened in store.peek_best()).
float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na <= 0.0f || nb <= 0.0f) return 0.0f;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void l2_normalise(std::vector<float>& v) {
    float n = 0.0f;
    for (float x : v) n += x * x;
    n = std::sqrt(n);
    if (n <= 0.0f) return;
    for (float& x : v) x /= n;
}

}  // namespace

// ── Concern ① : evidence-gated novelty ─────────────────────────────────────
bool RegistrationGate::offer(const std::vector<float>& emb, double now_sec,
                             const RegistrationConfig& cfg,
                             std::vector<float>& out_emb) {
    // Expire stale pending evidence: a novel voice that was heard once and
    // never recurred is not a speaker, it is noise / cross-talk.
    pending_.erase(
        std::remove_if(pending_.begin(), pending_.end(),
                       [&](const Pending& p) {
                           return now_sec - p.last_sec > cfg.ttl_sec;
                       }),
        pending_.end());

    // Find the pending bucket this embedding coalesces into (same novel voice).
    int   best     = -1;
    float best_sim = cfg.coalesce_sim;
    for (size_t i = 0; i < pending_.size(); ++i) {
        float s = cosine(emb, pending_[i].emb);
        if (s >= best_sim) {
            best_sim = s;
            best     = static_cast<int>(i);
        }
    }

    if (best < 0) {
        // First sighting of a new novel voice — start accumulating evidence.
        pending_.push_back({emb, 1, now_sec});
        return false;
    }

    // Coalesce: running-average the centroid (NOT EMA on the live store — this
    // is private pending evidence) and bump the hit count.
    Pending& p = pending_[static_cast<size_t>(best)];
    const float w = static_cast<float>(p.hits);
    for (size_t k = 0; k < p.emb.size(); ++k) {
        p.emb[k] = (p.emb[k] * w + emb[k]) / (w + 1.0f);
    }
    l2_normalise(p.emb);
    p.hits++;
    p.last_sec = now_sec;

    if (p.hits >= cfg.confirm_hits) {
        out_emb = p.emb;
        pending_.erase(pending_.begin() + best);
        return true;
    }
    return false;
}

// ── The unified online decision ────────────────────────────────────────────
OnlineDecision OratorOnline::decide(SpeakerVectorStore& store,
                                    const std::vector<float>& emb,
                                    double now_sec) {
    OnlineDecision d;

    // ② HOW to identify — read-only closed-set judgment (may abstain).
    SpeakerMatch     raw = store.peek_best(emb);
    IdentityJudgment j   = judge_identity(raw, cfg_.judge);
    d.confidence = j.confidence;

    if (j.matched()) {
        d.action     = OnlineAction::Broadcast;
        d.speaker_id = j.speaker_id;
        d.name       = raw.name;

        // ③ HOW to distinguish — gated exemplar admission, no EMA drift.
        if (cfg_.hygiene.exemplar_admit &&
            j.confidence >= cfg_.hygiene.admit_floor &&
            j.margin     >= cfg_.hygiene.admit_margin) {
            store.add_exemplar(j.speaker_id, emb);
        }
        return d;
    }

    // ① WHEN to register — only a genuinely novel (Unknown) embedding is a
    // new-speaker candidate. An Abstained verdict means "ambiguous between two
    // KNOWN speakers", which is not novelty; we stay humble and let the offline
    // finalize resolve it rather than minting a phantom cluster.
    if (j.unknown()) {
        std::vector<float> confirmed;
        if (gate_.offer(emb, now_sec, cfg_.reg, confirmed)) {
            int id = store.register_speaker("", confirmed);
            if (id >= 0) {
                d.action     = OnlineAction::Register;
                d.speaker_id = id;
                d.is_new     = true;
                d.confidence = 1.0f;
                return d;
            }
        }
    }

    // Still accumulating evidence, or ambiguous — the humble default.
    d.action = OnlineAction::Abstain;
    return d;
}

}  // namespace deusridet::orator
