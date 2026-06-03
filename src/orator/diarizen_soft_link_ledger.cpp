/**
 * @file diarizen_soft_link_ledger.cpp
 * @philosophical_role
 *   Implements a shadow-only confidence ledger for tentative->stable speaker
 *   ID association. The ledger adds hysteresis so identity links do not flip
 *   on one noisy relabel event.
 * @serves
 *   Runtime observability of ID continuity while keeping current online and
 *   periodic decisions untouched.
 */

#include "orator/diarizen_soft_link_ledger.h"

namespace deusridet {
namespace orator {

DiarizenSoftLinkLedger::Snapshot DiarizenSoftLinkLedger::observe_relabel(
    int live_id, int stable_id, float confidence) {
    Snapshot out;
    out.live_id = live_id;
    if (live_id < 0 || stable_id < 0) return out;

    std::lock_guard<std::mutex> lk(mu_);
    auto& edge = edges_[live_id][stable_id];
    edge.score = (edge.support == 0)
        ? confidence
        : ((1.0f - kEmaAlpha) * edge.score + kEmaAlpha * confidence);
    edge.support += 1;

    int best_sid = -1;
    float best_score = -1.0f;
    int best_support = 0;
    float second_score = -1.0f;
    for (const auto& kv : edges_[live_id]) {
        if (kv.second.score > best_score) {
            second_score = best_score;
            best_score = kv.second.score;
            best_support = kv.second.support;
            best_sid = kv.first;
        } else if (kv.second.score > second_score) {
            second_score = kv.second.score;
        }
    }

    if (best_sid >= 0) {
        const float margin = (second_score >= 0.0f) ? (best_score - second_score) : best_score;
        const bool strong_candidate =
            best_support >= kPromoteMinSupport &&
            best_score >= kPromoteMinScore &&
            margin >= kPromoteMinMargin;

        auto it_commit = committed_.find(live_id);
        if (it_commit == committed_.end()) {
            if (strong_candidate) {
                committed_[live_id] = best_sid;
                out.changed = true;
            }
        } else if (it_commit->second != best_sid) {
            const auto it_old = edges_[live_id].find(it_commit->second);
            const float old_score = (it_old == edges_[live_id].end()) ? 0.0f : it_old->second.score;
            if (strong_candidate && best_score >= old_score + kSwitchMinLead) {
                committed_[live_id] = best_sid;
                out.changed = true;
            }
        }
    }

    const bool changed = out.changed;
    out = snapshot_for_live_(live_id);
    out.changed = changed;
    return out;
}

DiarizenSoftLinkLedger::Snapshot DiarizenSoftLinkLedger::observe_online(
    int live_id, float similarity) {
    Snapshot out;
    out.live_id = live_id;
    if (live_id < 0) return out;

    std::lock_guard<std::mutex> lk(mu_);
    (void)similarity;
    return snapshot_for_live_(live_id);
}

DiarizenSoftLinkLedger::Snapshot DiarizenSoftLinkLedger::snapshot_for_live_(int live_id) const {
    Snapshot out;
    out.live_id = live_id;
    if (live_id < 0) return out;

    auto it_edges = edges_.find(live_id);
    if (it_edges == edges_.end() || it_edges->second.empty()) {
        auto it_commit = committed_.find(live_id);
        if (it_commit != committed_.end()) {
            out.stable_id = it_commit->second;
            out.committed = true;
        }
        return out;
    }

    int best_sid = -1;
    float best_score = -1.0f;
    int best_support = 0;
    float second_score = -1.0f;
    for (const auto& kv : it_edges->second) {
        if (kv.second.score > best_score) {
            second_score = best_score;
            best_score = kv.second.score;
            best_support = kv.second.support;
            best_sid = kv.first;
        } else if (kv.second.score > second_score) {
            second_score = kv.second.score;
        }
    }

    auto it_commit = committed_.find(live_id);
    if (it_commit != committed_.end()) {
        out.stable_id = it_commit->second;
        out.committed = true;
        auto it_comm_edge = it_edges->second.find(out.stable_id);
        if (it_comm_edge != it_edges->second.end()) {
            out.score = it_comm_edge->second.score;
            out.support = it_comm_edge->second.support;
        }
        out.margin = best_score - out.score;
        return out;
    }

    out.stable_id = best_sid;
    out.score = (best_score > 0.0f) ? best_score : 0.0f;
    out.support = best_support;
    out.margin = (second_score >= 0.0f) ? (best_score - second_score) : out.score;
    out.committed = false;
    return out;
}

}  // namespace orator
}  // namespace deusridet
