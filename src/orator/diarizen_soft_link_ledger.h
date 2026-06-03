/**
 * @file diarizen_soft_link_ledger.h
 * @philosophical_role
 *   Shadow bridge between online tentative speaker IDs and periodic stable
 *   identities. Maintains a soft association ledger with hysteresis so we can
 *   observe convergence quality without altering live decision outputs.
 * @serves
 *   auditus facade broadcast path and WebUI observability for ID continuity.
 */

#ifndef DEUSRIDET_ORATOR_DIARIZEN_SOFT_LINK_LEDGER_H
#define DEUSRIDET_ORATOR_DIARIZEN_SOFT_LINK_LEDGER_H

#include <unordered_map>
#include <mutex>

namespace deusridet {
namespace orator {

class DiarizenSoftLinkLedger {
public:
    struct Snapshot {
        int live_id = -1;
        int stable_id = -1;
        float score = 0.0f;
        float margin = 0.0f;
        int support = 0;
        bool committed = false;
        bool changed = false;
    };

    // Observe a periodic relabel edge old_id -> new_id and update the
    // soft association with hysteresis promotion rules.
    Snapshot observe_relabel(int live_id, int stable_id, float confidence);

    // Observe a live online decision; returns current best/committed link.
    Snapshot observe_online(int live_id, float similarity);

private:
    struct Edge {
        float score = 0.0f;
        int support = 0;
    };

    static constexpr float kEmaAlpha = 0.25f;
    static constexpr float kPromoteMinScore = 0.62f;
    static constexpr float kPromoteMinMargin = 0.08f;
    static constexpr int kPromoteMinSupport = 2;
    static constexpr float kSwitchMinLead = 0.10f;

    Snapshot snapshot_for_live_(int live_id) const;

    mutable std::mutex mu_;
    std::unordered_map<int, std::unordered_map<int, Edge>> edges_;
    std::unordered_map<int, int> committed_;
};

}  // namespace orator
}  // namespace deusridet

#endif  // DEUSRIDET_ORATOR_DIARIZEN_SOFT_LINK_LEDGER_H
