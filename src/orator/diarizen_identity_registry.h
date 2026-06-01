// @philosophical_role Orator — the census of persistent speaker identity.
//   A windowed DiariZen pass returns pipeline-local labels ("speaker0"…)
//   with no memory of earlier windows: the "speaker0" of one window and the
//   "speaker0" of the next are unrelated. This registry binds each new
//   window's local labels onto durable global identities ("S0","S1"…) by
//   time-overlap against each identity's recent occupancy, minting a fresh
//   identity only for a local label that overlaps no one.
//
//   It is the seed that will retire the greedy online speaker store: identity
//   here is decided by joint windowed clustering (DiariZen) and bound across
//   time by overlap, never by an irreversible per-segment nearest-centroid
//   commit. The sliding window's deliberate overlap (period < window) is what
//   makes the binding work — consecutive windows share audio, so the same
//   speaker is diarised in both and their intervals overlap.
// @serves DiarizenPeriodicWorker (stable labels for the live broadcast and,
//   later, for the LLM relabel path that today lives in TranscriptHoldback).
#ifndef DEUSRIDET_ORATOR_DIARIZEN_IDENTITY_REGISTRY_H
#define DEUSRIDET_ORATOR_DIARIZEN_IDENTITY_REGISTRY_H

#include "diarizen_pipeline.h"  // DiarizenSegment

#include <deque>
#include <unordered_map>
#include <vector>

namespace deusridet {
namespace orator {

// Cross-window persistent-identity registry. Not thread-safe by itself; the
// DiarizenPeriodicWorker owns exactly one and only ever stitches from its own
// single pass thread (passes are serialised by DiarizenPipeline::pass_mutex).
class DiarizenIdentityRegistry {
   public:
    struct Config {
        // Minimum cosine similarity between a local label's voiceprint and an
        // existing identity's EMA prototype to bind them. Below this the label
        // either falls back to time-overlap (if it has no voiceprint) or mints
        // a fresh identity. 0.50 is the ResNet34-256d same-speaker floor.
        double match_min_cos = 0.50;
        // EMA blend for updating a bound identity's prototype voiceprint.
        double ema_alpha = 0.30;
        // Minimum overlap (seconds) a *voiceprint-less* local label must share
        // with an existing identity's recent occupancy before binding to it.
        double match_min_overlap_sec = 0.5;
        // Occupancy older than (latest window end − this horizon) is pruned.
        // Must exceed the sliding-window length so the overlap zone between
        // adjacent windows is always retained; 90 s is generous for the
        // ≤30 s windows used live while keeping history bounded.
        double history_horizon_sec = 90.0;
    };

    DiarizenIdentityRegistry() = default;
    explicit DiarizenIdentityRegistry(Config cfg) : cfg_(cfg) {}

    // @role Rewrite the labels of `segs` in place to stable global identity
    //   strings ("S0","S1"…), binding each window-local label to the global
    //   identity whose EMA prototype voiceprint it most resembles (greedy
    //   one-to-one cosine so two simultaneous speakers never collapse), with a
    //   time-overlap fallback for labels that carry no usable embedding.
    //   `centroids[k]` is the L2-normalised 256-d voiceprint of pipeline
    //   cluster k (label "speaker<k>"), valid only for this pass. `origin_sec`
    //   is the absolute stream-time of the window start: seg start/end are
    //   capture-relative and left unchanged; only labels are rewritten.
    //   Updates prototypes + occupancy and prunes history past the horizon.
    //   Returns the number of distinct global identities touched by this pass.
    size_t stitch(std::vector<DiarizenSegment>& segs, double origin_sec,
                  const std::vector<std::vector<float>>& centroids);

    // Total number of distinct global identities ever minted.
    int num_identities() const { return next_gid_; }

   private:
    struct Occ {
        double start_sec = 0.0;  // absolute stream-time
        double end_sec = 0.0;
    };

    Config cfg_;
    int next_gid_ = 0;
    // Per global id → EMA prototype voiceprint (unit-length, 256-d).
    std::unordered_map<int, std::vector<float>> proto_;
    // Per global id → recent absolute-time intervals (pruned to horizon),
    // used only by the voiceprint-less time-overlap fallback.
    std::unordered_map<int, std::deque<Occ>> hist_;
};

}  // namespace orator
}  // namespace deusridet

#endif  // DEUSRIDET_ORATOR_DIARIZEN_IDENTITY_REGISTRY_H
