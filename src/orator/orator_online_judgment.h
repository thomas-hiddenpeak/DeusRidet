/**
 * @file orator_online_judgment.h
 * @philosophical_role The seam that splits ONE entangled online-speaker
 *     decision into the THREE distinct engineering problems it always was:
 *     ① WHEN to register (novelty / evidence), ② HOW to identify
 *     (closed-set judgment, may ABSTAIN), ③ HOW to distinguish (store
 *     hygiene, deferred to the offline finalize layer). This header owns
 *     concern ② as a first-class, pure, testable value — a judgment the
 *     online path is allowed to be HUMBLE about, because the offline
 *     reclusterer + transcript holdback correct it retroactively.
 * @serves Orator (online identity), Sensus/Auditus (SAAS full-extract),
 *     Nexus (speaker broadcast), Conscientia (LLM-injected speaker label).
 *
 * Design note (Jun 2, 2026, branch redesign/orator-online-three-concerns):
 * The legacy `process_saas_full_extract_()` collapsed all three concerns
 * into one greedy pass sharing single knobs (speaker_threshold doubling as
 * both match gate AND pending-confirm gate; margin_abstain doubling as both
 * ambiguity reject AND cross-speaker absorb guard; recency_bonus driving
 * match + register-disable + absorb-guard at once). Tuning one broke the
 * others. This file introduces the explicit ② boundary WITHOUT changing
 * behaviour: judge_identity() encodes the exact same match / margin / abstain
 * rule the monster already applies, but as one named function returning a
 * value that can carry an honest ABSTAIN/UNKNOWN verdict.
 */
#pragma once

#include "orator/speaker_db.h"  // SpeakerMatch

namespace deusridet::orator {

// ── Concern ② : HOW to identify (closed-set, may abstain) ──────────────────
//
// The online path classifies a completed-segment embedding against the
// already-registered identities. Unlike the legacy code, the verdict is a
// first-class value with an explicit ABSTAIN state: the online path is
// permitted to say "I am not sure" and let the offline finalize + holdback
// fill in the truth later.

enum class IdentityDecision {
    Matched,    // confident closed-set hit on an existing identity
    Abstained,  // a plausible top-1 exists but the margin is too thin to trust
    Unknown,    // nothing in the store is similar enough (top-1 below floor)
};

struct IdentityJudgment {
    IdentityDecision decision = IdentityDecision::Unknown;
    int   speaker_id   = -1;     // valid only when decision == Matched
    float confidence   = 0.0f;   // top-1 cosine similarity
    float margin       = 0.0f;   // top-1 minus top-2 (separation from runner-up)
    int   second_id    = -1;     // runner-up identity (for diagnostics)
    float second_sim   = 0.0f;   // runner-up similarity

    bool matched()   const { return decision == IdentityDecision::Matched; }
    bool abstained() const { return decision == IdentityDecision::Abstained; }
    bool unknown()   const { return decision == IdentityDecision::Unknown; }
};

// Thresholds that belong to concern ② ONLY. Deliberately separated from the
// registration (①) and hygiene (③) knobs so that tuning identification can
// no longer silently alter when a new speaker is born or how centroids drift.
struct JudgeThresholds {
    float match_floor    = 0.45f;  // top-1 must clear this to be a Matched hit
    float margin_abstain = 0.05f;  // (top1 - top2) below this ⇒ Abstained
};

// Pure, side-effect-free mapping from a raw store query (SpeakerMatch) to an
// explicit ② judgment. Encodes the SAME logic the legacy full-extract applied
// inline (floor check, then margin check), so wiring this in is behaviour-
// preserving until concern ② is deliberately allowed to abstain at call sites.
inline IdentityJudgment judge_identity(const SpeakerMatch& m,
                                       const JudgeThresholds& th) {
    IdentityJudgment j;
    j.confidence = m.similarity;
    j.second_id  = m.second_best_id;
    j.second_sim = m.second_best_sim;
    j.margin     = m.similarity - m.second_best_sim;

    if (m.similarity < th.match_floor) {
        j.decision = IdentityDecision::Unknown;
        return j;
    }
    if (m.second_best_id >= 0 && j.margin < th.margin_abstain) {
        j.decision = IdentityDecision::Abstained;
        j.second_id = m.second_best_id;
        return j;
    }
    j.decision    = IdentityDecision::Matched;
    j.speaker_id  = m.speaker_id;
    return j;
}

}  // namespace deusridet::orator
