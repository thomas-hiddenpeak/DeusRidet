/**
 * @file orator_online.h
 * @philosophical_role The clean online speaker facade that replaces the
 *     1080-line entangled `process_saas_full_extract_()` monster. It states
 *     the three engineering problems as three distinct, decoupled units and
 *     nothing else:
 *       ① WHEN to register  — evidence-gated novelty (RegistrationGate),
 *       ② HOW to identify    — read-only closed-set judgment that may ABSTAIN
 *                              (judge_identity, orator_online_judgment.h),
 *       ③ HOW to distinguish — minimal online store hygiene; GLOBAL
 *                              separation is deferred to the offline finalize.
 *     Everything the old path bolted on to force a confident label in one
 *     greedy forward pass — discovery phase, recency bonus, multi-gate probe,
 *     SHORT-IDENTIFY, SI-peek veto/rescue, retro-ring, inherit-broadcast — is
 *     deleted. The online path is now HUMBLE: it broadcasts only what it can
 *     defend, abstains otherwise, and lets the proven DiariZen-v2 finalize +
 *     transcript holdback rewrite the truth retroactively.
 * @serves Orator (online identity), Sensus/Auditus (SAAS full-extract),
 *     Nexus (speaker broadcast), Conscientia (LLM speaker label, via holdback).
 *
 * Design note (Jun 2, 2026, branch redesign/orator-online-three-concerns):
 * Clean-slate rebuild requested by the owner. The old code is recoverable from
 * git history (checkpoint 8ffefbc). No knob in this facade is reused across
 * concerns: identifying never moves the registration gate, registering never
 * loosens the identify floor, and neither drifts a centroid by EMA on a
 * low-similarity query (the historical silent cross-speaker absorption hazard).
 */
#pragma once

#include "orator/orator_online_judgment.h"  // ② judge_identity, IdentityJudgment
#include "orator/speaker_vector_store.h"     // the GPU identity store

#include <string>
#include <utility>
#include <vector>

namespace deusridet::orator {

// ── Concern ① : WHEN to register — evidence-gated novelty ──────────────────
//
// Registration is a discrete event driven by ACCUMULATED EVIDENCE, never by a
// single cosine compare. An embedding that concern ② judged Unknown is offered
// to the gate; the gate coalesces repeat sightings of the same novel voice and
// confirms a brand-new identity only after enough mutually-consistent hits.
// Its thresholds are its own — deliberately NOT the identify match floor — so
// that tuning identification can no longer silently change when a speaker is
// born (the failure mode that lost 徐子景 to 朱杰's cluster).
struct RegistrationConfig {
    float coalesce_sim = 0.60f;  // two pending embeddings within this ⇒ same novel voice
    int   confirm_hits = 2;      // coalescing hits required before a new id is born
    double ttl_sec     = 30.0;   // pending evidence older than this expires
};

class RegistrationGate {
public:
    // Offer an embedding that concern ② judged Unknown. Returns true and fills
    // `out_emb` with the averaged, confirmed centroid when evidence is enough
    // to register a NEW speaker; returns false while still accumulating.
    bool offer(const std::vector<float>& emb, double now_sec,
               const RegistrationConfig& cfg, std::vector<float>& out_emb);

    void reset() { pending_.clear(); }
    int  pending_count() const { return static_cast<int>(pending_.size()); }

private:
    struct Pending {
        std::vector<float> emb;   // running-averaged, L2-normalised centroid
        int    hits = 0;
        double last_sec = 0.0;
    };
    std::vector<Pending> pending_;
};

// ── Concern ③ : HOW to distinguish — online store hygiene only ─────────────
//
// The online half of distinction is conservative: a confidently-matched,
// well-separated segment MAY contribute a fresh exemplar (capturing genuine
// voice variation), but a low-similarity or ambiguous segment NEVER touches
// the store. There is no EMA drift. Global cluster separation — splitting a
// polluted cluster, recovering a cold-start tail, merging fragments — is the
// offline finalize's job, where the whole session is visible at once.
struct HygieneConfig {
    bool  exemplar_admit = true;   // confidently-matched segments may add an exemplar
    float admit_floor    = 0.60f;  // ... only if confidence ≥ this (no low-sim drift)
    float admit_margin   = 0.10f;  // ... and margin ≥ this (no ambiguous absorb)
};

// ── The unified online decision the broadcast layer consumes ───────────────
enum class OnlineAction {
    Broadcast,  // confident closed-set hit on an existing identity
    Register,   // a NEW identity was just born (concern ① confirmed)
    Abstain,    // humble: not sure — let the offline finalize fill it in
};

struct OnlineDecision {
    OnlineAction action     = OnlineAction::Abstain;
    int          speaker_id = -1;     // valid for Broadcast / Register
    float        confidence = 0.0f;   // top-1 similarity (1.0 for a fresh Register)
    bool         is_new     = false;  // true only for Register
    std::string  name;                // matched speaker name (empty for new/abstain)
};

struct OratorOnlineConfig {
    JudgeThresholds    judge;    // ②
    RegistrationConfig reg;      // ①
    HygieneConfig      hygiene;  // ③
};

// The clean online facade. Owns the concern-① gate state. It is READ-ONLY on
// the store for concern ② (peek_best), and mutates the store ONLY on a
// confirmed ① registration or a gated ③ exemplar admission.
class OratorOnline {
public:
    OratorOnline() = default;
    explicit OratorOnline(OratorOnlineConfig cfg) : cfg_(std::move(cfg)) {}

    // The single online entry: classify a completed-segment embedding against
    // `store` at wall/audio time `now_sec`, and return the decision to emit.
    OnlineDecision decide(SpeakerVectorStore& store,
                          const std::vector<float>& emb,
                          double now_sec);

    const OratorOnlineConfig& config() const { return cfg_; }
    void set_config(OratorOnlineConfig cfg) { cfg_ = std::move(cfg); }
    void reset() { gate_.reset(); }

private:
    OratorOnlineConfig cfg_{};
    RegistrationGate   gate_{};
};

}  // namespace deusridet::orator
