/**
 * @file vires_types.h
 * @philosophical_role The vocabulary of metabolic substrate. Names the scarce
 *         GPU *compute* resource and the priority classes through which it is
 *         delivered. Pure data — no policy, no CUDA, no consumer knowledge.
 * @serves Vires (the arterial compute substrate) and every consumer that
 *         declares its metabolic class to it.
 */
// vires_types.h — Compute-substrate vocabulary (Priority, Consumer, Snapshot).
//
// Vires governs GPU *compute* allocation only. GPU memory as a whole belongs
// to Memoria; nothing here models memory capacity. See
// docs/{en,zh}/architecture/13-vires.md.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace deusridet {
namespace vires {

// ── Metabolic priority class ──────────────────────────────────────────────────
// Two classes in V1. Foreground is the live perception → prefill → decode path
// that must never be starved; Background is refinement that yields to it
// (native DiariZen, Somnium consolidation).
enum class Priority : uint8_t {
    Foreground = 0,  // highest urgency — perception, prefill, decode
    Background = 1,  // lowest urgency  — refinement, consolidation
};

inline const char* priority_str(Priority p) {
    switch (p) {
        case Priority::Foreground: return "foreground";
        case Priority::Background: return "background";
    }
    return "?";
}

// ── Consumer identity ─────────────────────────────────────────────────────────
using ConsumerId = uint32_t;
constexpr ConsumerId kInvalidConsumer = 0;  // 0 is never a valid handle

// ── Telemetry ─────────────────────────────────────────────────────────────────
// One row per registered GPU consumer. This is the single observable
// "who is computing what" surface (observability rule). Compute-only — there
// is deliberately no memory field here.
struct ConsumerStat {
    ConsumerId  id        = kInvalidConsumer;
    std::string name;
    Priority    priority  = Priority::Background;
    uint64_t    submitted = 0;  // monotonic count of GPU passes submitted
    uint64_t    reclaimed = 0;  // monotonic count of glymphatic clearances (V3)
};

// A point-in-time view of the compute ledger.
struct Snapshot {
    // CUDA stream priority range queried from the device. On the Orin the
    // numerically *smaller* value is the *greater* priority (CUDA convention):
    // greatest_priority <= least_priority.
    int greatest_priority = 0;
    int least_priority    = 0;
    std::vector<ConsumerStat> consumers;

    // V2 back-pressure observability. `background_yielding` is true when a
    // foreground submission landed within the recent activity window, i.e.
    // background consumers should currently pause/chunk to let foreground
    // through. `foreground_idle_us` is the time since the last foreground
    // submission (UINT64_MAX if none ever observed).
    bool     background_yielding   = false;
    uint64_t foreground_idle_us    = UINT64_MAX;
};

} // namespace vires
} // namespace deusridet
