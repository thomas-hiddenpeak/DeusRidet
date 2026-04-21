// tempus.h — Unified temporal architecture (Tempus Reale)
//
// DeusRidet operates on a THREE-TIER clock hierarchy. Every event, every
// memory record, every cross-modal alignment MUST ultimately be expressed
// in T0 (real wall time). This is not merely systems engineering — it is
// the physical substrate of subjective continuity. A conscious entity
// without a unified real-time anchor cannot reason about "before" and
// "after" across modalities or across sleep cycles.
//
//   T0  Real time        — steady_clock nanoseconds, monotonic, system-wide
//                          unique. The ground truth of "when".
//   T1  Business time    — per-subsystem clock (audio sample index,
//                          video frame index, consciousness frame id,
//                          TTS codec step, dream cycle, ...).
//                          Each registers an anchor {t0_ns, t1=0} at its
//                          birth moment, then advances by arithmetic.
//   T2  Module time      — per-module counter (VAD window, OD segmenter
//                          frame, speaker embedding index, ASR token
//                          position, KV block id, ...). Always reducible
//                          to T1 via a known period, hence to T0.
//
// Cross-domain alignment ALWAYS goes through T0. Never convert
// audio_sample_index directly to video frame_index. Convert both to T0
// and compare there. This keeps new modalities zero-cost to integrate.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace deusridet {
namespace tempus {

// ----- T0: real time -------------------------------------------------------

// Monotonic nanoseconds since an unspecified epoch (process start in
// practice). This is THE system clock. Use this for every event timestamp
// that will be compared, stored, or exchanged across subsystems.
inline uint64_t now_t0_ns() noexcept {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now().time_since_epoch()).count());
}

// ----- Business domains ----------------------------------------------------
//
// Add new domains at the end; never renumber — values may be persisted in
// Memoria Longa records.
enum class Domain : uint8_t {
    AUDIO         = 0,  // sample_index @ 16 kHz by default
    VIDEO         = 1,  // frame_index  @ camera fps
    CONSCIOUSNESS = 2,  // Prefill frame_id
    TTS           = 3,  // codec step   @ 12 Hz
    DREAM         = 4,  // dream cycle (advances only in low-wakefulness)
    TOOL          = 5,  // MCP / tool invocation sequence
    COUNT         = 6
};

// ----- Clock anchor --------------------------------------------------------
//
// A domain "begins" by calling anchor_register(domain, t1_zero, period_ns).
// Thereafter t0_ns = t0_anchor + (t1 - t1_zero) * period_ns, with no
// further syscalls. Anchors are set-once, lock-free-readable. Re-registering
// a domain (e.g. after a pipeline restart) atomically replaces the anchor.
struct ClockAnchor {
    uint64_t t0_anchor_ns = 0;   // T0 at the moment t1 == t1_zero
    uint64_t t1_zero      = 0;   // the T1 value the anchor was taken at
    uint64_t period_ns    = 0;   // nanoseconds per unit of T1 (0 = unset)
};

namespace detail {
// One anchor slot per domain. Updated rarely, read frequently. The trick:
// we pack {t1_zero, period_ns} into an atomic<uint64_t*>-indirected struct
// only when absolutely necessary; for now per-field atomic loads/stores
// are sufficient because anchor changes are exceptional events (pipeline
// start / dream transition / process restart) and downstream readers
// tolerate a one-sample racey read.
struct AnchorSlot {
    std::atomic<uint64_t> t0_anchor_ns{0};
    std::atomic<uint64_t> t1_zero{0};
    std::atomic<uint64_t> period_ns{0};
};
inline std::array<AnchorSlot, static_cast<size_t>(Domain::COUNT)>& anchors() {
    static std::array<AnchorSlot, static_cast<size_t>(Domain::COUNT)> s;
    return s;
}
}  // namespace detail

// Register (or replace) the anchor for a domain.
//   t0_anchor_ns : T0 at the moment of anchoring. Pass now_t0_ns() unless
//                  you are reconstructing a historical anchor for replay.
//   t1_zero      : the T1 value the clock holds at this anchor moment.
//                  Usually 0 at pipeline start.
//   period_ns    : nanoseconds per T1 unit. Examples:
//                    AUDIO 16 kHz   → 62500
//                    AUDIO 16 kHz speed=2.0 replay → 31250
//                        (the business clock advances at source rate but
//                         T0 advances half as much per sample, because
//                         wall time passes twice as fast per source sec)
//                    VIDEO 30 fps   → 33'333'333
//                    CONSCIOUSNESS  → typical 100 ms frame = 100'000'000
//                    TTS 12 Hz codec→ 83'333'333
inline void anchor_register(Domain d,
                            uint64_t t0_anchor_ns,
                            uint64_t t1_zero,
                            uint64_t period_ns) noexcept {
    auto& slot = detail::anchors()[static_cast<size_t>(d)];
    slot.t0_anchor_ns.store(t0_anchor_ns, std::memory_order_release);
    slot.t1_zero.store(t1_zero,           std::memory_order_release);
    slot.period_ns.store(period_ns,       std::memory_order_release);
}

// Load the current anchor for inspection / serialization.
inline ClockAnchor anchor_of(Domain d) noexcept {
    const auto& slot = detail::anchors()[static_cast<size_t>(d)];
    ClockAnchor a;
    a.t0_anchor_ns = slot.t0_anchor_ns.load(std::memory_order_acquire);
    a.t1_zero      = slot.t1_zero.load(std::memory_order_acquire);
    a.period_ns    = slot.period_ns.load(std::memory_order_acquire);
    return a;
}

// ----- T1 <-> T0 conversion ------------------------------------------------
//
// These are arithmetic-only; safe to call in hot paths (no syscalls, no
// locks). Returns 0 if the domain's anchor has not been registered yet —
// callers should check anchor_of(d).period_ns != 0 during initialization
// if that matters.

inline uint64_t t1_to_t0(Domain d, uint64_t t1) noexcept {
    const ClockAnchor a = anchor_of(d);
    if (a.period_ns == 0) return 0;
    // Signed delta to support clocks where current t1 < t1_zero (should be
    // rare but not impossible during replay / checkpoint restore).
    const int64_t delta = static_cast<int64_t>(t1) - static_cast<int64_t>(a.t1_zero);
    return a.t0_anchor_ns + static_cast<uint64_t>(delta * static_cast<int64_t>(a.period_ns));
}

inline uint64_t t0_to_t1(Domain d, uint64_t t0_ns) noexcept {
    const ClockAnchor a = anchor_of(d);
    if (a.period_ns == 0) return 0;
    const int64_t delta_ns = static_cast<int64_t>(t0_ns) - static_cast<int64_t>(a.t0_anchor_ns);
    return a.t1_zero + static_cast<uint64_t>(delta_ns / static_cast<int64_t>(a.period_ns));
}

// ----- TimeStamp: the canonical event time triple -------------------------
//
// Every event emitted on the internal event bus / timeline carries this.
// t0_ns is authoritative. t1_business / t2_module are retained for
// intra-domain reasoning (sample-accurate ASR readback, KV block indexing)
// and for debugging. If a consumer does not care which domain produced
// the event, it looks only at t0_ns.
struct TimeStamp {
    uint64_t t0_ns        = 0;
    uint64_t t1_business  = 0;
    uint64_t t2_module    = 0;
    Domain   domain       = Domain::AUDIO;
    uint8_t  _pad[7]      = {0, 0, 0, 0, 0, 0, 0};
};
static_assert(sizeof(TimeStamp) == 32, "TimeStamp layout changed");

// Convenience constructor: stamp an event at the current T0, given a
// business T1 value (and optional module T2). Anchor must already exist.
inline TimeStamp stamp_from_t1(Domain d, uint64_t t1, uint64_t t2 = 0) noexcept {
    TimeStamp ts;
    ts.domain      = d;
    ts.t1_business = t1;
    ts.t2_module   = t2;
    ts.t0_ns       = t1_to_t0(d, t1);
    return ts;
}

// Convenience constructor: stamp NOW in T0 and derive the business T1
// after the fact. Prefer stamp_from_t1() when the producer already owns
// an authoritative T1 counter (audio sample index, frame id, ...).
inline TimeStamp stamp_now(Domain d, uint64_t t2 = 0) noexcept {
    TimeStamp ts;
    ts.domain      = d;
    ts.t0_ns       = now_t0_ns();
    ts.t1_business = t0_to_t1(d, ts.t0_ns);
    ts.t2_module   = t2;
    return ts;
}

}  // namespace tempus
}  // namespace deusridet
