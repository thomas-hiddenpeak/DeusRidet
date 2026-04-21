/**
 * @file scheduler.h
 * @philosophical_role Declaration of the frame-boundary scheduler interface. Enforces the invariant that only the scheduler advances the frame counter.
 * @serves ConscientiaStream and any subsystem that needs frame phase.
 */
// scheduler.h — Wakefulness-driven consciousness state machine
//
// Manages the consciousness state transitions based on input activity:
//
//   ACTIVE    — External input arriving. Continuous prefill + decode.
//               Prefill runs as fast as input arrives. Decode branches
//               produce speech, action, and thoughts interleaved.
//
//   DAYDREAM  — No external input for idle_threshold_ms.
//               Prefill stops. Decode runs freely: review, organize,
//               analyze what was perceived. Internal reflection.
//
//   DREAMING  — No external input for dream_threshold_ms.
//               Deep consolidation. Multiple decode branches explore
//               associations, strengthen memory, creative wandering.
//
// Any external input immediately transitions back to ACTIVE.
// Transitions are smooth: ACTIVE → DAYDREAM → DREAMING (with decay)
//                          DREAMING/DAYDREAM → ACTIVE (instant on input)

#pragma once

#include <chrono>
#include <atomic>
#include <cstdint>

namespace deusridet {

// ============================================================================
// Consciousness states
// ============================================================================

enum class WakefulnessState : uint8_t {
    ACTIVE   = 0,  // Input arriving → continuous prefill + decode
    DAYDREAM = 1,  // No input briefly → reflect, organize, analyze
    DREAMING = 2,  // No input prolonged → deep consolidation, creative
};

// ============================================================================
// Scheduler configuration
// ============================================================================

struct SchedulerConfig {
    // Time thresholds for state transitions (ms since last external input)
    float idle_threshold_ms   = 5000.0f;   // ACTIVE → DAYDREAM
    float dream_threshold_ms  = 60000.0f;  // DAYDREAM → DREAMING

    // Decode branch counts per state
    int active_decode_branches  = 1;  // speech/action (focused)
    int daydream_decode_branches = 2; // thinking + daydream
    int dream_decode_branches    = 4; // multiple consolidation branches

    // Max tokens per decode step per state
    int active_max_decode   = 128;   // short, responsive
    int daydream_max_decode = 256;   // more freedom to think
    int dream_max_decode    = 512;   // deep exploration

    // Wakefulness level parameters (0.0–1.0)
    float active_wakefulness   = 1.0f;
    float daydream_wakefulness = 0.4f;
    float dream_wakefulness    = 0.1f;

    // Exponential decay: wakefulness halves every half_life_sec seconds
    float wakefulness_half_life_sec = 8.0f;

    // Probe decode budget: number of tokens to let model decide speak/silence
    // Scales with wakefulness: probe_budget = probe_min + (probe_max - probe_min) * w
    float probe_threshold      = 0.3f;   // below this, don't probe (save GPU)
    int   probe_min_tokens     = 3;      // minimum probe budget
    int   probe_max_tokens     = 8;      // maximum probe budget

    // Wakefulness boost on probe silence (EOS output) — small negative
    float probe_silence_decay  = 0.1f;   // subtract from wakefulness on silence
};

// ============================================================================
// Scheduler — consciousness state machine
// ============================================================================

class Scheduler {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit Scheduler(const SchedulerConfig& cfg = {});

    // ── State queries ───────────────────────────────────────────────

    // Current consciousness state.
    WakefulnessState state() const { return state_.load(std::memory_order_relaxed); }

    // Current wakefulness level (0.0–1.0). Decays smoothly between states.
    float wakefulness() const { return wakefulness_.load(std::memory_order_relaxed); }

    // Time since last external input (ms).
    float idle_time_ms() const;

    // ── Input events ────────────────────────────────────────────────

    // Signal that external input was received. Immediately transitions to ACTIVE.
    void on_external_input();

    // Boost wakefulness by a delta (clamped to [0, 1]).
    // Use for attention events: name detected (+0.5), background ASR (+0.15), etc.
    void boost_wakefulness(float delta);

    // Signal that a probe decode produced silence (EOS).
    // Slightly decreases wakefulness (entity chose not to speak).
    void on_probe_silence();

    // Signal that a probe decode produced a response.
    // Sets wakefulness to 1.0 (entity decided to speak).
    void on_probe_response();

    // ── Per-tick update ─────────────────────────────────────────────

    // Call once per consciousness cycle. Updates state based on idle time.
    // Returns the current state after update.
    WakefulnessState tick();

    // ── Budget queries (per current state) ──────────────────────────

    // Recommended number of decode branches.
    int recommended_decode_branches() const;

    // Max decode tokens for current state.
    int max_decode_tokens() const;

    // Whether wakefulness is high enough to justify a probe decode.
    bool should_probe() const;

    // Probe decode token budget (scales with wakefulness).
    int probe_budget() const;

    // Whether prefill should run (true only in ACTIVE with pending input).
    bool should_prefill() const;

    // Config access
    const SchedulerConfig& config() const { return cfg_; }

private:
    SchedulerConfig cfg_;
    std::atomic<WakefulnessState> state_{WakefulnessState::ACTIVE};
    std::atomic<float> wakefulness_{1.0f};
    TimePoint last_external_input_;
    TimePoint last_tick_;
};

} // namespace deusridet
