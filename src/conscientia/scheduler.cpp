/**
 * @file scheduler.cpp
 * @philosophical_role Frame-boundary scheduler. Decides when the next frame starts, when a frame drops, and when idle becomes dreaming. The scheduler is NOT a request dispatcher — it is the entity's heartbeat.
 * @serves ConscientiaStream::run(), Vigilia, Somnium.
 */
// scheduler.cpp — Wakefulness-driven consciousness state machine

#include "scheduler.h"
#include <algorithm>
#include <cmath>

namespace deusridet {

using Ms = std::chrono::duration<float, std::milli>;

Scheduler::Scheduler(const SchedulerConfig& cfg) : cfg_(cfg) {
    last_external_input_ = Clock::now();
    last_tick_ = Clock::now();
}

float Scheduler::idle_time_ms() const {
    auto now = Clock::now();
    return std::chrono::duration_cast<Ms>(now - last_external_input_).count();
}

void Scheduler::on_external_input() {
    last_external_input_ = Clock::now();
    state_.store(WakefulnessState::ACTIVE, std::memory_order_relaxed);
    // Don't slam wakefulness to 1.0 — let boost_wakefulness() modulate.
    // Just ensure minimum ACTIVE level so tick() doesn't immediately transition.
    float w = wakefulness_.load(std::memory_order_relaxed);
    if (w < cfg_.daydream_wakefulness) {
        wakefulness_.store(cfg_.daydream_wakefulness, std::memory_order_relaxed);
    }
}

void Scheduler::boost_wakefulness(float delta) {
    float w = wakefulness_.load(std::memory_order_relaxed);
    w = std::min(1.0f, w + delta);
    wakefulness_.store(w, std::memory_order_relaxed);
}

void Scheduler::on_probe_silence() {
    float w = wakefulness_.load(std::memory_order_relaxed);
    w = std::max(0.0f, w - cfg_.probe_silence_decay);
    wakefulness_.store(w, std::memory_order_relaxed);
}

void Scheduler::on_probe_response() {
    wakefulness_.store(cfg_.active_wakefulness, std::memory_order_relaxed);
    // Do NOT reset last_external_input_ here — self-generated responses
    // should not prevent ACTIVE → DAYDREAM transition. Only real external
    // input (ASR/TEXT) resets the idle timer via on_external_input().
}

WakefulnessState Scheduler::tick() {
    auto now = Clock::now();
    float dt_sec = std::chrono::duration_cast<Ms>(now - last_tick_).count() / 1000.0f;
    last_tick_ = now;

    float idle_ms = std::chrono::duration_cast<Ms>(now - last_external_input_).count();
    WakefulnessState current = state_.load(std::memory_order_relaxed);

    // State transitions based on idle time
    WakefulnessState next = current;
    float target_w;

    if (idle_ms < cfg_.idle_threshold_ms) {
        next = WakefulnessState::ACTIVE;
        target_w = cfg_.active_wakefulness;
    } else if (idle_ms < cfg_.dream_threshold_ms) {
        next = WakefulnessState::DAYDREAM;
        target_w = cfg_.daydream_wakefulness;
    } else {
        next = WakefulnessState::DREAMING;
        target_w = cfg_.dream_wakefulness;
    }

    state_.store(next, std::memory_order_relaxed);

    // Exponential decay toward target wakefulness
    // w(t+dt) = target + (w - target) * exp(-ln2 / half_life * dt)
    float w = wakefulness_.load(std::memory_order_relaxed);
    if (std::fabs(w - target_w) > 0.001f) {
        float lambda = 0.693147f / cfg_.wakefulness_half_life_sec;  // ln(2)
        float factor = std::exp(-lambda * dt_sec);
        w = target_w + (w - target_w) * factor;
    }
    wakefulness_.store(w, std::memory_order_relaxed);

    return next;
}

int Scheduler::recommended_decode_branches() const {
    switch (state_.load(std::memory_order_relaxed)) {
        case WakefulnessState::ACTIVE:   return cfg_.active_decode_branches;
        case WakefulnessState::DAYDREAM: return cfg_.daydream_decode_branches;
        case WakefulnessState::DREAMING: return cfg_.dream_decode_branches;
    }
    return 1;
}

int Scheduler::max_decode_tokens() const {
    switch (state_.load(std::memory_order_relaxed)) {
        case WakefulnessState::ACTIVE:   return cfg_.active_max_decode;
        case WakefulnessState::DAYDREAM: return cfg_.daydream_max_decode;
        case WakefulnessState::DREAMING: return cfg_.dream_max_decode;
    }
    return 128;
}

bool Scheduler::should_probe() const {
    return wakefulness_.load(std::memory_order_relaxed) >= cfg_.probe_threshold;
}

int Scheduler::probe_budget() const {
    float w = wakefulness_.load(std::memory_order_relaxed);
    float t = (w - cfg_.probe_threshold) / (1.0f - cfg_.probe_threshold);
    t = std::max(0.0f, std::min(1.0f, t));
    return cfg_.probe_min_tokens +
           static_cast<int>(t * (cfg_.probe_max_tokens - cfg_.probe_min_tokens));
}

bool Scheduler::should_prefill() const {
    // Prefill only in ACTIVE state (when input is arriving)
    return state_.load(std::memory_order_relaxed) == WakefulnessState::ACTIVE;
}

} // namespace deusridet
