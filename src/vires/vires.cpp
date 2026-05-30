/**
 * @file vires.cpp
 * @philosophical_role The arterial delivery itself: query the device's priority
 *         range once, hand each consumer a stream tagged with its metabolic
 *         class, and keep the single ledger of who is computing. Below
 *         consciousness, demand-driven, never deciding what to think.
 * @serves Arbiter (vires_facade.h).
 */
// vires.cpp — Vires V1 (Delivery core) implementation.
//
// Compute substrate only. No GPU memory is sized, evicted, or relocated here —
// that is Memoria's charge. See docs/{en,zh}/architecture/13-vires.md.

#include "vires/vires_facade.h"

#include "communis/log.h"

#include <chrono>

namespace deusridet {
namespace vires {

namespace {
constexpr const char* kMod = "vires";
} // namespace

Arbiter& Arbiter::instance() {
    // Meyers singleton: constructed on first use, after a CUDA context exists.
    static Arbiter g_arbiter;
    return g_arbiter;
}

Arbiter::Arbiter() {
    // Query the device's cooperative stream-priority range. CUDA convention:
    // the numerically *smaller* value is the *greater* priority.
    cudaError_t err =
        cudaDeviceGetStreamPriorityRange(&least_priority_, &greatest_priority_);
    if (err != cudaSuccess) {
        // Boundary degradation: if the range is unavailable, fall back to a
        // single priority level (0). Streams still work; they just won't be
        // differentiated. Never crash the substrate layer.
        LOG_WARN(kMod, "cudaDeviceGetStreamPriorityRange failed (%s); "
                       "priority differentiation disabled",
                 cudaGetErrorString(err));
        least_priority_ = 0;
        greatest_priority_ = 0;
    }
    LOG_INFO(kMod, "arbiter online — priority range [greatest=%d, least=%d], "
                   "background slice %lu us",
             greatest_priority_, least_priority_,
             (unsigned long)background_slice_us_);
}

Arbiter::~Arbiter() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : consumers_) {
        if (kv.second.stream) cudaStreamDestroy(kv.second.stream);
    }
    consumers_.clear();
}

int Arbiter::priority_value_(Priority p) const {
    // Foreground → greatest priority (most urgent); Background → least.
    return (p == Priority::Foreground) ? greatest_priority_ : least_priority_;
}

ConsumerId Arbiter::register_consumer(const std::string& name,
                                      Priority priority,
                                      std::function<void()> reclaim_cb) {
    cudaStream_t s = nullptr;
    const int prio = priority_value_(priority);
    cudaError_t err =
        cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, prio);
    if (err != cudaSuccess) {
        LOG_WARN(kMod, "stream create failed for '%s' (%s); using default stream",
                 name.c_str(), cudaGetErrorString(err));
        s = nullptr;  // default stream — safe degradation
    }

    std::lock_guard<std::mutex> lk(mu_);
    const ConsumerId id = next_id_++;
    Consumer c;
    c.stat.id = id;
    c.stat.name = name;
    c.stat.priority = priority;
    c.stat.submitted = 0;
    c.stream = s;
    c.reclaim_cb = std::move(reclaim_cb);
    consumers_.emplace(id, std::move(c));

    LOG_INFO(kMod, "registered consumer #%u '%s' [%s] prio=%d",
             id, name.c_str(), priority_str(priority), prio);
    return id;
}

void Arbiter::unregister_consumer(ConsumerId id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = consumers_.find(id);
    if (it == consumers_.end()) return;
    if (it->second.stream) cudaStreamDestroy(it->second.stream);
    LOG_INFO(kMod, "unregistered consumer #%u '%s'",
             id, it->second.stat.name.c_str());
    consumers_.erase(it);
}

cudaStream_t Arbiter::stream(ConsumerId id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = consumers_.find(id);
    return (it == consumers_.end()) ? nullptr : it->second.stream;
}

void Arbiter::note_submit(ConsumerId id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = consumers_.find(id);
    if (it == consumers_.end()) return;
    ++it->second.stat.submitted;
    // V2 back-pressure: a foreground submission opens the activity window that
    // tells background consumers to yield. Recorded lock-free for cheap reads.
    if (it->second.stat.priority == Priority::Foreground) {
        last_foreground_submit_us_.store(now_us_(), std::memory_order_relaxed);
    }
}

bool Arbiter::background_should_yield() const {
    const uint64_t last =
        last_foreground_submit_us_.load(std::memory_order_relaxed);
    if (last == 0) return false;  // no foreground activity ever observed
    const uint64_t now = now_us_();
    return (now - last) < foreground_active_window_us_;
}

uint64_t Arbiter::now_us_() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

Snapshot Arbiter::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    Snapshot snap;
    snap.greatest_priority = greatest_priority_;
    snap.least_priority = least_priority_;
    snap.consumers.reserve(consumers_.size());
    for (const auto& kv : consumers_) snap.consumers.push_back(kv.second.stat);
    // V2 observability: surface the back-pressure state alongside the ledger.
    const uint64_t last =
        last_foreground_submit_us_.load(std::memory_order_relaxed);
    if (last == 0) {
        snap.foreground_idle_us = UINT64_MAX;
        snap.background_yielding = false;
    } else {
        const uint64_t now = now_us_();
        snap.foreground_idle_us = (now >= last) ? (now - last) : 0;
        snap.background_yielding =
            snap.foreground_idle_us < foreground_active_window_us_;
    }
    return snap;
}

} // namespace vires
} // namespace deusridet
