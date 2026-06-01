/**
 * @file vires_facade.h
 * @philosophical_role The single seam onto the arterial compute substrate.
 *         Consumers reach Vires only through here: they register a metabolic
 *         class, draw a priority-tagged CUDA stream, and report submissions.
 *         Vires never reaches back up into a consumer (one-way isolation).
 * @serves machina · auditus · orator · cogitatio · vox · somnium · conscientia —
 *         any subsystem that issues GPU compute.
 * @role: Arbiter::instance() — process-global metabolic substrate (autonomic,
 *        always present, like the GPU device itself).
 */
// vires_facade.h — Vires V1 (Delivery core): consumer registry + priority
// streams + bounded-yield hint. Compute only; Memoria owns GPU memory.
//
// Dependency direction is strictly downward: this header pulls in the CUDA
// runtime and communis only. It must never include a consumer/Vigilia header.
// See docs/{en,zh}/architecture/13-vires.md.

#pragma once

#include "vires/vires_types.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace deusridet {
namespace vires {

// The arterial compute substrate. One per process, lazily constructed on first
// use (after a CUDA context exists). Thread-safe: registration, stream lookup,
// telemetry, and submission notes may be called from any consumer thread.
class Arbiter {
public:
    // Process-global accessor. The substrate is as singular as the device.
    static Arbiter& instance();

    Arbiter(const Arbiter&) = delete;
    Arbiter& operator=(const Arbiter&) = delete;

    // Register a GPU consumer. Returns a handle used for every later call.
    // `reclaim_cb` is the consumer's non-LLM scratch release hook. It is stored
    // and, from V3 on, invoked by note_pass_complete() once a non-LLM GPU pass
    // finishes (glymphatic clearance). Pass nullptr for consumers that hold no
    // reclaimable transient scratch. A priority-tagged CUDA stream is created
    // for the consumer at this point.
    ConsumerId register_consumer(const std::string& name,
                                 Priority priority,
                                 std::function<void()> reclaim_cb = nullptr);

    // Release a consumer. Destroys its stream. Safe to call once at shutdown.
    void unregister_consumer(ConsumerId id);

    // The CUDA stream Vires created for this consumer, carrying its priority.
    // Returns nullptr (the default stream) for an unknown handle so callers
    // degrade safely rather than crash at a boundary.
    cudaStream_t stream(ConsumerId id) const;

    // Telemetry: bump the consumer's submitted-pass counter. Cheap; lock-free
    // per-consumer atomic. Call once per GPU pass for observability.
    void note_submit(ConsumerId id);

    // V3 glymphatic clearance (non-LLM only). Call once when a non-LLM GPU pass
    // finishes. If the consumer registered a reclaim callback, Vires invokes it
    // OUTSIDE its own lock to release the transient scratch it handed out for
    // that pass, and bumps the consumer's `reclaimed` counter. No-op for an
    // unknown id or a consumer with no callback. Never sizes, evicts, or
    // touches any LLM allocation — that remains Memoria's charge.
    void note_pass_complete(ConsumerId id);

    // Bounded-yield hint for background passes. A background consumer should
    // chunk its work so no single launch occupies the GPU longer than this many
    // microseconds, giving foreground a guaranteed cadence to interleave.
    // Foreground consumers may ignore it.
    uint64_t background_slice_us() const { return background_slice_us_; }

    // V2 back-pressure (compute-saturation admission). Returns true when a
    // foreground submission landed within the recent activity window — i.e. the
    // live perception/prefill/decode path is busy and a background consumer
    // should pause or chunk to stay out of its way. Foreground is always
    // admitted, so it never consults this. Cheap atomic read; no lock.
    bool background_should_yield() const;

    // A consistent snapshot of the compute ledger for Nexus/WebUI.
    Snapshot snapshot() const;

private:
    Arbiter();
    ~Arbiter();

    struct Consumer {
        ConsumerStat          stat;
        cudaStream_t          stream = nullptr;
        std::function<void()> reclaim_cb;  // stored for V3; unused in V1
    };

    int greatest_priority_ = 0;  // most urgent (numerically smallest)
    int least_priority_    = 0;  // least urgent (numerically largest)
    uint64_t background_slice_us_ = 2000;  // 2 ms default bounded slice

    // V2 back-pressure: the foreground-activity window. A background consumer
    // yields while a foreground submission landed within this many microseconds.
    uint64_t foreground_active_window_us_ = 50000;  // 50 ms
    // Steady-clock microseconds of the last foreground submission (0 = none).
    std::atomic<uint64_t> last_foreground_submit_us_{0};

    mutable std::mutex mu_;
    std::unordered_map<ConsumerId, Consumer> consumers_;
    ConsumerId next_id_ = 1;  // 0 reserved as kInvalidConsumer

    // Map a metabolic class to a concrete CUDA stream priority value.
    int priority_value_(Priority p) const;

    // Monotonic microsecond clock for the foreground-activity window.
    static uint64_t now_us_();
};

} // namespace vires
} // namespace deusridet
