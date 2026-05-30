/**
 * @file diarizen_periodic_worker.h
 * @philosophical_role Hybrid P2 — drives DiariZen-v2 reclustering on a
 *     periodic cadence and on-demand. Each pass copies the AudioPipeline
 *     session-capture PCM and runs the in-process native CUDA
 *     DiarizenPipeline, then asks the TranscriptHoldback to rewrite
 *     still-pending speaker_ids before they reach Conscientia.
 *     A `speaker_diarize_partial` (or `speaker_diarize_final` on finalize)
 *     WS message is also broadcast for debug overlay.
 *
 *     Public surface is HTTP-portable: `trigger_async()` and
 *     `drain_and_stop()` make no assumption about the transport. Today
 *     they're wired only to WS commands; a future HTTP handler can call
 *     the same methods without modification.
 * @serves awaken (lifecycle + WS triggers), nexus (broadcast).
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace deusridet {

class AudioPipeline;
class WsServer;

namespace auditus { class TranscriptHoldback; }

namespace orator {

class DiarizenPipeline;

class DiarizenPeriodicWorker {
public:
    DiarizenPeriodicWorker(AudioPipeline& audio,
                           DiarizenPipeline& pipeline,
                           auditus::TranscriptHoldback& holdback,
                           WsServer& server,
                           double period_sec);

    ~DiarizenPeriodicWorker();

    DiarizenPeriodicWorker(const DiarizenPeriodicWorker&) = delete;
    DiarizenPeriodicWorker& operator=(const DiarizenPeriodicWorker&) = delete;

    /// Spawn the worker thread. No-op if already running.
    void start();

    /// Request one extra diarisation pass right now. Returns immediately;
    /// the worker thread will pick it up on its next wakeup.
    void trigger_async();

    /// Run one synchronous diarisation pass, then drain remaining holdback
    /// into Conscientia, then stop the worker. Broadcasts the final pass
    /// as `speaker_diarize_final`. Safe to call multiple times.
    void finalize();

    /// Stop the worker without an extra pass; pending holdback is *not*
    /// drained here (caller decides). Idempotent.
    void stop();

private:
    void worker_loop_();
    /// Runs one pass. `is_final` controls the WS broadcast type.
    /// Returns true on success (DiariZen returned a non-empty segment list).
    bool run_one_pass_(bool is_final);

    AudioPipeline&               audio_;
    DiarizenPipeline&            pipeline_;
    auditus::TranscriptHoldback& holdback_;
    WsServer&                    server_;
    double                       period_sec_;
    std::atomic<uint64_t>        pass_seq_{0};

    std::mutex                   mu_;
    std::condition_variable      cv_;
    std::thread                  th_;
    bool                         running_ = false;
    bool                         stop_req_ = false;
    bool                         trigger_req_ = false;
};

}  // namespace orator
}  // namespace deusridet
