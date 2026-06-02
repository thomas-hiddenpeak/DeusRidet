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

#include "diarizen_identity_registry.h"

#include <atomic>
#include <chrono>
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
    struct StatusSnapshot {
        bool periodic_enabled = true;
        bool running = false;
        double period_sec = 0.0;
        double window_sec = 0.0;
        double cycle_progress = 0.0;
        uint64_t pass_seq = 0;
        std::string phase = "idle";
    };

    /// `holdback` is nullable: in audio-only sessions (LLM not loaded)
    /// there is no Conscientia stream to drain into, so the worker still
    /// runs diarisation passes and broadcasts `speaker_diarize_*` for the
    /// live WebUI, but skips the holdback rewrite/drain entirely.
    DiarizenPeriodicWorker(AudioPipeline& audio,
                           DiarizenPipeline& pipeline,
                           auditus::TranscriptHoldback* holdback,
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

    /// Thread-safe live status for WS/UI observability.
    StatusSnapshot snapshot_status();

private:
    void worker_loop_();
    /// Runs one pass. `is_final` controls the WS broadcast type.
    /// `window_sec` bounds the diarised audio to the most recent N seconds
    /// (0 = whole accumulated session). Returns true on success (DiariZen
    /// returned a non-empty segment list).
    bool run_one_pass_(bool is_final, double window_sec);

    AudioPipeline&               audio_;
    DiarizenPipeline&            pipeline_;
    auditus::TranscriptHoldback* holdback_;  // nullable — see ctor doc
    WsServer&                    server_;
    double                       period_sec_;
    std::atomic<uint64_t>        pass_seq_{0};

    // P2 — cross-window persistent identity. Windowed (partial) passes return
    // pipeline-local labels with no memory of earlier windows; the registry
    // binds them onto durable global identities by time-overlap before the
    // broadcast, so the live label stream is identity-stable. The full-session
    // finalize pass is canonical and is NOT stitched (its single clustering is
    // already globally consistent). Only ever touched from the pass thread.
    DiarizenIdentityRegistry     identity_;

    std::mutex                   mu_;
    std::condition_variable      cv_;
    std::thread                  th_;
    bool                         running_ = false;
    bool                         stop_req_ = false;
    bool                         trigger_req_ = false;
    std::string                  phase_ = "idle";
    // Timed live re-diarise is ON by default after the Jun 2 B1 live proof.
    // It is bounded by window_sec_ (120 s default) so each pass has finite GPU
    // wall time and cannot grow with session length. Set
    // DEUSRIDET_DIARIZEN_PERIODIC=0 to opt out; trigger_async() and finalize()
    // remain available either way.
    bool                         periodic_enabled_ = true;
    // Direction C — sliding-window live diarise. When > 0, periodic and
    // on-demand passes diarise only the most recent `window_sec_` seconds
    // of captured audio, bounding per-pass GPU wall regardless of session
    // length (a full-session pass is O(N) and on a long session blocks the
    // GPU long enough to overflow the live front-end audio buffer). The
    // canonical end-of-session finalize() always runs a FULL pass so the
    // accuracy verdict is unchanged. Set DEUSRIDET_DIARIZEN_WINDOW_SEC=0 to
    // force full-session live passes. Should be ≥ the holdback horizon so
    // live transcripts pending
    // for the LLM still fall inside the re-diarised window.
    double                       window_sec_ = 120.0;
    std::chrono::steady_clock::time_point next_due_ = {};
};

}  // namespace orator
}  // namespace deusridet
