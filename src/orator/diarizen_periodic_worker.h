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

    std::mutex                   mu_;
    std::condition_variable      cv_;
    std::thread                  th_;
    bool                         running_ = false;
    // DiarizenPipeline::diarize() is NOT re-entrant: it drives one shared
    // set of WavLM / Conformer GPU scratch buffers on one CUDA stream, so
    // two concurrent passes corrupt each other (observed as
    // CUDNN_STATUS_EXECUTION_FAILED → empty segmentation). Periodic and
    // on-demand passes all run on the worker thread and are serial by
    // construction, but finalize() runs on a detached WS-handler thread and
    // can be invoked more than once; pass_mutex_ guarantees at most one
    // diarize() pass executes at any instant across all callers.
    std::mutex                   pass_mutex_;
    bool                         stop_req_ = false;
    bool                         trigger_req_ = false;
    // Timed full-session re-diarise is OFF by default. A full re-diarise
    // of the whole accumulated session every period is O(N²) and, on a
    // long session, monopolises the GPU for minutes — starving the live
    // perception pipeline (FRCRN/VAD/speaker-id) that shares the same GPU
    // and poisoning the CUDA context. Set DEUSRIDET_DIARIZEN_PERIODIC=1 to
    // re-enable the timed cadence; otherwise the worker only runs on an
    // explicit trigger_async() or finalize().
    bool                         periodic_enabled_ = false;
    // Direction C — sliding-window live diarise. When > 0, periodic and
    // on-demand passes diarise only the most recent `window_sec_` seconds
    // of captured audio, bounding per-pass GPU wall regardless of session
    // length (a full-session pass is O(N) and on a long session blocks the
    // GPU long enough to overflow the live front-end audio buffer). The
    // canonical end-of-session finalize() always runs a FULL pass so the
    // accuracy verdict is unchanged. 0 (default) preserves full-session
    // behaviour on every pass. Set DEUSRIDET_DIARIZEN_WINDOW_SEC=<sec> to
    // opt in. Should be ≥ the holdback horizon so live transcripts pending
    // for the LLM still fall inside the re-diarised window.
    double                       window_sec_ = 0.0;
};

}  // namespace orator
}  // namespace deusridet
