/**
 * @file transcript_holdback.h
 * @philosophical_role Hybrid P2 — defers ASR→Conscientia injection by a
 *     small holdback window so that periodic DiariZen reclustering can
 *     rewrite the speaker_id (and speaker_name) the LLM actually sees,
 *     not just the WebUI display layer.
 *
 *     Plan A: a pending FIFO sits between the auditus transcript callback
 *     and ConscientiStream::inject_input. Items are released to Conscientia
 *     after `holdback_sec` of stream-time has elapsed past their end. Until
 *     then they are mutable: `apply_diarization()` rewrites their
 *     speaker_id/name from a fresh DiariZen-v2 pass.
 *
 *     Once an item leaves the holdback into Conscientia it is permanent
 *     (Conscientia tokenises and prefills into the KV cache; we deliberately
 *     refuse to invalidate that). Cross-run label stability is NOT computed
 *     here: the DiarizenPeriodicWorker runs the single identity authority —
 *     DiarizenIdentityRegistry — over each pass FIRST, binding pipeline-local
 *     clusters onto durable voiceprint-anchored global ids ("S<gid>") before
 *     handing the segments here. apply_diarization() then only overlap-maps
 *     each pending utterance onto the best-covering "S<gid>" and parses the
 *     gid directly. One stitcher, voiceprint-anchored, feeds both the LLM
 *     transcript path and the live WebUI broadcast.
 *
 * @serves auditus install_transcript_callback, DiarizenPeriodicWorker,
 *         awaken's diarizen_finalize/diarizen_trigger WS commands.
 */
#pragma once

#include "conscientia/frame.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace deusridet {

class ConscientiStream;

namespace orator {
struct DiarizenSegment;
}

namespace auditus {

struct PendingTranscript {
    InputItem item;
    double    stream_start_sec = 0.0;
    double    stream_end_sec   = 0.0;
};

/// Holdback FIFO between ASR transcripts and ConscientiStream::inject_input.
/// Thread-safe; one drainer thread.
class TranscriptHoldback {
public:
    /// @param cs            target Conscientia stream (must outlive this).
    /// @param holdback_sec  how many stream-time seconds an item must "age"
    ///                      past its end before it is allowed to drain into
    ///                      Conscientia. Recommend ≥ DiariZen period_sec so
    ///                      at least one full diarisation pass sees the
    ///                      utterance before commit.
    /// @param stream_clock_sec_fn  callback returning current stream-time
    ///                      seconds (audio_t1 / 16000). Used by the drainer
    ///                      to decide when a pending item is ripe.
    TranscriptHoldback(ConscientiStream& cs,
                       double holdback_sec,
                       std::function<double()> stream_clock_sec_fn);

    ~TranscriptHoldback();

    TranscriptHoldback(const TranscriptHoldback&) = delete;
    TranscriptHoldback& operator=(const TranscriptHoldback&) = delete;

    /// Start the drainer thread. Safe to call once; further calls no-op.
    void start();

    /// Drain everything still pending into Conscientia, then stop the
    /// drainer thread. Idempotent.
    void stop();

    /// Enqueue a transcript for delayed injection. `stream_start_sec` and
    /// `stream_end_sec` are absolute stream-time seconds (since pipeline
    /// boot). The item's current speaker_id/name are kept as the initial
    /// guess and may be overwritten by `apply_diarization()` before drain.
    void enqueue(InputItem item, double stream_start_sec, double stream_end_sec);

    /// Apply a fresh DiariZen-v2 pass to all currently-pending items.
    /// @param segs        segments ALREADY stitched by DiarizenIdentityRegistry
    ///                    (label = "S<gid>", durable voiceprint-anchored global
    ///                    id); times are in *capture-relative* sec.
    /// @param capture_origin_sec  add this to seg.start/end_sec to get
    ///                    absolute stream-time. (Comes from
    ///                    AudioPipeline::diarizen_capture_origin_sec.)
    /// Returns number of pending items whose speaker_id changed.
    size_t apply_diarization(const std::vector<orator::DiarizenSegment>& segs,
                             double capture_origin_sec);

    /// Force-drain every pending item into Conscientia *now*, ignoring the
    /// holdback window. Used at finalize.
    void drain_now();

    /// Set a callback invoked with the FINAL, post-holdback speaker_id/name
    /// of each transcript, immediately before it is committed to Conscientia.
    /// This lets the broadcast/capture layer observe the voiceprint-anchored
    /// id the LLM actually consumes — not the provisional online tracker id
    /// that was sent at ASR time. Arguments: (item, stream_start_sec,
    /// stream_end_sec). Set once before start().
    void set_on_commit(
        std::function<void(const InputItem&, double, double)> fn);

    /// Diagnostic.
    size_t pending_count() const;

private:
    void drainer_loop_();

    ConscientiStream&         cs_;
    double                    holdback_sec_;
    std::function<double()>   stream_clock_sec_fn_;

    mutable std::mutex        mu_;
    std::condition_variable   cv_;
    std::deque<PendingTranscript> q_;
    // Optional gid → human display name (enrollment hook for the persona
    // layer). Empty by default ⇒ names render as "Speaker <gid>".
    std::unordered_map<int, std::string> id_to_name_;

    // Invoked with the final speaker_id/name just before each item commits
    // to Conscientia. Set once before start(); read lock-free thereafter.
    std::function<void(const InputItem&, double, double)> on_commit_;

    std::thread               drainer_;
    bool                      running_ = false;
    bool                      stop_req_ = false;
};

}  // namespace auditus
}  // namespace deusridet
