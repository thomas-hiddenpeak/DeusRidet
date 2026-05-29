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
 *     refuse to invalidate that). Cross-run label stability is preserved by
 *     a small committed-history table: new DiariZen labels are mapped to
 *     existing global speaker_ids by overlap-seconds, with fresh ids only
 *     allocated for labels that have no overlap with any committed item.
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

struct CommittedSlot {
    double    stream_start_sec = 0.0;
    double    stream_end_sec   = 0.0;
    int       speaker_id       = -1;
    std::string speaker_name;
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
    /// @param segs        DiariZen pipeline-local segments (label = e.g.
    ///                    "speaker0"); times are in *capture-relative* sec.
    /// @param capture_origin_sec  add this to seg.start/end_sec to get
    ///                    absolute stream-time. (Comes from
    ///                    AudioPipeline::diarizen_capture_origin_sec.)
    /// Returns number of pending items whose speaker_id changed.
    size_t apply_diarization(const std::vector<orator::DiarizenSegment>& segs,
                             double capture_origin_sec);

    /// Force-drain every pending item into Conscientia *now*, ignoring the
    /// holdback window. Used at finalize.
    void drain_now();

    /// Diagnostic.
    size_t pending_count() const;

private:
    void drainer_loop_();

    /// Re-label segs into stable global speaker_ids by Hungarian-like
    /// overlap match against committed_. Each unique label in `segs` gets
    /// an assignment in the returned `label_to_global_` map.
    /// Updates `next_global_id_` when a label has no prior overlap.
    struct LabelAssignment {
        int  global_id = -1;
        std::string name;        // best-known display name for that id
    };
    std::unordered_map<std::string, LabelAssignment>
    assign_labels_(const std::vector<orator::DiarizenSegment>& segs,
                   double capture_origin_sec);

    ConscientiStream&         cs_;
    double                    holdback_sec_;
    std::function<double()>   stream_clock_sec_fn_;

    mutable std::mutex        mu_;
    std::condition_variable   cv_;
    std::deque<PendingTranscript> q_;
    std::deque<CommittedSlot>     committed_;   // capped FIFO of injected slots
    std::unordered_map<int, std::string> id_to_name_;
    int                       next_global_id_ = 0;

    std::thread               drainer_;
    bool                      running_ = false;
    bool                      stop_req_ = false;

    static constexpr size_t   kCommittedCap = 1024;
};

}  // namespace auditus
}  // namespace deusridet
