/**
 * @file orator_reclusterer.h
 * @philosophical_role The Orator's second hearing — after each segment receives
 *     a tentative label from the online state machine, a rolling window of the
 *     last W seconds is re-clustered offline. What the entity heard in haste,
 *     it now corrects in reflection. Tentative labels are retroactively
 *     overwritten by globally consistent identities.
 * @serves Diarisation pipeline (Phase 1 of the speaker-ID rework). Replaces
 *     the AHC/centroid-cosine matcher whose ceiling sits at macro ≈ 0.17 on
 *     tests/test.mp3 with rolling-window spectral clustering + Hungarian
 *     global-ID linking (PoC headroom: +0.28 macro on s1800, +0.19 on full).
 */
// orator_reclusterer.h — rolling-window spectral re-cluster with persistent
// global IDs. Synchronous, single-threaded, CPU-only. Designed to be driven
// from the audio pipeline thread after each segment is finalised.
//
// Lifecycle:
//   push(segment)           // O(1) — appends to ring buffer
//   tick(now_sec)           // O(W^3) spectral pass once per interval_sec
//   drain_relabels(out)     // pops pending RelabelEvent batches
//
// Memory budget: 300 segments * (384 floats embedding + bookkeeping) ≈ 500 KB
// in CPU RAM. No CUDA, no GPU residency.

#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace deusridet::orator {

// One segment fed into the reclusterer. Embedding is the L2-normalised
// per-segment vector used for similarity. For DeusRidet today this is the
// 384-D fused (CAM++ 192D || WL-ECAPA 192D) embedding.
struct ReclusterSegment {
    uint64_t segment_id   = 0;      // monotonic id allocated by audio pipeline
    double   t_center_sec = 0.0;    // midpoint of the segment on the wall clock
    double   t_start_sec  = 0.0;
    double   t_end_sec    = 0.0;
    int      tentative_speaker_id = -1;  // what the online state machine emitted
    std::vector<float> embedding;        // L2-normalised, dim must equal cfg.embedding_dim
};

// Emitted whenever a segment's globally-consistent speaker identity differs
// from the label that was previously broadcast. The audio pipeline / Nexus
// layer is expected to forward this to the WebUI as a `speaker_relabel`
// event so the timeline can be patched in place.
struct RelabelEvent {
    uint64_t segment_id      = 0;
    int      old_speaker_id  = -1;
    int      new_speaker_id  = -1;
    float    confidence      = 0.0f;   // cosine sim to the matched global centroid
};

struct OratorReclustererConfig {
    int    embedding_dim   = 384;     // fused CAM++ || WL-ECAPA
    double window_sec      = 180.0;   // W — rolling buffer length in seconds (Phase 6: smaller windows reduce K-means contamination; empirically peaks at 180s)
    double interval_sec    = 30.0;    // S — how often tick() runs the spectral pass
    int    min_segments    = 12;      // do not run unless at least this many segs in window
    int    max_segments    = 300;     // hard cap to keep spectral O(N^3) cheap
    int    max_k           = 6;       // upper bound on cluster count
    int    min_k           = 2;
    float  link_threshold  = 0.55f;   // cosine sim required to reuse an existing global id
    float  centroid_ema    = 0.20f;   // running-mean rate for global centroid update
    int    global_id_base  = 1000;    // first id assigned by the reclusterer (avoid collision with online ids)
    float  global_merge_threshold = -1.0f;  // cos sim above which two globals are merged into one (≤0 to disable)
    float  global_merge_support_ratio = 0.5f; // only merge if min_support <= ratio * max_support (protects mixed clusters)
    int    k_selection_mode = 0;      // 0=nme+rel_gap (legacy); 1=eigenvalue ratio (parameter-free, recommended)
};

// Internal book-keeping for a persistent speaker identity discovered by the
// reclusterer. Exposed for tests / introspection only.
struct GlobalSpeaker {
    int               id              = -1;
    std::vector<float> centroid;        // L2-normalised, embedding_dim
    int               support_count   = 0;     // number of segments ever assigned
    double            last_seen_sec   = 0.0;
};

class OratorReclusterer {
public:
    explicit OratorReclusterer(OratorReclustererConfig cfg = {});

    // Append a finalised segment to the rolling buffer. The segment's
    // tentative_speaker_id is recorded so the next tick() can compare and
    // emit a RelabelEvent if the global ID differs.
    void push(const ReclusterSegment& seg);

    // Drive the reclusterer's clock. If at least cfg.interval_sec has
    // elapsed since the last run, prune the buffer to the trailing window,
    // execute the spectral pass, run the Hungarian linker against the
    // persistent global speakers, update committed labels, and queue any
    // resulting RelabelEvents.
    //
    // Returns the number of RelabelEvents queued by this tick (0 if no pass
    // was run).
    int tick(double now_sec);

    // Pop all pending relabel events.
    void drain_relabels(std::vector<RelabelEvent>& out);

    // Inspection.
    int   global_speaker_count() const { return static_cast<int>(globals_.size()); }
    int   window_segment_count() const { return static_cast<int>(buffer_.size()); }
    double last_run_sec()        const { return last_run_sec_; }
    const std::unordered_map<int, GlobalSpeaker>& globals() const { return globals_; }

    // Force a recluster pass regardless of cfg.interval_sec. Returns number
    // of relabel events queued. Useful for tests.
    int force_run(double now_sec);

private:
    // Per-segment slot kept in the ring buffer. Stores the last committed
    // label so subsequent ticks know what was already broadcast.
    struct Slot {
        ReclusterSegment seg;
        int              committed_speaker_id = -1;   // last id we authored
        bool             ever_committed       = false;
    };

    OratorReclustererConfig                cfg_;
    std::deque<Slot>                       buffer_;
    std::unordered_map<int, GlobalSpeaker> globals_;
    std::unordered_map<uint64_t, int>      committed_history_; // segment_id -> last committed id (kept beyond the window so merges can retroactively relabel)
    std::vector<RelabelEvent>              pending_;

    int    next_global_id_ = 0;
    double last_run_sec_   = -1e18;

    // Prune segments older than now_sec - cfg_.window_sec.
    void prune_window(double now_sec);

    // Execute one spectral + Hungarian pass on the current buffer. Updates
    // committed labels and pushes RelabelEvents into pending_. Returns the
    // number of events appended.
    int run_pass(double now_sec);
};

} // namespace deusridet::orator
