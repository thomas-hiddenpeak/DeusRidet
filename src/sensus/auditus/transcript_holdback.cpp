/**
 * @file transcript_holdback.cpp
 * @philosophical_role Implementation of the Hybrid P2 ASR→Conscientia
 *     holdback (see transcript_holdback.h).
 * @serves DiarizenPeriodicWorker, auditus_facade_broadcasts.
 */
#include "transcript_holdback.h"

#include "conscientia/stream.h"
#include "orator/diarizen_pipeline.h"  // orator::DiarizenSegment

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

namespace deusridet::auditus {

namespace {

inline double overlap_seconds(double a0, double a1, double b0, double b1) {
    double lo = std::max(a0, b0);
    double hi = std::min(a1, b1);
    return hi > lo ? (hi - lo) : 0.0;
}

// Registry-stitched labels are "S<gid>". Return the gid, or -1 if the label
// is not in that durable identity space (e.g. an unstitched fallback).
inline int parse_global_id(const std::string& label) {
    if (label.size() < 2 || label[0] != 'S') return -1;
    int v = 0;
    for (size_t i = 1; i < label.size(); ++i) {
        if (label[i] < '0' || label[i] > '9') return -1;
        v = v * 10 + (label[i] - '0');
    }
    return v;
}

}  // namespace

TranscriptHoldback::TranscriptHoldback(ConscientiStream& cs,
                                       double holdback_sec,
                                       std::function<double()> stream_clock_sec_fn)
    : cs_(cs),
      holdback_sec_(holdback_sec),
      stream_clock_sec_fn_(std::move(stream_clock_sec_fn)) {}

TranscriptHoldback::~TranscriptHoldback() { stop(); }

void TranscriptHoldback::start() {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return;
    running_ = true;
    stop_req_ = false;
    drainer_ = std::thread(&TranscriptHoldback::drainer_loop_, this);
}

void TranscriptHoldback::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!running_) return;
        stop_req_ = true;
    }
    cv_.notify_all();
    if (drainer_.joinable()) drainer_.join();
    drain_now();
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
}

void TranscriptHoldback::enqueue(InputItem item,
                                 double stream_start_sec,
                                 double stream_end_sec) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(PendingTranscript{std::move(item),
                                       stream_start_sec,
                                       stream_end_sec});
    }
    cv_.notify_all();
}

size_t TranscriptHoldback::pending_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
}

void TranscriptHoldback::drainer_loop_() {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_req_) {
        cv_.wait_for(lk, 250ms);
        if (stop_req_) break;
        double now = stream_clock_sec_fn_ ? stream_clock_sec_fn_() : 0.0;
        while (!q_.empty()) {
            const auto& head = q_.front();
            if (now - head.stream_end_sec < holdback_sec_) break;
            PendingTranscript pt = std::move(q_.front());
            q_.pop_front();
            // Release lock around the (potentially heavy) inject call.
            lk.unlock();
            cs_.inject_input(std::move(pt.item));
            lk.lock();
        }
    }
}

void TranscriptHoldback::drain_now() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!q_.empty()) {
        PendingTranscript pt = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        cs_.inject_input(std::move(pt.item));
        lk.lock();
    }
}

size_t TranscriptHoldback::apply_diarization(
        const std::vector<orator::DiarizenSegment>& segs,
        double capture_origin_sec) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty() || segs.empty()) return 0;

    size_t changed = 0;
    for (auto& p : q_) {
        // Find label with max overlap against this pending item.
        std::unordered_map<std::string, double> per_label_ov;
        for (const auto& s : segs) {
            double a0 = s.start_sec + capture_origin_sec;
            double a1 = s.end_sec   + capture_origin_sec;
            double ov = overlap_seconds(a0, a1, p.stream_start_sec, p.stream_end_sec);
            if (ov > 0.0) per_label_ov[s.label] += ov;
        }
        if (per_label_ov.empty()) continue;
        std::string best;
        double best_ov = 0.0;
        for (const auto& kv : per_label_ov) {
            if (kv.second > best_ov) { best_ov = kv.second; best = kv.first; }
        }
        // segs were already stitched onto durable voiceprint-anchored global
        // identities ("S<gid>") by DiarizenIdentityRegistry — the single
        // identity authority. Parse the gid directly; no second stitcher.
        int gid = parse_global_id(best);
        if (gid < 0) continue;
        if (gid != p.item.speaker_id) {
            p.item.speaker_id = gid;
            auto it = id_to_name_.find(gid);
            p.item.speaker_name = (it != id_to_name_.end())
                ? it->second
                : (std::string("Speaker ") + std::to_string(gid));
            ++changed;
        }
    }
    return changed;
}

}  // namespace deusridet::auditus
