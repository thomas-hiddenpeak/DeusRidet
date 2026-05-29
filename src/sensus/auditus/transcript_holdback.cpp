/**
 * @file transcript_holdback.cpp
 * @philosophical_role Implementation of the Hybrid P2 ASR→Conscientia
 *     holdback (see transcript_holdback.h).
 * @serves DiarizenPeriodicWorker, auditus_facade_broadcasts.
 */
#include "transcript_holdback.h"

#include "conscientia/stream.h"
#include "orator/diarizen_facade.h"

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
        // Seed id_to_name_ with the initial guess so cross-run remap can
        // still find a friendly name even if no DiariZen pass runs first.
        const auto& enq = q_.back().item;
        if (enq.speaker_id >= 0 && id_to_name_.find(enq.speaker_id) == id_to_name_.end()) {
            id_to_name_[enq.speaker_id] = enq.speaker_name;
            if (enq.speaker_id >= next_global_id_) next_global_id_ = enq.speaker_id + 1;
        }
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
            // Commit slot history (used for label re-mapping next pass).
            committed_.push_back(CommittedSlot{pt.stream_start_sec,
                                               pt.stream_end_sec,
                                               pt.item.speaker_id,
                                               pt.item.speaker_name});
            while (committed_.size() > kCommittedCap) committed_.pop_front();
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
        committed_.push_back(CommittedSlot{pt.stream_start_sec,
                                           pt.stream_end_sec,
                                           pt.item.speaker_id,
                                           pt.item.speaker_name});
        while (committed_.size() > kCommittedCap) committed_.pop_front();
        lk.unlock();
        cs_.inject_input(std::move(pt.item));
        lk.lock();
    }
}

std::unordered_map<std::string, TranscriptHoldback::LabelAssignment>
TranscriptHoldback::assign_labels_(const std::vector<orator::DiarizenSegment>& segs,
                                   double capture_origin_sec) {
    // For each unique DiariZen label, score it against every existing
    // global_id by total overlap-seconds with committed slots. Take argmax
    // if positive; else allocate a fresh global_id.
    std::unordered_map<std::string, std::unordered_map<int, double>> label_id_overlap;
    for (const auto& s : segs) {
        double a0 = s.start_sec + capture_origin_sec;
        double a1 = s.end_sec   + capture_origin_sec;
        auto& m = label_id_overlap[s.label];
        for (const auto& c : committed_) {
            double ov = overlap_seconds(a0, a1, c.stream_start_sec, c.stream_end_sec);
            if (ov > 0.0) m[c.speaker_id] += ov;
        }
        // Also score against currently-pending items so cross-pass stable
        // ids can be recovered even when committed_ is still empty (first
        // diarisation pass on a brand-new session).
        for (const auto& p : q_) {
            double ov = overlap_seconds(a0, a1, p.stream_start_sec, p.stream_end_sec);
            if (ov > 0.0 && p.item.speaker_id >= 0) {
                m[p.item.speaker_id] += ov * 0.25;  // weak prior
            }
        }
    }

    // Greedy resolution: process labels in order of best score, pin each
    // to its top remaining global_id (so two labels don't collapse).
    std::unordered_map<std::string, LabelAssignment> out;
    std::unordered_map<int, std::string> taken_id_to_label;
    struct Cand { std::string label; int id; double score; };
    std::vector<Cand> cands;
    cands.reserve(64);
    for (const auto& kv : label_id_overlap) {
        for (const auto& kv2 : kv.second) {
            cands.push_back({kv.first, kv2.first, kv2.second});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    for (const auto& c : cands) {
        if (out.count(c.label)) continue;
        if (taken_id_to_label.count(c.id)) continue;
        if (c.score < 0.5) continue;  // need at least half a second
        LabelAssignment a;
        a.global_id = c.id;
        auto it = id_to_name_.find(c.id);
        a.name = (it != id_to_name_.end()) ? it->second : std::string("Speaker ") + std::to_string(c.id);
        out[c.label] = a;
        taken_id_to_label[c.id] = c.label;
    }
    // Anything still unassigned → fresh id.
    for (const auto& kv : label_id_overlap) {
        if (out.count(kv.first)) continue;
        LabelAssignment a;
        a.global_id = next_global_id_++;
        a.name = std::string("Speaker ") + std::to_string(a.global_id);
        out[kv.first] = a;
        id_to_name_[a.global_id] = a.name;
    }
    // Propagate any name updates back to id_to_name_.
    for (const auto& kv : out) {
        id_to_name_[kv.second.global_id] = kv.second.name;
    }
    return out;
}

size_t TranscriptHoldback::apply_diarization(
        const std::vector<orator::DiarizenSegment>& segs,
        double capture_origin_sec) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty() || segs.empty()) return 0;

    auto label_map = assign_labels_(segs, capture_origin_sec);

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
        auto it = label_map.find(best);
        if (it == label_map.end()) continue;
        int new_id = it->second.global_id;
        if (new_id != p.item.speaker_id) {
            p.item.speaker_id = new_id;
            p.item.speaker_name = it->second.name;
            ++changed;
        }
    }
    return changed;
}

}  // namespace deusridet::auditus
