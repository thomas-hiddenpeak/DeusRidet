// speaker_timeline.h — Speaker timeline types and fused resolution.
//
// Extracted from audio_pipeline.h to allow speaker_stream.h to include
// timeline types without circular dependency.

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>

namespace deusridet {

// ─── SpeakerTimeline: fused speaker resolution via event sourcing ───
//
// Speaker identification writes timestamped events to a shared timeline.
// When an ASR segment needs a speaker label, resolve() queries the timeline
// for the best label covering the audio range via weighted majority voting.
//
// Authority hierarchy: SPK_FULL > SPK_CHANGE > SPK_EARLY

enum class SpkEventSource : uint8_t {
    SPK_EARLY   = 0,  // Early extraction (auto_register=false)
    SPK_FULL    = 1,  // End-of-segment (auto_register=true)
    SPK_CHANGE  = 2,  // Intra-segment speaker change detection
};

struct SpeakerEvent {
    int64_t  audio_start;   // start of audio range this event covers (absolute sample)
    int64_t  audio_end;     // end of audio range this event covers (absolute sample)
    SpkEventSource source;
    int      speaker_id;    // -1 = unknown/no match
    float    similarity;
    char     name[64];
};

struct ResolvedSpeaker {
    int      speaker_id = -1;
    float    confidence = 0.0f;   // total weighted vote
    float    similarity = 0.0f;
    char     name[64] = {};
    SpkEventSource source = SpkEventSource::SPK_EARLY;
};

class SpeakerTimeline {
public:
    static constexpr int kMaxEvents = 2000;

    void push(const SpeakerEvent& ev) {
        events_[write_pos_] = ev;
        write_pos_ = (write_pos_ + 1) % kMaxEvents;
        if (count_ < kMaxEvents) ++count_;
    }

    // Resolve the best speaker label for a given audio sample range.
    // Uses weighted majority voting with overlap-proportional weighting.
    // If no events overlap the query range, performs context fill:
    // finds the nearest event before and after — if they agree on speaker, fills.
    ResolvedSpeaker resolve(int64_t start_sample, int64_t end_sample) const {
        if (count_ == 0 || end_sample <= start_sample)
            return {};

        // Source authority weights.
        static constexpr float kWeights[] = {
            0.70f,  // SPK_EARLY   — early extraction, might not match final
            1.00f,  // SPK_FULL    — end-of-segment, highest authority
            0.90f,  // SPK_CHANGE  — speaker change detection, high authority
        };

        int64_t query_len = end_sample - start_sample;

        // Accumulate weighted votes per speaker_id.
        static constexpr int kMaxSpk = 64;
        float votes[kMaxSpk] = {};
        float best_sim[kMaxSpk] = {};
        char  best_name[kMaxSpk][64] = {};
        SpkEventSource best_source[kMaxSpk] = {};
        float best_source_weight[kMaxSpk] = {};
        bool any_overlap = false;

        // Also track nearest non-overlapping events for context fill.
        int64_t nearest_before_dist = INT64_MAX;
        int     nearest_before_spk  = -1;
        float   nearest_before_sim  = 0.0f;
        char    nearest_before_name[64] = {};
        SpkEventSource nearest_before_src = SpkEventSource::SPK_EARLY;

        int64_t nearest_after_dist  = INT64_MAX;
        int     nearest_after_spk   = -1;
        float   nearest_after_sim   = 0.0f;
        char    nearest_after_name[64] = {};
        SpkEventSource nearest_after_src = SpkEventSource::SPK_EARLY;

        int start_idx = (count_ < kMaxEvents) ? 0 : write_pos_;
        for (int i = 0; i < count_; ++i) {
            int idx = (start_idx + i) % kMaxEvents;
            const auto& ev = events_[idx];
            if (ev.speaker_id < 0 || ev.speaker_id >= kMaxSpk) continue;

            // Compute overlap between event range and query range.
            int64_t ov_start = std::max(ev.audio_start, start_sample);
            int64_t ov_end   = std::min(ev.audio_end, end_sample);
            if (ov_end > ov_start) {
                any_overlap = true;
                float overlap_frac = (float)(ov_end - ov_start) / (float)query_len;
                float w = kWeights[static_cast<int>(ev.source)] * overlap_frac;
                votes[ev.speaker_id] += w;

                // Track best similarity per speaker (prefer highest-authority source).
                float sw = kWeights[static_cast<int>(ev.source)];
                if (sw > best_source_weight[ev.speaker_id] ||
                    (sw == best_source_weight[ev.speaker_id] &&
                     ev.similarity > best_sim[ev.speaker_id])) {
                    best_sim[ev.speaker_id] = ev.similarity;
                    memcpy(best_name[ev.speaker_id], ev.name, 64);
                    best_source[ev.speaker_id] = ev.source;
                    best_source_weight[ev.speaker_id] = sw;
                }
            } else {
                // No overlap — track nearest events for context fill.
                if (ev.audio_end <= start_sample) {
                    int64_t dist = start_sample - ev.audio_end;
                    if (dist < nearest_before_dist) {
                        nearest_before_dist = dist;
                        nearest_before_spk  = ev.speaker_id;
                        nearest_before_sim  = ev.similarity;
                        memcpy(nearest_before_name, ev.name, 64);
                        nearest_before_src  = ev.source;
                    }
                } else if (ev.audio_start >= end_sample) {
                    int64_t dist = ev.audio_start - end_sample;
                    if (dist < nearest_after_dist) {
                        nearest_after_dist = dist;
                        nearest_after_spk  = ev.speaker_id;
                        nearest_after_sim  = ev.similarity;
                        memcpy(nearest_after_name, ev.name, 64);
                        nearest_after_src  = ev.source;
                    }
                }
            }
        }

        if (any_overlap) {
            // Find speaker with highest vote.
            int best_id = -1;
            float best_vote = 0.0f;
            for (int i = 0; i < kMaxSpk; ++i) {
                if (votes[i] > best_vote) {
                    best_vote = votes[i];
                    best_id = i;
                }
            }

            if (best_id < 0) return {};

            ResolvedSpeaker result;
            result.speaker_id = best_id;
            result.confidence = best_vote;
            result.similarity = best_sim[best_id];
            memcpy(result.name, best_name[best_id], 64);
            result.source = best_source[best_id];
            return result;
        }

        // Context fill: no overlapping events. Check nearest before/after.
        // If both agree on same speaker AND gap is < 5s (80000 samples), fill.
        // If only one side has a result within 3s (48000 samples), use it.
        static constexpr int64_t kFillBothMaxGap  = 80000;  // 5s
        static constexpr int64_t kFillSingleMaxGap = 48000;  // 3s

        if (nearest_before_spk >= 0 && nearest_after_spk >= 0 &&
            nearest_before_spk == nearest_after_spk &&
            nearest_before_dist + nearest_after_dist < kFillBothMaxGap) {
            // Both neighbors agree — high confidence context fill.
            ResolvedSpeaker result;
            result.speaker_id = nearest_before_spk;
            result.confidence = 0.35f;  // lower confidence for context fill
            // Use the higher-authority source's similarity.
            float sw_b = kWeights[static_cast<int>(nearest_before_src)];
            float sw_a = kWeights[static_cast<int>(nearest_after_src)];
            if (sw_b >= sw_a) {
                result.similarity = nearest_before_sim;
                memcpy(result.name, nearest_before_name, 64);
                result.source = nearest_before_src;
            } else {
                result.similarity = nearest_after_sim;
                memcpy(result.name, nearest_after_name, 64);
                result.source = nearest_after_src;
            }
            return result;
        }

        // Single-side fill: only one neighbor within 3s.
        if (nearest_before_spk >= 0 && nearest_before_dist < kFillSingleMaxGap) {
            ResolvedSpeaker result;
            result.speaker_id = nearest_before_spk;
            result.confidence = 0.20f;  // low confidence single-side fill
            result.similarity = nearest_before_sim;
            memcpy(result.name, nearest_before_name, 64);
            result.source = nearest_before_src;
            return result;
        }
        if (nearest_after_spk >= 0 && nearest_after_dist < kFillSingleMaxGap) {
            ResolvedSpeaker result;
            result.speaker_id = nearest_after_spk;
            result.confidence = 0.15f;  // lowest confidence
            result.similarity = nearest_after_sim;
            memcpy(result.name, nearest_after_name, 64);
            result.source = nearest_after_src;
            return result;
        }

        return {};
    }

    int event_count() const { return count_; }
    void clear() { count_ = 0; write_pos_ = 0; }

private:
    SpeakerEvent events_[kMaxEvents];
    int write_pos_ = 0;
    int count_ = 0;
};

}  // namespace deusridet
