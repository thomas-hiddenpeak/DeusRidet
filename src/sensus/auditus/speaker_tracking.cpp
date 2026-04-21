/**
 * @file speaker_tracking.cpp
 * @philosophical_role SpeakerTimeline::resolve — the weighted-majority verdict that answers 'who owned this span of audio'. Moved out-of-line so the header stays a pure ledger of types.
 * @serves SpeakerTimeline.
 */
#include "speaker_tracking.h"

#include <algorithm>
#include <climits>
#include <cstring>

namespace deusridet {

// Resolve the best speaker label for a given audio sample range.
// Uses weighted majority voting with overlap-proportional weighting.
// If no events overlap the query range, performs context fill:
// finds the nearest event before and after — if they agree on speaker, fills.
ResolvedSpeaker SpeakerTimeline::resolve(int64_t start_sample, int64_t end_sample) const {
        if (count_ == 0 || end_sample <= start_sample)
            return {};

        // Source authority weights.
        static constexpr float kWeights[] = {
            0.70f,  // SAAS_EARLY   — early extraction, might not match final
            1.00f,  // SAAS_FULL    — end-of-segment, highest authority
            0.90f,  // SAAS_CHANGE  — speaker change detection, high authority
            0.50f,  // SAAS_INHERIT — inherited from previous, could be wrong
            0.40f,  // TRACKER      — independent pipeline, lower authority
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
        SpkEventSource nearest_before_src = SpkEventSource::SAAS_EARLY;

        int64_t nearest_after_dist  = INT64_MAX;
        int     nearest_after_spk   = -1;
        float   nearest_after_sim   = 0.0f;
        char    nearest_after_name[64] = {};
        SpkEventSource nearest_after_src = SpkEventSource::SAAS_EARLY;

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
        // Widened gaps to cover processing-lag-induced timeline holes.
        // If both agree on same speaker AND gap is < 30s, fill.
        // If only one side within 20s, use it.  Last resort: nearest event.
        static constexpr int64_t kFillBothMaxGap   = 480000;  // 30s
        static constexpr int64_t kFillSingleMaxGap  = 320000;  // 20s

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

        // Last resort: use nearest event regardless of distance.
        // This prevents SNAPSHOT fallback when processing lag creates
        // large timeline gaps (observed >100s in 60-min tests).
        if (nearest_before_spk >= 0 || nearest_after_spk >= 0) {
            ResolvedSpeaker result;
            if (nearest_before_spk >= 0 &&
                (nearest_after_spk < 0 || nearest_before_dist <= nearest_after_dist)) {
                result.speaker_id = nearest_before_spk;
                result.similarity = nearest_before_sim;
                memcpy(result.name, nearest_before_name, 64);
                result.source = nearest_before_src;
            } else {
                result.speaker_id = nearest_after_spk;
                result.similarity = nearest_after_sim;
                memcpy(result.name, nearest_after_name, 64);
                result.source = nearest_after_src;
            }
            result.confidence = 0.05f;  // very low — last resort
            return result;
        }

        return {};
    }

} // namespace deusridet
