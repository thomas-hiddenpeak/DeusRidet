/**
 * @file speaker_tracking.h
 * @philosophical_role The entity's memory of who-spoke-when. SpeakerTimeline is
 *   the event-sourced ledger that fuses SAAS branches (EARLY/FULL/CHANGE/INHERIT)
 *   into a single weighted verdict per audio span. The historical `SpeakerTracker`
 *   class (independent 500 ms tick-based A/B pipeline) was removed after test
 *   runs confirmed it contributed 0 labels under the dual-encoder path while
 *   adding ~1 100 lines of surface area and cognitive load.
 * @serves AudioPipeline speaker resolution, ASR label attribution.
 */
#pragma once

#include <cstdint>
#include <cstring>

namespace deusridet {

// ─── SpeakerTimeline: fused speaker resolution via event sourcing ───
//
// All SAAS speaker identification branches (early/full/change/inherit) write
// timestamped events to a shared timeline. When an ASR segment needs a speaker
// label, resolve() queries the timeline for the best label covering the audio
// range via weighted majority voting.
//
// Authority hierarchy: SAAS_FULL > SAAS_CHANGE > SAAS_EARLY > SAAS_INHERIT.

enum class SpkEventSource : uint8_t {
    SAAS_EARLY   = 0,  // SAAS early extraction (auto_register=false)
    SAAS_FULL    = 1,  // SAAS end-of-segment (auto_register=true)
    SAAS_CHANGE  = 2,  // SAAS intra-segment speaker change detection
    SAAS_INHERIT = 3,  // SAAS inheritance from previous segment
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
    SpkEventSource source = SpkEventSource::SAAS_EARLY;
};

class SpeakerTimeline {
public:
    static constexpr int kMaxEvents = 2000;

    void push(const SpeakerEvent& ev) {
        events_[write_pos_] = ev;
        write_pos_ = (write_pos_ + 1) % kMaxEvents;
        if (count_ < kMaxEvents) ++count_;
    }

    // See speaker_tracking.cpp for the definition.
    ResolvedSpeaker resolve(int64_t start_sample, int64_t end_sample) const;


    int event_count() const { return count_; }
    void clear() { count_ = 0; write_pos_ = 0; }

private:
    SpeakerEvent events_[kMaxEvents];
    int write_pos_ = 0;
    int count_ = 0;
};

} // namespace deusridet
