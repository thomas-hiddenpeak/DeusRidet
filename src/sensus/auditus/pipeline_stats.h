/**
 * @file pipeline_stats.h
 * @philosophical_role The dashboard of audition — a flat struct mirroring, tick by tick, every dial the pipeline turns. What is measured is what becomes visible; Actus reads this to narrate the entity's hearing.
 * @serves AudioPipeline, Actus (JSON export), Nexus (stats broadcast).
 */
#pragma once

#include <cstdint>

namespace deusridet {

struct AudioPipelineStats {
    // AUDIO business clock (T1) snapshot — sample indices.
    // Derive wall time via tempus::t1_to_t0(Domain::AUDIO, audio_t1_processed).
    uint64_t audio_t1_in;        // samples successfully enqueued into ring buffer
    uint64_t audio_t1_processed; // samples popped by process_loop (authoritative "now")
    uint64_t mel_frames;       // total mel frames computed
    uint64_t speech_frames;    // mel frames classified as speech (energy VAD)
    float    last_rms;         // latest frame RMS (linear)
    float    last_energy;      // latest frame mean log-energy
    bool     is_speech;        // current energy VAD state
    // FRCRN speech enhancement.
    bool     frcrn_active;     // true when FRCRN is processing
    float    frcrn_lat_ms;     // latest FRCRN inference latency (ms)
    // Overlap detection (P1).
    bool     overlap_detected;   // true when overlap detected in current window
    float    overlap_ratio;      // fraction of frames with overlap [0,1]
    float    od_latency_ms;      // OD inference latency (ms)
    // Speech separation (P2).
    bool     separation_active;    // true when separation is running
    float    separation_lat_ms;    // separation latency (ms)
    float    sep_source1_energy;   // RMS energy of separated source 1
    float    sep_source2_energy;   // RMS energy of separated source 2
    // Silero VAD.
    float    silero_prob;      // latest Silero speech probability [0,1]
    bool     silero_speech;    // Silero VAD speech state
    // Speaker identification (CAM++).
    int      speaker_id;       // current speaker ID (-1 = unknown)
    float    speaker_sim;      // best cosine similarity
    bool     speaker_new;      // true if newly registered speaker
    int      speaker_count;    // number of known speakers
    int      speaker_exemplars;  // exemplar count for matched speaker
    int      speaker_hits_above; // exemplars above threshold in this match
    char     speaker_name[64]; // current speaker name (empty if unnamed)
    // Speaker identification (WavLM-Large + ECAPA-TDNN, native GPU).
    int      wlecapa_id;
    float    wlecapa_sim;
    bool     wlecapa_new;
    int      wlecapa_count;
    int      wlecapa_exemplars;      // exemplar count for matched speaker
    int      wlecapa_hits_above;     // exemplars above threshold in this match
    char     wlecapa_name[64];
    // Active flags — true only on the tick when extraction happened.
    bool     speaker_active;
    bool     wlecapa_active;
    // WL-ECAPA latency breakdown (ms), set after each extraction.
    float    wlecapa_lat_cnn_ms;
    float    wlecapa_lat_encoder_ms;
    float    wlecapa_lat_ecapa_ms;
    float    wlecapa_lat_total_ms;
    bool     wlecapa_is_early;       // true if this was an early extraction (not end-of-segment)
    // Change detection: cosine similarity between consecutive segment embeddings.
    float    wlecapa_change_sim;     // -1 if no previous segment
    bool     wlecapa_change_valid;   // true when change_sim is meaningful
    // ASR (Qwen3-ASR).
    bool     asr_active;             // true on tick when new transcript is ready
    bool     asr_busy;               // true when ASR thread is processing
    float    asr_latency_ms;         // transcription latency
    float    asr_audio_duration_s;   // audio segment duration
    float    asr_buf_sec;            // current ASR buffer duration (seconds)
    bool     asr_buf_has_speech;     // buffer contains detected speech
    // SAAS: adaptive post-silence feedback.
    int      asr_effective_silence_ms; // currently effective post-silence threshold
    int      asr_post_silence_ms;    // current accumulated post-silence (ms)
};


} // namespace deusridet
