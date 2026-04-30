/**
 * @file auditus_facade.h
 * @philosophical_role The bridge between Auditus (hearing) and Nexus (voice to the outer world).
 *         Owns every audio-derived broadcast and every client message that mutates hearing state.
 *         Facade installs — it does not own. AudioPipeline and WsServer lifetimes remain with the
 *         caller (awaken); facade only wires callbacks across the seam.
 * @serves awaken and any future Actus verb that needs to expose Auditus over a WS channel.
 */
#pragma once

#include "communis/json_util.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace deusridet {

// Forward declarations — avoid dragging every audio/ws header into translation units
// that only need the facade surface.
class AudioPipeline;
class WsServer;
class TimelineLogger;
class ConscientiStream;

namespace auditus {

// ── JSON helpers ──────────────────────────────────────────────────────────────
// Canonical home is communis::json_util; re-exported into auditus:: so that the
// many existing `auditus::json_escape(...)` call sites stay untouched.
using communis::sanitize_utf8;
using communis::json_escape;

// ── Callback installers (Step 7a scope) ───────────────────────────────────────
// Each installer wires one AudioPipeline output onto a WS broadcast (and, where
// relevant, the timeline log). Callbacks run on the audio/ASR worker thread;
// the facade adds no threading of its own.

// VAD events → ws "vad" envelope + timeline log_vad.
void install_vad_callback(AudioPipeline& audio,
                          WsServer& server,
                          TimelineLogger& timeline);

// Streaming partial ASR text → ws "asr_partial" envelope (no timeline log).
void install_asr_partial_callback(AudioPipeline& audio,
                                  WsServer& server);

// Ring-buffer drops → ws "audio_drop" envelope + timeline log_drop.
void install_drop_callback(AudioPipeline& audio,
                           WsServer& server,
                           TimelineLogger& timeline);

// ── Callback installers (Step 7b scope) ───────────────────────────────────────

// Full ASR transcript → ws "asr_transcript" envelope + timeline log_asr +
// stdout trace + optional injection into the consciousness input stream.
// `llm_loaded` is a snapshot of whether a language model is available at the
// moment of install; the transcript→consciousness injection is gated on it.
void install_transcript_callback(AudioPipeline& audio,
                                 WsServer& server,
                                 TimelineLogger& timeline,
                                 ConscientiStream& consciousness,
                                 bool llm_loaded);

// ASR detail log (wrapped as "asr_log") → ws broadcast; fusion shadow logs persist to timeline.
void install_asr_log_callback(AudioPipeline& audio,
                              WsServer& server,
                              TimelineLogger& timeline);

// Per-tick audio-pipeline stats → ws "pipeline_stats" envelope + timeline log_stats.
// This is the largest single broadcast in the system — it carries VAD, ASR, speaker,
// tracker, overlap-detection, separation, and multi-speaker fusion state.
void install_stats_callback(AudioPipeline& audio,
                            WsServer& server,
                            TimelineLogger& timeline);

// One-shot speaker match (Legacy CAM++ path) → ws "speaker" envelope + stdout.
void install_speaker_match_callback(AudioPipeline& audio,
                                    WsServer& server);

// ── Callback installers (Step 7c scope) ───────────────────────────────────────

// Binary WS frames (16-bit PCM) → AudioPipeline ingress + periodic ws
// "audio_stats" envelopes + optional loopback echo + periodic stdout trace.
// `total_frames`, `total_bytes`, `loopback` are owned by the caller; the
// facade only reads/updates them via the references — this keeps the
// periodic stdout summary in awaken working with the same counter.
void install_ws_binary_callback(WsServer& server,
                                AudioPipeline& audio,
                                std::atomic<uint64_t>& total_frames,
                                std::atomic<uint64_t>& total_bytes,
                                std::atomic<bool>& loopback);

}  // namespace auditus
}  // namespace deusridet
