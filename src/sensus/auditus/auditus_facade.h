/**
 * @file auditus_facade.h
 * @philosophical_role The bridge between Auditus (hearing) and Nexus (voice to the outer world).
 *         Owns every audio-derived broadcast and every client message that mutates hearing state.
 *         Facade installs — it does not own. AudioPipeline and WsServer lifetimes remain with the
 *         caller (cmd_test_ws); facade only wires callbacks across the seam.
 * @serves cmd_test_ws and any future Actus verb that needs to expose Auditus over a WS channel.
 */
#pragma once

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

// ── JSON helpers (Auditus-local) ──────────────────────────────────────────────
// These are used by every callback that marshals ASR/VAD text into a WS frame.
// Kept inline so both the facade TU and cmd_test_ws (for callbacks not yet
// migrated) can share one definition with zero link-time coupling.

// Strip a trailing incomplete UTF-8 sequence from a byte string.
inline std::string sanitize_utf8(const std::string& s) {
    if (s.empty()) return s;
    size_t i = 0, last_good = 0;
    while (i < s.size()) {
        uint8_t c = (uint8_t)s[i];
        int expect;
        if (c < 0x80)       expect = 1;
        else if (c < 0xC0)  { i++; continue; }
        else if (c < 0xE0)  expect = 2;
        else if (c < 0xF0)  expect = 3;
        else if (c < 0xF8)  expect = 4;
        else                { i++; continue; }
        if (i + expect > s.size()) break;
        bool ok = true;
        for (int j = 1; j < expect; j++) {
            if (((uint8_t)s[i + j] & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) { i++; continue; }
        i += expect;
        last_good = i;
    }
    return s.substr(0, last_good);
}

// Escape a UTF-8 string for JSON, dropping invalid bytes and control chars.
inline std::string json_escape(const std::string& raw) {
    std::string clean = sanitize_utf8(raw);
    std::string out;
    out.reserve(clean.size() + 32);
    for (unsigned char c : clean) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20)  { /* drop control chars */ }
        else                out += (char)c;
    }
    return out;
}

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

// ASR detail log (wrapped as "asr_log") → ws broadcast only.
void install_asr_log_callback(AudioPipeline& audio,
                              WsServer& server);

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
// periodic stdout summary in cmd_test_ws working with the same counter.
void install_ws_binary_callback(WsServer& server,
                                AudioPipeline& audio,
                                std::atomic<uint64_t>& total_frames,
                                std::atomic<uint64_t>& total_bytes,
                                std::atomic<bool>& loopback);

}  // namespace auditus
}  // namespace deusridet
