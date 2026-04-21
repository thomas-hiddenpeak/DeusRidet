/**
 * @file auditus_facade.h
 * @philosophical_role The bridge between Auditus (hearing) and Nexus (voice to the outer world).
 *         Owns every audio-derived broadcast and every client message that mutates hearing state.
 *         Facade installs — it does not own. AudioPipeline and WsServer lifetimes remain with the
 *         caller (cmd_test_ws); facade only wires callbacks across the seam.
 * @serves cmd_test_ws and any future Actus verb that needs to expose Auditus over a WS channel.
 */
#pragma once

#include <cstdint>
#include <string>

namespace deusridet {

// Forward declarations — avoid dragging every audio/ws header into translation units
// that only need the facade surface.
class AudioPipeline;
class WsServer;
class TimelineLogger;

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

}  // namespace auditus
}  // namespace deusridet
