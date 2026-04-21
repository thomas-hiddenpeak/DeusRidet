/**
 * @file json_util.h
 * @philosophical_role UTF-8 safe JSON string escaping, shared across every
 *         subsystem that marshals textual content onto the wire. Lives in
 *         Communis because it belongs to every speaker but none exclusively.
 * @serves auditus_facade, conscientia_facade, and any future subsystem that
 *         emits JSON envelopes via WsServer.
 */
#pragma once

#include <cstdint>
#include <string>

namespace deusridet {
namespace communis {

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

}  // namespace communis
}  // namespace deusridet
