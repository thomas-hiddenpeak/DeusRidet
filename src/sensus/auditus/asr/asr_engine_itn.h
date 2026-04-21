/**
 * @file asr_engine_itn.h
 * @philosophical_role Internal ITN seam — declarations shared between asr_engine.cpp
 *         (the transcribe loop that calls ITN on its output) and asr_engine_itn.cpp
 *         (the peer TU that owns the Chinese-numeral → Arabic conversion and the
 *         small repeat collapser). Not exported outside the asr subfolder.
 * @serves asr_engine.cpp (caller), asr_engine_itn.cpp (implementation).
 */
#pragma once

#include <string>

namespace deusridet {
namespace asr {

// Replace Chinese numeral sequences in `text` with Arabic digits.
std::string apply_itn(const std::string& text);

// Collapse short repeated substrings (stutter suppression).
std::string collapse_repeats(const std::string& text);

} // namespace asr
} // namespace deusridet
