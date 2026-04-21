/**
 * @file asr_engine_itn.cpp
 * @philosophical_role Peer TU of asr_engine.cpp owning Inverse Text Normalization —
 *         Chinese numeral → Arabic conversion and simple stutter collapsing. Split
 *         out under R1 because the ITN block (~130 lines) pushed asr_engine.cpp
 *         over the 500-line .cpp hard cap; the rest of the file (weight loading,
 *         prompt building, autoregressive decode) is more cohesive without it.
 * @serves asr_engine.cpp via asr_engine_itn.h.
 */
#include "asr_engine_itn.h"

#include <cstring>
#include <cstdint>
#include <regex>

namespace deusridet {
namespace asr {

// =========================================================================
// ITN (Inverse Text Normalization) — Chinese numerals → Arabic
// =========================================================================

static const char* cn_digits[] = {
    "零","一","二","三","四","五","六","七","八","九"
};

// Chinese numeral string → integer. Returns -1 on failure.
static int64_t cn_to_arabic(const std::string& s) {
    // Single-char digit
    for (int i = 0; i < 10; i++) {
        if (s == cn_digits[i]) return i;
    }

    int64_t result = 0;
    int64_t current = 0;
    int64_t high_unit = 0;

    auto starts_with = [](const std::string& str, size_t pos, const char* prefix) -> bool {
        size_t plen = std::strlen(prefix);
        return pos + plen <= str.size() && str.compare(pos, plen, prefix) == 0;
    };

    size_t i = 0;
    while (i < s.size()) {
        // Check digit
        int digit = -1;
        for (int d = 0; d < 10; d++) {
            if (starts_with(s, i, cn_digits[d])) {
                digit = d;
                i += std::strlen(cn_digits[d]);
                break;
            }
        }
        if (digit >= 0) {
            current = digit;
            continue;
        }

        // Check units
        if (starts_with(s, i, "十")) {
            current = (current == 0) ? 10 : current * 10;
            i += 3; // UTF-8 "十" = 3 bytes
        } else if (starts_with(s, i, "百")) {
            current *= 100;
            i += 3;
        } else if (starts_with(s, i, "千")) {
            current *= 1000;
            i += 3;
        } else if (starts_with(s, i, "万")) {
            if (current == 0 && result > 0) {
                result *= 10000;
            } else {
                result = (result + current) * 10000;
                current = 0;
            }
            high_unit = 10000;
            i += 3;
        } else if (starts_with(s, i, "亿")) {
            result = (result + current) * 100000000LL;
            current = 0;
            high_unit = 100000000LL;
            i += 3;
        } else {
            return -1; // unrecognized
        }
        result += current;
        current = 0;
    }
    result += current;
    return result;
}

// Apply basic ITN: replace Chinese numeral patterns with Arabic numerals
std::string apply_itn(const std::string& text) {
    // Pattern: sequences of Chinese numeral chars
    static const std::regex cn_num_re(
        "([零一二三四五六七八九十百千万亿]+)");

    std::string result;
    std::sregex_iterator it(text.begin(), text.end(), cn_num_re);
    std::sregex_iterator end;
    size_t last = 0;

    while (it != end) {
        result += text.substr(last, it->position() - last);
        std::string cn = it->str();
        int64_t val = cn_to_arabic(cn);
        if (val >= 0) {
            result += std::to_string(val);
        } else {
            result += cn;  // keep original if parse fails
        }
        last = it->position() + it->length();
        ++it;
    }
    result += text.substr(last);
    return result;
}

// Collapse repeated phrases (simple: consecutive identical substrings)
std::string collapse_repeats(const std::string& text) {
    // Simple character-level repeat: "ABABAB" → "AB"
    // Only for very short repeated units (≤8 chars) repeated ≥3 times
    std::string result = text;
    for (int unit_len = 1; unit_len <= 8 && unit_len * 3 <= (int)result.size(); unit_len++) {
        size_t pos = 0;
        while (pos + unit_len * 3 <= result.size()) {
            std::string unit = result.substr(pos, unit_len);
            int count = 1;
            size_t check = pos + unit_len;
            while (check + unit_len <= result.size() &&
                   result.substr(check, unit_len) == unit) {
                count++;
                check += unit_len;
            }
            if (count >= 3) {
                // Replace N copies with 1
                result = result.substr(0, pos + unit_len) +
                         result.substr(pos + unit_len * count);
            } else {
                pos += unit_len;
            }
        }
    }
    return result;
}

} // namespace asr
} // namespace deusridet
