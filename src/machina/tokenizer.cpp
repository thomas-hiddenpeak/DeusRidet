/**
 * @file tokenizer.cpp
 * @philosophical_role BPE tokenizer. Words become tokens become embeddings become thought; the tokenizer is the entity's first translation from human text to internal state.
 * @serves Machina prefill, ASR decoder output, Cogitatio inputs.
 */
// tokenizer.cpp — GPT2-style Byte-Level BPE Tokenizer implementation
//
// Qwen3.5 tokenizer: vocab.json (248044 base entries) + merges.txt (247587
// merge rules) + 26 added/special tokens (248044–248069).
//
// Byte-Level encoding: each byte maps to a printable Unicode character
// (GPT2 byte_encoder scheme), so all vocab tokens are printable strings.
//
// Adapted from qwen35-orin (src/engine/tokenizer.cpp): BPE algorithm,
// byte-level encoding, pre-tokenization, and ChatML template handling.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstring>

namespace deusridet {

// ============================================================================
// UTF-8 utilities
// ============================================================================

static inline int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static std::string codepoint_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = utf8_char_len(static_cast<unsigned char>(s[i]));
        if (i + len > s.size()) len = 1;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// ============================================================================
// Minimal JSON string unescaping (for vocab.json)
// ============================================================================

static std::string unescape_json_string(const char* start, size_t len) {
    std::string result;
    result.reserve(len);
    for (size_t i = 0; i < len; i++) {
        if (start[i] == '\\' && i + 1 < len) {
            char next = start[i + 1];
            switch (next) {
                case '"':  result += '"';  i++; break;
                case '\\': result += '\\'; i++; break;
                case '/':  result += '/';  i++; break;
                case 'n':  result += '\n'; i++; break;
                case 'r':  result += '\r'; i++; break;
                case 't':  result += '\t'; i++; break;
                case 'b':  result += '\b'; i++; break;
                case 'f':  result += '\f'; i++; break;
                case 'u': {
                    if (i + 5 < len) {
                        char hex[5] = {start[i+2], start[i+3], start[i+4], start[i+5], 0};
                        int cp = (int)strtol(hex, nullptr, 16);
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 11 < len &&
                            start[i+6] == '\\' && start[i+7] == 'u') {
                            char hex2[5] = {start[i+8], start[i+9], start[i+10], start[i+11], 0};
                            int cp2 = (int)strtol(hex2, nullptr, 16);
                            if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                                i += 6;
                            }
                        }
                        result += codepoint_to_utf8(cp);
                        i += 5;
                    }
                    break;
                }
                default: result += start[i]; break;
            }
        } else {
            result += start[i];
        }
    }
    return result;
}

// ============================================================================
// Byte-level encoding (GPT2 byte_encoder)
// ============================================================================

void Tokenizer::init_byte_mapping() {
    // GPT2 byte_encoder: maps each byte (0-255) to a Unicode codepoint
    // Printable bytes (33-126, 161-172, 174-255) map to same codepoint
    // Remaining bytes (0-32, 127-160, 173) map to 256, 257, ...

    std::vector<int> bs, cs;

    for (int b = 33; b <= 126; b++)  { bs.push_back(b); cs.push_back(b); }
    for (int b = 161; b <= 172; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 174; b <= 255; b++) { bs.push_back(b); cs.push_back(b); }

    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool found = false;
        for (int x : bs) {
            if (x == b) { found = true; break; }
        }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        std::string utf8 = codepoint_to_utf8(cs[i]);
        byte_to_unicode_[bs[i]] = utf8;
        unicode_to_byte_[utf8] = static_cast<uint8_t>(bs[i]);
    }
}

std::string Tokenizer::byte_encode(const std::string& utf8_text) const {
    std::string result;
    for (unsigned char b : utf8_text) {
        result += byte_to_unicode_[b];
    }
    return result;
}

std::string Tokenizer::byte_decode(const std::string& piece) const {
    std::string result;
    size_t i = 0;
    while (i < piece.size()) {
        int len = utf8_char_len(static_cast<unsigned char>(piece[i]));
        if (i + len > piece.size()) break;
        std::string ch = piece.substr(i, len);
        auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        }
        i += len;
    }
    return result;
}

// ============================================================================
// Pre-tokenization (simplified GPT2 regex split)
// ============================================================================

static bool is_ascii_letter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Newlines
        if (c == '\n' || c == '\r') {
            std::string piece;
            while (i < n && (text[i] == '\n' || text[i] == '\r')) {
                piece += text[i++];
            }
            pieces.push_back(piece);
            continue;
        }

        // Whitespace (space/tab)
        if (c == ' ' || c == '\t') {
            if (i + 1 < n) {
                unsigned char next = static_cast<unsigned char>(text[i + 1]);
                // Space + ASCII letter → single piece
                if (is_ascii_letter(next)) {
                    std::string piece;
                    piece += text[i++];
                    while (i < n && is_ascii_letter(text[i])) {
                        piece += text[i++];
                    }
                    // Check for contraction ('s, 't, etc.)
                    if (i < n && text[i] == '\'') {
                        pieces.push_back(piece);
                        std::string contr = "'";
                        size_t j = i + 1;
                        while (j < n && is_ascii_letter(text[j]) && j - i <= 3) {
                            contr += text[j++];
                        }
                        std::string lower = contr.substr(1);
                        for (auto& ch : lower) ch = tolower(ch);
                        if (lower == "s" || lower == "t" || lower == "re" ||
                            lower == "ve" || lower == "m" || lower == "ll" ||
                            lower == "d") {
                            pieces.push_back(contr);
                            i = j;
                        }
                    } else {
                        pieces.push_back(piece);
                    }
                    continue;
                }
                // Space + non-ASCII
                if (next >= 0x80) {
                    std::string piece;
                    piece += text[i++];
                    while (i < n && static_cast<unsigned char>(text[i]) >= 0x80) {
                        int len = utf8_char_len(static_cast<unsigned char>(text[i]));
                        if (i + len > n) break;
                        piece.append(text, i, len);
                        i += len;
                    }
                    pieces.push_back(piece);
                    continue;
                }
                // Space + digit
                if (isdigit(next)) {
                    pieces.push_back(std::string(1, text[i++]));
                    continue;
                }
                // Space + punctuation
                if (!isspace(next)) {
                    std::string piece;
                    piece += text[i++];
                    while (i < n && !isspace(text[i]) && !is_ascii_letter(text[i]) &&
                           !isdigit(static_cast<unsigned char>(text[i])) &&
                           static_cast<unsigned char>(text[i]) < 0x80) {
                        piece += text[i++];
                    }
                    pieces.push_back(piece);
                    continue;
                }
            }
            // Trailing whitespace
            std::string piece;
            while (i < n && (text[i] == ' ' || text[i] == '\t')) {
                piece += text[i++];
            }
            pieces.push_back(piece);
            continue;
        }

        // Digits (single)
        if (isdigit(c)) {
            pieces.push_back(std::string(1, text[i++]));
            continue;
        }

        // ASCII letter sequences
        if (is_ascii_letter(c)) {
            std::string piece;
            while (i < n && is_ascii_letter(text[i])) {
                piece += text[i++];
            }
            // Check contraction
            if (i < n && text[i] == '\'') {
                std::string contr = "'";
                size_t j = i + 1;
                while (j < n && is_ascii_letter(text[j]) && j - i <= 3) {
                    contr += text[j++];
                }
                std::string lower = contr.substr(1);
                for (auto& ch : lower) ch = tolower(ch);
                if (lower == "s" || lower == "t" || lower == "re" ||
                    lower == "ve" || lower == "m" || lower == "ll" ||
                    lower == "d") {
                    pieces.push_back(piece);
                    pieces.push_back(contr);
                    i = j;
                    continue;
                }
            }
            pieces.push_back(piece);
            continue;
        }

        // Non-ASCII UTF-8 character sequences
        if (c >= 0x80) {
            std::string piece;
            while (i < n && static_cast<unsigned char>(text[i]) >= 0x80) {
                int len = utf8_char_len(static_cast<unsigned char>(text[i]));
                if (i + len > n) break;
                piece.append(text, i, len);
                i += len;
            }
            if (!piece.empty()) pieces.push_back(piece);
            continue;
        }

        // Other ASCII (punctuation/symbols)
        pieces.push_back(std::string(1, text[i++]));
    }

    return pieces;
}

// ============================================================================
// BPE merge on a single byte-encoded piece
// ============================================================================

std::vector<int> Tokenizer::bpe(const std::string& piece) const {
    std::vector<std::string> symbols = utf8_chars(piece);

    if (symbols.empty()) return {};
    if (symbols.size() == 1) {
        auto it = piece_to_id_.find(symbols[0]);
        if (it != piece_to_id_.end()) return {it->second};
        return {};
    }

    while (symbols.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (size_t j = 0; j + 1 < symbols.size(); j++) {
            std::string pair_key = symbols[j] + " " + symbols[j + 1];
            auto it = merges_.find(pair_key);
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_idx < 0) break;

        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto it = piece_to_id_.find(sym);
        if (it != piece_to_id_.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

// ============================================================================
// Encode
// ============================================================================

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!loaded_ || text.empty()) return {};

    std::vector<int> all_ids;

    // Split on special tokens (greedy: longest first)
    struct Segment {
        std::string text;
        bool is_special;
        int special_id;
    };
    std::vector<Segment> segments;

    size_t pos = 0;
    while (pos < text.size()) {
        bool found = false;
        for (const auto& st : special_token_list_) {
            if (pos + st.size() <= text.size() &&
                text.compare(pos, st.size(), st) == 0) {
                segments.push_back({st, true, special_tokens_.at(st)});
                pos += st.size();
                found = true;
                break;
            }
        }
        if (!found) {
            size_t start = pos;
            size_t next = std::string::npos;
            for (const auto& st : special_token_list_) {
                size_t f = text.find(st, pos);
                if (f != std::string::npos && f < next) {
                    next = f;
                }
            }
            size_t end = (next != std::string::npos) ? next : text.size();
            if (end > start) {
                segments.push_back({text.substr(start, end - start), false, -1});
            }
            pos = end;
        }
    }

    for (const auto& seg : segments) {
        if (seg.is_special) {
            all_ids.push_back(seg.special_id);
            continue;
        }

        auto pieces = pre_tokenize(seg.text);
        for (const auto& piece : pieces) {
            std::string encoded = byte_encode(piece);
            auto ids = bpe(encoded);
            all_ids.insert(all_ids.end(), ids.begin(), ids.end());
        }
    }

    return all_ids;
}

// ============================================================================
// Decode
// ============================================================================

std::string Tokenizer::decode(int token_id) const {
    if (!loaded_) return "";
    if (token_id < 0 || token_id >= static_cast<int>(id_to_piece_.size())) return "";

    const std::string& piece = id_to_piece_[token_id];
    if (special_tokens_.count(piece) > 0) return piece;

    return byte_decode(piece);
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) {
        result += decode(id);
    }
    return result;
}

// ============================================================================
// Chat template (Qwen3 ChatML)
// ============================================================================

std::vector<int> Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt,
    bool enable_thinking) const
{
    std::string prompt;
    for (const auto& [role, content] : messages) {
        if (role == "tool") {
            prompt += "<|im_start|>user\n<tool_response>\n" + content +
                      "\n</tool_response><|im_end|>\n";
        } else {
            prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
    }
    if (add_generation_prompt) {
        if (enable_thinking) {
            prompt += "<|im_start|>assistant\n<think>\n";
        } else {
            prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        }
    }
    return encode(prompt);
}

// ============================================================================
// Load
// ============================================================================

bool Tokenizer::load(const std::string& model_dir) {
    init_byte_mapping();

    std::string vocab_path  = model_dir + "/vocab.json";
    std::string merges_path = model_dir + "/merges.txt";

    if (!load_vocab(vocab_path)) {
        fprintf(stderr, "[Tokenizer] Failed to load vocab: %s\n", vocab_path.c_str());
        return false;
    }
    if (!load_merges(merges_path)) {
        fprintf(stderr, "[Tokenizer] Failed to load merges: %s\n", merges_path.c_str());
        return false;
    }

    // Register Qwen3.5 special tokens
    struct SpecialDef { const char* content; int fallback_id; };
    SpecialDef specials[] = {
        {"<|endoftext|>",         248044},
        {"<|im_start|>",          248045},
        {"<|im_end|>",            248046},
        {"<|object_ref_start|>",  248047},
        {"<|object_ref_end|>",    248048},
        {"<|box_start|>",         248049},
        {"<|box_end|>",           248050},
        {"<|quad_start|>",        248051},
        {"<|quad_end|>",          248052},
        {"<|vision_start|>",      248053},
        {"<|vision_end|>",        248054},
        {"<|vision_pad|>",        248055},
        {"<|image_pad|>",         248056},
        {"<|video_pad|>",         248057},
        {"<tool_call>",           248058},
        {"</tool_call>",          248059},
        {"<|fim_prefix|>",        248060},
        {"<|fim_middle|>",        248061},
        {"<|fim_suffix|>",        248062},
        {"<|fim_pad|>",           248063},
        {"<|repo_name|>",         248064},
        {"<|file_sep|>",          248065},
        {"<tool_response>",       248066},
        {"</tool_response>",      248067},
        {"<think>",               248068},
        {"</think>",              248069},
    };

    for (const auto& sp : specials) {
        auto it = piece_to_id_.find(sp.content);
        int id = (it != piece_to_id_.end()) ? it->second : sp.fallback_id;

        special_tokens_[sp.content] = id;
        special_token_list_.push_back(sp.content);

        if (id >= static_cast<int>(id_to_piece_.size())) {
            id_to_piece_.resize(id + 1);
        }
        id_to_piece_[id] = sp.content;
        piece_to_id_[sp.content] = id;
    }

    std::sort(special_token_list_.begin(), special_token_list_.end(),
              [](const std::string& a, const std::string& b) {
                  return a.size() > b.size();
              });

    eot_id_             = special_tokens_["<|endoftext|>"];
    im_start_id_        = special_tokens_["<|im_start|>"];
    im_end_id_          = special_tokens_["<|im_end|>"];
    eos_id_             = im_end_id_;
    think_start_id_     = special_tokens_["<think>"];
    think_end_id_       = special_tokens_["</think>"];
    tool_call_start_id_ = special_tokens_["<tool_call>"];
    tool_call_end_id_   = special_tokens_["</tool_call>"];

    loaded_ = true;
    fprintf(stderr, "[Tokenizer] Loaded: vocab=%d, merges=%d, specials=%d\n",
            static_cast<int>(piece_to_id_.size()),
            static_cast<int>(merges_.size()),
            static_cast<int>(special_tokens_.size()));
    return true;
}

// ============================================================================
// Load vocab.json
// ============================================================================

bool Tokenizer::load_vocab(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    ifs.read(&content[0], file_size);

    int max_id = 0;
    const char* p = content.c_str();
    const char* end = p + content.size();

    while (p < end && *p != '{') p++;
    if (p < end) p++;

    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ','))
            p++;
        if (p >= end || *p == '}') break;

        if (*p != '"') { p++; continue; }
        p++;

        const char* key_start = p;
        while (p < end) {
            if (*p == '"') {
                int bs_count = 0;
                const char* bp = p - 1;
                while (bp >= key_start && *bp == '\\') { bs_count++; bp--; }
                if (bs_count % 2 == 0) break;
            }
            p++;
        }
        size_t key_len = p - key_start;
        std::string key = unescape_json_string(key_start, key_len);
        if (p < end) p++;

        while (p < end && *p != ':') p++;
        if (p < end) p++;
        while (p < end && (*p == ' ' || *p == '\t')) p++;

        int value = 0;
        bool negative = false;
        if (p < end && *p == '-') { negative = true; p++; }
        while (p < end && *p >= '0' && *p <= '9') {
            value = value * 10 + (*p - '0');
            p++;
        }
        if (negative) value = -value;

        piece_to_id_[key] = value;
        if (value > max_id) max_id = value;
    }

    id_to_piece_.resize(max_id + 1);
    for (const auto& [piece, id] : piece_to_id_) {
        if (id >= 0 && id < static_cast<int>(id_to_piece_.size())) {
            id_to_piece_[id] = piece;
        }
    }

    return !piece_to_id_.empty();
}

// ============================================================================
// Load merges.txt
// ============================================================================

bool Tokenizer::load_merges(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return false;

    std::string line;
    int rank = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        merges_[line] = rank++;
    }

    return !merges_.empty();
}

} // namespace deusridet
