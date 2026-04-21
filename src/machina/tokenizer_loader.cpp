/**
 * @file tokenizer_loader.cpp
 * @philosophical_role Peer TU of tokenizer.cpp owning the disk-ingest side of
 *         the BPE pipeline: vocab.json, merges.txt, and the special-token
 *         registration that turns the Qwen3.5 files on disk into the in-memory
 *         tables the rest of Tokenizer assumes. Split out because the main
 *         tokenizer.cpp breached the R1 500-line hard cap at 665 lines, and
 *         file I/O is orthogonal to the pre-tokenisation / BPE / encode /
 *         decode hot path.
 * @serves Machina Tokenizer::load entry, invoked during awaken.
 */
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>

namespace deusridet {

// ---------------------------------------------------------------------------
// Local helpers (duplicated from tokenizer.cpp — kept static-TU-local to avoid
// a shared internal header; the two functions together are ~60 lines).
// ---------------------------------------------------------------------------

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
