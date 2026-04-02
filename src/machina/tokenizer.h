// tokenizer.h — GPT2-style Byte-Level BPE Tokenizer for Qwen3.5
//
// Loads vocab.json + merges.txt from model directory.
// Encoding: text → NFC → regex split → byte-level encode → BPE → vocab lookup
// Decoding: token_id → vocab lookup → byte-level decode → UTF-8 text
//
// Adapted from qwen35-orin (src/engine/tokenizer.h): tokenizer interface and
// BPE algorithm adapted for DeusRidet's consciousness-centric design.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace deusridet {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load vocab.json + merges.txt from model directory
    bool load(const std::string& model_dir);
    bool is_loaded() const { return loaded_; }

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text) const;

    // Decode token ID(s) to text
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    // Qwen3 ChatML template
    std::vector<int> apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = true) const;

    // Special token IDs
    int eos_token_id()   const { return eos_id_; }
    int im_start_id()    const { return im_start_id_; }
    int im_end_id()      const { return im_end_id_; }
    int eot_id()         const { return eot_id_; }
    int think_start_id() const { return think_start_id_; }
    int think_end_id()   const { return think_end_id_; }
    int tool_call_start_id() const { return tool_call_start_id_; }
    int tool_call_end_id()   const { return tool_call_end_id_; }
    int vocab_size()     const { return static_cast<int>(id_to_piece_.size()); }

private:
    void init_byte_mapping();
    std::string byte_encode(const std::string& utf8_text) const;
    std::string byte_decode(const std::string& piece) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    std::vector<int> bpe(const std::string& piece) const;
    bool load_vocab(const std::string& path);
    bool load_merges(const std::string& path);

    bool loaded_ = false;

    // Vocabulary: piece ↔ id
    std::unordered_map<std::string, int> piece_to_id_;
    std::vector<std::string> id_to_piece_;

    // Special/added tokens (matched before BPE)
    std::unordered_map<std::string, int> special_tokens_;
    std::vector<std::string> special_token_list_;  // sorted by length desc

    // BPE merge rules: "a b" → rank (lower = higher priority)
    std::unordered_map<std::string, int> merges_;

    // GPT2 byte-level mapping
    std::string byte_to_unicode_[256];
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    // Special token IDs
    int eos_id_             = -1;
    int im_start_id_        = -1;
    int im_end_id_          = -1;
    int eot_id_             = -1;
    int think_start_id_     = -1;
    int think_end_id_       = -1;
    int tool_call_start_id_ = -1;
    int tool_call_end_id_   = -1;
};

} // namespace deusridet
