// frame.h — Consciousness frame definition
//
// A consciousness frame is one "pulse" of the Prefill engine.
// It merges accumulated inputs (sensory, internal thought, memory retrieval)
// into a token sequence that gets prefilled through the model.
//
// The output of prefill produces updated KV cache state that decode
// branches then use to generate responses, thoughts, or daydreams.

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace deusridet {

// ============================================================================
// Input source types
// ============================================================================

enum class InputSource : uint8_t {
    TEXT     = 0,   // WebSocket text input from user
    ASR      = 1,   // Speech-to-text from audio pipeline
    THOUGHT  = 2,   // Internal thought from previous decode branch
    VISION   = 3,   // Vision features (future)
    MEMORY   = 4,   // Long-term memory retrieval
    SYSTEM   = 5,   // System prompt / context injection
};

// ============================================================================
// Input item — one contribution to a consciousness frame
// ============================================================================

struct InputItem {
    InputSource source;
    std::string text;           // tokenizable text content
    std::string speaker_name;   // for ASR: who said it
    int speaker_id = -1;
    float priority = 1.0f;     // higher = more important (affects ordering)
    std::chrono::steady_clock::time_point timestamp;
};

// ============================================================================
// Consciousness frame — assembled from accumulated inputs
// ============================================================================

struct ConscientiFrame {
    uint64_t frame_id = 0;      // monotonic frame counter

    // Merged inputs for this frame
    std::vector<InputItem> inputs;

    // Tokenized content (assembled from inputs)
    std::vector<int> token_ids;
    int num_tokens = 0;

    // Position in the global sequence
    int pos_start = 0;          // first token position in KV cache

    // Timing
    std::chrono::steady_clock::time_point assembled_at;
    float prefill_ms = 0.0f;   // prefill duration after processing

    // Wakefulness state at frame assembly time
    float wakefulness = 0.5f;   // 0.0 = deep dream, 1.0 = fully alert

    // Whether this frame contains external input (triggers action/speech decode)
    bool has_external_input() const {
        for (const auto& item : inputs) {
            if (item.source == InputSource::TEXT ||
                item.source == InputSource::ASR) return true;
        }
        return false;
    }
};

} // namespace deusridet
