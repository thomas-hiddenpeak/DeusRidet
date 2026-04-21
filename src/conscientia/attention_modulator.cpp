/**
 * @file attention_modulator.cpp
 * @philosophical_role Pulsed attention budget allocator. Consciousness is not constant-throughput — it is rhythmic: Prefill pulses are spaced to leave Decode room to think.
 * @serves ConscientiaStream::tick() and the Prefill/Decode budget split.
 */
// attention_modulator.cpp — Wakefulness modulation implementation
//
// Computes wakefulness boost from input signals.
// Does NOT decide whether to respond — that's the model's job via probe decode.

#include "attention_modulator.h"
#include "../communis/log.h"

namespace deusridet {

void AttentionModulator::init(const PersonaConfig& persona) {
    persona_ = persona;
    LOG_INFO("AttentionMod", "Entity: %s, aliases: %zu, boosts: text=%.1f name=%.2f asr=%.2f",
             persona_.name.c_str(), persona_.aliases.size(),
             boosts_.text_input, boosts_.name_detected, boosts_.asr_activity);
}

float AttentionModulator::compute_boost(const std::vector<InputItem>& inputs) const {
    float max_boost = 0.0f;

    for (const auto& item : inputs) {
        float boost = 0.0f;

        switch (item.source) {
        case InputSource::TEXT:
            // Direct text input — explicit addressing, max wakefulness
            boost = boosts_.text_input;
            break;

        case InputSource::ASR:
            // Speech detected — check for name (cocktail party effect)
            if (contains_name(item.text)) {
                boost = boosts_.name_detected;
                LOG_DEBUG("AttentionMod", "Name detected in ASR: [%.40s...]",
                          item.text.c_str());
            } else {
                boost = boosts_.asr_activity;
            }
            break;

        case InputSource::THOUGHT:
        case InputSource::MEMORY:
        case InputSource::SYSTEM:
        case InputSource::VISION:
            // Internal sources don't boost wakefulness
            break;
        }

        max_boost = std::max(max_boost, boost);
    }

    return max_boost;
}

bool AttentionModulator::has_direct_text(const std::vector<InputItem>& inputs) const {
    for (const auto& item : inputs) {
        if (item.source == InputSource::TEXT) return true;
    }
    return false;
}

bool AttentionModulator::contains_name(const std::string& text) const {
    if (!persona_.name.empty() && text.find(persona_.name) != std::string::npos) {
        return true;
    }
    for (const auto& alias : persona_.aliases) {
        if (!alias.empty() && text.find(alias) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace deusridet
