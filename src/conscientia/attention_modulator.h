/**
 * @file attention_modulator.h
 * @philosophical_role Declaration of the attention-budget interface. Defines who decides how the next 100 ms of compute is split between perception and thought.
 * @serves ConscientiaStream, Machina forward scheduler.
 */
// attention_modulator.h — Wakefulness modulation based on input significance
//
// The AttentionModulator does NOT decide whether to respond.
// That decision belongs to the model itself (via probe decode).
//
// This module adjusts the entity's wakefulness level based on input signals:
//   - Name/alias detected in ASR → large boost (cocktail party effect)
//   - Background ASR activity    → small boost (someone is talking)
//   - Direct TEXT input           → max wakefulness (explicit addressing)
//
// Higher wakefulness → probe decode happens → model decides speak or silence.
// Lower wakefulness → no probe, pure perception (prefill only, save GPU).

#pragma once

#include "frame.h"
#include "../communis/config.h"
#include <string>
#include <vector>

namespace deusridet {

// Wakefulness boost amounts for different input types
struct AttentionBoosts {
    float text_input      = 1.0f;   // TEXT → slam to max (explicit addressing)
    float name_detected   = 0.5f;   // name heard in ASR (cocktail party effect)
    float asr_activity    = 0.15f;  // background speech activity
};

class AttentionModulator {
public:
    AttentionModulator() = default;

    void init(const PersonaConfig& persona);

    // Compute the wakefulness boost for a batch of input items.
    // Returns the maximum boost across all items (boosts don't stack additively
    // across items in the same batch — take the most significant signal).
    float compute_boost(const std::vector<InputItem>& inputs) const;

    // Check if any input item is direct TEXT (bypasses probe, always decode)
    bool has_direct_text(const std::vector<InputItem>& inputs) const;

    const AttentionBoosts& boosts() const { return boosts_; }

    // Check if text contains the entity's name or aliases
    bool contains_name(const std::string& text) const;

private:
    PersonaConfig persona_;
    AttentionBoosts boosts_;
};

} // namespace deusridet
