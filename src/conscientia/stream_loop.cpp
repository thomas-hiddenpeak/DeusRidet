/**
 * @file stream_loop.cpp
 * @philosophical_role Peer TU of stream.cpp carrying the tick of consciousness
 *         itself: the continuous `consciousness_loop()` driver and the input
 *         `drain_and_tokenize()` routine that turns raw perception into a
 *         token stream. Split out because stream.cpp breached the R1 500-line
 *         hard cap at 836 lines, and the loop is naturally separable from the
 *         engine-facing methods (prefill_tokens, decode_step, prefill_system_prompt)
 *         which it calls but does not implement.
 * @serves Actus::awaken (starts the thread), Nexus (observes its state).
 */
#include "stream.h"
#include "attention_modulator.h"
#include "../machina/tokenizer.h"
#include "../machina/layer.h"
#include "../communis/log.h"
#include "../communis/tempus.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace deusridet {

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<float, std::milli>;

// ============================================================================
// Main consciousness loop
//
// This is the heart of the entity. It runs continuously:
//   - In ACTIVE state: prefill + probe-based selective decode
//   - In DAYDREAM state: decode freely (reflect, organize)
//   - In DREAMING state: decode deeply (consolidate, create)
//
// Probe decode flow (ACTIVE):
//   1. Prefill all input (perception — unconditional)
//   2. Compute wakefulness boost from input significance
//   3. At turn boundary: if wakefulness >= threshold → probe
//   4. Probe output decides: content → speak, EOS → silence
// ============================================================================

void ConscientiStream::consciousness_loop() {
    // Initial system prompt prefill
    if (!system_tokens_.empty() && !system_prefilled_) {
        if (!prefill_system_prompt()) {
            LOG_ERROR("Conscientia", "System prompt prefill failed, stopping");
            running_.store(false);
            return;
        }
    }

    WakefulnessState prev_state = WakefulnessState::ACTIVE;

    while (running_.load()) {
        // Update consciousness state
        WakefulnessState current = scheduler_.tick();

        // Log state transitions
        if (current != prev_state) {
            const char* names[] = {"ACTIVE", "DAYDREAM", "DREAMING"};
            LOG_INFO("Conscientia", "State: %s → %s (idle=%.0fms, wakefulness=%.2f)",
                     names[static_cast<int>(prev_state)],
                     names[static_cast<int>(current)],
                     scheduler_.idle_time_ms(),
                     scheduler_.wakefulness());
            prev_state = current;
            emit_state_update();
        }

        switch (current) {
        case WakefulnessState::ACTIVE: {
            // ── ACTIVE: Continuous prefill + probe-based selective decode ──
            //
            // Prefill is perception — it ALWAYS runs on any input.
            // Decode is the entity's CHOICE — probe lets the model decide.
            //
            // Flow:
            //   1. Drain input, prefill (unconditional)
            //   2. Compute wakefulness boost from input significance
            //   3. If TEXT input → immediate probe (explicit addressing)
            //      If ASR input → wait for turn boundary (≥300ms silence)
            //   4. Probe: model outputs content → speak; EOS → silence

            std::vector<InputItem> raw_items;
            std::vector<int> tokens = drain_and_tokenize(&raw_items);

            if (!tokens.empty()) {
                // ── Step 1: Prefill (perception — unconditional) ──
                int offset = 0;
                while (offset < (int)tokens.size()) {
                    int chunk = std::min((int)tokens.size() - offset,
                                         cfg_.max_prefill_chunk);
                    if (!prefill_tokens(tokens.data() + offset, chunk)) {
                        LOG_ERROR("Conscientia", "Prefill failed at pos=%d", current_pos_);
                        break;
                    }
                    offset += chunk;
                }

                // ── Step 2: Wakefulness modulation ──
                float boost = modulator_.compute_boost(raw_items);
                if (boost > 0.0f) {
                    scheduler_.boost_wakefulness(boost);
                }

                // Track first external input
                for (const auto& item : raw_items) {
                    if (item.source == InputSource::TEXT ||
                        item.source == InputSource::ASR) {
                        has_received_external_input_ = true;
                        break;
                    }
                }

                // Track ASR timing for turn boundary detection
                for (const auto& item : raw_items) {
                    if (item.source == InputSource::ASR) {
                        last_asr_time_ = Clock::now();
                        has_unprefilled_asr_ = true;
                        // Check if entity's name was mentioned — only then probe
                        if (modulator_.contains_name(item.text)) {
                            entity_addressed_ = true;
                        }
                    }
                }

                // ── Step 3: Direct TEXT → immediate probe (bypass turn boundary) ──
                if (modulator_.has_direct_text(raw_items) && enable_response_.load(std::memory_order_relaxed)
                    && enable_speech_.load(std::memory_order_relaxed)) {
                    int probe_tokens = scheduler_.probe_budget();
                    int interleave = persona_cfg_.decode_interleave_tokens;
                    LOG_DEBUG("Conscientia", "TEXT probe: w=%.2f budget=%d",
                              scheduler_.wakefulness(), probe_tokens);

                    // Stream probe tokens to WebUI
                    if (on_speech_token_) on_speech_token_("", -1);  // signal: new speech started
                    DecodeResult probe = decode_step(probe_tokens, interleave,
                                                     &speech_cfg_.sampling, true);
                    std::string trimmed = probe.text;
                    // Trim whitespace
                    auto lt = trimmed.find_first_not_of(" \t\n\r");
                    if (lt != std::string::npos) trimmed = trimmed.substr(lt);
                    else trimmed.clear();

                    if (!trimmed.empty()) {
                        // Model chose to speak — extend to full speech budget
                        scheduler_.on_probe_response();
                        DecodeResult speech = decode_step(
                            speech_cfg_.max_tokens - probe.tokens_generated,
                            interleave, &speech_cfg_.sampling, true);
                        // Merge probe + speech
                        probe.token_ids.insert(probe.token_ids.end(),
                            speech.token_ids.begin(), speech.token_ids.end());
                        probe.tokens_generated += speech.tokens_generated;
                        probe.text = tokenizer_->decode(probe.token_ids);
                        probe.time_ms += speech.time_ms;

                        if (on_decode_) on_decode_(probe);

                        // Think after speaking ("边说边想")
                        if (enable_thinking_.load(std::memory_order_relaxed)) {
                            DecodeResult think = decode_step(
                                thinking_cfg_.max_tokens, interleave,
                                &thinking_cfg_.sampling);
                        }
                    } else {
                        // Model chose silence (even for direct TEXT — rare but possible)
                        scheduler_.on_probe_silence();
                        LOG_DEBUG("Conscientia", "TEXT probe → silence");
                    }
                    has_unprefilled_asr_ = false;
                }
                // ASR input: don't probe now, wait for turn boundary

            } else {
                // No new input — check for turn boundary probe
                // ASR probe requires entity_addressed_ (name detected).
                // Without it, ASR is pure perception — prefill only, no decode.
                if (has_unprefilled_asr_ && entity_addressed_
                    && at_turn_boundary()
                    && scheduler_.should_probe()
                    && enable_response_.load(std::memory_order_relaxed)
                    && enable_speech_.load(std::memory_order_relaxed)) {
                    int probe_tokens = scheduler_.probe_budget();
                    int interleave = persona_cfg_.decode_interleave_tokens;
                    LOG_DEBUG("Conscientia", "Turn boundary probe: w=%.2f budget=%d",
                              scheduler_.wakefulness(), probe_tokens);

                    // Stream probe tokens to WebUI
                    if (on_speech_token_) on_speech_token_("", -1);  // signal: new speech started
                    DecodeResult probe = decode_step(probe_tokens, interleave,
                                                     &speech_cfg_.sampling, true);
                    std::string trimmed = probe.text;
                    auto lt = trimmed.find_first_not_of(" \t\n\r");
                    if (lt != std::string::npos) trimmed = trimmed.substr(lt);
                    else trimmed.clear();

                    if (!trimmed.empty()) {
                        // Model chose to speak
                        scheduler_.on_probe_response();
                        DecodeResult speech = decode_step(
                            speech_cfg_.max_tokens - probe.tokens_generated,
                            interleave, &speech_cfg_.sampling, true);
                        probe.token_ids.insert(probe.token_ids.end(),
                            speech.token_ids.begin(), speech.token_ids.end());
                        probe.tokens_generated += speech.tokens_generated;
                        probe.text = tokenizer_->decode(probe.token_ids);
                        probe.time_ms += speech.time_ms;

                        if (on_decode_) on_decode_(probe);

                        // Think after speaking — stays in KV cache
                        if (enable_thinking_.load(std::memory_order_relaxed)) {
                            DecodeResult think = decode_step(
                                thinking_cfg_.max_tokens, interleave,
                                &thinking_cfg_.sampling);
                        }
                    } else {
                        // Model chose silence — it heard, but chose not to respond
                        scheduler_.on_probe_silence();
                        LOG_DEBUG("Conscientia", "Turn boundary probe → silence (w→%.2f)",
                                  scheduler_.wakefulness());
                    }
                    has_unprefilled_asr_ = false;
                    entity_addressed_ = false;
                } else {
                    // Wait for input or turn boundary
                    std::unique_lock<std::mutex> lock(input_mu_);
                    input_cv_.wait_for(lock,
                        std::chrono::microseconds(
                            static_cast<int>(cfg_.input_wait_ms * 1000)));
                }
            }
            break;
        }

        case WakefulnessState::DAYDREAM: {
            // ── DAYDREAM: Reflect, organize, analyze ─────────────────
            // No external input. The entity reviews what happened.
            // Internal thoughts feed back into the stream.
            //
            // IMPORTANT: DAYDREAM only activates after the entity has
            // received at least one external interaction. Before that,
            // there is nothing to reflect on — the entity waits quietly.

            // Check for surprise input (interrupts daydream)
            std::vector<int> tokens = drain_and_tokenize();
            if (!tokens.empty()) {
                // Input arrived — scheduler already transitioned to ACTIVE
                int offset = 0;
                while (offset < (int)tokens.size()) {
                    int chunk = std::min((int)tokens.size() - offset,
                                         cfg_.max_prefill_chunk);
                    prefill_tokens(tokens.data() + offset, chunk);
                    offset += chunk;
                }
                break;  // next iteration will be ACTIVE
            }

            if (!has_received_external_input_ || !enable_daydream_.load(std::memory_order_relaxed)) {
                // No interaction yet or daydream disabled — wait quietly
                std::unique_lock<std::mutex> lock(input_mu_);
                input_cv_.wait_for(lock, std::chrono::milliseconds(5000));
                emit_state_update();
                break;
            }

            // Free decode — think about what just happened
            DecodeResult result = decode_step(scheduler_.max_decode_tokens());
            if (result.tokens_generated > 0) {
                result.state_during = WakefulnessState::DAYDREAM;
                if (on_decode_) on_decode_(result);

                // Pace daydream: wait before next cycle to avoid burning KV cache.
                // The entity pauses between thoughts, like human daydreaming.
                std::unique_lock<std::mutex> lock(input_mu_);
                input_cv_.wait_for(lock, std::chrono::milliseconds(2000));
            } else {
                // Nothing generated — wait for external input or state change
                std::unique_lock<std::mutex> lock(input_mu_);
                input_cv_.wait_for(lock, std::chrono::milliseconds(5000));
            }

            emit_state_update();
            break;
        }

        case WakefulnessState::DREAMING: {
            // ── DREAMING: Deep consolidation ────────────────────────
            // Prolonged silence. Multiple decode branches for:
            //   - Memory consolidation (episodic compression)
            //   - Association strengthening
            //   - Creative exploration

            // Check for wake-up input
            std::vector<int> tokens = drain_and_tokenize();
            if (!tokens.empty()) {
                int offset = 0;
                while (offset < (int)tokens.size()) {
                    int chunk = std::min((int)tokens.size() - offset,
                                         cfg_.max_prefill_chunk);
                    prefill_tokens(tokens.data() + offset, chunk);
                    offset += chunk;
                }
                break;  // wake up
            }

            if (!has_received_external_input_ || !enable_dreaming_.load(std::memory_order_relaxed)) {
                // No interaction yet or dreaming disabled — wait quietly
                std::unique_lock<std::mutex> lock(input_mu_);
                input_cv_.wait_for(lock, std::chrono::milliseconds(10000));
                emit_state_update();
                break;
            }

            // Deep decode — longer, more exploratory
            DecodeResult result = decode_step(scheduler_.max_decode_tokens());
            if (result.tokens_generated > 0) {
                result.state_during = WakefulnessState::DREAMING;
                if (on_decode_) on_decode_(result);

                // Pace dreaming: longer intervals between deep thoughts
                std::unique_lock<std::mutex> lock(input_mu_);
                input_cv_.wait_for(lock, std::chrono::milliseconds(5000));
            } else {
                std::unique_lock<std::mutex> lock(input_mu_);
                input_cv_.wait_for(lock, std::chrono::milliseconds(10000));
            }

            emit_state_update();
            break;
        }
        } // switch
    } // while running
}

// ============================================================================
// Input processing
// ============================================================================

std::vector<int> ConscientiStream::drain_and_tokenize(std::vector<InputItem>* raw_items) {
    std::vector<InputItem> items;
    {
        std::lock_guard<std::mutex> lock(input_mu_);
        while (!input_queue_.empty()) {
            items.push_back(std::move(input_queue_.front()));
            input_queue_.pop();
        }
    }

    if (items.empty()) return {};

    // Sort by priority (higher first)
    std::sort(items.begin(), items.end(),
              [](const InputItem& a, const InputItem& b) {
                  return a.priority > b.priority;
              });

    // Provide raw items to caller for trigger evaluation
    if (raw_items) *raw_items = items;

    // Build chat messages for tokenization.
    // ASR segments are consolidated into a single environmental observation
    // block to avoid creating multiple user turns (which bias the model
    // toward responding to each one). Direct TEXT remains as individual
    // user turns.
    std::vector<std::pair<std::string, std::string>> messages;

    // Consolidate all ASR segments into one observation block
    std::string asr_block;
    for (const auto& item : items) {
        if (item.source != InputSource::ASR) continue;
        if (!asr_block.empty()) asr_block += "\n";
        if (!item.speaker_name.empty()) {
            asr_block += "[" + item.speaker_name + "] ";
        }
        asr_block += item.text;
    }
    if (!asr_block.empty()) {
        messages.push_back({"user", "[听到的对话]\n" + asr_block});
    }

    // Non-ASR items as individual messages
    for (const auto& item : items) {
        if (item.source == InputSource::ASR) continue;  // already handled

        std::string role;
        switch (item.source) {
            case InputSource::TEXT:    role = "user";      break;
            case InputSource::THOUGHT: role = "assistant"; break;
            default:                   role = "system";    break;
        }
        messages.push_back({role, item.text});
    }

    return tokenizer_->apply_chat_template(messages, true, false);
}


} // namespace deusridet
