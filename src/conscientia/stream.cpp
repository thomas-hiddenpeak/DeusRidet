/**
 * @file stream.cpp
 * @philosophical_role The Conscientia stream — the continuous loop that IS the entity's awareness. Not a server, not a worker pool: a single persistent principal thread whose termination is the entity's end.
 * @serves Actus::awaken (starts it), Nexus (observes it), every other subsystem (driven by it).
 */
// stream.cpp — Consciousness stream: probe-based awareness engine
//
// Implements the consciousness model:
//   ACTIVE:   Prefill runs unconditionally (perception).
//             At turn boundaries, probe decode lets the MODEL decide
//             whether to speak or stay silent.
//   DAYDREAM: No input → prefill pauses, decode reflects/organizes.
//   DREAMING: Prolonged silence → deep consolidation, multi-branch decode.
//
// The entity is not a chatbot. Probe decode is the model's opportunity
// to decide — EOS means silence, content means speech.

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

ConscientiStream::~ConscientiStream() {
    stop();
}

bool ConscientiStream::init(const ConscientiConfig& cfg,
                            const PersonaConfig& persona,
                            ModelWeights& model,
                            InferenceState& state,
                            CacheManager& cache_mgr,
                            Tokenizer& tokenizer) {
    cfg_       = cfg;
    persona_cfg_ = persona;

    // Initialize per-pipeline configs from global defaults
    speech_cfg_.sampling.temperature = cfg.temperature;
    speech_cfg_.sampling.top_k = cfg.top_k;
    speech_cfg_.sampling.top_p = cfg.top_p;
    speech_cfg_.max_tokens = persona.speech_max_tokens;

    thinking_cfg_.sampling.temperature = cfg.temperature;
    thinking_cfg_.sampling.top_k = cfg.top_k;
    thinking_cfg_.sampling.top_p = cfg.top_p;
    thinking_cfg_.max_tokens = persona.thinking_max_tokens;

    action_cfg_.sampling.temperature = cfg.temperature;
    action_cfg_.sampling.top_k = cfg.top_k;
    action_cfg_.sampling.top_p = cfg.top_p;
    action_cfg_.max_tokens = 512;

    model_     = &model;
    state_     = &state;
    cache_mgr_ = &cache_mgr;
    tokenizer_ = &tokenizer;
    current_pos_ = 0;
    last_token_ = 0;
    system_prefilled_ = false;
    has_unprefilled_asr_ = false;
    last_asr_time_ = {};

    modulator_.init(persona_cfg_);

    LOG_INFO("Conscientia", "Stream initialized: entity=%s, max_chunk=%d, "
             "probe_threshold=%.2f, half_life=%.1fs",
             persona_cfg_.name.c_str(), cfg_.max_prefill_chunk,
             scheduler_.config().probe_threshold,
             scheduler_.config().wakefulness_half_life_sec);
    return true;
}

void ConscientiStream::start() {
    if (running_.load()) return;
    running_.store(true);
    // Register the CONSCIOUSNESS business-clock anchor. T1 = Prefill frame_id;
    // nominal pulse ~100 ms. Subjective time is legal at T1 (frames may skip
    // under Decode preemption), T0 always ticks wall-clock.
    tempus::anchor_register(tempus::Domain::CONSCIOUSNESS,
                            tempus::now_t0_ns(),
                            /*t1_zero=*/0,
                            /*period_ns=*/100'000'000ULL);
    loop_thread_ = std::thread(&ConscientiStream::consciousness_loop, this);
    LOG_INFO("Conscientia", "Consciousness stream started (tempus CONSCIOUSNESS anchored, 100 ms/frame)");
}

void ConscientiStream::stop() {
    if (!running_.load()) return;
    running_.store(false);
    input_cv_.notify_all();  // wake the loop so it can exit
    if (loop_thread_.joinable()) loop_thread_.join();
    LOG_INFO("Conscientia", "Consciousness stream stopped (pos=%d)", current_pos_);
}

void ConscientiStream::inject_input(InputItem item) {
    if (!enable_llm_.load(std::memory_order_relaxed)) return;
    item.timestamp = Clock::now();
    {
        std::lock_guard<std::mutex> lock(input_mu_);
        input_queue_.push(std::move(item));
    }
    scheduler_.on_external_input();  // immediately transition to ACTIVE
    input_cv_.notify_one();          // wake the consciousness loop
}

void ConscientiStream::set_identity_prompt(const std::string& prompt) {
    identity_prompt_ = prompt;
    recompose_system_prompt();
}

void ConscientiStream::recompose_system_prompt() {
    // Compose full system prompt: identity + per-pipeline guidance
    std::string combined = identity_prompt_;
    if (!speech_cfg_.prompt.empty()) {
        combined += "\n\n当你说话时：" + speech_cfg_.prompt;
    }
    if (!thinking_cfg_.prompt.empty()) {
        combined += "\n\n当你思考时：" + thinking_cfg_.prompt;
    }
    if (!action_cfg_.prompt.empty()) {
        combined += "\n\n当你行动时：" + action_cfg_.prompt;
    }
    set_system_prompt(combined);
}

void ConscientiStream::set_system_prompt(const std::string& prompt) {
    system_tokens_ = tokenizer_->apply_chat_template(
        {{"system", prompt}}, false, false);
    LOG_INFO("Conscientia", "System prompt: %zu tokens", system_tokens_.size());
}

// ============================================================================
// Prefill — process a chunk of tokens through all 64 layers
// ============================================================================

bool ConscientiStream::prefill_tokens(const int* token_ids, int num_tokens) {
    using MC = ModelConfig;
    int M = num_tokens;
    if (M <= 0) return true;

    // Ensure KV blocks are allocated for the new token range
    if (!cache_mgr_->ensure_blocks_for_range(current_pos_, M)) {
        LOG_ERROR("Conscientia", "Block allocation failed for pos=%d M=%d",
                  current_pos_, M);
        return false;
    }
    cache_mgr_->sync_block_table(state_->compute_stream);

    auto t0 = Clock::now();
    cudaStream_t stream = state_->compute_stream;

    // Process in sub-chunks to stay within activation buffer limits
    int chunk_start = 0;
    while (chunk_start < M) {
        int chunk_size = std::min(M - chunk_start, state_->max_seq_len);
        int pos_start  = current_pos_ + chunk_start;

        // Copy token IDs to device
        cudaMemcpyAsync(state_->token_ids,
                        token_ids + chunk_start,
                        chunk_size * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        // Embedding lookup
        embedding_lookup(model_->embed_tokens, state_->token_ids,
                         state_->hidden, chunk_size, MC::HIDDEN_SIZE, stream);

        // Copy to residual
        cudaMemcpyAsync(state_->residual, state_->hidden,
                        (size_t)chunk_size * MC::HIDDEN_SIZE * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);

        // Forward through all 64 layers
        int dn_layer_idx = 0;
        int fa_layer_idx = 0;
        int seq_len_after = pos_start + chunk_size;

        const int max_phys  = cache_mgr_->block_pool().max_blocks();
        const int blk_size  = cache_mgr_->block_pool().block_size();
        __half* kv_pool     = cache_mgr_->block_pool().pool_ptr();
        const int* blk_tbl  = cache_mgr_->d_block_table();

        for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
            const LayerWeights& lw = model_->layers[layer];

            rms_norm(state_->residual, lw.input_layernorm, state_->norm_out,
                     chunk_size, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

            if (lw.is_full_attention) {
                full_attention_forward_paged_prefill(
                    state_->norm_out, lw.full_attn,
                    kv_pool, blk_tbl, fa_layer_idx,
                    pos_start, chunk_size, seq_len_after,
                    max_phys, blk_size, *state_, stream);
                fa_layer_idx++;
            } else {
                deltanet_prefill(state_->norm_out, lw.delta_net,
                                 dn_layer_idx, chunk_size, *state_, stream);
                dn_layer_idx++;
            }

            residual_rms_norm(state_->residual, state_->norm_out,
                              lw.post_attn_layernorm, state_->norm_out,
                              chunk_size, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

            mlp_forward_prefill(state_->norm_out, lw.mlp, state_->residual,
                                chunk_size, *state_, stream);
        }

        chunk_start += chunk_size;
    }

    cudaError_t sync_err = cudaStreamSynchronize(state_->compute_stream);
    if (sync_err != cudaSuccess) {
        LOG_ERROR("Conscientia", "CUDA sync error in prefill: %s (code %d)",
                  cudaGetErrorString(sync_err), (int)sync_err);
        // Clear the error so subsequent operations aren't poisoned
        cudaGetLastError();
        return false;
    }
    // Also check for async errors that may not have been caught
    cudaError_t async_err = cudaGetLastError();
    if (async_err != cudaSuccess) {
        LOG_ERROR("Conscientia", "CUDA async error in prefill: %s (code %d)",
                  cudaGetErrorString(async_err), (int)async_err);
        return false;
    }

    current_pos_ += M;
    cache_mgr_->set_seq_len(current_pos_);

    auto t1 = Clock::now();
    float ms = std::chrono::duration_cast<Ms>(t1 - t0).count();
    float tps = M / (ms / 1000.0f);

    // Update metrics
    metrics_.last_prefill_ms = ms;
    metrics_.last_prefill_tps = tps;
    metrics_.last_prefill_tokens = M;
    metrics_.total_prefill_tokens += M;
    metrics_.total_prefill_ms += ms;

    LOG_INFO("Conscientia", "Prefill: %d tokens, pos→%d, %.1f ms (%.0f tok/s)",
             M, current_pos_, ms, tps);
    return true;
}

// ============================================================================
// Decode — generate tokens autoregressively
// ============================================================================

DecodeResult ConscientiStream::decode_step(int max_tokens, int interleave_check,
                                           const SamplingParams* override_params,
                                           bool stream_tokens) {
    using MC = ModelConfig;
    DecodeResult result;
    result.state_during = scheduler_.state();
    auto t0 = Clock::now();

    SamplingParams params;
    if (override_params) {
        params = *override_params;
    } else {
        params.temperature = cfg_.temperature;
        params.top_k = cfg_.top_k;
        params.top_p = cfg_.top_p;
    }

    const int max_phys = cache_mgr_->block_pool().max_blocks();
    const int blk_size = cache_mgr_->block_pool().block_size();
    __half* kv_pool    = cache_mgr_->block_pool().pool_ptr();
    cudaStream_t stream = state_->compute_stream;

    int eos_id  = tokenizer_->eos_token_id();
    int eot_id  = tokenizer_->eot_id();
    int prev_token = last_token_;

    for (int step = 0; step < max_tokens; step++) {
        // Interleave check: yield to prefill if new input arrived
        if (interleave_check > 0 && step > 0 && (step % interleave_check) == 0) {
            if (input_pending()) {
                LOG_DEBUG("Conscientia", "Decode interleaved at step %d: input pending", step);
                break;
            }
        }

        // Ensure block allocated
        if (!cache_mgr_->ensure_blocks_for(current_pos_)) {
            LOG_WARN("Conscientia", "Decode: block alloc failed at pos=%d", current_pos_);
            break;
        }
        cache_mgr_->sync_block_table(stream);

        // Update device position
        cudaMemcpyAsync(state_->d_pos, &current_pos_, sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        // Token embedding
        cudaMemcpyAsync(state_->token_ids, &prev_token, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        embedding_lookup(model_->embed_tokens, state_->token_ids,
                         state_->hidden, 1, MC::HIDDEN_SIZE, stream);
        cudaMemcpyAsync(state_->residual, state_->hidden,
                        MC::HIDDEN_SIZE * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);

        // Forward through all layers (single-token decode with paged attention)
        int dn_layer_idx = 0;
        int fa_layer_idx = 0;
        int seq_len = current_pos_ + 1;
        const int* blk_tbl = cache_mgr_->d_block_table();

        for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
            const LayerWeights& lw = model_->layers[layer];

            rms_norm(state_->residual, lw.input_layernorm, state_->norm_out,
                     1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

            if (lw.is_full_attention) {
                full_attention_forward_paged(
                    state_->norm_out, lw.full_attn,
                    kv_pool, blk_tbl, fa_layer_idx,
                    current_pos_, seq_len,
                    max_phys, blk_size, *state_, stream);
                fa_layer_idx++;
            } else {
                deltanet_forward(state_->norm_out, lw.delta_net,
                                 dn_layer_idx, *state_, stream);
                dn_layer_idx++;
            }

            residual_rms_norm(state_->residual, state_->norm_out,
                              lw.post_attn_layernorm, state_->norm_out,
                              1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

            mlp_forward(state_->norm_out, lw.mlp, state_->residual,
                        *state_, stream);
        }

        // Final norm + LM head
        rms_norm(state_->residual, model_->final_norm, state_->hidden,
                 1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);
        int8_linear_forward(state_->hidden, model_->lm_head_int8,
                            state_->logits, 1, stream);

        // Sample
        unsigned long long seed = ++state_->rng_counter;
        sample_top_k_top_p(state_->logits, state_->probs, MC::VOCAB_SIZE,
                            params, seed, state_->sample_out, stream);

        // Get result
        int token;
        cudaMemcpyAsync(&token, state_->sample_out, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Check stop conditions
        if (token == eos_id || token == eot_id) break;

        result.token_ids.push_back(token);
        prev_token = token;
        current_pos_++;
        cache_mgr_->set_seq_len(current_pos_);

        // Stream individual token to WebUI (for speech streaming)
        if (stream_tokens && on_speech_token_) {
            std::string tok_text = tokenizer_->decode({token});
            on_speech_token_(tok_text, token);
        }
    }

    last_token_ = prev_token;
    result.tokens_generated = static_cast<int>(result.token_ids.size());
    result.text = tokenizer_->decode(result.token_ids);
    result.time_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();

    if (result.tokens_generated > 0) {
        float ms_per_tok = result.time_ms / (float)result.tokens_generated;

        // Update metrics
        metrics_.last_decode_ms_per_tok = ms_per_tok;
        metrics_.last_decode_tokens = result.tokens_generated;
        metrics_.total_decode_tokens += result.tokens_generated;
        metrics_.total_decode_ms += result.time_ms;

        LOG_INFO("Conscientia", "Decode: %d tokens, %.1f ms (%.1f ms/tok), text='%.40s'",
                 result.tokens_generated, result.time_ms, ms_per_tok,
                 result.text.c_str());
    }

    return result;
}

// ============================================================================
// System prompt prefill
// ============================================================================

bool ConscientiStream::prefill_system_prompt() {
    if (system_tokens_.empty()) {
        system_prefilled_ = true;
        return true;
    }

    bool ok = prefill_tokens(system_tokens_.data(),
                             static_cast<int>(system_tokens_.size()));
    if (ok) {
        system_prefilled_ = true;
        LOG_INFO("Conscientia", "System prompt prefilled: %zu tokens",
                 system_tokens_.size());
    }
    return ok;
}

// ============================================================================
// Turn boundary detection (ASR silence ≥300ms)
// ============================================================================

bool ConscientiStream::at_turn_boundary() const {
    if (last_asr_time_ == Clock::time_point{}) return false;
    auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - last_asr_time_).count();
    return elapsed >= 1000.0f;  // 1s silence after last ASR = turn boundary
}

// ============================================================================
// Input pending check (non-blocking, for decode interleave)
// ============================================================================

bool ConscientiStream::input_pending() const {
    // Note: input_mu_ is mutable here; we use try_lock to avoid blocking
    // the decode loop. If lock is contended, assume no input (conservative).
    auto* mu = const_cast<std::mutex*>(&input_mu_);
    if (mu->try_lock()) {
        bool pending = !input_queue_.empty();
        mu->unlock();
        return pending;
    }
    return false;  // couldn't lock — don't interrupt decode
}

// ============================================================================
// State update callback
// ============================================================================

void ConscientiStream::emit_state_update() {
    if (on_state_) {
        on_state_(scheduler_.state(),
                  scheduler_.wakefulness(),
                  cache_mgr_->gpu_blocks_used(),
                  cache_mgr_->free_blocks(),
                  current_pos_,
                  metrics_);
    }
}

} // namespace deusridet
