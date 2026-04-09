// stream.h — Consciousness stream (continuous awareness engine)
//
// The ConscientiStream implements the consciousness model:
//
//   Prefill is continuous perception — it ALWAYS runs on input.
//   Decode is the entity's CHOICE to speak, think, or act.
//
//   The entity is not a chatbot. It perceives everything but speaks
//   only when it decides to — via probe decode. Hard-coded rules do
//   not decide whether to respond; the model itself does.
//
//   Probe decode flow (ACTIVE state):
//     1. Prefill input (unconditional perception)
//     2. AttentionModulator computes wakefulness boost
//     3. At turn boundary (ASR silence ≥300ms or TEXT input):
//        if wakefulness >= threshold → probe decode (3-8 tokens)
//     4. If probe produces content → model chose to speak → full decode
//        If probe produces EOS    → model chose silence → wait
//
//   DAYDREAM: No input → free decode (reflect, organize)
//   DREAMING: Prolonged silence → deep consolidation

#pragma once

#include "frame.h"
#include "scheduler.h"
#include "attention_modulator.h"
#include "../machina/model.h"
#include "../machina/forward.h"
#include "../machina/paged_attention.h"
#include "../memoria/cache_manager.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <functional>
#include <string>
#include <vector>

namespace deusridet {

// ============================================================================
// Configuration
// ============================================================================

struct ConscientiConfig {
    int   max_prefill_chunk = 512;     // max tokens per prefill burst
    int   max_context_len   = 131072;  // max total context (tokens)
    float temperature       = 0.7f;    // global fallback
    int   top_k             = 50;
    float top_p             = 0.9f;

    // How long to wait for input before yielding to decode (ms)
    float input_wait_ms     = 50.0f;
};

// Per-pipeline decode configuration (speech, thinking, action).
// Each pipeline has its own sampling params, max token limit, and
// an optional system prompt injected before decode.
struct PipelineConfig {
    SamplingParams sampling;    // temperature, top_k, top_p, rep_penalty
    int max_tokens = 256;
    std::string prompt;         // per-pipeline system prompt (empty = none)
};

// ============================================================================
// Decode result — output from a decode branch
// ============================================================================

struct DecodeResult {
    std::vector<int> token_ids;
    std::string text;
    float time_ms = 0.0f;
    int tokens_generated = 0;
    WakefulnessState state_during;  // which state produced this output
};

// ============================================================================
// Consciousness performance metrics (exposed to WebUI)
// ============================================================================

struct ConscientiMetrics {
    // Last prefill
    float last_prefill_ms    = 0.0f;
    float last_prefill_tps   = 0.0f;  // tokens per second
    int   last_prefill_tokens = 0;

    // Last decode
    float last_decode_ms_per_tok = 0.0f;
    int   last_decode_tokens     = 0;

    // Cumulative
    int   total_prefill_tokens   = 0;
    int   total_decode_tokens    = 0;
    float total_prefill_ms       = 0.0f;
    float total_decode_ms        = 0.0f;
};

// ============================================================================
// Callback types
// ============================================================================

// Called when a decode branch produces output text.
using OnDecodeOutput = std::function<void(const DecodeResult& result)>;

// Called per-token during speech decode for streaming to WebUI.
// token_text is the decoded text for this single token.
using OnSpeechToken = std::function<void(const std::string& token_text, int token_id)>;

// Called on state transitions and periodic updates.
using OnStateUpdate = std::function<void(WakefulnessState state,
                                          float wakefulness,
                                          int kv_blocks_used,
                                          int kv_blocks_free,
                                          int current_pos,
                                          const ConscientiMetrics& metrics)>;

// ============================================================================
// ConscientiStream — continuous consciousness engine
// ============================================================================

class ConscientiStream {
public:
    ConscientiStream() = default;
    ~ConscientiStream();

    ConscientiStream(const ConscientiStream&) = delete;
    ConscientiStream& operator=(const ConscientiStream&) = delete;

    // Initialize the consciousness stream.
    // Model and cache_manager must already be initialized.
    bool init(const ConscientiConfig& cfg,
              const PersonaConfig& persona,
              ModelWeights& model,
              InferenceState& state,
              CacheManager& cache_mgr,
              class Tokenizer& tokenizer);

    // Start the consciousness loop on a dedicated thread.
    void start();

    // Stop the consciousness loop (blocks until thread exits).
    void stop();

    // ── Input injection ─────────────────────────────────────────────

    // Inject external input (text, ASR, vision). Wakes the stream if idle.
    void inject_input(InputItem item);

    // Set identity prompt and recompose the full system prompt.
    // The full system prompt = identity + per-pipeline guidance.
    void set_identity_prompt(const std::string& prompt);
    const std::string& identity_prompt() const { return identity_prompt_; }

    // Recompose the full system prompt from identity + pipeline prompts.
    // Call after changing any pipeline prompt.
    void recompose_system_prompt();

    // ── Callbacks ───────────────────────────────────────────────────

    void set_on_decode(OnDecodeOutput cb) { on_decode_ = std::move(cb); }
    void set_on_state(OnStateUpdate cb)   { on_state_ = std::move(cb); }
    void set_on_speech_token(OnSpeechToken cb) { on_speech_token_ = std::move(cb); }

    // ── Mode enable/disable ─────────────────────────────────────────

    void set_response_enabled(bool v)  { enable_response_.store(v, std::memory_order_relaxed); }
    void set_daydream_enabled(bool v)  { enable_daydream_.store(v, std::memory_order_relaxed); }
    void set_dreaming_enabled(bool v)  { enable_dreaming_.store(v, std::memory_order_relaxed); }
    void set_llm_enabled(bool v)       { enable_llm_.store(v, std::memory_order_relaxed); }
    void set_speech_enabled(bool v)    { enable_speech_.store(v, std::memory_order_relaxed); }
    void set_thinking_enabled(bool v)  { enable_thinking_.store(v, std::memory_order_relaxed); }
    void set_action_enabled(bool v)    { enable_action_.store(v, std::memory_order_relaxed); }
    bool response_enabled() const { return enable_response_.load(std::memory_order_relaxed); }
    bool daydream_enabled() const { return enable_daydream_.load(std::memory_order_relaxed); }
    bool dreaming_enabled() const { return enable_dreaming_.load(std::memory_order_relaxed); }
    bool llm_enabled() const { return enable_llm_.load(std::memory_order_relaxed); }
    bool speech_enabled() const { return enable_speech_.load(std::memory_order_relaxed); }
    bool thinking_enabled() const { return enable_thinking_.load(std::memory_order_relaxed); }
    bool action_enabled() const { return enable_action_.load(std::memory_order_relaxed); }

    // ── Runtime hyperparameter update ───────────────────────────────

    // Global fallback params (used by daydream/dreaming)
    void set_temperature(float v) { cfg_.temperature = v; }
    void set_top_k(int v) { cfg_.top_k = v; }
    void set_top_p(float v) { cfg_.top_p = v; }

    // Per-pipeline config access
    PipelineConfig& speech_cfg()   { return speech_cfg_; }
    PipelineConfig& thinking_cfg() { return thinking_cfg_; }
    PipelineConfig& action_cfg()   { return action_cfg_; }
    const PipelineConfig& speech_cfg()   const { return speech_cfg_; }
    const PipelineConfig& thinking_cfg() const { return thinking_cfg_; }
    const PipelineConfig& action_cfg()   const { return action_cfg_; }

    float temperature() const { return cfg_.temperature; }
    int top_k() const { return cfg_.top_k; }
    float top_p() const { return cfg_.top_p; }
    int speech_max_tokens() const { return speech_cfg_.max_tokens; }
    int thinking_max_tokens() const { return thinking_cfg_.max_tokens; }

    // ── Performance metrics ─────────────────────────────────────────

    ConscientiMetrics metrics() const { return metrics_; }

    // ── State access ────────────────────────────────────────────────

    bool running() const { return running_.load(std::memory_order_relaxed); }
    int current_pos() const { return current_pos_; }
    const Scheduler& scheduler() const { return scheduler_; }
    Scheduler& scheduler() { return scheduler_; }
    WakefulnessState current_state() const { return scheduler_.state(); }

private:
    using Clock = std::chrono::steady_clock;

    ConscientiConfig cfg_;
    PersonaConfig   persona_cfg_;
    PipelineConfig  speech_cfg_;
    PipelineConfig  thinking_cfg_;
    PipelineConfig  action_cfg_;
    ModelWeights*   model_     = nullptr;
    InferenceState* state_     = nullptr;
    CacheManager*   cache_mgr_ = nullptr;
    class Tokenizer* tokenizer_ = nullptr;

    Scheduler scheduler_;
    AttentionModulator modulator_;

    // Turn boundary tracking for ASR (probe only at speech pauses)
    Clock::time_point last_asr_time_{};   // when last ASR input arrived
    bool has_unprefilled_asr_ = false;     // ASR prefilled but not yet probed
    bool entity_addressed_ = false;        // name detected in ASR or TEXT input

    // Consciousness loop thread
    std::thread loop_thread_;
    std::atomic<bool> running_{false};

    // Input queue (thread-safe, notifies loop via condition variable)
    std::queue<InputItem> input_queue_;
    std::mutex input_mu_;
    std::condition_variable input_cv_;  // wakes loop when input arrives

    // Identity prompt (stored separately for recomposition)
    std::string identity_prompt_;

    // System prompt tokens (prefilled once at start)
    // Composed from identity_prompt_ + pipeline prompts.
    std::vector<int> system_tokens_;
    bool system_prefilled_ = false;

    // Sequence position tracking
    int current_pos_ = 0;   // next write position in KV cache

    // Last generated token (for decode continuation)
    int last_token_ = 0;

    // Tracks whether ANY external input has ever been received.
    // DAYDREAM/DREAMING won't generate until first external interaction.
    bool has_received_external_input_ = false;

    // Mode enable flags (controllable from WebUI)
    std::atomic<bool> enable_response_{true};
    std::atomic<bool> enable_daydream_{false};  // default off for debugging
    std::atomic<bool> enable_dreaming_{false};  // default off for debugging
    std::atomic<bool> enable_llm_{true};        // master LLM switch (gates input injection + prefill/decode)
    std::atomic<bool> enable_speech_{true};     // speech decode branch
    std::atomic<bool> enable_thinking_{false};  // thinking decode branch (default off)
    std::atomic<bool> enable_action_{false};    // action decode branch (default off)

    // Performance metrics
    ConscientiMetrics metrics_;

    // Callbacks
    OnDecodeOutput on_decode_;
    OnStateUpdate  on_state_;
    OnSpeechToken  on_speech_token_;

    // ── Internal methods ────────────────────────────────────────────

    // Main consciousness loop (runs on dedicated thread)
    void consciousness_loop();

    // Drain input queue and tokenize into a token sequence.
    // Returns empty vector if no input pending.
    // Also populates raw_items with the original InputItems for evaluation.
    std::vector<int> drain_and_tokenize(std::vector<InputItem>* raw_items = nullptr);

    // Prefill a chunk of tokens through the model.
    bool prefill_tokens(const int* token_ids, int num_tokens);

    // Run one decode step. Returns generated tokens (may be empty).
    // If interleave_check > 0, checks input queue every N tokens
    // and breaks early if input is pending (yields to prefill).
    // If override_params is non-null, uses those instead of cfg_ defaults.
    // If stream_tokens is true, calls on_speech_token_ per token.
    DecodeResult decode_step(int max_tokens, int interleave_check = 0,
                             const SamplingParams* override_params = nullptr,
                             bool stream_tokens = false);

    // Check if we're at a turn boundary (silence after ASR, ≥300ms).
    bool at_turn_boundary() const;

    // Prefill system prompt (one-time at start).
    bool prefill_system_prompt();

    // Set tokenized system prompt from combined text (called by recompose).
    void set_system_prompt(const std::string& prompt);

    // Check if input is pending (non-blocking, for decode interleave).
    bool input_pending() const;

    // Emit state update callback.
    void emit_state_update();
};

} // namespace deusridet
