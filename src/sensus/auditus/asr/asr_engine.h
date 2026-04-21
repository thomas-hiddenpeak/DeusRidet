/**
 * @file asr_engine.h
 * @philosophical_role Declaration of the ASR engine facade.
 * @serves Auditus pipeline, Actus.
 */
// asr_engine.h — Qwen3-ASR inference engine
//
// Top-level orchestrator:
//   1. Load weights (SafetensorsLoader, cudaMalloc to device memory)
//   2. Orchestrate AudioEncoder + TextDecoder
//   3. Provide transcribe(): PCM float → text
//
// Usage:
//   ASREngine engine;
//   engine.load_model("/path/to/Qwen3-ASR-1.7B");
//   std::string text = engine.transcribe(pcm, num_samples, 16000);
//
// Adapted from qwen35-orin (src/plugins/asr/asr_engine.h): engine class
// with weight loading, prompt construction, and autoregressive decode loop.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include "asr_config.h"
#include "asr_encoder.h"
#include "asr_decoder.h"
#include "whisper_mel.h"
#include "machina/tokenizer.h"
#include "machina/safetensors.h"
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace deusridet {
namespace asr {

// Structured result from ASR transcription with timing breakdown.
struct ASRResult {
    std::string text;           // final post-processed text
    std::string raw_text;       // text before collapse_repeats / ITN
    float mel_ms = 0;           // mel spectrogram computation time
    float encoder_ms = 0;       // Whisper encoder forward time
    float decode_ms = 0;        // autoregressive decode loop time
    float postprocess_ms = 0;   // collapse_repeats + ITN time
    float total_ms = 0;         // end-to-end transcription time
    int mel_frames = 0;         // mel spectrogram frames
    int encoder_out_len = 0;    // encoder output sequence length
    int token_count = 0;        // decoded tokens count
};

class ASREngine {
public:
    ASREngine();
    ~ASREngine();

    // Load model: model_dir must contain config.json + safetensors + tokenizer
    void load_model(const std::string& model_dir);

    // Transcribe: PCM float → ASRResult with timing breakdown.
    // samples: mono float32, sample_rate: auto-resamples to 16kHz
    ASRResult transcribe(const float* samples, int num_samples,
                         int sample_rate = 16000,
                         int max_new_tokens = 448);

    bool is_loaded() const { return loaded_; }
    const ASRConfig& config() const { return config_; }

    void set_repetition_penalty(float p) { repetition_penalty_ = p; }
    float get_repetition_penalty() const { return repetition_penalty_; }

private:
    ASRConfig config_;
    std::unique_ptr<AudioEncoder> encoder_;
    std::unique_ptr<TextDecoder> decoder_;
    std::unique_ptr<WhisperMel> whisper_mel_;
    Tokenizer tokenizer_;
    std::string model_dir_;

    // GPU buffers
    __nv_bfloat16* mel_gpu_ = nullptr;       // [128, max_mel_frames] BF16
    __nv_bfloat16* encoder_out_ = nullptr;   // [max_tokens, 2048]
    __nv_bfloat16* input_embeds_ = nullptr;  // [max_prompt_len, 2048]
    __nv_bfloat16* logits_ = nullptr;        // [vocab_size]
    int* position_ids_ = nullptr;            // [3, max_seq_len]
    int* token_id_gpu_ = nullptr;            // [1]
    int* prompt_tokens_gpu_ = nullptr;       // [max_prompt_len]

    // Repetition penalty
    float repetition_penalty_ = 1.0f;
    int* rep_tokens_gpu_ = nullptr;          // [max_new_tokens]

    // Embed weight pointer (shared with decoder, not separately owned)
    __nv_bfloat16* embed_tokens_w_ = nullptr;

    // Device weight allocations (freed in destructor)
    std::vector<void*> device_weights_;

    int max_mel_frames_ = 0;
    int max_prompt_len_ = 0;

    cudaStream_t stream_ = 0;
    bool loaded_ = false;

    void load_weights(const std::string& model_dir);
    void build_prompt(int encoder_out_len, std::vector<int>& token_ids,
                      const std::string& language = "Chinese");
};

} // namespace asr
} // namespace deusridet
