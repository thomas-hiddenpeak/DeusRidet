// asr_engine.cpp — Qwen3-ASR inference engine implementation
//
// Weight loading via SafetensorsLoader (mmap → cudaMalloc device copy).
// Transcribe flow: PCM → Mel → Encoder → Prompt → Prefill → Decode loop → Text
//
// Adapted from qwen35-orin (src/plugins/asr/asr_engine.cpp): weight loading,
// prompt construction, autoregressive decode loop, and ITN post-processing.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "asr_engine.h"
#include "asr_ops.h"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <regex>

namespace deusridet {
namespace asr {

// =========================================================================
// ITN (Inverse Text Normalization) — Chinese numerals → Arabic
// =========================================================================

static const char* cn_digits[] = {
    "零","一","二","三","四","五","六","七","八","九"
};

// Chinese numeral string → integer. Returns -1 on failure.
static int64_t cn_to_arabic(const std::string& s) {
    // Single-char digit
    for (int i = 0; i < 10; i++) {
        if (s == cn_digits[i]) return i;
    }

    int64_t result = 0;
    int64_t current = 0;
    int64_t high_unit = 0;

    auto starts_with = [](const std::string& str, size_t pos, const char* prefix) -> bool {
        size_t plen = std::strlen(prefix);
        return pos + plen <= str.size() && str.compare(pos, plen, prefix) == 0;
    };

    size_t i = 0;
    while (i < s.size()) {
        // Check digit
        int digit = -1;
        for (int d = 0; d < 10; d++) {
            if (starts_with(s, i, cn_digits[d])) {
                digit = d;
                i += std::strlen(cn_digits[d]);
                break;
            }
        }
        if (digit >= 0) {
            current = digit;
            continue;
        }

        // Check units
        if (starts_with(s, i, "十")) {
            current = (current == 0) ? 10 : current * 10;
            i += 3; // UTF-8 "十" = 3 bytes
        } else if (starts_with(s, i, "百")) {
            current *= 100;
            i += 3;
        } else if (starts_with(s, i, "千")) {
            current *= 1000;
            i += 3;
        } else if (starts_with(s, i, "万")) {
            if (current == 0 && result > 0) {
                result *= 10000;
            } else {
                result = (result + current) * 10000;
                current = 0;
            }
            high_unit = 10000;
            i += 3;
        } else if (starts_with(s, i, "亿")) {
            result = (result + current) * 100000000LL;
            current = 0;
            high_unit = 100000000LL;
            i += 3;
        } else {
            return -1; // unrecognized
        }
        result += current;
        current = 0;
    }
    result += current;
    return result;
}

// Apply basic ITN: replace Chinese numeral patterns with Arabic numerals
static std::string apply_itn(const std::string& text) {
    // Pattern: sequences of Chinese numeral chars
    static const std::regex cn_num_re(
        "([零一二三四五六七八九十百千万亿]+)");

    std::string result;
    std::sregex_iterator it(text.begin(), text.end(), cn_num_re);
    std::sregex_iterator end;
    size_t last = 0;

    while (it != end) {
        result += text.substr(last, it->position() - last);
        std::string cn = it->str();
        int64_t val = cn_to_arabic(cn);
        if (val >= 0) {
            result += std::to_string(val);
        } else {
            result += cn;  // keep original if parse fails
        }
        last = it->position() + it->length();
        ++it;
    }
    result += text.substr(last);
    return result;
}

// Collapse repeated phrases (simple: consecutive identical substrings)
static std::string collapse_repeats(const std::string& text) {
    // Simple character-level repeat: "ABABAB" → "AB"
    // Only for very short repeated units (≤8 chars) repeated ≥3 times
    std::string result = text;
    for (int unit_len = 1; unit_len <= 8 && unit_len * 3 <= (int)result.size(); unit_len++) {
        size_t pos = 0;
        while (pos + unit_len * 3 <= result.size()) {
            std::string unit = result.substr(pos, unit_len);
            int count = 1;
            size_t check = pos + unit_len;
            while (check + unit_len <= result.size() &&
                   result.substr(check, unit_len) == unit) {
                count++;
                check += unit_len;
            }
            if (count >= 3) {
                // Replace N copies with 1
                result = result.substr(0, pos + unit_len) +
                         result.substr(pos + unit_len * count);
            } else {
                pos += unit_len;
            }
        }
    }
    return result;
}

// =========================================================================
// Helper: allocate device memory and copy from host data
// =========================================================================

static __nv_bfloat16* alloc_and_copy(const void* src, size_t nbytes,
                                       std::vector<void*>& allocs) {
    __nv_bfloat16* d_ptr = nullptr;
    cudaMalloc(&d_ptr, nbytes);
    cudaMemcpy(d_ptr, src, nbytes, cudaMemcpyHostToDevice);
    allocs.push_back(d_ptr);
    return d_ptr;
}

// =========================================================================
// ASREngine
// =========================================================================

ASREngine::ASREngine() = default;

ASREngine::~ASREngine() {
    for (void* p : device_weights_) {
        if (p) cudaFree(p);
    }
    if (mel_gpu_)           cudaFree(mel_gpu_);
    if (encoder_out_)       cudaFree(encoder_out_);
    if (input_embeds_)      cudaFree(input_embeds_);
    if (logits_)            cudaFree(logits_);
    if (position_ids_)      cudaFree(position_ids_);
    if (token_id_gpu_)      cudaFree(token_id_gpu_);
    if (prompt_tokens_gpu_) cudaFree(prompt_tokens_gpu_);
    if (rep_tokens_gpu_)    cudaFree(rep_tokens_gpu_);
    if (stream_)            cudaStreamDestroy(stream_);
}

// =========================================================================
// Prompt construction
// =========================================================================

void ASREngine::build_prompt(int encoder_out_len, std::vector<int>& token_ids,
                              const std::string& language) {
    // Qwen3-ASR prompt format:
    // <|im_start|>system\n<|im_end|>\n
    // <|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n
    // <|im_start|>assistant\nlanguage Chinese<asr_text>

    token_ids.clear();

    // System turn (empty)
    token_ids.push_back(ASRConfig::IM_START_TOKEN);
    // "system" = token 8948, "\n" = token 198
    token_ids.push_back(8948);
    token_ids.push_back(198);
    token_ids.push_back(ASRConfig::IM_END_TOKEN);
    token_ids.push_back(198);

    // User turn with audio
    token_ids.push_back(ASRConfig::IM_START_TOKEN);
    token_ids.push_back(872);  // "user"
    token_ids.push_back(198);  // "\n"
    token_ids.push_back(ASRConfig::AUDIO_START_TOKEN);
    for (int i = 0; i < encoder_out_len; i++) {
        token_ids.push_back(ASRConfig::AUDIO_PAD_TOKEN);
    }
    token_ids.push_back(ASRConfig::AUDIO_END_TOKEN);
    token_ids.push_back(ASRConfig::IM_END_TOKEN);
    token_ids.push_back(198);  // "\n"

    // Assistant turn with language prefix
    token_ids.push_back(ASRConfig::IM_START_TOKEN);
    token_ids.push_back(77091);  // "assistant"
    token_ids.push_back(198);    // "\n"

    // "language Chinese" — encode via tokenizer
    std::string lang_prefix = "language " + language;
    auto lang_tokens = tokenizer_.encode(lang_prefix);
    for (int t : lang_tokens) {
        token_ids.push_back(t);
    }

    token_ids.push_back(ASRConfig::ASR_TEXT_TOKEN);
}

// =========================================================================
// Weight loading
// =========================================================================

void ASREngine::load_weights(const std::string& model_dir) {
    SafetensorsLoader loader(model_dir);

    auto load_tensor = [&](const std::string& name) -> __nv_bfloat16* {
        if (!loader.has_tensor(name)) {
            fprintf(stderr, "ASR: missing tensor: %s\n", name.c_str());
            return nullptr;
        }
        auto t = loader.get_tensor(name);
        return alloc_and_copy(t->data(), t->nbytes(), device_weights_);
    };

    // --- Encoder Conv2D frontend ---
    auto conv2d1_w = load_tensor("thinker.audio_tower.conv2d1.weight");
    auto conv2d1_b = load_tensor("thinker.audio_tower.conv2d1.bias");
    auto conv2d2_w = load_tensor("thinker.audio_tower.conv2d2.weight");
    auto conv2d2_b = load_tensor("thinker.audio_tower.conv2d2.bias");
    auto conv2d3_w = load_tensor("thinker.audio_tower.conv2d3.weight");
    auto conv2d3_b = load_tensor("thinker.audio_tower.conv2d3.bias");
    auto conv_out_w = load_tensor("thinker.audio_tower.conv_out.weight");
    encoder_->set_conv_weights(conv2d1_w, conv2d1_b,
                                conv2d2_w, conv2d2_b,
                                conv2d3_w, conv2d3_b,
                                conv_out_w);

    // --- Encoder layers ---
    for (int i = 0; i < config_.encoder_layers; i++) {
        std::string prefix = "thinker.audio_tower.layers." + std::to_string(i) + ".";
        EncoderLayerWeights lw;
        lw.self_attn_layer_norm_w = load_tensor(prefix + "self_attn_layer_norm.weight");
        lw.self_attn_layer_norm_b = load_tensor(prefix + "self_attn_layer_norm.bias");
        lw.q_proj_w = load_tensor(prefix + "self_attn.q_proj.weight");
        lw.q_proj_b = load_tensor(prefix + "self_attn.q_proj.bias");
        lw.k_proj_w = load_tensor(prefix + "self_attn.k_proj.weight");
        lw.k_proj_b = load_tensor(prefix + "self_attn.k_proj.bias");
        lw.v_proj_w = load_tensor(prefix + "self_attn.v_proj.weight");
        lw.v_proj_b = load_tensor(prefix + "self_attn.v_proj.bias");
        lw.o_proj_w = load_tensor(prefix + "self_attn.out_proj.weight");
        lw.o_proj_b = load_tensor(prefix + "self_attn.out_proj.bias");
        lw.final_layer_norm_w = load_tensor(prefix + "final_layer_norm.weight");
        lw.final_layer_norm_b = load_tensor(prefix + "final_layer_norm.bias");
        lw.fc1_w = load_tensor(prefix + "fc1.weight");
        lw.fc1_b = load_tensor(prefix + "fc1.bias");
        lw.fc2_w = load_tensor(prefix + "fc2.weight");
        lw.fc2_b = load_tensor(prefix + "fc2.bias");
        encoder_->set_layer_weights(i, lw);
    }

    // --- Encoder post-processing ---
    auto ln_post_w = load_tensor("thinker.audio_tower.ln_post.weight");
    auto ln_post_b = load_tensor("thinker.audio_tower.ln_post.bias");
    auto proj1_w   = load_tensor("thinker.audio_tower.proj1.weight");
    auto proj1_b   = load_tensor("thinker.audio_tower.proj1.bias");
    auto proj2_w   = load_tensor("thinker.audio_tower.proj2.weight");
    auto proj2_b   = load_tensor("thinker.audio_tower.proj2.bias");
    encoder_->set_post_weights(ln_post_w, ln_post_b, proj1_w, proj1_b, proj2_w, proj2_b);

    // --- Decoder embeddings ---
    auto embed_w = load_tensor("thinker.model.embed_tokens.weight");
    embed_tokens_w_ = embed_w;

    __nv_bfloat16* lm_head_w = embed_w;  // tied weights
    if (!config_.tie_word_embeddings && loader.has_tensor("thinker.lm_head.weight")) {
        lm_head_w = load_tensor("thinker.lm_head.weight");
    }

    auto final_norm_w = load_tensor("thinker.model.norm.weight");
    decoder_->set_embed_weights(embed_w, lm_head_w, final_norm_w);

    // --- Decoder layers ---
    for (int i = 0; i < config_.decoder_layers; i++) {
        std::string prefix = "thinker.model.layers." + std::to_string(i) + ".";
        DecoderLayerWeights lw;
        lw.input_layernorm_w       = load_tensor(prefix + "input_layernorm.weight");
        lw.q_proj_w                = load_tensor(prefix + "self_attn.q_proj.weight");
        lw.k_proj_w                = load_tensor(prefix + "self_attn.k_proj.weight");
        lw.v_proj_w                = load_tensor(prefix + "self_attn.v_proj.weight");
        lw.o_proj_w                = load_tensor(prefix + "self_attn.o_proj.weight");
        lw.q_norm_w                = load_tensor(prefix + "self_attn.q_norm.weight");
        lw.k_norm_w                = load_tensor(prefix + "self_attn.k_norm.weight");
        lw.post_attention_layernorm_w = load_tensor(prefix + "post_attention_layernorm.weight");
        lw.gate_proj_w             = load_tensor(prefix + "mlp.gate_proj.weight");
        lw.up_proj_w               = load_tensor(prefix + "mlp.up_proj.weight");
        lw.down_proj_w             = load_tensor(prefix + "mlp.down_proj.weight");
        decoder_->set_layer_weights(i, lw);
    }
}

// =========================================================================
// load_model
// =========================================================================

void ASREngine::load_model(const std::string& model_dir) {
    model_dir_ = model_dir;

    // 1. Load config
    if (!config_.load_from_model_dir(model_dir)) {
        fprintf(stderr, "ASR: failed to load config from %s\n", model_dir.c_str());
        return;
    }
    fprintf(stderr, "ASR: config loaded — encoder %d layers, decoder %d layers, vocab %d\n",
            config_.encoder_layers, config_.decoder_layers, config_.vocab_size);

    // 2. Load tokenizer
    if (!tokenizer_.load(model_dir)) {
        fprintf(stderr, "ASR: failed to load tokenizer from %s\n", model_dir.c_str());
        return;
    }
    fprintf(stderr, "ASR: tokenizer loaded — vocab %d\n", tokenizer_.vocab_size());

    // 3. Create CUDA stream
    cudaStreamCreate(&stream_);

    // 4. Create components
    whisper_mel_ = std::make_unique<WhisperMel>();
    if (!whisper_mel_->init()) {
        fprintf(stderr, "ASR: failed to init WhisperMel\n");
        return;
    }

    encoder_ = std::make_unique<AudioEncoder>(config_);
    decoder_ = std::make_unique<TextDecoder>(config_);

    // 5. Load weights
    load_weights(model_dir);

    // 6. Initialize encoder + decoder
    encoder_->initialize(stream_);
    decoder_->initialize(stream_);
    decoder_->prepare_optimized_weights(stream_);

    // 7. Allocate engine-level GPU buffers
    // Max mel: 30s * 16000 / 160 = 3000 frames
    max_mel_frames_ = 3000;
    int max_encoder_out = ASRConfig::get_output_length(max_mel_frames_);
    // Prompt: ~20 fixed tokens + encoder_out_len audio_pads + max_new_tokens
    max_prompt_len_ = 64 + max_encoder_out + 512;

    size_t mel_bytes = 128 * max_mel_frames_ * sizeof(__nv_bfloat16);
    size_t enc_bytes = max_encoder_out * config_.output_dim * sizeof(__nv_bfloat16);
    size_t emb_bytes = max_prompt_len_ * config_.decoder_hidden_size * sizeof(__nv_bfloat16);
    size_t logit_bytes = config_.vocab_size * sizeof(__nv_bfloat16);
    size_t pos_bytes = 3 * max_prompt_len_ * sizeof(int);

    cudaMalloc(&mel_gpu_, mel_bytes);
    cudaMalloc(&encoder_out_, enc_bytes);
    cudaMalloc(&input_embeds_, emb_bytes);
    cudaMalloc(&logits_, logit_bytes);
    cudaMalloc(&position_ids_, pos_bytes);
    cudaMalloc(&token_id_gpu_, sizeof(int));
    cudaMalloc(&prompt_tokens_gpu_, max_prompt_len_ * sizeof(int));
    cudaMalloc(&rep_tokens_gpu_, 512 * sizeof(int));

    cudaStreamSynchronize(stream_);
    loaded_ = true;
    fprintf(stderr, "ASR: model loaded — max_mel=%d, max_enc=%d, max_prompt=%d\n",
            max_mel_frames_, max_encoder_out, max_prompt_len_);
}

// =========================================================================
// Transcribe
// =========================================================================

ASRResult ASREngine::transcribe(const float* samples, int num_samples,
                                   int sample_rate, int max_new_tokens) {
    ASRResult result;
    if (!loaded_) return result;
    if (num_samples <= 0 || !samples) return result;

    auto t_total_start = std::chrono::steady_clock::now();

    // TODO: resample if sample_rate != 16000. For now, assume 16kHz.

    // 1. Mel spectrogram (F32 on GPU)
    auto t0 = std::chrono::steady_clock::now();
    auto mel_result = whisper_mel_->compute(samples, num_samples);
    // CRITICAL: WhisperMel uses its own CUDA stream. Must synchronize before
    // the engine stream reads d_mel, otherwise F32→BF16 races with mel kernels.
    whisper_mel_->sync();
    float* d_mel_f32 = mel_result.d_mel;
    int mel_frames = mel_result.num_frames;
    auto t1 = std::chrono::steady_clock::now();
    result.mel_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    result.mel_frames = mel_frames;

    if (mel_frames <= 0) return result;
    if (mel_frames > max_mel_frames_) {
        fprintf(stderr, "ASR: mel_frames %d exceeds max %d, clamping\n",
                mel_frames, max_mel_frames_);
        mel_frames = max_mel_frames_;
        result.mel_frames = mel_frames;
    }

    // 2. Convert F32 mel → BF16
    int mel_elements = 128 * mel_frames;
    asr_ops::invoke_f32_to_bf16(mel_gpu_, d_mel_f32, mel_elements, stream_);

    // 3. Encoder forward → encoder_out [out_seq_len, 2048]
    t0 = std::chrono::steady_clock::now();
    int encoder_out_len = 0;
    encoder_->forward(mel_gpu_, mel_frames, encoder_out_, encoder_out_len, stream_);
    cudaStreamSynchronize(stream_);
    t1 = std::chrono::steady_clock::now();
    result.encoder_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    result.encoder_out_len = encoder_out_len;

    if (encoder_out_len <= 0) return result;

    // 4. Build prompt token IDs
    std::vector<int> prompt_tokens;
    build_prompt(encoder_out_len, prompt_tokens);
    int prompt_len = (int)prompt_tokens.size();

    if (prompt_len > max_prompt_len_) {
        fprintf(stderr, "ASR: prompt_len %d exceeds max %d\n", prompt_len, max_prompt_len_);
        return result;
    }

    // Copy prompt tokens to GPU
    cudaMemcpyAsync(prompt_tokens_gpu_, prompt_tokens.data(),
                    prompt_len * sizeof(int), cudaMemcpyHostToDevice, stream_);

    // 5. Embedding lookup: prompt_tokens → input_embeds
    asr_ops::invoke_embedding_lookup(
        input_embeds_, prompt_tokens_gpu_, embed_tokens_w_,
        prompt_len, config_.decoder_hidden_size, stream_);

    // 6. Replace audio_pad positions with encoder output
    // Find audio_pad range in prompt
    int audio_pad_start = -1;
    for (int i = 0; i < prompt_len; i++) {
        if (prompt_tokens[i] == ASRConfig::AUDIO_PAD_TOKEN) {
            audio_pad_start = i;
            break;
        }
    }
    if (audio_pad_start >= 0) {
        // Copy encoder_out into input_embeds at audio_pad positions
        size_t copy_bytes = encoder_out_len * config_.decoder_hidden_size * sizeof(__nv_bfloat16);
        __nv_bfloat16* dst = input_embeds_ + audio_pad_start * config_.decoder_hidden_size;
        cudaMemcpyAsync(dst, encoder_out_, copy_bytes,
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // 7. Build MRoPE position IDs (all 3 dims = sequential, same for ASR)
    std::vector<int> h_position_ids(3 * prompt_len);
    for (int d = 0; d < 3; d++) {
        for (int i = 0; i < prompt_len; i++) {
            h_position_ids[d * prompt_len + i] = i;
        }
    }
    cudaMemcpyAsync(position_ids_, h_position_ids.data(),
                    3 * prompt_len * sizeof(int), cudaMemcpyHostToDevice, stream_);

    // 8. Reset decoder KV cache and prefill
    decoder_->reset_cache();
    decoder_->forward_prefill(input_embeds_, position_ids_, prompt_len,
                               logits_, stream_);

    // 9. Autoregressive decode loop
    auto t_decode_start = std::chrono::steady_clock::now();
    std::vector<int> output_tokens;
    output_tokens.reserve(max_new_tokens);

    // EOS suppression: suppress IM_END and ENDOFTEXT for first few tokens
    const int eos_suppress_tokens = 3;

    // Repetition penalty tracking
    std::vector<int> all_tokens(prompt_tokens);

    int next_pos = prompt_len;
    for (int step = 0; step < max_new_tokens; step++) {
        // Apply EOS suppression
        if (step < eos_suppress_tokens) {
            asr_ops::invoke_suppress_eos(logits_, ASRConfig::IM_END_TOKEN,
                                         ASRConfig::ENDOFTEXT_TOKEN, stream_);
        }

        // Apply repetition penalty
        if (repetition_penalty_ > 1.0f && !all_tokens.empty()) {
            int n_rep = std::min((int)all_tokens.size(), 512);
            cudaMemcpyAsync(rep_tokens_gpu_,
                           all_tokens.data() + all_tokens.size() - n_rep,
                           n_rep * sizeof(int), cudaMemcpyHostToDevice, stream_);
            asr_ops::invoke_repetition_penalty(logits_, rep_tokens_gpu_, n_rep,
                                               repetition_penalty_, stream_);
        }

        // Argmax
        int token_id = 0;
        asr_ops::invoke_argmax(logits_, token_id_gpu_, config_.vocab_size, stream_);
        cudaMemcpyAsync(&token_id, token_id_gpu_, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Check EOS
        if (token_id == ASRConfig::IM_END_TOKEN ||
            token_id == ASRConfig::ENDOFTEXT_TOKEN) {
            break;
        }

        output_tokens.push_back(token_id);
        all_tokens.push_back(token_id);

        // Inline repetition check (bail if last N tokens are all the same)
        if (output_tokens.size() >= 6) {
            int last = output_tokens.back();
            bool all_same = true;
            for (int k = (int)output_tokens.size() - 6; k < (int)output_tokens.size() - 1; k++) {
                if (output_tokens[k] != last) { all_same = false; break; }
            }
            if (all_same) break;
        }

        // Build position IDs for decode step [3, 1]
        int h_pos[3] = { next_pos, next_pos, next_pos };
        cudaMemcpyAsync(position_ids_, h_pos, 3 * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);

        // Decode step
        decoder_->forward_decode(token_id, position_ids_, logits_, stream_);
        next_pos++;
    }

    // 10. Record decode timing and token count
    auto t_decode_end = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();
    result.token_count = (int)output_tokens.size();

    if (output_tokens.empty()) return result;

    std::string text = tokenizer_.decode(output_tokens);

    // 11. Strip leading language tag if present
    // The ASR_TEXT_TOKEN is in prompt; decoder output starts after it
    // But sometimes model echoes "language Chinese" prefix — strip it
    {
        size_t pos = text.find(">");
        if (pos != std::string::npos && pos < 20) {
            text = text.substr(pos + 1);
        }
    }

    // 12. Trim whitespace
    while (!text.empty() && (text.front() == ' ' || text.front() == '\n'))
        text.erase(text.begin());
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n'))
        text.pop_back();

    // Save raw text before post-processing.
    result.raw_text = text;

    // 13. ITN post-processing
    auto t_pp_start = std::chrono::steady_clock::now();
    text = collapse_repeats(text);
    text = apply_itn(text);

    auto t_pp_end = std::chrono::steady_clock::now();
    result.postprocess_ms = std::chrono::duration<float, std::milli>(t_pp_end - t_pp_start).count();

    result.text = text;
    auto t_total_end = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<float, std::milli>(t_total_end - t_total_start).count();

    return result;
}

} // namespace asr
} // namespace deusridet
