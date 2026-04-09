// asr_config.h — Qwen3-ASR model configuration
//
// Parses Qwen3-ASR config.json, provides Encoder + Decoder parameters.
// Audio Encoder: Whisper-style Conv2D + 24-layer bidirectional Transformer
// Text Decoder:  Qwen3 28-layer GQA + MRoPE + SwiGLU
//
// Adapted from qwen35-orin (src/plugins/asr/asr_config.h): config structure
// and JSON parsing for Qwen3-ASR model parameters.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include <string>
#include <array>
#include <cstdint>

namespace deusridet {
namespace asr {

struct ASRConfig {
    // ========== Audio Encoder (Whisper-style) ==========
    int num_mel_bins            = 128;
    int encoder_layers          = 24;
    int encoder_d_model         = 1024;
    int encoder_attention_heads = 16;
    int encoder_head_dim        = 64;     // = d_model / attention_heads
    int encoder_ffn_dim         = 4096;
    int downsample_hidden_size  = 480;
    int max_source_positions    = 1500;   // sinusoidal PE max length
    int n_window                = 50;     // training window
    int n_window_infer          = 800;    // inference attention window
    int conv_chunksize          = 500;
    int output_dim              = 2048;   // proj2 output → decoder hidden

    // Conv2D downsample: 3 layers stride=2, padding=1 → 8× downsample
    // mel_bins=128 → 64 → 32 → 16
    int freq_after_conv() const {
        return (((num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;  // = 16
    }
    int conv_out_features() const {
        return downsample_hidden_size * freq_after_conv();  // 480 * 16 = 7680
    }

    // Conv2D stride-2 output dimension: (x-1)/2+1
    static int conv_output_size(int x) {
        return x <= 0 ? 0 : (x - 1) / 2 + 1;
    }

    // Compute output sequence length after CNN frontend
    // Python: input_lengths_leave = input_lengths % 100
    //         feat_lengths = (input_lengths_leave - 1) // 2 + 1
    //         output = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
    //                  + (input_lengths // 100) * 13
    static int get_output_length(int mel_frames) {
        int leave = mel_frames % 100;
        int t = conv_output_size(conv_output_size(conv_output_size(leave)));
        return t + (mel_frames / 100) * 13;
    }

    // ========== Text Decoder (Qwen3) ==========
    int decoder_layers              = 28;
    int decoder_hidden_size         = 2048;
    int decoder_num_attention_heads = 16;
    int decoder_num_kv_heads        = 8;
    int decoder_head_dim            = 128;
    int decoder_intermediate_size   = 6144;
    int vocab_size                  = 151936;
    float rms_norm_eps              = 1e-6f;
    float rope_theta                = 1000000.0f;
    bool tie_word_embeddings        = true;

    // MRoPE
    bool mrope_interleaved          = true;
    std::array<int, 3> mrope_section = {24, 20, 20};

    // ========== Token IDs ==========
    static constexpr int AUDIO_START_TOKEN  = 151669;
    static constexpr int AUDIO_END_TOKEN    = 151670;
    static constexpr int AUDIO_PAD_TOKEN    = 151676;
    static constexpr int IM_START_TOKEN     = 151644;
    static constexpr int IM_END_TOKEN       = 151645;
    static constexpr int ENDOFTEXT_TOKEN    = 151643;
    static constexpr int ASR_TEXT_TOKEN     = 151704;

    // ========== Derived sizes ==========
    int decoder_q_dim() const { return decoder_num_attention_heads * decoder_head_dim; }
    int decoder_kv_dim() const { return decoder_num_kv_heads * decoder_head_dim; }

    // ========== Load from config.json ==========
    bool load_from_json(const std::string& config_path);

    bool load_from_model_dir(const std::string& model_dir) {
        std::string path = model_dir;
        if (path.back() != '/') path += '/';
        return load_from_json(path + "config.json");
    }
};

} // namespace asr
} // namespace deusridet
