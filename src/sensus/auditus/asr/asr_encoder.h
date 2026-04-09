// asr_encoder.h — Qwen3-ASR Audio Encoder
//
// Whisper-style encoder:
//   1. Conv2D frontend: 3 layers (1→480→480→480), stride=2, GELU, 8× downsample
//   2. conv_out: Linear(7680→1024, no bias)
//   3. Sinusoidal PE: [max_pos=1500, 1024]
//   4. Chunk + Window Attention (n_window_infer=800)
//   5. 24× Encoder Layer: PRE-LN LayerNorm → bidirectional MHA → LayerNorm → GELU FFN
//   6. ln_post + proj1(GELU) + proj2 → [total_tokens, 2048]
//
// Weight prefix: thinker.audio_tower.*
//
// Adapted from qwen35-orin (src/plugins/asr/asr_encoder.h): encoder class
// and weight binding interface.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include "asr_config.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>

namespace deusridet {
namespace asr {

// Per-layer encoder weight pointers (all externally owned)
struct EncoderLayerWeights {
    // Self-attention
    __nv_bfloat16* self_attn_layer_norm_w = nullptr;  // [d_model]
    __nv_bfloat16* self_attn_layer_norm_b = nullptr;  // [d_model]
    __nv_bfloat16* q_proj_w = nullptr;                // [d_model, d_model]
    __nv_bfloat16* q_proj_b = nullptr;                // [d_model]
    __nv_bfloat16* k_proj_w = nullptr;                // [d_model, d_model]
    __nv_bfloat16* k_proj_b = nullptr;                // [d_model]
    __nv_bfloat16* v_proj_w = nullptr;                // [d_model, d_model]
    __nv_bfloat16* v_proj_b = nullptr;                // [d_model]
    __nv_bfloat16* o_proj_w = nullptr;                // [d_model, d_model]
    __nv_bfloat16* o_proj_b = nullptr;                // [d_model]

    // FFN
    __nv_bfloat16* final_layer_norm_w = nullptr;      // [d_model]
    __nv_bfloat16* final_layer_norm_b = nullptr;      // [d_model]
    __nv_bfloat16* fc1_w = nullptr;                   // [ffn_dim, d_model]
    __nv_bfloat16* fc1_b = nullptr;                   // [ffn_dim]
    __nv_bfloat16* fc2_w = nullptr;                   // [d_model, ffn_dim]
    __nv_bfloat16* fc2_b = nullptr;                   // [d_model]
};

class AudioEncoder {
public:
    explicit AudioEncoder(const ASRConfig& config);
    ~AudioEncoder();

    // Bind Conv2D frontend weights
    void set_conv_weights(
        __nv_bfloat16* conv2d1_w, __nv_bfloat16* conv2d1_b,   // [480, 1, 3, 3], [480]
        __nv_bfloat16* conv2d2_w, __nv_bfloat16* conv2d2_b,   // [480, 480, 3, 3], [480]
        __nv_bfloat16* conv2d3_w, __nv_bfloat16* conv2d3_b,   // [480, 480, 3, 3], [480]
        __nv_bfloat16* conv_out_w                              // [1024, 7680], no bias
    );

    // Bind post-processing weights
    void set_post_weights(
        __nv_bfloat16* ln_post_w, __nv_bfloat16* ln_post_b,   // [1024], [1024]
        __nv_bfloat16* proj1_w, __nv_bfloat16* proj1_b,       // [1024, 1024], [1024]
        __nv_bfloat16* proj2_w, __nv_bfloat16* proj2_b        // [2048, 1024], [2048]
    );

    // Bind layer weights
    void set_layer_weights(int layer_idx, const EncoderLayerWeights& weights);

    // Initialize: sinusoidal PE + workspace allocation
    void initialize(cudaStream_t stream = 0);

    // Forward pass
    // Input:  mel [128, mel_frames] GPU BF16
    // Output: encoder_out [out_seq_len, 2048] GPU BF16
    void forward(const __nv_bfloat16* mel,
                 int mel_frames,
                 __nv_bfloat16* encoder_out,
                 int& out_seq_len,
                 cudaStream_t stream = 0);

    int get_output_length(int mel_frames) const {
        return ASRConfig::get_output_length(mel_frames);
    }

private:
    ASRConfig config_;
    cublasHandle_t cublas_ = nullptr;

    // Conv2D weights (externally owned)
    __nv_bfloat16* conv2d1_w_ = nullptr;
    __nv_bfloat16* conv2d1_b_ = nullptr;
    __nv_bfloat16* conv2d2_w_ = nullptr;
    __nv_bfloat16* conv2d2_b_ = nullptr;
    __nv_bfloat16* conv2d3_w_ = nullptr;
    __nv_bfloat16* conv2d3_b_ = nullptr;
    __nv_bfloat16* conv_out_w_ = nullptr;

    // Post-processing (externally owned)
    __nv_bfloat16* ln_post_w_ = nullptr;
    __nv_bfloat16* ln_post_b_ = nullptr;
    __nv_bfloat16* proj1_w_ = nullptr;
    __nv_bfloat16* proj1_b_ = nullptr;
    __nv_bfloat16* proj2_w_ = nullptr;
    __nv_bfloat16* proj2_b_ = nullptr;

    // Layers
    std::vector<EncoderLayerWeights> layer_weights_;

    // Sinusoidal PE (owned)
    __nv_bfloat16* pe_table_ = nullptr;  // [max_source_positions, d_model]

    // Workspace (owned)
    __nv_bfloat16* workspace_ = nullptr;
    size_t workspace_size_ = 0;
    __nv_bfloat16* im2col_buf_ = nullptr;

    bool initialized_ = false;

    // Conv2D: k=3, stride=2, pad=1; GELU fused
    void conv2d_forward(const __nv_bfloat16* input,
                        int batch, int C_in, int H_in, int W_in,
                        const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                        int C_out,
                        __nv_bfloat16* output,
                        cudaStream_t stream);

    // Single Transformer encoder layer
    void encoder_layer_forward(int layer_idx,
                               __nv_bfloat16* hidden_states,
                               int seq_len,
                               const int* cu_seqlens,
                               int num_segments,
                               __nv_bfloat16* workspace,
                               cudaStream_t stream);
};

} // namespace asr
} // namespace deusridet
