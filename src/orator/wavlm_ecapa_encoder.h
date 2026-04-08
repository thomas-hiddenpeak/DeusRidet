// wavlm_ecapa_encoder.h — WavLM-Large + ECAPA-TDNN Joint Speaker Encoder
//
// Adapted from ESPnet voxcelebs12_ecapa_wavlm_joint model:
//   WavLM-Large (s3prl, Microsoft MIT license) → learned weighted sum →
//   UtteranceMVN → ECAPA-TDNN (ESPnet Apache 2.0) → ChnAttnStatPool →
//   RawNet3 Projector → L2 normalize → 192-dim speaker embedding
//
// All computation on GPU via cuBLAS SGEMM + custom CUDA kernels.
// Weights loaded from safetensors (ESPnet checkpoint, zero-copy mmap).
//
// Reference: docs/ACKNOWLEDGMENTS.md for full attribution.

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace deusridet {

// ============================================================================
// WavLM-Large configuration (frozen at export time)
// ============================================================================
struct WavLMConfig {
    // CNN feature extractor
    // [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
    static constexpr int cnn_layers       = 7;
    static constexpr int cnn_dim          = 512;
    static constexpr int cnn_kernels[7]   = {10, 3, 3, 3, 3, 2, 2};
    static constexpr int cnn_strides[7]   = {5,  2, 2, 2, 2, 2, 2};
    // extractor_mode = "layer_norm" → every layer has LayerNorm

    // Encoder
    static constexpr int embed_dim        = 1024;
    static constexpr int ffn_dim          = 4096;
    static constexpr int num_heads        = 16;
    static constexpr int head_dim         = 64;
    static constexpr int num_layers       = 24;
    static constexpr bool layer_norm_first = true;

    // Positional conv
    static constexpr int pos_conv_kernel  = 128;
    static constexpr int pos_conv_groups  = 16;

    // Relative position embedding
    static constexpr bool rel_pos_embed   = true;
    static constexpr int num_buckets      = 320;
    static constexpr int max_distance     = 800;
    static constexpr bool gru_rel_pos     = true;

    // Input normalization
    static constexpr bool normalize_input = true;

    // Featurizer
    static constexpr int num_hidden_states = 25;  // 24 layers + 1 initial
};

// ============================================================================
// ECAPA-TDNN configuration
// ============================================================================
struct EcapaConfig {
    static constexpr int input_size   = 1024;
    static constexpr int ndim         = 1024;
    static constexpr int output_size  = 1536;
    static constexpr int model_scale  = 8;
    static constexpr int width        = 128;   // ndim / scale
    static constexpr int num_res2     = 7;     // scale - 1
    static constexpr int se_bottleneck = 128;

    // 3 blocks with different dilations
    static constexpr int block_dilations[3] = {2, 3, 4};
    static constexpr int block_kernel = 3;
};

// ============================================================================
// GPU weight buffer: simple float* + element count pair
// ============================================================================
struct GpuWeight {
    float*  ptr   = nullptr;
    __half* fp16  = nullptr;   // FP16 copy for Tensor Core GEMM
    int     numel = 0;
};

// ============================================================================
// WavLM + ECAPA-TDNN Joint Speaker Encoder
// ============================================================================
class WavLMEcapaEncoder {
public:
    WavLMEcapaEncoder();
    ~WavLMEcapaEncoder();

    WavLMEcapaEncoder(const WavLMEcapaEncoder&) = delete;
    WavLMEcapaEncoder& operator=(const WavLMEcapaEncoder&) = delete;

    // Load weights from safetensors file and allocate GPU buffers.
    // model_path: path to wavlm_ecapa.safetensors
    bool init(const std::string& model_path);
    bool initialized() const { return initialized_; }

    // Extract 192-dim L2-normalized speaker embedding from raw 16kHz PCM.
    // pcm: CPU float32 buffer, normalized [-1, 1] range.
    // n_samples: number of audio samples.
    // Returns empty vector on error.
    std::vector<float> extract(const float* pcm, int n_samples);

    // Same but PCM already on GPU.
    std::vector<float> extract_gpu(const float* d_pcm, int n_samples);

    static constexpr int embedding_dim() { return 192; }

    // Latency from last extract_gpu() call (ms).
    float last_lat_cnn_ms()     const { return last_lat_cnn_ms_; }
    float last_lat_encoder_ms() const { return last_lat_encoder_ms_; }
    float last_lat_ecapa_ms()   const { return last_lat_ecapa_ms_; }
    float last_lat_total_ms()   const { return last_lat_total_ms_; }

    // === Layer-by-layer test interface (for debugging) ===
    // Each returns GPU pointer + updates T (time dimension after this stage).
    // Caller must NOT free returned pointers (owned by scratch).

    // Stage 1: Waveform normalization + CNN feature extraction
    // Input: d_wav [1, n_samples] on GPU
    // Output: [cnn_dim, T'] on GPU
    float* test_cnn(const float* d_wav, int n_samples, int& T_out);

    // Stage 2: LayerNorm + post-extract projection
    // Input: d_cnn [cnn_dim, T'] on GPU (from test_cnn)
    // Output: [T', embed_dim] on GPU (row-major)
    float* test_projection(const float* d_cnn, int T, int& T_out);

    // Stage 3: Positional convolution + GELU + skip connection
    // Input: d_proj [T', embed_dim] on GPU (from test_projection)
    // Output: [T', embed_dim] on GPU (row-major)
    float* test_pos_conv(const float* d_proj, int T, int& T_out);

    // Stage 4: Full transformer encoder (24 layers) + final LN
    // Input: d_pos [T', embed_dim] on GPU (from test_pos_conv)
    // Output: d_hidden_states_ populated [25, T', embed_dim], returns final layer output
    float* test_encoder(const float* d_pos, int T, int& T_out);

    // Access hidden state from encoder (layer 0 = input, 1..24 = layer outputs)
    const float* get_hidden_state(int layer) const;
    const float* get_pos_bias() const { return d_pos_bias_; }

    // More stages added incrementally as implemented...

private:
    // Weight loading from safetensors
    bool load_weights(const std::string& model_path);
    GpuWeight upload_weight(const std::string& key, const float* data, int numel);
    const GpuWeight& w(const std::string& key) const;

    // Forward pass stages
    void forward_cnn(const float* d_wav, int n_samples, float* d_out, int& T_out);
    void forward_layer_norm(const float* d_in, float* d_out, int T, int dim,
                            const float* d_gamma, const float* d_beta);
    void forward_linear(const float* d_in, float* d_out,
                        int rows, int in_dim, int out_dim,
                        const float* d_weight, const float* d_bias,
                        const __half* d_weight_fp16 = nullptr);
    void forward_transformer_layer(float* d_x, int T, int layer_idx,
                                    float* d_pos_bias);
    void forward_conv1d(const float* d_in, float* d_out,
                        int C_in, int C_out, int T, int K,
                        int pad, int dilation,
                        const float* d_weight, const float* d_bias,
                        const __half* d_weight_fp16 = nullptr);
    // cuDNN-accelerated Conv1d (as Conv2d with H=1) for CNN extractor + pos conv
    void forward_conv1d_cudnn(const float* d_in, float* d_out,
                              int C_in, int C_out, int T, int K,
                              int stride, int pad, int groups, int dilation,
                              const float* d_weight, const float* d_bias);
    void forward_batch_norm_1d(const float* d_in, float* d_out,
                                int C, int T, const std::string& prefix);
    void forward_se_block(float* d_x, int C, int T, const std::string& prefix);
    void forward_ecapa_block(float* d_x, int C, int T,
                              int dilation, const std::string& prefix);

    // Activation scratch space management
    bool ensure_scratch(int n_samples);

    // Merge per-layer Q/K/V weights into single [3D, D] matrices
    void merge_qkv_weights();

    // Members
    bool initialized_ = false;
    std::string model_path_;

    // GPU weights indexed by safetensors key name
    std::unordered_map<std::string, GpuWeight> weights_;

    // cuBLAS + CUDA resources
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t   stream_ = nullptr;

    // cuDNN for CNN feature extractor + positional conv
    cudnnHandle_t  cudnn_  = nullptr;
    void*  d_cudnn_ws_     = nullptr;
    size_t cudnn_ws_size_  = 0;

    // Scratch GPU buffers (pre-allocated, grown as needed)
    float* scratch_a_ = nullptr;  // primary activation buffer
    float* scratch_b_ = nullptr;  // secondary activation buffer
    float* scratch_c_ = nullptr;  // tertiary (for attention etc.)
    size_t scratch_size_ = 0;     // current size in floats per buffer
    int    scratch_max_T_ = 0;    // max T' these buffers can handle

    // WavLM hidden states buffer for featurizer weighted sum
    // [num_hidden_states, T', embed_dim] stored contiguously
    float* d_hidden_states_ = nullptr;
    size_t hidden_states_size_ = 0;

    // Precomputed positional conv weight (weight_norm applied once)
    float* d_pos_conv_weight_ = nullptr;
    bool   pos_conv_weight_computed_ = false;

    // Im2col buffer for Conv1d kernel>1
    float* d_im2col_ = nullptr;
    size_t im2col_size_ = 0;

    // Pre-allocated PCM upload buffer (avoids per-call cudaMalloc)
    float* d_pcm_buf_ = nullptr;
    size_t pcm_buf_size_ = 0;  // current capacity in floats

    // Latency measurements from last extract_gpu() call (ms).
    float last_lat_cnn_ms_     = 0;
    float last_lat_encoder_ms_ = 0;
    float last_lat_ecapa_ms_   = 0;
    float last_lat_total_ms_   = 0;

    // FP16 scratch for cublasGemmEx Tensor Core path
    __half* d_gemm_a_ = nullptr;  // activation input  (max: T*4096)
    __half* d_gemm_b_ = nullptr;  // second input       (max: 16*T*T)
    size_t gemm_fp16_size_ = 0;   // current capacity in halfs per buffer

    // Precomputed relative position bias [num_heads, T, T]
    float* d_pos_bias_ = nullptr;
    int    pos_bias_T_ = 0;
};

} // namespace deusridet
