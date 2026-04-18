// pyannote_seg3.h — Native CUDA implementation of pyannote/segmentation-3.0.
//
// Architecture: PyanNet (SincNet + 4x BiLSTM + Linear head)
//   InstanceNorm(input) → SincConv(80,k=251,s=10) → |Abs| → MaxPool(3)
//   → [InstanceNorm → LeakyReLU → Conv(60,80,k=5) → MaxPool(3)]
//   → [InstanceNorm → LeakyReLU → Conv(60,60,k=5) → MaxPool(3)]
//   → 4× BiLSTM(hidden=128, output=256)
//   → Linear(256→128) → LeakyReLU
//   → Linear(128→128) → LeakyReLU
//   → Linear(128→7) → LogSoftmax
//
// Input:  (1, 1, 160000)  — 10s mono PCM @ 16kHz, float32
// Output: (589, 7)        — powerset log-probabilities
//
// Model weights: safetensors format (~5.7 MB, 1.49M params, all FP32).
// Uses cuDNN for BiLSTM, cuBLAS for linear layers, custom CUDA kernels
// for SincNet conv, InstanceNorm, MaxPool, LeakyReLU, and LogSoftmax.

#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <string>

namespace deusridet {

class PyannoteSeg3 {
public:
    PyannoteSeg3();
    ~PyannoteSeg3();

    PyannoteSeg3(const PyannoteSeg3&) = delete;
    PyannoteSeg3& operator=(const PyannoteSeg3&) = delete;

    // Load weights from safetensors and allocate GPU buffers.
    bool init(const std::string& model_path, cudaStream_t stream = nullptr);

    // Run forward pass. d_pcm: (chunk_samples,) float32 on GPU.
    // d_output: (num_frames, 7) float32 on GPU, log-softmax probabilities.
    // Returns num_frames (589 for 160000 samples).
    int forward(const float* d_pcm, float* d_output, int n_samples);

    bool initialized() const { return initialized_; }
    float last_latency_ms() const { return last_latency_ms_; }

    // Constants
    static constexpr int kChunkSamples = 160000;  // 10s @ 16kHz
    static constexpr int kNumFrames    = 589;      // output frames
    static constexpr int kNumClasses   = 7;        // powerset classes

    // SincNet constants
    static constexpr int kSincFilters   = 80;
    static constexpr int kSincKernel    = 251;
    static constexpr int kSincStride    = 10;
    static constexpr int kSincOutLen    = 15975;   // (160000 - 251) / 10 + 1

    static constexpr int kPool0OutLen   = 5325;    // 15975 / 3
    static constexpr int kConv1Filters  = 60;
    static constexpr int kConv1Kernel   = 5;
    static constexpr int kConv1OutLen   = 5321;    // 5325 - 5 + 1
    static constexpr int kPool1OutLen   = 1773;    // 5321 / 3

    static constexpr int kConv2Filters  = 60;
    static constexpr int kConv2Kernel   = 5;
    static constexpr int kConv2OutLen   = 1769;    // 1773 - 5 + 1
    static constexpr int kPool2OutLen   = 589;     // 1769 / 3

    // LSTM constants
    static constexpr int kLstmLayers    = 4;
    static constexpr int kLstmHidden    = 128;
    static constexpr int kLstmBiDir     = 256;     // 2 * hidden (bidirectional)
    static constexpr int kSeqLen        = 589;     // = kPool2OutLen

    // Linear head
    static constexpr int kLinear0Out    = 128;
    static constexpr int kLinear1Out    = 128;
    static constexpr float kLeakySlope  = 0.01f;

private:
    bool initialized_ = false;
    float last_latency_ms_ = 0.0f;

    cudaStream_t stream_ = nullptr;
    bool own_stream_ = false;
    cublasHandle_t cublas_ = nullptr;
    cudnnHandle_t  cudnn_  = nullptr;

    // ---- Weights (GPU) ----
    // SincNet
    float* d_wav_norm_w_ = nullptr;   // (1,)
    float* d_wav_norm_b_ = nullptr;   // (1,)
    float* d_sinc_filters_ = nullptr; // (80, 1, 251) precomputed
    float* d_norm0_w_ = nullptr;      // (80,)
    float* d_norm0_b_ = nullptr;      // (80,)
    float* d_conv1_w_ = nullptr;      // (60, 80, 5)
    float* d_conv1_b_ = nullptr;      // (60,)
    float* d_norm1_w_ = nullptr;      // (60,)
    float* d_norm1_b_ = nullptr;      // (60,)
    float* d_conv2_w_ = nullptr;      // (60, 60, 5)
    float* d_conv2_b_ = nullptr;      // (60,)
    float* d_norm2_w_ = nullptr;      // (60,)
    float* d_norm2_b_ = nullptr;      // (60,)

    // LSTM (4 layers, bidirectional)
    // Per layer: W_ih(2,512,in), W_hh(2,512,128), bias(2,1024)
    float* d_lstm_Wih_[kLstmLayers] = {};  // (2, 4*hidden, input_size)
    float* d_lstm_Whh_[kLstmLayers] = {};  // (2, 4*hidden, hidden_size)
    float* d_lstm_bias_[kLstmLayers] = {}; // (2, 4*hidden) combined bias

    // Linear head
    float* d_linear0_w_ = nullptr;    // (256, 128) = MatMul_915 transposed
    float* d_linear0_b_ = nullptr;    // (128,)
    float* d_linear1_w_ = nullptr;    // (128, 128)
    float* d_linear1_b_ = nullptr;    // (128,)
    float* d_classifier_w_ = nullptr; // (128, 7)
    float* d_classifier_b_ = nullptr; // (7,)

    // ---- Scratch buffers (GPU) ----
    float* d_sinc_out_ = nullptr;     // (80, 15975)
    float* d_pool0_ = nullptr;        // (80, 5325)
    float* d_norm_tmp_ = nullptr;     // (80, 5325) temp for instance norm
    float* d_conv1_out_ = nullptr;    // (60, 5321)
    float* d_pool1_ = nullptr;        // (60, 1773)
    float* d_conv2_out_ = nullptr;    // (60, 1769)
    float* d_pool2_ = nullptr;        // (60, 589)
    float* d_lstm_in_ = nullptr;      // (589, 60) or (589, 256)
    float* d_lstm_out_ = nullptr;     // (589, 256)
    float* d_linear_buf_ = nullptr;   // (589, 256)
    float* d_precomp_gates_ = nullptr; // (589, 512) LSTM precomputed gates

    // cuDNN LSTM workspace
    void*  d_lstm_workspace_ = nullptr;
    size_t lstm_workspace_bytes_ = 0;

    // cuDNN descriptors
    cudnnRNNDescriptor_t    rnn_desc_ = nullptr;
    cudnnRNNDataDescriptor_t x_desc_  = nullptr;
    cudnnRNNDataDescriptor_t y_desc_  = nullptr;
    cudnnTensorDescriptor_t  h_desc_  = nullptr;
    cudnnTensorDescriptor_t  c_desc_  = nullptr;

    // LSTM hidden/cell states
    float* d_hx_ = nullptr;  // (2, 128)
    float* d_cx_ = nullptr;  // (2, 128)

    // cuDNN weight descriptor for LSTM
    size_t lstm_weight_bytes_[kLstmLayers] = {};
    cudnnTensorDescriptor_t w_desc_ = nullptr;
    float* d_lstm_weights_[kLstmLayers] = {};  // packed for cuDNN

    // Forward sub-steps
    void forward_sincnet(const float* d_pcm, int n_samples);
    void forward_lstm();
    void forward_linear_head(float* d_output);

    bool load_weights(const std::string& model_path);
    bool setup_cudnn_lstm();
};

} // namespace deusridet
