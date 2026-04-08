// speaker_encoder.h — CAM++ GPU Speaker Encoder.
//
// Adapted from qwen35-orin speaker_encoder_gpu (Thomas Zhu)
// Architecture: FCM(ResNet) → TDNN → 3×CAMDenseTDNNBlock → StatsPool → Dense → L2
// Input:  80-dim Mel × T frames (Kaldi-compatible FBank)
// Output: 192-dim L2-normalized speaker embedding
//
// All computation on GPU via cuBLAS SGEMM + custom CUDA kernels.
// Weights loaded from safetensors (zero-copy mmap possible, currently read).
//
// Reference: FunASR CAM++ (https://github.com/modelscope/FunASR)

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace deusridet {

// Pre-allocated GPU scratch buffers, reused each DenseTDNN layer.
// Avoids per-call cudaMalloc/Free — critical for real-time pipeline.
struct SpeakerScratchPool {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    float* d = nullptr;
    float* e = nullptr;
    float* f = nullptr;
    float* concat[2] = {nullptr, nullptr};
    int which_concat = 0;
    size_t total_bytes = 0;

    bool alloc(int max_T, int max_spatial);
    void free();
    float* cur_concat()  { return concat[which_concat]; }
    float* next_concat() { return concat[1 - which_concat]; }
    void swap_concat()   { which_concat = 1 - which_concat; }
};

struct SpeakerEncoderConfig {
    std::string model_path;    // path to campplus.safetensors
    int mel_bins     = 80;     // must be 80 for CAM++
    int embedding_dim = 192;   // output embedding dimension
};

class SpeakerEncoder {
public:
    SpeakerEncoder();
    ~SpeakerEncoder();

    SpeakerEncoder(const SpeakerEncoder&) = delete;
    SpeakerEncoder& operator=(const SpeakerEncoder&) = delete;

    // Load safetensors weights to GPU.
    bool init(const SpeakerEncoderConfig& cfg);
    bool initialized() const { return initialized_; }

    // Extract 192-dim embedding from GPU Mel features.
    // d_mel: GPU pointer, [T, 80] row-major (from Mel GPU extractor).
    // Returns empty vector on error.
    std::vector<float> extract_gpu(const float* d_mel, int T);

    // Extract from CPU Mel features (uploads to GPU internally).
    // mel: [T, 80] row-major.
    std::vector<float> extract(const float* mel, int T);

    static constexpr int embedding_dim() { return 192; }

    static float cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b);

private:
    using TensorMap = std::unordered_map<std::string, std::vector<float>>;

    // Core forward pass (no sync). Writes 192-dim to d_emb_out.
    void forward_one(const float* d_mel, int T,
                     SpeakerScratchPool& sp,
                     cudaStream_t stream, cublasHandle_t cublas,
                     float* d_emb_out);

    // FCM ResBlock sub-forward.
    void gpu_res_block(const float* d_input, float* d_output,
                       int C, int H, int W,
                       const std::string& prefix, int stride,
                       float* scratch_a, float* scratch_b,
                       cudaStream_t stream);

    // CAM DenseTDNN block.
    void gpu_cam_dense_block(SpeakerScratchPool& sp, int in_dim, int T,
                             const std::string& prefix,
                             int num_layers, int dilation,
                             cublasHandle_t cublas,
                             cudaStream_t stream);

    // Single CAM layer within a DenseTDNN block.
    void gpu_cam_layer(SpeakerScratchPool& sp, int bn_ch, int out_ch,
                       int T, const std::string& prefix,
                       int k, int dilation, int padding,
                       cublasHandle_t cublas,
                       cudaStream_t stream);

    // Transit layer between DenseTDNN blocks.
    void gpu_transit(SpeakerScratchPool& sp, int in_dim, int T,
                     const std::string& prefix, int out_dim,
                     cublasHandle_t cublas,
                     cudaStream_t stream);

    // Ensure scratch is large enough for T frames (auto-grows).
    bool ensure_scratch(int T);

    // GPU weight accessors.
    const float* get_gpu(const std::string& name) const;
    TensorMap load_safetensors(const std::string& path);

    SpeakerEncoderConfig cfg_;
    bool initialized_ = false;

    // GPU weights.
    std::unordered_map<std::string, float*> gpu_tensors_;
    std::unordered_map<std::string, int> tensor_sizes_;
    cublasHandle_t cublas_ = nullptr;

    // Persistent scratch + stream.
    SpeakerScratchPool scratch_;
    cudaStream_t stream_ = nullptr;
    int scratch_max_T_ = 0;
};

} // namespace deusridet
