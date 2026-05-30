/**
 * @file diarizen_resnet34_embedder.h
 * @philosophical_role Native CUDA port of the WeSpeaker ResNet34-LM speaker
 *   embedder used by DiariZen-v2 — turns a 16 kHz chunk + speaker activity
 *   mask into a 256-d x-vector. Retires the Python embedding step of
 *   tools/diarizen_worker.py.
 * @serves Orator diarization clustering (P2b VBx consumes these embeddings).
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
#include <cudnn.h>

namespace deusridet {
namespace orator {

// Architecture constants for WeSpeaker ResNet34-LM (pyannote
// wespeaker-voxceleb-resnet34-LM): BasicBlock [3,4,6,3], base width 32,
// 80-bin Kaldi fbank, TSTP pooling, 256-d embedding, two_emb_layer=False.
struct DiarizenResnet34Arch {
    static constexpr int kSampleRate = 16000;
    static constexpr int kNumMel = 80;
    static constexpr int kFrameLen = 400;   // 25 ms
    static constexpr int kFrameShift = 160;  // 10 ms
    static constexpr int kNFft = 512;        // round_to_power_of_two(400)
    static constexpr int kBaseWidth = 32;
    static constexpr int kEmbedDim = 256;
    static constexpr int kStatsDim = 2560;   // (80/8) * 32 * 8
    static constexpr int kPoolDim = 5120;    // stats_dim * 2 (mean|std)
    static constexpr int kNumStages = 4;
    // Blocks per stage and the planes (output channels) per stage.
    static constexpr int kBlocks[4] = {3, 4, 6, 3};
    static constexpr int kPlanes[4] = {32, 64, 128, 256};
    static constexpr float kBnEps = 1e-5f;
};

// One ResNet BasicBlock's folded parameters (BN folded into per-channel
// scale/bias; convolutions carry no bias).
struct Resnet34Block {
    int in_planes = 0;
    int planes = 0;
    int stride = 1;
    bool has_shortcut = false;
    float* conv1_w = nullptr;  // [planes, in_planes, 3, 3]
    float* bn1_scale = nullptr;
    float* bn1_bias = nullptr;
    float* conv2_w = nullptr;  // [planes, planes, 3, 3]
    float* bn2_scale = nullptr;
    float* bn2_bias = nullptr;
    float* sc_w = nullptr;     // [planes, in_planes, 1, 1] (shortcut conv)
    float* sc_scale = nullptr;
    float* sc_bias = nullptr;
};

class DiarizenResnet34Embedder {
public:
    DiarizenResnet34Embedder();
    ~DiarizenResnet34Embedder();

    DiarizenResnet34Embedder(const DiarizenResnet34Embedder&) = delete;
    DiarizenResnet34Embedder& operator=(const DiarizenResnet34Embedder&) =
        delete;

    // Load fp16 safetensors (wespeaker_resnet34.safetensors). Weights are
    // converted to fp32 on device and BatchNorm is folded.
    bool load(const std::string& safetensors_path);
    bool is_loaded() const { return loaded_; }

    // Extract a 256-d embedding for one (chunk, speaker) pair.
    //   wave   : [n_samples] float at [-1, 1] scale (will be * 32768 for fbank)
    //   mask   : [num_frames] float activity weights at the segmentation frame
    //            rate (799 frames / 16 s); interpolated nearest to the pooling
    //            time axis. Pass nullptr for unweighted pooling.
    // Writes kEmbedDim floats into out_embed. Returns false on failure.
    bool embed(const float* wave, int n_samples, const float* mask,
               int num_frames, float* out_embed);

    // Split embed for the common case where many speakers share one chunk:
    // the ResNet34 backbone depends only on the waveform (the speaker mask
    // enters at pooling), so compute the backbone once per chunk via
    // embed_backbone(), then call embed_pool() per speaker. embed() above is
    // exactly embed_backbone() followed by embed_pool() and stays for the
    // bit-eq harness; the split path is numerically identical.
    bool embed_backbone(const float* wave, int n_samples);
    bool embed_pool(const float* mask, int num_frames, float* out_embed);

    // Diagnostic tap: compute the [T, 80] CMN fbank for the waveform and copy
    // to host (row-major, T rows). Used by the P2a bit-eq harness.
    bool debug_fbank(const float* wave, int n_samples,
                     std::vector<float>& out, int& out_frames);

private:
    struct HostTensor {
        std::vector<float> data;  // fp32
        std::vector<int> shape;
    };

    bool load_safetensors_(const std::string& path,
                           std::unordered_map<std::string, HostTensor>& tensors);
    float* upload_(const std::vector<float>& v);
    // Fold BN(weight,bias,mean,var) into scale/bias device buffers.
    void fold_bn_(const std::unordered_map<std::string, HostTensor>& t,
                  const std::string& prefix, float** scale_out,
                  float** bias_out);
    void build_block_(const std::unordered_map<std::string, HostTensor>& t,
                      const std::string& prefix, int in_planes, int planes,
                      int stride, Resnet34Block& blk);

    // fbank frontend
    void build_frontend_();
    int compute_fbank_(const float* wave, int n_samples, float* d_fbank_TM);

    // conv2d via cuDNN (NCHW, fp32, cross-correlation, no bias).
    void conv2d_(const float* d_in, float* d_out, int C_in, int C_out, int H,
                 int W, int K, int stride, int pad, const float* d_w,
                 int& H_out, int& W_out);
    void run_block_(const Resnet34Block& blk, float* d_in, int H, int W,
                    float* d_out, int& H_out, int& W_out, float* d_scratch);

    // Cached cuDNN convolution plan. ResNet34's conv configs are fixed once
    // the chunk length is fixed (the segmenter always feeds a full-window,
    // zero-padded 16 s chunk), so the ~36 distinct (C_in,C_out,H,W,K,stride,
    // pad) shapes recur identically across every (chunk, speaker) embed call.
    // Caching the descriptors + chosen algo removes a per-conv descriptor
    // create/destroy and a per-conv cudnnGetConvolutionForwardAlgorithm_v7
    // heuristic — the same per-iteration churn the WavLM/Conformer fixes
    // eliminated. The kernel chosen is unchanged, so results are bit-identical.
    struct ConvPlan {
        cudnnTensorDescriptor_t in_d = nullptr;
        cudnnTensorDescriptor_t out_d = nullptr;
        cudnnFilterDescriptor_t filt_d = nullptr;
        cudnnConvolutionDescriptor_t conv_d = nullptr;
        cudnnConvolutionFwdAlgo_t algo{};
        int H_out = 0;
        int W_out = 0;
    };

    bool loaded_ = false;
    cudnnHandle_t cudnn_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    std::unordered_map<std::uint64_t, ConvPlan> conv_cache_;

    // Device parameter buffers (owned).
    std::vector<float*> owned_;
    float* conv1_w_ = nullptr;
    float* bn1_scale_ = nullptr;
    float* bn1_bias_ = nullptr;
    float* seg1_w_ = nullptr;   // [256, 5120]
    float* seg1_b_ = nullptr;   // [256]
    std::vector<Resnet34Block> blocks_;

    // frontend device buffers
    float* d_window_ = nullptr;   // [400]
    float* d_mel_fb_ = nullptr;   // [80 * 257]

    // workspace
    float* d_ws_ = nullptr;
    size_t ws_bytes_ = 0;
    float* d_bufA_ = nullptr;
    float* d_bufB_ = nullptr;
    float* d_bufC_ = nullptr;
    float* d_bufD_ = nullptr;
    float* d_pool_ = nullptr;   // [5120]
    size_t buf_floats_ = 0;
    float* d_wave_ = nullptr;
    size_t wave_cap_ = 0;
    float* d_fbank_ = nullptr;
    size_t fbank_cap_floats_ = 0;

    // Persistent backbone features [kStatsDim, Tp] for the current chunk,
    // shared across that chunk's speakers (filled by embed_backbone, consumed
    // by embed_pool). Kept separate from d_bufA_..D_ because embed_pool reuses
    // those as scratch across speakers.
    float* d_backbone_ = nullptr;
    size_t backbone_cap_floats_ = 0;
    int backbone_Tp_ = 0;
};

}  // namespace orator
}  // namespace deusridet
