// mossformer2.h — MossFormer2 speech separation: native CUDA forward pass.
//
// Implements the full MossFormer2_SS_16K inference pipeline on GPU using
// cuBLAS for GEMM and custom CUDA kernels for everything else.
//
// Model: MossFormer2_SS_16K from ClearerVoice-Studio (Apache-2.0)
// Architecture:
//   Conv1d encoder (k=16, s=8) → 24× [FLASH attention + Gated FSMN] → mask
//   → ConvTranspose1d decoder per speaker
//
// Input:  float32 PCM [T], 16kHz mono (GPU-resident)
// Output: float32 PCM [T] × 2 separated speakers (GPU-resident)
//
// Weights: safetensors format, 55.7M params, ~213 MB FP32
//
// Adapted from ClearerVoice-Studio MossFormer2 (Apache-2.0).
// Original: https://github.com/modelscope/ClearerVoice-Studio

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <unordered_map>

namespace deusridet {

// ============================================================================
// MossFormer2 model constants
// ============================================================================
namespace mf2 {
    constexpr int kEncDim     = 512;     // Encoder output / feature dim
    constexpr int kEncKernel  = 16;      // Encoder Conv1d kernel size
    constexpr int kEncStride  = 8;       // Encoder Conv1d stride
    constexpr int kNumSpk     = 2;       // Number of output speakers
    constexpr int kNumLayers  = 24;      // FLASH + FSMN layer pairs
    constexpr int kHiddenDim  = 2048;    // FLASH expansion (512 * 4)
    constexpr int kVUDim      = 1024;    // Hidden / 2 (v, u split)
    constexpr int kQKDim      = 128;     // Query/key dimension
    constexpr int kGroupSize  = 256;     // FLASH attention group size
    constexpr int kHeads      = 4;       // OffsetScale heads
    constexpr int kRopeDim    = 32;      // Rotary embedding dimension
    constexpr int kRopeFreqs  = 16;      // kRopeDim / 2
    constexpr int kDWKernel   = 17;      // Depthwise conv kernel in ConvModule
    constexpr int kDWPad      = 8;       // (kDWKernel - 1) / 2
    constexpr int kFsmnInner  = 256;     // FSMN inner channels
    constexpr int kFsmnOrder  = 20;      // FSMN convolution order
    constexpr int kDDCDepth   = 2;       // DilatedDenseNet depth
    constexpr int kDDCKernel  = 39;      // 2 * kFsmnOrder - 1
}

// ============================================================================
// Weight reference (points into GPU memory)
// ============================================================================
struct MF2GpuWeight {
    float* ptr   = nullptr;
    int    numel = 0;
};

// ============================================================================
// Per-FLASH-layer weight pointers (resolved at init, zero-cost at runtime)
// ============================================================================
struct FlashWeights {
    // to_hidden: ScaleNorm(512) → Linear(512→2048) → SiLU → ConvModule(2048,k=17)
    float* hidden_norm_g;      // [1]
    float* hidden_linear_w;    // [2048, 512]
    float* hidden_linear_b;    // [2048]
    float* hidden_dw_w;        // [2048, 1, 17]

    // to_qk: ScaleNorm(512) → Linear(512→128) → SiLU → ConvModule(128,k=17)
    float* qk_norm_g;          // [1]
    float* qk_linear_w;        // [128, 512]
    float* qk_linear_b;        // [128]
    float* qk_dw_w;            // [128, 1, 17]

    // OffsetScale(128, heads=4)
    float* offset_gamma;       // [4, 128]
    float* offset_beta;        // [4, 128]

    // RotaryPosEmb
    float* rope_freqs;         // [16]

    // to_out: ScaleNorm(1024) → Linear(1024→512) → SiLU → ConvModule(512,k=17)
    float* out_norm_g;         // [1]
    float* out_linear_w;       // [512, 1024]
    float* out_linear_b;       // [512]
    float* out_dw_w;           // [512, 1, 17]
};

// ============================================================================
// Per-FSMN-block weight pointers
// ============================================================================
struct FsmnWeights {
    // conv1: Conv1d(512→256, k=1) + PReLU(1)
    float* conv1_w;            // [256, 512, 1]
    float* conv1_b;            // [256]
    float* conv1_prelu;        // [1]

    // norm1: CLayerNorm(256)
    float* norm1_w;            // [256]
    float* norm1_b;            // [256]

    // gated_fsmn.to_u: LayerNorm(256) → Linear(256→256) → SiLU → ConvModule(256,k=17)
    float* to_u_norm_w;        // [256]
    float* to_u_norm_b;        // [256]
    float* to_u_linear_w;      // [256, 256]
    float* to_u_linear_b;      // [256]
    float* to_u_dw_w;          // [256, 1, 17]

    // gated_fsmn.to_v: same as to_u
    float* to_v_norm_w;        // [256]
    float* to_v_norm_b;        // [256]
    float* to_v_linear_w;      // [256, 256]
    float* to_v_linear_b;      // [256]
    float* to_v_dw_w;          // [256, 1, 17]

    // gated_fsmn.fsmn: Linear(256→256) → ReLU → Project(256→256) → DDC → residual
    float* fsmn_linear_w;      // [256, 256]
    float* fsmn_linear_b;      // [256]
    float* fsmn_project_w;     // [256, 256] (no bias)

    // DilatedDenseNet: 2 layers
    float* ddc_conv1_w;        // [256, 1, 39, 1]   (dilation=1)
    float* ddc_norm1_w;        // [256]
    float* ddc_norm1_b;        // [256]
    float* ddc_prelu1_w;       // [256]
    float* ddc_conv2_w;        // [256, 2, 39, 1]   (dilation=2)
    float* ddc_norm2_w;        // [256]
    float* ddc_norm2_b;        // [256]
    float* ddc_prelu2_w;       // [256]

    // norm2: CLayerNorm(256)
    float* norm2_w;            // [256]
    float* norm2_b;            // [256]

    // conv2: Conv1d(256→512, k=1)
    float* conv2_w;            // [512, 256, 1]
    float* conv2_b;            // [512]
};

// ============================================================================
// MossFormer2 — Main GPU inference class
// ============================================================================
class MossFormer2 {
public:
    MossFormer2();
    ~MossFormer2();

    MossFormer2(const MossFormer2&) = delete;
    MossFormer2& operator=(const MossFormer2&) = delete;

    // Load weights from safetensors file and allocate GPU buffers.
    // model_path: path to .safetensors file (single file)
    // max_samples: maximum input PCM length in samples
    bool init(const std::string& model_path, int max_samples = 32000,
              cudaStream_t stream = nullptr);

    // Separate mixed PCM into 2 speaker streams.
    // All pointers are GPU-resident float32.
    // d_pcm_in:  [n_samples] mixed input
    // d_source1: [n_samples] separated speaker 1 output
    // d_source2: [n_samples] separated speaker 2 output
    // n_samples must be <= max_samples.
    bool forward(const float* d_pcm_in, float* d_source1, float* d_source2,
                 int n_samples);

    bool initialized() const { return initialized_; }
    float last_latency_ms() const { return last_lat_ms_; }

    // Debug: expose internal buffers for stage-by-stage validation.
    // All pointers are GPU-resident. Check null before use.
    float* dbg_enc_out() const { return d_enc_out_; }  // [kEncDim, max_L]
    float* dbg_x() const { return d_x_; }              // [max_L, kEncDim]
    float* dbg_skip() const { return d_skip_; }        // [max_L, kEncDim]
    float* dbg_hidden() const { return d_hidden_; }    // [max_L, kHiddenDim]
    float* dbg_work_a() const { return d_work_a_; }    // [max_L, kHiddenDim]
    float* dbg_work_c() const { return d_work_c_; }    // [max_L, kEncDim]
    float* dbg_masks() const { return d_masks_; }      // [kNumSpk*kEncDim, max_L]

    // Debug: run individual pipeline stages
    void dbg_forward_encoder(const float* d_pcm, int n_samples, int L) {
        forward_encoder(d_pcm, n_samples, L);
    }
    void dbg_forward_masknet(int L) { forward_masknet(L); }

    // Debug: run masknet up to a specific stage, then return.
    // Stages: 1=CLNorm, 2=Conv1d_enc, 3=Transpose+SinuEmb, 4=Skip,
    //         5=24 layers, 6=LayerNorm, 7=AddSkip, 8=Transpose, 9=IntraNorm,
    //         10=PReLU, 11=Conv1d_out, 12=Gating, 13=MaskApply
    // After stage 1: d_work_c_ has CLNorm [512,L]
    // After stage 2: d_work_a_ has conv_enc [512,L]
    // After stage 3: d_x_ has pos_enc [L,512]
    // After stage 5+N (N-th layer pair): run N layer pairs, d_x_ [L,512]
    void dbg_forward_masknet_stage(int L, int stage);

    // Debug: run a single FLASH layer + FSMN block pair
    void dbg_forward_layer_pair(int idx, int L) {
        forward_flash_layer(idx, L);
        forward_fsmn_block(idx, L);
    }
    void dbg_forward_flash_layer(int idx, int L) { forward_flash_layer(idx, L); }
    void dbg_forward_fsmn_block(int idx, int L) { forward_fsmn_block(idx, L); }

    // Debug: expose additional buffers for sub-step validation
    float* dbg_qk() const { return d_qk_; }
    float* dbg_work_b() const { return d_work_b_; }
    float* dbg_fsmn_a() const { return d_fsmn_a_; }
    float* dbg_fsmn_b() const { return d_fsmn_b_; }

    // Debug: run FLASH layer 0 up to a sub-step, then return.
    // Steps: 1=token_shift(d_x_), 2=to_hidden(d_hidden_), 3=to_qk(d_qk_),
    //        4=offset_scale, 5=attention, 6=gate(d_hidden_), 7=to_out(d_x_), 8=residual_add(d_x_)
    void dbg_forward_flash_substep(int idx, int L, int substep);

    // Debug: run FSMN block up to a sub-step, then return.
    // Steps: 1=conv1+prelu(d_fsmn_a_[256,L]), 2=norm1(d_fsmn_b_[256,L]),
    //        3=to_u(d_fsmn_b_[L,256]), 4=to_v(d_hidden_[L,256]),
    //        5=uni_fsmn(d_fsmn_b_[L,256]), 6=gate(d_fsmn_b_[L,256]),
    //        7=norm2(d_fsmn_b_[256,L]), 8=conv2+residual(d_x_[L,512])
    void dbg_forward_fsmn_substep(int idx, int L, int substep);

private:
    // Weight management
    bool load_weights(const std::string& path);
    float* wp(const std::string& name) const;  // get weight pointer
    bool resolve_layer_weights();

    // Buffer allocation
    bool alloc_scratch(int max_L);
    void free_scratch();

    // Forward pass stages
    void forward_encoder(const float* d_pcm, int n_samples, int L);
    void forward_masknet(int L);
    void forward_decoder(int L, int n_samples, float* d_out1, float* d_out2);

    // Layer-level operations
    void forward_flash_layer(int idx, int L);
    void forward_fsmn_block(int idx, int L);

    // FFConvM: ScaleNorm/LayerNorm → Linear → SiLU → ConvModule → (dropout=noop)
    // Reads from d_in [L, dim_in], writes to d_out [L, dim_out]
    // d_tmp: scratch [L, dim_out] for ConvModule intermediate
    void forward_ffconvm_scale(const float* d_in, float* d_out, float* d_tmp,
                               int L, int dim_in, int dim_out,
                               float* norm_g,
                               float* linear_w, float* linear_b,
                               float* dw_w);

    void forward_ffconvm_layer(const float* d_in, float* d_out, float* d_tmp,
                               int L, int dim_in, int dim_out,
                               float* norm_w, float* norm_b,
                               float* linear_w, float* linear_b,
                               float* dw_w);

    // Attention computation (quadratic + linear, in groups)
    void forward_attention(int L);

    // DilatedDenseNet forward
    void forward_ddc(const float* d_in, float* d_out, int L,
                     const FsmnWeights& fw);

    // UniDeepFsmn_dilated forward
    void forward_uni_fsmn(const float* d_in, float* d_out, int L,
                          const FsmnWeights& fw);

    // ---- State ----
    bool initialized_ = false;
    float last_lat_ms_ = 0.0f;
    int max_samples_ = 0;
    int max_L_ = 0;               // max encoder output length

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    // ---- GPU weight storage ----
    float* d_weights_ = nullptr;   // single contiguous GPU allocation
    size_t weights_bytes_ = 0;
    std::unordered_map<std::string, MF2GpuWeight> wmap_;

    // ---- Resolved per-layer weight pointers ----
    FlashWeights flash_w_[mf2::kNumLayers];
    FsmnWeights  fsmn_w_[mf2::kNumLayers];

    // ---- Top-level weights ----
    float* enc_w_     = nullptr;   // [512, 1, 16]  Encoder Conv1d
    float* dec_w_     = nullptr;   // [512, 1, 16]  Decoder ConvTranspose1d
    float* mn_norm_w_ = nullptr;   // [512]  MaskNet input LayerNorm
    float* mn_norm_b_ = nullptr;   // [512]
    float* mn_conv_enc_w_ = nullptr; // [512, 512, 1]  conv1d_encoder
    float* pos_inv_freq_  = nullptr; // [256]  ScaledSinuEmbedding
    float* pos_scale_     = nullptr; // [1]
    float* blk_norm_w_ = nullptr;  // [512]  intra_mdl.norm (block internal)
    float* blk_norm_b_ = nullptr;  // [512]
    float* intra_norm_w_ = nullptr; // [512]  intra_norm (after block)
    float* intra_norm_b_ = nullptr; // [512]
    float* prelu_w_    = nullptr;  // [1]   PReLU after computation block
    float* conv_out_w_ = nullptr;  // [1024, 512, 1]  conv1d_out
    float* conv_out_b_ = nullptr;  // [1024]
    float* output_w_   = nullptr;  // [512, 512, 1]   output (Tanh gate)
    float* output_b_   = nullptr;  // [512]
    float* outgate_w_  = nullptr;  // [512, 512, 1]   output_gate (Sigmoid gate)
    float* outgate_b_  = nullptr;  // [512]
    float* conv1_dec_w_ = nullptr; // [512, 512, 1]   conv1_decoder

    // ---- Scratch buffers ----
    // Encoder output [kEncDim, max_L] — preserved through masknet for mask apply
    float* d_enc_out_ = nullptr;

    // Main state tensor [max_L, kEncDim] — the "x" flowing through layers
    float* d_x_ = nullptr;

    // Skip connection buffer for computation block
    float* d_skip_ = nullptr;      // [max_L, kEncDim]

    // FLASH layer scratch
    float* d_hidden_ = nullptr;    // [max_L, kHiddenDim] — to_hidden output
    float* d_qk_ = nullptr;        // [max_L, kQKDim] — to_qk output
    float* d_work_a_ = nullptr;    // [max_L, kHiddenDim] — general scratch
    float* d_work_b_ = nullptr;    // [max_L, kHiddenDim] — attention output scratch
    float* d_work_c_ = nullptr;    // [max_L, kEncDim] — medium scratch

    // FSMN block scratch
    float* d_fsmn_a_ = nullptr;    // [max_L, kFsmnInner] — FSMN intermediates
    float* d_fsmn_b_ = nullptr;    // [max_L, kFsmnInner]
    float* d_fsmn_c_ = nullptr;    // [max_L, kFsmnInner]

    // DDC scratch (DilatedDenseNet needs concat buffer)
    float* d_ddc_cat_ = nullptr;   // [max_L, kFsmnInner * 3] — concat in DDC

    // Mask output [kNumSpk, kEncDim, max_L]
    float* d_masks_ = nullptr;

    // GroupNorm(1,C) reduction scratch — 2 floats (sum, sumsq)
    float* d_gn_stats_ = nullptr;

    // Decoder output staging
    float* d_dec_tmp_ = nullptr;   // [max_samples]
};

} // namespace deusridet
