// fsmn_vad.cpp — FunASR FSMN VAD native C++ inference (FP32, safetensors).
//
// Architecture: in_linear1(400->140) -> in_linear2(140->250) -> ReLU
//   -> 4x FSMN block: linear(250->128) -> depthwise_conv(128,k=20) + residual -> affine(128->250) -> ReLU
//   -> out_linear1(250->140) -> ReLU -> out_linear2(140->248) -> softmax
//
// ~430K FP32 params, pure CPU. On Tegra unified memory this shares the same DRAM.
//
// Adapted from FunASR runtime (https://github.com/modelscope/FunASR)
// Original: MIT License

#include "fsmn_vad.h"
#include "../../communis/log.h"
#include "../../machina/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace deusridet {

// ============================================================================
// CMVN loading from am.mvn (Kaldi-style nnet text format)
// ============================================================================

static bool load_cmvn(const std::string& path,
                      std::vector<float>& mean_shift,
                      std::vector<float>& rescale) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    int state = 0; // 0: searching, 1: in AddShift, 2: in Rescale
    while (std::getline(f, line)) {
        if (line.find("<AddShift>") != std::string::npos) {
            state = 1; continue;
        }
        if (line.find("<Rescale>") != std::string::npos) {
            state = 2; continue;
        }
        if (line.find("<LearnRateCoef>") != std::string::npos) {
            auto bracket_start = line.find('[');
            auto bracket_end   = line.rfind(']');
            if (bracket_start == std::string::npos) continue;
            std::string vals_str = line.substr(bracket_start + 1,
                bracket_end != std::string::npos
                    ? bracket_end - bracket_start - 1
                    : std::string::npos);
            std::istringstream iss(vals_str);
            float v;
            std::vector<float>& target = (state == 1) ? mean_shift : rescale;
            target.clear();
            while (iss >> v) {
                target.push_back(v);
            }
        }
    }
    return !mean_shift.empty() && !rescale.empty();
}

// ============================================================================
// FsmnVad implementation
// ============================================================================

FsmnVad::FsmnVad() = default;
FsmnVad::~FsmnVad() = default;

bool FsmnVad::init(const FsmnVadConfig& cfg) {
    cfg_ = cfg;
    threshold_ = cfg.threshold;

    int frame_len = cfg_.sample_rate * cfg_.frame_length_ms / 1000; // 400
    int hop       = cfg_.sample_rate * cfg_.frame_shift_ms / 1000;  // 160
    int n_fft = 512;
    while (n_fft < frame_len) n_fft *= 2;

    // Initialize GPU Fbank extractor.
    if (!fbank_gpu_.init(cfg_.n_mels, frame_len, hop, n_fft, cfg_.sample_rate)) {
        LOG_ERROR("FsmnVAD", "GPU Fbank init failed");
        return false;
    }

    // Load CMVN.
    if (!load_cmvn(cfg_.cmvn_path, cmvn_mean_, cmvn_istd_)) {
        LOG_ERROR("FsmnVAD", "Failed to load CMVN from %s",
                  cfg_.cmvn_path.c_str());
        return false;
    }
    int expected_dim = cfg_.n_mels * cfg_.lfr_m; // 80 * 5 = 400
    if ((int)cmvn_mean_.size() != expected_dim ||
        (int)cmvn_istd_.size() != expected_dim) {
        LOG_ERROR("FsmnVAD", "CMVN dim mismatch: got %zu/%zu, expected %d",
                  cmvn_mean_.size(), cmvn_istd_.size(), expected_dim);
        return false;
    }

    // Load weights from safetensors.
    SafetensorsFile sf(cfg_.model_path);

    auto load_vec = [&](const char* name, std::vector<float>& dst) -> bool {
        auto t = sf.get_tensor(name);
        if (!t) {
            LOG_ERROR("FsmnVAD", "Missing tensor: %s", name);
            return false;
        }
        dst.resize(t->numel());
        std::memcpy(dst.data(), t->data(), t->numel() * sizeof(float));
        return true;
    };

    // Input projection
    if (!load_vec("encoder.in_linear1.linear.weight", in_linear1_w_)) return false;
    if (!load_vec("encoder.in_linear1.linear.bias",   in_linear1_b_)) return false;
    if (!load_vec("encoder.in_linear2.linear.weight", in_linear2_w_)) return false;
    if (!load_vec("encoder.in_linear2.linear.bias",   in_linear2_b_)) return false;

    // FSMN blocks
    for (int i = 0; i < kFsmnLayers; i++) {
        char name[128];
        snprintf(name, sizeof(name), "encoder.fsmn.%d.linear.linear.weight", i);
        if (!load_vec(name, fsmn_[i].linear_w)) return false;

        snprintf(name, sizeof(name), "encoder.fsmn.%d.fsmn_block.conv_left.weight", i);
        if (!load_vec(name, fsmn_[i].conv_w)) return false;

        snprintf(name, sizeof(name), "encoder.fsmn.%d.affine.linear.weight", i);
        if (!load_vec(name, fsmn_[i].affine_w)) return false;

        snprintf(name, sizeof(name), "encoder.fsmn.%d.affine.linear.bias", i);
        if (!load_vec(name, fsmn_[i].affine_b)) return false;
    }

    // Output projection
    if (!load_vec("encoder.out_linear1.linear.weight", out_linear1_w_)) return false;
    if (!load_vec("encoder.out_linear1.linear.bias",   out_linear1_b_)) return false;
    if (!load_vec("encoder.out_linear2.linear.weight", out_linear2_w_)) return false;
    if (!load_vec("encoder.out_linear2.linear.bias",   out_linear2_b_)) return false;

    // Init streaming caches
    for (int i = 0; i < kFsmnLayers; i++) {
        caches_[i].assign(kProjDim * kCacheLen, 0.0f);
    }

    initialized_ = true;
    LOG_INFO("FsmnVAD", "Loaded (native FP32): %s (GPU Fbank, cmvn=%s, %d-bin, LFR=%dx)",
             cfg_.model_path.c_str(), cfg_.cmvn_path.c_str(),
             cfg_.n_mels, cfg_.lfr_m);
    return true;
}

// ============================================================================
// Forward pass helpers
// ============================================================================

// MatMul: out[m,n] = x[m,k] @ w[k,n] + bias[n]  (row-major)
static void matmul_bias(const float* x, int M, int K,
                        const float* w, const float* bias, int N,
                        float* out) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = bias ? bias[n] : 0.0f;
            const float* xr = x + m * K;
            for (int k = 0; k < K; k++) {
                sum += xr[k] * w[k * N + n];
            }
            out[m * N + n] = sum;
        }
    }
}

static void relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

static void softmax_inplace(float* data, int n) {
    float max_val = data[0];
    for (int i = 1; i < n; i++)
        if (data[i] > max_val) max_val = data[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++)
        data[i] *= inv_sum;
}

// ============================================================================
// Forward pass -- processes frame(s) sequentially (streaming)
// ============================================================================

float FsmnVad::forward(const float* feats, int n_frames, int feat_dim) {
    if (n_frames <= 0) return 0.0f;

    float prob = 0.0f;

    for (int f = 0; f < n_frames; f++) {
        const float* x_in = feats + f * feat_dim;

        // in_linear1: (1, 400) -> (1, 140)
        float a1[kAffineIn];
        matmul_bias(x_in, 1, kInputDim,
                    in_linear1_w_.data(), in_linear1_b_.data(), kAffineIn, a1);

        // in_linear2: (1, 140) -> (1, 250)
        float a2[kLinearDim];
        matmul_bias(a1, 1, kAffineIn,
                    in_linear2_w_.data(), in_linear2_b_.data(), kLinearDim, a2);
        relu_inplace(a2, kLinearDim);

        // Working buffer for FSMN blocks
        float x[kLinearDim];
        std::memcpy(x, a2, kLinearDim * sizeof(float));

        for (int i = 0; i < kFsmnLayers; i++) {
            // linear: (1, 250) -> (1, 128) -- no bias
            float h[kProjDim];
            matmul_bias(x, 1, kLinearDim,
                        fsmn_[i].linear_w.data(), nullptr, kProjDim, h);

            // Depthwise conv with left context cache.
            // cache: (128, 19), new h: 128 scalars -> concat -> (128, 20) -> conv -> 128
            float conv_out[kProjDim];
            for (int c = 0; c < kProjDim; c++) {
                float sum = 0.0f;
                const float* cache_row = caches_[i].data() + c * kCacheLen;
                const float* cw = fsmn_[i].conv_w.data() + c * kLorder;
                for (int k = 0; k < kCacheLen; k++)
                    sum += cache_row[k] * cw[k];
                sum += h[c] * cw[kCacheLen]; // last tap
                conv_out[c] = sum;
            }

            // Update cache: shift left by 1, append h
            for (int c = 0; c < kProjDim; c++) {
                float* cache_row = caches_[i].data() + c * kCacheLen;
                std::memmove(cache_row, cache_row + 1, (kCacheLen - 1) * sizeof(float));
                cache_row[kCacheLen - 1] = h[c];
            }

            // Residual + affine + ReLU
            float res[kProjDim];
            for (int c = 0; c < kProjDim; c++)
                res[c] = h[c] + conv_out[c];

            matmul_bias(res, 1, kProjDim,
                        fsmn_[i].affine_w.data(), fsmn_[i].affine_b.data(),
                        kLinearDim, x);
            relu_inplace(x, kLinearDim);
        }

        // out_linear1: (1, 250) -> (1, 140)
        float o1[kAffineOut];
        matmul_bias(x, 1, kLinearDim,
                    out_linear1_w_.data(), out_linear1_b_.data(), kAffineOut, o1);
        relu_inplace(o1, kAffineOut);

        // out_linear2: (1, 140) -> (1, 248)
        float logits[kOutputDim];
        matmul_bias(o1, 1, kAffineOut,
                    out_linear2_w_.data(), out_linear2_b_.data(), kOutputDim, logits);
        softmax_inplace(logits, kOutputDim);

        // P(speech) = 1 - P(silence), class 0 = silence
        prob = 1.0f - logits[0];
    }

    return prob;
}

// ============================================================================
// LFR + CMVN
// ============================================================================

int FsmnVad::apply_lfr_cmvn(std::vector<float>& out_feats) {
    int n_fbank = (int)fbank_buf_.size();
    int avail = n_fbank - lfr_consumed_;
    if (avail < cfg_.lfr_m) return 0;

    int feat_dim = cfg_.n_mels * cfg_.lfr_m; // 400
    int n_lfr = avail / cfg_.lfr_m;

    out_feats.resize(n_lfr * feat_dim);

    for (int i = 0; i < n_lfr; i++) {
        for (int j = 0; j < cfg_.lfr_m; j++) {
            int src = lfr_consumed_ + i * cfg_.lfr_m + j;
            std::memcpy(out_feats.data() + i * feat_dim + j * cfg_.n_mels,
                        fbank_buf_[src].data(),
                        cfg_.n_mels * sizeof(float));
        }
        for (int d = 0; d < feat_dim; d++) {
            float& v = out_feats[i * feat_dim + d];
            v = (v + cmvn_mean_[d]) * cmvn_istd_[d];
        }
    }

    lfr_consumed_ += n_lfr * cfg_.lfr_m;

    if (lfr_consumed_ > 1000) {
        fbank_buf_.erase(fbank_buf_.begin(),
                         fbank_buf_.begin() + lfr_consumed_);
        lfr_consumed_ = 0;
    }

    return n_lfr;
}

// ============================================================================
// Process: PCM -> Fbank -> LFR -> CMVN -> forward -> speech probability
// ============================================================================

FsmnVadResult FsmnVad::process(const int16_t* pcm, int n_samples) {
    FsmnVadResult result{};
    if (!initialized_ || !pcm || n_samples <= 0) return result;

    int new_frames = fbank_gpu_.push_pcm(pcm, n_samples);

    if (new_frames > 0) {
        std::vector<float> fbank_host(new_frames * cfg_.n_mels);
        int read = fbank_gpu_.read_fbank(fbank_host.data(), new_frames);
        for (int i = 0; i < read; i++) {
            std::vector<float> frame(cfg_.n_mels);
            std::memcpy(frame.data(),
                        fbank_host.data() + i * cfg_.n_mels,
                        cfg_.n_mels * sizeof(float));
            fbank_buf_.push_back(std::move(frame));
        }
    }

    std::vector<float> feats;
    int n_lfr = apply_lfr_cmvn(feats);
    if (n_lfr == 0) return result;

    int feat_dim = cfg_.n_mels * cfg_.lfr_m;
    float speech_prob = forward(feats.data(), n_lfr, feat_dim);

    result.probability = speech_prob;
    result.is_speech = speech_prob >= threshold_;
    return result;
}

void FsmnVad::reset_state() {
    for (int i = 0; i < kFsmnLayers; i++) {
        caches_[i].assign(kProjDim * kCacheLen, 0.0f);
    }
    fbank_buf_.clear();
    lfr_consumed_ = 0;
    fbank_gpu_.reset();
}

} // namespace deusridet
