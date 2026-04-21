/**
 * @file silero_vad.cpp
 * @philosophical_role Silero VAD — the lightweight VAD chosen as the default gate. Fast, robust, and noiseland-friendly.
 * @serves Auditus pipeline default VAD.
 */
// silero_vad.cpp — Silero VAD v5 native C++ inference.
//
// Pure CPU implementation using safetensors weights.
// Architecture: STFT(Conv1d) → 4×Conv1d+ReLU encoder → LSTM → Linear+Sigmoid
// ~310K params, ~600K FLOPs per 32ms window — too small for GPU kernels.

#include "silero_vad.h"
#include "../../communis/log.h"
#include "../../machina/safetensors.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace deusridet {

SileroVad::SileroVad() = default;
SileroVad::~SileroVad() = default;

bool SileroVad::init(const SileroVadConfig& cfg) {
    cfg_ = cfg;

    SafetensorsFile sf(cfg_.model_path);

    auto load_vec = [&](const char* name, std::vector<float>& dst) -> bool {
        auto t = sf.get_tensor(name);
        if (!t) {
            LOG_ERROR("SileroVAD", "Missing tensor: %s", name);
            return false;
        }
        size_t n = t->numel();
        dst.resize(n);
        std::memcpy(dst.data(), t->data(), n * sizeof(float));
        return true;
    };

    // STFT basis: (258, 1, 256) → stored flat
    if (!load_vec("stft.forward_basis_buffer", stft_basis_)) return false;

    // Encoder convolutions
    const char* enc_w_names[] = {
        "encoder.0.reparam_conv.weight", "encoder.1.reparam_conv.weight",
        "encoder.2.reparam_conv.weight", "encoder.3.reparam_conv.weight"
    };
    const char* enc_b_names[] = {
        "encoder.0.reparam_conv.bias", "encoder.1.reparam_conv.bias",
        "encoder.2.reparam_conv.bias", "encoder.3.reparam_conv.bias"
    };
    for (int i = 0; i < 4; i++) {
        if (!load_vec(enc_w_names[i], enc_w_[i])) return false;
        if (!load_vec(enc_b_names[i], enc_b_[i])) return false;
    }

    // LSTMCell weights (PyTorch gate order: i, f, g, o)
    if (!load_vec("decoder.rnn.weight_ih", lstm_wih_)) return false;
    if (!load_vec("decoder.rnn.weight_hh", lstm_whh_)) return false;
    if (!load_vec("decoder.rnn.bias_ih",   lstm_bih_)) return false;
    if (!load_vec("decoder.rnn.bias_hh",   lstm_bhh_)) return false;

    // Decoder: Conv(128→1, k=1) weight (1,128,1) → 128 values, + scalar bias
    {
        auto t = sf.get_tensor("decoder.decoder.2.weight");
        if (!t) { LOG_ERROR("SileroVAD", "Missing decoder.decoder.2.weight"); return false; }
        dec_w_.resize(t->numel());
        std::memcpy(dec_w_.data(), t->data(), t->numel() * sizeof(float));
    }
    {
        auto t = sf.get_tensor("decoder.decoder.2.bias");
        if (t) {
            dec_b_ = *static_cast<const float*>(t->data());
        } else {
            dec_b_ = 0.0f;
        }
    }

    // Init persistent state
    h_state_.assign(kLstmHidden, 0.0f);
    c_state_.assign(kLstmHidden, 0.0f);
    context_.assign(kContextSize, 0.0f);

    initialized_ = true;
    LOG_INFO("SileroVAD", "Loaded model (native): %s (sr=%d, window=%d)",
             cfg_.model_path.c_str(), cfg_.sample_rate, cfg_.window_samples);
    return true;
}

// ──────────────────────────────────────────────────────
// Forward pass helpers (all CPU, fixed sizes)
// ──────────────────────────────────────────────────────

// Right-only reflect pad: output[0..len-1] = input, output[len..len+pad-1] = reflected
static void reflect_pad_right(const float* in, int len, int pad, float* out) {
    std::memcpy(out, in, len * sizeof(float));
    for (int i = 0; i < pad; i++)
        out[len + i] = in[len - 2 - i];
}

// Conv1d: input(C_in, L_in) * weight(C_out, C_in, K) + bias → output(C_out, L_out)
static void conv1d(const float* input, int C_in, int L_in,
                   const float* weight, const float* bias,
                   int C_out, int K, int stride, int pad,
                   float* output, int L_out) {
    for (int co = 0; co < C_out; co++) {
        float b = bias ? bias[co] : 0.0f;
        for (int t = 0; t < L_out; t++) {
            float sum = b;
            int t_start = t * stride - pad;
            for (int ci = 0; ci < C_in; ci++) {
                for (int k = 0; k < K; k++) {
                    int pos = t_start + k;
                    if (pos >= 0 && pos < L_in) {
                        sum += input[ci * L_in + pos] *
                               weight[co * C_in * K + ci * K + k];
                    }
                }
            }
            output[co * L_out + t] = sum;
        }
    }
}

static void relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

// LSTMCell step with PyTorch gate order: i, f, g, o
static void lstm_cell_step(const float* x, int input_size,
                           const float* Wih, const float* Whh,
                           const float* bih, const float* bhh,
                           float* h, float* c, int hidden) {
    int H4 = 4 * hidden;
    float gates[512];  // 4*128 = 512
    assert(H4 <= 512);

    for (int g = 0; g < H4; g++) {
        float val = bih[g] + bhh[g];
        const float* wr = Wih + g * input_size;
        for (int j = 0; j < input_size; j++)
            val += wr[j] * x[j];
        const float* wh = Whh + g * hidden;
        for (int j = 0; j < hidden; j++)
            val += wh[j] * h[j];
        gates[g] = val;
    }

    // PyTorch LSTMCell: [i(0:H), f(H:2H), g(2H:3H), o(3H:4H)]
    for (int i = 0; i < hidden; i++) {
        float gi = 1.0f / (1.0f + expf(-gates[0 * hidden + i]));  // input
        float gf = 1.0f / (1.0f + expf(-gates[1 * hidden + i]));  // forget
        float gg = tanhf(gates[2 * hidden + i]);                    // cell gate
        float go = 1.0f / (1.0f + expf(-gates[3 * hidden + i]));  // output

        c[i] = gf * c[i] + gi * gg;
        h[i] = go * tanhf(c[i]);
    }
}

SileroVadResult SileroVad::process(const float* pcm, int n_samples) {
    SileroVadResult result{};
    if (!initialized_ || !pcm) return result;

    // 1. Assemble input: context(64) + pcm(512) = 576 samples
    float input[kTotalInput];
    std::memcpy(input, context_.data(), kContextSize * sizeof(float));
    int copy_len = std::min(n_samples, kWindowSize);
    std::memcpy(input + kContextSize, pcm, copy_len * sizeof(float));
    if (copy_len < kWindowSize)
        std::memset(input + kContextSize + copy_len, 0,
                    (kWindowSize - copy_len) * sizeof(float));

    // 2. Right-only reflect pad: 576 → 640 (pad [0, 64])
    float padded[kPaddedLen];
    reflect_pad_right(input, kTotalInput, kPadRight, padded);

    // 3. STFT: Conv1d(1→258, k=256, s=128, no pad) → (258, 4)
    constexpr int kStftOut = 258;
    float stft_out[kStftOut * kStftFrames];
    conv1d(padded, 1, kPaddedLen,
           stft_basis_.data(), nullptr,
           kStftOut, kNfft, kHopLength, 0,
           stft_out, kStftFrames);

    // 4. Magnitude: real[:129], imag[129:258], sqrt(r²+i²) → (129, 4)
    float magnitude[kStftBins * kStftFrames];
    for (int f = 0; f < kStftBins; f++) {
        for (int t = 0; t < kStftFrames; t++) {
            float re = stft_out[f * kStftFrames + t];
            float im = stft_out[(kStftBins + f) * kStftFrames + t];
            magnitude[f * kStftFrames + t] = sqrtf(re * re + im * im);
        }
    }

    // 5. Encoder: 4× Conv1d(k=3) + ReLU
    //    enc0: (129,4) → (128,4)  s=1
    //    enc1: (128,4) → (64,2)   s=2
    //    enc2: (64,2)  → (64,1)   s=2
    //    enc3: (64,1)  → (128,1)  s=1
    float enc0[kEnc0Out * kEnc0Frames];
    conv1d(magnitude, kStftBins, kStftFrames,
           enc_w_[0].data(), enc_b_[0].data(),
           kEnc0Out, 3, 1, 1, enc0, kEnc0Frames);
    relu_inplace(enc0, kEnc0Out * kEnc0Frames);

    float enc1[kEnc1Out * kEnc1Frames];
    conv1d(enc0, kEnc0Out, kEnc0Frames,
           enc_w_[1].data(), enc_b_[1].data(),
           kEnc1Out, 3, 2, 1, enc1, kEnc1Frames);
    relu_inplace(enc1, kEnc1Out * kEnc1Frames);

    float enc2[kEnc2Out * kEnc2Frames];
    conv1d(enc1, kEnc1Out, kEnc1Frames,
           enc_w_[2].data(), enc_b_[2].data(),
           kEnc2Out, 3, 2, 1, enc2, kEnc2Frames);
    relu_inplace(enc2, kEnc2Out * kEnc2Frames);

    float enc3[kEnc3Out * kEnc3Frames];
    conv1d(enc2, kEnc2Out, kEnc2Frames,
           enc_w_[3].data(), enc_b_[3].data(),
           kEnc3Out, 3, 1, 1, enc3, kEnc3Frames);
    relu_inplace(enc3, kEnc3Out * kEnc3Frames);

    // 6. LSTMCell: single timestep (encoder output is 128×1 → squeeze → 128)
    //    Encoder output layout: enc3[ch * 1 + 0] for ch=0..127
    //    With kEnc3Frames=1, the data is already [ch0, ch1, ...ch127]
    float lstm_in[kLstmHidden];
    for (int h = 0; h < kLstmHidden; h++)
        lstm_in[h] = enc3[h];  // enc3[h * 1 + 0]

    lstm_cell_step(lstm_in, kLstmHidden,
                   lstm_wih_.data(), lstm_whh_.data(),
                   lstm_bih_.data(), lstm_bhh_.data(),
                   h_state_.data(), c_state_.data(), kLstmHidden);

    // 7. Decoder: ReLU → Conv1d(128→1, k=1) → Sigmoid
    float dot = dec_b_;
    for (int h = 0; h < kLstmHidden; h++) {
        float v = h_state_[h];
        if (v < 0.0f) v = 0.0f;  // ReLU
        dot += v * dec_w_[h];
    }
    float prob = 1.0f / (1.0f + expf(-dot));  // sigmoid

    // Update context: last 64 samples of full input
    std::memcpy(context_.data(), input + kTotalInput - kContextSize,
                kContextSize * sizeof(float));

    result.probability = prob;
    result.is_speech = prob >= cfg_.threshold;

    // State machine: segment start/end detection
    if (result.is_speech) {
        silence_samples_ = 0;
        speech_samples_ += n_samples;
        int min_speech_samples = cfg_.min_speech_ms * cfg_.sample_rate / 1000;
        if (!in_speech_ && speech_samples_ >= min_speech_samples) {
            in_speech_ = true;
            result.segment_start = true;
        }
    } else {
        if (in_speech_) {
            silence_samples_ += n_samples;
            int min_silence_samples = cfg_.min_silence_ms * cfg_.sample_rate / 1000;
            if (silence_samples_ >= min_silence_samples) {
                in_speech_ = false;
                result.segment_end = true;
                speech_samples_ = 0;
            }
        } else {
            speech_samples_ = 0;
        }
    }

    return result;
}

void SileroVad::reset_state() {
    std::fill(h_state_.begin(), h_state_.end(), 0.0f);
    std::fill(c_state_.begin(), c_state_.end(), 0.0f);
    std::fill(context_.begin(), context_.end(), 0.0f);
    in_speech_ = false;
    speech_samples_ = 0;
    silence_samples_ = 0;
}

} // namespace deusridet
