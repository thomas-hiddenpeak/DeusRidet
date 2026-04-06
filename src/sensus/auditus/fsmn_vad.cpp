// fsmn_vad.cpp — FunASR FSMN VAD implementation (GPU Fbank + ORT CPU).
//
// Pipeline: PCM → GPU Fbank(80, Hamming+preemph, 25ms/10ms)
//         → LFR(5x) → CMVN → ORT CPU → speech prob
//
// model_quant.onnx uses DynamicQuantizeLinear/MatMulInteger (TRT incompatible).
// On Tegra unified memory, CPU EP shares the same physical DRAM — no penalty.
//
// Adapted from FunASR runtime (https://github.com/modelscope/FunASR)
// Original: MIT License

#include "fsmn_vad.h"
#include "../../communis/log.h"

#include <onnxruntime_cxx_api.h>

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

FsmnVad::~FsmnVad() {
    if (session_) {
        delete static_cast<Ort::Session*>(session_);
        session_ = nullptr;
    }
    if (env_) {
        delete static_cast<Ort::Env*>(env_);
        env_ = nullptr;
    }
}

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

    // Create ONNX Runtime session (CPU EP — unified memory on Tegra).
    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "fsmn_vad");
        env_ = env;

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto* session = new Ort::Session(*env, cfg_.model_path.c_str(), opts);
        session_ = session;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("FsmnVAD", "ORT init failed: %s", e.what());
        return false;
    }

    // Init caches to zeros: each [1, 128, 19].
    for (int i = 0; i < NUM_CACHES; i++) {
        caches_[i].assign(CACHE_DIM * CACHE_LEN, 0.0f);
    }

    initialized_ = true;
    LOG_INFO("FsmnVAD", "Loaded: %s (GPU Fbank + ORT CPU, cmvn=%s, %d-bin, LFR=%dx)",
             cfg_.model_path.c_str(), cfg_.cmvn_path.c_str(),
             cfg_.n_mels, cfg_.lfr_m);
    return true;
}

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

    // Trim consumed frames to prevent unbounded growth.
    if (lfr_consumed_ > 1000) {
        fbank_buf_.erase(fbank_buf_.begin(),
                         fbank_buf_.begin() + lfr_consumed_);
        lfr_consumed_ = 0;
    }

    return n_lfr;
}

float FsmnVad::run_onnx(const float* feats, int n_frames, int feat_dim) {
    auto* session = static_cast<Ort::Session*>(session_);
    if (!session || n_frames <= 0) return 0.0f;

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input: speech [1, T, 400].
    std::vector<int64_t> speech_shape = {1, n_frames, feat_dim};
    auto speech_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(feats),
        n_frames * feat_dim, speech_shape.data(), 3);

    // Input: in_cache0..3 [1, 128, 19, 1].
    std::vector<int64_t> cache_shape = {1, CACHE_DIM, CACHE_LEN, 1};
    Ort::Value cache_tensors[NUM_CACHES] = {
        Ort::Value::CreateTensor<float>(mem, caches_[0].data(),
            CACHE_DIM * CACHE_LEN, cache_shape.data(), 4),
        Ort::Value::CreateTensor<float>(mem, caches_[1].data(),
            CACHE_DIM * CACHE_LEN, cache_shape.data(), 4),
        Ort::Value::CreateTensor<float>(mem, caches_[2].data(),
            CACHE_DIM * CACHE_LEN, cache_shape.data(), 4),
        Ort::Value::CreateTensor<float>(mem, caches_[3].data(),
            CACHE_DIM * CACHE_LEN, cache_shape.data(), 4),
    };

    // Assemble inputs.
    const char* input_names[] = {
        "speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"
    };
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(speech_tensor));
    for (int i = 0; i < NUM_CACHES; i++)
        inputs.push_back(std::move(cache_tensors[i]));

    // Output names.
    const char* output_names[] = {
        "logits", "out_cache0", "out_cache1", "out_cache2", "out_cache3"
    };

    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names, inputs.data(), 5,
            output_names, 5);

        // Update caches from output: out_cache → caches_ for next call.
        for (int i = 0; i < NUM_CACHES; i++) {
            const float* out_cache = outputs[1 + i].GetTensorData<float>();
            std::memcpy(caches_[i].data(), out_cache,
                        CACHE_DIM * CACHE_LEN * sizeof(float));
        }

        // Output "logits" is already softmax probabilities [1, T_out, 248].
        // Class 0 = silence. P(speech) = 1 - P(silence).
        // Do NOT apply softmax again — the ONNX model output is post-softmax.
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int t_out = (shape.size() >= 2) ? (int)shape[1] : n_frames;
        int n_cls = (shape.size() >= 3) ? (int)shape[2] : 248;

        const float* probs = outputs[0].GetTensorData<float>();
        const float* last = probs + (t_out - 1) * n_cls;

        float sil_prob = last[0]; // class 0 = silence
        return 1.0f - sil_prob;
    } catch (const Ort::Exception& e) {
        LOG_ERROR("FsmnVAD", "ORT run failed: %s", e.what());
        return 0.0f;
    }
}

FsmnVadResult FsmnVad::process(const int16_t* pcm, int n_samples) {
    FsmnVadResult result{};
    if (!initialized_ || !pcm || n_samples <= 0) return result;

    // Push PCM to GPU Fbank — produces fbank frames on device.
    int new_frames = fbank_gpu_.push_pcm(pcm, n_samples);

    // Read new fbank frames from GPU to host.
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

    // Apply LFR + CMVN.
    std::vector<float> feats;
    int n_lfr = apply_lfr_cmvn(feats);
    if (n_lfr == 0) return result;

    // Run ONNX Runtime inference (CPU EP).
    int feat_dim = cfg_.n_mels * cfg_.lfr_m;
    float speech_prob = run_onnx(feats.data(), n_lfr, feat_dim);

    result.probability = speech_prob;
    result.is_speech = speech_prob >= threshold_;
    return result;
}

void FsmnVad::reset_state() {
    for (int i = 0; i < NUM_CACHES; i++) {
        caches_[i].assign(CACHE_DIM * CACHE_LEN, 0.0f);
    }
    fbank_buf_.clear();
    lfr_consumed_ = 0;
    fbank_gpu_.reset();
}

} // namespace deusridet
