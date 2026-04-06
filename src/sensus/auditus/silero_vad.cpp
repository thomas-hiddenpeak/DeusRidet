// silero_vad.cpp — Silero VAD v5 ONNX Runtime inference.
//
// Uses ONNX Runtime C++ API (CPU EP) for inference.
// Model is stateful (LSTM) — state carried between process() calls.

#include "silero_vad.h"
#include "../../communis/log.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <cstring>

namespace deusridet {

SileroVad::SileroVad() = default;

SileroVad::~SileroVad() {
    if (session_) {
        delete static_cast<Ort::Session*>(session_);
        session_ = nullptr;
    }
    if (env_) {
        delete static_cast<Ort::Env*>(env_);
        env_ = nullptr;
    }
}

bool SileroVad::init(const SileroVadConfig& cfg) {
    cfg_ = cfg;

    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "silero_vad");
        env_ = env;

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto* session = new Ort::Session(*env, cfg_.model_path.c_str(), opts);
        session_ = session;

        // Init LSTM state to zeros: [2, 1, 128].
        state_.assign(2 * 1 * STATE_DIM, 0.0f);

        // Init context buffer to zeros.
        int ctx_size = (cfg_.sample_rate == 16000) ? CONTEXT_SIZE_16K : CONTEXT_SIZE_8K;
        context_.assign(ctx_size, 0.0f);

        initialized_ = true;
        LOG_INFO("SileroVAD", "Loaded model: %s (sr=%d, window=%d)",
                 cfg_.model_path.c_str(), cfg_.sample_rate, cfg_.window_samples);
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("SileroVAD", "ONNX Runtime error: %s", e.what());
        return false;
    }
}

SileroVadResult SileroVad::process(const float* pcm, int n_samples) {
    SileroVadResult result{};
    if (!initialized_ || !pcm) return result;

    auto* session = static_cast<Ort::Session*>(session_);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // Prepend context (64 samples @16k) to input window.
    int ctx_size = (int)context_.size();
    int total = ctx_size + n_samples;
    std::vector<float> input_buf(total);
    std::memcpy(input_buf.data(), context_.data(), ctx_size * sizeof(float));
    std::memcpy(input_buf.data() + ctx_size, pcm, n_samples * sizeof(float));

    // Input tensor: "input" [1, total]
    int64_t input_shape[] = {1, (int64_t)total};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_buf.data(), total, input_shape, 2);

    // State tensor: "state" [2, 1, 128]
    int64_t state_shape[] = {2, 1, STATE_DIM};
    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        mem_info, state_.data(), state_.size(), state_shape, 3);

    // Sample rate tensor: "sr" scalar (0-d tensor).
    int64_t sr_val = cfg_.sample_rate;
    Ort::Value sr_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, &sr_val, 1, nullptr, 0);

    const char* input_names[]  = {"input", "state", "sr"};
    const char* output_names[] = {"output", "stateN"};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    inputs.push_back(std::move(state_tensor));
    inputs.push_back(std::move(sr_tensor));

    try {
        auto outputs = session->Run(Ort::RunOptions{nullptr},
                                    input_names, inputs.data(), 3,
                                    output_names, 2);

        float prob = outputs[0].GetTensorData<float>()[0];
        result.probability = prob;
        result.is_speech = prob >= cfg_.threshold;

        // Update LSTM state from "stateN".
        const float* new_state = outputs[1].GetTensorData<float>();
        auto state_info = outputs[1].GetTensorTypeAndShapeInfo();
        size_t state_size = state_info.GetElementCount();
        if (state_size == state_.size()) {
            std::memcpy(state_.data(), new_state, state_size * sizeof(float));
        }

        // Update context: last ctx_size samples.
        std::memcpy(context_.data(), input_buf.data() + total - ctx_size,
                    ctx_size * sizeof(float));

    } catch (const Ort::Exception& e) {
        LOG_ERROR("SileroVAD", "Inference error: %s", e.what());
        return result;
    }

    // State machine: segment start/end detection.
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
    std::fill(state_.begin(), state_.end(), 0.0f);
    std::fill(context_.begin(), context_.end(), 0.0f);
    in_speech_ = false;
    speech_samples_ = 0;
    silence_samples_ = 0;
}

} // namespace deusridet
