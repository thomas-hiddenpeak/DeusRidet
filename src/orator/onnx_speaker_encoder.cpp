// onnx_speaker_encoder.cpp — ONNX Runtime speaker encoder implementation.
//
// Wraps any ONNX speaker verification model (WavLM, UniSpeech-SAT, etc.)
// that takes raw float32 waveform [1, T] and outputs embeddings [1, D].

#include "onnx_speaker_encoder.h"
#include "../communis/log.h"

#include <cmath>
#include <cstring>

namespace deusridet {

OnnxSpeakerEncoder::OnnxSpeakerEncoder() {
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

OnnxSpeakerEncoder::~OnnxSpeakerEncoder() {
    if (session_) api_->ReleaseSession(session_);
    if (session_opts_) api_->ReleaseSessionOptions(session_opts_);
    if (mem_info_) api_->ReleaseMemoryInfo(mem_info_);
    if (env_) api_->ReleaseEnv(env_);
}

bool OnnxSpeakerEncoder::init(const OnnxSpeakerConfig& cfg) {
    cfg_ = cfg;

    OrtStatus* st = api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                     cfg_.name.c_str(), &env_);
    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] CreateEnv failed: %s",
                  cfg_.name.c_str(), api_->GetErrorMessage(st));
        api_->ReleaseStatus(st);
        return false;
    }

    st = api_->CreateSessionOptions(&session_opts_);
    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] CreateSessionOptions failed", cfg_.name.c_str());
        api_->ReleaseStatus(st);
        return false;
    }

    api_->SetIntraOpNumThreads(session_opts_, cfg_.num_threads);
    api_->SetSessionGraphOptimizationLevel(session_opts_, ORT_ENABLE_ALL);

    st = api_->CreateSession(env_, cfg_.model_path.c_str(),
                              session_opts_, &session_);
    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] CreateSession failed: %s",
                  cfg_.name.c_str(), api_->GetErrorMessage(st));
        api_->ReleaseStatus(st);
        return false;
    }

    st = api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                    &mem_info_);
    if (st) {
        api_->ReleaseStatus(st);
        return false;
    }

    LOG_INFO("OnnxSpk", "[%s] Loaded: %s (embed=%d)",
             cfg_.name.c_str(), cfg_.model_path.c_str(), cfg_.embedding_dim);
    return true;
}

std::vector<float> OnnxSpeakerEncoder::extract(const float* pcm_float,
                                                int n_samples) {
    if (!session_ || n_samples < 1600) return {};

    // RMS normalization: bring audio to a consistent energy level.
    // Browser mic audio is typically 4-6x quieter than training data
    // (RMS ~0.02 vs ~0.10), which degrades speaker discrimination.
    // Target RMS 0.1 matches typical clean speech levels.
    std::vector<float> normalized(n_samples);
    {
        double sum2 = 0;
        for (int i = 0; i < n_samples; i++) {
            sum2 += (double)pcm_float[i] * pcm_float[i];
        }
        float rms = sqrtf((float)(sum2 / n_samples));
        float gain = (rms > 1e-8f) ? (0.1f / rms) : 1.0f;
        // Clamp gain to avoid amplifying noise-only segments excessively.
        if (gain > 20.0f) gain = 20.0f;
        for (int i = 0; i < n_samples; i++) {
            normalized[i] = pcm_float[i] * gain;
        }
    }

    // Input: [1, n_samples]
    int64_t input_shape[2] = {1, (int64_t)n_samples};
    OrtValue* input_tensor = nullptr;
    OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
        mem_info_,
        normalized.data(),
        n_samples * sizeof(float),
        input_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] CreateTensor failed: %s",
                  cfg_.name.c_str(), api_->GetErrorMessage(st));
        api_->ReleaseStatus(st);
        return {};
    }

    const char* input_names[] = {cfg_.input_name.c_str()};
    const char* output_names[] = {cfg_.output_name.c_str()};
    OrtValue* output_tensor = nullptr;

    st = api_->Run(session_, nullptr,
                    input_names, &input_tensor, 1,
                    output_names, 1, &output_tensor);
    api_->ReleaseValue(input_tensor);

    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] Run failed: %s",
                  cfg_.name.c_str(), api_->GetErrorMessage(st));
        api_->ReleaseStatus(st);
        return {};
    }

    float* data = nullptr;
    api_->GetTensorMutableData(output_tensor, (void**)&data);

    OrtTensorTypeAndShapeInfo* info = nullptr;
    api_->GetTensorTypeAndShape(output_tensor, &info);
    size_t dim_count = 0;
    api_->GetDimensionsCount(info, &dim_count);
    std::vector<int64_t> dims(dim_count);
    api_->GetDimensions(info, dims.data(), dim_count);
    api_->ReleaseTensorTypeAndShapeInfo(info);

    int emb_dim = (dim_count >= 2) ? (int)dims[1] : (int)dims[0];
    std::vector<float> result(data, data + emb_dim);

    api_->ReleaseValue(output_tensor);

    // L2 normalize.
    float norm = 0;
    for (float v : result) norm += v * v;
    norm = 1.0f / (sqrtf(norm) + 1e-12f);
    for (float& v : result) v *= norm;

    return result;
}

std::vector<float> OnnxSpeakerEncoder::extract_int16(const int16_t* pcm,
                                                      int n_samples) {
    if (n_samples < 1600) return {};
    std::vector<float> pcm_float(n_samples);
    for (int i = 0; i < n_samples; i++) {
        pcm_float[i] = pcm[i] / 32768.0f;
    }
    return extract(pcm_float.data(), n_samples);
}

std::vector<float> OnnxSpeakerEncoder::extract_fbank(const float* fbank,
                                                      int T, int n_mels) {
    if (!session_ || T < 10 || n_mels < 1) return {};

    // Apply CMN (cepstral mean normalization): subtract utterance-level
    // mean per mel bin. Required by WeSpeaker ECAPA-TDNN models.
    std::vector<float> normalized(T * n_mels);
    for (int m = 0; m < n_mels; m++) {
        float sum = 0;
        for (int t = 0; t < T; t++) sum += fbank[t * n_mels + m];
        float mean = sum / T;
        for (int t = 0; t < T; t++) {
            normalized[t * n_mels + m] = fbank[t * n_mels + m] - mean;
        }
    }

    // Input: [1, T, n_mels]
    int64_t input_shape[3] = {1, (int64_t)T, (int64_t)n_mels};
    OrtValue* input_tensor = nullptr;
    OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
        mem_info_,
        normalized.data(),
        T * n_mels * sizeof(float),
        input_shape, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] CreateTensor(fbank) failed: %s",
                  cfg_.name.c_str(), api_->GetErrorMessage(st));
        api_->ReleaseStatus(st);
        return {};
    }

    const char* input_names[] = {cfg_.input_name.c_str()};
    const char* output_names[] = {cfg_.output_name.c_str()};
    OrtValue* output_tensor = nullptr;

    st = api_->Run(session_, nullptr,
                    input_names, &input_tensor, 1,
                    output_names, 1, &output_tensor);
    api_->ReleaseValue(input_tensor);

    if (st) {
        LOG_ERROR("OnnxSpk", "[%s] Run(fbank) failed: %s",
                  cfg_.name.c_str(), api_->GetErrorMessage(st));
        api_->ReleaseStatus(st);
        return {};
    }

    float* data = nullptr;
    api_->GetTensorMutableData(output_tensor, (void**)&data);

    OrtTensorTypeAndShapeInfo* info = nullptr;
    api_->GetTensorTypeAndShape(output_tensor, &info);
    size_t dim_count = 0;
    api_->GetDimensionsCount(info, &dim_count);
    std::vector<int64_t> dims(dim_count);
    api_->GetDimensions(info, dims.data(), dim_count);
    api_->ReleaseTensorTypeAndShapeInfo(info);

    int emb_dim = (dim_count >= 2) ? (int)dims[1] : (int)dims[0];
    std::vector<float> result(data, data + emb_dim);

    api_->ReleaseValue(output_tensor);

    // L2 normalize.
    float norm = 0;
    for (float v : result) norm += v * v;
    norm = 1.0f / (sqrtf(norm) + 1e-12f);
    for (float& v : result) v *= norm;

    return result;
}

float OnnxSpeakerEncoder::cosine_similarity(const std::vector<float>& a,
                                             const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

} // namespace deusridet
