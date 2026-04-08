// onnx_speaker_encoder.h — ONNX-based speaker encoder for waveform-input models.
//
// Supports any ONNX speaker verification model that takes raw waveform
// input and produces a fixed-size embedding vector. Examples:
//   - WavLM-Base-Plus-SV (microsoft, 512-dim)
//   - UniSpeech-SAT-Base-Plus-SV (microsoft, 512-dim)
//   - Data2Vec-Audio-Base-SV (facebook, 512-dim)
//
// Uses ONNX Runtime for inference (CPU). Models take float32 PCM [1, T].

#pragma once

#include <onnxruntime_c_api.h>

#include <string>
#include <vector>

namespace deusridet {

struct OnnxSpeakerConfig {
    std::string model_path;       // path to .onnx file
    std::string name;             // display name ("WavLM", "UniSpeech", etc.)
    std::string input_name  = "input_values";
    std::string output_name = "embeddings";
    int embedding_dim       = 512;
    int sample_rate         = 16000;
    int num_threads         = 2;  // ORT intra-op threads
};

class OnnxSpeakerEncoder {
public:
    OnnxSpeakerEncoder();
    ~OnnxSpeakerEncoder();

    OnnxSpeakerEncoder(const OnnxSpeakerEncoder&) = delete;
    OnnxSpeakerEncoder& operator=(const OnnxSpeakerEncoder&) = delete;

    bool init(const OnnxSpeakerConfig& cfg);
    bool initialized() const { return session_ != nullptr; }
    const std::string& name() const { return cfg_.name; }
    int embedding_dim() const { return cfg_.embedding_dim; }

    // Extract embedding from float32 PCM waveform.
    // pcm_float: [n_samples] float32, range roughly [-1, 1].
    std::vector<float> extract(const float* pcm_float, int n_samples);

    // Extract from int16 PCM (converts to float internally).
    std::vector<float> extract_int16(const int16_t* pcm, int n_samples);

    // Extract from pre-computed Fbank features [T, n_mels] row-major.
    // Applies CMN (cepstral mean normalization) internally.
    // For models like ECAPA-TDNN that take Fbank input instead of raw PCM.
    std::vector<float> extract_fbank(const float* fbank, int T, int n_mels);

    static float cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b);

private:
    OnnxSpeakerConfig cfg_;
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtSessionOptions* session_opts_ = nullptr;
    OrtMemoryInfo* mem_info_ = nullptr;
};

} // namespace deusridet
