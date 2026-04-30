/**
 * @file separatio_orator_probe.h
 * @philosophical_role Declaration of the shadow identity probe for separated Auditus streams. It lets divided hearing ask who spoke without changing the live speaker timeline.
 * @serves Auditus fusion shadow evidence, Orator read-only scoring.
 */
#pragma once

#include "povey_fbank_gpu.h"
#include "../../orator/speaker_encoder.h"

#include <mutex>
#include <string>
#include <vector>

namespace deusridet {

struct AudioPipelineConfig;
class SpeakerVectorStore;
class WavLMEcapaEncoder;

namespace asr {
struct ASRResult;
}

std::mutex& auditus_wlecapa_extract_mutex();

struct ShadowSpeakerEvidence {
    bool ready = false;
    bool extracted = false;
    bool accepted = false;
    std::string backend;
    std::string reason;
    int db_speakers = 0;
    int speaker_id = -1;
    int second_id = -1;
    int exemplar_count = 0;
    int match_count = 0;
    std::string name;
    float similarity = 0.0f;
    float second_similarity = 0.0f;
    float margin = 0.0f;
    float threshold = 0.0f;
    float min_margin = 0.0f;
    float extract_ms = 0.0f;
    float match_ms = 0.0f;
    bool stable = false;
    int stable_min_exemplars = 0;
    int stable_min_matches = 0;
};

class SeparatioOratorProbe {
public:
    bool init(const AudioPipelineConfig& cfg, bool use_dual);
    ShadowSpeakerEvidence score(const float* samples, int n_samples,
                                SpeakerVectorStore& db,
                                WavLMEcapaEncoder* wavlm,
                                float threshold, float min_margin,
                                int stable_min_exemplars,
                                int stable_min_matches);

private:
    std::vector<float> extract(const float* samples, int n_samples,
                               WavLMEcapaEncoder* wavlm);

    bool ready_ = false;
    bool use_dual_ = false;
    std::string backend_ = "campp";
    std::string reason_ = "not_initialized";
    SpeakerEncoder campp_;
    PoveyFbankGpu fbank_;
};

std::string speaker_evidence_json(const ShadowSpeakerEvidence& ev);
std::string source_result_json(const asr::ASRResult& result,
                               const ShadowSpeakerEvidence& evidence);
std::string fusion_arbitrium_json(const asr::ASRResult& src1_result,
                                  const ShadowSpeakerEvidence& src1_evidence,
                                  const asr::ASRResult& src2_result,
                                  const ShadowSpeakerEvidence& src2_evidence,
                                  int timeline_speaker_id);
std::string fusion_evidence_ledger_json(const asr::ASRResult& src1_result,
                                        const ShadowSpeakerEvidence& src1_evidence,
                                        const asr::ASRResult& src2_result,
                                        const ShadowSpeakerEvidence& src2_evidence,
                                        int timeline_speaker_id);

} // namespace deusridet
