/**
 * @file separatio_orator_probe.cpp
 * @philosophical_role Shadow speaker evidence for separated streams. It scores who a separated source sounds like while preserving the live speaker timeline as the authority.
 * @serves Auditus fusion shadow evidence, Orator read-only scoring.
 */
#include "separatio_orator_probe.h"
#include "audio_pipeline.h"
#include "asr/asr_engine.h"
#include "../../orator/speaker_vector_store.h"
#include "../../orator/wavlm_ecapa_encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace deusridet {

std::mutex& auditus_wlecapa_extract_mutex() {
    static std::mutex mutex;
    return mutex;
}

namespace {

std::string escape_json_probe(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char value : input) {
        if (value == '"') out += "\\\"";
        else if (value == '\\') out += "\\\\";
        else if (value == '\n') out += "\\n";
        else if (value == '\r') out += "\\r";
        else if (value == '\t') out += "\\t";
        else out += value;
    }
    return out;
}

} // namespace

bool SeparatioOratorProbe::init(const AudioPipelineConfig& cfg, bool use_dual) {
    use_dual_ = use_dual;
    backend_ = use_dual_ ? "dual" : "campp";
    if (cfg.speaker.model_path.empty()) {
        reason_ = "campp_model_missing";
        return false;
    }
    if (!campp_.init(cfg.speaker)) {
        reason_ = "campp_init_failed";
        return false;
    }
    if (!fbank_.init(80, 400, 160, 512, 16000,
                     FbankWindowType::POVEY, true)) {
        reason_ = "fbank_init_failed";
        return false;
    }
    ready_ = true;
    reason_ = "ready";
    return true;
}

ShadowSpeakerEvidence SeparatioOratorProbe::score(
        const float* samples, int n_samples, SpeakerVectorStore& db,
        WavLMEcapaEncoder* wavlm, float threshold, float min_margin,
        float stable_min_similarity, float stable_min_margin,
        int stable_min_exemplars, int stable_min_matches) {
    ShadowSpeakerEvidence ev;
    ev.ready = ready_;
    ev.backend = backend_;
    ev.reason = reason_;
    ev.threshold = threshold;
    ev.min_margin = min_margin;
    ev.stable_min_similarity = stable_min_similarity;
    ev.stable_min_margin = stable_min_margin;
    ev.stable_min_exemplars = stable_min_exemplars;
    ev.stable_min_matches = stable_min_matches;
    ev.db_speakers = db.count();
    if (!ready_) return ev;
    if (ev.db_speakers <= 0) {
        ev.reason = "speaker_store_empty";
        return ev;
    }
    if (n_samples < 4800) {
        ev.reason = "source_too_short";
        return ev;
    }

    auto extract_start = std::chrono::high_resolution_clock::now();
    std::vector<float> embedding = extract(samples, n_samples, wavlm);
    auto extract_end = std::chrono::high_resolution_clock::now();
    ev.extract_ms = std::chrono::duration<float, std::milli>(extract_end - extract_start).count();
    if (embedding.empty()) {
        ev.reason = use_dual_ ? "dual_embedding_empty" : "campp_embedding_empty";
        return ev;
    }
    ev.extracted = true;

    auto match_start = std::chrono::high_resolution_clock::now();
    SpeakerMatch match = db.peek_best(embedding);
    auto match_end = std::chrono::high_resolution_clock::now();
    ev.match_ms = std::chrono::duration<float, std::milli>(match_end - match_start).count();
    ev.speaker_id = match.speaker_id;
    ev.name = match.name;
    ev.similarity = match.similarity;
    ev.second_id = match.second_best_id;
    ev.second_similarity = match.second_best_sim;
    ev.margin = match.second_best_id >= 0
        ? match.similarity - match.second_best_sim
        : match.similarity;
    if (match.speaker_id >= 0) {
        std::vector<SpeakerInfo> speakers = db.all_speakers();
        for (const SpeakerInfo& speaker : speakers) {
            if (speaker.id != match.speaker_id) continue;
            ev.exemplar_count = speaker.exemplar_count;
            ev.match_count = speaker.match_count;
            break;
        }
    }
    ev.accepted = match.speaker_id >= 0 &&
        match.similarity >= threshold && ev.margin >= min_margin;
    ev.stable = ev.accepted &&
        match.similarity >= stable_min_similarity &&
        ev.margin >= stable_min_margin &&
        ev.exemplar_count >= stable_min_exemplars &&
        ev.match_count >= stable_min_matches;
    if (ev.accepted && !ev.stable) ev.reason = "speaker_unstable";
    else ev.reason = ev.accepted ? "accepted" : "below_threshold";
    return ev;
}

std::vector<float> SeparatioOratorProbe::extract(const float* samples,
                                                 int n_samples,
                                                 WavLMEcapaEncoder* wavlm) {
    fbank_.reset();
    std::vector<int16_t> int_samples(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float value = std::max(-1.0f, std::min(1.0f, samples[i]));
        int_samples[i] = (int16_t)std::lrintf(value * 32767.0f);
    }
    fbank_.push_pcm(int_samples.data(), n_samples);
    int frames = fbank_.frames_ready();
    if (frames < 30) return {};
    std::vector<float> mel(frames * 80);
    int read_frames = fbank_.read_fbank(mel.data(), frames);
    if (read_frames <= 0) return {};
    std::vector<float> campp = campp_.extract(mel.data(), read_frames);
    if (campp.size() != 192) return {};
    if (!use_dual_) return campp;
    if (!wavlm || !wavlm->initialized()) return {};
    std::vector<float> wavlm_emb;
    {
        std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
        wavlm_emb = wavlm->extract(samples, n_samples);
    }
    if (wavlm_emb.size() != 192) return {};
    std::vector<float> dual(384);
    std::copy(campp.begin(), campp.end(), dual.begin());
    std::copy(wavlm_emb.begin(), wavlm_emb.end(), dual.begin() + 192);
    float norm_sq = 0.0f;
    for (float value : dual) norm_sq += value * value;
    float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
    for (float& value : dual) value *= inv_norm;
    return dual;
}

std::string speaker_evidence_json(const ShadowSpeakerEvidence& ev) {
    char head[1152];
    snprintf(head, sizeof(head),
        R"({"ready":%s,"extracted":%s,"accepted":%s,"stable":%s,"backend":"%s",)"
        R"("reason":"%s","db_speakers":%d,"speaker_id":%d,)"
        R"("exemplar_count":%d,"match_count":%d,)"
        R"("similarity":%.3f,"second_id":%d,"second_similarity":%.3f,)"
        R"("margin":%.3f,"threshold":%.3f,"min_margin":%.3f,)"
        R"("stable_min_similarity":%.3f,"stable_min_margin":%.3f,)"
        R"("stable_min_exemplars":%d,"stable_min_matches":%d,)"
        R"("extract_ms":%.1f,"match_ms":%.1f)",
        ev.ready ? "true" : "false",
        ev.extracted ? "true" : "false",
        ev.accepted ? "true" : "false",
        ev.stable ? "true" : "false",
        ev.backend.c_str(), escape_json_probe(ev.reason).c_str(), ev.db_speakers,
        ev.speaker_id, ev.exemplar_count, ev.match_count,
        ev.similarity, ev.second_id, ev.second_similarity,
        ev.margin, ev.threshold, ev.min_margin,
        ev.stable_min_similarity, ev.stable_min_margin,
        ev.stable_min_exemplars, ev.stable_min_matches,
        ev.extract_ms, ev.match_ms);
    std::string out = head;
    out += ",\"name\":\"" + escape_json_probe(ev.name) + "\"}";
    return out;
}

std::string source_result_json(const asr::ASRResult& result,
                               const ShadowSpeakerEvidence& evidence) {
    return "{\"text\":\"" + escape_json_probe(result.text) +
        "\",\"text_nonempty\":" + std::string(result.text.empty() ? "false" : "true") +
        ",\"tokens\":" + std::to_string(result.token_count) +
        ",\"total_ms\":" + std::to_string(result.total_ms) +
        ",\"speaker\":" + speaker_evidence_json(evidence) + "}";
}

namespace {

bool source_has_stable_text(const asr::ASRResult& result,
                            const ShadowSpeakerEvidence& evidence) {
    return !result.text.empty() && evidence.accepted && evidence.stable;
}

bool source_has_accepted_text(const asr::ASRResult& result,
                              const ShadowSpeakerEvidence& evidence) {
    return !result.text.empty() && evidence.accepted;
}

const char* bool_json(bool value) {
    return value ? "true" : "false";
}

struct FusionArbitriumDecision {
    const char* action = "observe_only";
    const char* reason = "no_stable_source_evidence";
    int accepted_sources = 0;
    int stable_sources = 0;
    int candidate_speaker = -1;
    bool contradiction = false;
    bool split_candidate = false;
    bool src1_accepted_text = false;
    bool src2_accepted_text = false;
    bool src1_stable_text = false;
    bool src2_stable_text = false;
    bool direct_candidate = false;
};

FusionArbitriumDecision decide_fusion_arbitrium(
        const asr::ASRResult& src1_result,
        const ShadowSpeakerEvidence& src1_evidence,
        const asr::ASRResult& src2_result,
        const ShadowSpeakerEvidence& src2_evidence,
        int timeline_speaker_id) {
    FusionArbitriumDecision decision;
    decision.src1_accepted_text = source_has_accepted_text(src1_result, src1_evidence);
    decision.src2_accepted_text = source_has_accepted_text(src2_result, src2_evidence);
    decision.src1_stable_text = source_has_stable_text(src1_result, src1_evidence);
    decision.src2_stable_text = source_has_stable_text(src2_result, src2_evidence);
    decision.accepted_sources =
        (decision.src1_accepted_text ? 1 : 0) + (decision.src2_accepted_text ? 1 : 0);
    decision.stable_sources =
        (decision.src1_stable_text ? 1 : 0) + (decision.src2_stable_text ? 1 : 0);
    bool has_two_distinct = decision.src1_stable_text && decision.src2_stable_text &&
        src1_evidence.speaker_id >= 0 &&
        src2_evidence.speaker_id >= 0 &&
        src1_evidence.speaker_id != src2_evidence.speaker_id;
    bool has_two_same = decision.src1_stable_text && decision.src2_stable_text &&
        src1_evidence.speaker_id >= 0 &&
        src1_evidence.speaker_id == src2_evidence.speaker_id;

    if (decision.accepted_sources > 0 && decision.stable_sources == 0) {
        decision.reason = "speaker_store_unstable";
    } else if (has_two_distinct) {
        decision.action = "propose_split";
        decision.reason = "two_stable_sources_distinct";
        decision.split_candidate = true;
        decision.contradiction = timeline_speaker_id >= 0 &&
            timeline_speaker_id != src1_evidence.speaker_id &&
            timeline_speaker_id != src2_evidence.speaker_id;
    } else if (has_two_same) {
        decision.candidate_speaker = src1_evidence.speaker_id;
        decision.contradiction = timeline_speaker_id >= 0 &&
            timeline_speaker_id != decision.candidate_speaker;
        if (decision.contradiction) {
            decision.action = "record_contradiction";
            decision.reason = "two_stable_sources_conflict_timeline";
        } else {
            decision.action = "support_single_speaker";
            decision.reason = "two_stable_sources_same_speaker";
            decision.direct_candidate = true;
        }
    } else if (decision.stable_sources == 1) {
        const ShadowSpeakerEvidence& evidence =
            decision.src1_stable_text ? src1_evidence : src2_evidence;
        decision.candidate_speaker = evidence.speaker_id;
        if (timeline_speaker_id < 0) {
            decision.action = "fill_unknown_candidate";
            decision.reason = "one_stable_source_no_timeline_speaker";
            decision.direct_candidate = true;
        } else if (timeline_speaker_id == decision.candidate_speaker) {
            decision.action = "support_timeline";
            decision.reason = "one_stable_source_matches_timeline";
            decision.direct_candidate = true;
        } else {
            decision.action = "record_contradiction";
            decision.reason = "one_stable_source_conflicts_timeline";
            decision.contradiction = true;
        }
    } else if (src1_evidence.accepted || src2_evidence.accepted) {
        decision.reason = "source_asr_empty";
    }
    return decision;
}

const char* canary_blocker(const FusionArbitriumDecision& decision) {
    if (decision.direct_candidate && !decision.contradiction &&
        decision.candidate_speaker >= 0) {
        return "none";
    }
    if (decision.contradiction) return "contradiction";
    if (decision.split_candidate) return "split_candidate";
    if (decision.stable_sources == 0 && decision.accepted_sources > 0) {
        return "speaker_store_unstable";
    }
    if (decision.stable_sources == 0) return "no_stable_source_evidence";
    if (decision.candidate_speaker < 0) return "no_candidate_speaker";
    return "not_direct_attribution";
}

std::string stable_speaker_ids_json(const FusionArbitriumDecision& decision,
                                    const ShadowSpeakerEvidence& src1_evidence,
                                    const ShadowSpeakerEvidence& src2_evidence) {
    std::vector<int> ids;
    if (decision.src1_stable_text && src1_evidence.speaker_id >= 0) {
        ids.push_back(src1_evidence.speaker_id);
    }
    if (decision.src2_stable_text && src2_evidence.speaker_id >= 0 &&
        std::find(ids.begin(), ids.end(), src2_evidence.speaker_id) == ids.end()) {
        ids.push_back(src2_evidence.speaker_id);
    }
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out += ",";
        out += std::to_string(ids[i]);
    }
    out += "]";
    return out;
}

} // namespace

std::string fusion_arbitrium_json(const asr::ASRResult& src1_result,
                                  const ShadowSpeakerEvidence& src1_evidence,
                                  const asr::ASRResult& src2_result,
                                  const ShadowSpeakerEvidence& src2_evidence,
                                  int timeline_speaker_id) {
    FusionArbitriumDecision decision = decide_fusion_arbitrium(
        src1_result, src1_evidence, src2_result, src2_evidence,
        timeline_speaker_id);

    char json[768];
    snprintf(json, sizeof(json),
        R"({"shadow_only":true,"authoritative":false,"action":"%s",)"
        R"("reason":"%s","accepted_text_sources":%d,"stable_text_sources":%d,)"
        R"("candidate_speaker_id":%d,"timeline_speaker_id":%d,)"
        R"("split_candidate":%s,"contradiction":%s,)"
        R"("src1_accepted_text":%s,"src2_accepted_text":%s,)"
        R"("src1_stable_text":%s,"src2_stable_text":%s})",
        decision.action, decision.reason,
        decision.accepted_sources, decision.stable_sources,
        decision.candidate_speaker, timeline_speaker_id,
        bool_json(decision.split_candidate), bool_json(decision.contradiction),
        bool_json(decision.src1_accepted_text), bool_json(decision.src2_accepted_text),
        bool_json(decision.src1_stable_text), bool_json(decision.src2_stable_text));
    return json;
}

std::string fusion_evidence_ledger_json(const asr::ASRResult& src1_result,
                                        const ShadowSpeakerEvidence& src1_evidence,
                                        const asr::ASRResult& src2_result,
                                        const ShadowSpeakerEvidence& src2_evidence,
                                        int timeline_speaker_id,
                                        bool canary_enabled) {
    FusionArbitriumDecision decision = decide_fusion_arbitrium(
        src1_result, src1_evidence, src2_result, src2_evidence,
        timeline_speaker_id);
    bool canary_candidate = decision.direct_candidate &&
        !decision.contradiction && decision.candidate_speaker >= 0;
    bool canary_would_apply = canary_enabled && canary_candidate;
    int db_speakers = std::max(src1_evidence.db_speakers, src2_evidence.db_speakers);
    int unstable_accepted = decision.accepted_sources - decision.stable_sources;
    char head[1280];
    snprintf(head, sizeof(head),
        R"({"schema":"auditus_fusion_shadow_ledger.v1","authority":"shadow",)"
        R"("shadow_only":true,"canary_enabled":%s,"canary_candidate":%s,)"
        R"("canary_would_apply":%s,"canary_blocker":"%s",)"
        R"("timeline_present":%s,"timeline_speaker_id":%d,"db_speakers":%d,)"
        R"("accepted_text_sources":%d,"stable_text_sources":%d,)"
        R"("unstable_accepted_sources":%d,"candidate_speaker_id":%d,)"
        R"("split_candidate":%s,"contradiction":%s,)"
        R"("action":"%s","reason":"%s","stable_speaker_ids":)",
        bool_json(canary_enabled), bool_json(canary_candidate),
        bool_json(canary_would_apply), canary_blocker(decision),
        bool_json(timeline_speaker_id >= 0), timeline_speaker_id, db_speakers,
        decision.accepted_sources, decision.stable_sources, unstable_accepted,
        decision.candidate_speaker,
        bool_json(decision.split_candidate), bool_json(decision.contradiction),
        decision.action, decision.reason);
    std::string out = head;
    out += stable_speaker_ids_json(decision, src1_evidence, src2_evidence);
    out += "}";
    return out;
}

} // namespace deusridet
