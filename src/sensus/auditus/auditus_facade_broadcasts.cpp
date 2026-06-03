/**
 * @file auditus_facade_broadcasts.cpp
 * @philosophical_role Peer TU of auditus_facade.cpp that owns the four Step-7b broadcast
 *         installers: transcript, asr_log, stats, and speaker-match. Split out under R1 because
 *         install_stats_callback alone is ~280 lines of JSON assembly; keeping it in the core
 *         facade TU pushed the file past the 500-line hard cap. The declarations all still live
 *         in auditus_facade.h — this file is a pure weight-shedding peer, not a new seam.
 * @serves auditus_facade.h consumers (awaken) via the same four `install_*_callback` symbols.
 */

#include "auditus_facade.h"

#include "sensus/auditus/audio_pipeline.h"
#include "nexus/ws_server.h"
#include "communis/timeline_logger.h"
#include "conscientia/stream.h"
#include "conscientia/frame.h"
#include "orator/diarizen_soft_link_ledger.h"
#include "sensus/auditus/transcript_holdback.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace deusridet {
namespace auditus {

namespace {

template <class Db>
std::string speaker_name_by_id(Db& db, int id) {
    if (id < 0) return {};
    auto spks = db.all_speakers();
    for (const auto& spk : spks)
        if (spk.id == id) return spk.name;
    return {};
}

orator::DiarizenSoftLinkLedger& soft_link_ledger() {
    static orator::DiarizenSoftLinkLedger ledger;
    return ledger;
}

void broadcast_soft_link(WsServer& server,
                         const char* source,
                         uint64_t segment_id,
                         const orator::DiarizenSoftLinkLedger::Snapshot& snap) {
    if (snap.live_id < 0) return;
    char json[320];
    snprintf(json, sizeof(json),
        R"({"type":"speaker_id_link","source":"%s","segment_id":%llu,"live_id":%d,"stable_id":%d,"score":%.3f,"margin":%.3f,"support":%d,"committed":%s,"changed":%s})",
        source,
        (unsigned long long)segment_id,
        snap.live_id,
        snap.stable_id,
        snap.score,
        snap.margin,
        snap.support,
        snap.committed ? "true" : "false",
        snap.changed ? "true" : "false");
    server.broadcast_text(json);
}

}  // namespace

void install_transcript_callback(AudioPipeline& audio,
                                 WsServer& server,
                                 TimelineLogger& timeline,
                                 ConscientiStream& consciousness,
                                 bool llm_loaded,
                                 auditus::TranscriptHoldback* holdback) {
    // When the holdback is active, the speaker_id we broadcast at ASR time
    // is only the provisional ONLINE tracker id. The holdback rewrites it
    // from the voiceprint-anchored DiariZen identity registry up to
    // holdback_sec later, just before the transcript is committed to the
    // LLM. Emit an `asr_transcript_amend` envelope at that commit point so
    // the WebUI / capture layer can observe the FINAL speaker the LLM
    // actually consumes — closing the speaker↔content boundary that the
    // provisional broadcast leaves open.
    if (holdback) {
        holdback->set_on_commit(
            [&server](const InputItem& item,
                      double stream_start_sec,
                      double stream_end_sec) {
                std::string txt_escaped = json_escape(item.text);
                std::string spk_escaped = json_escape(item.speaker_name);
                char json[2048];
                snprintf(json, sizeof(json),
                    R"({"type":"asr_transcript_amend","text":"%s",)"
                    R"("stream_start_sec":%.2f,"stream_end_sec":%.2f,)"
                    R"("speaker_id":%d,"speaker_name":"%s"})",
                    txt_escaped.c_str(), stream_start_sec, stream_end_sec,
                    item.speaker_id, spk_escaped.c_str());
                server.broadcast_text(json);
            });
    }
    audio.set_on_transcript([&server, &timeline, &consciousness, llm_loaded, holdback]
                            (const asr::ASRResult& result, float audio_sec,
                             int speaker_id, const std::string& speaker_name,
                             float speaker_sim, float speaker_confidence,
                             const std::string& speaker_source,
                             const std::string& trigger_reason,
                             float stream_start_sec, float stream_end_sec) {
        std::string escaped = json_escape(result.text);
        std::string spk_escaped = json_escape(speaker_name);
        std::string src_escaped = json_escape(speaker_source);
        char json[2048];
        snprintf(json, sizeof(json),
            R"({"type":"asr_transcript","text":"%s","latency_ms":%.1f,"audio_sec":%.2f,)"
            R"("stream_start_sec":%.2f,"stream_end_sec":%.2f,)"
            R"("mel_ms":%.1f,"encoder_ms":%.1f,"decode_ms":%.1f,"tokens":%d,"mel_frames":%d,)"
            R"("speaker_id":%d,"speaker_name":"%s","speaker_sim":%.3f,"speaker_confidence":%.3f,"speaker_source":"%s",)"
            R"("trigger":"%s"})",
            escaped.c_str(), result.total_ms, audio_sec,
            stream_start_sec, stream_end_sec,
            result.mel_ms, result.encoder_ms, result.decode_ms,
            result.token_count, result.mel_frames,
            speaker_id, spk_escaped.c_str(), speaker_sim, speaker_confidence, src_escaped.c_str(),
            trigger_reason.c_str());
        server.broadcast_text(json);
        timeline.log_asr(result.text.c_str(), stream_start_sec, stream_end_sec,
                         result.total_ms, audio_sec, trigger_reason.c_str(),
                         speaker_id, speaker_name.c_str(), speaker_sim,
                         speaker_confidence, speaker_source.c_str());
        if (speaker_id >= 0)
            printf("[awaken] ASR: \"%s\" (%.1f ms, %.2f s) [spk=%d %s conf=%.2f src=%s]\n",
                   result.text.c_str(), result.total_ms, audio_sec,
                   speaker_id, speaker_name.c_str(), speaker_confidence, speaker_source.c_str());
        else
            printf("[awaken] ASR: \"%s\" (%.1f ms, %.2f s)\n",
                   result.text.c_str(), result.total_ms, audio_sec);

        // Inject ASR transcript into consciousness stream.
        if (!result.text.empty()) {
            InputItem item;
            item.source = InputSource::ASR;
            item.text = result.text;
            item.speaker_name = speaker_name;
            item.speaker_id = speaker_id;
            item.priority = 0.8f;
            if (holdback) {
                // Pre-prefill ("前置") mutable transcript buffer: ALWAYS
                // enqueue so DiariZen-v2 can rewrite speaker_id/name and the
                // WebUI observes `asr_transcript_amend` — even on prefill-free
                // observation runs where the LLM is not loaded. The actual
                // cs_.inject_input at drain time is gated by the stream's
                // enable_llm_ master switch, so it no-ops cleanly when prefill
                // is disabled.
                holdback->enqueue(std::move(item),
                                  (double)stream_start_sec,
                                  (double)stream_end_sec);
            } else if (llm_loaded) {
                consciousness.inject_input(std::move(item));
            }
        }
    });
}

void install_asr_log_callback(AudioPipeline& audio,
                              WsServer& server,
                              TimelineLogger& timeline) {
    audio.set_on_asr_log([&server, &timeline](const std::string& detail_json) {
        // Wrap the detail JSON inside an asr_log envelope.
        std::string msg = R"({"type":"asr_log",)" + detail_json.substr(1);
        server.broadcast_text(msg);
        if (detail_json.find(R"("stage":"fusion_shadow")") != std::string::npos) {
            timeline.log_fusion_shadow(detail_json.c_str());
        }
    });
}

void install_stats_callback(AudioPipeline& audio,
                            WsServer& server,
                            TimelineLogger& timeline) {
    // Per-install state for hysteretic multi-speaker ON/OFF logging.
    // Captured into the lambda via a heap-backed struct so each install_*
    // call gets its own fresh state (avoids static locals across reinstalls).
    struct State {
        bool multi_speaker_last = false;
        bool multi_speaker_initialized = false;
    };
    auto st_ptr = std::make_shared<State>();

    audio.set_on_stats([&audio, &server, &timeline, st_ptr]
                       (const AudioPipelineStats& st) {
        // The legacy `speaker_lists` block (CAM++ / CAM++Legacy / WL-ECAPA
        // stores) was removed 2026-06-03: in the default dual-encoder runtime
        // those three stores are never registered, so the block broadcast
        // three empty rosters from a SEPARATE id space than the live matcher.
        // The single speaker authority is now DiariZen's identity registry
        // ("S<gid>" via speaker_diarize_partial / asr_transcript_amend).
        char json[3200];
        snprintf(json, sizeof(json),
            R"({"type":"pipeline_stats","audio_t1":%lu,"audio_t1_in":%lu,"mel_frames":%lu,)"
            R"("rms":%.4f,"is_speech":%s,)"
            R"("gain":%.1f,)"
            R"("frcrn_active":%s,"frcrn_enabled":%s,"frcrn_loaded":%s,"frcrn_lat_ms":%.1f,)"
            R"("silero_prob":%.3f,"silero_speech":%s,"silero_threshold":%.2f,"silero_enabled":%s,)"
            R"("vad_source":%d,)"
            R"("speaker_id":%d,"speaker_sim":%.3f,"speaker_new":%s,"speaker_count":%d,)"
            R"("speaker_name":"%s","speaker_enabled":%s,"speaker_threshold":%.2f,"speaker_active":%s,)"
            R"("wlecapa_id":%d,"wlecapa_sim":%.3f,"wlecapa_new":%s,"wlecapa_count":%d,)"
            R"("wlecapa_exemplars":%d,"wlecapa_hits_above":%d,)"
            R"("wlecapa_name":"%s","wlecapa_enabled":%s,"wlecapa_threshold":%.2f,"wlecapa_active":%s)",
            (unsigned long)st.audio_t1_processed,
            (unsigned long)st.audio_t1_in,
            (unsigned long)st.mel_frames,
            st.last_rms,
            st.is_speech ? "true" : "false",
            audio.gain(),
            st.frcrn_active ? "true" : "false",
            audio.frcrn_enabled() ? "true" : "false",
            audio.frcrn_loaded() ? "true" : "false",
            st.frcrn_lat_ms,
            st.silero_prob, st.silero_speech ? "true" : "false",
            audio.silero_threshold(),
            audio.silero_enabled() ? "true" : "false",
            static_cast<int>(audio.vad_source()),
            st.speaker_id, st.speaker_sim,
            st.speaker_new ? "true" : "false",
            st.speaker_count, st.speaker_name,
            audio.speaker_enabled() ? "true" : "false",
            audio.speaker_threshold(),
            st.speaker_active ? "true" : "false",
            st.wlecapa_id, st.wlecapa_sim,
            st.wlecapa_new ? "true" : "false",
            st.wlecapa_count,
            st.wlecapa_exemplars, st.wlecapa_hits_above,
            st.wlecapa_name,
            audio.wlecapa_enabled() ? "true" : "false",
            audio.wlecapa_threshold(),
            st.wlecapa_active ? "true" : "false");

        // Append wlecapa margin guard value.
        std::string full_json(json);
        {
            char margin_buf[64];
            snprintf(margin_buf, sizeof(margin_buf),
                R"(,"wlecapa_margin":%.2f)", audio.wlecapa_db().min_margin());
            full_json += margin_buf;
        }

        // P1: Overlap detection stats.
        {
            char od_buf[256];
            snprintf(od_buf, sizeof(od_buf),
                R"(,"od_enabled":%s,"od_loaded":%s,"od_detected":%s,"od_ratio":%.3f,"od_lat_ms":%.1f)",
                audio.overlap_det_enabled() ? "true" : "false",
                audio.overlap_det_loaded() ? "true" : "false",
                st.overlap_detected ? "true" : "false",
                st.overlap_ratio,
                st.od_latency_ms);
            full_json += od_buf;
        }

        // P2: Speech separation stats.
        {
            char sep_buf[384];
            snprintf(sep_buf, sizeof(sep_buf),
                R"(,"sep_enabled":%s,"sep_loaded":%s,"sep_active":%s,"sep_lat_ms":%.1f,"sep_src1_rms":%.4f,"sep_src2_rms":%.4f)",
                audio.separator_enabled() ? "true" : "false",
                audio.separator_loaded() ? "true" : "false",
                st.separation_active ? "true" : "false",
                st.separation_lat_ms,
                st.sep_source1_energy,
                st.sep_source2_energy);
            full_json += sep_buf;
        }

        // ASR stats + tunable parameters.
        {
            char asr[768];
            snprintf(asr, sizeof(asr),
                R"(,"asr_enabled":%s,"asr_loaded":%s,"asr_active":%s,"asr_busy":%s,"asr_latency_ms":%.1f,"asr_audio_sec":%.2f)"
                R"(,"asr_buf_sec":%.2f,"asr_buf_has_speech":%s)"
                R"(,"asr_post_silence_ms":%d,"asr_max_buf_sec":%.1f,"asr_min_dur_sec":%.2f)"
                R"(,"asr_pre_roll_sec":%.2f,"asr_max_tokens":%d,"asr_rep_penalty":%.2f,"asr_min_energy":%.4f)"
                R"(,"asr_vad_source":%d,"asr_partial_sec":%.1f,"asr_min_speech_ratio":%.2f)"
                R"(,"asr_adaptive_silence":%s,"asr_effective_silence_ms":%d,"asr_current_silence_ms":%d)"
                R"(,"asr_adaptive_short_ms":%d,"asr_adaptive_long_ms":%d,"asr_adaptive_vlong_ms":%d)",
                audio.asr_enabled() ? "true" : "false",
                audio.asr_loaded() ? "true" : "false",
                st.asr_active ? "true" : "false",
                st.asr_busy ? "true" : "false",
                st.asr_latency_ms,
                st.asr_audio_duration_s,
                st.asr_buf_sec,
                st.asr_buf_has_speech ? "true" : "false",
                audio.asr_post_silence_ms(),
                audio.asr_max_buf_sec(),
                audio.asr_min_dur_sec(),
                audio.asr_pre_roll_sec(),
                audio.asr_max_tokens(),
                audio.asr_rep_penalty(),
                audio.asr_min_energy(),
                static_cast<int>(audio.asr_vad_source()),
                audio.asr_partial_sec(),
                audio.asr_min_speech_ratio(),
                audio.asr_adaptive_silence() ? "true" : "false",
                st.asr_effective_silence_ms,
                st.asr_post_silence_ms,
                audio.asr_adaptive_short_ms(),
                audio.asr_adaptive_long_ms(),
                audio.asr_adaptive_vlong_ms());
            full_json += asr;
        }

        // WL-ECAPA latency breakdown (when extraction happened this tick).
        if (st.wlecapa_active) {
            char lat[384];
            snprintf(lat, sizeof(lat),
                R"(,"lat_cnn_ms":%.1f,"lat_encoder_ms":%.1f,"lat_ecapa_ms":%.1f,"lat_total_ms":%.1f,"wlecapa_is_early":%s,"early_trigger_sec":%.2f,"early_enabled":%s,"min_speech_sec":%.2f)",
                st.wlecapa_lat_cnn_ms, st.wlecapa_lat_encoder_ms,
                st.wlecapa_lat_ecapa_ms, st.wlecapa_lat_total_ms,
                st.wlecapa_is_early ? "true" : "false",
                audio.early_trigger_sec(),
                audio.early_trigger_enabled() ? "true" : "false",
                audio.min_speech_sec());
            full_json += lat;

            // Change detection data.
            if (st.wlecapa_change_valid && !st.wlecapa_is_early) {
                char cd[128];
                snprintf(cd, sizeof(cd),
                    R"(,"change_similarity":%.4f)", st.wlecapa_change_sim);
                full_json += cd;
            }
        }

        // SpeakerTracker removed April 2026 — see devlog.

        // Multi-speaker assessment: fuse OD heuristic and CAM++ DB count.
        bool od_overlap = st.overlap_detected && st.overlap_ratio >= 0.15f;
        bool multi_by_count = (st.speaker_count >= 2);
        bool multi_speaker = od_overlap || multi_by_count;
        float multi_score = st.overlap_ratio;
        if (multi_by_count && multi_score < 0.50f) multi_score = 0.50f;

        char multi_source[64];
        multi_source[0] = '\0';
        if (od_overlap) strcat(multi_source, "od");
        if (multi_by_count) {
            if (multi_source[0] != '\0') strcat(multi_source, "+");
            strcat(multi_source, "speaker_count");
        }
        if (multi_source[0] == '\0') strcpy(multi_source, "none");

        char ms[128];
        snprintf(ms, sizeof(ms),
            R"(,"multi_speaker":%s,"multi_score":%.3f,"multi_source":"%s")",
            multi_speaker ? "true" : "false",
            multi_score,
            multi_source);


        if (!st_ptr->multi_speaker_initialized || multi_speaker != st_ptr->multi_speaker_last) {
            st_ptr->multi_speaker_initialized = true;
            st_ptr->multi_speaker_last = multi_speaker;
            printf("[awaken] MULTI-SPEAKER %s (score=%.2f source=%s)\n",
                   multi_speaker ? "ON" : "OFF",
                   multi_score,
                   multi_source);
        }

        full_json += ms;
        full_json += '}';
        server.broadcast_text(full_json);

        // Timeline log: compact stats.
        timeline.log_stats(st,
                           audio.wlecapa_db().min_margin(),
                           st.wlecapa_change_sim,
                           st.wlecapa_change_valid && !st.wlecapa_is_early);
    });
}

void install_speaker_match_callback(AudioPipeline& audio,
                                    WsServer& server) {
    audio.set_on_speaker([&server](const SpeakerMatch& match) {
        if (match.is_amend) {
            char json[384];
            snprintf(json, sizeof(json),
                R"({"type":"speaker_amend","target_t_close_sec":%.2f,"prior_id":%d,"prior_sim":%.3f,"id":%d,"sim":%.3f,"name":"%s"})",
                match.amend_t_close_sec,
                match.prior_speaker_id, match.prior_similarity,
                match.speaker_id, match.similarity,
                match.name.c_str());
            server.broadcast_text(json);
            printf("[awaken] Speaker amend: t=%.2f prior=%d(%.3f) -> id=%d sim=%.3f %s\n",
                   match.amend_t_close_sec,
                   match.prior_speaker_id, match.prior_similarity,
                   match.speaker_id, match.similarity,
                   match.name.empty() ? "(unnamed)" : match.name.c_str());

                 auto snap = soft_link_ledger().observe_online(match.speaker_id, match.similarity);
                 broadcast_soft_link(server, "online_amend", 0, snap);
            return;
        }
        char json[256];
        snprintf(json, sizeof(json),
            R"({"type":"speaker","id":%d,"sim":%.3f,"new":%s,"name":"%s"})",
            match.speaker_id, match.similarity,
            match.is_new ? "true" : "false",
            match.name.c_str());
        server.broadcast_text(json);
        printf("[awaken] Speaker: id=%d sim=%.3f %s%s\n",
               match.speaker_id, match.similarity,
               match.is_new ? "NEW " : "",
               match.name.empty() ? "(unnamed)" : match.name.c_str());

        auto snap = soft_link_ledger().observe_online(match.speaker_id, match.similarity);
        broadcast_soft_link(server, "online", 0, snap);
    });
}

void install_speaker_relabel_callback(AudioPipeline& audio,
                                      WsServer& server) {
    audio.set_on_speaker_relabel([&audio, &server](uint64_t segment_id,
                                                   int old_speaker_id,
                                                   int new_speaker_id,
                                                   float confidence) {
        if (old_speaker_id >= 0 && new_speaker_id >= 0) {
            // If reclustering split a previously named person onto a new ID,
            // inherit that manual label to preserve identity continuity.
            std::string old_name = speaker_name_by_id(audio.dual_db(), old_speaker_id);
            std::string new_name = speaker_name_by_id(audio.dual_db(), new_speaker_id);
            if (!old_name.empty() && new_name.empty()) {
                audio.set_speaker_name(new_speaker_id, old_name);
                char name_json[256];
                snprintf(name_json, sizeof(name_json),
                    R"({"type":"speaker_name","id":%d,"name":"%s"})",
                    new_speaker_id, old_name.c_str());
                server.broadcast_text(name_json);
                printf("[awaken] Speaker relabel inherited name: id=%d <- '%s' (from %d)\n",
                       new_speaker_id, old_name.c_str(), old_speaker_id);
            }
        }

        char json[256];
        snprintf(json, sizeof(json),
            R"({"type":"speaker_relabel","segment_id":%llu,"old_id":%d,"new_id":%d,"confidence":%.3f})",
            (unsigned long long)segment_id,
            old_speaker_id, new_speaker_id,
            (double)confidence);
        server.broadcast_text(json);

        auto snap = soft_link_ledger().observe_relabel(old_speaker_id, new_speaker_id, confidence);
        broadcast_soft_link(server, "relabel", segment_id, snap);

        printf("[awaken] Speaker relabel: seg=%llu %d -> %d conf=%.3f\n",
               (unsigned long long)segment_id,
               old_speaker_id, new_speaker_id,
               (double)confidence);
    });
}

}  // namespace auditus
}  // namespace deusridet
