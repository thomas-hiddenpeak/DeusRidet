// timeline_logger.cpp — JSONL timeline event serialization
//
// Compact, analysis-focused JSON — omits speaker roster, config params,
// and other data that doesn't change between ticks.

#include "communis/timeline_logger.h"
#include "communis/tempus.h"
#include "sensus/auditus/audio_pipeline.h"

namespace deusridet {

void TimelineLogger::log_stats(const AudioPipelineStats& st,
                               const TrackerStats& ts,
                                float wlecapa_margin,
                                float change_sim, bool change_valid) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!fp_) return;

    double stream_sec = st.audio_t1_processed / 16000.0;
    // Sample-accurate T0: map AUDIO T1 (sample index) back to wall time via
    // the tempus anchor. Falls back to 0 until AudioPipeline::start() has run.
    uint64_t t0_ns = tempus::t1_to_t0(tempus::Domain::AUDIO,
                                      st.audio_t1_processed);

    // Build compact JSON — only fields relevant for timeline analysis.
    char buf[1024];
    int n = snprintf(buf, sizeof(buf),
        R"({"t":"stats","t0":%lu,"audio_t1":%lu,"audio_t1_in":%lu,"s":%.4f,)"
        R"("speech":%s,"energy":%.2f,"rms":%.4f,)"
        R"("silero_p":%.3f,"silero_sp":%s,)"
        R"("fsmn_p":%.3f,"fsmn_sp":%s,)",
        (unsigned long)t0_ns,
        (unsigned long)st.audio_t1_processed,
        (unsigned long)st.audio_t1_in,
        stream_sec,
        st.is_speech ? "true" : "false", st.last_energy, st.last_rms,
        st.silero_prob, st.silero_speech ? "true" : "false",
        st.fsmn_prob, st.fsmn_speech ? "true" : "false");

    // WL-ECAPA (only when active).
    if (st.wlecapa_active) {
        n += snprintf(buf + n, sizeof(buf) - n,
            R"("wle_active":true,"wle_id":%d,"wle_name":"%s","wle_sim":%.3f,)"
            R"("wle_early":%s,"wle_margin":%.2f,)",
            st.wlecapa_id, st.wlecapa_name, st.wlecapa_sim,
            st.wlecapa_is_early ? "true" : "false",
            wlecapa_margin);
        if (change_valid) {
            n += snprintf(buf + n, sizeof(buf) - n,
                R"("change_sim":%.4f,)", change_sim);
        }
    }

    // Tracker.
    if (ts.check_active) {
        n += snprintf(buf + n, sizeof(buf) - n,
            R"("trk_check":true,"trk_state":%d,"trk_id":%d,"trk_name":"%s",)"
            R"("trk_sim":%.3f,"trk_avg":%.3f,"trk_conf":%d,)"
            R"("trk_switches":%d,"trk_f0":%.1f,"trk_jitter":%.3f,)",
            static_cast<int>(ts.state), ts.speaker_id, ts.speaker_name,
            ts.sim_to_ref, ts.sim_running_avg,
            static_cast<int>(ts.confidence),
            ts.switches, ts.f0_hz, ts.f0_jitter);
    }

    // Tracker registration event.
    if (ts.reg_event) {
        n += snprintf(buf + n, sizeof(buf) - n,
            R"("trk_reg":true,"trk_reg_id":%d,"trk_reg_name":"%s",)",
            ts.reg_id, ts.reg_name);
    }

    // ASR buffer state.
    n += snprintf(buf + n, sizeof(buf) - n,
        R"("asr_buf":%.2f,"asr_buf_sp":%s,"asr_busy":%s,)"
        R"("asr_sil_ms":%d,"asr_eff_sil":%d)",
        st.asr_buf_sec,
        st.asr_buf_has_speech ? "true" : "false",
        st.asr_busy ? "true" : "false",
        st.asr_post_silence_ms, st.asr_effective_silence_ms);

    // Close JSON.
    buf[n++] = '}';
    buf[n++] = '\n';
    buf[n] = '\0';

    fputs(buf, fp_);
    stats_count_++;
}

void TimelineLogger::log_asr(const char* text, float stream_start, float stream_end,
                               float latency_ms, float audio_sec,
                               const char* trigger,
                               int spk_id, const char* spk_name, float spk_sim,
                               float spk_conf, const char* spk_src,
                               int trk_id, const char* trk_name, float trk_sim) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!fp_) return;

    // Escape text for JSON (minimal: quotes and backslashes).
    std::string esc;
    esc.reserve(strlen(text) + 32);
    for (const char* p = text; *p; ++p) {
        if (*p == '"')       esc += "\\\"";
        else if (*p == '\\') esc += "\\\\";
        else if (*p == '\n') esc += "\\n";
        else if ((unsigned char)*p < 0x20) continue;
        else                 esc += *p;
    }

    // T0 stamp at serialization moment. The 's'/'e' business-clock range
    // remains the authoritative mapping back to source audio samples.
    uint64_t t0_ns = tempus::now_t0_ns();

    char buf[2048];
    int n = snprintf(buf, sizeof(buf),
        R"({"t":"asr","t0":%lu,"s":%.2f,"e":%.2f,"text":"%s",)"
        R"("trigger":"%s","latency":%.1f,"audio":%.2f,)"
        R"("spk_id":%d,"spk_name":"%s","spk_sim":%.3f,"spk_conf":%.3f,"spk_src":"%s",)"
        R"("trk_id":%d,"trk_name":"%s","trk_sim":%.3f})"
        "\n",
        (unsigned long)t0_ns,
        stream_start, stream_end, esc.c_str(),
        trigger ? trigger : "", latency_ms, audio_sec,
        spk_id, spk_name ? spk_name : "", spk_sim, spk_conf,
        spk_src ? spk_src : "",
        trk_id, trk_name ? trk_name : "", trk_sim);
    (void)n;

    fputs(buf, fp_);
    asr_count_++;
}

void TimelineLogger::log_vad(bool is_speech, bool segment_start, bool segment_end,
                               int frame_idx, float energy) {
    // Only log segment boundaries, not every frame.
    if (!segment_start && !segment_end) return;

    std::lock_guard<std::mutex> lk(mu_);
    if (!fp_) return;

    // VAD T1 is the VAD frame index (T2-equivalent for energy VAD); T0 is
    // stamped at emission. For sample-accurate VAD edges use log_stats'
    // sample field at the surrounding tick.
    uint64_t t0_ns = tempus::now_t0_ns();

    char buf[256];
    snprintf(buf, sizeof(buf),
        R"({"t":"vad","t0":%lu,"event":"%s","speech":%s,"frame":%d,"energy":%.2f})"
        "\n",
        (unsigned long)t0_ns,
        segment_start ? "start" : "end",
        is_speech ? "true" : "false",
        frame_idx, energy);

    fputs(buf, fp_);
    vad_count_++;
}

}  // namespace deusridet
