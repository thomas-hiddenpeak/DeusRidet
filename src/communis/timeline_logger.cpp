// timeline_logger.cpp — JSONL timeline event serialization
//
// Compact, analysis-focused JSON — omits speaker roster, config params,
// and other data that doesn't change between ticks.

#include "communis/timeline_logger.h"
#include "sensus/auditus/audio_pipeline.h"

namespace deusridet {

void TimelineLogger::log_stats(const AudioPipelineStats& st,
                                float wlecapa_margin,
                                float change_sim, bool change_valid) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!fp_) return;

    double stream_sec = st.pcm_samples_in / 16000.0;

    // Build compact JSON — only fields relevant for timeline analysis.
    char buf[1024];
    int n = snprintf(buf, sizeof(buf),
        R"({"t":"stats","s":%.4f,)"
        R"("speech":%s,"energy":%.2f,"rms":%.4f,)"
        R"("silero_p":%.3f,"silero_sp":%s,)"
        R"("fsmn_p":%.3f,"fsmn_sp":%s,)",
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
                               float spk_conf, const char* spk_src) {
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

    char buf[2048];
    int n = snprintf(buf, sizeof(buf),
        R"({"t":"asr","s":%.2f,"e":%.2f,"text":"%s",)"
        R"("trigger":"%s","latency":%.1f,"audio":%.2f,)"
        R"("spk_id":%d,"spk_name":"%s","spk_sim":%.3f,"spk_conf":%.3f,"spk_src":"%s"})"
        "\n",
        stream_start, stream_end, esc.c_str(),
        trigger ? trigger : "", latency_ms, audio_sec,
        spk_id, spk_name ? spk_name : "", spk_sim, spk_conf,
        spk_src ? spk_src : "");
    (void)n;

    fputs(buf, fp_);
    asr_count_++;
}

void TimelineLogger::log_spk(float stream_start, float stream_end,
                               const char* source,
                               int spk_id, const char* spk_name, float spk_sim,
                               bool is_new) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!fp_) return;

    char buf[512];
    snprintf(buf, sizeof(buf),
        R"({"t":"spk","s":%.4f,"e":%.4f,"src":"%s","id":%d,"name":"%s","sim":%.3f,"new":%s})"
        "\n",
        stream_start, stream_end,
        source ? source : "", spk_id,
        spk_name ? spk_name : "", spk_sim,
        is_new ? "true" : "false");

    fputs(buf, fp_);
    spk_count_++;
}

void TimelineLogger::log_vad(float stream_sec, bool segment_start, bool segment_end,
                               float energy) {
    // Only log segment boundaries, not every frame.
    if (!segment_start && !segment_end) return;

    std::lock_guard<std::mutex> lk(mu_);
    if (!fp_) return;

    char buf[256];
    snprintf(buf, sizeof(buf),
        R"({"t":"vad","s":%.4f,"event":"%s","energy":%.2f})"
        "\n",
        stream_sec,
        segment_start ? "start" : "end",
        energy);

    fputs(buf, fp_);
    vad_count_++;
}

}  // namespace deusridet
