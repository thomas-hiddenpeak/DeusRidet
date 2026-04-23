/**
 * @file auditus_facade.cpp
 * @philosophical_role Wires Auditus outputs onto Nexus WS broadcasts. Thin by design — any logic
 *         that belongs to hearing stays in AudioPipeline; any logic that belongs to transport stays
 *         in WsServer; the facade only binds the two across their declared seams. This TU owns the
 *         low-volume 7a installers (vad/asr_partial/drop) and the 7c binary intake installer; the
 *         heavier 7b broadcast installers (transcript/asr_log/stats/speaker_match) live in the peer
 *         TU auditus_facade_broadcasts.cpp to keep both files under the R1 500-line hard cap.
 * @serves auditus_facade.h consumers (currently awaken).
 */

#include "auditus_facade.h"

#include "sensus/auditus/audio_pipeline.h"
#include "nexus/ws_server.h"
#include "communis/timeline_logger.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>

namespace deusridet {
namespace auditus {

void install_vad_callback(AudioPipeline& audio,
                          WsServer& server,
                          TimelineLogger& timeline) {
    audio.set_on_vad([&server, &timeline](bool is_speech, bool segment_start,
                                          bool segment_end, float prob,
                                          int frame_idx, uint64_t audio_t1) {
        char json[256];
        snprintf(json, sizeof(json),
            R"({"type":"vad","audio_t1":%lu,"speech":%s,"event":"%s","frame":%d,"prob":%.3f})",
            (unsigned long)audio_t1,
            is_speech ? "true" : "false",
            segment_start ? "start" : (segment_end ? "end" : "none"),
            frame_idx, prob);
        server.broadcast_text(json);
        timeline.log_vad(is_speech, segment_start, segment_end,
                         frame_idx, prob, audio_t1);
        if (segment_start)
            printf("[awaken] VAD: speech START at frame %d (prob=%.3f)\n",
                   frame_idx, prob);
        if (segment_end)
            printf("[awaken] VAD: speech END at frame %d\n", frame_idx);
    });
}

void install_asr_partial_callback(AudioPipeline& audio,
                                  WsServer& server) {
    audio.set_on_asr_partial([&server](const std::string& text, float audio_sec) {
        std::string escaped = json_escape(text);
        char json[2048];
        snprintf(json, sizeof(json),
            R"({"type":"asr_partial","text":"%s","audio_sec":%.2f})",
            escaped.c_str(), audio_sec);
        server.broadcast_text(json);
    });
}

void install_drop_callback(AudioPipeline& audio,
                           WsServer& server,
                           TimelineLogger& timeline) {
    audio.set_on_drop([&server, &timeline](uint64_t t1_from, uint64_t t1_to,
                                           const char* reason, size_t bytes) {
        timeline.log_drop(t1_from, t1_to, reason, bytes);
        uint64_t n = (t1_to > t1_from) ? (t1_to - t1_from) : 0;
        char json[256];
        snprintf(json, sizeof(json),
            R"({"type":"audio_drop","t1_from":%lu,"t1_to":%lu,"samples":%lu,)"
            R"("sec":%.4f,"bytes":%lu,"reason":"%s"})",
            (unsigned long)t1_from, (unsigned long)t1_to,
            (unsigned long)n, n / 16000.0,
            (unsigned long)bytes, reason ? reason : "unknown");
        server.broadcast_text(json);
    });
}

// ── Step 7c installers ──────────────────────────────────────────────────────

void install_ws_binary_callback(WsServer& server,
                                AudioPipeline& audio,
                                std::atomic<uint64_t>& total_frames,
                                std::atomic<uint64_t>& total_bytes,
                                std::atomic<bool>& loopback) {
    server.set_on_binary([&server, &audio, &total_frames, &total_bytes, &loopback]
                         (int fd, const uint8_t* data, size_t len) {
        uint64_t f = total_frames.fetch_add(1, std::memory_order_relaxed) + 1;
        total_bytes.fetch_add(len, std::memory_order_relaxed);

        // Push PCM to audio pipeline.
        const int16_t* samples = reinterpret_cast<const int16_t*>(data);
        int n_samples = len / sizeof(int16_t);
        audio.push_pcm(samples, n_samples);

        // Quick RMS/peak for WS-level feedback (every 10 frames).
        if (f % 10 == 0) {
            double sum_sq = 0;
            int16_t peak_abs = 0;
            for (int i = 0; i < n_samples; i++) {
                int16_t s = samples[i];
                sum_sq += (double)s * s;
                int16_t a = s < 0 ? (int16_t)(-s) : s;
                if (a > peak_abs) peak_abs = a;
            }
            float rms  = n_samples > 0 ? sqrtf((float)(sum_sq / n_samples)) / 32768.0f : 0;
            float peak = peak_abs / 32768.0f;
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"audio_stats","frames":%lu,"rms":%.4f,"peak":%.4f})",
                (unsigned long)f, rms, peak);
            server.send_text(fd, json);
        }

        // Loopback.
        if (loopback.load(std::memory_order_relaxed)) {
            server.send_binary(fd, data, len);
        }

        if (f % 500 == 0) {
            auto& st = audio.stats();
            printf("[awaken] PCM: %lu frames | Mel: %lu | Speech: %lu | Energy: %.2f\n",
                   (unsigned long)f, (unsigned long)st.mel_frames,
                   (unsigned long)st.speech_frames, st.last_energy);
        }
    });
}

}  // namespace auditus
}  // namespace deusridet
