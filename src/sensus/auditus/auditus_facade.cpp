/**
 * @file auditus_facade.cpp
 * @philosophical_role Wires Auditus outputs onto Nexus WS broadcasts. Thin by design — any logic
 *         that belongs to hearing stays in AudioPipeline; any logic that belongs to transport stays
 *         in WsServer; the facade only binds the two across their declared seams.
 * @serves auditus_facade.h consumers (currently cmd_test_ws).
 */

#include "auditus_facade.h"

#include "sensus/auditus/audio_pipeline.h"
#include "nexus/ws_server.h"
#include "communis/timeline_logger.h"

#include <cstdio>

namespace deusridet {
namespace auditus {

void install_vad_callback(AudioPipeline& audio,
                          WsServer& server,
                          TimelineLogger& timeline) {
    audio.set_on_vad([&server, &timeline](const VadResult& vr, int frame_idx, uint64_t audio_t1) {
        char json[256];
        snprintf(json, sizeof(json),
            R"({"type":"vad","audio_t1":%lu,"speech":%s,"event":"%s","frame":%d,"energy":%.2f})",
            (unsigned long)audio_t1,
            vr.is_speech ? "true" : "false",
            vr.segment_start ? "start" : (vr.segment_end ? "end" : "none"),
            frame_idx, vr.energy);
        server.broadcast_text(json);
        timeline.log_vad(vr.is_speech, vr.segment_start, vr.segment_end,
                         frame_idx, vr.energy, audio_t1);
        if (vr.segment_start)
            printf("[test-ws] VAD: speech START at frame %d (energy=%.2f)\n",
                   frame_idx, vr.energy);
        if (vr.segment_end)
            printf("[test-ws] VAD: speech END at frame %d\n", frame_idx);
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

}  // namespace auditus
}  // namespace deusridet
