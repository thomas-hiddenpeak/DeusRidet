/**
 * @file awaken_router.cpp
 * @philosophical_role The bus that carries every WebUI knob-turn to the
 *         subsystem that owns that knob. Matching on command prefix, delegates
 *         to AudioPipeline / ConscientiStream and replies
 *         with a JSON envelope that names the same key the WebUI sent.
 *         Extracted from awaken.cpp (Step 7d) — behaviour identical.
 * @serves awaken_router.h.
 */
#include "awaken_router.h"

#include "sensus/auditus/audio_pipeline.h"
#include "nexus/ws_server.h"
#include "conscientia/stream.h"
#include "conscientia/frame.h"
#include "orator/diarizen_periodic_worker.h"
#include "orator/diarizen_pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace deusridet {
namespace actus {

void handle_ws_text_command(int fd,
                            const std::string& msg,
                            AudioPipeline& audio,
                            WsServer& server,
                            ConscientiStream& consciousness,
                            std::atomic<bool>& loopback,
                            bool llm_loaded,
                            orator::DiarizenPeriodicWorker* worker,
                            orator::DiarizenPipeline* native) {
    if (msg == "loopback:on") {
        loopback.store(true, std::memory_order_relaxed);
        server.send_text(fd, R"({"type":"loopback","enabled":true})");
        printf("[awaken] Loopback ON (fd=%d)\n", fd);
    } else if (msg == "loopback:off") {
        loopback.store(false, std::memory_order_relaxed);
        server.send_text(fd, R"({"type":"loopback","enabled":false})");
        printf("[awaken] Loopback OFF (fd=%d)\n", fd);
    } else if (msg.rfind("gain:", 0) == 0) {
        float g = std::strtof(msg.c_str() + 5, nullptr);
        if (g < 0.1f) g = 0.1f;
        if (g > 20.0f) g = 20.0f;
        audio.set_gain(g);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"gain","value":%.1f})", g);
        server.send_text(fd, json);
        printf("[awaken] Gain = %.1fx (fd=%d)\n", g, fd);
    } else if (msg.rfind("silero_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 17, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_silero_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"silero_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[awaken] Silero threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg == "silero_enable:on" || msg == "silero_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_silero_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"silero_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] Silero %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg == "frcrn_enable:on" || msg == "frcrn_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_frcrn_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"frcrn_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] FRCRN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("vad_source:", 0) == 0) {
        auto val = msg.substr(11);
        VadSource src = VadSource::ANY;
        if (val == "silero") src = VadSource::SILERO;
        else src = VadSource::ANY;
        audio.set_vad_source(src);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"vad_source","value":%d})", static_cast<int>(src));
        server.send_text(fd, json);
        printf("[awaken] VAD source = %s (%d) (fd=%d)\n", val.c_str(), static_cast<int>(src), fd);
    } else if (msg == "speaker_enable:on" || msg == "speaker_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_speaker_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"speaker_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] Speaker %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("speaker_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 18, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_speaker_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"speaker_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[awaken] Speaker threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg == "speaker_clear") {
        audio.clear_speaker_db();
        server.send_text(fd, R"({"type":"speaker_clear"})");
        printf("[awaken] Speaker (CAM++) DB cleared (fd=%d)\n", fd);
    } else if (msg.rfind("speaker_name:", 0) == 0) {
        // Format: speaker_name:ID:Name
        auto rest = msg.substr(13);
        auto colon = rest.find(':');
        if (colon != std::string::npos) {
            int id = std::stoi(rest.substr(0, colon));
            std::string name = rest.substr(colon + 1);
            audio.set_speaker_name(id, name);
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"speaker_name","id":%d,"name":"%s"})", id, name.c_str());
            server.send_text(fd, json);
            printf("[awaken] CAM++ Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
        }
    } else if (msg == "wlecapa_enable:on" || msg == "wlecapa_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_wlecapa_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] WL-ECAPA %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("wlecapa_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 18, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_wlecapa_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[awaken] WL-ECAPA threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg.rfind("wlecapa_margin:", 0) == 0) {
        float m = std::strtof(msg.c_str() + 15, nullptr);
        if (m < 0.0f) m = 0.0f;
        if (m > 0.5f) m = 0.5f;
        audio.wlecapa_db().set_min_margin(m);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_margin","value":%.2f})", m);
        server.send_text(fd, json);
        printf("[awaken] WL-ECAPA margin = %.2f (fd=%d)\n", m, fd);
    } else if (msg.rfind("early_trigger:", 0) == 0) {
        float s = std::strtof(msg.c_str() + 14, nullptr);
        if (s < 0.5f) s = 0.5f;
        if (s > 5.0f) s = 5.0f;
        audio.set_early_trigger_sec(s);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"early_trigger","value":%.2f})", s);
        server.send_text(fd, json);
        printf("[awaken] Early trigger = %.2fs (fd=%d)\n", s, fd);
    } else if (msg == "early_enable:on" || msg == "early_enable:off") {
        bool en = (msg == "early_enable:on");
        audio.set_early_trigger_enabled(en);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"early_enable","value":%s})", en ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] Early trigger %s (fd=%d)\n", en ? "enabled" : "disabled", fd);
    } else if (msg.rfind("min_speech:", 0) == 0) {
        float s = std::strtof(msg.c_str() + 11, nullptr);
        if (s < 0.5f) s = 0.5f;
        if (s > 10.0f) s = 10.0f;
        audio.set_min_speech_sec(s);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"min_speech","value":%.2f})", s);
        server.send_text(fd, json);
        printf("[awaken] Min speech = %.2fs (fd=%d)\n", s, fd);
    } else if (msg == "wlecapa_clear") {
        audio.clear_wlecapa_db();
        server.send_text(fd, R"({"type":"wlecapa_clear"})");
        printf("[awaken] WL-ECAPA DB cleared (fd=%d)\n", fd);
    } else if (msg.rfind("wlecapa_name:", 0) == 0) {
        auto rest = msg.substr(13);
        auto colon = rest.find(':');
        if (colon != std::string::npos) {
            int id = std::stoi(rest.substr(0, colon));
            std::string name = rest.substr(colon + 1);
            audio.set_wlecapa_name(id, name);
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"wlecapa_name","id":%d,"name":"%s"})", id, name.c_str());
            server.send_text(fd, json);
            printf("[awaken] WL-ECAPA Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
        }
    } else if (msg.rfind("wlecapa_delete:", 0) == 0) {
        // Format: wlecapa_delete:ID
        int id = std::stoi(msg.substr(15));
        bool ok = audio.remove_wlecapa_speaker(id);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_delete","id":%d,"ok":%s})", id, ok ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] WL-ECAPA delete #%d %s (fd=%d)\n", id, ok ? "OK" : "FAIL", fd);
    } else if (msg.rfind("wlecapa_merge:", 0) == 0) {
        // Format: wlecapa_merge:DST_ID:SRC_ID
        auto rest = msg.substr(14);
        auto colon = rest.find(':');
        if (colon != std::string::npos) {
            int dst = std::stoi(rest.substr(0, colon));
            int src = std::stoi(rest.substr(colon + 1));
            bool ok = audio.merge_wlecapa_speakers(dst, src);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"wlecapa_merge","dst":%d,"src":%d,"ok":%s})",
                dst, src, ok ? "true" : "false");
            server.send_text(fd, json);
            printf("[awaken] WL-ECAPA merge #%d←#%d %s (fd=%d)\n", dst, src, ok ? "OK" : "FAIL", fd);
        }
    } else if (msg == "asr_enable:on" || msg == "asr_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_asr_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"asr_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[awaken] ASR %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("asr_vad_source:", 0) == 0) {
        auto val = msg.substr(15);
        VadSource src = VadSource::ANY;
        if (val == "silero") src = VadSource::SILERO;
        else if (val == "direct") src = VadSource::DIRECT;
        else src = VadSource::ANY;
        audio.set_asr_vad_source(src);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"asr_vad_source","value":%d})", static_cast<int>(src));
        server.send_text(fd, json);
        printf("[awaken] ASR VAD source = %s (%d) (fd=%d)\n", val.c_str(), static_cast<int>(src), fd);
    } else if (msg.rfind("asr_param:", 0) == 0) {
        // Generic ASR parameter setter: "asr_param:<key>:<value>"
        auto rest = msg.substr(10);
        auto sep = rest.find(':');
        if (sep != std::string::npos) {
            auto key = rest.substr(0, sep);
            auto val = rest.substr(sep + 1);
            char json[256];
            if (key == "post_silence_ms") {
                int v = std::stoi(val);
                audio.set_asr_post_silence_ms(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"post_silence_ms","value":%d})",
                    audio.asr_post_silence_ms());
            } else if (key == "max_buf_sec") {
                float v = std::stof(val);
                audio.set_asr_max_buf_sec(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"max_buf_sec","value":%.1f})",
                    audio.asr_max_buf_sec());
            } else if (key == "min_dur_sec") {
                float v = std::stof(val);
                audio.set_asr_min_dur_sec(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"min_dur_sec","value":%.2f})",
                    audio.asr_min_dur_sec());
            } else if (key == "pre_roll_sec") {
                float v = std::stof(val);
                audio.set_asr_pre_roll_sec(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"pre_roll_sec","value":%.2f})",
                    audio.asr_pre_roll_sec());
            } else if (key == "max_tokens") {
                int v = std::stoi(val);
                audio.set_asr_max_tokens(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"max_tokens","value":%d})",
                    audio.asr_max_tokens());
            } else if (key == "rep_penalty") {
                float v = std::stof(val);
                audio.set_asr_rep_penalty(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"rep_penalty","value":%.2f})",
                    audio.asr_rep_penalty());
            } else if (key == "min_energy") {
                float v = std::stof(val);
                audio.set_asr_min_energy(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"min_energy","value":%.4f})",
                    audio.asr_min_energy());
            } else if (key == "partial_sec") {
                float v = std::stof(val);
                audio.set_asr_partial_sec(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"partial_sec","value":%.1f})",
                    audio.asr_partial_sec());
            } else if (key == "speech_ratio") {
                float v = std::stof(val);
                audio.set_asr_min_speech_ratio(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"speech_ratio","value":%.2f})",
                    audio.asr_min_speech_ratio());
            } else if (key == "adaptive_silence") {
                bool v = (val == "true" || val == "1" || val == "on");
                audio.set_asr_adaptive_silence(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"adaptive_silence","value":%s})",
                    audio.asr_adaptive_silence() ? "true" : "false");
            } else if (key == "adaptive_short_ms") {
                int v = std::stoi(val);
                audio.set_asr_adaptive_short_ms(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"adaptive_short_ms","value":%d})",
                    audio.asr_adaptive_short_ms());
            } else if (key == "adaptive_long_ms") {
                int v = std::stoi(val);
                audio.set_asr_adaptive_long_ms(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"adaptive_long_ms","value":%d})",
                    audio.asr_adaptive_long_ms());
            } else if (key == "adaptive_vlong_ms") {
                int v = std::stoi(val);
                audio.set_asr_adaptive_vlong_ms(v);
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"adaptive_vlong_ms","value":%d})",
                    audio.asr_adaptive_vlong_ms());
            } else {
                snprintf(json, sizeof(json),
                    R"({"type":"asr_param","key":"%s","error":"unknown"})",
                    key.c_str());
            }
            server.send_text(fd, json);
            printf("[awaken] ASR param %s=%s (fd=%d)\n",
                   key.c_str(), val.c_str(), fd);
        }
    } else if (llm_loaded &&
               handle_ws_consciousness_command(fd, msg, server, consciousness)) {
        // Handled by the consciousness peer router (awaken_router_consciousness.cpp).
    } else if (msg == "diarizen_trigger") {
        // Hybrid P2 — ask the periodic worker for an extra reclustering
        // pass right now. Returns immediately; result arrives later as a
        // `speaker_diarize_partial` broadcast.
        if (!worker) {
            server.send_text(fd, R"json({"type":"speaker_diarize_progress","ok":false,"error":"periodic worker disabled (LLM not loaded or DEUSRIDET_DIARIZEN_ENABLE!=1)"})json");
        } else {
            worker->trigger_async();
            server.send_text(fd, R"json({"type":"speaker_diarize_progress","status":"triggered"})json");
        }
    } else if (msg == "diarizen_finalize") {
        // Native path: when no holdback worker is wired (LLM not loaded),
        // finalise the whole session directly through the in-process CUDA
        // DiarizenPipeline — no wav dump, no Python worker.
        // Hybrid P2 path: if a periodic worker is wired, finalize through
        // it so the still-pending holdback is drained into Conscientia
        // with the freshest relabelling (the worker is itself native-backed).
        if (native && native->is_loaded() && !worker) {
            std::vector<float> pcm;
            size_t n = audio.diarizen_copy_pcm_f32(pcm);
            if (n == 0) {
                server.send_text(fd, R"({"type":"speaker_diarize_final","ok":false,"error":"capture buffer empty"})");
            } else {
                char ack[256];
                snprintf(ack, sizeof(ack),
                    R"({"type":"speaker_diarize_progress","status":"running","samples":%zu,"sec":%.2f})",
                    n, (double)n / 16000.0);
                server.send_text(fd, ack);
                auto* server_ptr = &server;
                auto* native_ptr = native;
                std::thread([server_ptr, native_ptr, pcm = std::move(pcm), n]() {
                    auto t0 = std::chrono::steady_clock::now();
                    auto segs = native_ptr->diarize(pcm.data(), pcm.size());
                    auto t1 = std::chrono::steady_clock::now();
                    double wall = std::chrono::duration<double>(t1 - t0).count();
                    if (segs.empty()) {
                        std::string err = native_ptr->last_error();
                        std::string j = std::string("{\"type\":\"speaker_diarize_final\",\"ok\":false,\"error\":\"")
                                      + (err.empty() ? "no segments" : err) + "\"}";
                        server_ptr->broadcast_text(j);
                        return;
                    }
                    std::string j = "{\"type\":\"speaker_diarize_final\",\"ok\":true,\"audio_sec\":";
                    char buf[96];
                    snprintf(buf, sizeof(buf), "%.3f,\"wall_sec\":%.3f,\"n_segments\":%zu,\"segments\":[",
                             (double)n / 16000.0, wall, segs.size());
                    j += buf;
                    for (size_t i = 0; i < segs.size(); ++i) {
                        if (i) j += ',';
                        snprintf(buf, sizeof(buf), "[%.3f,%.3f,\"", segs[i].start_sec, segs[i].end_sec);
                        j += buf;
                        j += segs[i].label;
                        j += "\"]";
                    }
                    j += "]}";
                    server_ptr->broadcast_text(j);
                }).detach();
            }
        } else if (worker) {
            char ack[160];
            size_t n = audio.diarizen_capture_samples();
            snprintf(ack, sizeof(ack),
                R"({"type":"speaker_diarize_progress","status":"finalizing","samples":%zu,"sec":%.2f})",
                n, (double)n / 16000.0);
            server.send_text(fd, ack);
            auto* worker_ptr = worker;
            std::thread([worker_ptr]() { worker_ptr->finalize(); }).detach();
        } else {
            server.send_text(fd, R"json({"type":"speaker_diarize_final","ok":false,"error":"diarizen disabled (set DEUSRIDET_DIARIZEN_ENABLE=1)"})json");
        }
    } else {
        printf("[awaken] Text from fd=%d: %s\n", fd, msg.c_str());
    }
}

}  // namespace actus
}  // namespace deusridet
