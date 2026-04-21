/**
 * @file cmd_test_ws_router.cpp
 * @philosophical_role The bus that carries every WebUI knob-turn to the
 *         subsystem that owns that knob. Matching on command prefix, delegates
 *         to AudioPipeline / SpeakerTracker / ConscientiStream and replies
 *         with a JSON envelope that names the same key the WebUI sent.
 *         Extracted from cmd_test_ws.cpp (Step 7d) — behaviour identical.
 * @serves cmd_test_ws_router.h.
 */
#include "cmd_test_ws_router.h"

#include "sensus/auditus/audio_pipeline.h"
#include "nexus/ws_server.h"
#include "conscientia/stream.h"
#include "conscientia/frame.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace deusridet {
namespace actus {

void handle_ws_text_command(int fd,
                            const std::string& msg,
                            AudioPipeline& audio,
                            WsServer& server,
                            ConscientiStream& consciousness,
                            std::atomic<bool>& loopback,
                            bool llm_loaded) {
    if (msg == "loopback:on") {
        loopback.store(true, std::memory_order_relaxed);
        server.send_text(fd, R"({"type":"loopback","enabled":true})");
        printf("[test-ws] Loopback ON (fd=%d)\n", fd);
    } else if (msg == "loopback:off") {
        loopback.store(false, std::memory_order_relaxed);
        server.send_text(fd, R"({"type":"loopback","enabled":false})");
        printf("[test-ws] Loopback OFF (fd=%d)\n", fd);
    } else if (msg.rfind("vad_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 14, nullptr);
        audio.set_vad_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"vad_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] VAD threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg.rfind("gain:", 0) == 0) {
        float g = std::strtof(msg.c_str() + 5, nullptr);
        if (g < 0.1f) g = 0.1f;
        if (g > 20.0f) g = 20.0f;
        audio.set_gain(g);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"gain","value":%.1f})", g);
        server.send_text(fd, json);
        printf("[test-ws] Gain = %.1fx (fd=%d)\n", g, fd);
    } else if (msg.rfind("silero_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 17, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_silero_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"silero_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] Silero threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg.rfind("fsmn_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 15, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_fsmn_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"fsmn_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] FSMN threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg == "silero_enable:on" || msg == "silero_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_silero_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"silero_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] Silero %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg == "frcrn_enable:on" || msg == "frcrn_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_frcrn_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"frcrn_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] FRCRN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg == "fsmn_enable:on" || msg == "fsmn_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_fsmn_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"fsmn_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] FSMN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("vad_source:", 0) == 0) {
        auto val = msg.substr(11);
        VadSource src = VadSource::ANY;
        if (val == "silero") src = VadSource::SILERO;
        else if (val == "fsmn") src = VadSource::FSMN;
        else src = VadSource::ANY;
        audio.set_vad_source(src);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"vad_source","value":%d})", static_cast<int>(src));
        server.send_text(fd, json);
        printf("[test-ws] VAD source = %s (%d) (fd=%d)\n", val.c_str(), static_cast<int>(src), fd);
    } else if (msg == "speaker_enable:on" || msg == "speaker_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_speaker_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"speaker_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] Speaker %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("speaker_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 18, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_speaker_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"speaker_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] Speaker threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg == "speaker_clear") {
        audio.clear_speaker_db();
        server.send_text(fd, R"({"type":"speaker_clear"})");
        printf("[test-ws] Speaker (CAM++) DB cleared (fd=%d)\n", fd);
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
            printf("[test-ws] CAM++ Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
        }
    } else if (msg == "wlecapa_enable:on" || msg == "wlecapa_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_wlecapa_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] WL-ECAPA %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("wlecapa_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 18, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.set_wlecapa_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] WL-ECAPA threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg.rfind("wlecapa_margin:", 0) == 0) {
        float m = std::strtof(msg.c_str() + 15, nullptr);
        if (m < 0.0f) m = 0.0f;
        if (m > 0.5f) m = 0.5f;
        audio.wlecapa_db().set_min_margin(m);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_margin","value":%.2f})", m);
        server.send_text(fd, json);
        printf("[test-ws] WL-ECAPA margin = %.2f (fd=%d)\n", m, fd);
    } else if (msg.rfind("early_trigger:", 0) == 0) {
        float s = std::strtof(msg.c_str() + 14, nullptr);
        if (s < 0.5f) s = 0.5f;
        if (s > 5.0f) s = 5.0f;
        audio.set_early_trigger_sec(s);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"early_trigger","value":%.2f})", s);
        server.send_text(fd, json);
        printf("[test-ws] Early trigger = %.2fs (fd=%d)\n", s, fd);
    } else if (msg == "early_enable:on" || msg == "early_enable:off") {
        bool en = (msg == "early_enable:on");
        audio.set_early_trigger_enabled(en);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"early_enable","value":%s})", en ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] Early trigger %s (fd=%d)\n", en ? "enabled" : "disabled", fd);
    } else if (msg.rfind("min_speech:", 0) == 0) {
        float s = std::strtof(msg.c_str() + 11, nullptr);
        if (s < 0.5f) s = 0.5f;
        if (s > 10.0f) s = 10.0f;
        audio.set_min_speech_sec(s);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"min_speech","value":%.2f})", s);
        server.send_text(fd, json);
        printf("[test-ws] Min speech = %.2fs (fd=%d)\n", s, fd);
    } else if (msg == "wlecapa_clear") {
        audio.clear_wlecapa_db();
        server.send_text(fd, R"({"type":"wlecapa_clear"})");
        printf("[test-ws] WL-ECAPA DB cleared (fd=%d)\n", fd);
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
            printf("[test-ws] WL-ECAPA Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
        }
    } else if (msg.rfind("wlecapa_delete:", 0) == 0) {
        // Format: wlecapa_delete:ID
        int id = std::stoi(msg.substr(15));
        bool ok = audio.remove_wlecapa_speaker(id);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"wlecapa_delete","id":%d,"ok":%s})", id, ok ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] WL-ECAPA delete #%d %s (fd=%d)\n", id, ok ? "OK" : "FAIL", fd);
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
            printf("[test-ws] WL-ECAPA merge #%d←#%d %s (fd=%d)\n", dst, src, ok ? "OK" : "FAIL", fd);
        }
    // ── SpeakerTracker controls ──
    } else if (msg == "tracker_enable:on" || msg == "tracker_enable:off") {
        bool on = msg.back() == 'n';
        audio.tracker().set_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"tracker_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] Tracker %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("tracker_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 18, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.tracker().set_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"tracker_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] Tracker threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg.rfind("tracker_change_threshold:", 0) == 0) {
        float t = std::strtof(msg.c_str() + 24, nullptr);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        audio.tracker().set_change_threshold(t);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"tracker_change_threshold","value":%.2f})", t);
        server.send_text(fd, json);
        printf("[test-ws] Tracker change threshold = %.2f (fd=%d)\n", t, fd);
    } else if (msg.rfind("tracker_interval:", 0) == 0) {
        int ms = std::stoi(msg.substr(17));
        if (ms < 250) ms = 250;
        if (ms > 5000) ms = 5000;
        audio.tracker().set_interval_ms(ms);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"tracker_interval","value":%d})", ms);
        server.send_text(fd, json);
        printf("[test-ws] Tracker interval = %d ms (fd=%d)\n", ms, fd);
    } else if (msg.rfind("tracker_window:", 0) == 0) {
        int ms = std::stoi(msg.substr(15));
        if (ms < 500) ms = 500;
        if (ms > 5000) ms = 5000;
        audio.tracker().set_window_ms(ms);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"tracker_window","value":%d})", ms);
        server.send_text(fd, json);
        printf("[test-ws] Tracker window = %d ms (fd=%d)\n", ms, fd);
    } else if (msg == "tracker_clear") {
        audio.tracker().clear();
        server.send_text(fd, R"({"type":"tracker_clear"})");
        printf("[test-ws] Tracker DB cleared (fd=%d)\n", fd);
    } else if (msg.rfind("tracker_name:", 0) == 0) {
        auto rest = msg.substr(13);
        auto colon = rest.find(':');
        if (colon != std::string::npos) {
            int id = std::stoi(rest.substr(0, colon));
            std::string name = rest.substr(colon + 1);
            audio.tracker().set_speaker_name(id, name);
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"tracker_name","id":%d,"name":"%s"})", id, name.c_str());
            server.send_text(fd, json);
            printf("[test-ws] Tracker Speaker %d named '%s' (fd=%d)\n", id, name.c_str(), fd);
        }
    } else if (msg == "asr_enable:on" || msg == "asr_enable:off") {
        bool on = msg.back() == 'n';
        audio.set_asr_enabled(on);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"asr_enable","enabled":%s})", on ? "true" : "false");
        server.send_text(fd, json);
        printf("[test-ws] ASR %s (fd=%d)\n", on ? "ON" : "OFF", fd);
    } else if (msg.rfind("asr_vad_source:", 0) == 0) {
        auto val = msg.substr(15);
        VadSource src = VadSource::ANY;
        if (val == "silero") src = VadSource::SILERO;
        else if (val == "fsmn") src = VadSource::FSMN;
        else if (val == "direct") src = VadSource::DIRECT;
        else src = VadSource::ANY;
        audio.set_asr_vad_source(src);
        char json[128];
        snprintf(json, sizeof(json),
            R"({"type":"asr_vad_source","value":%d})", static_cast<int>(src));
        server.send_text(fd, json);
        printf("[test-ws] ASR VAD source = %s (%d) (fd=%d)\n", val.c_str(), static_cast<int>(src), fd);
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
            printf("[test-ws] ASR param %s=%s (fd=%d)\n",
                   key.c_str(), val.c_str(), fd);
        }
    } else if (msg.rfind("consciousness_enable:", 0) == 0 && llm_loaded) {
        // Format: consciousness_enable:<mode>:<on|off>
        auto rest = msg.substr(21);
        auto sep = rest.find(':');
        if (sep != std::string::npos) {
            auto mode = rest.substr(0, sep);
            bool on = rest.substr(sep + 1) == "on";
            if (mode == "response") consciousness.set_response_enabled(on);
            else if (mode == "daydream") consciousness.set_daydream_enabled(on);
            else if (mode == "dreaming") consciousness.set_dreaming_enabled(on);
            else if (mode == "llm") consciousness.set_llm_enabled(on);
            else if (mode == "speech") consciousness.set_speech_enabled(on);
            else if (mode == "thinking") consciousness.set_thinking_enabled(on);
            else if (mode == "action") consciousness.set_action_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"consciousness_enable","mode":"%s","enabled":%s})",
                mode.c_str(), on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] Consciousness %s %s (fd=%d)\n",
                   mode.c_str(), on ? "ON" : "OFF", fd);
        }
    } else if (msg.rfind("consciousness_param:", 0) == 0 && llm_loaded) {
        // Format: consciousness_param:<key>:<value>
        // Key can be global (e.g. "temperature") or per-pipeline
        // (e.g. "speech.temperature", "thinking.top_k")
        auto rest = msg.substr(20);
        auto sep = rest.find(':');
        if (sep != std::string::npos) {
            auto key = rest.substr(0, sep);
            auto val = rest.substr(sep + 1);
            char json[256];

            // Determine pipeline target
            std::string pipeline;
            std::string param_name = key;
            auto dot = key.find('.');
            if (dot != std::string::npos) {
                pipeline = key.substr(0, dot);
                param_name = key.substr(dot + 1);
            }

            // Helper to get target pipeline config
            PipelineConfig* pcfg = nullptr;
            if (pipeline == "speech") pcfg = &consciousness.speech_cfg();
            else if (pipeline == "thinking") pcfg = &consciousness.thinking_cfg();
            else if (pipeline == "action") pcfg = &consciousness.action_cfg();

            if (param_name == "temperature") {
                float v = std::stof(val);
                if (v < 0.0f) v = 0.0f;
                if (v > 2.0f) v = 2.0f;
                if (pcfg) pcfg->sampling.temperature = v;
                else consciousness.set_temperature(v);
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_param","key":"%s","value":%.2f})",
                    key.c_str(), v);
            } else if (param_name == "top_k") {
                int v = std::stoi(val);
                if (v < 1) v = 1;
                if (v > 200) v = 200;
                if (pcfg) pcfg->sampling.top_k = v;
                else consciousness.set_top_k(v);
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_param","key":"%s","value":%d})",
                    key.c_str(), v);
            } else if (param_name == "top_p") {
                float v = std::stof(val);
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                if (pcfg) pcfg->sampling.top_p = v;
                else consciousness.set_top_p(v);
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_param","key":"%s","value":%.2f})",
                    key.c_str(), v);
            } else if (param_name == "max_tokens") {
                int v = std::stoi(val);
                if (v < 10) v = 10;
                if (v > 4096) v = 4096;
                if (pcfg) pcfg->max_tokens = v;
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_param","key":"%s","value":%d})",
                    key.c_str(), v);
            } else {
                snprintf(json, sizeof(json),
                    R"({"type":"consciousness_param","key":"%s","error":"unknown"})",
                    key.c_str());
            }
            server.send_text(fd, json);
            printf("[test-ws] Consciousness param %s=%s (fd=%d)\n",
                   key.c_str(), val.c_str(), fd);
        }
    } else if (msg.rfind("consciousness_prompt:", 0) == 0 && llm_loaded) {
        // Format: consciousness_prompt:<pipeline>:<text>
        // pipeline: "identity" (system prompt), "speech", "thinking", "action"
        auto rest = msg.substr(21);
        auto sep = rest.find(':');
        if (sep != std::string::npos) {
            auto pipeline = rest.substr(0, sep);
            auto text = rest.substr(sep + 1);
            if (pipeline == "identity") {
                consciousness.set_identity_prompt(text);
            } else if (pipeline == "speech") {
                consciousness.speech_cfg().prompt = text;
                consciousness.recompose_system_prompt();
            } else if (pipeline == "thinking") {
                consciousness.thinking_cfg().prompt = text;
                consciousness.recompose_system_prompt();
            } else if (pipeline == "action") {
                consciousness.action_cfg().prompt = text;
                consciousness.recompose_system_prompt();
            }
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"consciousness_prompt","pipeline":"%s","ok":true})",
                pipeline.c_str());
            server.send_text(fd, json);
            printf("[test-ws] %s prompt updated (%zu chars, fd=%d)\n",
                   pipeline.c_str(), text.size(), fd);
        } else {
            // Legacy: no pipeline prefix → identity prompt
            consciousness.set_identity_prompt(rest);
            server.send_text(fd, R"({"type":"consciousness_prompt","pipeline":"identity","ok":true})");
            printf("[test-ws] System prompt updated (%zu chars, fd=%d)\n",
                   rest.size(), fd);
        }
    } else if (msg.rfind("text_input:", 0) == 0 && llm_loaded) {
        // Inject text into consciousness stream
        std::string text = msg.substr(11);
        if (!text.empty()) {
            InputItem item;
            item.source = InputSource::TEXT;
            item.text = text;
            item.priority = 1.0f;
            consciousness.inject_input(std::move(item));
            // Echo back confirmation
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"text_input_ack","ok":true})");
            server.send_text(fd, json);
            printf("[test-ws] Text input injected: \"%s\" (fd=%d)\n",
                   text.c_str(), fd);
        }
    } else {
        printf("[test-ws] Text from fd=%d: %s\n", fd, msg.c_str());
    }
}

}  // namespace actus
}  // namespace deusridet
