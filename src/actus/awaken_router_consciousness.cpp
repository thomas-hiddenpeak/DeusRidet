/**
 * @file awaken_router_consciousness.cpp
 * @philosophical_role Peer TU of awaken_router.cpp owning the four
 *         consciousness_* text prefixes (enable, param, prompt) plus the
 *         text_input injector. Split out under R1 because these four
 *         branches together were ~150 lines of identity/sampling/pipeline
 *         plumbing; keeping them inside the main router pushed
 *         awaken_router.cpp past the 500-line hard cap. Behaviour-
 *         identical to the pre-split code — same prefix matching, same
 *         JSON envelope keys, same stdout lines.
 * @serves awaken_router.cpp (via handle_ws_consciousness_command).
 */
#include "awaken_router.h"

#include "nexus/ws_server.h"
#include "conscientia/stream.h"
#include "conscientia/frame.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace deusridet {
namespace actus {

bool handle_ws_consciousness_command(int fd,
                                     const std::string& msg,
                                     WsServer& server,
                                     ConscientiStream& consciousness) {
    if (msg.rfind("consciousness_enable:", 0) == 0) {
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
            printf("[awaken] Consciousness %s %s (fd=%d)\n",
                   mode.c_str(), on ? "ON" : "OFF", fd);
        }
        return true;
    }
    if (msg.rfind("consciousness_param:", 0) == 0) {
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
            printf("[awaken] Consciousness param %s=%s (fd=%d)\n",
                   key.c_str(), val.c_str(), fd);
        }
        return true;
    }
    if (msg.rfind("consciousness_prompt:", 0) == 0) {
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
            printf("[awaken] %s prompt updated (%zu chars, fd=%d)\n",
                   pipeline.c_str(), text.size(), fd);
        } else {
            // Legacy: no pipeline prefix → identity prompt
            consciousness.set_identity_prompt(rest);
            server.send_text(fd, R"({"type":"consciousness_prompt","pipeline":"identity","ok":true})");
            printf("[awaken] System prompt updated (%zu chars, fd=%d)\n",
                   rest.size(), fd);
        }
        return true;
    }
    if (msg.rfind("text_input:", 0) == 0) {
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
            printf("[awaken] Text input injected: \"%s\" (fd=%d)\n",
                   text.c_str(), fd);
        }
        return true;
    }
    return false;
}

}  // namespace actus
}  // namespace deusridet
