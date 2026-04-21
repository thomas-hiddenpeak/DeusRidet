/**
 * @file cmd_test_ws_hello.cpp
 * @philosophical_role Composes the consciousness_state + consciousness_prompts
 *         JSON envelope sent to every freshly-connected WS client. Logic
 *         byte-identical to the original inline lambda in cmd_test_ws.cpp.
 * @serves cmd_test_ws_hello.h.
 */
#include "cmd_test_ws_hello.h"

#include "nexus/ws_server.h"
#include "conscientia/stream.h"
#include "memoria/cache_manager.h"
#include "communis/config.h"
#include "communis/tegra.h"
#include "sensus/auditus/auditus_facade.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

namespace deusridet {
namespace actus {

using auditus::json_escape;

void send_consciousness_hello(int fd,
                              WsServer& server,
                              ConscientiStream& consciousness,
                              CacheManager& llm_cache,
                              const PersonaConfig& persona_cfg,
                              bool llm_loaded) {
    printf("[test-ws] WS client connected  (fd=%d)\n", fd);
    if (!llm_loaded) {
        server.send_text(fd, R"({"type":"consciousness_state","llm_loaded":false})");
        return;
    }

    const char* state_names[] = {"active", "daydream", "dreaming"};
    auto m = consciousness.metrics();

    size_t cuda_free = 0, cuda_total = 0;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    size_t mem_avail_kb = read_memavail_kb();
    size_t rss_kb = 0;
    {
        FILE* fp = fopen("/proc/self/status", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "VmRSS:", 6) == 0) {
                    rss_kb = strtoull(line + 6, nullptr, 10);
                    break;
                }
            }
            fclose(fp);
        }
    }

    char json[4096];
    snprintf(json, sizeof(json),
        R"({"type":"consciousness_state","state":"%s","wakefulness":%.3f,)"
        R"("kv_used":%d,"kv_free":%d,"pos":%d,)"
        R"("llm_loaded":true,"entity":"%s",)"
        R"("prefill_ms":%.1f,"prefill_tps":%.1f,"prefill_tokens":%d,)"
        R"("decode_ms_per_tok":%.1f,"decode_tokens":%d,)"
        R"("total_prefill_tok":%d,"total_decode_tok":%d,)"
        R"("total_prefill_ms":%.0f,"total_decode_ms":%.0f,)"
        R"("cuda_free_mb":%.0f,"cuda_total_mb":%.0f,)"
        R"("mem_avail_mb":%.0f,"rss_mb":%.0f,)"
        R"("enable_response":%s,"enable_daydream":%s,"enable_dreaming":%s,"enable_llm":%s,)"
        R"("enable_speech":%s,"enable_thinking":%s,"enable_action":%s,)"
        R"("temperature":%.2f,"top_k":%d,"top_p":%.2f,)"
        R"("speech_max_tokens":%d,"thinking_max_tokens":%d,)"
        R"("speech":{"temperature":%.2f,"top_k":%d,"top_p":%.2f,"max_tokens":%d},)"
        R"("thinking":{"temperature":%.2f,"top_k":%d,"top_p":%.2f,"max_tokens":%d},)"
        R"("action":{"temperature":%.2f,"top_k":%d,"top_p":%.2f,"max_tokens":%d}})",
        state_names[static_cast<int>(consciousness.current_state())],
        consciousness.scheduler().wakefulness(),
        llm_cache.gpu_blocks_used(), llm_cache.free_blocks(),
        consciousness.current_pos(),
        persona_cfg.name.c_str(),
        m.last_prefill_ms, m.last_prefill_tps, m.last_prefill_tokens,
        m.last_decode_ms_per_tok, m.last_decode_tokens,
        m.total_prefill_tokens, m.total_decode_tokens,
        m.total_prefill_ms, m.total_decode_ms,
        cuda_free / 1048576.0, cuda_total / 1048576.0,
        mem_avail_kb / 1024.0, rss_kb / 1024.0,
        consciousness.response_enabled() ? "true" : "false",
        consciousness.daydream_enabled() ? "true" : "false",
        consciousness.dreaming_enabled() ? "true" : "false",
        consciousness.llm_enabled() ? "true" : "false",
        consciousness.speech_enabled() ? "true" : "false",
        consciousness.thinking_enabled() ? "true" : "false",
        consciousness.action_enabled() ? "true" : "false",
        consciousness.temperature(), consciousness.top_k(),
        consciousness.top_p(),
        consciousness.speech_max_tokens(),
        consciousness.thinking_max_tokens(),
        consciousness.speech_cfg().sampling.temperature,
        consciousness.speech_cfg().sampling.top_k,
        consciousness.speech_cfg().sampling.top_p,
        consciousness.speech_cfg().max_tokens,
        consciousness.thinking_cfg().sampling.temperature,
        consciousness.thinking_cfg().sampling.top_k,
        consciousness.thinking_cfg().sampling.top_p,
        consciousness.thinking_cfg().max_tokens,
        consciousness.action_cfg().sampling.temperature,
        consciousness.action_cfg().sampling.top_k,
        consciousness.action_cfg().sampling.top_p,
        consciousness.action_cfg().max_tokens);
    server.send_text(fd, json);

    // Send prompt defaults as a separate message (prompts may contain
    // characters that break snprintf-assembled JSON).
    std::string pj = R"({"type":"consciousness_prompts",)"
        R"("identity":")" + json_escape(consciousness.identity_prompt()) + R"(",)"
        R"("speech":")" + json_escape(consciousness.speech_cfg().prompt) + R"(",)"
        R"("thinking":")" + json_escape(consciousness.thinking_cfg().prompt) + R"(",)"
        R"("action":")" + json_escape(consciousness.action_cfg().prompt) + R"("})";
    server.send_text(fd, pj);
}

}  // namespace actus
}  // namespace deusridet
