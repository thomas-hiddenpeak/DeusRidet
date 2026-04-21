/**
 * @file conscientia_facade.cpp
 * @philosophical_role Implementation of the Conscientia↔Nexus broadcast seam.
 *         Every snprintf format, field name, and field order here is a promise
 *         kept with the WebUI — changing any of them is a protocol break.
 */
#include "conscientia_facade.h"

#include "stream.h"
#include "nexus/ws_server.h"
#include "communis/json_util.h"
#include "communis/tegra.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

namespace deusridet {
namespace conscientia {

namespace {
const char* const kStateNames[] = {"active", "daydream", "dreaming"};
}  // namespace

void install_decode_callback(ConscientiStream& consciousness,
                             WsServer& server) {
    consciousness.set_on_decode([&server](const DecodeResult& result) {
        std::string escaped = communis::json_escape(result.text);
        char json[4096];
        snprintf(json, sizeof(json),
            R"({"type":"consciousness_decode","text":"%s","tokens":%d,"time_ms":%.1f,"state":"%s"})",
            escaped.c_str(), result.tokens_generated, result.time_ms,
            kStateNames[static_cast<int>(result.state_during)]);
        server.broadcast_text(json);
        printf("[consciousness] %s: \"%s\" (%d tok, %.0fms)\n",
               kStateNames[static_cast<int>(result.state_during)],
               result.text.c_str(), result.tokens_generated, result.time_ms);
    });
}

void install_speech_token_callback(ConscientiStream& consciousness,
                                   WsServer& server) {
    consciousness.set_on_speech_token([&server](const std::string& token_text, int token_id) {
        std::string escaped = communis::json_escape(token_text);
        char json[512];
        snprintf(json, sizeof(json),
            R"({"type":"speech_token","text":"%s","token_id":%d})",
            escaped.c_str(), token_id);
        server.broadcast_text(json);
    });
}

void install_state_callback(ConscientiStream& consciousness,
                            WsServer& server) {
    consciousness.set_on_state([&server](WakefulnessState state,
                                         float wakefulness,
                                         int kv_blocks_used,
                                         int kv_blocks_free,
                                         int current_pos,
                                         const ConscientiMetrics& metrics) {
        // System memory stats
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

        char json[1024];
        snprintf(json, sizeof(json),
            R"({"type":"consciousness_state","state":"%s","wakefulness":%.3f,)"
            R"("kv_used":%d,"kv_free":%d,"pos":%d,)"
            R"("prefill_ms":%.1f,"prefill_tps":%.1f,"prefill_tokens":%d,)"
            R"("decode_ms_per_tok":%.1f,"decode_tokens":%d,)"
            R"("total_prefill_tok":%d,"total_decode_tok":%d,)"
            R"("total_prefill_ms":%.0f,"total_decode_ms":%.0f,)"
            R"("cuda_free_mb":%.0f,"cuda_total_mb":%.0f,)"
            R"("mem_avail_mb":%.0f,"rss_mb":%.0f})",
            kStateNames[static_cast<int>(state)], wakefulness,
            kv_blocks_used, kv_blocks_free, current_pos,
            metrics.last_prefill_ms, metrics.last_prefill_tps,
            metrics.last_prefill_tokens,
            metrics.last_decode_ms_per_tok, metrics.last_decode_tokens,
            metrics.total_prefill_tokens, metrics.total_decode_tokens,
            metrics.total_prefill_ms, metrics.total_decode_ms,
            cuda_free / 1048576.0, cuda_total / 1048576.0,
            mem_avail_kb / 1024.0, rss_kb / 1024.0);
        server.broadcast_text(json);
    });
}

}  // namespace conscientia
}  // namespace deusridet
