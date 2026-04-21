/**
 * @file cmd_test_forward.cpp
 * @philosophical_role External command `cmd_test_forward`. An Actus function — one CLI verb, one finite
 *         act, one return code.
 * @serves main.cpp dispatch (declaration in actus.h).
 */


#include "actus/actus.h"
#include "communis/config.h"
#include "communis/log.h"
#include "communis/tegra.h"
#include "machina/gptq.h"
#include "machina/gptq_gemm_v2.h"
#include "machina/model.h"
#include "machina/forward.h"
#include "machina/allocator.h"
#include "machina/safetensors.h"
#include "machina/tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <signal.h>
#include "nexus/ws_server.h"
#include "sensus/auditus/audio_pipeline.h"
#include "orator/wavlm_ecapa_encoder.h"
#include "conscientia/stream.h"
#include "memoria/cache_manager.h"
#include "communis/timeline_logger.h"

namespace deusridet {

int cmd_test_forward(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[test-forward] Loading model weights...\n");

    // Drop caches
    drop_page_caches();
    cudaFree(0);

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[test-forward] Tokenizer load failed\n");
        return 1;
    }

    // Load weights
    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[test-forward] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);
    printf("[test-forward] Weights loaded: %.2f GB\n",
           weights.total_bytes / 1073741824.0);

    // Allocate inference state
    int max_seq = 128;  // Enough for chat template + generation
    InferenceState state;
    if (!state.allocate(max_seq)) {
        fprintf(stderr, "[test-forward] State allocation failed\n");
        free_model_weights(weights);
        return 1;
    }

    // Allocate KV cache for full attention layers
    // Layout: [num_full_attn_layers * 2 * num_kv_heads * max_kv_len * head_dim]
    int num_full_attn = 0;
    for (int i = 0; i < MC::NUM_LAYERS; i++)
        if (MC::is_full_attention(i)) num_full_attn++;

    int max_kv_len = max_seq;
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);
    cudaMemset(kv_cache, 0, kv_bytes);
    printf("[test-forward] KV cache: %.1f MB (max_kv=%d, %d full-attn layers)\n",
           kv_bytes / 1048576.0, max_kv_len, num_full_attn);

    // Encode prompt using ChatML template for proper model behavior
    std::string prompt = "Hello";
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    auto tokens = tokenizer.apply_chat_template(messages);
    printf("[test-forward] Prompt: \"%s\" → %zu tokens:", prompt.c_str(), tokens.size());
    for (int t : tokens) printf(" %d", t);
    printf("\n");

    if (tokens.empty()) {
        fprintf(stderr, "[test-forward] Tokenizer returned empty tokens\n");
        cudaFree(kv_cache);
        state.free();
        free_model_weights(weights);
        return 1;
    }

    // Generate tokens
    int max_gen = 16;
    printf("[test-forward] Generating %d tokens...\n", max_gen);

    std::vector<int> generated;

    // Process prompt tokens one by one (prefill)
    int pos = 0;
    int next_token = tokens[0];

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    // Batched prefill: process all prompt tokens in one pass (GEMM instead of 11× GEMV)
    next_token = forward_prefill(weights, state, kv_cache,
                                 tokens.data(), (int)tokens.size(),
                                 pos, max_kv_len);
    pos += (int)tokens.size();
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    generated.push_back(next_token);
    printf("[test-forward] Prefill: %zu tokens in %.1f ms (%.1f ms/token)\n",
           tokens.size(), prefill_ms, prefill_ms / tokens.size());
    printf("[test-forward] First generated token: %d = \"%s\"\n",
           next_token, tokenizer.decode(next_token).c_str());

    // Generate remaining tokens (decode)
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int g = 1; g < max_gen; g++) {
        if (g_shutdown_requested) break;
        if (pos >= max_kv_len - 1) break;  // Safety limit
        next_token = forward_one_token(weights, state, kv_cache,
                                       next_token, pos, max_kv_len);
        pos++;
        generated.push_back(next_token);

        // Check for EOS (token 151643 or 151645)
        if (next_token == 151643 || next_token == 151645) break;
    }
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    int decode_count = (int)generated.size() - 1;  // exclude first token (from prefill)

    // Print results
    double total_ms = prefill_ms + decode_ms;
    printf("\n[test-forward] Decode: %d tokens in %.1f ms (%.1f ms/token)\n",
           decode_count, decode_ms, decode_count > 0 ? decode_ms / decode_count : 0.0);
    printf("[test-forward] Total: %.1f ms  (prefill %.1f + decode %.1f)\n",
           total_ms, prefill_ms, decode_ms);
    printf("[test-forward] Output: ");
    for (int t : generated) {
        printf("%s", tokenizer.decode(t).c_str());
    }
    printf("\n");
    printf("[test-forward] Token IDs:");
    for (int t : generated) printf(" %d", t);
    printf("\n");

    // Cleanup
    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[test-forward] Done.\n");
    return 0;
}

} // namespace deusridet

#include "tools/dev_main_helper.h"
#include <signal.h>
namespace deusridet { volatile sig_atomic_t g_shutdown_requested = 0; }
int main(int argc, char** argv) {
    std::string model_dir = deusridet::dev::resolve_model_dir(argc, argv);
    int rc = deusridet::cmd_test_forward(model_dir);
    deusridet::tegra_cleanup();
    return rc;
}
