/**
 * @file cmd_test_sample.cpp
 * @philosophical_role External command `cmd_test_sample`. An Actus function — one CLI verb, one finite
 *         act, one return code.
 * @serves main.cpp dispatch (declaration in actus.h).
 */


#include "actus.h"
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

int cmd_test_sample(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[test-sample] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[test-sample] Tokenizer load failed\n");
        return 1;
    }

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[test-sample] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    InferenceState state;
    int max_kv_len = 128;
    if (!state.allocate(max_kv_len)) {
        fprintf(stderr, "[test-sample] State allocation failed\n");
        return 1;
    }

    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);
    cudaMemset(kv_cache, 0, kv_bytes);

    std::string prompt = "Hello";
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    auto tokens = tokenizer.apply_chat_template(messages);
    printf("[test-sample] Prompt: \"%s\" → %zu tokens\n", prompt.c_str(), tokens.size());

    if (tokens.empty()) {
        cudaFree(kv_cache);
        state.free();
        free_model_weights(weights);
        return 1;
    }

    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 50;
    params.top_p = 0.9f;

    int max_gen = 32;
    printf("[test-sample] Sampling: temp=%.1f, top_k=%d, top_p=%.2f, max_gen=%d\n",
           params.temperature, params.top_k, params.top_p, max_gen);

    std::vector<int> generated;
    int pos = 0;
    int next_token = tokens[0];

    // Prefill (batched)
    auto t_start = std::chrono::high_resolution_clock::now();
    next_token = forward_prefill(weights, state, kv_cache,
                                 tokens.data(), (int)tokens.size(),
                                 pos, max_kv_len);
    pos += (int)tokens.size();
    generated.push_back(next_token);

    // Decode
    for (int g = 1; g < max_gen; g++) {
        if (g_shutdown_requested) break;
        if (pos >= max_kv_len - 1) break;
        next_token = forward_one_token_sampled(weights, state, kv_cache,
                                                next_token, pos, max_kv_len, params);
        pos++;
        generated.push_back(next_token);
        if (next_token == 151643 || next_token == 151645) break;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("[test-sample] Generated %zu tokens in %.1f ms (%.1f ms/token)\n",
           generated.size(), total_ms, total_ms / generated.size());
    printf("[test-sample] Output: ");
    for (int t : generated) printf("%s", tokenizer.decode(t).c_str());
    printf("\n");
    printf("[test-sample] Token IDs:");
    for (int t : generated) printf(" %d", t);
    printf("\n");

    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[test-sample] Done.\n");
    return 0;
}

} // namespace deusridet
