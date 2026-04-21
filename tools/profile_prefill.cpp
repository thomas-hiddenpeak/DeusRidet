/**
 * @file profile_prefill.cpp
 * @philosophical_role Developer instrument — Marlin MLP prefill timing across
 *         a sweep of batch sizes (M) on loaded model weights. Not an Actus:
 *         pure engine measurement.
 * @serves performance regression tracking for `forward_prefill` and
 *         `profile_sublayer_prefill` in `src/machina/forward.{h,cu}`.
 */
#include "machina/model.h"
#include "machina/forward.h"
#include "machina/tokenizer.h"
#include "communis/tegra.h"
#include "tools/dev_main_helper.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <utility>
#include <vector>

namespace deusridet {

static int run_profile_prefill(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[profile-prefill] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[profile-prefill] Tokenizer load failed\n");
        return 1;
    }

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[profile-prefill] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    const int M_vals[] = {11, 32, 64, 128, 256, 512};
    const int N_M = sizeof(M_vals) / sizeof(M_vals[0]);
    int max_M = M_vals[N_M - 1];

    InferenceState state;
    int max_kv_len = max_M;
    if (!state.allocate(max_kv_len)) {
        fprintf(stderr, "[profile-prefill] State allocation failed\n");
        free_model_weights(weights);
        return 1;
    }

    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);

    std::string prompt = "Hello";
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    auto base_tokens = tokenizer.apply_chat_template(messages);
    printf("[profile-prefill] Base prompt: \"%s\" → %zu tokens\n",
           prompt.c_str(), base_tokens.size());

    for (int mi = 0; mi < N_M; mi++) {
        int M = M_vals[mi];

        std::vector<int> tokens(M);
        for (int i = 0; i < M; i++)
            tokens[i] = (i < (int)base_tokens.size()) ? base_tokens[i] : 198;

        cudaMemset(kv_cache, 0, kv_bytes);
        for (int i = 0; i < state.num_dn_layers; i++) {
            size_t state_bytes = (size_t)MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM
                               * MC::LIN_V_HEAD_DIM * sizeof(float);
            cudaMemset(state.dn_states[i], 0, state_bytes);
            size_t conv_bytes = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1) * sizeof(__half);
            cudaMemset(state.conv_states[i], 0, conv_bytes);
        }

        forward_prefill(weights, state, kv_cache,
                        tokens.data(), M,
                        0, max_kv_len);

        cudaMemset(kv_cache, 0, kv_bytes);
        for (int i = 0; i < state.num_dn_layers; i++) {
            size_t state_bytes = (size_t)MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM
                               * MC::LIN_V_HEAD_DIM * sizeof(float);
            cudaMemset(state.dn_states[i], 0, state_bytes);
            size_t conv_bytes = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1) * sizeof(__half);
            cudaMemset(state.conv_states[i], 0, conv_bytes);
        }

        profile_forward_prefill(weights, state, kv_cache,
                                tokens.data(), M,
                                0, max_kv_len);

        for (int i = 0; i < state.num_dn_layers; i++) {
            size_t state_bytes = (size_t)MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM
                               * MC::LIN_V_HEAD_DIM * sizeof(float);
            cudaMemset(state.dn_states[i], 0, state_bytes);
            size_t conv_bytes = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1) * sizeof(__half);
            cudaMemset(state.conv_states[i], 0, conv_bytes);
        }
        profile_sublayer_prefill(weights, state, kv_cache,
                                 M, 0, max_kv_len);
    }

    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[profile-prefill] Done.\n");
    return 0;
}

}  // namespace deusridet

int main(int argc, char** argv) {
    std::string model_dir = deusridet::dev::resolve_model_dir(argc, argv);
    int rc = deusridet::run_profile_prefill(model_dir);
    deusridet::tegra_cleanup();
    return rc;
}
