/**
 * @file profile_forward.cpp
 * @philosophical_role Developer instrument — single-token decode timing on
 *         loaded model weights. Not an Actus: pure engine measurement.
 * @serves performance regression tracking for `forward_decode` in
 *         `src/machina/forward.{h,cu}`.
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

namespace deusridet {

static int run_profile_forward(const std::string& model_dir) {
    using MC = ModelConfig;
    printf("[profile-forward] Loading model...\n");

    drop_page_caches();
    cudaFree(0);

    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[profile-forward] Tokenizer load failed\n");
        return 1;
    }

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[profile-forward] Weight load failed\n");
        return 1;
    }

    int max_seq = 32;
    InferenceState state;
    if (!state.allocate(max_seq)) {
        fprintf(stderr, "[profile-forward] State alloc failed\n");
        free_model_weights(weights);
        return 1;
    }

    int max_kv_len = max_seq;
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);
    cudaMemset(kv_cache, 0, kv_bytes);

    int token_id = 9419;  // "Hello"
    printf("[profile-forward] Profiling single-token decode...\n");

    profile_forward(weights, state, kv_cache, token_id, 0, max_kv_len);

    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();
    return 0;
}

}  // namespace deusridet

int main(int argc, char** argv) {
    std::string model_dir = deusridet::dev::resolve_model_dir(argc, argv);
    int rc = deusridet::run_profile_forward(model_dir);
    deusridet::tegra_cleanup();
    return rc;
}
