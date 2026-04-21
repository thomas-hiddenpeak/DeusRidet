/**
 * @file bench_prefill.cpp
 * @philosophical_role Developer instrument — measures Marlin vs cuBLAS FP16
 *         projection throughput across a Prefill batch sweep on real model
 *         weights. Not an Actus: pure engine measurement.
 * @serves performance regression tracking for prefill projections in
 *         `src/machina/forward.{h,cu}`.
 */
#include "machina/model.h"
#include "machina/forward.h"
#include "communis/tegra.h"
#include "tools/dev_main_helper.h"

#include <cstdio>
#include <string>

namespace deusridet {

static int run_bench_prefill(const std::string& model_dir) {
    printf("[bench-prefill] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[bench-prefill] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    InferenceState state;
    int max_seq = 2048;
    if (!state.allocate(max_seq)) {
        fprintf(stderr, "[bench-prefill] State allocation failed (max_seq=%d)\n", max_seq);
        free_model_weights(weights);
        return 1;
    }

    printf("[bench-prefill] State allocated (max_seq=%d)\n", max_seq);

    bench_prefill_projections(weights, state);

    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[bench-prefill] Done.\n");
    return 0;
}

}  // namespace deusridet

int main(int argc, char** argv) {
    std::string model_dir = deusridet::dev::resolve_model_dir(argc, argv);
    int rc = deusridet::run_bench_prefill(model_dir);
    deusridet::tegra_cleanup();
    return rc;
}
