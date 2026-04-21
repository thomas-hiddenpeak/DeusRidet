/**
 * @file cmd_bench_prefill.cpp
 * @philosophical_role External command `cmd_bench_prefill`. An Actus function — one CLI verb, one finite
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

int cmd_bench_prefill(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[bench-prefill] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[bench-prefill] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    // Allocate state with large max_seq to support up to M=2048
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

} // namespace deusridet
