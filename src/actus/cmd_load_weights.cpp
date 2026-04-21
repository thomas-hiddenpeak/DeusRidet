/**
 * @file cmd_load_weights.cpp
 * @philosophical_role External command `cmd_load_weights`. An Actus function — one CLI verb, one finite
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

int cmd_load_weights(const std::string& model_dir) {
    // Drop page caches for clean baseline
    drop_page_caches();
    cudaFree(0);

    ModelWeights weights;
    bool ok = load_model_weights(model_dir, weights);

    if (ok) {
        printf("\n[load-weights] Summary:\n");
        printf("  Total device memory: %.2f GB\n", weights.total_bytes / 1073741824.0);
        printf("  Full attention layers:");
        for (int i = 0; i < ModelConfig::NUM_LAYERS; i++) {
            if (weights.layers[i].is_full_attention) printf(" %d", i);
        }
        printf("\n");

        // Spot-check: print first layer MLP dimensions
        auto& mlp0 = weights.layers[0].mlp;
        printf("  Layer 0 MLP: gate[K=%d,N=%d] up[K=%d,N=%d] down[K=%d,N=%d]\n",
               mlp0.gate_proj.K, mlp0.gate_proj.N,
               mlp0.up_proj.K, mlp0.up_proj.N,
               mlp0.down_proj.K, mlp0.down_proj.N);

        // Spot-check: layer 0 DeltaNet dims
        auto& dn0 = weights.layers[0].delta_net;
        printf("  Layer 0 DeltaNet: qkv[%d→%d] z[%d→%d] out[%d→%d]\n",
               dn0.fp16_qkv.in_features, dn0.fp16_qkv.out_features,
               dn0.fp16_z.in_features, dn0.fp16_z.out_features,
               dn0.fp16_out.in_features, dn0.fp16_out.out_features);

        // Spot-check: layer 3 Full Attention dims
        auto& fa3 = weights.layers[3].full_attn;
        printf("  Layer 3 FullAttn: q[%d→%d] k[%d→%d] v[%d→%d] o[%d→%d]\n",
               fa3.fp16_q.in_features, fa3.fp16_q.out_features,
               fa3.fp16_k.in_features, fa3.fp16_k.out_features,
               fa3.fp16_v.in_features, fa3.fp16_v.out_features,
               fa3.fp16_o.in_features, fa3.fp16_o.out_features);
    }

    printf("\n[load-weights] Press Enter to release...\n");
    fflush(stdout);
    getchar();

    free_model_weights(weights);

    // Reclaim CMA pages
    drop_page_caches();

    printf("[load-weights] Released.\n");
    return ok ? 0 : 1;
}

} // namespace deusridet
