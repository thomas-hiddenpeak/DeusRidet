/**
 * @file load_model.cpp
 * @philosophical_role External command `load_model`. An Actus function — one CLI verb, one finite
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

int load_model(const std::string& model_dir) {
    using Clock = std::chrono::steady_clock;

    // Drop page caches for clean baseline measurement
    drop_page_caches();

    // Force CUDA context init
    cudaFree(0);
    size_t avail_baseline = read_memavail_kb() / 1024;
    printf("[Load Model] Baseline (after drop_caches + CUDA init): MemAvail=%zuMB\n", avail_baseline);

    printf("[Load Model] Loading from %s\n", model_dir.c_str());

    auto t0 = Clock::now();

    ModelWeights weights;
    bool ok = load_model_weights(model_dir, weights);
    if (!ok) {
        fprintf(stderr, "[Load Model] Weight loading failed\n");
        return 1;
    }

    double total_sec = std::chrono::duration<double>(Clock::now() - t0).count();

    printf("\n[Load Model] Done: %.2f GB in %d pool blocks, %.1fs (%.0f MB/s)\n",
           weights.total_bytes / 1073741824.0,
           (int)weights.pool_blocks.size(),
           total_sec,
           (weights.total_bytes / 1048576.0) / total_sec);

    // Memory breakdown: clean comparison against drop_caches baseline
    {
        size_t vm_rss_kb = 0, rss_anon_kb = 0, rss_file_kb = 0;
        FILE* st = fopen("/proc/self/status", "r");
        if (st) {
            char line[256];
            while (fgets(line, sizeof(line), st)) {
                if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line+6, " %zu", &vm_rss_kb);
                else if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line+8, " %zu", &rss_anon_kb);
                else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line+8, " %zu", &rss_file_kb);
            }
            fclose(st);
        }
        size_t cuda_free = 0, cuda_total = 0;
        cudaMemGetInfo(&cuda_free, &cuda_total);
        size_t avail_now = read_memavail_kb() / 1024;
        size_t consumed = (avail_baseline > avail_now) ? (avail_baseline - avail_now) : 0;

        printf("\n[Memory Breakdown]\n");
        printf("  Pool blocks (cudaMalloc): %8d\n", (int)weights.pool_blocks.size());
        printf("  cudaMemGetInfo free:      %8.1f MB / %.1f MB\n",
               cuda_free / 1048576.0, cuda_total / 1048576.0);
        printf("  Process VmRSS:            %8.1f MB\n", vm_rss_kb / 1024.0);
        printf("    RssAnon  (heap+GPU):    %8.1f MB\n", rss_anon_kb / 1024.0);
        printf("    RssFile  (file-backed): %8.1f MB\n", rss_file_kb / 1024.0);
        printf("  MemAvail baseline:        %8zu MB  (after drop_caches + CUDA init)\n", avail_baseline);
        printf("  MemAvail now:             %8zu MB\n", avail_now);
        printf("  System RAM consumed:      %8zu MB  (baseline - now)\n", consumed);
        printf("  Weight data loaded:       %8.1f MB\n", weights.total_bytes / 1048576.0);
        printf("  Overhead (consumed-load): %8ld MB\n",
               (long)consumed - (long)(weights.total_bytes / 1048576));
    }

    printf("\n[Load Model] Weights held in %d pool blocks. Press Enter to release...\n",
           (int)weights.pool_blocks.size());
    fflush(stdout);
    getchar();

    free_model_weights(weights);
    drop_page_caches();

    size_t avail_after = read_memavail_kb() / 1024;
    printf("[Load Model] Released. MemAvail=%zuMB (recovered ~%ldMB vs baseline)\n",
           avail_after, (long)avail_after - (long)avail_baseline);

    return 0;
}

} // namespace deusridet
