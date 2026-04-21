/**
 * @file cmd_bench_gptq.cpp
 * @philosophical_role External command `cmd_bench_gptq`. An Actus function — one CLI verb, one finite
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

int cmd_bench_gptq() {
    printf("[GPTQ Benchmark] — SM87 Jetson AGX Orin\n");
    printf("  GPTQ: bits=4, group_size=128, sym=true\n\n");

    // Dimensions from Qwen3.5-27B model
    struct BenchCase {
        const char* name;
        int K, N, M;
        int warmup, iters;
    };

    BenchCase cases[] = {
        // Decode (M=1)
        {"gate_proj GEMV  (5120→17408)",          5120, 17408,   1, 10, 50},
        {"down_proj GEMV  (17408→5120)",          17408,  5120,   1, 10, 50},
        // Prefill (various M)
        {"gate_proj GEMM M=32  (5120→17408)",     5120, 17408,  32,  5, 20},
        {"gate_proj GEMM M=128 (5120→17408)",     5120, 17408, 128,  3, 10},
        {"gate_proj GEMM M=512 (5120→17408)",     5120, 17408, 512,  2,  5},
        {"down_proj GEMM M=128 (17408→5120)",    17408,  5120, 128,  3, 10},
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);
    bool all_correct = true;

    printf("  %-45s %10s %10s %10s %8s\n",
           "Case", "Time(us)", "BW(GB/s)", "TFLOPS", "Correct");
    printf("  %s\n", std::string(90, '-').c_str());

    for (int c = 0; c < num_cases; c++) {
        auto& bc = cases[c];
        auto r = gptq_benchmark(bc.K, bc.N, bc.M, bc.warmup, bc.iters);

        if (bc.M == 1) {
            printf("  %-45s %10.1f %10.1f %10s %8s\n",
                   bc.name, r.gemv_us, r.gemv_gbps, "—",
                   r.correct ? "✓" : "✗");
        } else {
            printf("  %-45s %10.1f %10s %10.3f %8s\n",
                   bc.name, r.gemm_us, "—", r.gemm_tflops,
                   r.correct ? "✓" : "✗");
        }

        if (!r.correct) all_correct = false;
    }

    printf("\n  %s\n", all_correct ? "All correctness checks PASSED ✓" :
                                      "Some correctness checks FAILED ✗");
    return all_correct ? 0 : 1;
}

} // namespace deusridet
