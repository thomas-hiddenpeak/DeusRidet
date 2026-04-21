/**
 * @file actus.cpp
 * @philosophical_role Registry of the Actus subsystem. Holds the module-scope
 *         shutdown flag (the one shared datum between the signal handler and
 *         every running command) plus the two non-command entry points
 *         `print_version()` and `print_usage()`.
 * @serves main.cpp. Every `cmd_*` lives in its own translation unit in
 *         src/actus/; see src/actus/README.la for the subsystem-level anchor.
 *
 * Rationale: an act is transient, so the Actus subsystem holds almost no
 * state. What little remains — the shutdown flag, the version strings, the
 * usage banner — lives here because it is shared across all acts.
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

static const char* VERSION    = "0.1.0";
static const char* BUILD_DATE = __DATE__;

volatile sig_atomic_t g_shutdown_requested = 0;

// ============================================================================
// version / usage
// ============================================================================

void print_version() {
    printf("DeusRidet v%s (%s)\n", VERSION, BUILD_DATE);
    printf("  \"When humans think, God laughs; when AI thinks, humans stop laughing.\"\n\n");

    int driver_ver = 0, runtime_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("  Device:    %s\n", prop.name);
    printf("  SM:        %d.%d (%d SMs)\n", prop.major, prop.minor, prop.multiProcessorCount);
    printf("  Memory:    %.1f GB\n", prop.totalGlobalMem / 1073741824.0);
    printf("  CUDA:      Driver %d.%d, Runtime %d.%d\n",
           driver_ver / 1000, (driver_ver % 100) / 10,
           runtime_ver / 1000, (runtime_ver % 100) / 10);
    printf("  License:   GPLv3\n");
}

void print_usage() {
    printf("\n  DeusRidet v%s — Continuous Consciousness Engine\n\n", VERSION);
    printf("  Usage:\n");
    printf("    deusridet <command> [options]\n\n");
    printf("  Commands:\n");
    printf("    test-tokenizer <text>   Encode/decode round-trip test\n");
    printf("    test-weights            Load weights and print tensor summary\n");
    printf("    test-gptq               GPTQ kernel correctness test with model weights\n");
    printf("    load-model              Load all weights to device, hold for inspection\n");
    printf("    load-weights            Structured weight load (model.h) with validation\n");
    printf("    test-forward            Single-token forward pass test\n");
    printf("    test-sample             Sampling test (greedy + top-k/p)\n");
    printf("    test-ws                 Start WebSocket server + serve WebUI\n");
    printf("    version                 Print version and hardware info\n\n");
    printf("  Note: bench-gptq / bench-gptq-v2 / bench-prefill /\n");
    printf("        profile-forward / profile-prefill have moved to standalone\n");
    printf("        executables under build/ (run ./build/bench_gptq etc.)\n\n");
    printf("  Options:\n");
    printf("    --config <file>         Configuration file (default: configs/machina.conf)\n");
    printf("    --model-dir <path>      Override LLM model directory\n\n");
}

} // namespace deusridet
