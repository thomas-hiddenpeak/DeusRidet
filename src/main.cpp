// main.cpp — DeusRidet entry point
//
// Boot sequence: parse config → load weights → initialize tokenizer
//                → validate → report status
//
// Commands:
//   test-tokenizer   — Encode/decode round-trip test
//   test-weights     — Load and print weight tensor summary
//   version          — Print version and hardware info
//   (future)         — boot consciousness stream
//
// DeusRidet — consciousness should not be locked behind closed doors.
// Licensed under GPLv3.

#include "communis/config.h"
#include "communis/log.h"
#include "machina/safetensors.h"
#include "machina/tokenizer.h"
#include <iostream>
#include <cstring>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const char* VERSION    = "0.1.0";
static const char* BUILD_DATE = __DATE__;

// ============================================================================
// Crash handler
// ============================================================================

static void crash_handler(int sig) {
    const char* msg = "\n[CRASH] Signal: ";
    ssize_t r_ __attribute__((unused));
    r_ = write(STDERR_FILENO, msg, strlen(msg));
    char num[16];
    int len = snprintf(num, sizeof(num), "%d\n", sig);
    r_ = write(STDERR_FILENO, num, len);

    void* frames[32];
    int n = backtrace(frames, 32);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    signal(sig, SIG_DFL);
    raise(sig);
}

// ============================================================================
// Commands
// ============================================================================

static void print_version() {
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

static void print_usage() {
    printf("\n  DeusRidet v%s — Continuous Consciousness Engine\n\n", VERSION);
    printf("  Usage:\n");
    printf("    deusridet <command> [options]\n\n");
    printf("  Commands:\n");
    printf("    test-tokenizer <text>   Encode/decode round-trip test\n");
    printf("    test-weights            Load weights and print tensor summary\n");
    printf("    version                 Print version and hardware info\n\n");
    printf("  Options:\n");
    printf("    --config <file>         Configuration file (default: configs/machina.conf)\n");
    printf("    --model-dir <path>      Override LLM model directory\n\n");
}

static int cmd_test_tokenizer(const std::string& model_dir, const std::string& text) {
    deusridet::Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        LOG_ERROR("Main", "Failed to load tokenizer from %s", model_dir.c_str());
        return 1;
    }

    printf("[Tokenizer] vocab_size = %d\n", tokenizer.vocab_size());
    printf("[Tokenizer] eos_id = %d, im_start_id = %d, im_end_id = %d\n",
           tokenizer.eos_token_id(), tokenizer.im_start_id(), tokenizer.im_end_id());

    auto ids = tokenizer.encode(text);
    printf("\n[Encode] \"%s\"\n  → %zu tokens: [", text.c_str(), ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        printf("%d%s", ids[i], i + 1 < ids.size() ? ", " : "");
    }
    printf("]\n");

    std::string decoded = tokenizer.decode(ids);
    printf("\n[Decode] → \"%s\"\n", decoded.c_str());

    bool match = (decoded == text);
    printf("\n[Round-trip] %s\n", match ? "PASS ✓" : "MISMATCH ✗");

    // Chat template test
    std::vector<std::pair<std::string, std::string>> messages = {
        {"system", "You are a helpful assistant."},
        {"user",   text}
    };
    auto chat_ids = tokenizer.apply_chat_template(messages);
    printf("\n[ChatML] %zu tokens\n", chat_ids.size());

    return match ? 0 : 1;
}

static int cmd_test_weights(const std::string& model_dir) {
    LOG_INFO("Main", "Loading weights from %s ...", model_dir.c_str());

    deusridet::SafetensorsLoader loader(model_dir);
    auto names = loader.tensor_names();

    printf("\n[Weights] %zu tensors across %zu shards\n\n", names.size(), loader.shard_count());

    // Sort names for readable output
    std::sort(names.begin(), names.end());

    size_t total_bytes = 0;
    int shown = 0;
    for (const auto& name : names) {
        auto tensor = loader.get_tensor(name);
        size_t nb = tensor->nbytes();
        total_bytes += nb;

        // Print first 20 and last 5 for brevity
        if (shown < 20 || names.size() - shown <= 5) {
            printf("  %-60s  %6s  [", name.c_str(),
                   deusridet::dtype_name(tensor->dtype()));
            for (size_t i = 0; i < tensor->shape().size(); i++) {
                printf("%lld%s", (long long)tensor->shape()[i],
                       i + 1 < tensor->shape().size() ? ", " : "");
            }
            printf("]  %.2f MB\n", nb / 1048576.0);
        } else if (shown == 20) {
            printf("  ... (%zu more tensors) ...\n", names.size() - 25);
        }
        shown++;
    }

    printf("\n[Total] %.2f GB across %zu tensors\n", total_bytes / 1073741824.0, names.size());
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    signal(SIGSEGV, crash_handler);
    signal(SIGBUS,  crash_handler);
    signal(SIGFPE,  crash_handler);
    signal(SIGABRT, crash_handler);

    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string cmd = argv[1];

    // Parse global options
    std::string config_path = "configs/machina.conf";
    std::string model_dir_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir_override = argv[++i];
        }
    }

    // Load config
    deusridet::Config cfg;
    cfg.load(config_path);

    std::string model_dir = model_dir_override.empty()
        ? cfg.get_string("llm_model_dir")
        : model_dir_override;

    // Dispatch
    int rc = 0;
    if (cmd == "version" || cmd == "--version" || cmd == "-v") {
        print_version();
    }
    else if (cmd == "test-tokenizer") {
        std::string text = "Hello, world! 你好世界";
        // Use remaining args as text
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--config" || arg == "--model-dir") { i++; continue; }
            text = arg;
            break;
        }
        rc = cmd_test_tokenizer(model_dir, text);
    }
    else if (cmd == "test-weights") {
        rc = cmd_test_weights(model_dir);
    }
    else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
    }
    else {
        fprintf(stderr, "Unknown command: %s\nRun 'deusridet --help' for usage.\n", cmd.c_str());
        rc = 1;
    }

    cudaDeviceReset();
    return rc;
}
