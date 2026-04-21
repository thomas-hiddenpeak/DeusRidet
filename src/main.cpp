// main.cpp — DeusRidet entry point
//
// Thin entry: signal handling → config parsing → command dispatch → cleanup.
// All command implementations live in commands.cpp.
// Tegra platform utilities live in communis/tegra.h.
//
// DeusRidet — consciousness should not be locked behind closed doors.
// Licensed under GPLv3.

#include "commands.h"
#include "communis/config.h"
#include "communis/tegra.h"
#include <cstdio>
#include <cstring>
#include <csignal>
#include <string>
#include <climits>
#include <execinfo.h>
#include <unistd.h>

// ============================================================================
// Signal handlers
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

static void shutdown_handler(int sig) {
    if (deusridet::g_shutdown_requested) {
        // Second signal — user insists, hard exit after cleanup
        const char* msg = "\n[SHUTDOWN] Forced exit.\n";
        ssize_t r_ __attribute__((unused));
        r_ = write(STDERR_FILENO, msg, strlen(msg));
        deusridet::tegra_cleanup();
        _exit(128 + sig);
    }
    deusridet::g_shutdown_requested = 1;
    const char* msg = "\n[SHUTDOWN] Signal received, finishing current step...\n";
    ssize_t r_ __attribute__((unused));
    r_ = write(STDERR_FILENO, msg, strlen(msg));
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
    signal(SIGINT,  shutdown_handler);
    signal(SIGTERM, shutdown_handler);

    if (argc < 2) {
        deusridet::print_usage();
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
        } else if (model_dir_override.empty() && arg[0] != '-') {
            model_dir_override = arg;
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
        deusridet::print_version();
    }
    else if (cmd == "test-tokenizer") {
        std::string text = "Hello, world! 你好世界";
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--config" || arg == "--model-dir") { i++; continue; }
            text = arg;
            break;
        }
        rc = deusridet::cmd_test_tokenizer(model_dir, text);
    }
    else if (cmd == "test-weights") {
        rc = deusridet::cmd_test_weights(model_dir);
    }
    else if (cmd == "test-gptq") {
        rc = deusridet::cmd_test_gptq(model_dir);
    }
    else if (cmd == "bench-gptq") {
        rc = deusridet::cmd_bench_gptq();
    }
    else if (cmd == "bench-gptq-v2") {
        rc = deusridet::cmd_bench_gptq_v2();
    }
    else if (cmd == "load-model") {
        rc = deusridet::cmd_load_model(model_dir);
    }
    else if (cmd == "load-weights") {
        rc = deusridet::cmd_load_weights(model_dir);
    }
    else if (cmd == "test-forward") {
        rc = deusridet::cmd_test_forward(model_dir);
    }
    else if (cmd == "test-sample") {
        rc = deusridet::cmd_test_sample(model_dir);
    }
    else if (cmd == "profile-forward") {
        rc = deusridet::cmd_profile_forward(model_dir);
    }
    else if (cmd == "profile-prefill") {
        rc = deusridet::cmd_profile_prefill(model_dir);
    }
    else if (cmd == "profile-prefill-gptq-v2") {
        // Legacy alias: v2 is now the default kernel, redirect to profile-prefill
        rc = deusridet::cmd_profile_prefill(model_dir);
    }
    else if (cmd == "bench-prefill") {
        rc = deusridet::cmd_bench_prefill(model_dir);
    }
    else if (cmd == "test-wavlm-cnn") {
        rc = deusridet::cmd_test_wavlm_cnn();
    }
    else if (cmd == "test-ws") {
        // Default webui dir relative to executable location.
        std::string webui_dir = cfg.get_string("webui_dir", "");
        std::string persona_conf = "configs/persona.conf";
        float replay_speed = 1.0f;
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--webui" && i + 1 < argc) {
                webui_dir = argv[++i];
            } else if (arg == "--persona" && i + 1 < argc) {
                persona_conf = argv[++i];
            } else if (arg == "--test-replay-speed" && i + 1 < argc) {
                replay_speed = (float)std::atof(argv[++i]);
                if (!(replay_speed > 0.0f)) replay_speed = 1.0f;
            }
        }
        // If not set, resolve relative to executable's directory.
        if (webui_dir.empty()) {
            char exe_path[PATH_MAX] = {};
            ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
            if (n > 0) {
                exe_path[n] = '\0';
                // Strip executable name to get directory.
                char* last_slash = strrchr(exe_path, '/');
                if (last_slash) *(last_slash + 1) = '\0';
                webui_dir = std::string(exe_path) + "../src/nexus/webui";
            } else {
                webui_dir = "../src/nexus/webui";  // fallback
            }
        }
        rc = deusridet::cmd_test_ws(webui_dir, model_dir, persona_conf, replay_speed);
    }
    else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        deusridet::print_usage();
    }
    else {
        fprintf(stderr, "Unknown command: %s\nRun 'deusridet --help' for usage.\n", cmd.c_str());
        rc = 1;
    }

    deusridet::tegra_cleanup();

    return rc;
}
