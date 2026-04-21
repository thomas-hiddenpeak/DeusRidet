// commands.h — CLI command implementations
//
// Each command is a self-contained function returning 0 on success.
// Extracted from main.cpp to keep the entry point clean.

#pragma once

#include <string>
#include <csignal>

namespace deusridet {

// Shutdown flag — set by signal handler, checked by long-running commands.
extern volatile sig_atomic_t g_shutdown_requested;

void print_version();
void print_usage();

int cmd_test_tokenizer(const std::string& model_dir, const std::string& text);
int cmd_test_weights(const std::string& model_dir);
int cmd_test_gptq(const std::string& model_dir);
int cmd_bench_gptq();
int cmd_bench_gptq_v2();
int cmd_load_model(const std::string& model_dir);
int cmd_load_weights(const std::string& model_dir);
int cmd_test_forward(const std::string& model_dir);
int cmd_test_sample(const std::string& model_dir);
int cmd_profile_forward(const std::string& model_dir);
int cmd_profile_prefill(const std::string& model_dir);

int cmd_bench_prefill(const std::string& model_dir);

// WavLM+ECAPA layer-by-layer test
int cmd_test_wavlm_cnn();

// WebSocket server test (with optional LLM consciousness stream)
int cmd_test_ws(const std::string& webui_dir,
                const std::string& llm_model_dir = "",
                const std::string& persona_conf_path = "",
                float replay_speed = 1.0f);

} // namespace deusridet
