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
int cmd_load_model(const std::string& model_dir);
int cmd_load_weights(const std::string& model_dir);
int cmd_test_forward(const std::string& model_dir);
int cmd_test_sample(const std::string& model_dir);
int cmd_profile_forward(const std::string& model_dir);
int cmd_profile_prefill(const std::string& model_dir);

} // namespace deusridet
