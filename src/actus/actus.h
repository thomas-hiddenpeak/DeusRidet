/**
 * @file actus.h
 * @philosophical_role External-command entry points. The Latin `actus` means
 *         "act / deed" — the bridge between the user's verb (CLI argv) and
 *         the internal subsystems that carry out the deed.
 * @serves main.cpp as its dispatch target; every subcommand resolves to one
 *         function declared here.
 *
 * This header is the public contract of the Actus subsystem. It intentionally
 * contains only free functions in `namespace deusridet` — no classes, no
 * state — because an external act is, by definition, transient.
 *
 * Each command function is self-contained and returns 0 on success. The
 * shutdown flag is module-scoped state shared with the signal handler.
 */

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
int cmd_load_model(const std::string& model_dir);
int cmd_load_weights(const std::string& model_dir);
int cmd_test_forward(const std::string& model_dir);
int cmd_test_sample(const std::string& model_dir);

// WavLM+ECAPA layer-by-layer test
int cmd_test_wavlm_cnn();

// WebSocket server test (with optional LLM consciousness stream)
int cmd_test_ws(const std::string& webui_dir,
                const std::string& llm_model_dir = "",
                const std::string& persona_conf_path = "",
                float replay_speed = 1.0f);

} // namespace deusridet
