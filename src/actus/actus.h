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

int cmd_load_model(const std::string& model_dir);
int cmd_load_weights(const std::string& model_dir);

// Continuous consciousness loop: WebSocket server + WebUI + audio +
// LLM stream. The principal Actus verb — the entity becoming awake.
int awaken(const std::string& webui_dir,
           const std::string& llm_model_dir = "",
           const std::string& persona_conf_path = "",
           float replay_speed = 1.0f);

} // namespace deusridet
