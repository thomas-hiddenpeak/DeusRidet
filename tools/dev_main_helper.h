/**
 * @file dev_main_helper.h
 * @philosophical_role Argument-parse stub for developer instruments. These
 *         executables are *not* Actus — they are the developer's tools for
 *         poking at internals — and they intentionally repeat just enough
 *         of `main.cpp`'s arg conventions (--config / --model-dir / first
 *         positional) so that running e.g. `./bench_prefill` "feels like"
 *         the old `deusridet bench-prefill` did.
 * @serves every standalone executable under tools/ and tests/.
 */
#pragma once

#include "communis/config.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace deusridet {
namespace dev {

// Mirror of main.cpp's resolution rules:
//   --config <path>      override config file (default: configs/machina.conf)
//   --model-dir <path>   override model dir
//   <first non-flag arg> treated as model_dir if --model-dir not given
// Falls back to cfg.get_string("llm_model_dir") when nothing is supplied.
inline std::string resolve_model_dir(int argc, char** argv) {
    std::string config_path = "configs/machina.conf";
    std::string model_dir_override;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir_override = argv[++i];
        } else if (model_dir_override.empty() && !arg.empty() && arg[0] != '-') {
            model_dir_override = arg;
        }
    }
    Config cfg;
    cfg.load(config_path);
    return model_dir_override.empty()
               ? cfg.get_string("llm_model_dir")
               : model_dir_override;
}

}  // namespace dev
}  // namespace deusridet
