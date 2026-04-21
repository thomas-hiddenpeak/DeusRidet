/**
 * @file awaken_consciousness.h
 * @philosophical_role Bootstrap bundle for the four subsystems the
 *     principal verb `awaken` must bring online before any WebSocket or
 *     Auditus wiring can be installed: the LLM (machina), its paged
 *     KV cache (memoria), the Persona (persona_config), and the
 *     Conscientia stream itself. Grouping them in one struct keeps the
 *     `awaken()` orchestrator from having to name six separate locals
 *     and six separate `*_ptr` shadows just to hand them to callbacks.
 * @serves `awaken.cpp` as a peer Actus TU. The struct is the bootstrap
 *     outcome; `bootstrap_consciousness()` is the verb that fills it.
 *     Member instances own their resources; `awaken()` is still
 *     responsible for the matching destroy/free at shutdown (see
 *     existing call pattern in `awaken.cpp`).
 */
#pragma once

#include "communis/config.h"      // PersonaConfig
#include "conscientia/stream.h"   // ConscientiStream
#include "machina/model.h"        // ModelWeights, InferenceState
#include "machina/tokenizer.h"    // Tokenizer
#include "memoria/cache_manager.h" // CacheManager

#include <string>

namespace deusridet {

// All state the Conscientia stream needs to live. When `loaded` is
// false, only default-constructed members are present — `awaken()`
// must skip the decode callbacks, the hello envelope, and shutdown.
struct ConscientiaBootstrap {
    Tokenizer        tokenizer;
    ModelWeights     weights = {};
    InferenceState   state   = {};
    CacheManager     cache;
    ConscientiStream stream;
    PersonaConfig    persona_cfg;
    bool             loaded  = false;
};

// Optional bootstrap — governed by env `DEUSRIDET_TEST_WS_ENABLE_LLM=1`
// and a non-empty `llm_model_dir`. Returns 0 on success (with or
// without LLM loaded) and 1 on any fatal step failure; on failure the
// function undoes the partially-allocated resources before returning.
int bootstrap_consciousness(const std::string& llm_model_dir,
                            const std::string& persona_conf_path,
                            ConscientiaBootstrap& out);

} // namespace deusridet
