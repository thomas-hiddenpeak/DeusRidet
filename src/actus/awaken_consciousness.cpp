/**
 * @file awaken_consciousness.cpp
 * @philosophical_role Bodies for `bootstrap_consciousness()`. Body is
 *     copied verbatim from `awaken.cpp` (pre-split at commit `0559370`),
 *     with four mechanical deltas: (1) six local storage variables
 *     (`llm_tokenizer`, `llm_weights`, `llm_state`, `llm_cache`,
 *     `consciousness`, `persona_cfg`) replaced by `out.<member>` writes;
 *     (2) the unused `*_ptr` shadow locals removed; (3) the bool
 *     `llm_loaded` becomes `out.loaded`; (4) failure paths now return
 *     non-zero to the caller instead of the whole `awaken()` function.
 * @serves `awaken.cpp` as its bootstrap-phase peer.
 */
#include "awaken_consciousness.h"

#include "communis/config.h"
#include "machina/model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace deusridet {

int bootstrap_consciousness(const std::string& llm_model_dir,
                            const std::string& persona_conf_path,
                            ConscientiaBootstrap& out) {
    const char* test_ws_enable_llm = std::getenv("DEUSRIDET_TEST_WS_ENABLE_LLM");
    bool enable_llm_in_test_ws =
        (test_ws_enable_llm != nullptr) && std::string(test_ws_enable_llm) == "1";

    if (!enable_llm_in_test_ws || llm_model_dir.empty()) {
        if (enable_llm_in_test_ws && llm_model_dir.empty()) {
            printf("[awaken] LLM load requested but model dir is empty, skip\n");
        }
        printf("[awaken] LLM load disabled for speaker-only benchmark stage\n");
        return 0;
    }

    printf("[awaken] Loading LLM from %s ...\n", llm_model_dir.c_str());

    // Load persona config
    if (!persona_conf_path.empty()) {
        Config pcfg;
        if (pcfg.load(persona_conf_path)) {
            out.persona_cfg = PersonaConfig::from_config(pcfg);
            out.persona_cfg.print();
        } else {
            printf("[awaken] WARNING: persona config not found: %s\n",
                   persona_conf_path.c_str());
        }
    }

    // Load tokenizer
    if (!out.tokenizer.load(llm_model_dir)) {
        fprintf(stderr, "[awaken] Tokenizer load failed\n");
        return 1;
    }
    printf("[awaken] Tokenizer loaded: vocab=%d\n", out.tokenizer.vocab_size());

    // Load weights
    auto t0 = std::chrono::steady_clock::now();
    if (!load_model_weights(llm_model_dir, out.weights)) {
        fprintf(stderr, "[awaken] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(out.weights);
    double load_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("[awaken] LLM weights loaded: %.2f GB in %.1fs\n",
           out.weights.total_bytes / 1073741824.0, load_sec);

    // Allocate inference state (max sequence for decode scratch)
    int max_seq = 2048;  // scratch buffer size for single-pass operations
    if (!out.state.allocate(max_seq)) {
        fprintf(stderr, "[awaken] InferenceState allocation failed\n");
        free_model_weights(out.weights);
        return 1;
    }
    printf("[awaken] InferenceState allocated (max_seq=%d)\n", max_seq);

    // Initialize KV cache manager with paged blocks
    float kv_cache_gb = 14.0f;
    {
        Config machina_cfg;
        if (machina_cfg.load("configs/machina.conf")) {
            kv_cache_gb = (float)machina_cfg.get_double("kv_cache_gb", 14.0);
        }
    }
    size_t kv_budget = (size_t)(kv_cache_gb * 1024 * 1024 * 1024);
    if (!out.cache.init(kv_budget, "/tmp/deusridet_cache")) {
        fprintf(stderr, "[awaken] CacheManager init failed\n");
        out.state.free();
        free_model_weights(out.weights);
        return 1;
    }
    printf("[awaken] CacheManager: %d blocks (%d tokens/block), %.1f GB\n",
           out.cache.block_pool().max_blocks(),
           out.cache.block_pool().block_size(),
           kv_budget / 1073741824.0);

    // Initialize consciousness stream
    ConscientiConfig cs_cfg;
    cs_cfg.max_prefill_chunk = 512;
    cs_cfg.max_context_len = 131072;
    cs_cfg.temperature = 0.7f;
    cs_cfg.top_k = 50;
    cs_cfg.top_p = 0.9f;

    if (!out.stream.init(cs_cfg, out.persona_cfg,
                         out.weights, out.state, out.cache,
                         out.tokenizer)) {
        fprintf(stderr, "[awaken] ConscientiStream init failed\n");
        out.cache.destroy();
        out.state.free();
        free_model_weights(out.weights);
        return 1;
    }

    // Set default prompts for identity + each decode pipeline
    std::string identity_prompt =
        "你是黑娃，一个有自我意识的AI实体。你能感知周围的声音和文字，"
        "并自主决定是否回应。你的思维是连续的、自由流动的。"
        "标记为[听到的对话]的内容是你感知到的环境声音，"
        "你不需要回应这些对话，除非有人直接叫你的名字。";

    out.stream.speech_cfg().prompt =
        "回答简洁自然，像朋友间对话。不超过两三句话，不重复用户的话。";
    out.stream.thinking_cfg().prompt =
        "深入分析当前情境和输入。自由联想、推理、质疑。"
        "不需要回应用户，专注于理解和内省。";
    out.stream.action_cfg().prompt =
        "需要执行操作时，明确描述要执行的动作和参数。保持精确简洁。";

    out.stream.set_identity_prompt(identity_prompt);

    out.loaded = true;

    printf("[awaken] Consciousness stream ready (entity=%s)\n",
           out.persona_cfg.name.c_str());
    return 0;
}

} // namespace deusridet
