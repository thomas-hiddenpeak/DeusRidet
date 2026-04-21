/**
 * @file cmd_test_tokenizer.cpp
 * @philosophical_role External command `cmd_test_tokenizer`. An Actus function — one CLI verb, one finite
 *         act, one return code.
 * @serves main.cpp dispatch (declaration in actus.h).
 */


#include "actus/actus.h"
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

int cmd_test_tokenizer(const std::string& model_dir, const std::string& text) {
    Tokenizer tokenizer;
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

} // namespace deusridet

#include "tools/dev_main_helper.h"
int main(int argc, char** argv) {
    std::string model_dir = deusridet::dev::resolve_model_dir(argc, argv);
    std::string text = "Hello, world! 你好世界";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--config" || a == "--model-dir") { i++; continue; }
        if (!a.empty() && a[0] != '-') { text = a; break; }
    }
    int rc = deusridet::cmd_test_tokenizer(model_dir, text);
    deusridet::tegra_cleanup();
    return rc;
}
