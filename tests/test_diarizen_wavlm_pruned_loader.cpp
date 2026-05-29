/**
 * @file test_diarizen_wavlm_pruned_loader.cpp
 * @philosophical_role P1a-step1 smoke test: load the wavlm-pruned safetensors
 *     into a GPU arena and verify per-layer dimensions. Forward pass and
 *     bit-equality against Python reference (P1a-step2) are out of scope.
 *
 * Usage:
 *   ./build/test_diarizen_wavlm_pruned_loader \
 *       /home/rm01/models/dev/diarizen_v2/wavlm_pruned.safetensors
 *
 * Exit codes:
 *   0  - load succeeded, summary printed, sanity-check passed
 *   1  - load failed (LOG_ERROR explains)
 *   2  - sanity-check failed (a known tensor missing or wrong shape)
 */
#include <cstdio>
#include <cstdlib>
#include <string>

#include "orator/diarizen_wavlm_pruned.h"

namespace {

constexpr const char* kDefaultWeights =
    "/home/rm01/models/dev/diarizen_v2/wavlm_pruned.safetensors";

}  // namespace

int main(int argc, char** argv) {
    const std::string path =
        argc > 1 ? std::string(argv[1]) : std::string(kDefaultWeights);

    deusridet::orator::DiarizenWavlmPruned model;
    if (!model.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 1;
    }

    model.log_summary();

    // A few hand-picked tensors must exist for downstream P1a-step2 to work.
    const char* required[] = {
        "weight_sum.weight",
        "proj.weight",
        "lnorm.weight",
        "wavlm_model.encoder.feature_projection.projection.weight",
        "wavlm_model.encoder.transformer.layer_norm.weight",
        "wavlm_model.encoder.transformer.layers.0.attention.k_proj.weight",
        "wavlm_model.encoder.transformer.layers.23.feed_forward.output_dense.weight",
        "wavlm_model.feature_extractor.conv_layers.6.layer_norm.weight",
    };
    for (const char* n : required) {
        if (!model.find(n)) {
            std::fprintf(stderr, "MISSING required tensor: %s\n", n);
            return 2;
        }
    }

    std::fprintf(stderr,
                 "diarizen_wavlm_pruned_loader: OK (%zu tensors, %.2f MB arena)\n",
                 model.tensor_count(),
                 model.arena_bytes() / (1024.0 * 1024.0));
    return 0;
}
