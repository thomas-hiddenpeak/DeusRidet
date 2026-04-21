/**
 * @file cmd_test_wavlm_cnn.cpp
 * @philosophical_role External command `cmd_test_wavlm_cnn`. An Actus function — one CLI verb, one finite
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

int cmd_test_wavlm_cnn() {
    printf("[test-wavlm-cnn] WavLM CNN Feature Extractor validation\n");

    // Paths (workspace-local by default; override with DEUSRIDET_MODEL_ROOT)
    std::string model_root = getenv("DEUSRIDET_MODEL_ROOT")
                             ? getenv("DEUSRIDET_MODEL_ROOT")
                             : "/home/rm01/DeusRidet/models/dev";
    std::string model_path = model_root + "/speaker/espnet_wavlm_ecapa/wavlm_ecapa.safetensors";
    std::string ref_dir    = model_root + "/speaker/espnet_wavlm_ecapa/ref_dump/";

    // Helper: load reference binary tensor
    auto load_ref = [&](const char* name) -> std::vector<float> {
        std::string path = ref_dir + name;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "  ERROR: cannot open %s\n", path.c_str());
            return {};
        }
        fseek(f, 0, SEEK_END);
        size_t bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<float> data(bytes / sizeof(float));
        size_t r_ = fread(data.data(), 1, bytes, f);
        (void)r_;
        fclose(f);
        return data;
    };

    // Helper: compare GPU buffer vs CPU reference
    auto compare = [](const float* d_gpu, const std::vector<float>& ref,
                      const char* label) -> float {
        std::vector<float> gpu(ref.size());
        cudaMemcpy(gpu.data(), d_gpu, ref.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        float max_diff = 0;
        float max_val = 0;
        int max_idx = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            float diff = fabsf(gpu[i] - ref[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_val = ref[i];
                max_idx = (int)i;
            }
        }
        printf("  %-40s max_diff=%.6f (at [%d], ref=%.6f, gpu=%.6f)\n",
               label, max_diff, max_idx, max_val,
               max_idx < (int)gpu.size() ? gpu[max_idx] : 0.0f);
        return max_diff;
    };

    // 1. Init encoder
    WavLMEcapaEncoder enc;
    if (!enc.init(model_path)) {
        fprintf(stderr, "[test-wavlm-cnn] Failed to init encoder\n");
        return 1;
    }

    // 2. Load reference input
    auto ref_wav = load_ref("input_wav.bin");
    if (ref_wav.empty()) return 1;
    int n_samples = (int)ref_wav.size();
    printf("  Input: %d samples (%.2f sec)\n", n_samples, n_samples / 16000.0f);

    // Upload to GPU
    float* d_wav = nullptr;
    cudaMalloc(&d_wav, n_samples * sizeof(float));
    cudaMemcpy(d_wav, ref_wav.data(), n_samples * sizeof(float),
               cudaMemcpyHostToDevice);

    // 3. Run CNN
    int T_out = 0;
    float* d_cnn_out = enc.test_cnn(d_wav, n_samples, T_out);
    printf("  CNN output: T'=%d (512 channels)\n", T_out);

    // 4. Compare against reference tensors
    printf("\n  --- CNN layer comparisons ---\n");
    // We compare the final CNN output (layer6_out)
    auto ref_cnn6 = load_ref("wavlm_cnn_layer6_out.bin");
    if (!ref_cnn6.empty()) {
        compare(d_cnn_out, ref_cnn6, "wavlm/cnn/layer6_out");
    }

    // Also check input normalization
    auto ref_norm = load_ref("wavlm_input_norm.bin");
    if (!ref_norm.empty()) {
        printf("  (input normalization checked via CNN output diff)\n");
    }

    // 5. Test feature projection: LN(512) + Linear(512→1024)
    printf("\n  --- Feature Projection ---\n");
    int T_proj = 0;
    float* d_proj = enc.test_projection(d_cnn_out, T_out, T_proj);
    printf("  Projection output: [%d, 1024]\n", T_proj);

    auto ref_pre_ln = load_ref("wavlm_features_pre_ln.bin");
    // ref_pre_ln is [1, T', 512] → compare transposed CNN output (via projection intermediate)
    // We can't easily compare intermediates, but post_extract_proj is the final output

    auto ref_post_proj = load_ref("wavlm_post_extract_proj.bin");
    if (!ref_post_proj.empty()) {
        compare(d_proj, ref_post_proj, "wavlm/post_extract_proj");
    }

    // 6. Test positional convolution
    printf("\n  --- Positional Conv ---\n");
    int T_pos = 0;
    float* d_pos = enc.test_pos_conv(d_proj, T_proj, T_pos);
    printf("  Pos conv output: [%d, 1024]\n", T_pos);

    // d_pos is after residual add (input + GELU(conv))
    auto ref_after_pos = load_ref("wavlm_after_pos_add.bin");
    if (!ref_after_pos.empty()) {
        compare(d_pos, ref_after_pos, "wavlm/after_pos_add");
    }

    // 7. Test transformer encoder (24 layers + final LN)
    printf("\n  --- Transformer Encoder ---\n");

    // First check position bias
    auto ref_pos_bias = load_ref("wavlm_encoder_rel_pos_bias.bin");

    int T_enc = 0;
    float* d_enc = enc.test_encoder(d_pos, T_pos, T_enc);
    printf("  Encoder output: [%d, 1024]\n", T_enc);

    // Compare position bias
    if (!ref_pos_bias.empty()) {
        const float* gpu_pb = enc.get_pos_bias();
        if (gpu_pb) {
            printf("  ref_pos_bias size=%zu, first5: %.4f %.4f %.4f %.4f %.4f\n",
                   ref_pos_bias.size(),
                   ref_pos_bias[0], ref_pos_bias[1], ref_pos_bias[2],
                   ref_pos_bias[3], ref_pos_bias[4]);
            std::vector<float> gpu_pb_h(std::min(ref_pos_bias.size(), (size_t)10));
            cudaMemcpy(gpu_pb_h.data(), gpu_pb, gpu_pb_h.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            printf("  gpu_pos_bias first5: %.4f %.4f %.4f %.4f %.4f\n",
                   gpu_pb_h[0], gpu_pb_h[1], gpu_pb_h[2],
                   gpu_pb_h[3], gpu_pb_h[4]);
            compare(gpu_pb, ref_pos_bias, "rel_pos_bias");
        }
    }

    // Compare individual layer outputs
    for (int i = 0; i < 24; i++) {
        char ref_name[64];
        snprintf(ref_name, sizeof(ref_name), "wavlm_encoder_layer%d_out.bin", i);
        auto ref_layer = load_ref(ref_name);
        if (!ref_layer.empty()) {
            const float* hs = enc.get_hidden_state(i + 1);  // layer i output = hidden_state[i+1]
            if (hs) {
                char label[64];
                snprintf(label, sizeof(label), "wavlm/encoder/layer%d_out", i);
                float md = compare(hs, ref_layer, label);
                if (md > 0.01f && i < 3) {
                    // Print first few values for debugging
                    std::vector<float> gpu(10);
                    cudaMemcpy(gpu.data(), hs, 10 * sizeof(float), cudaMemcpyDeviceToHost);
                    printf("    gpu[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
                           gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]);
                    printf("    ref[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
                           ref_layer[0], ref_layer[1], ref_layer[2], ref_layer[3], ref_layer[4]);
                }
            }
        }
    }

    // Compare final layer norm output
    auto ref_final_ln = load_ref("wavlm_encoder_final_ln.bin");
    if (!ref_final_ln.empty()) {
        // Final LN output is stored in hidden_states[24]
        const float* hs24 = enc.get_hidden_state(24);
        if (hs24) compare(hs24, ref_final_ln, "wavlm/encoder/final_ln");
    }

    // ======================================================================
    // 8. Featurizer + MVN + ECAPA + Pooling + Projector (end-to-end)
    // ======================================================================
    printf("\n  --- Full Extract (end-to-end) ---\n");
    auto result = enc.extract_gpu(d_wav, n_samples);
    printf("  Embedding dim: %d\n", (int)result.size());

    // Compare intermediate stages
    auto ref_feat = load_ref("featurizer_output.bin");
    if (!ref_feat.empty()) {
        printf("  featurizer ref: %zu floats\n", ref_feat.size());
    }

    auto ref_mvn = load_ref("normalize_output.bin");
    if (!ref_mvn.empty()) {
        printf("  MVN ref: %zu floats\n", ref_mvn.size());
    }

    // Compare final embedding
    auto ref_emb = load_ref("output_embedding.bin");
    if (!ref_emb.empty()) {
        printf("  ref embedding dim=%zu\n", ref_emb.size());
        float max_diff = 0;
        int max_idx = 0;
        for (size_t i = 0; i < std::min(result.size(), ref_emb.size()); i++) {
            float diff = fabsf(result[i] - ref_emb[i]);
            if (diff > max_diff) { max_diff = diff; max_idx = (int)i; }
        }
        printf("  %-40s max_diff=%.6f (at [%d], ref=%.6f, gpu=%.6f)\n",
               "output_embedding", max_diff, max_idx,
               max_idx < (int)ref_emb.size() ? ref_emb[max_idx] : 0.0f,
               max_idx < (int)result.size() ? result[max_idx] : 0.0f);
        // Print first 8 values
        printf("  gpu[0..7]:");
        for (int i = 0; i < std::min(8, (int)result.size()); i++)
            printf(" %.6f", result[i]);
        printf("\n  ref[0..7]:");
        for (int i = 0; i < std::min(8, (int)ref_emb.size()); i++)
            printf(" %.6f", ref_emb[i]);
        printf("\n");
    }

    cudaFree(d_wav);
    printf("\n[test-wavlm-cnn] Done.\n");
    return 0;
}

} // namespace deusridet

#include "communis/tegra.h"
int main(int /*argc*/, char** /*argv*/) {
    int rc = deusridet::cmd_test_wavlm_cnn();
    deusridet::tegra_cleanup();
    return rc;
}
