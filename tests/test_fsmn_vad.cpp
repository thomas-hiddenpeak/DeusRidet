// test_fsmn_vad.cpp — Validate native FSMN VAD FP32 against PyTorch reference.
//
// Uses pre-computed feature vectors (bypassing GPU Fbank + LFR + CMVN)
// to test the forward pass directly against PyTorch FP32 reference.

#include "../src/sensus/auditus/fsmn_vad.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Minimal .npy loader (float32 C-contiguous only)
static std::vector<float> load_npy(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return {}; }

    char magic[6];
    fread(magic, 1, 6, f);

    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);

    uint16_t header_len;
    fread(&header_len, 2, 1, f);

    fseek(f, 6 + 2 + 2 + header_len, SEEK_SET);

    long cur = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, cur, SEEK_SET);

    size_t n = (end - cur) / sizeof(float);
    std::vector<float> data(n);
    fread(data.data(), sizeof(float), n, f);
    fclose(f);
    return data;
}

int main() {
    using namespace deusridet;

    // Load reference data
    auto feat1  = load_npy("/tmp/fsmn_ref_feat.npy");        // (1, 1, 400)
    auto probs1 = load_npy("/tmp/fsmn_ref_fp32_probs.npy");  // (1, 248)
    auto feat2  = load_npy("/tmp/fsmn_ref_feat2.npy");       // (1, 1, 400)
    auto probs2 = load_npy("/tmp/fsmn_ref_fp32_probs2.npy"); // (1, 248)

    if (feat1.empty() || probs1.empty() || feat2.empty() || probs2.empty()) {
        fprintf(stderr, "Failed to load reference data from /tmp/fsmn_ref_*\n");
        return 1;
    }

    float ref_p1 = 1.0f - probs1[0];  // P(speech) = 1 - P(silence)
    float ref_p2 = 1.0f - probs2[0];
    printf("Reference: step1=%.6f, step2=%.6f\n", ref_p1, ref_p2);

    // Init FSMN VAD (we only need model weights, not GPU Fbank for this test)
    FsmnVadConfig cfg;
    cfg.model_path = "/home/rm01/models/dev/vad/fsmn/fsmn_vad.safetensors";
    cfg.cmvn_path  = "/home/rm01/models/dev/vad/fsmn/am.mvn";

    FsmnVad vad;
    if (!vad.init(cfg)) {
        fprintf(stderr, "Failed to init FsmnVad\n");
        return 1;
    }

    // Step 1: Feed raw feature vector directly (bypass Fbank/LFR/CMVN)
    // feat1 is (1, 1, 400) — extract the 400 features
    float native_p1 = vad.forward(feat1.data(), 1, 400);
    float err1 = fabsf(native_p1 - ref_p1);
    printf("\n=== Step 1 ===\n");
    printf("Native: %.6f  PyTorch: %.6f  err: %.6f\n", native_p1, ref_p1, err1);

    // Step 2: state carries from step 1
    float native_p2 = vad.forward(feat2.data(), 1, 400);
    float err2 = fabsf(native_p2 - ref_p2);
    printf("\n=== Step 2 ===\n");
    printf("Native: %.6f  PyTorch: %.6f  err: %.6f\n", native_p2, ref_p2, err2);

    constexpr float tol = 0.001f;
    if (err1 < tol && err2 < tol) {
        printf("\nPASS (tol=%.4f, err1=%.6f, err2=%.6f)\n", tol, err1, err2);
        return 0;
    } else {
        printf("\nFAIL (tol=%.4f, err1=%.6f, err2=%.6f)\n", tol, err1, err2);
        return 1;
    }
}
