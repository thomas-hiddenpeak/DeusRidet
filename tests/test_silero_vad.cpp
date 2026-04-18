// test_silero_vad.cpp — Validate native Silero VAD against ONNX reference data.
//
// Reference arrays generated with zero initial context (matching native model).

#include "../src/sensus/auditus/silero_vad.h"
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

    auto pcm = load_npy("/tmp/silero_ref/pcm_512.npy");            // (512,)
    auto ref_out1 = load_npy("/tmp/silero_ref/output_v2.npy");     // (1, 1)
    auto ref_out2 = load_npy("/tmp/silero_ref/output_v2_step2.npy");

    if (pcm.empty() || ref_out1.empty()) {
        fprintf(stderr, "Failed to load reference data\n");
        return 1;
    }

    printf("Reference: step1=%.6f, step2=%.6f\n", ref_out1[0], ref_out2[0]);

    SileroVadConfig cfg;
    cfg.model_path = "/home/rm01/models/dev/vad/silero_vad.safetensors";
    cfg.sample_rate = 16000;
    cfg.window_samples = 512;

    SileroVad vad;
    if (!vad.init(cfg)) {
        fprintf(stderr, "Failed to init SileroVad\n");
        return 1;
    }

    // Step 1: context=zeros (default), pcm=512 samples
    SileroVadResult res1 = vad.process(pcm.data(), 512);
    printf("\n=== Step 1 ===\n");
    printf("Native: %.6f  ONNX: %.6f  err: %.6f\n",
           res1.probability, ref_out1[0],
           fabsf(res1.probability - ref_out1[0]));

    // Step 2: context updated from step 1, same pcm
    SileroVadResult res2 = vad.process(pcm.data(), 512);
    printf("\n=== Step 2 ===\n");
    printf("Native: %.6f  ONNX: %.6f  err: %.6f\n",
           res2.probability, ref_out2[0],
           fabsf(res2.probability - ref_out2[0]));

    float err1 = fabsf(res1.probability - ref_out1[0]);
    float err2 = fabsf(res2.probability - ref_out2[0]);
    float tol = 0.001f;
    bool pass = err1 < tol && err2 < tol;
    printf("\n%s (tol=%.4f, err1=%.6f, err2=%.6f)\n",
           pass ? "PASS" : "FAIL", tol, err1, err2);

    return pass ? 0 : 1;
}
