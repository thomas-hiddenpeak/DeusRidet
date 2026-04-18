// test_pyannote_seg3.cpp — Validate native CUDA PyannoteSeg3 against ONNX reference.
//
// Uses /tmp/seg3_ref/*.npy for layer-by-layer comparison.
// Build: make test_pyannote_seg3
// Run: ./test_pyannote_seg3

#include "../src/sensus/auditus/pyannote_seg3.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

// Load a .npy file (float32 only, little-endian, C-contiguous)
static bool load_npy(const char* path, std::vector<int>& shape, std::vector<float>& data) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || magic[0] != '\x93' || memcmp(magic + 1, "NUMPY", 5) != 0) {
        fclose(f); return false;
    }

    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);

    uint32_t header_len;
    if (major == 1) {
        uint16_t hl;
        fread(&hl, 2, 1, f);
        header_len = hl;
    } else {
        fread(&header_len, 4, 1, f);
    }

    std::string header(header_len, '\0');
    fread(&header[0], 1, header_len, f);

    // Parse shape
    auto sp = header.find("'shape': (");
    if (sp == std::string::npos) { fclose(f); return false; }
    sp += 10;
    auto se = header.find(')', sp);
    std::string ss = header.substr(sp, se - sp);

    shape.clear();
    size_t pos = 0;
    while (pos < ss.size()) {
        while (pos < ss.size() && (ss[pos] == ' ' || ss[pos] == ',')) pos++;
        if (pos >= ss.size()) break;
        shape.push_back(std::stoi(ss.substr(pos)));
        while (pos < ss.size() && ss[pos] != ',' && ss[pos] != ')') pos++;
    }

    int total = 1;
    for (int s : shape) total *= s;
    data.resize(total);
    fread(data.data(), sizeof(float), total, f);
    fclose(f);
    return true;
}

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

static float max_err(const float* a, const float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main() {
    printf("=== PyannoteSeg3 Native CUDA Validation ===\n\n");

    // Load input and reference logits
    std::vector<int> in_shape, out_shape;
    std::vector<float> in_data, ref_logits;
    if (!load_npy("/tmp/seg3_ref/input.npy", in_shape, in_data)) {
        printf("ERROR: Cannot load /tmp/seg3_ref/input.npy\n");
        return 1;
    }
    if (!load_npy("/tmp/seg3_ref/logits.npy", out_shape, ref_logits)) {
        printf("ERROR: Cannot load /tmp/seg3_ref/logits.npy\n");
        return 1;
    }

    printf("Input: %d samples\n", (int)in_data.size());
    printf("Expected output: (%d, %d)\n\n", out_shape[1], out_shape[2]);

    // Init model
    std::string model_path = std::string(getenv("HOME"))
                             + "/models/dev/vad/pyannote_seg3.safetensors";
    deusridet::PyannoteSeg3 model;
    if (!model.init(model_path)) {
        printf("ERROR: Failed to init model\n");
        return 1;
    }

    // Upload input to GPU
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, 160000 * sizeof(float));
    cudaMalloc(&d_output, 589 * 7 * sizeof(float));
    cudaMemcpy(d_input, in_data.data(), 160000 * sizeof(float),
               cudaMemcpyHostToDevice);

    // Run forward
    int num_frames = model.forward(d_input, d_output, 160000);
    printf("Forward: %d frames, latency=%.2f ms\n\n", num_frames, model.last_latency_ms());

    // Download output
    std::vector<float> h_output(589 * 7);
    cudaMemcpy(h_output.data(), d_output, 589 * 7 * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compare with reference
    int ref_total = out_shape[1] * out_shape[2];
    float cos = cosine_sim(h_output.data(), ref_logits.data(), ref_total);
    float me = max_err(h_output.data(), ref_logits.data(), ref_total);

    printf("=== Final Output Comparison ===\n");
    printf("  logits (%d, %d):  cos=%.8f  max_err=%.6f\n",
           out_shape[1], out_shape[2], cos, me);

    // Print first/last few values
    printf("\n  CUDA first 5 frames:\n");
    for (int f = 0; f < 5; f++) {
        printf("    [%d]:", f);
        for (int c = 0; c < 7; c++) printf(" %.4f", h_output[f * 7 + c]);
        printf("\n");
    }
    printf("  Ref first 5 frames:\n");
    for (int f = 0; f < 5; f++) {
        printf("    [%d]:", f);
        for (int c = 0; c < 7; c++) printf(" %.4f", ref_logits[f * 7 + c]);
        printf("\n");
    }

    bool pass = cos > 0.999f;
    printf("\n=== %s === (cos=%.6f, threshold=0.999)\n",
           pass ? "VALIDATION PASSED" : "VALIDATION FAILED", cos);

    cudaFree(d_input);
    cudaFree(d_output);
    return pass ? 0 : 1;
}
