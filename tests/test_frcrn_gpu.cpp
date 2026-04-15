// test_frcrn_gpu.cpp — Standalone test for FRCRN CUDA forward pass.
//
// Loads test PCM from file, runs GPU enhancement, validates output.
// Tests: weight loading, STFT/iSTFT, UNet forward, no NaN/Inf.
//
// Usage:
//   ./build/test_frcrn_gpu [input.raw] [output.raw]
//   Default input: /tmp/frcrn_test_input.raw (16000 float32 samples)

#include "src/sensus/auditus/frcrn_gpu.h"
#include "src/communis/log.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace deusridet;

int main(int argc, char** argv) {
    const char* input_path  = argc > 1 ? argv[1] : "/tmp/frcrn_test_input.raw";
    const char* output_path = argc > 2 ? argv[2] : "/tmp/frcrn_test_output.raw";
    const char* weights_dir = getenv("FRCRN_WEIGHTS");
    if (!weights_dir) {
        std::string home = getenv("HOME") ? getenv("HOME") : "/home/rm01";
        static std::string w = home + "/models/dev/vad/frcrn_weights";
        weights_dir = w.c_str();
    }

    // Load input PCM
    std::ifstream fin(input_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "Cannot open input: %s\n", input_path);
        return 1;
    }
    fin.seekg(0, std::ios::end);
    size_t file_bytes = fin.tellg();
    fin.seekg(0);
    int n_samples = file_bytes / sizeof(float);
    std::vector<float> pcm_in(n_samples);
    fin.read(reinterpret_cast<char*>(pcm_in.data()), file_bytes);
    fin.close();
    printf("Input: %d samples (%.3f s) from %s\n",
           n_samples, n_samples / 16000.0f, input_path);

    // Input stats
    float in_min = 1e9, in_max = -1e9, in_rms = 0;
    for (int i = 0; i < n_samples; i++) {
        in_min = std::min(in_min, pcm_in[i]);
        in_max = std::max(in_max, pcm_in[i]);
        in_rms += pcm_in[i] * pcm_in[i];
    }
    in_rms = sqrtf(in_rms / n_samples);
    printf("Input stats:  min=%.4f max=%.4f rms=%.4f\n", in_min, in_max, in_rms);

    // Initialize GPU
    printf("\n--- Initializing FrcrnGpu ---\n");
    FrcrnGpu gpu;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto t0 = std::chrono::steady_clock::now();
    bool ok = gpu.init(weights_dir, n_samples + 1000, stream);
    auto t1 = std::chrono::steady_clock::now();
    float init_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (!ok) {
        fprintf(stderr, "FrcrnGpu::init() failed!\n");
        return 1;
    }
    printf("Init OK: %.1f ms\n", init_ms);

    // Run enhance_host (handles H2D/D2H)
    std::vector<float> pcm_out(n_samples, 0.0f);

    printf("\n--- Running enhance_host ---\n");

    // Warmup
    int r = gpu.enhance_host(pcm_in.data(), pcm_out.data(), n_samples);
    printf("Warmup: %d samples returned, latency=%.1f ms\n", r, gpu.last_latency_ms());

    // Benchmark (5 runs)
    float total_ms = 0;
    int runs = 5;
    for (int i = 0; i < runs; i++) {
        std::fill(pcm_out.begin(), pcm_out.end(), 0.0f);
        r = gpu.enhance_host(pcm_in.data(), pcm_out.data(), n_samples);
        float lat = gpu.last_latency_ms();
        total_ms += lat;
        printf("  Run %d: %d samples, %.1f ms\n", i + 1, r, lat);
    }
    printf("Average latency: %.1f ms (%.1fx realtime for %.0f ms audio)\n",
           total_ms / runs,
           (n_samples / 16000.0f * 1000.0f) / (total_ms / runs),
           n_samples / 16.0f);

    // Validate output
    printf("\n--- Validation ---\n");
    int nan_count = 0, inf_count = 0, zero_count = 0;
    float out_min = 1e9, out_max = -1e9, out_rms = 0;
    for (int i = 0; i < n_samples; i++) {
        if (std::isnan(pcm_out[i])) nan_count++;
        if (std::isinf(pcm_out[i])) inf_count++;
        if (pcm_out[i] == 0.0f) zero_count++;
        out_min = std::min(out_min, pcm_out[i]);
        out_max = std::max(out_max, pcm_out[i]);
        out_rms += pcm_out[i] * pcm_out[i];
    }
    out_rms = sqrtf(out_rms / n_samples);

    printf("Output stats: min=%.4f max=%.4f rms=%.4f\n", out_min, out_max, out_rms);
    printf("NaN: %d  Inf: %d  Zero: %d/%d (%.1f%%)\n",
           nan_count, inf_count, zero_count, n_samples,
           100.0f * zero_count / n_samples);

    // Check for all-zero output (STFT/iSTFT pipeline broken)
    if (zero_count == n_samples) {
        printf("FAIL: Output is all zeros!\n");
    } else if (nan_count > 0) {
        printf("FAIL: Output contains NaN!\n");
    } else if (inf_count > 0) {
        printf("FAIL: Output contains Inf!\n");
    } else if (out_rms < 0.001f) {
        printf("WARN: Output RMS very low (%.6f) — possible near-zero output\n", out_rms);
    } else {
        printf("PASS: Output looks reasonable\n");
    }

    // Difference analysis
    float diff_rms = 0;
    for (int i = 0; i < n_samples; i++) {
        float d = pcm_out[i] - pcm_in[i];
        diff_rms += d * d;
    }
    diff_rms = sqrtf(diff_rms / n_samples);
    printf("In-Out diff RMS: %.6f\n", diff_rms);
    if (diff_rms < 1e-6f) {
        printf("WARN: Output ≈ Input (FRCRN may not be modifying signal)\n");
    }

    // Save output
    std::ofstream fout(output_path, std::ios::binary);
    fout.write(reinterpret_cast<char*>(pcm_out.data()), n_samples * sizeof(float));
    fout.close();
    printf("\nOutput saved: %s (%d samples)\n", output_path, n_samples);

    // Test smaller inputs (streaming chunks)
    printf("\n--- Streaming chunk test (100ms = 1600 samples) ---\n");
    int chunk = 1600;
    std::vector<float> chunk_in(chunk), chunk_out(chunk);
    std::memcpy(chunk_in.data(), pcm_in.data(), chunk * sizeof(float));

    r = gpu.enhance_host(chunk_in.data(), chunk_out.data(), chunk);
    printf("Chunk %d samples: returned %d, latency=%.1f ms\n",
           chunk, r, gpu.last_latency_ms());

    int chunk_nan = 0, chunk_zero = 0;
    float chunk_rms = 0;
    for (int i = 0; i < chunk; i++) {
        if (std::isnan(chunk_out[i])) chunk_nan++;
        if (chunk_out[i] == 0.0f) chunk_zero++;
        chunk_rms += chunk_out[i] * chunk_out[i];
    }
    chunk_rms = sqrtf(chunk_rms / chunk);
    printf("Chunk output: rms=%.4f nan=%d zero=%d/%d\n",
           chunk_rms, chunk_nan, chunk_zero, chunk);

    cudaStreamDestroy(stream);
    printf("\nDone.\n");
    return (nan_count > 0 || inf_count > 0 || zero_count == n_samples) ? 1 : 0;
}
