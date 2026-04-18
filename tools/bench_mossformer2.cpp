// bench_mossformer2.cpp — Performance benchmark for native CUDA MossFormer2.
//
// Measures latency and throughput across different audio lengths.
// Compares against real-time factor (RTF) for streaming viability.
//
// Usage: ./build/bench_mossformer2 [max_samples]

#include "src/sensus/auditus/speech_separator.h"
#include "src/sensus/auditus/mossformer2.h"
#include "src/communis/log.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace deusridet;

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

// Benchmark raw MossFormer2 forward pass (GPU→GPU, no H↔D copies)
static void bench_raw(int max_samples) {
    printf("=== RAW MossFormer2 forward (GPU→GPU) ===\n");
    printf("%-12s %8s %8s %8s %8s %8s\n",
           "samples", "dur(ms)", "lat(ms)", "std(ms)", "RTF", "x-RT");

    MossFormer2 mf2;
    if (!mf2.init(std::string(getenv("HOME")) +
                  "/models/dev/vad/mossformer2_ss_16k.safetensors", max_samples)) {
        fprintf(stderr, "MossFormer2 init failed\n");
        return;
    }

    float *d_in, *d_s1, *d_s2;
    cudaMalloc(&d_in, max_samples * sizeof(float));
    cudaMalloc(&d_s1, max_samples * sizeof(float));
    cudaMalloc(&d_s2, max_samples * sizeof(float));
    cudaMemset(d_in, 0, max_samples * sizeof(float));

    // Generate a simple test signal
    std::vector<float> pcm(max_samples);
    for (int i = 0; i < max_samples; i++)
        pcm[i] = 0.3f * sinf(2.0f * M_PI * 440.0f * i / 16000.0f) +
                 0.2f * sinf(2.0f * M_PI * 880.0f * i / 16000.0f);
    cudaMemcpy(d_in, pcm.data(), max_samples * sizeof(float), cudaMemcpyHostToDevice);

    // Test various audio lengths
    int lengths[] = {1600, 3200, 4800, 8000, 16000, 24000, 32000, 48000, 64000};
    for (int len : lengths) {
        if (len > max_samples) break;

        float dur_ms = len / 16.0f;  // audio duration in ms

        // Warm-up runs
        for (int w = 0; w < 3; w++)
            mf2.forward(d_in, d_s1, d_s2, len);

        // Timed runs
        const int N = 10;
        double times[N];
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            mf2.forward(d_in, d_s1, d_s2, len);
            times[i] = now_ms() - t0;
        }

        // Stats
        double sum = 0, sum2 = 0;
        for (int i = 0; i < N; i++) { sum += times[i]; sum2 += times[i] * times[i]; }
        double mean = sum / N;
        double std_dev = sqrt(sum2 / N - mean * mean);
        double rtf = mean / dur_ms;
        double xrt = dur_ms / mean;

        printf("%-12d %8.1f %8.2f %8.2f %8.4f %8.2f\n",
               len, dur_ms, mean, std_dev, rtf, xrt);
    }

    cudaFree(d_in);
    cudaFree(d_s1);
    cudaFree(d_s2);
}

// Benchmark SpeechSeparator (full pipeline: H→D, inference, D→H, segmentation)
static void bench_separator(int max_samples) {
    printf("\n=== SpeechSeparator pipeline (H→D + inference + D→H) ===\n");
    printf("%-12s %8s %8s %8s %8s %8s\n",
           "samples", "dur(ms)", "lat(ms)", "std(ms)", "RTF", "x-RT");

    SpeechSeparatorConfig cfg;
    cfg.model_path = std::string(getenv("HOME")) +
                     "/models/dev/vad/mossformer2_ss_16k.safetensors";
    cfg.max_chunk = std::min(max_samples, 32000);
    cfg.lazy_load = false;

    SpeechSeparator sep;
    if (!sep.init(cfg)) {
        fprintf(stderr, "SpeechSeparator init failed\n");
        return;
    }

    // Generate test signal
    std::vector<float> pcm(max_samples);
    for (int i = 0; i < max_samples; i++)
        pcm[i] = 0.3f * sinf(2.0f * M_PI * 300.0f * i / 16000.0f) +
                 0.2f * sinf(2.0f * M_PI * 800.0f * i / 16000.0f);

    int lengths[] = {1600, 3200, 4800, 8000, 16000, 24000, 32000, 48000, 64000};
    for (int len : lengths) {
        if (len > max_samples) break;

        float dur_ms = len / 16.0f;

        // Warm-up
        for (int w = 0; w < 2; w++)
            sep.separate(pcm.data(), len);

        // Timed runs
        const int N = 5;
        double times[N];
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            auto result = sep.separate(pcm.data(), len);
            times[i] = now_ms() - t0;
            if (!result.valid) {
                fprintf(stderr, "Separation failed at len=%d\n", len);
                return;
            }
        }

        double sum = 0, sum2 = 0;
        for (int i = 0; i < N; i++) { sum += times[i]; sum2 += times[i] * times[i]; }
        double mean = sum / N;
        double std_dev = sqrt(sum2 / N - mean * mean);
        double rtf = mean / dur_ms;
        double xrt = dur_ms / mean;

        printf("%-12d %8.1f %8.2f %8.2f %8.4f %8.2f\n",
               len, dur_ms, mean, std_dev, rtf, xrt);
    }
}

// Memory usage report
static void report_memory() {
    printf("\n=== GPU Memory ===\n");
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("  Total: %.1f MB\n", total_bytes / 1048576.0);
    printf("  Free:  %.1f MB\n", free_bytes / 1048576.0);
    printf("  Used:  %.1f MB\n", (total_bytes - free_bytes) / 1048576.0);

    // Read system memory for Tegra unified memory
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemTotal:", 9) == 0 ||
                strncmp(line, "MemAvailable:", 13) == 0) {
                printf("  %s", line);
            }
        }
        fclose(f);
    }
}

int main(int argc, char** argv) {
    int max_samples = 64000;  // 4s default
    if (argc > 1) max_samples = atoi(argv[1]);
    printf("MossFormer2 Performance Benchmark\n");
    printf("Max samples: %d (%.2fs @ 16kHz)\n\n", max_samples, max_samples / 16000.0f);

    report_memory();
    printf("\n");

    bench_raw(max_samples);
    bench_separator(max_samples);

    report_memory();
    printf("\nDone.\n");
    return 0;
}
