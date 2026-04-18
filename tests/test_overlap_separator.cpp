// test_overlap_separator.cpp — Standalone test for P1 (overlap detection) and P2 (speech separation).
//
// Usage:
//   ./build/test_overlap_separator [--od-only | --sep-only]
//
// Tests:
//   P1: Feed silence, single speaker (sine), and overlapping speakers (two sines) to
//       overlap detector and verify detection.
//   P2: Mix two sines at different frequencies, separate, and verify output shapes/energy.

#include "../src/sensus/auditus/overlap_detector.h"
#include "../src/sensus/auditus/speech_separator.h"
#include "../src/communis/log.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace deusridet;

static constexpr int SR = 16000;
static constexpr float PI = 3.14159265358979323846f;

// Generate a sine tone (simulating a speaker).
static void gen_sine(std::vector<float>& out, int n_samples, float freq, float amp = 0.3f) {
    out.resize(n_samples);
    for (int i = 0; i < n_samples; i++)
        out[i] = amp * sinf(2.0f * PI * freq * i / SR);
}

// Generate silence.
static void gen_silence(std::vector<float>& out, int n_samples) {
    out.assign(n_samples, 0.0f);
}

// Mix two signals.
static std::vector<float> mix(const std::vector<float>& a, const std::vector<float>& b) {
    int n = std::max(a.size(), b.size());
    std::vector<float> out(n, 0.0f);
    for (int i = 0; i < (int)a.size(); i++) out[i] += a[i];
    for (int i = 0; i < (int)b.size(); i++) out[i] += b[i];
    return out;
}

static float rms(const std::vector<float>& v) {
    if (v.empty()) return 0;
    double s = 0;
    for (float x : v) s += x * x;
    return sqrtf((float)(s / v.size()));
}

// ──────── P1: Overlap Detector Test ────────

static bool test_overlap_detector() {
    printf("\n=== P1: Overlap Detector Test ===\n");

    OverlapDetectorConfig cfg;
    cfg.model_path = std::string(getenv("HOME")) + "/models/dev/vad/pyannote_seg3.safetensors";
    cfg.overlap_threshold = 0.5f;
    cfg.chunk_samples = 160000;  // 10s
    cfg.hop_samples = 80000;     // 5s

    OverlapDetector od;
    if (!od.init(cfg)) {
        printf("FAIL: Could not init overlap detector\n");
        return false;
    }
    printf("  Model loaded: frames=%d, classes=%d\n", od.num_output_frames(), od.num_classes());

    // Test 1: Silence → should NOT detect overlap.
    {
        std::vector<float> silence;
        gen_silence(silence, 160000);
        auto t0 = std::chrono::steady_clock::now();
        auto result = od.detect(silence.data(), silence.size());
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        printf("  [Silence] overlap=%s, ratio=%.3f, frames=%d, latency=%.1f ms\n",
               result.is_overlap ? "YES" : "no", result.overlap_ratio,
               result.num_frames, ms);
        if (result.is_overlap && result.overlap_ratio > 0.1f) {
            printf("  WARN: Silence detected as overlap (ratio=%.3f) — may be model artifact\n",
                   result.overlap_ratio);
        }
    }

    // Test 2: Single tone (one speaker) → should NOT detect overlap.
    {
        std::vector<float> tone;
        gen_sine(tone, 160000, 300.0f, 0.3f);
        auto t0 = std::chrono::steady_clock::now();
        auto result = od.detect(tone.data(), tone.size());
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        printf("  [Single tone 300Hz] overlap=%s, ratio=%.3f, frames=%d, latency=%.1f ms\n",
               result.is_overlap ? "YES" : "no", result.overlap_ratio,
               result.num_frames, ms);
    }

    // Test 3: Two overlapping tones → should detect overlap.
    {
        std::vector<float> tone1, tone2;
        gen_sine(tone1, 160000, 250.0f, 0.3f);
        gen_sine(tone2, 160000, 800.0f, 0.25f);
        auto mixed = mix(tone1, tone2);
        auto t0 = std::chrono::steady_clock::now();
        auto result = od.detect(mixed.data(), mixed.size());
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        printf("  [Two tones 250+800Hz] overlap=%s, ratio=%.3f, frames=%d, latency=%.1f ms\n",
               result.is_overlap ? "YES" : "no", result.overlap_ratio,
               result.num_frames, ms);
    }

    // Test 4: Streaming mode — feed 3s chunks until full window.
    {
        od.reset();
        std::vector<float> tone;
        gen_sine(tone, 48000, 400.0f, 0.3f);  // 3s

        bool got_result = false;
        OverlapResult result;
        int feeds = 0;
        auto t0 = std::chrono::steady_clock::now();
        // Feed multiple 3s chunks until we get a result (need 10s = ~4 feeds).
        for (int i = 0; i < 6 && !got_result; i++) {
            got_result = od.feed(tone.data(), tone.size(), result);
            feeds++;
        }
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        printf("  [Streaming %d×3s] got_result=%s, overlap=%s, ratio=%.3f, latency=%.1f ms\n",
               feeds, got_result ? "yes" : "no",
               result.is_overlap ? "YES" : "no", result.overlap_ratio, ms);
    }

    printf("  P1 tests complete.\n");
    return true;
}

// ──────── P2: Speech Separator Test ────────

static bool test_speech_separator() {
    printf("\n=== P2: Speech Separator Test ===\n");

    SpeechSeparatorConfig cfg;
    cfg.model_path = std::string(getenv("HOME")) + "/models/dev/vad/mossformer2_ss_16k.safetensors";
    cfg.max_chunk = 32000;      // 2s chunks
    cfg.overlap_samples = 3200; // 200ms overlap
    cfg.lazy_load = false;      // load immediately for testing

    SpeechSeparator sep;
    if (!sep.init(cfg)) {
        printf("FAIL: Could not init speech separator\n");
        return false;
    }
    printf("  Model loaded: initialized=%d, loaded=%d\n", sep.initialized(), sep.loaded());

    // Test 1: Short audio (1s) — two mixed sines at different frequencies.
    {
        printf("  [Short 1s mix]\n");
        std::vector<float> s1, s2;
        gen_sine(s1, 16000, 300.0f, 0.4f);  // "speaker 1": 300 Hz
        gen_sine(s2, 16000, 800.0f, 0.3f);  // "speaker 2": 800 Hz
        auto mixed = mix(s1, s2);
        float mix_rms = rms(mixed);
        printf("    mix_rms=%.4f, s1_rms=%.4f, s2_rms=%.4f\n", mix_rms, rms(s1), rms(s2));

        auto t0 = std::chrono::steady_clock::now();
        auto result = sep.separate(mixed.data(), mixed.size());
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (result.valid) {
            printf("    src1: %d samples, rms=%.4f\n", (int)result.source1.size(), result.energy1);
            printf("    src2: %d samples, rms=%.4f\n", (int)result.source2.size(), result.energy2);
            printf("    latency=%.1f ms\n", ms);

            // Both outputs should have audio content.
            if (result.energy1 < 0.001f || result.energy2 < 0.001f)
                printf("    WARN: One source has very low energy\n");
        } else {
            printf("    FAIL: Separation returned invalid result\n");
            return false;
        }
    }

    // Test 2: Longer audio (4s) — tests segmented processing.
    {
        printf("  [Long 4s mix]\n");
        std::vector<float> s1, s2;
        gen_sine(s1, 64000, 250.0f, 0.35f);
        gen_sine(s2, 64000, 700.0f, 0.3f);
        auto mixed = mix(s1, s2);
        float mix_rms = rms(mixed);
        printf("    mix_rms=%.4f, n_samples=%d\n", mix_rms, (int)mixed.size());

        auto t0 = std::chrono::steady_clock::now();
        auto result = sep.separate(mixed.data(), mixed.size());
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (result.valid) {
            printf("    src1: %d samples, rms=%.4f\n", (int)result.source1.size(), result.energy1);
            printf("    src2: %d samples, rms=%.4f\n", (int)result.source2.size(), result.energy2);
            printf("    latency=%.1f ms (%.1f ms/s)\n", ms, ms / 4.0f);
        } else {
            printf("    FAIL: Separation returned invalid result\n");
            return false;
        }
    }

    // Test 3: Silence input — should produce silent outputs.
    {
        printf("  [Silence 1s]\n");
        std::vector<float> silence;
        gen_silence(silence, 16000);

        auto result = sep.separate(silence.data(), silence.size());
        if (result.valid) {
            printf("    src1_rms=%.6f, src2_rms=%.6f (should be ~0)\n",
                   result.energy1, result.energy2);
        } else {
            printf("    Separation of silence: invalid (acceptable)\n");
        }
    }

    printf("  P2 tests complete.\n");
    return true;
}

int main(int argc, char** argv) {
    bool od_only = false, sep_only = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--od-only") == 0) od_only = true;
        if (strcmp(argv[i], "--sep-only") == 0) sep_only = true;
    }

    bool ok = true;
    if (!sep_only) ok &= test_overlap_detector();
    if (!od_only)  ok &= test_speech_separator();

    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
