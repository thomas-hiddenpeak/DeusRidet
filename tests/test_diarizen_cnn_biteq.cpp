// @role: P1a-step2a bit-equality harness for the DiariZen WavLM-pruned CNN
//        feature extractor. Reads a raw float32 PCM window, runs
//        DiarizenWavlmPruned::debug_cnn_features, writes the [T,211] feature
//        map as raw float32. The Python driver tools/diarizen_bit_eq_p1a.py
//        feeds wave_in from the reference .npz and compares against the
//        cnn_out tap. Mechanical comparison only (physical bit-equality,
//        not a semantic quality score) per workflow.instructions.md.
#include "../src/orator/diarizen_wavlm_pruned.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using deusridet::orator::DiarizenWavlmPruned;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <safetensors> <pcm.bin> <n_samples> <out.bin> [--tap0]\n",
                     argv[0]);
        return 2;
    }
    const char* weights = argv[1];
    const char* pcm_path = argv[2];
    const int n_samples = std::atoi(argv[3]);
    const char* out_path = argv[4];
    const bool tap0 = (argc >= 6) && (std::string(argv[5]) == "--tap0");

    // Read raw float32 PCM.
    std::vector<float> pcm(n_samples);
    FILE* f = std::fopen(pcm_path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open pcm %s\n", pcm_path);
        return 1;
    }
    size_t got = std::fread(pcm.data(), sizeof(float), n_samples, f);
    std::fclose(f);
    if ((int)got != n_samples) {
        std::fprintf(stderr, "pcm short read: got %zu expected %d\n", got,
                     n_samples);
        return 1;
    }

    DiarizenWavlmPruned m;
    if (!m.load(weights)) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }

    int T = 0;
    std::vector<float> feats = tap0
                                  ? m.debug_tap0(pcm.data(), n_samples, T)
                                  : m.debug_cnn_features(pcm.data(), n_samples, T);
    if (feats.empty()) {
        std::fprintf(stderr, "%s failed\n",
                     tap0 ? "debug_tap0" : "debug_cnn_features");
        return 1;
    }
    std::fprintf(stderr, "%s: T=%d C=%d (%zu floats)\n",
                 tap0 ? "tap0" : "cnn features", T,
                 (int)(feats.size() / (T ? T : 1)), feats.size());

    FILE* o = std::fopen(out_path, "wb");
    if (!o) {
        std::fprintf(stderr, "cannot open out %s\n", out_path);
        return 1;
    }
    std::fwrite(feats.data(), sizeof(float), feats.size(), o);
    std::fclose(o);
    std::fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
