// @role: P1b bit-equality harness for the DiariZen Conformer head. Reads a
//        raw float32 [T,256] feature map (the WavLM-pruned lnorm tail output),
//        runs DiarizenConformerHead::debug_{conformer,logits,probs}, writes the
//        result as raw float32. The Python driver
//        tools/diarizen_bit_eq_conformer.py feeds wavlm_lnorm_out from the
//        reference .npz and compares against the conformer_out /
//        classifier_logits / classifier_probs taps. Mechanical comparison only
//        (physical bit-equality, not a semantic quality score) per
//        workflow.instructions.md.
#include "../src/orator/diarizen_conformer_head.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using deusridet::orator::DiarizenConformerHead;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <safetensors> <feat.bin> <T> <C> <out.bin> "
                     "[--conformer|--logits|--probs]\n",
                     argv[0]);
        return 2;
    }
    const char* weights = argv[1];
    const char* feat_path = argv[2];
    const int T = std::atoi(argv[3]);
    const int C = std::atoi(argv[4]);
    const char* out_path = argv[5];
    std::string mode_flag = (argc >= 7) ? std::string(argv[6]) : "--conformer";

    if (C != 256) {
        std::fprintf(stderr, "expected C=256 feature dim, got %d\n", C);
        return 1;
    }

    std::vector<float> feat((size_t)T * C);
    FILE* f = std::fopen(feat_path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open feat %s\n", feat_path);
        return 1;
    }
    size_t got = std::fread(feat.data(), sizeof(float), feat.size(), f);
    std::fclose(f);
    if (got != feat.size()) {
        std::fprintf(stderr, "feat short read: got %zu expected %zu\n", got,
                     feat.size());
        return 1;
    }

    DiarizenConformerHead m;
    if (!m.load(weights)) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }

    std::vector<float> out;
    const char* mode;
    if (mode_flag == "--logits") {
        out = m.debug_logits(feat.data(), T);
        mode = "debug_logits";
    } else if (mode_flag == "--probs") {
        out = m.debug_probs(feat.data(), T);
        mode = "debug_probs";
    } else {
        out = m.debug_conformer(feat.data(), T);
        mode = "debug_conformer";
    }
    if (out.empty()) {
        std::fprintf(stderr, "%s failed\n", mode);
        return 1;
    }
    std::fprintf(stderr, "%s: T=%d out_C=%d (%zu floats)\n", mode, T,
                 (int)(out.size() / (T ? T : 1)), out.size());

    FILE* o = std::fopen(out_path, "wb");
    if (!o) {
        std::fprintf(stderr, "cannot open out %s\n", out_path);
        return 1;
    }
    std::fwrite(out.data(), sizeof(float), out.size(), o);
    std::fclose(o);
    std::fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
