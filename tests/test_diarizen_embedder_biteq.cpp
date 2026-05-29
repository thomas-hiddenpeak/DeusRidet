// @role: P2a bit-equality harness for the WeSpeaker ResNet34-LM embedder.
//        Reads a float32 waveform ([-1,1]) and an optional float32 mask, runs
//        DiarizenResnet34Embedder, and writes the 256-d embedding (or the
//        [T,80] fbank with --fbank) as raw float32. The Python driver
//        tools/diarizen_bit_eq_embedder.py compares against the pyannote
//        WeSpeakerResNet34 reference. Mechanical bit-equality, not a semantic
//        score.
#include "../src/orator/diarizen_resnet34_embedder.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using deusridet::orator::DiarizenResnet34Embedder;

static std::vector<float> read_bin(const char* path, int n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    size_t got = std::fread(v.data(), sizeof(float), n, f);
    std::fclose(f);
    if ((int)got != n) {
        std::fprintf(stderr, "%s short read %zu != %d\n", path, got, n);
        std::exit(1);
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <resnet34.safetensors> <wave.bin> <n_samples> "
                     "<mask.bin|none> <num_frames> <out.bin> [--fbank]\n",
                     argv[0]);
        return 2;
    }
    const char* st_path = argv[1];
    const char* wave_path = argv[2];
    const int n_samples = std::atoi(argv[3]);
    const char* mask_path = argv[4];
    const int num_frames = std::atoi(argv[5]);
    const char* out_path = argv[6];
    const bool fbank = (argc >= 8) && (std::string(argv[7]) == "--fbank");

    std::vector<float> wave = read_bin(wave_path, n_samples);
    std::vector<float> mask;
    const bool have_mask = std::string(mask_path) != "none";
    if (have_mask) mask = read_bin(mask_path, num_frames);

    DiarizenResnet34Embedder emb;
    if (!emb.load(st_path)) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }

    FILE* o = std::fopen(out_path, "wb");
    if (!o) {
        std::fprintf(stderr, "cannot open out %s\n", out_path);
        return 1;
    }

    if (fbank) {
        std::vector<float> fb;
        int T = 0;
        if (!emb.debug_fbank(wave.data(), n_samples, fb, T)) {
            std::fprintf(stderr, "fbank failed\n");
            return 1;
        }
        const std::int32_t hdr[2] = {T, 80};
        std::fwrite(hdr, sizeof(std::int32_t), 2, o);
        std::fwrite(fb.data(), sizeof(float), fb.size(), o);
        std::fprintf(stderr, "fbank T=%d\n", T);
    } else {
        std::vector<float> e(256);
        if (!emb.embed(wave.data(), n_samples, have_mask ? mask.data() : nullptr,
                       num_frames, e.data())) {
            std::fprintf(stderr, "embed failed\n");
            return 1;
        }
        std::fwrite(e.data(), sizeof(float), e.size(), o);
        std::fprintf(stderr, "embed dim=256\n");
    }
    std::fclose(o);
    std::fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
