// @role: P3a get_embeddings bit-eq harness. Reads a 16 kHz mono waveform and
//   the binarized segmentation [C*F*S] from raw float32 files, runs the native
//   DiarizenPipeline::debug_get_embeddings, and writes the embeddings
//   [C*S*256] to a raw float32 file for the Python driver to compare against
//   the diarizen_dump_pipeline fixture.
//
// Usage:
//   test_diarizen_pipeline_biteq <wave.bin> <seg.bin> <C> <F> <S> <out.bin>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "orator/diarizen_pipeline.h"

namespace {
bool read_f32(const std::string& path, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(static_cast<std::size_t>(bytes) / sizeof(float));
    const std::size_t n = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    return n == out.size();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s <wave.bin> <seg.bin> <C> <F> <S> <out.bin>\n",
                     argv[0]);
        return 2;
    }
    const std::string wave_path = argv[1];
    const std::string seg_path  = argv[2];
    const int C = std::atoi(argv[3]);
    const int F = std::atoi(argv[4]);
    const int S = std::atoi(argv[5]);
    const std::string out_path = argv[6];

    std::vector<float> wave, seg;
    if (!read_f32(wave_path, wave) || !read_f32(seg_path, seg)) return 1;
    const std::size_t want_seg = static_cast<std::size_t>(C) * F * S;
    if (seg.size() != want_seg) {
        std::fprintf(stderr, "seg size mismatch: got %zu want %zu\n",
                     seg.size(), want_seg);
        return 1;
    }

    deusridet::orator::DiarizenPipeline pipe;
    deusridet::orator::DiarizenPipelineConfig cfg;
    if (!pipe.load(cfg)) {
        std::fprintf(stderr, "load failed: %s\n", pipe.last_error().c_str());
        return 1;
    }

    std::vector<float> emb;
    if (!pipe.debug_get_embeddings(wave.data(),
                                   static_cast<int>(wave.size()), seg.data(), C,
                                   F, S, emb)) {
        std::fprintf(stderr, "get_embeddings failed: %s\n",
                     pipe.last_error().c_str());
        return 1;
    }

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fwrite(emb.data(), sizeof(float), emb.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "wrote %zu floats (C=%d S=%d D=256) to %s\n",
                 emb.size(), C, S, out_path.c_str());
    return 0;
}
