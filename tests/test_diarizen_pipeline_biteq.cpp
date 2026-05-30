// @role: P3a bit-eq harness. Two modes:
//   embeddings <wave.bin> <seg.bin> <C> <F> <S> <out.bin>
//     run get_embeddings, write embeddings [C*S*256].
//   postproc <seg.bin> <hard.bin int32> <C> <F> <S> <count_out.bin> <bin_out.bin>
//     run reconstruct+count+to_diarization, write count [nf] + binary [nf*ncl].
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

bool read_i32(const std::string& path, std::vector<int>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(static_cast<std::size_t>(bytes) / sizeof(int));
    const std::size_t n = std::fread(out.data(), sizeof(int), out.size(), f);
    std::fclose(f);
    return n == out.size();
}

bool write_f32(const std::string& path, const std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return true;
}

int run_embeddings(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: %s embeddings <wave.bin> <seg.bin> <C> <F> <S> "
                     "<out.bin>\n",
                     argv[0]);
        return 2;
    }
    std::vector<float> wave, seg;
    if (!read_f32(argv[2], wave) || !read_f32(argv[3], seg)) return 1;
    const int C = std::atoi(argv[4]);
    const int F = std::atoi(argv[5]);
    const int S = std::atoi(argv[6]);
    if (seg.size() != static_cast<std::size_t>(C) * F * S) {
        std::fprintf(stderr, "seg size mismatch\n");
        return 1;
    }
    deusridet::orator::DiarizenPipeline pipe;
    deusridet::orator::DiarizenPipelineConfig cfg;
    if (!pipe.load(cfg)) {
        std::fprintf(stderr, "load failed: %s\n", pipe.last_error().c_str());
        return 1;
    }
    std::vector<float> emb;
    if (!pipe.debug_get_embeddings(wave.data(), static_cast<int>(wave.size()),
                                   seg.data(), C, F, S, emb)) {
        std::fprintf(stderr, "get_embeddings failed: %s\n",
                     pipe.last_error().c_str());
        return 1;
    }
    if (!write_f32(argv[7], emb)) return 1;
    std::fprintf(stderr, "wrote %zu floats (C=%d S=%d D=256)\n", emb.size(), C,
                 S);
    return 0;
}

int run_postproc(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(stderr,
                     "usage: %s postproc <seg.bin> <hard.bin> <C> <F> <S> "
                     "<count_out.bin> <bin_out.bin>\n",
                     argv[0]);
        return 2;
    }
    std::vector<float> seg;
    std::vector<int> hard;
    if (!read_f32(argv[2], seg) || !read_i32(argv[3], hard)) return 1;
    const int C = std::atoi(argv[4]);
    const int F = std::atoi(argv[5]);
    const int S = std::atoi(argv[6]);
    if (seg.size() != static_cast<std::size_t>(C) * F * S ||
        hard.size() != static_cast<std::size_t>(C) * S) {
        std::fprintf(stderr, "seg/hard size mismatch\n");
        return 1;
    }
    // post_process needs the embedder for nothing; load() still required to
    // instantiate (it loads all stages — acceptable for an offline harness).
    deusridet::orator::DiarizenPipeline pipe;
    deusridet::orator::DiarizenPipelineConfig cfg;
    if (!pipe.load(cfg)) {
        std::fprintf(stderr, "load failed: %s\n", pipe.last_error().c_str());
        return 1;
    }
    std::vector<float> count, binary;
    int nf = 0, ncl = 0;
    if (!pipe.debug_post_process(seg.data(), hard.data(), C, F, S, count,
                                 binary, nf, ncl)) {
        std::fprintf(stderr, "post_process failed: %s\n",
                     pipe.last_error().c_str());
        return 1;
    }
    if (!write_f32(argv[7], count) || !write_f32(argv[8], binary)) return 1;
    std::fprintf(stderr, "wrote count[%d] binary[%d*%d]\n", nf, nf, ncl);
    return 0;
}

int run_diarize(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s diarize <wave.bin> <out.bin>\n",
                     argv[0]);
        return 2;
    }
    std::vector<float> wave;
    if (!read_f32(argv[2], wave)) return 1;
    deusridet::orator::DiarizenPipeline pipe;
    deusridet::orator::DiarizenPipelineConfig cfg;
    if (!pipe.load(cfg)) {
        std::fprintf(stderr, "load failed: %s\n", pipe.last_error().c_str());
        return 1;
    }
    auto segs = pipe.diarize(wave.data(), static_cast<int>(wave.size()));
    if (segs.empty()) {
        std::fprintf(stderr, "diarize empty: %s\n", pipe.last_error().c_str());
        return 1;
    }
    // Write [start, end, label_index] float triples ("speakerK" -> K).
    std::vector<float> out;
    out.reserve(segs.size() * 3);
    for (const auto& s : segs) {
        out.push_back(static_cast<float>(s.start_sec));
        out.push_back(static_cast<float>(s.end_sec));
        const char* p = s.label.c_str();
        while (*p && (*p < '0' || *p > '9')) ++p;
        out.push_back(static_cast<float>(std::atoi(p)));
    }
    if (!write_f32(argv[3], out)) return 1;
    std::fprintf(stderr, "wrote %zu segments\n", segs.size());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <embeddings|postproc|diarize> ...\n",
                     argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "embeddings") return run_embeddings(argc, argv);
    if (mode == "postproc") return run_postproc(argc, argv);
    if (mode == "diarize") return run_diarize(argc, argv);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
