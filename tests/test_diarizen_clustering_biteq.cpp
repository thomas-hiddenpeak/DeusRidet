// @role: P2b bit-equality harness for the DiariZen VBx clustering stage.
//        Stages selectable: --fea (PLDA feature transform), --ahc (AHC
//        labels), --hard (full hard_clusters). Mechanical bit-equality.
#include "../src/orator/diarizen_clustering.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using deusridet::orator::DiarizenClustering;

static std::vector<float> read_f32(const char* path, std::size_t n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", path); std::exit(1); }
    std::size_t got = std::fread(v.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n) { std::fprintf(stderr, "short %s %zu!=%zu\n", path, got, n); std::exit(1); }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <plda_dir> <mode> <out.bin> [args...]\n"
            "  --fea  <train_emb.bin> <N> <xdim>\n"
            "  --ahc  <train_emb.bin> <N> <xdim>\n"
            "  --hard <emb.bin> <C> <S> <dim> <seg.bin> <F>\n", argv[0]);
        return 2;
    }
    const char* plda_dir = argv[1];
    const std::string mode = argv[2];
    const char* out_path = argv[3];

    DiarizenClustering clu;
    if (!clu.load_priors(plda_dir)) { std::fprintf(stderr, "load_priors failed\n"); return 1; }

    FILE* o = std::fopen(out_path, "wb");
    if (!o) { std::fprintf(stderr, "open out %s\n", out_path); return 1; }

    if (mode == "--fea") {
        const int N = std::atoi(argv[5]);
        const int xdim = std::atoi(argv[6]);
        auto emb = read_f32(argv[4], (std::size_t)N * xdim);
        std::vector<float> fea;
        if (!clu.debug_fea(emb.data(), N, xdim, fea)) return 1;
        std::fwrite(fea.data(), sizeof(float), fea.size(), o);
        std::fprintf(stderr, "fea N=%d pdim=%d\n", N, (int)(fea.size() / N));
    } else if (mode == "--ahc") {
        const int N = std::atoi(argv[5]);
        const int xdim = std::atoi(argv[6]);
        auto emb = read_f32(argv[4], (std::size_t)N * xdim);
        std::vector<int> ahc;
        if (!clu.debug_ahc(emb.data(), N, xdim, ahc)) return 1;
        std::vector<std::int32_t> a(ahc.begin(), ahc.end());
        std::fwrite(a.data(), sizeof(std::int32_t), a.size(), o);
        std::fprintf(stderr, "ahc N=%d\n", N);
    } else if (mode == "--hard") {
        const int C = std::atoi(argv[5]);
        const int S = std::atoi(argv[6]);
        const int dim = std::atoi(argv[7]);
        const int F = std::atoi(argv[9]);
        auto emb = read_f32(argv[4], (std::size_t)C * S * dim);
        auto seg = read_f32(argv[8], (std::size_t)C * F * S);
        std::vector<std::int8_t> hard;
        if (!clu.cluster(emb.data(), C, S, dim, seg.data(), F, hard)) return 1;
        std::fwrite(hard.data(), sizeof(std::int8_t), hard.size(), o);
        std::fprintf(stderr, "hard C=%d S=%d\n", C, S);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    std::fclose(o);
    return 0;
}
