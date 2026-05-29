// @role: P1c bit-equality harness for the DiariZen segmentation orchestrator.
//        Reads a raw float32 16 kHz mono waveform, runs DiarizenSegmenter over
//        the sliding window, and writes the [num_chunks, 799, 4] multilabel map
//        as raw float32 (with a 3-int header: num_chunks, num_frames,
//        num_speakers). The Python driver tools/diarizen_bit_eq_segmentation.py
//        feeds wave_full from the reference .npz and compares against seg_raw
//        (--raw) or seg_med (default). Mechanical bit-equality, not a semantic
//        quality score, per workflow.instructions.md.
#include "../src/orator/diarizen_segmenter.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using deusridet::orator::DiarizenSegmentation;
using deusridet::orator::DiarizenSegmenter;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <wavlm.safetensors> <conformer.safetensors> "
                     "<wave.bin> <n_samples> <out.bin> [--raw]\n",
                     argv[0]);
        return 2;
    }
    const char* wavlm_path = argv[1];
    const char* conf_path = argv[2];
    const char* wave_path = argv[3];
    const int n_samples = std::atoi(argv[4]);
    const char* out_path = argv[5];
    const bool raw = (argc >= 7) && (std::string(argv[6]) == "--raw");

    std::vector<float> wave(n_samples);
    FILE* f = std::fopen(wave_path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open wave %s\n", wave_path);
        return 1;
    }
    size_t got = std::fread(wave.data(), sizeof(float), n_samples, f);
    std::fclose(f);
    if ((int)got != n_samples) {
        std::fprintf(stderr, "wave short read: got %zu expected %d\n", got,
                     n_samples);
        return 1;
    }

    DiarizenSegmenter seg;
    if (!seg.load(wavlm_path, conf_path)) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }

    DiarizenSegmentation out =
        raw ? seg.segment_raw(wave.data(), n_samples)
            : seg.segment(wave.data(), n_samples, /*apply_median=*/true);
    if (out.empty()) {
        std::fprintf(stderr, "segment failed\n");
        return 1;
    }
    std::fprintf(stderr, "segment%s: chunks=%d frames=%d speakers=%d\n",
                 raw ? "_raw" : "", out.num_chunks, out.num_frames,
                 out.num_speakers);

    FILE* o = std::fopen(out_path, "wb");
    if (!o) {
        std::fprintf(stderr, "cannot open out %s\n", out_path);
        return 1;
    }
    const std::int32_t hdr[3] = {out.num_chunks, out.num_frames,
                                 out.num_speakers};
    std::fwrite(hdr, sizeof(std::int32_t), 3, o);
    std::fwrite(out.data.data(), sizeof(float), out.data.size(), o);
    std::fclose(o);
    std::fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
