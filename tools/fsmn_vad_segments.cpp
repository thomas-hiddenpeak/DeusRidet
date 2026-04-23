/**
 * @file fsmn_vad_segments.cpp
 * @philosophical_role Offline harness — emits voiced regions using DeusRidet's
 *                     native FSMN VAD (CAM++'s official companion). Companion
 *                     to silero_vad_segments for the Step 2a² VAD×Encoder
 *                     matrix evaluation.
 * @serves tools/refine_gt.py (with --vad fsmn)
 *
 * Usage:
 *   fsmn_vad_segments <pcm16k_mono_f32.raw> <out.json>
 *                     [--threshold T] [--chunk-ms N]
 *                     [--min-speech-ms N] [--min-silence-ms N] [--pad-ms N]
 *
 * FSMN returns one speech probability per process() call (over all new LFR
 * frames accumulated in that call). We chunk the audio at fixed granularity
 * (default 100 ms) and apply a simple hysteresis state machine.
 */

#include "../src/sensus/auditus/fsmn_vad.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void usage(const char* a0) {
    fprintf(stderr,
        "Usage: %s <pcm16k_f32.raw> <out.json>\n"
        "       [--threshold T] [--chunk-ms N]\n"
        "       [--min-speech-ms N] [--min-silence-ms N] [--pad-ms N]\n", a0);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const char* pcm_path = argv[1];
    const char* out_path = argv[2];

    deusridet::FsmnVadConfig cfg;
    cfg.model_path = "/home/rm01/models/dev/vad/fsmn/fsmn_vad.safetensors";
    cfg.cmvn_path  = "/home/rm01/models/dev/vad/fsmn/am.mvn";
    cfg.threshold  = 0.5f;

    int chunk_ms      = 100;
    int min_speech_ms = 250;
    int min_silence_ms= 100;
    int pad_ms        = 30;

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "Missing %s\n", n); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--threshold")      cfg.threshold  = std::atof(nxt("--threshold"));
        else if (a == "--chunk-ms")       chunk_ms       = std::atoi(nxt("--chunk-ms"));
        else if (a == "--min-speech-ms")  min_speech_ms  = std::atoi(nxt("--min-speech-ms"));
        else if (a == "--min-silence-ms") min_silence_ms = std::atoi(nxt("--min-silence-ms"));
        else if (a == "--pad-ms")         pad_ms         = std::atoi(nxt("--pad-ms"));
        else { fprintf(stderr, "Unknown %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    std::ifstream in(pcm_path, std::ios::binary | std::ios::ate);
    if (!in) { fprintf(stderr, "Cannot open %s\n", pcm_path); return 1; }
    auto bytes = in.tellg();
    in.seekg(0);
    if (bytes <= 0 || (bytes % sizeof(float)) != 0) {
        fprintf(stderr, "Bad PCM size %lld\n", (long long)bytes); return 1;
    }
    std::vector<float> pcm_f(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(pcm_f.data()), bytes);
    fprintf(stderr, "Loaded %zu samples (%.2f s)\n",
            pcm_f.size(), pcm_f.size() / 16000.0);

    // Convert to int16.
    std::vector<int16_t> pcm16(pcm_f.size());
    for (size_t i = 0; i < pcm_f.size(); i++) {
        float v = pcm_f[i] * 32768.0f;
        if (v >  32767.f) v =  32767.f;
        if (v < -32768.f) v = -32768.f;
        pcm16[i] = (int16_t)v;
    }

    deusridet::FsmnVad vad;
    if (!vad.init(cfg)) {
        fprintf(stderr, "FsmnVad init failed (model=%s)\n", cfg.model_path.c_str());
        return 1;
    }
    vad.set_threshold(cfg.threshold);

    const int SR    = cfg.sample_rate;
    const int CH    = chunk_ms * SR / 1000;
    const int total = (int)pcm16.size();
    const int n_chunks = total / CH;

    // Per-chunk decisions (after threshold applied in process()).
    std::vector<uint8_t> speech(n_chunks, 0);
    for (int c = 0; c < n_chunks; c++) {
        auto r = vad.process(pcm16.data() + c * CH, CH);
        speech[c] = r.is_speech ? 1 : 0;
    }
    fprintf(stderr, "Decoded %d chunks @ %d ms (%.1f%% speech)\n",
            n_chunks, chunk_ms,
            100.0 * std::count(speech.begin(), speech.end(), (uint8_t)1) / std::max(1, n_chunks));

    // Hysteresis state machine:
    //   need >= min_speech_chunks contiguous speech to open a segment
    //   need >= min_silence_chunks contiguous non-speech to close it
    const int min_speech_c  = std::max(1, min_speech_ms / chunk_ms);
    const int min_silence_c = std::max(1, min_silence_ms / chunk_ms);
    const int pad_c         = pad_ms / chunk_ms;

    std::vector<std::pair<long, long>> segs;  // sample ranges
    bool in_speech = false;
    int  run = 0;
    long seg_start_chunk = -1;
    int  last_speech_chunk = -1;

    for (int c = 0; c < n_chunks; c++) {
        if (!in_speech) {
            if (speech[c]) {
                if (run == 0) seg_start_chunk = c;
                run++;
                if (run >= min_speech_c) {
                    in_speech = true;
                    last_speech_chunk = c;
                }
            } else {
                run = 0;
                seg_start_chunk = -1;
            }
        } else {
            if (speech[c]) {
                last_speech_chunk = c;
                run = 0;
            } else {
                run++;
                if (run >= min_silence_c) {
                    long s = std::max<long>(0, seg_start_chunk - pad_c) * CH;
                    long e = std::min<long>(n_chunks, last_speech_chunk + 1 + pad_c) * CH;
                    segs.emplace_back(s, e);
                    in_speech = false;
                    run = 0;
                    seg_start_chunk = -1;
                }
            }
        }
    }
    if (in_speech && seg_start_chunk >= 0) {
        long s = std::max<long>(0, seg_start_chunk - pad_c) * CH;
        long e = std::min<long>(n_chunks, last_speech_chunk + 1 + pad_c) * CH;
        segs.emplace_back(s, e);
    }

    fprintf(stderr, "Found %zu voiced segments\n", segs.size());

    FILE* out = std::fopen(out_path, "w");
    if (!out) { fprintf(stderr, "Cannot write %s\n", out_path); return 1; }
    std::fprintf(out, "[\n");
    for (size_t i = 0; i < segs.size(); i++) {
        long s_ms = segs[i].first  * 1000 / SR;
        long e_ms = segs[i].second * 1000 / SR;
        std::fprintf(out, "  {\"start_ms\":%ld,\"end_ms\":%ld}%s\n",
                     s_ms, e_ms, (i + 1 == segs.size() ? "" : ","));
    }
    std::fprintf(out, "]\n");
    std::fclose(out);
    fprintf(stderr, "Wrote %s\n", out_path);
    return 0;
}
