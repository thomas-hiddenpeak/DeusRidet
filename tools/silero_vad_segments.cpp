/**
 * @file silero_vad_segments.cpp
 * @philosophical_role Offline harness — emits voiced regions of an audio file
 *                     using DeusRidet's native SileroVad. Used by Step 2a
 *                     ground-truth refinement (CAM++ baseline evaluation).
 * @serves tools/refine_gt.py
 *
 * Usage:
 *   silero_vad_segments <pcm16k_mono_f32.raw> <out.json>
 *                       [--threshold T] [--min-speech-ms N] [--min-silence-ms N]
 *                       [--pad-ms N]
 *
 * Input  : raw float32 PCM, mono, 16 kHz (decode beforehand with ffmpeg)
 * Output : JSON array [{"start_ms":N,"end_ms":M}, ...]
 *
 * The model lives at /home/rm01/models/dev/vad/silero_vad.safetensors
 * (matches tests/test_silero_vad.cpp).
 */

#include "../src/sensus/auditus/silero_vad.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s <pcm16k_f32.raw> <out.json>\n"
        "       [--threshold T] [--min-speech-ms N] [--min-silence-ms N] [--pad-ms N]\n",
        argv0);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* pcm_path = argv[1];
    const char* out_path = argv[2];

    deusridet::SileroVadConfig cfg;
    cfg.model_path     = "/home/rm01/models/dev/vad/silero_vad.safetensors";
    cfg.sample_rate    = 16000;
    cfg.window_samples = 512;
    cfg.threshold      = 0.5f;
    cfg.min_speech_ms  = 250;
    cfg.min_silence_ms = 100;
    cfg.speech_pad_ms  = 30;

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--threshold")      cfg.threshold      = std::atof(next("--threshold"));
        else if (a == "--min-speech-ms")  cfg.min_speech_ms  = std::atoi(next("--min-speech-ms"));
        else if (a == "--min-silence-ms") cfg.min_silence_ms = std::atoi(next("--min-silence-ms"));
        else if (a == "--pad-ms")         cfg.speech_pad_ms  = std::atoi(next("--pad-ms"));
        else { fprintf(stderr, "Unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    // Load PCM
    std::ifstream in(pcm_path, std::ios::binary | std::ios::ate);
    if (!in) { fprintf(stderr, "Cannot open %s\n", pcm_path); return 1; }
    auto bytes = in.tellg();
    in.seekg(0);
    if (bytes <= 0 || (bytes % sizeof(float)) != 0) {
        fprintf(stderr, "Bad PCM size %lld\n", (long long)bytes); return 1;
    }
    std::vector<float> pcm(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(pcm.data()), bytes);
    fprintf(stderr, "Loaded %zu samples (%.2f s @16k)\n",
            pcm.size(), pcm.size() / 16000.0);

    deusridet::SileroVad vad;
    if (!vad.init(cfg)) {
        fprintf(stderr, "SileroVad init failed (model=%s)\n", cfg.model_path.c_str());
        return 1;
    }

    const int W = cfg.window_samples;       // 512
    const int total = (int)pcm.size();

    // Walk windows; emit (start_sample, end_sample) pairs on rising/falling edge.
    // Sample index of segment edge = sample index of LAST sample fed to process()
    // when the edge fires. Convert to ms using sample_rate.
    std::vector<std::pair<long, long>> segs;     // (start_sample, end_sample)
    long cur_start = -1;

    for (int s = 0; s + W <= total; s += W) {
        auto r = vad.process(pcm.data() + s, W);
        long sample_at_end = (long)s + W;
        if (r.segment_start) {
            // Trigger fires after min_speech_ms of speech accumulated.
            // Approximate start = sample_at_end - min_speech_samples.
            long min_s = (long)cfg.min_speech_ms * cfg.sample_rate / 1000;
            cur_start = sample_at_end - min_s;
            if (cur_start < 0) cur_start = 0;
        }
        if (r.segment_end && cur_start >= 0) {
            // End fires after min_silence_ms of silence; subtract that to get
            // the actual speech tail.
            long min_sil = (long)cfg.min_silence_ms * cfg.sample_rate / 1000;
            long end_sample = sample_at_end - min_sil;
            if (end_sample <= cur_start) end_sample = cur_start + W;
            segs.emplace_back(cur_start, end_sample);
            cur_start = -1;
        }
    }
    // If still in speech at EOF, close the segment.
    if (cur_start >= 0) {
        segs.emplace_back(cur_start, (long)total);
    }

    fprintf(stderr, "Found %zu voiced segments\n", segs.size());

    // Emit JSON
    FILE* out = std::fopen(out_path, "w");
    if (!out) { fprintf(stderr, "Cannot write %s\n", out_path); return 1; }
    std::fprintf(out, "[\n");
    for (size_t i = 0; i < segs.size(); i++) {
        long start_ms = segs[i].first  * 1000 / cfg.sample_rate;
        long end_ms   = segs[i].second * 1000 / cfg.sample_rate;
        std::fprintf(out, "  {\"start_ms\":%ld,\"end_ms\":%ld}%s\n",
                     start_ms, end_ms, (i + 1 == segs.size() ? "" : ","));
    }
    std::fprintf(out, "]\n");
    std::fclose(out);
    fprintf(stderr, "Wrote %s\n", out_path);
    return 0;
}
