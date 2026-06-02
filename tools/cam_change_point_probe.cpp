/**
 * @file cam_change_point_probe.cpp
 * @philosophical_role Step 19a — Offline sub-segment change-point probe.
 *                     For every VAD interval >= min_vad_sec, slides a
 *                     CAM++ window across the interval and emits the
 *                     adjacent-window cosine similarity sequence.
 *                     Diagnostic only: does not modify online behavior.
 * @serves offline change-point inspection (former
 *          tools/score_change_points.py deleted 2026-06-02 as a
 *          forbidden scoring script; this probe only emits the cosine
 *          similarity sequence for eyes-on reading, no F1/score).
 *
 * I/O:
 *   input : --pcm        /tmp/test_mp3_16k_mono.f32   (f32 mono 16k)
 *           --timeline   logs/timeline/<run>.jsonl    (online VAD events)
 *           --model      .../campplus.safetensors
 *   output: --out        /tmp/cam_change_points.jsonl
 *
 * Encoder + fbank settings mirror cam_extract_embeddings.cpp exactly so
 * results are directly comparable to the online CAM++ pipeline.
 */

#include "../src/sensus/auditus/povey_fbank_gpu.h"
#include "../src/orator/speaker_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSR     = 16000;
constexpr int kEmbDim = 192;

struct VadSeg {
    int   idx;          // sequential index, 0..N-1
    double start_sec;
    double end_sec;
};

// Parse online timeline VAD events into closed [start_sec, end_sec] intervals.
// Mirrors tools/segment_homogeneity_audit.py:parse_timeline_vad_segments.
// Each "vad" event has fields:
//   "event": "start" | "end"
//   "audio_t1": int64 samples processed at this point
// audio_t1 / kSR == seconds since stream start.
static std::vector<VadSeg> load_vad_timeline(const std::string& path,
                                             double max_sec) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }

    auto find_str = [](const std::string& line, const char* key,
                       std::string& v) -> bool {
        std::string k = std::string("\"") + key + "\":\"";
        auto p = line.find(k);
        if (p == std::string::npos) return false;
        p += k.size();
        auto q = line.find('"', p);
        if (q == std::string::npos) return false;
        v.assign(line.begin() + p, line.begin() + q);
        return true;
    };
    auto find_i64 = [](const std::string& line, const char* key,
                       long long& v) -> bool {
        std::string k = std::string("\"") + key + "\":";
        auto p = line.find(k);
        if (p == std::string::npos) return false;
        p += k.size();
        // skip spaces
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
        char* end = nullptr;
        v = std::strtoll(line.c_str() + p, &end, 10);
        return end != line.c_str() + p;
    };

    std::vector<VadSeg> segs;
    std::string line;
    bool in_seg = false;
    double cur_start = 0.0;
    int next_idx = 0;
    while (std::getline(f, line)) {
        if (line.find("\"t\":\"vad\"") == std::string::npos) continue;
        std::string event;
        if (!find_str(line, "event", event)) continue;
        long long audio_t1 = 0;
        if (!find_i64(line, "audio_t1", audio_t1)) continue;
        double sec = (double)audio_t1 / (double)kSR;
        if (event == "start" && !in_seg) {
            cur_start = sec;
            in_seg = true;
        } else if (event == "end" && in_seg) {
            VadSeg v{next_idx++, cur_start, sec};
            if (v.start_sec <= max_sec + 2.0) segs.push_back(v);
            in_seg = false;
        }
    }
    return segs;
}

static std::vector<float> load_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    auto n = f.tellg();
    f.seekg(0);
    std::vector<float> out(n / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

static double cosine(const float* a, const float* b, int d) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < d; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Extract one CAM++ embedding from PCM samples [s0, s0+n).
// Uses a fresh PoveyFbankGpu to avoid residual state.
static bool extract_emb(const int16_t* pcm, long total, long s0, long n,
                        deusridet::SpeakerEncoder& enc,
                        std::vector<float>& out) {
    if (s0 < 0 || n <= 0 || s0 + n > total) return false;
    deusridet::PoveyFbankGpu fb;
    if (!fb.init(80, 400, 160, 512, kSR,
                 deusridet::FbankWindowType::POVEY,
                 /*normalize_pcm=*/true)) return false;
    int produced = fb.push_pcm(pcm + s0, (int)n);
    if (produced <= 0) return false;
    std::vector<float> mel((size_t)produced * 80);
    int got = fb.read_fbank(mel.data(), produced);
    if (got <= 0) return false;
    out = enc.extract(mel.data(), got);
    return (int)out.size() == kEmbDim;
}

} // namespace

int main(int argc, char** argv) {
    std::string pcm_path  = "/tmp/test_mp3_16k_mono.f32";
    std::string tl_path;
    std::string model_path =
        "/home/rm01/models/dev/speaker/campplus/campplus.safetensors";
    std::string out_path  = "/tmp/cam_change_points.jsonl";
    double win_sec     = 1.5;
    double hop_sec     = 0.5;
    double min_vad_sec = 2.0;
    double max_sec     = 600.0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&](const char* n) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing arg for %s\n", n); std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (a == "--pcm")         pcm_path  = nxt("--pcm");
        else if (a == "--timeline")    tl_path   = nxt("--timeline");
        else if (a == "--model")       model_path = nxt("--model");
        else if (a == "--out")         out_path  = nxt("--out");
        else if (a == "--win-sec")     win_sec   = std::atof(nxt("--win-sec").c_str());
        else if (a == "--hop-sec")     hop_sec   = std::atof(nxt("--hop-sec").c_str());
        else if (a == "--min-vad-sec") min_vad_sec = std::atof(nxt("--min-vad-sec").c_str());
        else if (a == "--max-sec")     max_sec   = std::atof(nxt("--max-sec").c_str());
        else { fprintf(stderr, "Unknown arg %s\n", a.c_str()); return 1; }
    }
    if (tl_path.empty()) {
        fprintf(stderr, "Usage: --timeline <jsonl> is required\n");
        return 1;
    }

    auto pcm_f = load_f32(pcm_path);
    std::vector<int16_t> pcm(pcm_f.size());
    for (size_t i = 0; i < pcm_f.size(); i++) {
        float v = pcm_f[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    pcm_f.clear(); pcm_f.shrink_to_fit();
    const long total = (long)pcm.size();
    fprintf(stderr, "[pcm] %ld samples (%.2f s)\n", total, total / (double)kSR);

    auto vads = load_vad_timeline(tl_path, max_sec);
    fprintf(stderr, "[vad] %zu intervals from %s\n", vads.size(), tl_path.c_str());

    deusridet::SpeakerEncoderConfig sc;
    sc.model_path = model_path;
    deusridet::SpeakerEncoder enc;
    if (!enc.init(sc)) {
        fprintf(stderr, "SpeakerEncoder init failed (%s)\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "[encoder] CAM++ initialized\n");

    std::ofstream out(out_path);
    if (!out) { fprintf(stderr, "Cannot open %s for write\n", out_path.c_str()); return 1; }

    int n_long = 0, n_emit = 0;
    const long win_samp = (long)(win_sec * kSR);
    const long hop_samp = (long)(hop_sec * kSR);

    for (const auto& v : vads) {
        if (v.start_sec > max_sec) break;
        double dur = v.end_sec - v.start_sec;
        if (dur < min_vad_sec) continue;
        n_long++;

        long seg_s0 = (long)(v.start_sec * kSR);
        long seg_s1 = (long)(v.end_sec   * kSR);
        seg_s0 = std::max<long>(0, seg_s0);
        seg_s1 = std::min<long>(total, seg_s1);
        if (seg_s1 - seg_s0 < win_samp) continue;

        std::vector<double> centers;
        std::vector<std::vector<float>> embs;
        for (long ws = seg_s0; ws + win_samp <= seg_s1; ws += hop_samp) {
            std::vector<float> e;
            if (!extract_emb(pcm.data(), total, ws, win_samp, enc, e)) continue;
            centers.push_back((ws + win_samp / 2.0) / (double)kSR);
            embs.push_back(std::move(e));
        }
        if (embs.size() < 2) continue;

        std::vector<double> sims;
        sims.reserve(embs.size() - 1);
        for (size_t i = 1; i < embs.size(); i++) {
            sims.push_back(cosine(embs[i - 1].data(), embs[i].data(), kEmbDim));
        }

        out << "{\"vad_idx\":" << v.idx
            << ",\"start_sec\":" << v.start_sec
            << ",\"end_sec\":" << v.end_sec
            << ",\"win_sec\":" << win_sec
            << ",\"hop_sec\":" << hop_sec
            << ",\"n_win\":" << embs.size()
            << ",\"centers\":[";
        for (size_t i = 0; i < centers.size(); i++) {
            if (i) out << ',';
            out << centers[i];
        }
        out << "],\"adj_cos\":[";
        for (size_t i = 0; i < sims.size(); i++) {
            if (i) out << ',';
            out << sims[i];
        }
        out << "]}\n";
        n_emit++;

        if (n_emit % 20 == 0) {
            fprintf(stderr, "[progress] emitted %d / long_vad %d\n", n_emit, n_long);
        }
    }

    fprintf(stderr, "[done] long_vad=%d emitted=%d -> %s\n",
            n_long, n_emit, out_path.c_str());
    return 0;
}
