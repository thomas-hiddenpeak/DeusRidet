/**
 * @file build_overlap_patched_pcm.cpp
 * @philosophical_role Surgical overlap remediation. For every refined-GT
 *                     segment, ask pyannote-seg3 whether two voices co-occur;
 *                     if they do, route the segment through MossFormer2 and
 *                     keep only the dominant (higher-RMS) source. The output
 *                     is a patched PCM byte-identical to the original outside
 *                     overlap segments — drop-in replacement for the
 *                     CAM++/WL-ECAPA extractors that built fused_v1.bin.
 * @serves Orator reclusterer Phase 12 — overlap-aware fixture (fused_v2).
 *
 * Inputs:
 *   /tmp/test_mp3_16k_mono.f32                    (raw f32 PCM 16k mono)
 *   tests/fixtures/test_ground_truth_v1.jsonl     (refined GT JSONL)
 *   ~/models/dev/vad/pyannote_seg3.safetensors
 *   ~/models/dev/vad/mossformer2_ss_16k.safetensors
 *
 * Outputs:
 *   /tmp/test_mp3_16k_mono_overlap_dom.f32        (patched PCM)
 *   tests/fixtures/overlap_patch_v1.jsonl         (per-segment log)
 */

#include "../src/sensus/auditus/overlap_detector.h"
#include "../src/sensus/auditus/speech_separator.h"
#include "../src/communis/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace deusridet;

namespace {

constexpr int kSR = 16000;

struct GtSeg {
    int idx = -1;
    long start_ms = 0;
    long end_ms = 0;
    std::string speaker;
};

static bool parse_long(const std::string& line, const char* key, long& v) {
    std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
    char* end = nullptr;
    v = std::strtol(line.c_str() + p, &end, 10);
    return end != line.c_str() + p;
}

static bool parse_str(const std::string& line, const char* key, std::string& v) {
    std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
    if (p >= line.size() || line[p] != '"') return false;
    p++;
    auto q = line.find('"', p);
    if (q == std::string::npos) return false;
    v.assign(line.begin() + p, line.begin() + q);
    return true;
}

static std::vector<GtSeg> load_gt(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<GtSeg> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        GtSeg g;
        long v;
        if (!parse_long(line, "idx", v))      continue; g.idx = (int)v;
        if (!parse_long(line, "start_ms", v)) continue; g.start_ms = v;
        if (!parse_long(line, "end_ms", v))   continue; g.end_ms = v;
        parse_str(line, "speaker", g.speaker);
        out.push_back(std::move(g));
    }
    return out;
}

static std::vector<float> load_pcm_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    auto n = f.tellg();
    f.seekg(0);
    std::vector<float> out(n / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

static float rms(const float* x, long n) {
    if (n <= 0) return 0.0f;
    double s = 0.0;
    for (long i = 0; i < n; i++) s += (double)x[i] * x[i];
    return (float)std::sqrt(s / n);
}

static std::string home_dir() {
    const char* h = std::getenv("HOME");
    return h ? std::string(h) : std::string("/home/rm01");
}

} // namespace

int main(int argc, char** argv) {
    std::string pcm_path  = "/tmp/test_mp3_16k_mono.f32";
    std::string gt_path   = "tests/fixtures/test_ground_truth_v1.jsonl";
    std::string out_pcm   = "/tmp/test_mp3_16k_mono_overlap_dom.f32";
    std::string out_log   = "tests/fixtures/overlap_patch_v1.jsonl";
    std::string od_model  = home_dir() + "/models/dev/vad/pyannote_seg3.safetensors";
    std::string sep_model = home_dir() + "/models/dev/vad/mossformer2_ss_16k.safetensors";
    float overlap_thresh  = 0.20f;       // segment-level ratio gate
    int max_segments      = -1;          // debug cap
    bool dry_run          = false;       // detect only, do not separate

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&](const char* n) {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", n); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (a == "--pcm")             pcm_path = nxt("--pcm");
        else if (a == "--gt")              gt_path = nxt("--gt");
        else if (a == "--out-pcm")         out_pcm = nxt("--out-pcm");
        else if (a == "--out-log")         out_log = nxt("--out-log");
        else if (a == "--od-model")        od_model = nxt("--od-model");
        else if (a == "--sep-model")       sep_model = nxt("--sep-model");
        else if (a == "--overlap-thresh")  overlap_thresh = std::stof(nxt("--overlap-thresh"));
        else if (a == "--max-segments")    max_segments = std::stoi(nxt("--max-segments"));
        else if (a == "--dry-run")         dry_run = true;
        else { std::fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return 1; }
    }

    auto pcm = load_pcm_f32(pcm_path);
    const long total = (long)pcm.size();
    std::fprintf(stderr, "[pcm] %ld samples (%.2f s)\n", total, total / (double)kSR);

    auto gts = load_gt(gt_path);
    if (max_segments > 0 && (int)gts.size() > max_segments) gts.resize(max_segments);
    std::fprintf(stderr, "[gt] %zu segments (overlap_thresh=%.2f, dry_run=%d)\n",
                 gts.size(), overlap_thresh, (int)dry_run);

    OverlapDetectorConfig od_cfg;
    od_cfg.model_path = od_model;
    od_cfg.overlap_threshold = 0.5f;
    od_cfg.chunk_samples = 160000;
    od_cfg.hop_samples = 80000;
    OverlapDetector od;
    if (!od.init(od_cfg)) {
        std::fprintf(stderr, "OverlapDetector init failed (%s)\n", od_model.c_str());
        return 1;
    }
    std::fprintf(stderr, "[od] frames=%d classes=%d\n", od.num_output_frames(), od.num_classes());

    SpeechSeparator sep;
    if (!dry_run) {
        SpeechSeparatorConfig sep_cfg;
        sep_cfg.model_path = sep_model;
        sep_cfg.lazy_load = false;
        if (!sep.init(sep_cfg)) {
            std::fprintf(stderr, "SpeechSeparator init failed (%s)\n", sep_model.c_str());
            return 1;
        }
        std::fprintf(stderr, "[sep] MossFormer2 loaded\n");
    }

    // Patched PCM starts as a copy of the original.
    std::vector<float> patched(pcm.begin(), pcm.end());

    std::ofstream log(out_log);
    if (!log) { std::fprintf(stderr, "Cannot write %s\n", out_log.c_str()); return 1; }

    int n_detected = 0;
    int n_separated = 0;
    int n_replaced  = 0;

    std::vector<float> win_buf(160000);

    for (size_t gi = 0; gi < gts.size(); gi++) {
        const auto& g = gts[gi];
        long s0 = g.start_ms * kSR / 1000;
        long s1 = g.end_ms   * kSR / 1000;
        if (s0 < 0) s0 = 0;
        if (s1 > total) s1 = total;
        long dur = s1 - s0;
        if (dur < kSR / 5) {  // <200ms : skip
            log << "{\"idx\":" << g.idx
                << ",\"start_ms\":" << g.start_ms
                << ",\"end_ms\":" << g.end_ms
                << ",\"speaker\":\"" << g.speaker << "\""
                << ",\"overlap_ratio\":0.0,\"separated\":false,\"reason\":\"too_short\"}\n";
            continue;
        }

        // Build 10s window centered on segment, padded with zeros if needed.
        long center = (s0 + s1) / 2;
        long win_s = center - 80000;
        long win_e = win_s + 160000;
        if (win_s < 0) { win_s = 0; win_e = 160000; }
        if (win_e > total) { win_e = total; win_s = total - 160000; if (win_s < 0) win_s = 0; }
        std::fill(win_buf.begin(), win_buf.end(), 0.0f);
        long copy_n = std::min<long>(160000, win_e - win_s);
        std::memcpy(win_buf.data(), pcm.data() + win_s, copy_n * sizeof(float));

        auto od_res = od.detect(win_buf.data(), 160000);

        // Frames covering [s0, s1] inside this 10s window.
        const int n_frames = od_res.num_frames > 0 ? od_res.num_frames : (int)od_res.frame_overlap.size();
        if (n_frames <= 0) continue;
        const double spf = 160000.0 / (double)n_frames;
        int fs = (int)std::floor((s0 - win_s) / spf);
        int fe = (int)std::ceil ((s1 - win_s) / spf);
        if (fs < 0) fs = 0;
        if (fe > n_frames) fe = n_frames;
        if (fe <= fs) fe = fs + 1;

        int n_ov = 0;
        for (int f = fs; f < fe; f++)
            if (f < (int)od_res.frame_overlap.size() && od_res.frame_overlap[f]) n_ov++;
        const float seg_ov_ratio = (float)n_ov / (float)std::max(1, fe - fs);

        log << "{\"idx\":" << g.idx
            << ",\"start_ms\":" << g.start_ms
            << ",\"end_ms\":" << g.end_ms
            << ",\"speaker\":\"" << g.speaker << "\""
            << ",\"overlap_ratio\":" << seg_ov_ratio;

        if (seg_ov_ratio < overlap_thresh) {
            log << ",\"separated\":false}\n";
            continue;
        }
        n_detected++;

        if (dry_run) {
            log << ",\"separated\":false,\"reason\":\"dry_run\"}\n";
            continue;
        }

        // Separate this segment.
        SeparationResult sr = sep.separate(pcm.data() + s0, (int)dur);
        if (!sr.valid) {
            log << ",\"separated\":false,\"reason\":\"sep_invalid\"}\n";
            continue;
        }
        n_separated++;

        const std::vector<float>& dom = (sr.energy1 >= sr.energy2) ? sr.source1 : sr.source2;
        const int dom_src = (sr.energy1 >= sr.energy2) ? 1 : 2;
        if ((long)dom.size() < dur) {
            log << ",\"separated\":false,\"reason\":\"size_mismatch\""
                << ",\"src1_rms\":" << sr.energy1
                << ",\"src2_rms\":" << sr.energy2 << "}\n";
            continue;
        }

        const float orig_rms = rms(pcm.data() + s0, dur);
        const float dom_rms  = rms(dom.data(), dur);
        const float scale = (dom_rms > 1e-6f) ? (orig_rms / dom_rms) : 1.0f;

        for (long i = 0; i < dur; i++) {
            float v = dom[(size_t)i] * scale;
            if (v >  1.0f) v =  1.0f;
            if (v < -1.0f) v = -1.0f;
            patched[(size_t)(s0 + i)] = v;
        }
        n_replaced++;

        log << ",\"separated\":true"
            << ",\"dom_source\":" << dom_src
            << ",\"src1_rms\":"   << sr.energy1
            << ",\"src2_rms\":"   << sr.energy2
            << ",\"orig_rms\":"   << orig_rms
            << ",\"dom_rms\":"    << dom_rms
            << ",\"scale\":"      << scale
            << "}\n";

        if ((int)gi % 50 == 49) {
            std::fprintf(stderr, "[progress] %zu/%zu  ov_seg=%d  replaced=%d\n",
                         gi + 1, gts.size(), n_detected, n_replaced);
        }
    }
    log.close();

    // Write patched PCM.
    {
        std::ofstream f(out_pcm, std::ios::binary);
        if (!f) { std::fprintf(stderr, "Cannot write %s\n", out_pcm.c_str()); return 1; }
        f.write(reinterpret_cast<const char*>(patched.data()), patched.size() * sizeof(float));
    }

    std::fprintf(stderr, "[done] n_gt=%zu  ov_detected=%d  separated_ok=%d  replaced=%d\n",
                 gts.size(), n_detected, n_separated, n_replaced);
    std::fprintf(stderr, "[done] patched PCM -> %s (%.1f MB)\n",
                 out_pcm.c_str(), patched.size() * sizeof(float) / 1048576.0);
    std::fprintf(stderr, "[done] per-seg log -> %s\n", out_log.c_str());
    return 0;
}
