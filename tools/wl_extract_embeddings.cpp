/**
 * @file wl_extract_embeddings.cpp
 * @philosophical_role Offline WavLM+ECAPA embedding extractor, companion to
 *                     cam_extract_embeddings. Runs the same segment list and
 *                     window strategies so downstream ablation can compare
 *                     CAM++ and WL head-to-head on identical inputs.
 * @serves tools/wl_ablation.py
 *
 * I/O mirrors cam_extract_embeddings but outputs to wl_embeddings_v1.*.
 */

#include "../src/orator/wavlm_ecapa_encoder.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int   kSR       = 16000;
constexpr int   kEmbDim   = 192;
constexpr float kWinsSec[] = {0.0f, 1.5f, 2.0f, 3.0f, 4.0f};
constexpr const char* kWinsName[] = {"full", "1.5s_center", "2.0s_center",
                                     "3.0s_center", "4.0s_center"};
constexpr int kNStrat = 5;

struct GtSeg {
    int    idx;
    long   start_ms;
    long   end_ms;
    int    duration_ms;
    std::string speaker;
    int    src_utt_idx;
};

// Same parser as cam_extract_embeddings.
static bool parse_jsonl_line(const std::string& line, GtSeg& out) {
    auto skip_ws = [](const std::string& s, size_t p) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
        return p;
    };
    auto find_long = [&](const char* key, long& v) -> bool {
        std::string k = std::string("\"") + key + "\":";
        auto p = line.find(k);
        if (p == std::string::npos) return false;
        p = skip_ws(line, p + k.size());
        char* end = nullptr;
        v = std::strtol(line.c_str() + p, &end, 10);
        return end != line.c_str() + p;
    };
    auto find_str = [&](const char* key, std::string& v) -> bool {
        std::string k = std::string("\"") + key + "\":";
        auto p = line.find(k);
        if (p == std::string::npos) return false;
        p = skip_ws(line, p + k.size());
        if (p >= line.size() || line[p] != '"') return false;
        p++;
        auto q = line.find('"', p);
        if (q == std::string::npos) return false;
        v.assign(line.begin() + p, line.begin() + q);
        return true;
    };
    long v;
    if (!find_long("idx", v))          return false; out.idx = (int)v;
    if (!find_long("start_ms", v))     return false; out.start_ms = v;
    if (!find_long("end_ms", v))       return false; out.end_ms = v;
    if (!find_long("duration_ms", v))  return false; out.duration_ms = (int)v;
    if (!find_str ("speaker", out.speaker)) return false;
    if (!find_long("src_utt_idx", v))  return false; out.src_utt_idx = (int)v;
    return true;
}

static std::vector<GtSeg> load_gt(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<GtSeg> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        GtSeg g;
        if (parse_jsonl_line(line, g)) v.push_back(std::move(g));
    }
    return v;
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

static std::pair<long, long> window_range(long start_ms, long end_ms,
                                          float win_sec, long total_samples) {
    long s0 = start_ms * kSR / 1000;
    long s1 = end_ms   * kSR / 1000;
    if (win_sec <= 0.0f) {
        long a = std::max<long>(0, s0);
        long b = std::min<long>(total_samples, s1);
        return {a, b - a};
    }
    long win = (long)(win_sec * kSR);
    long dur = s1 - s0;
    if (dur <= win) {
        long a = std::max<long>(0, s0);
        long b = std::min<long>(total_samples, s1);
        return {a, b - a};
    }
    long mid = (s0 + s1) / 2;
    long ws  = mid - win / 2;
    long we  = ws + win;
    if (ws < 0) { we -= ws; ws = 0; }
    if (we > total_samples) { ws -= (we - total_samples); we = total_samples; }
    if (ws < 0) ws = 0;
    return {ws, we - ws};
}

} // namespace

int main(int argc, char** argv) {
    std::string pcm_path  = "/tmp/test_mp3_16k_mono.f32";
    std::string gt_path   = "tests/fixtures/test_ground_truth_v1.jsonl";
    std::string model_path =
        "/home/rm01/models/dev/speaker/espnet_wavlm_ecapa/wavlm_ecapa.safetensors";
    std::string out_f32   = "tests/fixtures/wl_embeddings_v1.f32";
    std::string out_meta  = "tests/fixtures/wl_embeddings_v1.meta.json";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&](const char* n) {
            if (i + 1 >= argc) { fprintf(stderr, "missing %s\n", n); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (a == "--pcm")      pcm_path   = nxt("--pcm");
        else if (a == "--gt")       gt_path    = nxt("--gt");
        else if (a == "--model")    model_path = nxt("--model");
        else if (a == "--out-f32")  out_f32    = nxt("--out-f32");
        else if (a == "--out-meta") out_meta   = nxt("--out-meta");
        else { fprintf(stderr, "Unknown arg %s\n", a.c_str()); return 1; }
    }

    auto pcm = load_f32(pcm_path);
    fprintf(stderr, "[pcm] %zu samples (%.2f s)\n",
            pcm.size(), pcm.size() / (double)kSR);
    const long total_samples = (long)pcm.size();

    auto gts = load_gt(gt_path);
    fprintf(stderr, "[gt] %zu segments\n", gts.size());

    deusridet::WavLMEcapaEncoder enc;
    if (!enc.init(model_path)) {
        fprintf(stderr, "WavLMEcapaEncoder init failed (%s)\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "[encoder] WL-ECAPA initialized\n");

    const size_t total_embs = gts.size() * (size_t)kNStrat;
    std::vector<float> embs(total_embs * kEmbDim, 0.0f);

    int done = 0;
    int next_log = 50;
    for (size_t gi = 0; gi < gts.size(); gi++) {
        const auto& g = gts[gi];
        for (int s = 0; s < kNStrat; s++) {
            auto [ws, wn] = window_range(g.start_ms, g.end_ms,
                                         kWinsSec[s], total_samples);
            if (wn < kSR / 10) continue;

            auto e = enc.extract(pcm.data() + ws, (int)wn);
            if ((int)e.size() != kEmbDim) continue;
            std::memcpy(embs.data() +
                        (gi * kNStrat + s) * kEmbDim,
                        e.data(), kEmbDim * sizeof(float));
        }
        done++;
        if (done >= next_log) {
            fprintf(stderr, "[progress] %d/%zu\n", done, gts.size());
            next_log = done + 50;
        }
    }

    {
        std::ofstream f(out_f32, std::ios::binary);
        f.write(reinterpret_cast<const char*>(embs.data()),
                embs.size() * sizeof(float));
    }
    {
        std::ofstream f(out_meta);
        f << "{\n";
        f << "  \"dim\": " << kEmbDim << ",\n";
        f << "  \"n_segments\": " << gts.size() << ",\n";
        f << "  \"strategies\": [";
        for (int s = 0; s < kNStrat; s++) {
            f << (s ? ", " : "") << "\"" << kWinsName[s] << "\"";
        }
        f << "],\n";
        f << "  \"layout\": \"float32 [n_segments, n_strategies, dim] row-major\",\n";
        f << "  \"segments\": [\n";
        for (size_t i = 0; i < gts.size(); i++) {
            const auto& g = gts[i];
            f << "    {\"idx\":" << g.idx
              << ",\"start_ms\":" << g.start_ms
              << ",\"end_ms\":"   << g.end_ms
              << ",\"duration_ms\":" << g.duration_ms
              << ",\"speaker\":\"" << g.speaker << "\""
              << ",\"src_utt_idx\":" << g.src_utt_idx << "}"
              << (i + 1 == gts.size() ? "" : ",") << "\n";
        }
        f << "  ]\n";
        f << "}\n";
    }
    fprintf(stderr, "[done] %zu segs x %d strategies -> %s (%zu MB)\n",
            gts.size(), kNStrat, out_f32.c_str(),
            embs.size() * sizeof(float) / (1024 * 1024));
    return 0;
}
