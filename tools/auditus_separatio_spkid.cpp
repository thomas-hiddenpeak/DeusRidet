/**
 * @file auditus_separatio_spkid.cpp
 * @philosophical_role Offline identity probe for separated Auditus streams. It asks whether divided hearing still carries the correct voice.
 * @serves Auditus separation evaluation, Orator speaker-attribution tuning.
 */

#include "src/orator/speaker_encoder.h"
#include "src/orator/speaker_vector_store.h"
#include "src/orator/wavlm_ecapa_encoder.h"
#include "src/sensus/auditus/povey_fbank_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace deusridet;

namespace {

constexpr int kSampleRate = 16000;
constexpr int kDualDim = 384;

struct GtRow {
    int idx = -1;
    long start_ms = 0;
    long end_ms = 0;
    std::string speaker;
};

struct ClipWindow {
    std::string clip_name;
    double start_sec = 0.0;
    double end_sec = 0.0;
};

static std::string home_dir() {
    const char* value = std::getenv("HOME");
    return value ? std::string(value) : "/home/rm01";
}

static bool find_number(const std::string& line, const char* key, double& out) {
    std::string token = std::string("\"") + key + "\":";
    size_t pos = line.find(token);
    if (pos == std::string::npos) return false;
    pos += token.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    char* end_ptr = nullptr;
    out = std::strtod(line.c_str() + pos, &end_ptr);
    return end_ptr != line.c_str() + pos;
}

static bool find_string(const std::string& line, const char* key, std::string& out) {
    std::string token = std::string("\"") + key + "\":";
    size_t pos = line.find(token);
    if (pos == std::string::npos) return false;
    pos += token.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size() || line[pos] != '"') return false;
    pos++;
    std::string value;
    for (; pos < line.size(); pos++) {
        char ch = line[pos];
        if (ch == '"') {
            out = value;
            return true;
        }
        if (ch == '\\' && pos + 1 < line.size()) {
            value.push_back(line[++pos]);
        } else {
            value.push_back(ch);
        }
    }
    return false;
}

static std::string json_escape(const std::string& text) {
    std::string out;
    for (char ch : text) {
        if (ch == '\\') out += "\\\\";
        else if (ch == '"') out += "\\\"";
        else if (ch == '\n') out += "\\n";
        else out += ch;
    }
    return out;
}

static std::vector<GtRow> load_gt(const fs::path& path) {
    std::ifstream input(path);
    std::vector<GtRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        double number_value = 0.0;
        GtRow row;
        if (!find_number(line, "idx", number_value)) continue;
        row.idx = (int)number_value;
        if (!find_number(line, "start_ms", number_value)) continue;
        row.start_ms = (long)number_value;
        if (!find_number(line, "end_ms", number_value)) continue;
        row.end_ms = (long)number_value;
        if (!find_string(line, "speaker", row.speaker)) continue;
        rows.push_back(std::move(row));
    }
    return rows;
}

static std::vector<ClipWindow> load_manifest_windows(const fs::path& path) {
    std::ifstream input(path);
    std::vector<ClipWindow> rows;
    std::string line;
    while (std::getline(input, line)) {
        ClipWindow row;
        if (!find_string(line, "clip_name", row.clip_name)) continue;
        if (!find_number(line, "clip_start_sec", row.start_sec)) continue;
        if (!find_number(line, "clip_end_sec", row.end_sec)) continue;
        rows.push_back(std::move(row));
    }
    return rows;
}

static double overlap_sec(double left_start, double left_end, double right_start, double right_end) {
    return std::max(0.0, std::min(left_end, right_end) - std::max(left_start, right_start));
}

static bool overlaps_manifest(const GtRow& row, const std::vector<ClipWindow>& windows) {
    double start_sec = row.start_ms / 1000.0;
    double end_sec = row.end_ms / 1000.0;
    for (const auto& window : windows) {
        if (overlap_sec(start_sec, end_sec, window.start_sec, window.end_sec) >= 0.05) return true;
    }
    return false;
}

static bool has_other_speaker_overlap(const GtRow& row, const std::vector<GtRow>& all_rows) {
    double start_sec = row.start_ms / 1000.0;
    double end_sec = row.end_ms / 1000.0;
    for (const auto& other : all_rows) {
        if (other.idx == row.idx || other.speaker == row.speaker) continue;
        double other_start = other.start_ms / 1000.0;
        double other_end = other.end_ms / 1000.0;
        if (overlap_sec(start_sec, end_sec, other_start, other_end) >= 0.05) return true;
    }
    return false;
}

static std::string shell_quote(const fs::path& path) {
    std::string input = path.string();
    std::string out = "'";
    for (char ch : input) {
        if (ch == '\'') out += "'\\''";
        else out += ch;
    }
    out += "'";
    return out;
}

static bool read_audio_ffmpeg(const fs::path& path, std::vector<float>& pcm_out) {
    std::string command = "ffmpeg -hide_banner -loglevel error -i " + shell_quote(path) +
                          " -ar 16000 -ac 1 -f s16le -";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return false;
    pcm_out.clear();
    int16_t buffer[4096];
    size_t read_count = 0;
    while ((read_count = fread(buffer, sizeof(int16_t), 4096, pipe)) > 0) {
        size_t old_size = pcm_out.size();
        pcm_out.resize(old_size + read_count);
        for (size_t sample_idx = 0; sample_idx < read_count; sample_idx++) {
            pcm_out[old_size + sample_idx] = buffer[sample_idx] / 32768.0f;
        }
    }
    int status = pclose(pipe);
    return status == 0 && !pcm_out.empty();
}

static float rms(const float* samples, int n_samples) {
    if (n_samples <= 0) return 0.0f;
    double sum_sq = 0.0;
    for (int sample_idx = 0; sample_idx < n_samples; sample_idx++) {
        sum_sq += (double)samples[sample_idx] * samples[sample_idx];
    }
    return std::sqrt((float)(sum_sq / n_samples));
}

static std::pair<int, int> center_window(int start_sample, int n_samples,
                                         float window_sec, int total_samples) {
    int clipped_start = std::max(0, start_sample);
    int clipped_end = std::min(total_samples, start_sample + n_samples);
    int clipped_n = clipped_end - clipped_start;
    if (window_sec <= 0.0f || clipped_n <= (int)(window_sec * kSampleRate)) {
        return {clipped_start, std::max(0, clipped_n)};
    }
    int window_samples = (int)(window_sec * kSampleRate);
    int center = clipped_start + clipped_n / 2;
    int window_start = std::max(0, center - window_samples / 2);
    if (window_start + window_samples > total_samples) {
        window_start = std::max(0, total_samples - window_samples);
    }
    return {window_start, std::min(window_samples, total_samples - window_start)};
}

class DualExtractor {
public:
    bool init(const std::string& campp_path, const std::string& wavlm_path) {
        SpeakerEncoderConfig campp_cfg;
        campp_cfg.model_path = campp_path;
        if (!campp_.init(campp_cfg)) return false;
        if (!fbank_.init(80, 400, 160, 512, kSampleRate, FbankWindowType::POVEY, true)) return false;
        return wavlm_.init(wavlm_path);
    }

    std::vector<float> extract(const float* samples, int n_samples) {
        if (n_samples < 4800) return {};
        fbank_.reset();
        std::vector<int16_t> int_samples(n_samples);
        for (int sample_idx = 0; sample_idx < n_samples; sample_idx++) {
            float value = std::max(-1.0f, std::min(1.0f, samples[sample_idx]));
            int_samples[sample_idx] = (int16_t)std::lrintf(value * 32767.0f);
        }
        fbank_.push_pcm(int_samples.data(), n_samples);
        int frames = fbank_.frames_ready();
        if (frames < 30) return {};
        std::vector<float> mel(frames * 80);
        int read_frames = fbank_.read_fbank(mel.data(), frames);
        if (read_frames <= 0) return {};
        auto campp_emb = campp_.extract(mel.data(), read_frames);
        auto wavlm_emb = wavlm_.extract(samples, n_samples);
        if (campp_emb.size() != 192 || wavlm_emb.size() != 192) return {};
        std::vector<float> dual(kDualDim);
        std::copy(campp_emb.begin(), campp_emb.end(), dual.begin());
        std::copy(wavlm_emb.begin(), wavlm_emb.end(), dual.begin() + 192);
        float norm_sq = 0.0f;
        for (float value : dual) norm_sq += value * value;
        float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
        for (float& value : dual) value *= inv_norm;
        return dual;
    }

private:
    SpeakerEncoder campp_;
    WavLMEcapaEncoder wavlm_;
    PoveyFbankGpu fbank_;
};

static std::vector<fs::path> collect_source_wavs(const fs::path& audio_dir, int limit) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(audio_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") continue;
        std::string name = entry.path().filename().string();
        if (name.find("_src1.wav") == std::string::npos &&
            name.find("_src2.wav") == std::string::npos) continue;
        int rank = 0;
        size_t rank_pos = name.find("rank_");
        if (rank_pos != std::string::npos) rank = std::atoi(name.substr(rank_pos + 5, 2).c_str());
        if (limit > 0 && rank > limit) continue;
        paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

static std::string clip_key_from_name(const std::string& name) {
    std::string key = name;
    const char* suffixes[] = {"_src1.wav", "_src2.wav"};
    for (const char* suffix : suffixes) {
        size_t pos = key.rfind(suffix);
        if (pos != std::string::npos) return key.substr(0, pos) + ".wav";
    }
    return key;
}

static std::string stream_from_name(const std::string& name) {
    if (name.find("_src1.wav") != std::string::npos) return "src1";
    if (name.find("_src2.wav") != std::string::npos) return "src2";
    return "unknown";
}

} // namespace

int main(int argc, char** argv) {
    fs::path audio_dir = "logs/separatio_param_sweep_r1/separatio_raw_o16000/audio";
    fs::path out_jsonl = "logs/separatio_param_sweep_r1/asr_raw_o16000/spkid_sources.jsonl";
    fs::path gt_path = "tests/fixtures/test_ground_truth_v1.jsonl";
    fs::path manifest_path = "logs/segment_homogeneity_clips_r3/clip_manifest.jsonl";
    fs::path reference_audio = "tests/test.mp3";
    std::string campp_model = home_dir() + "/models/dev/speaker/campplus/campplus.safetensors";
    std::string wavlm_model = home_dir() + "/models/dev/speaker/espnet_wavlm_ecapa/wavlm_ecapa.safetensors";
    int limit = 30;
    int max_ref_per_speaker = 8;
    int ref_min_ms = 1200;
    float ref_window_sec = 2.5f;
    float source_window_sec = 0.0f;
    float min_rms = 0.005f;
    float match_threshold = 0.35f;
    float min_margin = 0.03f;

    for (int arg_idx = 1; arg_idx < argc; arg_idx++) {
        std::string arg = argv[arg_idx];
        auto next = [&](const char* name) -> std::string {
            if (arg_idx + 1 >= argc) { fprintf(stderr, "missing %s\n", name); std::exit(2); }
            return argv[++arg_idx];
        };
        if (arg == "--audio-dir") audio_dir = next("--audio-dir");
        else if (arg == "--out") out_jsonl = next("--out");
        else if (arg == "--gt") gt_path = next("--gt");
        else if (arg == "--manifest") manifest_path = next("--manifest");
        else if (arg == "--reference-audio") reference_audio = next("--reference-audio");
        else if (arg == "--campp-model") campp_model = next("--campp-model");
        else if (arg == "--wavlm-model") wavlm_model = next("--wavlm-model");
        else if (arg == "--limit") limit = std::atoi(next("--limit").c_str());
        else if (arg == "--max-ref-per-speaker") max_ref_per_speaker = std::atoi(next("--max-ref-per-speaker").c_str());
        else if (arg == "--ref-min-ms") ref_min_ms = std::atoi(next("--ref-min-ms").c_str());
        else if (arg == "--ref-window-sec") ref_window_sec = std::atof(next("--ref-window-sec").c_str());
        else if (arg == "--source-window-sec") source_window_sec = std::atof(next("--source-window-sec").c_str());
        else if (arg == "--min-rms") min_rms = std::atof(next("--min-rms").c_str());
        else if (arg == "--match-threshold") match_threshold = std::atof(next("--match-threshold").c_str());
        else if (arg == "--min-margin") min_margin = std::atof(next("--min-margin").c_str());
    }

    auto gt_rows = load_gt(gt_path);
    auto clip_windows = load_manifest_windows(manifest_path);
    std::vector<float> reference_pcm;
    if (!read_audio_ffmpeg(reference_audio, reference_pcm)) {
        fprintf(stderr, "reference decode failed: %s\n", reference_audio.string().c_str());
        return 1;
    }

    DualExtractor extractor;
    if (!extractor.init(campp_model, wavlm_model)) {
        fprintf(stderr, "speaker encoder init failed\n");
        return 1;
    }

    SpeakerVectorStore speaker_db{"SeparatioSpkIdDb", kDualDim, 0.15f};
    std::map<std::string, int> speaker_to_id;
    std::map<int, std::string> id_to_speaker;
    std::map<std::string, int> ref_counts;
    int refs_attempted = 0;
    int refs_used = 0;

    for (const auto& row : gt_rows) {
        if (ref_counts[row.speaker] >= max_ref_per_speaker) continue;
        if (row.end_ms - row.start_ms < ref_min_ms) continue;
        if (overlaps_manifest(row, clip_windows)) continue;
        if (has_other_speaker_overlap(row, gt_rows)) continue;
        auto window = center_window((int)(row.start_ms * kSampleRate / 1000),
                                    (int)((row.end_ms - row.start_ms) * kSampleRate / 1000),
                                    ref_window_sec, (int)reference_pcm.size());
        if (window.second < 4800) continue;
        if (rms(reference_pcm.data() + window.first, window.second) < min_rms) continue;
        refs_attempted++;
        auto embedding = extractor.extract(reference_pcm.data() + window.first, window.second);
        if (embedding.empty()) continue;
        auto id_iter = speaker_to_id.find(row.speaker);
        if (id_iter == speaker_to_id.end()) {
            int speaker_id = speaker_db.register_speaker(row.speaker, embedding);
            speaker_to_id[row.speaker] = speaker_id;
            id_to_speaker[speaker_id] = row.speaker;
        } else {
            speaker_db.add_exemplar(id_iter->second, embedding);
        }
        ref_counts[row.speaker]++;
        refs_used++;
    }

    auto source_paths = collect_source_wavs(audio_dir, limit);
    fs::create_directories(out_jsonl.parent_path());
    std::ofstream output(out_jsonl);
    if (!output) {
        fprintf(stderr, "cannot write %s\n", out_jsonl.string().c_str());
        return 1;
    }

    printf("[refs] attempted=%d used=%d speakers=%d\n", refs_attempted, refs_used, speaker_db.count());
    for (const auto& item : ref_counts) printf("[ref] %s=%d\n", item.first.c_str(), item.second);

    int processed = 0;
    for (const auto& path : source_paths) {
        std::vector<float> pcm;
        if (!read_audio_ffmpeg(path, pcm)) continue;
        int total_samples = (int)pcm.size();
        auto window = center_window(0, total_samples, source_window_sec, total_samples);
        float source_rms = rms(pcm.data() + window.first, window.second);
        SpeakerMatch match;
        bool extracted = false;
        if (source_rms >= min_rms) {
            auto embedding = extractor.extract(pcm.data() + window.first, window.second);
            if (!embedding.empty()) {
                extracted = true;
                match = speaker_db.peek_best(embedding);
            }
        }
        float margin = match.similarity - match.second_best_sim;
        bool accepted = extracted && match.speaker_id >= 0 &&
                        match.similarity >= match_threshold && margin >= min_margin;
        std::string pred_name = accepted ? match.name : "?";
        std::string second_name = "?";
        auto second_iter = id_to_speaker.find(match.second_best_id);
        if (second_iter != id_to_speaker.end()) second_name = second_iter->second;
        std::string name = path.filename().string();
        output << "{"
               << "\"clip\":\"" << json_escape(clip_key_from_name(name)) << "\","
               << "\"stream\":\"" << stream_from_name(name) << "\","
               << "\"file\":\"" << json_escape(path.string()) << "\","
               << "\"duration_sec\":" << (total_samples / (float)kSampleRate) << ","
               << "\"window_sec\":" << (window.second / (float)kSampleRate) << ","
               << "\"rms\":" << source_rms << ","
               << "\"extracted\":" << (extracted ? "true" : "false") << ","
               << "\"accepted\":" << (accepted ? "true" : "false") << ","
               << "\"speaker_id\":" << (accepted ? match.speaker_id : -1) << ","
               << "\"pred_speaker\":\"" << json_escape(pred_name) << "\","
               << "\"best_raw_speaker\":\"" << json_escape(match.name) << "\","
               << "\"similarity\":" << match.similarity << ","
               << "\"second_speaker\":\"" << json_escape(second_name) << "\","
               << "\"second_similarity\":" << match.second_best_sim << ","
               << "\"margin\":" << margin << ","
               << "\"match_threshold\":" << match_threshold << ","
               << "\"min_margin\":" << min_margin
               << "}\n";
        processed++;
        printf("[%03d/%03zu] %s %s pred=%s sim=%.3f margin=%.3f rms=%.4f\n",
               processed, source_paths.size(), clip_key_from_name(name).c_str(),
               stream_from_name(name).c_str(), pred_name.c_str(), match.similarity,
               margin, source_rms);
    }
    printf("[out] %s\n", out_jsonl.string().c_str());
    return 0;
}