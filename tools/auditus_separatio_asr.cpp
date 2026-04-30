/**
 * @file auditus_separatio_asr.cpp
 * @philosophical_role Offline ASR probe for separated Auditus streams. It asks whether divided hearing still carries words.
 * @serves Auditus separation evaluation, ASR/Orator shadow strategy.
 */

#include "src/sensus/auditus/asr/asr_engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace deusridet;

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

static bool read_audio_ffmpeg(const fs::path& path, std::vector<float>& pcm_out) {
    std::string cmd = "ffmpeg -hide_banner -loglevel error -i " + shell_quote(path) +
                      " -ar 16000 -ac 1 -f s16le -";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;

    pcm_out.clear();
    int16_t buffer[4096];
    size_t read_count = 0;
    while ((read_count = fread(buffer, sizeof(int16_t), 4096, fp)) > 0) {
        size_t old_size = pcm_out.size();
        pcm_out.resize(old_size + read_count);
        for (size_t i = 0; i < read_count; i++) {
            pcm_out[old_size + i] = buffer[i] / 32768.0f;
        }
    }
    int status = pclose(fp);
    return status == 0 && !pcm_out.empty();
}

static int rank_from_name(const std::string& name) {
    size_t rank_pos = name.find("rank_");
    if (rank_pos == std::string::npos || name.size() < rank_pos + 7) return 0;
    return std::atoi(name.substr(rank_pos + 5, 2).c_str());
}

static std::vector<fs::path> collect_wavs(const fs::path& audio_dir, bool include_mix, int limit, int only_rank) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(audio_dir)) {
        if (!entry.is_regular_file()) continue;
        fs::path path = entry.path();
        std::string name = path.filename().string();
        if (path.extension() != ".wav") continue;
        bool is_source = name.find("_src1.wav") != std::string::npos ||
                         name.find("_src2.wav") != std::string::npos;
        bool is_mix = name.find("_mix.wav") != std::string::npos &&
                      name.find("_rawmix.wav") == std::string::npos;
        if (!is_source && !(include_mix && is_mix)) continue;
        int rank = rank_from_name(name);
        if (only_rank > 0 && rank != only_rank) continue;
        paths.push_back(path);
    }
    std::sort(paths.begin(), paths.end());
    if (limit > 0) {
        std::vector<fs::path> limited;
        for (const auto& path : paths) {
            std::string name = path.filename().string();
            int rank = rank_from_name(name);
            if (rank > 0 && rank <= limit) limited.push_back(path);
        }
        paths.swap(limited);
    }
    return paths;
}

static std::string clip_key_from_name(const std::string& name) {
    std::string key = name;
    const char* suffixes[] = {"_src1.wav", "_src2.wav", "_mix.wav"};
    for (const char* suffix : suffixes) {
        size_t pos = key.rfind(suffix);
        if (pos != std::string::npos) return key.substr(0, pos) + ".wav";
    }
    return key;
}

static std::string stream_from_name(const std::string& name) {
    if (name.find("_src1.wav") != std::string::npos) return "src1";
    if (name.find("_src2.wav") != std::string::npos) return "src2";
    if (name.find("_mix.wav") != std::string::npos) return "mix";
    return "unknown";
}

int main(int argc, char** argv) {
    fs::path audio_dir = "logs/separatio_examen_window_r1/audio";
    fs::path out_jsonl = "logs/separatio_asr_window_r1/asr_sources.jsonl";
    std::string asr_model = std::string(getenv("HOME") ? getenv("HOME") : "/home/rm01") +
                            "/models/dev/asr/Qwen/Qwen3-ASR-1.7B";
    int limit = 30;
    int only_rank = 0;
    int max_new_tokens = 96;
    bool include_mix = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--audio-dir" && i + 1 < argc) audio_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_jsonl = argv[++i];
        else if (arg == "--asr-model" && i + 1 < argc) asr_model = argv[++i];
        else if (arg == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
        else if (arg == "--rank" && i + 1 < argc) only_rank = std::atoi(argv[++i]);
        else if (arg == "--max-new-tokens" && i + 1 < argc) max_new_tokens = std::atoi(argv[++i]);
        else if (arg == "--sources-only") include_mix = false;
    }

    auto paths = collect_wavs(audio_dir, include_mix, limit, only_rank);
    if (paths.empty()) {
        fprintf(stderr, "No source wavs found in %s\n", audio_dir.string().c_str());
        return 1;
    }

    fs::create_directories(out_jsonl.parent_path());
    std::ofstream out(out_jsonl);
    if (!out) {
        fprintf(stderr, "Cannot write %s\n", out_jsonl.string().c_str());
        return 1;
    }

    asr::ASREngine engine;
    engine.load_model(asr_model);
    if (!engine.is_loaded()) {
        fprintf(stderr, "ASR load failed: %s\n", asr_model.c_str());
        return 1;
    }
    engine.set_repetition_penalty(1.05f);

    for (size_t index = 0; index < paths.size(); index++) {
        const fs::path& path = paths[index];
        std::vector<float> pcm;
        if (!read_audio_ffmpeg(path, pcm)) {
            fprintf(stderr, "decode failed: %s\n", path.string().c_str());
            continue;
        }
        auto result = engine.transcribe(pcm.data(), (int)pcm.size(), 16000, max_new_tokens);
        std::string name = path.filename().string();
        std::string clip_key = clip_key_from_name(name);
        std::string stream = stream_from_name(name);
        float duration_sec = pcm.size() / 16000.0f;

        out << "{"
            << "\"clip\":\"" << json_escape(clip_key) << "\","
            << "\"stream\":\"" << stream << "\","
            << "\"file\":\"" << json_escape(path.string()) << "\","
            << "\"duration_sec\":" << duration_sec << ","
            << "\"text\":\"" << json_escape(result.text) << "\","
            << "\"raw_text\":\"" << json_escape(result.raw_text) << "\","
            << "\"total_ms\":" << result.total_ms << ","
            << "\"mel_ms\":" << result.mel_ms << ","
            << "\"encoder_ms\":" << result.encoder_ms << ","
            << "\"decode_ms\":" << result.decode_ms << ","
            << "\"tokens\":" << result.token_count
            << "}\n";
        out.flush();

        printf("[%03zu/%03zu] %s/%s dur=%.2fs tokens=%d text=%s\n",
               index + 1, paths.size(), clip_key.c_str(), stream.c_str(),
               duration_sec, result.token_count, result.text.c_str());
    }

    printf("[out] %s\n", out_jsonl.string().c_str());
    return 0;
}
