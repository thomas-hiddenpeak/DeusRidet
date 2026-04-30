/**
 * @file auditus_separatio_examen.cpp
 * @philosophical_role Offline examination of whether high-risk Auditus clips can be separated into two audible streams before any online behavior changes.
 * @serves Auditus overlap/separation evaluation, Orator speaker-attribution strategy.
 */

#include "src/sensus/auditus/speech_separator.h"
#include "src/sensus/auditus/frcrn_enhancer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace deusridet;

static constexpr int kSampleRate = 16000;

struct ClipResult {
    std::string name;
    int samples = 0;
    float duration_sec = 0.0f;
    bool valid = false;
    float latency_ms = 0.0f;
    float frcrn_latency_ms = 0.0f;
    float rtf = 0.0f;
    float raw_input_rms = 0.0f;
    float input_rms = 0.0f;
    float source1_rms = 0.0f;
    float source2_rms = 0.0f;
    float source1_peak = 0.0f;
    float source2_peak = 0.0f;
    float raw_energy1 = 0.0f;
    float raw_energy2 = 0.0f;
    float energy_balance = 0.0f;
    float mix_source1_corr = 0.0f;
    float mix_source2_corr = 0.0f;
    float source_corr = 0.0f;
    bool two_active = false;
};

static float rms(const std::vector<float>& pcm) {
    if (pcm.empty()) return 0.0f;
    double sum = 0.0;
    for (float value : pcm) sum += (double)value * value;
    return std::sqrt((float)(sum / pcm.size()));
}

static float peak_abs(const std::vector<float>& pcm) {
    float peak = 0.0f;
    for (float value : pcm) peak = std::max(peak, std::abs(value));
    return peak;
}

static void scale_in_place(std::vector<float>& pcm, float scale) {
    for (float& value : pcm) value *= scale;
}

static float corr(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0f;
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    if (aa <= 1e-12 || bb <= 1e-12) return 0.0f;
    return (float)(dot / std::sqrt(aa * bb));
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

static void write_wav_scaled(const fs::path& path, const std::vector<float>& pcm, float scale) {
    FILE* fp = fopen(path.string().c_str(), "wb");
    if (!fp) return;

    int32_t data_size = (int32_t)pcm.size() * 2;
    int32_t file_size = 36 + data_size;
    int32_t sample_rate = kSampleRate;
    int32_t byte_rate = kSampleRate * 2;
    int16_t channels = 1;
    int16_t block_align = 2;
    int16_t bits_per_sample = 16;
    int32_t fmt_size = 16;
    int16_t fmt = 1;

    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVEfmt ", 1, 8, fp);
    fwrite(&fmt_size, 4, 1, fp);
    fwrite(&fmt, 2, 1, fp);
    fwrite(&channels, 2, 1, fp);
    fwrite(&sample_rate, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bits_per_sample, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);
    for (float value : pcm) {
        float scaled = std::max(-1.0f, std::min(1.0f, value * scale));
        int16_t sample = (int16_t)std::lrintf(scaled * 32767.0f);
        fwrite(&sample, 2, 1, fp);
    }
    fclose(fp);
}

static void write_wav_auto_peak(const fs::path& path, const std::vector<float>& pcm) {
    float peak = peak_abs(pcm);
    float scale = peak > 0.98f ? 0.98f / peak : 1.0f;
    write_wav_scaled(path, pcm, scale);
}

static float pair_peak_scale(const std::vector<float>& a, const std::vector<float>& b) {
    float peak = std::max(peak_abs(a), peak_abs(b));
    if (peak <= 1e-8f) return 1.0f;
    return 0.98f / peak;
}

static std::vector<float> normalize_to_rms(const std::vector<float>& source,
                                           float target_rms) {
    std::vector<float> normalized = source;
    float source_rms = rms(normalized);
    if (source_rms > 1e-8f) scale_in_place(normalized, target_rms / source_rms);
    return normalized;
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

static std::vector<fs::path> collect_wavs(const fs::path& dir, int limit) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".wav") paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (limit > 0 && (int)paths.size() > limit) paths.resize(limit);
    return paths;
}

static void write_reports(const fs::path& out_dir, const std::vector<ClipResult>& results,
                          int max_chunk, int overlap_samples, bool use_frcrn,
                          float two_source_balance, float two_source_corr) {
    std::ofstream jsonl(out_dir / "separatio_results.jsonl");
    std::ofstream md(out_dir / "separatio_summary.md");

    int valid = 0;
    int two_active = 0;
    float total_duration = 0.0f;
    float total_latency = 0.0f;
    float total_frcrn_latency = 0.0f;
    for (const auto& row : results) {
        if (row.valid) valid++;
        if (row.two_active) two_active++;
        total_duration += row.duration_sec;
        total_latency += row.latency_ms;
        total_frcrn_latency += row.frcrn_latency_ms;
    }

    md << "# Auditus Separatio Examen\n\n";
    md << "Offline MossFormer2 probe over exported high-risk Auditus clips. "
       << "Outputs are inspection artifacts only; no online speaker label is changed. "
       << "`audio/` preserves source energy balance for listening; "
       << "`audio_official_rms/` keeps ClearVoice-style per-source RMS normalization.\n\n";
    md << "## Summary\n\n";
    md << "- clips: " << results.size() << "\n";
    md << "- valid separations: " << valid << "\n";
    md << "- two-source candidates: " << two_active << "\n";
    md << "- max chunk: " << max_chunk << " samples (" << std::fixed << std::setprecision(2)
       << max_chunk / (float)kSampleRate << "s)\n";
    md << "- overlap: " << overlap_samples << " samples (" << std::fixed << std::setprecision(2)
       << overlap_samples / (float)kSampleRate << "s)\n";
     md << "- two-source balance threshold: " << std::fixed << std::setprecision(3)
         << two_source_balance << "\n";
     md << "- two-source corr threshold: " << std::fixed << std::setprecision(3)
         << two_source_corr << "\n";
    md << "- FRCRN pre-enhancement: " << (use_frcrn ? "on" : "off") << "\n";
    md << "- audio duration: " << std::fixed << std::setprecision(2) << total_duration << "s\n";
    if (use_frcrn) {
        md << "- total FRCRN latency: " << std::fixed << std::setprecision(1)
           << total_frcrn_latency << "ms\n";
    }
    md << "- total latency: " << std::fixed << std::setprecision(1) << total_latency << "ms\n";
    if (total_duration > 0.0f) {
        md << "- aggregate RTF: " << std::fixed << std::setprecision(3)
           << total_latency / (total_duration * 1000.0f) << "\n";
    }
    md << "\n## Clips\n\n";
    md << "| Clip | Dur | RTF | In RMS | Raw E1 | Raw E2 | Balance | Src Corr | Two-Source |\n";
    md << "|------|----:|----:|-------:|-------:|-------:|--------:|---------:|:----------:|\n";

    for (const auto& row : results) {
        jsonl << "{"
              << "\"clip\":\"" << json_escape(row.name) << "\","
              << "\"samples\":" << row.samples << ","
              << "\"duration_sec\":" << row.duration_sec << ","
              << "\"valid\":" << (row.valid ? "true" : "false") << ","
              << "\"latency_ms\":" << row.latency_ms << ","
              << "\"frcrn_latency_ms\":" << row.frcrn_latency_ms << ","
              << "\"rtf\":" << row.rtf << ","
              << "\"raw_input_rms\":" << row.raw_input_rms << ","
              << "\"input_rms\":" << row.input_rms << ","
              << "\"source1_rms\":" << row.source1_rms << ","
              << "\"source2_rms\":" << row.source2_rms << ","
              << "\"source1_peak\":" << row.source1_peak << ","
              << "\"source2_peak\":" << row.source2_peak << ","
              << "\"raw_energy1\":" << row.raw_energy1 << ","
              << "\"raw_energy2\":" << row.raw_energy2 << ","
              << "\"energy_balance\":" << row.energy_balance << ","
              << "\"mix_source1_corr\":" << row.mix_source1_corr << ","
              << "\"mix_source2_corr\":" << row.mix_source2_corr << ","
              << "\"source_corr\":" << row.source_corr << ","
              << "\"two_active\":" << (row.two_active ? "true" : "false")
              << "}\n";

        md << "| " << row.name << " | " << std::fixed << std::setprecision(2)
           << row.duration_sec << " | " << std::setprecision(3) << row.rtf << " | "
              << std::setprecision(4) << row.input_rms << " | " << row.raw_energy1 << " | "
              << row.raw_energy2 << " | " << row.energy_balance << " | " << row.source_corr << " | "
              << (row.two_active ? "yes" : "no") << " |\n";
    }
}

int main(int argc, char** argv) {
    fs::path clips_dir = "logs/segment_homogeneity_clips_r3/clips";
    fs::path out_dir = "logs/separatio_examen_r1";
    std::string model_path = std::string(getenv("HOME")) + "/models/dev/vad/mossformer2_ss_16k.safetensors";
    std::string frcrn_weights = std::string(getenv("HOME")) + "/models/dev/vad/frcrn_weights";
    int limit = 30;
    int max_chunk = 32000;
    int overlap_samples = 16000;
    float two_source_balance = 0.05f;
    float two_source_corr = 0.92f;
    bool use_frcrn = false;
    constexpr int frcrn_window_samples = 160000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--clips-dir" && i + 1 < argc) clips_dir = argv[++i];
        else if (arg == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--frcrn-weights" && i + 1 < argc) frcrn_weights = argv[++i];
        else if (arg == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
        else if (arg == "--max-chunk" && i + 1 < argc) max_chunk = std::atoi(argv[++i]);
        else if (arg == "--overlap-samples" && i + 1 < argc) overlap_samples = std::atoi(argv[++i]);
        else if (arg == "--two-source-balance" && i + 1 < argc) two_source_balance = std::atof(argv[++i]);
        else if (arg == "--two-source-corr" && i + 1 < argc) two_source_corr = std::atof(argv[++i]);
        else if (arg == "--frcrn") use_frcrn = true;
    }

    if (max_chunk <= 0 || overlap_samples < 0 || overlap_samples >= max_chunk) {
        fprintf(stderr, "Invalid chunk parameters: max_chunk=%d overlap_samples=%d\n",
                max_chunk, overlap_samples);
        return 1;
    }

    auto clips = collect_wavs(clips_dir, limit);
    if (clips.empty()) {
        fprintf(stderr, "No wav clips found in %s\n", clips_dir.string().c_str());
        return 1;
    }

    fs::create_directories(out_dir / "audio");
    fs::create_directories(out_dir / "audio_official_rms");

    SpeechSeparatorConfig cfg;
    cfg.model_path = model_path;
    cfg.max_chunk = max_chunk;
    cfg.overlap_samples = overlap_samples;
    cfg.lazy_load = false;

    SpeechSeparator separator;
    if (!separator.init(cfg)) {
        fprintf(stderr, "SpeechSeparator init failed: %s\n", model_path.c_str());
        return 1;
    }

    FrcrnEnhancer frcrn;
    if (use_frcrn) {
        FrcrnConfig frcrn_cfg;
        frcrn_cfg.weights_dir = frcrn_weights;
        frcrn_cfg.chunk_samples = frcrn_window_samples;
        frcrn_cfg.hop_samples = 320;
        frcrn_cfg.enabled = true;
        if (!frcrn.init(frcrn_cfg)) {
            fprintf(stderr, "FRCRN init failed: %s\n", frcrn_weights.c_str());
            return 1;
        }
    }

    std::vector<ClipResult> results;
    for (const auto& clip : clips) {
        std::vector<float> mix;
        ClipResult row;
        row.name = clip.filename().string();
        if (!read_audio_ffmpeg(clip, mix)) {
            fprintf(stderr, "Failed to decode %s\n", clip.string().c_str());
            results.push_back(row);
            continue;
        }

        row.samples = (int)mix.size();
        row.duration_sec = row.samples / (float)kSampleRate;
        row.raw_input_rms = rms(mix);

        std::vector<float> model_input = mix;
        if (use_frcrn) {
            if (row.samples > frcrn_window_samples) {
                fprintf(stderr, "Clip exceeds FRCRN offline window: %s samples=%d max=%d\n",
                        clip.string().c_str(), row.samples, frcrn_window_samples);
                return 1;
            }
            model_input = frcrn.enhance(mix.data(), row.samples);
            row.frcrn_latency_ms = frcrn.last_latency_ms();
        }
        row.input_rms = rms(model_input);

        auto t0 = std::chrono::steady_clock::now();
        auto separated = separator.separate(model_input.data(), row.samples);
        auto t1 = std::chrono::steady_clock::now();
        row.latency_ms = row.frcrn_latency_ms + std::chrono::duration<float, std::milli>(t1 - t0).count();
        row.rtf = row.duration_sec > 0.0f ? row.latency_ms / (row.duration_sec * 1000.0f) : 0.0f;
        row.valid = separated.valid;

        if (separated.valid) {
            row.raw_energy1 = separated.energy1;
            row.raw_energy2 = separated.energy2;
            row.source1_rms = rms(separated.source1);
            row.source2_rms = rms(separated.source2);
            row.source1_peak = peak_abs(separated.source1);
            row.source2_peak = peak_abs(separated.source2);
            float larger_energy = std::max(row.raw_energy1, row.raw_energy2);
            float smaller_energy = std::min(row.raw_energy1, row.raw_energy2);
            row.energy_balance = larger_energy > 1e-8f ? smaller_energy / larger_energy : 0.0f;
            row.mix_source1_corr = corr(model_input, separated.source1);
            row.mix_source2_corr = corr(model_input, separated.source2);
            row.source_corr = corr(separated.source1, separated.source2);
            row.two_active = row.energy_balance >= two_source_balance &&
                             std::abs(row.source_corr) < two_source_corr;

            std::string stem = clip.stem().string();
            auto official1 = normalize_to_rms(separated.source1, row.input_rms);
            auto official2 = normalize_to_rms(separated.source2, row.input_rms);
            write_wav_auto_peak(out_dir / "audio_official_rms" / (stem + "_mix.wav"), model_input);
            write_wav_auto_peak(out_dir / "audio_official_rms" / (stem + "_src1.wav"), official1);
            write_wav_auto_peak(out_dir / "audio_official_rms" / (stem + "_src2.wav"), official2);

            auto raw1 = separated.source1;
            auto raw2 = separated.source2;
            float balance_scale = pair_peak_scale(raw1, raw2);
            if (use_frcrn) write_wav_auto_peak(out_dir / "audio" / (stem + "_rawmix.wav"), mix);
            write_wav_auto_peak(out_dir / "audio" / (stem + "_mix.wav"), model_input);
            write_wav_scaled(out_dir / "audio" / (stem + "_src1.wav"), raw1, balance_scale);
            write_wav_scaled(out_dir / "audio" / (stem + "_src2.wav"), raw2, balance_scale);
        }

        printf("[%02zu/%02zu] %s valid=%d dur=%.2fs rtf=%.3f frcrn=%.1fms raw=(%.4f,%.4f) balance=%.3f corr=%.3f two_source=%d\n",
               results.size() + 1, clips.size(), row.name.c_str(), row.valid ? 1 : 0,
               row.duration_sec, row.rtf, row.frcrn_latency_ms, row.raw_energy1, row.raw_energy2,
             row.energy_balance, row.source_corr, row.two_active ? 1 : 0);
        results.push_back(row);
    }

    write_reports(out_dir, results, cfg.max_chunk, cfg.overlap_samples, use_frcrn,
                  two_source_balance, two_source_corr);
    printf("[out] %s\n", (out_dir / "separatio_results.jsonl").string().c_str());
    printf("[out] %s\n", (out_dir / "separatio_summary.md").string().c_str());
    printf("[out] %s\n", (out_dir / "audio").string().c_str());
    printf("[out] %s\n", (out_dir / "audio_official_rms").string().c_str());
    return 0;
}