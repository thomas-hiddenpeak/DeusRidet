// test_overlap_spkid.cpp — test overlap detection → separation → speaker ID pipeline.
//
// Reads an audio file via ffmpeg, scans for overlapping regions using P1 (pyannote),
// separates overlapping segments using P2 (MossFormer2), identifies speakers on
// each separated source using CAM++ and WavLM-ECAPA dual encoder, and reports
// accuracy against the known speaker count (4 speakers in test.mp3).
//
// Usage:
//   ./build/test_overlap_spkid <audio_file> [--dump-wav]
//
// Output:
//   For each overlap region:
//     - Time range, overlap ratio
//     - P2 separated source energies
//     - Speaker ID for each source (CAM++ 192D and WL-ECAPA 192D)
//     - Similarity scores

#include "../src/sensus/auditus/overlap_detector.h"
#include "../src/sensus/auditus/speech_separator.h"
#include "../src/sensus/auditus/fsmn_fbank_gpu.h"
#include "../src/orator/speaker_encoder.h"
#include "../src/orator/wavlm_ecapa_encoder.h"
#include "../src/orator/speaker_vector_store.h"
#include "../src/communis/log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace deusridet;

static constexpr int SR = 16000;

// ──────── Read audio via ffmpeg ────────

static bool read_audio_ffmpeg(const std::string& path,
                              std::vector<int16_t>& pcm_out) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -hide_banner -loglevel error -i \"%s\" "
             "-ar 16000 -ac 1 -f s16le -",
             path.c_str());
    FILE* fp = popen(cmd, "r");
    if (!fp) return false;

    pcm_out.clear();
    pcm_out.reserve(SR * 3700);  // ~60 min
    int16_t buf[4096];
    size_t n;
    while ((n = fread(buf, sizeof(int16_t), 4096, fp)) > 0)
        pcm_out.insert(pcm_out.end(), buf, buf + n);
    pclose(fp);
    return !pcm_out.empty();
}

// ──────── WAV writer ────────

static void write_wav(const std::string& path, const float* pcm, int n_samples) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return;
    int32_t data_size = n_samples * 2;
    int16_t bps = 16;
    int32_t file_size = 36 + data_size;
    int32_t sr = SR, byte_rate = SR * 2;
    int16_t channels = 1, block = 2;
    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVEfmt ", 1, 8, fp);
    int32_t fmt_size = 16; fwrite(&fmt_size, 4, 1, fp);
    int16_t fmt = 1; fwrite(&fmt, 2, 1, fp);
    fwrite(&channels, 2, 1, fp);
    fwrite(&sr, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block, 2, 1, fp);
    fwrite(&bps, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);
    for (int i = 0; i < n_samples; i++) {
        int16_t s = (int16_t)std::max(-32768.0f, std::min(32767.0f, pcm[i] * 32767.0f));
        fwrite(&s, 2, 1, fp);
    }
    fclose(fp);
}

static float compute_rms(const float* pcm, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)pcm[i] * pcm[i];
    return sqrtf((float)(s / n));
}

// ──────── Audio region ────────

struct OverlapRegion {
    int start_sample;  // absolute sample offset in full audio
    int end_sample;
    float overlap_ratio;  // ratio of overlapping frames in this region
};

struct SingleSpkRegion {
    int start_sample;
    int end_sample;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio_file> [--dump-wav]\n", argv[0]);
        return 1;
    }

    std::string audio_path = argv[1];
    bool dump_wav = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dump-wav") == 0) dump_wav = true;
    }

    // ──── 1. Read audio ────
    printf("Loading audio: %s\n", audio_path.c_str());
    std::vector<int16_t> pcm_i16;
    if (!read_audio_ffmpeg(audio_path, pcm_i16)) {
        fprintf(stderr, "Failed to read audio\n");
        return 1;
    }
    int total_samples = (int)pcm_i16.size();
    float duration_s = (float)total_samples / SR;
    printf("  Samples: %d (%.1fs)\n", total_samples, duration_s);

    // Convert to float32.
    std::vector<float> pcm_f32(total_samples);
    for (int i = 0; i < total_samples; i++)
        pcm_f32[i] = pcm_i16[i] / 32768.0f;
    pcm_i16.clear();  // free memory

    // ──── 2. Init P1: Overlap Detector ────
    printf("\nInitializing P1 (Overlap Detector)...\n");
    OverlapDetectorConfig od_cfg;
    od_cfg.model_path = std::string(getenv("HOME")) + "/models/dev/vad/pyannote_seg3.safetensors";
    od_cfg.overlap_threshold = 0.5f;
    od_cfg.chunk_samples = 160000;  // 10s
    od_cfg.hop_samples   = 80000;   // 5s

    OverlapDetector od;
    if (!od.init(od_cfg)) {
        fprintf(stderr, "Failed to init overlap detector\n");
        return 1;
    }
    printf("  P1 ready: %d frames, %d classes\n",
           od.num_output_frames(), od.num_classes());

    // ──── 3. Init P2: Speech Separator ────
    printf("Initializing P2 (Speech Separator)...\n");
    SpeechSeparatorConfig sep_cfg;
    sep_cfg.model_path = std::string(getenv("HOME")) + "/models/dev/vad/mossformer2_ss_16k.safetensors";
    sep_cfg.max_chunk = 32000;
    sep_cfg.overlap_samples = 3200;
    sep_cfg.lazy_load = false;  // load now

    SpeechSeparator sep;
    if (!sep.init(sep_cfg)) {
        fprintf(stderr, "Failed to init speech separator\n");
        return 1;
    }
    printf("  P2 ready\n");

    // ──── 4. Init Speaker Encoders ────
    printf("Initializing speaker encoders...\n");

    // CAM++ (192D) — needs 80-dim fbank
    SpeakerEncoderConfig campp_cfg;
    campp_cfg.model_path = std::string(getenv("HOME")) +
        "/models/dev/speaker/campplus/campplus.safetensors";
    SpeakerEncoder campp_enc;
    if (!campp_enc.init(campp_cfg)) {
        fprintf(stderr, "Failed to init CAM++ encoder\n");
        return 1;
    }
    printf("  CAM++ ready (192D)\n");

    // FBank for CAM++ (80-dim, Povey window)
    FsmnFbankGpu fbank;
    if (!fbank.init(80, 400, 160, 512, SR, FbankWindowType::POVEY, true)) {
        fprintf(stderr, "Failed to init fbank\n");
        return 1;
    }

    // WavLM-ECAPA (192D) — takes raw PCM
    WavLMEcapaEncoder wlecapa_enc;
    if (!wlecapa_enc.init(std::string(getenv("HOME")) +
        "/models/dev/speaker/espnet_wavlm_ecapa/wavlm_ecapa.safetensors")) {
        fprintf(stderr, "Failed to init WavLM-ECAPA encoder\n");
        return 1;
    }
    printf("  WavLM-ECAPA ready (192D)\n");

    // Dual 384D speaker store
    SpeakerVectorStore spk_db{"OverlapTestDb", 384, 0.15f};

    // ──── 5. Scan for overlaps with P1 ────
    printf("\nScanning for overlaps (P1)...\n");
    auto t0 = std::chrono::steady_clock::now();

    // Process 10s windows with 5s hop.
    int chunk = od_cfg.chunk_samples;   // 160000 (10s)
    int hop   = od_cfg.hop_samples;     // 80000 (5s)
    int num_windows = 0;
    int total_overlap_frames = 0;
    int total_frames = 0;

    // Per-frame overlap map (1 frame = 16ms for pyannote default).
    // pyannote seg3 produces 587 frames per 10s window.
    // With 5s hop, center 5s = ~294 frames per window.
    float frame_dur = 10.0f / od.num_output_frames();  // seconds per frame
    int frames_per_hop = od.num_output_frames() / 2;    // center frames to keep

    // Collect overlap regions as contiguous blocks.
    // Also collect single-speaker regions for pre-enrollment.
    std::vector<OverlapRegion> regions;
    std::vector<SingleSpkRegion> single_regions;
    bool in_overlap = false;
    int region_start = 0;
    int overlap_frame_count = 0;
    int region_frame_count = 0;

    bool in_single = false;
    int single_start = 0;

    for (int offset = 0; offset + chunk <= total_samples; offset += hop) {
        auto result = od.detect(pcm_f32.data() + offset, chunk);
        num_windows++;

        // Use center frames (skip border artifacts).
        int frame_start = (offset == 0) ? 0 : frames_per_hop / 2;
        int frame_end = std::min(result.num_frames,
                                 frame_start + frames_per_hop);

        for (int f = frame_start; f < frame_end; f++) {
            total_frames++;
            bool is_ovlp = result.frame_overlap[f];
            if (is_ovlp) total_overlap_frames++;

            int num_spk = (f < (int)result.frame_num_spk.size())
                          ? result.frame_num_spk[f] : 0;

            // Compute absolute sample position for this frame.
            int abs_sample = offset + (int)(f * frame_dur * SR);

            // Track overlap regions.
            if (is_ovlp && !in_overlap) {
                in_overlap = true;
                region_start = abs_sample;
                overlap_frame_count = 1;
                region_frame_count = 1;
            } else if (is_ovlp && in_overlap) {
                overlap_frame_count++;
                region_frame_count++;
            } else if (!is_ovlp && in_overlap) {
                int region_end = abs_sample;
                float ratio = (float)overlap_frame_count / region_frame_count;
                if (region_end - region_start >= (int)(0.3f * SR)) {
                    regions.push_back({region_start, region_end, ratio});
                }
                in_overlap = false;
                overlap_frame_count = 0;
                region_frame_count = 0;
            }

            // Track single-speaker regions.
            if (num_spk == 1 && !in_single) {
                in_single = true;
                single_start = abs_sample;
            } else if (num_spk != 1 && in_single) {
                int single_end = abs_sample;
                // Keep segments >= 1.5s for stable embeddings.
                if (single_end - single_start >= (int)(1.5f * SR)) {
                    single_regions.push_back({single_start, single_end});
                }
                in_single = false;
            }
        }
    }

    // Close any open region.
    if (in_overlap && overlap_frame_count > 0) {
        int region_end = total_samples;
        float ratio = (float)overlap_frame_count / region_frame_count;
        if (region_end - region_start >= (int)(0.3f * SR)) {
            regions.push_back({region_start, region_end, ratio});
        }
    }
    if (in_single) {
        int single_end = total_samples;
        if (single_end - single_start >= (int)(1.5f * SR)) {
            single_regions.push_back({single_start, single_end});
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    float scan_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("  Scanned %d windows in %.1fs\n", num_windows, scan_ms / 1000.0f);
    printf("  Total frames: %d, overlap frames: %d (%.1f%%)\n",
           total_frames, total_overlap_frames,
           100.0f * total_overlap_frames / std::max(1, total_frames));
    printf("  Overlap regions (>0.3s): %d\n", (int)regions.size());

    // Merge close regions (gap < 0.5s).
    std::vector<OverlapRegion> merged;
    for (auto& r : regions) {
        if (!merged.empty() && r.start_sample - merged.back().end_sample < (int)(0.5f * SR)) {
            merged.back().end_sample = r.end_sample;
            merged.back().overlap_ratio = std::max(merged.back().overlap_ratio, r.overlap_ratio);
        } else {
            merged.push_back(r);
        }
    }
    printf("  After merge: %d regions\n", (int)merged.size());
    printf("  Single-speaker regions (>=1.5s): %d\n", (int)single_regions.size());

    if (merged.empty()) {
        printf("\nNo overlap regions found. Done.\n");
        return 0;
    }

    // ──── 6. Phase 1: Pre-enroll speakers from clean single-speaker audio ────
    printf("\n=== Phase 1: Speaker enrollment from clean audio ===\n");

    // Helper: extract embedding from float PCM using dual encoder.
    auto extract_dual = [&](const float* pcm_data, int n) -> std::vector<float> {
        if (n < 4800) return {};  // need at least 0.3s

        // CAM++ path: PCM → int16 → fbank → CAM++
        fbank.reset();
        std::vector<int16_t> tmp_i16(n);
        for (int i = 0; i < n; i++)
            tmp_i16[i] = (int16_t)std::max(-32768.0f, std::min(32767.0f, pcm_data[i] * 32767.0f));
        fbank.push_pcm(tmp_i16.data(), n);
        int nf = fbank.frames_ready();
        if (nf < 30) return {};  // too few frames
        std::vector<float> fb(nf * 80);
        fbank.read_fbank(fb.data(), nf);
        auto campp_emb = campp_enc.extract(fb.data(), nf);
        if (campp_emb.empty()) return {};

        // WavLM-ECAPA: raw PCM → embedding
        auto wl_emb = wlecapa_enc.extract(pcm_data, n);
        if (wl_emb.empty()) return {};

        // Concatenate + L2 normalize → 384D
        std::vector<float> dual(384);
        std::copy(campp_emb.begin(), campp_emb.end(), dual.begin());
        std::copy(wl_emb.begin(), wl_emb.end(), dual.begin() + 192);
        float norm2 = 0;
        for (float v : dual) norm2 += v * v;
        float inv = 1.0f / sqrtf(norm2 + 1e-12f);
        for (float& v : dual) v *= inv;
        return dual;
    };

    {
        // Process all single-speaker segments (up to 200) for robust enrollment.
        // Use lower thresholds since segments are non-consecutive and may vary.
        int max_enroll = std::min((int)single_regions.size(), 200);
        int step = std::max(1, (int)single_regions.size() / max_enroll);
        int enrolled = 0;
        int skipped = 0;
        for (int i = 0; i < (int)single_regions.size() && enrolled < max_enroll; i += step) {
            auto& sr = single_regions[i];
            int n = sr.end_sample - sr.start_sample;
            // Use center 2s (or full segment if shorter) for best embedding.
            int use_n = std::min(n, 2 * SR);
            int use_start = sr.start_sample + (n - use_n) / 2;
            const float* pcm = pcm_f32.data() + use_start;

            float rms = compute_rms(pcm, use_n);
            if (rms < 0.01f) { skipped++; continue; }

            auto emb = extract_dual(pcm, use_n);
            if (emb.empty()) { skipped++; continue; }

            auto match = spk_db.identify(emb, 0.40f, true, 0.45f);
            enrolled++;

            if (enrolled <= 10 || match.is_new) {
                printf("  Enroll #%d @ %.1fs: spk=%d sim=%.3f%s\n",
                       enrolled, (float)use_start / SR,
                       match.speaker_id, match.similarity,
                       match.is_new ? " (NEW)" : "");
            }
        }
        printf("  Enrolled %d segments (%d skipped), Speaker DB: %d speakers, %d exemplars\n",
               enrolled, skipped, spk_db.count(), spk_db.total_exemplars());
    }

    // ──── 7. Phase 2: Process each overlap region: P2 → Speaker ID ────
    printf("\n=== Phase 2: Overlap separation + speaker identification ===\n\n");

    int overlap_idx = 0;
    int total_separated = 0;
    int both_identified = 0;
    int both_diff = 0;
    int at_least_one_identified = 0;
    float total_overlap_duration = 0;

    for (auto& region : merged) {
        int n = region.end_sample - region.start_sample;
        float dur = (float)n / SR;
        float t_start = (float)region.start_sample / SR;
        float t_end = (float)region.end_sample / SR;
        total_overlap_duration += dur;

        printf("Overlap #%d: %.1fs–%.1fs (%.1fs, ratio=%.0f%%)\n",
               overlap_idx, t_start, t_end, dur, region.overlap_ratio * 100);

        // Add context: extend 0.5s before and after for better separation.
        int ctx = (int)(0.5f * SR);
        int ext_start = std::max(0, region.start_sample - ctx);
        int ext_end = std::min(total_samples, region.end_sample + ctx);
        int ext_n = ext_end - ext_start;

        // P2: Separate.
        auto sep_t0 = std::chrono::steady_clock::now();
        auto sep_result = sep.separate(pcm_f32.data() + ext_start, ext_n);
        auto sep_t1 = std::chrono::steady_clock::now();
        float sep_ms = std::chrono::duration<float, std::milli>(sep_t1 - sep_t0).count();

        if (!sep_result.valid) {
            printf("  P2 separation FAILED\n\n");
            overlap_idx++;
            continue;
        }
        total_separated++;

        // Trim back to original region (remove context).
        int trim_start = region.start_sample - ext_start;
        int trim_n = n;
        const float* s1 = sep_result.source1.data() + trim_start;
        const float* s2 = sep_result.source2.data() + trim_start;

        float rms1 = compute_rms(s1, trim_n);
        float rms2 = compute_rms(s2, trim_n);

        printf("  P2: src1_rms=%.4f src2_rms=%.4f (%.1fms)\n", rms1, rms2, sep_ms);

        // Dump wavs if requested.
        if (dump_wav) {
            char fname[256];
            snprintf(fname, sizeof(fname), "/tmp/overlap_%03d_mix.wav", overlap_idx);
            write_wav(fname, pcm_f32.data() + region.start_sample, trim_n);
            snprintf(fname, sizeof(fname), "/tmp/overlap_%03d_src1.wav", overlap_idx);
            write_wav(fname, s1, trim_n);
            snprintf(fname, sizeof(fname), "/tmp/overlap_%03d_src2.wav", overlap_idx);
            write_wav(fname, s2, trim_n);
            printf("  Dumped: /tmp/overlap_%03d_{mix,src1,src2}.wav\n", overlap_idx);
        }

        // Speaker ID on each source.
        // Skip if source has very low energy (likely silence/noise).
        float min_energy = 0.005f;
        int spk1_id = -1, spk2_id = -1;
        float spk1_sim = 0, spk2_sim = 0;

        if (rms1 > min_energy) {
            auto emb1 = extract_dual(s1, trim_n);
            if (!emb1.empty()) {
                // Match only — no auto-register from separated audio.
                auto match1 = spk_db.identify(emb1, 0.35f, false);
                spk1_id = match1.speaker_id;
                spk1_sim = match1.similarity;
                printf("  Src1: spk=%d sim=%.3f\n", spk1_id, spk1_sim);
            } else {
                printf("  Src1: embedding extraction failed (too short?)\n");
            }
        } else {
            printf("  Src1: too quiet (rms=%.4f)\n", rms1);
        }

        if (rms2 > min_energy) {
            auto emb2 = extract_dual(s2, trim_n);
            if (!emb2.empty()) {
                // Match only — no auto-register from separated audio.
                auto match2 = spk_db.identify(emb2, 0.35f, false);
                spk2_id = match2.speaker_id;
                spk2_sim = match2.similarity;
                printf("  Src2: spk=%d sim=%.3f\n", spk2_id, spk2_sim);
            } else {
                printf("  Src2: embedding extraction failed (too short?)\n");
            }
        } else {
            printf("  Src2: too quiet (rms=%.4f)\n", rms2);
        }

        if (spk1_id >= 0 && spk2_id >= 0) {
            both_identified++;
            bool diff = (spk1_id != spk2_id);
            if (diff) both_diff++;
            printf("  → Two speakers identified: spk%d + spk%d %s\n",
                   spk1_id, spk2_id,
                   diff ? "(DIFFERENT)" : "(SAME)");
        } else if (spk1_id >= 0 || spk2_id >= 0) {
            at_least_one_identified++;
            printf("  → One speaker identified\n");
        } else {
            printf("  → No speakers identified\n");
        }

        printf("\n");
        overlap_idx++;
    }

    // ──── 8. Summary ────
    printf("=== SUMMARY ===\n");
    printf("Total audio: %.1fs\n", duration_s);
    printf("Overlap: %.1f%% of frames\n",
           100.0f * total_overlap_frames / std::max(1, total_frames));
    printf("Overlap regions: %d (total %.1fs)\n",
           (int)merged.size(), total_overlap_duration);
    printf("P2 separated: %d / %d\n", total_separated, (int)merged.size());
    printf("Both speakers identified: %d / %d (%.1f%%)\n",
           both_identified, total_separated,
           total_separated > 0 ? 100.0f * both_identified / total_separated : 0.0f);
    printf("  of which DIFFERENT: %d, SAME: %d\n",
           both_diff, both_identified - both_diff);
    printf("At least one identified: %d / %d (%.1f%%)\n",
           both_identified + at_least_one_identified, total_separated,
           total_separated > 0 ?
           100.0f * (both_identified + at_least_one_identified) / total_separated : 0.0f);
    printf("Speaker DB: %d speakers, %d total exemplars\n",
           spk_db.count(), spk_db.total_exemplars());

    // Print speaker distribution.
    auto speakers = spk_db.all_speakers();
    for (auto& s : speakers) {
        printf("  spk%d: %d exemplars, %d matches\n",
               s.id, s.exemplar_count, s.match_count);
    }

    return 0;
}
