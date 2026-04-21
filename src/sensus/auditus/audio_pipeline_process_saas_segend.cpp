/**
 * @file src/sensus/auditus/audio_pipeline_process_saas_segend.cpp
 * @philosophical_role
 *   Stage-extract of AudioPipeline::process_loop (Step 11 A1c-2).
 *   End-of-segment SAAS bookkeeping: save prev-speaker state for inheritance,
 *   optional debug WAV dump, flush remaining fbank frames, dispatch to the
 *   CAM++ FULL extract stage (Step 11 A1b), and run the WL-ECAPA native path
 *   + change-detection + timeline push for the just-ended segment.
 * @serves
 *   Sensus auditus — SAAS end-of-segment identity arm (excluding the FULL
 *   extract body itself, which lives in audio_pipeline_process_saas_full.cpp).
 */
#include "audio_pipeline.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: close out a just-ended speech segment — save state, FULL CAM++, WL native, timeline.
void AudioPipeline::process_saas_segment_end_() {
                in_speech_segment_ = false;
                int speech_samples = (int)speech_pcm_buf_.size();
                float speech_duration = speech_samples / 16000.0f;

                // SAAS: save speaker state for short-segment inheritance.
                prev_seg_end_t1_ = audio_t1_processed_;
                if (seg_ref_speaker_id_ >= 0) {
                    prev_seg_speaker_id_ = seg_ref_speaker_id_;
                    prev_seg_speaker_name_ = seg_ref_speaker_name_;
                    prev_seg_speaker_sim_ = seg_ref_speaker_sim_;
                } else if (stats_.speaker_id >= 0) {
                    // Fallback: use CAM++ stats (primary encoder).
                    prev_seg_speaker_id_ = stats_.speaker_id;
                    prev_seg_speaker_name_ = stats_.speaker_name;
                    prev_seg_speaker_sim_ = stats_.speaker_sim;
                } else if (!use_dual_encoder_ && stats_.wlecapa_id >= 0) {
                    // Further fallback: use whatever WL-ECAPA ID was last reported.
                    // Skip when dual encoder active — wlecapa_db_ IDs != dual_db_ IDs.
                    prev_seg_speaker_id_ = stats_.wlecapa_id;
                    prev_seg_speaker_name_ = stats_.wlecapa_name;
                    prev_seg_speaker_sim_ = stats_.wlecapa_sim;
                }

                LOG_INFO("AudioPipe", "Speech segment ended: %.2fs (%d samples, spk=%d %s)",
                         speech_duration, speech_samples, prev_seg_speaker_id_,
                         prev_seg_speaker_name_.c_str());

                // Debug: dump first 10 speech segments to WAV for analysis.
                {
                    static int dump_count = 0;
                    if (dump_count < 10 && speech_samples >= 16000) {
                        char path[128];
                        snprintf(path, sizeof(path), "/tmp/spk_seg_%d.wav", dump_count);
                        FILE* f = fopen(path, "wb");
                        if (f) {
                            // Write minimal WAV header (16-bit mono 16kHz).
                            uint32_t data_sz = speech_samples * 2;
                            uint32_t file_sz = 36 + data_sz;
                            uint16_t fmt = 1; // PCM
                            uint16_t ch = 1;
                            uint32_t sr = 16000;
                            uint32_t bps = 32000;
                            uint16_t ba = 2;
                            uint16_t bits = 16;
                            fwrite("RIFF", 1, 4, f);
                            fwrite(&file_sz, 4, 1, f);
                            fwrite("WAVEfmt ", 1, 8, f);
                            uint32_t fmt_sz = 16;
                            fwrite(&fmt_sz, 4, 1, f);
                            fwrite(&fmt, 2, 1, f);
                            fwrite(&ch, 2, 1, f);
                            fwrite(&sr, 4, 1, f);
                            fwrite(&bps, 4, 1, f);
                            fwrite(&ba, 2, 1, f);
                            fwrite(&bits, 2, 1, f);
                            fwrite("data", 1, 4, f);
                            fwrite(&data_sz, 4, 1, f);
                            fwrite(speech_pcm_buf_.data(), 2, speech_samples, f);
                            fclose(f);
                            LOG_INFO("AudioPipe", "Dumped segment %d: %s (%.2fs)",
                                     dump_count, path, speech_duration);
                        }
                        dump_count++;
                    }
                }

                // Read remaining fbank features and append to segment accumulator.
                {
                    int avail = speaker_fbank_.frames_ready();
                    if (avail > 0) {
                        size_t old_sz = seg_fbank_buf_.size();
                        seg_fbank_buf_.resize(old_sz + avail * 80);
                        speaker_fbank_.read_fbank(seg_fbank_buf_.data() + old_sz, avail);
                    }
                }
                int fbank_frames = (int)(seg_fbank_buf_.size() / 80);

                // CAM++ FULL extract + dual-encoder fuse + spectral warmup.
                // Extracted to audio_pipeline_process_saas_full.cpp (Step 11 A1b).
                process_saas_full_extract_(fbank_frames);


                // WavLM-Large + ECAPA-TDNN native GPU speaker encoder (uses raw PCM).
                int min_spk_samples = min_speech_samples_.load(std::memory_order_relaxed);
                if (wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed) &&
                    speech_samples >= min_spk_samples) {
                    // Convert int16 PCM to float32 [-1, 1].
                    std::vector<float> pcm_f32(speech_samples);
                    for (int i = 0; i < speech_samples; i++)
                        pcm_f32[i] = speech_pcm_buf_[i] / 32768.0f;
                    auto emb = wlecapa_enc_.extract(pcm_f32.data(), speech_samples);
                    if (!emb.empty()) {
                        float enorm = 0;
                        for (float v : emb) enorm += v * v;
                        enorm = sqrtf(enorm);
                        LOG_INFO("AudioPipe", "WL-ECAPA emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                 enorm, emb[0], emb[1], emb[2], emb[3]);
                        float thresh = wlecapa_threshold_.load(std::memory_order_relaxed);
                        SpeakerMatch match = wlecapa_db_.identify(emb, thresh);

                        stats_.wlecapa_id = match.speaker_id;
                        stats_.wlecapa_sim = match.similarity;
                        stats_.wlecapa_new = match.is_new;
                        stats_.wlecapa_count = wlecapa_db_.count();
                        stats_.wlecapa_exemplars = match.exemplar_count;
                        stats_.wlecapa_hits_above = match.hits_above;
                        stats_.wlecapa_active = true;
                        stats_.wlecapa_is_early = false;
                        stats_.wlecapa_lat_cnn_ms     = wlecapa_enc_.last_lat_cnn_ms();
                        stats_.wlecapa_lat_encoder_ms = wlecapa_enc_.last_lat_encoder_ms();
                        stats_.wlecapa_lat_ecapa_ms   = wlecapa_enc_.last_lat_ecapa_ms();
                        stats_.wlecapa_lat_total_ms   = wlecapa_enc_.last_lat_total_ms();

                        // SAAS: track speaker ref for ASR annotation and inheritance.
                        // When dual encoder active, dual_db_ FULL path already sets seg_ref.
                        // Don't overwrite with wlecapa_db_ IDs.
                        if (!use_dual_encoder_) {
                            seg_ref_speaker_id_ = match.speaker_id;
                            seg_ref_speaker_name_ = match.name;
                            seg_ref_speaker_sim_ = match.similarity;
                        }

                        // Change detection: cosine similarity with previous segment embedding.
                        if (!prev_wlecapa_emb_.empty() && prev_wlecapa_emb_.size() == emb.size()) {
                            float dot = 0;
                            for (size_t j = 0; j < emb.size(); j++)
                                dot += emb[j] * prev_wlecapa_emb_[j];
                            stats_.wlecapa_change_sim = dot;  // both L2-normed → dot = cosine
                            stats_.wlecapa_change_valid = true;
                        } else {
                            stats_.wlecapa_change_sim = -1.0f;
                            stats_.wlecapa_change_valid = false;
                        }
                        prev_wlecapa_emb_ = emb;

                        strncpy(stats_.wlecapa_name, match.name.c_str(),
                                sizeof(stats_.wlecapa_name) - 1);
                        stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                        LOG_INFO("AudioPipe", "WL-ECAPA: id=%d sim=%.3f %s%s (%d samples, %.1fms)",
                                 match.speaker_id, match.similarity,
                                 match.is_new ? "NEW " : "",
                                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                 speech_samples, wlecapa_enc_.last_lat_total_ms());
                        if (on_speaker_) on_speaker_(match);
                        // Timeline: SAAS full end-of-segment event (highest authority).
                        // Skip when dual encoder active — dual_db_ FULL path already pushed.
                        if (!use_dual_encoder_) {
                            int64_t seg_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                            SpeakerEvent ev{};
                            ev.audio_start = seg_start;
                            ev.audio_end   = audio_t1_processed_;
                            ev.source      = SpkEventSource::SAAS_FULL;
                            ev.speaker_id  = match.speaker_id;
                            ev.similarity  = match.similarity;
                            strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                            spk_timeline_.push(ev);
                        }
                    }
                }

                speech_pcm_buf_.clear();
}

} // namespace deusridet
