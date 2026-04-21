/**
 * @file src/sensus/auditus/audio_pipeline_process.cpp
 * @philosophical_role
 *   AudioPipeline::process_loop — the auditus mainloop.
 *   Per-chunk phases (gain/RMS, FRCRN, Silero, FSMN, speaker pipeline,
 *   tracker pipe, ASR, Mel/VAD, stats) are invoked here in order; heavy
 *   phases are extracted to peer TUs so each stage stays independently
 *   inspectable and trace-taggable (Step 11 A1).
 *     A1a — process_asr_pipeline_ → audio_pipeline_process_asr.cpp
 *     A1b — speaker pipeline (next)
 * @serves
 *   Sensus auditus pipeline.
 */
#include "audio_pipeline.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"
#include "../../orator/spectral_cluster.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

void AudioPipeline::process_loop() {
    // Process chunk size in samples.
    int chunk_samples = cfg_.mel.sample_rate * cfg_.process_chunk_ms / 1000;
    size_t chunk_bytes = chunk_samples * sizeof(int16_t);
    std::vector<int16_t> pcm_buf(chunk_samples);

    // Host buffer for reading back Mel frames for VAD.
    int n_mels = cfg_.mel.n_mels;
    std::vector<float> mel_host(n_mels);

    // Silero VAD processes 512-sample windows from float PCM.
    int silero_window = silero_.initialized() ? cfg_.silero.window_samples : 0;
    std::vector<float> pcm_float;  // reused buffer for gain-applied float PCM
    std::vector<float> silero_buf; // carries remainder samples across chunks

    LOG_INFO("AudioPipe", "Process loop: chunk=%d samples (%d ms), frcrn_loaded=%s frcrn_enabled=%s silero_loaded=%s silero_enabled=%s fsmn_loaded=%s fsmn_enabled=%s",
             chunk_samples, cfg_.process_chunk_ms,
             frcrn_.initialized() ? "ON" : "OFF",
             enable_frcrn_.load(std::memory_order_relaxed) ? "ON" : "OFF",
             silero_.initialized() ? "ON" : "OFF",
             enable_silero_.load(std::memory_order_relaxed) ? "ON" : "OFF",
             fsmn_.initialized() ? "ON" : "OFF",
             enable_fsmn_.load(std::memory_order_relaxed) ? "ON" : "OFF");

    int diag_counter = 0;

    while (running_.load(std::memory_order_relaxed)) {
        size_t avail = ring_->available();

        if (avail < chunk_bytes) {
            // Not enough data yet — sleep briefly.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Pull PCM from ring.
        size_t got = ring_->pop(reinterpret_cast<uint8_t*>(pcm_buf.data()),
                                chunk_bytes);
        int n_samples = got / sizeof(int16_t);
        // Advance AUDIO T1: the authoritative "now" on the processing side.
        audio_t1_processed_ += (uint64_t)n_samples;
        stats_.audio_t1_processed = audio_t1_processed_;
        stats_.audio_t1_in        = audio_t1_in_.load(std::memory_order_relaxed);

        // Apply gain before processing.
        float g = gain_.load(std::memory_order_relaxed);
        if (g != 1.0f) {
            for (int i = 0; i < n_samples; i++) {
                int32_t s = (int32_t)(pcm_buf[i] * g);
                pcm_buf[i] = (int16_t)std::max(-32768, std::min(32767, s));
            }
        }

        // Compute RMS from PCM for stats (fast, on host).
        double sum_sq = 0;
        for (int i = 0; i < n_samples; i++) {
            float s = pcm_buf[i] / 32768.0f;
            sum_sq += s * s;
        }
        stats_.last_rms = n_samples > 0 ? sqrtf((float)(sum_sq / n_samples)) : 0;

        // FRCRN speech enhancement: denoise PCM before all downstream processing.
        // Uses direct per-chunk enhancement (no accumulation latency).
        // FRCRN internally pads to valid STFT alignment.
        //
        // NOTE: the speaker tracker also needs the *pre-FRCRN* signal for
        // MossFormer2 separation in overlap regions (S3: FRCRN suppresses
        // the weaker speaker, which defeats separation). We stash a raw
        // copy here and hand it to tracker_.feed() alongside the denoised
        // buffer below. The copy is skipped entirely when FRCRN is off.
        stats_.frcrn_active = false;
        std::vector<int16_t> pcm_raw_buf;
        if (frcrn_.initialized() && enable_frcrn_.load(std::memory_order_relaxed)) {
            pcm_raw_buf.assign(pcm_buf.begin(), pcm_buf.begin() + n_samples);
            frcrn_.enhance_inplace(pcm_buf.data(), n_samples);
            stats_.frcrn_active = true;
            stats_.frcrn_lat_ms = frcrn_.last_latency_ms();
        }

        // Run Silero VAD on raw PCM (512-sample windows).
        if (silero_.initialized() && silero_window > 0 &&
            enable_silero_.load(std::memory_order_relaxed)) {
            // Convert int16 -> float for Silero (reuse buffer).
            pcm_float.resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                pcm_float[i] = pcm_buf[i] / 32768.0f;
            }
            // Append new samples to remainder from previous chunk.
            silero_buf.insert(silero_buf.end(), pcm_float.begin(),
                              pcm_float.begin() + n_samples);
            // Process in silero_window-sized chunks.
            int consumed = 0;
            while (consumed + silero_window <= (int)silero_buf.size()) {
                SileroVadResult svr = silero_.process(
                    silero_buf.data() + consumed, silero_window);
                consumed += silero_window;
                stats_.silero_prob = svr.probability;
                stats_.silero_speech = svr.is_speech;
                // Use Silero result as authoritative VAD if available.
                stats_.is_speech = svr.is_speech;
                if (on_vad_ && (svr.segment_start || svr.segment_end)) {
                    VadResult vr{};
                    vr.is_speech = svr.is_speech;
                    vr.segment_start = svr.segment_start;
                    vr.segment_end = svr.segment_end;
                    vr.energy = svr.probability;  // repurpose energy field for prob
                    on_vad_(vr, (int)stats_.mel_frames, audio_t1_processed_);
                }
            }
            // Keep remainder for next chunk.
            if (consumed > 0) {
                silero_buf.erase(silero_buf.begin(),
                                 silero_buf.begin() + consumed);
            }
        }

        // Run FSMN VAD on raw PCM chunk (accumulates fbank internally).
        if (fsmn_.initialized() &&
            enable_fsmn_.load(std::memory_order_relaxed)) {
            FsmnVadResult fvr = fsmn_.process(pcm_buf.data(), n_samples);
            stats_.fsmn_prob = fvr.probability;
            stats_.fsmn_speech = fvr.is_speech;
        }

        // Buffer PCM for speaker identification during speech (VAD-gated).
        bool any_speaker_enabled =
            (speaker_enc_.initialized() && enable_speaker_.load(std::memory_order_relaxed)) ||
            (wlecapa_enc_.initialized() && enable_wlecapa_.load(std::memory_order_relaxed));
        bool need_segment_pcm = any_speaker_enabled;

        // Clear active flags each tick — only set true when extraction happens.
        stats_.speaker_active = false;
        stats_.wlecapa_active = false;
        stats_.wlecapa_change_valid = false;
        stats_.asr_active = false;

        if (need_segment_pcm) {
            // Determine speech state from selected VAD source.
            VadSource src = static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed));
            bool vad_speech = false;
            switch (src) {
                case VadSource::SILERO: vad_speech = stats_.silero_speech; break;
                case VadSource::FSMN:   vad_speech = stats_.fsmn_speech; break;
                case VadSource::ANY:
                default:
                    vad_speech = stats_.is_speech || stats_.silero_speech ||
                                 stats_.fsmn_speech;
                    break;
            }
            if (vad_speech && !in_speech_segment_) {
                in_speech_segment_ = true;
                early_extracted_   = false;
                campp_early_extracted_ = false;
                seg_has_ref_       = false;
                seg_last_recheck_at_ = 0;
                seg_ref_emb_.clear();
                seg_ref_speaker_id_ = -1;
                seg_ref_speaker_name_.clear();
                seg_ref_speaker_sim_ = 0.0f;
                speech_pcm_buf_.clear();
                speaker_fbank_.reset();
                seg_fbank_buf_.clear();
                seg_overlap_chunks_ = 0;
                seg_total_chunks_ = 0;

                // SAAS: short-segment speaker inheritance.
                // If this new segment starts within a short gap of the previous one,
                // pre-populate the speaker ID so that very short utterances (< 1.0s)
                // that can't extract their own embedding get a reasonable speaker label.
                int64_t gap_samples = audio_t1_processed_ - prev_seg_end_t1_;
                float gap_sec = gap_samples / 16000.0f;
                if (prev_seg_speaker_id_ >= 0 && gap_sec < 0.8f) {
                    // Populate both CAM++ and WL-ECAPA stats for inheritance.
                    stats_.speaker_id = prev_seg_speaker_id_;
                    stats_.speaker_sim = prev_seg_speaker_sim_;
                    stats_.speaker_new = false;
                    strncpy(stats_.speaker_name, prev_seg_speaker_name_.c_str(),
                            sizeof(stats_.speaker_name) - 1);
                    stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                    stats_.wlecapa_id = prev_seg_speaker_id_;
                    stats_.wlecapa_sim = prev_seg_speaker_sim_;
                    stats_.wlecapa_new = false;
                    strncpy(stats_.wlecapa_name, prev_seg_speaker_name_.c_str(),
                            sizeof(stats_.wlecapa_name) - 1);
                    stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                    seg_ref_speaker_id_ = prev_seg_speaker_id_;
                    seg_ref_speaker_name_ = prev_seg_speaker_name_;
                    seg_ref_speaker_sim_ = prev_seg_speaker_sim_;
                    LOG_INFO("AudioPipe", "SAAS: inherited spk=%d (%s) from prev segment (gap=%.3fs)",
                             prev_seg_speaker_id_, prev_seg_speaker_name_.c_str(), gap_sec);
                    // Timeline: SAAS inheritance event (covers ~2s from onset).
                    {
                        SpeakerEvent ev{};
                        ev.audio_start = audio_t1_processed_;
                        ev.audio_end   = audio_t1_processed_ + 32000;  // 2s look-ahead
                        ev.source      = SpkEventSource::SAAS_INHERIT;
                        ev.speaker_id  = prev_seg_speaker_id_;
                        ev.similarity  = prev_seg_speaker_sim_;
                        strncpy(ev.name, prev_seg_speaker_name_.c_str(), sizeof(ev.name) - 1);
                        spk_timeline_.push(ev);
                    }
                } else {
                    // Reset speaker ID for new segment.
                    stats_.speaker_id = -1;
                    stats_.speaker_sim = 0.0f;
                    stats_.speaker_new = false;
                    stats_.speaker_name[0] = '\0';
                    stats_.wlecapa_id = -1;
                    stats_.wlecapa_sim = 0.0f;
                    stats_.wlecapa_new = false;
                    strncpy(stats_.wlecapa_name, "", sizeof(stats_.wlecapa_name));
                }
            }
            if (in_speech_segment_) {
                speech_pcm_buf_.insert(speech_pcm_buf_.end(),
                                       pcm_buf.data(), pcm_buf.data() + n_samples);
                // Track overlap during this speech segment.
                seg_total_chunks_++;
                if (stats_.overlap_detected) seg_overlap_chunks_++;
                if ((speaker_enc_.initialized() &&
                     enable_speaker_.load(std::memory_order_relaxed))) {
                    speaker_fbank_.push_pcm(pcm_buf.data(), n_samples);
                    // Accumulate fbank frames for CAM++ EARLY/FULL extraction.
                    int avail = speaker_fbank_.frames_ready();
                    if (avail > 0) {
                        size_t old_sz = seg_fbank_buf_.size();
                        seg_fbank_buf_.resize(old_sz + avail * 80);
                        speaker_fbank_.read_fbank(seg_fbank_buf_.data() + old_sz, avail);
                    }
                }
                // Limit to 10 seconds (160000 samples @ 16kHz).
                if (speech_pcm_buf_.size() > 160000) {
                    speech_pcm_buf_.erase(speech_pcm_buf_.begin(),
                                          speech_pcm_buf_.begin() + n_samples);
                }

                // CAM++ EARLY extraction: when >= 150 fbank frames accumulated
                // during speech, extract embedding and match against existing speakers.
                // No auto-register — avoids spurious registrations from short clips.
                if (!campp_early_extracted_ &&
                    !use_dual_encoder_ &&  // skip EARLY speaker when dual-encoder active
                    enable_early_.load(std::memory_order_relaxed) &&
                    speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed)) {
                    int fbank_frames = (int)(seg_fbank_buf_.size() / 80);
                    if (fbank_frames >= 150) {
                        campp_early_extracted_ = true;
                        auto emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
                        if (!emb.empty()) {
                            float thresh = speaker_threshold_.load(std::memory_order_relaxed);
                            float reg_thresh = speaker_register_threshold_.load(std::memory_order_relaxed);
                            SpeakerMatch match = campp_db_.identify(emb, thresh, /*auto_register=*/false, reg_thresh);
                            if (match.speaker_id >= 0) {
                                stats_.speaker_id = match.speaker_id;
                                stats_.speaker_sim = match.similarity;
                                stats_.speaker_new = false;
                                stats_.speaker_count = campp_db_.count();
                                strncpy(stats_.speaker_name, match.name.c_str(),
                                        sizeof(stats_.speaker_name) - 1);
                                stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                                seg_ref_speaker_id_ = match.speaker_id;
                                seg_ref_speaker_name_ = match.name;
                                seg_ref_speaker_sim_ = match.similarity;
                                LOG_INFO("AudioPipe", "CAM++(early): id=%d sim=%.3f %s (fbank=%d)",
                                         match.speaker_id, match.similarity,
                                         match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                         fbank_frames);
                                if (on_speaker_) on_speaker_(match);
                                // Timeline: SAAS early event.
                                SpeakerEvent ev{};
                                ev.audio_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                ev.audio_end   = audio_t1_processed_;
                                ev.source      = SpkEventSource::SAAS_EARLY;
                                ev.speaker_id  = match.speaker_id;
                                ev.similarity  = match.similarity;
                                strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                                spk_timeline_.push(ev);
                            } else {
                                LOG_INFO("AudioPipe", "CAM++(early): no match (best_sim=%.3f, fbank=%d)",
                                         match.similarity, fbank_frames);
                            }
                        }
                    }
                }

                // Early extraction: run WL-ECAPA once we have enough speech,
                // without waiting for end-of-segment. This reduces "time to light-up".
                // IMPORTANT: auto_register=false — never create new speakers from
                // short early clips. If no match, report "identifying" state.
                int early_thresh = early_trigger_samples_.load(std::memory_order_relaxed);
                if (!early_extracted_ &&
                    enable_early_.load(std::memory_order_relaxed) &&
                    wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed) &&
                    (int)speech_pcm_buf_.size() >= early_thresh) {
                    early_extracted_ = true;
                    int early_samples = (int)speech_pcm_buf_.size();
                    std::vector<float> pcm_f32(early_samples);
                    for (int i = 0; i < early_samples; i++)
                        pcm_f32[i] = speech_pcm_buf_[i] / 32768.0f;
                    auto emb = wlecapa_enc_.extract(pcm_f32.data(), early_samples);
                    if (!emb.empty()) {
                        // Store as reference for intra-segment speaker change detection.
                        seg_ref_emb_ = emb;
                        seg_has_ref_ = true;
                        seg_last_recheck_at_ = (int)speech_pcm_buf_.size();

                        float thresh = wlecapa_threshold_.load(std::memory_order_relaxed);
                        // No auto-registration: match only against existing speakers.
                        SpeakerMatch match = wlecapa_db_.identify(emb, thresh, /*auto_register=*/false);
                        stats_.wlecapa_active = true;
                        stats_.wlecapa_is_early = true;
                        stats_.wlecapa_lat_cnn_ms     = wlecapa_enc_.last_lat_cnn_ms();
                        stats_.wlecapa_lat_encoder_ms = wlecapa_enc_.last_lat_encoder_ms();
                        stats_.wlecapa_lat_ecapa_ms   = wlecapa_enc_.last_lat_ecapa_ms();
                        stats_.wlecapa_lat_total_ms   = wlecapa_enc_.last_lat_total_ms();
                        if (match.speaker_id >= 0) {
                            // Matched an existing speaker — light up immediately.
                            stats_.wlecapa_id = match.speaker_id;
                            stats_.wlecapa_sim = match.similarity;
                            stats_.wlecapa_new = false;
                            stats_.wlecapa_count = wlecapa_db_.count();
                            stats_.wlecapa_exemplars = match.exemplar_count;
                            stats_.wlecapa_hits_above = match.hits_above;
                            // SAAS: track speaker ref for ASR annotation and inheritance.
                            // When dual encoder active, EARLY uses wlecapa_db_ (different ID space
                            // from dual_db_). Don't contaminate seg_ref or timeline.
                            if (!use_dual_encoder_) {
                                seg_ref_speaker_id_ = match.speaker_id;
                                seg_ref_speaker_name_ = match.name;
                                seg_ref_speaker_sim_ = match.similarity;
                            }
                            strncpy(stats_.wlecapa_name, match.name.c_str(),
                                    sizeof(stats_.wlecapa_name) - 1);
                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "WL-ECAPA(early): id=%d sim=%.3f %s (%.2fs, %.1fms)%s",
                                     match.speaker_id, match.similarity,
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     early_samples / 16000.0f,
                                     wlecapa_enc_.last_lat_total_ms(),
                                     use_dual_encoder_ ? " [skip timeline: dual mode]" : "");
                            if (on_speaker_) on_speaker_(match);
                            // Timeline: SAAS early extraction event.
                            // Skip when dual encoder active — wlecapa_db_ IDs != dual_db_ IDs.
                            if (!use_dual_encoder_) {
                                SpeakerEvent ev{};
                                ev.audio_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                ev.audio_end   = audio_t1_processed_;
                                ev.source      = SpkEventSource::SAAS_EARLY;
                                ev.speaker_id  = match.speaker_id;
                                ev.similarity  = match.similarity;
                                strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
                                spk_timeline_.push(ev);
                            }
                        } else {
                            // No match — signal "identifying" to UI.
                            stats_.wlecapa_id = -1;
                            stats_.wlecapa_sim = match.similarity;
                            stats_.wlecapa_new = false;
                            strncpy(stats_.wlecapa_name, "(identifying)",
                                    sizeof(stats_.wlecapa_name) - 1);
                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "WL-ECAPA(early): no match (best_sim=%.3f, %.2fs, %.1fms) — awaiting full segment",
                                     match.similarity, early_samples / 16000.0f,
                                     wlecapa_enc_.last_lat_total_ms());
                        }
                    }
                }

                // Intra-segment speaker change detection: periodically re-extract
                // an embedding from the recent audio window and compare against the
                // segment's reference speaker. If similarity drops below threshold,
                // force a segment boundary — this catches speaker transitions that
                // VAD misses (rapid turn-taking without silence).
                if (early_extracted_ && seg_has_ref_ &&
                    enable_spk_recheck_.load(std::memory_order_relaxed) &&
                    wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed)) {
                    int recheck_interval = spk_recheck_samples_.load(std::memory_order_relaxed);
                    int buf_sz = (int)speech_pcm_buf_.size();
                    if (buf_sz - seg_last_recheck_at_ >= recheck_interval) {
                        seg_last_recheck_at_ = buf_sz;
                        int win_samples = spk_recheck_window_samples_.load(std::memory_order_relaxed);
                        int start = std::max(0, buf_sz - win_samples);
                        int len = buf_sz - start;
                        if (len >= 16000) { // at least 1s for meaningful embedding
                            std::vector<float> pcm_f32(len);
                            for (int i = 0; i < len; i++)
                                pcm_f32[i] = speech_pcm_buf_[start + i] / 32768.0f;
                            auto emb = wlecapa_enc_.extract(pcm_f32.data(), len);
                            if (!emb.empty() && emb.size() == seg_ref_emb_.size()) {
                                // Cosine similarity between reference and current window.
                                float dot = 0, na = 0, nb = 0;
                                for (size_t i = 0; i < emb.size(); i++) {
                                    dot += seg_ref_emb_[i] * emb[i];
                                    na  += seg_ref_emb_[i] * seg_ref_emb_[i];
                                    nb  += emb[i] * emb[i];
                                }
                                float sim = (na > 0 && nb > 0) ? dot / (sqrtf(na) * sqrtf(nb)) : 0.0f;
                                float change_thresh = spk_change_threshold_.load(std::memory_order_relaxed);
                                LOG_INFO("AudioPipe", "SPK-RECHECK: sim=%.3f (thresh=%.3f) at %.2fs in segment",
                                         sim, change_thresh, buf_sz / 16000.0f);
                                if (sim < change_thresh) {
                                    // Speaker changed mid-segment! Soft restart:
                                    // 1. Run full WL-ECAPA on pre-change audio (auto_register=true)
                                    // 2. Save speaker state for inheritance
                                    // 3. Trigger ASR split
                                    // 4. Keep in_speech_segment_=true, carry tail audio forward
                                    LOG_INFO("AudioPipe", "SPK-CHANGE detected (sim=%.3f < %.3f) — soft restart at %.2fs",
                                             sim, change_thresh, buf_sz / 16000.0f);

                                    // --- 1. Full WL-ECAPA on pre-change audio ---
                                    int pre_samples = start;
                                    int pre_min = min_speech_samples_.load(std::memory_order_relaxed);
                                    if (pre_samples >= pre_min) {
                                        std::vector<float> pre_f32(pre_samples);
                                        for (int i = 0; i < pre_samples; i++)
                                            pre_f32[i] = speech_pcm_buf_[i] / 32768.0f;
                                        auto pre_emb = wlecapa_enc_.extract(pre_f32.data(), pre_samples);
                                        if (!pre_emb.empty()) {
                                            float wt = wlecapa_threshold_.load(std::memory_order_relaxed);
                                            SpeakerMatch m = wlecapa_db_.identify(pre_emb, wt);
                                            // When dual encoder active, don't pollute seg_ref with
                                            // wlecapa_db_ IDs (different ID space from dual_db_).
                                            if (!use_dual_encoder_) {
                                                seg_ref_speaker_id_ = m.speaker_id;
                                                seg_ref_speaker_name_ = m.name;
                                                seg_ref_speaker_sim_ = m.similarity;
                                            }
                                            stats_.wlecapa_id = m.speaker_id;
                                            stats_.wlecapa_sim = m.similarity;
                                            stats_.wlecapa_new = m.is_new;
                                            stats_.wlecapa_count = wlecapa_db_.count();
                                            stats_.wlecapa_exemplars = m.exemplar_count;
                                            stats_.wlecapa_hits_above = m.hits_above;
                                            stats_.wlecapa_is_early = false;
                                            strncpy(stats_.wlecapa_name, m.name.c_str(),
                                                    sizeof(stats_.wlecapa_name) - 1);
                                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                                            LOG_INFO("AudioPipe", "SPK-CHANGE full-extract: id=%d sim=%.3f %s%s (%.2fs)",
                                                     m.speaker_id, m.similarity,
                                                     m.is_new ? "NEW " : "",
                                                     m.name.empty() ? "(unnamed)" : m.name.c_str(),
                                                     pre_samples / 16000.0f);
                                            if (on_speaker_) on_speaker_(m);
                                            prev_wlecapa_emb_ = pre_emb;
                                            // Timeline: SAAS speaker change event.
                                            // Skip when dual encoder active — wlecapa_db_ IDs != dual_db_ IDs.
                                            if (!use_dual_encoder_) {
                                                int64_t seg_start = audio_t1_processed_ - (int64_t)speech_pcm_buf_.size();
                                                SpeakerEvent ev{};
                                                ev.audio_start = seg_start;
                                                ev.audio_end   = seg_start + pre_samples;
                                                ev.source      = SpkEventSource::SAAS_CHANGE;
                                                ev.speaker_id  = m.speaker_id;
                                                ev.similarity  = m.similarity;
                                                strncpy(ev.name, m.name.c_str(), sizeof(ev.name) - 1);
                                                spk_timeline_.push(ev);
                                            }
                                        }
                                    }

                                    // --- 2. Save speaker state for inheritance ---
                                    prev_seg_end_t1_ = audio_t1_processed_;
                                    if (seg_ref_speaker_id_ >= 0) {
                                        prev_seg_speaker_id_ = seg_ref_speaker_id_;
                                        prev_seg_speaker_name_ = seg_ref_speaker_name_;
                                        prev_seg_speaker_sim_ = seg_ref_speaker_sim_;
                                    }

                                    // --- 3. Trigger ASR split ---
                                    int tail_samples = buf_sz - start;
                                    int asr_buf_sz = (int)asr_pcm_buf_.size();
                                    int split_at = std::max(0, asr_buf_sz - tail_samples);
                                    if (asr_saw_speech_ && split_at > 0 && split_at < asr_buf_sz) {
                                        asr_spk_change_pending_ = true;
                                        asr_spk_change_split_at_ = split_at;
                                        LOG_INFO("AudioPipe", "SAAS: ASR split queued at sample %d/%d (%.2fs)",
                                                 split_at, asr_buf_sz, split_at / 16000.0f);
                                    }

                                    // --- 4. Soft restart: carry tail audio, reset for new speaker ---
                                    std::vector<int16_t> tail(speech_pcm_buf_.begin() + start,
                                                              speech_pcm_buf_.end());
                                    speech_pcm_buf_ = std::move(tail);
                                    // New speaker's embedding becomes ref
                                    seg_ref_emb_ = emb;
                                    seg_has_ref_ = true;
                                    early_extracted_ = true;
                                    seg_last_recheck_at_ = (int)speech_pcm_buf_.size();
                                    seg_ref_speaker_id_ = -1;
                                    seg_ref_speaker_name_.clear();
                                    seg_ref_speaker_sim_ = 0.0f;
                                    // Reset fbank for new sub-segment
                                    speaker_fbank_.reset();
                                    if (((speaker_enc_.initialized() &&
                                        enable_speaker_.load(std::memory_order_relaxed))) &&
                                        !speech_pcm_buf_.empty()) {
                                        speaker_fbank_.push_pcm(speech_pcm_buf_.data(),
                                                                (int)speech_pcm_buf_.size());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (!vad_speech && in_speech_segment_) {
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
        }

        // SpeakerTracker: independent continuous pipeline.
        // Uses same VAD source as SAAS for consistency but tracks independently.
        {
            VadSource src = static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed));
            bool tracker_vad = false;
            switch (src) {
                case VadSource::SILERO: tracker_vad = stats_.silero_speech; break;
                case VadSource::FSMN:   tracker_vad = stats_.fsmn_speech; break;
                case VadSource::ANY:
                default:
                    tracker_vad = stats_.is_speech || stats_.silero_speech ||
                                  stats_.fsmn_speech;
                    break;
            }
            tracker_.feed(pcm_buf.data(), n_samples, tracker_vad,
                          pcm_raw_buf.empty() ? nullptr : pcm_raw_buf.data());
            if (tracker_.check()) {
                // Timeline: Tracker identification event.
                // Skip when dual encoder active — tracker uses its own DB with
                // different speaker IDs from dual_db_.
                auto& ts = tracker_.stats();
                if (ts.speaker_id >= 0 && !use_dual_encoder_) {
                    int win = tracker_.window_ms() * 16;  // window in samples
                    SpeakerEvent ev{};
                    ev.audio_start = audio_t1_processed_ - win;
                    ev.audio_end   = audio_t1_processed_;
                    ev.source      = SpkEventSource::TRACKER;
                    ev.speaker_id  = ts.speaker_id;
                    ev.similarity  = ts.speaker_sim;
                    strncpy(ev.name, ts.speaker_name, sizeof(ev.name) - 1);
                    spk_timeline_.push(ev);
                }
            }
            // Copy P1/P2 stats from tracker to pipeline stats.
            {
                auto& ts = tracker_.stats();
                stats_.overlap_detected   = ts.overlap_detected;
                stats_.overlap_ratio      = ts.overlap_ratio;
                stats_.od_latency_ms      = ts.od_latency_ms;
                stats_.separation_active  = ts.separation_active;
                stats_.separation_lat_ms  = ts.separation_lat_ms;
                stats_.sep_source1_energy = ts.sep_source1_energy;
                stats_.sep_source2_energy = ts.sep_source2_energy;
            }
        }

        // ASR pipeline — continuous accumulation + VAD/speaker-change triggered split.
        // Extracted to audio_pipeline_process_asr.cpp (Step 11 A1a).
        process_asr_pipeline_(pcm_buf.data(), n_samples);

        // Update ASR buffer stats.
        stats_.asr_buf_sec = asr_pcm_buf_.size() / 16000.0f;
        stats_.asr_buf_has_speech = asr_saw_speech_;
        stats_.asr_busy = asr_busy_.load(std::memory_order_relaxed);

        // Push to Mel spectrogram (GPU).
        int new_frames = mel_.push_pcm(pcm_buf.data(), n_samples);
        stats_.mel_frames += new_frames;

        if (new_frames <= 0) continue;

        // Run VAD on new frames.
        //   Copy new Mel frames back to host one at a time for state machine.
        int start_frame = mel_.frames_ready() - new_frames;
        for (int i = 0; i < new_frames; i++) {
            int frame_idx = start_frame + i;
            cudaMemcpy(mel_host.data(),
                       mel_.mel_buffer() + frame_idx * n_mels,
                       n_mels * sizeof(float),
                       cudaMemcpyDeviceToHost);

            VadResult vr = vad_.process_frame(mel_host.data(), n_mels);
            stats_.last_energy = vr.energy;
            if (vr.is_speech) stats_.speech_frames++;

            // Energy VAD drives is_speech only when Silero is not available.
            if (!silero_.initialized()) {
                stats_.is_speech = vr.is_speech;
                if (on_vad_ && (vr.segment_start || vr.segment_end)) {
                    on_vad_(vr, frame_idx, audio_t1_processed_);
                }
            }
        }

        // Report stats.
        if (on_stats_) {
            on_stats_(stats_);
        }

        // Periodic diagnostic log (~every 1s = 10 chunks at 100ms).
        if (++diag_counter % 10 == 0) {
            LOG_INFO("AudioPipe", "DIAG rms=%.4f silero=%.3f fsmn=%.3f speech=%d gain=%.1f spk=%d(%.2f)",
                     stats_.last_rms, stats_.silero_prob,
                     stats_.fsmn_prob,
                     (int)stats_.is_speech,
                     gain_.load(std::memory_order_relaxed),
                     stats_.speaker_id, stats_.speaker_sim);
        }
    }
}


} // namespace deusridet
