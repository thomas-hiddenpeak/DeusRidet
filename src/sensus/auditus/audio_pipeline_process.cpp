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
                // SAAS during-speech: buffer + fbank + CAM++ EARLY + WL-EARLY + recheck.
                // Extracted to audio_pipeline_process_saas_during.cpp (Step 11 A1c-1).
                process_saas_during_speech_(pcm_buf.data(), n_samples);
            }
            if (!vad_speech && in_speech_segment_) {
                // SAAS end-of-segment bookkeeping + FULL extract dispatch + WL native.
                // Extracted to audio_pipeline_process_saas_segend.cpp (Step 11 A1c-2).
                process_saas_segment_end_();
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
