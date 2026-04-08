// audio_pipeline.cpp — Real-time audio processing pipeline implementation.
//
// Processing thread: pull PCM from ring buffer → push to MelSpectrogram (GPU)
// → VAD on computed frames → report stats/events via callbacks.

#include "audio_pipeline.h"
#include "../../communis/log.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

AudioPipeline::AudioPipeline() = default;

AudioPipeline::~AudioPipeline() {
    stop();
    delete ring_;
}

bool AudioPipeline::start(const AudioPipelineConfig& cfg) {
    if (running_.load()) return false;
    cfg_ = cfg;

    // Initialize ring buffer.
    ring_ = new RingBuffer(cfg_.ring_buffer_bytes);

    // Initialize Mel spectrogram.
    if (!mel_.init(cfg_.mel)) {
        LOG_ERROR("AudioPipe", "Failed to init Mel spectrogram");
        return false;
    }

    // Initialize VAD.
    vad_.init(cfg_.vad);

    // Initialize Silero VAD (optional — non-fatal if model not found).
    if (!cfg_.silero.model_path.empty()) {
        if (!silero_.init(cfg_.silero)) {
            LOG_WARN("AudioPipe", "Silero VAD init failed — running energy-only VAD");
        }
    }

    // Initialize FSMN VAD (optional — non-fatal).
    if (!cfg_.fsmn.model_path.empty()) {
        if (!fsmn_.init(cfg_.fsmn)) {
            LOG_WARN("AudioPipe", "FSMN VAD init failed");
        }
    }

    // Initialize TEN VAD (optional — non-fatal).
    if (!cfg_.ten.model_path.empty()) {
        if (!ten_.init(cfg_.ten)) {
            LOG_WARN("AudioPipe", "TEN VAD init failed");
        }
    }

    // Initialize speaker encoder (optional — non-fatal).
    bool need_fbank = false;
    if (!cfg_.speaker.model_path.empty()) {
        if (!speaker_enc_.init(cfg_.speaker)) {
            LOG_WARN("AudioPipe", "Speaker encoder init failed");
        } else {
            need_fbank = true;
        }
    }

    // ECAPA-TDNN also requires fbank features.
    if (!cfg_.unispeech.model_path.empty()) {
        need_fbank = true;
    }

    if (need_fbank) {
        // Initialize 80-dim fbank (shared between CAM++ and ECAPA-TDNN).
        // Povey window + [-1,1] PCM normalization — matches WeSpeaker/Kaldi defaults.
        if (!speaker_fbank_.init(80, 400, 160, 512, 16000,
                                 FbankWindowType::POVEY, true)) {
            LOG_WARN("AudioPipe", "Speaker fbank init failed");
        }
    }

    // Initialize WavLM ONNX speaker encoder (optional — non-fatal).
    if (!cfg_.wavlm.model_path.empty()) {
        if (!wavlm_enc_.init(cfg_.wavlm)) {
            LOG_WARN("AudioPipe", "WavLM speaker encoder init failed");
        }
    }

    // Initialize UniSpeech-SAT ONNX speaker encoder (optional — non-fatal).
    if (!cfg_.unispeech.model_path.empty()) {
        if (!unispeech_enc_.init(cfg_.unispeech)) {
            LOG_WARN("AudioPipe", "UniSpeech-SAT speaker encoder init failed");
        }
    }

    // Initialize WavLM-Large + ECAPA-TDNN native GPU speaker encoder (optional).
    if (!cfg_.wavlm_ecapa_model.empty()) {
        if (!wlecapa_enc_.init(cfg_.wavlm_ecapa_model)) {
            LOG_WARN("AudioPipe", "WavLM-ECAPA native GPU init failed");
        } else {
            LOG_INFO("AudioPipe", "WavLM-ECAPA native GPU encoder ready (192-dim)");
        }
    }

    speaker_threshold_.store(cfg_.speaker_threshold, std::memory_order_relaxed);
    wavlm_threshold_.store(cfg_.wavlm_threshold, std::memory_order_relaxed);
    unispeech_threshold_.store(cfg_.unispeech_threshold, std::memory_order_relaxed);
    wlecapa_threshold_.store(cfg_.wavlm_ecapa_threshold, std::memory_order_relaxed);

    // Reset stats.
    memset(&stats_, 0, sizeof(stats_));

    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&AudioPipeline::process_loop, this);

    LOG_INFO("AudioPipe", "Started (ring=%zu KB, chunk=%d ms)",
             cfg_.ring_buffer_bytes / 1024, cfg_.process_chunk_ms);
    return true;
}

void AudioPipeline::stop() {
    if (!running_.load()) return;
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    LOG_INFO("AudioPipe", "Stopped (total: %lu samples, %lu mel frames, %lu speech)",
             stats_.pcm_samples_in, stats_.mel_frames, stats_.speech_frames);
}

void AudioPipeline::push_pcm(const int16_t* data, int n_samples) {
    if (!ring_ || n_samples <= 0) return;
    size_t bytes = n_samples * sizeof(int16_t);
    size_t written = ring_->push(reinterpret_cast<const uint8_t*>(data), bytes);
    if (written < bytes) {
        LOG_WARN("AudioPipe", "Ring buffer overflow, dropped %zu bytes",
                 bytes - written);
    }
}

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

    // TEN VAD processes 160-sample hops (10ms).
    int ten_hop = ten_.initialized() ? cfg_.ten.hop_size : 0;
    std::vector<int16_t> ten_buf;  // carries remainder for TEN VAD

    LOG_INFO("AudioPipe", "Process loop: chunk=%d samples (%d ms), silero=%s fsmn=%s ten=%s",
             chunk_samples, cfg_.process_chunk_ms,
             silero_.initialized() ? "ON" : "OFF",
             fsmn_.initialized() ? "ON" : "OFF",
             ten_.initialized() ? "ON" : "OFF");

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
        stats_.pcm_samples_in += n_samples;

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
                    on_vad_(vr, (int)stats_.mel_frames);
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

        // Run TEN VAD on raw PCM (160-sample hops).
        if (ten_.initialized() && ten_hop > 0 &&
            enable_ten_.load(std::memory_order_relaxed)) {
            ten_buf.insert(ten_buf.end(), pcm_buf.data(),
                           pcm_buf.data() + n_samples);
            int consumed = 0;
            while (consumed + ten_hop <= (int)ten_buf.size()) {
                TenVadResult tvr = ten_.process(
                    ten_buf.data() + consumed, ten_hop);
                consumed += ten_hop;
                stats_.ten_prob = tvr.probability;
                stats_.ten_speech = tvr.is_speech;
            }
            if (consumed > 0) {
                ten_buf.erase(ten_buf.begin(),
                              ten_buf.begin() + consumed);
            }
        }

        // Buffer PCM for speaker identification during speech.
        bool any_speaker_enabled =
            (speaker_enc_.initialized() && enable_speaker_.load(std::memory_order_relaxed)) ||
            (wavlm_enc_.initialized() && enable_wavlm_.load(std::memory_order_relaxed)) ||
            (unispeech_enc_.initialized() && enable_unispeech_.load(std::memory_order_relaxed)) ||
            (wlecapa_enc_.initialized() && enable_wlecapa_.load(std::memory_order_relaxed));

        // Clear active flags each tick — only set true when extraction happens.
        stats_.speaker_active = false;
        stats_.wavlm_active = false;
        stats_.unispeech_active = false;
        stats_.wlecapa_active = false;
        stats_.wlecapa_change_valid = false;

        if (any_speaker_enabled) {
            // Determine speech state from selected VAD source.
            VadSource src = static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed));
            bool vad_speech = false;
            switch (src) {
                case VadSource::SILERO: vad_speech = stats_.silero_speech; break;
                case VadSource::FSMN:   vad_speech = stats_.fsmn_speech; break;
                case VadSource::TEN:    vad_speech = stats_.ten_speech; break;
                case VadSource::ANY:
                default:
                    vad_speech = stats_.is_speech || stats_.silero_speech ||
                                 stats_.fsmn_speech || stats_.ten_speech;
                    break;
            }
            if (vad_speech && !in_speech_segment_) {
                in_speech_segment_ = true;
                early_extracted_   = false;
                speech_pcm_buf_.clear();
                speaker_fbank_.reset();
            }
            if (in_speech_segment_) {
                speech_pcm_buf_.insert(speech_pcm_buf_.end(),
                                       pcm_buf.data(), pcm_buf.data() + n_samples);
                if ((speaker_enc_.initialized() &&
                     enable_speaker_.load(std::memory_order_relaxed)) ||
                    (unispeech_enc_.initialized() &&
                     enable_unispeech_.load(std::memory_order_relaxed))) {
                    speaker_fbank_.push_pcm(pcm_buf.data(), n_samples);
                }
                // Limit to 10 seconds (160000 samples @ 16kHz).
                if (speech_pcm_buf_.size() > 160000) {
                    speech_pcm_buf_.erase(speech_pcm_buf_.begin(),
                                          speech_pcm_buf_.begin() + n_samples);
                }

                // Early extraction: run WL-ECAPA once we have enough speech,
                // without waiting for end-of-segment. This reduces "time to light-up".
                // IMPORTANT: auto_register=false — never create new speakers from
                // short early clips. If no match, report "identifying" state.
                int early_thresh = early_trigger_samples_.load(std::memory_order_relaxed);
                if (!early_extracted_ &&
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
                            strncpy(stats_.wlecapa_name, match.name.c_str(),
                                    sizeof(stats_.wlecapa_name) - 1);
                            stats_.wlecapa_name[sizeof(stats_.wlecapa_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "WL-ECAPA(early): id=%d sim=%.3f %s (%.2fs, %.1fms)",
                                     match.speaker_id, match.similarity,
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     early_samples / 16000.0f,
                                     wlecapa_enc_.last_lat_total_ms());
                            if (on_speaker_) on_speaker_(match);
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
            }
            if (!vad_speech && in_speech_segment_) {
                in_speech_segment_ = false;
                int speech_samples = (int)speech_pcm_buf_.size();
                float speech_duration = speech_samples / 16000.0f;

                LOG_INFO("AudioPipe", "Speech segment ended: %.2fs (%d samples)",
                         speech_duration, speech_samples);

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

                // Read fbank features (shared between CAM++ and ECAPA-TDNN).
                // CAM++ needs >= 150 frames (~1.5s), ECAPA-TDNN >= 100 frames (~1.0s).
                int fbank_frames = speaker_fbank_.frames_ready();
                std::vector<float> fbank_host;
                if (fbank_frames >= 100) {
                    fbank_host.resize(fbank_frames * 80);
                    speaker_fbank_.read_fbank(fbank_host.data(), fbank_frames);
                }

                // CAM++ speaker encoder (uses fbank features).
                if (speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    fbank_frames >= 150) {
                        float thresh = speaker_threshold_.load(std::memory_order_relaxed);

                        // Diagnostic: fbank value statistics.
                        {
                            float fmin = 1e30f, fmax = -1e30f, fsum = 0;
                            int total = fbank_frames * 80;
                            for (int i = 0; i < total; i++) {
                                fmin = std::min(fmin, fbank_host[i]);
                                fmax = std::max(fmax, fbank_host[i]);
                                fsum += fbank_host[i];
                            }
                            LOG_INFO("AudioPipe", "CAM++ fbank: frames=%d min=%.3f max=%.3f mean=%.3f",
                                     fbank_frames, fmin, fmax, fsum / total);
                        }

                        auto emb = speaker_enc_.extract(fbank_host.data(), fbank_frames);
                        if (!emb.empty()) {
                            // Validate embedding norm (should be ~1.0 if L2-normalized).
                            float enorm = 0;
                            for (float v : emb) enorm += v * v;
                            enorm = sqrtf(enorm);
                            LOG_INFO("AudioPipe", "CAM++ emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                     enorm, emb[0], emb[1], emb[2], emb[3]);
                            SpeakerMatch match = speaker_db_.identify(emb, thresh);
                            stats_.speaker_id = match.speaker_id;
                            stats_.speaker_sim = match.similarity;
                            stats_.speaker_new = match.is_new;
                            stats_.speaker_count = speaker_db_.count();
                            stats_.speaker_active = true;
                            strncpy(stats_.speaker_name, match.name.c_str(),
                                    sizeof(stats_.speaker_name) - 1);
                            stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "CAM++: id=%d sim=%.3f %s%s (fbank=%d)",
                                     match.speaker_id, match.similarity,
                                     match.is_new ? "NEW " : "",
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     fbank_frames);
                            if (on_speaker_) on_speaker_(match);
                        }
                }

                // WavLM speaker encoder (uses raw PCM waveform).
                if (wavlm_enc_.initialized() &&
                    enable_wavlm_.load(std::memory_order_relaxed) &&
                    speech_samples >= 24000) {  // minimum ~1.5s for Gemm output
                    auto emb = wavlm_enc_.extract_int16(speech_pcm_buf_.data(), speech_samples);
                    if (!emb.empty()) {
                        float enorm = 0;
                        for (float v : emb) enorm += v * v;
                        enorm = sqrtf(enorm);
                        LOG_INFO("AudioPipe", "WavLM emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                 enorm, emb[0], emb[1], emb[2], emb[3]);
                        float thresh = wavlm_threshold_.load(std::memory_order_relaxed);
                        SpeakerMatch match = wavlm_db_.identify(emb, thresh);
                        stats_.wavlm_id = match.speaker_id;
                        stats_.wavlm_sim = match.similarity;
                        stats_.wavlm_new = match.is_new;
                        stats_.wavlm_count = wavlm_db_.count();
                        stats_.wavlm_active = true;
                        strncpy(stats_.wavlm_name, match.name.c_str(),
                                sizeof(stats_.wavlm_name) - 1);
                        stats_.wavlm_name[sizeof(stats_.wavlm_name) - 1] = '\0';
                        LOG_INFO("AudioPipe", "WavLM: id=%d sim=%.3f %s%s (%d samples)",
                                 match.speaker_id, match.similarity,
                                 match.is_new ? "NEW " : "",
                                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                 speech_samples);
                    }
                }

                // ECAPA-TDNN speaker encoder (uses fbank features, not raw PCM).
                // Adapted from WeSpeaker ECAPA-TDNN-1024-LM with ASTP attention pooling.
                if (unispeech_enc_.initialized() &&
                    enable_unispeech_.load(std::memory_order_relaxed) &&
                    fbank_frames >= 100) {  // minimum ~1.0s for ECAPA-TDNN
                    auto emb = unispeech_enc_.extract_fbank(fbank_host.data(), fbank_frames, 80);
                    if (!emb.empty()) {
                        float enorm = 0;
                        for (float v : emb) enorm += v * v;
                        enorm = sqrtf(enorm);
                        LOG_INFO("AudioPipe", "ECAPA emb: norm=%.4f e[0..3]=[%.4f,%.4f,%.4f,%.4f]",
                                 enorm, emb[0], emb[1], emb[2], emb[3]);
                        float thresh = unispeech_threshold_.load(std::memory_order_relaxed);
                        SpeakerMatch match = unispeech_db_.identify(emb, thresh);
                        stats_.unispeech_id = match.speaker_id;
                        stats_.unispeech_sim = match.similarity;
                        stats_.unispeech_new = match.is_new;
                        stats_.unispeech_count = unispeech_db_.count();
                        stats_.unispeech_active = true;
                        strncpy(stats_.unispeech_name, match.name.c_str(),
                                sizeof(stats_.unispeech_name) - 1);
                        stats_.unispeech_name[sizeof(stats_.unispeech_name) - 1] = '\0';
                        LOG_INFO("AudioPipe", "ECAPA: id=%d sim=%.3f %s%s (fbank=%d)",
                                 match.speaker_id, match.similarity,
                                 match.is_new ? "NEW " : "",
                                 match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                 fbank_frames);
                    }
                }

                // WavLM-Large + ECAPA-TDNN native GPU speaker encoder (uses raw PCM).
                if (wlecapa_enc_.initialized() &&
                    enable_wlecapa_.load(std::memory_order_relaxed) &&
                    speech_samples >= 16000) {  // minimum ~1.0s
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
                    }
                }

                speech_pcm_buf_.clear();
            }
        }

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
                    on_vad_(vr, frame_idx);
                }
            }
        }

        // Report stats.
        if (on_stats_) {
            on_stats_(stats_);
        }

        // Periodic diagnostic log (~every 1s = 10 chunks at 100ms).
        if (++diag_counter % 10 == 0) {
            LOG_INFO("AudioPipe", "DIAG rms=%.4f silero=%.3f fsmn=%.3f ten=%.3f speech=%d gain=%.1f spk=%d(%.2f)",
                     stats_.last_rms, stats_.silero_prob,
                     stats_.fsmn_prob, stats_.ten_prob,
                     (int)stats_.is_speech,
                     gain_.load(std::memory_order_relaxed),
                     stats_.speaker_id, stats_.speaker_sim);
        }
    }
}

} // namespace deusridet
