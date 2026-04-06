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
            LOG_INFO("AudioPipe", "DIAG rms=%.4f silero=%.3f fsmn=%.3f ten=%.3f speech=%d gain=%.1f",
                     stats_.last_rms, stats_.silero_prob,
                     stats_.fsmn_prob, stats_.ten_prob,
                     (int)stats_.is_speech,
                     gain_.load(std::memory_order_relaxed));
        }
    }
}

} // namespace deusridet
