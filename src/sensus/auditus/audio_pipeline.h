// audio_pipeline.h — Real-time audio processing pipeline.
//
// Wires: WS PCM input → Ring Buffer → Gain → Mel (GPU) → Energy VAD
//                                          └→ Silero VAD (ONNX, CPU)
// Runs a processing thread that pulls from the ring buffer, computes
// Mel frames on GPU, runs both VAD engines, and reports results.

#pragma once

#include "mel_gpu.h"
#include "silero_vad.h"
#include "fsmn_vad.h"
#include "ten_vad_wrapper.h"
#include "vad.h"
#include "../../communis/ring_buffer.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

namespace deusridet {

struct AudioPipelineConfig {
    MelConfig mel;
    VadConfig vad;
    SileroVadConfig silero;             // Silero VAD model config
    FsmnVadConfig fsmn;                 // FSMN VAD model config
    TenVadConfig  ten;                  // TEN VAD model config
    size_t ring_buffer_bytes = 1 << 20;  // 1 MB (~32 seconds of int16 mono 16kHz)
    int process_chunk_ms     = 100;      // process in 100ms chunks (10 mel frames)
};

struct AudioPipelineStats {
    uint64_t pcm_samples_in;   // total PCM samples received
    uint64_t mel_frames;       // total mel frames computed
    uint64_t speech_frames;    // mel frames classified as speech (energy VAD)
    float    last_rms;         // latest frame RMS (linear)
    float    last_energy;      // latest frame mean log-energy
    bool     is_speech;        // current energy VAD state
    // Silero VAD.
    float    silero_prob;      // latest Silero speech probability [0,1]
    bool     silero_speech;    // Silero VAD speech state
    // FSMN VAD.
    float    fsmn_prob;        // latest FSMN speech probability [0,1]
    bool     fsmn_speech;      // FSMN VAD speech state
    // TEN VAD.
    float    ten_prob;         // latest TEN speech probability [0,1]
    bool     ten_speech;       // TEN VAD speech state
};

class AudioPipeline {
public:
    using OnVadEvent = std::function<void(const VadResult&, int frame_idx)>;
    using OnStats    = std::function<void(const AudioPipelineStats&)>;

    AudioPipeline();
    ~AudioPipeline();

    AudioPipeline(const AudioPipeline&) = delete;
    AudioPipeline& operator=(const AudioPipeline&) = delete;

    bool start(const AudioPipelineConfig& cfg);
    void stop();
    bool running() const { return running_.load(std::memory_order_relaxed); }

    // Push raw int16 PCM from WS callback (producer thread). Non-blocking.
    void push_pcm(const int16_t* data, int n_samples);

    // Register callbacks.
    void set_on_vad(OnVadEvent cb) { on_vad_ = std::move(cb); }
    void set_on_stats(OnStats cb)  { on_stats_ = std::move(cb); }

    const AudioPipelineStats& stats() const { return stats_; }

    // Runtime VAD threshold adjustment (thread-safe: atomic float write).
    void set_vad_threshold(float t) { vad_.set_threshold(t); }
    float vad_threshold() const { return vad_.config().energy_threshold; }
    float vad_noise_floor() const { return vad_.noise_floor(); }

    // Silero VAD threshold.
    void set_silero_threshold(float t) { silero_.set_threshold(t); }
    float silero_threshold() const { return silero_.threshold(); }
    float silero_prob() const { return stats_.silero_prob; }

    // FSMN VAD threshold.
    void set_fsmn_threshold(float t) { fsmn_.set_threshold(t); }
    float fsmn_threshold() const { return fsmn_.threshold(); }

    // TEN VAD threshold.
    void set_ten_threshold(float t) { ten_.set_threshold(t); }
    float ten_threshold() const { return ten_.threshold(); }

    // Per-VAD enable/disable (thread-safe).
    void set_silero_enabled(bool e) { enable_silero_.store(e, std::memory_order_relaxed); }
    bool silero_enabled() const { return enable_silero_.load(std::memory_order_relaxed); }
    void set_fsmn_enabled(bool e) { enable_fsmn_.store(e, std::memory_order_relaxed); }
    bool fsmn_enabled() const { return enable_fsmn_.load(std::memory_order_relaxed); }
    void set_ten_enabled(bool e) { enable_ten_.store(e, std::memory_order_relaxed); }
    bool ten_enabled() const { return enable_ten_.load(std::memory_order_relaxed); }

    // Input gain (applied before Mel + VAD). 1.0 = unity.
    void set_gain(float g) { gain_.store(g, std::memory_order_relaxed); }
    float gain() const { return gain_.load(std::memory_order_relaxed); }

private:
    void process_loop();

    AudioPipelineConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    RingBuffer* ring_ = nullptr;
    MelSpectrogram mel_;
    VoiceActivityDetector vad_;
    SileroVad silero_;
    FsmnVad fsmn_;
    TenVadWrapper ten_;

    AudioPipelineStats stats_{};
    std::atomic<float> gain_{1.0f};
    std::atomic<bool> enable_silero_{true};
    std::atomic<bool> enable_fsmn_{true};
    std::atomic<bool> enable_ten_{true};

    OnVadEvent on_vad_;
    OnStats    on_stats_;
};

} // namespace deusridet
