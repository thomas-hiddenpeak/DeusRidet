// audio_pipeline.h — Real-time audio processing pipeline.
//
// Wires: WS PCM input → Ring Buffer → Gain → Mel (GPU) → Energy VAD
//                                          └→ Silero VAD (ONNX, CPU)
//                                          └→ FSMN VAD (ONNX, GPU fbank)
//                                          └→ TEN VAD (ONNX, CPU)
//                                          └→ Speaker Encoder (CAM++ GPU)
//                                          └→ ASR (Qwen3-ASR, native CUDA)
// Runs a processing thread that pulls from the ring buffer, computes
// Mel frames on GPU, runs VAD engines, extracts speaker embeddings, and reports results.

#pragma once

#include "mel_gpu.h"
#include "silero_vad.h"
#include "fsmn_vad.h"
#include "fsmn_fbank_gpu.h"
#include "ten_vad_wrapper.h"
#include "vad.h"
#include "asr/asr_engine.h"
#include "../../communis/ring_buffer.h"
#include "../../orator/speaker_encoder.h"
#include "../../orator/onnx_speaker_encoder.h"
#include "../../orator/wavlm_ecapa_encoder.h"
#include "../../orator/speaker_db.h"
#include "../../orator/speaker_vector_store.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <string>

namespace deusridet {

// Which VAD engine drives the speech detection for speaker extraction.
enum class VadSource : int {
    SILERO = 0,
    FSMN   = 1,
    TEN    = 2,
    ANY    = 3,  // OR of all enabled VADs
    DIRECT = 4,  // bypass VAD — ASR triggers on buffer duration only
};

struct AudioPipelineConfig {
    MelConfig mel;
    VadConfig vad;
    SileroVadConfig silero;             // Silero VAD model config
    FsmnVadConfig fsmn;                 // FSMN VAD model config
    TenVadConfig  ten;                  // TEN VAD model config
    SpeakerEncoderConfig speaker;       // CAM++ speaker encoder config
    OnnxSpeakerConfig wavlm;              // WavLM speaker encoder config
    OnnxSpeakerConfig unispeech;          // ECAPA-TDNN speaker encoder config (uses fbank, not raw PCM)
    std::string wavlm_ecapa_model;         // WavLM-Large+ECAPA-TDNN safetensors path (native GPU)
    float wavlm_ecapa_threshold = 0.55f;   // default cosine sim threshold
    std::string asr_model_path;            // Qwen3-ASR model directory (empty = disabled)
    size_t ring_buffer_bytes = 1 << 20;  // 1 MB (~32 seconds of int16 mono 16kHz)
    int process_chunk_ms     = 100;      // process in 100ms chunks (10 mel frames)
    float speaker_threshold  = 0.50f;    // CAM++ cosine sim threshold (same ~0.67, diff ~0.07)
    float wavlm_threshold    = 0.80f;    // WavLM Gemm threshold (same ~0.86-0.93, diff ~0.36-0.76)
    float unispeech_threshold= 0.55f;    // ECAPA-TDNN threshold (same ~0.57, diff ~0.03-0.45)
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
    // Speaker identification (CAM++).
    int      speaker_id;       // current speaker ID (-1 = unknown)
    float    speaker_sim;      // best cosine similarity
    bool     speaker_new;      // true if newly registered speaker
    int      speaker_count;    // number of known speakers
    char     speaker_name[64]; // current speaker name (empty if unnamed)
    // Speaker identification (WavLM).
    int      wavlm_id;         // WavLM speaker ID
    float    wavlm_sim;        // WavLM best similarity
    bool     wavlm_new;
    int      wavlm_count;
    char     wavlm_name[64];
    // Speaker identification (UniSpeech-SAT).
    int      unispeech_id;
    float    unispeech_sim;
    bool     unispeech_new;
    int      unispeech_count;
    char     unispeech_name[64];
    // Speaker identification (WavLM-Large + ECAPA-TDNN, native GPU).
    int      wlecapa_id;
    float    wlecapa_sim;
    bool     wlecapa_new;
    int      wlecapa_count;
    int      wlecapa_exemplars;      // exemplar count for matched speaker
    int      wlecapa_hits_above;     // exemplars above threshold in this match
    char     wlecapa_name[64];
    // Active flags — true only on the tick when extraction happened.
    bool     speaker_active;
    bool     wavlm_active;
    bool     unispeech_active;
    bool     wlecapa_active;
    // WL-ECAPA latency breakdown (ms), set after each extraction.
    float    wlecapa_lat_cnn_ms;
    float    wlecapa_lat_encoder_ms;
    float    wlecapa_lat_ecapa_ms;
    float    wlecapa_lat_total_ms;
    bool     wlecapa_is_early;       // true if this was an early extraction (not end-of-segment)
    // Change detection: cosine similarity between consecutive segment embeddings.
    float    wlecapa_change_sim;     // -1 if no previous segment
    bool     wlecapa_change_valid;   // true when change_sim is meaningful
    // ASR (Qwen3-ASR).
    bool     asr_active;             // true on tick when new transcript is ready
    bool     asr_busy;               // true when ASR thread is processing
    float    asr_latency_ms;         // transcription latency
    float    asr_audio_duration_s;   // audio segment duration
    float    asr_buf_sec;            // current ASR buffer duration (seconds)
    bool     asr_buf_has_speech;     // buffer contains detected speech
};

class AudioPipeline {
public:
    using OnVadEvent  = std::function<void(const VadResult&, int frame_idx)>;
    using OnStats     = std::function<void(const AudioPipelineStats&)>;
    using OnSpeaker   = std::function<void(const SpeakerMatch&)>;
    using OnTranscript = std::function<void(const asr::ASRResult& result, float audio_sec,
                                             int speaker_id, const std::string& speaker_name,
                                             float speaker_sim,
                                             const std::string& trigger_reason)>;
    using OnAsrLog = std::function<void(const std::string& json)>;
    using OnAsrPartial = std::function<void(const std::string& text, float audio_sec)>;

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
    void set_on_speaker(OnSpeaker cb) { on_speaker_ = std::move(cb); }
    void set_on_transcript(OnTranscript cb) { on_transcript_ = std::move(cb); }
    void set_on_asr_log(OnAsrLog cb) { on_asr_log_ = std::move(cb); }
    void set_on_asr_partial(OnAsrPartial cb) { on_asr_partial_ = std::move(cb); }

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

    // Speaker encoder enable/disable (thread-safe).
    void set_speaker_enabled(bool e) { enable_speaker_.store(e, std::memory_order_relaxed); }
    bool speaker_enabled() const { return enable_speaker_.load(std::memory_order_relaxed); }
    void set_wavlm_enabled(bool e) { enable_wavlm_.store(e, std::memory_order_relaxed); }
    bool wavlm_enabled() const { return enable_wavlm_.load(std::memory_order_relaxed); }
    void set_unispeech_enabled(bool e) { enable_unispeech_.store(e, std::memory_order_relaxed); }
    bool unispeech_enabled() const { return enable_unispeech_.load(std::memory_order_relaxed); }
    void set_wlecapa_enabled(bool e) { enable_wlecapa_.store(e, std::memory_order_relaxed); }
    bool wlecapa_enabled() const { return enable_wlecapa_.load(std::memory_order_relaxed); }

    // Per-backend threshold control.
    void set_speaker_threshold(float t) { speaker_threshold_.store(t, std::memory_order_relaxed); }
    float speaker_threshold() const { return speaker_threshold_.load(std::memory_order_relaxed); }
    void set_wavlm_threshold(float t) { wavlm_threshold_.store(t, std::memory_order_relaxed); }
    float wavlm_threshold() const { return wavlm_threshold_.load(std::memory_order_relaxed); }
    void set_unispeech_threshold(float t) { unispeech_threshold_.store(t, std::memory_order_relaxed); }
    float unispeech_threshold() const { return unispeech_threshold_.load(std::memory_order_relaxed); }
    void set_wlecapa_threshold(float t) { wlecapa_threshold_.store(t, std::memory_order_relaxed); }
    float wlecapa_threshold() const { return wlecapa_threshold_.load(std::memory_order_relaxed); }

    // Early extraction trigger (in seconds of speech).
    void set_early_trigger_sec(float s) { early_trigger_samples_.store((int)(s * 16000), std::memory_order_relaxed); }
    float early_trigger_sec() const { return early_trigger_samples_.load(std::memory_order_relaxed) / 16000.0f; }
    void set_early_trigger_enabled(bool e) { enable_early_.store(e, std::memory_order_relaxed); }
    bool early_trigger_enabled() const { return enable_early_.load(std::memory_order_relaxed); }

    // Minimum speech duration for full-segment speaker ID (in seconds).
    void set_min_speech_sec(float s) { min_speech_samples_.store(std::max(1, (int)(s * 16000)), std::memory_order_relaxed); }
    float min_speech_sec() const { return min_speech_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // ASR (Qwen3-ASR) enable/disable and tunable parameters.
    void set_asr_enabled(bool e) { enable_asr_.store(e, std::memory_order_relaxed); }
    bool asr_enabled() const { return enable_asr_.load(std::memory_order_relaxed); }
    bool asr_loaded() const { return asr_engine_ && asr_engine_->is_loaded(); }

    // Post-silence trigger: ms of silence after speech before triggering ASR.
    void set_asr_post_silence_ms(int ms) { asr_post_silence_ms_.store(std::max(100, ms), std::memory_order_relaxed); }
    int  asr_post_silence_ms() const { return asr_post_silence_ms_.load(std::memory_order_relaxed); }

    // Max buffer duration before forced transcription (seconds).
    void set_asr_max_buf_sec(float s) { asr_max_buf_samples_.store(std::max(16000, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_max_buf_sec() const { return asr_max_buf_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Minimum audio duration to trigger ASR (seconds).
    void set_asr_min_dur_sec(float s) { asr_min_samples_.store(std::max(1600, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_min_dur_sec() const { return asr_min_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Pre-roll: seconds of audio retained after transcription as context.
    void set_asr_pre_roll_sec(float s) { asr_pre_roll_samples_.store(std::max(0, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_pre_roll_sec() const { return asr_pre_roll_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Max new tokens for ASR decoder.
    void set_asr_max_tokens(int t) { asr_max_tokens_.store(std::max(1, std::min(4096, t)), std::memory_order_relaxed); }
    int  asr_max_tokens() const { return asr_max_tokens_.load(std::memory_order_relaxed); }

    // Repetition penalty for ASR decoder.
    void set_asr_rep_penalty(float p);
    float asr_rep_penalty() const { return asr_rep_penalty_.load(std::memory_order_relaxed); }

    // Minimum average energy for ASR segment (reject silence/noise).
    // Adapted from qwen35-thor (voice_session.cpp): min_avg_energy rejection.
    void set_asr_min_energy(float e) { asr_min_energy_.store(std::max(0.0f, e), std::memory_order_relaxed); }
    float asr_min_energy() const { return asr_min_energy_.load(std::memory_order_relaxed); }

    // Streaming ASR partial interval (seconds). 0 = disabled.
    // Adapted from qwen35-thor: STREAMING_ASR_CHUNK_S (~2s partial transcriptions).
    void set_asr_partial_sec(float s) { asr_partial_samples_.store(std::max(0, (int)(s * 16000)), std::memory_order_relaxed); }
    float asr_partial_sec() const { return asr_partial_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // Minimum speech ratio for ASR trigger (0.0–1.0). Segments with speech_sec / buf_sec
    // below this ratio are skipped (when buffer > 2s). Default 0.15 (15%).
    void set_asr_min_speech_ratio(float r) { asr_min_speech_ratio_.store(std::max(0.0f, std::min(1.0f, r)), std::memory_order_relaxed); }
    float asr_min_speech_ratio() const { return asr_min_speech_ratio_.load(std::memory_order_relaxed); }

    // Hallucination filter toggle. When ON, Whisper subtitle artifacts are
    // suppressed. When OFF, all ASR output passes through (rely on VAD pre-filtering).
    void set_asr_halluc_filter(bool e) { asr_halluc_filter_.store(e, std::memory_order_relaxed); }
    bool asr_halluc_filter() const { return asr_halluc_filter_.load(std::memory_order_relaxed); }

    // Per-backend speaker database access.
    SpeakerDb& speaker_db() { return speaker_db_; }
    SpeakerDb& wavlm_db() { return wavlm_db_; }
    SpeakerDb& unispeech_db() { return unispeech_db_; }
    SpeakerVectorStore& wlecapa_db() { return wlecapa_db_; }

    // Per-backend clear and name.
    void clear_speaker_db() {
        speaker_db_.clear();
        stats_.speaker_id = -1; stats_.speaker_sim = 0;
        stats_.speaker_new = false; stats_.speaker_count = 0;
        stats_.speaker_active = true;  // trigger UI refresh
        stats_.speaker_name[0] = '\0';
    }
    void clear_wavlm_db() {
        wavlm_db_.clear();
        stats_.wavlm_id = -1; stats_.wavlm_sim = 0;
        stats_.wavlm_new = false; stats_.wavlm_count = 0;
        stats_.wavlm_active = true;  // trigger UI refresh
        stats_.wavlm_name[0] = '\0';
    }
    void clear_unispeech_db() {
        unispeech_db_.clear();
        stats_.unispeech_id = -1; stats_.unispeech_sim = 0;
        stats_.unispeech_new = false; stats_.unispeech_count = 0;
        stats_.unispeech_active = true;  // trigger UI refresh
        stats_.unispeech_name[0] = '\0';
    }
    void clear_wlecapa_db() {
        wlecapa_db_.clear();
        stats_.wlecapa_id = -1; stats_.wlecapa_sim = 0;
        stats_.wlecapa_new = false; stats_.wlecapa_count = 0;
        stats_.wlecapa_exemplars = 0; stats_.wlecapa_hits_above = 0;
        stats_.wlecapa_active = true;
        stats_.wlecapa_name[0] = '\0';
    }
    void set_speaker_name(int id, const std::string& name) { speaker_db_.set_name(id, name); }
    void set_wavlm_name(int id, const std::string& name) { wavlm_db_.set_name(id, name); }
    void set_unispeech_name(int id, const std::string& name) { unispeech_db_.set_name(id, name); }
    void set_wlecapa_name(int id, const std::string& name) { wlecapa_db_.set_name(id, name); }
    bool remove_wlecapa_speaker(int id) { return wlecapa_db_.remove_speaker(id); }
    bool merge_wlecapa_speakers(int dst_id, int src_id) { return wlecapa_db_.merge_speakers(dst_id, src_id); }

    // Input gain (applied before Mel + VAD). 1.0 = unity.
    void set_gain(float g) { gain_.store(g, std::memory_order_relaxed); }
    float gain() const { return gain_.load(std::memory_order_relaxed); }

    // VAD source selection for speaker extraction pipeline routing.
    void set_vad_source(VadSource s) { vad_source_.store(static_cast<int>(s), std::memory_order_relaxed); }
    VadSource vad_source() const { return static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed)); }

    // VAD source selection for ASR pipeline (independent from speaker).
    void set_asr_vad_source(VadSource s) { asr_vad_source_.store(static_cast<int>(s), std::memory_order_relaxed); }
    VadSource asr_vad_source() const { return static_cast<VadSource>(asr_vad_source_.load(std::memory_order_relaxed)); }

private:
    void process_loop();
    void asr_loop();

    AudioPipelineConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    RingBuffer* ring_ = nullptr;
    MelSpectrogram mel_;
    VoiceActivityDetector vad_;
    SileroVad silero_;
    FsmnVad fsmn_;
    TenVadWrapper ten_;
    SpeakerEncoder speaker_enc_;
    SpeakerDb speaker_db_{"CAM++Db"};
    FsmnFbankGpu speaker_fbank_;  // 80-dim fbank for CAM++
    OnnxSpeakerEncoder wavlm_enc_;
    SpeakerDb wavlm_db_{"WavLMDb", 0.1f};       // low EMA to resist centroid contamination
    OnnxSpeakerEncoder unispeech_enc_;
    SpeakerDb unispeech_db_{"ECAPADb", 0.15f};  // ECAPA-TDNN: higher EMA for stable centroids
    WavLMEcapaEncoder wlecapa_enc_;
    SpeakerVectorStore wlecapa_db_{"WLEcapaDb", 192, 0.15f};

    AudioPipelineStats stats_{};
    std::atomic<float> gain_{1.0f};
    std::atomic<int> vad_source_{static_cast<int>(VadSource::SILERO)};
    std::atomic<int> asr_vad_source_{static_cast<int>(VadSource::SILERO)};  // ASR defaults to SILERO (same as speaker)
    std::atomic<bool> enable_silero_{true};
    std::atomic<bool> enable_fsmn_{false};
    std::atomic<bool> enable_ten_{false};
    std::atomic<bool> enable_speaker_{false};
    std::atomic<bool> enable_wavlm_{false};
    std::atomic<bool> enable_unispeech_{false};
    std::atomic<bool> enable_wlecapa_{true};
    std::atomic<float> speaker_threshold_{0.50f};
    std::atomic<float> wavlm_threshold_{0.80f};
    std::atomic<float> unispeech_threshold_{0.55f};
    std::atomic<float> wlecapa_threshold_{0.55f};
    std::atomic<int>   early_trigger_samples_{27200};  // 1.7s default
    std::atomic<bool>  enable_early_{true};              // early trigger on/off
    std::atomic<int>   min_speech_samples_{16000};       // 1.0s default for full-segment ID
    std::atomic<bool>  enable_asr_{true};                // ASR on/off
    std::atomic<int>   asr_post_silence_ms_{300};         // post-silence trigger (ms)
    std::atomic<int>   asr_max_buf_samples_{480000};      // max buffer (30s @ 16kHz)
    std::atomic<int>   asr_min_samples_{8000};            // min audio for ASR (0.5s)
    std::atomic<int>   asr_pre_roll_samples_{16000};      // pre-roll retention (1s)
    std::atomic<int>   asr_max_tokens_{448};              // decoder max new tokens
    std::atomic<float> asr_rep_penalty_{1.0f};            // repetition penalty
    std::atomic<float> asr_min_energy_{0.008f};           // min avg energy for ASR segment
    std::atomic<int>   asr_partial_samples_{32000};       // streaming partial interval (2s default)
    std::atomic<float> asr_min_speech_ratio_{0.15f};      // min speech / buffer ratio for trigger
    std::atomic<bool>  asr_halluc_filter_{true};           // hallucination filter on/off

    // ASR engine (Qwen3-ASR).
    std::unique_ptr<asr::ASREngine> asr_engine_;

    // PCM buffer for speech segments (accumulated for speaker embedding).
    std::vector<int16_t> speech_pcm_buf_;
    bool in_speech_segment_ = false;
    bool early_extracted_   = false;   // true after early extraction during speech

    // ASR audio accumulation: ALL audio is accumulated (Whisper handles silence
    // naturally). VAD is used only to decide WHEN to trigger transcription and
    // to track speech content ratio for filtering mostly-silence segments.
    std::vector<int16_t> asr_pcm_buf_;
    bool asr_saw_speech_    = false;   // any speech detected in current accumulation window
    int  asr_post_silence_  = 0;       // silence chunks after last speech (for trigger)
    int  asr_speech_samples_ = 0;      // samples accumulated while VAD=speech (content quality metric)
    int  asr_partial_sent_at_ = 0;     // buffer size (samples) at last partial submission

    // ASR async thread — transcription runs off-process_loop to avoid blocking.
    struct ASRJob {
        std::vector<float> pcm_f32;     // speech audio, already int16→float32
        float audio_duration_sec;
        std::string trigger_reason;     // "post_silence" or "buffer_full" or "streaming_partial"
        bool is_partial = false;        // streaming partial — don't count as final transcript
        // Speaker identification snapshot captured at trigger time.
        int speaker_id = -1;            // wlecapa speaker ID (-1 = unknown)
        std::string speaker_name;       // wlecapa speaker name (empty = unnamed)
        float speaker_sim = 0.0f;       // wlecapa cosine similarity
    };
    std::thread asr_thread_;
    std::mutex asr_mutex_;
    std::condition_variable asr_cv_;
    std::queue<ASRJob> asr_queue_;
    std::atomic<bool> asr_busy_{false};

    // Change detection: previous segment embedding for inter-segment cosine similarity.
    std::vector<float> prev_wlecapa_emb_;  // 192-dim, empty if first segment

    OnVadEvent on_vad_;
    OnStats    on_stats_;
    OnSpeaker  on_speaker_;
    OnTranscript on_transcript_;
    OnAsrLog   on_asr_log_;
    OnAsrPartial on_asr_partial_;
};

} // namespace deusridet
