/**
 * @file audio_pipeline.h
 * @philosophical_role Declaration of the auditory pipeline facade. Single seam through which PCM enters the entity; stats fields here are read by Actus verbs for JSON export.
 * @serves Nexus, Conscientia, Actus (awaken).
 */
// audio_pipeline.h — Real-time audio processing pipeline.
// PCM → ring buffer → GPU Mel → Silero VAD → CAM++/WL-ECAPA speaker
// encoders → Qwen3-ASR. A processing thread pulls from the ring buffer
// and reports stats; see the @serves consumers above.

#pragma once

// Engine-config headers arrive via audio_pipeline_config.h (its canonical owner).
#include "povey_fbank_gpu.h"  // generic Povey/Hamming GPU Fbank (CAM++ frontend)
#include "asr/asr_engine.h"
#include "../../communis/ring_buffer.h"
#include "../../orator/wavlm_ecapa_encoder.h"
#include "../../orator/speaker_db.h"
#include "../../orator/speaker_vector_store.h"
#include "../../orator/orator_reclusterer.h"
#include "../../orator/orator_online.h"  // clean three-concern online facade

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <string>

#include "pipeline_stats.h"
#include "speaker_tracking.h"
#include "audio_pipeline_config.h"

namespace deusridet {

class AudioPipeline {
public:
    // VAD callback fires only on speech segment boundaries (start/end).
    //   prob = Silero probability for this window (energy-VAD removed April 2026).
    using OnVadEvent  = std::function<void(bool is_speech, bool segment_start,
                                           bool segment_end, float prob,
                                           int frame_idx, uint64_t audio_t1)>;
    using OnStats     = std::function<void(const AudioPipelineStats&)>;
    using OnSpeaker   = std::function<void(const SpeakerMatch&)>;
    using OnTranscript = std::function<void(const asr::ASRResult& result, float audio_sec,
                                             int speaker_id, const std::string& speaker_name,
                                             float speaker_sim, float speaker_confidence,
                                             const std::string& speaker_source,
                                             const std::string& trigger_reason,
                                             float stream_start_sec, float stream_end_sec)>;
    using OnAsrLog = std::function<void(const std::string& json)>;
    using OnAsrPartial = std::function<void(const std::string& text, float audio_sec)>;
    // Drop event: ring-buffer overflow at push_pcm. Carries the AUDIO T1
    // range of the samples that were discarded so consumers can splice a
    // gap marker into the timeline instead of silently losing audio.
    using OnDrop = std::function<void(uint64_t t1_from, uint64_t t1_to,
                                      const char* reason, size_t bytes)>;
    // Phase 4 — OratorReclusterer relabel event. Fired when the rolling-window
    // re-clusterer disagrees with the online tentative id for a finalized
    // segment. Consumers (Nexus) should forward this to a
    // `speaker_relabel` WS event so the WebUI timeline can be patched.
    using OnSpeakerRelabel = std::function<void(uint64_t segment_id,
                                                int old_speaker_id,
                                                int new_speaker_id,
                                                float confidence)>;

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
    void set_on_drop(OnDrop cb) { on_drop_ = std::move(cb); }
    void set_on_speaker_relabel(OnSpeakerRelabel cb) { on_speaker_relabel_ = std::move(cb); }

    const AudioPipelineStats& stats() const { return stats_; }

    // Silero VAD threshold.
    void set_silero_threshold(float t) { silero_.set_threshold(t); }
    float silero_threshold() const { return silero_.threshold(); }
    float silero_prob() const { return stats_.silero_prob; }

    // FRCRN speech enhancement enable/disable (thread-safe).
    void set_frcrn_enabled(bool e) { enable_frcrn_.store(e, std::memory_order_relaxed); }
    bool frcrn_enabled() const { return enable_frcrn_.load(std::memory_order_relaxed); }
    bool frcrn_loaded() const { return frcrn_.initialized(); }

    // P1: Overlap detection enable/disable (model loaded but currently not
    // wired into the process loop — retained for future reintroduction).
    void set_overlap_det_enabled(bool e) { enable_overlap_det_.store(e, std::memory_order_relaxed); }
    bool overlap_det_enabled() const { return enable_overlap_det_.load(std::memory_order_relaxed); }
    bool overlap_det_loaded() const { return overlap_det_.initialized(); }

    // P2: Speech separation enable/disable (see overlap note above).
    void set_separator_enabled(bool e) { enable_separator_.store(e, std::memory_order_relaxed); }
    bool separator_enabled() const { return enable_separator_.load(std::memory_order_relaxed); }
    bool separator_loaded() const { return separator_.loaded(); }

    // Silero enable/disable (thread-safe). FSMN was removed April 2026.
    void set_silero_enabled(bool e) { enable_silero_.store(e, std::memory_order_relaxed); }
    bool silero_enabled() const { return enable_silero_.load(std::memory_order_relaxed); }

    // Speaker encoder enable/disable (thread-safe).
    void set_speaker_enabled(bool e) { enable_speaker_.store(e, std::memory_order_relaxed); }
    bool speaker_enabled() const { return enable_speaker_.load(std::memory_order_relaxed); }
    void set_wlecapa_enabled(bool e) { enable_wlecapa_.store(e, std::memory_order_relaxed); }
    bool wlecapa_enabled() const { return enable_wlecapa_.load(std::memory_order_relaxed); }

    // Per-backend threshold control.
    void set_speaker_threshold(float t) { speaker_threshold_.store(t, std::memory_order_relaxed); }
    float speaker_threshold() const { return speaker_threshold_.load(std::memory_order_relaxed); }
    // Ambiguity guard: min (top1 - top2) margin required to trust a closed-set
    // match. Runtime-tunable (like the match/register thresholds) so the
    // cross-speaker absorption guard can be exercised live, not frozen in a
    // fixture-driven config value.
    void set_speaker_margin_abstain(float m) { speaker_margin_abstain_.store(m, std::memory_order_relaxed); }
    float speaker_margin_abstain() const { return speaker_margin_abstain_.load(std::memory_order_relaxed); }
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

    // Intra-segment speaker change detection: re-check speaker identity
    // within long segments to detect speaker transitions missed by VAD.
    void set_spk_recheck_sec(float s) { spk_recheck_samples_.store(std::max(16000, (int)(s * 16000)), std::memory_order_relaxed); }
    float spk_recheck_sec() const { return spk_recheck_samples_.load(std::memory_order_relaxed) / 16000.0f; }
    void set_spk_recheck_enabled(bool e) { enable_spk_recheck_.store(e, std::memory_order_relaxed); }
    bool spk_recheck_enabled() const { return enable_spk_recheck_.load(std::memory_order_relaxed); }
    // Cosine similarity threshold below which we declare a speaker change.
    void set_spk_change_threshold(float t) { spk_change_threshold_.store(t, std::memory_order_relaxed); }
    float spk_change_threshold() const { return spk_change_threshold_.load(std::memory_order_relaxed); }
    // Window size (seconds) for the re-check embedding extraction.
    void set_spk_recheck_window_sec(float s) { spk_recheck_window_samples_.store(std::max(8000, (int)(s * 16000)), std::memory_order_relaxed); }
    float spk_recheck_window_sec() const { return spk_recheck_window_samples_.load(std::memory_order_relaxed) / 16000.0f; }

    // ASR (Qwen3-ASR) enable/disable and tunable parameters.
    void set_asr_enabled(bool e) { enable_asr_.store(e, std::memory_order_relaxed); }
    bool asr_enabled() const { return enable_asr_.load(std::memory_order_relaxed); }
    bool asr_loaded() const { return asr_engine_ && asr_engine_->is_loaded(); }

    // Post-silence trigger: ms of silence after speech before triggering ASR.
    void set_asr_post_silence_ms(int ms) { asr_post_silence_ms_.store(std::max(100, ms), std::memory_order_relaxed); }
    int  asr_post_silence_ms() const { return asr_post_silence_ms_.load(std::memory_order_relaxed); }

    // SAAS: adaptive post-silence — dynamically adjusts based on segment length.
    void set_asr_adaptive_silence(bool e) { asr_adaptive_silence_.store(e, std::memory_order_relaxed); }
    bool asr_adaptive_silence() const { return asr_adaptive_silence_.load(std::memory_order_relaxed); }
    void set_asr_adaptive_short_ms(int ms) { asr_adaptive_short_ms_.store(std::max(100, ms), std::memory_order_relaxed); }
    int  asr_adaptive_short_ms() const { return asr_adaptive_short_ms_.load(std::memory_order_relaxed); }
    void set_asr_adaptive_long_ms(int ms) { asr_adaptive_long_ms_.store(std::max(50, ms), std::memory_order_relaxed); }
    int  asr_adaptive_long_ms() const { return asr_adaptive_long_ms_.load(std::memory_order_relaxed); }
    void set_asr_adaptive_vlong_ms(int ms) { asr_adaptive_vlong_ms_.store(std::max(50, ms), std::memory_order_relaxed); }
    int  asr_adaptive_vlong_ms() const { return asr_adaptive_vlong_ms_.load(std::memory_order_relaxed); }

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

    // Per-backend speaker database access.
    SpeakerDb& speaker_db() { return speaker_db_; }
    SpeakerVectorStore& campp_db() { return campp_db_; }
    SpeakerVectorStore& wlecapa_db() { return wlecapa_db_; }
    SpeakerVectorStore& dual_db() { return dual_db_; }

    // Per-backend clear and name.
    void clear_speaker_db() {
        speaker_db_.clear();
        campp_db_.clear();
        stats_.speaker_id = -1; stats_.speaker_sim = 0;
        stats_.speaker_new = false; stats_.speaker_count = 0;
        stats_.speaker_active = true;  // trigger UI refresh
        stats_.speaker_name[0] = '\0';
    }
    void clear_wlecapa_db() {
        wlecapa_db_.clear();
        stats_.wlecapa_id = -1; stats_.wlecapa_sim = 0;
        stats_.wlecapa_new = false; stats_.wlecapa_count = 0;
        stats_.wlecapa_exemplars = 0; stats_.wlecapa_hits_above = 0;
        stats_.wlecapa_active = true;
        stats_.wlecapa_name[0] = '\0';
    }
    void set_speaker_name(int id, const std::string& name) {
        // Keep legacy + active online stores in sync so manual naming survives
        // whichever matcher path is currently producing IDs.
        speaker_db_.set_name(id, name);
        campp_db_.set_name(id, name);
        wlecapa_db_.set_name(id, name);
        dual_db_.set_name(id, name);
    }
    void set_wlecapa_name(int id, const std::string& name) { wlecapa_db_.set_name(id, name); }
    bool remove_wlecapa_speaker(int id) { return wlecapa_db_.remove_speaker(id); }
    bool merge_wlecapa_speakers(int dst_id, int src_id) { return wlecapa_db_.merge_speakers(dst_id, src_id); }

    // Forget a speaker everywhere. The live online matcher reads dual_db_ (or
    // campp_db_ when the dual encoder is off), so deleting only the legacy
    // wlecapa_db_ left the prototype alive on the hot path — a deleted person
    // kept absorbing strangers. Remove the prototype from every store the
    // identifier can consult, mirroring set_speaker_name's fan-out.
    bool remove_speaker(int id) {
        bool any = false;
        any |= dual_db_.remove_speaker(id);
        any |= campp_db_.remove_speaker(id);
        any |= wlecapa_db_.remove_speaker(id);
        return any;
    }

    // Input gain (applied before Mel + VAD). 1.0 = unity.
    void set_gain(float g) { gain_.store(g, std::memory_order_relaxed); }
    float gain() const { return gain_.load(std::memory_order_relaxed); }

    // VAD source selection for speaker extraction pipeline routing.
    void set_vad_source(VadSource s) { vad_source_.store(static_cast<int>(s), std::memory_order_relaxed); }
    VadSource vad_source() const { return static_cast<VadSource>(vad_source_.load(std::memory_order_relaxed)); }

    // VAD source selection for ASR pipeline (independent from speaker).
    void set_asr_vad_source(VadSource s) { asr_vad_source_.store(static_cast<int>(s), std::memory_order_relaxed); }
    VadSource asr_vad_source() const { return static_cast<VadSource>(asr_vad_source_.load(std::memory_order_relaxed)); }

    // DiariZen-v2 session-level capture (Hybrid P1).
    //   push_pcm() also appends to an in-RAM session buffer (mono 16 kHz
    //   int16). `max_seconds` caps RAM (default 4000 s ≈ 125 MB); on
    //   overflow further samples are dropped from capture only (live
    //   pipeline unaffected). `dump_wav` writes a minimal 16-kHz mono WAV
    //   and returns samples written (0 on failure/empty). `clear` resets
    //   the buffer between sessions to avoid cross-run concatenation.
    void diarizen_capture_enable(bool on, double max_seconds = 4000.0);
    bool diarizen_capture_enabled() const;
    size_t diarizen_capture_samples() const;
    size_t diarizen_dump_wav(const std::string& path) const;
    // Native-pipeline path (P3b): snapshot the capture buffer as float32 in
    // [-1, 1] (int16/32768), matching torchaudio/soundfile scale. Replaces
    // the dump-to-wav + reload step for the in-process DiarizenPipeline.
    // Returns the number of samples copied; 0 on empty buffer.
    size_t diarizen_copy_pcm_f32(std::vector<float>& out) const;

    // Hybrid P2 / direction C — sliding-window snapshot. Copies only the
    // tail `window_samples` of the capture buffer (the most recent audio)
    // as float32, and writes the *absolute* stream-time origin of the first
    // copied sample into `origin_sec_out`. With `window_samples == 0` or a
    // window larger than the buffer, copies the whole buffer (identical to
    // diarizen_copy_pcm_f32, origin == diarizen_capture_origin_sec). Both
    // the copy and the matching origin are taken under one lock, so the
    // returned PCM and its origin can never race against push_pcm. Returns
    // the number of samples copied; 0 on empty buffer.
    size_t diarizen_copy_pcm_f32_window(std::vector<float>& out,
                                        size_t window_samples,
                                        double& origin_sec_out) const;
    void diarizen_capture_clear();

    // Hybrid P2 — origin of the capture buffer in stream-time seconds.
    // The capture buffer's sample index 0 corresponds to this audio_t1
    // sample count. Adding this offset to a DiariZen segment's start_sec
    // recovers the absolute stream-time second the segment refers to.
    // Returns 0.0 when capture has never been enabled.
    double diarizen_capture_origin_sec() const;

    // Current ingress audio_t1 in seconds (samples / 16000). Useful for
    // any consumer that needs to reason about "the latest second of audio
    // the producer has handed us" (e.g. transcript holdback drainer).
    double audio_t1_in_sec() const {
        return audio_t1_in_.load(std::memory_order_relaxed) / 16000.0;
    }

private:
    void process_loop();
    void asr_loop();

    // DiariZen-v2 capture tap (called from push_pcm). Defined in
    // audio_pipeline_diarizen_capture.cpp.
    void diarizen_capture_tap_(const int16_t* data, int n_samples);

    // process_loop() stage decomposition (Step 11 A1).
    //   See docs/{en,zh}/architecture/00-overview.md §"Step 11" for the
    //   stage map; each helper lives in its own peer TU by the same name.
    void process_asr_pipeline_(const int16_t* pcm_buf, int n_samples);
    void process_saas_full_extract_(int fbank_frames);
    void process_saas_during_speech_(const int16_t* pcm_buf, int n_samples);
    void process_saas_segment_end_();

    AudioPipelineConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    RingBuffer* ring_ = nullptr;
    MelSpectrogram mel_;
    FrcrnEnhancer frcrn_;

    SileroVad silero_;
    SpeakerEncoder speaker_enc_;
    SpeakerVectorStore campp_db_{"CamppDb", 192, 0.15f};
    SpeakerDb speaker_db_{"CAM++Db"};  // legacy — kept for UI/API backward compat
    PoveyFbankGpu speaker_fbank_;  // 80-dim fbank for CAM++
    std::vector<float> seg_fbank_buf_;   // accumulated fbank frames for current speech segment
    bool campp_early_extracted_ = false; // whether CAM++ EARLY has fired this segment

    // CAM++ temporal smoothing: majority voting over recent identifications.
    // Prevents rapid flip-backs between speakers.
    static constexpr int kSmoothWindowSize = 3;
    int smooth_ring_[kSmoothWindowSize] = {-1, -1, -1};  // recent speaker IDs
    int smooth_ring_pos_ = 0;
    int smoothed_speaker_id_ = -1;       // current smoothed speaker
    int campp_full_count_ = 0;           // count FULL extractions for periodic absorption

    int seg_overlap_chunks_ = 0;  // count overlap-detected chunks in current segment
    int seg_total_chunks_ = 0;    // total chunks in current segment

    WavLMEcapaEncoder wlecapa_enc_;
    SpeakerVectorStore wlecapa_db_{"WLEcapaDb", 192, 0.15f};

    // Dual-encoder 384D store: CAM++ (192D) + WL-ECAPA (192D) concatenated.
    // Uses both encoders for better discrimination of similar voices.
    SpeakerVectorStore dual_db_{"DualDb", 384, 0.15f};
    bool use_dual_encoder_ = false;  // set true once WL-ECAPA is confirmed initialized

    // Clean three-concern online speaker decision facade (replaces the legacy
    // entangled greedy logic in process_saas_full_extract_). Owns concern-①
    // registration-gate state; read-only on the store for concern ②.
    orator::OratorOnline orator_online_{};

    AudioPipelineStats stats_{};
    std::atomic<float> gain_{1.0f};
    std::atomic<int> vad_source_{static_cast<int>(VadSource::SILERO)};
    std::atomic<int> asr_vad_source_{static_cast<int>(VadSource::SILERO)};  // ASR defaults to SILERO (same as speaker)
    std::atomic<bool> enable_silero_{true};
    std::atomic<bool> enable_frcrn_{true};     // FRCRN speech enhancement

    std::atomic<bool> enable_speaker_{true};    // CAM++ — primary SAAS encoder
    std::atomic<bool> enable_wlecapa_{false};    // WL-ECAPA — disabled (CAM++ is primary)
    std::atomic<float> speaker_threshold_{0.50f}; // CAM++ matching threshold
    std::atomic<float> speaker_register_threshold_{0.60f}; // pending pool confirmation threshold (0.60: separates true 4-spk at 0.62+ from false splits at 0.56-0.58)
    std::atomic<float> speaker_margin_abstain_{0.05f}; // min (top1-top2) margin to trust a match (ambiguity guard, runtime-tunable)
    std::atomic<float> wlecapa_threshold_{0.55f};
    std::atomic<int>   early_trigger_samples_{27200};  // 1.7s default
    std::atomic<bool>  enable_early_{true};              // early trigger on/off
    std::atomic<int>   min_speech_samples_{16000};       // 1.0s default for full-segment ID
    // Intra-segment speaker change detection.
    std::atomic<int>   spk_recheck_samples_{48000};      // re-check every 3.0s of speech
    std::atomic<bool>  enable_spk_recheck_{true};         // on by default
    std::atomic<float> spk_change_threshold_{0.35f};      // cosine sim below this = different speaker
    std::atomic<int>   spk_recheck_window_samples_{24000}; // 1.5s window for re-check embedding
    std::atomic<bool>  enable_asr_{false};                // ASR off for speaker-only testing
    std::atomic<int>   asr_post_silence_ms_{500};         // post-silence trigger (ms) — base value
    std::atomic<int>   asr_max_buf_samples_{480000};      // max buffer (30s @ 16kHz)

    // SAAS: adaptive post-silence parameters.
    // Actual post-silence = base * multiplier, where multiplier depends on segment length.
    std::atomic<bool>  asr_adaptive_silence_{true};        // enable adaptive post-silence
    std::atomic<int>   asr_adaptive_short_ms_{800};        // post-silence for short segments (<0.8s)
    std::atomic<int>   asr_adaptive_long_ms_{200};         // post-silence for long segments (5-15s)
    std::atomic<int>   asr_adaptive_vlong_ms_{150};        // post-silence for very long segments (>15s)
    std::atomic<int>   asr_min_samples_{8000};            // min audio for ASR (0.5s)
    std::atomic<int>   asr_pre_roll_samples_{1600};       // pre-roll retention (0.1s)
    std::atomic<int>   asr_max_tokens_{448};              // decoder max new tokens
    std::atomic<float> asr_rep_penalty_{1.0f};            // repetition penalty
    std::atomic<float> asr_min_energy_{0.008f};           // min avg energy for ASR segment
    std::atomic<int>   asr_partial_samples_{32000};       // streaming partial interval (2s default)
    std::atomic<float> asr_min_speech_ratio_{0.15f};      // min speech / buffer ratio for trigger

    // ASR engine (Qwen3-ASR).
    std::unique_ptr<asr::ASREngine> asr_engine_;

    // PCM buffer for speech segments (accumulated for speaker embedding).
    std::vector<int16_t> speech_pcm_buf_;
    bool in_speech_segment_ = false;
    bool early_extracted_   = false;   // true after early extraction during speech
    // Intra-segment speaker change detection state.
    std::vector<float> seg_ref_emb_;     // reference embedding for current segment's speaker
    int seg_ref_speaker_id_ = -1;        // speaker ID from early/full extraction
    std::string seg_ref_speaker_name_;   // speaker name from early/full extraction
    float seg_ref_speaker_sim_ = 0.0f;   // speaker similarity from early/full extraction
    int seg_last_recheck_at_ = 0;        // sample position of last re-check
    bool seg_has_ref_ = false;           // true after initial speaker identified

    // SAAS: short-segment speaker inheritance state.
    int prev_seg_speaker_id_ = -1;       // speaker ID from the previous segment
    std::string prev_seg_speaker_name_;  // speaker name from the previous segment
    float prev_seg_speaker_sim_ = 0.0f;  // speaker similarity from the previous segment
    uint64_t prev_seg_end_t1_ = 0;       // AUDIO T1 at end of previous segment

    // AUDIO business clock (T1) — single source of truth.
    //   audio_t1_in_        : samples successfully pushed into the ring buffer
    //                         (written by push_pcm from any thread)
    //   audio_t1_processed_ : samples popped by process_loop and handed to
    //                         downstream stages (touched only by the loop thread)
    // The difference = current ring-buffer occupancy = perception latency.
    std::atomic<uint64_t> audio_t1_in_{0};
    uint64_t audio_t1_processed_ = 0;

    // SAAS: speaker-change ASR split flag.
    bool asr_spk_change_pending_ = false;   // speaker change detected, trigger ASR split
    int  asr_spk_change_split_at_ = 0;      // sample position in asr_pcm_buf_ to split at

    // Speaker timeline: fused speaker resolution across SAAS + Tracker events.
    SpeakerTimeline spk_timeline_;

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
        // Stream position (absolute time from start of audio stream, in seconds).
        float stream_start_sec = 0.0f;  // start of this segment in stream time
        float stream_end_sec   = 0.0f;  // end of this segment in stream time
        // Speaker identification from timeline fusion.
        int speaker_id = -1;            // resolved speaker ID (-1 = unknown)
        std::string speaker_name;       // resolved speaker name
        float speaker_sim = 0.0f;       // similarity from best-authority source
        float speaker_confidence = 0.0f; // timeline fusion confidence (weighted vote)
        std::string speaker_source;      // source name ("SAAS_FULL", "SAAS_CHANGE", etc.)
    };
    std::thread asr_thread_;
    std::mutex asr_mutex_;
    std::condition_variable asr_cv_;
    std::queue<ASRJob> asr_queue_;
    std::atomic<bool> asr_busy_{false};

    // Change detection: previous segment embedding for inter-segment cosine similarity.
    std::vector<float> prev_wlecapa_emb_;  // 192-dim, empty if first segment

    // Overlap detection (pyannote) + speech separator (MossFormer2). Loaded
    // from config but currently not wired into the process loop — they were
    // previously invoked via SpeakerTracker. Retained as owned members so
    // the WebUI toggles, config paths, and test suites continue to work and
    // so future reintroduction has a clean home.
    OverlapDetector overlap_det_;
    SpeechSeparator separator_;
    std::atomic<bool> enable_overlap_det_{true};
    std::atomic<bool> enable_separator_{true};

    OnVadEvent on_vad_;
    OnStats    on_stats_;
    OnSpeaker  on_speaker_;
    OnTranscript on_transcript_;
    OnAsrLog   on_asr_log_;
    OnDrop     on_drop_;
    OnAsrPartial on_asr_partial_;
    // Phase 4 — OratorReclusterer rolling-window re-clustering. Owned by
    // pipeline; instantiated in start() iff cfg_.speaker_reclusterer_enable
    // (or env DEUSRIDET_RECLUSTERER_ENABLE=1). Lives next to dual_db_.
    std::unique_ptr<orator::OratorReclusterer> reclusterer_;
    uint64_t reclusterer_seg_id_ = 0;
    OnSpeakerRelabel on_speaker_relabel_;

    // DiariZen-v2 Hybrid P1 — optional session-level PCM capture.
    // Guarded by diarizen_capture_mu_ because push_pcm() runs on the WS
    // producer thread and dump_wav/clear may be called from the router
    // thread that handles control messages. The enable bit is also exposed
    // as an atomic so push_pcm()'s hot path can short-circuit without
    // taking the mutex when capture is off (the common case).
    mutable std::mutex      diarizen_capture_mu_;
    std::atomic<bool>       diarizen_capture_on_atomic_{false};
    bool                    diarizen_capture_on_ = false;
    size_t                  diarizen_capture_cap_samples_ = 0;
    std::vector<int16_t>    diarizen_capture_buf_;
    // P2: audio_t1 in samples at the moment capture was enabled. Buffer
    // index 0 maps to (origin_samples_ / 16000) stream seconds.
    uint64_t                diarizen_capture_origin_samples_ = 0;
};

} // namespace deusridet
