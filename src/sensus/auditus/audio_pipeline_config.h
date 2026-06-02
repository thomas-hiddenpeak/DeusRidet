/**
 * @file audio_pipeline_config.h
 * @philosophical_role Declares the static shape of the auditory seam — which VAD drives perception and which knobs the online speaker facade consumes. Split from audio_pipeline.h under R1 (2026-06-02).
 * @serves Nexus, Conscientia, Actus (awaken).
 */
#pragma once

#include "mel_gpu.h"
#include "frcrn_enhancer.h"
#include "overlap_detector.h"
#include "speech_separator.h"
#include "silero_vad.h"
#include "../../orator/speaker_encoder.h"

#include <cstddef>
#include <string>

namespace deusridet {

// Which VAD engine drives the speech detection for speaker extraction.
// Historical enum values preserved to keep WebUI + router wire format stable;
// only SILERO and DIRECT are live after FSMN was removed (April 2026,
// data-driven: FSMN lost to Silero at every tested threshold).
enum class VadSource : int {
    SILERO = 0,
    ANY    = 2,  // alias for SILERO (kept for backward-compat WS messages)
    DIRECT = 3,  // bypass VAD — ASR triggers on buffer duration only
};

struct AudioPipelineConfig {
    MelConfig mel;
    FrcrnConfig frcrn;                  // FRCRN speech enhancement (P0)
    OverlapDetectorConfig overlap_det;  // pyannote overlap detection (P1)
    SpeechSeparatorConfig separator;    // MossFormer2 speech separation (P2)
    SileroVadConfig silero;             // Silero VAD model config
    SpeakerEncoderConfig speaker;       // CAM++ speaker encoder config
    std::string wavlm_ecapa_model;         // WavLM-Large+ECAPA-TDNN safetensors path (native GPU)
    float wavlm_ecapa_threshold = 0.55f;   // default cosine sim threshold
    std::string asr_model_path;            // Qwen3-ASR model directory (empty = disabled)
    size_t ring_buffer_bytes = 1 << 22;  // 4 MB (~128 seconds of int16 mono 16kHz)
    int process_chunk_ms     = 100;      // process in 100ms chunks (10 mel frames)
    float speaker_threshold  = 0.45f;    // dual 384D cosine sim match threshold (v22c level)
    // Diarization knobs — overridable via configs/auditus.conf (no rebuild
    // required). Only the four gates the clean three-concern online facade
    // (orator_online) actually consumes survive; the long tail of
    // discovery/recency/short-identify/multi-gate/campp-shadow/inherit-peek
    // knobs from the pre-redesign EMA-library era were removed (2026-06-02
    // anti-entropy pass — all were 0-consumption after the rewrite).
    float speaker_register_threshold = 0.55f; // pending-pool confirmation sim
    float speaker_margin_abstain     = 0.05f; // min (top1 - top2) to trust a match
    // Minimum fbank-frame count required to run CAM++ FULL extraction on a
    // completed speech segment. 150 (~1.5 s): lower floods the library with
    // noisy short-segment embeddings (Step 4b negative result).
    int   speaker_min_fbank_frames   = 150;
    // Phase 4 — OratorReclusterer runtime wiring.
    // When enabled, the segment-end FULL path pushes the fused 384D
    // (CAM++ ⊕ WL-ECAPA) embedding to a rolling-window spectral
    // reclusterer (src/orator/orator_reclusterer.{h,cpp}). Each tick
    // produces a globally-consistent speaker assignment over the last
    // window_sec seconds; whenever the new global id disagrees with
    // the tentative online id, a RelabelEvent is logged (and, in a
    // future revision, surfaced as a Nexus `speaker_relabel` WS event
    // so the WebUI timeline can be patched in place).
    // 2026-05-26 — DEFAULT FLIPPED BACK TO OFF.
    //
    // The previous flip to ON was justified by a fixture-based macro_f1
    // jump (0.5476 → 0.7025 on the fused 60-min fixture). On the live
    // pipeline against tests/test.mp3 + ground_truth.json, however,
    // every reclusterer config (LINK_THRESH ∈ {0.55, 0.60, 0.65, 0.70,
    // 0.75}) collapses all 4 GT speakers into a single global id —
    // 4-way speaker-attribution accuracy = 0%. With the reclusterer
    // OFF, the same audio yields 25.4% best-mapping accuracy and the
    // 4 speakers occupy 4 distinct raw ids. The reclusterer is
    // therefore strictly destructive on multi-speaker meeting audio
    // until its run_pass collapse mechanism is found and fixed.
    // See docs/{en,zh}/devlog/2026-05-27.md "Layer 2c" for the live
    // numbers and the constitutional rule that mandated this revert
    // (accuracy on tests/test.mp3 is the sole metric).
    // Env override: DEUSRIDET_RECLUSTERER_ENABLE=1 re-enables.
    bool  speaker_reclusterer_enable        = false;
    float speaker_reclusterer_window_sec    = 180.0f;
    float speaker_reclusterer_interval_sec  = 30.0f;
    float speaker_reclusterer_link_threshold = 0.55f;
    float speaker_reclusterer_centroid_ema  = 0.20f;
    int   speaker_reclusterer_min_segments  = 12;
    int   speaker_reclusterer_max_segments  = 300;
    int   speaker_reclusterer_min_k         = 2;
    int   speaker_reclusterer_max_k         = 6;
    // Replay speed for benchmark/testing input. 1.0 = real-time; >1.0 means
    // the upstream driver feeds samples faster than wall time (e.g. speed=2.0
    // pushes two seconds of source audio per wall second). This ONLY affects
    // the AUDIO T1 <-> T0 anchor: period_ns is scaled so that T0 tracks wall
    // time regardless of replay rate, keeping cross-domain alignment honest.
    // All pipeline logic (VAD, ASR, thresholds) remains invariant.
    float replay_speed       = 1.0f;
};

}  // namespace deusridet
