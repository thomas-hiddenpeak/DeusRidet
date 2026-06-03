/**
 * @file src/sensus/auditus/audio_pipeline_process_saas_full.cpp
 * @philosophical_role
 *   End-of-segment speaker extraction for the SAAS arm of
 *   AudioPipeline::process_loop. Clean-slate rebuild (Jun 2, 2026,
 *   branch redesign/orator-online-three-concerns).
 *
 *   The legacy implementation collapsed three distinct engineering
 *   problems — WHEN to register, HOW to identify, HOW to distinguish —
 *   into one 973-line greedy forward pass that shared single knobs across
 *   all three (discovery boost, recency bonus, multi-gate probe,
 *   SHORT-IDENTIFY, SI-peek veto/rescue, retro-ring, inherit-broadcast).
 *   Tuning one concern broke the others. It is deleted (recoverable from
 *   git checkpoint 8ffefbc).
 *
 *   The replacement is HUMBLE. It extracts one robust per-segment
 *   embedding, asks orator::OratorOnline for a single decoupled decision
 *   (② read-only judgment that may ABSTAIN, ① evidence-gated registration,
 *   ③ minimal store hygiene), broadcasts only what it can defend, and
 *   pushes every embedding to the offline DiariZen-v2 reclusterer — the
 *   proven layer that owns GLOBAL speaker separation and corrects the
 *   online tentative labels retroactively via the transcript holdback.
 * @serves
 *   Sensus auditus — SAAS end-of-segment identity arm.
 */
#include "audio_pipeline.h"
#include "separatio_orator_probe.h"
#include "../../orator/orator_online.h"
#include "../../communis/log.h"
#include "../../communis/tempus.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: run FULL speaker extraction (CAM++ ⊕ WL-ECAPA fuse) then one
//        decoupled three-concern online decision for an ended segment.
// @param fbank_frames  number of 80-dim fbank frames accumulated in seg_fbank_buf_.
void AudioPipeline::process_saas_full_extract_(int fbank_frames) {
    // Concern boundary: segments too short for a trustworthy FULL embedding are
    // NOT force-labelled online any more. The offline finalize, which sees the
    // whole session at once, owns them. This deletes the entire SHORT-IDENTIFY /
    // INHERIT-BROADCAST machinery that existed only to fake a live label.
    const int kMinFbankFrames = cfg_.speaker_min_fbank_frames;
    if (!(speaker_enc_.initialized() &&
          enable_speaker_.load(std::memory_order_relaxed) &&
          fbank_frames >= kMinFbankFrames)) {
        return;
    }

    auto emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
    if (emb.empty()) return;

    // Build the vector the identity store is keyed on: dual 384D when WL-ECAPA
    // is live, else CAM++ 192D. The fused embedding is also what the offline
    // reclusterer consumes, so it is computed once here.
    std::vector<float> id_emb;
    if (use_dual_encoder_) {
        std::vector<float> wl_emb;
        const int speech_samples = static_cast<int>(speech_pcm_buf_.size());
        if (speech_samples >= 16000) {
            std::vector<float> pcm_f32(speech_samples);
            for (int si = 0; si < speech_samples; ++si)
                pcm_f32[si] = speech_pcm_buf_[si] / 32768.0f;
            std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
            wl_emb = wlecapa_enc_.extract(pcm_f32.data(), speech_samples);
        }
        if (wl_emb.empty() || emb.size() != 192) {
            // No WL-ECAPA this segment ⇒ cannot key the dual store. Stay humble;
            // the offline finalize will resolve this segment's identity.
            LOG_INFO("AudioPipe",
                     "FULL: WL-ECAPA unavailable (samples=%d), abstain — offline finalize owns it",
                     speech_samples);
            return;
        }
        id_emb.resize(384);
        std::copy(emb.begin(), emb.end(), id_emb.begin());
        std::copy(wl_emb.begin(), wl_emb.end(), id_emb.begin() + 192);
        float n2 = 0.0f;
        for (float v : id_emb) n2 += v * v;
        const float inv = 1.0f / std::sqrt(n2 + 1e-12f);
        for (float& v : id_emb) v *= inv;
    } else {
        id_emb = emb;
    }

    SpeakerVectorStore& store = use_dual_encoder_ ? dual_db_ : campp_db_;

    // ── The clean three-concern online decision ──────────────────────────────
    // Thresholds are refreshed from the live atomics each call (they are
    // runtime-configurable); the gate's pending evidence persists in the member.
    orator::OratorOnlineConfig oc;
    oc.judge.match_floor    = speaker_threshold_.load(std::memory_order_relaxed);
    oc.judge.margin_abstain = speaker_margin_abstain_.load(std::memory_order_relaxed);
    oc.reg.coalesce_sim     = speaker_register_threshold_.load(std::memory_order_relaxed);
    oc.reg.confirm_hits     = 2;
    orator_online_.set_config(oc);

    const double now_sec = static_cast<double>(audio_t1_processed_) / 16000.0;
    const orator::OnlineDecision dec = orator_online_.decide(store, id_emb, now_sec);
    campp_full_count_++;

    // Map the decoupled decision onto the SpeakerMatch the broadcast layer
    // consumes. An Abstain leaves speaker_id = -1 (no event emitted).
    SpeakerMatch match{};
    match.speaker_id = (dec.action == orator::OnlineAction::Abstain) ? -1 : dec.speaker_id;
    match.similarity = dec.confidence;
    match.is_new     = dec.is_new;
    match.name       = dec.name;

    stats_.speaker_id      = match.speaker_id;
    stats_.speaker_sim     = match.similarity;
    stats_.speaker_new     = match.is_new;
    stats_.speaker_count   = store.count();
    stats_.speaker_active  = true;
    std::strncpy(stats_.speaker_name, match.name.c_str(), sizeof(stats_.speaker_name) - 1);
    stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';

    LOG_INFO("AudioPipe",
             "FULL: action=%s id=%d sim=%.3f %s(fbank=%d, roster=%d)",
             dec.action == orator::OnlineAction::Broadcast ? "broadcast"
               : dec.action == orator::OnlineAction::Register ? "register" : "abstain",
             match.speaker_id, match.similarity,
             match.is_new ? "NEW " : "", fbank_frames, store.count());

    // Broadcast only a defensible identity. Abstain ⇒ silence on the wire; the
    // offline finalize + holdback fills the label retroactively.
    if (match.speaker_id >= 0 && on_speaker_) on_speaker_(match);

    // ── Concern ③ backstop: push the fused embedding to the offline reclusterer.
    // Every FULL segment is pushed (including abstained ones, with tentative
    // id = -1) so global separation sees the whole stream, not just the online
    // linker's confident subset.
    if (reclusterer_ && use_dual_encoder_ && !id_emb.empty()) {
        orator::ReclusterSegment rseg;
        rseg.segment_id = ++reclusterer_seg_id_;
        const int64_t seg_end_samples   = audio_t1_processed_;
        int64_t       seg_start_samples = seg_end_samples -
                                          static_cast<int64_t>(speech_pcm_buf_.size());
        if (seg_start_samples < 0) seg_start_samples = 0;
        rseg.t_start_sec        = static_cast<double>(seg_start_samples) / 16000.0;
        rseg.t_end_sec          = static_cast<double>(seg_end_samples) / 16000.0;
        rseg.t_center_sec       = 0.5 * (rseg.t_start_sec + rseg.t_end_sec);
        rseg.tentative_speaker_id = match.speaker_id;
        rseg.embedding          = id_emb;  // already L2-normalised
        reclusterer_->push(rseg);
        if (reclusterer_->tick(rseg.t_end_sec) > 0) {
            std::vector<orator::RelabelEvent> evs;
            reclusterer_->drain_relabels(evs);
            for (const auto& ev : evs) {
                LOG_INFO("AudioPipe",
                         "Reclusterer relabel: seg=%lu old=%d new=%d conf=%.3f",
                         static_cast<unsigned long>(ev.segment_id),
                         ev.old_speaker_id, ev.new_speaker_id,
                         static_cast<double>(ev.confidence));
                if (on_speaker_relabel_)
                    on_speaker_relabel_(ev.segment_id, ev.old_speaker_id,
                                        ev.new_speaker_id, ev.confidence);
            }
        }
    }

    // ── Timeline: record only a decided identity (fusion / holdback consumer).
    if (match.speaker_id >= 0) {
        seg_ref_speaker_id_   = match.speaker_id;
        seg_ref_speaker_name_ = match.name;
        seg_ref_speaker_sim_  = match.similarity;
        const int64_t seg_start = audio_t1_processed_ -
                                  static_cast<int64_t>(speech_pcm_buf_.size());
        SpeakerEvent ev{};
        ev.audio_start = seg_start;
        ev.audio_end   = audio_t1_processed_;
        ev.source      = SpkEventSource::SAAS_FULL;
        ev.speaker_id  = match.speaker_id;
        ev.similarity  = match.similarity;
        std::strncpy(ev.name, match.name.c_str(), sizeof(ev.name) - 1);
        spk_timeline_.push(ev);
    }
}

}  // namespace deusridet
