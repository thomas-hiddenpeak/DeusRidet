/**
 * @file src/sensus/auditus/audio_pipeline_process_saas_full.cpp
 * @philosophical_role
 *   Stage-extract of AudioPipeline::process_loop (Step 11 A1b).
 *   End-of-segment CAM++ FULL speaker extraction.
 *
 *   When a speech segment ends, fbank frames accumulated during the
 *   segment are handed to the CAM++ encoder to produce a single robust
 *   embedding. This embedding feeds (a) overlap-guarded match/register
 *   against the speaker DB, and (b) dual-encoder CAM++||WL-ECAPA fusion
 *   when enabled.
 *
 *   (Historical note: an online spectral-clustering warm-up lived here
 *   through Step 11 A1b. It was disabled by test results and removed in
 *   Step 14a — see docs/{en,zh}/devlog/ for the failure record.)
 * @serves
 *   Sensus auditus — SAAS end-of-segment identity arm.
 */
#include "audio_pipeline.h"
#include "separatio_orator_probe.h"
#include "../../orator/orator_online_judgment.h"  // ② concern seam (redesign S1)
#include "../../communis/log.h"
#include "../../communis/tempus.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace deusridet {

// @role: run FULL CAM++ extraction + dual-encoder fuse + spectral warmup for an ended segment.
// @param fbank_frames  number of 80-dim fbank frames accumulated in seg_fbank_buf_.
void AudioPipeline::process_saas_full_extract_(int fbank_frames) {
                // CAM++ speaker encoder — FULL extraction using accumulated fbank.
                // Minimum-frames gate is configurable (speaker_min_fbank_frames);
                // default 50 (~500 ms) is the ECAPA-style stat-pooling floor,
                // not a test-recording-specific choice. Segments below this
                // threshold are dropped outright: no embedding, no identity.
                const int kMinFbankFrames = cfg_.speaker_min_fbank_frames;
                if (speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    fbank_frames >= kMinFbankFrames) {
                        float thresh = speaker_threshold_.load(std::memory_order_relaxed);
                        float reg_thresh = speaker_register_threshold_.load(std::memory_order_relaxed);

                        auto emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
                        if (!emb.empty()) {
                            bool auto_reg = true;

                            // Step 19c — VAD-internal multi-speaker probe.
                            // Slides a 1.5 s / 0.5 s CAM++ window over
                            // seg_fbank_buf_ and finds the minimum adjacent
                            // window cosine. Empirical AUC 0.819 vs GT
                            // single/multi label on 47 long VADs (devlog
                            // 2026-05-22). When ON and min_cos drops below
                            // speaker_multi_gate_threshold, the identify
                            // path switches to peek_best (no exemplar
                            // admission, no auto-register) and the retro
                            // ring cache + speaker timeline updates are
                            // skipped — so a mixed-speaker VAD can never
                            // pollute cluster centroids.
                            bool multi_speaker_suspect = false;
                            float multi_gate_min_cos = 1.0f;
                            int   multi_gate_n_windows = 0;
                            if (cfg_.speaker_multi_gate_enable &&
                                fbank_frames >= cfg_.speaker_multi_gate_min_fbank) {
                                const int win_f = 150;  // 1.5 s
                                const int hop_f = 50;   // 0.5 s
                                std::vector<std::vector<float>> wembs;
                                for (int ws = 0; ws + win_f <= fbank_frames; ws += hop_f) {
                                    auto we = speaker_enc_.extract(
                                        seg_fbank_buf_.data() + (size_t)ws * 80, win_f);
                                    if ((int)we.size() != 192) continue;
                                    float n2 = 0.0f;
                                    for (float v : we) n2 += v * v;
                                    float inv = 1.0f / std::sqrt(n2 + 1e-12f);
                                    for (float& v : we) v *= inv;
                                    wembs.emplace_back(std::move(we));
                                }
                                multi_gate_n_windows = (int)wembs.size();
                                for (size_t i = 1; i < wembs.size(); ++i) {
                                    float dot = 0.0f;
                                    for (int d = 0; d < 192; ++d)
                                        dot += wembs[i - 1][d] * wembs[i][d];
                                    if (dot < multi_gate_min_cos) multi_gate_min_cos = dot;
                                }
                                if (multi_gate_n_windows >= 2 &&
                                    multi_gate_min_cos < cfg_.speaker_multi_gate_threshold) {
                                    multi_speaker_suspect = true;
                                    LOG_INFO("AudioPipe",
                                             "MULTI-GATE flagged: min_cos=%.3f < %.2f windows=%d fbank=%d",
                                             multi_gate_min_cos,
                                             cfg_.speaker_multi_gate_threshold,
                                             multi_gate_n_windows, fbank_frames);
                                    auto_reg = false;
                                }
                            }

                            // Late registration cap — after N FULL identifications
                            // every legitimate speaker should already be registered.
                            // Further registrations have historically been drift
                            // clones (observed in v8/v9 test runs at count ≈ 180).
                            // Tunable: configs/auditus.conf:speaker_max_auto_reg_count
                            const int kMaxAutoRegCount = cfg_.speaker_max_auto_reg_count;
                            if (campp_full_count_ >= kMaxAutoRegCount) {
                                auto_reg = false;
                            }
                            float match_thresh = thresh;

                            // v24d: Discovery phase — use higher threshold during
                            // early extractions to force speaker separation.
                            // Without this, similar speakers (e.g. 徐子景/朱杰)
                            // get absorbed into the first registered speaker.
                            // Tunable: configs/auditus.conf:speaker_discovery_{count,boost}
                            const int   kDiscoveryCount = cfg_.speaker_discovery_count;
                            const float kDiscoveryBoost = cfg_.speaker_discovery_boost;
                            const float kDiscoveryRegRelax = cfg_.speaker_discovery_reg_relax;
                            if (campp_full_count_ < kDiscoveryCount) {
                                match_thresh += kDiscoveryBoost;
                                // Step 16e: Symmetric relaxation of pending-pool
                                // confirmation threshold during discovery. The
                                // discovery_boost makes matching to existing
                                // clusters stricter; without a symmetric relax
                                // on pending confirmation, quiet or tonally
                                // similar new speakers (self-sim ~0.52) never
                                // coalesce into their own cluster and get
                                // silently absorbed into neighbors. Match
                                // harder + pending softer is the right shape
                                // for cold-start. Recency guard (Step 16c)
                                // still prevents low-sim absorptions later.
                                reg_thresh -= kDiscoveryRegRelax;
                            }                            // v24: Temporal recency bonus — lower threshold when recent
                            // speaker still active, reducing false negatives (fragmentation).
                            float seg_mid_time = (float)(audio_t1_processed_ - (int64_t)speech_pcm_buf_.size() / 2) / 16000.0f;
                            float time_since_prev = seg_mid_time - prev_full_time_;
                            // Tunable: configs/auditus.conf:speaker_recency_{window_sec,bonus}
                            const float kRecencyWindow = cfg_.speaker_recency_window_sec;
                            const float kRecencyBonus  = cfg_.speaker_recency_bonus;
                            // Step 16 iter 1: recency is a post-discovery
                            // stabilizer only. During discovery (first
                            // kDiscoveryCount FULL extractions) its combined
                            // effect — lowered threshold + auto_reg=false —
                            // absorbs every cold-start speaker into spk0
                            // (baseline: 48.4% GT-side on seg0 0–600s). We
                            // therefore gate recency on post-discovery so
                            // newcomers within 15 s of another speaker still
                            // get a clean registration chance at reg_thresh.
                            bool recency_active =
                                (prev_full_speaker_id_ >= 0 &&
                                 time_since_prev < kRecencyWindow &&
                                 campp_full_count_ >= kDiscoveryCount);
                            if (recency_active) {
                                match_thresh -= kRecencyBonus;
                                // v32: restored from v30 — lowered threshold
                                // must NOT allow new-speaker registration.
                                // v12 showed spk1 registering at mt=0.47 and
                                // merging into spk0, scrambling all mappings.
                                auto_reg = false;
                            }

                            SpeakerMatch match;
                            std::vector<float> wl_emb;  // hoisted for warmup reuse
                            std::vector<float> dual_emb;  // hoisted for reclusterer push (Phase 4)
                            if (use_dual_encoder_) {
                                // Dual-encoder: concatenate CAM++ + WL-ECAPA → 384D.
                                int speech_samples = (int)speech_pcm_buf_.size();
                                if (speech_samples >= 16000) {
                                    std::vector<float> pcm_f32(speech_samples);
                                    for (int si = 0; si < speech_samples; si++)
                                        pcm_f32[si] = speech_pcm_buf_[si] / 32768.0f;
                                    std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
                                    wl_emb = wlecapa_enc_.extract(pcm_f32.data(), speech_samples);
                                }
                                if (!wl_emb.empty()) {
                                    // Build 384D vector: [CAM++ 192D | WL-ECAPA 192D], L2-normalized.
                                    std::vector<float> dual(384);
                                    std::copy(emb.begin(), emb.end(), dual.begin());
                                    std::copy(wl_emb.begin(), wl_emb.end(), dual.begin() + 192);
                                    float n2 = 0;
                                    for (float v : dual) n2 += v * v;
                                    float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                    for (float& v : dual) v *= inv;
                                    if (multi_speaker_suspect) {
                                        match = dual_db_.peek_best(dual);
                                    } else {
                                        match = dual_db_.identify(dual, match_thresh,
                                                                  auto_reg, reg_thresh);
                                    }
                                    dual_emb = std::move(dual);  // Phase 4: preserve for reclusterer
                                } else {
                                    // WL-ECAPA extraction failed (segment too short).
                                    // Skip — don't fallback to different ID space.
                                    LOG_INFO("AudioPipe", "CAM++ FULL: WL-ECAPA failed, skip dual identify");
                                }
                            } else {
                                if (multi_speaker_suspect) {
                                    match = campp_db_.peek_best(emb);
                                } else {
                                    match = campp_db_.identify(emb, match_thresh,
                                                               auto_reg, reg_thresh);
                                }
                            }

                            // v24: Recency validation — if threshold was lowered and matched
                            // a DIFFERENT speaker than the recent one, discard the match and
                            // re-run at standard threshold to avoid false positives.
                            if (recency_active && match.speaker_id >= 0 &&
                                match.speaker_id != prev_full_speaker_id_ &&
                                match.similarity < thresh) {
                                LOG_INFO("AudioPipe", "Recency: matched #%d(%.3f) != prev #%d, re-check at %.2f",
                                         match.speaker_id, match.similarity, prev_full_speaker_id_, thresh);
                                // Re-identify at standard threshold (reuse wl_emb).
                                if (use_dual_encoder_ && !wl_emb.empty()) {
                                    std::vector<float> dual(384);
                                    std::copy(emb.begin(), emb.end(), dual.begin());
                                    std::copy(wl_emb.begin(), wl_emb.end(), dual.begin() + 192);
                                    float n2 = 0;
                                    for (float v : dual) n2 += v * v;
                                    float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                    for (float& v : dual) v *= inv;
                                    match = dual_db_.identify(dual, thresh, auto_reg, reg_thresh);
                                } else if (!use_dual_encoder_) {
                                    match = campp_db_.identify(emb, thresh, auto_reg, reg_thresh);
                                }
                            }

                            // Step 16c: recency-absorb guard — when recency was
                            // active AND we accepted a match below the unmodified
                            // threshold, abstain even if the matched id equals
                            // prev. Manual review of 1x baseline on tests/test.mp3
                            // showed 徐子景 @05:21 being absorbed into the 唐云峰
                            // cluster at sim<0.50 because the recency bonus
                            // dropped match_thresh to 0.45 AND the v24 re-check
                            // above only fires when matched id != prev. That
                            // asymmetry lets cross-talk paint the wrong cluster.
                            // Abstaining here keeps the next FULL extraction free
                            // to register the true incoming speaker.
                            if (recency_active && match.speaker_id >= 0 && !match.is_new &&
                                match.similarity < thresh) {
                                LOG_INFO("AudioPipe", "Recency absorb-guard: sim=%.3f < thresh=%.2f (matched #%d=prev); abstain",
                                         match.similarity, thresh, match.speaker_id);
                                match.speaker_id = -1;
                                match.similarity = 0;
                                match.name.clear();
                            }

                            // Margin gate: abstain on ambiguous matches where
                            // top-1 and top-2 are too close to distinguish.
                            // Tunable: configs/auditus.conf:speaker_margin_abstain
                            const float kMarginAbstainThresh = cfg_.speaker_margin_abstain;
                            if (match.speaker_id >= 0 && !match.is_new &&
                                match.second_best_id >= 0 &&
                                (match.similarity - match.second_best_sim) < kMarginAbstainThresh) {
                                LOG_INFO("AudioPipe", "FULL margin-abstain: id=%d sim=%.3f 2nd=#%d(%.3f) margin=%.3f < %.2f",
                                         match.speaker_id, match.similarity,
                                         match.second_best_id, match.second_best_sim,
                                         match.similarity - match.second_best_sim, kMarginAbstainThresh);
                                match.speaker_id = -1;
                                match.similarity = 0;
                                match.name.clear();
                            }

                            stats_.speaker_id = match.speaker_id;
                            stats_.speaker_sim = match.similarity;
                            stats_.speaker_new = match.is_new;
                            stats_.speaker_count = use_dual_encoder_ ? dual_db_.count() : campp_db_.count();
                            stats_.speaker_active = true;
                            stats_.speaker_exemplars = match.exemplar_count;
                            stats_.speaker_hits_above = match.hits_above;

                            campp_full_count_++;

                            // Step 24: CAM++ shadow store. In dual-encoder mode,
                            // mirror this admission into campp_db_ using the same
                            // external_id so the SI-skip-wl path (samples<16000,
                            // WL empty) can fall back to campp_db_.peek_best(si_emb)
                            // and emit a directly-usable label. Without this mirror
                            // 76% of the 108 no_segment GTs on the 1800s fixture
                            // never get labeled. Only fires when the FULL path
                            // produced a confirmed speaker_id (post-abstain).
                            if (use_dual_encoder_ && match.speaker_id >= 0 &&
                                cfg_.speaker_campp_shadow_enable) {
                                if (match.is_new) {
                                    int sid = campp_db_.register_speaker_with_id(
                                        match.speaker_id, emb);
                                    if (sid < 0) {
                                        // Already registered (raced or rebuilt) →
                                        // fall through to add_exemplar.
                                        campp_db_.add_exemplar(match.speaker_id, emb);
                                    }
                                } else {
                                    // Existing dual_db_ speaker → add a 192-D
                                    // CAM++ exemplar to the shadow. add_exemplar
                                    // is a no-op (returns false) when the id has
                                    // not yet been mirrored — first SI-skip
                                    // events will simply miss until a FULL with
                                    // is_new=true registers the speaker, which
                                    // matches dual_db_'s own coldstart timing.
                                    campp_db_.add_exemplar(match.speaker_id, emb);
                                }
                            }

                            strncpy(stats_.speaker_name, match.name.c_str(),
                                    sizeof(stats_.speaker_name) - 1);
                            stats_.speaker_name[sizeof(stats_.speaker_name) - 1] = '\0';
                            LOG_INFO("AudioPipe", "FULL: id=%d sim=%.3f 2nd=#%d(%.3f) m=%.3f %s%s (fbank=%d, ex=%d, recency=%s, mt=%.2f, rt=%.2f)",
                                     match.speaker_id, match.similarity,
                                     match.second_best_id, match.second_best_sim,
                                     match.similarity - match.second_best_sim,
                                     match.is_new ? "NEW " : "",
                                     match.name.empty() ? "(unnamed)" : match.name.c_str(),
                                     fbank_frames, match.exemplar_count,
                                     recency_active ? "ON" : "off", match_thresh, reg_thresh);
                            if (on_speaker_) on_speaker_(match);

                            // Phase 4 — OratorReclusterer push + tick + drain.
                            // When enabled, push the finalized fused embedding to
                            // the rolling-window re-clusterer; periodically the
                            // tick() pass runs spectral re-clustering over the
                            // last window_sec seconds and surfaces RelabelEvents
                            // whenever the global identity differs from the
                            // online tentative id. The events are logged here
                            // and forwarded to on_speaker_relabel_ for the Nexus
                            // layer to publish as a `speaker_relabel` WS event.
                            //
                            // Phase 3b.2 audit (env-gated). DEUSRIDET_LIVE_PUSH_DEBUG=1
                            // enables a per-FULL-extraction stderr line and a
                            // rolling counter so we can compare the segment
                            // stream the LIVE pipeline feeds to the reclusterer
                            // against the GT-aligned fused_v1.bin fixture
                            // (which is gate-free by construction). Zero cost
                            // when the env var is unset.
                            static const bool kLivePushDebug = []{
                                const char* e = std::getenv("DEUSRIDET_LIVE_PUSH_DEBUG");
                                return e != nullptr && e[0] != '\0' && e[0] != '0';
                            }();
                            // Phase 3b.3 A/B. DEUSRIDET_RECLUSTERER_ACCEPT_ABSTAINED=1
                            // lifts the `match.speaker_id >= 0` clause from the push
                            // gate so abstain-paths (identify(-1), recency absorb-guard,
                            // margin-abstain) still hand their embedding to the
                            // reclusterer. The 3b.2 audit on tests/test.mp3 s600 showed
                            // 38 / 72 FULL extractions (53%) were dropped by this
                            // gate. The reclusterer's whole purpose is to override
                            // the online linker's judgement; filtering its input
                            // by the linker's verdict is a structural inversion.
                            // Abstain segments enter with tentative_speaker_id=-1,
                            // so emitted RelabelEvents will carry old_id=-1 — the
                            // WebUI patcher already tolerates unknown old ids.
                            static const bool kAcceptAbstained = []{
                                const char* e = std::getenv("DEUSRIDET_RECLUSTERER_ACCEPT_ABSTAINED");
                                return e != nullptr && e[0] != '\0' && e[0] != '0';
                            }();
                            const bool push_gate_ok =
                                reclusterer_ && use_dual_encoder_ && !dual_emb.empty() &&
                                (kAcceptAbstained || match.speaker_id >= 0);
                            if (kLivePushDebug) {
                                const char* reject_reason = nullptr;
                                if (!reclusterer_)        reject_reason = "no_recluster";
                                else if (!use_dual_encoder_) reject_reason = "no_dual_enc";
                                else if (dual_emb.empty())   reject_reason = "empty_dual_emb";
                                else if (!push_gate_ok)      reject_reason = "sid_negative";
                                if (reject_reason) {
                                    fprintf(stderr,
                                            "[live-push-audit] REJECT reason=%s sid=%d is_new=%d sim=%.3f fbank=%d t_end=%.3fs\n",
                                            reject_reason,
                                            match.speaker_id,
                                            match.is_new ? 1 : 0,
                                            (double)match.similarity,
                                            fbank_frames,
                                            (double)audio_t1_processed_ / 16000.0);
                                }
                            }
                            if (push_gate_ok) {
                                orator::ReclusterSegment rseg;
                                rseg.segment_id = ++reclusterer_seg_id_;
                                int64_t seg_end_samples   = audio_t1_processed_;
                                int64_t seg_start_samples = seg_end_samples -
                                                            (int64_t)speech_pcm_buf_.size();
                                if (seg_start_samples < 0) seg_start_samples = 0;
                                rseg.t_start_sec  = (double)seg_start_samples / 16000.0;
                                rseg.t_end_sec    = (double)seg_end_samples   / 16000.0;
                                rseg.t_center_sec = 0.5 * (rseg.t_start_sec + rseg.t_end_sec);
                                rseg.tentative_speaker_id = match.speaker_id;
                                rseg.embedding = dual_emb;  // already L2-normalised
                                if (kLivePushDebug) {
                                    double l2_sq = 0.0;
                                    for (float v : dual_emb) l2_sq += (double)v * v;
                                    fprintf(stderr,
                                            "[live-push-audit] ACCEPT seg=%lu raw=%d t=[%.3f,%.3f] is_new=%d sim=%.3f L2=%.6f dim=%zu\n",
                                            (unsigned long)reclusterer_seg_id_,
                                            match.speaker_id,
                                            rseg.t_start_sec, rseg.t_end_sec,
                                            match.is_new ? 1 : 0,
                                            (double)match.similarity,
                                            std::sqrt(l2_sq),
                                            dual_emb.size());
                                }
                                reclusterer_->push(rseg);
                                int n_emit = reclusterer_->tick(rseg.t_end_sec);
                                if (n_emit > 0) {
                                    std::vector<orator::RelabelEvent> evs;
                                    reclusterer_->drain_relabels(evs);
                                    for (const auto& ev : evs) {
                                        LOG_INFO("AudioPipe",
                                                 "Reclusterer relabel: seg=%lu old=%d new=%d conf=%.3f",
                                                 (unsigned long)ev.segment_id,
                                                 ev.old_speaker_id, ev.new_speaker_id,
                                                 (double)ev.confidence);
                                        if (on_speaker_relabel_) {
                                            on_speaker_relabel_(ev.segment_id,
                                                                ev.old_speaker_id,
                                                                ev.new_speaker_id,
                                                                ev.confidence);
                                        }
                                    }
                                }
                            }

                            // Step 17a — retro-relabel ring & scan.
                            //
                            // (1) Cache this FULL extraction's embedding so a
                            // future cluster birth can re-evaluate it. We
                            // store the same 384D dual vector the identify
                            // path consumed, or the 192D CAM++ embedding when
                            // dual-encoder is disabled. Slots are L2-normalised
                            // to keep peek_best a pure cosine search.
                            // (2) When match.is_new is true, a brand-new
                            // cluster was just confirmed from the pending
                            // pool. The store now contains an exemplar that
                            // didn't exist when the cached entries were
                            // identified. We peek_best each cached embedding
                            // against the freshened store; entries that now
                            // match the new cluster — especially those that
                            // abstained or were pulled into a neighbour at
                            // borderline margin — are the cold-start tail
                            // we cannot fix with any threshold knob (Step 16g
                            // ceiling analysis, devlog 05ce56a).
                            //
                            // 17a is diagnostic-only: candidates are LOGged,
                            // not amended on the wire. 17b will broadcast a
                            // speaker_amend frame and update the replay
                            // scorer.
                            // Step 19c: skip retro-ring caching for
                            // multi-speaker-suspect segments — their
                            // embedding is a fusion of multiple voices and
                            // would propagate wrong amend candidates.
                            if (!multi_speaker_suspect) {
                                RetroFullSlot slot;
                                slot.audio_end_samples = audio_t1_processed_;
                                slot.decided_id  = match.speaker_id;
                                slot.decided_sim = match.similarity;
                                slot.abstained   = (match.speaker_id < 0);
                                if (use_dual_encoder_ && !wl_emb.empty() &&
                                    emb.size() == 192) {
                                    slot.embedding.resize(384);
                                    std::copy(emb.begin(), emb.end(),
                                              slot.embedding.begin());
                                    std::copy(wl_emb.begin(), wl_emb.end(),
                                              slot.embedding.begin() + 192);
                                    float n2 = 0.0f;
                                    for (float v : slot.embedding) n2 += v * v;
                                    float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                    for (float& v : slot.embedding) v *= inv;
                                } else if (!use_dual_encoder_ &&
                                           emb.size() == 192) {
                                    slot.embedding = emb;
                                }
                                if (!slot.embedding.empty()) {
                                    retro_full_ring_.push(std::move(slot));
                                }
                            }

                            if (match.is_new && match.speaker_id >= 0) {
                                int new_cluster = match.speaker_id;
                                retro_full_ring_.for_each(
                                    [&](const RetroFullSlot& s) {
                                    if (s.decided_id == new_cluster) return;
                                    SpeakerMatch peek = use_dual_encoder_
                                        ? dual_db_.peek_best(s.embedding)
                                        : campp_db_.peek_best(s.embedding);
                                    if (peek.speaker_id != new_cluster) return;
                                    // Abstained: any plausible match is news.
                                    // Mis-routed: require the new cluster to
                                    // win by a clear margin over the prior
                                    // decision so we don't oscillate.
                                    bool worth_logging =
                                        s.abstained
                                            ? (peek.similarity >= thresh - 0.05f)
                                            : (peek.similarity >= thresh &&
                                               peek.similarity - s.decided_sim
                                                   >= 0.05f);
                                    if (!worth_logging) return;
                                    float t_close_sec =
                                        (float)s.audio_end_samples / 16000.0f;
                                    LOG_INFO("AudioPipe",
                                             "RETRO-CANDIDATE t_close=%.2fs "
                                             "prior_id=%d prior_sim=%.3f "
                                             "-> new_id=%d new_sim=%.3f "
                                             "2nd=#%d(%.3f) abstain=%d",
                                             t_close_sec,
                                             s.decided_id, s.decided_sim,
                                             new_cluster, peek.similarity,
                                             peek.second_best_id,
                                             peek.second_best_sim,
                                             (int)s.abstained);

                                    const float amend_min_sim =
                                        std::max(reg_thresh, thresh + 0.10f);
                                    const float amend_gain =
                                        peek.similarity - s.decided_sim;
                                    const float amend_margin =
                                        peek.second_best_id >= 0
                                            ? peek.similarity - peek.second_best_sim
                                            : 1.0f;
                                    bool worth_amending =
                                        s.abstained
                                            ? (peek.similarity >= amend_min_sim &&
                                               amend_margin >= kMarginAbstainThresh)
                                            : (peek.similarity >= amend_min_sim &&
                                               amend_gain >= 0.10f &&
                                               amend_margin >= kMarginAbstainThresh);
                                    if (!worth_amending) return;

                                    SpeakerMatch amend = peek;
                                    amend.is_amend = true;
                                    amend.is_new = false;
                                    amend.prior_speaker_id = s.decided_id;
                                    amend.prior_similarity = s.decided_sim;
                                    amend.amend_t_close_sec = t_close_sec;
                                    LOG_INFO("AudioPipe",
                                             "RETRO-AMEND t_close=%.2fs "
                                             "prior_id=%d prior_sim=%.3f "
                                             "-> new_id=%d new_sim=%.3f "
                                             "margin=%.3f gain=%.3f",
                                             t_close_sec,
                                             s.decided_id, s.decided_sim,
                                             new_cluster, peek.similarity,
                                             amend_margin, amend_gain);
                                    if (on_speaker_) on_speaker_(amend);
                                });
                            }

                            // DEBUG: dump embedding for offline clustering analysis.
                            // Format per record (1560 bytes):
                            //   float32 timestamp, int32 speaker_id, int32 fbank_frames,
                            //   float32 similarity, float32[192] campp, float32[192] wavlm
                            {
                                static FILE* emb_fp = nullptr;
                                if (!emb_fp) emb_fp = fopen("/tmp/spk_embeddings.bin", "ab");
                                if (emb_fp) {
                                    float ts_val = seg_mid_time;
                                    int32_t sid = match.speaker_id;
                                    int32_t fb = fbank_frames;
                                    float sim = match.similarity;
                                    fwrite(&ts_val, 4, 1, emb_fp);
                                    fwrite(&sid, 4, 1, emb_fp);
                                    fwrite(&fb, 4, 1, emb_fp);
                                    fwrite(&sim, 4, 1, emb_fp);
                                    // CAM++ 192D (already L2-normalized by encoder)
                                    if (emb.size() == 192) {
                                        fwrite(emb.data(), 4, 192, emb_fp);
                                    } else {
                                        float zeros[192] = {};
                                        fwrite(zeros, 4, 192, emb_fp);
                                    }
                                    // WavLM-ECAPA 192D
                                    if (wl_emb.size() == 192) {
                                        fwrite(wl_emb.data(), 4, 192, emb_fp);
                                    } else {
                                        float zeros[192] = {};
                                        fwrite(zeros, 4, 192, emb_fp);
                                    }
                                    fflush(emb_fp);
                                }
                            }

                            // Update recency tracking + run-length.
                            // Step 19c: skip recency update on multi-speaker
                            // suspect — a mixed-VAD's best peek match is
                            // not a reliable "previous speaker" anchor.
                            if (match.speaker_id >= 0 && !multi_speaker_suspect) {
                                if (match.speaker_id == prev_full_speaker_id_) {
                                    speaker_run_length_++;
                                } else {
                                    speaker_run_length_ = 1;
                                }
                                prev_full_speaker_id_ = match.speaker_id;
                                prev_full_time_ = seg_mid_time;
                                prev_full_speaker_name_ = match.name;  // v29
                            }

                            // SAAS: feed result into speaker timeline.
                            if (match.speaker_id >= 0) {
                                seg_ref_speaker_id_ = match.speaker_id;
                                seg_ref_speaker_name_ = match.name;
                                seg_ref_speaker_sim_ = match.similarity;
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

                // === Step 19b: SHORT-IDENTIFY-ONLY rescue ====================
                // Empirical (devlog 2026-05-22): 54.5% of all-GT segments are
                // isolated speech with VAD duration 0.5–1.5 s, which falls
                // below kMinFbankFrames (150 = 1.5 s) and so never produces
                // a runtime decision. When speaker_short_identify_enable is
                // ON and fbank_frames falls in the band
                // [speaker_min_fbank_frames_identify, kMinFbankFrames), we
                // still run the encoder but use SpeakerVectorStore::peek_best
                // — read-only cosine search against existing clusters, NO
                // register, NO EMA, NO pending, NO exemplar admission. This
                // can ONLY produce matches against speakers the FULL path
                // has already learned, so it cannot pollute centroids
                // (avoiding the Step 4b regression).
                const int  kMinFbankFramesIdent = cfg_.speaker_min_fbank_frames_identify;
                bool short_identify_broadcast = false;
                // Step 25a: capture SI peek (best id/sim) into outer scope
                // so the INHERIT-BROADCAST block can veto on a confidently-
                // different opinion. Stays at -1/0.0 when SI didn't run or
                // produced no candidate.
                int         si_peek_id  = -1;
                float       si_peek_sim = 0.0f;
                std::string si_peek_name;
                if (cfg_.speaker_short_identify_enable &&
                    speaker_enc_.initialized() &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    fbank_frames < kMinFbankFrames &&
                    fbank_frames >= kMinFbankFramesIdent) {
                    float si_thresh = cfg_.speaker_short_identify_threshold;
                    float si_margin = cfg_.speaker_short_identify_margin;
                    auto si_emb = speaker_enc_.extract(seg_fbank_buf_.data(), fbank_frames);
                    if (!si_emb.empty()) {
                        SpeakerMatch peek;
                        peek.speaker_id = -1;
                        peek.similarity = 0.0f;
                        bool peek_ok = false;
                        // Step 24-b: SI-skip-wl shadow fallback uses
                        // its own (tighter) gates — CAM++ single-encoder
                        // discrimination is weaker than dual 384-D.
                        bool peek_from_shadow = false;

                        if (use_dual_encoder_) {
                            // Dual-encoder mode populates only dual_db_. We
                            // must produce a 384-D fused vector or abstain;
                            // falling back to campp_db_ would be a different
                            // ID namespace.
                            // Step 19d note: zero-padding short PCM to 1 s so
                            // WL-ECAPA can run is tempting but produces an
                            // uninformative embedding (silence dominates the
                            // stat pool), which pulls peek_best toward the
                            // most-trained centroid. That cascades through
                            // INHERIT-BROADCAST and collapses macro accuracy.
                            // We hard-require ≥1 s of real speech instead.
                            //
                            // Step 23 — tile-padding for sub-1 s speech.
                            // Zero-padding fails because silence dominates
                            // the ECAPA stat pool. **Tile-padding** (loop
                            // the actual speech samples up to 1 s) preserves
                            // the mean/variance/energy stats while filling
                            // the required temporal context. Diagnostic on
                            // the 1800 s fixture (Step 22 sweep log) showed
                            // 223 "WL-ECAPA empty (samples<16000)" skips —
                            // the largest single source of lost short-band
                            // VADs. Tile-padding lets these VADs participate
                            // in SI peek_best (still read-only — no admit,
                            // no EMA, no exemplar). Floor at
                            // speaker_si_wl_tile_min_samples (default 8000 =
                            // 0.5 s of real speech) to avoid extreme
                            // duplication factors.
                            int speech_samples = (int)speech_pcm_buf_.size();
                            const int kTileTarget    = 16000;
                            // Step 23 r1/r2: kTileMinSource=8000 (factor up
                            // to 2×) recovered ~30-50 SI events of the 223
                            // skipped but doubled run-to-run cov σ from
                            // 0.013 to 0.022, washing out the signal. ECAPA
                            // stat pool is sensitive to the periodicity
                            // artefact of low-factor repetition. Tighten to
                            // ≥12000 source (max 1.33× repetition) to keep
                            // the tile-pad path strictly low-distortion.
                            const int kTileMinSource = 12000;  // 0.75 s
                            bool tile_pad_enable = cfg_.speaker_si_wl_tile_pad_enable;
                            if (const char* env = std::getenv("DEUSRIDET_SI_WL_TILE_PAD")) {
                                tile_pad_enable = (env[0] == '1' || env[0] == 't' || env[0] == 'T');
                            }
                            std::vector<float> wl_emb;
                            if (speech_samples >= kTileTarget) {
                                std::vector<float> pcm_f32(speech_samples);
                                for (int si = 0; si < speech_samples; si++)
                                    pcm_f32[si] = speech_pcm_buf_[si] / 32768.0f;
                                std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
                                wl_emb = wlecapa_enc_.extract(pcm_f32.data(), speech_samples);
                            } else if (tile_pad_enable &&
                                       speech_samples >= kTileMinSource) {
                                // Tile-pad: repeat the speech to reach 1 s.
                                std::vector<float> pcm_f32(kTileTarget);
                                for (int si = 0; si < kTileTarget; si++)
                                    pcm_f32[si] = speech_pcm_buf_[si % speech_samples] / 32768.0f;
                                std::lock_guard<std::mutex> lock(auditus_wlecapa_extract_mutex());
                                wl_emb = wlecapa_enc_.extract(pcm_f32.data(), kTileTarget);
                                LOG_INFO("AudioPipe",
                                         "SHORT-IDENTIFY tile-pad: src=%d -> %d (factor=%.2f)",
                                         speech_samples, kTileTarget,
                                         (float)kTileTarget / (float)speech_samples);
                            }
                            if (!wl_emb.empty()) {
                                std::vector<float> dual(384);
                                std::copy(si_emb.begin(), si_emb.end(), dual.begin());
                                std::copy(wl_emb.begin(), wl_emb.end(), dual.begin() + 192);
                                float n2 = 0.0f;
                                for (float v : dual) n2 += v * v;
                                float inv = 1.0f / sqrtf(n2 + 1e-12f);
                                for (float& v : dual) v *= inv;
                                peek = dual_db_.peek_best(dual);
                                peek_ok = true;
                                si_peek_id   = peek.speaker_id;
                                si_peek_sim  = peek.similarity;
                                si_peek_name = peek.name;
                            } else if (cfg_.speaker_campp_shadow_enable &&
                                       campp_db_.count() > 0) {
                                // Step 24: CAM++ shadow store fallback. WL-ECAPA
                                // unavailable for samples<16000 — but campp_db_
                                // has been mirrored from every FULL admission
                                // under the SAME external_id, so a CAM++-only
                                // peek_best returns a directly-usable label.
                                peek = campp_db_.peek_best(si_emb);
                                peek_ok = true;
                                peek_from_shadow = true;
                                si_peek_id   = peek.speaker_id;
                                si_peek_sim  = peek.similarity;
                                si_peek_name = peek.name;
                                LOG_INFO("AudioPipe",
                                         "SHORT-IDENTIFY campp-shadow: samples=%d id=%d sim=%.3f 2nd=#%d(%.3f)",
                                         speech_samples, peek.speaker_id, peek.similarity,
                                         peek.second_best_id, peek.second_best_sim);
                            } else {
                                LOG_INFO("AudioPipe",
                                         "SHORT-IDENTIFY skip: WL-ECAPA empty (samples=%d < 16000)",
                                         speech_samples);
                            }
                        } else {
                            peek = campp_db_.peek_best(si_emb);
                            peek_ok = true;
                            si_peek_id   = peek.speaker_id;
                            si_peek_sim  = peek.similarity;
                            si_peek_name = peek.name;
                        }

                        if (peek_ok && peek.speaker_id >= 0 && peek.similarity >= si_thresh) {
                            // Step 24-b: apply tighter gates for shadow path.
                            float eff_thresh = si_thresh;
                            float eff_margin = si_margin;
                            if (peek_from_shadow) {
                                if (cfg_.speaker_campp_shadow_threshold > 0.0f)
                                    eff_thresh = cfg_.speaker_campp_shadow_threshold;
                                if (cfg_.speaker_campp_shadow_margin > 0.0f)
                                    eff_margin = cfg_.speaker_campp_shadow_margin;
                            }
                            if (peek.similarity < eff_thresh) {
                                LOG_INFO("AudioPipe",
                                         "SHORT-IDENTIFY abstain (shadow-thresh): id=%d sim=%.3f < %.3f",
                                         peek.speaker_id, peek.similarity, eff_thresh);
                            } else {
                            // Margin gate — defend against absorb-into-dominant.
                            float margin = peek.similarity - peek.second_best_sim;
                            if (eff_margin > 0.0f && peek.second_best_id >= 0 && margin < eff_margin) {
                                LOG_INFO("AudioPipe",
                                         "SHORT-IDENTIFY abstain (margin%s): id=%d sim=%.3f vs id=%d sim=%.3f margin=%.3f < %.3f",
                                         peek_from_shadow ? "-shadow" : "",
                                         peek.speaker_id, peek.similarity,
                                         peek.second_best_id, peek.second_best_sim,
                                         margin, eff_margin);
                            } else {
                            SpeakerMatch match_si{};
                            match_si.speaker_id     = peek.speaker_id;
                            match_si.similarity     = peek.similarity;
                            match_si.is_new         = false;
                            match_si.name           = peek.name;
                            match_si.exemplar_count = 0;
                            match_si.hits_above     = 0;
                            short_identify_broadcast = true;
                            LOG_INFO("AudioPipe",
                                     "SHORT-IDENTIFY match: id=%d sim=%.3f %s (fbank=%d in [%d,%d) thresh=%.3f margin=%.3f)",
                                     peek.speaker_id, peek.similarity,
                                     peek.name.empty() ? "(unnamed)" : peek.name.c_str(),
                                     fbank_frames, kMinFbankFramesIdent, kMinFbankFrames,
                                     si_thresh, margin);
                            if (on_speaker_) on_speaker_(match_si);

                            // Step 19d note: do NOT update
                            // prev_seg_speaker_id_ from a SHORT-IDENTIFY
                            // hit. SI is a low-confidence label that
                            // applies only to its own segment. Allowing
                            // it to seed the INHERIT-BROADCAST chain
                            // turns one wrong match into a cascade of
                            // wrong labels (observed: 12 SI matches →
                            // ~150 inherited mislabels, dec_macro
                            // 0.95 → 0.27). Each subsequent short
                            // segment must earn its own SI hit or
                            // inherit from a real FULL identify.
                            float si_now = (float)audio_t1_processed_ / 16000.0f;
                            // Step 21 — SHORT-IDENTIFY → prev_full refresh.
                            // Config-driven (speaker_si_refresh_prev_full_threshold,
                            // default 0.60); env DEUSRIDET_SI_REFRESH_PREVFULL_THR
                            // overrides for sweeps. 0.0 = disabled (= Step 20
                            // behaviour). When ON, a strong SI hit refreshes
                            // prev_full so the inherit-recency window (4 s,
                            // Step 20) restarts under the matched identity.
                            // Step 21 sweep result on tests/test.mp3
                            // (600 s, replay 1x, 198 GT segs):
                            //   thr=0.55 → cov=0.591 dec_macro=0.894
                            //   thr=0.60 → cov=0.611 dec_macro=0.903  (chosen)
                            // vs Step 20 baseline cov=0.571 dec_macro=0.921.
                            // The margin gate (≥0.05) already filters
                            // absorb-into-dominant on the SI side.
                            float si_refresh_thr = cfg_.speaker_si_refresh_prev_full_threshold;
                            if (const char* env = std::getenv("DEUSRIDET_SI_REFRESH_PREVFULL_THR")) {
                                if (*env) si_refresh_thr = (float)std::atof(env);
                            }
                            if (si_refresh_thr > 0.0f &&
                                peek.similarity >= si_refresh_thr) {
                                prev_full_speaker_id_   = peek.speaker_id;
                                prev_full_speaker_name_ = peek.name;
                                prev_full_time_         = si_now;
                                LOG_INFO("AudioPipe",
                                         "SI refresh prev_full: id=%d sim=%.3f thr=%.3f",
                                         peek.speaker_id, peek.similarity,
                                         si_refresh_thr);
                            }
                            }
                            }
                        } else if (peek_ok) {
                            LOG_INFO("AudioPipe",
                                     "SHORT-IDENTIFY abstain: best id=%d sim=%.3f thresh=%.3f (fbank=%d)",
                                     peek.speaker_id, peek.similarity, si_thresh, fbank_frames);
                        }
                    }
                }
                // === end Step 19b ============================================

                // Step 4c: short-segment INHERIT-BROADCAST.
                // prev_seg_speaker_id_ is reset to -1 whenever the previous
                // segend took this same short-skip path (no FULL → no
                // seg_ref / stats). To stay continuous across a run of
                // short segments we fall back to prev_full_speaker_id_,
                // which persists from the last successful FULL identify.
                //
                // RECENCY GATE: only inherit from prev_full when it is
                // temporally fresh (≤ 2.0 s since prev FULL midpoint).
                // Without this, every short segment for the rest of the
                // timeline would inherit whichever speaker happened to be
                // last identified by FULL, collapsing all clusters onto
                // that one identity (observed empirically: decided_macro
                // crashes from 0.92 to 0.32).
                int         inh_id = -1;
                std::string inh_name;
                float       inh_sim = 0.0f;
                const char* inh_src = "";
                if (prev_seg_speaker_id_ >= 0) {
                    inh_id   = prev_seg_speaker_id_;
                    inh_name = prev_seg_speaker_name_;
                    inh_sim  = prev_seg_speaker_sim_;
                    inh_src  = "seg";
                } else if (prev_full_speaker_id_ >= 0) {
                    float now_sec = (float)audio_t1_processed_ / 16000.0f;
                    float age     = now_sec - prev_full_time_;
                    // Step 20a: extended prev_full recency 2.0 → 4.0 s.
                    // Diagnostic on /tmp/replay_step19d_si040m05_final
                    // showed 84/137 unpaired VADs are within 4 s of the
                    // nearest FULL event; 47/137 within 2 s. 3.0 s landed
                    // cov +9.6 pts (0.439→0.535) with dec_macro −1.4 pts.
                    // Widening to 4.0 s should pick up the remaining
                    // [3,4)s slice (~30 more saveable inherits at
                    // slightly increased stale-label risk).
                    if (age <= 4.0f) {
                        inh_id   = prev_full_speaker_id_;
                        inh_name = prev_full_speaker_name_;
                        inh_sim  = 0.0f;
                        inh_src  = "full";
                    }
                }
                if (!(speaker_enc_.initialized() &&
                      enable_speaker_.load(std::memory_order_relaxed) &&
                      fbank_frames >= kMinFbankFrames) &&
                    !short_identify_broadcast &&
                    cfg_.speaker_short_inherit_enable &&
                    enable_speaker_.load(std::memory_order_relaxed) &&
                    inh_id >= 0) {
                    // Step 25a: SI-peek veto on inherit-broadcast.
                    // When SI ran but abstained, if peek had a CONFIDENT
                    // DIFFERENT opinion from inh_id, suppress the inherit
                    // broadcast (emit no_segment instead of a likely-wrong
                    // inherited label). See audio_pipeline.h for rationale.
                    //
                    // Step 25b: when peek-rescue is enabled (default on),
                    // instead of dropping the broadcast, SUBSTITUTE peek's
                    // identity — converting wrong inherits into correct
                    // decisions (lifts both macro and dec_macro).
                    bool peek_disagrees =
                        (cfg_.speaker_inherit_peek_veto_enable &&
                         cfg_.speaker_inherit_peek_veto_threshold > 0.0f &&
                         si_peek_id >= 0 &&
                         si_peek_id != inh_id &&
                         si_peek_sim >= cfg_.speaker_inherit_peek_veto_threshold);
                    if (peek_disagrees && cfg_.speaker_inherit_peek_rescue_enable) {
                        // Rescue: replace inherit identity with peek's.
                        LOG_INFO("AudioPipe",
                                 "INHERIT rescued by SI peek: peek_id=%d sim=%.3f >= %.3f replaces inh_id=%d (%s, src=%s)",
                                 si_peek_id, si_peek_sim,
                                 cfg_.speaker_inherit_peek_veto_threshold,
                                 inh_id,
                                 inh_name.empty() ? "(unnamed)" : inh_name.c_str(),
                                 inh_src);
                        inh_id   = si_peek_id;
                        inh_name = si_peek_name;
                        inh_sim  = si_peek_sim;
                        inh_src  = "peek-rescue";
                        peek_disagrees = false;  // fall through to broadcast
                    }
                    if (peek_disagrees) {
                        LOG_INFO("AudioPipe",
                                 "INHERIT vetoed by SI peek: peek_id=%d sim=%.3f >= %.3f differs from inh_id=%d (%s, src=%s)",
                                 si_peek_id, si_peek_sim,
                                 cfg_.speaker_inherit_peek_veto_threshold,
                                 inh_id,
                                 inh_name.empty() ? "(unnamed)" : inh_name.c_str(),
                                 inh_src);
                    } else {
                    // Short-segment inheritance broadcast.
                    //
                    // This segment is too short for CAM++ FULL to produce a
                    // trustworthy embedding. Rather than drop it silently
                    // (→ "no_segment" in the replay scorer), forward the
                    // last successfully-identified speaker as a best-effort
                    // label. prev_seg_speaker_id_ is updated at the START
                    // of every segend, so it holds whichever identity the
                    // most recent long-enough segment produced (or the
                    // VAD-start 0.8 s inheritance when that fired).
                    //
                    // Critically: we do NOT call campp_db_.identify(),
                    // dual_db_.identify(), or register_speaker() here. The
                    // speaker library is untouched. This is pure continuity
                    // propagation at the broadcast layer. Wrong labels on
                    // isolated short utterances by a new speaker will show
                    // up in macro(all) but CANNOT pollute centroids.
                    SpeakerMatch inh{};
                    inh.speaker_id     = inh_id;
                    inh.similarity     = inh_sim;
                    inh.is_new         = false;
                    inh.name           = inh_name;
                    inh.exemplar_count = 0;
                    inh.hits_above     = 0;
                    LOG_INFO("AudioPipe",
                             "FULL-skip INHERIT-BROADCAST: id=%d sim=%.3f %s (fbank=%d < %d, src=%s)",
                             inh.speaker_id, inh.similarity,
                             inh.name.empty() ? "(unnamed)" : inh.name.c_str(),
                             fbank_frames, kMinFbankFrames,
                             inh_src);
                    if (on_speaker_) on_speaker_(inh);
                    } // end Step 25a veto else-branch
                }
}

} // namespace deusridet
