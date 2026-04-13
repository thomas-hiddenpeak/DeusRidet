// speaker_stream.cpp — Independent speaker identification stream with Bayesian tracking.
// VAD metadata used internally; caller pushes ALL audio unconditionally.

#include "speaker_stream.h"
#include "../../communis/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace deusridet {

void SpeakerStream::init(WavLMEcapaEncoder* encoder,
                          SpeakerVectorStore* store,
                          SpeakerTimeline* timeline,
                          const SpeakerStreamConfig& cfg,
                          SpeakerEncoder* cam_encoder) {
    encoder_  = encoder;
    store_    = store;
    timeline_ = timeline;
    cfg_      = cfg;
    cam_encoder_ = cam_encoder;

    // Initialize dedicated FBank for CAM++ if dual-encoder mode.
    if (cam_encoder_ && cam_encoder_->initialized()) {
        // Povey window + PCM normalization matches WeSpeaker/Kaldi defaults.
        if (!cam_fbank_.init(80, 400, 160, 512, 16000,
                             FbankWindowType::POVEY, true)) {
            LOG_WARN("SpkStream", "CAM++ fbank init failed, falling back to single encoder");
            cam_encoder_ = nullptr;
        } else {
            LOG_INFO("SpkStream", "Dual-encoder mode: WL-ECAPA(192) + CAM++(192) → %d-dim fused",
                     encoder_ ? 384 : 192);
        }
    } else {
        cam_encoder_ = nullptr;
    }

    reset_posteriors();
    initialized_ = true;
    LOG_INFO("SpkStream", "Initialized: stride=%.2fs window=%.2fs alpha=%.1f eps=%.3f forget=%.2f confirm=%d%s",
             cfg_.stride_samples / 16000.0f, cfg_.window_samples / 16000.0f,
             cfg_.bayesian_alpha, cfg_.transition_prob,
             cfg_.forgetting_factor, cfg_.change_confirm_steps,
             cam_encoder_ ? " [DUAL-ENCODER]" : "");
}

bool SpeakerStream::push_audio(const int16_t* pcm, int samples,
                                int64_t abs_position, bool is_speech) {
    if (!initialized_ || !encoder_ || !store_ || samples <= 0) return false;

    last_was_full_ = false;
    bool extracted = false;

    // Speech→silence transition: if we had enough speech, do a final
    // authoritative extraction (auto_reg=true → SPK_FULL, new speaker registration).
    if (was_speech_ && !is_speech) {
        int buf_sz = (int)speech_buf_.size();
        if (buf_sz >= cfg_.min_samples) {
            int64_t seg_start = abs_position - samples - buf_sz;
            int64_t seg_end   = abs_position - samples;  // end = start of silence
            identify_step(seg_start, seg_end, /*auto_reg=*/true);
            last_was_full_ = true;
            extracted = true;
        }
        // Buffer is cleared at next silence→speech transition (see below).
        samples_since_last_id_ = 0;
    }

    bool was_prev_speech = was_speech_;
    was_speech_ = is_speech;

    // Only accumulate speech PCM — silence is noise for embedding extraction.
    if (!is_speech) return extracted;

    // Silence→speech transition: clear the buffer to ensure this segment
    // only contains the current speaker's speech.  Without this, the buffer
    // retains audio from the previous speaker's turn, contaminating embeddings
    // when using windows larger than the current utterance.
    if (speech_buf_.empty() || (!was_prev_speech && is_speech)) {
        speech_buf_.clear();
        seg_start_abs_ = abs_position - samples;
    }

    speech_buf_.insert(speech_buf_.end(), pcm, pcm + samples);
    samples_since_last_id_ += samples;

    // Cap buffer at 10s (keep most recent).
    const int max_buf = 160000;
    if ((int)speech_buf_.size() > max_buf) {
        int excess = (int)speech_buf_.size() - max_buf;
        speech_buf_.erase(speech_buf_.begin(), speech_buf_.begin() + excess);
        seg_start_abs_ += excess;
    }

    // Check stride: enough new speech for an identification step?
    if (samples_since_last_id_ < cfg_.stride_samples) return extracted;
    if ((int)speech_buf_.size() < cfg_.min_samples) return extracted;

    int64_t seg_end = abs_position;
    int64_t seg_start = seg_end - (int64_t)speech_buf_.size();

    identify_step(seg_start, seg_end, /*auto_reg=*/false);
    return true;
}

bool SpeakerStream::identify_step(int64_t seg_start_abs, int64_t seg_end_abs, bool auto_reg) {
    samples_since_last_id_ = 0;

    int buf_sz = (int)speech_buf_.size();
    // For end-of-segment (auto_reg=true, SPK_FULL), use the full accumulated
    // speech buffer (up to 5s) for more discriminative embeddings.
    // Longer audio → more stable speaker characteristics.
    // For mid-speech (SPK_EARLY), use the smaller sliding window for responsiveness.
    int max_win = auto_reg ? std::min(buf_sz, 80000) : cfg_.window_samples;
    int win = std::min(buf_sz, max_win);
    int offset = buf_sz - win;

    // Convert to float32.
    std::vector<float> pcm_f32(win);
    for (int i = 0; i < win; i++)
        pcm_f32[i] = speech_buf_[offset + i] / 32768.0f;

    // Extract WL-ECAPA embedding (192-dim from raw PCM).
    auto emb_wl = encoder_->extract(pcm_f32.data(), win);
    if (emb_wl.empty()) return false;

    std::vector<float> emb;

    // Dual-encoder fusion: concatenate WL-ECAPA + CAM++ embeddings.
    if (cam_encoder_) {
        // Compute FBank from int16 PCM for CAM++ (80-dim Povey window).
        cam_fbank_.reset();
        cam_fbank_.push_pcm(speech_buf_.data() + offset, win);
        int fb_frames = cam_fbank_.frames_ready();
        if (fb_frames > 0) {
            std::vector<float> fbank_host(fb_frames * 80);
            cam_fbank_.read_fbank(fbank_host.data(), fb_frames);

            auto emb_cam = cam_encoder_->extract(fbank_host.data(), fb_frames);
            if (!emb_cam.empty() && (int)emb_cam.size() == (int)emb_wl.size()) {
                // Concatenate [WL-ECAPA || CAM++] → 384-dim.
                int dim1 = (int)emb_wl.size();
                int dim2 = (int)emb_cam.size();
                emb.resize(dim1 + dim2);
                memcpy(emb.data(), emb_wl.data(), dim1 * sizeof(float));
                memcpy(emb.data() + dim1, emb_cam.data(), dim2 * sizeof(float));

                // L2 normalize the fused vector.
                float norm_sq = 0.0f;
                for (float v : emb) norm_sq += v * v;
                float inv_norm = 1.0f / sqrtf(norm_sq + 1e-8f);
                for (float& v : emb) v *= inv_norm;
            } else {
                // CAM++ failed — fall back to WL-ECAPA only (zero-padded to match dim).
                emb.resize(emb_wl.size() * 2, 0.0f);
                memcpy(emb.data(), emb_wl.data(), emb_wl.size() * sizeof(float));
                float norm_sq = 0.0f;
                for (float v : emb) norm_sq += v * v;
                float inv_norm = 1.0f / sqrtf(norm_sq + 1e-8f);
                for (float& v : emb) v *= inv_norm;
            }
        } else {
            // No FBank frames — fall back to WL-ECAPA only (zero-padded).
            emb.resize(emb_wl.size() * 2, 0.0f);
            memcpy(emb.data(), emb_wl.data(), emb_wl.size() * sizeof(float));
            float norm_sq = 0.0f;
            for (float v : emb) norm_sq += v * v;
            float inv_norm = 1.0f / sqrtf(norm_sq + 1e-8f);
            for (float& v : emb) v *= inv_norm;
        }
    } else {
        emb = std::move(emb_wl);
    }

    int64_t win_start_abs = seg_end_abs - win;
    int64_t win_end_abs   = seg_end_abs;

    // ─── Warmup phase: collect embeddings for batch clustering ───
    if (!warmup_complete_ && cfg_.warmup_embeddings > 0) {
        warmup_embs_.push_back(emb);
        warmup_starts_.push_back(win_start_abs);
        warmup_ends_.push_back(win_end_abs);

        LOG_INFO("SpkStream", "WARMUP: collected %d/%d embeddings at %.2fs",
                 (int)warmup_embs_.size(), cfg_.warmup_embeddings,
                 win_end_abs / 16000.0f);

        if ((int)warmup_embs_.size() >= cfg_.warmup_embeddings) {
            run_warmup_clustering();
            warmup_complete_ = true;

            // Re-identify all buffered embeddings with the new clusters.
            for (int i = 0; i < (int)warmup_embs_.size(); i++) {
                auto res = store_->search_all(warmup_embs_[i]);
                if (!res.empty()) {
                    int best_id = res[0].speaker_id;
                    float best_sim = res[0].similarity;
                    current_speaker_id_ = best_id;

                    // Emit SPK_FULL events for warmup segments.
                    std::string name;
                    auto all = store_->all_speakers();
                    for (auto& si : all) {
                        if (si.id == best_id) { name = si.name; break; }
                    }

                    SpeakerEvent ev{};
                    ev.audio_start = warmup_starts_[i];
                    ev.audio_end   = warmup_ends_[i];
                    ev.source      = SpkEventSource::SPK_FULL;
                    ev.speaker_id  = best_id;
                    ev.similarity  = best_sim;
                    strncpy(ev.name, name.c_str(), sizeof(ev.name) - 1);
                    if (timeline_) timeline_->push(ev);
                    if (on_spk_event_) on_spk_event_(ev);
                    if (on_speaker_) {
                        SpeakerMatch m;
                        m.speaker_id = best_id;
                        m.similarity = best_sim;
                        m.name = name;
                        m.is_new = false;
                        on_speaker_(m);
                    }
                }
            }
            warmup_embs_.clear();
            warmup_starts_.clear();
            warmup_ends_.clear();

            reset_posteriors();
            LOG_INFO("SpkStream", "WARMUP complete — %d speakers registered, switching to online mode",
                     store_->count());
        }

        return false;  // No speaker change during warmup.
    }

    // Search all known speakers for similarity scores.
    auto results = store_->search_all(emb);
    int n_known = (int)results.size();

    // Recursive Bayesian HMM update (accumulates evidence over time).
    int old_speaker = current_speaker_id_;
    int map_speaker = bayesian_update(results);

    // Determine the best similarity for the MAP speaker.
    float map_sim = 0.0f;
    std::string map_name;
    for (auto& r : results) {
        if (r.speaker_id == map_speaker) {
            map_sim = r.similarity;
            break;
        }
    }

    // Get name from store.
    if (map_speaker >= 0) {
        auto all = store_->all_speakers();
        for (auto& si : all) {
            if (si.id == map_speaker) {
                map_name = si.name;
                break;
            }
        }
    }

    bool speaker_changed = false;

    // ─── Speaker change detection with confirmation ───
    // Instead of immediately switching on a single observation, require
    // change_confirm_steps consecutive steps where the new speaker leads.
    if (map_speaker != old_speaker && map_speaker >= 0 && old_speaker >= 0) {
        if (map_speaker == pending_change_id_) {
            // Same candidate as before — accumulate evidence.
            pending_change_steps_++;
            if (current_confidence_ > pending_change_conf_)
                pending_change_conf_ = current_confidence_;
        } else {
            // Different candidate — reset confirmation.
            pending_change_id_ = map_speaker;
            pending_change_steps_ = 1;
            pending_change_conf_ = current_confidence_;
        }

        // Check if confirmation threshold is met.
        bool confirmed = false;
        if (auto_reg) {
            // End-of-segment: accept with 1 step (no time to accumulate more).
            confirmed = (pending_change_conf_ >= cfg_.change_threshold);
        } else {
            confirmed = (pending_change_steps_ >= cfg_.change_confirm_steps &&
                         pending_change_conf_ >= cfg_.change_threshold);
        }

        if (confirmed) {
            speaker_changed = true;
            current_speaker_id_ = map_speaker;
            steps_since_change_ = 0;
            pending_change_id_ = -1;
            pending_change_steps_ = 0;
            pending_change_conf_ = 0.0f;

            LOG_INFO("SpkStream", "CHANGE: spk %d→%d (posterior=%.3f, sim=%.3f, %s, confirm=%d) at %.2fs",
                     old_speaker, map_speaker, current_confidence_, map_sim,
                     map_name.empty() ? "(unnamed)" : map_name.c_str(),
                     cfg_.change_confirm_steps,
                     win_end_abs / 16000.0f);

            if (on_changed_) {
                on_changed_(old_speaker, map_speaker, map_sim, win_start_abs);
            }
        } else {
            LOG_INFO("SpkStream", "PENDING: spk %d→%d? (posterior=%.3f, steps=%d/%d) at %.2fs",
                     old_speaker, map_speaker, current_confidence_,
                     pending_change_steps_, cfg_.change_confirm_steps,
                     win_end_abs / 16000.0f);
        }
    } else {
        // MAP agrees with current speaker — reset any pending change.
        if (pending_change_id_ >= 0) {
            LOG_INFO("SpkStream", "CANCEL: pending change to spk %d cancelled (current spk %d reconfirmed)",
                     pending_change_id_, current_speaker_id_);
        }
        pending_change_id_ = -1;
        pending_change_steps_ = 0;
        pending_change_conf_ = 0.0f;

        if (map_speaker >= 0 && old_speaker < 0) {
            // First identification — no "change" event, just initial assignment.
            current_speaker_id_ = map_speaker;
            steps_since_change_ = 0;
        }
    }

    steps_since_change_++;

    // ─── Emit events and update stats ───
    //
    // auto_reg=true  (end-of-segment): Always call identify() for authoritative
    //   speaker matching. identify() handles:
    //   - Threshold-based new speaker registration
    //   - Adding exemplars to known speakers (diversity-gated)
    //   - Returning full SpeakerMatch with exemplar_count, hits_above, etc.
    //   The Bayesian estimate above is used ONLY for mid-speech change detection.
    //
    // auto_reg=false (mid-speech): Use Bayesian MAP estimate for SPK_EARLY events.
    //   identify() is NOT called — no exemplar updates during speech.

    if (auto_reg && !emb.empty()) {
        // End-of-segment: authoritative identification via identify().
        float thresh = cfg_.confirm_threshold;
        SpeakerMatch m = store_->identify(emb, thresh, true);
        if (m.speaker_id >= 0) {
            // identify() may register a new speaker or match an existing one.
            if (m.speaker_id != current_speaker_id_ && current_speaker_id_ >= 0) {
                int old_id = current_speaker_id_;
                current_speaker_id_ = m.speaker_id;
                steps_since_change_ = 0;
                LOG_INFO("SpkStream", "IDENTIFY: spk %d→%d (sim=%.3f, %s%s) at %.2fs",
                         old_id, m.speaker_id, m.similarity,
                         m.is_new ? "NEW " : "",
                         m.name.empty() ? "(unnamed)" : m.name.c_str(),
                         win_end_abs / 16000.0f);

                if (m.is_new || old_id != m.speaker_id) {
                    if (on_changed_) {
                        on_changed_(old_id, m.speaker_id, m.similarity, win_start_abs);
                    }
                }
            } else if (current_speaker_id_ < 0) {
                // First identification ever.
                current_speaker_id_ = m.speaker_id;
                steps_since_change_ = 0;
            }

            SpeakerEvent ev{};
            ev.audio_start = seg_start_abs;
            ev.audio_end   = seg_end_abs;
            ev.source      = SpkEventSource::SPK_FULL;
            ev.speaker_id  = m.speaker_id;
            ev.similarity  = m.similarity;
            strncpy(ev.name, m.name.c_str(), sizeof(ev.name) - 1);
            if (timeline_) timeline_->push(ev);
            if (on_spk_event_) on_spk_event_(ev);
            if (on_speaker_) on_speaker_(m);
        }

        // ─── Partial posterior reset at turn boundaries ───
        // After end-of-segment processing, partially reset Bayesian posteriors
        // toward uniform. This prevents the next speech segment from starting
        // with excessive bias toward the previous speaker, allowing faster
        // speaker switching at turn boundaries while preserving some context.
        // Retain 40% of current posteriors — enough to bias toward the same
        // speaker if they continue, weak enough to allow switching on 1-2 obs.
        partial_reset_posteriors(0.40f);
    } else {
        // Mid-speech: use Bayesian MAP estimate for SPK_EARLY events.
        int emit_speaker = current_speaker_id_;
        float emit_sim = 0.0f;
        std::string emit_name;
        if (emit_speaker >= 0) {
            for (auto& r : results) {
                if (r.speaker_id == emit_speaker) { emit_sim = r.similarity; break; }
            }
            auto all = store_->all_speakers();
            for (auto& si : all) {
                if (si.id == emit_speaker) { emit_name = si.name; break; }
            }

            SpeakerEvent ev{};
            ev.audio_start = win_start_abs;
            ev.audio_end   = win_end_abs;
            ev.source      = SpkEventSource::SPK_EARLY;
            ev.speaker_id  = emit_speaker;
            ev.similarity  = emit_sim;
            strncpy(ev.name, emit_name.c_str(), sizeof(ev.name) - 1);
            if (timeline_) timeline_->push(ev);
            if (on_spk_event_) on_spk_event_(ev);

            // Report match to on_speaker_ callback for stats.
            if (on_speaker_) {
                SpeakerMatch match;
                match.speaker_id = emit_speaker;
                match.similarity = emit_sim;
                match.name = emit_name;
                match.is_new = false;
                on_speaker_(match);
            }
        }
    }

    LOG_INFO("SpkStream", "ID: spk=%d sim=%.3f post=%.3f n_spk=%d win=%.2fs%s",
             map_speaker >= 0 ? map_speaker : -1, map_sim,
             current_confidence_, n_known,
             win / 16000.0f,
             speaker_changed ? " [CHANGED]" : "");

    return speaker_changed;
}

int SpeakerStream::bayesian_update(const std::vector<SpeakerVectorStore::SearchResult>& results) {
    int n_spk = (int)results.size();
    if (n_spk == 0) {
        current_confidence_ = 0.0f;
        return -1;
    }

    float alpha = cfg_.bayesian_alpha;
    float eps   = cfg_.transition_prob;
    int   N     = n_spk + 1;  // known speakers + unknown class

    // ─── Step 1: Observation likelihood ───
    // P(obs | spk=k) ∝ exp(α · sim_k)
    float log_likelihoods[kMaxTracked] = {};
    float max_sim = -1.0f;

    for (auto& r : results) {
        int id = r.speaker_id;
        if (id < 0 || id >= kMaxTracked) continue;
        log_likelihoods[id] = alpha * r.similarity;
        if (r.similarity > max_sim) max_sim = r.similarity;
    }

    // Adaptive unknown class likelihood:
    // sim_unknown = max(max_sim - γ, floor)
    float unknown_sim = std::max(max_sim - cfg_.unknown_margin, cfg_.unknown_floor);
    float ll_unknown = alpha * unknown_sim;

    // ─── Step 2: Predict (HMM forward: transition × previous posterior) ───
    // If we have accumulated posteriors from previous steps, use them as prior.
    // Otherwise (first observation), use uniform prior.
    float log_prior[kMaxTracked] = {};
    float log_prior_unknown = 0.0f;

    if (has_prior_) {
        // Apply forgetting factor first to prevent extreme posteriors.
        apply_forgetting();

        // Full HMM predict step:
        // P(spk=k at t | obs_1..t-1) = Σ_j T(j→k) × P(spk=j at t-1 | obs_1..t-1)
        //
        // For efficiency with N≤64, compute transition matrix multiplication:
        //   T(j→k) = (1-ε) if j==k, else ε/(N-1)
        float log_stay   = logf(1.0f - eps);
        float log_switch = logf(eps / (float)(N - 1));

        for (auto& r : results) {
            int k = r.speaker_id;
            if (k < 0 || k >= kMaxTracked) continue;

            // log P(k at t) = log_sum_exp over j of {log T(j→k) + log P(j at t-1)}
            // Dominant terms: stay (j==k) and switch (any j≠k)
            float log_stay_term = log_stay + log_posterior_[k];

            // sum of switch contributions from all other speakers
            float max_switch = -1e30f;
            float switch_terms[kMaxTracked + 1];
            int n_switch = 0;
            for (auto& r2 : results) {
                int j = r2.speaker_id;
                if (j < 0 || j >= kMaxTracked || j == k) continue;
                float val = log_switch + log_posterior_[j];
                switch_terms[n_switch++] = val;
                if (val > max_switch) max_switch = val;
            }
            // Unknown → k transition
            {
                float val = log_switch + log_unknown_;
                switch_terms[n_switch++] = val;
                if (val > max_switch) max_switch = val;
            }

            // log-sum-exp of all switch terms
            if (n_switch > 0 && max_switch > -1e20f) {
                float sum_exp_sw = 0.0f;
                for (int i = 0; i < n_switch; i++)
                    sum_exp_sw += expf(switch_terms[i] - max_switch);
                float log_switch_total = max_switch + logf(sum_exp_sw);

                // log-sum-exp of stay + switch_total
                float mx = std::max(log_stay_term, log_switch_total);
                log_prior[k] = mx + logf(expf(log_stay_term - mx) + expf(log_switch_total - mx));
            } else {
                log_prior[k] = log_stay_term;
            }
        }

        // Unknown class prior
        {
            float log_stay_term = log_stay + log_unknown_;
            float max_switch_u = -1e30f;
            float switch_terms_u[kMaxTracked];
            int n_switch_u = 0;
            for (auto& r2 : results) {
                int j = r2.speaker_id;
                if (j < 0 || j >= kMaxTracked) continue;
                float val = log_switch + log_posterior_[j];
                switch_terms_u[n_switch_u++] = val;
                if (val > max_switch_u) max_switch_u = val;
            }
            if (n_switch_u > 0 && max_switch_u > -1e20f) {
                float sum_exp_sw = 0.0f;
                for (int i = 0; i < n_switch_u; i++)
                    sum_exp_sw += expf(switch_terms_u[i] - max_switch_u);
                float log_switch_total = max_switch_u + logf(sum_exp_sw);
                float mx = std::max(log_stay_term, log_switch_total);
                log_prior_unknown = mx + logf(expf(log_stay_term - mx) + expf(log_switch_total - mx));
            } else {
                log_prior_unknown = log_stay_term;
            }
        }
    } else {
        // First observation: uniform prior.
        float log_uniform = -logf((float)N);
        for (auto& r : results) {
            int id = r.speaker_id;
            if (id < 0 || id >= kMaxTracked) continue;
            log_prior[id] = log_uniform;
        }
        log_prior_unknown = log_uniform;
    }

    // ─── Step 3: Update (likelihood × prior) ───
    float log_posts[kMaxTracked] = {};
    float log_post_unknown = ll_unknown + log_prior_unknown;
    float max_lp = log_post_unknown;

    for (auto& r : results) {
        int id = r.speaker_id;
        if (id < 0 || id >= kMaxTracked) continue;
        log_posts[id] = log_likelihoods[id] + log_prior[id];
        if (log_posts[id] > max_lp) max_lp = log_posts[id];
    }

    // ─── Step 4: Normalize (log-sum-exp) ───
    float sum_exp = 0.0f;
    for (auto& r : results) {
        int id = r.speaker_id;
        if (id < 0 || id >= kMaxTracked) continue;
        sum_exp += expf(log_posts[id] - max_lp);
    }
    sum_exp += expf(log_post_unknown - max_lp);
    float log_Z = max_lp + logf(sum_exp);

    // ─── Step 5: Store normalized posteriors for next step ───
    for (auto& r : results) {
        int id = r.speaker_id;
        if (id < 0 || id >= kMaxTracked) continue;
        log_posterior_[id] = log_posts[id] - log_Z;
    }
    log_unknown_ = log_post_unknown - log_Z;
    has_prior_ = true;

    // ─── Step 6: Find MAP speaker ───
    int map_id = -1;
    float map_posterior = 0.0f;

    for (auto& r : results) {
        int id = r.speaker_id;
        if (id < 0 || id >= kMaxTracked) continue;
        float posterior = expf(log_posterior_[id]);
        if (posterior > map_posterior) {
            map_posterior = posterior;
            map_id = id;
        }
    }

    float unknown_posterior = expf(log_unknown_);
    if (unknown_posterior > map_posterior) {
        map_id = -1;
        map_posterior = unknown_posterior;
    }

    current_confidence_ = map_posterior;
    return map_id;
}

void SpeakerStream::apply_forgetting() {
    // Shrink posteriors toward uniform to prevent evidence lock-in.
    // log_post[k] = β * log_post[k] + (1-β) * log(1/N)
    // where N = n_active_speakers + 1 (unknown)
    float beta = cfg_.forgetting_factor;
    int n_active = store_ ? store_->count() : 0;
    if (n_active <= 0) return;

    float log_uniform = -logf((float)(n_active + 1));

    for (int i = 0; i < kMaxTracked; i++) {
        if (log_posterior_[i] < -50.0f) continue;  // skip dead slots
        log_posterior_[i] = beta * log_posterior_[i] + (1.0f - beta) * log_uniform;
    }
    log_unknown_ = beta * log_unknown_ + (1.0f - beta) * log_uniform;
}

void SpeakerStream::reset_posteriors() {
    for (int i = 0; i < kMaxTracked; i++) log_posterior_[i] = -50.0f;
    log_unknown_ = 0.0f;
    has_prior_ = false;
    current_speaker_id_ = -1;
    current_confidence_ = 0.0f;
    steps_since_change_ = 0;
    pending_change_id_ = -1;
    pending_change_steps_ = 0;
    pending_change_conf_ = 0.0f;
}

void SpeakerStream::partial_reset_posteriors(float retain) {
    if (!has_prior_) return;

    int n_active = store_ ? store_->count() : 0;
    if (n_active <= 0) return;

    float log_uniform = -logf((float)(n_active + 1));

    for (int i = 0; i < kMaxTracked; i++) {
        if (log_posterior_[i] < -50.0f) continue;
        log_posterior_[i] = retain * log_posterior_[i] + (1.0f - retain) * log_uniform;
    }
    log_unknown_ = retain * log_unknown_ + (1.0f - retain) * log_uniform;

    // Also reset pending change tracking — turn boundary should start clean.
    pending_change_id_ = -1;
    pending_change_steps_ = 0;
    pending_change_conf_ = 0.0f;
}

SpeakerStream::BayesianState SpeakerStream::bayesian_state() const {
    BayesianState st;
    st.map_speaker_id = current_speaker_id_;
    st.map_posterior = current_confidence_;
    st.unknown_posterior = expf(log_unknown_);
    st.n_speakers = store_ ? store_->count() : 0;
    st.steps_since_change = steps_since_change_;
    st.pending_change_id = pending_change_id_;
    st.pending_change_steps = pending_change_steps_;
    return st;
}

// ============================================================================
// Warmup batch clustering — agglomerative clustering to find initial speakers.
// ============================================================================

void SpeakerStream::run_warmup_clustering() {
    int N = (int)warmup_embs_.size();
    if (N == 0 || !store_) return;
    int dim = (int)warmup_embs_[0].size();

    LOG_INFO("SpkStream", "WARMUP: clustering %d embeddings (merge_thresh=%.2f, min_cluster=%d)",
             N, cfg_.warmup_merge_thresh, cfg_.warmup_min_cluster);

    // ─── Agglomerative clustering ───
    // Each element starts as its own cluster. Iterate: find closest pair,
    // merge if similarity > threshold. Repeat until no merge possible.

    struct Cluster {
        std::vector<float> centroid;
        int count;
        std::vector<int> members;  // indices into warmup_embs_
    };

    std::vector<Cluster> clusters(N);
    for (int i = 0; i < N; i++) {
        clusters[i].centroid = warmup_embs_[i];
        clusters[i].count = 1;
        clusters[i].members.push_back(i);
    }

    auto cosine_sim = [&](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float dot = 0, na = 0, nb = 0;
        for (int d = 0; d < dim; d++) {
            dot += a[d] * b[d];
            na  += a[d] * a[d];
            nb  += b[d] * b[d];
        }
        return dot / (sqrtf(na * nb) + 1e-8f);
    };

    while (clusters.size() > 1) {
        // Find most similar pair.
        float best_sim = -1.0f;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < (int)clusters.size(); i++) {
            for (int j = i + 1; j < (int)clusters.size(); j++) {
                float s = cosine_sim(clusters[i].centroid, clusters[j].centroid);
                if (s > best_sim) {
                    best_sim = s;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_sim < cfg_.warmup_merge_thresh) break;

        // Merge j into i (weighted centroid average).
        int total = clusters[best_i].count + clusters[best_j].count;
        float w_i = (float)clusters[best_i].count / total;
        float w_j = (float)clusters[best_j].count / total;
        for (int d = 0; d < dim; d++) {
            clusters[best_i].centroid[d] = w_i * clusters[best_i].centroid[d] +
                                           w_j * clusters[best_j].centroid[d];
        }
        // L2 normalize the merged centroid.
        float norm_sq = 0.0f;
        for (float v : clusters[best_i].centroid) norm_sq += v * v;
        float inv_norm = 1.0f / sqrtf(norm_sq + 1e-8f);
        for (float& v : clusters[best_i].centroid) v *= inv_norm;

        clusters[best_i].count = total;
        clusters[best_i].members.insert(clusters[best_i].members.end(),
                                         clusters[best_j].members.begin(),
                                         clusters[best_j].members.end());
        clusters.erase(clusters.begin() + best_j);
    }

    // ─── Register clusters as speakers ───
    int registered = 0;
    for (auto& c : clusters) {
        if (c.count < cfg_.warmup_min_cluster) {
            LOG_INFO("SpkStream", "WARMUP: skip cluster (count=%d < min=%d)",
                     c.count, cfg_.warmup_min_cluster);
            continue;
        }

        // Register with the centroid embedding.
        int id = store_->register_speaker("", c.centroid);
        if (id >= 0) {
            LOG_INFO("SpkStream", "WARMUP: registered spk%d (count=%d members)",
                     id, c.count);
            registered++;
        }
    }

    LOG_INFO("SpkStream", "WARMUP: %d clusters → %d speakers registered (from %d embeddings)",
             (int)clusters.size(), registered, N);
}

} // namespace deusridet
