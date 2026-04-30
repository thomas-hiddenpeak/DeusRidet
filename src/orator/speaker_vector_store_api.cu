/**
 * @file src/orator/speaker_vector_store_api.cu
 * @philosophical_role
 *   Public API for the speaker store — identify, register, delete, rename,
 *   list. The "who-is-speaking" decisions live here, above the GPU kernels.
 *   Split from speaker_vector_store.cu under R1 800-line hard cap.
 * @serves
 *   Orator identification path; Auditus tracker via facade.
 */
// speaker_vector_store_api.cu — peer TU of speaker_vector_store.cu.
//
// Contains the Core API (SpeakerVectorStore methods below gpu_remove_rows),
// starting at identify and running through clear.

#include "speaker_vector_store.h"
#include "../communis/log.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>

namespace deusridet {

namespace {

int pending_similarity_bucket(float similarity) {
    return (int)std::lround(similarity * 1000.0f);
}

bool pending_slot_better(float similarity, int slot, int miss_seq,
                         float best_similarity, int best_slot,
                         int best_miss_seq) {
    if (best_slot < 0) return true;
    int bucket = pending_similarity_bucket(similarity);
    int best_bucket = pending_similarity_bucket(best_similarity);
    if (bucket != best_bucket) return bucket > best_bucket;
    if (miss_seq != best_miss_seq) return miss_seq < best_miss_seq;
    return slot < best_slot;
}

} // namespace

// Forward declarations of kernels defined in speaker_vector_store.cu.
__global__ void ema_normalize_kernel(float4* __restrict__ embedding_row,
                                     const float4* __restrict__ query,
                                     int d_f4, float alpha);
__global__ void l2_normalize_kernel(float4* __restrict__ vec, int d_f4);

// ============================================================================

SpeakerMatch SpeakerVectorStore::identify(const std::vector<float>& embedding,
                                           float match_threshold,
                                           bool auto_register,
                                           float register_threshold) {
    // If no explicit register threshold, use match threshold.
    float reg_thresh = (register_threshold > 0.0f) ? register_threshold : match_threshold;

    if ((int)embedding.size() != dim_) {
        LOG_INFO(label_.c_str(), "identify: dim mismatch (%d vs %d)",
                 (int)embedding.size(), dim_);
        return SpeakerMatch{};
    }

    std::lock_guard<std::mutex> lk(mu_);

    // Upload query to GPU.
    upload_query(embedding.data());

    SpeakerMatch best;
    best.speaker_id = -1;
    best.similarity = 0.0f;
    float second_best_sim = 0.0f;
    int   second_best_id  = -1;

    if (!speakers_.empty()) {
        GpuSearchResult sr = gpu_search(n_total_, (int)speakers_.size());

        if (sr.spk_idx >= 0) {
            best.speaker_id = speakers_[sr.spk_idx].external_id;
            best.similarity = sr.similarity;
            best.name       = speakers_[sr.spk_idx].name;

            // Find second best (from d_spk_sims_ — copy small array).
            int n_spk = (int)speakers_.size();
            if (n_spk > 1) {
                std::vector<float> spk_sims(n_spk);
                cudaMemcpyAsync(spk_sims.data(), d_spk_sims_,
                                n_spk * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_);
                cudaStreamSynchronize(stream_);
                for (int i = 0; i < n_spk; ++i) {
                    if (i != sr.spk_idx && spk_sims[i] > second_best_sim) {
                        second_best_sim = spk_sims[i];
                        second_best_id  = speakers_[i].external_id;
                    }
                }
            }

            LOG_INFO(label_.c_str(),
                     "Match: best=#%d(%.3f) 2nd=#%d(%.3f) m_thresh=%.2f r_thresh=%.2f db=%d ex=%d",
                     best.speaker_id, best.similarity,
                     second_best_id, second_best_sim,
                     match_threshold, reg_thresh, n_spk, n_total_);

            // Store second-best info in result for margin-based decisions.
            best.second_best_sim = second_best_sim;
            best.second_best_id  = second_best_id;
        }

        if (best.similarity >= match_threshold && sr.spk_idx >= 0) {
            // === MATCHED ===
            // Margin guard is NOT applied here — if best exceeds threshold,
            // it's the best match regardless of how close second-best is.
            // Margin is only used during registration (below) to prevent
            // creating a new speaker that's too close to an existing one.

            // DO NOT clear pending slots on match. Pending slots hold evidence
            // for unknown speakers that haven't been confirmed yet. Clearing
            // them on every match destroys that evidence, making it nearly
            // impossible to register new speakers in active conversations
            // where matches are frequent.

            // Count how many exemplars exceeded threshold for this speaker.
            best.hits_above     = count_hits_above(sr.spk_idx, match_threshold);
            best.exemplar_count = speakers_[sr.spk_idx].exemplar_count;

            // Frozen anchor strategy: never EMA-update existing exemplars.
            // Instead, consider adding a new exemplar if it brings diversity.
            // Gate: only admit exemplars from high-confidence matches to prevent
            // borderline matches from contaminating the speaker profile.
            auto& spk = speakers_[sr.spk_idx];
            float div = min_diversity(sr.spk_idx);

            // Dynamic admission margin: stricter when exemplar set is small
            // (early exemplars define the speaker's cluster center and must
            // be high-quality to prevent cross-speaker contamination).
            // v13b: restored to 0.10 default (v13's 0.15 froze centroids).
            float admit_margin = kExemplarAdmitMargin;  // default 0.10
            if (spk.exemplar_count < 5)
                admit_margin = 0.20f;  // very strict early on
            else if (spk.exemplar_count < 10)
                admit_margin = 0.15f;  // moderate
            float admit_thresh = match_threshold + admit_margin;

            // Hit-ratio gate: when we have enough exemplars, require that
            // a meaningful fraction actually matched (not just the closest one).
            // Prevents cross-speaker contamination where a different speaker's
            // embedding happens to match 1-2 outlier exemplars but misses most.
            bool hit_ratio_ok = true;
            if (spk.exemplar_count >= 5) {
                float hit_ratio = (float)best.hits_above / spk.exemplar_count;
                if (hit_ratio < 0.3f) {
                    hit_ratio_ok = false;
                    LOG_INFO(label_.c_str(),
                             "Exemplar blocked by hit-ratio for #%d: "
                             "hits=%d/%d (%.1f%%) < 30%%",
                             spk.external_id, best.hits_above,
                             spk.exemplar_count, hit_ratio * 100.0f);
                }
            }

            // v13: Margin gate — when best and second-best are close,
            // this embedding is ambiguous and could contaminate the centroid.
            // Only add exemplars when the speaker identity is clear.
            // v13b: relaxed from 0.10 to 0.05 — 0.10 blocked 454 exemplars
            // and froze centroids, dropping accuracy to 39%.
            bool margin_ok = true;
            if (second_best_id >= 0 && (int)speakers_.size() >= 3) {
                float margin = best.similarity - second_best_sim;
                if (margin < 0.05f) {
                    margin_ok = false;
                    LOG_INFO(label_.c_str(),
                             "Exemplar blocked by margin for #%d: "
                             "best=%.3f 2nd=#%d(%.3f) margin=%.3f < 0.05",
                             spk.external_id, best.similarity,
                             second_best_id, second_best_sim, margin);
                }
            }

            if (best.similarity >= admit_thresh && div >= kDiversityThresh && hit_ratio_ok && margin_ok) {
                // This embedding is sufficiently different from all existing exemplars.
                if (spk.exemplar_count < max_exemplars_) {
                    // Room available — add directly.
                    gpu_add_exemplar(sr.spk_idx);
                    LOG_INFO(label_.c_str(),
                             "Added exemplar for #%d (div=%.3f, now %d exemplars)",
                             spk.external_id, div, spk.exemplar_count);
                } else {
                    // At capacity — replace most redundant exemplar.
                    int redundant = most_redundant_exemplar(sr.spk_idx);
                    gpu_replace_exemplar(redundant);
                    LOG_INFO(label_.c_str(),
                             "Replaced redundant exemplar row %d for #%d (div=%.3f)",
                             redundant, spk.external_id, div);
                }
            }
            // If div < kDiversityThresh, the embedding is too similar to an
            // existing anchor — skip it silently (no EMA drift).

            spk.match_count++;
            best.is_new = false;
            return best;
        }
    }

    // === NO MATCH ===
    if (!auto_register) {
        best.speaker_id = -1;
        return best;
    }

    // Multi-pending pool: find if current query matches any existing pending slot.
    int matched_slot = -1;
    float best_pending_sim = -1.0f;
    int best_pending_seq = INT_MAX;
    for (int s = 0; s < kMaxPending; s++) {
        if (!pending_slots_[s].active) continue;
        float sim = gpu_pending_dot(s);
        LOG_INFO(label_.c_str(),
                 "Pending[%d] vs query: sim=%.3f bucket=%d (reg_thresh=%.2f, age=%d)",
                 s, sim, pending_similarity_bucket(sim), reg_thresh,
                 pending_miss_seq_ - pending_slots_[s].miss_seq);
        if (sim >= reg_thresh &&
            pending_slot_better(sim, s, pending_slots_[s].miss_seq,
                                best_pending_sim, matched_slot,
                                best_pending_seq)) {
            best_pending_sim = sim;
            matched_slot = s;
            best_pending_seq = pending_slots_[s].miss_seq;
        }
    }

    if (matched_slot >= 0) {
        // Confirmed: pending slot matches current query — same unknown speaker twice.
        float pending_sim = best_pending_sim;

        // Margin guard at REGISTRATION time: check if this pending embedding is
        // too close to two existing speakers (would create a confusing duplicate).
        if (!speakers_.empty() && second_best_id >= 0) {
            float margin = best.similarity - second_best_sim;
            if (margin >= 0 && margin < min_margin_ && best.similarity > reg_thresh * 0.8f) {
                LOG_INFO(label_.c_str(),
                         "Registration blocked by margin guard: "
                         "best_db=#%d(%.3f) 2nd_db=#%d(%.3f) margin=%.3f < %.3f, pending_sim=%.3f slot=%d",
                         best.speaker_id, best.similarity,
                         second_best_id, second_best_sim,
                         margin, min_margin_, pending_sim, matched_slot);
                // Don't register, but keep the pending slot alive — the speaker
                // might accumulate a better embedding next time.
                best.speaker_id = -1;
                best.similarity = 0.0f;
                best.is_new = false;
                return best;
            }
        }

        // Average pending + current on GPU, then register.
        float* slot_ptr = d_pending_pool_ + matched_slot * dim_;
        {
            int d_f4 = dim_ / 4;
            float4* prow = reinterpret_cast<float4*>(slot_ptr);
            const float4* q = reinterpret_cast<const float4*>(d_query_);
            ema_normalize_kernel<<<1, 32, 0, stream_>>>(prow, q, d_f4, 0.5f);
        }

        // Register new speaker.
        int new_idx = (int)speakers_.size();
        SpeakerMeta meta;
        meta.external_id    = next_id_++;
        meta.exemplar_count = 1;
        meta.match_count    = 2;
        speakers_.push_back(std::move(meta));
        id_to_idx_[speakers_.back().external_id] = new_idx;

        ensure_capacity(n_total_ + 1);
        cudaMemcpyAsync(d_embeddings_ + n_total_ * dim_,
                        slot_ptr, dim_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        n_total_++;

        offsets_.push_back(n_total_);
        upload_offsets();

        // Clear confirmed slot.
        pending_slots_[matched_slot].active = false;
        // Update legacy flag.
        has_pending_ = false;
        for (int s = 0; s < kMaxPending; s++)
            if (pending_slots_[s].active) { has_pending_ = true; break; }

        LOG_INFO(label_.c_str(),
                 "Confirmed new speaker id=%d (pending_sim=%.3f, slot=%d, pool=[%d,%d,%d,%d,%d])",
                 speakers_.back().external_id, pending_sim, matched_slot,
                 (int)pending_slots_[0].active,
                 (int)pending_slots_[1].active,
                 (int)pending_slots_[2].active,
                 (int)pending_slots_[3].active,
                 (int)pending_slots_[4].active);

        best.speaker_id = speakers_.back().external_id;
        best.similarity = 1.0f;
        best.is_new     = true;
        best.name.clear();
        return best;
    }

    // No pending slot matched — store current query in a free or LRU slot.
    pending_miss_seq_++;
    int target_slot = -1;

    // Prefer an empty slot.
    for (int s = 0; s < kMaxPending; s++) {
        if (!pending_slots_[s].active) { target_slot = s; break; }
    }

    // No empty slot — evict the oldest (lowest miss_seq).
    if (target_slot < 0) {
        int oldest_seq = INT_MAX;
        for (int s = 0; s < kMaxPending; s++) {
            if (pending_slots_[s].miss_seq < oldest_seq ||
                (pending_slots_[s].miss_seq == oldest_seq &&
                 (target_slot < 0 || s < target_slot))) {
                oldest_seq = pending_slots_[s].miss_seq;
                target_slot = s;
            }
        }
        LOG_INFO(label_.c_str(),
                 "Pending pool full — evicting slot %d (age=%d)",
                 target_slot, pending_miss_seq_ - oldest_seq);
    }

    // Store query in target slot.
    float* slot_ptr = d_pending_pool_ + target_slot * dim_;
    cudaMemcpyAsync(slot_ptr, d_query_, dim_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    pending_slots_[target_slot].active = true;
    pending_slots_[target_slot].miss_seq = pending_miss_seq_;
    has_pending_ = true;

    LOG_INFO(label_.c_str(),
             "Pending new speaker in slot %d (pool=[%d,%d,%d,%d,%d], miss_seq=%d)",
             target_slot,
             (int)pending_slots_[0].active,
             (int)pending_slots_[1].active,
             (int)pending_slots_[2].active,
             (int)pending_slots_[3].active,
             (int)pending_slots_[4].active,
             pending_miss_seq_);

    best.speaker_id = -1;
    best.similarity = 0.0f;
    best.is_new = false;
    return best;
}

bool SpeakerVectorStore::add_exemplar(int id, const std::vector<float>& embedding) {
    if ((int)embedding.size() != dim_) return false;
    std::lock_guard<std::mutex> lk(mu_);
    int idx = id_to_idx(id);
    if (idx < 0) return false;
    upload_query(embedding.data());
    gpu_add_exemplar(idx);
    cudaStreamSynchronize(stream_);
    return true;
}

// Step 17a — read-only score against all clusters. Mirrors the search
// half of identify() but performs no exemplar admission, no pending
// churn, no EMA. Returns best.speaker_id == -1 only when the store is
// empty; otherwise reports best/second even below match thresholds.
SpeakerMatch SpeakerVectorStore::peek_best(const std::vector<float>& embedding) {
    SpeakerMatch best;
    best.speaker_id = -1;
    best.similarity = 0.0f;
    best.is_new     = false;

    if ((int)embedding.size() != dim_) return best;

    std::lock_guard<std::mutex> lk(mu_);
    if (speakers_.empty() || n_total_ == 0) return best;

    upload_query(embedding.data());
    GpuSearchResult sr = gpu_search(n_total_, (int)speakers_.size());
    if (sr.spk_idx < 0) return best;

    best.speaker_id = speakers_[sr.spk_idx].external_id;
    best.similarity = sr.similarity;
    best.name       = speakers_[sr.spk_idx].name;

    int n_spk = (int)speakers_.size();
    if (n_spk > 1) {
        std::vector<float> spk_sims(n_spk);
        cudaMemcpyAsync(spk_sims.data(), d_spk_sims_,
                        n_spk * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        float second_sim = 0.0f;
        int   second_id  = -1;
        for (int i = 0; i < n_spk; ++i) {
            if (i != sr.spk_idx && spk_sims[i] > second_sim) {
                second_sim = spk_sims[i];
                second_id  = speakers_[i].external_id;
            }
        }
        best.second_best_sim = second_sim;
        best.second_best_id  = second_id;
    }
    return best;
}

int SpeakerVectorStore::register_speaker(const std::string& name,
                                          const std::vector<float>& embedding) {
    if ((int)embedding.size() != dim_) return -1;

    std::lock_guard<std::mutex> lk(mu_);

    int new_idx = (int)speakers_.size();
    SpeakerMeta meta;
    meta.external_id    = next_id_++;
    meta.name           = name;
    meta.exemplar_count = 1;
    meta.match_count    = 1;
    speakers_.push_back(std::move(meta));
    id_to_idx_[speakers_.back().external_id] = new_idx;

    // Upload embedding and L2-normalize on GPU.
    ensure_capacity(n_total_ + 1);
    upload_query(embedding.data());
    cudaMemcpyAsync(d_embeddings_ + n_total_ * dim_,
                    d_query_, dim_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    int d_f4 = dim_ / 4;
    float4* row = reinterpret_cast<float4*>(d_embeddings_ + n_total_ * dim_);
    l2_normalize_kernel<<<1, 32, 0, stream_>>>(row, d_f4);

    n_total_++;
    offsets_.push_back(n_total_);
    upload_offsets();

    return speakers_.back().external_id;
}

void SpeakerVectorStore::set_name(int id, const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    int idx = id_to_idx(id);
    if (idx >= 0) speakers_[idx].name = name;
}

std::vector<SpeakerInfo> SpeakerVectorStore::all_speakers() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<SpeakerInfo> out;
    out.reserve(speakers_.size());
    for (int i = 0; i < (int)speakers_.size(); ++i) {
        auto& m = speakers_[i];
        SpeakerInfo si;
        si.id             = m.external_id;
        si.name           = m.name;
        si.match_count    = m.match_count;
        si.exemplar_count = m.exemplar_count;
        si.min_diversity  = (m.exemplar_count >= 2)
            ? const_cast<SpeakerVectorStore*>(this)->min_diversity(i) : -1.0f;
        // embedding left empty — data is on GPU.
        out.push_back(std::move(si));
    }
    return out;
}

int SpeakerVectorStore::count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)speakers_.size();
}

int SpeakerVectorStore::total_exemplars() const {
    std::lock_guard<std::mutex> lk(mu_);
    return n_total_;
}

void SpeakerVectorStore::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    n_total_ = 0;
    next_id_ = 0;
    has_pending_ = false;
    for (int s = 0; s < kMaxPending; s++)
        pending_slots_[s].active = false;
    pending_miss_seq_ = 0;
    speakers_.clear();
    offsets_.clear();
    offsets_.push_back(0);
    id_to_idx_.clear();

    // Zero GPU buffers (not strictly required, but clean).
    cudaMemsetAsync(d_embeddings_, 0, (size_t)capacity_ * dim_ * sizeof(float), stream_);
    cudaMemsetAsync(d_offsets_, 0, sizeof(int), stream_);  // single zero sentinel
    cudaMemsetAsync(d_pending_pool_, 0, (size_t)kMaxPending * dim_ * sizeof(float), stream_);
    LOG_INFO(label_.c_str(), "Cleared");
}

bool SpeakerVectorStore::remove_speaker(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    int idx = id_to_idx(id);
    if (idx < 0) return false;

    int begin_row = offsets_[idx];
    int end_row   = offsets_[idx + 1];
    int n_ex      = end_row - begin_row;

    // Remove GPU rows.
    gpu_remove_rows(begin_row, end_row);

    // Erase from host vectors and rebuild offsets + id_to_idx_.
    speakers_.erase(speakers_.begin() + idx);
    offsets_.clear();
    id_to_idx_.clear();
    int off = 0;
    for (int i = 0; i < (int)speakers_.size(); ++i) {
        offsets_.push_back(off);
        id_to_idx_[speakers_[i].external_id] = i;
        off += speakers_[i].exemplar_count;
    }
    offsets_.push_back(off);
    upload_offsets();

    LOG_INFO(label_.c_str(), "Removed speaker #%d (%d exemplars, %d remain)",
             id, n_ex, (int)speakers_.size());
    return true;
}

bool SpeakerVectorStore::merge_speakers(int dst_id, int src_id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (dst_id == src_id) return false;
    int dst_idx = id_to_idx(dst_id);
    int src_idx = id_to_idx(src_id);
    if (dst_idx < 0 || src_idx < 0) return false;

    auto& dst = speakers_[dst_idx];
    auto& src = speakers_[src_idx];
    int total_ex = dst.exemplar_count + src.exemplar_count;

    if (total_ex <= max_exemplars_) {
        // Simple case: all exemplars fit.  Move src rows to after dst rows.
        // Strategy: copy src exemplars to temp GPU buffer, remove src rows,
        // then insert after dst's last exemplar.

        int src_begin = offsets_[src_idx];
        int src_n     = src.exemplar_count;
        size_t src_bytes = (size_t)src_n * dim_ * sizeof(float);

        // Copy src exemplars to d_pending_ area (reuse, or temp alloc).
        float* d_tmp;
        cudaMalloc(&d_tmp, src_bytes);
        cudaMemcpyAsync(d_tmp, d_embeddings_ + src_begin * dim_, src_bytes,
                        cudaMemcpyDeviceToDevice, stream_);
        cudaStreamSynchronize(stream_);

        // Remove src from meta (must do before gpu_remove_rows messes with offsets).
        // But we need to be careful: removing src changes dst_idx if src < dst.
        int src_begin_row = offsets_[src_idx];
        int src_end_row   = offsets_[src_idx + 1];

        // Remove src GPU rows.
        gpu_remove_rows(src_begin_row, src_end_row);

        // Erase src from host.
        speakers_.erase(speakers_.begin() + src_idx);

        // Rebuild offsets and id_to_idx.
        offsets_.clear();
        id_to_idx_.clear();
        int off = 0;
        for (int i = 0; i < (int)speakers_.size(); ++i) {
            offsets_.push_back(off);
            id_to_idx_[speakers_[i].external_id] = i;
            off += speakers_[i].exemplar_count;
        }
        offsets_.push_back(off);

        // Find new dst_idx after erasure.
        int new_dst_idx = id_to_idx(dst_id);
        if (new_dst_idx < 0) { cudaFree(d_tmp); return false; }

        // Insert src exemplars after dst's block.
        int insert_pos = offsets_[new_dst_idx + 1];
        int tail_count = n_total_ - insert_pos;
        ensure_capacity(n_total_ + src_n);

        // Shift tail.
        if (tail_count > 0) {
            cudaMemcpyAsync(d_embeddings_ + (insert_pos + src_n) * dim_,
                            d_embeddings_ + insert_pos * dim_,
                            (size_t)tail_count * dim_ * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_);
        }
        // Copy tmp into gap.
        cudaMemcpyAsync(d_embeddings_ + insert_pos * dim_, d_tmp, src_bytes,
                        cudaMemcpyDeviceToDevice, stream_);
        cudaFree(d_tmp);

        n_total_ += src_n;
        speakers_[new_dst_idx].exemplar_count += src_n;
        speakers_[new_dst_idx].match_count += src.match_count;

        // Rebuild offsets again.
        offsets_.clear();
        id_to_idx_.clear();
        off = 0;
        for (int i = 0; i < (int)speakers_.size(); ++i) {
            offsets_.push_back(off);
            id_to_idx_[speakers_[i].external_id] = i;
            off += speakers_[i].exemplar_count;
        }
        offsets_.push_back(off);
        upload_offsets();
    } else {
        // Need to prune: keep dst exemplars, add src exemplars up to limit,
        // discarding least useful ones (lowest inter-exemplar distance = most redundant).

        // Download all dst + src exemplars to host for pruning.
        int dst_begin = offsets_[dst_idx];
        int dst_n     = dst.exemplar_count;
        int src_begin = offsets_[src_idx];
        int src_n     = src.exemplar_count;

        std::vector<float> all_emb((dst_n + src_n) * dim_);
        cudaMemcpyAsync(all_emb.data(),
                        d_embeddings_ + dst_begin * dim_,
                        (size_t)dst_n * dim_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(all_emb.data() + dst_n * dim_,
                        d_embeddings_ + src_begin * dim_,
                        (size_t)src_n * dim_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Greedy farthest-point selection: start with first dst exemplar,
        // iteratively pick the exemplar farthest from all selected.
        int total = dst_n + src_n;
        std::vector<bool> selected(total, false);
        std::vector<float> min_dist(total, 1e30f);
        selected[0] = true;
        int n_selected = 1;

        // Compute distances from first exemplar.
        auto dot_fn = [&](int i, int j) -> float {
            float d = 0;
            const float* a = all_emb.data() + i * dim_;
            const float* b = all_emb.data() + j * dim_;
            for (int k = 0; k < dim_; k++) d += a[k] * b[k];
            return d;
        };

        for (int i = 1; i < total; ++i)
            min_dist[i] = 1.0f - dot_fn(0, i);  // cosine distance

        while (n_selected < max_exemplars_ && n_selected < total) {
            // Find farthest unselected.
            int best_i = -1;
            float best_d = -1.0f;
            for (int i = 0; i < total; ++i) {
                if (!selected[i] && min_dist[i] > best_d) {
                    best_d = min_dist[i];
                    best_i = i;
                }
            }
            if (best_i < 0) break;
            selected[best_i] = true;
            n_selected++;
            // Update min_dist.
            for (int i = 0; i < total; ++i) {
                if (!selected[i]) {
                    float d = 1.0f - dot_fn(best_i, i);
                    if (d < min_dist[i]) min_dist[i] = d;
                }
            }
        }

        // Build pruned embedding set.
        std::vector<float> pruned(n_selected * dim_);
        int pi = 0;
        for (int i = 0; i < total; ++i) {
            if (selected[i]) {
                memcpy(pruned.data() + pi * dim_, all_emb.data() + i * dim_,
                       dim_ * sizeof(float));
                pi++;
            }
        }

        // Remove src from GPU + meta.
        int src_begin_row = offsets_[src_idx];
        int src_end_row   = offsets_[src_idx + 1];
        gpu_remove_rows(src_begin_row, src_end_row);
        speakers_.erase(speakers_.begin() + src_idx);

        // Rebuild offsets.
        offsets_.clear();
        id_to_idx_.clear();
        int off2 = 0;
        for (int i = 0; i < (int)speakers_.size(); ++i) {
            offsets_.push_back(off2);
            id_to_idx_[speakers_[i].external_id] = i;
            off2 += speakers_[i].exemplar_count;
        }
        offsets_.push_back(off2);

        // Replace dst exemplars with pruned set.
        int new_dst_idx = id_to_idx(dst_id);
        if (new_dst_idx < 0) return false;
        int old_dst_begin = offsets_[new_dst_idx];
        int old_dst_n     = speakers_[new_dst_idx].exemplar_count;
        int delta         = n_selected - old_dst_n;

        if (delta != 0) {
            int old_end  = old_dst_begin + old_dst_n;
            int tail_cnt = n_total_ - old_end;
            if (delta > 0) {
                ensure_capacity(n_total_ + delta);
                if (tail_cnt > 0) {
                    cudaMemcpyAsync(d_embeddings_ + (old_end + delta) * dim_,
                                    d_embeddings_ + old_end * dim_,
                                    (size_t)tail_cnt * dim_ * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_);
                }
            } else {
                if (tail_cnt > 0) {
                    cudaMemcpyAsync(d_embeddings_ + (old_end + delta) * dim_,
                                    d_embeddings_ + old_end * dim_,
                                    (size_t)tail_cnt * dim_ * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_);
                }
            }
            n_total_ += delta;
        }

        // Upload pruned embeddings.
        cudaMemcpyAsync(d_embeddings_ + old_dst_begin * dim_,
                        pruned.data(), (size_t)n_selected * dim_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        speakers_[new_dst_idx].exemplar_count = n_selected;
        speakers_[new_dst_idx].match_count += src.match_count;

        // Final offset rebuild.
        offsets_.clear();
        id_to_idx_.clear();
        int off3 = 0;
        for (int i = 0; i < (int)speakers_.size(); ++i) {
            offsets_.push_back(off3);
            id_to_idx_[speakers_[i].external_id] = i;
            off3 += speakers_[i].exemplar_count;
        }
        offsets_.push_back(off3);
        upload_offsets();
    }

    LOG_INFO(label_.c_str(), "Merged speaker #%d into #%d (%d exemplars)",
             src_id, dst_id, speakers_[id_to_idx(dst_id)].exemplar_count);
    return true;
}


} // namespace deusridet
