/**
 * @file speaker_db.cpp
 * @philosophical_role Persistent speaker database — names, exemplars, aliases. The entity remembers voices across restarts; forgetting a known speaker is a bug, not an optimization.
 * @serves Orator identification, Nexus speaker-list broadcasts, Actus profile commands.
 */
// speaker_db.cpp — Online speaker identification and clustering.

#include "speaker_db.h"
#include "../communis/log.h"

namespace deusridet {

float SpeakerDb::cosine_sim(const std::vector<float>& a,
                            const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

SpeakerMatch SpeakerDb::identify(const std::vector<float>& embedding,
                                  float threshold, bool auto_register) {
    std::lock_guard<std::mutex> lk(mu_);

    SpeakerMatch best;
    best.speaker_id = -1;
    best.similarity = 0.0f;
    float second_best_sim = 0.0f;
    int second_best_id = -1;

    for (auto& spk : speakers_) {
        float sim = cosine_sim(embedding, spk.embedding);
        if (sim > best.similarity) {
            second_best_sim = best.similarity;
            second_best_id = best.speaker_id;
            best.similarity = sim;
            best.speaker_id = spk.id;
            best.name = spk.name;
        } else if (sim > second_best_sim) {
            second_best_sim = sim;
            second_best_id = spk.id;
        }
    }

    if (!speakers_.empty()) {
        LOG_INFO(label_.c_str(), "Match: best=#%d(%.3f) 2nd=#%d(%.3f) thresh=%.2f db=%d",
                 best.speaker_id, best.similarity,
                 second_best_id, second_best_sim,
                 threshold, (int)speakers_.size());
    }

    if (best.similarity >= threshold && best.speaker_id >= 0) {
        // Matched an existing speaker — clear any pending.
        has_pending_ = false;

        // Update embedding via EMA (using db-level alpha for consistency).
        auto& spk = speakers_[best.speaker_id];
        float alpha = ema_alpha_;
        for (size_t i = 0; i < embedding.size(); ++i) {
            spk.embedding[i] = (1.0f - alpha) * spk.embedding[i]
                              + alpha * embedding[i];
        }
        // Re-normalize L2.
        float norm = 0;
        for (auto v : spk.embedding) norm += v * v;
        norm = 1.0f / (sqrtf(norm) + 1e-12f);
        for (auto& v : spk.embedding) v *= norm;

        spk.match_count++;
        best.is_new = false;
        return best;
    }

    if (auto_register) {
        if (!has_pending_) {
            // First miss: store as pending, don't register yet.
            pending_embedding_ = embedding;
            has_pending_ = true;
            LOG_INFO(label_.c_str(), "Pending new speaker (waiting for confirmation)");
            best.speaker_id = -1;
            best.similarity = 0.0f;
            best.is_new = false;
            return best;
        }

        // Second consecutive miss. Check if pending and current are from the
        // same speaker — if not, this is two different unknown speakers in
        // succession. Merging them would create a chimeric embedding.
        float pending_sim = cosine_sim(pending_embedding_, embedding);
        if (pending_sim < threshold) {
            // Different speakers. Replace pending with current, don't register.
            pending_embedding_ = embedding;
            LOG_INFO(label_.c_str(), "Pending replaced (prev vs cur sim=%.3f < %.2f)",
                     pending_sim, threshold);
            best.speaker_id = -1;
            best.similarity = 0.0f;
            best.is_new = false;
            return best;
        }

        // Same speaker confirmed: average pending + current, register.
        std::vector<float> avg_emb(embedding.size());
        for (size_t i = 0; i < embedding.size(); ++i)
            avg_emb[i] = 0.5f * pending_embedding_[i] + 0.5f * embedding[i];
        // L2-normalize the averaged embedding.
        float norm = 0;
        for (auto v : avg_emb) norm += v * v;
        norm = 1.0f / (sqrtf(norm) + 1e-12f);
        for (auto& v : avg_emb) v *= norm;

        SpeakerInfo info;
        info.id = next_id_++;
        info.embedding = std::move(avg_emb);
        info.match_count = 2;
        speakers_.push_back(std::move(info));

        has_pending_ = false;
        pending_embedding_.clear();

        LOG_INFO(label_.c_str(), "Confirmed new speaker id=%d (pending_sim=%.3f)",
                 speakers_.back().id, pending_sim);
        best.speaker_id = speakers_.back().id;
        best.similarity = 1.0f;
        best.is_new = true;
        return best;
    }

    best.speaker_id = -1;
    return best;
}

int SpeakerDb::register_speaker(const std::string& name,
                                 const std::vector<float>& embedding) {
    std::lock_guard<std::mutex> lk(mu_);
    SpeakerInfo info;
    info.id = next_id_++;
    info.name = name;
    info.embedding = embedding;
    info.match_count = 1;
    speakers_.push_back(std::move(info));
    return speakers_.back().id;
}

void SpeakerDb::set_name(int id, const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& spk : speakers_) {
        if (spk.id == id) { spk.name = name; return; }
    }
}

std::vector<SpeakerInfo> SpeakerDb::all_speakers() const {
    std::lock_guard<std::mutex> lk(mu_);
    return speakers_;
}

int SpeakerDb::count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)speakers_.size();
}

void SpeakerDb::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    speakers_.clear();
    next_id_ = 0;
    has_pending_ = false;
    pending_embedding_.clear();
}

} // namespace deusridet
