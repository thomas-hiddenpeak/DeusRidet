/**
 * @file speaker_db.h
 * @philosophical_role Declaration of the speaker DB facade. Single owner of the on-disk speaker identity store.
 * @serves Orator, Actus, Nexus.
 */
// speaker_db.h — Online speaker identification and clustering.
//
// Maintains a database of known speaker embeddings. New embeddings are matched
// against existing speakers via cosine similarity. Unknown speakers are
// auto-registered. Embeddings are updated via exponential moving average.
//
// Adapted from qwen35-orin SpeakerManager (Thomas Zhu)

#pragma once

#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace deusridet {

struct SpeakerMatch {
    int    speaker_id   = -1;     // -1 = unknown / new
    float  similarity   = 0.0f;   // best cosine similarity
    bool   is_new       = false;  // true if auto-registered
    std::string name;             // empty if unnamed
    int    exemplar_count = 0;    // total exemplars for matched speaker
    int    hits_above     = 0;    // exemplars above threshold in this query
    float  second_best_sim = 0.0f; // second-best speaker similarity (for margin)
    int    second_best_id  = -1;   // second-best speaker ID
    bool   is_amend        = false; // true when this is a retroactive relabel
    int    prior_speaker_id = -1;   // original emitted speaker for amend target
    float  prior_similarity = 0.0f;
    float  amend_t_close_sec = 0.0f; // source-time close of target segment
};

struct SpeakerInfo {
    int id;
    std::string name;                 // user-assigned name (empty = unnamed)
    std::vector<float> embedding;     // moving-average embedding (L2-normed)
    int match_count = 0;              // number of times matched
    int exemplar_count = 1;           // number of stored exemplars (SpeakerDb always 1)
    float ema_alpha = 0.3f;           // EMA weight for embedding update
    float min_diversity = -1.0f;      // minimum pairwise exemplar diversity (-1 = N/A)
};

class SpeakerDb {
public:
    SpeakerDb() = default;
    explicit SpeakerDb(const std::string& label, float ema_alpha = 0.3f)
        : label_(label), ema_alpha_(ema_alpha) {}

    // Identify speaker from an embedding. If no match exceeds threshold,
    // auto-register as new speaker (if auto_register=true).
    // Uses a pending mechanism: first unmatched embedding is held in pending;
    // only registers on second consecutive miss (with averaged embedding).
    SpeakerMatch identify(const std::vector<float>& embedding,
                          float threshold = 0.65f,
                          bool auto_register = true);

    // Register a named speaker with a known embedding.
    int register_speaker(const std::string& name,
                         const std::vector<float>& embedding);

    // Rename a speaker.
    void set_name(int id, const std::string& name);

    // Get all speakers.
    std::vector<SpeakerInfo> all_speakers() const;

    // Get speaker count.
    int count() const;

    // Reset all speakers.
    void clear();

    // Cosine similarity (static utility).
    static float cosine_sim(const std::vector<float>& a,
                            const std::vector<float>& b);

private:
    mutable std::mutex mu_;
    std::vector<SpeakerInfo> speakers_;
    int next_id_ = 0;

    // Pending new speaker: hold embedding until a second consecutive miss confirms.
    bool has_pending_ = false;
    std::vector<float> pending_embedding_;
    std::string label_ = "SpeakerDb";
    float ema_alpha_ = 0.3f;       // EMA weight for centroid updates
};

} // namespace deusridet
