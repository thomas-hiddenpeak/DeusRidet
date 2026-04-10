// speaker_vector_store.h — GPU-accelerated speaker vector store with multi-exemplar support.
//
// Replaces SpeakerDb for production scenarios with thousands of speakers.
// All similarity computation happens on GPU via custom CUDA kernels.
// Embeddings stored in a contiguous GPU buffer, sorted by speaker.
//
// Design:
//   - Each speaker holds 1–max_exemplars embeddings (capturing voice variations)
//   - Search: batch dot product over all exemplars + per-speaker max reduction
//   - Update: GPU-side EMA on best exemplar, or add new exemplar if dissimilar
//   - Persistence: binary embedding file + binary metadata
//
// All vector math (dot products, norms, EMA, search) runs on GPU.
// CPU only touches metadata strings and file I/O.

#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "speaker_db.h"  // SpeakerMatch, SpeakerInfo

namespace deusridet {

// GPU search result — written by the reduce kernel, read back on host.
struct alignas(16) GpuSearchResult {
    int   spk_idx;       // internal speaker index (-1 if empty)
    float similarity;    // best cosine similarity
    int   exemplar_row;  // which row in d_embeddings_ was best
    int   _pad;
};

// Host-side metadata for one speaker (no embeddings — those live on GPU).
struct SpeakerMeta {
    int  external_id;        // user-visible speaker ID
    std::string name;        // user-assigned name (empty = unnamed)
    int  exemplar_count = 0; // number of stored exemplars (1–max_exemplars)
    int  match_count    = 0; // lifetime match count
};

class SpeakerVectorStore {
public:
    // dim: embedding dimension (must be divisible by 4 for float4 vectorization).
    // ema_alpha: EMA weight for centroid updates when exemplar is similar.
    // max_exemplars: maximum exemplar embeddings per speaker (1–15).
    // initial_capacity: initial GPU buffer size in exemplar rows.
    explicit SpeakerVectorStore(const std::string& label,
                                int dim,
                                float ema_alpha       = 0.15f,
                                int   max_exemplars   = 15,
                                int   initial_capacity = 256);
    ~SpeakerVectorStore();

    SpeakerVectorStore(const SpeakerVectorStore&) = delete;
    SpeakerVectorStore& operator=(const SpeakerVectorStore&) = delete;

    // ---- Core API (drop-in compatible with SpeakerDb) ----

    // Identify speaker from embedding. Returns match or auto-registers new
    // speaker after 2-miss confirmation (same as SpeakerDb pending logic).
    SpeakerMatch identify(const std::vector<float>& embedding,
                          float threshold     = 0.65f,
                          bool  auto_register = true);

    // Register a named speaker with a known embedding.
    int register_speaker(const std::string& name,
                         const std::vector<float>& embedding);

    // Rename a speaker by external ID.
    void set_name(int id, const std::string& name);

    // Get all speakers (embedding field left empty — data is on GPU).
    std::vector<SpeakerInfo> all_speakers() const;

    // Speaker count.
    int count() const;

    // Total exemplar count across all speakers.
    int total_exemplars() const;

    // Reset everything, free GPU buffers and reallocate at initial capacity.
    void clear();

    // Remove a speaker by external ID. Shifts GPU buffer, rebuilds offsets.
    // Returns true if found and removed.
    bool remove_speaker(int id);

    // Merge speaker src_id into dst_id. Moves exemplars from src to dst
    // (up to max_exemplars, pruning most-redundant if over limit).
    // src is deleted after merge. Returns true on success.
    bool merge_speakers(int dst_id, int src_id);

    // ---- Margin guard ----
    void  set_min_margin(float m) { min_margin_ = m; }
    float min_margin() const { return min_margin_; }

    // ---- Persistence ----

    // Save store to directory (creates dir if needed).
    // Files: meta.bin (speaker metadata) + embeddings.bin (raw float32).
    bool save(const std::string& dir) const;

    // Load store from directory. Replaces current contents.
    bool load(const std::string& dir);

    // ---- Accessors ----

    int dim() const { return dim_; }
    const std::string& label() const { return label_; }

private:
    // Internal GPU search over all exemplars.
    // Must be called with mu_ held.  d_query_ must contain the query.
    GpuSearchResult gpu_search(int n_total, int n_speakers);

    // GPU EMA update: blend d_query_ into row exemplar_row, re-normalize.
    void gpu_ema_update(int exemplar_row, float alpha);

    // Insert a new exemplar for spk_idx.  d_query_ holds the embedding.
    // Shifts tail of d_embeddings_, updates offsets.
    void gpu_add_exemplar(int spk_idx);

    // Upload query embedding from host to d_query_.
    void upload_query(const float* host_emb);

    // Grow GPU embedding buffer if needed.
    void ensure_capacity(int needed);

    // Sync host offsets_ → d_offsets_.
    void upload_offsets();

    // Compute single dot product between d_pending_pool_[0] and d_query_ on GPU.
    // Legacy — use gpu_pending_dot(int slot) for multi-pending.
    float gpu_pending_dot();

    // Count how many exemplars of spk_idx scored above threshold.
    // Must be called after gpu_search() (reads d_sims_).
    int count_hits_above(int spk_idx, float threshold);

    // Compute minimum cosine distance from d_query_ to all exemplars of spk_idx.
    // Returns min diversity (smallest distance to any existing exemplar).
    // Must be called after gpu_search() (reads d_sims_).
    float min_diversity(int spk_idx);

    // Find the most redundant exemplar in spk_idx (closest to its nearest neighbor).
    // Downloads exemplars to host for pairwise comparison. Returns row index.
    int most_redundant_exemplar(int spk_idx);

    // Replace a single exemplar row with d_query_ content (frozen, no EMA).
    void gpu_replace_exemplar(int row);

    // Remove exemplar rows [begin, end) from GPU buffer, shift tail, update offsets.
    void gpu_remove_rows(int begin_row, int end_row);

    // External ID → internal index.
    int id_to_idx(int id) const;

    // Compute dot product between d_pending_pool_[slot] and d_query_.
    float gpu_pending_dot(int slot);

    // ---- Device buffers ----
    float* d_embeddings_ = nullptr;  // [capacity_ × dim_], L2-normed
    int*   d_offsets_    = nullptr;  // [spk_alloc_ + 1], exemplar prefix sums
    float* d_query_      = nullptr;  // [dim_]
    float* d_sims_       = nullptr;  // [capacity_], per-exemplar dot products
    float* d_spk_sims_   = nullptr;  // [spk_alloc_], per-speaker best sim
    int*   d_spk_ex_     = nullptr;  // [spk_alloc_], best exemplar row per spk
    float* d_pending_    = nullptr;  // [dim_], legacy alias → d_pending_pool_[0]
    GpuSearchResult* d_result_ = nullptr;  // single-element result

    // Multi-pending pool: up to kMaxPending slots for simultaneous unknown candidates.
    static constexpr int kMaxPending = 3;
    float* d_pending_pool_ = nullptr;  // [kMaxPending × dim_], GPU embeddings

    // ---- Pinned host buffer for result readback ----
    GpuSearchResult* h_result_ = nullptr;

    // ---- Host state ----
    std::string label_;
    int   dim_;
    float ema_alpha_;
    int   max_exemplars_;
    int   initial_capacity_;
    int   capacity_    = 0;   // allocated rows in d_embeddings_
    int   n_total_     = 0;   // total exemplar count
    int   spk_alloc_   = 512; // allocated speaker slots in GPU arrays
    int   next_id_     = 0;

    mutable std::mutex mu_;
    std::vector<SpeakerMeta> speakers_;
    std::vector<int> offsets_;  // host mirror of d_offsets_, size = speakers_.size()+1
    std::unordered_map<int, int> id_to_idx_;

    // Multi-pending pool state.
    struct PendingSlot {
        bool   active   = false;
        int    miss_seq = 0;  // monotonic sequence number at creation (for LRU)
    };
    PendingSlot pending_slots_[kMaxPending] = {};
    int pending_miss_seq_ = 0;  // global miss counter for LRU

    // Legacy single-pending compat.
    bool has_pending_ = false;

    // Minimum cosine distance for a new exemplar to be considered "diverse enough".
    // If the closest existing exemplar is nearer than this, the new sample is redundant.
    static constexpr float kDiversityThresh = 0.08f;

    // Exemplar admission margin: match_sim must exceed threshold by at least
    // this amount before a new exemplar is added.  Prevents borderline matches
    // (just above threshold) from contaminating the speaker profile.
    static constexpr float kExemplarAdmitMargin = 0.10f;

    // Margin guard for registration: if pending embedding is too close to two
    // existing speakers, reject registration to avoid creating duplicates.
    float min_margin_ = 0.08f;

    cudaStream_t stream_ = nullptr;
};

} // namespace deusridet
