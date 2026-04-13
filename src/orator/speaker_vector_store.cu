// speaker_vector_store.cu — GPU kernels and implementation for SpeakerVectorStore.
//
// Kernels:
//   batch_dot_kernel      — one warp per exemplar, float4 vectorized, query in SMEM
//   speaker_reduce_kernel — per-speaker max similarity + global argmax (single block)
//   ema_normalize_kernel  — EMA blend + L2 re-normalize (single warp)
//   single_dot_kernel     — single dot product (pending confirmation)

#include "speaker_vector_store.h"
#include "../communis/log.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <sys/stat.h>
#include <cerrno>

namespace deusridet {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Batch dot product: one warp computes dot(exemplar[warp_id], query).
// Embeddings are L2-normed, so dot == cosine similarity.
// query is loaded to shared memory once per block (broadcast optimization).
__global__ void batch_dot_kernel(const float4* __restrict__ embeddings,
                                 const float4* __restrict__ query,
                                 float* __restrict__ sims,
                                 int n_exemplars,
                                 int d_f4)   // dim / 4
{
    extern __shared__ float4 s_query[];

    // Cooperative load of query into shared memory.
    for (int i = threadIdx.x; i < d_f4; i += blockDim.x)
        s_query[i] = query[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;
    if (warp_id >= n_exemplars) return;

    const float4* row = embeddings + warp_id * d_f4;
    float sum = 0.0f;
    for (int i = lane; i < d_f4; i += 32) {
        float4 e = row[i];
        float4 q = s_query[i];
        sum += e.x * q.x + e.y * q.y + e.z * q.z + e.w * q.w;
    }

    // Warp shuffle reduction.
    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);

    if (lane == 0)
        sims[warp_id] = sum;
}

// Per-speaker max similarity + global argmax.  Single block, 256 threads.
// Each thread scans a subset of speakers, then block-level reduction.
__global__ void speaker_reduce_kernel(const float* __restrict__ sims,
                                       const int*   __restrict__ offsets,
                                       int n_speakers,
                                       float* __restrict__ spk_best_sim,
                                       int*   __restrict__ spk_best_ex,
                                       GpuSearchResult* __restrict__ result)
{
    __shared__ float s_sim[256];
    __shared__ int   s_spk[256];
    __shared__ int   s_ex[256];

    int tid = threadIdx.x;

    float my_best_sim = -1.0f;
    int   my_best_spk = -1;
    int   my_best_ex  = -1;

    // Phase 1: each thread handles some speakers.
    for (int s = tid; s < n_speakers; s += blockDim.x) {
        int begin = offsets[s];
        int end   = offsets[s + 1];
        float best = -1.0f;
        int   best_ex = begin;
        for (int e = begin; e < end; ++e) {
            if (sims[e] > best) {
                best    = sims[e];
                best_ex = e;
            }
        }
        spk_best_sim[s] = best;
        spk_best_ex[s]  = best_ex;

        if (best > my_best_sim) {
            my_best_sim = best;
            my_best_spk = s;
            my_best_ex  = best_ex;
        }
    }

    // Phase 2: block reduction for global best.
    s_sim[tid] = my_best_sim;
    s_spk[tid] = my_best_spk;
    s_ex[tid]  = my_best_ex;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && s_sim[tid + stride] > s_sim[tid]) {
            s_sim[tid] = s_sim[tid + stride];
            s_spk[tid] = s_spk[tid + stride];
            s_ex[tid]  = s_ex[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result->spk_idx      = s_spk[0];
        result->similarity   = s_sim[0];
        result->exemplar_row = s_ex[0];
    }
}

// EMA update + L2 re-normalize a single exemplar row in-place.
// new_embedding = (1-alpha)*old + alpha*query, then normalize.
// Single warp, float4 vectorized.
__global__ void ema_normalize_kernel(float4* __restrict__ embedding_row,
                                     const float4* __restrict__ query,
                                     int d_f4,
                                     float alpha)
{
    int lane = threadIdx.x;  // 0..31
    float partial_norm = 0.0f;

    for (int i = lane; i < d_f4; i += 32) {
        float4 e = embedding_row[i];
        float4 q = query[i];
        float4 blended;
        blended.x = (1.0f - alpha) * e.x + alpha * q.x;
        blended.y = (1.0f - alpha) * e.y + alpha * q.y;
        blended.z = (1.0f - alpha) * e.z + alpha * q.z;
        blended.w = (1.0f - alpha) * e.w + alpha * q.w;
        embedding_row[i] = blended;
        partial_norm += blended.x * blended.x + blended.y * blended.y
                      + blended.z * blended.z + blended.w * blended.w;
    }

    // Warp reduce sum.
    for (int mask = 16; mask > 0; mask >>= 1)
        partial_norm += __shfl_xor_sync(0xFFFFFFFF, partial_norm, mask);

    float inv_norm = rsqrtf(partial_norm + 1e-12f);

    // Second pass: scale.
    for (int i = lane; i < d_f4; i += 32) {
        float4 v = embedding_row[i];
        v.x *= inv_norm; v.y *= inv_norm;
        v.z *= inv_norm; v.w *= inv_norm;
        embedding_row[i] = v;
    }
}

// L2-normalize a single vector.  Single warp.
__global__ void l2_normalize_kernel(float4* __restrict__ vec, int d_f4)
{
    int lane = threadIdx.x;
    float partial = 0.0f;
    for (int i = lane; i < d_f4; i += 32) {
        float4 v = vec[i];
        partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, mask);
    float inv = rsqrtf(partial + 1e-12f);
    for (int i = lane; i < d_f4; i += 32) {
        float4 v = vec[i];
        v.x *= inv; v.y *= inv; v.z *= inv; v.w *= inv;
        vec[i] = v;
    }
}

// Single dot product between two vectors.  Result in sims[0].
__global__ void single_dot_kernel(const float4* __restrict__ a,
                                  const float4* __restrict__ b,
                                  float* __restrict__ out,
                                  int d_f4)
{
    int lane = threadIdx.x;
    float sum = 0.0f;
    for (int i = lane; i < d_f4; i += 32) {
        float4 va = a[i], vb = b[i];
        sum += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);
    if (lane == 0) out[0] = sum;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

SpeakerVectorStore::SpeakerVectorStore(const std::string& label, int dim,
                                       float ema_alpha, int max_exemplars,
                                       int initial_capacity)
    : label_(label), dim_(dim), ema_alpha_(ema_alpha),
      max_exemplars_(max_exemplars), initial_capacity_(initial_capacity)
{
    assert(dim % 4 == 0 && "dim must be divisible by 4 for float4 vectorization");
    assert(max_exemplars >= 1 && max_exemplars <= 64);

    cudaStreamCreate(&stream_);

    capacity_ = initial_capacity;
    size_t emb_bytes = (size_t)capacity_ * dim_ * sizeof(float);
    cudaMalloc(&d_embeddings_, emb_bytes);
    cudaMemsetAsync(d_embeddings_, 0, emb_bytes, stream_);

    cudaMalloc(&d_offsets_, (spk_alloc_ + 1) * sizeof(int));
    cudaMemsetAsync(d_offsets_, 0, (spk_alloc_ + 1) * sizeof(int), stream_);

    cudaMalloc(&d_query_,    dim_ * sizeof(float));
    cudaMalloc(&d_sims_,     capacity_ * sizeof(float));
    cudaMalloc(&d_spk_sims_, spk_alloc_ * sizeof(float));
    cudaMalloc(&d_spk_ex_,   spk_alloc_ * sizeof(int));
    cudaMalloc(&d_pending_pool_, (size_t)kMaxPending * dim_ * sizeof(float));
    cudaMemsetAsync(d_pending_pool_, 0, (size_t)kMaxPending * dim_ * sizeof(float), stream_);
    d_pending_ = d_pending_pool_;  // legacy alias → slot 0
    cudaMalloc(&d_result_,   sizeof(GpuSearchResult));

    cudaMallocHost(&h_result_, sizeof(GpuSearchResult));

    offsets_.push_back(0);  // empty: single sentinel

    LOG_INFO(label_.c_str(), "SpeakerVectorStore created: dim=%d cap=%d max_ex=%d",
             dim_, capacity_, max_exemplars_);
}

SpeakerVectorStore::~SpeakerVectorStore() {
    if (stream_) cudaStreamSynchronize(stream_);

    cudaFree(d_embeddings_);
    cudaFree(d_offsets_);
    cudaFree(d_query_);
    cudaFree(d_sims_);
    cudaFree(d_spk_sims_);
    cudaFree(d_spk_ex_);
    cudaFree(d_pending_pool_);
    d_pending_ = nullptr;  // was alias into pool
    cudaFree(d_result_);

    if (h_result_) cudaFreeHost(h_result_);
    if (stream_)   cudaStreamDestroy(stream_);
}

// ============================================================================
// GPU helper methods
// ============================================================================

void SpeakerVectorStore::upload_query(const float* host_emb) {
    cudaMemcpyAsync(d_query_, host_emb, dim_ * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
}

void SpeakerVectorStore::upload_offsets() {
    int n = (int)offsets_.size();
    if (n > spk_alloc_ + 1) {
        // Grow speaker slot arrays.
        int new_alloc = std::max(spk_alloc_ * 2, n);
        int* new_off;     cudaMalloc(&new_off,     (new_alloc + 1) * sizeof(int));
        float* new_sims;  cudaMalloc(&new_sims,    new_alloc * sizeof(float));
        int* new_ex;      cudaMalloc(&new_ex,      new_alloc * sizeof(int));
        cudaFree(d_offsets_);  d_offsets_  = new_off;
        cudaFree(d_spk_sims_); d_spk_sims_ = new_sims;
        cudaFree(d_spk_ex_);   d_spk_ex_   = new_ex;
        spk_alloc_ = new_alloc;
    }
    cudaMemcpyAsync(d_offsets_, offsets_.data(), n * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);
}

void SpeakerVectorStore::ensure_capacity(int needed) {
    if (needed <= capacity_) return;
    int new_cap = std::max(capacity_ * 2, needed);
    float* new_emb;
    cudaMalloc(&new_emb, (size_t)new_cap * dim_ * sizeof(float));
    if (n_total_ > 0) {
        cudaMemcpyAsync(new_emb, d_embeddings_,
                        (size_t)n_total_ * dim_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
    cudaFree(d_embeddings_);
    d_embeddings_ = new_emb;

    // Also grow d_sims_.
    cudaFree(d_sims_);
    cudaMalloc(&d_sims_, new_cap * sizeof(float));

    capacity_ = new_cap;
    LOG_INFO(label_.c_str(), "GPU buffer grown to capacity=%d", capacity_);
}

GpuSearchResult SpeakerVectorStore::gpu_search(int n_total, int n_speakers) {
    int d_f4 = dim_ / 4;

    // Kernel 1: batch dot product.
    {
        int warps_per_block = 8;  // 256 threads per block
        int threads = warps_per_block * 32;
        int blocks = (n_total + warps_per_block - 1) / warps_per_block;
        size_t smem = d_f4 * sizeof(float4);
        batch_dot_kernel<<<blocks, threads, smem, stream_>>>(
            reinterpret_cast<const float4*>(d_embeddings_),
            reinterpret_cast<const float4*>(d_query_),
            d_sims_, n_total, d_f4);
    }

    // Kernel 2: per-speaker max + global argmax.
    speaker_reduce_kernel<<<1, 256, 0, stream_>>>(
        d_sims_, d_offsets_, n_speakers,
        d_spk_sims_, d_spk_ex_, d_result_);

    // Readback result (12 bytes).
    cudaMemcpyAsync(h_result_, d_result_, sizeof(GpuSearchResult),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return *h_result_;
}

void SpeakerVectorStore::gpu_ema_update(int exemplar_row, float alpha) {
    int d_f4 = dim_ / 4;
    float4* row = reinterpret_cast<float4*>(d_embeddings_ + exemplar_row * dim_);
    const float4* q = reinterpret_cast<const float4*>(d_query_);
    ema_normalize_kernel<<<1, 32, 0, stream_>>>(row, q, d_f4, alpha);
}

void SpeakerVectorStore::gpu_add_exemplar(int spk_idx) {
    // Insert position: right after this speaker's last exemplar.
    int insert_pos = offsets_[spk_idx + 1];
    int tail_count = n_total_ - insert_pos;

    ensure_capacity(n_total_ + 1);

    // Shift tail by one row on GPU (device-to-device DMA).
    if (tail_count > 0) {
        cudaMemcpyAsync(d_embeddings_ + (insert_pos + 1) * dim_,
                        d_embeddings_ + insert_pos * dim_,
                        (size_t)tail_count * dim_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // Copy query into the gap.
    cudaMemcpyAsync(d_embeddings_ + insert_pos * dim_,
                    d_query_, dim_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // L2-normalize the new exemplar on GPU.
    int d_f4 = dim_ / 4;
    float4* row = reinterpret_cast<float4*>(d_embeddings_ + insert_pos * dim_);
    l2_normalize_kernel<<<1, 32, 0, stream_>>>(row, d_f4);

    n_total_++;

    // Update host offsets: all speakers after spk_idx shift +1.
    for (int i = spk_idx + 1; i < (int)offsets_.size(); ++i)
        offsets_[i]++;
    speakers_[spk_idx].exemplar_count++;

    upload_offsets();
}

float SpeakerVectorStore::gpu_pending_dot() {
    return gpu_pending_dot(0);
}

float SpeakerVectorStore::gpu_pending_dot(int slot) {
    int d_f4 = dim_ / 4;
    float* slot_ptr = d_pending_pool_ + slot * dim_;
    single_dot_kernel<<<1, 32, 0, stream_>>>(
        reinterpret_cast<const float4*>(slot_ptr),
        reinterpret_cast<const float4*>(d_query_),
        d_sims_,  // reuse first slot as scratch
        d_f4);
    float dot;
    cudaMemcpyAsync(&dot, d_sims_, sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return dot;
}

int SpeakerVectorStore::id_to_idx(int id) const {
    auto it = id_to_idx_.find(id);
    return (it != id_to_idx_.end()) ? it->second : -1;
}

int SpeakerVectorStore::count_hits_above(int spk_idx, float threshold) {
    int begin = offsets_[spk_idx];
    int end   = offsets_[spk_idx + 1];
    int n = end - begin;
    if (n <= 0) return 0;
    // Read this speaker's per-exemplar similarities from GPU.
    std::vector<float> sims(n);
    cudaMemcpyAsync(sims.data(), d_sims_ + begin, n * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    int hits = 0;
    for (float s : sims) if (s >= threshold) hits++;
    return hits;
}

float SpeakerVectorStore::min_diversity(int spk_idx) {
    // Returns the minimum cosine distance from d_query_ to any exemplar
    // of this speaker. d_sims_ already has per-exemplar dot products from gpu_search().
    int begin = offsets_[spk_idx];
    int end   = offsets_[spk_idx + 1];
    int n = end - begin;
    if (n <= 0) return 1.0f;  // no exemplars → maximally diverse
    std::vector<float> sims(n);
    cudaMemcpyAsync(sims.data(), d_sims_ + begin, n * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    float max_sim = -1.0f;
    for (float s : sims) if (s > max_sim) max_sim = s;
    return 1.0f - max_sim;  // cosine distance = 1 - similarity
}

int SpeakerVectorStore::most_redundant_exemplar(int spk_idx) {
    // Find the exemplar whose nearest neighbor (within the same speaker)
    // is closest — i.e., the one contributing least unique coverage.
    // Downloads exemplars to host for pairwise comparison.
    int begin = offsets_[spk_idx];
    int end   = offsets_[spk_idx + 1];
    int n = end - begin;
    if (n <= 1) return begin;

    std::vector<float> embs(n * dim_);
    cudaMemcpyAsync(embs.data(), d_embeddings_ + begin * dim_,
                    (size_t)n * dim_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // For each exemplar, find its nearest neighbor distance.
    auto dot_fn = [&](int i, int j) -> float {
        float d = 0;
        const float* a = embs.data() + i * dim_;
        const float* b = embs.data() + j * dim_;
        for (int k = 0; k < dim_; k++) d += a[k] * b[k];
        return d;
    };

    int most_redundant = 0;
    float smallest_nn_dist = 1e30f;
    for (int i = 0; i < n; ++i) {
        float best_sim = -1.0f;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            float s = dot_fn(i, j);
            if (s > best_sim) best_sim = s;
        }
        float nn_dist = 1.0f - best_sim;
        if (nn_dist < smallest_nn_dist) {
            smallest_nn_dist = nn_dist;
            most_redundant = i;
        }
    }
    return begin + most_redundant;
}

void SpeakerVectorStore::gpu_replace_exemplar(int row) {
    // Replace exemplar at row with d_query_ (frozen copy, L2 normalized).
    cudaMemcpyAsync(d_embeddings_ + row * dim_, d_query_,
                    dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    int d_f4 = dim_ / 4;
    float4* r = reinterpret_cast<float4*>(d_embeddings_ + row * dim_);
    l2_normalize_kernel<<<1, 32, 0, stream_>>>(r, d_f4);
}

void SpeakerVectorStore::gpu_remove_rows(int begin_row, int end_row) {
    int remove_count = end_row - begin_row;
    if (remove_count <= 0) return;
    int tail_count = n_total_ - end_row;
    if (tail_count > 0) {
        cudaMemcpyAsync(d_embeddings_ + begin_row * dim_,
                        d_embeddings_ + end_row * dim_,
                        (size_t)tail_count * dim_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
    n_total_ -= remove_count;
}

// ============================================================================
// Core API
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

            if (best.similarity >= admit_thresh && div >= kDiversityThresh && hit_ratio_ok) {
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
    for (int s = 0; s < kMaxPending; s++) {
        if (!pending_slots_[s].active) continue;
        float sim = gpu_pending_dot(s);
        LOG_INFO(label_.c_str(),
                 "Pending[%d] vs query: sim=%.3f (reg_thresh=%.2f, age=%d)",
                 s, sim, reg_thresh, pending_miss_seq_ - pending_slots_[s].miss_seq);
        if (sim >= reg_thresh && sim > best_pending_sim) {
            best_pending_sim = sim;
            matched_slot = s;
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
            if (pending_slots_[s].miss_seq < oldest_seq) {
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

// ============================================================================
// Persistence
// ============================================================================

// Binary format:
//   meta.bin:
//     int32  n_speakers
//     int32  dim
//     int32  next_id
//     for each speaker:
//       int32  external_id
//       int32  name_len
//       char[] name (no null terminator)
//       int32  exemplar_count
//       int32  match_count
//   embeddings.bin:
//     float32 × (n_total × dim)  — raw, contiguous, sorted by speaker

static bool ensure_dir(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    return mkdir(path.c_str(), 0755) == 0;
}

bool SpeakerVectorStore::save(const std::string& dir) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!ensure_dir(dir)) {
        LOG_INFO(label_.c_str(), "save: cannot create dir %s: %s",
                 dir.c_str(), strerror(errno));
        return false;
    }

    // --- meta.bin ---
    {
        std::string path = dir + "/meta.bin";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { LOG_INFO(label_.c_str(), "save: fopen %s: %s", path.c_str(), strerror(errno)); return false; }

        int32_t ns = (int32_t)speakers_.size();
        int32_t d  = (int32_t)dim_;
        int32_t ni = (int32_t)next_id_;
        fwrite(&ns, 4, 1, f);
        fwrite(&d,  4, 1, f);
        fwrite(&ni, 4, 1, f);

        for (auto& m : speakers_) {
            int32_t eid = m.external_id;
            int32_t nl  = (int32_t)m.name.size();
            int32_t ec  = m.exemplar_count;
            int32_t mc  = m.match_count;
            fwrite(&eid, 4, 1, f);
            fwrite(&nl,  4, 1, f);
            if (nl > 0) fwrite(m.name.data(), 1, nl, f);
            fwrite(&ec, 4, 1, f);
            fwrite(&mc, 4, 1, f);
        }
        fclose(f);
    }

    // --- embeddings.bin ---
    if (n_total_ > 0) {
        std::string path = dir + "/embeddings.bin";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { LOG_INFO(label_.c_str(), "save: fopen %s: %s", path.c_str(), strerror(errno)); return false; }

        size_t bytes = (size_t)n_total_ * dim_ * sizeof(float);
        std::vector<float> host_buf(n_total_ * dim_);
        cudaMemcpy(host_buf.data(), d_embeddings_, bytes, cudaMemcpyDeviceToHost);
        fwrite(host_buf.data(), sizeof(float), n_total_ * dim_, f);
        fclose(f);
    }

    LOG_INFO(label_.c_str(), "Saved %d speakers (%d exemplars) to %s",
             (int)speakers_.size(), n_total_, dir.c_str());
    return true;
}

bool SpeakerVectorStore::load(const std::string& dir) {
    std::lock_guard<std::mutex> lk(mu_);

    // --- meta.bin ---
    std::string meta_path = dir + "/meta.bin";
    FILE* mf = fopen(meta_path.c_str(), "rb");
    if (!mf) {
        LOG_INFO(label_.c_str(), "load: no meta.bin in %s", dir.c_str());
        return false;
    }

    int32_t ns, d, ni;
    if (fread(&ns, 4, 1, mf) != 1 || fread(&d, 4, 1, mf) != 1 ||
        fread(&ni, 4, 1, mf) != 1) {
        fclose(mf); return false;
    }
    if (d != dim_) {
        LOG_INFO(label_.c_str(), "load: dim mismatch (file %d vs store %d)", d, dim_);
        fclose(mf); return false;
    }

    std::vector<SpeakerMeta> new_speakers(ns);
    int total_ex = 0;
    for (int i = 0; i < ns; ++i) {
        int32_t eid, nl, ec, mc;
        if (fread(&eid, 4, 1, mf) != 1 || fread(&nl, 4, 1, mf) != 1) {
            fclose(mf); return false;
        }
        new_speakers[i].external_id = eid;
        if (nl > 0) {
            new_speakers[i].name.resize(nl);
            if ((int)fread(&new_speakers[i].name[0], 1, nl, mf) != nl) {
                fclose(mf); return false;
            }
        }
        if (fread(&ec, 4, 1, mf) != 1 || fread(&mc, 4, 1, mf) != 1) {
            fclose(mf); return false;
        }
        new_speakers[i].exemplar_count = ec;
        new_speakers[i].match_count    = mc;
        total_ex += ec;
    }
    fclose(mf);

    // --- embeddings.bin ---
    std::vector<float> host_emb;
    if (total_ex > 0) {
        std::string emb_path = dir + "/embeddings.bin";
        FILE* ef = fopen(emb_path.c_str(), "rb");
        if (!ef) { LOG_INFO(label_.c_str(), "load: no embeddings.bin"); return false; }
        host_emb.resize(total_ex * dim_);
        size_t read = fread(host_emb.data(), sizeof(float), total_ex * dim_, ef);
        fclose(ef);
        if ((int)read != total_ex * dim_) {
            LOG_INFO(label_.c_str(), "load: embeddings.bin truncated");
            return false;
        }
    }

    // Rebuild state.
    speakers_ = std::move(new_speakers);
    next_id_  = ni;
    n_total_  = total_ex;
    has_pending_ = false;

    id_to_idx_.clear();
    offsets_.clear();
    offsets_.reserve(speakers_.size() + 1);
    int off = 0;
    for (int i = 0; i < (int)speakers_.size(); ++i) {
        offsets_.push_back(off);
        id_to_idx_[speakers_[i].external_id] = i;
        off += speakers_[i].exemplar_count;
    }
    offsets_.push_back(off);

    // Upload to GPU.
    ensure_capacity(total_ex);
    if (total_ex > 0) {
        cudaMemcpyAsync(d_embeddings_, host_emb.data(),
                        (size_t)total_ex * dim_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }
    upload_offsets();
    cudaStreamSynchronize(stream_);

    LOG_INFO(label_.c_str(), "Loaded %d speakers (%d exemplars) from %s",
             (int)speakers_.size(), n_total_, dir.c_str());
    return true;
}

int SpeakerVectorStore::absorb_fragments(float absorption_threshold) {
    // Fragment absorption: iteratively merge speaker pairs with high centroid similarity.
    // Uses mean of all exemplars (centroid) instead of single anchor for robustness.
    // Absorbs smaller (fewer matches) into larger speaker.
    int merges = 0;
    while (true) {
        int dst_id = -1, src_id = -1;
        float best_sim = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            int n_spk = (int)speakers_.size();
            if (n_spk < 2) break;

            // Compute centroid (mean embedding) for each speaker.
            std::vector<std::vector<float>> centroids(n_spk, std::vector<float>(dim_, 0.0f));
            for (int i = 0; i < n_spk; ++i) {
                int start = offsets_[i];
                int end   = offsets_[i + 1];
                int n_ex  = end - start;
                if (n_ex == 0) continue;

                // Download all exemplars for this speaker.
                std::vector<float> buf(n_ex * dim_);
                cudaMemcpy(buf.data(),
                           d_embeddings_ + start * dim_,
                           n_ex * dim_ * sizeof(float),
                           cudaMemcpyDeviceToHost);

                // Compute mean.
                for (int e = 0; e < n_ex; ++e)
                    for (int d = 0; d < dim_; ++d)
                        centroids[i][d] += buf[e * dim_ + d];
                float inv_n = 1.0f / n_ex;
                float norm = 0;
                for (int d = 0; d < dim_; ++d) {
                    centroids[i][d] *= inv_n;
                    norm += centroids[i][d] * centroids[i][d];
                }
                // L2-normalize centroid.
                if (norm > 1e-12f) {
                    float inv_norm = 1.0f / sqrtf(norm);
                    for (int d = 0; d < dim_; ++d)
                        centroids[i][d] *= inv_norm;
                }
            }

            // Find pair with highest mutual centroid similarity.
            for (int i = 0; i < n_spk; ++i) {
                for (int j = i + 1; j < n_spk; ++j) {
                    float sim = 0;
                    for (int d = 0; d < dim_; ++d)
                        sim += centroids[i][d] * centroids[j][d];
                    if (sim > best_sim) {
                        best_sim = sim;
                        // Larger (more matches) becomes destination.
                        if (speakers_[i].match_count >= speakers_[j].match_count) {
                            dst_id = speakers_[i].external_id;
                            src_id = speakers_[j].external_id;
                        } else {
                            dst_id = speakers_[j].external_id;
                            src_id = speakers_[i].external_id;
                        }
                    }
                }
            }
        }
        // Check if we found a pair above threshold (lock released).
        if (best_sim < absorption_threshold || dst_id < 0) break;
        LOG_INFO(label_.c_str(),
                 "Absorb: merge spk%d → spk%d (centroid_sim=%.3f)",
                 src_id, dst_id, best_sim);
        merge_speakers(dst_id, src_id);
        merges++;
    }
    return merges;
}

} // namespace deusridet
