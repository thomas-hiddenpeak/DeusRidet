/**
 * @file src/orator/speaker_vector_store.cu
 * @philosophical_role
 *   On-GPU speaker-embedding store and nearest-neighbour kernel — the substrate of who-is-speaking. Long-term memory about perception itself, not about content.
 * @serves
 *   Orator speaker-identification path fed by wavlm_ecapa_encoder; Auditus tracker (audio_pipeline.cpp) consumes via facade.
 */
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

} // namespace deusridet
