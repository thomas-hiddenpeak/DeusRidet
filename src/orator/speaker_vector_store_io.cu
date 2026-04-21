/**
 * @file src/orator/speaker_vector_store_io.cu
 * @philosophical_role
 *   Persistence for the speaker store — the memory that survives restarts.
 *   Split from speaker_vector_store.cu under R1 800-line hard cap.
 * @serves
 *   Orator identification path; awaken/shutdown ritual.
 */
// speaker_vector_store_io.cu — peer TU of speaker_vector_store.cu.
//
// Contains the Persistence section (save/load in-place below clear).

#include "speaker_vector_store.h"
#include "../communis/log.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sys/stat.h>
#include <cerrno>

namespace deusridet {

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

int SpeakerVectorStore::absorb_fragments(float absorption_threshold, int max_minor_matches) {
    // Fragment absorption: iteratively merge speaker pairs with high centroid similarity.
    // Uses mean of all exemplars (centroid) instead of single anchor for robustness.
    // Absorbs smaller (fewer matches) into larger speaker.
    // If max_minor_matches > 0, only merge if the smaller speaker has
    // <= max_minor_matches matches (prevents merging established speakers).
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
                    // Guard: skip if smaller speaker is well-established.
                    int minor_count = std::min(speakers_[i].match_count,
                                               speakers_[j].match_count);
                    if (max_minor_matches > 0 && minor_count > max_minor_matches)
                        continue;

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
