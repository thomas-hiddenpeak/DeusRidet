/**
 * @file src/sensus/auditus/retro_full_ring.h
 * @philosophical_role
 *   Step 17a — Memoria of the recent past, used when the future has just
 *   clarified itself. A small ring of recent FULL identifications cached
 *   so that, the moment a new cluster is born from the pending pool, we
 *   can revisit segments that were emitted as abstain or as borderline
 *   matches against the wrong neighbour and surface them as retroactive
 *   re-label candidates.
 * @serves
 *   Sensus auditus — diagnostic-only in 17a; promoted to live amend in 17b.
 */
#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace deusridet {

struct RetroFullSlot {
    int64_t audio_end_samples = 0;     // segment end in 16 kHz sample count
    int     decided_id        = -1;    // speaker id finally emitted (-1 = abstain)
    float   decided_sim       = 0.0f;
    bool    abstained         = false;
    std::vector<float> embedding;      // 384D dual or 192D single, L2-normalised
};

class RetroFullRing {
public:
    static constexpr int kSize = 16;

    void push(RetroFullSlot s) {
        ring_[pos_] = std::move(s);
        pos_ = (pos_ + 1) % kSize;
    }

    // Visit slots in chronological order (oldest first), skipping empty ones.
    template <typename F>
    void for_each(F&& fn) const {
        for (int i = 0; i < kSize; ++i) {
            int idx = (pos_ + i) % kSize;
            if (!ring_[idx].embedding.empty()) fn(ring_[idx]);
        }
    }

private:
    std::array<RetroFullSlot, kSize> ring_{};
    int pos_ = 0;
};

} // namespace deusridet
