/**
 * @file audio_pipeline_diarizen_capture.cpp
 * @philosophical_role Session-level PCM tap for the DiariZen-v2 reclusterer
 *     (Hybrid P1). The live pipeline remains streaming; this file just adds
 *     a side-channel that mirrors push_pcm() into a capped in-RAM buffer and
 *     dumps it to a 16 kHz mono WAV on request, so the offline DiariZen
 *     facade can score the whole session.
 * @serves AudioPipeline::diarizen_capture_* and ::diarizen_dump_wav.
 */
#include "audio_pipeline.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace deusridet {

namespace {

// Hook called from push_pcm() — kept as a free function so we don't pollute
// the hot path with another method-call frame. Inlined at -O2.
inline void diarizen_capture_append(
    std::mutex& mu,
    bool on,
    size_t cap_samples,
    std::vector<int16_t>& buf,
    const int16_t* data,
    int n_samples) {
    if (!on || n_samples <= 0) return;
    std::lock_guard<std::mutex> lk(mu);
    if (buf.size() >= cap_samples) return;
    size_t take = (size_t)n_samples;
    if (buf.size() + take > cap_samples) take = cap_samples - buf.size();
    if (take == 0) return;
    size_t old = buf.size();
    buf.resize(old + take);
    std::memcpy(buf.data() + old, data, take * sizeof(int16_t));
}

}  // namespace

void AudioPipeline::diarizen_capture_enable(bool on, double max_seconds) {
    std::lock_guard<std::mutex> lk(diarizen_capture_mu_);
    diarizen_capture_on_ = on;
    diarizen_capture_on_atomic_.store(on, std::memory_order_release);
    if (max_seconds < 1.0) max_seconds = 1.0;
    // 16 kHz, int16 mono = 2 bytes per sample.
    diarizen_capture_cap_samples_ =
        static_cast<size_t>(max_seconds * 16000.0);
    if (on) {
        diarizen_capture_buf_.reserve(
            std::min<size_t>(diarizen_capture_cap_samples_, 1 << 24));
        // P2: record the stream-time origin of buffer index 0. Reads the
        // ingress counter (push_pcm's only writer; uses release/acquire),
        // which is monotonic and lock-free.
        diarizen_capture_origin_samples_ =
            audio_t1_in_.load(std::memory_order_acquire);
    } else {
        diarizen_capture_buf_.clear();
        diarizen_capture_buf_.shrink_to_fit();
        diarizen_capture_origin_samples_ = 0;
    }
}

bool AudioPipeline::diarizen_capture_enabled() const {
    std::lock_guard<std::mutex> lk(diarizen_capture_mu_);
    return diarizen_capture_on_;
}

size_t AudioPipeline::diarizen_capture_samples() const {
    std::lock_guard<std::mutex> lk(diarizen_capture_mu_);
    return diarizen_capture_buf_.size();
}

void AudioPipeline::diarizen_capture_clear() {
    std::lock_guard<std::mutex> lk(diarizen_capture_mu_);
    diarizen_capture_buf_.clear();
    // Re-anchor origin to the current ingress, since the buffer is now
    // empty and the next sample appended will live at index 0 again.
    diarizen_capture_origin_samples_ =
        audio_t1_in_.load(std::memory_order_acquire);
}

double AudioPipeline::diarizen_capture_origin_sec() const {
    std::lock_guard<std::mutex> lk(diarizen_capture_mu_);
    return diarizen_capture_origin_samples_ / 16000.0;
}

size_t AudioPipeline::diarizen_dump_wav(const std::string& path) const {
    // Snapshot under the lock so push_pcm cannot race against fwrite.
    std::vector<int16_t> snapshot;
    {
        std::lock_guard<std::mutex> lk(diarizen_capture_mu_);
        if (diarizen_capture_buf_.empty()) return 0;
        snapshot = diarizen_capture_buf_;  // O(N) copy; ~125 MB worst case
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return 0;
    uint32_t n = static_cast<uint32_t>(snapshot.size());
    uint32_t data_sz = n * 2;
    uint32_t file_sz = 36 + data_sz;
    uint16_t fmt = 1, ch = 1, ba = 2, bits = 16;
    uint32_t sr = 16000, bps = 32000, fmt_sz = 16;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&file_sz, 4, 1, f);
    std::fwrite("WAVEfmt ", 1, 8, f);
    std::fwrite(&fmt_sz, 4, 1, f);
    std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&ch, 2, 1, f);
    std::fwrite(&sr, 4, 1, f);
    std::fwrite(&bps, 4, 1, f);
    std::fwrite(&ba, 2, 1, f);
    std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_sz, 4, 1, f);
    std::fwrite(snapshot.data(), 2, n, f);
    std::fclose(f);
    return n;
}

// Public hook for audio_pipeline.cpp::push_pcm so capture is appended on
// every successful WS binary frame. Implemented as a member so it can
// touch the private buffer/lock fields directly.
void AudioPipeline::diarizen_capture_tap_(const int16_t* data, int n_samples) {
    // Hot-path short-circuit: avoid the mutex when capture is off.
    if (!diarizen_capture_on_atomic_.load(std::memory_order_acquire)) return;
    diarizen_capture_append(diarizen_capture_mu_,
                            diarizen_capture_on_,
                            diarizen_capture_cap_samples_,
                            diarizen_capture_buf_,
                            data, n_samples);
}

}  // namespace deusridet
