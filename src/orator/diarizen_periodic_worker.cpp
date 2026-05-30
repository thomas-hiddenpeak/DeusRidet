/**
 * @file diarizen_periodic_worker.cpp
 * @philosophical_role Implementation of the DiariZen-v2 periodic/on-demand
 *     worker (see diarizen_periodic_worker.h).
 * @serves awaken, nexus broadcast layer, transcript holdback.
 */
#include "diarizen_periodic_worker.h"

#include "diarizen_pipeline.h"
#include "nexus/ws_server.h"
#include "sensus/auditus/audio_pipeline.h"
#include "sensus/auditus/transcript_holdback.h"

#include <chrono>
#include <cstdio>
#include <sstream>
#include <utility>
#include <vector>

namespace deusridet::orator {

DiarizenPeriodicWorker::DiarizenPeriodicWorker(
    AudioPipeline& audio,
    DiarizenPipeline& pipeline,
    auditus::TranscriptHoldback& holdback,
    WsServer& server,
    double period_sec)
    : audio_(audio),
      pipeline_(pipeline),
      holdback_(holdback),
      server_(server),
      period_sec_(period_sec < 5.0 ? 5.0 : period_sec) {}

DiarizenPeriodicWorker::~DiarizenPeriodicWorker() { stop(); }

void DiarizenPeriodicWorker::start() {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return;
    running_ = true;
    stop_req_ = false;
    trigger_req_ = false;
    th_ = std::thread(&DiarizenPeriodicWorker::worker_loop_, this);
}

void DiarizenPeriodicWorker::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!running_) return;
        stop_req_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
}

void DiarizenPeriodicWorker::trigger_async() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        trigger_req_ = true;
    }
    cv_.notify_all();
}

void DiarizenPeriodicWorker::finalize() {
    // One final pass, broadcast as `speaker_diarize_final`, then drain
    // the holdback so the LLM sees every pending transcript with the
    // freshly-relabelled speaker_id.
    run_one_pass_(/*is_final=*/true);
    holdback_.drain_now();
    stop();
}

void DiarizenPeriodicWorker::worker_loop_() {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> lk(mu_);
    auto period = std::chrono::duration<double>(period_sec_);
    while (!stop_req_) {
        cv_.wait_for(lk, period, [this] { return stop_req_ || trigger_req_; });
        if (stop_req_) break;
        bool was_triggered = trigger_req_;
        trigger_req_ = false;
        lk.unlock();
        // Only pay the diarisation cost when there is actual audio to
        // diarise — first pass on an empty buffer wastes ~30 s on Orin.
        size_t samples = audio_.diarizen_capture_samples();
        if (samples >= 16000 * 8) {  // need ≥ 8 s of audio
            (void)was_triggered;
            run_one_pass_(/*is_final=*/false);
        }
        lk.lock();
    }
}

bool DiarizenPeriodicWorker::run_one_pass_(bool is_final) {
    const uint64_t seq = pass_seq_.fetch_add(1, std::memory_order_relaxed);

    std::vector<float> pcm;
    size_t n = audio_.diarizen_copy_pcm_f32(pcm);
    if (n == 0) {
        std::fprintf(stderr, "[diarizen-worker] copy_pcm_f32 returned 0 samples; skipping\n");
        return false;
    }

    double origin_sec = audio_.diarizen_capture_origin_sec();

    auto segs = pipeline_.diarize(pcm.data(), (int)pcm.size());
    if (segs.empty()) {
        std::fprintf(stderr, "[diarizen-worker] pipeline.diarize returned empty: %s\n",
                     pipeline_.last_error().c_str());
        return false;
    }

    size_t changed = holdback_.apply_diarization(segs, origin_sec);
    std::fprintf(stderr,
                 "[diarizen-worker] pass=%llu segs=%zu changed_pending=%zu origin=%.2fs final=%d\n",
                 (unsigned long long)seq, segs.size(), changed, origin_sec, (int)is_final);

    // Broadcast WS message (matches the P1 finalize format; only `type`
    // differs for the periodic case).
    std::ostringstream js;
    js << "{\"type\":\"" << (is_final ? "speaker_diarize_final" : "speaker_diarize_partial")
       << "\",\"pass\":" << seq
       << ",\"origin_sec\":" << origin_sec
       << ",\"segment_count\":" << segs.size()
       << ",\"changed_pending\":" << changed
       << ",\"segments\":[";
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];
        if (i) js << ',';
        js << "{\"start\":" << (s.start_sec + origin_sec)
           << ",\"end\":"   << (s.end_sec   + origin_sec)
           << ",\"label\":\"" << s.label << "\"}";
    }
    js << "]}";
    server_.broadcast_text(js.str());
    return true;
}

}  // namespace deusridet::orator
