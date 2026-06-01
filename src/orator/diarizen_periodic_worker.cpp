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
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace deusridet::orator {

DiarizenPeriodicWorker::DiarizenPeriodicWorker(
    AudioPipeline& audio,
    DiarizenPipeline& pipeline,
    auditus::TranscriptHoldback* holdback,
    WsServer& server,
    double period_sec)
    : audio_(audio),
      pipeline_(pipeline),
      holdback_(holdback),
      server_(server),
      period_sec_(period_sec < 5.0 ? 5.0 : period_sec) {
    if (const char* e = std::getenv("DEUSRIDET_DIARIZEN_PERIODIC")) {
        periodic_enabled_ = (std::string(e) == "1");
    }
}

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
    // freshly-relabelled speaker_id. With no holdback (audio-only
    // session) the final broadcast is the whole contribution.
    run_one_pass_(/*is_final=*/true);
    if (holdback_) holdback_->drain_now();
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
        // Skip bare periodic wakeups unless the timed cadence is explicitly
        // re-enabled. Full-session re-diarise on the shared GPU starves the
        // live perception pipeline (see periodic_enabled_ in the header) —
        // the default path only diarises on an explicit trigger / finalize.
        if (was_triggered || periodic_enabled_) {
            // Only pay the diarisation cost when there is actual audio to
            // diarise — first pass on an empty buffer wastes ~30 s on Orin.
            size_t samples = audio_.diarizen_capture_samples();
            if (samples >= 16000 * 8) {  // need ≥ 8 s of audio
                run_one_pass_(/*is_final=*/false);
            }
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

    auto t0 = std::chrono::steady_clock::now();
    auto segs = pipeline_.diarize(pcm.data(), (int)pcm.size());
    auto t1 = std::chrono::steady_clock::now();
    double wall_sec = std::chrono::duration<double>(t1 - t0).count();
    if (segs.empty()) {
        std::fprintf(stderr, "[diarizen-worker] pipeline.diarize returned empty: %s\n",
                     pipeline_.last_error().c_str());
        // The score client / WebUI must still get a terminal reply on a final
        // pass, otherwise they hang waiting for `speaker_diarize_final`.
        if (is_final) {
            std::string err = pipeline_.last_error();
            std::string j = std::string("{\"type\":\"speaker_diarize_final\",\"ok\":false,\"error\":\"")
                          + (err.empty() ? "no segments" : err) + "\"}";
            server_.broadcast_text(j);
        }
        return false;
    }

    size_t changed = holdback_ ? holdback_->apply_diarization(segs, origin_sec) : 0;
    std::fprintf(stderr,
                 "[diarizen-worker] pass=%llu segs=%zu changed_pending=%zu origin=%.2fs final=%d\n",
                 (unsigned long long)seq, segs.size(), changed, origin_sec, (int)is_final);

    // Broadcast WS message. Use the array-segment schema understood by BOTH
    // the live score client (tools/diarizen_live_score.py expects `ok` +
    // `segments` as [start,end,label]) AND the WebUI panel (which accepts
    // either array or object segments and reads `segment_count`/`pass`/
    // `origin_sec`/`changed_pending`).
    std::ostringstream js;
    js << "{\"type\":\"" << (is_final ? "speaker_diarize_final" : "speaker_diarize_partial")
       << "\",\"ok\":true"
       << ",\"pass\":" << seq
       << ",\"origin_sec\":" << origin_sec
       << ",\"audio_sec\":" << ((double)n / 16000.0)
       << ",\"wall_sec\":" << wall_sec
       << ",\"segment_count\":" << segs.size()
       << ",\"n_segments\":" << segs.size()
       << ",\"changed_pending\":" << changed
       << ",\"segments\":[";
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];
        if (i) js << ',';
        js << '[' << (s.start_sec + origin_sec)
           << ',' << (s.end_sec   + origin_sec)
           << ",\"" << s.label << "\"]";
    }
    js << "]}";
    server_.broadcast_text(js.str());
    return true;
}

}  // namespace deusridet::orator
