/**
 * @file diarizen_periodic_worker.cpp
 * @philosophical_role Implementation of the DiariZen-v2 periodic/on-demand
 *     worker (see diarizen_periodic_worker.h).
 * @serves awaken, nexus broadcast layer, transcript holdback.
 */
#include "diarizen_periodic_worker.h"

#include "diarizen_facade.h"
#include "nexus/ws_server.h"
#include "sensus/auditus/audio_pipeline.h"
#include "sensus/auditus/transcript_holdback.h"

#include <chrono>
#include <cstdio>
#include <sstream>
#include <utility>

namespace deusridet::orator {

DiarizenPeriodicWorker::DiarizenPeriodicWorker(
    AudioPipeline& audio,
    DiarizenFacade& facade,
    auditus::TranscriptHoldback& holdback,
    WsServer& server,
    double period_sec,
    std::string wav_dir)
    : audio_(audio),
      facade_(facade),
      holdback_(holdback),
      server_(server),
      period_sec_(period_sec < 5.0 ? 5.0 : period_sec),
      wav_dir_(std::move(wav_dir)) {}

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
    std::ostringstream path_oss;
    path_oss << wav_dir_ << "/diarizen_partial_" << seq << ".wav";
    const std::string wav_path = path_oss.str();

    size_t n = audio_.diarizen_dump_wav(wav_path);
    if (n == 0) {
        std::fprintf(stderr, "[diarizen-worker] dump_wav returned 0 samples; skipping\n");
        return false;
    }

    double origin_sec = audio_.diarizen_capture_origin_sec();

    auto segs = facade_.diarize(wav_path);
    if (segs.empty()) {
        std::fprintf(stderr, "[diarizen-worker] facade.diarize returned empty: %s\n",
                     facade_.last_error().c_str());
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
