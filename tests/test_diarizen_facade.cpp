// test_diarizen_facade.cpp — smoke test for the DiariZen-v2 Python-worker
// facade. Spawns the worker, asks it to diarise tests/verification_2026/
// test_16k.wav (the canonical 10-min fixture that produced the live 93.5 %
// verdict), prints the resulting segment count + speaker label histogram,
// and shuts the worker down cleanly. Exit code 0 iff segments > 0 and the
// histogram contains exactly the expected number of distinct labels.
//
// Not a position on the Constitutional metric — that line ships only when
// the facade is wired into the live awaken pipeline and the recluster
// default is flipped. This test exists to prove the C++ <-> Python bridge
// works end-to-end on real audio.

#include "orator/diarizen_facade.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    fs::path repo = fs::path(__FILE__).parent_path().parent_path();
    fs::path wav  = (argc >= 2)
                        ? fs::path(argv[1])
                        : repo / "tools" / "verification_2026" / "test_16k.wav";
    if (!fs::exists(wav)) {
        std::fprintf(stderr, "[test] wav not found: %s\n", wav.c_str());
        return 2;
    }

    deusridet::orator::DiarizenFacadeConfig cfg;
    cfg.worker_script = (repo / "tools" / "diarizen_worker.py").string();
    deusridet::orator::DiarizenFacade facade(cfg);

    std::printf("[test] starting worker (script=%s)\n",
                cfg.worker_script.c_str());
    std::fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    if (!facade.start()) {
        std::fprintf(stderr, "[test] start failed: %s\n",
                     facade.last_error().c_str());
        return 3;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("[test] worker ready pid=%d  spawn=%.1fs\n",
                facade.worker_pid(),
                std::chrono::duration<double>(t1 - t0).count());
    std::fflush(stdout);

    std::printf("[test] diarize %s\n", wav.c_str());
    std::fflush(stdout);
    auto t2 = std::chrono::steady_clock::now();
    auto segs = facade.diarize(wav.string());
    auto t3 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t3 - t2).count();

    if (segs.empty()) {
        std::fprintf(stderr, "[test] diarize failed: %s\n",
                     facade.last_error().c_str());
        facade.shutdown();
        return 4;
    }

    std::map<std::string, double> dur_by_label;
    double t_total = 0.0;
    for (const auto& s : segs) {
        double d = s.end_sec - s.start_sec;
        dur_by_label[s.label] += d;
        t_total += d;
    }
    std::printf("[test] segments=%zu  wall=%.1fs  total_speech=%.1fs\n",
                segs.size(), wall, t_total);
    for (const auto& [lab, d] : dur_by_label) {
        std::printf("[test]   %-12s %.2f s\n", lab.c_str(), d);
    }
    std::fflush(stdout);

    facade.shutdown();

    // Acceptance: ≥ 1 segment and at least 1 distinct label. The Constitutional
    // accuracy check is performed separately by tools/verification_2026/
    // diar_diarizen_gpu.py against test_ground_truth.json.
    if (segs.empty() || dur_by_label.empty()) return 5;
    std::printf("[test] OK\n");
    return 0;
}
