/**
 * @file diarizen_facade.h
 * @philosophical_role Orator's deferred ear — a transparent gateway to the
 *     DiariZen-v2 reclusterer (WavLM-pruned + 4-layer Conformer EEND head +
 *     WeSpeaker ResNet34-LM + VBx variational HMM). Behind this facade today
 *     sits a persistent Python worker that owns the model weights and runs
 *     them on CUDA via the same code path that produced the 93.5 % live
 *     accuracy verdict on tests/test.mp3. Tomorrow the facade is unchanged
 *     and the worker is replaced, stage by stage, with native CUDA kernels
 *     under src/orator/diarizen_* (see docs/{en,zh}/architecture/12-diarizen.md
 *     phases P1a–P3c).
 * @serves Orator subsystem. The facade is intentionally narrow — one entry
 *     point (`diarize`) and one lifecycle pair (`start`/`shutdown`) — so the
 *     callers never need to know whether the computation is delegated or
 *     native. R3 boundary: no Python types leak across this header.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deusridet::orator {

/// One labelled time interval produced by DiariZen-v2 diarisation. Labels
/// are pipeline-local strings (e.g. "speaker0", "speaker1"); the caller is
/// responsible for mapping them onto persistent global identities.
struct DiarizenSegment {
    double      start_sec = 0.0;
    double      end_sec   = 0.0;
    std::string label;
};

struct DiarizenFacadeConfig {
    /// Absolute path to the Python interpreter that has the diarizen + torch
    /// + pyannote stack installed. Default: the verified py310_diarizen env.
    std::string python_bin =
        "/home/rm01/miniconda3/envs/py310_diarizen/bin/python";
    /// Absolute path to tools/diarizen_worker.py.
    std::string worker_script;  // empty -> auto-derived from cwd
    /// HuggingFace model id, kept identical to the verified verdict.
    std::string model_name =
        "BUT-FIT/diarizen-wavlm-large-s80-md-v2";
    /// Seconds to wait for the initial "ready" greeting from the worker.
    int spawn_timeout_sec = 30;
    /// Seconds to wait for a single diarisation reply. 16 s of audio takes
    /// ~10 s on Orin; a full 10-min file ~740 s. Default leaves slack for
    /// the live test fixture.
    int diarize_timeout_sec = 1500;
};

/// Persistent IPC handle to the DiariZen-v2 Python worker. Non-copyable;
/// move-only.
class DiarizenFacade {
public:
    explicit DiarizenFacade(DiarizenFacadeConfig cfg = {});
    ~DiarizenFacade();

    DiarizenFacade(const DiarizenFacade&) = delete;
    DiarizenFacade& operator=(const DiarizenFacade&) = delete;
    DiarizenFacade(DiarizenFacade&&) noexcept;
    DiarizenFacade& operator=(DiarizenFacade&&) noexcept;

    /// @role Spawn the worker, wait for `ready`, and load the model. Safe
    ///       to call multiple times; subsequent calls are no-ops. Returns
    ///       true on success; on failure, `last_error()` describes why.
    bool start();

    /// @role Run diarisation on a 16 kHz mono WAV file. Returns the list of
    ///       labelled intervals. Empty vector on failure (check
    ///       `last_error()`). Blocking; thread-unsafe (one outstanding
    ///       request per facade instance).
    std::vector<DiarizenSegment> diarize(const std::string& wav_path);

    /// @role Politely shut the worker down and reap it. Idempotent.
    void shutdown() noexcept;

    /// Last diagnostic message; empty when the previous call succeeded.
    const std::string& last_error() const noexcept;

    /// Worker PID (0 if not running). Useful for tests and for the Vigilia
    /// process monitor.
    int worker_pid() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace deusridet::orator
