---
applyTo: "tests/**,tools/**"
---

# Benchmarks — Acceptance Criteria & Evaluation Rules

## Speaker Separation & Identification (primary benchmark)

This is a **first-class acceptance criterion** for DeusRidet. Goal:
production-ready quality, not a demo.

- **Baseline audio (canonical, non-negotiable)**: `tests/test.mp3`
  (3615 s, 4 speakers: 朱杰, 徐子景, 石一, 唐云峰). This is the
  **only** acceptable behavioural reference. No discount, no
  substitution. Every ship/no-ship decision and every default-value
  change in any sensus / orator / auditus path must cite a live run
  against this audio.
- **Ground truth**: `tests/fixtures/test_ground_truth.json` (machine-
  readable, 556 utterances with `t0_start_sec` / `t0_end_sec` /
  `speaker` / `text`) — primary; `tests/test.txt` — legacy free-form
  transcript, kept for cross-reference.
- **Target**: ≥ **90%** speaker-attribution accuracy across the full recording.
  Short tokens the pipeline marks as unknown (`spk-1` / `?`) are allowed
  and are **not** counted against accuracy — better to abstain than to
  misattribute.
- **Streaming speed**: If the stream rate overwhelms the pipeline (ring
  buffer overflow, KV saturation, dropped ASR segments), **lower the speed**
  (e.g. `--speed 4.0`, `2.0`, `1.0`) until the backend keeps up. Coverage
  matters more than throughput.
- **Evaluation method**: The agent reads the pipeline log **directly** and
  compares against `test.txt` segment by segment. **No grep/awk/Python
  evaluation scripts — direct, human-style reading only.** Non-negotiable
  per owner's explicit instruction.
- **Authority & scope**: Until the 90% target is met, the agent has full
  authority to iterate on thresholds, kernels, VAD tuning, register/match
  logic, scorer weights, diarizer clustering — anything in the audio path.
  Every tuning cycle must be logged in DEVLOG with delta and accuracy impact.
- **Forbidden shortcuts**: Do not raise accuracy by suppressing hard cases
  (always predict most frequent speaker, always abstain). Abstain rate must
  itself stay reasonable — spot-check that abstentions are concentrated on
  genuinely short / overlapping / low-SNR regions.

## Pre-Run Checklist (mandatory)

Before every benchmark invocation:
1. Kill previous processes: `sudo kill -9 $(pgrep -f deusridet) 2>/dev/null`
2. Drop page caches: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
3. Verify clean state.

Skipping these steps produces polluted baselines and meaningless measurements.
This applies to every single invocation, not just the first.

## No Detached / No Half-System Tests (project-wide rule)

Any evaluator that bypasses the live `awaken` pipeline — i.e. that
reads pre-computed embeddings, skips VAD, skips dual_db, skips the
identify/match gates, or otherwise observes only a slice of the real
production path — is **internal-check-only**:

- Allowed uses: GPU-vs-CPU bit-equality regression; kernel
  micro-benchmarks (latency / throughput); one-shot algorithmic
  sanity. The number it produces describes that slice only.
- **Forbidden uses**: setting any default value, flipping any
  always-on/off behaviour, declaring a phase "positive" or
  "negative", or claiming a quality regression / improvement.
- Every such tool must carry an `internal-check-only` banner in its
  header comment and its DEVLOG entries.

Examples currently in the tree (all `internal-check-only`):
`tools/orator_reclusterer_eval*`, `tests/fixtures/fused_v1.bin`,
`tests/fixtures/fused_v2_dominant.bin`, ablation CSVs derived from
them.

The **only** acceptable source of behavioural evidence is a full
`awaken` run streaming `tests/test.mp3` over WS, captured by
`tools/replay_to_transcript.py` and read by the agent.

## Tests vs Tools Boundary

| Directory | Contents | Language |
|-----------|----------|----------|
| `tests/` | Correctness tests + baseline audio (`test.mp3`, `test.txt`) | C++ only |
| `tools/` | Benchmarks, probes, offline evaluators | C++ preferred; Python allowed for offline export/conversion |

Test binaries produced during development (logs, mapped outputs,
`test_output_*.txt`) are **build artifacts**, not source. They must be
gitignored. Never commit run results into the repo — write them to `logs/`
(also gitignored).

## DEVLOG Integration

Every benchmark run that affects the 90% target records in
`docs/{en,zh}/devlog/YYYY-MM-DD.md`:
- Baseline commit hash
- Configuration deltas (thresholds, VAD, kernel flags)
- Observed accuracy (or "abstain rate / misattribution rate" if sub-metric)
- Log file path (if retained) or reason for discard

Never run a benchmark without a DEVLOG entry. Silent benchmarks are noise.
