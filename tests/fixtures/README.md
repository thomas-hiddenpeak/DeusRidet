# tests/fixtures — canonical and internal-check-only artefacts

This directory holds two **categorically different** kinds of files.
Read this before using anything here.

## Canonical behavioural reference (use freely)

| File | Role |
|------|------|
| `test_ground_truth.json` | Machine-readable transcript of `tests/test.mp3` (4 speakers · 556 utterances · 3615 s). Pairs `t0_start_sec`/`t0_end_sec`/`speaker`/`text`. |
| `test_ground_truth_v1.jsonl` | Legacy line-oriented form of the same content. |

These pair with `tests/test.mp3` and `tests/test.txt`. Together they
are the **only** acceptable behavioural reference for any sensus /
orator / auditus decision. No discount, no substitution. See
`.github/instructions/benchmarks.instructions.md`.

## Internal-check-only fixtures (do NOT use to set defaults or ship)

| File | Role | Restriction |
|------|------|-------------|
| `fused_v1.bin` + `.speakers.txt` | GT-cut, gate-bypassing embedding dump used by `tools/orator_reclusterer_eval`. | `internal-check-only` |
| `fused_v2_dominant.bin` + `.speakers.txt` | Same, dominant-speaker variant. | `internal-check-only` |
| `orator_reclusterer_ablation*.csv` | Numeric scans derived from the above. | `internal-check-only` |

These files **bypass** the live `awaken` pipeline (no VAD, no dual_db,
no identify/match gates). Numbers computed against them describe a
slice of the system, not the system. They MAY be used for:

- GPU-vs-CPU bit-equality regression
- Kernel micro-benchmarks (latency / throughput)
- One-shot algorithmic sanity

They MUST NOT be used to:

- set or change any default value
- flip an always-on / always-off behaviour
- declare a phase positive or negative
- claim a quality regression or improvement

For any behavioural verdict, use `tools/replay_to_transcript.py`
against a live `awaken` server streaming `tests/test.mp3`, and read
the resulting pairing report directly. See
`.github/instructions/workflow.instructions.md` (Semantic Evaluation,
Not Scripted Scoring).
