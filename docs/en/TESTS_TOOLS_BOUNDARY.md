# Tests vs Tools Boundary

## Purpose
This document defines stable placement rules for files under `tests/` and `tools/` to reduce overlap and keep repository structure predictable.

## Definition

### `tests/`
Use `tests/` for correctness and regression validation code.

Typical content:
- Unit / integration / numerical validation sources (`test_*.cpp`, `test_*.py`)
- Fixed test fixtures and sample inputs under `tests/audio/`
- Speaker/ASR regression test harnesses intended to validate behavior changes

Rules:
- Must have clear pass/fail or measurable validation intent.
- Should be reproducible and versioned with code.
- Should not primarily serve model export/download tasks.

### `tools/`
Use `tools/` for operational utilities and developer workflows.

Typical content:
- Model export/conversion scripts (`export_*`)
- Runtime drivers and inspectors (`test_audio_ws.py`, `timeline_logger.py`)
- Performance probes and kernel micro-benchmarks (`probe_*`, low-level CUDA probes)
- One-off offline processing helpers

Rules:
- Utility-first, not strict pass/fail test suites.
- May depend on external runtime/model environment.
- Avoid committing generated binaries or cache artifacts in `tools/`.

## Migration Applied

This restructuring includes one concrete move to align the boundary:
- `tools/test_overlap_spkid.cpp` -> `tests/test_overlap_spkid.cpp`

And CMake target path was updated accordingly.

## Naming Conventions

- `tests/`: `test_<component>.<ext>`
- `tools/`: verb-driven names (`export_*`, `probe_*`, `compare_*`, `timeline_*`)

## Future Placement Checklist

Before adding a new file:
1. Does it assert correctness (pass/fail) for code changes?
2. Is it a reusable utility/driver without strict test assertions?
3. Does it require external model export/download behavior?

Decision:
- If (1) -> `tests/`
- If (2) or (3) -> `tools/`
