#!/usr/bin/env python3
# TODO(native-cuda-port): replace with diarizen_pipeline.cpp once native
# CUDA P1–P3 (S-WavLM-s80-md tap + Conformer EEND head + ResNet34-LM
# embedder + VBx clustering) lands. This Python worker is the only
# remaining philosophy violation tracked in the active codebase. See
# docs/{en,zh}/architecture/12-diarizen.md §"Architectural anchor".
"""Persistent DiariZen-v2 worker (Orator/DiariZen facade backend).

Protocol (one JSON object per line on stdin/stdout; logs to stderr):

  request  -> {"op": "ping"}
  response <- {"ok": true, "pong": true}

  request  -> {"op": "load", "model": "BUT-FIT/diarizen-wavlm-large-s80-md-v2"}
  response <- {"ok": true, "loaded": "<model>", "load_sec": <float>}
              (idempotent; second call is a no-op if same model)

  request  -> {"op": "diarize", "wav": "/abs/path/to/wav.wav"}
  response <- {"ok": true,
               "segments": [[start_sec, end_sec, "label"], ...],
               "wall_clock_sec": <float>}

  request  -> {"op": "shutdown"}
  response <- {"ok": true, "bye": true}   # then process exits 0

Any malformed input -> {"ok": false, "error": "<msg>"} (worker keeps
running so the C++ side can retry).

The worker is intentionally minimal: it owns the model, nothing else.
Speaker→GT alignment, scoring, and amend-broadcast all live on the
C++ side (Orator). This keeps the boundary clean for the eventual
native-CUDA replacement of the WavLM/Conformer/ResNet/VBx stack
(see docs/{en,zh}/architecture/12-diarizen.md, phases P1a–P3c).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

# ---------------------------------------------------------------------------
# Hijack fd 1: many libraries (pyannote, diarizen, tqdm-ish things) call
# print() unconditionally. Our line-protocol cannot tolerate noise on stdout.
# Solution: dup the real stdout to a private fd, then redirect fd 1 onto
# stderr's fd. All library prints land in stderr where the C++ side ignores
# them; protocol replies go out through _PROTO via the saved fd.
# ---------------------------------------------------------------------------
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)  # fd 1 now points where fd 2 points (stderr)
sys.stdout = os.fdopen(1, "w", buffering=1)  # rebind sys.stdout to stderr
_PROTO = os.fdopen(_REAL_STDOUT_FD, "w", buffering=1)


def _log(msg: str) -> None:
    print(f"[diarizen-worker] {msg}", file=sys.stderr, flush=True)


def _reply(obj: dict[str, Any]) -> None:
    _PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO.flush()


# ---------------------------------------------------------------------------
# Lazy heavyweight imports — only when first 'load' arrives. This keeps the
# 'ping' round-trip cheap and lets the C++ side discover process liveness
# without paying torch import time on every spawn.
# ---------------------------------------------------------------------------
_PIPE = None
_LOADED_NAME: str | None = None
_DEVICE = None


def _ensure_loaded(model_name: str) -> tuple[float, bool]:
    """Load DiariZen pipeline (idempotent). Returns (load_sec, did_load)."""
    global _PIPE, _LOADED_NAME, _DEVICE
    if _PIPE is not None and _LOADED_NAME == model_name:
        return 0.0, False
    t0 = time.time()

    import torch  # noqa: PLC0415  — intentional lazy
    _orig_torch_load = torch.load

    def _patched_load(*a, **kw):  # noqa: ANN001
        kw["weights_only"] = False
        return _orig_torch_load(*a, **kw)

    torch.load = _patched_load  # type: ignore[assignment]

    import torchaudio  # noqa: PLC0415
    import soundfile as _sf  # noqa: PLC0415

    def _ta_load(path, *a, **kw):  # noqa: ANN001
        data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _ta_load  # type: ignore[assignment]

    from diarizen.pipelines.inference import DiariZenPipeline  # noqa: PLC0415

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"loading {model_name} on {_DEVICE}")
    pipe = DiariZenPipeline.from_pretrained(model_name).to(_DEVICE)
    _PIPE = pipe
    _LOADED_NAME = model_name
    dt = time.time() - t0
    _log(f"loaded in {dt:.1f}s")
    return dt, True


def _diarize(wav_path: str) -> tuple[list[tuple[float, float, str]], float]:
    if _PIPE is None:
        raise RuntimeError("model not loaded; send {'op':'load',...} first")
    t0 = time.time()
    annot = _PIPE(wav_path)
    segs: list[tuple[float, float, str]] = []
    for turn, _trk, label in annot.itertracks(yield_label=True):
        segs.append((float(turn.start), float(turn.end), str(label)))
    return segs, time.time() - t0


def _handle(req: dict[str, Any]) -> dict[str, Any]:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "pong": True, "pid": os.getpid()}
    if op == "load":
        model = req.get("model", "BUT-FIT/diarizen-wavlm-large-s80-md-v2")
        dt, did = _ensure_loaded(model)
        return {"ok": True, "loaded": model, "load_sec": dt, "fresh": did}
    if op == "diarize":
        wav = req.get("wav")
        if not isinstance(wav, str) or not os.path.isfile(wav):
            return {"ok": False, "error": f"wav not found: {wav!r}"}
        segs, dt = _diarize(wav)
        return {"ok": True, "segments": segs, "wall_clock_sec": dt}
    if op == "shutdown":
        return {"ok": True, "bye": True}
    return {"ok": False, "error": f"unknown op: {op!r}"}


def main() -> int:
    _log(f"started pid={os.getpid()}")
    # Greeting so the C++ side can synchronise on a known token.
    _reply({"ok": True, "ready": True, "pid": os.getpid()})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            _reply({"ok": False, "error": f"json decode: {e}"})
            continue
        try:
            resp = _handle(req)
        except Exception as e:  # noqa: BLE001 — surface to C++ caller
            tb = traceback.format_exc(limit=3)
            _log(f"handler error: {e}\n{tb}")
            _reply({"ok": False, "error": str(e)})
            continue
        _reply(resp)
        if req.get("op") == "shutdown":
            _log("shutdown requested")
            return 0
    _log("stdin closed; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
