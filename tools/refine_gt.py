#!/usr/bin/env python3
"""
refine_gt.py — Refine the coarse Ground Truth (test_ground_truth.json) into
voiced-only, tightly-bounded segments suitable for CAM++ baseline evaluation.

Pipeline:
  1. Decode tests/test.mp3 -> 16kHz mono float32 PCM (ffmpeg, cached in /tmp).
  2. Run build/silero_vad_segments to get all voiced regions of the file.
  3. For each coarse utterance (start, end, speaker):
       - Intersect with voiced regions
       - Drop intersections shorter than --min-seg-ms (default 500 ms)
       - Each surviving piece becomes one refined GT entry
  4. Write tests/fixtures/test_ground_truth_v1.jsonl (one JSON object per line):
       {"idx":N, "start_ms":S, "end_ms":E, "duration_ms":D,
        "speaker":"...", "src_utt_idx":K}

Per Step 2a-2 of the SAAS evaluation plan
(/.github + DEVLOG). This is a one-shot prep tool, not a runtime dep.

Usage:
  python3 tools/refine_gt.py [--min-seg-ms 500]
                             [--vad-threshold 0.5]
                             [--out tests/fixtures/test_ground_truth_v1.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COARSE_GT = ROOT / "tests" / "fixtures" / "test_ground_truth.json"
AUDIO     = ROOT / "tests" / "test.mp3"
PCM_CACHE = Path("/tmp") / "test_mp3_16k_mono.f32"
VAD_BINS  = {
    "silero": ROOT / "build" / "silero_vad_segments",
    "fsmn":   ROOT / "build" / "fsmn_vad_segments",
}


def decode_audio() -> None:
    """Decode test.mp3 -> 16kHz mono float32 PCM (cached)."""
    if PCM_CACHE.exists() and PCM_CACHE.stat().st_size > 0:
        print(f"[decode] Using cached {PCM_CACHE} "
              f"({PCM_CACHE.stat().st_size / 1e6:.1f} MB)")
        return
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found in PATH")
    print(f"[decode] {AUDIO} -> {PCM_CACHE}")
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(AUDIO),
        "-ac", "1", "-ar", "16000",
        "-f", "f32le",
        str(PCM_CACHE),
    ])


def run_vad(kind: str, threshold: float, min_speech_ms: int,
            min_silence_ms: int, pad_ms: int) -> list[dict]:
    if kind not in VAD_BINS:
        sys.exit(f"Unknown --vad {kind}; expected silero|fsmn")
    binp = VAD_BINS[kind]
    if not binp.exists():
        sys.exit(f"VAD binary not built: {binp}\n"
                 f"  cd build && make {binp.name} -j$(nproc)")
    segs_json = Path("/tmp") / f"test_mp3_{kind}_segments.json"
    print(f"[{kind}] running {binp.name} (thr={threshold}, "
          f"min_speech={min_speech_ms}ms, min_silence={min_silence_ms}ms, "
          f"pad={pad_ms}ms)")
    subprocess.check_call([
        str(binp), str(PCM_CACHE), str(segs_json),
        "--threshold",      str(threshold),
        "--min-speech-ms",  str(min_speech_ms),
        "--min-silence-ms", str(min_silence_ms),
        "--pad-ms",         str(pad_ms),
    ])
    with open(segs_json) as f:
        return json.load(f)


def intersect(utt_start: int, utt_end: int,
              segs: list[dict], cursor: int) -> tuple[list[tuple[int,int]], int]:
    """Return list of (s,e) intersections of [utt_start, utt_end] with `segs`.
    `segs` is sorted by start_ms; advance `cursor` for amortised O(N+M)."""
    out: list[tuple[int,int]] = []
    # Rewind cursor while previous segment still ends after utt_start
    while cursor > 0 and segs[cursor - 1]["end_ms"] > utt_start:
        cursor -= 1
    i = cursor
    while i < len(segs) and segs[i]["start_ms"] < utt_end:
        s = max(utt_start, segs[i]["start_ms"])
        e = min(utt_end,   segs[i]["end_ms"])
        if e > s:
            out.append((s, e))
        i += 1
    # Move cursor forward to the first segment fully past this utt
    while cursor < len(segs) and segs[cursor]["end_ms"] <= utt_start:
        cursor += 1
    return out, cursor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vad",            choices=["silero", "fsmn"], default="silero")
    ap.add_argument("--min-seg-ms",     type=int,   default=500)
    ap.add_argument("--vad-threshold",  type=float, default=0.5)
    ap.add_argument("--vad-min-speech-ms",  type=int, default=250)
    ap.add_argument("--vad-min-silence-ms", type=int, default=200)
    ap.add_argument("--vad-pad-ms",         type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.vad == "silero" else f"_{args.vad}"
        args.out = ROOT / "tests" / "fixtures" / f"test_ground_truth_v1{suffix}.jsonl"

    if not COARSE_GT.exists():
        sys.exit(f"Coarse GT not found: {COARSE_GT}\n"
                 f"  python3 tools/parse_ground_truth.py")

    decode_audio()
    segs = run_vad(args.vad, args.vad_threshold, args.vad_min_speech_ms,
                   args.vad_min_silence_ms, args.vad_pad_ms)
    segs.sort(key=lambda s: s["start_ms"])
    print(f"[{args.vad}] {len(segs)} voiced regions, "
          f"total {sum(s['end_ms']-s['start_ms'] for s in segs)/1000:.1f}s")

    coarse = json.loads(COARSE_GT.read_text())
    utts = coarse["utterances"]
    print(f"[gt] {len(utts)} coarse utterances over {coarse['duration_sec']:.1f}s")

    refined = []
    cursor = 0
    dropped_short = 0
    coverage_ms = 0
    for u in utts:
        u_start = int(u["t0_start_sec"] * 1000)
        u_end   = int(u["t0_end_sec"]   * 1000)
        pieces, cursor = intersect(u_start, u_end, segs, cursor)
        for s, e in pieces:
            dur = e - s
            if dur < args.min_seg_ms:
                dropped_short += 1
                continue
            refined.append({
                "idx": len(refined),
                "start_ms": s,
                "end_ms":   e,
                "duration_ms": dur,
                "speaker":  u["speaker"],
                "src_utt_idx": u["idx"],
            })
            coverage_ms += dur

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in refined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-speaker stats
    by_spk: dict[str, list[int]] = {}
    for r in refined:
        by_spk.setdefault(r["speaker"], []).append(r["duration_ms"])
    print(f"\n[refine] kept {len(refined)} segments, "
          f"dropped {dropped_short} short (<{args.min_seg_ms}ms)")
    print(f"[refine] total speech coverage: {coverage_ms/1000:.1f}s")
    print(f"[refine] -> {args.out}")
    print("\nPer-speaker:")
    for spk, durs in sorted(by_spk.items()):
        print(f"  {spk:8s}  segs={len(durs):4d}  "
              f"total={sum(durs)/1000:6.1f}s  "
              f"mean={sum(durs)/len(durs)/1000:5.2f}s  "
              f"min={min(durs)/1000:.2f}s  max={max(durs)/1000:5.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
