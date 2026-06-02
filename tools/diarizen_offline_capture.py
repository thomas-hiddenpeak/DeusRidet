#!/usr/bin/env python3
"""
diarizen_offline_capture.py — Mechanical OFFLINE direct-feed capture.

PURPOSE
    The DiariZen full-session reclusterer ("offline processing") consumes
    the WHOLE captured audio at once and replies with
    ``speaker_diarize_final``. Offline processing has NO replay-rate: we
    feed the entire audio as fast as the socket accepts (the capture
    buffer is appended in push_pcm() *before* the online ring buffer, so
    feeding fast cannot drop capture samples), then send
    ``diarizen_finalize`` and WAIT for the single final pass to complete.
    This is the correct offline methodology — NOT the WS speed-paced
    streaming that the deleted tools/diarizen_live_score.py used.

STRICTLY MECHANICAL — NO SCORING.
    This tool only: decodes audio, feeds it, triggers finalize, captures
    the returned segments, time-aligns them against the ground-truth
    utterances, and renders a human-readable report. It deliberately does
    NOT compute accuracy / macro-F1 / micro / DER / any "auto-judged
    correctness number". Per .github/instructions/benchmarks.md and
    workflow.instructions.md, the agent reads the produced report
    top-to-bottom and judges speaker-attribution correctness by eye.
    Emitting a percentage here re-introduces the exact violation that
    deleted diarizen_live_score.py + compute_accuracy.py on 2026-06-02.

USAGE (with awaken already running on :8080, diarizen enabled by default):
    python3 tools/diarizen_offline_capture.py \\
        --audio tests/test.mp3 \\
        --gt    tests/fixtures/test_ground_truth.json \\
        --out-dir runs/$(date +%Y%m%dT%H%M%S)_diarizen_offline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import websocket  # type: ignore


def decode_to_pcm(path: str) -> bytes:
    """Decode any audio file to s16le 16 kHz mono PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"ffmpeg error: {r.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    return r.stdout


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def fmt_t(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="tests/test.mp3")
    ap.add_argument("--gt", default="tests/fixtures/test_ground_truth.json")
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    ap.add_argument("--chunk", type=int, default=16000,
                    help="samples per binary frame (1s @16k). Fed with no "
                         "real-time pacing — offline has no replay rate.")
    ap.add_argument("--max-sec", type=float, default=0.0,
                    help="0 = feed the whole file (offline default).")
    ap.add_argument("--finalize-timeout", type=float, default=1800.0,
                    help="seconds to wait for the single final pass.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm = decode_to_pcm(args.audio)
    total_samples = len(pcm) // 2
    if args.max_sec > 0:
        total_samples = min(total_samples, int(args.max_sec * 16000))
    audio_sec = total_samples / 16000.0
    print(f"[offline] decoded {audio_sec:.1f}s of audio "
          f"({total_samples} samples)")

    ws = websocket.create_connection(args.url, max_size=None)
    print(f"[offline] connected {args.url}")

    # ── Feed the ENTIRE audio as fast as the socket accepts (no pacing). ──
    t_feed0 = time.time()
    chunk = args.chunk
    sent = 0
    while sent < total_samples:
        take = min(chunk, total_samples - sent)
        ws.send_binary(pcm[sent * 2:(sent + take) * 2])
        sent += take
    t_feed1 = time.time()
    print(f"[offline] fed {sent} samples in {t_feed1 - t_feed0:.1f}s wall "
          f"(no replay pacing)")

    # Give the ingress a moment to drain into the capture buffer, then
    # trigger the single offline full-session pass.
    time.sleep(2.0)
    ws.send("diarizen_finalize")
    print("[offline] sent diarizen_finalize; waiting for "
          "speaker_diarize_final ...")

    # ── Wait for the final pass. ──
    final = None
    ws.settimeout(5.0)
    deadline = time.time() + args.finalize_timeout
    raw_path = out_dir / "raw_messages.jsonl"
    with raw_path.open("w") as rawf:
        while time.time() < deadline:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:  # connection closed etc.
                print(f"[offline] recv error: {e}", file=sys.stderr)
                break
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                obj = json.loads(msg)
            except Exception:
                continue
            t = obj.get("type")
            if t in ("speaker_diarize_final", "speaker_diarize_partial",
                     "speaker_diarize_progress"):
                rawf.write(msg + "\n")
            if t == "speaker_diarize_progress":
                print(f"[offline] progress: {obj.get('status')} "
                      f"samples={obj.get('samples')} sec={obj.get('sec')}")
            if t == "speaker_diarize_final":
                if obj.get("ok"):
                    final = obj
                    break
                else:
                    print(f"[offline] finalize FAILED: {obj.get('error')}",
                          file=sys.stderr)
                    ws.close()
                    return 2
    try:
        ws.close()
    except Exception:
        pass

    if final is None:
        print("[offline] TIMEOUT — no speaker_diarize_final received",
              file=sys.stderr)
        return 3

    segs = final.get("segments", [])
    print(f"[offline] final pass: n_segments={len(segs)} "
          f"audio_sec={final.get('audio_sec')} "
          f"wall_sec={final.get('wall_sec')}")
    (out_dir / "speaker_diarize_final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2))

    # ── Load GT utterances. ──
    gt = json.loads(Path(args.gt).read_text())
    utts = [u for u in gt.get("utterances", [])
            if float(u.get("t0_start_sec", 0)) < audio_sec + 5.0]

    # Normalise segments to (start, end, label).
    norm = []
    for s in segs:
        if isinstance(s, list) and len(s) >= 3:
            norm.append((float(s[0]), float(s[1]), str(s[2])))
        elif isinstance(s, dict):
            norm.append((float(s.get("start_sec", s.get("start", 0))),
                         float(s.get("end_sec", s.get("end", 0))),
                         str(s.get("label", s.get("speaker", "?")))))
    norm.sort(key=lambda x: x[0])

    # ── DISPLAY-ONLY label → GT-name hint (NOT a score). ──
    # For readability we note, per offline label, which GT speaker it most
    # often overlaps. This is a naming aid; the agent still judges by eye.
    from collections import defaultdict
    label_overlap: dict = defaultdict(lambda: defaultdict(float))
    for (s0, s1, lab) in norm:
        for u in utts:
            ov = overlap_sec(s0, s1, float(u["t0_start_sec"]),
                             float(u["t0_end_sec"]))
            if ov > 0:
                label_overlap[lab][u["speaker"]] += ov
    label_hint = {}
    for lab, d in label_overlap.items():
        label_hint[lab] = max(d.items(), key=lambda kv: kv[1])[0] if d else "?"

    # ── Render the eyes-on report. ──
    rep = out_dir / "offline_report.md"
    with rep.open("w") as f:
        f.write("# DiariZen OFFLINE direct-feed capture\n\n")
        f.write(f"- audio: `{args.audio}` ({audio_sec:.1f}s, fed with NO "
                f"replay pacing)\n")
        f.write(f"- final pass wall: {final.get('wall_sec')}s, "
                f"segments: {len(norm)}\n")
        f.write(f"- distinct offline labels: "
                f"{sorted({l for _,_,l in norm})}\n")
        f.write(f"- GT speakers: {gt.get('speakers')}\n\n")
        f.write("> NO accuracy percentage is computed. Read the alignment\n"
                "> below by eye and judge identify / register / distinguish.\n\n")

        f.write("## Offline label → GT-name hint (display only, NOT scored)\n\n")
        for lab in sorted(label_hint):
            f.write(f"- `{lab}` most-often overlaps GT **{label_hint[lab]}**\n")
        f.write("\n")

        # Segment-oriented chronological alignment.
        f.write("## Diarized segments ↔ GT (chronological)\n\n")
        f.write("Each line: offline segment window, its label (≈hint), and\n"
                "the GT utterance(s) it overlaps. No ✓/✗ — agent judges.\n\n")
        for (s0, s1, lab) in norm:
            hits = []
            for u in utts:
                ov = overlap_sec(s0, s1, float(u["t0_start_sec"]),
                                 float(u["t0_end_sec"]))
                if ov >= 0.2:
                    hits.append((ov, u))
            hits.sort(key=lambda x: -x[0])
            gt_desc = ", ".join(f"{u['speaker']}" for _, u in hits[:3]) or "—"
            f.write(f"- [{fmt_t(s0)} – {fmt_t(s1)}]  `{lab}`"
                    f" (≈{label_hint.get(lab,'?')})  GT={gt_desc}\n")
        f.write("\n")

        # GT-oriented narrative.
        f.write("## GT-oriented narrative\n\n")
        f.write("For each GT utterance: speaker, window, text, then the\n"
                "offline labels overlapping it.\n\n")
        for u in utts:
            us = float(u["t0_start_sec"]); ue = float(u["t0_end_sec"])
            txt = (u.get("text", "") or "").replace("\n", " ").strip()
            f.write(f"### [{fmt_t(us)} – {fmt_t(ue)}]  GT={u['speaker']}\n\n")
            f.write(f"> {txt}\n\n")
            labs = []
            for (s0, s1, lab) in norm:
                ov = overlap_sec(us, ue, s0, s1)
                if ov >= 0.2:
                    labs.append((ov, lab, s0, s1))
            labs.sort(key=lambda x: -x[0])
            if not labs:
                f.write("_(no offline segment overlaps this utterance)_\n\n")
                continue
            for ov, lab, s0, s1 in labs:
                f.write(f"- `{lab}` (≈{label_hint.get(lab,'?')})  "
                        f"[{fmt_t(s0)} – {fmt_t(s1)}]  overlap={ov:.1f}s\n")
            f.write("\n")

    print(f"[offline] report written: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
