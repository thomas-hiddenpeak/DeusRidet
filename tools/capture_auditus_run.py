#!/usr/bin/env python3
"""Replay an audio file through a running DeusRidet instance and capture
every asr_transcript event into a hypothesis JSONL (Step 15b).

Assumes the awaken server is already up on ws://localhost:8080/ws. The
audio is decoded to 16 kHz mono int16 PCM via ffmpeg and streamed in
512-sample frames at a configurable speed multiplier (`--speed`). The
server broadcasts asr_transcript JSON messages whose stream_start_sec /
stream_end_sec are derived from source-audio sample counts, so they are
already source-audio seconds -- i.e. directly comparable to the T0
fields in tests/fixtures/test_ground_truth.json regardless of replay
speed.

Part of the Auditus Evaluation Harness. Conversion-only: this tool does
NOT compute any metrics. The paired hyp+gt timeline is read directly by
the agent in Step 15d.

Output schema (runs/<ts>/hyp.jsonl, one record per line):
    {
      "t0_start_sec":        <float>,   # stream_start_sec from broadcast
      "t0_end_sec":          <float>,   # stream_end_sec
      "speaker_id":          <int>,     # CAM++ / WL-ECAPA speaker id (-1 = none)
      "speaker_name":        <str>,     # auto-assigned or registered name
      "speaker_sim":         <float>,   # cosine similarity to best match
      "speaker_confidence":  <float>,
      "speaker_source":      <str>,     # e.g. "camxx_full", "wlecapa_early"
      "trigger":             <str>,     # "vad_end" / "force_flush" / ...
      "text":                <str>,
      "latency_ms":          <float>,
      "recv_wall_sec":       <float>    # time since capture start (debug)
    }

Usage:
    python3 tools/capture_auditus_run.py tests/test.mp3 \\
            --speed 4.0 --out runs/$(date +%Y%m%dT%H%M%S)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import websocket  # type: ignore


def decode_to_pcm(path: str) -> bytes:
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


class Capture:
    def __init__(self, hyp_path: Path, raw_path: Path):
        self.hyp_fh = hyp_path.open("w", encoding="utf-8")
        self.raw_fh = raw_path.open("w", encoding="utf-8")
        self.t0_wall = time.time()
        self.n_transcripts = 0
        self.n_raw = 0
        self.lock = threading.Lock()

    def close(self) -> None:
        with self.lock:
            self.hyp_fh.close()
            self.raw_fh.close()

    def on_text(self, raw: str) -> None:
        with self.lock:
            self.raw_fh.write(raw.rstrip("\n") + "\n")
            self.raw_fh.flush()
            self.n_raw += 1
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if data.get("type") != "asr_transcript":
            return
        rec = {
            "t0_start_sec":       float(data.get("stream_start_sec", 0.0)),
            "t0_end_sec":         float(data.get("stream_end_sec", 0.0)),
            "speaker_id":         int(data.get("speaker_id", -1)),
            "speaker_name":       data.get("speaker_name", ""),
            "speaker_sim":        float(data.get("speaker_sim", 0.0)),
            "speaker_confidence": float(data.get("speaker_confidence", 0.0)),
            "speaker_source":     data.get("speaker_source", ""),
            "trigger":            data.get("trigger", ""),
            "text":               data.get("text", ""),
            "latency_ms":         float(data.get("latency_ms", 0.0)),
            "recv_wall_sec":      round(time.time() - self.t0_wall, 3),
        }
        with self.lock:
            self.hyp_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.hyp_fh.flush()
            self.n_transcripts += 1
            n = self.n_transcripts
        print(
            f"  [{n:4d}] {rec['t0_start_sec']:7.2f}-{rec['t0_end_sec']:7.2f}s "
            f"spk={rec['speaker_name'] or f'spk{rec[\"speaker_id\"]}':<10} "
            f"src={rec['speaker_source']:<14} {rec['text']}",
            flush=True,
        )


def stream(ws: websocket.WebSocketApp, pcm: bytes, chunk: int, speed: float) -> None:
    """Feed PCM frames to the server at `speed` multiplier."""
    # Each 16-bit sample = 2 bytes. `chunk` counts samples.
    total_samples = len(pcm) // 2
    n_chunks = (total_samples + chunk - 1) // chunk
    frame_wall_s = (chunk / 16000.0) / max(speed, 1e-6)
    t_start = time.time()
    for i in range(n_chunks):
        sbeg = i * chunk
        send = min(sbeg + chunk, total_samples)
        frame = pcm[sbeg * 2: send * 2]
        try:
            ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(f"WS send failed at chunk {i}: {e}", file=sys.stderr)
            return
        # Pace the stream. t_target is the wall time by which chunk (i+1)
        # should have been sent. If we're ahead, sleep; if behind, do not
        # sleep (the server will see a faster-than-configured stream, but
        # ring buffer plus replay_speed still tracks T0 correctly).
        t_target = t_start + (i + 1) * frame_wall_s
        dt = t_target - time.time()
        if dt > 0:
            time.sleep(dt)
    print(
        f"--- finished feeding {total_samples} samples "
        f"({total_samples/16000.0:.1f}s source audio) in "
        f"{time.time()-t_start:.1f}s wall ---",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="source audio file (mp3/wav/...)")
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier (default 1.0 = real-time)")
    ap.add_argument("--chunk", type=int, default=512,
                    help="samples per WS frame (default 512 = 32ms)")
    ap.add_argument("--out", required=True,
                    help="output directory (will be created)")
    ap.add_argument("--drain-sec", type=float, default=15.0,
                    help="seconds to wait after sending to drain tail ASR jobs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Decoding {args.audio} ...", flush=True)
    pcm = decode_to_pcm(args.audio)
    n = len(pcm) // 2
    dur = n / 16000.0
    eta_wall = dur / max(args.speed, 1e-6) + args.drain_sec
    print(
        f"  {n} samples ({dur:.1f}s); speed={args.speed}x; "
        f"ETA wall ~{eta_wall:.0f}s (+ drain {args.drain_sec}s)",
        flush=True,
    )

    hyp_path = out / "hyp.jsonl"
    raw_path = out / "raw_events.jsonl"
    cap = Capture(hyp_path, raw_path)

    meta = {
        "source_audio":     os.fspath(args.audio),
        "duration_sec":     dur,
        "speed":            args.speed,
        "chunk_samples":    args.chunk,
        "ws_url":           args.url,
        "started_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    connected = threading.Event()
    closed    = threading.Event()

    def on_open(ws):
        print("--- WS connected ---", flush=True)
        connected.set()

    def on_message(ws, message):
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8", errors="replace")
            except Exception:
                return
        cap.on_text(message)

    def on_error(ws, error):
        print(f"WS error: {error}", file=sys.stderr, flush=True)

    def on_close(ws, code, reason):
        print(f"--- WS closed ({code}) ---", flush=True)
        closed.set()

    ws = websocket.WebSocketApp(
        args.url,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close,
    )
    t_ws = threading.Thread(target=ws.run_forever, daemon=True)
    t_ws.start()

    if not connected.wait(timeout=10):
        print("ERROR: WS did not connect within 10s", file=sys.stderr)
        return 2

    try:
        stream(ws, pcm, args.chunk, args.speed)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)

    # Drain tail: the last ASR segment may still be in flight.
    if args.drain_sec > 0:
        print(f"--- draining {args.drain_sec:.1f}s for tail ASR ---", flush=True)
        time.sleep(args.drain_sec)

    try:
        ws.close()
    except Exception:
        pass
    closed.wait(timeout=5)
    cap.close()

    print(
        f"\nCaptured {cap.n_transcripts} asr_transcript events "
        f"({cap.n_raw} raw messages).\n"
        f"  hyp:  {hyp_path}\n"
        f"  raw:  {raw_path}\n"
        f"  meta: {out / 'run_meta.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
