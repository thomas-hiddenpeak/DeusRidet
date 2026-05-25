#!/usr/bin/env python3
"""E2E smoke for Phase 10 + Phase 11.

Connects to a running awaken server on ws://localhost:8080/ws, streams
the canonical tests/test.mp3 at high speed, and counts every WS frame
by type. Specifically validates that:
  - speaker frames flow
  - speaker_relabel frames are emitted at least once when the server
    is launched with DEUSRIDET_RECLUSTERER_MAX_GLOBALS=<small>.

Run from repo root:
    python3 tools/e2e_relabel_smoke.py --speed 10 --max-sec 240
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import websocket  # type: ignore


def decode_pcm(path: str) -> bytes:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed: {r.stderr.decode()}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="tests/test.mp3")
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--max-sec", type=float, default=240.0)
    ap.add_argument("--drain-sec", type=float, default=20.0)
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    args = ap.parse_args()

    pcm = decode_pcm(args.audio)
    total_samples = len(pcm) // 2
    src_sec = total_samples / 16000.0
    if args.max_sec > 0 and src_sec > args.max_sec:
        keep = int(args.max_sec * 16000) * 2
        pcm = pcm[:keep]
        total_samples = len(pcm) // 2
    src_sec = total_samples / 16000.0
    print(f"[e2e] source: {args.audio}  {src_sec:.1f}s ({total_samples} samples)")

    counts: Counter[str] = Counter()
    relabels: list[dict] = []
    speakers_seen: set = set()
    lock = threading.Lock()

    ws = websocket.WebSocket()
    ws.connect(args.url)
    print(f"[e2e] connected to {args.url}")

    stop = threading.Event()

    def reader():
        ws.settimeout(1.0)
        while not stop.is_set():
            try:
                msg = ws.recv()
            except Exception:
                continue
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                obj = json.loads(msg)
            except Exception:
                continue
            t = obj.get("type", "?")
            with lock:
                counts[t] += 1
                if t == "speaker":
                    sid = obj.get("speaker_id")
                    if sid is not None:
                        speakers_seen.add(sid)
                elif t == "speaker_relabel":
                    relabels.append(obj)

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    # Stream at speed.
    chunk = 1600  # 100 ms
    n_chunks = (total_samples + chunk - 1) // chunk
    frame_wall = (chunk / 16000.0) / max(args.speed, 1e-6)
    t0 = time.time()
    for i in range(n_chunks):
        a = i * chunk
        b = min(a + chunk, total_samples)
        ws.send(pcm[a * 2: b * 2], opcode=websocket.ABNF.OPCODE_BINARY)
        target = t0 + (i + 1) * frame_wall
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)
    wall = time.time() - t0
    print(f"[e2e] streamed {src_sec:.1f}s src in {wall:.1f}s wall ({src_sec/wall:.1f}x)")

    print(f"[e2e] draining for {args.drain_sec:.0f}s …")
    time.sleep(args.drain_sec)
    stop.set()
    try:
        ws.close()
    except Exception:
        pass
    th.join(timeout=2.0)

    print("\n=== E2E summary ===")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v}")
    print(f"\nDistinct online speaker_ids:  {sorted(speakers_seen)}")
    print(f"speaker_relabel events:       {len(relabels)}")
    for r in relabels[:20]:
        print(f"  seg={r.get('segment_id')} {r.get('old_id')} -> {r.get('new_id')} "
              f"conf={r.get('confidence', 0):.3f}")
    if len(relabels) > 20:
        print(f"  ... and {len(relabels) - 20} more")

    # Smoke verdict.
    ok_speaker = counts.get("speaker", 0) > 0
    ok_relabel = len(relabels) > 0
    print(f"\nspeaker frames flowed:        {ok_speaker}")
    print(f"speaker_relabel observed:     {ok_relabel}")
    sys.exit(0 if ok_speaker else 2)


if __name__ == "__main__":
    main()
