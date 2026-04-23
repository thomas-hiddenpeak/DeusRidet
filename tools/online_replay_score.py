#!/usr/bin/env python3
"""
online_replay_score.py — Step 5 sanity check.

Streams a source audio file (default tests/test.mp3) into a running
awaken server over WebSocket, captures every ``{"type":"speaker",...}``
broadcast in the order they arrive, and reports how well that order of
predicted cluster IDs matches the ground-truth segment order under a
first-seen cluster → speaker mapping.

Purpose: Step 3 and Step 4 were pure offline simulators built on
pre-extracted CAM++ embeddings. Step 5 is the 2-pp reality check — we
want to know whether the live runtime (Silero VAD + CAM++ FULL path
with all its exemplar/hit-ratio/margin gates) produces a per-segment
speaker sequence whose macro accuracy is within a couple of points of
the offline 'baseline' configuration.

Assumptions (documented, not tuned):
  1. The runtime emits exactly one ``speaker`` broadcast per speech
     segment on the SAAS_FULL path. If runtime produces N segments and
     GT has M, we compare the first min(N,M) and print a length delta.
  2. We scale audio replay by --speed so the test completes faster;
     server-side stream_start_sec is source-audio seconds regardless
     of replay speed, but we do not rely on timestamps — just order.
  3. We run against the default runtime config in configs/auditus.conf
     (baseline upper-layer gates). Changing the runtime config to match
     Step 3 'all_off' would require an auditus.conf override plus a
     restart — out of scope for this initial sanity check.

Usage:
    python3 tools/online_replay_score.py \\
        --audio tests/test.mp3 \\
        --gt tests/fixtures/test_ground_truth_v1.jsonl \\
        --speed 4.0 --drain-sec 15 \\
        --out-dir runs/online_$(date +%Y%m%dT%H%M%S)

The awaken server must already be running on ws://localhost:8080/ws.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

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


def stream_pcm(ws, pcm: bytes, chunk: int, speed: float) -> None:
    total = len(pcm) // 2
    n_chunks = (total + chunk - 1) // chunk
    frame_wall = (chunk / 16000.0) / max(speed, 1e-6)
    t0 = time.time()
    for i in range(n_chunks):
        a = i * chunk
        b = min(a + chunk, total)
        frame = pcm[a * 2: b * 2]
        ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
        target = t0 + (i + 1) * frame_wall
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)
    print(
        f"[stream] sent {total} samples ({total/16000.0:.1f}s src) "
        f"in {time.time()-t0:.1f}s wall",
        flush=True,
    )


def score_sequence(pred_clusters, gt_speakers):
    """First-seen cluster → speaker mapping, per-speaker macro accuracy.
    Lengths may differ; we compare aligned prefix."""
    n = min(len(pred_clusters), len(gt_speakers))
    mapping = {}
    for i in range(n):
        c = pred_clusters[i]
        if c < 0:
            continue
        mapping.setdefault(c, gt_speakers[i])

    speakers = sorted(set(gt_speakers))
    per_spk_correct = {s: 0 for s in speakers}
    per_spk_total   = {s: 0 for s in speakers}
    for i in range(n):
        true = gt_speakers[i]
        per_spk_total[true] += 1
        c = pred_clusters[i]
        pred_name = mapping.get(c, "__unk__") if c >= 0 else "__unk__"
        if pred_name == true:
            per_spk_correct[true] += 1

    per_spk_acc = {
        s: (per_spk_correct[s] / per_spk_total[s])
        if per_spk_total[s] else 0.0
        for s in speakers
    }
    macro = sum(per_spk_acc.values()) / max(1, len(per_spk_acc))
    micro = sum(per_spk_correct.values()) / max(1, sum(per_spk_total.values()))
    return {
        "macro": macro, "micro": micro, "per_spk": per_spk_acc,
        "n_aligned": n, "n_pred": len(pred_clusters), "n_gt": len(gt_speakers),
        "mapping": mapping,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default="tests/test.mp3")
    ap.add_argument("--gt", default="tests/fixtures/test_ground_truth_v1.jsonl")
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--drain-sec", type=float, default=20.0)
    ap.add_argument("--max-sec", type=float, default=0.0,
                    help="if >0, truncate both audio and GT to this many "
                         "seconds of source audio (sanity-check mode)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"runs/online_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT.
    gt = [json.loads(l) for l in Path(args.gt).read_text().splitlines() if l.strip()]
    if args.max_sec > 0:
        cap_ms = int(args.max_sec * 1000)
        gt = [g for g in gt if g["end_ms"] <= cap_ms]
    gt_speakers = [g["speaker"] for g in gt]
    print(f"[gt] {len(gt)} segments, {len(set(gt_speakers))} speakers "
          f"(max_sec={args.max_sec})")

    # Decode audio.
    pcm = decode_to_pcm(args.audio)
    if args.max_sec > 0:
        pcm = pcm[: int(args.max_sec * 16000) * 2]
    dur = len(pcm) / 2 / 16000.0
    print(f"[audio] {dur:.1f}s @ 16kHz; replay at {args.speed}x")

    # WS setup.
    speaker_events: list[dict] = []  # order-preserving capture
    stats_count = 0
    lock = threading.Lock()

    raw_fh = (out_dir / "raw_events.jsonl").open("w", encoding="utf-8")

    def on_message(ws, msg):
        nonlocal stats_count
        if isinstance(msg, bytes):
            try:
                msg = msg.decode("utf-8", errors="replace")
            except Exception:
                return
        with lock:
            raw_fh.write(msg.rstrip("\n") + "\n")
        try:
            obj = json.loads(msg)
        except json.JSONDecodeError:
            return
        t = obj.get("type")
        if t == "speaker":
            with lock:
                speaker_events.append({
                    "id": int(obj.get("id", -1)),
                    "sim": float(obj.get("sim", 0.0)),
                    "new": bool(obj.get("new", False)),
                    "name": obj.get("name", ""),
                    "order": len(speaker_events) + 1,
                })
        elif t == "pipeline_stats":
            stats_count += 1

    connected = threading.Event()

    def on_open(ws):
        print("[ws] connected", flush=True)
        connected.set()

    def on_error(ws, err):
        print(f"[ws] error: {err}", file=sys.stderr, flush=True)

    ws = websocket.WebSocketApp(
        args.url, on_open=on_open, on_message=on_message, on_error=on_error,
    )
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    if not connected.wait(timeout=10):
        print("[ws] connect timeout", file=sys.stderr)
        return 2

    stream_pcm(ws, pcm, args.chunk, args.speed)
    print(f"[drain] waiting {args.drain_sec}s for tail segments...", flush=True)
    time.sleep(args.drain_sec)

    try:
        ws.close()
    except Exception:
        pass
    raw_fh.close()

    # Save events snapshot.
    with (out_dir / "speaker_events.jsonl").open("w", encoding="utf-8") as f:
        for e in speaker_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n[capture] speaker events={len(speaker_events)} "
          f"(pipeline_stats broadcasts seen: {stats_count})")

    # Score. Use only SAAS_FULL-like events — any with id>=0 count; id=-1
    # would mean an abstain broadcast, but the runtime currently does not
    # emit those via the `speaker` callback, so all events have id>=0.
    pred_clusters = [e["id"] for e in speaker_events]
    res = score_sequence(pred_clusters, gt_speakers)

    print("\n=== Online replay result ===")
    print(f"aligned prefix: {res['n_aligned']} "
          f"(runtime emitted {res['n_pred']}, GT has {res['n_gt']})")
    print(f"macro={res['macro']:.3f}  micro={res['micro']:.3f}")
    print("per-speaker accuracy:")
    for s, a in sorted(res["per_spk"].items()):
        print(f"  {s}: {a:.3f}")
    print(f"cluster→speaker mapping (first-seen): "
          f"{len(res['mapping'])} clusters")

    # Persist summary.
    (out_dir / "summary.json").write_text(json.dumps({
        "n_pred": res["n_pred"], "n_gt": res["n_gt"],
        "n_aligned": res["n_aligned"],
        "macro": round(res["macro"], 4),
        "micro": round(res["micro"], 4),
        "per_spk": {s: round(v, 4) for s, v in res["per_spk"].items()},
        "mapping": {str(k): v for k, v in res["mapping"].items()},
        "args": vars(args),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
