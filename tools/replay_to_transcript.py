#!/usr/bin/env python3
"""
replay_to_transcript.py — Mechanical capture + patch + GT pairing.

This tool is **strictly mechanical** — it captures the WS stream of a
running awaken server, applies the same forward-map that the WebUI
applies (see src/nexus/webui/js/components/timeline-panel.js
::onSpeakerRelabel), pairs each runtime VAD segment with the
overlapping ground-truth utterance (text + speaker name), and emits a
human-readable markdown timeline report.

It does NOT compute macro-F1, fuzzy string matching, or any
"auto-judged correctness number". Per
.github/instructions/workflow.instructions.md (Semantic Evaluation,
Not Scripted Scoring): the agent must read the produced report
top-to-bottom and report patterns. Numeric "quality scores" are
forbidden here.

Usage (with awaken already running on :8080):
    python3 tools/replay_to_transcript.py \\
        --audio tests/test.mp3 \\
        --gt    tests/fixtures/test_ground_truth.json \\
        --max-sec 600 --speed 4.0 \\
        --out-dir runs/$(date +%Y%m%dT%H%M%S)_replay_transcript
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import websocket  # type: ignore


# ─── audio + WS plumbing ─────────────────────────────────────────────

def decode_to_pcm(path: str) -> bytes:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed: {r.stderr.decode()}")
    return r.stdout


def stream_pcm(ws, pcm: bytes, chunk: int, speed: float) -> None:
    total = len(pcm) // 2
    n_chunks = (total + chunk - 1) // chunk
    frame_wall = (chunk / 16000.0) / max(speed, 1e-6)
    t0 = time.time()
    for i in range(n_chunks):
        a = i * chunk
        b = min(a + chunk, total)
        ws.send(pcm[a * 2: b * 2], opcode=websocket.ABNF.OPCODE_BINARY)
        target = t0 + (i + 1) * frame_wall
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)
    print(f"[stream] sent {total/16000.0:.1f}s src in {time.time()-t0:.1f}s wall",
          flush=True)


# ─── timeline jsonl reader ───────────────────────────────────────────

def parse_vad_segments(timeline_path: Path, sr: int = 16000) -> list[dict]:
    segs = []
    open_start = None
    for line in timeline_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("t") != "vad":
            continue
        ev = obj.get("event")
        t1 = int(obj.get("audio_t1", 0))
        if ev == "start":
            open_start = t1
        elif ev == "end" and open_start is not None:
            segs.append({"start_sec": open_start / float(sr),
                         "end_sec":   t1 / float(sr)})
            open_start = None
    return segs


def newest_timeline_log(d: Path = Path("logs/timeline")) -> Path | None:
    if not d.exists():
        return None
    cands = sorted(d.glob("tl_*.jsonl"))
    return cands[-1] if cands else None


# ─── forward-map (JS-parity) ─────────────────────────────────────────

class RelabelMap:
    """Replay timeline-panel.js::onSpeakerRelabel semantics."""

    def __init__(self):
        self.map: dict[int, int] = {}
        self.relabel_log: list[dict] = []  # for transparency

    def resolve(self, cid: int) -> int:
        if cid is None or cid < 0:
            return cid
        cur = cid
        visited = set()
        while cur in self.map and cur not in visited:
            visited.add(cur)
            cur = self.map[cur]
        return cur

    def apply(self, old_id: int, new_id: int, t_now_sec: float,
              confidence: float = 0.0) -> None:
        if old_id is None or new_id is None:
            return
        if old_id < 0 or new_id < 0 or old_id == new_id:
            return
        # Resolve new_id transitively first.
        target = new_id
        visited = {old_id}
        while target in self.map and target not in visited:
            visited.add(target)
            target = self.map[target]
        self.map[old_id] = target
        # Rewrite any entries that pointed at old_id.
        for k in list(self.map.keys()):
            if self.map[k] == old_id:
                self.map[k] = target
        self.relabel_log.append({
            "t_relabel_sec": t_now_sec,
            "old_id": old_id, "new_id": new_id, "resolved_to": target,
            "confidence": confidence,
        })


# ─── pairing ─────────────────────────────────────────────────────────

def pair_runtime_with_vad(vad_segs: list[dict],
                          speaker_events: list[dict]) -> list[dict]:
    """For each speaker broadcast, snap to the nearest unused VAD end."""
    vad_sorted = sorted(vad_segs, key=lambda v: v["end_sec"])
    used = [False] * len(vad_sorted)
    out = []
    for e in speaker_events:
        t_close = float(e.get("t_close_sec", 0.0))
        best_i, best_d = -1, float("inf")
        for i, v in enumerate(vad_sorted):
            if used[i]:
                continue
            d = abs(v["end_sec"] - t_close)
            if d < best_d:
                best_d = d
                best_i = i
            if v["end_sec"] > t_close + 5.0:
                break
        if best_i < 0:
            continue
        used[best_i] = True
        v = vad_sorted[best_i]
        out.append({
            "start_sec":   v["start_sec"],
            "end_sec":     v["end_sec"],
            "raw_id":      e["raw_id"],
            "current_id":  e["current_id"],
            "sim":         e.get("sim", 0.0),
            "name":        e.get("name", ""),
            "amended":     e.get("amended", False),
            "order":       e.get("order", 0),
            "pair_dt":     best_d,
            "relabel_chain": e.get("relabel_chain", []),
        })
    out.sort(key=lambda r: r["start_sec"])
    return out


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ─── main ────────────────────────────────────────────────────────────

def fmt_t(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:06.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="tests/test.mp3")
    ap.add_argument("--gt", default="tests/fixtures/test_ground_truth.json",
                    help="utterance-level GT with text")
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=1600)
    ap.add_argument("--drain-sec", type=float, default=25.0)
    ap.add_argument("--max-sec", type=float, default=600.0,
                    help="truncate audio + GT to first N seconds")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT.
    gt_obj = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    utts_all = gt_obj.get("utterances", [])
    if args.max_sec > 0:
        utts = [u for u in utts_all
                if u.get("t0_end_sec", 0.0) <= args.max_sec + 1.0]
    else:
        utts = utts_all
    print(f"[gt] {len(utts)} utterances (max_sec={args.max_sec}); "
          f"speakers={gt_obj.get('speakers')}")

    # Decode audio.
    pcm = decode_to_pcm(args.audio)
    if args.max_sec > 0:
        pcm = pcm[: int(args.max_sec * 16000) * 2]
    dur_sec = len(pcm) / 2 / 16000.0
    print(f"[audio] {dur_sec:.1f}s @16k; replay {args.speed}x")

    # WS capture state.
    raw_path = out_dir / "raw_events.jsonl"
    raw_fh = raw_path.open("w", encoding="utf-8")

    relabel_map = RelabelMap()
    speaker_events: list[dict] = []
    asr_events: list[dict] = []
    last_audio_t1_in = 0
    stats_count = 0
    lock = threading.Lock()

    def on_message(_ws, msg):
        nonlocal last_audio_t1_in, stats_count
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
        if t == "pipeline_stats":
            stats_count += 1
            try:
                last_audio_t1_in = int(obj.get("audio_t1_in", last_audio_t1_in))
            except (TypeError, ValueError):
                pass
        elif t == "speaker":
            raw_id = int(obj.get("id", -1))
            with lock:
                current = relabel_map.resolve(raw_id)
                speaker_events.append({
                    "raw_id":     raw_id,
                    "current_id": current,
                    "sim":        float(obj.get("sim", 0.0)),
                    "name":       obj.get("name", ""),
                    "new":        bool(obj.get("new", False)),
                    "order":      len(speaker_events) + 1,
                    "t_close_sec": last_audio_t1_in / 16000.0,
                    "relabel_chain": [],
                })
        elif t == "speaker_amend":
            with lock:
                # Update the matching speaker event in-place (closest
                # by prior_id + target_t_close_sec, within 2s).
                tgt_t = float(obj.get("target_t_close_sec", 0.0))
                prior = int(obj.get("prior_id", -1))
                new_id = int(obj.get("id", -1))
                best_i, best_dt = -1, 2.0
                for i, ev in enumerate(speaker_events):
                    if prior >= 0 and ev["raw_id"] != prior:
                        continue
                    dt = abs(ev["t_close_sec"] - tgt_t)
                    if dt <= best_dt:
                        best_dt = dt; best_i = i
                if best_i >= 0:
                    speaker_events[best_i]["raw_id"] = new_id
                    # Re-resolve current under map.
                    speaker_events[best_i]["current_id"] = relabel_map.resolve(new_id)
                    speaker_events[best_i]["amended"] = True
                    speaker_events[best_i]["amend_from"] = prior
        elif t == "speaker_relabel":
            old_id = int(obj.get("old_id", -1))
            new_id = int(obj.get("new_id", -1))
            conf = float(obj.get("confidence", 0.0))
            with lock:
                relabel_map.apply(old_id, new_id,
                                  t_now_sec=last_audio_t1_in / 16000.0,
                                  confidence=conf)
                # Patch already-emitted speaker_events whose current_id == old_id.
                for ev in speaker_events:
                    if ev["current_id"] == old_id:
                        ev["current_id"] = relabel_map.resolve(old_id)
                        ev["relabel_chain"].append(
                            {"old": old_id,
                             "new": ev["current_id"],
                             "t_sec": last_audio_t1_in / 16000.0,
                             "conf": conf})
        elif t == "asr_transcript":
            with lock:
                asr_events.append({
                    "start_sec":   float(obj.get("stream_start_sec", 0.0)),
                    "end_sec":     float(obj.get("stream_end_sec", 0.0)),
                    "text":        obj.get("text", ""),
                    "speaker_id":  int(obj.get("speaker_id", -1)),
                    "speaker_name": obj.get("speaker_name", ""),
                })

    connected = threading.Event()

    def on_open(_ws):
        print("[ws] connected", flush=True)
        connected.set()

    def on_error(_ws, err):
        print(f"[ws] error: {err}", file=sys.stderr, flush=True)

    ws = websocket.WebSocketApp(args.url,
                                on_open=on_open, on_message=on_message,
                                on_error=on_error)
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    if not connected.wait(timeout=10):
        print("[ws] connect timeout", file=sys.stderr)
        return 2

    stream_pcm(ws, pcm, args.chunk, args.speed)
    print(f"[drain] waiting {args.drain_sec:.0f}s …", flush=True)
    time.sleep(args.drain_sec)

    try:
        ws.close()
    except Exception:
        pass
    raw_fh.close()

    with lock:
        n_spk = len(speaker_events)
        n_amend = sum(1 for e in speaker_events if e.get("amended"))
        n_relabel = len(relabel_map.relabel_log)
        n_asr = len(asr_events)
        print(f"[capture] speaker={n_spk} amend={n_amend} "
              f"relabel={n_relabel} asr={n_asr} stats={stats_count}")

    # Pair with timeline VAD segments.
    tl = newest_timeline_log()
    if not tl:
        print("[err] no timeline jsonl found", file=sys.stderr)
        return 3
    print(f"[timeline] {tl}")
    vad_segs = parse_vad_segments(tl)
    if args.max_sec > 0:
        vad_segs = [v for v in vad_segs if v["end_sec"] <= args.max_sec + 5.0]
    print(f"[vad] {len(vad_segs)} segments")

    rt_segs = pair_runtime_with_vad(vad_segs, speaker_events)
    print(f"[pair] runtime_segments={len(rt_segs)}")

    # ── emit JSON for transparency ─────────────────────────────
    (out_dir / "runtime_segments.json").write_text(
        json.dumps(rt_segs, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "relabel_log.json").write_text(
        json.dumps(relabel_map.relabel_log, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "forward_map.json").write_text(
        json.dumps(relabel_map.map, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "asr_events.json").write_text(
        json.dumps(asr_events, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── emit human-readable markdown ───────────────────────────
    md = out_dir / "transcript.md"
    with md.open("w", encoding="utf-8") as f:
        f.write(f"# Replay transcript\n\n")
        f.write(f"- source audio: `{args.audio}` (first {args.max_sec:.0f}s)\n")
        f.write(f"- GT utterances paired: {len(utts)}\n")
        f.write(f"- runtime VAD segments: {len(vad_segs)}\n")
        f.write(f"- runtime speaker decisions: {len(rt_segs)}\n")
        f.write(f"- speaker_amend in-place: {n_amend}\n")
        f.write(f"- speaker_relabel (forward-map merges): {n_relabel}\n")
        f.write(f"- GT speakers: {gt_obj.get('speakers')}\n\n")

        # ── relabel log ──
        f.write("## Reclusterer forward-map events\n\n")
        if not relabel_map.relabel_log:
            f.write("_no speaker_relabel events fired in this slice._\n\n")
        else:
            for r in relabel_map.relabel_log:
                f.write(f"- t={fmt_t(r['t_relabel_sec'])}  "
                        f"{r['old_id']} → {r['new_id']} "
                        f"(resolved={r['resolved_to']}, "
                        f"conf={r['confidence']:.3f})\n")
            f.write("\nFinal forward-map (raw → resolved):\n\n")
            for k, v in sorted(relabel_map.map.items()):
                f.write(f"- {k} → {v}\n")
            f.write("\n")

        # ── GT-oriented narrative ──
        f.write("## GT-oriented narrative\n\n")
        f.write("For each GT utterance: speaker name, time window, full text,\n"
                "then the runtime VAD segments overlapping it with their\n"
                "post-relabel predicted cluster id and the raw chain.\n\n")

        for u in utts:
            us = float(u["t0_start_sec"])
            ue = float(u["t0_end_sec"])
            who = u["speaker"]
            text = (u.get("text", "") or "").replace("\n", " ").strip()
            f.write(f"### [{fmt_t(us)} – {fmt_t(ue)}]  GT={who}\n\n")
            f.write(f"> {text}\n\n")

            # find overlapping runtime segments (>= 0.1s overlap or
            # whose midpoint is inside the utterance).
            cands = []
            for r in rt_segs:
                ov = overlap_sec(us, ue, r["start_sec"], r["end_sec"])
                mid = 0.5 * (r["start_sec"] + r["end_sec"])
                if ov >= 0.1 or (us <= mid <= ue):
                    cands.append((r, ov))
            if not cands:
                f.write("_(no runtime segment overlaps this utterance — "
                        "either VAD missed it or it was outside the slice)_\n\n")
                continue
            f.write("runtime decisions:\n\n")
            for r, ov in cands:
                chain = r.get("relabel_chain", [])
                chain_str = ""
                if chain:
                    parts = [f"{c['old']}→{c['new']}@{fmt_t(c['t_sec'])}"
                             for c in chain]
                    chain_str = f"  relabel=[{'; '.join(parts)}]"
                amend_tag = " (amended)" if r.get("amended") else ""
                f.write(f"- [{fmt_t(r['start_sec'])} – {fmt_t(r['end_sec'])}]"
                        f"  pred={r['current_id']}"
                        f"  raw={r['raw_id']}{amend_tag}"
                        f"  sim={r['sim']:.3f}  ov={ov:.2f}s{chain_str}\n")
            f.write("\n")

        # ── runtime-oriented appendix ──
        f.write("## Runtime-oriented appendix (every emitted decision)\n\n")
        for r in rt_segs:
            # Find the GT utterance with the largest overlap.
            best_u, best_ov = None, 0.0
            for u in utts:
                us = float(u["t0_start_sec"]); ue = float(u["t0_end_sec"])
                ov = overlap_sec(us, ue, r["start_sec"], r["end_sec"])
                if ov > best_ov:
                    best_ov = ov; best_u = u
            gt_who = best_u["speaker"] if best_u else "—"
            chain = r.get("relabel_chain", [])
            chain_str = ""
            if chain:
                parts = [f"{c['old']}→{c['new']}@{fmt_t(c['t_sec'])}" for c in chain]
                chain_str = f"  relabel=[{'; '.join(parts)}]"
            f.write(f"- [{fmt_t(r['start_sec'])} – {fmt_t(r['end_sec'])}]  "
                    f"pred={r['current_id']}  raw={r['raw_id']}  "
                    f"GT={gt_who}  ov={best_ov:.2f}s{chain_str}\n")

    print(f"[done] {md}")
    print(f"[done] {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
