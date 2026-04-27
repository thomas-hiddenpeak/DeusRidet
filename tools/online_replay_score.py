#!/usr/bin/env python3
"""
online_replay_score.py — Step 5 online replay evaluation.

Streams a source audio file into a running awaken server over
WebSocket, captures every ``{"type":"speaker",...}`` broadcast, then
reads the TimelineLogger jsonl to recover each runtime segment's
``[start_sec, end_sec]`` from the VAD entries. Each runtime segment is
paired in order with its corresponding speaker broadcast, producing a
list of (interval, cluster_id) tuples.

Scoring: for each GT segment, we pick the runtime segment with the
greatest time overlap; a first-seen cluster → speaker mapping (the
first GT segment matched to cluster k fixes cluster k → that GT
segment's true speaker) converts cluster ids into predicted names.
GT segments with zero overlap are counted as wrong (runtime emitted
no decision covering that interval).

This methodology produces a macro-accuracy number that is directly
comparable to the Step 3 offline ``baseline`` configuration because
both score over the same GT segment set; the difference is only
whether the cluster id for each GT segment comes from a live runtime
pass or from a simulated one.

Usage:
    python3 tools/online_replay_score.py \\
        --audio tests/test.mp3 \\
        --gt tests/fixtures/test_ground_truth_v1.jsonl \\
        --speed 1.0 --drain-sec 15 --max-sec 600 \\
        --out-dir runs/online_$(date +%Y%m%dT%H%M%S)

The awaken server must already be running on ws://localhost:8080/ws
with a fresh TimelineLogger (logs/timeline/tl_*.jsonl).
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


def parse_timeline_vad_segments(timeline_path: Path, sample_rate: int = 16000):
    """Parse VAD segment boundaries from a TimelineLogger jsonl file.

    Returns a list of {"start_sec": float, "end_sec": float} pairs,
    built by pairing consecutive segment_start / segment_end VAD entries
    in emission order. audio_t1 is in source-audio sample count, so
    seconds = audio_t1 / sample_rate.
    """
    segs = []
    open_start = None  # audio_t1 of last unmatched segment_start
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
            segs.append({
                "start_sec": open_start / float(sample_rate),
                "end_sec":   t1 / float(sample_rate),
            })
            open_start = None
    return segs


def pair_segments_with_events(vad_segs, speaker_events):
    """Pair runtime VAD segments with speaker broadcasts by close-time
    nearest-neighbour.

    The SAAS_FULL path skips VAD segments that are too short to extract
    a reliable embedding, so ``len(speaker_events) <= len(vad_segs)``.
    Each speaker broadcast carries an approximate ``t_close_sec`` (the
    audio_t1_in seen in the nearest preceding pipeline_stats frame),
    which we snap to the VAD segment whose ``end_sec`` is closest
    (within a small tolerance). Unpaired VAD segments are reported in
    ``n_unpaired_vad`` and do NOT contribute to the runtime_segs list.
    """
    vad_sorted = sorted(vad_segs, key=lambda v: v["end_sec"])
    used = [False] * len(vad_sorted)
    runtime = []
    for e in speaker_events:
        t_close = float(e.get("t_close_sec", 0.0))
        # Find nearest unused VAD end.
        best_i, best_d = -1, float("inf")
        for i, v in enumerate(vad_sorted):
            if used[i]:
                continue
            d = abs(v["end_sec"] - t_close)
            if d < best_d:
                best_d = d
                best_i = i
            # Early exit: sorted list, once we cross t_close + margin, stop.
            if v["end_sec"] > t_close + 5.0:
                break
        if best_i < 0:
            continue
        used[best_i] = True
        v = vad_sorted[best_i]
        runtime.append({
            "start_sec": v["start_sec"],
            "end_sec":   v["end_sec"],
            "cluster":   e["id"],
            "name":      e.get("name", ""),
            "sim":       e.get("sim", 0.0),
            "order":     e.get("order", 0),
            "pair_dt":   best_d,
        })
    # Order runtime segs by their true interval start.
    runtime.sort(key=lambda r: r["start_sec"])
    return runtime, {
        "n_vad": len(vad_segs),
        "n_spk": len(speaker_events),
        "paired": len(runtime),
        "n_unpaired_vad": len(vad_segs) - len(runtime),
    }


def apply_speaker_amends(speaker_events, amend_events, tolerance_sec: float = 2.0):
    applied = 0
    for amend in amend_events:
        target_t = amend.get("target_t_close_sec", 0.0)
        prior_id = amend.get("prior_id", -1)
        best_i = -1
        best_dt = tolerance_sec
        for i, ev in enumerate(speaker_events):
            if prior_id >= 0 and ev.get("id", -1) != prior_id:
                continue
            dt = abs(ev.get("t_close_sec", 0.0) - target_t)
            if dt <= best_dt:
                best_dt = dt
                best_i = i
        if best_i < 0:
            continue
        ev = speaker_events[best_i]
        ev["amended_from_id"] = ev.get("id", -1)
        ev["amended_from_sim"] = ev.get("sim", 0.0)
        ev["amended_by_t_close_sec"] = target_t
        ev["id"] = amend.get("id", ev.get("id", -1))
        ev["sim"] = amend.get("sim", ev.get("sim", 0.0))
        ev["name"] = amend.get("name", ev.get("name", ""))
        ev["amended"] = True
        applied += 1
    return applied


def score_by_overlap(runtime_segs, gt_rows):
    """For each GT segment, find the runtime segment with the largest
    time-overlap (intersection length). First-seen cluster → speaker
    mapping over GT segments (the first GT segment matched to cluster k
    fixes cluster k → that GT segment's true speaker name). Compute
    per-speaker accuracy, macro, micro.

    GT segments with zero overlap to any runtime segment are counted
    as wrong (the runtime simply did not emit a decision covering that
    interval).
    """
    # Pre-sort runtime by start for a mild speed-up (still O(NM) worst).
    runtime_sorted = sorted(runtime_segs, key=lambda r: r["start_sec"])

    def best_match(gs, ge):
        best = None
        best_ov = 0.0
        for r in runtime_sorted:
            if r["end_sec"] <= gs:
                continue
            if r["start_sec"] >= ge:
                break
            ov = max(0.0, min(ge, r["end_sec"]) - max(gs, r["start_sec"]))
            if ov > best_ov:
                best_ov = ov
                best = r
        return best, best_ov

    matched = []
    for g in gt_rows:
        gs = g["start_ms"] / 1000.0
        ge = g["end_ms"]   / 1000.0
        rt, ov = best_match(gs, ge)
        matched.append({
            "gt_idx":     g["idx"],
            "gt_start":   gs,
            "gt_end":     ge,
            "gt_speaker": g["speaker"],
            "rt_cluster": rt["cluster"] if rt else -1,
            "rt_start":   rt["start_sec"] if rt else None,
            "rt_end":     rt["end_sec"]   if rt else None,
            "overlap":    ov,
            "status":     ("decided" if (rt and rt["cluster"] >= 0)
                           else ("abstain" if rt else "no_segment")),
        })

    # First-seen mapping (by GT index order, matches online semantics).
    mapping = {}
    for m in matched:
        c = m["rt_cluster"]
        if c < 0:
            continue
        mapping.setdefault(c, m["gt_speaker"])

    speakers = sorted({m["gt_speaker"] for m in matched})
    per_spk_correct = {s: 0 for s in speakers}
    per_spk_total   = {s: 0 for s in speakers}
    per_spk_decided = {s: 0 for s in speakers}
    per_spk_decided_correct = {s: 0 for s in speakers}
    n_no_seg  = 0
    n_abstain = 0
    for m in matched:
        true = m["gt_speaker"]
        per_spk_total[true] += 1
        st = m["status"]
        if st == "no_segment":
            n_no_seg += 1
            continue
        if st == "abstain":
            n_abstain += 1
            continue
        # decided
        per_spk_decided[true] += 1
        pred = mapping.get(m["rt_cluster"], "__unk__")
        if pred == true:
            per_spk_correct[true] += 1
            per_spk_decided_correct[true] += 1

    per_spk_acc = {
        s: per_spk_correct[s] / per_spk_total[s] if per_spk_total[s] else 0.0
        for s in speakers
    }
    per_spk_decided_acc = {
        s: (per_spk_decided_correct[s] / per_spk_decided[s])
            if per_spk_decided[s] else 0.0
        for s in speakers
    }
    macro = sum(per_spk_acc.values()) / max(1, len(per_spk_acc))
    micro = sum(per_spk_correct.values()) / max(1, sum(per_spk_total.values()))
    decided_macro = (
        sum(per_spk_decided_acc.values()) / max(1, len(per_spk_decided_acc))
    )
    decided_micro = (
        sum(per_spk_decided_correct.values())
        / max(1, sum(per_spk_decided.values()))
    )
    coverage = sum(per_spk_decided.values()) / max(1, sum(per_spk_total.values()))
    return {
        "macro": macro, "micro": micro,
        "decided_macro": decided_macro, "decided_micro": decided_micro,
        "coverage": coverage,
        "per_spk": per_spk_acc,
        "per_spk_decided": per_spk_decided_acc,
        "mapping": mapping,
        "n_gt":       len(matched),
        "n_no_seg":   n_no_seg,
        "n_abstain":  n_abstain,
        "n_decided":  sum(per_spk_decided.values()),
        "matched":    matched,
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
    ap.add_argument("--enable-asr", action="store_true",
                    help="After WS connect, send 'asr_enable:on' so the server "
                         "emits asr_transcript events. Server must have been "
                         "launched with DEUSRIDET_TEST_WS_ENABLE_ASR=1 so the "
                         "ASR model is actually loaded. With ASR on, replay "
                         "speed > 2.0x is unreliable.")
    ap.add_argument("--timeline", default=None,
                    help="Path to the TimelineLogger jsonl produced by this "
                         "awaken run (VAD segment boundaries). If omitted, "
                         "the newest logs/timeline/tl_*.jsonl is used.")
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
    amend_events: list[dict] = []    # retroactive speaker relabels
    asr_events: list[dict] = []      # asr_transcript captures (Step 5e)
    stats_count = 0
    last_audio_t1_in = 0  # updated from every pipeline_stats broadcast
    lock = threading.Lock()

    raw_fh = (out_dir / "raw_events.jsonl").open("w", encoding="utf-8")

    def on_message(ws, msg):
        nonlocal stats_count, last_audio_t1_in
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
                # Stamp each speaker broadcast with the most recent
                # audio_t1_in seen in a pipeline_stats broadcast. The
                # runtime emits pipeline_stats at ~10 Hz and speaker at
                # segment-end on the same thread, so the last-seen
                # audio_t1_in is within ~100 ms of the segment close.
                speaker_events.append({
                    "id": int(obj.get("id", -1)),
                    "sim": float(obj.get("sim", 0.0)),
                    "new": bool(obj.get("new", False)),
                    "name": obj.get("name", ""),
                    "order": len(speaker_events) + 1,
                    "t_close_sec": last_audio_t1_in / 16000.0,
                })
        elif t == "speaker_amend":
            with lock:
                amend_events.append({
                    "target_t_close_sec": float(obj.get("target_t_close_sec", 0.0)),
                    "prior_id": int(obj.get("prior_id", -1)),
                    "prior_sim": float(obj.get("prior_sim", 0.0)),
                    "id": int(obj.get("id", -1)),
                    "sim": float(obj.get("sim", 0.0)),
                    "name": obj.get("name", ""),
                    "order": len(amend_events) + 1,
                })
        elif t == "pipeline_stats":
            stats_count += 1
            try:
                last_audio_t1_in = int(obj.get("audio_t1_in", last_audio_t1_in))
            except (TypeError, ValueError):
                pass
        elif t == "asr_transcript":
            with lock:
                asr_events.append({
                    "start_sec":  float(obj.get("stream_start_sec", 0.0)),
                    "end_sec":    float(obj.get("stream_end_sec", 0.0)),
                    "text":       obj.get("text", ""),
                    "speaker_id": int(obj.get("speaker_id", -1)),
                    "speaker_name": obj.get("speaker_name", ""),
                    "speaker_sim": float(obj.get("speaker_sim", 0.0)),
                    "speaker_source": obj.get("speaker_source", ""),
                    "trigger":    obj.get("trigger", ""),
                    "latency_ms": float(obj.get("latency_ms", 0.0)),
                    "audio_sec":  float(obj.get("audio_sec", 0.0)),
                })

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

    if args.enable_asr:
        if args.speed > 2.0:
            print(f"[asr] WARNING: --speed={args.speed} > 2.0 with ASR on; "
                  f"transcripts will be incomplete.", flush=True)
        try:
            ws.send("asr_enable:on", opcode=websocket.ABNF.OPCODE_TEXT)
            print("[asr] sent asr_enable:on", flush=True)
        except Exception as e:
            print(f"[asr] failed to enable: {e}", file=sys.stderr, flush=True)

    stream_pcm(ws, pcm, args.chunk, args.speed)
    print(f"[drain] waiting {args.drain_sec}s for tail segments...", flush=True)
    time.sleep(args.drain_sec)

    # Anchor the ASR transcripts back to source-audio time. The pipeline's
    # audio_t1_in clock started counting at server boot; by the time
    # streaming ends it equals (boot_offset + total_src_samples). We
    # subtract total_src_samples to recover the offset.
    total_src_sec = len(pcm) / 2 / 16000.0
    with lock:
        final_audio_t1_sec = last_audio_t1_in / 16000.0
    stream_anchor_sec = max(0.0, final_audio_t1_sec - total_src_sec)
    print(f"[anchor] final_audio_t1_sec={final_audio_t1_sec:.2f} "
          f"total_src_sec={total_src_sec:.2f} -> "
          f"stream_anchor_sec={stream_anchor_sec:.2f}", flush=True)

    try:
        ws.close()
    except Exception:
        pass
    raw_fh.close()

    n_amend_applied = apply_speaker_amends(speaker_events, amend_events)

    # Save events snapshot.
    with (out_dir / "speaker_events.jsonl").open("w", encoding="utf-8") as f:
        for e in speaker_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if amend_events:
        with (out_dir / "speaker_amends.jsonl").open("w", encoding="utf-8") as f:
            for e in amend_events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Save ASR transcripts (if any) and a human-readable interleaved
    # transcript ready for manual diff against tests/test.txt. Both the
    # source-relative (src_*) and raw pipeline (start_sec/end_sec) times
    # are stored so the offset can be re-derived later.
    if asr_events:
        for e in asr_events:
            e["src_start_sec"] = e["start_sec"] - stream_anchor_sec
            e["src_end_sec"]   = e["end_sec"]   - stream_anchor_sec
        with (out_dir / "asr_transcripts.jsonl").open("w", encoding="utf-8") as f:
            for e in asr_events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        # Sort by source-relative start for comparison with tests/test.txt.
        asr_sorted = sorted(asr_events, key=lambda e: e["src_start_sec"])
        with (out_dir / "asr_transcripts.txt").open("w", encoding="utf-8") as f:
            f.write(f"# source-relative time (anchor_sec={stream_anchor_sec:.2f} "
                    f"subtracted from raw stream_start)\n")
            f.write("# HH:MM:SS  [src_start -> src_end]  spk_id  spk_name  text\n")
            for e in asr_sorted:
                s = max(0.0, e["src_start_sec"])
                hh = int(s) // 3600
                mm = (int(s) // 60) % 60
                ss = int(s) % 60
                f.write(f"{hh:02d}:{mm:02d}:{ss:02d}  "
                        f"[{e['src_start_sec']:7.2f} -> {e['src_end_sec']:7.2f}]  "
                        f"id={e['speaker_id']:>2}  "
                        f"{e['speaker_name'] or '?':<10}  "
                        f"{e['text']}\n")
        print(f"[capture] asr_transcript events={len(asr_events)} -> "
              f"asr_transcripts.{{jsonl,txt}}")

    print(f"\n[capture] speaker events={len(speaker_events)} "
          f"(pipeline_stats broadcasts seen: {stats_count})")
    if amend_events:
        print(f"[capture] speaker_amend events={len(amend_events)} "
              f"applied={n_amend_applied}")

    # Resolve timeline log: CLI arg wins, else newest logs/timeline/tl_*.jsonl.
    if args.timeline:
        tl_path = Path(args.timeline)
    else:
        tl_dir = Path("logs/timeline")
        candidates = sorted(
            tl_dir.glob("tl_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print(f"[err] no logs/timeline/tl_*.jsonl found", file=sys.stderr)
            return 3
        tl_path = candidates[-1]
    print(f"[timeline] using {tl_path}")

    vad_segs = parse_timeline_vad_segments(tl_path)
    print(f"[timeline] parsed {len(vad_segs)} VAD segments")

    # Filter VAD segments to the replayed window if --max-sec was used.
    if args.max_sec > 0:
        vad_segs = [v for v in vad_segs if v["end_sec"] <= args.max_sec + 2.0]

    runtime_segs, pair_stats = pair_segments_with_events(vad_segs, speaker_events)
    print(f"[pair] vad={pair_stats['n_vad']}  speaker={pair_stats['n_spk']}  "
          f"paired(nearest-close)={pair_stats['paired']}  "
          f"unpaired_vad={pair_stats['n_unpaired_vad']}")

    res = score_by_overlap(runtime_segs, gt)

    print("\n=== Online replay result (time-overlap scoring) ===")
    print(f"GT segments: {res['n_gt']}   decided: {res['n_decided']}   "
          f"abstain: {res['n_abstain']}   no_segment: {res['n_no_seg']}")
    print(f"coverage(decided/gt): {res['coverage']:.3f}")
    print(f"macro={res['macro']:.3f}  micro={res['micro']:.3f}   "
          f"(all GT; no_segment & abstain count as wrong)")
    print(f"decided_macro={res['decided_macro']:.3f}  "
          f"decided_micro={res['decided_micro']:.3f}   "
          f"(only GT segs with a decided runtime cluster)")
    print("per-speaker accuracy (all / decided-only):")
    for s in sorted(res["per_spk"]):
        print(f"  {s}: {res['per_spk'][s]:.3f} / "
              f"{res['per_spk_decided'][s]:.3f}")
    print(f"first-seen mapping: {len(res['mapping'])} cluster(s) -> "
          f"{len(set(res['mapping'].values()))} unique name(s)")
    for c, n in sorted(res["mapping"].items()):
        print(f"  cluster {c} -> {n}")

    # Persist per-GT match detail + summary.
    with (out_dir / "matched.jsonl").open("w", encoding="utf-8") as f:
        for m in res["matched"]:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(json.dumps({
        "method":   "time_overlap_first_seen",
        "n_gt":     res["n_gt"],
        "n_vad":    pair_stats["n_vad"],
        "n_spk":    pair_stats["n_spk"],
        "n_amend":  len(amend_events),
        "n_amend_applied": n_amend_applied,
        "n_paired": pair_stats["paired"],
        "n_decided":  res["n_decided"],
        "n_abstain":  res["n_abstain"],
        "n_no_seg":   res["n_no_seg"],
        "coverage":   round(res["coverage"], 4),
        "macro":          round(res["macro"], 4),
        "micro":          round(res["micro"], 4),
        "decided_macro":  round(res["decided_macro"], 4),
        "decided_micro":  round(res["decided_micro"], 4),
        "per_spk":        {s: round(v, 4) for s, v in res["per_spk"].items()},
        "per_spk_decided":{s: round(v, 4) for s, v in res["per_spk_decided"].items()},
        "mapping":  {str(k): v for k, v in res["mapping"].items()},
        "timeline": str(tl_path),
        "args":     vars(args),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
