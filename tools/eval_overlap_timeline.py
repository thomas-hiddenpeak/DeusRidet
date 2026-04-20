#!/usr/bin/env python3
"""Evaluate multi-speaker timeline against overlap reference intervals.

Inputs:
- Reference: tests/asrTest2OverlapTimeline.txt
- Prediction: JSONL produced by tools/test_audio_ws.py --stats-log

This is a blind, relative-time evaluation by second.
"""

import argparse
import json
import math
import re
from pathlib import Path


def parse_hms(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def parse_reference(path: Path, min_conf: str):
    allowed = {"low", "medium", "high"}
    if min_conf not in allowed:
        raise ValueError("min_conf must be one of low|medium|high")

    conf_rank = {"low": 1, "medium": 2, "high": 3}
    min_rank = conf_rank[min_conf]

    truth_seconds = set()
    intervals = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or not raw[0].isdigit():
            continue
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        start, end, conf = parts[0], parts[1], parts[2].strip().lower()
        if conf_rank.get(conf, 0) < min_rank:
            continue
        s0 = parse_hms(start)
        s1 = parse_hms(end)
        if s1 < s0:
            s0, s1 = s1, s0
        intervals.append((s0, s1, conf))
        for t in range(s0, s1 + 1):
            truth_seconds.add(t)

    return truth_seconds, intervals


def parse_prediction(path: Path):
    points = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "pipeline_stats":
            continue
        t = float(d.get("audio_sec", 0.0) or 0.0)
        state = bool(d.get("multi_speaker", False))
        points.append((t, state))

    points.sort(key=lambda x: x[0])
    return points


def expand_pred_seconds(points, end_sec):
    pred = set()
    if not points:
        return pred

    for i in range(len(points) - 1):
        t0, s0 = points[i]
        t1, _ = points[i + 1]
        a = int(math.floor(t0))
        b = int(math.floor(t1))
        if s0:
            for t in range(a, b + 1):
                pred.add(t)

    t_last, s_last = points[-1]
    if s_last:
        for t in range(int(math.floor(t_last)), end_sec + 1):
            pred.add(t)

    return pred


def event_starts(seconds_set):
    starts = []
    for t in sorted(seconds_set):
        if (t - 1) not in seconds_set:
            starts.append(t)
    return starts


def main():
    ap = argparse.ArgumentParser(description="Blind overlap timeline evaluator")
    ap.add_argument("--reference", default="tests/asrTest2OverlapTimeline.txt")
    ap.add_argument("--pred", required=True, help="JSONL from test_audio_ws --stats-log")
    ap.add_argument("--min-conf", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--tol", type=int, default=2, help="event latency tolerance in seconds")
    args = ap.parse_args()

    ref_path = Path(args.reference)
    pred_path = Path(args.pred)

    truth_seconds, truth_intervals = parse_reference(ref_path, args.min_conf)
    points = parse_prediction(pred_path)

    if not truth_seconds:
        print("No reference seconds after confidence filter.")
        return
    if not points:
        print("No prediction points found.")
        return

    end_sec = max(max(truth_seconds), int(math.floor(points[-1][0])))
    pred_seconds = expand_pred_seconds(points, end_sec)

    all_sec = set(range(0, end_sec + 1))

    tp = len(truth_seconds & pred_seconds)
    fp = len(pred_seconds - truth_seconds)
    fn = len(truth_seconds - pred_seconds)
    tn = len(all_sec - (truth_seconds | pred_seconds))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Event-level latency (pred ON vs nearest truth ON)
    truth_starts = [s for s, _, _ in truth_intervals]
    pred_starts = event_starts(pred_seconds)

    matched = 0
    latencies = []
    for p in pred_starts:
        best = None
        for t in truth_starts:
            d = p - t
            if abs(d) <= args.tol:
                if best is None or abs(d) < abs(best):
                    best = d
        if best is not None:
            matched += 1
            latencies.append(best)

    print("== Blind Overlap Evaluation ==")
    print(f"reference: {ref_path}")
    print(f"prediction: {pred_path}")
    print(f"confidence>= {args.min_conf}")
    print(f"seconds: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
    print(f"truth_intervals={len(truth_intervals)} pred_on_events={len(pred_starts)}")
    print(f"event_matched_within_{args.tol}s={matched}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"latency_sec: avg={avg:.3f} min={min(latencies)} max={max(latencies)}")


if __name__ == "__main__":
    main()
