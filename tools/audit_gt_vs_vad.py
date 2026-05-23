#!/usr/bin/env python3
"""Step 19a addendum — Map GT coverage to VAD intervals.

For every GT utterance in `tests/fixtures/test_ground_truth_v1.jsonl` within
[0, max_sec], classify how it overlaps with online VAD intervals parsed from
a timeline:

  - "no_vad"      : zero overlap with any VAD interval
  - "isolated"    : overlaps exactly one VAD interval, alone
  - "shared_2"    : overlaps a VAD interval shared with 1 other speaker's GT
  - "shared_3+"   : overlaps a VAD interval shared with >=2 other speakers
  - "multi_vad"   : the GT utterance spans across >1 VAD intervals

Outputs a count table + per-speaker breakdown + a list of the shared_2+
intervals with their speakers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_vad(p: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    in_seg = False
    cur = 0.0
    for line in p.read_text().splitlines():
        if '"t":"vad"' not in line:
            continue
        obj = json.loads(line)
        sec = float(obj["audio_t1"]) / 16000.0
        if obj.get("event") == "start" and not in_seg:
            cur = sec
            in_seg = True
        elif obj.get("event") == "end" and in_seg:
            out.append((cur, sec))
            in_seg = False
    return out


def parse_gt(p: Path) -> list[dict]:
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(
            {
                "start": float(obj["start_ms"]) / 1000.0,
                "end": float(obj["end_ms"]) / 1000.0,
                "speaker": str(obj["speaker"]),
                "dur": float(obj["duration_ms"]) / 1000.0,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--timeline",
        default="logs/timeline/tl_20260426_182711.jsonl",
    )
    ap.add_argument(
        "--gt",
        default="tests/fixtures/test_ground_truth_v1.jsonl",
    )
    ap.add_argument("--max-sec", type=float, default=600.0)
    ap.add_argument("--min-overlap-sec", type=float, default=0.10)
    args = ap.parse_args()

    vad = parse_vad(Path(args.timeline))
    gt = parse_gt(Path(args.gt))
    gt = [g for g in gt if g["start"] < args.max_sec]
    print(
        f"[inputs] vad={len(vad)} gt={len(gt)} (start<{args.max_sec}s)"
    )

    # For each VAD interval, collect overlapping GT speakers (distinct names).
    vad_speakers: list[set[str]] = [set() for _ in vad]
    for g in gt:
        for vi, (vs, ve) in enumerate(vad):
            ov = min(g["end"], ve) - max(g["start"], vs)
            if ov >= args.min_overlap_sec:
                vad_speakers[vi].add(g["speaker"])

    # Classify each GT utterance.
    counts: Counter[str] = Counter()
    per_spk: dict[str, Counter[str]] = defaultdict(Counter)
    multi_vad_examples = []
    shared_examples = []
    no_vad_examples = []

    for g in gt:
        ovs = []
        for vi, (vs, ve) in enumerate(vad):
            ov = min(g["end"], ve) - max(g["start"], vs)
            if ov >= args.min_overlap_sec:
                ovs.append((vi, ov))
        if not ovs:
            cls = "no_vad"
            if len(no_vad_examples) < 5:
                no_vad_examples.append(g)
        elif len(ovs) > 1:
            cls = "multi_vad"
            if len(multi_vad_examples) < 5:
                multi_vad_examples.append((g, ovs))
        else:
            vi = ovs[0][0]
            n = len(vad_speakers[vi])
            if n <= 1:
                cls = "isolated"
            elif n == 2:
                cls = "shared_2"
                if len(shared_examples) < 8:
                    shared_examples.append((g, vi, vad_speakers[vi]))
            else:
                cls = "shared_3+"
                if len(shared_examples) < 8:
                    shared_examples.append((g, vi, vad_speakers[vi]))
        counts[cls] += 1
        per_spk[g["speaker"]][cls] += 1

    print("\n=== Overall coverage class ===")
    total = sum(counts.values())
    for k in ["isolated", "shared_2", "shared_3+", "multi_vad", "no_vad"]:
        v = counts[k]
        print(f"  {k:>10}: {v:5d}  ({100.0*v/total:5.1f}%)")

    print("\n=== Per-speaker breakdown ===")
    print(
        f"{'speaker':10} {'tot':>4} {'isol':>5} {'sh2':>4} {'sh3+':>5} "
        f"{'mvad':>4} {'no_vad':>6}"
    )
    for spk, c in sorted(per_spk.items()):
        t = sum(c.values())
        print(
            f"{spk:10} {t:>4d} {c['isolated']:>5d} {c['shared_2']:>4d} "
            f"{c['shared_3+']:>5d} {c['multi_vad']:>4d} {c['no_vad']:>6d}"
        )

    if no_vad_examples:
        print("\n=== Examples of GT with NO overlapping VAD ===")
        for g in no_vad_examples:
            print(
                f"  [{g['start']:7.2f}-{g['end']:7.2f}s dur={g['dur']:.2f}s] "
                f"{g['speaker']}"
            )
    if shared_examples:
        print("\n=== Examples of GT in shared (multi-speaker) VAD ===")
        for g, vi, spks in shared_examples:
            vs, ve = vad[vi]
            print(
                f"  vad#{vi} [{vs:7.2f}-{ve:7.2f}s]  "
                f"speakers={sorted(spks)}  "
                f"this_gt[{g['start']:.2f}-{g['end']:.2f}]={g['speaker']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
