#!/usr/bin/env python3
"""Step 19a — Cross-tab GT decision status × VAD coverage class.

Joins `matched.jsonl` (per-GT decision from online_replay_score.py) with the
VAD-coverage classifier output (isolated / shared_2 / shared_3+ / multi_vad /
no_vad). Reveals where the coverage gap actually lives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
                "idx": int(obj["idx"]),
                "start": float(obj["start_ms"]) / 1000.0,
                "end": float(obj["end_ms"]) / 1000.0,
                "speaker": str(obj["speaker"]),
                "dur": float(obj["duration_ms"]) / 1000.0,
            }
        )
    return out


def classify_gt(
    gt: list[dict], vad: list[tuple[float, float]], min_ov: float = 0.10
) -> dict[int, dict]:
    # vad_speakers[i] = set of GT speaker names overlapping vad i
    vad_speakers: list[set[str]] = [set() for _ in vad]
    for g in gt:
        for vi, (vs, ve) in enumerate(vad):
            ov = min(g["end"], ve) - max(g["start"], vs)
            if ov >= min_ov:
                vad_speakers[vi].add(g["speaker"])
    out: dict[int, dict] = {}
    for g in gt:
        ovs = []
        for vi, (vs, ve) in enumerate(vad):
            ov = min(g["end"], ve) - max(g["start"], vs)
            if ov >= min_ov:
                ovs.append(vi)
        if not ovs:
            cls = "no_vad"
        elif len(ovs) > 1:
            cls = "multi_vad"
        else:
            n = len(vad_speakers[ovs[0]])
            if n <= 1:
                cls = "isolated"
            elif n == 2:
                cls = "shared_2"
            else:
                cls = "shared_3+"
        out[g["idx"]] = {"class": cls, "n_vads": len(ovs)}
    return out


def load_matched(p: Path) -> list[dict]:
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True, help="matched.jsonl from replay")
    ap.add_argument("--timeline", required=True)
    ap.add_argument(
        "--gt", default="tests/fixtures/test_ground_truth_v1.jsonl"
    )
    ap.add_argument("--max-sec", type=float, default=600.0)
    args = ap.parse_args()

    vad = parse_vad(Path(args.timeline))
    gt = [g for g in parse_gt(Path(args.gt)) if g["start"] < args.max_sec]
    cls = classify_gt(gt, vad)
    matched = load_matched(Path(args.matched))
    # status: decided / abstain / no_segment
    # correctness: decided + (cluster maps to true speaker)

    # Build cluster->speaker first-seen mapping (same logic as scorer).
    mapping: dict[int, str] = {}
    by_idx = {m["gt_idx"]: m for m in matched}
    for m in matched:
        c = m["rt_cluster"]
        if c < 0:
            continue
        mapping.setdefault(c, m["gt_speaker"])

    # Cross-tab.
    classes = ["isolated", "shared_2", "shared_3+", "multi_vad", "no_vad"]
    statuses = ["decided_correct", "decided_wrong", "abstain", "no_segment"]
    grid: dict[str, Counter[str]] = {c: Counter() for c in classes}
    per_class_total: Counter[str] = Counter()

    for g in gt:
        c = cls[g["idx"]]["class"]
        per_class_total[c] += 1
        m = by_idx.get(g["idx"])
        if m is None:
            grid[c]["no_segment"] += 1
            continue
        st = m["status"]
        if st == "no_segment":
            grid[c]["no_segment"] += 1
        elif st == "abstain":
            grid[c]["abstain"] += 1
        else:
            pred = mapping.get(m["rt_cluster"])
            if pred == g["speaker"]:
                grid[c]["decided_correct"] += 1
            else:
                grid[c]["decided_wrong"] += 1

    total = sum(per_class_total.values())
    print(f"[inputs] gt={total} vad={len(vad)} matched={len(matched)}")
    print("\n=== Cross-tab: VAD-class × decision-status ===")
    print(
        f"{'class':>11} {'tot':>4} {'dec_ok':>7} {'dec_wr':>7} "
        f"{'absta':>6} {'no_seg':>7}  recover_target"
    )
    for c in classes:
        t = per_class_total[c]
        if t == 0:
            continue
        ok = grid[c]["decided_correct"]
        wr = grid[c]["decided_wrong"]
        ab = grid[c]["abstain"]
        ns = grid[c]["no_segment"]
        # "Recover target" = utterances that are currently lost but
        # technically still resolvable for this class.
        recover = ab + ns
        print(
            f"{c:>11} {t:>4d} {ok:>7d} {wr:>7d} {ab:>6d} {ns:>7d}  "
            f"{recover:>4d} (={100.0*recover/t:5.1f}% of class, "
            f"={100.0*recover/total:5.1f}% of all-GT)"
        )

    # Totals row.
    ok = sum(grid[c]["decided_correct"] for c in classes)
    wr = sum(grid[c]["decided_wrong"] for c in classes)
    ab = sum(grid[c]["abstain"] for c in classes)
    ns = sum(grid[c]["no_segment"] for c in classes)
    print(
        f"{'TOTAL':>11} {total:>4d} {ok:>7d} {wr:>7d} {ab:>6d} {ns:>7d}"
    )
    print(
        f"\n  current decided coverage = {100.0*(ok+wr)/total:.1f}% "
        f"({ok+wr}/{total})"
    )
    print(
        f"  current decided macro-acc on covered = "
        f"{100.0*ok/max(1,ok+wr):.1f}% ({ok}/{ok+wr})"
    )
    print(
        f"  if we could decide every isolated GT correctly, gain = +"
        f"{grid['isolated']['abstain']+grid['isolated']['no_segment']} GT "
        f"({100.0*(grid['isolated']['abstain']+grid['isolated']['no_segment'])/total:.1f}%)"
    )
    print(
        f"  if we could split every shared/multi_vad correctly, gain = +"
        f"{sum(grid[c]['abstain']+grid[c]['no_segment'] for c in ('shared_2','shared_3+','multi_vad'))} GT "
        f"({100.0*sum(grid[c]['abstain']+grid[c]['no_segment'] for c in ('shared_2','shared_3+','multi_vad'))/total:.1f}%)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
