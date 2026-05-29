#!/usr/bin/env python3
"""offline_score.py — score a candidate prediction file with the same
overlap-second best-mapping logic as tools/compute_accuracy.py, so
candidate numbers are directly comparable to the live 31.0% baseline.

Input
-----
run_dir/predictions.jsonl  — one JSON per line:
    {"t0": <sec>, "t1": <sec>, "speaker_id": <int>}
  (speaker_id may be any integer; -1 means "unknown / abstain".)

Output
------
run_dir/accuracy.json      — same shape as live compute_accuracy.py
stdout last line           — accuracy(tests/test.mp3, speaker-id 4-way): X%
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_predictions(p: Path) -> list[dict]:
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--gt",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "tests/fixtures/test_ground_truth.json",
    )
    args = ap.parse_args()

    pred_path = args.run_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found", file=sys.stderr)
        return 2
    preds = load_predictions(pred_path)
    gt = json.loads(args.gt.read_text())
    utt = gt["utterances"]
    speakers = gt["speakers"]

    spk_pred: dict[str, Counter[int]] = defaultdict(Counter)
    spk_time: Counter[str] = Counter()
    for u in utt:
        spk = u["speaker"]
        gs, ge = u["t0_start_sec"], u["t0_end_sec"]
        spk_time[spk] += ge - gs
        for s in preds:
            rs, re_ = s["t0"], s["t1"]
            ov = max(0.0, min(re_, ge) - max(rs, gs))
            if ov > 0:
                spk_pred[spk][s["speaker_id"]] += ov

    triples = []
    for spk, ct in spk_pred.items():
        for pid, sec in ct.items():
            if pid == -1:
                continue
            triples.append((sec, spk, pid))
    triples.sort(reverse=True)
    spk_to_id: dict[str, int] = {}
    id_to_spk: dict[int, str] = {}
    for sec, spk, pid in triples:
        if spk in spk_to_id or pid in id_to_spk:
            continue
        spk_to_id[spk] = pid
        id_to_spk[pid] = spk

    per_speaker = []
    total_correct = 0.0
    total_speech = 0.0
    for spk in speakers:
        tot = spk_time.get(spk, 0.0)
        matched_id = spk_to_id.get(spk)
        sec = spk_pred[spk].get(matched_id, 0.0) if matched_id is not None else 0.0
        per_speaker.append({
            "gt": spk,
            "matched_pred_id": matched_id,
            "speech_sec": round(tot, 1),
            "correct_sec": round(sec, 1),
            "accuracy_pct": round(sec / tot * 100, 1) if tot > 0 else 0.0,
            "top_pred_overlaps": [
                {"id": pid, "sec": round(sc, 1)}
                for pid, sc in spk_pred[spk].most_common(6)
            ],
        })
        total_correct += sec
        total_speech += tot

    overall = round(total_correct / total_speech * 100, 1) if total_speech > 0 else 0.0

    # Diagnostic: optimal (Hungarian) one-to-one mapping for the ceiling.
    hungarian_pct = overall
    hungarian_map: dict[str, int] = dict(spk_to_id)
    try:
        from scipy.optimize import linear_sum_assignment
        pids = sorted({pid for ct in spk_pred.values() for pid in ct if pid != -1})
        if pids and speakers:
            import numpy as np
            cost = np.zeros((len(speakers), len(pids)), dtype=np.float64)
            for i, spk in enumerate(speakers):
                for j, pid in enumerate(pids):
                    cost[i, j] = -spk_pred[spk].get(pid, 0.0)
            ri, ci = linear_sum_assignment(cost)
            opt_map = {speakers[i]: pids[j] for i, j in zip(ri, ci)}
            opt_correct = sum(
                spk_pred[spk].get(pid, 0.0) for spk, pid in opt_map.items()
            )
            hungarian_pct = round(opt_correct / total_speech * 100, 1)
            hungarian_map = opt_map
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Hungarian unavailable: {exc}", file=sys.stderr)
    out = {
        "task": "speaker-id 4-way",
        "audio": "tests/test.mp3",
        "ground_truth": str(args.gt),
        "n_predictions": len(preds),
        "n_gt_utterances": len(utt),
        "total_speech_sec": round(total_speech, 1),
        "speaker_mapping": spk_to_id,
        "per_speaker": per_speaker,
        "overall_accuracy_pct": overall,
        "hungarian_accuracy_pct": hungarian_pct,
        "hungarian_mapping": hungarian_map,
    }
    (args.run_dir / "accuracy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )

    print(f"run: {args.run_dir}")
    for ps in per_speaker:
        print(
            f"  {ps['gt']:<8} id={ps['matched_pred_id']!s:<5} "
            f"{ps['correct_sec']:6.1f}s / {ps['speech_sec']:6.1f}s "
            f"= {ps['accuracy_pct']:5.1f}%"
        )
    print(f"accuracy(tests/test.mp3, speaker-id 4-way): {overall}%")
    print(f"  hungarian-ceiling: {hungarian_pct}%   map={hungarian_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
