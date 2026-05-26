#!/usr/bin/env python3
"""compute_accuracy.py — the only post-run number that matters.

Reads runtime_segments.json from a live `awaken` replay and
tests/fixtures/test_ground_truth.json, computes best-one-to-one mapping
between predicted ids and GT speakers (by overlap-second mass), and
emits a single `accuracy.json` next to the input plus a one-line
summary to stdout suitable for pasting into commit messages.

Output JSON shape:
{
  "task": "speaker-id 4-way",
  "audio": "tests/test.mp3",
  "n_segments": 1119,
  "n_gt_utterances": 556,
  "total_speech_sec": 3612.0,
  "speaker_mapping": {"石一": 7, "唐云峰": 3, ...},
  "per_speaker": [
    {"gt": "唐云峰", "speech_sec": 1115.0, "correct_sec": 171.0, "accuracy_pct": 15.3},
    ...
  ],
  "overall_accuracy_pct": 25.4
}

Stdout (last line, machine-readable):
  accuracy(tests/test.mp3, speaker-id 4-way): 25.4%

Usage:
  python3 tools/compute_accuracy.py <run_dir>
where <run_dir> contains runtime_segments.json.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def final_id(seg: dict) -> int:
    cur = seg["current_id"]
    for r in seg.get("relabel_chain", []):
        if r["old"] == cur:
            cur = r["new"]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--gt",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "tests/fixtures/test_ground_truth.json",
    )
    args = ap.parse_args()

    seg_path = args.run_dir / "runtime_segments.json"
    if not seg_path.exists():
        print(f"ERROR: {seg_path} not found", file=sys.stderr)
        return 2
    segs = json.loads(seg_path.read_text())
    gt = json.loads(args.gt.read_text())
    utt = gt["utterances"]
    speakers = gt["speakers"]

    spk_pred: dict[str, Counter[int]] = defaultdict(Counter)
    spk_time: Counter[str] = Counter()
    for u in utt:
        spk = u["speaker"]
        gs, ge = u["t0_start_sec"], u["t0_end_sec"]
        spk_time[spk] += ge - gs
        for s in segs:
            rs, re_ = s["start_sec"], s["end_sec"]
            ov = max(0.0, min(re_, ge) - max(rs, gs))
            if ov > 0:
                spk_pred[spk][final_id(s)] += ov

    # Greedy best one-to-one mapping by overlap mass.
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
    out = {
        "task": "speaker-id 4-way",
        "audio": "tests/test.mp3",
        "ground_truth": str(args.gt),
        "n_segments": len(segs),
        "n_gt_utterances": len(utt),
        "total_speech_sec": round(total_speech, 1),
        "speaker_mapping": spk_to_id,
        "per_speaker": per_speaker,
        "overall_accuracy_pct": overall,
    }
    (args.run_dir / "accuracy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )

    # Human-readable verdict.
    print(f"run: {args.run_dir}")
    for ps in per_speaker:
        print(
            f"  {ps['gt']:<8} id={ps['matched_pred_id']!s:<5} "
            f"{ps['correct_sec']:6.1f}s / {ps['speech_sec']:6.1f}s "
            f"= {ps['accuracy_pct']:5.1f}%"
        )
    # Last line: machine-readable.
    print(f"accuracy(tests/test.mp3, speaker-id 4-way): {overall}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
