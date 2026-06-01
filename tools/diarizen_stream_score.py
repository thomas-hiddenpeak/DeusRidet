#!/usr/bin/env python3
"""Streaming-stitch diarization scorer (mechanical only).

Reads ``raw_events.jsonl`` captured by ``tools/diarizen_live_score.py`` and
reconstructs the *live* label stream a listener would actually have seen from
the windowed ``speaker_diarize_partial`` broadcasts — i.e. the P2 stitched
labels, NOT the offline ``speaker_diarize_final`` reclustering.

Because consecutive windows overlap (window 30 s, period 10 s ⇒ 20 s overlap),
the honest streaming timeline keeps, for each window, only the *newly revealed*
tail past the furthest point any earlier window reached (the first window keeps
its full extent). That concatenation is non-overlapping and reflects what the
system commits over time.

Scoring is the canonical overlap-second + Hungarian/first-seen
``score_diarization`` imported verbatim from ``diarizen_live_score.py`` — the
only metric permitted by the project benchmarks rule. No macro-F1, fuzzy
matching, or edit distance is computed here.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diarizen_live_score import score_diarization, load_gt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="raw_events.jsonl path")
    ap.add_argument("--gt", required=True, help="ground-truth JSON path")
    ap.add_argument("--max-sec", type=float, default=600.0,
                    help="score only GT utterances within [0, max-sec]")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()

    partials = []
    with open(args.raw, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "speaker_diarize_partial":
                continue
            if not obj.get("ok"):
                continue
            partials.append(obj)

    if not partials:
        print("[stream] no speaker_diarize_partial events found", file=sys.stderr)
        return 2

    # Build the non-overlapping streaming timeline: each window contributes
    # only the region beyond the furthest end any earlier window reached.
    stream_segs = []
    revealed_to = 0.0
    n_windows = 0
    for p in partials:
        n_windows += 1
        segs = p.get("segments", [])
        win_end = 0.0
        for s in segs:
            win_end = max(win_end, float(s[1]))
        for s in segs:
            ss, se, lbl = float(s[0]), float(s[1]), str(s[2])
            cs = max(ss, revealed_to)
            if se > cs:
                stream_segs.append([cs, se, lbl])
        revealed_to = max(revealed_to, win_end)

    gt_rows = [g for g in load_gt(args.gt)
               if g["start_sec"] < args.max_sec]

    res = score_diarization(stream_segs, gt_rows)

    n_labels = len({s[2] for s in stream_segs})
    print(f"[stream] windows={n_windows}  stitched_segments={len(stream_segs)}  "
          f"distinct_labels={n_labels}  covered_to={revealed_to:.1f}s  "
          f"gt_utts={len(gt_rows)}")
    acc = res.get("accuracy", res.get("micro"))
    print(f"[stream] STREAMING accuracy={acc:.4f}  macro={res.get('macro'):.4f}  "
          f"coverage={res.get('coverage'):.4f}  "
          f"decided={res.get('n_decided')}/{res.get('n_gt')}")
    per = res.get("per_spk", {})
    for spk in sorted(per):
        print(f"    {spk}: {per[spk]:.3f}")
    print(f"[stream] first-seen labels mapped: {res.get('mapping')}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
