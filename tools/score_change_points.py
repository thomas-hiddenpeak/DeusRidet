#!/usr/bin/env python3
"""Step 19a — Score sliding-window CAM++ change-point candidates against GT.

Reads adjacent-window cosine sequences emitted by cam_change_point_probe and
the refined GT (`tests/fixtures/test_ground_truth_v1.jsonl`). For every long
VAD interval, identifies GT speaker boundaries that fall inside it and tries
to recover them by thresholding adj_cos (with local-minimum gating).

Outputs a per-threshold sweep with precision/recall/F1 plus a per-interval
breakdown to stdout. No file mutation, diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CandSeg:
    vad_idx: int
    start_sec: float
    end_sec: float
    centers: list[float]
    adj_cos: list[float]


@dataclass
class GtUtt:
    start_sec: float
    end_sec: float
    speaker: str


def load_candidates(p: Path) -> list[CandSeg]:
    out: list[CandSeg] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(
            CandSeg(
                vad_idx=int(obj["vad_idx"]),
                start_sec=float(obj["start_sec"]),
                end_sec=float(obj["end_sec"]),
                centers=[float(x) for x in obj["centers"]],
                adj_cos=[float(x) for x in obj["adj_cos"]],
            )
        )
    return out


def load_gt(p: Path) -> list[GtUtt]:
    out: list[GtUtt] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(
            GtUtt(
                start_sec=float(obj["start_ms"]) / 1000.0,
                end_sec=float(obj["end_ms"]) / 1000.0,
                speaker=str(obj["speaker"]),
            )
        )
    out.sort(key=lambda u: u.start_sec)
    return out


def gt_boundaries_in(
    seg: CandSeg, gt: list[GtUtt], edge_margin: float = 0.25
) -> list[tuple[float, str, str]]:
    """Return GT speaker-change time-points strictly inside the VAD interval.

    A boundary is the midpoint between two adjacent GT utterances with
    different speakers when *both* lie at least partly inside the interval and
    the midpoint sits at least `edge_margin` seconds from the VAD edges.
    """
    inside = [
        u for u in gt
        if u.end_sec > seg.start_sec + edge_margin
        and u.start_sec < seg.end_sec - edge_margin
    ]
    inside.sort(key=lambda u: u.start_sec)
    bounds: list[tuple[float, str, str]] = []
    for i in range(1, len(inside)):
        a, b = inside[i - 1], inside[i]
        if a.speaker == b.speaker:
            continue
        # Boundary placed at the midpoint between a.end and b.start (or their
        # overlap midpoint when interleaved).
        t = 0.5 * (max(a.end_sec, b.start_sec) + min(a.end_sec, b.start_sec))
        if seg.start_sec + edge_margin <= t <= seg.end_sec - edge_margin:
            bounds.append((t, a.speaker, b.speaker))
    return bounds


def predicted_change_points(
    seg: CandSeg, threshold: float
) -> list[tuple[float, float]]:
    """Return predicted (time_sec, sim) for each adj_cos[i] that is a local
    minimum AND falls below threshold."""
    cps: list[tuple[float, float]] = []
    n = len(seg.adj_cos)
    for i, s in enumerate(seg.adj_cos):
        if s >= threshold:
            continue
        left = seg.adj_cos[i - 1] if i > 0 else float("inf")
        right = seg.adj_cos[i + 1] if i + 1 < n else float("inf")
        # Strict local minimum (or equal-tied at boundaries)
        if s <= left and s <= right:
            # Predicted boundary lies between center[i] and center[i+1]
            t = 0.5 * (seg.centers[i] + seg.centers[i + 1])
            cps.append((t, s))
    return cps


def score(
    cands: list[CandSeg],
    gt: list[GtUtt],
    threshold: float,
    tol_sec: float,
) -> dict:
    tp = 0
    fp = 0
    fn = 0
    abs_errs: list[float] = []
    n_intervals_with_gt = 0
    n_intervals_recovered = 0
    n_total_gt_bounds = 0
    n_total_pred = 0

    for seg in cands:
        bounds = gt_boundaries_in(seg, gt)
        preds = predicted_change_points(seg, threshold)
        n_total_gt_bounds += len(bounds)
        n_total_pred += len(preds)
        if bounds:
            n_intervals_with_gt += 1
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        # Greedy by smallest time distance.
        pairs = []
        for pi, (pt, _) in enumerate(preds):
            for gi, (gt_t, _, _) in enumerate(bounds):
                d = abs(pt - gt_t)
                if d <= tol_sec:
                    pairs.append((d, pi, gi))
        pairs.sort()
        for d, pi, gi in pairs:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
            abs_errs.append(d)
            tp += 1
        fp += len(preds) - len(matched_pred)
        fn += len(bounds) - len(matched_gt)
        if bounds and matched_gt:
            n_intervals_recovered += 1

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "threshold": threshold,
        "tol_sec": tol_sec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "median_abs_err_sec": (
            statistics.median(abs_errs) if abs_errs else None
        ),
        "n_intervals_with_gt": n_intervals_with_gt,
        "n_intervals_recovered": n_intervals_recovered,
        "n_total_gt_bounds": n_total_gt_bounds,
        "n_total_pred": n_total_pred,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default="/tmp/cam_change_points_r3.jsonl",
        help="JSONL from cam_change_point_probe",
    )
    ap.add_argument(
        "--gt",
        default="tests/fixtures/test_ground_truth_v1.jsonl",
        help="Refined GT JSONL",
    )
    ap.add_argument(
        "--tol-sec",
        type=float,
        default=0.75,
        help="Time tolerance for a predicted CP to count as a TP",
    )
    ap.add_argument(
        "--thresholds",
        default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated cosine thresholds to sweep",
    )
    ap.add_argument(
        "--per-interval",
        action="store_true",
        help="Print per-interval breakdown at the best-F1 threshold",
    )
    args = ap.parse_args()

    cands = load_candidates(Path(args.candidates))
    gt = load_gt(Path(args.gt))
    print(
        f"[inputs] candidates={len(cands)} gt_utts={len(gt)} "
        f"tol={args.tol_sec}s"
    )

    rows = []
    for t_str in args.thresholds.split(","):
        t = float(t_str)
        rows.append(score(cands, gt, t, args.tol_sec))

    print("\n=== Threshold sweep ===")
    print(
        f"{'thresh':>7} {'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'prec':>6} {'rec':>6} {'F1':>6} {'med_err_s':>9} "
        f"{'cov_int':>8} {'gt_bnd':>7} {'pred':>6}"
    )
    for r in rows:
        me = (
            f"{r['median_abs_err_sec']:.3f}"
            if r["median_abs_err_sec"] is not None
            else "  n/a"
        )
        cov = f"{r['n_intervals_recovered']}/{r['n_intervals_with_gt']}"
        print(
            f"{r['threshold']:>7.2f} {r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
            f"{me:>9} {cov:>8} {r['n_total_gt_bounds']:>7d} "
            f"{r['n_total_pred']:>6d}"
        )

    best = max(rows, key=lambda r: r["f1"])
    print(
        f"\n[best] threshold={best['threshold']:.2f} "
        f"F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f}"
    )

    if args.per_interval:
        print(
            f"\n=== Per-interval breakdown @ thresh={best['threshold']:.2f} ==="
        )
        print(
            f"{'vad_idx':>7} {'start':>7} {'end':>7} {'dur':>5} "
            f"{'gt_b':>4} {'pred':>4} {'matched':>7}  details"
        )
        for seg in cands:
            bounds = gt_boundaries_in(seg, gt)
            preds = predicted_change_points(seg, best["threshold"])
            if not bounds and not preds:
                continue
            # Match greedy.
            pairs = []
            for pi, (pt, _) in enumerate(preds):
                for gi, (gt_t, _, _) in enumerate(bounds):
                    d = abs(pt - gt_t)
                    if d <= args.tol_sec:
                        pairs.append((d, pi, gi))
            pairs.sort()
            mp: set[int] = set()
            mg: set[int] = set()
            matched: list[tuple[float, float, float]] = []  # (gt_t, pred_t, err)
            for d, pi, gi in pairs:
                if pi in mp or gi in mg:
                    continue
                mp.add(pi)
                mg.add(gi)
                matched.append((bounds[gi][0], preds[pi][0], d))
            dur = seg.end_sec - seg.start_sec
            det_parts = []
            for gt_t, pred_t, e in matched:
                det_parts.append(f"GT@{gt_t:.2f}~pred@{pred_t:.2f}(Δ{e:.2f})")
            for gi, (gt_t, sa, sb) in enumerate(bounds):
                if gi in mg:
                    continue
                det_parts.append(f"MISS_GT@{gt_t:.2f}({sa}->{sb})")
            for pi, (pt, sim) in enumerate(preds):
                if pi in mp:
                    continue
                det_parts.append(f"FP@{pt:.2f}(sim={sim:.2f})")
            print(
                f"{seg.vad_idx:>7d} {seg.start_sec:>7.2f} {seg.end_sec:>7.2f} "
                f"{dur:>5.2f} {len(bounds):>4d} {len(preds):>4d} "
                f"{len(matched):>7d}  {'; '.join(det_parts)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
