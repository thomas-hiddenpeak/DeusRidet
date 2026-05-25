#!/usr/bin/env python3
"""
Phase 13 — error cohort audit.

Joins per-seg prediction dump (orator_reclusterer_eval --dump-pred) with
the overlap-patch log (tests/fixtures/overlap_patch_v1.jsonl) and reports
statistics on the misclassified cohort:

  * overlap_ratio distribution (errors vs corrects)
  * segment-length distribution
  * GT speaker × pred cluster confusion
  * fraction of errors above selected overlap thresholds

Usage:
  python3 tools/audit_error_segs.py \
      --pred /tmp/pred_v1_s1800.jsonl \
      --patch tests/fixtures/overlap_patch_v1.jsonl \
      --speakers tests/fixtures/fused_v1.speakers.txt \
      [--out /tmp/error_audit_v1_s1800.json]
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_speakers(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        return json.loads(txt.replace("'", '"'))
    except Exception:
        return [s.strip() for s in txt.strip("[]").split(",")]


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(name, xs):
    if not xs:
        return f"  {name}: n=0"
    return (
        f"  {name}: n={len(xs)} "
        f"mean={statistics.mean(xs):.3f} "
        f"median={statistics.median(xs):.3f} "
        f"p25={percentile(xs, 0.25):.3f} "
        f"p75={percentile(xs, 0.75):.3f} "
        f"p90={percentile(xs, 0.90):.3f} "
        f"max={max(xs):.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    preds = load_jsonl(args.pred)
    patch = load_jsonl(args.patch)
    spks = load_speakers(args.speakers) or [f"spk{i}" for i in range(4)]

    # Index patch by (start_ms, end_ms) for robust join — pred rows carry
    # t_start/t_end in seconds. Patch rows are by GT seg idx; pred rows have
    # 'idx' that lines up with the fixture row order, which equals patch idx.
    patch_by_idx = {row["idx"]: row for row in patch}

    rows = []
    for p in preds:
        idx = p["idx"]
        seg_dur = p["t_end"] - p["t_start"]
        ov = patch_by_idx.get(idx, {})
        rows.append({
            "idx": idx,
            "t_center": p["t_center"],
            "dur": seg_dur,
            "gt": p["gt"],
            "pred": p["pred"],
            "mapped_gt": p["mapped_gt"],
            "correct": bool(p["correct"]),
            "overlap_ratio": float(ov.get("overlap_ratio", 0.0)),
            "separated": bool(ov.get("separated", False)),
        })

    n = len(rows)
    n_err = sum(1 for r in rows if not r["correct"])
    print(f"== {os.path.basename(args.pred)} ==")
    print(f"total scored = {n}, errors = {n_err}, "
          f"acc = {100.0 * (n - n_err) / max(n,1):.2f}%")

    err = [r for r in rows if not r["correct"]]
    ok  = [r for r in rows if     r["correct"]]

    print("\n[overlap_ratio]")
    print(summarize("errors  ", [r["overlap_ratio"] for r in err]))
    print(summarize("corrects", [r["overlap_ratio"] for r in ok]))

    print("\n[segment duration sec]")
    print(summarize("errors  ", [r["dur"] for r in err]))
    print(summarize("corrects", [r["dur"] for r in ok]))

    print("\n[overlap threshold counts on errors]")
    for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
        ne = sum(1 for r in err if r["overlap_ratio"] >= thr)
        no = sum(1 for r in ok  if r["overlap_ratio"] >= thr)
        ratio_err = 100.0 * ne / max(len(err), 1)
        ratio_ok  = 100.0 * no / max(len(ok),  1)
        print(f"  ov >= {thr:.2f}: err={ne}/{len(err)} ({ratio_err:5.2f}%)  "
              f"ok={no}/{len(ok)} ({ratio_ok:5.2f}%)")

    print("\n[short-segment counts on errors]")
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        ne = sum(1 for r in err if r["dur"] < d)
        no = sum(1 for r in ok  if r["dur"] < d)
        print(f"  dur <  {d:.1f}s: err={ne}/{len(err)} "
              f"({100.0*ne/max(len(err),1):5.2f}%)  "
              f"ok={no}/{len(ok)} ({100.0*no/max(len(ok),1):5.2f}%)")

    print("\n[per-GT speaker error breakdown]")
    by_gt = defaultdict(lambda: {"n": 0, "err": 0})
    for r in rows:
        s = by_gt[r["gt"]]
        s["n"] += 1
        if not r["correct"]:
            s["err"] += 1
    for gt in sorted(by_gt.keys()):
        s = by_gt[gt]
        name = spks[gt] if gt < len(spks) else f"spk{gt}"
        rate = 100.0 * s["err"] / max(s["n"], 1)
        print(f"  gt={gt} ({name:>6}): n={s['n']:3d}  err={s['err']:3d}  ({rate:5.2f}%)")

    print("\n[error confusion: gt -> mapped_pred]")
    conf = Counter()
    for r in err:
        conf[(r["gt"], r["mapped_gt"])] += 1
    for (gt, mp), c in sorted(conf.items(), key=lambda x: -x[1]):
        g_name = spks[gt] if 0 <= gt < len(spks) else f"spk{gt}"
        p_name = spks[mp] if 0 <= mp < len(spks) else (f"spk{mp}" if mp >= 0 else "UNMAPPED")
        print(f"  gt={gt}({g_name}) -> mapped={mp}({p_name})  n={c}")

    print("\n[errors with separator candidate] (overlap_ratio >= 0.20 in patch)")
    err_with_sep = [r for r in err if r["separated"]]
    print(f"  count = {len(err_with_sep)} / {len(err)} "
          f"({100.0*len(err_with_sep)/max(len(err),1):.2f}%)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "n_total": n,
                "n_err": n_err,
                "errors": err,
                "corrects_sample": ok[:50],
                "speakers": spks,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
