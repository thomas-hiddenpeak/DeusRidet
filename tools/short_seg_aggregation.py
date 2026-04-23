#!/usr/bin/env python3
"""
short_seg_aggregation.py — Step 4 offline study.

Hypothesis: the <1.5s macro ceiling (~0.77 from Step 2, ~0.44 with
current runtime gates per Step 3) is intrinsic to 192-D speaker
encoders fed too-short windows. Merging consecutive short segments
separated by a short silence into one aggregate might raise short-seg
accuracy without hurting long-seg accuracy — at the cost of resolving
speaker identity slightly more coarsely in time.

We simulate aggregation at the embedding-mean level (L2-normalized
average of per-segment embeddings). This is an approximation of
re-extracting the encoder on merged PCM: the real on-device change
would run the encoder once over the merged window, whose fbank context
differs slightly from the mean-of-embeddings proxy. The direction of
the effect is expected to match, but the magnitude is a ceiling
estimate.

Grouping rule (deliberately GT-blind):
  A segment joins the current group iff
    (a) prev_end exists in this group,
    (b) gap_ms = seg.start_ms - prev_end <= GAP_MS,
    (c) current group's aggregated duration is still short
        (< MAX_GROUP_MS), otherwise close the group.
  Seeds (the first segment in a group) can be either short or long —
  a long seed never triggers grouping because (c) fails immediately,
  so long segs pass through untouched. Short seeds absorb subsequent
  short neighbours until they hit MAX_GROUP_MS or the gap widens.

Metrics:
  - n_groups, n_singletons, encoder-call reduction
  - macro / short_macro / decided_macro under the Step 3 baseline vs
    the Step 3 'no_recency' vs 'all_off' upper-layer configs
  - "wrong-speaker-merge rate": fraction of groups spanning >1 GT
    speaker (pure diagnostic, the aggregator never sees GT)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Reuse Step 3 simulator + scorer.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import upper_layer_ablation as ula  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
F32  = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.f32"
META = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.meta.json"
OUT_CSV = ROOT / "tests" / "fixtures" / "short_seg_aggregation.csv"


def group_segments(segs, *, gap_ms, max_group_ms, short_ms):
    """Pure temporal grouping. Returns list of groups; each group is a
    list of segment indices ordered by time."""
    order = sorted(range(len(segs)), key=lambda i: segs[i]["start_ms"])
    groups = []
    cur = []
    cur_end = -1
    cur_start = -1
    for i in order:
        s = segs[i]
        dur = s["duration_ms"]
        if not cur:
            cur = [i]
            cur_start = s["start_ms"]
            cur_end = s["end_ms"]
            continue
        gap = s["start_ms"] - cur_end
        group_dur = cur_end - cur_start
        # Only extend groups whose seed + tail remain short.
        if (dur < short_ms and group_dur < max_group_ms and
                gap >= 0 and gap <= gap_ms):
            cur.append(i)
            cur_end = s["end_ms"]
        else:
            groups.append(cur)
            cur = [i]
            cur_start = s["start_ms"]
            cur_end = s["end_ms"]
    if cur:
        groups.append(cur)
    return groups


def l2(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def build_aggregated(embs_s, segs, groups):
    """Produce per-group synthetic embeddings + segment-like metadata.
    Each group inherits the time extent of its members, and its GT
    speaker is decided by majority (purely for scoring, never touched
    by the simulator)."""
    agg_embs = []
    agg_segs = []
    for g in groups:
        vecs = embs_s[g]
        mean = l2(np.mean(vecs, axis=0))
        agg_embs.append(mean)
        starts = [segs[i]["start_ms"] for i in g]
        ends   = [segs[i]["end_ms"]   for i in g]
        dur    = max(ends) - min(starts)
        # Majority speaker (diagnostic only for scoring).
        votes = {}
        for i in g:
            votes[segs[i]["speaker"]] = votes.get(segs[i]["speaker"], 0) + 1
        top_spk = max(votes.items(), key=lambda kv: kv[1])[0]
        agg_segs.append({
            "idx": len(agg_segs),
            "start_ms": min(starts),
            "end_ms": max(ends),
            "duration_ms": dur,
            "speaker": top_spk,
            "members": g,
            "n_members": len(g),
            "pure": len(votes) == 1,
        })
    return np.stack(agg_embs, axis=0), agg_segs


def run_cfg(agg_embs, agg_segs, *, match_thresh, **overrides):
    cfg = ula.base()
    cfg.update(overrides)
    preds, n_cluster = ula.run_stream(
        agg_embs, agg_segs, match_thresh=match_thresh,
        reg_thresh=ula.BASE_REG_THRESH, **cfg)
    mapping, sp = ula.first_seen_map(
        preds, [g["speaker"] for g in agg_segs], agg_segs, n_cluster)
    return preds, mapping, sp, n_cluster


def score_groups(preds, agg_segs, mapping, orig_segs, speakers, short_ms):
    """Project group-level predictions back onto per-segment GT, so
    macro is comparable with Step 3's original-stream numbers."""
    per_spk_correct = {s: 0 for s in speakers}
    per_spk_total   = {s: 0 for s in speakers}
    short_correct = 0
    short_total = 0
    decided_correct = 0
    decided_total = 0
    for gi, g in enumerate(agg_segs):
        p_cluster = preds[gi]
        pred_name = mapping.get(p_cluster, "__unk__") if p_cluster >= 0 else "__unk__"
        for mem in g["members"]:
            true = orig_segs[mem]["speaker"]
            if true not in per_spk_total:
                continue
            per_spk_total[true] += 1
            is_correct = (pred_name == true)
            if is_correct:
                per_spk_correct[true] += 1
            if p_cluster >= 0:
                decided_total += 1
                if is_correct:
                    decided_correct += 1
            if orig_segs[mem]["duration_ms"] < short_ms:
                short_total += 1
                if is_correct:
                    short_correct += 1
    per_spk_acc = {s: (per_spk_correct[s] / per_spk_total[s])
                   if per_spk_total[s] else 0.0 for s in speakers}
    macro = float(np.mean(list(per_spk_acc.values())))
    micro = sum(per_spk_correct.values()) / max(1, sum(per_spk_total.values()))
    decided_macro = decided_correct / max(1, decided_total)
    short_macro = short_correct / max(1, short_total)
    return macro, micro, short_macro, decided_macro, per_spk_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-ms", type=int, nargs="+",
                    default=[100, 200, 400, 800])
    ap.add_argument("--max-group-ms", type=int, default=3000)
    ap.add_argument("--short-ms", type=int, default=1500)
    ap.add_argument("--strategy", default="2.0s_center")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.45, 0.50])
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    embs, orig_segs, strategies = ula.load_data()
    si = strategies.index(args.strategy)
    Es = embs[:, si, :]
    speakers_all = sorted({g["speaker"] for g in orig_segs})

    # Upper-layer configs to compare (same semantics as Step 3).
    UL_CFGS = [
        ("baseline",    {}),
        ("no_recency",  dict(recency_window=0.0, recency_bonus=0.0)),
        ("all_off",     dict(discovery_count=0, discovery_boost=0.0,
                             recency_window=0.0, recency_bonus=0.0,
                             margin_abstain=0.0, max_auto_reg=0)),
    ]

    rows = []
    for gap in [0] + list(args.gap_ms):
        # gap=0 → pass-through (one group per segment).
        if gap == 0:
            groups = [[i] for i in sorted(
                range(len(orig_segs)),
                key=lambda i: orig_segs[i]["start_ms"])]
        else:
            groups = group_segments(
                orig_segs, gap_ms=gap,
                max_group_ms=args.max_group_ms,
                short_ms=args.short_ms)
        agg_embs, agg_segs = build_aggregated(Es, orig_segs, groups)
        n_groups = len(agg_segs)
        n_multi  = sum(1 for g in agg_segs if g["n_members"] > 1)
        n_impure = sum(1 for g in agg_segs if not g["pure"])
        reduction = 1.0 - n_groups / len(orig_segs)

        for thr in args.thresholds:
            for label, overrides in UL_CFGS:
                preds, mapping, sp, n_cluster = run_cfg(
                    agg_embs, agg_segs, match_thresh=thr, **overrides)
                macro, micro, shortm, dec, per_spk = score_groups(
                    preds, agg_segs, mapping, orig_segs, speakers_all,
                    args.short_ms)
                abstained = sum(1 for p in preds if p < 0)
                rows.append({
                    "gap_ms": gap,
                    "strategy": args.strategy, "threshold": thr,
                    "config": label,
                    "n_groups": n_groups, "n_multi": n_multi,
                    "n_impure": n_impure,
                    "reduction": round(reduction, 4),
                    "abstained_groups": abstained,
                    "macro": round(macro, 4),
                    "micro": round(micro, 4),
                    "short_macro": round(shortm, 4),
                    "decided_macro": round(dec, 4),
                    **{f"acc_{s}": round(per_spk.get(s, 0.0), 3)
                       for s in speakers_all},
                })

    # Write CSV.
    fields = ["gap_ms", "strategy", "threshold", "config",
              "n_groups", "n_multi", "n_impure", "reduction",
              "abstained_groups",
              "macro", "micro", "short_macro", "decided_macro"] + \
             [f"acc_{s}" for s in speakers_all]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] {len(rows)} rows -> {args.out_csv}")

    # Pretty-print: per config, gap-sweep summary.
    print(f"\n=== Gap sweep (strategy={args.strategy}, short<{args.short_ms}ms) ===")
    for thr in args.thresholds:
        for label, _ in UL_CFGS:
            print(f"\n-- thr={thr:.2f}  config={label} --")
            print(f"{'gap':>4s} {'groups':>6s} {'multi':>6s} {'impure':>6s} "
                  f"{'red%':>6s} {'abst':>4s} {'macro':>7s} {'short':>7s} "
                  f"{'decided':>8s}")
            for r in rows:
                if r["strategy"] != args.strategy: continue
                if r["threshold"] != thr: continue
                if r["config"] != label: continue
                print(f"{r['gap_ms']:>4d} {r['n_groups']:>6d} "
                      f"{r['n_multi']:>6d} {r['n_impure']:>6d} "
                      f"{r['reduction']*100:>5.1f}% "
                      f"{r['abstained_groups']:>4d} "
                      f"{r['macro']:>7.3f} {r['short_macro']:>7.3f} "
                      f"{r['decided_macro']:>8.3f}")


if __name__ == "__main__":
    main()
