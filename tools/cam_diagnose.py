#!/usr/bin/env python3
"""
cam_diagnose.py — Diagnose why two speakers get confused by CAM++.

Analyses for a given speaker pair (default 石一 vs 唐云峰):

 1. Per-speaker centroid + variance (is one speaker much more scattered?)
 2. Pair cosine distribution: within-A, within-B, A-vs-B
 3. For every segment of speaker A, rank its cosine similarity to the
    centroids of all speakers; how often is B ranked #1 or #2?
 4. Time-series of mis-classifications: are errors clustered (suggesting
    drift) or uniformly distributed?
 5. Duration breakdown of mis-classifications: is the error concentrated
    on short segments?

Usage:
    python3 tools/cam_diagnose.py                # default 石一 vs 唐云峰
    python3 tools/cam_diagnose.py --pair A B     # any two speakers
    python3 tools/cam_diagnose.py --strategy full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F32  = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.f32"
META = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.meta.json"


def l2(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["石一", "唐云峰"])
    ap.add_argument("--strategy", default="2.0s_center")
    args = ap.parse_args()
    A, B = args.pair

    meta = json.loads(META.read_text())
    dim = meta["dim"]
    n   = meta["n_segments"]
    strategies = meta["strategies"]
    si = strategies.index(args.strategy)
    raw = np.fromfile(F32, dtype=np.float32)
    embs = raw.reshape(n, len(strategies), dim)[:, si, :]
    segs = meta["segments"]
    speakers = sorted({g["speaker"] for g in segs})

    E = l2(embs)
    by_spk = {s: np.array([i for i, g in enumerate(segs) if g["speaker"] == s])
              for s in speakers}
    for s in speakers:
        print(f"  {s:8s}: n={len(by_spk[s])}")
    print(f"Strategy: {args.strategy}\n")

    # --- 1. Centroid + variance -------------------------------------------
    print("=== 1. Centroid stats ===")
    centroids = {}
    for s in speakers:
        m = l2(E[by_spk[s]].mean(axis=0, keepdims=True))[0]
        # Mean cosine of each segment to own centroid
        sims = E[by_spk[s]] @ m
        centroids[s] = m
        print(f"  {s:8s}  self-sim mean={sims.mean():.3f}  "
              f"std={sims.std():.3f}  "
              f"p05={np.percentile(sims,5):.3f}  "
              f"p95={np.percentile(sims,95):.3f}")

    # --- 2. Pair cosine distribution --------------------------------------
    print(f"\n=== 2. Pair distributions: {A} vs {B} ===")
    rng = np.random.default_rng(0)
    def sample(pairs, n=5000):
        idx = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
        return pairs[idx]
    # Within A
    ia = by_spk[A]; ib = by_spk[B]
    AA = np.array([(i, j) for i in ia for j in ia if i < j])
    BB = np.array([(i, j) for i in ib for j in ib if i < j])
    AB = np.array([(i, j) for i in ia for j in ib])
    for label, pairs in [("intra-"+A, AA), ("intra-"+B, BB), (A+"-vs-"+B, AB)]:
        pairs = sample(pairs, 5000)
        sims = np.einsum("ij,ij->i", E[pairs[:, 0]], E[pairs[:, 1]])
        print(f"  {label:20s}  n={len(pairs):5d}  "
              f"mean={sims.mean():.3f}  std={sims.std():.3f}  "
              f"p05={np.percentile(sims,5):.3f}  "
              f"p95={np.percentile(sims,95):.3f}")

    # --- 3. Per-A rank against centroids ----------------------------------
    print(f"\n=== 3. For every segment of {A}, "
          f"where do the speakers rank by cosine? ===")
    cent_mat = np.stack([centroids[s] for s in speakers])      # (S, D)
    names = list(speakers)
    for target, ids in [(A, by_spk[A]), (B, by_spk[B])]:
        sims = E[ids] @ cent_mat.T                             # (N, S)
        # Top-1 speaker for each segment
        top1 = np.array([names[j] for j in sims.argmax(axis=1)])
        counts = {n: int((top1 == n).sum()) for n in names}
        # Rank of the "other" speaker (B if target==A, else A)
        other = B if target == A else A
        oi = names.index(other)
        ti = names.index(target)
        ranks = (-sims).argsort(axis=1)
        rank_of_other = np.array([list(r).index(oi) + 1 for r in ranks])
        rank_of_self  = np.array([list(r).index(ti) + 1 for r in ranks])
        gap = sims[:, ti] - sims[:, oi]
        print(f"  true={target}  (n={len(ids)})")
        print(f"    top-1 counts:            {counts}")
        print(f"    rank-of-{other:8s} mean={rank_of_other.mean():.2f}  "
              f"(1st:{(rank_of_other==1).sum()}, 2nd:{(rank_of_other==2).sum()}, "
              f"3rd:{(rank_of_other==3).sum()}, 4th:{(rank_of_other==4).sum()})")
        print(f"    gap(self - {other}): mean={gap.mean():+.3f}  "
              f"std={gap.std():.3f}  p05={np.percentile(gap,5):+.3f}  "
              f"frac(gap<0)={(gap<0).mean():.3f}")

    # --- 4. Time distribution of errors + 5. duration vs error ------------
    print(f"\n=== 4+5. Error vs time, error vs duration ({A} and {B}) ===")
    # Use centroid-based classifier (no streaming updates) for clean diagnosis
    for target, ids in [(A, by_spk[A]), (B, by_spk[B])]:
        sims = E[ids] @ cent_mat.T
        pred = np.array([names[j] for j in sims.argmax(axis=1)])
        wrong_mask = pred != target
        ts = np.array([segs[i]["start_ms"] / 1000 for i in ids])
        durs = np.array([segs[i]["duration_ms"] / 1000 for i in ids])
        if wrong_mask.any():
            print(f"  {target}: {wrong_mask.sum()}/{len(ids)} wrong "
                  f"({wrong_mask.mean():.1%})")
            # Error clustering: chi-square-like test by dividing time into quintiles
            t_bins = np.linspace(ts.min(), ts.max() + 1, 6)
            for k in range(5):
                m = (ts >= t_bins[k]) & (ts < t_bins[k+1])
                if m.any():
                    print(f"    time [{t_bins[k]:6.0f}s, {t_bins[k+1]:6.0f}s)  "
                          f"n={m.sum():4d}  err={wrong_mask[m].sum():4d}  "
                          f"rate={wrong_mask[m].mean():.1%}")
            # Duration-conditional error
            d_bins = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100.0]
            for k in range(len(d_bins)-1):
                m = (durs >= d_bins[k]) & (durs < d_bins[k+1])
                if m.any():
                    print(f"    dur [{d_bins[k]:.1f}s, {d_bins[k+1]:.1f}s)  "
                          f"n={m.sum():4d}  err={wrong_mask[m].sum():4d}  "
                          f"rate={wrong_mask[m].mean():.1%}")
        else:
            print(f"  {target}: all correct")

    # --- 6. Interpretation ---------------------------------------------------
    print("\n=== 6. Interpretation heuristics ===")
    # Is the pair-cosine overlap with intra-A or intra-B?
    AA_s = np.einsum("ij,ij->i", E[AA[:, 0]], E[AA[:, 1]])
    BB_s = np.einsum("ij,ij->i", E[BB[:, 0]], E[BB[:, 1]])
    AB_s = np.einsum("ij,ij->i", E[AB[:min(5000,len(AB)), 0]],
                                 E[AB[:min(5000,len(AB)), 1]])
    print(f"  intra-{A} mean {AA_s.mean():.3f} "
          f"vs {A}-{B} mean {AB_s.mean():.3f}: "
          f"gap = {AA_s.mean() - AB_s.mean():+.3f}")
    print(f"  intra-{B} mean {BB_s.mean():.3f} "
          f"vs {A}-{B} mean {AB_s.mean():.3f}: "
          f"gap = {BB_s.mean() - AB_s.mean():+.3f}")
    # Overlap fraction: intra-A samples that are below inter-AB p95
    ab_p95 = np.percentile(AB_s, 95)
    aa_below = (AA_s < ab_p95).mean()
    bb_below = (BB_s < ab_p95).mean()
    print(f"  fraction of intra-{A} pairs with sim < (A-B p95={ab_p95:.3f}): "
          f"{aa_below:.1%}")
    print(f"  fraction of intra-{B} pairs with sim < (A-B p95={ab_p95:.3f}): "
          f"{bb_below:.1%}")


if __name__ == "__main__":
    sys.exit(main())
