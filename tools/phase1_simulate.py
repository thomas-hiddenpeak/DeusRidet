#!/usr/bin/env python3
"""
phase1_simulate.py — Phase 1 streaming spectral re-cluster simulation.

Simulates: at "now" time T, a background re-clusterer runs spectral
clustering on the last W seconds of fused embeddings, then commits
labels for the rolling window. Global speaker IDs persist across
windows via Hungarian centroid matching.

This is the offline-PoC for Phase 1. End-to-end macro tells us what to
expect after C++ implementation.

Modes:
  prefix  — at end, run one spectral on ALL segs in [0, T_eval].
            Pure oracle ceiling (matches offline_ceiling.py).
  window  — rolling window of W sec, step S sec, persistent global IDs.
            Realistic Phase 1.

Inputs: tests/fixtures/{cam,wl}_embeddings_v1.{f32,meta.json}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
STRAT = ["full", "1.5s_center", "2.0s_center", "3.0s_center", "4.0s_center"]


def load_embeddings(prefix):
    meta = json.loads((FIX / f"{prefix}_embeddings_v1.meta.json").read_text())
    n = meta["n_segments"]
    dim = meta["dim"]
    arr = np.fromfile(FIX / f"{prefix}_embeddings_v1.f32", dtype=np.float32)
    return arr.reshape(n, len(STRAT), dim), meta["segments"]


def l2(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def macro_f1(gt, pred):
    gt_classes = np.unique(gt)
    pred_classes = np.unique(pred)
    cost = np.zeros((len(pred_classes), len(gt_classes)), dtype=np.int64)
    for i, p in enumerate(pred_classes):
        for j, g in enumerate(gt_classes):
            cost[i, j] = -np.sum((pred == p) & (gt == g))
    n = max(cost.shape)
    pad = np.zeros((n, n), dtype=np.int64)
    pad[: cost.shape[0], : cost.shape[1]] = cost
    ri, ci = linear_sum_assignment(pad)
    mapping = {}
    for r, c in zip(ri, ci):
        if r < len(pred_classes) and c < len(gt_classes):
            mapping[pred_classes[r]] = gt_classes[c]
    mapped = np.array([mapping.get(p, "_") for p in pred])
    f1s = {}
    for g in gt_classes:
        tp = np.sum((mapped == g) & (gt == g))
        fp = np.sum((mapped == g) & (gt != g))
        fn = np.sum((mapped != g) & (gt == g))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[g] = (
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
            int(tp),
            int(fp),
            int(fn),
        )
    macro = float(np.mean([v[0] for v in f1s.values()]))
    return macro, f1s


def spectral_cosine(X, K):
    Xn = l2(X)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = (sim + 1.0) * 0.5
    np.fill_diagonal(aff, 0.0)
    sc = SpectralClustering(
        n_clusters=K, affinity="precomputed", assign_labels="kmeans", random_state=0
    )
    return sc.fit_predict(aff)


def auto_K(X, K_max=6):
    """Eigengap on normalized Laplacian. Returns K in [2, K_max]."""
    Xn = l2(X)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = (sim + 1.0) * 0.5
    np.fill_diagonal(aff, 0.0)
    d = aff.sum(axis=1)
    dis = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(aff.shape[0]) - (dis[:, None] * aff * dis[None, :])
    ev = np.linalg.eigvalsh(L)
    ev.sort()
    if len(ev) <= 2:
        return 2
    # eigvals[0] ≈ 0 (connected component). Skip gap[0] which is huge by
    # construction; real candidate K starts at gap index >= 1.
    gaps = np.diff(ev[: K_max + 1])
    if len(gaps) <= 1:
        return 2
    K = int(np.argmax(gaps[1:])) + 2  # +1 for skip, +1 for K = idx+1
    return max(2, min(K, K_max))


def build_fused(cam, wl, strat_idx):
    cam_s = cam[:, strat_idx, :]
    wl_s = wl[:, strat_idx, :]
    fused = np.concatenate([l2(cam_s), l2(wl_s)], axis=1)
    return l2(fused)


def filter_subset(meta_segs, end_max_ms=None, dur_min_ms=None):
    idx = []
    for i, s in enumerate(meta_segs):
        if end_max_ms is not None and s["end_ms"] > end_max_ms:
            continue
        if dur_min_ms is not None and s["duration_ms"] < dur_min_ms:
            continue
        idx.append(i)
    return np.array(idx, dtype=np.int64)


def simulate_window(emb, gt, t_center_ms, W_sec, S_sec, K_mode="oracle", K=4, link_thr=0.55):
    """Rolling-window streaming simulation with persistent global IDs.

    For each tick t in steps of S sec:
      - take all segs whose center is in [t-W, t]
      - run spectral with K (oracle) or auto-K
      - Hungarian-match new clusters to global IDs by centroid similarity
      - commit labels for window-local segs (overwriting any tentative)
    Final labels are read out for ALL segs.
    """
    N = emb.shape[0]
    final_label = np.full(N, -1, dtype=np.int64)
    next_global = 0
    global_centroids = {}  # id -> running mean unit vector
    global_counts = {}  # id -> n exemplars

    t_max = int(t_center_ms.max())
    t_step = S_sec * 1000
    t_win = W_sec * 1000

    for t in range(t_step, t_max + t_step, t_step):
        mask = (t_center_ms > t - t_win) & (t_center_ms <= t)
        idx = np.where(mask)[0]
        if len(idx) < max(4, K + 1):
            continue
        X = emb[idx]
        if K_mode == "oracle":
            k_use = K
        else:
            k_use = auto_K(X, K_max=6)
        try:
            labels = spectral_cosine(X, k_use)
        except Exception:
            continue
        # compute local centroids
        local_ids = np.unique(labels)
        local_cents = {}
        for lid in local_ids:
            v = X[labels == lid].mean(axis=0)
            v = v / max(np.linalg.norm(v), 1e-12)
            local_cents[lid] = v

        # Hungarian match local -> global
        glob_ids = list(global_centroids.keys())
        if not glob_ids:
            # bootstrap
            mapping = {}
            for lid in local_ids:
                mapping[lid] = next_global
                global_centroids[next_global] = local_cents[lid]
                global_counts[next_global] = int(np.sum(labels == lid))
                next_global += 1
        else:
            # cost = -cos(local, glob)
            n_local = len(local_ids)
            n_glob = len(glob_ids)
            n_pad = max(n_local, n_glob)
            cost = np.zeros((n_pad, n_pad), dtype=np.float64)
            for i, lid in enumerate(local_ids):
                for j, gid in enumerate(glob_ids):
                    cost[i, j] = -float(local_cents[lid] @ global_centroids[gid])
            ri, ci = linear_sum_assignment(cost)
            mapping = {}
            for r, c in zip(ri, ci):
                if r >= n_local:
                    continue
                lid = local_ids[r]
                if c < n_glob and -cost[r, c] >= link_thr:  # match threshold
                    gid = glob_ids[c]
                    mapping[lid] = gid
                    # update centroid (running mean)
                    n_old = global_counts[gid]
                    n_new = int(np.sum(labels == lid))
                    v = (
                        global_centroids[gid] * n_old
                        + local_cents[lid] * n_new
                    )
                    v = v / max(np.linalg.norm(v), 1e-12)
                    global_centroids[gid] = v
                    global_counts[gid] = n_old + n_new
                else:
                    # new speaker
                    mapping[lid] = next_global
                    global_centroids[next_global] = local_cents[lid]
                    global_counts[next_global] = int(np.sum(labels == lid))
                    next_global += 1

        # commit labels for window segs
        for k, seg_i in enumerate(idx):
            final_label[seg_i] = mapping[labels[k]]

    return final_label, next_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-ms", type=int, default=1800_000)
    ap.add_argument("--dur-min-ms", type=int, default=0)
    ap.add_argument("--strat", default="full", choices=STRAT)
    ap.add_argument("--encoder", default="fused", choices=["cam", "wl", "fused"])
    ap.add_argument("--window-sec", type=int, default=120)
    ap.add_argument("--step-sec", type=int, default=30)
    ap.add_argument("--k-mode", default="oracle", choices=["oracle", "auto"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--link-thr", type=float, default=0.55)
    args = ap.parse_args()

    cam, segs = load_embeddings("cam")
    wl, _ = load_embeddings("wl")
    sidx = STRAT.index(args.strat)

    if args.encoder == "cam":
        emb_all = l2(cam[:, sidx, :])
    elif args.encoder == "wl":
        emb_all = l2(wl[:, sidx, :])
    else:
        emb_all = build_fused(cam, wl, sidx)

    sub = filter_subset(segs, end_max_ms=args.end_ms, dur_min_ms=args.dur_min_ms)
    emb = emb_all[sub]
    gt = np.array([segs[i]["speaker"] for i in sub])
    t_center = np.array(
        [(segs[i]["start_ms"] + segs[i]["end_ms"]) // 2 for i in sub],
        dtype=np.int64,
    )
    print(
        f"subset: n={len(sub)}  enc={args.encoder}  strat={args.strat}  "
        f"end_ms<={args.end_ms} dur_min={args.dur_min_ms}ms"
    )
    print(f"  gt dist: {dict(Counter(gt))}")

    # PREFIX mode = run one spectral on all segs (== oracle ceiling)
    print("\n[PREFIX / oracle K=4 — same as offline_ceiling.py]")
    lab = spectral_cosine(emb, args.k)
    macro, f1s = macro_f1(gt, lab)
    print(f"  macro={macro:.3f}")
    for k, v in f1s.items():
        print(f"    {k}: F1={v[0]:.3f}  tp={v[1]} fp={v[2]} fn={v[3]}")

    # WINDOW mode = streaming rolling-window simulation
    print(
        f"\n[WINDOW W={args.window_sec}s step={args.step_sec}s "
        f"K_mode={args.k_mode} K={args.k}]"
    )
    final, n_speakers = simulate_window(
        emb, gt, t_center, args.window_sec, args.step_sec, args.k_mode, args.k,
        link_thr=args.link_thr,
    )
    # remaining -1 (not covered by any window) — assign to a "noise" class
    n_uncovered = int(np.sum(final < 0))
    print(f"  speakers discovered: {n_speakers}  uncovered_segs: {n_uncovered}")
    # macro restricted to segs with a label assigned
    mask = final >= 0
    if mask.sum() > 0:
        macro, f1s = macro_f1(gt[mask], final[mask])
        print(f"  macro (decided only) = {macro:.3f}  n_decided={mask.sum()}")
        for k, v in f1s.items():
            print(f"    {k}: F1={v[0]:.3f}  tp={v[1]} fp={v[2]} fn={v[3]}")
    # macro including uncovered as wrong
    final_full = final.copy()
    final_full[~mask] = 9999
    macro_full, _ = macro_f1(gt, final_full)
    print(
        f"  macro (incl uncovered as wrong) = {macro_full:.3f}  "
        f"coverage={mask.mean():.3f}"
    )


if __name__ == "__main__":
    main()
