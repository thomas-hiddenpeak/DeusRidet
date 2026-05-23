#!/usr/bin/env python3
"""
offline_ceiling.py — Phase 0 ceiling estimator.

Question: ignoring the online state machine entirely, what is the BEST
macro F1 the current encoders can achieve on tests/test.mp3?

If even an oracle offline clustering can't clear ~0.70, the encoders
themselves are the bottleneck and no amount of state-machine tuning
will save us. If oracle ≥ 0.70, then online sequential matching is
the bug and we should add a 2-pass offline re-cluster.

Inputs (already on disk):
  tests/fixtures/cam_embeddings_v1.f32     [n_seg, 5, 192]
  tests/fixtures/wl_embeddings_v1.f32      [n_seg, 5, 192]
  tests/fixtures/cam_embeddings_v1.meta.json   (GT labels per seg)

Outputs:
  prints macro / dec_macro / coverage for each (encoder × clusterer ×
  subset) cell.

Subsets:
  full     — all 1169 GT segments (60 min)
  s1800    — segments fully within [0, 1800000] ms (matches runtime eval)
  long     — duration_ms >= 1500 (where CAM++/WL-ECAPA are both stable)

Clusterers:
  AHC-K4       — agglomerative cosine, oracle K=4
  AHC-NME      — agglomerative cosine, NME-estimated K
  Spectral-K4  — spectral clustering on cosine affinity, oracle K=4

Encoder choices:
  cam_full     — CAM++ 'full' window only (matches runtime)
  wl_full      — WL-ECAPA 'full' window only
  fused_full   — concat CAM++ ⊕ WL-ECAPA, both L2-normed (runtime fusion)
  fused_2s     — concat at 2.0s_center window (runtime SI peek)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"

STRAT = ["full", "1.5s_center", "2.0s_center", "3.0s_center", "4.0s_center"]


def load_embeddings(prefix: str):
    meta = json.loads((FIX / f"{prefix}_embeddings_v1.meta.json").read_text())
    n = meta["n_segments"]
    dim = meta["dim"]
    arr = np.fromfile(FIX / f"{prefix}_embeddings_v1.f32", dtype=np.float32)
    arr = arr.reshape(n, len(STRAT), dim)
    return arr, meta["segments"]


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def macro_f1(gt_labels: np.ndarray, pred_labels: np.ndarray):
    """Hungarian-match predicted clusters to GT speakers, then macro F1.

    Returns (macro_f1, per_speaker_f1_dict, n_classes_matched)
    """
    gt_classes = np.unique(gt_labels)
    pred_classes = np.unique(pred_labels)
    # cost matrix: rows = pred clusters, cols = gt speakers, value = -overlap
    cost = np.zeros((len(pred_classes), len(gt_classes)), dtype=np.int64)
    for i, p in enumerate(pred_classes):
        for j, g in enumerate(gt_classes):
            cost[i, j] = -np.sum((pred_labels == p) & (gt_labels == g))
    # pad to square
    n = max(cost.shape)
    pad = np.zeros((n, n), dtype=np.int64)
    pad[: cost.shape[0], : cost.shape[1]] = cost
    row_ind, col_ind = linear_sum_assignment(pad)
    # mapping: pred_classes[i] -> gt_classes[j] if both indices valid
    mapping = {}
    for ri, ci in zip(row_ind, col_ind):
        if ri < len(pred_classes) and ci < len(gt_classes):
            mapping[pred_classes[ri]] = gt_classes[ci]
    mapped = np.array([mapping.get(p, -1) for p in pred_labels])
    # per-speaker F1
    f1s = {}
    for g in gt_classes:
        tp = np.sum((mapped == g) & (gt_labels == g))
        fp = np.sum((mapped == g) & (gt_labels != g))
        fn = np.sum((mapped != g) & (gt_labels == g))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s[g] = (f1, prec, rec, int(tp), int(fp), int(fn))
    macro = float(np.mean([v[0] for v in f1s.values()]))
    return macro, f1s


def ahc_cosine(X: np.ndarray, K: int) -> np.ndarray:
    # cosine distance via L2-normed -> Euclidean: equivalent ordering
    Xn = l2norm(X)
    # 1 - cos = 0.5 * ||a-b||^2 for unit vectors; AgglomerativeClustering supports cosine directly
    ac = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    )
    return ac.fit_predict(Xn)


def ahc_nme(X: np.ndarray, Kmax: int = 8) -> tuple[np.ndarray, int]:
    """NME-style auto-K: pick K maximizing eigengap of normalized Laplacian."""
    Xn = l2norm(X)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = (sim + 1.0) * 0.5  # map to [0,1]
    np.fill_diagonal(aff, 0.0)
    d = aff.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L_sym = np.eye(aff.shape[0]) - (d_inv_sqrt[:, None] * aff * d_inv_sqrt[None, :])
    eigvals = np.linalg.eigvalsh(L_sym)
    eigvals.sort()
    # eigengaps among first Kmax+1 eigenvalues
    gaps = np.diff(eigvals[: Kmax + 1])
    K = int(np.argmax(gaps)) + 1
    K = max(2, min(K, Kmax))
    labels = ahc_cosine(X, K)
    return labels, K


def spectral_K(X: np.ndarray, K: int) -> np.ndarray:
    Xn = l2norm(X)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = (sim + 1.0) * 0.5
    np.fill_diagonal(aff, 0.0)
    sc = SpectralClustering(
        n_clusters=K, affinity="precomputed", assign_labels="kmeans", random_state=0
    )
    return sc.fit_predict(aff)


def build_subsets(meta_segs):
    n = len(meta_segs)
    full_idx = np.arange(n)
    s1800_idx = np.array(
        [i for i, s in enumerate(meta_segs) if s["end_ms"] <= 1800_000]
    )
    long_idx = np.array(
        [i for i, s in enumerate(meta_segs) if s["duration_ms"] >= 1500]
    )
    long_s1800_idx = np.array(
        [
            i
            for i, s in enumerate(meta_segs)
            if s["duration_ms"] >= 1500 and s["end_ms"] <= 1800_000
        ]
    )
    return {
        "full_60m": full_idx,
        "s1800": s1800_idx,
        "long_ge1500ms": long_idx,
        "long_s1800": long_s1800_idx,
    }


def build_encoder_inputs(cam, wl, strat_idx: int):
    cam_s = cam[:, strat_idx, :]
    wl_s = wl[:, strat_idx, :]
    cam_n = l2norm(cam_s)
    wl_n = l2norm(wl_s)
    fused = np.concatenate([cam_n, wl_n], axis=1)
    fused_n = l2norm(fused)
    return {"cam": cam_n, "wl": wl_n, "fused": fused_n}


def run_cell(name: str, X: np.ndarray, gt: np.ndarray):
    out = {}
    # AHC oracle K=4
    lab = ahc_cosine(X, 4)
    macro, f1s = macro_f1(gt, lab)
    out["AHC-K4"] = (macro, 4, f1s)
    # Spectral K=4
    lab = spectral_K(X, 4)
    macro, f1s = macro_f1(gt, lab)
    out["Spectral-K4"] = (macro, 4, f1s)
    # NME auto-K
    lab, K = ahc_nme(X, Kmax=8)
    macro, f1s = macro_f1(gt, lab)
    out["AHC-NME"] = (macro, K, f1s)
    return out


def main():
    print("Loading embeddings…")
    cam, meta_segs_cam = load_embeddings("cam")
    wl, meta_segs_wl = load_embeddings("wl")
    assert len(meta_segs_cam) == len(meta_segs_wl)
    n = len(meta_segs_cam)
    print(f"  n_segments={n}  cam.shape={cam.shape}  wl.shape={wl.shape}")

    gt_str = np.array([s["speaker"] for s in meta_segs_cam])
    subsets = build_subsets(meta_segs_cam)
    for k, v in subsets.items():
        from collections import Counter
        print(f"  subset {k:16s}  n={len(v):4d}  dist={dict(Counter(gt_str[v]))}")

    # strategy choices: 'full' is the natural one matching runtime
    strat_choices = [("full", 0), ("2.0s_center", 2)]

    encoder_kinds = ["cam", "wl", "fused"]

    rows = []
    for strat_name, sidx in strat_choices:
        enc = build_encoder_inputs(cam, wl, sidx)
        for ek in encoder_kinds:
            X_all = enc[ek]
            for sub_name, sub_idx in subsets.items():
                X = X_all[sub_idx]
                gt = gt_str[sub_idx]
                if len(np.unique(gt)) < 2:
                    continue
                res = run_cell(f"{ek}/{strat_name}/{sub_name}", X, gt)
                for clust_name, (macro, K, f1s) in res.items():
                    rows.append(
                        (
                            strat_name,
                            ek,
                            sub_name,
                            clust_name,
                            K,
                            macro,
                            f1s,
                        )
                    )

    # print summary
    print()
    print(
        f"{'strat':12s} {'enc':6s} {'subset':16s} {'clust':12s} {'K':>2s}  {'macro':>6s}  per-spk F1"
    )
    print("-" * 110)
    for strat_name, ek, sub_name, clust_name, K, macro, f1s in rows:
        per = "  ".join(f"{k}:{v[0]:.3f}" for k, v in f1s.items())
        print(
            f"{strat_name:12s} {ek:6s} {sub_name:16s} {clust_name:12s} {K:>2d}  {macro:6.3f}  {per}"
        )

    # headline
    print()
    print("=== HEADLINE (sorted by macro on long_s1800) ===")
    head = [r for r in rows if r[2] == "long_s1800"]
    head.sort(key=lambda r: -r[5])
    for strat_name, ek, sub_name, clust_name, K, macro, f1s in head:
        print(f"  {ek:6s} {strat_name:12s} {clust_name:12s} K={K}  macro={macro:.3f}")


if __name__ == "__main__":
    main()
