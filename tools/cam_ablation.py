#!/usr/bin/env python3
"""
cam_ablation.py — CAM++ baseline ablation on closed-set speaker classification.

Reads embeddings produced by tools/cam_extract_embeddings:
  tests/fixtures/cam_embeddings_v1.f32       (N_seg x N_strat x 192 float32)
  tests/fixtures/cam_embeddings_v1.meta.json

Runs two protocols on every (window_strategy, threshold, update_policy) tuple:

  Protocol A — enrollment-only
    For each speaker, sort segments by start_ms, take first K as enrollment
    (mean-pool, L2-normalized). Remaining segments are test. Pure closed-set
    classification by argmax cosine. No library updates after enrollment.
    K defaults to 3 (configurable).

  Protocol B — streaming with updates
    Process all segments in global time order. Each segment is classified by
    argmax cosine against the current library. Apply the update policy to
    the matched entry.  *Cheat*: identity is always taken from GT, so "update
    policy" is evaluated in isolation from threshold failures. (We care here
    about whether updates help or hurt embedding stability, not about
    registration logic.)

Metrics:
  - macro-avg accuracy (mean per-speaker accuracy) — primary
  - micro-avg accuracy (over all test segments)
  - per-speaker accuracy
  - confusion matrix at best config

Also prints intra vs inter cosine distribution at one "best" config.

Per Step 2b of SAAS ablation plan. Non-trivial Python but no runtime
dependencies beyond numpy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F32  = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.f32"
META = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.meta.json"
OUT_CSV = ROOT / "tests" / "fixtures" / "cam_ablation_v1.csv"


def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def load_data():
    meta = json.loads(META.read_text())
    dim = meta["dim"]
    n = meta["n_segments"]
    strategies = meta["strategies"]
    raw = np.fromfile(F32, dtype=np.float32)
    assert raw.size == n * len(strategies) * dim, (
        f"size mismatch {raw.size} vs {n*len(strategies)*dim}")
    embs = raw.reshape(n, len(strategies), dim)
    segs = meta["segments"]
    return embs, segs, strategies


# ---- library update policies ----------------------------------------------

class LibReplace:
    def __init__(self, dim):
        self.vec: dict[str, np.ndarray] = {}
    def get(self, sid): return self.vec.get(sid)
    def update(self, sid, emb):
        self.vec[sid] = l2norm(emb)


class LibEMA:
    def __init__(self, dim, alpha=0.9):
        self.alpha = alpha
        self.vec: dict[str, np.ndarray] = {}
    def get(self, sid): return self.vec.get(sid)
    def update(self, sid, emb):
        if sid in self.vec:
            v = self.alpha * self.vec[sid] + (1 - self.alpha) * l2norm(emb)
            self.vec[sid] = l2norm(v)
        else:
            self.vec[sid] = l2norm(emb)


class LibTopK:
    """Keep the most recent K embeddings per speaker; return their mean."""
    def __init__(self, dim, k=3):
        self.k = k
        self.hist: dict[str, list[np.ndarray]] = {}
    def get(self, sid):
        h = self.hist.get(sid)
        if not h: return None
        m = np.mean(h, axis=0)
        return l2norm(m)
    def update(self, sid, emb):
        e = l2norm(emb)
        h = self.hist.setdefault(sid, [])
        h.append(e)
        if len(h) > self.k: h.pop(0)


UPDATE_POLICIES = {
    "replace":   lambda dim: LibReplace(dim),
    "ema_0.7":   lambda dim: LibEMA(dim, 0.7),
    "ema_0.8":   lambda dim: LibEMA(dim, 0.8),
    "ema_0.9":   lambda dim: LibEMA(dim, 0.9),
    "ema_0.95":  lambda dim: LibEMA(dim, 0.95),
    "ema_0.98":  lambda dim: LibEMA(dim, 0.98),
    "top_3":     lambda dim: LibTopK(dim, 3),
    "top_5":     lambda dim: LibTopK(dim, 5),
}


# ---- protocols ------------------------------------------------------------

def protocol_a(embs_s: np.ndarray, segs: list, speakers: list,
               threshold: float, enroll_k: int):
    """Enrollment-only. Returns per-spk (correct, tested) and confusion dict."""
    # Group segment indices by speaker, sorted by start_ms.
    by_spk: dict[str, list[int]] = {s: [] for s in speakers}
    for i, g in enumerate(segs):
        by_spk[g["speaker"]].append(i)
    for s in speakers:
        by_spk[s].sort(key=lambda i: segs[i]["start_ms"])

    # Enroll: mean of first K segments' embeddings (L2-normed).
    lib = {}
    test_ids = []
    for s in speakers:
        ids = by_spk[s]
        if len(ids) < enroll_k + 1:
            continue
        lib[s] = l2norm(np.mean(embs_s[ids[:enroll_k]], axis=0))
        test_ids.extend((s, i) for i in ids[enroll_k:])

    if not lib or not test_ids:
        return None

    names = list(lib.keys())
    matrix = np.stack([lib[n] for n in names], axis=0)  # (S, D)

    # Classify.
    per_spk_correct = {s: 0 for s in names}
    per_spk_total   = {s: 0 for s in names}
    confusion: dict[tuple[str, str], int] = {}
    unknown_per_spk = {s: 0 for s in names}
    for true_spk, i in test_ids:
        if true_spk not in per_spk_total: continue
        e = l2norm(embs_s[i])
        sims = matrix @ e
        j = int(np.argmax(sims))
        max_sim = float(sims[j])
        pred = names[j] if max_sim >= threshold else "__unk__"
        per_spk_total[true_spk] += 1
        if pred == true_spk:
            per_spk_correct[true_spk] += 1
        elif pred == "__unk__":
            unknown_per_spk[true_spk] += 1
        confusion[(true_spk, pred)] = confusion.get((true_spk, pred), 0) + 1

    per_spk_acc = {s: (per_spk_correct[s] / per_spk_total[s])
                   if per_spk_total[s] else 0.0 for s in names}
    macro = float(np.mean(list(per_spk_acc.values())))
    total_test = sum(per_spk_total.values())
    total_correct = sum(per_spk_correct.values())
    micro = total_correct / total_test if total_test else 0.0
    unk_rate = sum(unknown_per_spk.values()) / total_test if total_test else 0.0
    return {
        "macro": macro, "micro": micro, "unk_rate": unk_rate,
        "per_spk": per_spk_acc, "confusion": confusion,
        "test_count": total_test,
    }


def protocol_b(embs_s: np.ndarray, segs: list, speakers: list,
               threshold: float, policy_name: str, enroll_k: int = 1):
    """Streaming with update policy. Same enrollment bootstrap as A(K=1),
    then stream: classify, update lib entry per policy."""
    dim = embs_s.shape[1]
    by_spk: dict[str, list[int]] = {s: [] for s in speakers}
    for i, g in enumerate(segs):
        by_spk[g["speaker"]].append(i)
    for s in speakers:
        by_spk[s].sort(key=lambda i: segs[i]["start_ms"])

    # Enroll: first K segs per speaker in time order. Initialize lib.
    enroll_ids: set[int] = set()
    lib = UPDATE_POLICIES[policy_name](dim)
    for s in speakers:
        ids = by_spk[s][:enroll_k]
        if len(ids) < enroll_k:
            continue
        mean = l2norm(np.mean(embs_s[ids], axis=0))
        lib.update(s, mean)
        enroll_ids.update(ids)

    # Stream remaining segments in global time order.
    test_ids = sorted([i for i in range(len(segs)) if i not in enroll_ids],
                      key=lambda i: segs[i]["start_ms"])

    names = [s for s in speakers if lib.get(s) is not None]
    if not names: return None

    per_spk_correct = {s: 0 for s in names}
    per_spk_total   = {s: 0 for s in names}
    confusion: dict[tuple[str, str], int] = {}
    unknown_per_spk = {s: 0 for s in names}

    for i in test_ids:
        true_spk = segs[i]["speaker"]
        if true_spk not in per_spk_total: continue
        e = l2norm(embs_s[i])
        sims = [float(lib.get(n) @ e) for n in names]
        j = int(np.argmax(sims))
        max_sim = sims[j]
        pred = names[j] if max_sim >= threshold else "__unk__"
        per_spk_total[true_spk] += 1
        if pred == true_spk:
            per_spk_correct[true_spk] += 1
            # Update only on correct match (typical streaming policy).
            lib.update(true_spk, embs_s[i])
        elif pred == "__unk__":
            unknown_per_spk[true_spk] += 1
        confusion[(true_spk, pred)] = confusion.get((true_spk, pred), 0) + 1

    per_spk_acc = {s: (per_spk_correct[s] / per_spk_total[s])
                   if per_spk_total[s] else 0.0 for s in names}
    macro = float(np.mean(list(per_spk_acc.values())))
    total_test = sum(per_spk_total.values())
    total_correct = sum(per_spk_correct.values())
    micro = total_correct / total_test if total_test else 0.0
    unk_rate = sum(unknown_per_spk.values()) / total_test if total_test else 0.0
    return {
        "macro": macro, "micro": micro, "unk_rate": unk_rate,
        "per_spk": per_spk_acc, "confusion": confusion,
        "test_count": total_test,
    }


def distrib_intra_inter(embs_s: np.ndarray, segs: list):
    """Pairwise cosine distributions within-speaker vs across-speaker.
    Samples at most 20k pairs per bucket to stay tractable."""
    rng = np.random.default_rng(0)
    by_spk: dict[str, list[int]] = {}
    for i, g in enumerate(segs):
        by_spk.setdefault(g["speaker"], []).append(i)

    E = l2norm(embs_s)
    intra, inter = [], []
    # Intra: sample pairs within each speaker
    for s, ids in by_spk.items():
        if len(ids) < 2: continue
        ids = np.array(ids)
        n_pairs = min(5000, len(ids) * (len(ids) - 1) // 2)
        a = rng.choice(ids, n_pairs)
        b = rng.choice(ids, n_pairs)
        mask = a != b
        sims = np.einsum("ij,ij->i", E[a[mask]], E[b[mask]])
        intra.extend(sims.tolist())
    # Inter: sample across different speakers
    all_ids = np.arange(len(segs))
    for _ in range(20000):
        i, j = rng.choice(all_ids, 2, replace=False)
        if segs[i]["speaker"] != segs[j]["speaker"]:
            inter.append(float(E[i] @ E[j]))
    return np.array(intra), np.array(inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    ap.add_argument("--enroll-k", type=int, default=3)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    embs, segs, strategies = load_data()
    speakers = sorted({g["speaker"] for g in segs})
    print(f"[data] {len(segs)} segments, {len(strategies)} strategies, "
          f"{len(speakers)} speakers: {speakers}")

    rows = []
    # Protocol A: window x threshold, no update policy needed
    for si, strat in enumerate(strategies):
        Es = embs[:, si, :]
        for thr in args.thresholds:
            r = protocol_a(Es, segs, speakers, thr, args.enroll_k)
            if r is None: continue
            rows.append({
                "protocol": f"A(K={args.enroll_k})",
                "strategy": strat, "threshold": thr, "update": "-",
                "macro": r["macro"], "micro": r["micro"],
                "unk_rate": r["unk_rate"], "test": r["test_count"],
                **{f"acc_{s}": r["per_spk"].get(s, 0.0) for s in speakers},
            })

    # Protocol B: window x threshold x update
    for si, strat in enumerate(strategies):
        Es = embs[:, si, :]
        for thr in args.thresholds:
            for pol in UPDATE_POLICIES:
                r = protocol_b(Es, segs, speakers, thr, pol, enroll_k=1)
                if r is None: continue
                rows.append({
                    "protocol": "B(stream)",
                    "strategy": strat, "threshold": thr, "update": pol,
                    "macro": r["macro"], "micro": r["micro"],
                    "unk_rate": r["unk_rate"], "test": r["test_count"],
                    **{f"acc_{s}": r["per_spk"].get(s, 0.0) for s in speakers},
                })

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = ["protocol", "strategy", "threshold", "update",
                  "macro", "micro", "unk_rate", "test"] + \
                 [f"acc_{s}" for s in speakers]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[csv] {len(rows)} rows -> {args.out_csv}")

    # Top-10 by macro
    rows.sort(key=lambda r: r["macro"], reverse=True)
    print("\n=== Top 10 configs by macro-avg accuracy ===")
    print(f"{'protocol':11s} {'strategy':12s} {'thr':>5s} {'update':>8s} "
          f"{'macro':>7s} {'micro':>7s} {'unk':>6s} " +
          "  ".join(f"{s:>6s}" for s in speakers))
    for r in rows[:10]:
        print(f"{r['protocol']:11s} {r['strategy']:12s} "
              f"{r['threshold']:>5.2f} {r['update']:>8s} "
              f"{r['macro']:>7.3f} {r['micro']:>7.3f} {r['unk_rate']:>6.3f} " +
              "  ".join(f"{r['acc_'+s]:>6.3f}" for s in speakers))

    # Per-strategy best (protocol A only for comparability)
    print("\n=== Best Protocol A per strategy ===")
    for strat in strategies:
        sub = [r for r in rows if r["protocol"].startswith("A") and r["strategy"] == strat]
        if not sub: continue
        b = max(sub, key=lambda r: r["macro"])
        print(f"  {strat:12s}  macro={b['macro']:.3f}  "
              f"thr={b['threshold']:.2f}  micro={b['micro']:.3f}  "
              f"unk={b['unk_rate']:.3f}")

    # Intra/inter cosine distribution on best strategy
    best_strat = max(strategies,
        key=lambda s: max((r["macro"] for r in rows
                           if r["strategy"] == s and r["protocol"].startswith("A")),
                          default=0.0))
    si = strategies.index(best_strat)
    intra, inter = distrib_intra_inter(embs[:, si, :], segs)
    print(f"\n=== Cosine distribution @ {best_strat} ===")
    print(f"  intra (same spk):  n={len(intra):5d}  "
          f"mean={intra.mean():.3f}  std={intra.std():.3f}  "
          f"p05={np.percentile(intra,5):.3f}  p50={np.percentile(intra,50):.3f}")
    print(f"  inter (diff spk):  n={len(inter):5d}  "
          f"mean={inter.mean():.3f}  std={inter.std():.3f}  "
          f"p95={np.percentile(inter,95):.3f}  p50={np.percentile(inter,50):.3f}")
    sep = intra.mean() - inter.mean()
    pooled = np.sqrt(0.5 * (intra.var() + inter.var()))
    d = sep / pooled if pooled > 0 else 0.0
    print(f"  Cohen's d (separability): {d:.2f}")

    # Confusion at overall best
    if rows:
        b = rows[0]
        print(f"\n=== Confusion matrix @ best config "
              f"({b['protocol']} / {b['strategy']} / thr={b['threshold']} / {b['update']}) ===")
        Es = embs[:, strategies.index(b["strategy"]), :]
        if b["protocol"].startswith("A"):
            res = protocol_a(Es, segs, speakers, b["threshold"], args.enroll_k)
        else:
            res = protocol_b(Es, segs, speakers, b["threshold"], b["update"], 1)
        conf = res["confusion"]
        labels = speakers + ["__unk__"]
        print("  true \\ pred  " + "  ".join(f"{l:>8s}" for l in labels))
        for t in speakers:
            row = [conf.get((t, p), 0) for p in labels]
            print(f"  {t:10s} " + "  ".join(f"{v:>8d}" for v in row))


if __name__ == "__main__":
    sys.exit(main())
