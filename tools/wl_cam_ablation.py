#!/usr/bin/env python3
"""
wl_cam_ablation.py — Ablate WL-ECAPA alone AND CAM+WL fusion strategies,
re-using the closed-set evaluation protocols from cam_ablation.py.

For each configuration runs:
  - WL-only (B streaming, EMA)
  - CAM-only (reference, same config)
  - dual-concat 384-D (emb = [CAM; WL], L2-normed)
  - score-fusion: sim = mean(cos_cam, cos_wl)
  - score-fusion: sim = min(cos_cam, cos_wl)
  - score-fusion: sim = w*cos_cam + (1-w)*cos_wl for w ∈ {0.3, 0.5, 0.7}

Computes the same macro/micro accuracy + per-speaker + duration-conditional
accuracy as cam_ablation. Drills down on the <1.5 s segments where CAM++
was shown (cam_diagnose.py) to fail hardest, to see which fusion helps.

Assumes cam_embeddings_v1.f32 and wl_embeddings_v1.f32 both exist, produced
from the same GT jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAM_F32_DEFAULT  = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.f32"
CAM_META_DEFAULT = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.meta.json"
WL_F32_DEFAULT   = ROOT / "tests" / "fixtures" / "wl_embeddings_v1.f32"
WL_META_DEFAULT  = ROOT / "tests" / "fixtures" / "wl_embeddings_v1.meta.json"
OUT_CSV_DEFAULT  = ROOT / "tests" / "fixtures" / "wl_cam_ablation_v1.csv"


def l2(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def load(cam_f32, cam_meta, wl_f32, wl_meta):
    cm = json.loads(Path(cam_meta).read_text())
    wm = json.loads(Path(wl_meta).read_text())
    assert cm["n_segments"] == wm["n_segments"]
    assert cm["strategies"] == wm["strategies"]
    segs = cm["segments"]
    for a, b in zip(cm["segments"], wm["segments"]):
        assert a["idx"] == b["idx"] and a["start_ms"] == b["start_ms"]
    n = cm["n_segments"]; s = len(cm["strategies"])
    c = np.fromfile(cam_f32, dtype=np.float32).reshape(n, s, 192)
    w = np.fromfile(wl_f32,  dtype=np.float32).reshape(n, s, 192)
    return c, w, segs, cm["strategies"]


# ---- library update policies (reuse from cam_ablation) --------------------

class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.vec = {}
    def get(self, s): return self.vec.get(s)
    def update(self, s, emb):
        e = l2(emb[None, :])[0]
        if s in self.vec:
            v = self.alpha * self.vec[s] + (1 - self.alpha) * e
            self.vec[s] = l2(v[None, :])[0]
        else:
            self.vec[s] = e


def run_protocol_b(embs_s, segs, speakers, threshold, alpha,
                   sim_fn=None, dual_mode="single"):
    """Protocol B streaming with EMA.
    embs_s: dict {cam: (N,D), wl: (N,D)} for dual modes, or (N,D) direct.
    sim_fn : if given, callable (cam_emb, wl_emb, cam_lib, wl_lib) -> similarity.
    """
    by_spk = {s: [] for s in speakers}
    for i, g in enumerate(segs):
        by_spk[g["speaker"]].append(i)
    for s in speakers:
        by_spk[s].sort(key=lambda i: segs[i]["start_ms"])

    if dual_mode in ("cam", "wl", "concat"):
        # Single library with combined embedding.
        if dual_mode == "cam":
            E = l2(embs_s["cam"])
        elif dual_mode == "wl":
            E = l2(embs_s["wl"])
        else:  # concat
            E = l2(np.concatenate([embs_s["cam"], embs_s["wl"]], axis=1))
        lib = EMA(alpha)
        enroll_ids = set()
        for s in speakers:
            ids = by_spk[s][:1]
            if not ids: continue
            lib.update(s, E[ids[0]])
            enroll_ids.update(ids)
        test_ids = sorted([i for i in range(len(segs)) if i not in enroll_ids],
                          key=lambda i: segs[i]["start_ms"])
        names = [s for s in speakers if lib.get(s) is not None]
        per_spk_correct = {s: 0 for s in names}
        per_spk_total   = {s: 0 for s in names}
        per_spk_dur = {s: {"short": [0, 0], "mid": [0, 0], "long": [0, 0]} for s in names}
        confusion = {}
        for i in test_ids:
            t = segs[i]["speaker"]
            if t not in per_spk_total: continue
            e = E[i]
            sims = [float(lib.get(n) @ e) for n in names]
            j = int(np.argmax(sims))
            pred = names[j] if sims[j] >= threshold else "__unk__"
            per_spk_total[t] += 1
            dms = segs[i]["duration_ms"]
            bucket = "short" if dms < 1500 else ("mid" if dms < 2500 else "long")
            per_spk_dur[t][bucket][1] += 1
            if pred == t:
                per_spk_correct[t] += 1
                per_spk_dur[t][bucket][0] += 1
                lib.update(t, e)
            confusion[(t, pred)] = confusion.get((t, pred), 0) + 1
    else:
        # Dual library + custom sim_fn.
        cam_E = l2(embs_s["cam"])
        wl_E  = l2(embs_s["wl"])
        cam_lib = EMA(alpha); wl_lib = EMA(alpha)
        enroll_ids = set()
        for s in speakers:
            ids = by_spk[s][:1]
            if not ids: continue
            cam_lib.update(s, cam_E[ids[0]])
            wl_lib.update(s,  wl_E[ids[0]])
            enroll_ids.update(ids)
        test_ids = sorted([i for i in range(len(segs)) if i not in enroll_ids],
                          key=lambda i: segs[i]["start_ms"])
        names = [s for s in speakers if cam_lib.get(s) is not None and wl_lib.get(s) is not None]
        per_spk_correct = {s: 0 for s in names}
        per_spk_total   = {s: 0 for s in names}
        per_spk_dur = {s: {"short": [0, 0], "mid": [0, 0], "long": [0, 0]} for s in names}
        confusion = {}
        for i in test_ids:
            t = segs[i]["speaker"]
            if t not in per_spk_total: continue
            ce = cam_E[i]; we = wl_E[i]
            sims = [sim_fn(ce, we, cam_lib.get(n), wl_lib.get(n)) for n in names]
            j = int(np.argmax(sims))
            pred = names[j] if sims[j] >= threshold else "__unk__"
            per_spk_total[t] += 1
            dms = segs[i]["duration_ms"]
            bucket = "short" if dms < 1500 else ("mid" if dms < 2500 else "long")
            per_spk_dur[t][bucket][1] += 1
            if pred == t:
                per_spk_correct[t] += 1
                per_spk_dur[t][bucket][0] += 1
                cam_lib.update(t, ce)
                wl_lib.update(t, we)
            confusion[(t, pred)] = confusion.get((t, pred), 0) + 1

    per_spk_acc = {s: (per_spk_correct[s]/per_spk_total[s]) if per_spk_total[s] else 0 for s in names}
    macro = float(np.mean(list(per_spk_acc.values())))
    total = sum(per_spk_total.values())
    micro = sum(per_spk_correct.values())/total if total else 0
    # Macro per duration bucket: mean over speakers of (correct/total in bucket)
    dur_macro = {}
    for b in ("short", "mid", "long"):
        vals = []
        for s in names:
            c, t = per_spk_dur[s][b]
            if t > 0: vals.append(c/t)
        dur_macro[b] = float(np.mean(vals)) if vals else 0.0
    return {
        "macro": macro, "micro": micro,
        "per_spk": per_spk_acc,
        "dur_macro": dur_macro,
        "test_count": total,
        "confusion": confusion,
    }


# ---- similarity fusion functions ------------------------------------------

def sim_mean(ce, we, cl, wl_):
    return 0.5 * (cl @ ce) + 0.5 * (wl_ @ we)

def sim_min(ce, we, cl, wl_):
    return float(min(cl @ ce, wl_ @ we))

def sim_weighted(w_cam):
    def f(ce, we, cl, wl_):
        return w_cam * (cl @ ce) + (1 - w_cam) * (wl_ @ we)
    return f


def intra_inter(E, segs, speakers, label):
    rng = np.random.default_rng(0)
    by = {s: np.array([i for i, g in enumerate(segs) if g["speaker"] == s]) for s in speakers}
    intra = []
    for s in speakers:
        ids = by[s]
        if len(ids) < 2: continue
        a = rng.choice(ids, 2000); b = rng.choice(ids, 2000)
        m = a != b
        intra.extend((E[a[m]] * E[b[m]]).sum(-1).tolist())
    inter = []
    ids = np.arange(len(segs))
    tries = 0
    while len(inter) < 5000 and tries < 40000:
        i, j = rng.choice(ids, 2, replace=False)
        if segs[i]["speaker"] != segs[j]["speaker"]:
            inter.append(float(E[i] @ E[j]))
        tries += 1
    a, b = np.array(intra), np.array(inter)
    sep = a.mean() - b.mean()
    d = sep / np.sqrt(0.5 * (a.var() + b.var())) if sep > 0 else 0.0
    print(f"  {label:20s}  intra {a.mean():.3f}±{a.std():.3f}  "
          f"inter {b.mean():.3f}±{b.std():.3f}  "
          f"gap {sep:+.3f}  Cohen's d {d:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34])
    ap.add_argument("--strategies", nargs="+",
                    default=["2.0s_center", "full"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.8, 0.9])
    ap.add_argument("--cam-f32",  type=Path, default=CAM_F32_DEFAULT)
    ap.add_argument("--cam-meta", type=Path, default=CAM_META_DEFAULT)
    ap.add_argument("--wl-f32",   type=Path, default=WL_F32_DEFAULT)
    ap.add_argument("--wl-meta",  type=Path, default=WL_META_DEFAULT)
    ap.add_argument("--out-csv",  type=Path, default=OUT_CSV_DEFAULT)
    args = ap.parse_args()

    cam, wl, segs, strategies = load(args.cam_f32, args.cam_meta,
                                     args.wl_f32,  args.wl_meta)
    speakers = sorted({g["speaker"] for g in segs})
    print(f"[data] {len(segs)} segs, speakers={speakers}")

    # Intra/inter distribution summary
    print("\n=== Separability by mode (strategy=2.0s_center) ===")
    si = strategies.index("2.0s_center")
    intra_inter(l2(cam[:, si, :]),         segs, speakers, "CAM-only")
    intra_inter(l2(wl[:, si, :]),          segs, speakers, "WL-only")
    intra_inter(l2(np.concatenate([cam[:, si, :], wl[:, si, :]], axis=1)),
                segs, speakers, "concat-384")

    rows = []
    MODES = [
        ("cam",       "single"),
        ("wl",        "single"),
        ("concat",    "single"),
        ("fuse_mean", "dual", sim_mean),
        ("fuse_min",  "dual", sim_min),
        ("fuse_w0.3", "dual", sim_weighted(0.3)),
        ("fuse_w0.5", "dual", sim_weighted(0.5)),
        ("fuse_w0.7", "dual", sim_weighted(0.7)),
    ]
    for strat in args.strategies:
        si = strategies.index(strat)
        for thr in args.thresholds:
            for alpha in args.alphas:
                for m in MODES:
                    name, kind = m[0], m[1]
                    sim_fn = m[2] if len(m) > 2 else None
                    E = {"cam": cam[:, si, :], "wl": wl[:, si, :]}
                    r = run_protocol_b(E, segs, speakers, thr, alpha,
                                       sim_fn=sim_fn,
                                       dual_mode=(name if kind == "single" else "dual"))
                    rows.append({
                        "mode": name, "strategy": strat,
                        "threshold": thr, "alpha": alpha,
                        "macro": r["macro"], "micro": r["micro"],
                        "acc_short(<1.5s)": r["dur_macro"]["short"],
                        "acc_mid(1.5-2.5s)": r["dur_macro"]["mid"],
                        "acc_long(>2.5s)": r["dur_macro"]["long"],
                        "test": r["test_count"],
                        **{f"acc_{s}": r["per_spk"].get(s, 0.0) for s in speakers},
                    })

    if rows:
        fields = ["mode", "strategy", "threshold", "alpha",
                  "macro", "micro",
                  "acc_short(<1.5s)", "acc_mid(1.5-2.5s)", "acc_long(>2.5s)",
                  "test"] + [f"acc_{s}" for s in speakers]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows: w.writerow(r)
    print(f"\n[csv] {len(rows)} rows -> {args.out_csv}")

    # Best per mode
    print("\n=== Best config per mode (by macro) ===")
    modes = set(r["mode"] for r in rows)
    summary = []
    for m in sorted(modes):
        sub = [r for r in rows if r["mode"] == m]
        b = max(sub, key=lambda r: r["macro"])
        summary.append((m, b))
        print(f"  {m:11s}  macro={b['macro']:.3f}  micro={b['micro']:.3f}  "
              f"thr={b['threshold']:.2f} α={b['alpha']:.2f} strat={b['strategy']}   "
              f"short={b['acc_short(<1.5s)']:.3f} "
              f"mid={b['acc_mid(1.5-2.5s)']:.3f} "
              f"long={b['acc_long(>2.5s)']:.3f}")

    # Top 10 overall
    rows.sort(key=lambda r: r["macro"], reverse=True)
    print("\n=== Top 10 overall ===")
    print(f"{'mode':11s} {'strat':12s} {'thr':>5s} {'α':>4s} "
          f"{'macro':>7s} {'micro':>7s} {'short':>7s} {'mid':>7s} {'long':>7s}")
    for r in rows[:10]:
        print(f"{r['mode']:11s} {r['strategy']:12s} "
              f"{r['threshold']:>5.2f} {r['alpha']:>4.2f} "
              f"{r['macro']:>7.3f} {r['micro']:>7.3f} "
              f"{r['acc_short(<1.5s)']:>7.3f} "
              f"{r['acc_mid(1.5-2.5s)']:>7.3f} "
              f"{r['acc_long(>2.5s)']:>7.3f}")


if __name__ == "__main__":
    sys.exit(main())
