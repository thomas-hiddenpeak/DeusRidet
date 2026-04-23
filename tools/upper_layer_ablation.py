#!/usr/bin/env python3
"""
upper_layer_ablation.py — Ablate the FULL-path upper-layer toggles in
src/sensus/auditus/audio_pipeline_process_saas_full.cpp against the
offline embedding fixtures.

Purpose (Step 3 of the evaluation roadmap): quantify whether each
upper-layer heuristic — discovery boost, recency bonus + recheck,
margin-abstain, max-auto-reg cap — helps or hurts macro accuracy
before we add structural changes (Step 4). Drops out the exemplar-bank
and hit-ratio policies (those live inside SpeakerVectorStore and would
require a C++ harness to simulate faithfully); we model the library as
EMA-over-matched-embeddings, which Step 2's Protocol-B Top-1 config
('ema_0.9') already proved is close to the ceiling for this data.

Pipeline (per config):
  1. Stream segments in ascending start_ms.
  2. For each embedding, compute cos-sim against every library centroid.
  3. Apply enabled toggles:
       discovery     — first N events: match_thresh += boost
       recency       — within window_sec of last match: match_thresh
                       -= bonus, auto_reg=false; if the lowered match
                       points to a different speaker, re-run at base
                       threshold (matches C++ recency validation).
       margin_abstain — if best - 2nd < margin, return -1 (no update).
       max_auto_reg  — after N registrations, auto_reg=false.
  4. On match: update the centroid with EMA(alpha=0.9).
     On no-match with auto_reg and best < reg_thresh: register new id.
     Else: abstain.
  5. After the whole stream, run Hungarian matching (SciPy-free, greedy
     as fallback) on the cluster-id × speaker count matrix to produce
     the best name mapping; compute macro/micro/short-seg metrics.

Output:
  tests/fixtures/upper_layer_ablation.csv — all configs
  stdout: top-5 configs and per-speaker/per-duration breakdowns

Deliberately cheap (<10 s). Short-segment bin: duration < 1500 ms.

NOTE (anti-overfit): test.mp3/test.txt is one conversation from one
environment; per the project directive we do NOT tune hyper-parameters
to best-on-this-dataset. We publish the grid and the winner, and we
only change defaults when (a) the winner beats the current config by
≥ 1.5 pp AND (b) per-speaker accuracy is not dragged down on any
speaker by > 2 pp.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F32  = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.f32"
META = ROOT / "tests" / "fixtures" / "cam_embeddings_v1.meta.json"
OUT_CSV = ROOT / "tests" / "fixtures" / "upper_layer_ablation.csv"

# Defaults pulled from configs/auditus.conf (runtime baseline).
BASE_MATCH_THRESH   = 0.50
BASE_REG_THRESH     = 0.60
DISCOVERY_COUNT     = 50
DISCOVERY_BOOST     = 0.07
RECENCY_WIN_SEC     = 15.0
RECENCY_BONUS       = 0.05
MARGIN_ABSTAIN      = 0.05
MAX_AUTO_REG        = 1000
EMA_ALPHA           = 0.9
SHORT_MS            = 1500


def l2(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def load_data():
    meta = json.loads(META.read_text())
    dim = meta["dim"]
    n = meta["n_segments"]
    strategies = meta["strategies"]
    raw = np.fromfile(F32, dtype=np.float32)
    embs = raw.reshape(n, len(strategies), dim)
    segs = meta["segments"]
    return embs, segs, strategies


# --- Simulator -------------------------------------------------------------

def run_stream(embs_s, segs, *,
               match_thresh=BASE_MATCH_THRESH,
               reg_thresh=BASE_REG_THRESH,
               discovery_count=DISCOVERY_COUNT,
               discovery_boost=DISCOVERY_BOOST,
               recency_window=RECENCY_WIN_SEC,
               recency_bonus=RECENCY_BONUS,
               margin_abstain=MARGIN_ABSTAIN,
               max_auto_reg=MAX_AUTO_REG,
               ema_alpha=EMA_ALPHA):
    """Run the simulated upper-layer FULL path. Returns predicted cluster
    IDs aligned with `segs` order (list of ints; -1 = abstained)."""
    order = sorted(range(len(segs)), key=lambda i: segs[i]["start_ms"])
    lib_names = []       # cluster id index → centroid vector
    lib_vecs  = []
    preds = [-1] * len(segs)
    event_count = 0
    reg_count = 0
    prev_time = -1e9
    prev_id   = -1

    for i in order:
        e = l2(embs_s[i].astype(np.float64))
        t_mid = 0.5 * (segs[i]["start_ms"] + segs[i]["end_ms"]) / 1000.0
        event_count += 1

        # Compute sims against current library.
        if lib_vecs:
            M = np.stack(lib_vecs, axis=0)
            sims = M @ e
            top = int(np.argmax(sims))
            best_sim = float(sims[top])
            if len(sims) > 1:
                rest = np.sort(sims)[-2]
                second_sim = float(rest)
            else:
                second_sim = 0.0
        else:
            top = -1
            best_sim = 0.0
            second_sim = 0.0

        # Threshold adjustments (the upper layer).
        mt = match_thresh
        if discovery_count > 0 and event_count <= discovery_count:
            mt += discovery_boost
        time_since_prev = t_mid - prev_time
        recency_active = (
            recency_window > 0 and
            prev_id >= 0 and
            time_since_prev < recency_window and
            event_count > discovery_count
        )
        if recency_active:
            mt -= recency_bonus

        auto_reg = (max_auto_reg <= 0 or reg_count < max_auto_reg)
        if recency_active:
            auto_reg = False  # matches C++ v24 recency policy

        pred = -1
        if top >= 0 and best_sim >= mt:
            # Recency validation: if lowered threshold caught a different
            # speaker than the recent one AND best < base_thresh, re-check.
            if (recency_active and top != prev_id and
                    best_sim < match_thresh):
                # Re-run at base threshold — i.e. require full confidence.
                if best_sim >= match_thresh:
                    pred = top
                # else abstain.
            else:
                pred = top
            # Margin gate (apply after match, mirrors FULL margin-abstain).
            if (pred >= 0 and margin_abstain > 0 and
                    len(lib_vecs) > 1 and
                    (best_sim - second_sim) < margin_abstain):
                pred = -1
        elif auto_reg and (top < 0 or best_sim < reg_thresh):
            # Register new cluster.
            lib_vecs.append(e.copy())
            lib_names.append(len(lib_vecs) - 1)
            pred = len(lib_vecs) - 1
            reg_count += 1

        # EMA update on confirmed match (not on new-register — that
        # already IS the centroid).
        if pred >= 0 and pred < len(lib_vecs) and lib_vecs[pred] is not e:
            v = ema_alpha * lib_vecs[pred] + (1 - ema_alpha) * e
            lib_vecs[pred] = l2(v)

        preds[i] = pred
        if pred >= 0:
            prev_time = t_mid
            prev_id = pred

    return preds, len(lib_vecs)


# --- Online "first-seen" cluster → speaker mapping ------------------------
#
# Rationale: during real operation, the system doesn't know GT speaker
# names. The consciousness fixes a cluster's identity the moment it is
# founded (the founder's embedding defines the centroid). Any subsequent
# segment assigned to that cluster inherits the founder's true speaker
# label for scoring purposes — even if the founder was wrong.
#
# This penalises BOTH over-registration (each extra cluster steals
# segments from the "correct" cluster for that speaker) AND false
# merges (a cluster founded by A that absorbs B's segments). It is
# closer to the actual closed-set semantics of diarization.

def first_seen_map(preds, gt_names, segs, n_clusters):
    speakers = sorted(set(gt_names))
    order = sorted(range(len(segs)), key=lambda i: segs[i]["start_ms"])
    mapping = {}
    for i in order:
        c = preds[i]
        if c < 0 or c in mapping:
            continue
        mapping[c] = gt_names[i]
    for c in range(n_clusters):
        mapping.setdefault(c, "__unk__")
    return mapping, speakers


def score(preds, segs, mapping, speakers, short_ms=SHORT_MS):
    per_spk_correct = {s: 0 for s in speakers}
    per_spk_total   = {s: 0 for s in speakers}
    # Decided accuracy: only over segments where a cluster was actually
    # predicted (pred >= 0). Lets us separate "the gate abstains a lot"
    # from "the prediction quality is bad".
    per_spk_decided_correct = {s: 0 for s in speakers}
    per_spk_decided_total   = {s: 0 for s in speakers}
    short_correct = 0
    short_total = 0
    for i, g in enumerate(segs):
        true = g["speaker"]
        if true not in per_spk_total:
            continue
        per_spk_total[true] += 1
        pred_cluster = preds[i]
        pred = mapping.get(pred_cluster, "__unk__") if pred_cluster >= 0 else "__unk__"
        is_correct = (pred == true)
        if is_correct:
            per_spk_correct[true] += 1
        if pred_cluster >= 0:
            per_spk_decided_total[true] += 1
            if is_correct:
                per_spk_decided_correct[true] += 1
        if g["duration_ms"] < short_ms:
            short_total += 1
            if is_correct:
                short_correct += 1
    per_spk_acc = {s: (per_spk_correct[s] / per_spk_total[s])
                   if per_spk_total[s] else 0.0 for s in speakers}
    macro = float(np.mean(list(per_spk_acc.values())))
    micro = sum(per_spk_correct.values()) / max(1, sum(per_spk_total.values()))
    per_spk_decided_acc = {
        s: (per_spk_decided_correct[s] / per_spk_decided_total[s])
        if per_spk_decided_total[s] else 0.0 for s in speakers}
    decided_macro = float(np.mean(list(per_spk_decided_acc.values())))
    short_macro = short_correct / max(1, short_total)
    return macro, micro, short_macro, decided_macro, per_spk_acc


# --- Config grid -----------------------------------------------------------

def base():
    return dict(
        discovery_count=DISCOVERY_COUNT,
        discovery_boost=DISCOVERY_BOOST,
        recency_window=RECENCY_WIN_SEC,
        recency_bonus=RECENCY_BONUS,
        margin_abstain=MARGIN_ABSTAIN,
        max_auto_reg=MAX_AUTO_REG,
    )

# Each config is a label + a dict of overrides on top of base().
CONFIGS = [
    ("baseline",            {}),
    ("no_discovery",        dict(discovery_count=0, discovery_boost=0.0)),
    ("no_recency",          dict(recency_window=0.0, recency_bonus=0.0)),
    ("no_margin",           dict(margin_abstain=0.0)),
    ("no_max_reg",          dict(max_auto_reg=0)),                          # unlimited
    ("all_off",             dict(discovery_count=0, discovery_boost=0.0,
                                 recency_window=0.0, recency_bonus=0.0,
                                 margin_abstain=0.0, max_auto_reg=0)),
    # Parameter sensitivities of potentially-keeper toggles.
    ("disc_0.03",           dict(discovery_boost=0.03)),
    ("disc_0.10",           dict(discovery_boost=0.10)),
    ("recency_0.02",        dict(recency_bonus=0.02)),
    ("recency_0.10",        dict(recency_bonus=0.10)),
    ("margin_0.02",         dict(margin_abstain=0.02)),
    ("margin_0.10",         dict(margin_abstain=0.10)),
]

# Which (strategy, threshold) to evaluate under. Step 2 winner was
# ('2.0s_center', thr=0.22 when EMA_0.8, but Protocol-B grid reported
# macro=0.833 at 2.0s_center with ema_0.9). We pin thr=0.50 (runtime
# config default) *and* also try 0.45/0.40 so we don't silently hide a
# lower threshold as the "real" upper-layer issue.
# NOTE: deliberately do NOT use thr=0.22 — that's the Step 2 oracle,
# tuned exactly on this dataset; using it here would defeat the point
# of ablation.
STRATEGIES = ["2.0s_center", "3.0s_center"]
THRESHOLDS = [0.40, 0.45, 0.50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    embs, segs, strategies = load_data()
    speakers_all = sorted({g["speaker"] for g in segs})
    n_short = sum(1 for g in segs if g["duration_ms"] < SHORT_MS)
    print(f"[data] {len(segs)} segments ({n_short} short <{SHORT_MS}ms), "
          f"{len(speakers_all)} speakers: {speakers_all}")

    rows = []
    for strat in STRATEGIES:
        si = strategies.index(strat)
        Es = embs[:, si, :]
        for thr in THRESHOLDS:
            for label, overrides in CONFIGS:
                cfg = base()
                cfg.update(overrides)
                preds, n_cluster = run_stream(
                    Es, segs, match_thresh=thr, reg_thresh=BASE_REG_THRESH,
                    **cfg)
                mapping, sp = first_seen_map(
                    preds, [g["speaker"] for g in segs], segs, n_cluster)
                macro, micro, shortm, decided_macro, per_spk = score(
                    preds, segs, mapping, sp)
                abstained = sum(1 for p in preds if p < 0)
                rows.append({
                    "strategy": strat, "threshold": thr, "config": label,
                    "n_clusters": n_cluster, "abstained": abstained,
                    "macro": round(macro, 4),
                    "micro": round(micro, 4),
                    "short_macro": round(shortm, 4),
                    "decided_macro": round(decided_macro, 4),
                    **{f"acc_{s}": round(per_spk.get(s, 0.0), 3)
                       for s in speakers_all},
                })

    # Write CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["strategy", "threshold", "config",
              "n_clusters", "abstained",
              "macro", "micro", "short_macro", "decided_macro"] + \
             [f"acc_{s}" for s in speakers_all]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] {len(rows)} rows -> {args.out_csv}")

    # Top-5.
    rows.sort(key=lambda r: r["macro"], reverse=True)
    print("\n=== Top 5 by macro ===")
    print(f"{'strategy':13s} {'thr':>5s} {'config':18s} "
          f"{'clu':>4s} {'abs':>4s} {'macro':>7s} {'micro':>7s} "
          f"{'short':>7s} {'decided':>8s}")
    for r in rows[:5]:
        print(f"{r['strategy']:13s} {r['threshold']:>5.2f} "
              f"{r['config']:18s} "
              f"{r['n_clusters']:>4d} {r['abstained']:>4d} "
              f"{r['macro']:>7.3f} {r['micro']:>7.3f} "
              f"{r['short_macro']:>7.3f} {r['decided_macro']:>8.3f}")

    # Per-config delta vs baseline at thr=0.50, 2.0s_center.
    print("\n=== Ablation deltas vs baseline (strategy=2.0s_center, thr=0.50) ===")
    pivot = {r["config"]: r for r in rows
             if r["strategy"] == "2.0s_center" and r["threshold"] == 0.50}
    if "baseline" in pivot:
        b = pivot["baseline"]
        print(f"{'config':18s} {'macro':>7s} {'Δ':>6s} "
              f"{'short':>7s} {'Δ':>6s} {'decided':>8s} "
              f"{'abstained':>10s}")
        for label, _ in CONFIGS:
            if label not in pivot:
                continue
            r = pivot[label]
            dm = r["macro"] - b["macro"]
            ds = r["short_macro"] - b["short_macro"]
            print(f"{label:18s} {r['macro']:>7.3f} {dm:>+6.3f} "
                  f"{r['short_macro']:>7.3f} {ds:>+6.3f} "
                  f"{r['decided_macro']:>8.3f} "
                  f"{r['abstained']:>10d}")


if __name__ == "__main__":
    main()
