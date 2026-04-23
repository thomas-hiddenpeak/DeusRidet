#!/usr/bin/env python3
"""
online_hybrid_ceiling.py — Step 5.1 "what if every short segment made
it to the encoder?" ceiling estimate.

Rationale:
  The Step 5 online run shows SAAS_FULL skips 54 % of GT segments —
  they never reach CAM++ at all. We want to know how much macro we
  could recover if the runtime simply gave each GT-sized interval an
  embedding (short-segment rescue) before passing it to the existing
  upper-layer gates. We do NOT want to tune parameters here; we just
  want an upper bound on the size of the prize.

Method (no runtime change):
  1. Load the Step 5 online capture (speaker_events.jsonl + timeline)
     exactly as online_replay_score.py does, producing the same
     (runtime_interval, cluster_id) list.
  2. For each GT segment, classify by overlap as decided / abstain /
     no_segment (reusing Step 5's logic).
  3. Replay the upper-layer simulator (tools/upper_layer_ablation.py)
     in time order across the FULL GT stream:
        - decided GT: feed the runtime-assigned cluster id straight
          into the mapping machinery (we keep the online identity
          choice, only the mapping attribution is re-derived).
        - abstain / no_segment GT: synthesize an 'as-if encoded'
          decision by pulling the fixture CAM++ embedding at a fixed
          strategy ('2.0s_center') and running the same EMA-centroid
          identify() the offline simulator uses. This is intentionally
          the simplest possible rescue — no new gates, no tuning.
  4. Report macro/micro with and without rescue, so the diff is the
     upper bound on what a runtime short-segment rescue could buy us.

Outputs:
  runs/<out>/hybrid_summary.json   — headline numbers
  runs/<out>/hybrid_matched.jsonl  — per-GT detail with rescue flag

The offline rescue uses the current runtime defaults (match=0.50,
reg=0.60, all upper gates ON) as the baseline config so the rescue
itself is not an 'all_off' oracle — we are measuring the ceiling at
the *current* upper-layer policy, not at an over-fit one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Re-use the online capture parser + pairer and the offline simulator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_replay_score import (   # noqa: E402
    parse_timeline_vad_segments,
    pair_segments_with_events,
)
import upper_layer_ablation as ula   # noqa: E402


def load_fixture(fixture_f32: Path, meta_json: Path, strategy: str):
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    n, s, dim = meta["n_segments"], len(meta["strategies"]), meta["dim"]
    arr = np.fromfile(fixture_f32, dtype=np.float32).reshape(n, s, dim)
    si = meta["strategies"].index(strategy)
    return meta, arr[:, si, :]  # [n, dim]


def gt_index_for_time(meta_segments, start_ms, end_ms):
    """Match an online GT row (start_ms,end_ms) to a fixture segment idx
    by exact boundary equality. The Step 5 GT jsonl is produced from
    the same segmenter fixture, so every row has a unique fixture idx."""
    for s in meta_segments:
        if s["start_ms"] == start_ms and s["end_ms"] == end_ms:
            return s["idx"]
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="runs/<online_stepNc_600s> from Step 5")
    ap.add_argument("--timeline", required=True,
                    help="logs/timeline/tl_*.jsonl for that run")
    ap.add_argument("--gt", default="tests/fixtures/test_ground_truth_v1.jsonl")
    ap.add_argument("--fixture", default="tests/fixtures/cam_embeddings_v1.f32")
    ap.add_argument("--fixture-meta",
                    default="tests/fixtures/cam_embeddings_v1.meta.json")
    ap.add_argument("--strategy", default="2.0s_center")
    ap.add_argument("--max-sec", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load online capture ------------------------------------------------
    evs = [
        json.loads(l) for l in
        (run_dir / "speaker_events.jsonl").read_text().splitlines() if l.strip()
    ]
    vads = parse_timeline_vad_segments(Path(args.timeline))
    if args.max_sec > 0:
        vads = [v for v in vads if v["end_sec"] <= args.max_sec + 2.0]
    runtime, pair_stats = pair_segments_with_events(vads, evs)

    # --- Load GT + fixture --------------------------------------------------
    gt = [
        json.loads(l) for l in
        Path(args.gt).read_text().splitlines() if l.strip()
    ]
    if args.max_sec > 0:
        cap_ms = int(args.max_sec * 1000)
        gt = [g for g in gt if g["end_ms"] <= cap_ms]
    print(f"[gt] {len(gt)} segments within {args.max_sec}s window")

    meta, fx = load_fixture(
        Path(args.fixture), Path(args.fixture_meta), args.strategy)
    idx_by_bounds = {(s["start_ms"], s["end_ms"]): s["idx"]
                     for s in meta["segments"]}

    # --- Per-GT classification against the runtime --------------------------
    runtime_sorted = sorted(runtime, key=lambda r: r["start_sec"])

    def best_rt(gs, ge):
        best, best_ov = None, 0.0
        for r in runtime_sorted:
            if r["end_sec"] <= gs:
                continue
            if r["start_sec"] >= ge:
                break
            ov = max(0.0, min(ge, r["end_sec"]) - max(gs, r["start_sec"]))
            if ov > best_ov:
                best_ov, best = ov, r
        return best, best_ov

    classified = []
    for g in gt:
        gs, ge = g["start_ms"] / 1000.0, g["end_ms"] / 1000.0
        rt, ov = best_rt(gs, ge)
        if rt is None:
            status = "no_segment"
            cluster = -1
        elif rt["cluster"] < 0:
            status = "abstain"
            cluster = -1
        else:
            status = "decided"
            cluster = rt["cluster"]
        classified.append({
            "gt_idx": g["idx"], "speaker": g["speaker"],
            "start_ms": g["start_ms"], "end_ms": g["end_ms"],
            "status": status, "online_cluster": cluster,
            "overlap": ov,
        })

    n_decided = sum(1 for c in classified if c["status"] == "decided")
    n_abstain = sum(1 for c in classified if c["status"] == "abstain")
    n_no_seg  = sum(1 for c in classified if c["status"] == "no_segment")
    print(f"[online] decided={n_decided} abstain={n_abstain} "
          f"no_segment={n_no_seg}")

    # --- Hybrid simulation --------------------------------------------------
    # Walk GT in time order. Maintain an EMA centroid library keyed by a
    # HYBRID cluster id so that online cluster IDs (integers >=0 from the
    # runtime) and rescued "offline" ids (new ints starting at 10000) do
    # not collide. For every GT row:
    #   decided: treat its online_cluster as the predicted id; also update
    #            the EMA using the fixture embedding for that row so
    #            rescued rows downstream see a consistent library.
    #   abstain / no_segment: run ula's upper-layer simulator's identify()
    #            against the same library at the baseline config.
    cfg = ula.base()  # current runtime defaults (all gates ON)
    ema_alpha    = ula.EMA_ALPHA
    match_thresh = ula.BASE_MATCH_THRESH
    reg_thresh   = ula.BASE_REG_THRESH
    disc_count   = cfg["discovery_count"]
    disc_boost   = cfg["discovery_boost"]
    rec_window   = cfg["recency_window"]
    rec_bonus    = cfg["recency_bonus"]
    margin_abs   = cfg["margin_abstain"]
    max_auto_reg = cfg["max_auto_reg"]

    library = []       # list of dict: {"id":int, "centroid":np.array, "hits":int}
    last_seen = {}     # id -> segment midpoint time (sec) of last match
    decisions = []     # per classified row
    next_offline_id = 10000
    rescued_count = 0
    rescued_ok    = 0
    auto_reg_used = 0

    def identify(emb, t_mid):
        # Current runtime 'identify' model — centroid cosine + discovery
        # boost + recency bonus + margin-abstain. Returns (cluster_id, ok).
        if not library:
            return None, None
        sims = []
        for entry in library:
            c = entry["centroid"]
            s = float(np.dot(emb, c))  # both L2-normalised
            # discovery boost for under-seen speakers
            if entry["hits"] < disc_count:
                s += disc_boost
            # recency bonus for recently seen
            if entry["id"] in last_seen:
                if t_mid - last_seen[entry["id"]] <= rec_window:
                    s += rec_bonus
            sims.append(s)
        order = np.argsort(sims)[::-1]
        top1 = sims[order[0]]
        top2 = sims[order[1]] if len(sims) > 1 else -1.0
        if top1 < match_thresh:
            return None, "below_match"
        if (top1 - top2) < margin_abs:
            return None, "margin"
        return library[order[0]], None

    for c in classified:
        gs = c["start_ms"] / 1000.0
        ge = c["end_ms"]   / 1000.0
        t_mid = 0.5 * (gs + ge)

        # Fixture embedding for this GT row (used by rescue *and* by
        # decided rows to keep the library's EMA consistent).
        fx_idx = idx_by_bounds.get((c["start_ms"], c["end_ms"]), -1)
        if fx_idx < 0:
            # Should not happen: GT rows and fixture rows share bounds.
            c["predicted"] = "__missing_fixture__"
            c["rescued"]   = False
            decisions.append(c)
            continue
        emb = fx[fx_idx].astype(np.float32)
        # L2 normalise (fixture may or may not be normalised — Step 3
        # does so; match that.)
        n = np.linalg.norm(emb) + 1e-9
        emb = emb / n

        if c["status"] == "decided":
            # Keep the online identity choice, but also refresh the
            # EMA library so rescues downstream compete fairly.
            cid = c["online_cluster"]
            entry = next((e for e in library if e["id"] == cid), None)
            if entry is None:
                library.append({"id": cid, "centroid": emb.copy(), "hits": 1})
            else:
                entry["centroid"] = (ema_alpha * emb +
                                     (1.0 - ema_alpha) * entry["centroid"])
                n = np.linalg.norm(entry["centroid"]) + 1e-9
                entry["centroid"] /= n
                entry["hits"] += 1
            last_seen[cid] = t_mid
            c["predicted_cluster"] = cid
            c["rescued"] = False
        else:
            # RESCUE: ask the library who this is.
            hit, reason = identify(emb, t_mid)
            if hit is not None:
                entry = hit
                entry["centroid"] = (ema_alpha * emb +
                                     (1.0 - ema_alpha) * entry["centroid"])
                n = np.linalg.norm(entry["centroid"]) + 1e-9
                entry["centroid"] /= n
                entry["hits"] += 1
                last_seen[entry["id"]] = t_mid
                c["predicted_cluster"] = entry["id"]
                c["rescue_reason"] = "matched"
                rescued_count += 1
            else:
                # Treat like the runtime would: register new speaker if
                # we still have budget, else abstain.
                if auto_reg_used < max_auto_reg:
                    new_id = next_offline_id
                    next_offline_id += 1
                    library.append({"id": new_id,
                                    "centroid": emb.copy(), "hits": 1})
                    last_seen[new_id] = t_mid
                    auto_reg_used += 1
                    c["predicted_cluster"] = new_id
                    c["rescue_reason"] = (reason or "new_register")
                    rescued_count += 1
                else:
                    c["predicted_cluster"] = -1
                    c["rescue_reason"] = "budget"
            c["rescued"] = True
        decisions.append(c)

    # --- Score --------------------------------------------------------------
    # first-seen mapping from the hybrid stream (online + rescued).
    mapping = {}
    for d in decisions:
        cid = d.get("predicted_cluster", -1)
        if cid >= 0:
            mapping.setdefault(cid, d["speaker"])

    speakers = sorted({d["speaker"] for d in decisions})
    totals  = {s: 0 for s in speakers}
    correct = {s: 0 for s in speakers}
    online_correct = {s: 0 for s in speakers}
    rescue_correct = {s: 0 for s in speakers}
    rescue_total   = {s: 0 for s in speakers}
    for d in decisions:
        s = d["speaker"]
        totals[s] += 1
        cid = d.get("predicted_cluster", -1)
        pred = mapping.get(cid, "__unk__") if cid >= 0 else "__unk__"
        ok = (pred == s)
        if ok:
            correct[s] += 1
            if d.get("rescued"):
                rescue_correct[s] += 1
            else:
                online_correct[s] += 1
        if d.get("rescued"):
            rescue_total[s] += 1
            if ok:
                rescued_ok += 1

    def macro(d_correct, d_total):
        vals = []
        for s in d_total:
            if d_total[s] > 0:
                vals.append(d_correct[s] / d_total[s])
        return sum(vals) / max(1, len(vals))

    per_spk = {s: (correct[s] / totals[s] if totals[s] else 0.0)
               for s in speakers}
    macro_all = macro(correct, totals)
    micro_all = sum(correct.values()) / max(1, sum(totals.values()))

    # Online-only (no rescue) baseline for context.
    # A row counts as 'online-correct' only if it was decided AND correct.
    online_total = {s: 0 for s in speakers}
    for d in decisions:
        if not d.get("rescued"):
            online_total[d["speaker"]] += 1
    macro_online = macro(online_correct, totals)  # denom = all GT
    macro_online_decided = macro(online_correct, online_total)

    rescue_per_spk_acc = {s: (rescue_correct[s] / rescue_total[s]
                              if rescue_total[s] else 0.0)
                          for s in speakers}

    print("\n=== Hybrid ceiling (online decided + offline rescue of short segs) ===")
    print(f"GT total                 : {sum(totals.values())}")
    print(f"online decided (kept)    : {sum(online_total.values())}")
    print(f"offline rescued          : {rescued_count}")
    print(f"coverage after rescue    : {(sum(online_total.values()) + rescued_count) / max(1, sum(totals.values())):.3f}")
    print(f"macro(all,online-only)   : {macro_online:.3f}")
    print(f"macro(all, hybrid)       : {macro_all:.3f}   <-- ceiling at current policy")
    print(f"micro(all, hybrid)       : {micro_all:.3f}")
    print(f"macro(rescue-only)       : "
          f"{macro(rescue_correct, rescue_total):.3f}   "
          f"(on rescued rows only)")
    print("per-speaker accuracy (hybrid):")
    for s in speakers:
        print(f"  {s}: total={totals[s]:3d} "
              f"rescued={rescue_total[s]:3d} "
              f"acc_all={per_spk[s]:.3f} "
              f"acc_rescue={rescue_per_spk_acc[s]:.3f}")
    print(f"first-seen mapping: {len(mapping)} cluster(s) -> "
          f"{len(set(mapping.values()))} unique name(s)")

    # Persist.
    with (out_dir / "hybrid_matched.jsonl").open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (out_dir / "hybrid_summary.json").write_text(json.dumps({
        "macro_all_online":  round(macro_online, 4),
        "macro_all_hybrid":  round(macro_all, 4),
        "micro_all_hybrid":  round(micro_all, 4),
        "macro_rescue_only": round(macro(rescue_correct, rescue_total), 4),
        "n_gt":       sum(totals.values()),
        "n_online":   sum(online_total.values()),
        "n_rescued":  rescued_count,
        "per_spk_hybrid":   {s: round(per_spk[s], 4) for s in speakers},
        "per_spk_rescue":   {s: round(rescue_per_spk_acc[s], 4)
                             for s in speakers},
        "mapping":    {str(k): v for k, v in mapping.items()},
        "config":     {k: cfg[k] for k in cfg},
        "strategy":   args.strategy,
        "pair_stats": pair_stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out_dir}/hybrid_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
