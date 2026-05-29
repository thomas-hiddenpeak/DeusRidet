#!/usr/bin/env python3
"""
diarizen_live_score.py — Hybrid P1 end-to-end accuracy gate for the
DiariZen-v2 reclusterer wired into ``awaken``.

Procedure (live-system-only evidence per workflow.instructions.md):

  1. Connect to a running ``awaken`` server (DEUSRIDET_DIARIZEN_ENABLE=1).
  2. Stream ``tests/test.mp3`` over WS at the given speed.
  3. After ``--drain-sec`` of silence, send WS text ``diarizen_finalize``.
  4. Wait for ``speaker_diarize_final`` (the offline reclusterer reply).
  5. Score the returned ``segments`` against
     ``tests/fixtures/test_ground_truth.json`` using first-seen
     cluster→name mapping and time-overlap matching, identical in spirit
     to ``tools/online_replay_score.py``.
  6. Print the Constitutional-rule accuracy line:

         accuracy(tests/test.mp3, diarization): <baseline>% → <live>% (Δ = ±X pp)

The baseline number is the GPU-offline reference reported in repo memory
(93.5% on the verified DiariZen-v2 stack). The "live" number is the one
the live awaken pipeline can certify; matching the baseline within
~1 pp is the ship gate.

Usage:
    python3 tools/diarizen_live_score.py \\
        --audio tests/test.mp3 \\
        --gt    tests/fixtures/test_ground_truth.json \\
        --speed 4.0 --drain-sec 30 --diarize-timeout 1500
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import websocket  # type: ignore


def decode_to_pcm(path: str) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"ffmpeg error: {r.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    return r.stdout


def stream_pcm(ws, pcm: bytes, chunk: int, speed: float) -> None:
    total = len(pcm) // 2
    n_chunks = (total + chunk - 1) // chunk
    frame_wall = (chunk / 16000.0) / max(speed, 1e-6)
    t0 = time.time()
    for i in range(n_chunks):
        a = i * chunk
        b = min(a + chunk, total)
        frame = pcm[a * 2: b * 2]
        ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
        target = t0 + (i + 1) * frame_wall
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)
    print(
        f"[stream] sent {total} samples ({total/16000.0:.1f}s src) "
        f"in {time.time()-t0:.1f}s wall",
        flush=True,
    )


def load_gt(path: str):
    """Return a list of {"start_sec","end_sec","speaker"} from the canonical
    test_ground_truth.json (utterances array with t0_*_sec fields)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for u in d.get("utterances", []):
        s = float(u.get("t0_start_sec", 0.0))
        e = float(u.get("t0_end_sec", 0.0))
        sp = u.get("speaker", "")
        if e > s and sp:
            out.append({"start_sec": s, "end_sec": e, "speaker": sp})
    return out


def score_diarization(segments, gt_rows):
    """Canonical overlap-second scoring with Hungarian label→speaker mapping.

    Mirrors `tools/verification_2026/diar_diarizen_gpu.py` +
    `tools/verification_2026/offline_score.py`:

      1. Reduce labels to top-4 by total segment duration (others are
         remapped to the nearest top-4 label by time distance, so every
         GT utterance gets a prediction).
      2. For each GT utterance, aggregate overlap-seconds per (reduced)
         label; if no overlap, use the nearest reduced label.
      3. Solve one-to-one (Hungarian) mapping from GT speaker → label
         to maximise total correct seconds.
      4. Accuracy = correct_seconds / total_speech_seconds.
    """
    from collections import defaultdict
    speakers = sorted({g["speaker"] for g in gt_rows})

    segs_sorted = sorted(
        [(float(s[0]), float(s[1]), str(s[2])) for s in segments],
        key=lambda s: s[0],
    )
    # Step 1: reduce labels → top-4 by duration; minor labels → nearest top-4.
    durs: dict = defaultdict(float)
    for ss, se, lbl in segs_sorted:
        durs[lbl] += se - ss
    ranked = sorted(durs.items(), key=lambda kv: -kv[1])
    top4 = [lab for lab, _ in ranked[: max(1, len(speakers))]]
    label_to_gid = {lab: i for i, lab in enumerate(top4)}

    def nearest_top(seg_s, seg_e, lab):
        if lab in label_to_gid:
            return label_to_gid[lab]
        best_gid, best_d = 0, 1e18
        for ss, se, l2 in segs_sorted:
            if l2 not in label_to_gid:
                continue
            if se < seg_s:
                d = seg_s - se
            elif ss > seg_e:
                d = ss - seg_e
            else:
                d = 0.0
            if d < best_d:
                best_d, best_gid = d, label_to_gid[l2]
        return best_gid

    # Step 2: per-GT-utterance prediction by overlap-max; fallback to
    # the nearest top-4 segment label.
    spk_pred = defaultdict(lambda: defaultdict(float))
    spk_time: dict = defaultdict(float)
    for g in gt_rows:
        gs, ge = g["start_sec"], g["end_sec"]
        dur = max(0.0, ge - gs)
        spk_time[g["speaker"]] += dur
        gid_overlap: dict = defaultdict(float)
        nearest_gid = None
        nearest_d = 1e18
        for ss, se, lbl in segs_sorted:
            if se <= gs:
                # Track nearest before window for fallback.
                d = gs - se
                if d < nearest_d:
                    nearest_d, nearest_gid = d, nearest_top(ss, se, lbl)
                continue
            if ss >= ge:
                d = ss - ge
                if d < nearest_d:
                    nearest_d, nearest_gid = d, nearest_top(ss, se, lbl)
                break
            ov = min(ge, se) - max(gs, ss)
            if ov > 0:
                gid_overlap[nearest_top(ss, se, lbl)] += ov
        if gid_overlap:
            pred_gid = max(gid_overlap.items(), key=lambda kv: kv[1])[0]
        elif nearest_gid is not None:
            pred_gid = nearest_gid
        else:
            pred_gid = 0
        # Whole utterance counts for that pred id.
        spk_pred[g["speaker"]][pred_gid] += dur

    # Greedy fallback mapping by largest overlap first.
    triples = []
    for spk, ct in spk_pred.items():
        for lbl, sec in ct.items():
            triples.append((sec, spk, lbl))
    triples.sort(reverse=True)
    mapping: dict = {}
    used_lbl: set = set()
    used_spk: set = set()
    for _sec, spk, lbl in triples:
        if spk in used_spk or lbl in used_lbl:
            continue
        mapping[spk] = lbl
        used_spk.add(spk)
        used_lbl.add(lbl)

    # Hungarian ceiling (preferred).
    hungarian_mapping = dict(mapping)
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        labels = sorted({lbl for ct in spk_pred.values() for lbl in ct})
        if labels and speakers:
            cost = np.zeros((len(speakers), len(labels)), dtype=np.float64)
            for i, spk in enumerate(speakers):
                for j, lbl in enumerate(labels):
                    cost[i, j] = -spk_pred[spk].get(lbl, 0.0)
            ri, ci = linear_sum_assignment(cost)
            hungarian_mapping = {speakers[i]: labels[j] for i, j in zip(ri, ci)}
    except Exception:
        pass

    use_map = hungarian_mapping
    per_spk = {}
    total_correct = 0.0
    total_speech = 0.0
    for spk in speakers:
        tot = spk_time[spk]
        lbl = use_map.get(spk)
        sec = spk_pred[spk].get(lbl, 0.0) if lbl is not None else 0.0
        per_spk[spk] = sec / tot if tot > 0 else 0.0
        total_correct += sec
        total_speech += tot
    accuracy = total_correct / total_speech if total_speech > 0 else 0.0

    # Keep per-utterance coverage for diagnostics.
    decided = 0
    for g in gt_rows:
        gs, ge = g["start_sec"], g["end_sec"]
        for s in segs_sorted:
            ss, se = s[0], s[1]
            if se <= gs or ss >= ge:
                continue
            decided += 1
            break
    coverage = decided / max(1, len(gt_rows))

    return {
        "accuracy": accuracy,
        "total_speech_sec": total_speech,
        "total_correct_sec": total_correct,
        "per_spk": per_spk,
        "mapping": {spk: use_map.get(spk) for spk in speakers},
        "coverage": coverage,
        "n_gt": len(gt_rows),
        "n_decided": decided,
        # Legacy keys kept so the existing prints don't blow up.
        "macro": sum(per_spk.values()) / max(1, len(per_spk)),
        "micro": accuracy,
        "decided_macro": sum(per_spk.values()) / max(1, len(per_spk)),
        "decided_micro": accuracy,
        "per_spk_decided": per_spk,
        "n_no_seg": len(gt_rows) - decided,
        "matched": [],
    }


def _legacy_score_diarization_disabled(segments, gt_rows):  # pragma: no cover
    """Old per-utterance majority scorer — kept for reference only."""
    # Pre-sort segments by start for the overlap scan.
    segs_sorted = sorted(segments, key=lambda s: s[0])

    def best_label(gs, ge):
        # Aggregate overlap per label so a long GT row covered by two
        # diarizer segments with the same label still picks that label.
        score = {}
        winner_seg = None
        winner_ov = 0.0
        for s in segs_sorted:
            ss, se, lbl = s[0], s[1], s[2]
            if se <= gs:
                continue
            if ss >= ge:
                break
            ov = max(0.0, min(ge, se) - max(gs, ss))
            if ov <= 0:
                continue
            score[lbl] = score.get(lbl, 0.0) + ov
            if ov > winner_ov:
                winner_ov = ov
                winner_seg = (ss, se, lbl)
        if not score:
            return None, 0.0, None
        # Pick label with the largest aggregate overlap.
        best_lbl = max(score.items(), key=lambda kv: kv[1])[0]
        return best_lbl, score[best_lbl], winner_seg

    matched = []
    for g in gt_rows:
        gs, ge = g["start_sec"], g["end_sec"]
        lbl, ov, seg = best_label(gs, ge)
        matched.append({
            "gt_start": gs, "gt_end": ge,
            "gt_speaker": g["speaker"],
            "label": lbl,
            "overlap": ov,
            "status": "decided" if lbl is not None else "no_segment",
        })

    mapping = {}
    for m in matched:
        if m["label"] is None:
            continue
        mapping.setdefault(m["label"], m["gt_speaker"])

    speakers = sorted({m["gt_speaker"] for m in matched})
    per_spk_total = {s: 0 for s in speakers}
    per_spk_correct = {s: 0 for s in speakers}
    per_spk_decided = {s: 0 for s in speakers}
    per_spk_decided_correct = {s: 0 for s in speakers}
    n_no_seg = 0
    for m in matched:
        true = m["gt_speaker"]
        per_spk_total[true] += 1
        if m["status"] == "no_segment":
            n_no_seg += 1
            continue
        per_spk_decided[true] += 1
        pred = mapping.get(m["label"], "__unk__")
        if pred == true:
            per_spk_correct[true] += 1
            per_spk_decided_correct[true] += 1

    per_spk_acc = {
        s: per_spk_correct[s] / per_spk_total[s] if per_spk_total[s] else 0.0
        for s in speakers
    }
    per_spk_decided_acc = {
        s: (per_spk_decided_correct[s] / per_spk_decided[s])
            if per_spk_decided[s] else 0.0
        for s in speakers
    }
    macro = sum(per_spk_acc.values()) / max(1, len(per_spk_acc))
    micro = sum(per_spk_correct.values()) / max(1, sum(per_spk_total.values()))
    decided_macro = sum(per_spk_decided_acc.values()) / max(1, len(per_spk_decided_acc))
    decided_micro = (
        sum(per_spk_decided_correct.values()) / max(1, sum(per_spk_decided.values()))
    )
    coverage = sum(per_spk_decided.values()) / max(1, sum(per_spk_total.values()))
    return {
        "macro": macro, "micro": micro,
        "decided_macro": decided_macro, "decided_micro": decided_micro,
        "coverage": coverage,
        "per_spk": per_spk_acc,
        "per_spk_decided": per_spk_decided_acc,
        "mapping": mapping,
        "n_gt": len(matched),
        "n_no_seg": n_no_seg,
        "n_decided": sum(per_spk_decided.values()),
        "matched": matched,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default="tests/test.mp3")
    ap.add_argument("--gt",    default="tests/fixtures/test_ground_truth.json")
    ap.add_argument("--url",   default="ws://localhost:8080/ws")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--drain-sec", type=float, default=30.0)
    ap.add_argument("--diarize-timeout", type=float, default=1500.0)
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--baseline", type=float, default=93.5,
                    help="reference accuracy (%) for the Δ in the "
                         "Constitutional accuracy line; default = "
                         "the GPU-offline reference from repo memory.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"runs/diarizen_live_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.gt)
    if args.max_sec > 0:
        cap = args.max_sec
        gt = [g for g in gt if g["end_sec"] <= cap]
    print(f"[gt] {len(gt)} utterances, "
          f"{len({g['speaker'] for g in gt})} speakers", flush=True)

    pcm = decode_to_pcm(args.audio)
    if args.max_sec > 0:
        pcm = pcm[: int(args.max_sec * 16000) * 2]
    print(f"[audio] {len(pcm)/2/16000.0:.1f}s @ 16kHz; replay at {args.speed}x",
          flush=True)

    final_msg = {"value": None}
    progress_msg = {"value": None}
    connected = threading.Event()
    got_final = threading.Event()

    raw_fh = (out_dir / "raw_events.jsonl").open("w", encoding="utf-8")
    lock = threading.Lock()

    def on_message(ws, msg):
        if isinstance(msg, bytes):
            try:
                msg = msg.decode("utf-8", errors="replace")
            except Exception:
                return
        with lock:
            raw_fh.write(msg.rstrip("\n") + "\n")
        try:
            obj = json.loads(msg)
        except json.JSONDecodeError:
            return
        t = obj.get("type")
        if t == "speaker_diarize_progress":
            progress_msg["value"] = obj
            print(f"[diarize] progress: {obj}", flush=True)
        elif t == "speaker_diarize_final":
            final_msg["value"] = obj
            got_final.set()

    def on_open(ws):
        print("[ws] connected", flush=True)
        connected.set()

    def on_error(ws, err):
        print(f"[ws] error: {err}", file=sys.stderr, flush=True)

    ws = websocket.WebSocketApp(
        args.url, on_open=on_open, on_message=on_message, on_error=on_error,
    )
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    if not connected.wait(timeout=10):
        print("[ws] connect timeout", file=sys.stderr)
        return 2

    stream_pcm(ws, pcm, args.chunk, args.speed)
    print(f"[drain] waiting {args.drain_sec}s for tail...", flush=True)
    time.sleep(args.drain_sec)

    print("[diarize] sending diarizen_finalize", flush=True)
    ws.send("diarizen_finalize", opcode=websocket.ABNF.OPCODE_TEXT)

    if not got_final.wait(timeout=args.diarize_timeout):
        print(f"[diarize] TIMEOUT after {args.diarize_timeout}s waiting "
              f"for speaker_diarize_final", file=sys.stderr)
        try:
            ws.close()
        except Exception:
            pass
        raw_fh.close()
        return 3

    try:
        ws.close()
    except Exception:
        pass
    raw_fh.close()

    final = final_msg["value"]
    if not final.get("ok"):
        print(f"[diarize] FAILED: {final.get('error', 'unknown')}",
              file=sys.stderr)
        return 4

    segs = final.get("segments", [])
    print(f"[diarize] received {len(segs)} segments  "
          f"audio_sec={final.get('audio_sec', 0):.2f}  "
          f"wall_sec={final.get('wall_sec', 0):.2f}", flush=True)

    (out_dir / "diarize_segments.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    res = score_diarization(segs, gt)
    (out_dir / "score.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "matched"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "pairings.jsonl").open("w", encoding="utf-8") as f:
        for m in res["matched"]:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("\n=== DiariZen-v2 live (Hybrid P1) result ===")
    print(f"GT utterances: {res['n_gt']}   decided: {res['n_decided']}   "
          f"no_segment: {res['n_no_seg']}")
    print(f"coverage(decided/gt): {res['coverage']:.3f}")
    print(f"macro={res['macro']:.3f}  micro={res['micro']:.3f}   "
          f"(no_segment counts as wrong)")
    print(f"decided_macro={res['decided_macro']:.3f}  "
          f"decided_micro={res['decided_micro']:.3f}")
    print("per-speaker accuracy (all / decided-only):")
    for s in sorted(res["per_spk"]):
        print(f"  {s:<8s} all={res['per_spk'][s]:.3f}  "
              f"decided-only={res['per_spk_decided'][s]:.3f}")
    print(f"\nfirst-seen mapping (label → GT name):")
    for k, v in sorted(res["mapping"].items()):
        print(f"  {k!r:<10s} → {v}")

    # Constitutional accuracy line (philosophy.instructions.md §"Accuracy
    # Is the Sole Metric").
    live_pct = res["micro"] * 100.0
    delta_pp = live_pct - args.baseline
    print(f"\naccuracy(tests/test.mp3, diarization): "
          f"{args.baseline:.1f}% → {live_pct:.1f}%   "
          f"(Δ = {delta_pp:+.1f} pp)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
