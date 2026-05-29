#!/usr/bin/env python3
"""Reuse cached diar_raw.jsonl → produce predictions.jsonl for Cand #7."""
import json, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
GT = REPO / "tests/fixtures/test_ground_truth.json"
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    REPO / "tools/verification_2026/runs/07_diarizen_diarizen-wavlm-large-s80-md")

segs = []
with (RUN / "diar_raw.jsonl").open() as f:
    for line in f:
        d = json.loads(line)
        segs.append((float(d["start"]), float(d["end"]), str(d["label"])))
print(f"[raw] {len(segs)} segments")

durs = defaultdict(float)
for s, e, lab in segs:
    durs[lab] += e - s
ranked = sorted(durs.items(), key=lambda kv: -kv[1])
top4 = [lab for lab, _ in ranked[:4]]
label_to_gid = {lab: i for i, lab in enumerate(top4)}
print(f"[map] top-4: {[(l, round(durs[l],1)) for l in top4]}")
print(f"[map] minor ({len(ranked)-4}): {[(l, round(d,1)) for l,d in ranked[4:]]}")

def nearest_top(seg_s, seg_e, lab):
    if lab in label_to_gid: return label_to_gid[lab]
    best_gid, best_d = 0, 1e18
    for s, e, l in segs:
        if l not in label_to_gid: continue
        if e < seg_s: d = seg_s - e
        elif s > seg_e: d = s - seg_e
        else: d = 0.0
        if d < best_d:
            best_d, best_gid = d, label_to_gid[l]
    return best_gid

gt = json.loads(GT.read_text())
utts = gt["utterances"]
pred_path = RUN / "predictions.jsonl"
n_unk = 0
with pred_path.open("w") as f:
    for u in utts:
        u_s, u_e = float(u["t0_start_sec"]), float(u["t0_end_sec"])
        gid_overlap = defaultdict(float)
        for s, e, lab in segs:
            ov = max(0.0, min(e, u_e) - max(s, u_s))
            if ov > 0:
                gid_overlap[nearest_top(s, e, lab)] += ov
        if not gid_overlap:
            n_unk += 1
            pred_gid = 0
        else:
            pred_gid = max(gid_overlap.items(), key=lambda kv: kv[1])[0]
        f.write(json.dumps({
            "t0": u_s, "t1": u_e,
            "gt_speaker": u.get("speaker"),
            "speaker_id": int(pred_gid),
        }) + "\n")
print(f"[pred] {pred_path}  unknown_utts={n_unk}/{len(utts)}")
