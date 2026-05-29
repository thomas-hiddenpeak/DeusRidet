#!/usr/bin/env python3
"""Cand #7-v2 GPU: DiariZen WavLM-large-s80-md-v2 on CUDA."""
import json, sys, time, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GT = REPO / "tests/fixtures/test_ground_truth.json"
WAV = REPO / "tools/verification_2026/test_16k.wav"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
OUT = REPO / "tools/verification_2026/runs" / ("07v2_diarizen_" + MODEL.split("/")[-1].replace(".", "_"))
OUT.mkdir(parents=True, exist_ok=True)

import torch
# torch>=2.6 changed default weights_only=True; old pyannote ckpts need False
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

# torchaudio 2.10 requires torchcodec for load(); fall back to soundfile.
import torchaudio, soundfile as _sf, numpy as _np
def _ta_load(path, *a, **kw):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr
torchaudio.load = _ta_load

from diarizen.pipelines.inference import DiariZenPipeline

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {dev}", flush=True)
print(f"[load] {MODEL}", flush=True)
pipe = DiariZenPipeline.from_pretrained(MODEL).to(dev)

print(f"[diarize] {WAV}", flush=True)
t0 = time.time()
annot = pipe(str(WAV))
dt = time.time() - t0
print(f"[diarize] done in {dt:.1f}s", flush=True)

raw_path = OUT / "diar_raw.jsonl"
segs = []
with raw_path.open("w") as f:
    for turn, _, label in annot.itertracks(yield_label=True):
        s, e = float(turn.start), float(turn.end)
        segs.append((s, e, str(label)))
        f.write(json.dumps({"start": s, "end": e, "label": str(label)}) + "\n")
print(f"[raw] {len(segs)} segments  raw={raw_path}", flush=True)

from collections import defaultdict
durs = defaultdict(float)
for s, e, lab in segs:
    durs[lab] += e - s
ranked = sorted(durs.items(), key=lambda kv: -kv[1])
top4 = [lab for lab, _ in ranked[:4]]
label_to_gid = {lab: i for i, lab in enumerate(top4)}
print(f"[map] top-4: {[(l, round(durs[l],1)) for l in top4]}", flush=True)
print(f"[map] minor ({len(ranked)-4}): {[(l, round(d,1)) for l,d in ranked[4:]]}", flush=True)

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
pred_path = OUT / "predictions.jsonl"
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
print(f"[pred] {pred_path}  unknown_utts={n_unk}/{len(utts)}", flush=True)
print(f"[wall_clock_sec] {dt:.1f}", flush=True)
print(f"[done] run_dir={OUT}", flush=True)
