#!/usr/bin/env python3
"""diar_sortformer.py — run NeMo Sortformer end-to-end diarization on
the test audio, then assign GT oracle VAD segments by max-overlap.
Writes predictions.jsonl in the standard {t0,t1,speaker_id} format.
"""
import json, time, argparse, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--audio", default="tools/verification_2026/test_16k.wav")
ap.add_argument("--gt", default="tests/fixtures/test_ground_truth.json")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
ap.add_argument("--session-len", type=float, default=600.0)
args = ap.parse_args()

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "run.log"
def log(m):
    print(m, flush=True)
    with open(log_path, "a") as f: f.write(m + "\n")

log(f"[sortformer] loading {args.model}")
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams
m = SortformerEncLabelModel.from_pretrained(args.model)
m = m.eval().to("cuda")

log(f"[sortformer] diarize {args.audio} (session_len={args.session_len}s)")
t0 = time.time()
from omegaconf import OmegaConf
pp = OmegaConf.structured(PostProcessingParams())
cfg = DiarizeConfig(session_len_sec=args.session_len, batch_size=1, verbose=True,
                    postprocessing_params=pp)
result = m.diarize(audio=[args.audio], override_config=cfg)
dt = time.time() - t0
log(f"[sortformer] diarize done in {dt:.1f}s")

# result[0] is list of strings like "0.00 1.23 speaker_0"
segs = []
for row in result[0]:
    parts = row.strip().split()
    if len(parts) < 3: continue
    b = float(parts[0]); e = float(parts[1])
    spk = parts[2]
    # parse "speaker_N" → int
    sid = int(spk.split("_")[-1]) if "_" in spk else int(spk)
    segs.append((b, e, sid))
log(f"[sortformer] {len(segs)} predicted intervals")

# save raw diar
with open(out_dir / "diar_raw.jsonl", "w") as f:
    for b,e,s in segs:
        f.write(json.dumps({"t0":b,"t1":e,"speaker_id":s})+"\n")

# load GT VAD segments
gt = json.load(open(args.gt))
utts = gt["utterances"] if "utterances" in gt else gt
log(f"[sortformer] {len(utts)} GT utterances")

# for each GT utt, find dominant predicted speaker by overlap
def overlap(a0,a1,b0,b1):
    return max(0.0, min(a1,b1)-max(a0,b0))

with open(out_dir / "predictions.jsonl", "w") as f:
    nass = 0
    for u in utts:
        t0_ = u["t0_start_sec"]; t1_ = u["t0_end_sec"]
        scores = {}
        for b,e,s in segs:
            ov = overlap(t0_,t1_,b,e)
            if ov > 0:
                scores[s] = scores.get(s,0.0) + ov
        if scores:
            sid = max(scores.items(), key=lambda x: x[1])[0]
            nass += 1
        else:
            sid = 0  # fallback
        f.write(json.dumps({"t0":t0_,"t1":t1_,"speaker_id":int(sid)})+"\n")
    log(f"[sortformer] assigned {nass}/{len(utts)} (rest fallback=0)")

log("[sortformer] DONE")
