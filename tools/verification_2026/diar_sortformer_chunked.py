#!/usr/bin/env python3
"""diar_sortformer_chunked.py — sliding-window NeMo Sortformer diarization
for long audio. Chunks audio into windows the model can handle, then stitches
per-chunk speaker labels into a global identity space via overlap voting.
"""
import json, time, argparse, os
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--audio", default="tools/verification_2026/test_16k.wav")
ap.add_argument("--gt", default="tests/fixtures/test_ground_truth.json")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
ap.add_argument("--chunk-sec", type=float, default=240.0)
ap.add_argument("--overlap-sec", type=float, default=30.0)
ap.add_argument("--session-len", type=float, default=300.0)
args = ap.parse_args()

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "run.log"
def log(m):
    print(m, flush=True)
    with open(log_path, "a") as f: f.write(m + "\n")

log(f"[chunked] loading {args.model}")
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams
from omegaconf import OmegaConf
m = SortformerEncLabelModel.from_pretrained(args.model)
m = m.eval().to("cuda")

pp = OmegaConf.structured(PostProcessingParams())

audio, sr = sf.read(args.audio)
if audio.ndim > 1: audio = audio.mean(1)
assert sr == 16000, f"need 16kHz, got {sr}"
T = len(audio) / sr
log(f"[chunked] audio {T:.1f}s sr={sr}")

# build chunks
chunk_n = int(args.chunk_sec * sr)
step_n = int((args.chunk_sec - args.overlap_sec) * sr)
chunks = []  # (start_sec, end_sec, pcm)
pos = 0
while pos < len(audio):
    end = min(pos + chunk_n, len(audio))
    chunks.append((pos / sr, end / sr, audio[pos:end].astype(np.float32)))
    if end >= len(audio): break
    pos += step_n
log(f"[chunked] {len(chunks)} chunks, chunk={args.chunk_sec}s overlap={args.overlap_sec}s")

# diarize each chunk → list of (t0, t1, local_spk_id) in absolute time
per_chunk_segs = []
t0 = time.time()
import tempfile
for ci, (s0, s1, pcm) in enumerate(chunks):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, pcm, sr)
        tmp_path = tmp.name
    try:
        cfg = DiarizeConfig(session_len_sec=args.session_len, batch_size=1,
                            verbose=False, postprocessing_params=pp)
        result = m.diarize(audio=[tmp_path], override_config=cfg)
    finally:
        os.unlink(tmp_path)
    segs = []
    for row in result[0]:
        parts = row.strip().split()
        if len(parts) < 3: continue
        b = float(parts[0]); e = float(parts[1])
        sid_local = int(parts[2].split("_")[-1]) if "_" in parts[2] else int(parts[2])
        segs.append((s0 + b, s0 + e, sid_local))
    per_chunk_segs.append(segs)
    log(f"[chunked] chunk {ci+1}/{len(chunks)} [{s0:.1f}-{s1:.1f}] -> {len(segs)} segs")
log(f"[chunked] diarize all chunks done in {time.time()-t0:.1f}s")

# stitch: assign global IDs via overlap of consecutive chunks
def overlap(a0,a1,b0,b1):
    return max(0.0, min(a1,b1)-max(a0,b0))

# global_id_for[(chunk_idx, local_id)] = global_id
mapping = {}
next_gid = 0
# First chunk: take all locals as own globals
for s in per_chunk_segs[0]:
    key = (0, s[2])
    if key not in mapping:
        mapping[key] = next_gid; next_gid += 1
for ci in range(1, len(per_chunk_segs)):
    prev_segs = per_chunk_segs[ci-1]
    cur_segs = per_chunk_segs[ci]
    # for each local id in current chunk, compute overlap with each global id
    # (from prev chunk's segments) in the overlap region
    overlap_start = chunks[ci][0]
    overlap_end = chunks[ci-1][1]
    cur_locals = set(s[2] for s in cur_segs)
    for cl in cur_locals:
        # overlap-weighted votes per global id
        votes = {}
        for cs in cur_segs:
            if cs[2] != cl: continue
            for ps in prev_segs:
                ov = overlap(max(cs[0], overlap_start), min(cs[1], overlap_end),
                             max(ps[0], overlap_start), min(ps[1], overlap_end))
                if ov > 0:
                    gid = mapping.get((ci-1, ps[2]))
                    if gid is None: continue
                    votes[gid] = votes.get(gid, 0.0) + ov
        if votes:
            best = max(votes.items(), key=lambda x: x[1])[0]
            mapping[(ci, cl)] = best
        else:
            mapping[(ci, cl)] = next_gid; next_gid += 1
log(f"[chunked] stitched {next_gid} global speakers")

# flatten to global timeline
all_segs = []
for ci, segs in enumerate(per_chunk_segs):
    for (b, e, lid) in segs:
        gid = mapping[(ci, lid)]
        all_segs.append((b, e, gid))
all_segs.sort()

with open(out_dir / "diar_raw.jsonl", "w") as f:
    for b,e,s in all_segs:
        f.write(json.dumps({"t0":b,"t1":e,"speaker_id":int(s)})+"\n")

# If more than 4 globals, merge smallest into nearest by total duration overlap
from collections import Counter, defaultdict
totals = Counter()
for b,e,s in all_segs:
    totals[s] += (e-b)
log(f"[chunked] global totals: {dict(totals)}")

# Keep top-4 by duration; map remaining minor ids to closest (by joint co-occurrence in audio time) top-4
top4 = [k for k,_ in totals.most_common(4)]
top4_set = set(top4)
remap = {gid: gid for gid in top4}
for gid in totals:
    if gid in top4_set: continue
    # find which top-4 has greatest temporal proximity (shortest avg gap) — simple heuristic
    # use the top-4 whose intervals are most often adjacent in time
    my_intervals = [(b,e) for b,e,s in all_segs if s == gid]
    best_gid, best_score = top4[0], -1.0
    for cand in top4:
        cand_intervals = [(b,e) for b,e,s in all_segs if s == cand]
        # score = inverse of min gap
        score = 0.0
        for mb,me in my_intervals:
            mgap = min((abs(mb-ce) if ce<=mb else abs(cb-me) if cb>=me else 0.0)
                       for cb,ce in cand_intervals) if cand_intervals else 1e9
            score += 1.0/(1.0+mgap)
        if score > best_score:
            best_score = score; best_gid = cand
    remap[gid] = best_gid
log(f"[chunked] remap minor->top4: {remap}")

# load GT
gt = json.load(open(args.gt))
utts = gt["utterances"] if "utterances" in gt else gt
log(f"[chunked] {len(utts)} GT utterances")

# Re-assign top4 → 0..3
top4_to_idx = {g:i for i,g in enumerate(top4)}

with open(out_dir / "predictions.jsonl", "w") as f:
    nass = 0
    for u in utts:
        ut0 = u["t0_start_sec"]; ut1 = u["t0_end_sec"]
        scores = {}
        for b,e,s in all_segs:
            ov = overlap(ut0,ut1,b,e)
            if ov > 0:
                g = remap.get(s, s)
                scores[g] = scores.get(g,0.0) + ov
        if scores:
            best = max(scores.items(), key=lambda x: x[1])[0]
            sid = top4_to_idx.get(best, 0)
            nass += 1
        else:
            sid = 0
        f.write(json.dumps({"t0":ut0,"t1":ut1,"speaker_id":int(sid)})+"\n")
    log(f"[chunked] assigned {nass}/{len(utts)}")

log("[chunked] DONE")
