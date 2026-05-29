#!/usr/bin/env python3
"""Cand #7-v2 + ReDimNet embedding head replacement, GPU.

Drop-in swaps DiariZen-v2's WeSpeaker-ResNet34-LM (192-d, VoxCeleb)
embedding head for IDRnD ReDimNet `M / ft_mix / vb2+vox2+cnc` (192-d,
trained on VoxBlink2 + VoxCeleb2 + CN-Celeb). Same 192-d dimension =>
VBx clustering hyper-params (lda_dim=128, Fa=0.07, Fb=0.8) remain valid.
"""
import json, sys, time, os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
GT = REPO / "tests/fixtures/test_ground_truth.json"
WAV = REPO / "tools/verification_2026/test_16k.wav"
MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
REDIM_NAME = os.environ.get("REDIM_NAME", "M")
REDIM_TRAIN = os.environ.get("REDIM_TRAIN", "ft_mix")
REDIM_DATA = os.environ.get("REDIM_DATA", "vb2+vox2+cnc")
TAG = f"07v2_diarizen_v2_redimnet_{REDIM_NAME}_{REDIM_TRAIN}_{REDIM_DATA}"
OUT = REPO / "tools/verification_2026/runs" / TAG
OUT.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# torch>=2.6 forces weights_only=True; old pyannote ckpts trip on TorchVersion.
_orig_torch_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)
torch.load = _patched_load

# torchaudio 2.10 requires torchcodec for load(); fall back to soundfile.
import torchaudio, soundfile as _sf
def _ta_load(path, *a, **kw):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr
torchaudio.load = _ta_load

from diarizen.pipelines.inference import DiariZenPipeline

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {dev}", flush=True)

# --- load ReDimNet ---
print(f"[redimnet] loading {REDIM_NAME}/{REDIM_TRAIN}/{REDIM_DATA}", flush=True)
redim = torch.hub.load(
    "IDRnD/ReDimNet", "ReDimNet",
    model_name=REDIM_NAME, train_type=REDIM_TRAIN, dataset=REDIM_DATA,
)
redim = redim.to(dev).eval()
# Probe dimension
with torch.inference_mode():
    _probe = redim(torch.randn(1, 16000, device=dev))
print(f"[redimnet] output shape = {tuple(_probe.shape)} dtype={_probe.dtype}", flush=True)
REDIM_DIM = int(_probe.shape[-1])


class ReDimNetEmbedding:
    """pyannote-compatible embedding head wrapping ReDimNet.

    Mirrors `SpeechBrain_PretrainedSpeakerEmbedding.__call__` mask handling:
    interpolate frame-mask -> sample-mask, gather active samples, pad to
    same length, run model, return numpy [B, D].
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, dim: int):
        self.model = model
        self.device = device
        self.dimension = dim
        self.sample_rate = 16000
        self.metric = "cosine"
        # ReDimNet works on any >=~0.25s; mirror SpeechBrain default ~640.
        self.min_num_samples = 640

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None) -> np.ndarray:
        # waveforms: [B, 1, T]
        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1
        waveforms = waveforms.squeeze(dim=1)  # [B, T]

        if masks is None:
            signals = waveforms
            wav_lens = torch.full((batch_size,), num_samples, dtype=torch.long)
        else:
            imasks = F.interpolate(
                masks.unsqueeze(1).float(), size=num_samples, mode="nearest"
            ).squeeze(1) > 0.5
            sigs = [w[m].contiguous() for w, m in zip(waveforms, imasks)]
            wav_lens = torch.tensor([s.shape[0] for s in sigs], dtype=torch.long)
            # pad to max
            max_len = int(wav_lens.max().item()) if int(wav_lens.max().item()) > 0 else 1
            signals = torch.zeros(batch_size, max_len, dtype=waveforms.dtype, device=waveforms.device)
            for i, s in enumerate(sigs):
                if s.shape[0] > 0:
                    signals[i, :s.shape[0]] = s

        too_short = wav_lens < self.min_num_samples
        if bool(too_short.all().item()):
            return np.full((batch_size, self.dimension), np.nan, dtype=np.float32)

        signals = signals.to(self.device)
        out = np.full((batch_size, self.dimension), np.nan, dtype=np.float32)

        # Run only valid rows individually (variable length): ReDimNet handles
        # arbitrary T, but we still cannot batch across very different lengths
        # without re-padding/masking. Per-row inference is simple and avoids
        # silent zero-padding contaminating statistics-pool features.
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == "cuda" else torch.float32):
            for i in range(batch_size):
                if bool(too_short[i].item()):
                    continue
                L = int(wav_lens[i].item())
                emb = self.model(signals[i:i+1, :L])  # [1, D]
                out[i] = emb.squeeze(0).float().cpu().numpy()
        return out


# --- load DiariZen pipeline ---
print(f"[load] {MODEL}", flush=True)
pipe = DiariZenPipeline.from_pretrained(MODEL).to(dev)
print(f"[swap] replacing _embedding (was dim={pipe._embedding.dimension}, sr={pipe._embedding.sample_rate})", flush=True)
assert REDIM_DIM == pipe._embedding.dimension, (
    f"ReDimNet dim {REDIM_DIM} != original {pipe._embedding.dimension}; "
    "VBx params are tuned for the original; aborting."
)
pipe._embedding = ReDimNetEmbedding(redim, dev, REDIM_DIM)
# pyannote's diarization rebuilds `_audio` from `_embedding.sample_rate`;
# we already set it to 16000 so reuse the same Audio object.
from pyannote.audio.core.io import Audio
pipe._audio = Audio(sample_rate=pipe._embedding.sample_rate, mono="downmix")

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
