#!/usr/bin/env python3
# TODO(native-cuda-port): produces the bit-equality ground truth for the
# DiariZen-v2 WeSpeaker ResNet34-LM embedder (P2a). Lives under tools/ because
# it runs in the py310_diarizen env, not in awaken.
"""Dump the DiariZen-v2 embedding stage reference for P2a.

For a handful of (chunk, speaker) pairs the native CUDA embedder must
reproduce the pyannote WeSpeakerResNet34 output. This script replicates the
exact crop + mask logic of ``SpeakerDiarization.get_embeddings`` and calls the
real ``pipe._embedding(waveform, masks=mask)`` to capture, per pair:

    wave   float32 [num_samples]  cropped chunk waveform at [-1, 1] (the model
                                  multiplies by 2**15 internally)
    mask   float32 [num_frames]   the used activity mask (clean or full)
    embed  float32 [256]          reference embedding

A fbank reference (CMN log-mel) is also dumped for the first pair via the exact
torchaudio.compliance.kaldi recipe the model uses.

Keys: n_pairs, num_samples, num_frames, wave[P,Ns], mask[P,Nf], embed[P,256],
      chunk_idx[P], spk_idx[P], fbank0[T,80], fbank0_T.

Run inside py310_diarizen:
    python tools/diarizen_dump_embedder.py --audio tests/test.mp3 \
        --offset 0.0 --duration 30.0 \
        --out tests/fixtures/diarizen_p2a_embedder.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio", required=True)
    p.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--chunks", default="0,4",
                   help="comma-separated chunk indices to dump")
    return p.parse_args()


def main() -> int:
    a = _args()
    out_path = Path(a.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import librosa
    import numpy as np
    import torch
    import torchaudio.compliance.kaldi as kaldi

    _o = torch.load
    torch.load = lambda *aa, **kk: (kk.update(weights_only=False) or _o(*aa, **kk))

    from diarizen.pipelines.inference import DiariZenPipeline

    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu"
                       else "cpu")
    print(f"[dump] loading {a.model} on {dev}", file=sys.stderr)
    pipe = DiariZenPipeline.from_pretrained(a.model).to(dev)

    wav, _ = librosa.load(str(a.audio), sr=16000, mono=True, offset=a.offset,
                          duration=a.duration)
    wav = np.asarray(wav, dtype=np.float32)
    file = {"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": 16000}

    seg = pipe.get_segmentations(file, soft=False)  # binarized (powerset)
    import numpy as _np
    from pyannote.core import SlidingWindowFeature
    binseg = SlidingWindowFeature(_np.asarray(seg.data, dtype=_np.float32),
                                  seg.sliding_window)

    excl = bool(getattr(pipe, "embedding_exclude_overlap", True))
    print(f"[dump] exclude_overlap={excl} seg={binseg.data.shape}",
          file=sys.stderr)

    duration = binseg.sliding_window.duration
    num_chunks, num_frames, num_speakers = binseg.data.shape

    # Replicate get_embeddings clean-mask logic.
    min_num_samples = pipe._embedding.min_num_samples
    num_samples_dur = duration * pipe._embedding.sample_rate
    import math
    min_num_frames = math.ceil(num_frames * min_num_samples / num_samples_dur)
    clean = 1.0 * (np.sum(binseg.data, axis=2, keepdims=True) < 2)
    clean_seg = binseg.data * clean

    want = [int(x) for x in a.chunks.split(",") if x.strip() != ""]
    chunks_list = list(binseg)  # [(chunk, masks), ...]

    waves, masks, embeds, cidx, sidx = [], [], [], [], []
    Ns = None
    for c in want:
        chunk, _m = chunks_list[c]
        waveform, _ = pipe._audio.crop(file, chunk, duration=duration, mode="pad")
        wf = waveform.detach().to("cpu", torch.float32)  # [1, Ns]
        if Ns is None:
            Ns = wf.shape[-1]
        m_full = np.nan_to_num(binseg.data[c], nan=0.0).astype(np.float32)
        m_clean = np.nan_to_num(clean_seg[c], nan=0.0).astype(np.float32)
        for s in range(num_speakers):
            cm = m_clean[:, s]
            fm = m_full[:, s]
            used = cm if np.sum(cm) > min_num_frames else fm
            if np.sum(used) <= 0:
                continue
            mb = torch.from_numpy(used)[None]
            e = pipe._embedding(wf[None], masks=mb)  # [1, 256] np
            waves.append(wf.numpy().reshape(-1))
            masks.append(used)
            embeds.append(np.asarray(e, dtype=np.float32).reshape(-1))
            cidx.append(c)
            sidx.append(s)
            print(f"[dump] chunk={c} spk={s} active={int(np.sum(used))} "
                  f"emb[:3]={embeds[-1][:3]}", file=sys.stderr)

    # fbank reference for the first pair (matches WeSpeakerResNet34.compute_fbank)
    w0 = torch.from_numpy(waves[0]).unsqueeze(0) * (1 << 15)
    fb = kaldi.fbank(w0, num_mel_bins=80, frame_length=25, frame_shift=10,
                     dither=0.0, sample_frequency=16000, window_type="hamming",
                     use_energy=False)
    fb = fb - torch.mean(fb, dim=0, keepdim=True)
    fb = fb.numpy().astype(np.float32)  # [T, 80]

    np.savez_compressed(
        out_path,
        n_pairs=np.int64(len(waves)),
        num_samples=np.int64(Ns),
        num_frames=np.int64(num_frames),
        wave=np.stack(waves).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        embed=np.stack(embeds).astype(np.float32),
        chunk_idx=np.asarray(cidx, dtype=np.int64),
        spk_idx=np.asarray(sidx, dtype=np.int64),
        fbank0=fb,
        fbank0_T=np.int64(fb.shape[0]),
    )
    print(f"[dump] wrote {out_path}: pairs={len(waves)} Ns={Ns} "
          f"fbank0={fb.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
