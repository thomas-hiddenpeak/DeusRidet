#!/usr/bin/env python3
"""embed_oracle_vad_wespeaker.py — same as embed_oracle_vad.py but uses
WeSpeaker's pretrained models.

Usage:
  python3 embed_oracle_vad_wespeaker.py --tag english --out runs/04_we_resnet221/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="wespeaker Hub tag: english|chinese|campplus|"
                         "eres2net|vblinkp|vblinkf|w2vbert2_mfa")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "tests/fixtures/test_ground_truth.json")
    ap.add_argument("--audio", type=Path,
                    default=Path(__file__).resolve().parent / "test_16k.wav")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cluster", default="kmeans",
                    choices=["spectral", "kmeans", "agglo"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-sec", type=float, default=10.0,
                    help="truncate utterances longer than this (centered)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        log_lines.append(s)

    log(f"[load] audio={args.audio}")
    wav_np, sr = sf.read(str(args.audio), dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    assert sr == 16000, f"expected 16kHz, got {sr}"
    log(f"[load] {wav_np.shape[0]/sr:.1f}s @ {sr}Hz")

    gt = json.loads(args.gt.read_text())
    utts = gt["utterances"]
    log(f"[load] utts={len(utts)} speakers={gt['speakers']}")

    import wespeaker
    log(f"[model] loading wespeaker tag={args.tag}")
    sp = wespeaker.load_model(args.tag)
    sp.set_device(args.device)
    sp.model.eval()

    def _embed(pcm: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(pcm).unsqueeze(0).to(args.device)
        feats = sp.compute_features(t, sample_rate=sr, cmn=True)
        if isinstance(feats, torch.Tensor):
            feats = feats.to(args.device)
        with torch.no_grad():
            out = sp.model(feats)
            out = out[-1] if isinstance(out, tuple) else out
        return out[0].detach().cpu().numpy()

    # Probe.
    probe = _embed(wav_np[:16000])
    D = probe.shape[-1]
    log(f"[model] embed dim={D}")

    embeddings = np.zeros((len(utts), D), dtype=np.float32)
    seg_meta: list[tuple[float, float, str]] = []
    t0 = time.time()
    with torch.no_grad():
        for i, u in enumerate(utts):
            s = int(u["t0_start_sec"] * sr)
            e = int(u["t0_end_sec"] * sr)
            s = max(0, s)
            e = min(wav_np.shape[0], e)
            # Fixed-length window: every input must have the same shape,
            # otherwise each new shape forces a multi-second cuDNN/JIT
            # re-autotune on Tegra (observed 20-40 s per shape).
            fixed_n = int(args.max_sec * sr)
            if e - s >= fixed_n:
                mid = (s + e) // 2
                s = max(0, mid - fixed_n // 2)
                e = min(wav_np.shape[0], s + fixed_n)
            chunk = wav_np[s:e]
            if chunk.shape[0] < fixed_n:
                chunk = np.concatenate(
                    [chunk, np.zeros(fixed_n - chunk.shape[0],
                                     dtype=np.float32)])
            elif chunk.shape[0] > fixed_n:
                chunk = chunk[:fixed_n]
            emb = _embed(chunk)
            emb = np.asarray(emb).squeeze()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            embeddings[i] = emb
            seg_meta.append((u["t0_start_sec"], u["t0_end_sec"], u["speaker"]))
            if (i + 1) % 25 == 0:
                log(f"[embed] {i+1}/{len(utts)} ({time.time()-t0:.1f}s)")
    log(f"[embed] done {len(utts)} in {time.time()-t0:.1f}s")

    log(f"[cluster] method={args.cluster} K={args.k}")
    if args.cluster == "spectral":
        from sklearn.cluster import SpectralClustering
        sim = embeddings @ embeddings.T
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
        labels = SpectralClustering(
            n_clusters=args.k, affinity="precomputed",
            assign_labels="kmeans", random_state=0).fit_predict(sim)
    elif args.cluster == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=args.k, random_state=0, n_init=10
                        ).fit_predict(embeddings)
    else:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(
            n_clusters=args.k, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
    log(f"[cluster] hist: "
        f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    pred_path = args.out / "predictions.jsonl"
    with pred_path.open("w") as fp:
        for (t0_, t1_, _), lab in zip(seg_meta, labels):
            fp.write(json.dumps({"t0": t0_, "t1": t1_,
                                 "speaker_id": int(lab)}) + "\n")
    log(f"[out] {pred_path}")
    np.save(args.out / "embeddings.npy", embeddings)
    (args.out / "run.log").write_text("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
