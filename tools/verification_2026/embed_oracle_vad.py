#!/usr/bin/env python3
"""embed_oracle_vad.py — extract one embedding per GT utterance using
a chosen modelscope speaker model, then cluster into K=4 with
spectral clustering. Writes predictions.jsonl for offline_score.py.

Usage:
  python3 embed_oracle_vad.py \
      --model iic/speech_eres2netv2_sv_zh-cn_16k-common \
      --out runs/02_eres2netv2/
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--gt",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "tests/fixtures/test_ground_truth.json",
    )
    ap.add_argument(
        "--audio",
        type=Path,
        default=Path(__file__).resolve().parent / "test_16k.wav",
    )
    ap.add_argument("--k", type=int, default=4, help="forced cluster count")
    ap.add_argument("--cluster", default="spectral",
                    choices=["spectral", "kmeans", "agglo"])
    ap.add_argument("--device", default="cuda")
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
    log(f"[load] audio loaded: {wav_np.shape[0]/sr:.1f}s @ {sr}Hz")

    gt = json.loads(args.gt.read_text())
    utts = gt["utterances"]
    log(f"[load] gt utterances={len(utts)}, speakers={gt['speakers']}")

    log(f"[model] loading {args.model} on {args.device}")
    from modelscope.pipelines import pipeline
    pipe = pipeline(task="speaker-verification", model=args.model,
                    device=args.device)
    # The pipeline's underlying `model` is a Module whose forward takes
    # raw audio [N, T] and does fbank internally.
    embedding_model = pipe.model
    log(f"[model] using {type(embedding_model).__name__}")

    embeddings = np.zeros((len(utts), 192), dtype=np.float32)
    seg_meta: list[tuple[float, float, str]] = []
    t0 = time.time()
    with torch.no_grad():
        for i, u in enumerate(utts):
            s = int(u["t0_start_sec"] * sr)
            e = int(u["t0_end_sec"] * sr)
            s = max(0, s)
            e = min(wav_np.shape[0], e)
            if e - s < int(0.40 * sr):
                need = int(0.40 * sr) - (e - s)
                chunk = np.concatenate(
                    [wav_np[s:e], np.zeros(need, dtype=np.float32)])
            else:
                chunk = wav_np[s:e]
            a = torch.from_numpy(chunk).unsqueeze(0)  # [1, T]
            emb = embedding_model(a)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            emb = emb.squeeze().detach().cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            embeddings[i] = emb
            seg_meta.append((u["t0_start_sec"], u["t0_end_sec"], u["speaker"]))
            if (i + 1) % 50 == 0:
                log(f"[embed] {i+1}/{len(utts)} ({(time.time()-t0):.1f}s)")
    log(f"[embed] done {len(utts)} utts in {time.time()-t0:.1f}s")

    # Cluster.
    log(f"[cluster] method={args.cluster} K={args.k}")
    if args.cluster == "spectral":
        from sklearn.cluster import SpectralClustering
        # Build cosine-affinity matrix.
        sim = embeddings @ embeddings.T
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
        labels = SpectralClustering(
            n_clusters=args.k, affinity="precomputed",
            assign_labels="kmeans", random_state=0,
        ).fit_predict(sim)
    elif args.cluster == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=args.k, random_state=0, n_init=10
                        ).fit_predict(embeddings)
    else:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(
            n_clusters=args.k, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
    log(f"[cluster] label hist: "
        f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    pred_path = args.out / "predictions.jsonl"
    with pred_path.open("w") as fp:
        for (t0_, t1_, _spk), lab in zip(seg_meta, labels):
            fp.write(json.dumps({"t0": t0_, "t1": t1_,
                                 "speaker_id": int(lab)}) + "\n")
    log(f"[out] {pred_path}")

    np.save(args.out / "embeddings.npy", embeddings)
    (args.out / "run.log").write_text("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
