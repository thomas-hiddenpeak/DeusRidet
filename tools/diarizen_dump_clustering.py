#!/usr/bin/env python3
# TODO(native-cuda-port): produces the bit-equality ground truth for the
# DiariZen-v2 VBx clustering stage (P2b). Runs in py310_diarizen.
"""Dump the DiariZen-v2 VBx clustering reference for P2b.

The native C++ clustering (small-N serial glue: AHC + PLDA transform + VBx
HMM/GMM EM + constrained assignment — an external-library algorithm with no
GPU entry point) must reproduce ``hard_clusters``. This script dumps the real
pipeline embeddings + binarized segmentation as the C++ INPUT, plus every
intermediate tap as the reference, by replicating VBxClustering.__call__ with
the exact upstream functions.

Inputs for C++ (npz keys):
    embeddings   float32 [C, S, 256]   per (chunk, local-speaker) embedding,
                                        NaN rows for inactive speakers
    seg          float32 [C, F, S]     binarized (median-filtered) segmentation
    num_chunks, num_local_speakers, num_frames, dim   int64 scalars

Reference taps:
    train_idx_chunk  int64 [N]
    train_idx_spk    int64 [N]
    train_emb        float32 [N, 256]
    fea              float32 [N, 128]   PLDA-space features
    ahc              int64   [N]        AHC labels (0-based, renumbered)
    gamma            float32 [N, K0]    VBx responsibilities
    pi               float32 [K0]       VBx priors
    centroids        float32 [K, 256]
    hard             int8    [C, S]     final hard_clusters (the C++ target)

Run inside py310_diarizen:
    python tools/diarizen_dump_clustering.py --audio tests/test.mp3 \
        --offset 0.0 --duration 120.0 \
        --out tests/fixtures/diarizen_p2b_clustering.npz
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
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


def main() -> int:
    a = _args()
    out_path = Path(a.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import librosa
    import numpy as np
    import torch
    from scipy.ndimage import median_filter
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from pyannote.core import SlidingWindowFeature

    _o = torch.load
    torch.load = lambda *aa, **kk: (kk.update(weights_only=False) or _o(*aa, **kk))

    from diarizen.pipelines.inference import DiariZenPipeline
    from diarizen.clustering.VBx import vbx_setup, cluster_vbx

    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu"
                       else "cpu")
    print(f"[dump] loading {a.model} on {dev}", file=sys.stderr)
    pipe = DiariZenPipeline.from_pretrained(a.model).to(dev)

    wav, _ = librosa.load(str(a.audio), sr=16000, mono=True, offset=a.offset,
                          duration=a.duration)
    wav = np.asarray(wav, dtype=np.float32)
    file = {"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": 16000}

    seg = pipe.get_segmentations(file, soft=False)
    seg_med = median_filter(np.asarray(seg.data, dtype=np.float32),
                            size=(1, 11, 1), mode="reflect").astype(np.float32)
    binseg = SlidingWindowFeature(seg_med, seg.sliding_window)

    embeddings = pipe.get_embeddings(file, binseg,
                                     exclude_overlap=pipe.embedding_exclude_overlap)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    C, S, D = embeddings.shape
    F = seg_med.shape[1]
    print(f"[dump] embeddings={embeddings.shape} seg={seg_med.shape}",
          file=sys.stderr)

    clu = pipe.clustering  # the VBxClustering instance (already instantiated)

    # --- Replicate VBxClustering.__call__ capturing taps -------------------
    train_emb, tci, tsi = clu.filter_embeddings(embeddings, segmentations=binseg,
                                                min_frames_ratio=0.1)
    print(f"[dump] train_emb={train_emb.shape}", file=sys.stderr)

    tn = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    dendro = linkage(tn, method="centroid", metric="euclidean")
    ahc = fcluster(dendro, clu.ahc_threshold, criterion=clu.ahc_criterion) - 1
    _, ahc = np.unique(ahc, return_inverse=True)

    x_tf, plda_tf, plda_psi = vbx_setup(clu.plda_dir)
    fea = plda_tf(x_tf(train_emb), lda_dim=clu.lda_dim).astype(np.float64)
    Phi = plda_psi[:clu.lda_dim]
    q, sp = cluster_vbx(ahc, fea, Phi, Fa=clu.Fa, Fb=clu.Fb, maxIters=clu.maxIters)

    centroids = (q[:, sp > 1e-7].T @ train_emb.reshape(-1, D))
    e2k = cdist(embeddings.reshape(-1, D), centroids, metric="cosine")
    e2k = e2k.reshape(C, S, -1)
    soft = 2 - e2k
    soft = np.nan_to_num(soft, nan=np.nanmin(soft))
    hard = -2 * np.ones((C, S), dtype=np.int8)
    for c in range(C):
        rs, ks = linear_sum_assignment(soft[c], maximize=True)
        for s, k in zip(rs, ks):
            hard[c, s] = k
    _, hflat = np.unique(hard, return_inverse=True)
    hard = hflat.reshape(C, S).astype(np.int8)
    # match the pipeline's -2-for-inactive bookkeeping done OUTSIDE clustering:
    # here we keep raw hard_clusters as the clustering() return (no inactive -2).

    # cross-check against the real clustering() call
    ref_hard, _, _ = clu(embeddings=embeddings, segmentations=binseg,
                         min_clusters=pipe.min_speakers,
                         max_clusters=pipe.max_speakers)
    agree = float(np.mean(ref_hard == hard))
    print(f"[dump] replication vs clustering() agreement={agree:.4f} "
          f"K={centroids.shape[0]} ahc_K={int(ahc.max())+1}", file=sys.stderr)

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        seg=seg_med,
        num_chunks=np.int64(C),
        num_local_speakers=np.int64(S),
        num_frames=np.int64(F),
        dim=np.int64(D),
        train_idx_chunk=tci.astype(np.int64),
        train_idx_spk=tsi.astype(np.int64),
        train_emb=train_emb.astype(np.float32),
        fea=fea.astype(np.float32),
        ahc=ahc.astype(np.int64),
        gamma=q.astype(np.float32),
        pi=sp.astype(np.float32),
        centroids=centroids.astype(np.float32),
        hard=ref_hard.astype(np.int8),
        ahc_threshold=np.float64(clu.ahc_threshold),
        Fa=np.float64(clu.Fa),
        Fb=np.float64(clu.Fb),
        lda_dim=np.int64(clu.lda_dim),
        max_iters=np.int64(clu.maxIters),
    )
    print(f"[dump] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
