#!/usr/bin/env python3
# TODO(native-cuda-port): produces the bit-equality ground truth for the
# DiariZen-v2 *pipeline post-processing* (P3a): get_embeddings windowing,
# reconstruct, Inference.aggregate overlap-add, speaker_count, and Binarize.
# Lives under tools/ because it runs in the py310_diarizen env, not in awaken.
# See docs/{en,zh}/architecture/12-diarizen.md phase P3a.
"""Dump the full DiariZen-v2 pipeline intermediates from the live model.

Replicates `DiariZenPipeline.__call__` inline (see diarizen/pipelines/
inference.py) on a fixed slice of tests/test.mp3, capturing every
intermediate the native `DiarizenPipeline` (src/orator/diarizen_pipeline.cpp)
must reproduce:

    wave_in              float32 [N]            16 kHz mono input slice
    seg_data             float32 [C, F, S]      raw segmentation (powerset
                                                 binarized; pre median filter
                                                 is identical to post here
                                                 because median is applied
                                                 in-place below)
    seg_window           float64 [3]            chunk SlidingWindow:
                                                 (start, duration, step)
    frames_window        float64 [3]            receptive-field SlidingWindow:
                                                 (start, duration, step) — the
                                                 frame resolution used by
                                                 aggregate / speaker_count
    count_data           uint8   [Tg, 1]        instantaneous speaker count
    count_window         float64 [3]            count SlidingWindow
    embeddings           float32 [C, S, D]      per (chunk, speaker) x-vectors
                                                 (NaN rows for inactive spk)
    hard_clusters        int64   [C, S]         clustering output (-2 inactive)
    discrete_data        float32 [Td, K]        reconstructed discrete diar
    discrete_window      float64 [3]            discrete SlidingWindow
    segments             float64 [M, 2]         final (start, end) intervals
    segment_labels       int64   [M]            integer speaker label per seg

Run inside the py310_diarizen env:

    conda activate py310_diarizen
    python tools/diarizen_dump_pipeline.py \
        --audio tests/test.mp3 \
        --offset 0.0 --duration 30.0 \
        --out tests/fixtures/diarizen_p3a_pipeline.npz

The output file is NOT committed - it is regenerable. The script is
committed because it pins the exact recipe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio", required=True, help="path to test.mp3 (or wav)")
    p.add_argument(
        "--model",
        default="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        help="HF model id used by DiariZenPipeline.from_pretrained",
    )
    p.add_argument("--offset", type=float, default=0.0, help="start in seconds")
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="slice length in seconds (>=16 to exercise multi-chunk logic)",
    )
    p.add_argument("--out", required=True, help="output .npz path (overwritten)")
    p.add_argument(
        "--device", default="cuda", choices=("cuda", "cpu"),
        help="torch device (default cuda)",
    )
    return p.parse_args()


def _load_audio_window(path: str, offset: float, duration: float):
    """Load a mono 16 kHz float32 slice (matches dump_reference recipe)."""
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    if info.samplerate != 16000:
        import librosa
        wav, _ = librosa.load(path, sr=16000, mono=True, offset=offset,
                              duration=duration)
    else:
        start = int(round(offset * 16000))
        nsamp = int(round(duration * 16000))
        wav, _ = sf.read(path, start=start, frames=nsamp, dtype="float32",
                         always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32)


def _sw_triple(sw):
    """SlidingWindow -> (start, duration, step) float64 triple."""
    import numpy as np
    return np.asarray([sw.start, sw.duration, sw.step], dtype=np.float64)


def main() -> int:
    args = _parse_args()
    audio_path = Path(args.audio).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from scipy.ndimage import median_filter

    # Mirror the worker's torch.load patch so the tool runs standalone.
    _orig = torch.load

    def _patched(*a, **kw):  # noqa: ANN001
        kw["weights_only"] = False
        return _orig(*a, **kw)

    torch.load = _patched  # type: ignore[assignment]

    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.core import SlidingWindowFeature
    from pyannote.audio.utils.signal import Binarize

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu")
        else "cpu")
    print(f"[p3a] loading {args.model} on {device}", file=sys.stderr)
    pipe = DiariZenPipeline.from_pretrained(args.model).to(device)

    print(f"[p3a] audio slice offset={args.offset}s duration={args.duration}s",
          file=sys.stderr)
    wav = _load_audio_window(str(audio_path), args.offset, args.duration)
    waveform = torch.from_numpy(wav).unsqueeze(0)  # [1, N] mono (SDM channel 0)
    sample_rate = 16000

    # --- replicate DiariZenPipeline.__call__ -----------------------------
    print("[p3a] segmentations", file=sys.stderr)
    segmentations = pipe.get_segmentations(
        {"waveform": waveform, "sample_rate": sample_rate}, soft=False)

    if pipe.apply_median_filtering:
        segmentations.data = median_filter(
            segmentations.data, size=(1, 11, 1), mode="reflect")

    binarized_segmentations = segmentations  # powerset

    frames = pipe._segmentation.model._receptive_field
    count = pipe.speaker_count(
        binarized_segmentations, frames, warm_up=(0.0, 0.0))

    print("[p3a] embeddings", file=sys.stderr)
    embeddings = pipe.get_embeddings(
        {"waveform": waveform, "sample_rate": sample_rate},
        binarized_segmentations,
        exclude_overlap=pipe.embedding_exclude_overlap,
    )  # (C, S, D)

    print("[p3a] clustering", file=sys.stderr)
    hard_clusters, _, _ = pipe.clustering(
        embeddings=embeddings,
        segmentations=binarized_segmentations,
        min_clusters=pipe.min_speakers,
        max_clusters=pipe.max_speakers,
    )

    count.data = np.minimum(count.data, pipe.max_speakers).astype(np.int8)

    inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    hard_clusters[inactive_speakers] = -2

    discrete_diarization, _ = pipe.reconstruct(
        segmentations, hard_clusters, count)

    to_annotation = Binarize(
        onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)
    result = to_annotation(discrete_diarization)

    # Flatten the final annotation into (start, end, int_label) rows. The
    # Binarize tracks are per-speaker-column; the label is the discrete
    # speaker index (column k), recovered from the track name.
    seg_rows = []
    seg_labels = []
    for segment, _track, label in result.itertracks(yield_label=True):
        seg_rows.append([float(segment.start), float(segment.end)])
        seg_labels.append(int(label))

    # --- save ------------------------------------------------------------
    out = {
        "wave_in": wav.astype(np.float32),
        "seg_data": np.asarray(binarized_segmentations.data, dtype=np.float32),
        "seg_window": _sw_triple(binarized_segmentations.sliding_window),
        "frames_window": _sw_triple(frames),
        "count_data": np.asarray(count.data, dtype=np.uint8),
        "count_window": _sw_triple(count.sliding_window),
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "hard_clusters": np.asarray(hard_clusters, dtype=np.int64),
        "discrete_data": np.asarray(discrete_diarization.data, dtype=np.float32),
        "discrete_window": _sw_triple(discrete_diarization.sliding_window),
        "segments": (np.asarray(seg_rows, dtype=np.float64)
                     if seg_rows else np.zeros((0, 2), dtype=np.float64)),
        "segment_labels": (np.asarray(seg_labels, dtype=np.int64)
                           if seg_labels else np.zeros((0,), dtype=np.int64)),
    }

    np.savez_compressed(out_path, **out)
    print(f"[p3a] wrote {out_path}", file=sys.stderr)
    for k, v in out.items():
        print(f"  {k:18s} {str(v.shape):18s} {v.dtype}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
