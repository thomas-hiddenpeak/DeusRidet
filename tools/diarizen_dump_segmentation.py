#!/usr/bin/env python3
# TODO(native-cuda-port): produces the bit-equality ground truth for the
# DiariZen-v2 segmentation orchestrator (P1c): sliding-window chunking +
# per-chunk powerset decode + median filtering. Lives under tools/ because it
# runs in the py310_diarizen env, not in awaken. See docs/{en,zh}/architecture/
# 12-diarizen.md "Architectural anchor".
"""Dump the DiariZen-v2 segmentation stage reference for P1c.

The native CUDA orchestrator must reproduce ``segmentations.data`` exactly:

    1. resample/downmix the input to 16 kHz mono (done here by librosa so the
       C++ side receives the identical waveform via ``wave_full``);
    2. slide a 16 s window with a 1.6 s step (step = 0.1 * 16 s), unfolding the
       complete chunks and zero-padding one final partial chunk when the audio
       length is not an exact multiple of the step;
    3. per chunk, run WavLM-pruned -> Conformer -> classifier -> 16-way powerset
       logits, take a per-frame argmax, and map through the powerset->multilabel
       matrix to a [799, 4] binary map (soft=False);
    4. stack to [num_chunks, 799, 4] and apply
       ``scipy.ndimage.median_filter(size=(1, 11, 1), mode='reflect')``.

Keys written into the .npz (the C++ verification targets):

    wave_full        float32 [N]                 16 kHz mono waveform fed to the
                                                  segmenter (after model.audio)
    seg_raw          float32 [num_chunks, 799, 4] per-chunk multilabel, pre
                                                  median filter
    seg_med          float32 [num_chunks, 799, 4] post median filter (the
                                                  binarized_segmentations the
                                                  clustering stage consumes)
    num_chunks       int64   scalar
    window_size      int64   scalar (= 256000)
    step_size        int64   scalar (= 25600)

Run inside the py310_diarizen env:

    conda activate py310_diarizen
    python tools/diarizen_dump_segmentation.py \
        --audio tests/test.mp3 --offset 0.0 --duration 30.0 \
        --out tests/fixtures/diarizen_p1c_segmentation.npz

The output file is NOT committed - it is regenerable. The script is committed
because it pins the exact recipe.
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
        help="window length in seconds (>=16 s exercises >1 chunk + last chunk)",
    )
    p.add_argument("--out", required=True, help="output .npz path (overwritten)")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    audio_path = Path(args.audio).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import librosa
    import numpy as np
    import torch
    from scipy.ndimage import median_filter

    _orig_torch_load = torch.load

    def _patched_load(*a, **kw):  # noqa: ANN001
        kw["weights_only"] = False
        return _orig_torch_load(*a, **kw)

    torch.load = _patched_load  # type: ignore[assignment]

    from diarizen.pipelines.inference import DiariZenPipeline  # noqa: PLC0415

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    print(f"[dump] loading {args.model} on {device}", file=sys.stderr)
    pipe = DiariZenPipeline.from_pretrained(args.model).to(device)

    # 16 kHz mono window so torchaudio/model.audio resampling is a no-op and the
    # C++ side can chunk the identical samples.
    print(f"[dump] loading audio: offset={args.offset}s dur={args.duration}s",
          file=sys.stderr)
    wav, _ = librosa.load(str(audio_path), sr=16000, mono=True,
                          offset=args.offset, duration=args.duration)
    wav = np.asarray(wav, dtype=np.float32)
    waveform = torch.from_numpy(wav).unsqueeze(0)  # [1, N]
    file = {"waveform": waveform, "sample_rate": 16000}

    # Exact waveform the segmenter unfolds (after downmix/resample).
    wav_proc, sr = pipe.model.audio(file)
    wav_proc = wav_proc.detach().to("cpu", torch.float32).numpy().reshape(-1)
    assert sr == 16000, sr

    seg = pipe.get_segmentations(file, soft=False)
    seg_raw = np.asarray(seg.data, dtype=np.float32).copy()
    seg_med = median_filter(seg_raw, size=(1, 11, 1), mode="reflect").astype(
        np.float32)

    num_chunks = seg_raw.shape[0]
    window_size = pipe.model.audio.get_num_samples(pipe._segmentation.duration)
    step_size = round(pipe._segmentation.step * 16000)
    print(f"[dump] wave_full={wav_proc.shape} seg={seg_raw.shape} "
          f"num_chunks={num_chunks} window={window_size} step={step_size}",
          file=sys.stderr)

    np.savez_compressed(
        out_path,
        wave_full=wav_proc,
        seg_raw=seg_raw,
        seg_med=seg_med,
        num_chunks=np.int64(num_chunks),
        window_size=np.int64(window_size),
        step_size=np.int64(step_size),
    )
    print(f"[dump] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
